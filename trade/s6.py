"""職能 2 のための模型 s6。s5（職能 1 で凍結）は変えず、s5.run の本文を写して、聞いた言い値だけを知る仲介者を差し込む口を三か所だけ足した。
仲介者を置かない局（運用役なし、または s5 の運用役だけ）では、s5.run と同じ乱数を同じ順に使い、同じ記録を返す（tests/test_s6.py）。

仲介者（Mediator、s5.Broker の子。種類 "broker"、名 "broker"）
  知ること   住み手から聞いた言い値（損をする世帯の補償の額と、得をする世帯の出してよい額）だけ。性向は知らない。
  いつ動くか  ある年に改変を求めた世帯（提案者）が、自分の手続き（TP+top3・改良版top3）で手を打てなかったときだけ、提案者が相談に来る。
  何を検討するか（聞く範囲 scope）
             提案者が自分で検討した手（自分の得の大きい順に上から 3 手まで）のうち、次の手を、同じ順に検討する。
             "stuck"（届かなかった手だけ。主）：損をする世帯に提案者の届かない戸がいた手（近い戸にしか聞けない場合）と、
                     自分の得が言い値の合計を上回ったのに、工事の費用を足すと上回らなかった手（費用がある場合）。
             "all"（決まらなければ相談）：上から 3 手すべて。
  誰に何を聞くか
             その手で満足が動く世帯すべてに聞く。遠い戸にも届く。損をする世帯には補償の額（言い値）、得をする世帯には出してよい額
             （自分の得）、提案者には自分の得を聞く。聞いた額から分かる係数（s1.revealed と同じ規則）を learn で記録に残す。
  何を出すか 得をする世帯の得の合計が、損をする世帯の言い値の合計と工事の費用を上回る手があれば、最初のその手を束として出す。
             損をする世帯は言い値ちょうどを受け取り、言い値の合計と費用は、得をする世帯（提案者を含む）が得の大きさに比べて割り勘にする。
             得をする世帯はどれも、払った後も得が残る。
  決めるのは住民
             満足の動く世帯すべてが受け入れたときだけ打つ（打たれた手の同意の世帯は、満足の動く世帯すべて）。〔貸〕の上限は、自分の手続きと同じ。
  住み手が知ること
             住み手は、仲介者から自分の払う額か受け取る額だけを聞く。束に加わる世帯が二戸だけなら、自分の払う額から相手の額が分かるので、
             相手の係数を知る（確証の市場型 M と同じ近似）。三戸以上なら、ほかの世帯の額は分からない。
全知の仲介者 BR（名付けの偽装、regime "BR"・"BRdyn"）は s5 と同じ（s1 の central の写し）。BR との違いは、誰が何を知るかだけである。
BR は全員の係数を知るので、頼まれなくても全知の最適へ部位ごとに組み替える。仲介者は聞いた額しか知らないので、持ち込まれた手しか検討できない。
"""
from __future__ import annotations
import numpy as np
from trade import s5
from trade.s5 import *                    # noqa: F401,F403  s5.run の本文が使う名をすべて写す
from trade.s5 import _reconsent           # noqa: F401
from trade.s0 import MU
from trade.s1 import NEAR, revealed, getp, move_cost


class Mediator(s5.Broker):
    """聞いた言い値だけを知る仲介者（職能 2）。種類は s5.Broker と同じ（broker）"""

    def __init__(self, scope="stuck", name="broker"):
        super().__init__(name)
        assert scope in ("stuck", "all")
        self.scope = scope
        self.n_consult = 0          # 相談の回数
        self.n_ask = 0              # 聞いた世帯の延べ数（提案者を含む）
        self.n_bundle = 0           # 出して打たれた束の数
        self.n_bundle3 = 0          # そのうち三戸以上の束
        self.n_far = 0              # そのうち提案者から遠い戸を含む束
        self.n_copay = 0            # そのうち提案者のほかにも払う世帯がいる束

    def consult(self, t, i, evals, env):
        """提案者 i の相談。evals は提案者が自分で検討した手（聞く範囲で絞った後）。束を出せれば dict、出せなければ None。
        env["reach"] は提案者に届く戸（近い戸にしか聞けない場合。None なら全員に届く）"""
        ctx, st, cost, rcost, acc, lim, quotes, reach = (env[k] for k in ("ctx", "st", "cost", "rcost", "acc", "lim", "quotes", "reach"))
        brought = []
        for x in evals:
            mv, dS, dC, nxt = x
            p = s5.part_of5(mv); nv = int(getp(nxt, p))
            cst = move_cost(p, nv, st, cost, rcost)
            if mv[0] == "swap":
                cst += move_cost(("e", mv[2]), 0, st, cost, rcost)
            harmed = [j for j in range(9) if j != i and dS[j] < -1e-12]
            if dS[i] <= 1e-12:
                continue
            if self.scope == "stuck":
                far = reach is not None and any(not reach[j] for j in harmed)
                net = dS[i] / MU - sum(-dS[j] / MU for j in harmed)
                by_cost = (not far) and net > 1e-12 and net - cst <= 1e-12
                if not (far or by_cost):
                    continue
            brought.append((x, cst, harmed))
        if not brought:
            return None
        self.n_consult += 1
        for (mv, dS, dC, nxt), cst, harmed in brought:
            gain = [j for j in range(9) if dS[j] > 1e-12]
            parts = sorted(set(gain) | set(harmed))
            # 聞く：満足の動く世帯すべて（提案者を含む）。聞いた額から分かる係数を記録に残す
            for j in parts:
                for c_ in s5.rev_of(mv, j):
                    self.learn(ctx, t, j, c_)
                self.n_ask += 1
            G = np.array([max(0.0, dS[j] / MU) for j in range(9)])
            L = np.array([max(0.0, -dS[j] / MU) if j in harmed else 0.0 for j in range(9)])
            C = L.sum() + cst
            if G.sum() <= C + 1e-12:
                continue
            s = C * G / G.sum()                       # 得の大きさに比べた割り勘
            co = [j for j in gain if j != i]
            pay = np.zeros(9)
            for j in harmed:
                pay[j] = L[j]                         # 提案者 → 損をする世帯（言い値ちょうど）
            for j in co:
                pay[j] = -s[j]                        # ともに得をする世帯 → 提案者（割り勘の分）
            val = G[i] - s[i]
            if lim is not None:
                out_i = max(0.0, pay.sum()) + cst
                if acc[i] - out_i < -lim[i] - 1e-9:
                    continue
                if any(pay[j] < 0 and acc[j] + pay[j] < -lim[j] - 1e-9 for j in range(9)):
                    continue
            # 住み手が知ること：束が二戸だけなら、自分の払う額（受け取る額）から相手の額が分かる
            if len(parts) == 2:
                other = [j for j in parts if j != i][0]
                for c_ in s5.rev_of(mv, other):
                    quotes.append((t, int(i), other, c_))
                if other in co:
                    for c_ in s5.rev_of(mv, i):
                        quotes.append((t, other, int(i), c_))
            self.n_bundle += 1
            self.n_bundle3 += len(parts) >= 3
            self.n_far += any(j != i and j not in NEAR[i] for j in parts)
            self.n_copay += len(co) > 0
            return dict(best=(float(val), True, pay, mv, dS, dC, nxt, cst, 0.0, 0.0), co=co, parts=parts)
        return None


def run(W, regime, seed, years=30, p_req=0.3, form="linear", pi=0.5, start=None,
        theta=0.0, budget=None, conv_scale=1.0, nu=0.0, newW=None,
        split=False, credit=None, settle=False, reconsent=None,
        protocol="full", ask=3, cost=None, rcost=None, turn_at=None, ref=None,
        turnover=None, restore=None, amort=10, horizon=None, owner=None, venue=None, actors=None):
    actors = list(actors or [])
    meds = [a for a in actors if isinstance(a, Mediator)]
    if len(meds) > 1:
        raise ValueError("仲介者は一人まで")
    med = meds[0] if meds else None
    if med is not None and (owner is not None or horizon is not None or regime not in ("T1+", "T1")):
        raise NotImplementedError("仲介者を置けるのは、言い値の取引（T1・T1+）で、所有者と見通しのない場合だけ")
    boards = [a.board for a in actors if a.board is not None]
    if len(boards) > 1:
        raise ValueError("盤の設計者は一人まで")
    B = boards[0] if boards else BASE
    SA = B.S
    if B is not BASE and (owner or venue or restore or regime in ("P", "BR", "BRdyn", "R", "Rc") or form != "linear"):
        if B is RICH:
            raise NotImplementedError("戻せる手の多い盤で使えるのは、言い値の取引・入れ替わり・改良版の部品だけ")
    W = np.array(W, float).copy()
    ss = np.random.SeedSequence([seed, 7])
    r_req, r_ord, r_vote = [np.random.default_rng(s) for s in ss.spawn(3)]
    REQ = r_req.random((years, 9))
    ORD = np.array([r_ord.permutation(9) for _ in range(years)])
    VOT = r_vote.random((years, 9, 9))
    r_turn = np.random.default_rng(np.random.SeedSequence([seed, 7, 99]))
    TURN = r_turn.random((years, 9))
    r_null = np.random.default_rng(np.random.SeedSequence([seed, 7, 123]))
    r_type = np.random.default_rng(np.random.SeedSequence([seed, 7, 77]))
    r_ven = np.random.default_rng(np.random.SeedSequence([seed, 7, 55]))
    e, k, o = start if start is not None else B.start(form)
    e, k, o = e.copy(), k.copy(), o.copy()
    X0 = (e.copy(), k.copy(), o.copy())
    st = (e.copy(), k.copy(), o.copy()) if ref is None else tuple(a.copy() for a in ref)
    c = np.zeros(9); m = np.zeros(9); b = np.zeros(9)
    n_recon = n_revert = 0; n_exit = 0; exit_bal = 0.0
    moved = np.zeros(9, bool)
    Wbar = np.tile(W_BAR * np.array([1.0, conv_scale, conv_scale, conv_scale]), (9, 1))
    events, quotes, turns, recon = [], [], [], []
    spent = 0.0
    extra = dict(owner_in=0.0, owner_out=0.0, tcost=0.0, vetoed=0, unreached=0, restored=0, refund=0.0)
    Wt = [SA(W, e, k, o, form).sum()]
    acc = c if credit is not None else m
    lim = np.full(9, credit) if credit is not None else budget

    # ---- 運用役の文脈
    ledger, seqlog = [], []          # ledger：(年, 戸, 係数, 順)、seqlog：入れ替わりと記録の順（推定器が読む順を決める）
    _seq = [0]

    def nextseq():
        _seq[0] += 1
        return _seq[0]
    ctx = dict(S=SA, form=form, Wbar=Wbar, start=st, ledger=ledger, seq=nextseq, board=B, blocked=[], mixed=[])
    ctx["cfg"] = lambda: (e, k, o)
    if actors:
        for a in actors:
            ov = a.init(ctx) or {}
            protocol = ov.get("protocol", protocol); ask = ov.get("ask", ask)
        if not all(a.allows_reconsent for a in actors):
            reconsent = None
        if not all(a.allows_settle for a in actors):
            settle = False

    kind = turnover["kind"] if turnover else None
    if kind == "geo":
        nu = turnover["nu"]
    if kind == "mix":
        sig = np.where(r_type.random(9) < turnover.get("piS", 0.72), 0, 1)
        nus = (turnover.get("nuS", 0.32), turnover.get("nuL", 0.087))
    if kind == "renew":
        h0 = renew_base(turnover["nu"], turnover.get("rho", 2.0)); rho = turnover.get("rho", 2.0)
        nu = turnover["nu"]
    tenure = np.zeros(9, int)
    tenant = np.arange(9); nxt_tenant = 9

    def mean_nu(i):
        if kind == "mix":
            return nus[sig[i]]
        return nu

    def hazard(i):
        if kind == "mix":
            return nus[sig[i]]
        if kind == "renew":
            return min(1.0, h0 * (rho if (tenure[i] + 1) % 2 == 0 else 1.0))
        return nu
    turning = (nu > 0 or turn_at or kind is not None) and newW is not None

    author = {}
    own = owner or None
    Lam = set(own.get("Lam", set())) if own else set()

    def owner_value(e0, k0, o0, e1, k1, o1):
        D0, X0_ = market(e0, k0, o0); D1, X1_ = market(e1, k1, o1)
        gam = own.get("gamma", 1.0) * (own.get("M", 10) if own["kind"] == "sub" else 1.0)
        return own.get("beta", 0.5) * float((D1 - D0).sum()) / MU - gam * float((X1_ - X0_).sum())

    def d4_list():
        L = set()
        for typ in ("ext", "store", "open"):
            cnt = 0
            S0_ = SA(W, e, k, o, form)
            for i in range(9):
                for mv in B.legal(i, e, k, o):
                    if TYPE[mv[0]] != typ:
                        continue
                    e1, k1, o1 = apply(mv, i, e, k, o)
                    if SA(W, e1, k1, o1, form)[i] - S0_[i] > 1e-12:
                        cnt += 1; break
            if cnt >= 5:
                L.add(typ)
        return L

    def central(t, W_now):
        nonlocal e, k, o, spent
        if regime == "P":
            e2, k2, o2, cst = best_net(W_now, e, k, o, st, cost, rcost, form)
            events.append(dict(t=t, i=None, part=None, ok=True, planner=True, cons=False, supp=set(), drafter="planner",
                               changed=bool(any((a != b_).any() for a, b_ in zip((e2, k2, o2), (e, k, o))))))
            e, k, o = e2, k2, o2; spent += cst
        else:
            ez, kz, oz, _ = best_net(W_now, e, k, o, st, cost, rcost, form)
            tgt = (ez, kz, oz)
            events.append(dict(t=t, i=None, part=None, ok=False, cons=False, supp=set(), drafter="broker", look=True))
            for p in KEYS:
                v = getp(tgt, p)
                if getp((e, k, o), p) == v:
                    continue
                S0_ = SA(W_now, e, k, o, form)
                nx = [e.copy(), k.copy(), o.copy()]; {"e": nx[0], "k": nx[1], "o": nx[2]}[p[0]][p[1]] = v
                dS = SA(W_now, *nx, form) - S0_
                harmed = {j for j in range(9) if dS[j] < -1e-12}; gain = {j for j in range(9) if dS[j] > 1e-12}
                spent += move_cost(p, v, st, cost, rcost)
                events.append(dict(t=t, i=None, part=p, nv=int(v), ok=True, dS=dS, pay=np.zeros(9), cons=True,
                                   supp=harmed | gain, drafter="broker", broker=True))
                author.pop(p, None)
                e, k, o = nx

    if regime in ("P", "BR", "BRdyn"):
        central(-1, W)
    for t in range(years):
        if own and own.get("fix") == "D4":
            Lam = d4_list()
        if venue and venue.get("G") == 1:
            fee = venue.get("tauG", 0.5) / 9
            m -= fee; extra["tcost"] += fee * 9
        if regime in ("R", "Rc"):
            for i in ORD[t]:
                if REQ[t, i] >= p_req:
                    continue
                cands = B.moves(e, k, o)
                mv = cands[int(r_null.integers(len(cands)))]
                S0_ = SA(W, e, k, o, form)
                e1, k1, o1 = apply(mv, i, e, k, o)
                dS = SA(W, e1, k1, o1, form) - S0_
                pay = np.zeros(9)
                if regime == "Rc":
                    for j in range(9):
                        if j != i:
                            pay[j] = max(0.0, -dS[j] / MU)
                    m[i] -= pay.sum(); m += pay
                p = part_of(mv); nv = int(getp((e1, k1, o1), p))
                cst = move_cost(p, nv, st, cost, rcost); spent += cst; m[i] -= cst
                events.append(dict(t=t, i=int(i), part=p, nv=nv, ok=True, dS=dS, pay=pay, cons=False, supp=set(),
                                   drafter="random", cost=cst))
                author.pop(p, None)
                e, k, o = e1, k1, o1
        elif regime not in ("F", "P", "BR", "BRdyn"):
            plus = regime.endswith("+")
            base = regime.rstrip("+")
            if base == "AI":
                base = "T1"
            gv = np.array([g_of(mean_nu(j), horizon) for j in range(9)])
            for i in ORD[t]:
                if REQ[t, i] >= p_req:
                    continue
                S0_ = SA(W, e, k, o, form); C0 = SA(Wbar, e, k, o, form)
                cands = B.moves(e, k, o) if plus else B.legal(i, e, k, o)
                tr = base in ("T0", "T1", "T1h")
                reach = None
                if venue and venue.get("G") == 0 and tr:
                    u = r_ven.random(9)
                    reach = np.array([j == i or j in NEAR[i] or u[j] < venue.get("q_far", 0.4) for j in range(9)])
                evals = []
                for mv in cands:
                    e1, k1, o1 = apply(mv, i, e, k, o)
                    dS = SA(W, e1, k1, o1, form) - S0_
                    dC = SA(Wbar, e1, k1, o1, form) - C0
                    evals.append((mv, dS, dC, (e1, k1, o1)))
                if actors:
                    ctx["W"] = W
                    for a in actors:
                        evals = a.evals(t, int(i), evals, ctx)
                drafter = "self"
                if regime == "AI+":
                    good = []
                    for x in evals:
                        p = part_of5(x[0]); nv = int(getp(x[3], p))
                        val, ok, pay = decide("T1", i, x[1], x[2], None, pi, theta, False)
                        val -= move_cost(p, nv, st, cost, rcost)
                        if ok and val > 1e-12:
                            good.append((x[1].sum() - MU * move_cost(p, nv, st, cost, rcost), x))
                    evals = [max(good, key=lambda z: z[0])[1]] if good else []
                    drafter = "ai"
                elif protocol == "topk" and tr:
                    evals = [x for x in evals if x[1][i] * gv[i] > 1e-12]
                    evals.sort(key=lambda x: -x[1][i])
                    evals = evals[:ask]
                elif protocol == "near" and tr:
                    evals = [x for x in evals if {j for j in range(9) if j != i and abs(x[1][j]) > 1e-12} <= NEAR[i]]
                best = None
                for mv, dS, dC, nxt in evals:
                    dSe = dS * gv if horizon is not None else dS
                    harmed_set = [j for j in range(9) if j != i and dS[j] < -1e-12]
                    if reach is not None and any(not reach[j] for j in harmed_set):
                        extra["unreached"] += 1
                        continue
                    val, ok, pay = decide(base, i, dSe, dC, None, pi, theta, split)
                    p = part_of5(mv); nv = int(getp(nxt, p))
                    cst = move_cost(p, nv, st, cost, rcost)
                    if mv[0] == "swap":
                        cst += move_cost(("e", mv[2]), 0, st, cost, rcost)
                    ant = 0.0
                    if horizon is not None and nv != getp(st, p):
                        if restore == "R0" and rcost is not None:
                            ant += float(rcost.get(p[0], 0.0)) * (1 - (1 - mean_nu(i)) ** horizon)
                        if restore == "Rdk" and cost is not None:
                            ant -= float(cost.get(p[0], 0.0)) * refund_frac(mean_nu(i), amort)
                    tc = venue.get("tau", 0.5) * len(harmed_set) if (reach is not None) else 0.0
                    opay = 0.0
                    if own is not None:
                        typ = TYPE[mv[0]]
                        if own["kind"] == "mgmt":
                            okd = typ in Lam
                            if not okd and own.get("fix") == "D5":
                                okd = owner_value(e, k, o, *nxt) + (val - cst - ant - tc) >= -1e-12
                            if not okd:
                                ok = False; val = -1.0
                        else:
                            Vo = owner_value(e, k, o, *nxt)
                            net = val - cst - ant - tc
                            if venue and venue.get("G") == 0.5:
                                okd = Vo + venue.get("alpha", 0.5) * net >= -1e-12
                            elif own.get("fix") == "D5":
                                okd = Vo >= -1e-12 or Vo + net >= -1e-12
                            else:
                                okd = Vo >= -1e-12
                            if not okd and own.get("role") == "trade":
                                opay = -Vo; okd = True
                            if not okd:
                                ok = False; val = -1.0
                    val = val - cst - ant - tc - opay
                    if tr:
                        for j in range(9):
                            if j != i and dS[j] < -1e-12:
                                if mv[0] == "swap":
                                    cs_ = rev_of(mv, j)
                                    for c_ in cs_:
                                        if len(cs_) > 1:
                                            ctx["mixed"].append(len(quotes))
                                        quotes.append((t, int(i), j, c_))
                                else:
                                    quotes.append((t, int(i), j, revealed(mv, j)))
                    if ok and lim is not None:
                        out_i = max(0.0, pay.sum()) + cst + tc + opay
                        if acc[i] - out_i < -lim[i] - 1e-9:
                            ok = False
                        for j in range(9):
                            if pay[j] < 0 and acc[j] + pay[j] < -lim[j] - 1e-9:
                                ok = False
                        if not ok:
                            val = -1.0
                    if best is None or val > best[0]:
                        best = (val, ok, pay, mv, dS, dC, nxt, cst, tc, opay)
                    if protocol == "topk" and tr and ok and val > 1e-12:
                        break
                bundle = None
                if best is None or best[0] <= 1e-12:
                    if best is not None and own is not None and best[1] is False:
                        extra["vetoed"] += 1
                    if med is not None and tr:
                        # 世帯が自分で聞いて決まらなかったときだけ、仲介者に相談する
                        bundle = med.consult(t, int(i), evals, dict(st=st, cost=cost, rcost=rcost, acc=acc, lim=lim,
                                                                    quotes=quotes, ctx=ctx, reach=reach))
                    if bundle is None:
                        continue
                    best = bundle["best"]
                val, ok, pay, mv, dS, dC, nxt, cst, tc, opay = best
                nyes = None
                if base in ("V½", "V¾"):
                    ok, nyes = vote(base, i, dS, VOT[t, i], pi)
                harmed = {j for j in range(9) if j != i and dS[j] < -1e-12}
                cons = tr
                if base in ("U", "V½", "V¾"):
                    cons = len(harmed) == 0
                p = part_of5(mv)
                lab = B.label(mv, i, e) if B is not BASE or mv[0] == "swap" else move_label(mv, i, e)
                mty = B.mtype(mv, e) if B is not BASE or mv[0] == "swap" else move_type(mv, e)
                ev = dict(t=t, i=int(i), move=lab, type=mty, ok=bool(ok), val=float(val),
                          yes=nyes, part=p, nv=int(getp(nxt, p)), dS=dS, pay=pay.copy(), cons=cons,
                          supp=({int(i)} | harmed) if cons else set(), drafter=drafter, cost=cst if ok else 0.0)
                if bundle is not None:
                    # 仲介者の束：満足の動くすべての世帯（提案者・ともに払う世帯・補償を受ける世帯）の同意で打つ
                    ev.update(supp={int(i)} | harmed | set(bundle["co"]), drafter="mediator", bundle=len(bundle["parts"]))
                ev2 = None
                if mv[0] == "swap":
                    # 組の手は、外す部位の記録を二つ目の手として並べる（推定器は部位ごとに同意を数える）
                    ev2 = dict(t=t, i=int(i), move=lab, type="外す（組の手）", ok=bool(ok), val=0.0, yes=nyes,
                               part=("e", mv[2]), nv=0, dS=np.zeros(9), pay=np.zeros(9), cons=cons,
                               supp=set(ev["supp"]), drafter=drafter, cost=0.0, pair=True)
                if ok:
                    for j in range(9):
                        if j != i:
                            b[j] -= dC[j] / MU
                            b[i] += dC[j] / MU
                    for j in range(9):
                        if pay[j] != 0:
                            b[i] += pay[j]; b[j] -= pay[j]
                            acc[i] -= pay[j]; acc[j] += pay[j]
                    acc[i] -= cst; spent += cst
                    if tc or opay:
                        acc[i] -= tc + opay; extra["tcost"] += tc; extra["owner_in"] += opay
                    for pp, evx in ((p, ev), (("e", mv[2]) if ev2 else None, ev2)):
                        if pp is None:
                            continue
                        if evx["nv"] == getp(st, pp):
                            author.pop(pp, None)
                        else:
                            author[pp] = (int(i), int(tenant[i]), t)
                    e, k, o = nxt
                    if actors:
                        for a in actors:
                            a.trade(t, int(i), mv, dS, pay, ctx)
                events.append(ev)
                if ev2 is not None:
                    events.append(ev2)
        changed = False
        if turning:
            for i in range(9):
                if turn_at is not None:
                    out = i in turn_at.get(t, ())
                elif kind is not None:
                    out = TURN[t, i] < hazard(i)
                else:
                    out = TURN[t, i] < nu
                if out:
                    mine = [p for p, (u_, a_, _) in author.items() if u_ == i and a_ == tenant[i]]
                    for p in mine:
                        if restore == "R0":
                            s_ = getp(st, p)
                            if getp((e, k, o), p) != s_:
                                nx = [e.copy(), k.copy(), o.copy()]; {"e": nx[0], "k": nx[1], "o": nx[2]}[p[0]][p[1]] = s_
                                dS = SA(W, *nx, form) - SA(W, e, k, o, form)
                                rc = 0.0 if rcost is None else float(rcost.get(p[0], 0.0))
                                acc[i] -= rc; spent += rc
                                events.append(dict(t=t, i=int(i), part=p, nv=int(s_), ok=True, dS=dS, pay=np.zeros(9),
                                                   cons=False, supp=set(), drafter="restore", cost=rc))
                                e, k, o = nx; extra["restored"] += 1
                            author.pop(p, None)
                        elif restore == "Rdk" and cost is not None:
                            age = t - author[p][2]
                            rf = float(cost.get(p[0], 0.0)) * max(0.0, 1 - age / amort)
                            acc[i] += rf; extra["refund"] += rf; extra["owner_out"] += rf
                    n_exit += 1; exit_bal += abs(b[i]); moved[i] = True
                    turns.append((t, i)); changed = True
                    if actors:
                        seqlog.append(("turn", t, i, nextseq()))
                    W[i] = newW(r_turn, i)
                    tenant[i] = nxt_tenant; nxt_tenant += 1; tenure[i] = 0
                    if kind == "mix":
                        sig[i] = 0 if r_type.random() < turnover.get("piS", 0.72) else 1
                    if settle:
                        m[i] = 0.0; c[i] = 0.0; b[i] = 0.0
                    if reconsent:
                        gn = g_of(mean_nu(i), horizon) if horizon is not None else 1.0
                        e, k, o, nr, nv_, sp = _reconsent(i, t, W, e, k, o, st, form, reconsent, acc, lim, recon, rcost, gn,
                                                          author, events, B, actors, ctx)
                        n_recon += nr; n_revert += nv_; spent += sp
                else:
                    tenure[i] += 1
        if changed and regime in ("P", "BRdyn"):
            central(t, W)
        Wt.append(SA(W, e, k, o, form).sum())
    S_fin = SA(W, e, k, o, form)
    out = dict(e=e, k=k, o=o, S=S_fin, U=S_fin + MU * (m + c), b=b, m=m, c=c, Wt=np.array(Wt), events=events,
               W_final=W, n_exit=n_exit, exit_bal=exit_bal, n_recon=n_recon, n_revert=n_revert, moved=moved,
               quotes=quotes, turns=turns, recon=recon, spent=spent, start=X0, ref=st, extra=extra)
    if actors:
        out.update(board=B.name, ledger=ledger, seqlog=seqlog, blocked=ctx["blocked"], mixed_quotes=ctx["mixed"],
                   actor_facts={a.name: list(a.facts) for a in actors},
                   actor_full={a.name: list(a.full) for a in actors},
                   actor_kind={a.name: a.kind for a in actors},
                   actor_stats={a.name: {k_: v for k_, v in vars(a).items() if k_.startswith("n_")} for a in actors})
    return out
