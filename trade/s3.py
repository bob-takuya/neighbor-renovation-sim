"""段 1 の第二報のための模型 s3。s1（段 2 で凍結）は変えず、写して賃貸の制約を四つの部品として足した。
どの部品も既定では切れていて、既定の引数では s1.run と同じ乱数を同じ順に使い、同じ記録を返す（tests/test_s3.py）。

1. 入れ替わりの型 turnover（既定 None ＝ s1 の nu をそのまま使う）
     dict(kind="geo", nu=ν)                     毎年、各戸が確率 ν で出る（s1 と同じ）
     dict(kind="renew", nu=ν, rho=2)            2 年ごとの更新の年だけ退去の確率が rho 倍。年あたりの平均の退去率が ν になるように基準を合わせる
     dict(kind="mix", nuS=0.32, nuL=0.087, piS=0.72)
                                                短く住む型（年 0.32）と長く住む型（年 0.087）の混合。入居する世帯が短く住む型である確率 piS
   性向を引き継ぐ対照は、newW に前の住み手の性向を返す関数を渡して作る（模型は変えない）。
2. 原状回復の型 restore（既定 None ＝ 何も起きない。ゼロ精算と同じ）
     "R0"   退去する世帯が自分の起こした手（最後に変えたのがその世帯である部位）を始まりの状態に戻し、戻す費用を払う
     "Rdk"  改変は残る。退去する世帯は、自分の起こした手の施工費の未償却分（amort 年の直線）を所有者から受け取る
3. 見通し horizon（既定 None）
     H 年を見通す。各世帯は、自分の満足の変化を「H 年のうち住んでいると見込む年の割合」g＝(1−(1−ν)^H)/(Hν) で割り引いて評価する。
     R0 なら戻す費用を H 年のうちに出る確率で、Rdk なら受け取る見込みの未償却分を、提案者の費用の見込みに入れる。
4. 所有者 owner（既定 None ＝ 所有者なし）
     dict(kind="ind"|"sub"|"mgmt", role="veto"|"trade", beta=, gamma=, M=10, Lam=set(), fix=None|"D4"|"D5")
     ind   個人の所有者。手が建物の市場での評判を上げるかで承諾する：
           V_o ＝ beta·Σ_j ΔΔ_j/μ − gamma·Σ_j Δξ_j
           Δ_j は市場の平均的な新しい住み手（参照の母集団）が住戸 j の今の形を F と比べてどれだけ好むか、ξ_j はそのうち F を好む割合（癖）。
     sub   サブリース。ind と同じ式で、癖の損を M 棟分（gamma×M）に見る
     mgmt  管理会社に委託。一覧 Lam に載る種類の手（"ext"・"store"・"open"）だけを承諾する。価格には反応しない
     role="trade" なら、V_o＜0 のとき提案者が −V_o を所有者に払って承諾を買える（mgmt では効かない）
     fix="D4"  承諾の一覧を住み手が決める（毎年、今の住み手の過半が得をする手を一つでも持つ種類を一覧に載せる）。mgmt に使う
     fix="D5"  目的限定の承諾請求権。所有者は、自分の損が住み手の正味の得を上回るときだけ拒める（V_o＋val ≥ 0 なら承諾）。
               mgmt では、一覧の外の手を、ind と同じ式の V_o で裁く
5. 決める場 venue（既定 None ＝ s1 と同じ。全員に届き、手間はない）
     dict(G=0, q_far=0.4, tau=0.5)   場なし。近い戸（同じ階の隣・同じ列の上下一つ）以外の相手には確率 q_far でしか届かない。
                                     損をする相手が一人でも届かない手は打てない。相手一人あたり tau（㎡）の手間を提案者が払う
     dict(G=1, tauG=0.5)            年に一度の寄り合い。全員に届く。手間は年に tauG を 9 戸で割って払う（判断には効かない）
     dict(G=0.5, alpha=0.5)         区分所有法 44 条型。全員に届くが、決めるのは所有者で、所有者は住み手の正味の得に重み alpha をかけて足す
                                     （V_o＋alpha·val ≥ 0 なら承諾）
"""
from __future__ import annotations
import numpy as np
from trade.s0 import (S_all, MU, W_BAR, legal_moves, all_moves, apply, move_label, move_type, decide, vote, f_design,
                      quantities, phi)
from trade.s1 import KEYS, NEAR, part_of, revealed, getp, move_cost, best_net

TYPE = {"ext": "ext", "take": "store", "shift": "store", "open": "open"}
F0 = f_design()


def _ref_pop(n=300):
    """市場の参照の母集団：平均の性向のまわりのディリクレ（集中度 3）。所有者は個々の住み手の性向を知らず、この分布だけを知る"""
    rng = np.random.default_rng(424242)
    P = rng.dirichlet(W_BAR * 3.0, n)
    return np.clip(P, 1e-9, None) / np.clip(P, 1e-9, None).sum(1, keepdims=True)


POP = _ref_pop()
POPM = POP.mean(0)
PHI_F = phi(*quantities(*F0))


def market(e, k, o):
    """各住戸について (Δ_j, ξ_j)。Δ_j は平均的な新しい住み手が今の形を F と比べて好む度合い（満足）、ξ_j は F を好む割合"""
    D = phi(*quantities(e, k, o)) - PHI_F          # (9, 4)
    V = POP @ D.T                                   # (n, 9)
    return D @ POPM, (V < -1e-12).mean(0)


def g_of(nu, H):
    if H is None:
        return 1.0
    if nu <= 0:
        return 1.0
    return (1 - (1 - nu) ** H) / (H * nu)


def refund_frac(nu, A):
    """幾何分布の居住で、施工費の未償却分を受け取る見込みの割合 E[(1−T/A)⁺]（T は施工から退去までの年）"""
    if nu <= 0:
        return 0.0
    return float(sum(nu * (1 - nu) ** (t - 1) * max(0.0, 1 - t / A) for t in range(1, int(A) + 1)))


def renew_base(nu, rho, horizon=400):
    """更新の山つきの居住で、平均の居住年数が 1/ν になる基準の退去率 h0 を二分法で求める"""
    def mean_T(h0):
        s, surv = 0.0, 1.0
        for a in range(1, horizon + 1):
            h = min(1.0, h0 * (rho if a % 2 == 0 else 1.0))
            s += a * surv * h; surv *= (1 - h)
        return s + horizon * surv
    lo, hi = 1e-6, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if mean_T(mid) > 1 / nu:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run(W, regime, seed, years=30, p_req=0.3, form="linear", pi=0.5, start=None,
        theta=0.0, budget=None, conv_scale=1.0, nu=0.0, newW=None,
        split=False, credit=None, settle=False, reconsent=None,
        protocol="full", ask=3, cost=None, rcost=None, turn_at=None, ref=None,
        turnover=None, restore=None, amort=10, horizon=None, owner=None, venue=None):
    W = np.array(W, float).copy()
    ss = np.random.SeedSequence([seed, 7])
    r_req, r_ord, r_vote = [np.random.default_rng(s) for s in ss.spawn(3)]
    REQ = r_req.random((years, 9))
    ORD = np.array([r_ord.permutation(9) for _ in range(years)])
    VOT = r_vote.random((years, 9, 9))
    r_turn = np.random.default_rng(np.random.SeedSequence([seed, 7, 99]))
    TURN = r_turn.random((years, 9))
    r_null = np.random.default_rng(np.random.SeedSequence([seed, 7, 123]))
    r_type = np.random.default_rng(np.random.SeedSequence([seed, 7, 77]))     # 入れ替わりの型（s1 の乱数とは別）
    r_ven = np.random.default_rng(np.random.SeedSequence([seed, 7, 55]))      # 決める場で相手に届くか
    e, k, o = start if start is not None else f_design(form)
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
    Wt = [S_all(W, e, k, o, form).sum()]
    acc = c if credit is not None else m
    lim = np.full(9, credit) if credit is not None else budget

    # ---- 入れ替わりの型
    kind = turnover["kind"] if turnover else None
    if kind == "geo":
        nu = turnover["nu"]
    if kind == "mix":
        sig = np.where(r_type.random(9) < turnover.get("piS", 0.72), 0, 1)     # 0＝短く住む型, 1＝長く住む型
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

    # ---- 作者の記録（原状回復のため）：部位 → (戸, 住み手の番号, 年)
    author = {}

    # ---- 所有者
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
            S0_ = S_all(W, e, k, o, form)
            for i in range(9):
                for mv in legal_moves(i, e, k, o):
                    if TYPE[mv[0]] != typ:
                        continue
                    e1, k1, o1 = apply(mv, i, e, k, o)
                    if S_all(W, e1, k1, o1, form)[i] - S0_[i] > 1e-12:
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
                S0_ = S_all(W_now, e, k, o, form)
                nx = [e.copy(), k.copy(), o.copy()]; {"e": nx[0], "k": nx[1], "o": nx[2]}[p[0]][p[1]] = v
                dS = S_all(W_now, *nx, form) - S0_
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
                cands = all_moves(e, k, o)
                mv = cands[int(r_null.integers(len(cands)))]
                S0_ = S_all(W, e, k, o, form)
                e1, k1, o1 = apply(mv, i, e, k, o)
                dS = S_all(W, e1, k1, o1, form) - S0_
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
                S0_ = S_all(W, e, k, o, form); C0 = S_all(Wbar, e, k, o, form)
                cands = all_moves(e, k, o) if plus else legal_moves(i, e, k, o)
                tr = base in ("T0", "T1", "T1h")
                reach = None
                if venue and venue.get("G") == 0 and tr:
                    u = r_ven.random(9)
                    reach = np.array([j == i or j in NEAR[i] or u[j] < venue.get("q_far", 0.4) for j in range(9)])
                evals = []
                for mv in cands:
                    e1, k1, o1 = apply(mv, i, e, k, o)
                    dS = S_all(W, e1, k1, o1, form) - S0_
                    dC = S_all(Wbar, e1, k1, o1, form) - C0
                    evals.append((mv, dS, dC, (e1, k1, o1)))
                drafter = "self"
                if regime == "AI+":
                    good = []
                    for x in evals:
                        p = part_of(x[0]); nv = int(getp(x[3], p))
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
                    p = part_of(mv); nv = int(getp(nxt, p))
                    cst = move_cost(p, nv, st, cost, rcost)
                    ant = 0.0                                   # 見通しに入る、後で払う・受け取る見込み
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
                                # 一覧の外でも、所有者の市場での損（ind と同じ式）が住み手の正味の得を上回らなければ拒めない
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
                if best is None:
                    continue
                val, ok, pay, mv, dS, dC, nxt, cst, tc, opay = best
                if val <= 1e-12:
                    if own is not None and ok is False:
                        extra["vetoed"] += 1
                    continue
                nyes = None
                if base in ("V½", "V¾"):
                    ok, nyes = vote(base, i, dS, VOT[t, i], pi)
                harmed = {j for j in range(9) if j != i and dS[j] < -1e-12}
                cons = tr
                if base in ("U", "V½", "V¾"):
                    cons = len(harmed) == 0
                p = part_of(mv)
                ev = dict(t=t, i=int(i), move=move_label(mv, i, e), type=move_type(mv, e), ok=bool(ok), val=float(val),
                          yes=nyes, part=p, nv=int(getp(nxt, p)), dS=dS, pay=pay.copy(), cons=cons,
                          supp=({int(i)} | harmed) if cons else set(), drafter=drafter, cost=cst if ok else 0.0)
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
                    if ev["nv"] == getp(st, p):
                        author.pop(p, None)
                    else:
                        author[p] = (int(i), int(tenant[i]), t)
                    e, k, o = nxt
                events.append(ev)
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
                    # 原状回復：退去する世帯が起こした手
                    mine = [p for p, (u_, a_, _) in author.items() if u_ == i and a_ == tenant[i]]
                    for p in mine:
                        if restore == "R0":
                            s_ = getp(st, p)
                            if getp((e, k, o), p) != s_:
                                nx = [e.copy(), k.copy(), o.copy()]; {"e": nx[0], "k": nx[1], "o": nx[2]}[p[0]][p[1]] = s_
                                dS = S_all(W, *nx, form) - S_all(W, e, k, o, form)
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
                    W[i] = newW(r_turn, i)
                    tenant[i] = nxt_tenant; nxt_tenant += 1; tenure[i] = 0
                    if kind == "mix":
                        sig[i] = 0 if r_type.random() < turnover.get("piS", 0.72) else 1
                    if settle:
                        m[i] = 0.0; c[i] = 0.0; b[i] = 0.0
                    if reconsent:
                        gn = g_of(mean_nu(i), horizon) if horizon is not None else 1.0
                        e, k, o, nr, nv_, sp = _reconsent(i, t, W, e, k, o, st, form, reconsent, acc, lim, recon, rcost, gn,
                                                          author, events)
                        n_recon += nr; n_revert += nv_; spent += sp
                else:
                    tenure[i] += 1
        if changed and regime in ("P", "BRdyn"):
            central(t, W)
        Wt.append(S_all(W, e, k, o, form).sum())
    S_fin = S_all(W, e, k, o, form)
    return dict(e=e, k=k, o=o, S=S_fin, U=S_fin + MU * (m + c), b=b, m=m, c=c, Wt=np.array(Wt), events=events,
                W_final=W, n_exit=n_exit, exit_bal=exit_bal, n_recon=n_recon, n_revert=n_revert, moved=moved,
                quotes=quotes, turns=turns, recon=recon, spent=spent, start=X0, ref=st, extra=extra)


def _reconsent(n, t, W, e, k, o, st, form, scope, acc, lim, recon, rcost, gn=1.0, author=None, events=None):
    """〔再〕入居した世帯 n の再同意（s1._reconsent と同じ規則。見通しがあれば満足の変化を g で割り引く）"""
    parts = [("o", x) for x in range(6)] + [("k", x) for x in range(6)]
    if scope == "all":
        parts += [("e", x) for x in range(9)]
    nr = nv = 0; sp = 0.0
    for p in parts:
        cur = getp((e, k, o), p); s_ = getp(st, p)
        if cur == s_:
            continue
        nx = [e.copy(), k.copy(), o.copy()]; {"e": nx[0], "k": nx[1], "o": nx[2]}[p[0]][p[1]] = s_
        dS = S_all(W, *nx, form) - S_all(W, e, k, o, form)
        rc = 0.0 if rcost is None else float(rcost.get(p[0], 0.0))
        G = gn * dS[n] / MU - rc
        if G <= 1e-12:
            recon.append((t, n, p, "accept"))
            continue
        nr += 1
        L = np.maximum(0.0, -gn * dS / MU) if gn != 1.0 else np.maximum(0.0, -dS / MU); L[n] = 0.0
        keep = False
        if L.sum() > G + 1e-12:
            s = G * L / L.sum()
            if lim is None or all(acc[j] - s[j] >= -lim[j] - 1e-9 for j in range(9) if s[j] > 0):
                keep = True
                for j in range(9):
                    acc[j] -= s[j]
                acc[n] += G
        recon.append((t, n, p, "paid" if keep else "revert"))
        if not keep:
            e, k, o = nx; nv += 1; acc[n] -= rc; sp += rc
            if author is not None:
                author.pop(p, None)
    return e, k, o, nr, nv, sp
