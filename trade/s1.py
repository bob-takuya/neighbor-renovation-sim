"""段 2 のための模型 s1。s0（段 1 の本走行で凍結）は変えず、写して次を足した。

1. 記録：手ごとの部位・新しい値・各戸の満足の変化・支払い・同意の有無と支える世帯・起案者、
   言い値の問い合わせ（年, 聞いた戸, 答えた戸, 明かした係数）、入れ替わり、再同意の結果、費用。
2. 聞く範囲 protocol：
     "full"  提案者は盤のすべての手について、損をする全員の言い値を聞いてから最良を選ぶ（s0 と同じ）
     "topk"  提案者は自分の得の大きい順に手を並べ、上から ask 個まで、損をする戸にだけ聞く。打てる手が見つかればそこでやめる
     "near"  提案者は、影響（満足の変化）がすべて自戸と近い戸（同じ階の隣・同じ列の上下一つ）に収まる手だけを考え、その戸にだけ聞く
3. 改変の費用：cost＝{"e","k","o"}（始まりの状態から変える施工費、㎡ 換算）、rcost＝同（始まりの状態へ戻す費用）。
   手を打った提案者が払う（お金 m か、〔貸〕なら c から）。再同意で戻すときは、戻したい新しい世帯が払う。
4. 帰無と偽装の決め方：
     "R"     N4：要求のたびにランダムな手を打つ（償いなし）
     "Rc"    N4c：ランダムな手に、損をする戸へ言い値を払う形だけの取引を付ける
     "P"     N3：全知の計画者が、始まりと入れ替わりのたびに、今の住み手にとっての最良（費用込み）へ組み替える
     "BR"    全知の仲介者が、始まりの年に最良（費用込み）への手を一つずつ、損をする全員に言い値をちょうど払う形で集める
     "BRdyn" BR を入れ替わりのたびにくり返す
     "AI+"   性向を知る起案 AI が、要求した世帯に「この一手だけ聞けばよい」と教える TP+（提案者は AI の示した手の損をする戸にだけ聞く）
   N2（全知の最適から始める）は start で、N6（わざと悪い盤から始める）も start で与える。
既定の引数では s0.run と同じ乱数を同じ順に使い、最終の形・満足・お金・貸し借り・帳簿が一致する（tests/test_s1.py）。
"""
from __future__ import annotations
import numpy as np
from trade.s0 import (S_all, MU, UNITS, W_BAR, FLOOR, COL, legal_moves, all_moves, apply, move_label, move_type,
                      decide, vote, f_design, b_units, wstar_linear)

KEYS = [("e", x) for x in range(9)] + [("k", x) for x in range(6)] + [("o", x) for x in range(6)]
NEAR = [{j for j in range(9) if j != i and ((FLOOR[j] == FLOOR[i] and abs(COL[j] - COL[i]) == 1) or
                                         (COL[j] == COL[i] and abs(FLOOR[j] - FLOOR[i]) == 1))} for i in range(9)]


def part_of(mv):
    kind, x = mv[0], mv[1]
    return ("e", x) if kind == "ext" else (("o", x) if kind == "open" else ("k", x))


def revealed(mv, j):
    """戸 j が損の言い値を明かしたとき、どの係数が分かるか（a＝広さ, s＝日照, q＝静けさ−近さ）"""
    kind, x = mv[0], mv[1]
    if kind == "ext":
        return "a" if j == x else "s"
    if kind == "open":
        return "q"
    return "a"


def getp(cfg, p):
    return {"e": cfg[0], "k": cfg[1], "o": cfg[2]}[p[0]][p[1]]


def move_cost(p, new, start, cost, rcost):
    if p is None:
        return 0.0
    if new == getp(start, p):
        return 0.0 if rcost is None else float(rcost.get(p[0], 0.0))
    return 0.0 if cost is None else float(cost.get(p[0], 0.0))


def best_net(W, e, k, o, start, cost=None, rcost=None, form="linear"):
    """全知の組み替え：今の盤から、費用を差し引いた満足の合計が最大になる盤へ。線形の満足では部位ごとに分かれるので、
    部位ごとに値を選ぶ（cost も rcost もなければ wstar_linear と同じ）。返り値 (e, k, o, 払う費用の合計)"""
    if cost is None and rcost is None and form == "linear":
        _, ez, kz, oz = wstar_linear(W)
        return ez, kz, oz, 0.0
    cur = [e.copy(), k.copy(), o.copy()]
    base = S_all(W, *cur, form).sum()
    tot = 0.0
    for p in KEYS:
        vals = (0, 1) if p[0] in "eo" else (-1, 0, 1)
        v0 = getp(cur, p)
        best_v, best_g = v0, 0.0
        for v in vals:
            if v == v0:
                continue
            nx = [a.copy() for a in cur]; {"e": nx[0], "k": nx[1], "o": nx[2]}[p[0]][p[1]] = v
            g = S_all(W, *nx, form).sum() - base - MU * move_cost(p, v, start, cost, rcost)
            if g > best_g + 1e-12:
                best_v, best_g = v, g
        if best_v != v0:
            {"e": cur[0], "k": cur[1], "o": cur[2]}[p[0]][p[1]] = best_v
            tot += move_cost(p, best_v, start, cost, rcost)
            base = S_all(W, *cur, form).sum()
    return cur[0], cur[1], cur[2], tot


def run(W, regime, seed, years=30, p_req=0.3, form="linear", pi=0.5, start=None,
        theta=0.0, budget=None, conv_scale=1.0, nu=0.0, newW=None,
        split=False, credit=None, settle=False, reconsent=None,
        protocol="full", ask=3, cost=None, rcost=None, turn_at=None, ref=None):
    W = np.array(W, float).copy()
    ss = np.random.SeedSequence([seed, 7])
    r_req, r_ord, r_vote = [np.random.default_rng(s) for s in ss.spawn(3)]
    REQ = r_req.random((years, 9))
    ORD = np.array([r_ord.permutation(9) for _ in range(years)])
    VOT = r_vote.random((years, 9, 9))
    r_turn = np.random.default_rng(np.random.SeedSequence([seed, 7, 99]))
    TURN = r_turn.random((years, 9))
    r_null = np.random.default_rng(np.random.SeedSequence([seed, 7, 123]))
    e, k, o = start if start is not None else f_design(form)
    e, k, o = e.copy(), k.copy(), o.copy()
    X0 = (e.copy(), k.copy(), o.copy())
    st = (e.copy(), k.copy(), o.copy()) if ref is None else tuple(a.copy() for a in ref)   # 費用と再同意の基準＝始まりの状態
    c = np.zeros(9); m = np.zeros(9); b = np.zeros(9)
    n_recon = n_revert = 0; n_exit = 0; exit_bal = 0.0
    moved = np.zeros(9, bool)
    Wbar = np.tile(W_BAR * np.array([1.0, conv_scale, conv_scale, conv_scale]), (9, 1))
    events, quotes, turns, recon = [], [], [], []
    spent = 0.0
    Wt = [S_all(W, e, k, o, form).sum()]
    acc = c if credit is not None else m
    lim = np.full(9, credit) if credit is not None else budget

    def central(t, W_now):
        nonlocal e, k, o, spent
        if regime == "P":
            e2, k2, o2, cst = best_net(W_now, e, k, o, st, cost, rcost, form)
            # 計画者は、盤を変えなくても、そのたびに全員の性向を見ている（知識の記録のため、変えなくても記録する）
            events.append(dict(t=t, i=None, part=None, ok=True, planner=True, cons=False, supp=set(), drafter="planner",
                               changed=bool(any((a != b_).any() for a, b_ in zip((e2, k2, o2), (e, k, o))))))
            e, k, o = e2, k2, o2; spent += cst
        else:   # BR・BRdyn：仲介者が部位ごとに、損をする全員へ言い値をちょうど払う形で同意を集める
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
                e, k, o = nx

    if regime in ("P", "BR", "BRdyn"):
        central(-1, W)
    for t in range(years):
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
                e, k, o = e1, k1, o1
        elif regime not in ("F", "P", "BR", "BRdyn"):
            plus = regime.endswith("+")
            base = regime.rstrip("+")
            if base == "AI":
                base = "T1"
            for i in ORD[t]:
                if REQ[t, i] >= p_req:
                    continue
                S0_ = S_all(W, e, k, o, form); C0 = S_all(Wbar, e, k, o, form)
                cands = all_moves(e, k, o) if plus else legal_moves(i, e, k, o)
                evals = []
                for mv in cands:
                    e1, k1, o1 = apply(mv, i, e, k, o)
                    dS = S_all(W, e1, k1, o1, form) - S0_
                    dC = S_all(Wbar, e1, k1, o1, form) - C0
                    evals.append((mv, dS, dC, (e1, k1, o1)))
                tr = base in ("T0", "T1", "T1h")
                drafter = "self"
                if regime == "AI+":
                    # 起案 AI：全員の性向を知り、提案者が言い値を払っても得をする手のうち、満足の合計を最も上げる一手を示す
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
                    evals = [x for x in evals if x[1][i] > 1e-12]
                    evals.sort(key=lambda x: -x[1][i])
                    evals = evals[:ask]
                elif protocol == "near" and tr:
                    evals = [x for x in evals if {j for j in range(9) if j != i and abs(x[1][j]) > 1e-12} <= NEAR[i]]
                best = None
                for mv, dS, dC, nxt in evals:
                    val, ok, pay = decide(base, i, dS, dC, None, pi, theta, split)
                    p = part_of(mv); nv = int(getp(nxt, p))
                    cst = move_cost(p, nv, st, cost, rcost)
                    val = val - cst
                    if tr:
                        for j in range(9):
                            if j != i and dS[j] < -1e-12:
                                quotes.append((t, int(i), j, revealed(mv, j)))
                    if ok and lim is not None:
                        out_i = max(0.0, pay.sum()) + cst
                        if acc[i] - out_i < -lim[i] - 1e-9:
                            ok = False
                        for j in range(9):
                            if pay[j] < 0 and acc[j] + pay[j] < -lim[j] - 1e-9:
                                ok = False
                        if not ok:
                            val = -1.0
                    if best is None or val > best[0]:
                        best = (val, ok, pay, mv, dS, dC, nxt, cst)
                    if protocol == "topk" and tr and ok and val > 1e-12:
                        break
                if best is None:
                    continue
                val, ok, pay, mv, dS, dC, nxt, cst = best
                if val <= 1e-12:
                    continue
                nyes = None
                if base in ("V½", "V¾"):
                    ok, nyes = vote(base, i, dS, VOT[t, i], pi)
                harmed = {j for j in range(9) if j != i and dS[j] < -1e-12}
                cons = tr if base != "T0" else tr      # 取引では損をする全員が言い値以上を受け取るか断れる
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
                    e, k, o = nxt
                events.append(ev)
        changed = False
        if (nu > 0 or turn_at) and newW is not None:
            for i in range(9):
                if (turn_at is not None and i in turn_at.get(t, ())) or (turn_at is None and TURN[t, i] < nu):
                    n_exit += 1; exit_bal += abs(b[i]); moved[i] = True
                    turns.append((t, i)); changed = True
                    W[i] = newW(r_turn, i)
                    if settle:
                        m[i] = 0.0; c[i] = 0.0; b[i] = 0.0
                    if reconsent:
                        e, k, o, nr, nv_, sp = _reconsent(i, t, W, e, k, o, st, form, reconsent, acc, lim, recon, rcost)
                        n_recon += nr; n_revert += nv_; spent += sp
        if changed and regime in ("P", "BRdyn"):
            central(t, W)
        Wt.append(S_all(W, e, k, o, form).sum())
    S_fin = S_all(W, e, k, o, form)
    return dict(e=e, k=k, o=o, S=S_fin, U=S_fin + MU * (m + c), b=b, m=m, c=c, Wt=np.array(Wt), events=events,
                W_final=W, n_exit=n_exit, exit_bal=exit_bal, n_recon=n_recon, n_revert=n_revert, moved=moved,
                quotes=quotes, turns=turns, recon=recon, spent=spent, start=X0, ref=st)


def _reconsent(n, t, W, e, k, o, st, form, scope, acc, lim, recon, rcost):
    """〔再〕入居した世帯 n の再同意（s0._reconsent と同じ規則に、記録と戻す費用を足した）。
    n が戻して得る分 G（戻す費用を差し引く）が正なら、残したい戸が G を損の大きさに比例して割り勘で払えば残り、払えなければ戻る"""
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
        G = dS[n] / MU - rc
        if G <= 1e-12:
            recon.append((t, n, p, "accept"))
            continue
        nr += 1
        L = np.maximum(0.0, -dS / MU); L[n] = 0.0
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
    return e, k, o, nr, nv, sp
