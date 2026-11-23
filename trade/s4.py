"""確証の実験のための模型 s4。s3（観察 2 で凍結）を import し、変えずに、話し合いを経ない二つの囮の決め方を足した。

囮は、言い値の取引 TP+（s0.decide の "T1"）と同じ言い値・同じ支払い・同じ手の選び方を使う。
違うのは「誰が誰の言い値を知るか」だけである。そのため、同じ構成・同じ局の種なら、囮の手の並び・形・お金は
TP+（protocol が同じもの）と一致する（tests/test_s4.py で確かめる）。

  "M+"    市場型 M（匿名の清算所）
          損をする戸は、損の額（言い値）を清算所に出す。提案者は、検討する手ごとに、清算所から合計の額だけを知る。
          提案者は、自分の得（費用を引いたもの）が合計を上回る手のうち正味の得が最大の手を打ち、合計を清算所に払う。
          損をする戸は、清算所からそれぞれ自分の言い値を受け取る（損をする側の同意あり、cons＝True）。
          提案者が知る事実：合計から一意に分かるとき、すなわち損をする戸が一戸だけの手では、その戸の係数を知る。それ以外は知らない。
          清算所は、出された言い値をすべて個別に知る。記録 ch_quotes＝(年, 戸, 明かした係数)。
          清算所を作った者（作者）に数えるかどうかは推定器（trade/vstar2.py）で切り替える。
  "Rep+"  代表型 Rep（代表に決めてもらう）
          局の始めに、住み手の中から代表を一人ランダムに選ぶ（専用の乱数。他の乱数の並びは変えない）。
          各年、改変を求める世帯は、要求（検討する手ごとの自分の得）を代表に伝える。代表は損をする戸の言い値を聞き、
          言い値で測った満足の合計（要求した世帯の得 − 損をする戸の言い値の合計 − 費用）が最大で正の手を決める。
          要求した世帯が言い値の合計を払い、損をする戸はそれぞれ自分の言い値を受け取る（cons＝True）。
          代表の知識：要求した世帯の得で明かされた係数と、聞いた言い値で明かされた係数（記録 quotes の聞いた戸＝代表の住戸）。
          要求した世帯の知識：打った手の損をする戸が一戸だけなら、払った合計からその戸の係数を知る。
          代表が入れ替わりで出ていった年は、その年の入れ替わりがすべて済んだ後、その時の住み手 9 戸から次の代表をランダムに選ぶ。
          rep_term＝n を与えると、n 年ごとにも代表を選び直す（選ぶ用での確かめにだけ使う。既定 None は選び直さない）。
          代表は今の住み手なので、推定器では今の住み手（去れば去った住み手）として作者に入る。
  聞く範囲 protocol は "full"（盤のすべての手、TP+ と同じ）と "topk"（自分の得の大きい順に上から ask 手、TP+top3 と同じ）。

既定の設定（囮でない決め方）では、s3.run をそのまま呼ぶので、s3・s1 と同じ乱数を同じ順に使い、同じ記録を返す。
"""
from __future__ import annotations
import numpy as np
from trade import s3
from trade.s0 import MU as MU_, S_all, W_BAR, all_moves, apply, move_label, move_type, decide, f_design
from trade.s1 import part_of, revealed, getp, move_cost, _reconsent

DECOYS = ("M+", "Rep+")


def run(W, regime, seed, **kw):
    if regime not in DECOYS:
        return s3.run(W, regime, seed, **kw)
    return run_decoy(W, regime, seed, **kw)


def run_decoy(W, regime, seed, years=30, p_req=0.3, form="linear", pi=0.5, start=None,
              theta=0.0, budget=None, conv_scale=1.0, nu=0.0, newW=None,
              split=False, credit=None, settle=False, reconsent=None,
              protocol="full", ask=3, cost=None, rcost=None, turn_at=None, ref=None, rep_term=None):
    if protocol not in ("full", "topk"):
        raise ValueError("囮の聞く範囲は full と topk だけ")
    if split:
        raise ValueError("囮に〔割〕は付けない")
    W = np.array(W, float).copy()
    # ---- s1・s3 と同じ乱数（同じ順に作る）
    ss = np.random.SeedSequence([seed, 7])
    r_req, r_ord, r_vote = [np.random.default_rng(s) for s in ss.spawn(3)]
    REQ = r_req.random((years, 9))
    ORD = np.array([r_ord.permutation(9) for _ in range(years)])
    VOT = r_vote.random((years, 9, 9))          # 使わないが、s1 と同じに作っておく
    r_turn = np.random.default_rng(np.random.SeedSequence([seed, 7, 99]))
    TURN = r_turn.random((years, 9))
    r_rep = np.random.default_rng(np.random.SeedSequence([seed, 7, 321]))    # 代表を選ぶ乱数（囮だけが使う）
    e, k, o = start if start is not None else f_design(form)
    e, k, o = e.copy(), k.copy(), o.copy()
    X0 = (e.copy(), k.copy(), o.copy())
    st = (e.copy(), k.copy(), o.copy()) if ref is None else tuple(a.copy() for a in ref)
    c = np.zeros(9); m = np.zeros(9); b = np.zeros(9)
    n_recon = n_revert = 0; n_exit = 0; exit_bal = 0.0
    moved = np.zeros(9, bool)
    Wbar = np.tile(W_BAR * np.array([1.0, conv_scale, conv_scale, conv_scale]), (9, 1))
    events, quotes, turns, recon = [], [], [], []
    ch_quotes = []                     # M：清算所が個別に知った言い値 (年, 戸, 係数)
    reps = []                          # Rep：(年, 代表の住戸)。年 −1 は局の始め
    spent = 0.0
    Wt = [S_all(W, e, k, o, form).sum()]
    acc = c if credit is not None else m
    lim = np.full(9, credit) if credit is not None else budget
    rep = int(r_rep.integers(9)) if regime == "Rep+" else None
    if rep is not None:
        reps.append((-1, rep))

    for t in range(years):
        for i in ORD[t]:
            if REQ[t, i] >= p_req:
                continue
            S0_ = S_all(W, e, k, o, form); C0 = S_all(Wbar, e, k, o, form)
            evals = []
            for mv in all_moves(e, k, o):
                e1, k1, o1 = apply(mv, i, e, k, o)
                dS = S_all(W, e1, k1, o1, form) - S0_
                dC = S_all(Wbar, e1, k1, o1, form) - C0
                evals.append((mv, dS, dC, (e1, k1, o1)))
            if protocol == "topk":
                evals = [x for x in evals if x[1][i] > 1e-12]
                evals.sort(key=lambda x: -x[1][i])
                evals = evals[:ask]
            best = None
            for mv, dS, dC, nxt in evals:
                val, ok, pay = decide("T1", i, dS, dC, None, pi, theta, False)
                p = part_of(mv); nv = int(getp(nxt, p))
                cst = move_cost(p, nv, st, cost, rcost)
                val = val - cst
                harmed = [j for j in range(9) if j != i and dS[j] < -1e-12]
                # ---- 誰が誰の言い値を知るか（囮で違うのはここだけ）
                if regime == "M+":
                    for j in harmed:
                        ch_quotes.append((t, j, revealed(mv, j)))           # 清算所は個別に知る
                    if len(harmed) == 1:                                    # 合計から一意に分かる
                        quotes.append((t, int(i), harmed[0], revealed(mv, harmed[0])))
                else:   # Rep+
                    if rep != i and abs(dS[i]) > 1e-12:                     # 要求した世帯の得を代表が聞く
                        quotes.append((t, rep, int(i), revealed(mv, int(i))))
                    for j in harmed:
                        if j != rep:
                            quotes.append((t, rep, j, revealed(mv, j)))     # 損をする戸の言い値を代表が聞く
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
                if protocol == "topk" and ok and val > 1e-12:
                    break
            if best is None:
                continue
            val, ok, pay, mv, dS, dC, nxt, cst = best
            if val <= 1e-12:
                continue
            harmed = {j for j in range(9) if j != i and dS[j] < -1e-12}
            if regime == "Rep+" and ok and len(harmed) == 1:
                j = next(iter(harmed))
                quotes.append((t, int(i), j, revealed(mv, j)))              # 払った合計から一意に分かる
            p = part_of(mv)
            ev = dict(t=t, i=int(i), move=move_label(mv, i, e), type=move_type(mv, e), ok=bool(ok), val=float(val),
                      yes=None, part=p, nv=int(getp(nxt, p)), dS=dS, pay=pay.copy(), cons=True,
                      supp={int(i)} | harmed, drafter="self" if regime == "M+" else "rep", rep=rep,
                      cost=cst if ok else 0.0)
            if ok:
                for j in range(9):
                    if j != i:
                        b[j] -= dC[j] / MU_
                        b[i] += dC[j] / MU_
                for j in range(9):
                    if pay[j] != 0:
                        b[i] += pay[j]; b[j] -= pay[j]
                        acc[i] -= pay[j]; acc[j] += pay[j]
                acc[i] -= cst; spent += cst
                e, k, o = nxt
            events.append(ev)
        rep_left = False
        if (nu > 0 or turn_at) and newW is not None:
            for i in range(9):
                if (turn_at is not None and i in turn_at.get(t, ())) or (turn_at is None and TURN[t, i] < nu):
                    n_exit += 1; exit_bal += abs(b[i]); moved[i] = True
                    turns.append((t, i))
                    W[i] = newW(r_turn, i)
                    if i == rep:
                        rep_left = True
                    if settle:
                        m[i] = 0.0; c[i] = 0.0; b[i] = 0.0
                    if reconsent:
                        e, k, o, nr, nv_, sp = _reconsent(i, t, W, e, k, o, st, form, reconsent, acc, lim, recon, rcost)
                        n_recon += nr; n_revert += nv_; spent += sp
        if rep_left or (rep is not None and rep_term is not None and (t + 1) % rep_term == 0):
            rep = int(r_rep.integers(9)); reps.append((t, rep))
        Wt.append(S_all(W, e, k, o, form).sum())
    S_fin = S_all(W, e, k, o, form)
    return dict(e=e, k=k, o=o, S=S_fin, U=S_fin + MU_ * (m + c), b=b, m=m, c=c, Wt=np.array(Wt), events=events,
                W_final=W, n_exit=n_exit, exit_bal=exit_bal, n_recon=n_recon, n_revert=n_revert, moved=moved,
                quotes=quotes, turns=turns, recon=recon, spent=spent, start=X0, ref=st,
                decoy=regime, ch_quotes=ch_quotes, reps=reps)

