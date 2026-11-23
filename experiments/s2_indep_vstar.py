"""分散知 V_H・V*・V*n を定義から計算し直す別の実装（trade/vstar.py との突き合わせ用）。runs.pkl を読んで値を出す。"""
import csv, math, pickle, sys
from collections import Counter
import numpy as np
from trade import s0, s1

D = "out/s2/"
KEYS = s1.KEYS
WBAR = s0.W_BAR
DEF = {"a": WBAR[0], "s": WBAR[1], "q": WBAR[2] - WBAR[3]}       # 0.35, 0.30, 0.05
CENTRAL_DRAFTERS = {"planner", "broker", "ai"}
EPS = 1e-12


# ------------------------------------------------------------ 情報量（plug-in ＋ Miller–Madow、ビット）
def H_mm(xs):
    n = len(xs)
    c = Counter(xs)
    h = -sum(v / n * math.log2(v / n) for v in c.values())
    return h + (len(c) - 1) / (2 * n * math.log(2))


def I_mm(xs, ys):
    return max(0.0, H_mm(xs) + H_mm(ys) - H_mm(list(zip(xs, ys))))


def vec(cfg):
    e, k, o = cfg
    return tuple(int(s1.getp((e, k, o), p)) for p in KEYS)


# ------------------------------------------------------------ 知識から推定した最適
def coef_true(W, j, c):
    return {"a": W[j, 0], "s": W[j, 1], "q": W[j, 2] - W[j, 3]}[c]


def zhat(W_final, known):
    """known：(住戸, 係数) の集合（30 年目に有効なもの）。知らない係数は平均の性向から"""
    Wh = np.zeros((9, 4))
    for j in range(9):
        v = {c: (coef_true(W_final, j, c) if (j, c) in known else DEF[c]) for c in "asq"}
        r = max(1e-9, 1 - v["a"] - v["s"])
        Wh[j] = (v["a"], v["s"], (r + v["q"]) / 2, (r - v["q"]) / 2)
    _, e, k, o = s0.wstar_linear(Wh)
    return vec((e, k, o))


# ------------------------------------------------------------ 作者ごとの知識
def authors(rec, cond):
    """返り値：dict 枠名 → 30 年目に有効な (住戸, 係数) の集合。None は「何も知らない／F の形」"""
    turns = rec["turns"]
    tt = {i: sorted(t for t, u in turns if u == i) for i in range(9)}

    def valid(j, t):            # 年 t の住戸 j の住み手についての事実が 30 年目に有効か（入れ替わりは年の手の後）
        return all(tu < t for tu in tt[j])

    def tenant_idx(i, t):       # 年 t に住戸 i に住んでいた住み手の番号（0＝初めの住み手）
        return sum(1 for tu in tt[i] if tu < t)

    # 住み手（住戸, 番号）ごとの聞いた事実
    heard = {}
    for (t, i, j, c) in rec["quotes"]:
        key = (i, tenant_idx(i, t))
        if valid(j, t):
            heard.setdefault(key, set()).add((j, c))
    A = {}
    for i in range(9):          # 今の住み手 9 戸
        cur = (i, len(tt[i]))
        A[f"res{i}"] = {(i, c) for c in "asq"} | heard.get(cur, set())
    # 中央の作者
    cen = set()
    has_central = cond in ("N2", "N3", "BR", "AI")
    know_years = set()
    for ev in rec["events"]:    # ok=False の「見た」記録（新しい s1 の BR）も含める
        if ev.get("drafter") in CENTRAL_DRAFTERS:
            know_years.add(ev["t"])
    if cond in ("N2", "N3", "BR"):   # 始まりの盤を置く／見る中央の作者（年 −1、初めの住み手全員）
        know_years.add(-1)
    if cond == "N3":            # 計画者は入れ替わりのたびに全員の係数を知る（古い s1 の記録でも）
        know_years |= {t for t, _ in turns}
    for t in know_years:
        cen |= {(j, c) for j in range(9) for c in "asq" if valid(j, t)}
    A["central"] = cen if has_central else None        # None → F の形
    # 去った住み手：有効な知識を最も多く持つ一人（同数なら先に去った方）
    best, bn = set(), -1
    for (t, i) in sorted(turns):
        r = tenant_idx(i, t)    # 年 t に去った住み手の番号
        kn = heard.get((i, r), set())   # 自分の係数は入れ替わりで無効
        if len(kn) > bn:
            best, bn = kn, len(kn)
    A["departed"] = best        # 去った住み手がいなければ空＝何も知らない
    return A


# ------------------------------------------------------------ 同意で支えられた部位
def supported_cfg(rec):
    """返り値 (X̃, X̃n)：部位ごとの値の組"""
    ref = rec["ref"]; refv = vec(ref)
    XT = (rec["e"], rec["k"], rec["o"]); xv = vec(XT)
    W = rec["W_final"]
    recd = {p: None for p in KEYS}          # p → (cons, supp)

    def do_event(ev):
        if not ev.get("ok", False):
            return
        if ev.get("planner"):
            for p in KEYS:
                recd[p] = None
            return
        p = tuple(ev["part"])
        if ev["nv"] == s1.getp(ref, p):
            recd[p] = None
        else:
            recd[p] = (bool(ev["cons"]), set(ev["supp"]) if ev["cons"] else set())

    evs = rec["events"]; turns = rec["turns"]; rc = rec["recon"]
    late = lambda ev: ev.get("drafter") in ("planner", "broker") and ev["t"] >= 0   # 入れ替わりの後の組み替え
    for ev in evs:
        if ev["t"] < 0:
            do_event(ev)
    for t in range(30):
        for ev in evs:
            if ev["t"] == t and not late(ev):
                do_event(ev)
        for (tu, i) in turns:
            if tu != t:
                continue
            for p in KEYS:
                if recd[p] is not None:
                    recd[p][1].discard(i)
            for (tr, n, p, res) in rc:
                if tr == t and n == i:
                    p = tuple(p)
                    if res == "revert":
                        recd[p] = None
                    elif recd[p] is not None:
                        recd[p][1].add(n)
        for ev in evs:
            if ev["t"] == t and late(ev):
                do_event(ev)
    S0 = s0.S_all(W, *XT)
    xs, xn = list(refv), list(refv)
    for idx, p in enumerate(KEYS):
        if xv[idx] == refv[idx] or recd[p] is None or not recd[p][0]:
            continue
        nx = [a.copy() for a in XT]; {"e": nx[0], "k": nx[1], "o": nx[2]}[p[0]][p[1]] = s1.getp(ref, p)
        dS = s0.S_all(W, *nx) - S0
        gain = {j for j in range(9) if dS[j] > EPS}
        if gain <= recd[p][1]:
            xs[idx] = xv[idx]
            if gain:
                xn[idx] = xv[idx]
    return tuple(xs), tuple(xn)


# ------------------------------------------------------------ 集団での V
def evaluate(recs, cond):
    n = len(recs)
    Z = []; X = []; Xs = []; Xn = []; AU = {}
    fz = vec(s0.f_design())
    for rec in recs:
        _, e, k, o = s0.wstar_linear(rec["W_final"])
        Z.append(vec((e, k, o)))
        X.append(vec((rec["e"], rec["k"], rec["o"])))
        a, b = supported_cfg(rec); Xs.append(a); Xn.append(b)
        for name, kn in authors(rec, cond).items():
            AU.setdefault(name, []).append(fz if kn is None else zhat(rec["W_final"], kn))
    col = lambda M, p: [m[p] for m in M]
    P = range(len(KEYS))
    HZ = sum(H_mm(col(Z, p)) for p in P)
    IX = sum(I_mm(col(Z, p), col(X, p)) for p in P)
    IXs = sum(I_mm(col(Z, p), col(Xs, p)) for p in P)
    IXn = sum(I_mm(col(Z, p), col(Xn, p)) for p in P)
    IA = {a: sum(I_mm(col(Z, p), col(Za, p)) for p in P) for a, Za in AU.items()}
    amax = max(IA, key=IA.get); Imax = IA[amax]
    ns = sum(1 for r in range(n) for p in P if Xs[r][p] != vec(recs[r]["ref"])[p])
    nn = sum(1 for r in range(n) for p in P if Xn[r][p] != vec(recs[r]["ref"])[p])
    nd = sum(1 for r in range(n) for p in P if X[r][p] != vec(recs[r]["ref"])[p])
    return dict(HZ=HZ, IX=IX / HZ, IXs=IXs / HZ, IXn=IXn / HZ, IA=Imax / HZ, amax=amax,
                VH=max(0.0, IX - Imax) / HZ, Vs=max(0.0, IXs - Imax) / HZ, Vn=max(0.0, IXn - Imax) / HZ,
                IA_res=max(IA[f"res{i}"] for i in range(9)) / HZ, IA_cen=IA["central"] / HZ,
                IA_dep=IA["departed"] / HZ, chg=nd / n, chg_s=ns / n, chg_n=nn / n)


if __name__ == "__main__":
    with open(D + "runs.pkl", "rb") as f:
        runs = pickle.load(f)
    rows = []
    for (cond, pat), recs in runs.items():
        r = evaluate(recs, cond)
        r.update(cond=cond, pattern=pat); rows.append(r)
        print(f"{cond:10s} {pat:4s} VH={r['VH']:.3f} V*={r['Vs']:.3f} V*n={r['Vn']:.3f} | "
              f"I(Z;X)={r['IX']:.3f} I(Z;X~)={r['IXs']:.3f} I(Z;X~n)={r['IXn']:.3f} maxA={r['IA']:.3f}({r['amax']}) "
              f"res={r['IA_res']:.3f} cen={r['IA_cen']:.3f} dep={r['IA_dep']:.3f} ΣH={r['HZ']:.2f} "
              f"chg={r['chg']:.1f}/{r['chg_s']:.1f}/{r['chg_n']:.1f}")
    cols = ["cond", "pattern", "VH", "Vs", "Vn", "IX", "IXs", "IXn", "IA", "amax", "IA_res", "IA_cen", "IA_dep",
            "HZ", "chg", "chg_s", "chg_n"]
    with open(D + "indep_values.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r[c] if isinstance(r[c], str) else f"{r[c]:.4f}" for c in cols])
