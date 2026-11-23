"""段 2 の評価の候補を、s1.run の記録から計算する。

一局ごとに作るもの（record）：
  Z      30 年目の住み手にとっての全知の最適（費用なし）の部位の値（21）
  X      30 年目の形
  Xc     同意で支えられた部位だけ X、それ以外は始まりの値（V* 用）
  Xn     損をする側の同意で支えられた部位だけ X、それ以外は始まりの値（V*n 用）
  G      作者ごとの推定 Ẑ^a（今の住み手 9 戸の枠・中央の作者・去った作者のうち最も多く知る者）
  ほか   手の数・問い合わせの数・Λc・Λn・権利の量 Vp・誤配・記録のエントロピー・η など
集団の量（population）：
  V_H ＝ [Σ_p I(Z_p; X_p) − max_a Σ_p I(Z_p; Ẑ^a_p)]⁺ ／ Σ_p H(Z_p)
  V*  ＝ 同じ式で X を Xc に替えたもの、V*n ＝ Xn に替えたもの
相互情報量は構成の集団についての数え上げ（Miller–Madow の補正つき）。
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from trade import s0
from trade.s1 import KEYS, getp

WB = s0.W_BAR
PB = dict(a=WB[0], s=WB[1], q=WB[2] - WB[3])


def cfgvec(e, k, o):
    return np.concatenate([e, k, o]).astype(int)


def guess(W, known):
    """known：分かっている (戸, 係数) の集合。知らない係数には平均の性向を入れて解いた最適"""
    G = np.zeros((9, 4))
    for j in range(9):
        a = W[j, 0] if (j, "a") in known else PB["a"]
        s = W[j, 1] if (j, "s") in known else PB["s"]
        q = W[j, 2] - W[j, 3] if (j, "q") in known else PB["q"]
        r = max(1e-9, 1 - a - s)
        G[j] = [a, s, (r + q) / 2, (r - q) / 2]
    _, e, k, o = s0.wstar_linear(G)
    return cfgvec(e, k, o)


ALLF = [(j, p) for j in range(9) for p in "asq"]


def knowledge(r, start_author=None):
    """作者ごとの、30 年目に有効な知識。事実は (戸, 係数, その戸の住み手の番号) で持ち、
    住み手が入れ替わった戸についての事実は無効になる。返り値：(今の住み手 9 戸の知識, 中央の作者の知識 or None, 去った作者のうち最大の知識)"""
    tenant = list(range(9)); nxt = 9
    facts = {a: {(j, p, a) for p in "asq"} for a in range(9) for j in [a]}   # 住み手 a（番号）は自分の係数を知る
    central = None
    if start_author == "planner":
        central = {(j, p, j) for j, p in ALLF}
    timeline = [(q[0], 0, q) for q in r["quotes"]] + [(t, 1, u) for (t, u) in r["turns"]]
    # 中央の作者が手を打った年：起案 AI はその年の入れ替わりの前、計画者・仲介者は入れ替わりの後に動く
    cent_ai = sorted({ev["t"] for ev in r["events"] if ev.get("drafter") == "ai"})
    cent_pl = sorted({ev["t"] for ev in r["events"] if ev.get("drafter") in ("planner", "broker")})
    timeline += [(t, 0.5, None) for t in cent_ai] + [(t, 2, None) for t in cent_pl]
    timeline.sort(key=lambda x: (x[0], x[1]))
    for t, typ, x in timeline:
        if typ in (0.5, 2):                # 中央の作者が手を打った年：その時点の全員の係数を知る
            central = (central or set()) | {(j, p, tenant[j]) for j, p in ALLF}
        elif typ == 0:
            _, i, j, p = x
            facts.setdefault(tenant[i], set()).add((j, p, tenant[j]))
        else:
            tenant[x] = nxt; facts[nxt] = {(x, p, nxt) for p in "asq"}; nxt += 1
    valid = lambda F: {(j, p) for (j, p, a) in F if tenant[j] == a}
    cur = [valid(facts[tenant[u]]) for u in range(9)]
    former = [valid(F) for a, F in facts.items() if a not in tenant and _authored(r, a)]
    best_former = max(former, key=len) if former else set()
    return cur, (valid(central) if central is not None else None), best_former


def _authored(r, a):
    return True    # 去った住み手は、手を打っていなくても棋譜に住んだ記録が残る（保守的に作者に入れる）


def supported(r, W_T, start):
    """同意で支えられた部位（Λc）と、そのうち損をする側の同意で支えられた部位（Λn）"""
    X0d = {"e": start[0], "k": start[1], "o": start[2]}
    st = {}; byt = {}
    for ev in r["events"]:
        byt.setdefault(ev["t"], []).append(("ev", ev))
    for (t, u) in r["turns"]:
        byt.setdefault(t, []).append(("turn", u))
    rc = {}
    for (t, n, p, kind) in r["recon"]:
        rc.setdefault((t, n), []).append((p, kind))
    for t in sorted(byt):
        for kind, x in byt[t]:
            if kind == "ev":
                ev = x
                if ev.get("planner"):
                    if ev.get("changed", True):
                        st = {}
                    continue
                if not ev["ok"] or ev.get("part") is None:
                    continue
                p = ev["part"]
                if ev["nv"] == X0d[p[0]][p[1]]:
                    st.pop(p, None); continue
                st[p] = [ev["nv"], set(ev["supp"]) if ev["cons"] else None]
            else:
                for p, v in st.items():
                    if v[1] is not None:
                        v[1].discard(x)
                for p, how in rc.get((t, x), []):
                    if how == "revert":
                        st.pop(p, None)
                    elif p in st and st[p][1] is not None:
                        st[p][1].add(x)
    e, k, o = r["e"], r["k"], r["o"]
    sup, neg = set(), set()
    for p, (v, C) in st.items():
        if getp((e, k, o), p) != v or C is None:
            continue
        nx = [e.copy(), k.copy(), o.copy()]; {"e": nx[0], "k": nx[1], "o": nx[2]}[p[0]][p[1]] = X0d[p[0]][p[1]]
        dS = s0.S_all(W_T, *nx) - s0.S_all(W_T, e, k, o)
        hn = {j for j in range(9) if dS[j] > 1e-12}        # 戻すと得をする今の住み手（＝今の値で損をしている者）
        if hn <= C:
            sup.add(p)
            if hn:
                neg.add(p)
    return sup, neg


def entropy_tokens(tokens):
    if not tokens:
        return 0.0
    c = Counter(tokens); n = len(tokens); p = np.array(list(c.values())) / n
    return float(-(p * np.log2(p)).sum())


def record(r, W0, start_author=None, F=None):
    """一局の記録から、評価の候補を計算するための値をまとめる"""
    F = F if F is not None else s0.f_design()
    WT = r["W_final"]
    _, ez, kz, oz = s0.wstar_linear(WT)
    Z = cfgvec(ez, kz, oz)
    X = cfgvec(r["e"], r["k"], r["o"])
    start = r["ref"]
    S0v = cfgvec(*start)
    cur, cen, former = knowledge(r, start_author)
    G = [guess(WT, kn) for kn in cur]
    G.append(guess(WT, cen) if cen is not None else cfgvec(*F))
    G.append(guess(WT, former))
    sup, neg = supported(r, WT, start)
    Xc = np.array([X[n] if KEYS[n] in sup else S0v[n] for n in range(21)])
    Xn = np.array([X[n] if KEYS[n] in neg else S0v[n] for n in range(21)])
    WF = s0.S_all(WT, *F).sum(); Ws = s0.wstar_linear(WT)[0]; Wx = s0.S_all(WT, r["e"], r["k"], r["o"]).sum()
    oks = [ev for ev in r["events"] if ev.get("ok") and not ev.get("planner") and ev.get("part") is not None]
    loss_all = loss_cons = 0.0; third = 0.0
    for ev in oks:
        if "dS" not in ev:
            continue
        dS = ev["dS"]; harmed = [j for j in range(9) if j != ev.get("i") and dS[j] < -1e-12]
        L = -dS[harmed].sum() if harmed else 0.0
        loss_all += L
        if ev["cons"]:
            loss_cons += L
        inv = set(harmed) | ({ev["i"]} if ev.get("i") is not None else set())
        third += sum(max(0.0, dS[j]) for j in range(9) if j not in inv)
    U = r["U"]; SF_i = s0.S_all(WT, *F)
    return dict(Z=Z.tolist(), X=X.tolist(), Xc=Xc.tolist(), Xn=Xn.tolist(), G=[g.tolist() for g in G],
                eta=float((Wx - WF) / (Ws - WF)) if Ws - WF > 1e-6 else float("nan"),
                dSstar=float(Ws - WF), L=float(np.mean(U < SF_i - 1e-9)), B=float(np.abs(r["b"]).sum() / 9),
                moves=len(oks), quotes=len(r["quotes"]), Lc=len(sup) / 21, Ln=len(neg) / 21,
                Vp=float(loss_cons / loss_all) if loss_all > 1e-12 else float("nan"),
                gohai=float(third / s0.MU), ent=entropy_tokens([str(ev.get("part")) for ev in oks]),
                rebind=sum(1 for x in r["recon"] if x[3] == "paid"), spent=float(r.get("spent", 0.0)),
                H=float(np.mean(np.abs(W0 - WB).sum(1) / 2)))


# ------------------------------------------------------------ 相互情報量
def _Hcodes(x):
    n = len(x)
    c = np.bincount(x); c = c[c > 0]; p = c / n
    return float(-(p * np.log2(p)).sum() + (len(c) - 1) / (2 * n * np.log(2)))


def H_(xs):
    return _Hcodes(np.asarray(xs, int) + 1)


def MI(a, b):
    a = np.asarray(a, int) + 1; b = np.asarray(b, int) + 1     # 値は −1・0・1
    return max(0.0, _Hcodes(a) + _Hcodes(b) - _Hcodes(a * 3 + b))


def population(recs, key="Xc", idx=None):
    """構成の集団について V を求める。key＝"X"（V_H）・"Xc"（V*）・"Xn"（V*n）"""
    if idx is None:
        idx = np.arange(len(recs))
    Z = np.array([recs[i]["Z"] for i in idx]); X = np.array([recs[i][key] for i in idx])
    G = np.array([recs[i]["G"] for i in idx])          # (n, 作者, 21)
    HZ = sum(H_(list(Z[:, p])) for p in range(21))
    if HZ < 1e-9:
        return dict(V=0.0, I=0.0, Imax=0.0, HZ=0.0)
    I = sum(MI(list(Z[:, p]), list(X[:, p])) for p in range(21))
    Ia = [sum(MI(list(Z[:, p]), list(G[:, a, p])) for p in range(21)) for a in range(G.shape[1])]
    return dict(V=max(0.0, I - max(Ia)) / HZ, I=I / HZ, Imax=max(Ia) / HZ, HZ=HZ)


def boot(recs, key, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(recs)
    v = [population(recs, key, rng.integers(0, N, N))["V"] for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def perm_floor(recs, key, n=200, seed=1):
    """ε：Z を局のあいだで入れ替えて依存を壊したときの値の 95 パーセンタイル（推定の偏りの床）"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        perm = rng.permutation(len(recs))
        sh = [dict(recs[i], Z=recs[perm[i]]["Z"]) for i in range(len(recs))]
        out.append(population(sh, key)["V"])
    return float(np.percentile(out, 95))


def pointwise(recs, key="Xc"):
    """一局ごとの近似：Σ_p pmi(Z_p, X_p) − max_a Σ_p pmi(Z_p, Ẑ^a_p)、集団の ΣH(Z_p) で割る"""
    Z = np.array([r["Z"] for r in recs]); X = np.array([r[key] for r in recs]); G = np.array([r["G"] for r in recs])
    n = len(recs); HZ = sum(H_(list(Z[:, p])) for p in range(21))
    if HZ < 1e-9:
        return np.zeros(n)

    def pmi_table(a, b):
        ca, cb, cab = Counter(a), Counter(b), Counter(zip(a, b))
        return np.array([np.log2(cab[(x, y)] * n / (ca[x] * cb[y])) for x, y in zip(a, b)])
    tx = sum(pmi_table(list(Z[:, p]), list(X[:, p])) for p in range(21))
    ta = np.max([sum(pmi_table(list(Z[:, p]), list(G[:, a, p])) for p in range(21)) for a in range(G.shape[1])], axis=0)
    return (tx - ta) / HZ
