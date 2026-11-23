"""段 1 第二報の集計：セルごとの η・L・V*n、予言の判定、図 231〜234。
PYTHONPATH=. python3 -m experiments.s3_analyze dev|main  → out/s3/{mode}_cells.json・{mode}_judge.json、図は main のときだけ
"""
import sys, json
from collections import defaultdict
import numpy as np
from trade import vstar

MODE = sys.argv[1] if len(sys.argv) > 1 else "main"
rng = np.random.default_rng(20261107)
cells = defaultdict(list)
for line in open(f"out/s3/{MODE}.jsonl"):
    r = json.loads(line)
    cells[(r["grp"], r["cond"], r["reg"], r["pat"])].append(r)
for k in cells:
    cells[k].sort(key=lambda r: r["cs"])
PATS = ["α=3", "二群"]


def recs(g, c, rg, pat=None):
    if pat:
        return cells[(g, c, rg, pat)]
    return cells[(g, c, rg, "α=3")] + cells[(g, c, rg, "二群")]


def mean(g, c, rg, col):
    x = [r[col] for r in recs(g, c, rg)]
    x = [v for v in x if v == v]
    return float(np.mean(x)) if x else float("nan")


def paired(g1, c1, r1, g2, c2, r2, col="eta", n=2000):
    """(1) − (2) の対の差：平均と 95 % 区間（同じ pat・cs で対にする）"""
    A = {(r["pat"], r["cs"]): r[col] for r in recs(g1, c1, r1)}
    B = {(r["pat"], r["cs"]): r[col] for r in recs(g2, c2, r2)}
    d = np.array([A[k] - B[k] for k in A if k in B and A[k] == A[k] and B[k] == B[k]])
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return dict(mean=float(d.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), n=len(d))


VN = {}


def vn(g, c, rg, pat):
    k = (g, c, rg, pat)
    if k not in VN:
        R = cells[k]
        v = vstar.population(R, "Xn")["V"]
        lo, hi = vstar.boot(R, "Xn", n=100)
        VN[k] = (v, lo, hi)
    return VN[k]


def vn_diff(a, b, pat, n=300):
    """V*n(a) − V*n(b)：同じ構成の組を再抽出して差の区間"""
    A, B = cells[a + (pat,)], cells[b + (pat,)]
    ka = {r["cs"]: r for r in A}; kb = {r["cs"]: r for r in B}
    cs = sorted(set(ka) & set(kb)); RA = [ka[c] for c in cs]; RB = [kb[c] for c in cs]
    d0 = vstar.population(RA, "Xn")["V"] - vstar.population(RB, "Xn")["V"]
    out = []
    for _ in range(n):
        ix = rng.integers(0, len(cs), len(cs))
        out.append(vstar.population(RA, "Xn", ix)["V"] - vstar.population(RB, "Xn", ix)["V"])
    return dict(mean=float(d0), lo=float(np.quantile(out, .025)), hi=float(np.quantile(out, .975)))


# ---------------------------------------------------------------- ε（命名の段階と同じ規則：住民主体の三つの決め方の最大）
eps_cells = [k for k in cells if k[0] == "T" and k[2] in ("U", "TP+top3", "改良版top3") and k[1].startswith("幾何")]
EPS = max(vstar.perm_floor(cells[k], "Xn", n=100) for k in eps_cells)

# ---------------------------------------------------------------- セルの表
table = {}
for (g, c, rg, pat) in cells:
    key = f"{g}|{c}|{rg}"
    if key in table:
        continue
    row = dict(grp=g, cond=c, reg=rg)
    for col in ["eta", "L", "L_stay", "L_new", "changed", "exit_rate", "moves", "quotes", "owner_in", "restored", "refund", "recon_paid", "recon_revert", "unreached"]:
        row[col] = mean(g, c, rg, col)
    for p in PATS:
        row[f"Vn_{p}"] = vn(g, c, rg, p)
    table[key] = row
json.dump(dict(eps=EPS, cells=table), open(f"out/s3/{MODE}_cells.json", "w"), ensure_ascii=False, indent=0)

# ---------------------------------------------------------------- 予言の判定
J = {}
nu = lambda x: f"幾何 ν={x:.2f}"
a = paired("T", nu(0.30), "TP+top3", "T", nu(0.05), "TP+top3")
J["PT1"] = dict(ok=a["hi"] < 0 and mean("T", nu(0.30), "TP+top3", "eta") < 0.5, diff=a, eta30=mean("T", nu(0.30), "TP+top3", "eta"))
J["PT2"] = dict(rows=[dict(nu=x, **paired("T", nu(x), "改良版top3", "T", nu(x), "TP+top3")) for x in (0.18, 0.30)])
J["PT2"]["ok"] = all(r["lo"] > 0 for r in J["PT2"]["rows"])
J["PT3"] = dict(rows=[dict(nu=x, pat=p, **vn_diff(("T", nu(x), "改良版top3"), ("T", nu(x), "TP+top3"), p)) for x in (0.18, 0.30) for p in PATS])
J["PT3"]["ok"] = all(r["lo"] > 0 for r in J["PT3"]["rows"])
a = paired("T", "性向を引き継ぐ ν=0.30", "TP+top3", "T", nu(0.30), "TP+top3")
J["PT4"] = dict(ok=a["lo"] > 0, diff=a)
J["PT5"] = dict(rows=[dict(reg=rg, **paired("T", "更新の山 ν=0.18", rg, "T", nu(0.18), rg)) for rg in ("TP+top3", "改良版top3")])
J["PT5"]["ok"] = all(r["lo"] > -0.05 and r["hi"] < 0.05 for r in J["PT5"]["rows"])
J["PT5"]["mix"] = {rg: dict(eta=mean("T", "二型の混合", rg, "eta"), exit=mean("T", "二型の混合", rg, "exit_rate"),
                            d=paired("T", "二型の混合", rg, "T", nu(0.18), rg)) for rg in ("TP+top3", "改良版top3")}
fails = [(c, rg, p, vn("T", c, rg, p)) for (g, c, rg, p) in cells if g == "T" and rg in ("N3", "BRdyn", "N4") and vn("T", c, rg, p)[2] > EPS]
J["PT6"] = dict(ok=not fails, fails=fails)
J["PR1"] = dict(rows=[dict(reg=rg, **paired("R", "R0 費用あり ν=0.30", rg, "R", "R∅ 費用あり ν=0.30", rg)) for rg in ("TP+top3", "改良版top3")])
J["PR1"]["ok"] = all(r["hi"] < 0 for r in J["PR1"]["rows"])
J["PR2"] = dict(rows=[dict(reg=rg, **paired("R", "R0 費用あり ν=0.30", rg, "R", "R∅ 費用あり ν=0.30", rg, col="L")) for rg in ("TP+top3", "改良版top3")])
J["PR2"]["ok"] = all(r["hi"] < 0.05 for r in J["PR2"]["rows"])
a = paired("R", "Rdk10 費用あり ν=0.30", "TP+top3", "R", "R0 費用あり ν=0.30", "TP+top3")
b = paired("R", "Rdk20 費用あり ν=0.30", "TP+top3", "R", "Rdk10 費用あり ν=0.30", "TP+top3")
J["PR3"] = dict(ok=a["lo"] > 0 and b["lo"] > -0.02, rdk10_r0=a, rdk20_rdk10=b)
J["PR4"] = dict(rows=[dict(pat=p, **vn_diff(("R", "R0 費用あり ν=0.30", "改良版top3"), ("R", "R∅ 費用あり ν=0.30", "改良版top3"), p)) for p in PATS])
J["PR4"]["ok"] = all(r["hi"] < 0 for r in J["PR4"]["rows"])


def oc(own, G, x):
    return f"{own}・{G} ν={x:.2f}"


def effect(pairs, n=2000):
    """対の差の平均を、構成ごとに平均してからブートストラップ"""
    per = defaultdict(list)
    for (c1, c2) in pairs:
        A = {(r["pat"], r["cs"]): r["eta"] for r in recs("O", c1, "TP+top3")}
        B = {(r["pat"], r["cs"]): r["eta"] for r in recs("O", c2, "TP+top3")}
        for k in A:
            if k in B:
                per[k].append(A[k] - B[k])
    return {k: np.mean(v) for k, v in per.items()}


NUS = (0.04, 0.18, 0.30)
eG = effect([(oc(o, "G1", x), oc(o, "G0", x)) for o in ("借家", "所有") for x in NUS])
eN = effect([(oc(o, G, 0.04), oc(o, G, 0.30)) for o in ("借家", "所有") for G in ("G0", "G1")])
eO = effect([(oc("所有", G, x), oc("借家", G, x)) for G in ("G0", "G1") for x in NUS])
ks = sorted(set(eG) & set(eN) & set(eO))
M = np.array([[eG[k], eN[k], eO[k]] for k in ks])
bs = M[rng.integers(0, len(ks), (2000, len(ks)))].mean(1)
J["PO1"] = dict(G=float(M[:, 0].mean()), nu=float(M[:, 1].mean()), own=float(M[:, 2].mean()),
                G_minus_nu=[float(np.quantile(bs[:, 0] - bs[:, 1], q)) for q in (.025, .975)],
                nu_minus_own=[float(np.quantile(bs[:, 1] - bs[:, 2], q)) for q in (.025, .975)])
J["PO1"]["ok"] = J["PO1"]["G_minus_nu"][0] > 0 and J["PO1"]["nu_minus_own"][0] > 0
h = effect([(oc("借家", "G½", x), oc("借家", "G0", x)) for x in NUS]); f = effect([(oc("借家", "G1", x), oc("借家", "G0", x)) for x in NUS])
ks = sorted(set(h) & set(f)); D = np.array([h[k] - 0.5 * f[k] for k in ks])
bs = D[rng.integers(0, len(ks), (2000, len(ks)))].mean(1)
J["PO2"] = dict(half=float(np.mean([h[k] for k in ks])), full=float(np.mean([f[k] for k in ks])), diff=[float(np.quantile(bs, q)) for q in (.025, .975)])
J["PO2"]["ok"] = J["PO2"]["diff"][1] < 0
W30 = lambda o: f"{o} ν=0.30"
order = ["なし", "個人・取引", "個人・拒否", "管理開口・拒否", "管理∅・拒否"]
rows = []
for x, y in zip(order, order[1:]):
    d = paired("W", W30(x), "TP+top3", "W", W30(y), "TP+top3")
    strict = (x, y) in (("なし", "個人・取引"), ("個人・取引", "個人・拒否"))
    rows.append(dict(a=x, b=y, strict=strict, ok=(d["lo"] > 0) if strict else (d["lo"] > -0.02), **d))
J["PW1"] = dict(rows=rows, ok=all(r["ok"] for r in rows), eta={o: mean("W", W30(o), "TP+top3", "eta") for o in order})
d = paired("W", W30("管理∅・取引"), "TP+top3", "W", W30("管理∅・拒否"), "TP+top3")
J["PW2"] = dict(ok=abs(d["mean"]) < 1e-12 and abs(d["lo"]) < 1e-12, diff=d)
d = paired("W", W30("管理∅・D4"), "TP+top3", "W", W30("管理∅・拒否"), "TP+top3")
J["PW3"] = dict(ok=d["lo"] > 0.1, diff=d)
rows = [dict(own=o, **paired("W", W30(f"{o}・D5"), "TP+top3", "W", W30(f"{o}・拒否"), "TP+top3")) for o in ("個人", "管理∅", "サブ")]
J["PW4"] = dict(ok=all(r["lo"] > 0 for r in rows[:2]), rows=rows)
rows = [dict(pat=p, **vn_diff(("W", W30("個人・拒否"), "TP+top3"), ("W", W30("なし"), "TP+top3"), p)) for p in PATS]
J["PW5"] = dict(ok=all(r["hi"] < 0 for r in rows), rows=rows)
J["epsilon"] = EPS
json.dump(J, open(f"out/s3/{MODE}_judge.json", "w"), ensure_ascii=False, indent=1, default=float)
for k, v in J.items():
    if isinstance(v, dict):
        print(k, "支持" if v.get("ok") else "不支持")
print("ε", round(EPS, 4))
