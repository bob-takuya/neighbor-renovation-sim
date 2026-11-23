"""s8 の試験：既定の引数で s3.run と一致すること（記録まで）、帳簿が残高と合うこと、退去の効果と収支の所有者の性質。
PYTHONPATH=. python3 -m tests.test_s8"""
import numpy as np
from trade import s0, s1, s3, s8
from experiments.explore import composition

ok = True


def check(name, cond, info=""):
    global ok
    print(("合格 " if cond else "不合格 ") + name + (f"  {info}" if str(info) else ""))
    ok &= bool(cond)


def same(a, b):
    if not (all(np.array_equal(a[x], b[x]) for x in "eko") and all(np.allclose(a[x], b[x]) for x in ["S", "m", "c", "b", "Wt", "U"])):
        return False
    if len(a["events"]) != len(b["events"]) or a["quotes"] != b["quotes"] or a["turns"] != b["turns"] or a["recon"] != b["recon"]:
        return False
    for x, y in zip(a["events"], b["events"]):
        if x.get("part") != y.get("part") or x.get("ok") != y.get("ok") or x.get("nv") != y.get("nv") or x.get("supp") != y.get("supp"):
            return False
    if a.get("extra") is not None and any(abs(a["extra"][q] - b["extra"][q]) > 1e-9 for q in a["extra"]):
        return False
    return abs(a["spent"] - b["spent"]) < 1e-9


def ledger_ok(r, settle):
    """帳簿の授受を世帯ごとに足し、住戸に残っている世帯の分を合わせると、30 年目のお金＋貸し借りに一致する"""
    bal = {}
    for (t, kd, f, to, amt, part, med) in r["ledger"]:
        if kd == "rent":
            continue                       # 家賃の上乗せは模型のお金 m に入れない（帳簿だけ）
        if isinstance(f, (int, np.integer)):
            bal[f] = bal.get(f, 0.0) - amt
        if isinstance(to, (int, np.integer)):
            bal[to] = bal.get(to, 0.0) + amt
    tot = np.zeros(9)
    for h, v in r["hh"].items():
        if settle and v["t_out"] is not None:
            continue
        tot[v["unit"]] += bal.get(h, 0.0)
    return float(np.abs(tot - (r["m"] + r["c"])).max())


IMP = dict(credit=20.0, settle=True, reconsent="all")
C1 = (dict(e=1.0, k=0.25, o=0.25), dict(e=0.5, k=0.1, o=0.1))
IND = dict(kind="ind", role="veto", beta=0.5, gamma=1.0)
# 1. 既定の引数で s3 と一致（観察 1・観察 2 の条件の形を網羅）
n = bad = 0; worst = 0.0
for pat in ["α=3", "二群"]:
    for cs in range(2):
        W = composition(pat, cs)
        newW = lambda rng, i, pat=pat: composition(pat, int(rng.integers(10**6, 10**7)))[i]
        geo = lambda v: dict(turnover=dict(kind="geo", nu=v), newW=newW)
        settings = [{}, dict(IMP), dict(protocol="topk", ask=3), dict(IMP, protocol="topk", ask=3, **geo(0.1)),
                    dict(protocol="topk", ask=3, **geo(0.3)),
                    dict(IMP, protocol="topk", ask=3, cost=C1[0], rcost=C1[1], horizon=10, restore="R0", **geo(0.3)),
                    dict(protocol="topk", ask=3, cost=C1[0], rcost=C1[1], horizon=10, restore="Rdk", amort=10, **geo(0.3)),
                    dict(protocol="topk", ask=3, cost=C1[0], rcost=C1[1], horizon=10, restore="R0", owner=dict(IND, role="trade"), **geo(0.1)),
                    dict(protocol="topk", ask=3, cost=C1[0], rcost=C1[1], horizon=10, restore="R0", owner=IND, venue=dict(G=0.5, alpha=0.5), **geo(0.1)),
                    dict(protocol="topk", ask=3, cost=C1[0], rcost=C1[1], horizon=10, venue=dict(G=0, q_far=0.4, tau=0.5), **geo(0.18))]
        for reg in ["U", "V½", "V¾", "T0+", "T1+"]:
            for kw in settings:
                a = s3.run(W, reg, cs, **kw); b = s8.run(W, reg, cs, **kw); n += 1
                if not same(a, b):
                    bad += 1
                worst = max(worst, ledger_ok(b, kw.get("settle", False)))
check("s8 は既定の引数で s3 と一致（記録まで）", bad == 0, f"{n - bad}/{n} 局")
check("帳簿の授受を世帯ごとに足すと、住戸のお金＋貸し借りに一致する", worst < 1e-9, f"最大の差 {worst:.1e}")
# 観察 1 の主条件：s0.run と一致
W = composition("α=3", 105)
bad = 0
for reg, kw in [("U", {}), ("V½", {}), ("V¾", {}), ("T0+", {}), ("T1+", {}), ("T1+", IMP)]:
    for sd in range(3):
        a = s0.run(W, reg, seed=105000 + sd, **kw); b = s8.run(W, reg, 105000 + sd, **kw)
        if not (all(np.array_equal(a[x], b[x]) for x in "eko") and all(np.allclose(a[x], b[x]) for x in ["S", "m", "c", "b", "U"])):
            bad += 1
check("観察 1 の主条件の決め方で s0 と一致", bad == 0, f"{18 - bad}/18 局")

W = composition("α=3", 7)
newW = lambda rng, i: composition("α=3", int(rng.integers(10**6, 10**7)))[i]
base = dict(protocol="topk", ask=3, cost=C1[0], rcost=C1[1], horizon=10, restore="R0", turnover=dict(kind="geo", nu=0.3), newW=newW)
# 2. 世帯の表
r = s8.run(W, "T1+", 7, **base)
check("世帯の数 ＝ 9 ＋ 退去の数", len(r["hh"]) == 9 + r["n_exit"], (len(r["hh"]), r["n_exit"]))
check("住み続けた世帯の毎年の満足は 30 個、退去した世帯は退去の年まで",
      all(len(v["Sy"]) == (30 - v["t_in"] if v["t_out"] is None else v["t_out"] - v["t_in"] + 1) for v in r["hh"].values()))
# 3. 退去の効果：φ＝ψ＝0 なら s3 と同じ。φ＞0 なら、住み心地の良い世帯の退去が減る
a = s3.run(W, "T1+", 7, **base); b = s8.run(W, "T1+", 7, exitfx=dict(phi=0.0, psi=0.0), **base)
check("φ＝ψ＝0 の退去の効果は s3 と同じ", same(a, b))
ex = [np.mean([s8.run(W, "T1+", s, exitfx=dict(phi=ph, psi=0.0), **base)["n_exit"] for s in range(40)]) for ph in (0.0, 0.15)]
check("φ＝0.15 では、手で住み心地が上がるので退去が減る（TP+top3、40 局の平均）", ex[1] < ex[0], [round(x, 2) for x in ex])
exF = [np.mean([s8.run(W, "F", s, exitfx=dict(phi=ph, psi=0.5), **base)["n_exit"] for s in range(40)]) for ph in (0.0, 0.15)]
check("F では住み心地が F と同じなので、φ によらず退去は同じ", abs(exF[0] - exF[1]) < 1e-12, [round(x, 2) for x in exF])
# 4. 収支の所有者
zero = dict(kind="econ", role="veto", lam=0.0, v0=0.0, v1=0.0, cturn=0.0)
a = s8.run(W, "T1+", 7, **base); b = s8.run(W, "T1+", 7, owner=zero, **base)
check("家賃・空室・募集の費用がすべて 0 の収支の所有者は、所有者なしと同じ手を通す", same(a, b))
X = s0.f_design()
d, sg = s8.market2(*X)
check("F の盤では d・σ はどちらも 0", np.allclose(d, 0) and np.allclose(sg, 0))
Y, P = s8.stay_years(0.3, 10)
check("見通し 10 年の住む年数の期待値は g と合う（ν＝0.3）", abs(Y / 10 - s3.g_of(0.3, 10)) < 1e-12 and 0 < P < 1, (round(Y, 3), round(P, 3)))
mv = lambda r: sum(1 for ev in r["events"] if ev["ok"])
econ = dict(kind="econ", role="veto")
nv_ = [np.mean([mv(s8.run(composition(pt, cs), "T1+", cs, owner=dict(econ, role=ro), **dict(base, newW=lambda rng, i, pt=pt: composition(pt, int(rng.integers(10**6, 10**7)))[i])))
                for pt in ("α=3", "二群") for cs in range(10)]) for ro in ("veto", "trade")]
check("退去の効果なしでは、収支の所有者（拒否権のみ）はどの手も通さず、取引の相手にすると手が通る（20 局の平均）", nv_[0] == 0 and nv_[1] > 0, nv_)
r = s8.run(W, "T1+", 7, owner=dict(econ, role="trade"), exitfx=dict(phi=0.15, psi=0.5), **base)
check("取引の相手では、所有者が受け取った額が帳簿と合う", abs(r["owner_pl"]["paid_in"] - r["extra"]["owner_in"]) < 1e-9,
      round(r["owner_pl"]["paid_in"], 3))
r = s8.run(W, "F", 7, landlord={}, **base)
check("F では家賃の上乗せは 0、空室の損は退去ごとに v0·r0", abs(r["owner_pl"]["rent"]) < 1e-12 and
      abs(r["owner_pl"]["vacancy"] - 0.25 * 5.0 * r["n_exit"]) < 1e-9, r["owner_pl"])
print("すべて合格" if ok else "不合格あり")
