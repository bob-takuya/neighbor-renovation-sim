"""s1 の試験：既定の引数で s0.run と一致すること、費用と帰無の決め方の性質。
PYTHONPATH=. python3 -m tests.test_s1"""
import sys
import numpy as np
from trade import s0, s1
from experiments.explore import composition

ok = True


def check(name, cond, info=""):
    global ok
    print(("合格 " if cond else "不合格 ") + name + (f"  {info}" if str(info) else ""))
    ok &= bool(cond)


def same(a, b):
    return all(np.array_equal(a[x], b[x]) for x in "eko") and np.allclose(a["S"], b["S"]) and np.allclose(a["m"], b["m"]) \
        and np.allclose(a["c"], b["c"]) and np.allclose(a["b"], b["b"]) and np.allclose(a["Wt"], b["Wt"])


# 1. 既定の引数で s0 と一致（決め方 9 × 改良の部品 × 前提を崩す条件）
n = bad = 0
for pat in ["α=3", "二群", "α=0.3"]:
    for cs in range(4):
        W = composition(pat, cs)
        newW = lambda rng, i, pat=pat: composition(pat, int(rng.integers(10**6, 10**7)))[i]
        for reg in ["U", "V½", "V¾", "T0", "T0+", "T1", "T1+"]:
            for kw in [{}, dict(theta=0.5), dict(budget=np.full(9, 5.0)), dict(credit=20.0, settle=True, reconsent="all", nu=0.1, newW=newW),
                       dict(nu=0.1, newW=newW), dict(split=True), dict(form="concave"), dict(conv_scale=2.0)]:
                if kw.get("split") and reg != "T1":
                    continue
                a = s0.run(W, reg, cs, **kw); b = s1.run(W, reg, cs, **kw); n += 1
                if not same(a, b):
                    bad += 1
check("s1 は既定の引数で s0 と一致", bad == 0, f"{n - bad}/{n} 局")

# 2. 費用：費用 0 を明示しても変わらない。費用を上げると手の数は減り、全知の組み替えの費用込みの値は wstar 以下
W = composition("α=3", 5)
a = s1.run(W, "T1+", 5); b = s1.run(W, "T1+", 5, cost=dict(e=0, k=0, o=0), rcost=dict(e=0, k=0, o=0))
check("費用 0 を明示しても同じ", same(a, b))
nm = [sum(ev["ok"] for ev in s1.run(W, "T1+", 5, cost=dict(e=c, k=c / 4, o=c / 4), rcost=dict(e=c / 2, k=c / 8, o=c / 8))["events"]) for c in (0, 1, 4, 16)]
check("費用を上げると手は減る（増えない）", all(nm[i] >= nm[i + 1] for i in range(3)), nm)
F = s0.f_design()
e, k, o, cst = s1.best_net(W, *[x.copy() for x in F], F, dict(e=1, k=0.25, o=0.25), dict(e=0.5, k=0.1, o=0.1))
check("費用込みの全知の組み替えは費用なしの最適を超えない", s0.S_all(W, e, k, o).sum() - s0.MU * cst <= s0.wstar_linear(W)[0] + 1e-9)
e0, k0, o0, _ = s1.best_net(W, *[x.copy() for x in F], F)
check("費用なしの組み替え＝wstar_linear", np.isclose(s0.S_all(W, e0, k0, o0).sum(), s0.wstar_linear(W)[0]))

# 3. 帰無・偽装の決め方
r = s1.run(W, "P", 5)
check("N3（P）は全知の最適に置く", np.isclose(r["S"].sum(), s0.wstar_linear(W)[0]))
r = s1.run(W, "BR", 5)
check("BR は全知の最適に置く", np.isclose(r["S"].sum(), s0.wstar_linear(W)[0]))
r = s1.run(W, "AI+", 5)
check("起案 AI は問い合わせを 1 手に絞る", all(ev["drafter"] == "ai" for ev in r["events"]), len(r["quotes"]))
r1 = s1.run(W, "T1+", 5, protocol="near")
check("near は近い戸にだけ聞く", all(j in s1.NEAR[i] for (_, i, j, _) in r1["quotes"]))
print("すべて合格" if ok else "不合格あり")
