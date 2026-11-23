"""S0 の試験。python3 -m tests.test_s0 で実行"""
import time
import numpy as np
from trade import s0, enumerate as EN

ok = True


def check(name, cond, info=""):
    global ok
    print(("合格 " if cond else "不合格 ") + name + (f"  {info}" if str(info) else ""))
    ok &= bool(cond)


# T-S0-5 F の形
e, k, o = s0.f_design("linear")
check("T-S0-5 F の形（線形）", list(np.array(s0.UNITS)[e == 1]) == ["2W", "2M", "2E", "1W", "1M", "1E"], str(list(np.array(s0.UNITS)[e == 1])))
e2, _, _ = s0.f_design("concave")
check("T-S0-5 F の形（逓減）", list(np.array(s0.UNITS)[e2 == 1]) == ["1W", "1M", "1E"], str(list(np.array(s0.UNITS)[e2 == 1])))

# T-S0-1 一手の例（何もない状態から 3W が張り出す）
W = np.tile(s0.W_BAR, (9, 1))
W[0] = [0.60, 0.15, 0.15, 0.10]; W[3] = [0.20, 0.50, 0.20, 0.10]; W[6] = [0.40, 0.20, 0.30, 0.10]
z9, z6 = np.zeros(9, int), np.zeros(6, int)
e1, k1, o1 = s0.apply(("ext", 0), 0, z9, z6, z6)
dS = s0.S_all(W, e1, k1, o1) - s0.S_all(W, z9, z6, z6)
dC = s0.S_all(np.tile(s0.W_BAR, (9, 1)), e1, k1, o1) - s0.S_all(np.tile(s0.W_BAR, (9, 1)), z9, z6, z6)
tok = dS / s0.MU
check("T-S0-1 自分の物差し", np.allclose(tok[[0, 3, 6]], [10.2857, -8.5714, -1.1429], atol=1e-3), np.round(tok[[0, 3, 6]], 2))
check("T-S0-1 共通の物差し", np.allclose((dC / s0.MU)[[3, 6]], [-5.1429, -1.7143], atol=1e-3), np.round((dC / s0.MU)[[3, 6]], 2))
v0, ok0, _ = s0.decide("T0", 0, dS, dC, None)
v1, ok1, p1 = s0.decide("T1", 0, dS, dC, None)
check("T-S0-1 T0 は却下", not ok0)
check("T-S0-1 T1 は成立・払い 9.71", ok1 and abs(p1.sum() - 9.714) < 1e-2, f"{p1.sum():.2f}")

# T-S0-2 不変量、T-S0-7 決定性
rng = np.random.default_rng(0)
Wr = rng.dirichlet(s0.W_BAR * 1.0, 9)
inv = True; det = True
for reg in s0.REGIMES:
    r = s0.run(Wr, reg, seed=3)
    inv &= abs(r["b"].sum()) < 1e-9 and abs(r["m"].sum()) < 1e-9
    r2 = s0.run(Wr, reg, seed=3)
    det &= np.array_equal(r["Wt"], r2["Wt"]) and len(r["events"]) == len(r2["events"])
check("T-S0-2 Σb=0, Σm=0（全決め方）", inv)
check("T-S0-7 決定性", det)

# T-S0-3 同質の対照
Wh = np.tile(s0.W_BAR, (9, 1))
rT0 = s0.run(Wh, "T0", 1); rT1 = s0.run(Wh, "T1", 1); rU = s0.run(Wh, "U", 1); rF = s0.run(Wh, "F", 1)
check("T-S0-3 同質で T0・T1 は動かない", len([x for x in rT0["events"] if x["ok"]]) == 0 and len([x for x in rT1["events"] if x["ok"]]) == 0,
      f"T0 {len(rT0['events'])} 件, T1 {len(rT1['events'])} 件")
check("T-S0-3 同質で U は F より悪い", rU["Wt"][-1] < rF["Wt"][-1] - 1e-9, f"ΔW={rU['Wt'][-1]-rF['Wt'][-1]:.4f}")

# T-S0-4 上界、総当たりとの一致
t0 = time.time()
Ws, Sm = EN.enumerate_all(Wr)
te = time.time() - t0
wstar = float(Ws.max())
fin = [s0.run(Wr, reg, seed=sd)["Wt"][-1] for reg in s0.REGIMES for sd in range(5)]
check("T-S0-4 W ≤ W*", max(fin) <= wstar + 1e-5, f"max W={max(fin):.4f}, W*={wstar:.4f}, 総当たり {te:.1f}s")
# 符号化の一致：ある局の最終状態を総当たりの配列で引く
r = s0.run(Wr, "T1", seed=2)
io, ie, ik = s0.encode(r["e"], r["k"], r["o"])
check("符号化と総当たりの添字の一致", abs(Ws[io, ie, ik] - r["S"].sum()) < 1e-5, f"{Ws[io, ie, ik]:.6f} vs {r['S'].sum():.6f}")

# T-S0-6 条件が空でない
outs = {reg: tuple(s0.run(Wr, reg, seed=4)["Wt"].round(9)) for reg in s0.REGIMES}
distinct = len(set(outs.values()))
check("T-S0-6 決め方が違う走行を生む", distinct >= 5, f"{distinct} 通り / {len(s0.REGIMES)}")
nrej = sum(1 for sd in range(10) for x in s0.run(Wr, "V¾", seed=sd)["events"] if not x["ok"])
check("T-S0-6 V¾ で否決が起きる", nrej > 0, f"否決 {nrej} 件")

print("\n全体：", "合格" if ok else "不合格あり")
