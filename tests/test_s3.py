"""s3 の試験：既定の引数で s1.run と一致すること（記録まで）、賃貸の制約の四つの部品の性質。
PYTHONPATH=. python3 -m tests.test_s3"""
import numpy as np
from trade import s0, s1, s3
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
    return abs(a["spent"] - b["spent"]) < 1e-9


IMP = dict(credit=20.0, settle=True, reconsent="all")
C1 = (dict(e=1.0, k=0.25, o=0.25), dict(e=0.5, k=0.1, o=0.1))
# 1. 既定の引数で s1 と一致
n = bad = 0
for pat in ["α=3", "二群"]:
    for cs in range(3):
        W = composition(pat, cs)
        newW = lambda rng, i, pat=pat: composition(pat, int(rng.integers(10**6, 10**7)))[i]
        for reg in ["U", "V½", "T0+", "T1+", "AI+", "P", "BR", "BRdyn", "R", "Rc"]:
            for kw in [{}, dict(protocol="topk", ask=3), dict(protocol="near"), dict(IMP, nu=0.1, newW=newW),
                       dict(IMP, protocol="topk", ask=3, nu=0.1, newW=newW, cost=C1[0], rcost=C1[1]), dict(nu=0.3, newW=newW, cost=C1[0], rcost=C1[1])]:
                a = s1.run(W, reg, cs, **kw); b = s3.run(W, reg, cs, **kw); n += 1
                if not same(a, b):
                    bad += 1
check("s3 は既定の引数で s1 と一致（記録まで）", bad == 0, f"{n - bad}/{n} 局")

W = composition("α=3", 7)
newW = lambda rng, i: composition("α=3", int(rng.integers(10**6, 10**7)))[i]
# 2. 入れ替わりの型：geo は nu と同じ。mix・renew の実現した退去率
a = s1.run(W, "T1+", 7, nu=0.18, newW=newW); b = s3.run(W, "T1+", 7, turnover=dict(kind="geo", nu=0.18), newW=newW)
check("turnover geo は nu と同じ", same(a, b))
ex = {kd: np.mean([s3.run(W, "F", s, turnover=tv, newW=newW)["n_exit"] / (9 * 30) for s in range(300)])
      for kd, tv in [("geo", dict(kind="geo", nu=0.18)), ("renew", dict(kind="renew", nu=0.18, rho=2.0)),
                     ("mix", dict(kind="mix"))]}
check("geo・renew の実現した退去率は 0.18 に近い", abs(ex["geo"] - 0.18) < 0.01 and abs(ex["renew"] - 0.18) < 0.02, {k_: round(v, 3) for k_, v in ex.items()})
h0 = s3.renew_base(0.18, 2.0)
check("renew の基準の退去率は ν と 2ν の間", h0 < 0.18 < 2 * h0, round(h0, 4))
# 3. 原状回復 R0：退去のたびに、その世帯の起こした手が戻る。退去がなければ R∅ と同じ
a = s3.run(W, "T1+", 7, protocol="topk", ask=3); b = s3.run(W, "T1+", 7, protocol="topk", ask=3, restore="R0")
check("退去がなければ R0 は R∅ と同じ", same(a, b))
r = s3.run(W, "T1+", 7, protocol="topk", ask=3, restore="R0", nu=0.3, newW=newW, cost=C1[0], rcost=C1[1])
rest = [ev for ev in r["events"] if ev.get("drafter") == "restore"]
check("R0 は手を戻し、戻す費用を払う", len(rest) > 0 and all(ev["nv"] == s1.getp(r["ref"], ev["part"]) for ev in rest), len(rest))
# 30 年目に残っている改変は、今の住み手の誰かが起こしたもの（再同意で残したものを除けば）
r = s3.run(W, "T1+", 11, protocol="topk", ask=3, restore="R0", nu=0.3, newW=newW)
chg = sum(1 for p in s1.KEYS if s1.getp((r["e"], r["k"], r["o"]), p) != s1.getp(r["ref"], p))
r0 = s3.run(W, "T1+", 11, protocol="topk", ask=3, nu=0.3, newW=newW)
chg0 = sum(1 for p in s1.KEYS if s1.getp((r0["e"], r0["k"], r0["o"]), p) != s1.getp(r0["ref"], p))
check("R0 では 30 年目に残る改変が R∅ より少ない（同じ乱数）", chg <= chg0, (chg, chg0))
# 4. Rdk：受け取る見込みの割合
check("Rdk の見込み：ν＝0 で 0、ν が大きいほど大きい、A が長いほど大きい",
      s3.refund_frac(0, 10) == 0 and s3.refund_frac(0.3, 10) > s3.refund_frac(0.1, 10) and s3.refund_frac(0.3, 20) > s3.refund_frac(0.3, 10),
      [round(s3.refund_frac(v, A), 3) for v, A in [(0.1, 10), (0.3, 10), (0.3, 20)]])
r = s3.run(W, "T1+", 7, protocol="topk", ask=3, restore="Rdk", amort=10, nu=0.3, newW=newW, cost=C1[0], rcost=C1[1], horizon=10)
check("Rdk は退去で未償却分を受け取る", r["extra"]["refund"] > 0, round(r["extra"]["refund"], 2))
# 5. 見通し：ν＝0 なら g＝1 で、見通しを付けても同じ
a = s3.run(W, "T1+", 7, cost=C1[0], rcost=C1[1]); b = s3.run(W, "T1+", 7, cost=C1[0], rcost=C1[1], horizon=10)
check("入れ替わりがなければ見通しを付けても同じ", same(a, b))
check("g は ν とともに下がる", s3.g_of(0.3, 10) < s3.g_of(0.1, 10) < 1, [round(s3.g_of(v, 10), 3) for v in (0.1, 0.18, 0.3)])
# 6. 所有者：mgmt で Lam＝∅ なら何も変わらない。gamma＝beta＝0 の ind は所有者なしと同じ。trade は veto 以上に手を通す
r = s3.run(W, "T1+", 7, protocol="topk", ask=3, owner=dict(kind="mgmt", Lam=set()))
check("管理会社・一覧なしでは何も変わらない", all(np.array_equal(r[x], r["ref"][j]) for j, x in enumerate("eko")))
a = s3.run(W, "T1+", 7, protocol="topk", ask=3); b = s3.run(W, "T1+", 7, protocol="topk", ask=3, owner=dict(kind="ind", beta=0.0, gamma=0.0))
check("無関心な所有者（beta＝gamma＝0）は所有者なしと同じ", same(a, b))
mv = lambda r: sum(1 for ev in r["events"] if ev["ok"])
nv_ = [mv(s3.run(W, "T1+", 7, protocol="topk", ask=3, owner=dict(kind="ind", role=ro))) for ro in ("veto", "trade")]
check("所有者を取引の相手にすると、手は増える（減らない）", nv_[1] >= nv_[0], nv_)
a = s3.run(W, "T1+", 7, protocol="topk", ask=3, owner=dict(kind="mgmt", Lam=set(), role="veto"))
b = s3.run(W, "T1+", 7, protocol="topk", ask=3, owner=dict(kind="mgmt", Lam=set(), role="trade"))
check("管理会社では取引の相手にしても同じ（価格に反応しない）", same(a, b))
D, X = s3.market(*s0.f_design())
check("F の盤では市場の評価は 0", np.allclose(D, 0) and np.allclose(X, 0))
# 7. 決める場：G＝0 で q_far＝1・tau＝0 なら場の制約なしと同じ。G＝0 は届かない手を打てない
a = s3.run(W, "T1+", 7, protocol="topk", ask=3); b = s3.run(W, "T1+", 7, protocol="topk", ask=3, venue=dict(G=0, q_far=1.0, tau=0.0))
check("G＝0・q＝1・τ＝0 は制約なしと同じ", same(a, b))
r = s3.run(W, "T1+", 7, protocol="topk", ask=3, venue=dict(G=0, q_far=0.0, tau=0.0))
far = [ev for ev in r["events"] if ev["ok"] and any(j not in s1.NEAR[ev["i"]] and j != ev["i"] and ev["dS"][j] < -1e-12 for j in range(9))]
check("q_far＝0 では、遠い戸に損をさせる手は打たれない", len(far) == 0)
print("すべて合格" if ok else "不合格あり")
