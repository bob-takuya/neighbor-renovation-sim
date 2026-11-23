"""s4・vstar2 の試験。
1. 囮でない決め方では、s4.run は s1.run・s3.run と同じ記録を返す（既定の設定で一字一句同じ）。
2. 実験の手順 s4_confirm.one は、名付けの確かめの記録 out/s2/name.jsonl の行と一字一句同じ JSON を返す。
3. 囮（M・Rep）の手の並び・形・お金は、同じ聞く範囲の TP+ と一致する（違うのは誰が誰の言い値を知るかだけ）。
4. M：提案者が知るのは損をする戸が一戸だけの手の係数だけで、清算所は TP+ の提案者が聞く言い値と同じものを個別に知る。
5. Rep：代表が入れ替わりで出ると、次の代表が選ばれる。代表の知識は、聞いた言い値と要求で明かされた係数。
6. vstar2：囮でない局では vstar.record と同じ。清算所を作者に数えた値は数えない値以下（定義から）。
PYTHONPATH=. python3 -m tests.test_s4"""
import json
import numpy as np
from trade import s0, s1, s3, s4, vstar, vstar2
from experiments.explore import composition
from experiments import s2_cand, s4_confirm

ok = True


def check(name, cond, info=""):
    global ok
    print(("合格 " if cond else "不合格 ") + name + (f"  {info}" if str(info) else ""))
    ok &= bool(cond)


def same(a, b, quotes=True):
    if not (all(np.array_equal(a[x], b[x]) for x in "eko") and all(np.allclose(a[x], b[x]) for x in ["S", "m", "c", "b", "Wt", "U"])):
        return False
    if len(a["events"]) != len(b["events"]) or a["turns"] != b["turns"] or a["recon"] != b["recon"]:
        return False
    if quotes and a["quotes"] != b["quotes"]:
        return False
    for x, y in zip(a["events"], b["events"]):
        for f in ("part", "ok", "nv", "supp", "i", "t", "cons"):
            if x.get(f) != y.get(f):
                return False
        if "pay" in x and not np.allclose(x["pay"], y["pay"]):
            return False
    return abs(a["spent"] - b["spent"]) < 1e-9


IMP = dict(credit=20.0, settle=True, reconsent="all")
C1 = (dict(e=1.0, k=0.25, o=0.25), dict(e=0.5, k=0.1, o=0.1))

# 1. 既定の設定で s1・s3 と一致
n = bad = 0
for pat in ["α=3", "二群"]:
    for cs in range(3):
        W = composition(pat, cs)
        newW = lambda rng, i, pat=pat: composition(pat, int(rng.integers(10**6, 10**7)))[i]
        for reg in ["U", "V½", "V¾", "T0+", "T1+", "AI+", "P", "BR", "BRdyn", "R", "Rc", "F"]:
            for kw in [{}, dict(protocol="topk", ask=3), dict(protocol="topk", ask=1), dict(protocol="near"), dict(IMP, nu=0.1, newW=newW),
                       dict(IMP, protocol="topk", ask=3, nu=0.1, newW=newW, cost=C1[0], rcost=C1[1]),
                       dict(nu=0.1, newW=newW, cost=C1[0], rcost=C1[1])]:
                a = s1.run(W, reg, cs, **kw); b = s4.run(W, reg, cs, **kw); c_ = s3.run(W, reg, cs, **kw); n += 1
                if not (same(a, b) and same(c_, b)):
                    bad += 1
check("s4 は囮でない決め方で s1・s3 と一致（記録まで）", bad == 0, f"{n - bad}/{n} 局")

# 2. 名付けの確かめの記録と一字一句同じ
want = {}
for line in open("out/s2/name.jsonl"):
    r = json.loads(line)
    if r["cs"] in (200, 213, 249) and r["pat"] in ("α=3", "二群"):
        want[(r["pat"], r["cs"], r["nu"], r["cost"], r["reg"])] = line.strip()
n = bad = 0
for (pat, cs, nu, c, reg), line in sorted(want.items()):
    got = json.dumps(s4_confirm.one((pat, cs, nu, c, reg, 0)), ensure_ascii=False)
    n += 1
    if got != line:
        bad += 1
check("s4_confirm.one は out/s2/name.jsonl と一字一句同じ（cseed 200・213・249、20 条件 × 8 セル）", bad == 0 and n == 480, f"{n - bad}/{n} 行")
# s2_cand.one とも同じ（選ぶ用の種）
n = bad = 0
for pat in ["α=3", "二群"]:
    for cs in (0, 17):
        for nu in (0.0, 0.1):
            for reg in s2_cand.REGS:
                a = json.dumps(s2_cand.one((pat, cs, nu, "C1", reg, 0)), ensure_ascii=False)
                b = json.dumps(s4_confirm.one((pat, cs, nu, "C1", reg, 0)), ensure_ascii=False)
                n += 1; bad += a != b
check("s4_confirm.one は s2_cand.one と同じ（選ぶ用、20 条件）", bad == 0, f"{n - bad}/{n}")

# 3. 囮の手の並びは TP+ と同じ
n = bad = 0
for pat in ["α=3", "二群"]:
    for cs in range(4):
        W = composition(pat, cs)
        newW = lambda rng, i, pat=pat: composition(pat, int(rng.integers(10**6, 10**7)))[i]
        for kw in [{}, dict(protocol="topk", ask=3), dict(nu=0.1, newW=newW), dict(protocol="topk", ask=3, nu=0.1, newW=newW, cost=C1[0], rcost=C1[1]),
                   dict(cost=C1[0], rcost=C1[1])]:
            a = s1.run(W, "T1+", cs, **kw)
            for reg in s4.DECOYS:
                b = s4.run(W, reg, cs, **kw); n += 1
                if not same(a, b, quotes=False):
                    bad += 1
check("囮 M・Rep の手の並び・形・お金・支払いは同じ聞く範囲の TP+ と一致", bad == 0, f"{n - bad}/{n} 局")

# 4. M の知識
W = composition("二群", 5)
a = s1.run(W, "T1+", 5); b = s4.run(W, "M+", 5)
check("M：清算所が知る言い値は、TP+ の提案者が聞く言い値と同じ（年・戸・係数）",
      sorted((t, j, p) for (t, i, j, p) in a["quotes"]) == sorted(b["ch_quotes"]), (len(a["quotes"]), len(b["ch_quotes"])))
check("M：提案者が知る係数は、TP+ の提案者が聞く係数の一部", set(b["quotes"]) <= set(a["quotes"]) and len(b["quotes"]) < len(a["quotes"]),
      (len(b["quotes"]), len(a["quotes"])))
# 一戸だけが損をする手でだけ知る：手ごとの損をする戸の数を数え直す
evs = [ev for ev in b["events"]]
check("M：手には損をする側の同意がある（cons、支える世帯に損をする全員）",
      all(ev["cons"] and {j for j in range(9) if j != ev["i"] and ev["dS"][j] < -1e-12} <= ev["supp"] for ev in evs if ev["ok"]))
check("M：損をする戸はそれぞれ自分の言い値を受け取る",
      all(abs(ev["pay"][j] - max(0.0, -ev["dS"][j] / s0.MU)) < 1e-9 for ev in evs if ev["ok"] for j in range(9) if j != ev["i"]))

# 5. Rep
newW = lambda rng, i: composition("α=3", int(rng.integers(10**6, 10**7)))[i]
W = composition("α=3", 3)
moved = 0; ok5 = True
for cs in range(20):
    r = s4.run(W, "Rep+", cs, nu=0.1, newW=newW)
    reps = r["reps"]
    for (t, rp), (t2, rp2) in zip(reps[:-1], reps[1:]):
        if (t2, rp) not in [(tt, u) for (tt, u) in r["turns"]]:
            ok5 = False                       # 代表が替わったのは、前の代表の住戸が入れ替わった年だけ
    # 代表の住戸が入れ替わった年には、必ず代表が替わっている
    cur = reps[0][1]; chg = {t2: rp2 for (t2, rp2) in reps[1:]}
    for t in range(30):
        if (t, cur) in r["turns"] and t not in chg:
            ok5 = False
        cur = chg.get(t, cur)
    moved += len(reps) - 1
    if any(ev["ok"] and ev["rep"] is None for ev in r["events"]):
        ok5 = False
check("Rep：代表が替わるのは代表の住戸が入れ替わった年だけ", ok5, f"20 局で代表の交代 {moved} 回")
r = s4.run(W, "Rep+", 3)
rp = r["reps"][0][1]
check("Rep：入れ替わりがなければ、聞いた戸は代表か、打った手の損をする戸が一戸だけの要求した世帯",
      all(i == rp or sum(1 for j2 in range(9) if j2 != i and ev["dS"][j2] < -1e-12) == 1
          for (t, i, j, p) in r["quotes"] for ev in r["events"] if ev["t"] == t and ev["i"] == i and ev["ok"]),
      (sum(1 for q in r["quotes"] if q[1] == rp), len(r["quotes"])))
a = s1.run(W, "T1+", 3)
know_rep = {(j, p) for (t, i, j, p) in r["quotes"] if i == rp}
know_tp = {(j, p) for (t, i, j, p) in a["quotes"]}
check("Rep：代表は TP+ の提案者たちが聞いた係数（の和集合）をすべて知る", know_tp - {(rp, x) for x in "asq"} <= know_rep, (len(know_rep), len(know_tp)))

# 6. vstar2
for reg in ["TP+", "TP+top3", "N3", "BR", "改良版top3"]:
    W = composition("二群", 7); rr, kw, sa = s2_cand.REGS[reg]
    r1 = s1.run(W, rr, 7, **kw); a = vstar.record(r1, W, sa, s2_cand.F0); b = vstar2.record2(r1, W, sa, s2_cand.F0)
    check(f"vstar2.record2 は囮でない局で vstar.record と同じ（{reg}）", json.dumps(a) == json.dumps(b))
recs = [s4_confirm.one(("二群", cs, 0.0, "なし", "M", 0)) for cs in range(30)]
v0 = vstar2.value(recs, "Xn")["V"]; v1 = vstar2.value(recs, "Xn", count_ch=True)["V"]
check("清算所を作者に数えた V*n は、数えない値以下", v1 <= v0 + 1e-12, (round(v0, 3), round(v1, 3)))
tp = [s4_confirm.one(("二群", cs, 0.0, "なし", "TP+", 0)) for cs in range(30)]
check("M と TP+ は形の項（Xn）と η が同じで、作者の項だけが違う",
      all(x["Xn"] == y["Xn"] and x["Xc"] == y["Xc"] and x["Z"] == y["Z"] for x, y in zip(recs, tp)))
# 7. 感度の分析のための Gbest を足しても、ほかの値は変わらない。代表の役の枠は代表の知識を持つ
a = s4_confirm.one(("α=3", 5, 0.1, "C1", "TP+top3", 0)); b = s4_confirm.one(("α=3", 5, 0.1, "C1", "TP+top3", 0, True))
check("Gbest を足しても記録のほかの値は同じ", "Gbest" in b and json.dumps(a) == json.dumps({k: v for k, v in b.items() if k != "Gbest"}))
W = composition("二群", 2)
r = s4.run(W, "Rep+", 2, protocol="topk", ask=3)
kn = vstar2.rep_knowledge(r); cur, _, _ = vstar.knowledge(r)
check("代表の役の枠の知識は、代表の住戸の知識と同じ（入れ替わりなし）", kn == cur[r["reps"][0][1]], (len(kn), max(len(x) for x in cur)))
print("すべて合格" if ok else "不合格あり")
