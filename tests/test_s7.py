"""s7（性向を知りうる運用役）の試験。tests/test_s6.py が今のまま合格することを先に確かめてから使う。
1. 運用役を置かない s7.run は、s6.run・s5.run・s3.run と記録まで一致する（13 の決め方 × 部品の組。仲介者・s5 の運用役を置いた局を含む）。
   運用役を置かない腕の一局の記録（record3）は、職能 2 の確かめ用の記録（out/s6/main.jsonl、凍結）と同じ（同じ種・同じ条件）。
2. 聞く範囲を選ぶ者を置いた局は、選んだ範囲を固定した運用役なしの局と記録まで一致する（選ぶ者は手を打たない）。
   性向を知る選ぶ者は、竣工の時の 27 の係数を知り、入れ替わった世帯の係数は無効になる。分布だけを知る選ぶ者は何も知らない。
3. 影響を見せる道具は、全員に届く条件では局を変えない（定義から）。近い戸だけの条件では、全員に届く条件の局と記録まで一致し、
   近い戸だけの基準の局（届かない手がある）とは違う。物理の量で見せる道具は何も知らず、満足の量で見せる道具は使われた年ごとに全員の係数を知る。
4. 全知の計画者（regime P、帰無 N3）は、費用なしで η＝1、〔数〕＝0（中央の作者）、〔外〕＝0（損をする側の同意がないので、
   同意で支えられた部位がない）。
5. 推定器：〔数〕≤〔外〕。何も知らない運用役では〔数〕＝〔外〕。入れ替わりなしでは、性向を知る選ぶ者と満足の量で見せる道具の〔数〕＝0。
   actor_I は役の枠だけの相互情報量で、何も知らない運用役では 0、全員を知る運用役では 1。
6. 選び方：choose_knowing は点ごとの値の最大、choose_distribution は集団の値の最大（同じなら基準の上位3手、なければ前の範囲）。
PYTHONPATH=. python3 tests/test_s7.py   （約 10 分。2 つの核を使う）
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
from trade import s0, s3, s5, s6, s7, vstar, vstar3, vstar4, vstar5
from experiments.explore import composition
from tests.test_s5 import same

ok = True
IMP = dict(credit=20.0, settle=True, reconsent="all")
C1 = (dict(e=1.0, k=0.25, o=0.25), dict(e=0.5, k=0.1, o=0.1))
NEARV = dict(G=0, q_far=0.0, tau=0.0)
F0 = s0.f_design()


def check(name, cond, info=""):
    global ok
    print(("合格 " if cond else "不合格 ") + name + (f"  {info}" if str(info) else ""), flush=True)
    ok &= bool(cond)


def nw(pat):
    return lambda rng, i: composition(pat, int(rng.integers(10**6, 10**7)))[i]


def t1(args):
    pat, cs = args
    W = composition(pat, cs)
    newW = nw(pat)
    n = bad = 0
    for reg in ["U", "V½", "V¾", "T0+", "T1+", "AI+", "P", "BR", "BRdyn", "R", "Rc", "F", "T1"]:
        for kw in [{}, dict(protocol="topk", ask=3), dict(protocol="topk", ask=1), dict(protocol="near"), dict(IMP, nu=0.1, newW=newW),
                   dict(IMP, protocol="topk", ask=3, turnover=dict(kind="geo", nu=0.3), newW=newW),
                   dict(protocol="topk", ask=3, cost=C1[0], rcost=C1[1], nu=0.1, newW=newW),
                   dict(protocol="topk", ask=3, venue=NEARV, nu=0.1, newW=newW),
                   dict(protocol="topk", ask=3, nu=0.1, newW=newW, owner=dict(kind="ind", role="trade", beta=0.5, gamma=1.0),
                        venue=dict(G=0, q_far=0.4, tau=0.5))]:
            a = s6.run(W, reg, cs, **kw); b = s7.run(W, reg, cs, **kw); n += 1
            bad += not same(a, b)
            if reg in ("T1+", "P", "BR"):
                c_ = s5.run(W, reg, cs, **kw); d_ = s3.run(W, reg, cs, **kw); n += 2
                bad += (not same(c_, b)) + (not same(d_, b))
    for acts in (lambda: [s6.Mediator("stuck")], lambda: [s6.Mediator("all")], lambda: [s5.Recorder("amount")], lambda: [s5.Verifier("alt")]):
        for kw in [dict(protocol="topk", ask=3, venue=NEARV), dict(IMP, protocol="topk", ask=3, turnover=dict(kind="geo", nu=0.1), newW=newW)]:
            a = s6.run(W, "T1+", cs, actors=acts(), **kw); b = s7.run(W, "T1+", cs, actors=acts(), **kw); n += 1
            bad += not same(a, b) or a["actor_facts"] != b["actor_facts"]
    return n, bad


def t_scope(args):
    """選ぶ者を置いた局と、選んだ範囲を固定した局"""
    pat, cs, nu, base, scope, knows, cond = args
    W = composition(pat, cs)
    kw = dict(IMP) if base == "改良版top3" else {}
    if cond == "近い戸だけ":
        kw["venue"] = NEARV
    if nu > 0:
        kw.update(turnover=dict(kind="geo", nu=nu), newW=nw(pat))
    a = s7.run(W, "T1+", cs, **kw, **s7.PROTOCOLS[scope])
    sc = s7.Scope(scope, knows)
    b = s7.run(W, "T1+", cs, actors=[sc], protocol="topk", ask=3, **kw)
    rec = vstar3.record3(b, W, None, F0)
    stay = [j for j in range(9) if not any(u == j for (_, u) in b["turns"])]
    want = {(j, c) for j in stay for c in "asq"} if knows else set()
    return dict(same=same(a, b), kn_ok=vstar3.actor_knowledge(b, "scope") == want, nf=rec["actor_nfacts"]["scope"])


def t_tool(args):
    pat, cs, nu, base, shows = args
    W = composition(pat, cs)
    kw = dict(IMP) if base == "改良版top3" else {}
    kw.update(protocol="topk", ask=3)
    if nu > 0:
        kw.update(turnover=dict(kind="geo", nu=nu), newW=nw(pat))
    full = s7.run(W, "T1+", cs, **kw)
    near = s7.run(W, "T1+", cs, venue=NEARV, **kw)
    t_full = s7.run(W, "T1+", cs, actors=[s7.Tool(shows)], **kw)
    tool = s7.Tool(shows)
    t_near = s7.run(W, "T1+", cs, actors=[tool], venue=NEARV, **kw)
    kn = vstar3.actor_knowledge(t_near, "tool")
    years = sorted({t for (t, _) in tool.full})
    used = sorted({ev["t"] for ev in t_near["events"] if ev.get("drafter") == "self"})
    return dict(same_full=same(full, t_full), same_near_full=same(full, t_near), near_differs=not same(near, full) or near["extra"]["unreached"] == 0,
                unreached_near=near["extra"]["unreached"], facts=len(tool.facts), nfull=len(tool.full), nkn=len(kn), years=years,
                years_ok=set(used) <= set(years), n_shown=tool.n_shown)


def t_planner(args):
    pat, cs, nu = args
    W = composition(pat, cs)
    kw = dict(turnover=dict(kind="geo", nu=nu), newW=nw(pat)) if nu > 0 else {}
    r = s7.run(W, "P", cs, **kw)
    rec = vstar3.record3(r, W, None, F0)
    rec.update(cs=cs)
    return rec


def one_rec(a):
    from experiments import s7_prof3 as P
    return P.one(a)


if __name__ == "__main__":
    pool = Pool(2)
    # 1
    res = pool.map(t1, [(p, cs) for p in ["α=3", "二群"] for cs in range(3)])
    n = sum(x[0] for x in res); bad = sum(x[1] for x in res)
    check("役者を置かない s7.run は s6.run（と s5.run・s3.run）と記録まで一致（13 の決め方 × 9 の部品の組、仲介者・s5 の役者を置いた局を含む）",
          bad == 0, f"{n - bad}/{n} 局")
    want = {}
    for line in open("out/s6/main.jsonl"):
        r = json.loads(line)
        if r["arm"] in ("基準", "仲介者") and r["cs"] < 504 and r["cond"] in ("基準", "近い戸だけ") and r["nu"] in (0.0, 0.1):
            want[(r["base"], r["arm"], r["cond"], r["pat"], r["nu"], r["cs"])] = r
    T = [(k[3], k[5], k[4], k[2], k[0], k[1], None) for k in want]
    got = pool.map(one_rec, T, chunksize=2)
    bad = 0
    for k, g in zip(want, got):
        w = want[k]
        bad += any(g[f] != w[f] for f in ("Z", "X", "Xc", "Xn", "G", "eta", "moves", "quotes", "unreached")) or g.get("GA") != w.get("GA")
    check("役者を置かない腕と仲介者の腕の一局の記録は、職能 2 の確かめ用の記録（out/s6/main.jsonl）と同じ", bad == 0 and len(want) == 96, f"{len(want) - bad}/{len(want)} 局")
    # 2
    T = [(p, cs, nu, base, sc, kn, cond) for p in ["α=3", "二群"] for cs in range(4) for nu in (0.0, 0.1) for base in ("TP+top3", "改良版top3")
         for sc in s7.ORDER for kn in (False, True) for cond in ("基準", "近い戸だけ")]
    R = pool.map(t_scope, T, chunksize=4)
    check("聞く範囲を選ぶ者を置いた局は、選んだ範囲を固定した役者なしの局と記録まで一致（手を打たない）", all(r["same"] for r in R), f"{len(R)} 局")
    check("性向を知る選ぶ者は竣工の時の 27 の係数を知り、入れ替わった世帯の係数は無効。分布だけを知る選ぶ者は何も知らない",
          all(r["kn_ok"] for r in R) and all(r["nf"] == 27 for a, r in zip(T, R) if a[5] and a[2] == 0.0)
          and all(r["nf"] == 0 for a, r in zip(T, R) if not a[5]))
    # 3
    T = [(p, cs, nu, base, sh) for p in ["α=3", "二群"] for cs in range(6) for nu in (0.0, 0.1, 0.3) for base in ("TP+top3", "改良版top3")
         for sh in ("physical", "satisfaction")]
    R = pool.map(t_tool, T, chunksize=4)
    check("影響を見せる道具は、全員に届く条件では局を変えない（定義から）", all(r["same_full"] for r in R), f"{len(R)} 局")
    check("近い戸だけの条件で道具を置いた局は、全員に届く条件の局と記録まで一致（届かない戸への影響が見え、損をする世帯が言い値を言いに来る）",
          all(r["same_near_full"] for r in R))
    nun = sum(r["unreached_near"] for r in R)
    check("近い戸だけの基準の局には、届かなかった手がある（道具の効き目が 0 でない）", nun > 0 and all(r["near_differs"] for r in R), f"届かなかった手 {nun}")
    check("物理の量で見せる道具は何も知らない（facts・full とも空）", all(r["facts"] == 0 and r["nfull"] == 0 and r["nkn"] == 0 for a, r in zip(T, R) if a[4] == "physical"))
    check("満足の量で見せる道具は、使われた年ごとに全員の係数を読み、入れ替わりなしでは 30 年目に 27 の係数を知る",
          all(r["years_ok"] and r["nfull"] == len(r["years"]) and r["facts"] == 0 for a, r in zip(T, R) if a[4] == "satisfaction")
          and all(r["nkn"] == 27 for a, r in zip(T, R) if a[4] == "satisfaction" and a[2] == 0.0))
    # 4
    okp = True
    TP = [(p, cs, nu) for p in ["α=3", "二群"] for cs in range(30) for nu in (0.0, 0.1)]
    byc = {}
    for (p, cs, nu), r in zip(TP, pool.map(t_planner, TP)):
        byc.setdefault((p, nu), []).append(r)
    for k_, R_ in byc.items():
        okp &= all(abs(r["eta"] - 1) < 1e-9 for r in R_)
        okp &= vstar3.population3(R_, "Xn")["V"] == 0.0
        okp &= vstar3.population3(vstar4.drop_central(R_), "Xn")["V"] == 0.0
        okp &= all(r["Ln"] == 0.0 for r in R_)
    check("全知の計画者は費用なしで η＝1、〔数〕＝0、〔外〕＝0（同意で支えられた部位がない）", okp)
    # 5
    from experiments import s7_prof3 as E
    T = [(p, cs, nu, "基準", "改良版top3", arm, (s7.ORDER[cs % 4] if arm.startswith("選ぶ者") else None))
         for p in ["α=3", "二群"] for cs in range(30) for nu in (0.0, 0.1) for arm in ("選ぶ者:分布", "選ぶ者:性向", "道具:物理", "道具:満足")]
    recs = pool.map(one_rec, T, chunksize=4)
    by = {}
    for a, r in zip(T, recs):
        by.setdefault((a[0], a[2], a[5]), []).append(r)
    okc = ok0 = okz = oki = True
    for (p, nu, arm), R_ in by.items():
        R_.sort(key=lambda r: r["cs"])
        vo = vstar3.population3(R_, "Xn")["V"]; vc = vstar3.population3(R_, "Xn", count=(E.KIND[arm],))["V"]
        okc &= vc <= vo + 1e-12
        if arm in ("選ぶ者:分布", "道具:物理"):
            ok0 &= vc == vo
            oki &= vstar5.actor_I(R_, E.KIND[arm]) == 0.0
        elif nu == 0.0:
            okz &= vc == 0.0
            oki &= abs(vstar5.actor_I(R_, E.KIND[arm]) - 1.0) < 1e-9
    check("〔数〕≤〔外〕（役者は役の枠で数える）", okc)
    check("何も知らない役者（分布だけを知る選ぶ者・物理の量で見せる道具）では〔数〕＝〔外〕、役の枠の相互情報量は 0", ok0 and oki)
    check("入れ替わりなしでは、性向を知る選ぶ者と満足の量で見せる道具の〔数〕＝0、役の枠の相互情報量は 1（定義から）", okz)
    # 6
    fake = {s: [dict(cs=c) for c in range(5)] for s in s7.ORDER}
    import trade.vstar5 as V5
    orig = vstar.pointwise
    tab = {"全員": [0, 1, 0, 0, 2], "上位1手": [0, 2, 0, 1, 2], "上位3手": [1, 2, 0, 1, 1], "近い戸だけ": [0, 0, 0, 3, 0]}
    vstar.pointwise = lambda recs, key: np.array(tab[[s for s in s7.ORDER if fake[s] is recs][0]], float)
    ch, _ = V5.choose_knowing(fake, s7.ORDER)
    vstar.pointwise = orig
    check("choose_knowing は建物ごとに点ごとの値の最大（同じなら上位3手、上位3手が最大でなければ前の範囲）を選ぶ",
          ch == {0: "上位3手", 1: "上位3手", 2: "上位3手", 3: "近い戸だけ", 4: "全員"}, ch)
    R0 = sorted([r for r in recs if r["arm"] == "選ぶ者:分布" and r["pat"] == "二群" and r["nu"] == 0.0], key=lambda r: r["cs"])
    by2 = {s: R0 for s in s7.ORDER}
    best, v = V5.choose_distribution(by2, s7.ORDER)
    check("choose_distribution は集団の値の最大（同じなら上位3手）を選ぶ", best == "上位3手" and len(set(v.values())) == 1)
    pool.close()
    print("すべて合格" if ok else "不合格あり")
