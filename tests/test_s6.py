"""s6（聞いた言い値だけを知る仲介者）の試験。tests/test_s5.py が今のまま合格することを先に確かめてから使う。
1. 仲介者を置かない s6.run は、s5.run・s3.run と記録まで一致する（13 の決め方 × 部品の組。s5 の運用役を置いた局も含む）。
2. 全知の仲介者 BR は、近い戸だけの条件でも全員に届く条件と同じ局になる（BR は聞く範囲を使わない）。
3. 届かなかった手だけを持ち込む仲介者は、費用なし・全員に届く条件では一度も相談されず、局は基準と同じ（定義から）。
4. 仲介者の束：損をする世帯は言い値ちょうどを受け取り、得をする世帯はどれも払った後も得が残り、満足の動く世帯すべてが同意の世帯に入る。
   近い戸だけの条件では、世帯が自分で打つ手の損をする相手はいつも近い戸で、遠い戸を含む手は仲介者の束だけ。
5. 仲介者が知るのは、相談で持ち込まれた手で満足の動く世帯の係数だけ。住み手が束から知るのは、束が二戸のときの相手の係数だけ。
6. 推定器：〔数〕≤〔外〕。BR は入れ替わりなしで〔数〕＝0、費用なしで η＝1（定義から）。drop_central は BR でない局の値を変えない。
PYTHONPATH=. python3 tests/test_s6.py   （約 10 分。2 つの核を使う）
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
from trade import s0, s1, s3, s5, s6, vstar, vstar3, vstar4
from trade.s1 import NEAR, revealed
from experiments.explore import composition
from tests.test_s5 import same

ok = True
IMP = dict(credit=20.0, settle=True, reconsent="all")
C1 = (dict(e=1.0, k=0.25, o=0.25), dict(e=0.5, k=0.1, o=0.1))
NEARV = dict(G=0, q_far=0.0, tau=0.0)


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
        for kw in [{}, dict(protocol="topk", ask=3), dict(protocol="near"), dict(IMP, nu=0.1, newW=newW),
                   dict(IMP, protocol="topk", ask=3, turnover=dict(kind="geo", nu=0.3), newW=newW),
                   dict(protocol="topk", ask=3, cost=C1[0], rcost=C1[1], nu=0.1, newW=newW),
                   dict(protocol="topk", ask=3, venue=NEARV, nu=0.1, newW=newW),
                   dict(protocol="topk", ask=3, turnover=dict(kind="mix"), newW=newW, restore="R0", cost=C1[0], rcost=C1[1], horizon=10),
                   dict(protocol="topk", ask=3, nu=0.1, newW=newW, owner=dict(kind="ind", role="trade", beta=0.5, gamma=1.0),
                        venue=dict(G=0, q_far=0.4, tau=0.5))]:
            a = s5.run(W, reg, cs, **kw); b = s6.run(W, reg, cs, **kw); n += 1
            bad += not same(a, b)
            c_ = s3.run(W, reg, cs, **kw); n += 1
            bad += not same(c_, b)
    # s5 の運用役を置いた局
    for acts in (lambda: [s5.Recorder("amount")], lambda: [s5.Verifier("alt")], lambda: [s5.Designer(s5.RICH)], lambda: [s5.Recorder("none")]):
        for kw in [dict(protocol="topk", ask=3), dict(IMP, protocol="topk", ask=3, turnover=dict(kind="geo", nu=0.1), newW=newW)]:
            a = s5.run(W, "T1+", cs, actors=acts(), **kw); b = s6.run(W, "T1+", cs, actors=acts(), **kw); n += 1
            bad += not same(a, b) or a["actor_facts"] != b["actor_facts"] or a["ledger"] != b["ledger"]
    return n, bad


def t_bundle(args):
    """仲介者の局を走らせ、束の約束を確かめる"""
    pat, cs, nu, cond, base, scope = args
    W = composition(pat, cs)
    kw = dict(protocol="topk", ask=3)
    if base == "改良版top3":
        kw.update(IMP)
    if cond == "費用1倍":
        kw.update(cost=C1[0], rcost=C1[1])
    if cond == "近い戸だけ":
        kw.update(venue=NEARV)
    if nu > 0:
        kw.update(turnover=dict(kind="geo", nu=nu), newW=nw(pat))
    m = s6.Mediator(scope)
    r = s6.run(W, "T1+", cs, actors=[m], **kw)
    base_r = s6.run(W, "T1+", cs, **kw)
    res = dict(n_bundle=m.n_bundle, n_consult=m.n_consult, pay_ok=True, supp_ok=True, near_ok=True, far_in_bundle=0,
               learn_ok=True, quote_ok=True, same_as_base=same(r, base_r), nfacts=len(m.facts))
    MU = s0.MU
    # 束の手については、仲介者が束の世帯すべての係数を知っていることを確かめる
    known = {(t, j, c) for (t, j, c, _) in m.facts}
    for ev in r["events"]:
        if not ev.get("ok") or ev.get("part") is None or "dS" not in ev:
            continue
        i, dS, pay = ev["i"], ev["dS"], ev["pay"]
        harmed = {j for j in range(9) if j != i and dS[j] < -1e-12}
        gain = {j for j in range(9) if dS[j] > 1e-12}
        if ev.get("drafter") == "mediator":
            for j in harmed:
                res["pay_ok"] &= abs(pay[j] - (-dS[j] / MU)) < 1e-12
            for g in gain - {i}:
                res["pay_ok"] &= pay[g] <= 0 and -pay[g] < dS[g] / MU + 1e-12
            res["pay_ok"] &= dS[i] / MU - (pay.sum() + ev["cost"]) > 1e-12
            res["supp_ok"] &= ev["supp"] == ({i} | harmed | gain) and ev["cons"] and ev["bundle"] == len(harmed | gain | {i})
            res["far_in_bundle"] += any(j not in NEAR[i] for j in harmed)
            mv_kind = ev["part"][0]
            for j in harmed | gain:
                c = "a" if (mv_kind == "e" and j == ev["part"][1]) or mv_kind == "k" else ("s" if mv_kind == "e" else "q")
                res["learn_ok"] &= (ev["t"], j, c) in known
        elif ev.get("drafter") == "self" and cond == "近い戸だけ":
            res["near_ok"] &= harmed <= NEAR[i]
    return res


if __name__ == "__main__":
    pool = Pool(2)
    # 1
    res = pool.map(t1, [(p, cs) for p in ["α=3", "二群"] for cs in range(3)])
    n = sum(x[0] for x in res); bad = sum(x[1] for x in res)
    check("仲介者を置かない s6.run は s5.run・s3.run と記録まで一致（13 の決め方 × 9 の部品の組、s5 の役者を置いた局を含む）", bad == 0, f"{n - bad}/{n} 局")
    # 2
    bad = n = 0
    for pat in ["α=3", "二群"]:
        for cs in range(6):
            W = composition(pat, cs)
            for reg in ("BR", "BRdyn"):
                for kw in ({}, dict(cost=C1[0], rcost=C1[1]), dict(nu=0.1, newW=nw(pat))):
                    a = s6.run(W, reg, cs, **kw); b = s6.run(W, reg, cs, venue=NEARV, protocol="topk", ask=3, **kw); n += 1
                    bad += not same(a, b)
    check("BR・BRdyn は、近い戸だけの条件でも全員に届く条件と同じ局（聞く範囲を使わない）", bad == 0, f"{n - bad}/{n} 局")
    # 3〜5
    T = [(p, cs, nu, cond, base, scope) for p in ["α=3", "二群"] for cs in range(8) for nu in (0.0, 0.1)
         for cond in ("基準", "費用1倍", "近い戸だけ") for base in ("TP+top3", "改良版top3") for scope in ("stuck", "all")]
    R = pool.map(t_bundle, T, chunksize=4)
    st0 = [r for a, r in zip(T, R) if a[3] == "基準" and a[5] == "stuck"]
    check("届かなかった手だけを持ち込む仲介者は、費用なし・全員に届く条件では相談されず、局は基準と同じ（定義から）",
          all(r["n_consult"] == 0 and r["nfacts"] == 0 and r["same_as_base"] for r in st0), len(st0))
    nb = sum(r["n_bundle"] for r in R)
    check("束：損をする世帯は言い値ちょうど、得をする世帯はどれも払った後も得が残る", all(r["pay_ok"] for r in R) and nb > 0, f"束 {nb}")
    check("束：満足の動く世帯すべてが同意の世帯に入る（束の大きさも記録と一致）", all(r["supp_ok"] for r in R))
    check("近い戸だけの条件で、世帯が自分で打つ手の損をする相手はいつも近い戸", all(r["near_ok"] for r in R))
    nf = sum(r["far_in_bundle"] for a, r in zip(T, R) if a[3] == "近い戸だけ")
    check("近い戸だけの条件で、遠い戸の損を含む手は仲介者の束で打たれる", nf > 0, f"遠い戸を含む束 {nf}")
    check("仲介者は、束の世帯すべての係数を知っている（learn で記録）", all(r["learn_ok"] for r in R))
    # 5 住み手が束から知ること：二戸の束だけ。三戸以上の束では、言い値の記録（quotes）が仲介者のない手続きより増えない
    okq = True; n3 = n2 = 0
    for cs in range(10):
        W = composition("α=3", cs)
        for cond_kw in (dict(venue=NEARV), dict(cost=C1[0], rcost=C1[1])):
            m = s6.Mediator("all")
            r = s6.run(W, "T1+", cs, actors=[m], protocol="topk", ask=3, nu=0.1, newW=nw("α=3"), **cond_kw)
            byt = {}
            for q in r["quotes"]:
                byt.setdefault((q[0], q[1]), []).append(q)
            for ev in r["events"]:
                if ev.get("drafter") == "mediator":
                    if ev["bundle"] >= 3:
                        n3 += 1
                        others = (ev["supp"] - {ev["i"]})
                        # 三戸以上の束の相手の係数は、提案者の言い値の記録に「束から」は足されない
                        # （同じ年に提案者が自分で聞いた手の分は残るので、束の世帯のうち近くない戸は記録にない）
                        if cond_kw.get("venue"):
                            okq &= not any(q[2] in others and q[2] not in NEAR[ev["i"]] for q in byt.get((ev["t"], ev["i"]), []))
                    else:
                        n2 += 1
                        other = list(ev["supp"] - {ev["i"]})[0]
                        okq &= any(q[2] == other for q in byt.get((ev["t"], ev["i"]), []))
    check("住み手が束から知るのは、二戸の束の相手の係数だけ（三戸以上の束では遠い戸の係数を知らない）", okq and n3 > 0, f"三戸以上 {n3}・二戸 {n2}")
    # 6 推定器
    T2 = [("α=3", cs, nu, cond, "改良版top3", sc) for cs in range(30) for nu in (0.0, 0.1) for cond in ("近い戸だけ",) for sc in ("stuck", "all")]
    from experiments import s6_prof2 as P
    recs = pool.map(P.one, [(a[0], a[1], a[2], a[3], a[4], "仲介者" if a[5] == "stuck" else "広く聞く仲介者") for a in T2], chunksize=4)
    okc = True
    for nu in (0.0, 0.1):
        for arm in ("仲介者", "広く聞く仲介者"):
            R_ = sorted([r for r in recs if r["nu"] == nu and r["arm"] == arm], key=lambda r: r["cs"])
            okc &= vstar3.population3(R_, "Xn", count=("broker",))["V"] <= vstar3.population3(R_, "Xn")["V"] + 1e-12
            okc &= "broker" in R_[0]["GA"]
    check("〔数〕≤〔外〕（仲介者は役の枠 broker で数える）", okc)
    brs = pool.map(P.one, [(p, cs, 0.0, c, "参照", "BR") for p in ["α=3", "二群"] for cs in range(30) for c in ("基準", "費用1倍")])
    okb = True; oke = True
    for p in ["α=3", "二群"]:
        for c in ("基準", "費用1倍"):
            R_ = sorted([r for r in brs if r["pat"] == p and r["cond"] == c], key=lambda r: r["cs"])
            okb &= vstar3.population3(R_, "Xn")["V"] == 0.0
            if c == "基準":
                oke &= all(abs(r["eta"] - 1) < 1e-9 for r in R_)
    check("BR は入れ替わりなしで〔数〕＝0（全員の係数を知る中央の作者）、費用なしで η＝1（定義から）", okb and oke)
    base = sorted(pool.map(P.one, [("α=3", cs, 0.0, "近い戸だけ", "改良版top3", "基準") for cs in range(30)]), key=lambda r: r["cs"])
    check("drop_central は BR でない局の値を変えない", vstar3.population3(vstar4.drop_central(base), "Xn")["V"] == vstar3.population3(base, "Xn")["V"])
    pool.close()
    print("すべて合格" if ok else "不合格あり")
