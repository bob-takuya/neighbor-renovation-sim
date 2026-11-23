"""s5・vstar3 の試験（一致の試験を含む）。
1. 運用役を置かない既定の設定で、s5.run は s3.run・s1.run と同じ記録を返す（多くの決め方と部品の組み合わせで）。
2. 観察 2 の局：s5 で写した手順は、観察 2 の確かめ用の記録 out/s3/main.jsonl の行と一字一句同じ JSON を返す
   （四つの群のすべての条件と決め方、cseed 400・459）。主なセル（入れ替わりの群の TP+top3・改良版top3、年 10 %・30 %、両パターン、
   60 構成）では、V*n の点推定と区間、η の平均が out/s3/main_cells.json と一字一句同じ。
3. 確証の局：s5 で写した手順は、確証の記録 out/s4/main.jsonl の TP+top3・改良版top3 の行と一字一句同じ（cseed 300・357・399、8 セル）。
   α＝3・入れ替わりなし・費用なしの TP+top3（100 構成）では、V*n と区間が out/s4/judge.json と一字一句同じ。
4. 効かない運用役（今の盤の設計者・同意だけ記録・額まで記録・TP+top3 の記録なし）を置いても、手の並び・形・お金は基準と同じ。
5. 狭い盤では張り出しが変わらない。検証者のもとでは、どの局でも同じ列の張り出しが三層にならない。代わりの手は二つの部位の記録を残す。
6. 戻せる手の多い盤：上下の部位が 0 なら満足は今の盤と同じ。最適 wstar は総当たりの最適と一致（部位ごとに分かれる）。
7. vstar3：運用役のない局で vstar.record と同じ。記録を読まない読み方で vstar.knowledge と同じ。〔数〕≤〔外〕。
   性向を知らない運用役（盤の設計者・同意だけの記録者・検証者）の役の枠は、定義から値を変えない（〔数〕＝〔外〕）。
8. 記録者：額の記録は、提案者が聞いた言い値の一部（打たれた手の分）。記録なしでは〔再〕〔清〕が起きない。
PYTHONPATH=. python3 tests/test_s5.py         （約 15 分。2 つの核を使う）
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
from trade import s0, s1, s3, s5, vstar, vstar3
from experiments.explore import composition

ok = True


def check(name, cond, info=""):
    global ok
    print(("合格 " if cond else "不合格 ") + name + (f"  {info}" if str(info) else ""), flush=True)
    ok &= bool(cond)


def same(a, b, quotes=True):
    if not (all(np.array_equal(a[x], b[x]) for x in "eko") and all(np.array_equal(a[x], b[x]) for x in ["S", "m", "c", "b", "Wt", "U"])):
        return False
    if len(a["events"]) != len(b["events"]) or a["turns"] != b["turns"] or a["recon"] != b["recon"]:
        return False
    if quotes and a["quotes"] != b["quotes"]:
        return False
    for x, y in zip(a["events"], b["events"]):
        for f in ("part", "ok", "nv", "supp", "i", "t", "cons", "move", "type", "val", "drafter"):
            if x.get(f) != y.get(f):
                return False
        if "pay" in x and not np.array_equal(x["pay"], y["pay"]):
            return False
    return a["spent"] == b["spent"] and (a.get("extra") is None or b.get("extra") is None or a["extra"] == b["extra"])   # s1 は extra を返さない


IMP = dict(credit=20.0, settle=True, reconsent="all")
C1 = (dict(e=1.0, k=0.25, o=0.25), dict(e=0.5, k=0.1, o=0.1))


def t1(args):
    pat, cs = args
    W = composition(pat, cs)
    newW = lambda rng, i, pat=pat: composition(pat, int(rng.integers(10**6, 10**7)))[i]
    n = bad = 0
    for reg in ["U", "V½", "V¾", "T0+", "T1+", "AI+", "P", "BR", "BRdyn", "R", "Rc", "F", "T1"]:
        for kw in [{}, dict(protocol="topk", ask=3), dict(protocol="near"), dict(IMP, nu=0.1, newW=newW),
                   dict(IMP, protocol="topk", ask=3, turnover=dict(kind="geo", nu=0.3), newW=newW),
                   dict(protocol="topk", ask=3, turnover=dict(kind="mix"), newW=newW, restore="R0", cost=C1[0], rcost=C1[1], horizon=10),
                   dict(protocol="topk", ask=3, turnover=dict(kind="renew", nu=0.18), newW=newW, restore="Rdk", cost=C1[0], rcost=C1[1], horizon=10),
                   dict(protocol="topk", ask=3, nu=0.1, newW=newW, owner=dict(kind="ind", role="trade", beta=0.5, gamma=1.0),
                        venue=dict(G=0, q_far=0.4, tau=0.5)),
                   dict(protocol="topk", ask=3, nu=0.1, newW=newW, owner=dict(kind="mgmt", role="veto", Lam=set(), fix="D4"),
                        venue=dict(G=0.5, alpha=0.5))]:
            if reg == "T1" and ("owner" in kw or "IMP" in kw):
                pass
            a = s3.run(W, reg, cs, **kw); b = s5.run(W, reg, cs, **kw); n += 1
            if not same(a, b):
                bad += 1
            if not kw.get("turnover") and not kw.get("owner"):
                c_ = s1.run(W, reg, cs, **kw); n += 1
                if not same(c_, b):
                    bad += 1
    return n, bad


def obs2_line(a):
    from experiments import s5_prof1 as P
    return json.dumps(P.one_obs2(a), ensure_ascii=False)


def conf_line(a):
    from experiments import s5_prof1 as P
    return json.dumps(P.one_conf(a), ensure_ascii=False)


def run_one(a):
    from experiments import s5_prof1 as P
    return P.one(a)


if __name__ == "__main__":
    from experiments import s3_rent, s5_prof1 as P
    pool = Pool(2)
    # 1. s3・s1 と一致
    res = pool.map(t1, [(p, cs) for p in ["α=3", "二群"] for cs in range(3)])
    n = sum(x[0] for x in res); bad = sum(x[1] for x in res)
    check("役者なしの s5.run は s3.run・s1.run と記録まで一致（13 の決め方 × 9 の部品の組）", bad == 0, f"{n - bad}/{n} 局")

    # 2. 観察 2 の記録と一字一句
    want = {}
    for line in open("out/s3/main.jsonl"):
        r = json.loads(line)
        if r["cs"] in (400, 459):
            want[(r["grp"], r["cond"], r["reg"], r["pat"], r["cs"])] = line.strip()
    T = [(g, c, rg, sp, p, cs) for (g, c, rg, sp) in s3_rent.conds() for p in s3_rent.PATS for cs in (400, 459)]
    got = pool.map(obs2_line, T, chunksize=8)
    bad = sum(1 for a, gl in zip(T, got) if want[(a[0], a[1], a[2], a[4], a[5])] != gl)
    check("観察 2 の手順を s5 で写すと out/s3/main.jsonl と一字一句同じ（四つの群のすべての条件、cseed 400・459）",
          bad == 0 and len(T) == 1000, f"{len(T) - bad}/{len(T)} 行")
    cells = json.load(open("out/s3/main_cells.json"))["cells"]
    for cond in ["幾何 ν=0.10", "幾何 ν=0.30"]:
        for rg in ["TP+top3", "改良版top3"]:
            spec = [sp for (g, c, r_, sp) in s3_rent.conds() if g == "T" and c == cond and r_ == rg][0]
            T = [("T", cond, rg, spec, p, cs) for p in s3_rent.PATS for cs in range(400, 460)]
            R = [json.loads(x) for x in pool.map(obs2_line, T, chunksize=8)]
            row = cells[f"T|{cond}|{rg}"]
            okv = True
            for p in s3_rent.PATS:
                Rp = sorted([r for r in R if r["pat"] == p], key=lambda r: r["cs"])
                v = vstar.population(Rp, "Xn")["V"]; lo, hi = vstar.boot(Rp, "Xn", n=100)
                okv &= [v, lo, hi] == row[f"Vn_{p}"]
            eta = float(np.mean([r["eta"] for r in R]))
            check(f"観察 2 の主なセル（{cond}・{rg}）：V*n の点推定と区間、η の平均が main_cells.json と一字一句同じ",
                  okv and eta == row["eta"], (row["Vn_α=3"][0], row["Vn_二群"][0], round(eta, 4)))

    # 3. 確証の記録と一字一句
    want = {}
    for line in open("out/s4/main.jsonl"):
        r = json.loads(line)
        if r["cs"] in (300, 357, 399) and r["reg"] in ("TP+top3", "改良版top3"):
            want[(r["pat"], r["cs"], r["nu"], r["cost"], r["reg"])] = line.strip()
    T = [(p, cs, nu, c, rg, 0, True) for (p, cs, nu, c, rg) in sorted(want)]
    got = pool.map(conf_line, T)
    bad = sum(1 for a, gl in zip(T, got) if want[a[:5]] != gl)
    check("確証の手順を s5 で写すと out/s4/main.jsonl の TP+top3・改良版top3 と一字一句同じ（cseed 300・357・399、8 セル）",
          bad == 0 and len(T) == 48, f"{len(T) - bad}/{len(T)} 行")
    J = json.load(open("out/s4/judge.json"))["values"]
    T = [("α=3", cs, 0.0, "なし", "TP+top3", 0, True) for cs in range(300, 400)]
    R = sorted([json.loads(x) for x in pool.map(conf_line, T, chunksize=8)], key=lambda r: r["cs"])
    v = vstar.population(R, "Xn")["V"]; lo, hi = vstar.boot(R, "Xn", n=1000, seed=0)
    check("確証の TP+top3（α＝3・入れ替わりなし・費用なし、100 構成）の V*n と区間が judge.json と一字一句同じ",
          [v, lo, hi] == J["TP+top3|α=3|0.0|なし"]["Vn"], (round(v, 4), round(lo, 4), round(hi, 4)))

    # 4. 効かない運用役
    n = bad = 0
    for pat in ["α=3", "二群"]:
        for cs in range(4):
            W = composition(pat, cs)
            for nu in (0.0, 0.1, 0.3):
                for base in P.BASES:
                    kw = P.kw_of(base, nu, pat)
                    a = s5.run(W, "T1+", cs, **kw)
                    acts = [[s5.Designer(s5.BASE)], [s5.Recorder("consent")], [s5.Recorder("amount")]]
                    if base == "TP+top3":
                        acts.append([s5.Recorder("none")])
                    for ac in acts:
                        b = s5.run(W, "T1+", cs, actors=ac, **kw); n += 1
                        bad += not same(a, b)
    check("効かない役者（今の盤の設計者・同意だけ・額まで・TP+top3 の記録なし）では手の並び・形・お金が基準と同じ", bad == 0, f"{n - bad}/{n} 局")

    # 5. 狭い盤・検証者
    T = [(p, cs, nu, b, arm) for p in ["α=3", "二群"] for cs in range(10) for nu in (0.0, 0.1, 0.3) for b in P.BASES
         for arm in ("狭い盤", "一律に禁止", "代わりの手")]
    R = pool.map(run_one, T, chunksize=8)
    nar = [r for r in R if r["arm"] == "狭い盤"]
    check("狭い盤では張り出しが始まりの形のまま", all(r["fin_e"] == [int(x) for x in s0.f_design()[0]] for r in nar), len(nar))
    ver = [r for r in R if r["arm"] != "狭い盤"]
    check("検証者のもとでは、30 年目に同じ列の張り出しが三層の局はない",
          all(not s5.Verifier.violates(np.array(r["fin_e"])) for r in ver), len(ver))
    pairs_all = 0; ok_p = True; ok_t = True
    for pat in ["α=3", "二群"]:
        for cs in range(12):
            W = composition(pat, cs)
            for nu in (0.0, 0.1):
                kw = P.kw_of("TP+top3", nu, pat)
                r = s5.run(W, "T1+", cs, actors=[s5.Verifier("alt")], **kw)
                evs = r["events"]
                for n_, ev in enumerate(evs):
                    if ev.get("pair"):
                        pairs_all += 1
                        prev = evs[n_ - 1]           # 組の手の一つ目（張り出す部位）は、すぐ前に同じ年・同じ提案者で並ぶ
                        ok_p &= prev["t"] == ev["t"] and prev["i"] == ev["i"] and prev["part"][0] == "e" and prev["nv"] == 1 and ev["nv"] == 0
                        ok_p &= prev["supp"] == ev["supp"] and prev["cons"] and ev["cons"]
                # 打たれた手を順に当てて、どの時点でも三層にならない（組の手は二つの記録を当て終えた時点で確かめる）
                e = s0.f_design()[0].copy()
                for n_, ev in enumerate(evs):
                    if ev["ok"] and ev.get("part") is not None and ev["part"][0] == "e":
                        e[ev["part"][1]] = ev["nv"]
                        nxt_pair = n_ + 1 < len(evs) and evs[n_ + 1].get("pair")
                        if not nxt_pair:
                            ok_t &= not s5.Verifier.violates(e)
                    elif ev["ok"] is False:
                        pass
                # 入れ替わりの再同意で戻した張り出しも当てる
                ok_t &= not s5.Verifier.violates(r["e"])
    check("代わりの手（組の手）は、張り出す部位と外す部位の二つの記録を、同じ年・同じ提案者・同じ同意で残す", ok_p and pairs_all > 0, f"組の手 {pairs_all}")
    check("代わりの手の局で、手を一つずつ当てても三層にならない", ok_t)

    # 6. 戻せる手の多い盤
    rng = np.random.default_rng(5)
    okS = okW = True
    for _ in range(30):
        W = rng.dirichlet(s0.W_BAR * 3, 9)
        e = rng.integers(0, 2, 9); k = rng.integers(-1, 2, 6); o = rng.integers(0, 2, 6)
        okS &= np.allclose(s5.RICH.S(W, e, np.concatenate([k, np.zeros(4, int)]), np.concatenate([o, np.zeros(6, int)])), s0.S_all(W, e, k, o))
    for _ in range(6):
        W = rng.dirichlet(s0.W_BAR * 3, 9)
        v, e, k, o = s5.RICH.wstar(W)
        best = -1e9
        for kv in itertools.product((-1, 0, 1), repeat=4):
            for ov in itertools.product((0, 1), repeat=6):
                kk = k.copy(); oo = o.copy(); kk[6:] = kv; oo[6:] = ov
                best = max(best, s5.RICH.S(W, e, kk, oo).sum())
        okW &= abs(best - v) < 1e-9
    check("戻せる手の多い盤：上下の部位が 0 なら満足は今の盤と同じ", okS)
    check("戻せる手の多い盤：最適は上下の部位の総当たり（3^4×2^6 通り）の最大と一致", okW)

    # 7. vstar3
    n = bad = 0
    for reg, kw in [("T1+", dict(protocol="topk", ask=3)), ("T1+", dict(IMP, protocol="topk", ask=3, nu=0.1, newW=lambda rng, i: composition("二群", int(rng.integers(10**6, 10**7)))[i])),
                    ("P", {}), ("AI+", {})]:
        for cs in range(4):
            W = composition("二群", cs)
            r = s5.run(W, reg, cs, **kw)
            n += 1; bad += json.dumps(vstar.record(r, W, None, s0.f_design())) != json.dumps(vstar3.record3(r, W, None, s0.f_design()))
            ra = s5.run(W, reg, cs, actors=[s5.Recorder("amount")], **kw)
            n += 1; bad += vstar3.knowledge3(ra, None, None) != vstar.knowledge(ra, None)
    check("vstar3：役者のない局で vstar.record と同じ。記録を読まない読み方で vstar.knowledge と同じ", bad == 0, f"{n - bad}/{n}")
    T = [(p, cs, nu, b, arm) for p in ["α=3", "二群"] for cs in range(30) for nu in (0.0, 0.1) for b in ["改良版top3"]
         for arm in ("基準", "額まで記録", "戻せる手の多い盤", "一律に禁止", "同意だけ記録")]
    R = pool.map(run_one, T, chunksize=8)
    okc = True; okd = True; oke = True
    for p in ["α=3", "二群"]:
        for nu in (0.0, 0.1):
            for arm, act in (("額まで記録", "recorder"), ("戻せる手の多い盤", "designer"), ("一律に禁止", "verifier"), ("同意だけ記録", "recorder")):
                recs = sorted([r for r in R if r["pat"] == p and r["nu"] == nu and r["arm"] == arm], key=lambda r: r["cs"])
                vo = vstar3.population3(recs, "Xn")["V"]; vc = vstar3.population3(recs, "Xn", count=(act,))["V"]
                okc &= vc <= vo + 1e-12
                if arm != "額まで記録":
                    okd &= vc == vo
            base = sorted([r for r in R if r["pat"] == p and r["nu"] == nu and r["arm"] == "基準"], key=lambda r: r["cs"])
            cons = sorted([r for r in R if r["pat"] == p and r["nu"] == nu and r["arm"] == "同意だけ記録"], key=lambda r: r["cs"])
            oke &= vstar3.population3(base, "Xn")["V"] == vstar3.population3(cons, "Xn", count=("recorder",))["V"]
            amt = sorted([r for r in R if r["pat"] == p and r["nu"] == nu and r["arm"] == "額まで記録"], key=lambda r: r["cs"])
            oke &= vstar3.population3(base, "Xn")["V"] == vstar3.population3(amt, "Xn", G_key="None")["V"]
    check("〔数〕≤〔外〕（記録者・設計者・検証者）", okc)
    check("係数を知らない役者（設計者・検証者・同意だけの記録者）は、作者に数えても値を変えない（定義から〔数〕＝〔外〕）", okd)
    check("同意だけの記録者は基準と同じ値、額まで記録しても住み手が読まなければ基準と同じ値", oke)
    pool.close()

    # 8. 記録者
    W = composition("α=3", 2)
    nwf = lambda rng, i: composition("α=3", int(rng.integers(10**6, 10**7)))[i]
    r = s5.run(W, "T1+", 2, actors=[s5.Recorder("amount")], protocol="topk", ask=3)
    qs = {(t, j, c) for (t, i, j, c) in r["quotes"]}
    check("額の記録（入れ替わりなし）は、提案者が聞いた言い値の一部", {(t, j, c) for (t, j, c, _) in r["ledger"]} <= qs, (len(r["ledger"]), len(qs)))
    r = s5.run(W, "T1+", 2, actors=[s5.Recorder("none")], nu=0.3, newW=nwf, **IMP)
    r2 = s5.run(W, "T1+", 2, nu=0.3, newW=nwf, **IMP)
    check("記録なしでは〔再〕〔清〕が起きない（基準では起きる）", r["recon"] == [] and len(r2["recon"]) > 0, (len(r["recon"]), len(r2["recon"])))
    print("すべて合格" if ok else "不合格あり")
