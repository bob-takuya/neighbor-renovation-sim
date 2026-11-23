"""確証の実験：名付けた量 V*n（住民で話し合う価値）を、新しい種と二つの囮（市場型 M・代表型 Rep）で確かめる。
事前登録 v6（experiments/prereg_s4.json、ハッシュ out/s4_frozen.sha256）。

PYTHONPATH=. python3 -m experiments.s4_confirm dev     選ぶ用（cseed 0〜39）で囮が定義から 0 にならないことを確かめる → out/s4/dev.jsonl・dev_values.json
PYTHONPATH=. python3 -m experiments.s4_confirm run     確証の本走行（cseed は事前登録の seeds）→ out/s4/main.jsonl
PYTHONPATH=. python3 -m experiments.s4_confirm judge   凍結した規則で判定 → out/s4/judge.json
"""
import sys, json, os, hashlib
import numpy as np
from multiprocessing import Pool
from trade import s0, s1, s4, vstar, vstar2
from experiments.explore import composition
from experiments import s2_cand

PRE = "experiments/prereg_s4.json"
DRY = os.environ.get("S4_DRY") == "1"      # 凍結の前の試し：選ぶ用の cseed 0〜9 で run と judge の全経路を通す（回数を減らす）
MAIN = "out/s4/dry.jsonl" if DRY else "out/s4/main.jsonl"
JUDGE = "out/s4/dry_judge.json" if DRY else "out/s4/judge.json"
PRE_V4 = "experiments/prereg_s2.json"
PATS = ["α=3", "二群"]
NUS = [0.0, 0.1]
COSTS = s2_cand.COSTS
# v4 の 20 の決め方（s2_cand.REGS と同じ）に、囮を足す。M と Mch は同じ局で、清算所を作者に数えるかだけが違う
REGS = dict(s2_cand.REGS)
REGS.update({
    "M": ("M+", {}, None), "Mtop3": ("M+", dict(protocol="topk", ask=3), None),
    "Rep": ("Rep+", {}, None), "Reptop3": ("Rep+", dict(protocol="topk", ask=3), None),
})
V4_REGS = list(s2_cand.REGS)
DECOY_RUNS = ["M", "Mtop3", "Rep", "Reptop3"]
F0 = s2_cand.F0
WORST = s2_cand.WORST


def one(a):
    """s2_cand.one と同じ手順（模型だけ s4.run、記録だけ vstar2.record2）。囮でない決め方では s2_cand.one と同じ記録を返す。
    a の 7 番目が True なら、感度の分析のための Gbest（今の住み手と去った住み手のうち最も多く知る者の知識で解いた最適）を足す"""
    pat, cs, nu, cname, reg, rep = a[:6]
    extra = len(a) > 6 and a[6]
    rr, kw, sa = REGS[reg]
    kw = dict(kw)
    W = composition(pat, cs)
    seed = cs + 50000 * rep
    st = kw.pop("_start", None)
    if st == "wstar":
        _, e, k, o = s0.wstar_linear(W); kw.update(start=(e, k, o), ref=F0)
    elif st == "worst":
        kw.update(start=tuple(x.copy() for x in WORST))
    cost, rcost = COSTS[cname]
    newW = (lambda rng, i: composition(pat, int(rng.integers(10**6, 10**7)))[i]) if nu > 0 else None
    r = s4.run(W, rr, seed, nu=nu, newW=newW, cost=cost, rcost=rcost, **kw)
    rec = vstar2.record2(r, W, sa, F0)
    rec.update(pat=pat, cs=cs, nu=nu, cost=cname, reg=reg, rep=rep)
    if extra:
        cur, cen, former = vstar.knowledge(r, sa)
        rec["Gbest"] = vstar.guess(r["W_final"], max(cur + [former], key=len)).tolist()
    return rec


def _pool_run(T, path, every=2000):
    os.makedirs("out/s4", exist_ok=True)
    print(len(T), "局", flush=True)
    with Pool(2) as pool, open(path, "w") as f:
        for i, rec in enumerate(pool.imap_unordered(one, T, chunksize=10)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % every == 0:
                print(i, flush=True)
    print("done", flush=True)


def load(path):
    cells = {}
    for line in open(path):
        r = json.loads(line)
        cells.setdefault((r["reg"], r["pat"], r["nu"], r["cost"]), []).append(r)
    for k in cells:
        cells[k].sort(key=lambda r: r["cs"])
    return cells


# ------------------------------------------------------------ 選ぶ用での確かめ（事前登録の前）
DEV_CS = list(range(0, 40))
DEV_REGS = ["F", "N3", "U", "TP+", "TP+top3", "改良版top3"] + DECOY_RUNS
REGS["Reptop3輪番"] = ("Rep+", dict(protocol="topk", ask=3, rep_term=1), None)     # 選ぶ用での確かめだけ：代表を毎年選び直す


def dev():
    T = [(p, cs, 0.0, c, rg, 0) for p in PATS for c in COSTS for rg in DEV_REGS + ["Reptop3輪番"] for cs in DEV_CS]
    T += [(p, cs, 0.1, c, rg, 0) for p in PATS for c in COSTS for rg in ["TP+top3", "改良版top3", "Mtop3", "Reptop3"] for cs in DEV_CS]
    _pool_run(T, "out/s4/dev.jsonl", every=500)
    dev_values()


def dev_values():
    cells = load("out/s4/dev.jsonl")
    out = {}
    for (rg, pat, nu, c), recs in sorted(cells.items()):
        row = dict(eta=float(np.nanmean([r["eta"] for r in recs])), quotes=float(np.mean([r["quotes"] for r in recs])))
        for key, nm in (("Xn", "Vn"), ("Xc", "Vs")):
            pv = vstar2.value(recs, key)
            row[nm] = pv["V"]; row[nm + "_I"] = pv["I"]; row[nm + "_Imax"] = pv["Imax"]
            row[nm + "_ci"] = vstar2.boot(recs, key, n=200)
            if rg.startswith("Rep"):       # 役の枠を足さない値（比べるため）
                row[nm + "_noslot"] = vstar.population(recs, key)["V"]
            if rg.startswith("M"):
                pc = vstar2.value(recs, key, count_ch=True)
                row[nm + "_ch"] = pc["V"]; row[nm + "_ch_Imax"] = pc["Imax"]
                row[nm + "_ch_ci"] = vstar2.boot(recs, key, n=200, count_ch=True)
        if rg.startswith("M"):
            row["ch_n"] = float(np.mean([r["ch_n"] for r in recs]))
        out[f"{rg}|{pat}|{nu}|{c}"] = row
    json.dump(out, open("out/s4/dev_values.json", "w"), ensure_ascii=False, indent=1)
    for k, v in out.items():
        s = f"{k:28s} η={v['eta']:.3f} 問={v['quotes']:7.1f}  V*n={v['Vn']:.3f}［{v['Vn_ci'][0]:.3f},{v['Vn_ci'][1]:.3f}］ V*={v['Vs']:.3f}"
        s += f"  I={v['Vn_I']:.3f} Imax={v['Vn_Imax']:.3f}"
        if "Vn_noslot" in v:
            s += f"  ｜役の枠なし V*n={v['Vn_noslot']:.3f}"
        if "Vn_ch" in v:
            s += f"  ｜清算所を数える V*n={v['Vn_ch']:.3f} V*={v['Vs_ch']:.3f} Imax={v['Vn_ch_Imax']:.3f} 清算所の問={v['ch_n']:.0f}"
        print(s)


# ------------------------------------------------------------ 確証の本走行
def run():
    pre = json.load(open(PRE))
    CS = list(range(pre["seeds"][0], pre["seeds"][1] + 1)) if not DRY else list(range(0, 10))
    regs = pre["regimes"]["v4"] + pre["regimes"]["decoy_runs"]
    T = [(p, cs, nu, c, rg, 0, True) for p in pre["patterns"] for nu in pre["nus"] for c in pre["costs"] for rg in regs for cs in CS]
    _pool_run(T, MAIN)


# ------------------------------------------------------------ 判定（凍結した規則どおり）
def spearman(a, b):
    from scipy.stats import spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).correlation)


def _cellval(a):
    recs, key, ch, B, BS = a
    v = vstar2.value(recs, key, count_ch=ch)["V"]
    lo, hi = vstar2.boot(recs, key, n=B, seed=BS, count_ch=ch)
    return (v, lo, hi)


def judge():
    from experiments.s2_tests import n7_records
    pre = json.load(open(PRE)); v4 = json.load(open(PRE_V4))
    B = pre["bootstrap"]["n"] if not DRY else 30; BS = pre["bootstrap"]["seed"]
    cells = load(MAIN)
    CS = list(range(pre["seeds"][0], pre["seeds"][1] + 1)) if not DRY else list(range(0, 10))
    SET = [(pat, nu, c) for pat in pre["patterns"] for nu in pre["nus"] for c in pre["costs"]]
    for pat, nu, c in SET:
        cells[("N7", pat, nu, c)] = n7_records(pat, CS)
    for k in cells:
        cells[k].sort(key=lambda r: r.get("cs", 0))
        assert len(cells[k]) == len(CS), (k, len(cells[k]))
    # ---- ε（v4 の規則どおり、このデータで求める）
    er = pre["epsilon"]
    eps = max(vstar.perm_floor(cells[(rg, pat, nu, c)], "Xn", n=er["perm"] if not DRY else 20, seed=er["seed"])
              for rg in er["regimes"] for (pat, nu, c) in SET)
    # ---- 値と区間：V*n（Xn）と V*（Xc）、M は清算所を数える／数えない
    val = {}
    names = sorted({k[0] for k in cells}) + [x + "ch" for x in pre["regimes"]["decoy_runs"] if x.startswith("M")]
    jobs = []
    for (pat, nu, c) in SET:
        for nm in names:
            base = nm[:-2] if nm.endswith("ch") and nm[:-2] in ("M", "Mtop3") else nm
            ch = base != nm
            for key in ("Xn", "Xc"):
                jobs.append(((nm, pat, nu, c), key, (cells[(base, pat, nu, c)], key, ch, B, BS)))
    with Pool(2) as pool:      # 計算を二つの核に分けるだけで、値は一つずつ計算したものと同じ
        res_ = pool.map(_cellval, [j[2] for j in jobs], chunksize=4)
    for (k, key, _), r in zip(jobs, res_):
        recs = cells[(k[0][:-2] if k[0].endswith("ch") and k[0][:-2] in ("M", "Mtop3") else k[0],) + k[1:]]
        row = val.setdefault(k, dict(eta=float(np.nanmean([x["eta"] for x in recs])), quotes=float(np.mean([x["quotes"] for x in recs]))))
        row["Vn" if key == "Xn" else "Vs"] = r
    V = lambda nm, cell: val[(nm,) + cell]["Vn"]
    out = dict(epsilon=eps, n_configs=len(CS), bootstrap=B)
    # ---- v4-P1〜P5（文は v4 のまま。判定の式は s2_name.judge と同じ）
    nullset = [x for x in v4["nulls"] if "@" not in x]
    f1 = [f"{k[0]}|{k[1]}|ν{k[2]}|{k[3]}: {V(k[0], k[1:])[0]:.3f}［{V(k[0], k[1:])[1]:.3f}, {V(k[0], k[1:])[2]:.3f}］"
          for k in val if (k[0] in nullset or (k[0] == "N2" and k[2] == 0.0)) and V(k[0], k[1:])[2] > eps]
    n1 = sum(1 for k in val if (k[0] in nullset or (k[0] == "N2" and k[2] == 0.0)))
    out["P1"] = dict(ok=not f1, fails=f1, n_cells=n1)
    res = []
    for pairs, nus in ((v4["order_pairs"], v4["order_nus"]), (v4["order_pairs_turnover"], [x for x in pre["nus"] if x > 0])):
        for a, b in pairs:
            for pat in pre["patterns"]:
                for nu in nus:
                    for c in pre["costs"]:
                        A, Bv = V(a, (pat, nu, c)), V(b, (pat, nu, c))
                        res.append(dict(pair=f"{a} ＞ {b}", cell=f"{pat}|ν{nu}|{c}", a=A[0], b=Bv[0], ok=bool(A[0] > Bv[0])))
    share = float(np.mean([r["ok"] for r in res]))
    out["P2"] = dict(ok=share >= v4["order_share"], share=share, n=len(res), n_ok=int(sum(r["ok"] for r in res)), rows=res)
    f3 = [f"{k[0]}|{k[1]}|ν{k[2]}|{k[3]}: {V(k[0], k[1:])[0]:.3f}［{V(k[0], k[1:])[1]:.3f}, {V(k[0], k[1:])[2]:.3f}］"
          for k in val if k[0] in v4["no_consent"] and V(k[0], k[1:])[2] > eps]
    out["P3"] = dict(ok=not f3, fails=f3)
    f4 = []
    for pat in pre["patterns"]:
        for c in pre["costs"]:
            A, Bv = V("TP+top3", (pat, 0.0, c)), V("TP+", (pat, 0.0, c))
            f4.append(dict(cell=f"{pat}|{c}", top3=A, full=Bv, eta=(val[("TP+top3", pat, 0.0, c)]["eta"], val[("TP+", pat, 0.0, c)]["eta"]),
                           ok=bool(A[1] > Bv[2])))
    out["P4"] = dict(ok=all(x["ok"] for x in f4), rows=f4)
    f5 = []
    for (pat, nu, c) in SET:
        A, Bv = V("N6", (pat, nu, c)), V("TP+top3", (pat, nu, c))
        f5.append(dict(cell=f"{pat}|ν{nu}|{c}", n6=A, top3=Bv, ok=bool(A[1] <= Bv[2] + eps)))
    out["P5"] = dict(ok=all(x["ok"] for x in f5), rows=f5)
    # ---- 囮の判定（事前登録 v6 の D1〜D3）
    dr = pre["decoy_rules"]
    resident = dr["resident"]; Mm = dr["M_main"]; Rm = dr["Rep_main"]; ref = dr["M_reference"]
    margin = eps if dr["D1"]["b_margin"] == "epsilon" else float(dr["D1"]["b_margin"])
    d1 = []
    for cell in SET:
        Mv = V(Mm, cell); T3 = V(ref, cell)
        a_ = bool(Mv[1] > T3[2])                                    # (a) M の区間の下端 ＞ TP+top3 の区間の上端
        mx = max(resident, key=lambda r: V(r, cell)[0])
        b_ = bool(Mv[0] > V(mx, cell)[0] + margin)                  # (b) M の点推定 ＞ 住民主体の最大 ＋ ε
        lower = bool(Mv[2] < T3[1])                                  # M の区間の上端 ＜ TP+top3 の区間の下端
        counts = bool(T3[1] > eps)                                   # TP+top3 の区間の下端が ε を超えるセル（支持の判定に数える）
        d1.append(dict(cell="|".join(map(str, cell)), M=Mv, ref=T3, a=a_, best_resident=mx, best_value=V(mx, cell)[0], b=b_,
                       M_lower=lower, ref_above_eps=counts))
    hit = [x["cell"] for x in d1 if x["a"] or x["b"]]
    sup_cells = [x for x in d1 if x["ref_above_eps"]]
    if hit:
        verdict = "不適切"
    elif sup_cells and all(x["M_lower"] for x in sup_cells):
        verdict = "反証されない"
    else:
        verdict = "範囲を限定"
    out["D1"] = dict(M=Mm, reference=ref, margin=margin, rows=d1, cells_hit=hit, n_ref_cells=len(sup_cells),
                     n_M_lower=sum(x["M_lower"] for x in sup_cells), verdict=verdict)
    d2 = [dict(cell="|".join(map(str, cell)), Rep=V(Rm, cell), below_eps=bool(V(Rm, cell)[2] <= eps)) for cell in SET]
    allb = all(x["below_eps"] for x in d2)
    out["D2"] = dict(Rep=Rm, rows=d2, all_below_eps=allb,
                     verdict="名の範囲を、住民どうしが直接取引する手続きに限定する" if allb else "代表型でも値が残る（代表型の結果では名の範囲を限定しない）")
    d3 = []
    for cell in SET:
        for nm in [Mm, Mm + "ch", ref, "TP+", "M", "Mch"]:
            d3.append(dict(cell="|".join(map(str, cell)), reg=nm, Vs=val[(nm,) + cell]["Vs"], Vn=val[(nm,) + cell]["Vn"],
                           eta=val[(nm,) + cell]["eta"]))
    out["D3"] = dict(rows=d3)
    # ---- 感度の分析（判定に使わない）
    sens = []
    for cell in SET:
        for nm in dr["sensitivity"]["resident_role_slot"]:
            recs = cells[(nm,) + cell]
            withb = [dict(x, G=x["G"] + [x["Gbest"]]) for x in vstar2.with_roles(recs)]
            sens.append(dict(cell="|".join(map(str, cell)), reg=nm, kind="今の住み手と去った住み手のうち最も多く知る者の役の枠を足す",
                             Vn=V(nm, cell)[0], Vn_sens=vstar.population(withb, "Xn")["V"]))
        for nm in dr["sensitivity"]["rep_without_role_slot"]:
            recs = cells[(nm,) + cell]
            sens.append(dict(cell="|".join(map(str, cell)), reg=nm, kind="代表の役の枠を足さない（住戸の枠だけ）",
                             Vn=V(nm, cell)[0], Vn_sens=vstar.population(recs, "Xn")["V"]))
    out["sensitivity"] = sens
    # ---- 再現の基準（関係の一致）：名付けの確かめ（cseed 200〜249）との比較
    old = json.load(open("out/s2/name_judge.json"))
    oldv = {tuple(k.split("|")[:1]) + (k.split("|")[1], float(k.split("|")[2]), k.split("|")[3]): v for k, v in old["values"].items()}
    agree = []
    for r_old in old["P2"]["rows"]:
        r_new = [x for x in res if x["pair"] == r_old["pair"] and x["cell"].replace("|ν", "・ν").replace("|", "・費用") == r_old["cell"]]
        assert len(r_new) == 1, r_old
        agree.append(bool(r_new[0]["ok"] == r_old["ok"]))
    rhos = []
    for (pat, nu, c) in SET:
        xs = [V(r, (pat, nu, c))[0] for r in resident]
        ys = [oldv[(r, pat, nu, c)][0] for r in resident]
        xs = [0.0 if x <= eps else x for x in xs]; ys = [0.0 if y <= old["epsilon"] else y for y in ys]
        rhos.append(spearman(xs, ys))
    rc = pre["replication"]
    out["replication"] = dict(direction_share=float(np.mean(agree)), direction_ok=bool(np.mean(agree) >= rc["direction_share"]),
                              rho_by_cell=rhos, rho_median=float(np.nanmedian(rhos)),
                              rank_ok=bool(np.nanmedian(rhos) >= rc["rank_rho_median"]),
                              verdicts_same=all(out[p]["ok"] == old[p]["ok"] for p in ["P1", "P2", "P3", "P4", "P5"]))
    out["values"] = {"|".join(map(str, k)): v for k, v in val.items()}
    out["prereg_sha256"] = hashlib.sha256(open(PRE, "rb").read()).hexdigest()
    json.dump(out, open(JUDGE, "w"), ensure_ascii=False, indent=1)
    print("ε =", round(eps, 4))
    for p in ["P1", "P2", "P3", "P4", "P5"]:
        print(p, "支持" if out[p]["ok"] else "不支持", {k: v for k, v in out[p].items() if k in ("fails", "share")})
    print("D1", out["D1"]["verdict"], out["D1"]["cells_hit"], out["D1"]["n_M_lower"], "/", out["D1"]["n_ref_cells"]); print("D2", out["D2"]["verdict"])
    print("再現", {k: v for k, v in out["replication"].items()})


if __name__ == "__main__":
    {"dev": dev, "dev_values": dev_values, "run": run, "judge": judge}[sys.argv[1]]()
