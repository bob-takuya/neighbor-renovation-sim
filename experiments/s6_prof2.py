"""職能 2：聞いた言い値だけを知る仲介者。事前登録 v7b（experiments/prereg_s6.json、ハッシュ out/s6_frozen.sha256）。

基準は運用役を置かない TP+top3 と改良版top3。同じ構成・同じ乱数のもとで運用役だけを替え、構成ごとの組ごとの差をとる。
取引の条件は三つ：基準（費用なし・全員に届く）、費用 1 倍（確証の C1）、近い戸だけ（同じ階の隣と同じ列の上下一つにしか聞けない）。
退去率は 入れ替わりなし と 年 5 %・10 %・30 %（観察 2 の入れ替わりの群と同じ幾何の入れ替わり）、パターンは α＝3 と二群。
比較の参照は全知の仲介者 BR（名付けの偽装、s1 の central）。入れ替わりのあるセルでは、入れ替わりのたびにくり返す BRdyn もあわせて走らせる。

PYTHONPATH=. python3 -m experiments.s6_prof2 dev       選ぶ用（cseed 0〜39）→ out/s6/dev_parts/・dev.jsonl
PYTHONPATH=. python3 -m experiments.s6_prof2 dev_values 選ぶ用の値と差（ブートストラップ 200 回）→ out/s6/dev_values.json
PYTHONPATH=. python3 -m experiments.s6_prof2 run       確かめ用（cseed は事前登録の seeds）→ out/s6/main_parts/・main.jsonl
PYTHONPATH=. python3 -m experiments.s6_prof2 judge     凍結した規則で判定 → out/s6/judge_parts/・judge.json
（S6_DRY=1 なら、選ぶ用の cseed 0〜9 で run と judge の全経路を試す → out/s6/dry_*）
どれも組ごと・計算ごとのファイルに書くので、途中で止まっても、そろった分は走らせ直さない。
"""
import sys, json, os, hashlib
import numpy as np
from multiprocessing import Pool
from trade import s0, s6, vstar, vstar3, vstar4
from experiments.explore import composition

PRE = "experiments/prereg_s6.json"
DRY = os.environ.get("S6_DRY") == "1"
OUT = "out/s6"
PATS = ["α=3", "二群"]
NUS = [0.0, 0.05, 0.10, 0.30]
C1 = (dict(e=1.0, k=0.25, o=0.25), dict(e=0.5, k=0.1, o=0.1))          # 確証の「費用 1 倍」（s2_cand.COSTS["C1"]）
CONDS = {"基準": {}, "費用1倍": dict(cost=C1[0], rcost=C1[1]), "近い戸だけ": dict(venue=dict(G=0, q_far=0.0, tau=0.0))}
IMP = dict(credit=20.0, settle=True, reconsent="all")
BASES = {"TP+top3": dict(protocol="topk", ask=3), "改良版top3": dict(IMP, protocol="topk", ask=3)}
BROKERS = {"仲介者": "stuck", "広く聞く仲介者": "all"}
REFS = ["BR", "BRdyn"]
F0 = s0.f_design()


def newW_of(pat):
    return lambda rng, i: composition(pat, int(rng.integers(10**6, 10**7)))[i]


def one(a):
    """一局。a ＝ (パターン, cseed, 退去率, 取引の条件, 基準 or "参照", 腕)"""
    pat, cs, nu, cond, base, arm = a
    W = composition(pat, cs)
    kw = dict(CONDS[cond])
    if nu > 0:
        kw.update(turnover=dict(kind="geo", nu=nu), newW=newW_of(pat))
    if base == "参照":
        kw.pop("venue", None)                       # BR は全員に届く（近い戸だけの条件でも同じ局になる）
        r = s6.run(W, arm, cs, **kw)
    else:
        kw.update(BASES[base])
        acts = [s6.Mediator(BROKERS[arm])] if arm in BROKERS else []
        r = s6.run(W, "T1+", cs, actors=acts, **kw)
    rec = vstar3.record3(r, W, None, F0)
    st = r.get("actor_stats", {}).get("broker", {})
    evs = [ev for ev in r["events"] if ev.get("ok") and ev.get("part") is not None]
    rec.update(pat=pat, cs=cs, nu=nu, cond=cond, base=base, arm=arm, n_exit=r["n_exit"],
               recon_paid=sum(1 for x in r["recon"] if x[3] == "paid"), unreached=r["extra"]["unreached"],
               n_mediated=sum(1 for ev in evs if ev.get("drafter") == "mediator"),
               bundle_sizes=[ev["bundle"] for ev in evs if ev.get("drafter") == "mediator"],
               spent=float(r["spent"]), **{k: v for k, v in st.items()})
    return rec


# ------------------------------------------------------------ 組（基準 × 腕 × 条件 × パターン × 退去率）
def groups(spec, pats, nus):
    """spec ＝ {基準 or "参照": {腕: [条件, …]}}。BRdyn は入れ替わりのあるセルだけ（入れ替わりなしでは BR と同じ局）"""
    G = []
    for b, arms in spec.items():
        for arm, conds in arms.items():
            for cond in conds:
                for p in pats:
                    for nu in nus:
                        if arm == "BRdyn" and nu == 0:
                            continue
                        G.append((b, arm, cond, p, nu))
    return G


DEV_SPEC = {"TP+top3": {"基準": list(CONDS), "仲介者": list(CONDS), "広く聞く仲介者": list(CONDS)},
            "改良版top3": {"基準": list(CONDS), "仲介者": list(CONDS), "広く聞く仲介者": list(CONDS)},
            "参照": {"BR": ["基準", "費用1倍"], "BRdyn": ["基準", "費用1倍"]}}
DEV_CS = list(range(0, 40))


def run_groups(G, CS, parts, merged):
    os.makedirs(parts, exist_ok=True)
    with Pool(2) as pool:
        for g in G:
            fn = f"{parts}/{'_'.join(map(str, g))}.jsonl"
            if os.path.exists(fn) and sum(1 for _ in open(fn)) == len(CS):
                continue
            T = [(g[3], cs, g[4], g[2], g[0], g[1]) for cs in CS]
            recs = pool.map(one, T, chunksize=4)
            with open(fn + ".tmp", "w") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(fn + ".tmp", fn)
            print("組", g, flush=True)
    with open(merged + ".tmp", "w") as f:
        for g in G:
            for line in open(f"{parts}/{'_'.join(map(str, g))}.jsonl"):
                f.write(line)
    os.replace(merged + ".tmp", merged)
    print("done", len(G) * len(CS), "局", flush=True)


def load(path):
    cells = {}
    for line in open(path):
        r = json.loads(line)
        cells.setdefault((r["base"], r["arm"], r["cond"], r["pat"], r["nu"]), []).append(r)
    for k in cells:
        cells[k].sort(key=lambda r: r["cs"])
    return cells


def getter(cells):
    """C(基準, 腕, 条件, パターン, 退去率)。BR・BRdyn は近い戸だけの条件でも基準の条件と同じ局（全員に届く）。
    BRdyn は入れ替わりなしでは BR と同じ局"""
    def C(b, arm, cond, p, nu):
        if b == "参照":
            if cond == "近い戸だけ":
                cond = "基準"
            if arm == "BRdyn" and nu == 0:
                arm = "BR"
        return cells[(b, arm, cond, p, nu)]
    return C


def prep(recs, arm, q):
    """量 q（"Vn外" か "Vn数"）のための (局の写し, 作者に数える運用役)。
    聞いた言い値だけを知る仲介者は役の枠（GA、種類 broker）で数える。全知の BR は中央の作者の枠で数え、〔外〕ではその枠を外す"""
    if arm in REFS:
        return (recs if q == "Vn数" else vstar4.drop_central(recs)), ()
    if arm in BROKERS and q == "Vn数":
        return recs, ("broker",)
    return recs, ()


# ------------------------------------------------------------ 選ぶ用
def dev():
    run_groups(groups(DEV_SPEC, PATS, NUS), DEV_CS, f"{OUT}/dev_parts", f"{OUT}/dev.jsonl")


def eta_diff(A, B, n, seed, col="eta"):
    """η の組ごとの差（A − B、同じ構成）の平均と、構成を引き直すブートストラップの 95 % 区間（s5_prof1.eta_diff と同じ式）"""
    assert [x["cs"] for x in A] == [x["cs"] for x in B]
    d = np.array([a[col] - b[col] for a, b in zip(A, B)])
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return dict(d=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))


def _job(a):
    kind = a[0]
    if kind == "eps":
        _, R, n, seed = a
        return vstar.perm_floor(R, "Xn", n=n, seed=seed)
    if kind == "val":
        _, R, cnt, B, BS = a
        v = vstar3.population3(R, "Xn", None, cnt)
        lo, hi = vstar3.boot3(R, "Xn", n=B, seed=BS, count=cnt)
        return dict(v=v["V"], lo=lo, hi=hi, I=v["I"], Imax=v["Imax"])
    _, RA, RB, cA, cB, B, BS = a
    return vstar3.diff3(RA, RB, "Xn", n=B, seed=BS, countA=cA, countB=cB)


def cached(jobs, cache_dir):
    """計算ごとのキャッシュ。jobs ＝ [(鍵, 引数)]"""
    os.makedirs(cache_dir, exist_ok=True)
    fns = [f"{cache_dir}/" + hashlib.sha1("|".join(map(str, j[0])).encode()).hexdigest()[:16] + ".json" for j in jobs]
    todo = [n for n, fn in enumerate(fns) if not os.path.exists(fn)]
    print(len(jobs), "の計算のうち", len(todo), "を計算する", flush=True)
    with Pool(2) as pool:
        for n, r in zip(todo, pool.imap(_job, [jobs[n][1] for n in todo], chunksize=1)):
            json.dump(dict(key=list(map(str, jobs[n][0])), res=r), open(fns[n] + ".tmp", "w"), ensure_ascii=False)
            os.replace(fns[n] + ".tmp", fns[n])
    return [json.load(open(fn))["res"] for fn in fns]


def value_jobs(G, C, B, BS):
    """組ごとの値：〔外〕と、運用役のある腕では〔数〕"""
    jobs = []
    for (b, arm, cond, p, nu) in G:
        R = C(b, arm, cond, p, nu)
        for q in (("Vn外", "Vn数") if (arm in BROKERS or arm in REFS) else ("Vn外",)):
            R2, cnt = prep(R, arm, q)
            jobs.append((("val", b, arm, cond, p, nu, q), ("val", R2, cnt, B, BS)))
    return jobs


def diff_job(C, b, arm, ref, cond, p, nu, q, B, BS, ref_base=None):
    """差（腕 − 比べる腕、同じ構成の組）。ref が BR・BRdyn のときの基準は "参照"。q は "Vn外"・"Vn数"（〔数〕は両方の腕の運用役を数える）"""
    rb = "参照" if ref in REFS else (ref_base or b)
    ab = "参照" if arm in REFS else b
    RA, cA = prep(C(ab, arm, cond, p, nu), arm, q)
    RB, cB = prep(C(rb, ref, cond, p, nu), ref, q)
    return (("diff", b, arm, ref, cond, p, nu, q), ("diff", RA, RB, cA, cB, B, BS))


def summarize(G, C):
    means = {}
    for (b, arm, cond, p, nu) in G:
        R = C(b, arm, cond, p, nu)
        means["|".join(map(str, (b, arm, cond, p, nu)))] = {
            c: float(np.nanmean([r.get(c, 0) for r in R]))
            for c in ("eta", "L", "moves", "quotes", "n_mediated", "n_consult", "n_ask", "n_bundle", "n_bundle3", "n_far",
                      "n_copay", "unreached", "recon_paid", "spent")}
        means["|".join(map(str, (b, arm, cond, p, nu)))]["nfacts"] = float(np.mean([sum(r.get("actor_nfacts", {}).values()) for r in R]))
    return means


def dev_values():
    cells = load(f"{OUT}/dev.jsonl")
    C = getter(cells)
    G = groups(DEV_SPEC, PATS, NUS)
    jobs = value_jobs(G, C, 200, 0)
    for b in BASES:
        for cond in CONDS:
            for p in PATS:
                for nu in NUS:
                    jobs.append((("eps", b, cond, p, nu), ("eps", C(b, "基準", cond, p, nu), 100, 1)))
                    for arm in BROKERS:
                        for q in ("Vn外", "Vn数"):
                            jobs.append(diff_job(C, b, arm, "基準", cond, p, nu, q, 200, 0))
                    jobs.append(diff_job(C, b, "仲介者", "広く聞く仲介者", cond, p, nu, "Vn数", 200, 0))
                    jobs.append(diff_job(C, b, "仲介者", "広く聞く仲介者", cond, p, nu, "Vn外", 200, 0))
                    if cond != "近い戸だけ" or True:
                        jobs.append(diff_job(C, b, "BR", "基準", cond, p, nu, "Vn外", 200, 0))
                        jobs.append(diff_job(C, b, "仲介者", "BR", cond, p, nu, "Vn外", 200, 0))
    res = cached(jobs, f"{OUT}/dev_cache")
    out = dict(values={}, diffs={}, eps={}, eta_diffs={})
    for (key, _), r in zip(jobs, res):
        k = "|".join(map(str, key[1:]))
        out[{"val": "values", "diff": "diffs", "eps": "eps"}[key[0]]][k] = r
    for b in BASES:
        for cond in CONDS:
            for p in PATS:
                for nu in NUS:
                    for arm in BROKERS:
                        out["eta_diffs"][f"{b}|{arm}|基準|{cond}|{p}|{nu}"] = eta_diff(C(b, arm, cond, p, nu), C(b, "基準", cond, p, nu), 2000, 1)
                    out["eta_diffs"][f"{b}|BR|基準|{cond}|{p}|{nu}"] = eta_diff(C("参照", "BR", cond, p, nu), C(b, "基準", cond, p, nu), 2000, 1)
                    out["eta_diffs"][f"{b}|BRdyn|基準|{cond}|{p}|{nu}"] = eta_diff(C("参照", "BRdyn", cond, p, nu), C(b, "基準", cond, p, nu), 2000, 1)
    out["means"] = summarize(G, C)
    out["epsilon_dev"] = max(out["eps"].values())
    json.dump(out, open(f"{OUT}/dev_values.json", "w"), ensure_ascii=False, indent=1)
    print("ε（選ぶ用）", round(out["epsilon_dev"], 4))


# ------------------------------------------------------------ 確かめ用
PARTS = f"{OUT}/dry_parts" if DRY else f"{OUT}/main_parts"
MAIN = f"{OUT}/dry.jsonl" if DRY else f"{OUT}/main.jsonl"
JUDGE = f"{OUT}/dry_judge.json" if DRY else f"{OUT}/judge.json"


def pre_groups(pre):
    return groups(pre["arms"], pre["patterns"], pre["nus"])


def run():
    pre = json.load(open(PRE))
    CS = list(range(pre["seeds"][0], pre["seeds"][1] + 1)) if not DRY else list(range(0, 10))
    run_groups(pre_groups(pre), CS, PARTS, MAIN)


def passes(rule, x, margin):
    if rule == "lower":
        return x["hi"] < 0
    if rule == "higher":
        return x["lo"] > 0
    if rule == "same":
        return x["lo"] >= -margin and x["hi"] <= margin
    raise ValueError(rule)


def judge():
    pre = json.load(open(PRE))
    bs = pre["bootstrap"]
    B = bs["n"] if not DRY else 30; BS = bs["seed"]; BE = bs["eta_n"] if not DRY else 200; BES = bs["eta_seed"]
    cells = load(MAIN)
    C = getter(cells)
    CS = list(range(pre["seeds"][0], pre["seeds"][1] + 1)) if not DRY else list(range(0, 10))
    G = pre_groups(pre)
    for k in cells:
        assert len(cells[k]) == len(CS), (k, len(cells[k]))
    er = pre["epsilon"]
    jobs = [(("eps", b, cond, p, nu), ("eps", C(b, "基準", cond, p, nu), er["perm"] if not DRY else 20, er["seed"]))
            for b in er["bases"] for cond in er["conds"] for p in pre["patterns"] for nu in pre["nus"]]
    jobs += value_jobs(G, C, B, BS)
    # 予言に使う差と、表のための差
    want = {}
    for pr in pre["predictions"]:
        if pr["quantity"].startswith("Vn"):
            for b in pr["bases"]:
                for (cond, p, nu) in [tuple(c) for c in pr["cells"]]:
                    want[(b, pr["arm"], pr["ref"], cond, p, nu, pr["quantity"])] = None
    for b in BASES:
        for (bb, arm, cond, p, nu) in G:
            if bb == b and arm in BROKERS:
                for q in ("Vn外", "Vn数"):
                    want.setdefault((b, arm, "基準", cond, p, nu, q), None)
    for (b, arm, ref, cond, p, nu, q) in want:
        jobs.append(diff_job(C, b, arm, ref, cond, p, nu, q, B, BS))
    res = cached(jobs, f"{OUT}/dry_judge_parts" if DRY else f"{OUT}/judge_parts")
    val, dif, epsd = {}, {}, {}
    for (key, _), r in zip(jobs, res):
        k = "|".join(map(str, key[1:]))
        {"val": val, "diff": dif, "eps": epsd}[key[0]][k] = r
    eps = max(epsd.values())
    # η の組ごとの差
    edif = {}
    for b in BASES:
        for (bb, arm, cond, p, nu) in G:
            if bb == b and arm in BROKERS:
                edif[f"{b}|{arm}|基準|{cond}|{p}|{nu}|eta"] = eta_diff(C(b, arm, cond, p, nu), C(b, "基準", cond, p, nu), BE, BES)
    for pr in pre["predictions"]:
        if pr["quantity"] == "eta":
            for b in pr["bases"]:
                for (cond, p, nu) in [tuple(c) for c in pr["cells"]]:
                    k = f"{b}|{pr['arm']}|{pr['ref']}|{cond}|{p}|{nu}|eta"
                    if k not in edif:
                        ab = "参照" if pr["arm"] in REFS else b
                        rb = "参照" if pr["ref"] in REFS else b
                        edif[k] = eta_diff(C(ab, pr["arm"], cond, p, nu), C(rb, pr["ref"], cond, p, nu), BE, BES)
    # 予言の判定
    out = dict(epsilon=eps, n_configs=len(CS), bootstrap=B, eta_bootstrap=BE)
    J = {}
    for pr in pre["predictions"]:
        rows = []
        for b in pr["bases"]:
            for (cond, p, nu) in [tuple(c) for c in pr["cells"]]:
                k = f"{b}|{pr['arm']}|{pr['ref']}|{cond}|{p}|{nu}|{pr['quantity']}"
                x = dif[k] if pr["quantity"].startswith("Vn") else edif[k]
                margin = eps if pr.get("margin") == "epsilon" else float(pr.get("margin", 0.0))
                rows.append(dict(base=b, cond=cond, pat=p, nu=nu, **x, ok=bool(passes(pr["rule"], x, margin))))
        n_ok = sum(r["ok"] for r in rows)
        verdict = "支持" if n_ok == len(rows) else ("一部支持" if n_ok > 0 else "不支持")
        J[pr["id"]] = dict(text=pr["text"], rule=pr["rule"], quantity=pr["quantity"], n_cells=len(rows), n_ok=n_ok, verdict=verdict, rows=rows)
    out["predictions"] = J
    # 定義からの確かめ（判定ではない）
    kb1 = [dict(cond=cond, pat=p, v=val[f"参照|BR|{cond}|{p}|0.0|Vn数"]["v"]) for cond in ("基準", "費用1倍") for p in pre["patterns"]]
    kb1 += [dict(cond=cond, pat=p, nu=nu, v=val[f"参照|BRdyn|{cond}|{p}|{nu}|Vn数"]["v"]) for cond in ("基準", "費用1倍")
            for p in pre["patterns"] for nu in pre["nus"] if nu > 0]
    kb2 = [dict(pat=p, eta=float(np.mean([r["eta"] for r in C("参照", "BR", "基準", p, 0.0)]))) for p in pre["patterns"]]
    k3 = [dict(key=k, out=val[k.replace("Vn数", "Vn外")]["v"], cnt=v["v"]) for k, v in val.items()
          if k.endswith("Vn数") and not k.startswith("参照")]
    out["definition_check"] = dict(
        KB1_rows=kb1, KB1=all(x["v"] == 0.0 for x in kb1),
        KB2_rows=kb2, KB2=all(abs(x["eta"] - 1.0) < 1e-9 for x in kb2),
        K3_rows=k3, K3=all(x["cnt"] <= x["out"] + 1e-12 for x in k3))
    out["values"] = val; out["diffs"] = dif; out["eta_diffs"] = edif; out["eps_cells"] = epsd
    out["means"] = summarize(G, C)
    out["prereg_sha256"] = hashlib.sha256(open(PRE, "rb").read()).hexdigest()
    json.dump(out, open(JUDGE, "w"), ensure_ascii=False, indent=1)
    print("ε =", round(eps, 4))
    for k, v in J.items():
        print(k, v["verdict"], f"{v['n_ok']}/{v['n_cells']}", v["text"][:70])
    print("定義からの確かめ", {k: v for k, v in out["definition_check"].items() if not k.endswith("_rows")})


if __name__ == "__main__":
    {"dev": dev, "dev_values": dev_values, "run": run, "judge": judge}[sys.argv[1]]()
