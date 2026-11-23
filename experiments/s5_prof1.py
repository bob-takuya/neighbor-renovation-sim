"""職能 1：性向を知らない運用役（盤の設計者・記録者・検証者）。事前登録 v7a（experiments/prereg_s5a.json、ハッシュ out/s5a_frozen.sha256）。

基準は運用役を置かない TP+top3 と改良版top3。同じ構成・同じ乱数のもとで運用役だけを替え、構成ごとの組ごとの差をとる。
条件は 入れ替わりなし と 退去率 年 5 %・10 %・30 %（観察 2 の入れ替わりの群と同じ幾何の入れ替わり、費用なし）、パターンは α＝3 と二群。

PYTHONPATH=. python3 -m experiments.s5_prof1 dev       選ぶ用（cseed 0〜39）で部品を確かめる → out/s5a/dev.jsonl・dev_values.json
PYTHONPATH=. python3 -m experiments.s5_prof1 run       確かめ用（cseed は事前登録の seeds）→ out/s5a/main.jsonl
PYTHONPATH=. python3 -m experiments.s5_prof1 judge     凍結した規則で判定 → out/s5a/judge.json
（S5_DRY=1 なら、選ぶ用の cseed 0〜9 で run と judge の全経路を試す → out/s5a/dry.jsonl・dry_judge.json）
"""
import sys, json, os, hashlib
import numpy as np
from multiprocessing import Pool
from trade import s0, s3, s5, vstar, vstar3
from experiments.explore import composition

PRE = "experiments/prereg_s5a.json"
DRY = os.environ.get("S5_DRY") == "1"
OUT = "out/s5a"
MAIN = f"{OUT}/dry.jsonl" if DRY else f"{OUT}/main.jsonl"
JUDGE = f"{OUT}/dry_judge.json" if DRY else f"{OUT}/judge.json"
PATS = ["α=3", "二群"]
NUS = [0.0, 0.05, 0.10, 0.30]
IMP = dict(credit=20.0, settle=True, reconsent="all")
BASES = {"TP+top3": dict(protocol="topk", ask=3), "改良版top3": dict(IMP, protocol="topk", ask=3)}
F0 = s0.f_design()


def actors_of(arm):
    """腕（運用役の置き方）→ 運用役の一覧。名は記録の GA の鍵になる"""
    return {
        "基準": [],
        "狭い盤": [s5.Designer(s5.NARROW)],
        "戻せる手の多い盤": [s5.Designer(s5.RICH)],
        "記録なし": [s5.Recorder("none")],
        "同意だけ記録": [s5.Recorder("consent")],
        "額まで記録": [s5.Recorder("amount")],
        "一律に禁止": [s5.Verifier("ban")],
        "代わりの手": [s5.Verifier("alt")],
    }[arm]


ARMS = ["基準", "狭い盤", "戻せる手の多い盤", "記録なし", "同意だけ記録", "額まで記録", "一律に禁止", "代わりの手"]
ACTOR_OF = {"狭い盤": "designer", "戻せる手の多い盤": "designer", "記録なし": "recorder", "同意だけ記録": "recorder",
            "額まで記録": "recorder", "一律に禁止": "verifier", "代わりの手": "verifier"}


def newW_of(pat):
    return lambda rng, i: composition(pat, int(rng.integers(10**6, 10**7)))[i]


def kw_of(base, nu, pat):
    kw = dict(BASES[base])
    if nu > 0:
        kw.update(turnover=dict(kind="geo", nu=nu), newW=newW_of(pat))
    return kw


def one(a):
    pat, cs, nu, base, arm = a
    W = composition(pat, cs)
    r = s5.run(W, "T1+", cs, actors=actors_of(arm), **kw_of(base, nu, pat))
    rec = vstar3.record3(r, W, None, F0)
    if "Zc" not in rec:                        # 運用役のない局にも、制約のもとの物差し（感度の分析）を足す
        vstar3.cap_fields(rec, r, s5.BASE)
    rec.update(pat=pat, cs=cs, nu=nu, base=base, arm=arm, n_exit=r["n_exit"],
               recon_paid=sum(1 for x in r["recon"] if x[3] == "paid"), recon_revert=sum(1 for x in r["recon"] if x[3] == "revert"),
               n_ledger=len(r.get("ledger", [])), fin_e=[int(x) for x in r["e"]])
    return rec


# ------------------------------------------------------------ 観察 2・確証の手順を s5 で写したもの（一致の試験に使う）
def one_obs2(a):
    """experiments/s3_rent.one と同じ手順で、模型だけ s5.run（運用役なし）、記録だけ vstar3.record3"""
    from experiments import s3_rent as S3
    grp, cname, reg, spec, pat, cs = a
    W = composition(pat, cs)
    rr, kw0 = S3.REG[reg]
    kw = dict(kw0); kw.update(S3.kwargs_of(spec, W, pat))
    r = s5.run(W, rr, cs, **kw)
    rec = vstar3.record3(r, W, None, S3.F0)
    ex = r["extra"]
    U = r["U"]; SF = s0.S_all(r["W_final"], *S3.F0)
    stay = ~r["moved"]
    rec.update(grp=grp, cond=cname, reg=reg, pat=pat, cs=cs, exit_rate=r["n_exit"] / 270,
               L_stay=float(np.mean(U[stay] < SF[stay] - 1e-9)) if stay.any() else float("nan"),
               L_new=float(np.mean(U[~stay] < SF[~stay] - 1e-9)) if (~stay).any() else float("nan"),
               changed=int(sum(1 for p in range(21) if rec["X"][p] != vstar.cfgvec(*r["ref"])[p])),
               owner_in=ex["owner_in"], refund=ex["refund"], restored=ex["restored"], tcost=ex["tcost"],
               unreached=ex["unreached"], recon_paid=sum(1 for x in r["recon"] if x[3] == "paid"),
               recon_revert=sum(1 for x in r["recon"] if x[3] == "revert"))
    return rec


def one_conf(a):
    """experiments/s4_confirm.one と同じ手順（TP+top3・改良版top3 のとき）で、模型だけ s5.run"""
    from experiments import s4_confirm as S4
    pat, cs, nu, cname, reg, rep = a[:6]
    extra = len(a) > 6 and a[6]
    rr, kw, sa = S4.REGS[reg]
    assert reg in ("TP+top3", "改良版top3")
    kw = dict(kw)
    W = composition(pat, cs)
    seed = cs + 50000 * rep
    cost, rcost = S4.COSTS[cname]
    newW = (lambda rng, i: composition(pat, int(rng.integers(10**6, 10**7)))[i]) if nu > 0 else None
    r = s5.run(W, rr, seed, nu=nu, newW=newW, cost=cost, rcost=rcost, **kw)
    from trade import vstar2
    rec = vstar2.record2(r, W, sa, S4.F0)
    rec.update(pat=pat, cs=cs, nu=nu, cost=cname, reg=reg, rep=rep)
    if extra:
        cur, cen, former = vstar.knowledge(r, sa)
        rec["Gbest"] = vstar.guess(r["W_final"], max(cur + [former], key=len)).tolist()
    return rec


def _pool_run(T, path, every=1000):
    os.makedirs(OUT, exist_ok=True)
    print(len(T), "局", flush=True)
    with Pool(2) as pool, open(path, "w") as f:
        for i, rec in enumerate(pool.imap_unordered(one, T, chunksize=8)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % every == 0:
                print(i, flush=True)
    print("done", flush=True)


def load(path):
    cells = {}
    for line in open(path):
        r = json.loads(line)
        cells.setdefault((r["base"], r["arm"], r["pat"], r["nu"]), []).append(r)
    for k in cells:
        cells[k].sort(key=lambda r: r["cs"])
    return cells


# ------------------------------------------------------------ 選ぶ用での確かめ（事前登録の前）
DEV_CS = list(range(0, 40))


def dev():
    T = [(p, cs, nu, b, arm) for p in PATS for nu in NUS for b in BASES for arm in ARMS for cs in DEV_CS]
    _pool_run(T, f"{OUT}/dev.jsonl", every=1000)
    dev_values()


def _val(a):
    recs, count, G_key, B = a
    pv = vstar3.population3(recs, "Xn", None, count, G_key)
    lo, hi = vstar3.boot3(recs, "Xn", n=B, seed=0, count=count, G_key=G_key)
    return pv["V"], lo, hi, pv["V_own"], pv["I"], pv["Imax"]


def dev_values():
    cells = load(f"{OUT}/dev.jsonl")
    keys = sorted(cells)
    jobs = []
    for k in keys:
        recs = cells[k]
        act = ACTOR_OF.get(k[1])
        jobs.append((k, "外", (recs, (), None, 200)))
        if act:
            jobs.append((k, "数", (recs, (act,), None, 200)))
        if k[1] == "額まで記録":
            jobs.append((k, "外・全員が読む", (recs, (), "all", 200)))
            jobs.append((k, "外・読まない", (recs, (), "None", 200)))
    with Pool(2) as pool:
        res = pool.map(_val, [j[2] for j in jobs], chunksize=2)
    out = {}
    for (k, nm, _), v in zip(jobs, res):
        recs = cells[k]
        row = out.setdefault("|".join(map(str, k)), dict(
            eta=float(np.nanmean([r["eta"] for r in recs])), eta_own=float(np.nanmean([r.get("eta_own", r["eta"]) for r in recs])),
            L=float(np.mean([r["L"] for r in recs])), moves=float(np.mean([r["moves"] for r in recs])),
            quotes=float(np.mean([r["quotes"] for r in recs])), n=len(recs),
            nfacts=float(np.mean([sum(r.get("actor_nfacts", {}).values()) for r in recs])),
            n_blocked=float(np.mean([r.get("n_blocked", 0) for r in recs])),
            n_banned=float(np.mean([sum(s.get("n_banned", 0) for s in r.get("actor_stats", {}).values()) for r in recs])),
            n_alt=float(np.mean([sum(s.get("n_alt", 0) for s in r.get("actor_stats", {}).values()) for r in recs])),
            recon_paid=float(np.mean([r["recon_paid"] for r in recs])), n_ledger=float(np.mean([r["n_ledger"] for r in recs]))))
        row["Vn" + nm] = v
    json.dump(out, open(f"{OUT}/dev_values.json", "w"), ensure_ascii=False, indent=1)
    for k, v in out.items():
        s = f"{k:34s} η={v['eta']:.3f}（自盤 {v['eta_own']:.3f}） 手={v['moves']:5.1f} 問={v['quotes']:6.1f}  V*n〔外〕={v['Vn外'][0]:.3f}［{v['Vn外'][1]:.3f},{v['Vn外'][2]:.3f}］"
        if "Vn数" in v:
            s += f" 〔数〕={v['Vn数'][0]:.3f}"
        if "Vn外・全員が読む" in v:
            s += f" 全員が読む={v['Vn外・全員が読む'][0]:.3f} 読まない={v['Vn外・読まない'][0]:.3f}"
        if v["n_banned"]:
            s += f" 禁={v['n_banned']:.0f} 代={v['n_alt']:.0f} 止={v['n_blocked']:.1f}"
        print(s)


# ------------------------------------------------------------ 確かめ用の走行
def tasks_main(pre, CS):
    return [(p, cs, nu, b, arm) for p in pre["patterns"] for nu in pre["nus"] for b in pre["bases"]
            for arm in pre["arms"][b] for cs in CS]


PARTS = f"{OUT}/dry_parts" if DRY else f"{OUT}/main_parts"


def run():
    """基準 × 腕 × パターン × 退去率 の組ごとに別のファイルへ書く（途中で止まっても、そろった組は走らせ直さない）。
    すべての組がそろったら、決まった順に main.jsonl へまとめる"""
    pre = json.load(open(PRE))
    CS = list(range(pre["seeds"][0], pre["seeds"][1] + 1)) if not DRY else list(range(0, 10))
    os.makedirs(PARTS, exist_ok=True)
    groups = [(b, arm, p, nu) for b in pre["bases"] for arm in pre["arms"][b] for p in pre["patterns"] for nu in pre["nus"]]
    with Pool(2) as pool:
        for g in groups:
            fn = f"{PARTS}/{g[0]}_{g[1]}_{g[2]}_{g[3]}.jsonl"
            if os.path.exists(fn) and sum(1 for _ in open(fn)) == len(CS):
                continue
            T = [(g[2], cs, g[3], g[0], g[1]) for cs in CS]
            recs = pool.map(one, T, chunksize=5)
            with open(fn + ".tmp", "w") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(fn + ".tmp", fn)
            print("組", g, flush=True)
    with open(MAIN, "w") as f:
        for g in groups:
            for line in open(f"{PARTS}/{g[0]}_{g[1]}_{g[2]}_{g[3]}.jsonl"):
                f.write(line)
    print("done", len(groups) * len(CS), "局", flush=True)


# ------------------------------------------------------------ 判定（凍結した規則どおり）
def eta_diff(A, B, n, seed, col="eta"):
    """η の組ごとの差（A − B、同じ構成）の平均と、構成を引き直すブートストラップの 95 % 区間"""
    assert [x["cs"] for x in A] == [x["cs"] for x in B]
    d = np.array([a[col] - b[col] for a, b in zip(A, B)])
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return dict(d=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))


def _vn_job(a):
    kind, recsA, recsB, cA, cB, gA, gB, B, BS, zk = a
    if kind == "val":
        v = vstar3.population3(recsA, "Xn", None, cA, gA, zk)
        lo, hi = vstar3.boot3(recsA, "Xn", n=B, seed=BS, count=cA, G_key=gA, Z_key=zk)
        return dict(v=v["V"], lo=lo, hi=hi, v_own=v["V_own"], I=v["I"], Imax=v["Imax"])
    return vstar3.diff3(recsA, recsB, "Xn", n=B, seed=BS, countA=cA, countB=cB, GA_key=gA, GB_key=gB, Z_key=zk,
                        field="V_own" if kind == "diff_own" else "V")


def passes(rule, x, margin):
    if rule == "lower":
        return x["hi"] < 0
    if rule == "higher":
        return x["lo"] > 0
    if rule == "not_lower":
        return x["lo"] > -margin
    if rule == "not_higher":
        return x["hi"] < margin
    if rule == "same":
        return x["lo"] >= -margin and x["hi"] <= margin
    raise ValueError(rule)


def judge():
    pre = json.load(open(PRE))
    bs = pre["bootstrap"]
    B = bs["n"] if not DRY else 30; BS = bs["seed"]; BE = bs["eta_n"] if not DRY else 200; BES = bs["eta_seed"]
    cells = load(MAIN)
    CS = list(range(pre["seeds"][0], pre["seeds"][1] + 1)) if not DRY else list(range(0, 10))
    for k in cells:
        assert len(cells[k]) == len(CS), (k, len(cells[k]))
    SET = [(p, nu) for p in pre["patterns"] for nu in pre["nus"]]
    C = lambda b, arm, p, nu: cells[(b, arm, p, nu)]
    # ---- ε（v4・v6 と同じ規則を、このデータの基準の腕に当てる）
    er = pre["epsilon"]
    eps = max(vstar.perm_floor(C(b, "基準", p, nu), "Xn", n=er["perm"] if not DRY else 20, seed=er["seed"])
              for b in er["bases"] for (p, nu) in SET)
    # ---- 値と区間：腕ごとに〔外〕、運用役の腕では〔数〕も。記録の額の腕では読み方の感度も
    jobs = []
    for b in pre["bases"]:
        for arm in pre["arms"][b]:
            act = ACTOR_OF.get(arm)
            for (p, nu) in SET:
                R = C(b, arm, p, nu)
                jobs.append((("val", b, arm, p, nu, "外"), ("val", R, None, (), (), None, None, B, BS, "Z")))
                if act:
                    jobs.append((("val", b, arm, p, nu, "数"), ("val", R, None, (act,), (), None, None, B, BS, "Z")))
                if arm == "額まで記録":
                    for gk, nm in (("all", "外・全員が読む"), ("None", "外・読まない")):
                        jobs.append((("val", b, arm, p, nu, nm), ("val", R, None, (), (), gk, None, B, BS, "Z")))
                if ACTOR_OF.get(arm) == "verifier" or arm == "基準":
                    jobs.append((("val", b, arm, p, nu, "外・制約のもとの最適"), ("val", R, None, (), (), None, None, B, BS, "Zc")))
    # ---- 差：予言に使う差と、表のための差（腕 − 比べる腕、同じ構成の組）
    diffs = {}
    for pr in pre["predictions"]:
        if not pr["quantity"].startswith("Vn"):
            continue
        for b in pr["bases"]:
            for (p, nu) in [tuple(c) for c in pr["cells"]]:
                diffs[(b, pr["arm"], pr["ref"], p, nu, pr["quantity"])] = None
    for b in pre["bases"]:
        for arm in pre["arms"][b]:
            if arm == "基準":
                continue
            for (p, nu) in SET:
                diffs.setdefault((b, arm, "基準", p, nu, "Vn外"), None)
                if arm == "額まで記録":
                    diffs.setdefault((b, arm, "基準", p, nu, "Vn数"), None)
    for (b, arm, ref, p, nu, q) in diffs:
        act = ACTOR_OF.get(arm)
        cA = (act,) if q == "Vn数" else ()
        cB = (ACTOR_OF[ref],) if (q == "Vn数" and ACTOR_OF.get(ref)) else ()
        jobs.append((("diff", b, arm, ref, p, nu, q), ("diff", C(b, arm, p, nu), C(b, ref, p, nu), cA, cB, None, None, B, BS, "Z")))
    for b in pre["bases"]:
        for (p, nu) in SET:
            jobs.append((("diff", b, "戻せる手の多い盤", "基準", p, nu, "Vn外_own"),
                         ("diff_own", C(b, "戻せる手の多い盤", p, nu), C(b, "基準", p, nu), (), (), None, None, B, BS, "Z")))
    cache_dir = f"{OUT}/dry_judge_parts" if DRY else f"{OUT}/judge_parts"
    os.makedirs(cache_dir, exist_ok=True)
    fns = [f"{cache_dir}/" + hashlib.sha1("|".join(map(str, j[0])).encode()).hexdigest()[:16] + ".json" for j in jobs]
    todo = [n for n, fn in enumerate(fns) if not os.path.exists(fn)]
    print(len(jobs), "の計算のうち", len(todo), "を計算する", flush=True)
    with Pool(2) as pool:
        for n, r in zip(todo, pool.imap(_vn_job, [jobs[n][1] for n in todo], chunksize=1)):
            json.dump(dict(key=list(map(str, jobs[n][0])), res=r), open(fns[n] + ".tmp", "w"), ensure_ascii=False)
            os.replace(fns[n] + ".tmp", fns[n])
    res = [json.load(open(fn))["res"] for fn in fns]
    val, dif = {}, {}
    for (key, _), r in zip(jobs, res):
        if key[0] == "val":
            val["|".join(map(str, key[1:]))] = r
        else:
            dif["|".join(map(str, key[1:]))] = r
    # ---- η・取り残し率など（組ごとの差）
    means, edif = {}, {}
    for b in pre["bases"]:
        for arm in pre["arms"][b]:
            for (p, nu) in SET:
                R = C(b, arm, p, nu)
                means[f"{b}|{arm}|{p}|{nu}"] = {c: float(np.nanmean([r.get(c, r["eta"]) if c.startswith("eta") else r.get(c, 0) for r in R]))
                                               for c in ("eta", "eta_own", "eta_cap", "L", "moves", "quotes", "n_ledger", "recon_paid", "n_blocked")}
                if arm != "基準":
                    for col in ("eta", "eta_cap", "L"):
                        edif[f"{b}|{arm}|基準|{p}|{nu}|{col}"] = eta_diff(R, C(b, "基準", p, nu), BE, BES, col)
    for pr in pre["predictions"]:
        if pr["quantity"] in ("eta", "eta_cap", "L"):
            for b in pr["bases"]:
                for (p, nu) in [tuple(c) for c in pr["cells"]]:
                    k = f"{b}|{pr['arm']}|{pr['ref']}|{p}|{nu}|{pr['quantity']}"
                    if k not in edif:
                        edif[k] = eta_diff(C(b, pr["arm"], p, nu), C(b, pr["ref"], p, nu), BE, BES, pr["quantity"])
    # ---- 予言の判定
    out = dict(epsilon=eps, n_configs=len(CS), bootstrap=B, eta_bootstrap=BE)
    J = {}
    for pr in pre["predictions"]:
        rows = []
        for b in pr["bases"]:
            for (p, nu) in [tuple(c) for c in pr["cells"]]:
                if pr["quantity"].startswith("Vn"):
                    x = dif[f"{b}|{pr['arm']}|{pr['ref']}|{p}|{nu}|{pr['quantity']}"]
                else:
                    x = edif[f"{b}|{pr['arm']}|{pr['ref']}|{p}|{nu}|{pr['quantity']}"]
                margin = eps if pr.get("margin") == "epsilon" else float(pr.get("margin", 0.0))
                rows.append(dict(base=b, pat=p, nu=nu, **x, ok=bool(passes(pr["rule"], x, margin))))
        n_ok = sum(r["ok"] for r in rows)
        verdict = "支持" if n_ok == len(rows) else ("一部支持" if n_ok > 0 else "不支持")
        J[pr["id"]] = dict(text=pr["text"], rule=pr["rule"], quantity=pr["quantity"], n_cells=len(rows), n_ok=n_ok, verdict=verdict, rows=rows)
    out["predictions"] = J
    # ---- 定義からの確かめ（判定ではない）：性向を知らない運用役は〔数〕＝〔外〕
    kchk = []
    for b in pre["bases"]:
        for arm in pre["arms"][b]:
            if ACTOR_OF.get(arm) and arm != "額まで記録":
                for (p, nu) in SET:
                    a_, c_ = val[f"{b}|{arm}|{p}|{nu}|外"], val[f"{b}|{arm}|{p}|{nu}|数"]
                    kchk.append(dict(base=b, arm=arm, pat=p, nu=nu, out=a_["v"], cnt=c_["v"], same=a_["v"] == c_["v"]))
    k2 = all(edif[f"{b}|額まで記録|基準|{p}|{nu}|eta"]["d"] == 0.0 for b in pre["bases"] for (p, nu) in SET)
    k3 = all(edif[f"改良版top3|記録なし|基準|{p}|0.0|eta"]["d"] == 0.0 and
             val[f"改良版top3|記録なし|{p}|0.0|外"]["v"] == val[f"改良版top3|基準|{p}|0.0|外"]["v"] for p in pre["patterns"])
    out["definition_check"] = dict(K1_rows=kchk, K1=all(x["same"] for x in kchk), K2=bool(k2), K3=bool(k3))
    out["values"] = val; out["diffs"] = dif; out["means"] = means; out["eta_diffs"] = edif
    out["prereg_sha256"] = hashlib.sha256(open(PRE, "rb").read()).hexdigest()
    json.dump(out, open(JUDGE, "w"), ensure_ascii=False, indent=1)
    print("ε =", round(eps, 4))
    for k, v in J.items():
        print(k, v["verdict"], f"{v['n_ok']}/{v['n_cells']}", v["text"][:60])
    print("定義からの確かめ", {k: v for k, v in out["definition_check"].items() if k != "K1_rows"})


if __name__ == "__main__":
    {"dev": dev, "dev_values": dev_values, "run": run, "judge": judge}[sys.argv[1]]()
