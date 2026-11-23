"""職能 3：性向を知りうる運用役。事前登録 v7c（experiments/prereg_s7.json、ハッシュ out/s7_frozen.sha256）。

運用役は知ることの少ない順に三つ：聞く範囲を選ぶ者（分布だけを知る・性向を知る）、AI を道具として使う建築家
（物理の量で見せる・満足の量で見せる）、全知の計画者（名付けの帰無の条件 N3 と同じ置き方）。
基準は運用役を置かない TP+top3 と改良版top3。同じ構成・同じ乱数（局の種 cseed）のもとで運用役だけを替え、構成ごとの組ごとの差をとる。
取引の条件は二つ：基準（費用なし・全員に届く）と、近い戸だけ（同じ階の隣と同じ列の上下一つにしか聞けない。職能 2 と同じ）。
退去率は 入れ替わりなし と 年 5 %・10 %・30 %（幾何の入れ替わり）、パターンは α＝3 と二群。
参照は、職能 2 の仲介者（届かなかった手だけ、近い戸だけの条件）と、全知の仲介者 BR・BRdyn。

聞く範囲を選ぶ者は、選ぶのに 1 局目（局の種 cseed＋50000、四つの範囲を固定して走らせる）を、測るのに 2 局目（局の種 cseed）を使う。
  順：(1) 運用役を置かない腕・固定した範囲の腕・1 局目・道具・仲介者・参照を走らせる → (2) 1 局目の記録から建物ごとの範囲を決める
  （out/s7/{dev,main}_choice.json）→ (3) 選ぶ者の腕を、選んだ範囲で走らせる。

PYTHONPATH=. python3 -m experiments.s7_prof3 dev        選ぶ用（cseed 0〜39）→ out/s7/dev_parts/・dev.jsonl
PYTHONPATH=. python3 -m experiments.s7_prof3 dev_values 選ぶ用の値と差（ブートストラップ 200 回）→ out/s7/dev_values.json
PYTHONPATH=. python3 -m experiments.s7_prof3 run        確かめ用（cseed は事前登録の seeds）→ out/s7/main_parts/・main.jsonl
PYTHONPATH=. python3 -m experiments.s7_prof3 judge      凍結した規則で判定 → out/s7/judge_parts/・judge.json
（S7_DRY=1 なら、選ぶ用の cseed 0〜9 で run と judge の全経路を試す → out/s7/dry_*）
どれも組ごと・計算ごとのファイルに書くので、途中で止まっても、そろった分は走らせ直さない。
"""
import sys, json, os, hashlib
import numpy as np
from multiprocessing import Pool
from trade import s0, s6, s7, vstar, vstar3, vstar4, vstar5
from experiments.explore import composition

PRE = "experiments/prereg_s7.json"
DRY = os.environ.get("S7_DRY") == "1"
OUT = "out/s7"
PATS = ["α=3", "二群"]
NUS = [0.0, 0.05, 0.10, 0.30]
NEARV = dict(G=0, q_far=0.0, tau=0.0)
CONDS = {"基準": {}, "近い戸だけ": dict(venue=NEARV)}
IMP = dict(credit=20.0, settle=True, reconsent="all")
BASES = {"TP+top3": dict(protocol="topk", ask=3), "改良版top3": dict(IMP, protocol="topk", ask=3)}
SCOPES = s7.ORDER                                       # 全員・上位1手・上位3手・近い戸だけ
G1_SHIFT = 50000                                        # 1 局目の局の種 ＝ cseed ＋ 50000（名付けの抵抗の試験の 2 局目と同じずらし）
FIXED = [f"範囲:{s}" for s in SCOPES if s != "上位3手"]  # 固定した範囲の 2 局目（上位3手は基準そのもの）
G1 = [f"1局目:{s}" for s in SCOPES]
SEL = {"選ぶ者:分布": False, "選ぶ者:性向": True}
TOOLS = {"道具:物理": "physical", "道具:満足": "satisfaction"}
MED = "仲介者"
REFS = {"計画者": "P", "BR": "BR", "BRdyn": "BRdyn"}
KIND = {**{a: "scope" for a in SEL}, **{a: "tool" for a in TOOLS}, MED: "broker"}
F0 = s0.f_design()


def newW_of(pat):
    return lambda rng, i: composition(pat, int(rng.integers(10**6, 10**7)))[i]


def one(a):
    """一局。a ＝ (パターン, cseed, 退去率, 取引の条件, 基準 or "参照", 腕, 選ぶ者の範囲 or None)"""
    pat, cs, nu, cond, base, arm, choice = a
    W = composition(pat, cs)
    kw = dict(CONDS[cond])
    seed = cs + G1_SHIFT if arm in G1 else cs
    if nu > 0:
        kw.update(turnover=dict(kind="geo", nu=nu), newW=newW_of(pat))
    acts = []
    if base == "参照":
        kw.pop("venue", None)                       # 計画者・BR は聞く範囲を使わない（近い戸だけの条件でも同じ局）
        r = s7.run(W, REFS[arm], seed, **kw)
    else:
        kw.update(BASES[base])
        if arm in FIXED or arm in G1:
            kw.update(s7.PROTOCOLS[arm.split(":")[1]])
        elif arm in SEL:
            acts = [s7.Scope(choice, SEL[arm])]
        elif arm in TOOLS:
            acts = [s7.Tool(TOOLS[arm])]
        elif arm == MED:
            acts = [s6.Mediator("stuck")]
        r = s7.run(W, "T1+", seed, actors=acts, **kw)
    rec = vstar3.record3(r, W, None, F0)
    st = {}
    for nm, d in r.get("actor_stats", {}).items():
        st.update(d)
    evs = [ev for ev in r["events"] if ev.get("ok") and ev.get("part") is not None]
    rec.update(pat=pat, cs=cs, nu=nu, cond=cond, base=base, arm=arm, seed=seed, choice=choice, n_exit=r["n_exit"],
               recon_paid=sum(1 for x in r["recon"] if x[3] == "paid"), unreached=r["extra"]["unreached"],
               n_mediated=sum(1 for ev in evs if ev.get("drafter") == "mediator"), spent=float(r["spent"]), **st)
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


def _spec(scope_conds):
    s = {}
    for b in BASES:
        s[b] = {"基準": list(CONDS)}
        for a in FIXED + G1:
            s[b][a] = list(scope_conds)
        for a in TOOLS:
            s[b][a] = list(CONDS)
        s[b][MED] = ["近い戸だけ"]
    s["参照"] = {a: ["基準"] for a in REFS}
    return s


def _sel_spec(scope_conds):
    return {b: {a: list(scope_conds) for a in SEL} for b in BASES}


DEV_SPEC = _spec(CONDS)                # 選ぶ用では、範囲を選ぶ者と道具を二つの条件で試す
DEV_SEL = _sel_spec(CONDS)
DEV_CS = list(range(0, 40))


def _main_spec():
    """確かめ用（選ぶ用を見た後に決めた）：範囲を選ぶ者は全員に届く条件だけ。物理の量で見せる道具は近い戸だけの条件だけ
    （全員に届く条件では定義から基準と同じ局。試験で確かめた）。満足の量で見せる道具は両方の条件（局は基準と同じでも、知ることが違う）"""
    s = {}
    for b in BASES:
        s[b] = {"基準": list(CONDS)}
        for a in FIXED + G1:
            s[b][a] = ["基準"]
        s[b]["道具:物理"] = ["近い戸だけ"]
        s[b]["道具:満足"] = list(CONDS)
        s[b][MED] = ["近い戸だけ"]
    s["参照"] = {a: ["基準"] for a in REFS}
    return s


MAIN_SPEC = _main_spec()
MAIN_SEL = _sel_spec(["基準"])


def gfile(parts, g):
    return f"{parts}/{'_'.join(map(str, g)).replace(':', '-')}.jsonl"


def run_groups(G, CS, parts, choices=None):
    os.makedirs(parts, exist_ok=True)
    with Pool(2) as pool:
        for g in G:
            fn = gfile(parts, g)
            if os.path.exists(fn) and sum(1 for _ in open(fn)) == len(CS):
                continue
            b, arm, cond, p, nu = g
            T = [(p, cs, nu, cond, b, arm, (choices[f"{b}|{arm}|{cond}|{p}|{nu}"][str(cs)] if arm in SEL else None)) for cs in CS]
            recs = pool.map(one, T, chunksize=4)
            with open(fn + ".tmp", "w") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(fn + ".tmp", fn)
            print("組", g, flush=True)


def merge(G, parts, merged):
    with open(merged + ".tmp", "w") as f:
        for g in G:
            for line in open(gfile(parts, g)):
                f.write(line)
    os.replace(merged + ".tmp", merged)


def load_parts(G, parts):
    cells = {}
    for g in G:
        for line in open(gfile(parts, g)):
            r = json.loads(line)
            cells.setdefault((r["base"], r["arm"], r["cond"], r["pat"], r["nu"]), []).append(r)
    for k in cells:
        cells[k].sort(key=lambda r: r["cs"])
    return cells


def load(path):
    cells = {}
    for line in open(path):
        r = json.loads(line)
        cells.setdefault((r["base"], r["arm"], r["cond"], r["pat"], r["nu"]), []).append(r)
    for k in cells:
        cells[k].sort(key=lambda r: r["cs"])
    return cells


def getter(cells):
    """C(基準, 腕, 条件, パターン, 退去率)。固定した範囲の上位3手は基準そのもの。参照は近い戸だけでも基準の条件の局。
    BRdyn は入れ替わりなしでは BR と同じ局。道具は全員に届く条件では基準と同じ局（走らせていない腕は基準を使う）"""
    def C(b, arm, cond, p, nu):
        if b == "参照":
            if cond == "近い戸だけ":
                cond = "基準"
            if arm == "BRdyn" and nu == 0:
                arm = "BR"
        if arm == "範囲:上位3手":
            arm = "基準"
        return cells[(b, arm, cond, p, nu)]
    return C


def choose(G1groups, cells, path):
    """1 局目の記録から、建物ごとの範囲を決める（性向を知る者）と、セルごとに一つの範囲を決める（分布だけを知る者）"""
    out = {}; info = {}
    keys = sorted({(b, cond, p, nu) for (b, arm, cond, p, nu) in G1groups})
    for (b, cond, p, nu) in keys:
        by = {s: cells[(b, f"1局目:{s}", cond, p, nu)] for s in SCOPES}
        kn, pw = vstar5.choose_knowing(by, SCOPES)
        dist, dv = vstar5.choose_distribution(by, SCOPES)
        out[f"{b}|選ぶ者:性向|{cond}|{p}|{nu}"] = {str(cs): s for cs, s in kn.items()}
        out[f"{b}|選ぶ者:分布|{cond}|{p}|{nu}"] = {str(cs): dist for cs in kn}
        info[f"{b}|{cond}|{p}|{nu}"] = dict(knowing_counts={s: sum(v == s for v in kn.values()) for s in SCOPES},
                                            distribution=dist, g1_values=dv)
    json.dump(dict(choices=out, info=info), open(path + ".tmp", "w"), ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    return out


def three_phase(spec, sel_spec, pats, nus, CS, parts, choice_path, merged):
    G = groups(spec, pats, nus)
    run_groups(G, CS, parts)                                                     # (1)
    cells = load_parts(G, parts)
    G1g = [g for g in G if g[1] in G1]
    ch = choose(G1g, cells, choice_path)                                         # (2)
    S = groups(sel_spec, pats, nus)
    run_groups(S, CS, parts, ch)                                                 # (3)
    merge(G + S, parts, merged)
    print("done", len(G + S) * len(CS), "局", flush=True)


# ------------------------------------------------------------ 選ぶ用
def dev():
    three_phase(DEV_SPEC, DEV_SEL, PATS, NUS, DEV_CS, f"{OUT}/dev_parts", f"{OUT}/dev_choice.json", f"{OUT}/dev.jsonl")


def eta_diff(A, B, n, seed, col="eta"):
    """η の組ごとの差（A − B、同じ構成）の平均と、構成を引き直すブートストラップの 95 % 区間（s6_prof2.eta_diff と同じ式）"""
    assert [x["cs"] for x in A] == [x["cs"] for x in B]
    d = np.array([a[col] - b[col] for a, b in zip(A, B)])
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return dict(d=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))


def prep(recs, arm, q):
    """量 q（"Vn外" か "Vn数"）のための (局の写し, 作者に数える運用役)。
    運用役（範囲を選ぶ者・道具・仲介者）は役の枠（GA）で数える。計画者・BR は中央の作者の枠で数え、〔外〕ではその枠を外す"""
    if arm in REFS:
        return (recs if q == "Vn数" else vstar4.drop_central(recs)), ()
    if arm in KIND and q == "Vn数":
        return recs, (KIND[arm],)
    return recs, ()


def has_actor(arm):
    return arm in KIND or arm in REFS


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
    """組ごとの値：〔外〕と、運用役のある腕では〔数〕。1 局目の腕は値を求めない（選ぶのにだけ使う）"""
    jobs = []
    for (b, arm, cond, p, nu) in G:
        if arm in G1:
            continue
        R = C(b, arm, cond, p, nu)
        for q in (("Vn外", "Vn数") if has_actor(arm) else ("Vn外",)):
            R2, cnt = prep(R, arm, q)
            jobs.append((("val", b, arm, cond, p, nu, q), ("val", R2, cnt, B, BS)))
    return jobs


def diff_job(C, b, arm, ref, cond, p, nu, q, B, BS):
    """差（腕 − 比べる腕、同じ構成の組）。参照の腕の基準は "参照"。q は "Vn外"・"Vn数"（〔数〕は両方の腕の運用役を数える）。
    q が "Vn数対外" のときは、同じ腕の〔数〕−〔外〕（ref は arm と同じ）。作者の数え方の約束に依る分"""
    ab = "参照" if arm in REFS else b
    rb = "参照" if ref in REFS else b
    if q == "Vn数対外":
        assert ref == arm
        RA, cA = prep(C(ab, arm, cond, p, nu), arm, "Vn数")
        RB, cB = prep(C(rb, ref, cond, p, nu), ref, "Vn外")
        return (("diff", b, arm, ref, cond, p, nu, q), ("diff", RA, RB, cA, cB, B, BS))
    RA, cA = prep(C(ab, arm, cond, p, nu), arm, q)
    RB, cB = prep(C(rb, ref, cond, p, nu), ref, q)
    return (("diff", b, arm, ref, cond, p, nu, q), ("diff", RA, RB, cA, cB, B, BS))


STATS = ("eta", "L", "moves", "quotes", "unreached", "recon_paid", "n_mediated", "n_bundle", "n_consult", "n_shown")


def summarize(G, C):
    means = {}
    for (b, arm, cond, p, nu) in G:
        R = C(b, arm, cond, p, nu)
        k = "|".join(map(str, (b, arm, cond, p, nu)))
        means[k] = {c: float(np.nanmean([r.get(c, 0) for r in R])) for c in STATS}
        means[k]["nfacts"] = float(np.mean([sum(r.get("actor_nfacts", {}).values()) for r in R]))
        if arm in KIND:
            means[k]["I_actor"] = vstar5.actor_I(R, KIND[arm])
        if arm in SEL:
            means[k]["choice_counts"] = {s: sum(r["choice"] == s for r in R) for s in SCOPES}
    return means


# 選ぶ用で見る差（予言の候補）：腕 − 比べる腕
def table_diffs(G):
    D = []
    Gset = set(G)
    for (b, arm, cond, p, nu) in G:
        if b == "参照" or arm in G1 or arm == "基準":
            continue
        for q in ("Vn外", "Vn数") if arm in KIND else ("Vn外",):
            D.append((b, arm, "基準", cond, p, nu, q))
        if arm in ("選ぶ者:性向", "道具:満足", MED):
            D.append((b, arm, arm, cond, p, nu, "Vn数対外"))
        if arm == "選ぶ者:性向":
            for q in ("Vn外", "Vn数"):
                D.append((b, arm, "選ぶ者:分布", cond, p, nu, q))
            if nu == 0.0:
                for f in FIXED:
                    D.append((b, arm, f, cond, p, nu, "Vn外"))
        if arm == "道具:満足" and (b, "道具:物理", cond, p, nu) in Gset:
            D.append((b, arm, "道具:物理", cond, p, nu, "Vn数"))
        if arm in TOOLS and cond == "近い戸だけ":
            D.append((b, arm, MED, cond, p, nu, "Vn外"))
    for b in BASES:
        for p in PATS:
            for nu in NUS:
                for ref in REFS:
                    if ref == "BRdyn" and nu == 0:
                        continue
                    D.append((b, ref, "基準", "基準", p, nu, "Vn外"))
    return D


def eta_pairs(G):
    E = []
    for (b, arm, cond, p, nu) in G:
        if b == "参照" or arm in G1 or arm == "基準":
            continue
        E.append((b, arm, "基準", cond, p, nu))
        if arm in TOOLS and cond == "近い戸だけ":
            E.append((b, arm, MED, cond, p, nu))
    for b in BASES:
        for p in PATS:
            for nu in NUS:
                for ref in REFS:
                    if ref == "BRdyn" and nu == 0:
                        continue
                    E.append((b, ref, "基準", "基準", p, nu))
    return E


def eta_of(C, b, arm, ref, cond, p, nu, n, seed):
    ab = "参照" if arm in REFS else b
    rb = "参照" if ref in REFS else b
    return eta_diff(C(ab, arm, cond, p, nu), C(rb, ref, cond, p, nu), n, seed)


def dev_values():
    G = groups(DEV_SPEC, PATS, NUS) + groups(DEV_SEL, PATS, NUS)
    cells = load(f"{OUT}/dev.jsonl")
    C = getter(cells)
    jobs = value_jobs(G, C, 200, 0)
    for b in BASES:
        for cond in CONDS:
            for p in PATS:
                for nu in NUS:
                    jobs.append((("eps", b, cond, p, nu), ("eps", C(b, "基準", cond, p, nu), 100, 1)))
    for (b, arm, ref, cond, p, nu, q) in table_diffs(G):
        jobs.append(diff_job(C, b, arm, ref, cond, p, nu, q, 200, 0))
    res = cached(jobs, f"{OUT}/dev_cache")
    out = dict(values={}, diffs={}, eps={}, eta_diffs={})
    for (key, _), r in zip(jobs, res):
        k = "|".join(map(str, key[1:]))
        out[{"val": "values", "diff": "diffs", "eps": "eps"}[key[0]]][k] = r
    for (b, arm, ref, cond, p, nu) in eta_pairs(G):
        out["eta_diffs"][f"{b}|{arm}|{ref}|{cond}|{p}|{nu}|eta"] = eta_of(C, b, arm, ref, cond, p, nu, 2000, 1)
    out["means"] = summarize(G, C)
    out["epsilon_dev"] = max(out["eps"].values())
    out["choice_info"] = json.load(open(f"{OUT}/dev_choice.json"))["info"]
    json.dump(out, open(f"{OUT}/dev_values.json", "w"), ensure_ascii=False, indent=1)
    print("ε（選ぶ用）", round(out["epsilon_dev"], 4))


# ------------------------------------------------------------ 確かめ用
PARTS = f"{OUT}/dry_parts" if DRY else f"{OUT}/main_parts"
MAIN = f"{OUT}/dry.jsonl" if DRY else f"{OUT}/main.jsonl"
CHOICE = f"{OUT}/dry_choice.json" if DRY else f"{OUT}/main_choice.json"
JUDGE = f"{OUT}/dry_judge.json" if DRY else f"{OUT}/judge.json"


def pre_groups(pre):
    return groups(pre["arms"], pre["patterns"], pre["nus"]), groups(pre["selector_arms"], pre["patterns"], pre["nus"])


def pre_cs(pre):
    return list(range(pre["seeds"][0], pre["seeds"][1] + 1)) if not DRY else list(range(0, 10))


def run():
    pre = json.load(open(PRE))
    G, S = pre_groups(pre)
    three_phase(pre["arms"], pre["selector_arms"], pre["patterns"], pre["nus"], pre_cs(pre), PARTS, CHOICE, MAIN)


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
    CS = pre_cs(pre)
    G0, S = pre_groups(pre)
    G = G0 + S
    for k in cells:
        assert len(cells[k]) == len(CS), (k, len(cells[k]))
    er = pre["epsilon"]
    jobs = [(("eps", b, cond, p, nu), ("eps", C(b, "基準", cond, p, nu), er["perm"] if not DRY else 20, er["seed"]))
            for b in er["bases"] for cond in er["conds"] for p in pre["patterns"] for nu in pre["nus"]]
    jobs += value_jobs(G, C, B, BS)
    want = {}
    for pr in pre["predictions"]:
        if pr["quantity"].startswith("Vn"):
            for b in pr["bases"]:
                for (cond, p, nu) in [tuple(c) for c in pr["cells"]]:
                    want[(b, pr["arm"], pr["ref"], cond, p, nu, pr["quantity"])] = None
    for d in table_diffs(G):
        want.setdefault(d, None)
    for (b, arm, ref, cond, p, nu, q) in want:
        jobs.append(diff_job(C, b, arm, ref, cond, p, nu, q, B, BS))
    res = cached(jobs, f"{OUT}/dry_judge_parts" if DRY else f"{OUT}/judge_parts")
    val, dif, epsd = {}, {}, {}
    for (key, _), r in zip(jobs, res):
        k = "|".join(map(str, key[1:]))
        {"val": val, "diff": dif, "eps": epsd}[key[0]][k] = r
    eps = max(epsd.values())
    edif = {}
    for (b, arm, ref, cond, p, nu) in eta_pairs(G):
        edif[f"{b}|{arm}|{ref}|{cond}|{p}|{nu}|eta"] = eta_of(C, b, arm, ref, cond, p, nu, BE, BES)
    for pr in pre["predictions"]:
        if pr["quantity"] == "eta":
            for b in pr["bases"]:
                for (cond, p, nu) in [tuple(c) for c in pr["cells"]]:
                    k = f"{b}|{pr['arm']}|{pr['ref']}|{cond}|{p}|{nu}|eta"
                    if k not in edif:
                        edif[k] = eta_of(C, b, pr["arm"], pr["ref"], cond, p, nu, BE, BES)
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
    k1 = [dict(pat=p, nu=nu, v=val[f"参照|計画者|基準|{p}|{nu}|Vn数"]["v"]) for p in pre["patterns"] for nu in pre["nus"]]
    k2 = [dict(pat=p, nu=nu, eta=float(np.mean([r["eta"] for r in C("参照", "計画者", "基準", p, nu)]))) for p in pre["patterns"] for nu in pre["nus"]]
    k3 = [dict(pat=p, nu=nu, v=val[f"参照|計画者|基準|{p}|{nu}|Vn外"]["v"]) for p in pre["patterns"] for nu in pre["nus"]]
    k4 = [dict(key=k, out=val[k.replace("Vn数", "Vn外")]["v"], cnt=v["v"]) for k, v in val.items()
          if k.endswith("Vn数") and not k.startswith("参照")]
    k5 = [dict(key=k, out=val[k.replace("Vn数", "Vn外")]["v"], cnt=v["v"]) for k, v in val.items()
          if k.endswith("Vn数") and ("|選ぶ者:分布|" in k or "|道具:物理|" in k)]
    k6 = [dict(base=b, p=p, v=val[f"{b}|選ぶ者:性向|基準|{p}|0.0|Vn数"]["v"]) for b in BASES for p in pre["patterns"]
          if f"{b}|選ぶ者:性向|基準|{p}|0.0|Vn数" in val]
    k6 += [dict(base=b, p=p, cond=cond, v=val[f"{b}|道具:満足|{cond}|{p}|0.0|Vn数"]["v"]) for b in BASES for p in pre["patterns"]
           for cond in CONDS if f"{b}|道具:満足|{cond}|{p}|0.0|Vn数" in val]
    out["definition_check"] = dict(
        K1_rows=k1, K1=all(x["v"] == 0.0 for x in k1),
        K2_rows=k2, K2=all(abs(x["eta"] - 1.0) < 1e-9 for x in k2),
        K3_rows=k3, K3=all(x["v"] == 0.0 for x in k3),
        K4_rows=k4, K4=all(x["cnt"] <= x["out"] + 1e-12 for x in k4),
        K5_rows=k5, K5=all(abs(x["cnt"] - x["out"]) < 1e-12 for x in k5),
        K6_rows=k6, K6=all(x["v"] == 0.0 for x in k6))
    out["values"] = val; out["diffs"] = dif; out["eta_diffs"] = edif; out["eps_cells"] = epsd
    out["means"] = summarize(G, C)
    out["choice_info"] = json.load(open(CHOICE))["info"]
    out["prereg_sha256"] = hashlib.sha256(open(PRE, "rb").read()).hexdigest()
    json.dump(out, open(JUDGE, "w"), ensure_ascii=False, indent=1)
    print("ε =", round(eps, 4))
    for k, v in J.items():
        print(k, v["verdict"], f"{v['n_ok']}/{v['n_cells']}", v["text"][:70])
    print("定義からの確かめ", {k: v for k, v in out["definition_check"].items() if not k.endswith("_rows")})


if __name__ == "__main__":
    {"dev": dev, "dev_values": dev_values, "run": run, "judge": judge}[sys.argv[1]]()
