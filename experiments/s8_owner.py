"""観察 3：自分の収支で承諾を決める所有者（事前登録 v8）。模型 trade/s8.py。
  dev    選ぶ用の構成 cseed 0〜39 で、規則が意図どおり働くことと実行時間を確かめる → out/s8/dev.jsonl
  run    検証用の構成 cseed 600〜659（事前登録 v8 を凍結した後に一度だけ）→ out/s8/main.jsonl
  judge  experiments/prereg_s8.json の予言を、凍結した規則で判定 → out/s8/judge.json・cells.json
PYTHONPATH=. python3 -m experiments.s8_owner dev|run|judge
条件（主）：世界 E0・E1・E2（退去の効果なし・小・大）× 所有者 4（なし・評判・拒否・収支・拒否・収支・取引）× ν 0.10・0.30
           × 決め方 U・TP+top3・改良版top3。F（所有者の収支の基準）を ν ごとに。すべて R0（日本の既定）、費用 1 倍、見通し 10 年。
感度（判定しない）：収支・拒否 × E2 で、単価を一つずつ変える（λ・v1・c_turn・r0）、原状回復を R∅ にする、退去の効果の片方だけ。
"""
import sys, json, os, time, hashlib
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from trade import s0, s8, vstar
from experiments.explore import composition

PATS = ["α=3", "二群"]
C1 = (dict(e=1.0, k=0.25, o=0.25), dict(e=0.5, k=0.1, o=0.1))
COSTX = 3.0          # 主条件の費用は 3 倍（現実の工事費の下限に近いのは費用 3 倍の条件）。1 倍は感度に回す
IMP = dict(credit=20.0, settle=True, reconsent="all")
REG = {"F": ("F", {}), "U": ("U", {}), "TP+top3": ("T1+", dict(protocol="topk", ask=3)),
       "改良版top3": ("T1+", dict(IMP, protocol="topk", ask=3))}
OWN = {"なし": None,
       "評判・拒否": dict(kind="ind", role="veto", beta=0.5, gamma=1.0),
       "収支・拒否": dict(kind="econ", role="veto"),
       "収支・取引": dict(kind="econ", role="trade")}
WORLD = {"E0": None, "E1": dict(phi=0.05, psi=0.25), "E2": dict(phi=0.15, psi=0.5),
         "E2φ": dict(phi=0.15, psi=0.0), "E2ψ": dict(phi=0.0, psi=0.5)}
HZ = 10
# 単価（㎡ 換算）。1 単位＝平均的な世帯が広さ 1 ㎡ を見通しの 10 年使う価値、と読む（満足の差 dS/μ は 10 年住む世帯の値で、
# 見通し g はそれを住む年の割合で割り引く。補償はその一回払い。家賃は市場が平均的な世帯の値打ちで ㎡ を値付けするとして年に 0.1／㎡）。
# v0（空室 0.25 年）は入居率と平均居住期間から逆算した 0.22〜0.35 年の範囲。c_turn は家賃 1〜2 か月分（0.4〜0.8 単位）の中ほど 0.6。λ・v1 は根拠なしの仮の値
PRICE = dict(r0=5.0, lam=0.5, v0=0.25, v1=0.02, cturn=0.6, Ho=10, Hv=10)
SENS = {"λ=0.25": dict(lam=0.25), "λ=1": dict(lam=1.0), "v1=0": dict(v1=0.0), "v1=0.06": dict(v1=0.06),
        "c_turn=0.4": dict(cturn=0.4), "c_turn=1": dict(cturn=1.0), "v0=0.35": dict(v0=0.35),
        "30年読み": dict(r0=50 / 30, Hv=30)}
F0 = s0.f_design()


def conds():
    """(群, 条件名, 決め方, 指定)"""
    out = []
    for nu in (0.10, 0.30):
        out.append(("M", f"F ν={nu:.2f}", "F", dict(nu=nu, world="E0", own="なし")))
        for wd in ("E0", "E1", "E2"):
            for own in OWN:
                for rg in ("U", "TP+top3", "改良版top3"):
                    out.append(("M", f"{wd}・{own} ν={nu:.2f}", rg, dict(nu=nu, world=wd, own=own)))
        for rg in ("TP+top3", "改良版top3"):
            for sname, sv in SENS.items():
                out.append(("S", f"E2・収支・拒否・{sname} ν={nu:.2f}", rg, dict(nu=nu, world="E2", own="収支・拒否", price=sv)))
            out.append(("S", f"E2・収支・拒否・R∅ ν={nu:.2f}", rg, dict(nu=nu, world="E2", own="収支・拒否", R="R∅")))
            out.append(("S", f"E2・なし・R∅ ν={nu:.2f}", rg, dict(nu=nu, world="E2", own="なし", R="R∅")))
            for own in ("なし", "収支・拒否"):
                out.append(("S", f"E2・{own}・費用1倍 ν={nu:.2f}", rg, dict(nu=nu, world="E2", own=own, costx=1.0)))
            for wd in ("E2φ", "E2ψ"):
                out.append(("S", f"{wd}・収支・拒否 ν={nu:.2f}", rg, dict(nu=nu, world=wd, own="収支・拒否")))
    return out


def kwargs_of(spec, pat):
    price = dict(PRICE); price.update(spec.get("price", {}))
    kw = dict(turnover=dict(kind="geo", nu=spec["nu"]),
              newW=lambda rng, i: composition(pat, int(rng.integers(10**6, 10**7)))[i],
              cost={q: v * spec.get("costx", COSTX) for q, v in C1[0].items()},
              rcost={q: v * spec.get("costx", COSTX) for q, v in C1[1].items()}, horizon=HZ, landlord=price)
    R = spec.get("R", "R0")
    if R == "R0":
        kw["restore"] = "R0"
    own = OWN[spec["own"]]
    if own is not None:
        kw["owner"] = dict(own, **(price if own["kind"] == "econ" else {}))
    if WORLD[spec["world"]] is not None:
        kw["exitfx"] = WORLD[spec["world"]]
    return kw


def one(a):
    grp, cname, reg, spec, pat, cs = a
    W = composition(pat, cs)
    rr, kw0 = REG[reg]
    kw = dict(kw0); kw.update(kwargs_of(spec, pat))
    t0 = time.time()
    r = s8.run(W, rr, cs, **kw)
    rec = vstar.record(r, W, None, F0)
    U = r["U"]; SF = s0.S_all(r["W_final"], *F0)
    # 今の世帯が払った家賃の上乗せ（帳簿から）
    rent_now = np.zeros(9)
    cur = {int(r["tenant"][i]): i for i in range(9)}
    for (t, kd, f, to, amt, part, med) in r["ledger"]:
        if kd == "rent" and f in cur:
            rent_now[cur[f]] += amt
    opl = r["owner_pl"]
    rec.update(grp=grp, cond=cname, reg=reg, pat=pat, cs=cs, exit_rate=r["n_exit"] / 270, n_exit=r["n_exit"],
               L_rent=float(np.mean(U - s0.MU * rent_now < SF - 1e-9)),
               vetoed=r["extra"]["vetoed"], owner_in=r["extra"]["owner_in"],
               o_rent=opl["rent"], o_vac=opl["vacancy"], o_turn=opl["turn"], o_refund=opl["refund"], o_in=opl["paid_in"],
               o_net=opl["rent"] - opl["vacancy"] - opl["turn"] - opl["refund"] + opl["paid_in"],
               changed=int(sum(1 for p in range(21) if rec["X"][p] != vstar.cfgvec(*r["ref"])[p])),
               sec=time.time() - t0)
    return rec


def tasks(cs_list, groups="MS"):
    return [(g, c, rg, sp, p, cs) for (g, c, rg, sp) in conds() if g in groups for p in PATS for cs in cs_list]


def execute(T, fn):
    os.makedirs("out/s8", exist_ok=True)
    print(len(T), "局", flush=True)
    t0 = time.time()
    with Pool(2) as pool, open(fn, "w") as f:
        for i, rec in enumerate(pool.imap_unordered(one, T, chunksize=4)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                print(i, f"{time.time()-t0:.0f}s", flush=True)
    print("done", f"{time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------- 判定（事前登録 v8 の規則）
def load(fn):
    cells = defaultdict(list)
    for line in open(fn):
        r = json.loads(line)
        cells[(r["cond"], r["reg"], r["pat"])].append(r)
    for k in cells:
        cells[k].sort(key=lambda r: r["cs"])
    return cells


def judge(fn="out/s8/main.jsonl", prereg="experiments/prereg_s8.json", out="out/s8"):
    P = json.load(open(prereg, encoding="utf-8"))
    rule = P["判定の規則"]
    rng = np.random.default_rng(rule["乱数の種"])
    NB = rule["ブートストラップの回数"]; EQ = rule["同等の幅"]; DSMALL = rule["小さい差の効果量"]
    cells = load(fn)

    def recs(c, rg):
        return cells[(c, rg, "α=3")] + cells[(c, rg, "二群")]

    def paired(c1, r1, c2, r2, col):
        A = {(r["pat"], r["cs"]): r[col] for r in recs(c1, r1)}
        B = {(r["pat"], r["cs"]): r[col] for r in recs(c2, r2)}
        d = np.array([A[k] - B[k] for k in A if k in B and A[k] == A[k] and B[k] == B[k]])
        bs = d[rng.integers(0, len(d), (NB, len(d)))].mean(1)
        sd = d.std(ddof=1) if len(d) > 1 else 0.0
        return dict(mean=float(d.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), n=int(len(d)),
                    d=float(d.mean() / sd) if sd > 0 else (0.0 if d.mean() == 0 else float("inf")))

    def verdict(x, kind):
        if kind == "正":
            return x["lo"] > 0
        if kind == "負":
            return x["hi"] < 0
        if kind == "同等":
            return x["lo"] > -EQ and x["hi"] < EQ
        if kind == "非劣":
            return x["lo"] > -EQ
        raise ValueError(kind)

    J = {}
    for pid, pr in P["予言"].items():
        rows = []
        for cmpd in pr["比較"]:
            x = paired(cmpd["A"], cmpd["決め方A"], cmpd["B"], cmpd["決め方B"], cmpd["量"])
            x.update(A=cmpd["A"], B=cmpd["B"], reg=cmpd["決め方A"], col=cmpd["量"], 向き=cmpd["向き"], ok=bool(verdict(x, cmpd["向き"])),
                     small=bool(abs(x["d"]) < DSMALL))
            rows.append(x)
        J[pid] = dict(文=pr["文"], 支持=all(r["ok"] for r in rows), rows=rows)
    # セルの表（η・L・退去率・所有者の収支・V*n）
    eps_cells = [k for k in cells if "・なし " in k[0] and k[1] in ("U", "TP+top3", "改良版top3") and k[0].startswith("E")]
    EPS = max(vstar.perm_floor(cells[k], "Xn", n=100) for k in eps_cells) if eps_cells else float("nan")
    table = {}
    for (c, rg, pat) in cells:
        key = f"{c}|{rg}"
        if key in table:
            continue
        R = recs(c, rg)
        row = dict(cond=c, reg=rg)
        for col in ["eta", "L", "L_rent", "exit_rate", "moves", "vetoed", "owner_in", "o_rent", "o_vac", "o_turn", "o_net", "changed", "sec"]:
            v = [r[col] for r in R if r[col] == r[col]]
            row[col] = float(np.mean(v)) if v else float("nan")
        for p in PATS:
            Rp = cells[(c, rg, p)]
            v = vstar.population(Rp, "Xn")["V"]
            row[f"Vn_{p}"] = v
        table[key] = row
    h = hashlib.sha256(open(prereg, "rb").read()).hexdigest()
    json.dump(dict(prereg_sha256=h, eps=EPS, judge=J, n_support=sum(v["支持"] for v in J.values()), n=len(J)),
              open(f"{out}/judge.json", "w"), ensure_ascii=False, indent=1)
    json.dump(dict(eps=EPS, cells=table), open(f"{out}/cells.json", "w"), ensure_ascii=False, indent=0)
    for pid, v in J.items():
        print(pid, "支持" if v["支持"] else "不支持", [(round(r["mean"], 3), round(r["lo"], 3), round(r["hi"], 3)) for r in v["rows"]])


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "dev":
        execute(tasks(list(range(0, 40)), "M"), "out/s8/dev.jsonl")
    elif mode == "devS":
        execute(tasks(list(range(0, 20)), "S"), "out/s8/dev_sens.jsonl")
    elif mode == "run":
        execute(tasks(list(range(600, 660))), "out/s8/main.jsonl")
    elif mode == "judge":
        judge()
    elif mode == "judgedev":
        judge("out/s8/dev.jsonl", out="out/s8/devjudge")
