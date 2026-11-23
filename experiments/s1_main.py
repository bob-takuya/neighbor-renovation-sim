"""第一段階・本走行（検証用の構成）。改良版の中身と判定の規則は prereg_s1.json に凍結してから走らせる。
python3 -m experiments.s1_main cache   # 非線形の満足のための総当たり（A 条件の W*）
python3 -m experiments.s1_main main    # 主条件：16 パターン × 30 構成 × 10 局 × 9 決め方
python3 -m experiments.s1_main robust  # 前提を崩した 7 条件：16 パターン × 10 構成 × 5 局 × 6 決め方
検証用の構成は cseed 100〜129。規則を選ぶのに使った cseed 0〜19 とは重ならない。
"""
import sys, os, csv, json, time
from multiprocessing import Pool
import numpy as np
from trade import s0, enumerate as EN
from experiments.explore import composition, hetero

OUT = "out/s1"; CACHE = f"{OUT}/cache"
os.makedirs(CACHE, exist_ok=True)
PRE = json.load(open("experiments/prereg_s1.json", encoding="utf-8"))
PATS = PRE["patterns"]
IMP = PRE["improved"]["name"]
# 記号 → (内部の決め方, 追加の引数)
RULES = {"F": ("F", {}), "U": ("U", {}), "V50": ("V½", {}), "V75": ("V¾", {}),
         "TR": ("T0", {}), "TR+": ("T0+", {}), "TP": ("T1", {}), "TP+": ("T1+", {}),
         IMP: ("T1+", PRE["improved"]["kwargs"])}
MAIN_RULES = list(RULES)
ROB_RULES = ["F", "U", "V75", "TR+", "TP+", IMP]
ROB = {  # 条件 → run() への引数
    "A_逓減": dict(form="concave"), "A_下限": dict(form="floor"),
    "B_盛り1.5": dict(theta=0.5), "C_半数払えない": dict(budget="poor"),
    "D_入替10%": dict(nu=0.10), "E_白紙から": dict(start="blank"), "G_換算2": dict(conv=2.0),
}
C_MAIN = range(100, 130); C_ROB = range(100, 110)
NS_MAIN, NS_ROB = 10, 5


def cache_one(args):
    pattern, cseed, form = args
    path = f"{CACHE}/{pattern}_{cseed}_{form}.npz"
    if not os.path.exists(path):
        Ws, Sm = EN.enumerate_all(composition(pattern, cseed), form)
        np.savez(path, wstar=float(Ws.max()))
    return path


def one_run(W, rule, seed, cond_kw, pattern, cseed):
    reg, kw = RULES[rule]
    form = cond_kw.get("form", "linear")
    start = (np.zeros(9, int), np.zeros(6, int), np.zeros(6, int)) if cond_kw.get("start") == "blank" and reg != "F" else None
    budget = None
    if cond_kw.get("budget") == "poor":
        rng = np.random.default_rng([cseed, 5])
        budget = np.where(rng.permutation(9) < 4, 0.0, 1e9)
    nu = cond_kw.get("nu", 0.0)
    newW = None
    if nu > 0:   # 新しい世帯は、同じパターンの別の構成の、同じ住戸の性向から引く
        newW = lambda rng, i: composition(pattern, int(rng.integers(10**6, 10**7)))[i]
    r = s0.run(W, reg, seed=seed, form=form, start=start, theta=cond_kw.get("theta", 0.0), budget=budget,
               conv_scale=cond_kw.get("conv", 1.0), nu=nu, newW=newW, **kw)
    eF, kF, oF = s0.f_design(form)
    Wf = r["W_final"]
    SF = s0.S_all(Wf, eF, kF, oF, form)
    if form == "linear":
        wstar = s0.wstar_linear(Wf)[0]
    else:
        wstar = float(np.load(f"{CACHE}/{pattern}_{cseed}_{form}.npz")["wstar"])
    Wfin = float(r["S"].sum()); WF = float(SF.sum())
    worse = r["U"] < SF - 1e-9
    d = dict(W=Wfin, WF=WF, Wstar=wstar, dW=Wfin - WF,
             eta=(Wfin - WF) / (wstar - WF) if wstar - WF > 1e-6 else np.nan,
             L=float(worse.mean()), B=float(np.abs(r["b"]).sum() / 9),
             minS=float(r["S"].min()), minS_F=float(SF.min()),
             debt_max=float(-min(0.0, (r["m"] + r["c"]).min())),
             n_moves=sum(1 for x in r["events"] if x["ok"]), n_fail=sum(1 for x in r["events"] if not x["ok"]),
             n_exit=r["n_exit"], n_revert=r["n_revert"],
             cfg="".join(map(str, r["e"])) + "|" + "".join(str(x + 1) for x in r["k"]) + "|" + "".join(map(str, r["o"])))
    if budget is not None:
        poor = budget == 0.0
        d["L_poor"] = float(worse[poor].mean())
    return d


def task(args):
    cond, pattern, cseed, rules, ns = args
    kw = {} if cond == "主条件" else ROB[cond]
    W = composition(pattern, cseed)
    rows = []
    for rule in rules:
        for sd in range(ns):
            rows.append(dict(cond=cond, pattern=pattern, cseed=cseed, H=hetero(W), rule=rule, seed=sd,
                             **one_run(W, rule, 1000 * cseed + sd, kw, pattern, cseed)))
    return rows


def _pool(tasks, path, tag):
    t0 = time.time(); allrows = []
    with Pool(2) as pool:
        for n, rows in enumerate(pool.imap_unordered(task, tasks)):
            allrows += rows
            if n % 40 == 0:
                print(tag, n, len(tasks), f"{time.time()-t0:.0f}s", flush=True)
    keys = []
    for r in allrows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(allrows)
    print(tag, "done", f"{time.time()-t0:.0f}s", len(allrows), flush=True)


def cache():
    tasks = [(p, c, f) for f in ("concave", "floor") for p in PATS for c in C_ROB]
    t0 = time.time()
    with Pool(2) as pool:
        for n, _ in enumerate(pool.imap_unordered(cache_one, tasks)):
            if n % 40 == 0:
                print("cache", n, len(tasks), f"{time.time()-t0:.0f}s", flush=True)
    print("cache done", flush=True)


def main():
    _pool([("主条件", p, c, MAIN_RULES, NS_MAIN) for p in PATS for c in C_MAIN], f"{OUT}/main.csv", "main")


def robust():
    _pool([(cond, p, c, ROB_RULES, NS_ROB) for cond in ROB for p in PATS for c in C_ROB], f"{OUT}/robust.csv", "robust")


if __name__ == "__main__":
    {"cache": cache, "main": main, "robust": robust}[sys.argv[1]]()
