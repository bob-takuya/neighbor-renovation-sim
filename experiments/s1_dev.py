"""第一段階・開発：改良の部品（〔割〕〔貸〕〔清〕〔再〕）を、規則を選ぶ用の構成（cseed 0〜19）で試す。
python3 -m experiments.s1_dev run   → out/s1dev/runs.csv
規則を選ぶのに使った構成は、本走行（検証用、cseed 100〜）では使わない。
"""
import sys, os, csv, time
from multiprocessing import Pool
import numpy as np
from trade import s0
from experiments.explore import composition, hetero

OUT = "out/s1dev"; CACHE = "out/e1/cache"
os.makedirs(OUT, exist_ok=True)
PATS = ["α=30", "α=3", "α=0.5", "二群", "少数の極端", "階の不一致"]
NC, NS = 20, 3

# 候補：名前 → run() への引数（base は T1+ ＝ TP+）
CAND = {
    "TP+": dict(),
    "TP+割": dict(split=True),
    "TP+貸5": dict(credit=5.0),
    "TP+貸10": dict(credit=10.0),
    "TP+貸20": dict(credit=20.0),
    "TP+割貸10": dict(split=True, credit=10.0),
    "TP+清": dict(settle=True),
    "TP+清再": dict(settle=True, reconsent="rev"),
    "TP+清再全": dict(settle=True, reconsent="all"),
    "TP+割貸10清再全": dict(split=True, credit=10.0, settle=True, reconsent="all"),
    "TP+貸20清再": dict(credit=20.0, settle=True, reconsent="rev"),
    "TP+貸20清再全": dict(credit=20.0, settle=True, reconsent="all"),
}
# 条件 → (条件の引数, 試す候補)
PAY = ["TP+", "TP+割", "TP+貸5", "TP+貸10", "TP+貸20", "TP+割貸10"]
TURN = ["TP+", "TP+清", "TP+清再", "TP+清再全", "TP+割貸10清再全"]
CONDS = {
    "基準": (dict(), PAY),
    "C_下限20": (dict(budget="20"), PAY),
    "C_下限5": (dict(budget="5"), PAY),
    "C_半数払えない": (dict(budget="poor"), PAY),
    "D_入替3%": (dict(nu=0.03, pats=["α=30", "α=3", "α=0.5"]), TURN),
    "D_入替10%": (dict(nu=0.10, pats=["α=30", "α=3", "α=0.5"]), TURN),
}


def make_budget(kind, cseed):
    if kind == "20":
        return np.full(9, 20.0)
    if kind == "5":
        return np.full(9, 5.0)
    if kind == "poor":
        rng = np.random.default_rng([cseed, 5])
        return np.where(rng.permutation(9) < 4, 0.0, 1e9)
    return None


def evaluate(W, r, form="linear", cz=None, poor=None):
    eF, kF, oF = s0.f_design(form)
    Wf = r["W_final"]
    Wfin = float(r["S"].sum())
    SF = s0.S_all(Wf, eF, kF, oF, form); WF = float(SF.sum())
    wstar = s0.wstar_linear(Wf)[0] if cz is None else float(cz["wstar"])
    worse = r["U"] < SF - 1e-9
    d = dict(W=Wfin, WF=WF, Wstar=wstar,
             eta=(Wfin - WF) / (wstar - WF) if wstar - WF > 1e-6 else np.nan,
             L=float(worse.mean()), B=float(np.abs(r["b"]).sum() / 9),
             debt_max=float(-min(0.0, (r["m"] + r["c"]).min())),
             n_moves=sum(1 for x in r["events"] if x["ok"]), n_exit=r["n_exit"],
             n_recon=r["n_recon"], n_revert=r["n_revert"])
    mv = r["moved"]
    d["L_new"] = float(worse[mv].mean()) if mv.any() else np.nan
    d["L_stay"] = float(worse[~mv].mean()) if (~mv).any() else np.nan
    if poor is not None:
        d["L_poor"] = float(worse[poor].mean()); d["L_rich"] = float(worse[~poor].mean())
    return d


def task(args):
    cond, pattern, cseed = args
    C, cands = CONDS[cond]
    W = composition(pattern, cseed)
    budget = make_budget(C.get("budget"), cseed)
    poor = (budget == 0.0) if C.get("budget") == "poor" else None
    nu = C.get("nu", 0.0)
    alpha = float(pattern[2:]) if pattern.startswith("α=") else None
    newW = (lambda rng, i: rng.dirichlet(s0.W_BAR * alpha)) if nu > 0 else None
    cz = None if nu > 0 else np.load(f"{CACHE}/{pattern}_{cseed}_linear.npz")
    rows = []
    for name in cands:
        for sd in range(NS):
            r = s0.run(W, "T1+", seed=1000 * cseed + sd, budget=budget, nu=nu, newW=newW, **CAND[name])
            rows.append(dict(cond=cond, pattern=pattern, cseed=cseed, H=hetero(W), cand=name, seed=sd,
                             **evaluate(W, r, cz=cz, poor=poor)))
    return rows


def run():
    tasks = [(cond, p, c) for cond, (C, _) in CONDS.items() for p in C.get("pats", PATS) for c in range(NC)]
    t0 = time.time(); allrows = []
    with Pool(2) as pool:
        for n, rows in enumerate(pool.imap_unordered(task, tasks)):
            allrows += rows
            if n % 60 == 0:
                print("run", n, len(tasks), f"{time.time()-t0:.0f}s", flush=True)
    keys = sorted({k for r in allrows for k in r}, key=lambda k: list(allrows[0].keys()).index(k) if k in allrows[0] else 99)
    with open(f"{OUT}/runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(allrows)
    print("done", time.time() - t0, flush=True)


def run2():
    """組み合わせの候補だけを、6 条件すべてで"""
    global CONDS
    keep = ["TP+", "TP+清再全", "TP+貸20清再", "TP+貸20清再全"]
    CONDS = {k: (v[0], keep) for k, v in CONDS.items()}
    tasks = [(cond, p, c) for cond, (C, _) in CONDS.items() for p in C.get("pats", PATS) for c in range(NC)]
    t0 = time.time(); allrows = []
    with Pool(2) as pool:
        for n, rows in enumerate(pool.imap_unordered(task, tasks)):
            allrows += rows
            if n % 60 == 0:
                print("run2", n, len(tasks), f"{time.time()-t0:.0f}s", flush=True)
    keys = []
    for r in allrows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(f"{OUT}/runs2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(allrows)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    {"run": run, "run2": run2}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
