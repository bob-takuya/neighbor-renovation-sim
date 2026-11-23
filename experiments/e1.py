"""探索 E1：E0 の結果が自明か、仮定のせいかを確かめる。
python3 -m experiments.e1 cache   # 構成ごとの総当たりの要約をつくる（線形・逓減・下限つき）
python3 -m experiments.e1 run     # 条件ごとに走らせる → out/e1/runs.csv

条件（いずれも E0 と同じ構成 cseed 0〜19、同じ乱数 seed 1000·cseed＋0〜2 で対応をとる）
  A  満足の関数：逓減（concave）・下限つき（floor）。線形では最適が部位ごとに分かれ、一手ずつの改善で必ず届く。形を変えると届かなくなるか
  B  駆け引き：損を 1.2 倍・1.5 倍に言う
  C  払えない：持ち点の下限 20・5 ㎡ 換算、半数の世帯が払えない（下限 0）
  D  入れ替わり：毎年 3 %・10 % が転出し、新しい性向の世帯が入る（ディリクレの 3 パターンのみ）
  E  始まりの状態：F の形ではなく、何もない状態から
  G  換算の誤り：共通の物差しで、日照・静けさ・近さを広さに対して 0.5 倍・2 倍に見る（T0 系だけに効く）
"""
import sys, os, json, time, csv
from multiprocessing import Pool
import numpy as np
from trade import s0, enumerate as EN
from experiments.explore import composition, hetero

OUT = "out/e1"; CACHE = f"{OUT}/cache"
os.makedirs(CACHE, exist_ok=True)
PATS = ["α=30", "α=3", "α=0.5", "二群", "少数の極端", "階の不一致"]
NC, NS = 20, 3
R8 = ["F", "U", "V½", "V¾", "T0", "T1", "T0+", "T1+"]

CONDS = {
    "A_逓減": dict(form="concave", regimes=R8),
    "A_下限": dict(form="floor", regimes=R8),
    "B_盛り1.2": dict(theta=0.2, regimes=["T0", "T1", "T0+", "T1+"]),
    "B_盛り1.5": dict(theta=0.5, regimes=["T0", "T1", "T0+", "T1+"]),
    "C_下限20": dict(budget="20", regimes=["T0", "T1", "T0+", "T1+"]),
    "C_下限5": dict(budget="5", regimes=["T0", "T1", "T0+", "T1+"]),
    "C_半数払えない": dict(budget="poor", regimes=["T0", "T1", "T0+", "T1+"]),
    "D_入替3%": dict(nu=0.03, regimes=["F", "U", "V¾", "T0+", "T1", "T1+"], pats=["α=30", "α=3", "α=0.5"]),
    "D_入替10%": dict(nu=0.10, regimes=["F", "U", "V¾", "T0+", "T1", "T1+"], pats=["α=30", "α=3", "α=0.5"]),
    "E_白紙から": dict(start="blank", regimes=R8),
    "G_換算0.5": dict(conv=0.5, regimes=["T0", "T0+"]),
    "G_換算2": dict(conv=2.0, regimes=["T0", "T0+"]),
}


def cache_one(args):
    pattern, cseed, form = args
    path = f"{CACHE}/{pattern}_{cseed}_{form}.npz"
    if os.path.exists(path):
        return path
    W = composition(pattern, cseed)
    Ws, Sm = EN.enumerate_all(W, form)
    flat = Ws.ravel()
    am = int(np.argmax(flat))
    q = np.quantile(flat, np.linspace(0, 1, 2001))
    np.savez(path, wstar=float(flat[am]), am=am, q=q, maxmin=float(Sm.max()))
    return path


def pct_of(q, w):
    return float(np.interp(w, q, np.linspace(0, 1, len(q))))


def task(args):
    cond, pattern, cseed = args
    C = CONDS[cond]
    form = C.get("form", "linear")
    W = composition(pattern, cseed)
    rows = []
    start = (np.zeros(9, int), np.zeros(6, int), np.zeros(6, int)) if C.get("start") == "blank" else None
    eF, kF, oF = s0.f_design(form)
    budget = None
    if C.get("budget") == "20":
        budget = np.full(9, 20.0)
    elif C.get("budget") == "5":
        budget = np.full(9, 5.0)
    elif C.get("budget") == "poor":
        rng = np.random.default_rng([cseed, 5])
        budget = np.where(rng.permutation(9) < 4, 0.0, 1e9)   # 4 戸が払えない
    nu = C.get("nu", 0.0)
    alpha = float(pattern[2:]) if pattern.startswith("α=") else None
    newW = (lambda rng, i: rng.dirichlet(s0.W_BAR * alpha)) if (nu > 0 and alpha) else None
    cz = None
    if nu == 0:
        cz = np.load(f"{CACHE}/{pattern}_{cseed}_{form}.npz")
    for reg in C["regimes"]:
        for sd in range(NS):
            r = s0.run(W, reg, seed=1000 * cseed + sd, form=form, start=None if reg == "F" else start, theta=C.get("theta", 0.0),
                       budget=budget, conv_scale=C.get("conv", 1.0), nu=nu, newW=newW)
            Wf = r["W_final"]
            Wfin = float(r["S"].sum())
            SF = s0.S_all(Wf, eF, kF, oF, form)
            WF = float(SF.sum())
            if nu > 0:
                wstar = s0.wstar_linear(Wf)[0]; pct = np.nan
            else:
                wstar = float(cz["wstar"]); pct = pct_of(cz["q"], Wfin)
            done = [x for x in r["events"] if x["ok"]]
            rows.append(dict(cond=cond, pattern=pattern, cseed=cseed, H=hetero(W), regime=reg, seed=sd,
                             W=Wfin, WF=WF, Wstar=wstar, dW=Wfin - WF,
                             eta=(Wfin - WF) / (wstar - WF) if wstar - WF > 1e-6 else np.nan,
                             gap=wstar - Wfin, pct=pct,
                             worse=float(np.mean(r["U"] < SF - 1e-9)), minS=float(r["S"].min()), minS_F=float(SF.min()),
                             bal=float(np.abs(r["b"]).sum() / 9), m_min=float(r["m"].min()),
                             n_moves=len(done), n_fail=len(r["events"]) - len(done),
                             n_exit=r["n_exit"], exit_bal=r["exit_bal"],
                             cfg="".join(map(str, r["e"])) + "|" + "".join(str(x + 1) for x in r["k"]) + "|" + "".join(map(str, r["o"]))))
    return rows


def cache():
    tasks = [(p, c, f) for f in ("linear", "concave", "floor") for p in PATS for c in range(NC)]
    t0 = time.time()
    with Pool(2) as pool:
        for n, _ in enumerate(pool.imap_unordered(cache_one, tasks)):
            if n % 30 == 0:
                print("cache", n, len(tasks), f"{time.time()-t0:.0f}s", flush=True)
    print("cache done", flush=True)


def run():
    tasks = []
    for cond, C in CONDS.items():
        for p in C.get("pats", PATS):
            for c in range(NC):
                tasks.append((cond, p, c))
    t0 = time.time(); allrows = []
    with Pool(2) as pool:
        for n, rows in enumerate(pool.imap_unordered(task, tasks)):
            allrows += rows
            if n % 50 == 0:
                print("run", n, len(tasks), f"{time.time()-t0:.0f}s", flush=True)
    with open(f"{OUT}/runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    print("run done", time.time() - t0, flush=True)


if __name__ == "__main__":
    {"cache": cache, "run": run}[sys.argv[1]]()
