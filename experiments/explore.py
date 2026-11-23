"""探索 E0：住み手の構成 × 決め方で、全配置の分布の中のどこにたどり着き、どんな手が打たれるか。
python3 -m experiments.explore main     # 7 パターン × 40 構成 × 7 決め方 × 5 局
python3 -m experiments.explore show     # 代表の 3 構成 × 7 決め方 × 100 局（分布の図用）
"""
import sys, json, time, os
from collections import Counter
from multiprocessing import Pool
import numpy as np
from trade import s0, enumerate as EN

OUT = "out/e0"
os.makedirs(OUT, exist_ok=True)
PATTERNS = ["同質", "α=30", "α=3", "α=0.5", "二群", "少数の極端", "階の不一致"]


def composition(pattern, seed):
    rng = np.random.default_rng([seed, 11])
    wb = s0.W_BAR
    if pattern == "同質":
        return np.tile(wb, (9, 1))
    if pattern.startswith("α="):
        a = float(pattern[2:])
        W = rng.dirichlet(wb * a, 9)
        return np.clip(W, 1e-9, None) / np.clip(W, 1e-9, None).sum(1, keepdims=True)
    if pattern in ("二群", "二群7:3", "二群9:1"):
        A = np.array([0.6, 0.2, 0.1, 0.1]); B = np.array([0.15, 0.25, 0.5, 0.1])
        nA = {"二群": 5, "二群7:3": 6, "二群9:1": 8}[pattern]
        who = rng.permutation(9)
        W = np.zeros((9, 4)); W[who[:nA]] = A; W[who[nA:]] = B
        return W
    if pattern == "日照派と近さ派":
        A = np.array([0.15, 0.65, 0.1, 0.1]); B = np.array([0.2, 0.1, 0.1, 0.6])
        who = rng.permutation(9)
        W = np.zeros((9, 4)); W[who[:5]] = A; W[who[5:]] = B
        return W
    if pattern in ("少数の極端", "少数の極端2"):
        W = np.tile(wb, (9, 1))
        if pattern == "少数の極端":          # E0・E1 と同じ乱数の使い方（再現のため変えない）
            W[rng.integers(9)] = [0.05, 0.05, 0.1, 0.8]
        else:
            W[rng.permutation(9)[:2]] = [0.05, 0.05, 0.1, 0.8]
        return W
    if pattern == "階の不一致":
        W = np.tile(wb, (9, 1)); W[6:9] = [0.15, 0.65, 0.1, 0.1]
        return W
    if pattern == "階の一致":
        W = np.tile(wb, (9, 1)); W[0:3] = [0.15, 0.65, 0.1, 0.1]
        return W
    raise ValueError(pattern)


def hetero(W):
    return float(np.mean(np.abs(W - s0.W_BAR).sum(1)) / 2)


def config_vec(e, k, o):
    return np.concatenate([e, k, o])


def one(args):
    pattern, cseed, nrun = args[:3]
    regimes = args[3] if len(args) > 3 else s0.REGIMES
    W = composition(pattern, cseed)
    Ws, Sm = EN.enumerate_all(W)
    flat = Ws.ravel()
    N = flat.size
    am = int(np.argmax(flat)); wstar = float(flat[am])
    eS, kS, oS = EN.decode(am)
    vstar = config_vec(eS, kS, oS)
    e0, k0, o0 = s0.f_design()
    SF = s0.S_all(W, e0, k0, o0); WF = float(SF.sum())
    sorted_ = np.sort(flat)
    near1 = int(np.count_nonzero(flat >= wstar - 0.01 * abs(wstar)))
    maxmin = float(Sm.max())
    rows = []
    for reg in regimes:
        for sd in range(nrun):
            r = s0.run(W, reg, seed=1000 * cseed + sd)
            Wfin = float(r["S"].sum())
            pct = float(np.searchsorted(sorted_, Wfin + 1e-6, side="right") / N)
            v = config_vec(r["e"], r["k"], r["o"])
            done = [x for x in r["events"] if x["ok"]]
            types = Counter(x["type"] for x in done)
            rows.append(dict(pattern=pattern, cseed=cseed, H=hetero(W), regime=reg, seed=sd,
                             WF=WF, Wstar=wstar, W=Wfin, dW=Wfin - WF,
                             eta=(Wfin - WF) / (wstar - WF) if wstar - WF > 1e-6 else np.nan,
                             pct=pct, near1=int(Wfin >= wstar - 0.01 * abs(wstar)), n_near1=near1,
                             ham=int(np.sum(v != vstar)), minS=float(r["S"].min()), minS_F=float(SF.min()), maxmin=maxmin,
                             minU=float(r["U"].min()), worse=float(np.mean(r["U"] < SF - 1e-9)),
                             bal=float(np.abs(r["b"]).sum() / 9), bal_max=float(np.abs(r["b"]).max()),
                             tok=float(np.abs(r["m"]).sum() / 2),
                             n_moves=len(done), n_fail=len(r["events"]) - len(done),
                             cfg="".join(map(str, r["e"])) + "|" + "".join(str(x + 1) for x in r["k"]) + "|" + "".join(map(str, r["o"])),
                             cfg_star="".join(map(str, eS)) + "|" + "".join(str(x + 1) for x in kS) + "|" + "".join(map(str, oS)),
                             types=json.dumps(types, ensure_ascii=False),
                             Wt=json.dumps([round(float(x), 5) for x in r["Wt"]])))
    return rows


def main():
    import csv
    tasks = [(p, c, 5) for p in PATTERNS for c in range(40)]
    t0 = time.time(); allrows = []
    with Pool(2) as pool:
        for n, rows in enumerate(pool.imap_unordered(one, tasks)):
            allrows += rows
            if n % 20 == 0:
                print(n, len(tasks), f"{time.time()-t0:.0f}s", flush=True)
    with open(f"{OUT}/runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    print("done", time.time() - t0)


def runs_for(W, reg, n=100):
    with Pool(2) as pool:
        res = pool.starmap(s0.run, [(W, reg, 10_000 + sd) for sd in range(n)])
    return [dict(W=float(r["S"].sum()), minS=float(r["S"].min()), minU=float(r["U"].min()),
                 bal=float(np.abs(r["b"]).sum() / 9), e=r["e"].tolist(), k=r["k"].tolist(), o=r["o"].tolist(),
                 types=dict(Counter(x["type"] for x in r["events"] if x["ok"])), Wt=[float(x) for x in r["Wt"]],
                 moves=[x["move"] for x in r["events"] if x["ok"]]) for r in res]


def show_plus():
    """既存の代表に T0+・T1+ を足し、階の不一致を代表に加える"""
    for pattern, cseed in [("α=30", 3), ("α=0.5", 5), ("二群", 2)]:
        path = f"{OUT}/show_{pattern}_{cseed}.json"
        D = json.load(open(path, encoding="utf-8"))
        W = np.array(D["W"])
        for reg in s0.REGIMES_BUY:
            D["runs"][reg] = runs_for(W, reg)
        D["runs"]["T1"] = runs_for(W, "T1")     # 手の名前を残すため取り直す（乱数は同じ）
        json.dump(D, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        print("show+", pattern, flush=True)
    show(picks=(("階の不一致", 0),), regimes=s0.REGIMES + s0.REGIMES_BUY)
    print("show 階の不一致", flush=True)


def main_plus():
    """他戸の手も買える T0+・T1+ を同じ構成・同じ乱数で"""
    import csv
    tasks = [(p, c, 5, s0.REGIMES_BUY) for p in PATTERNS for c in range(40)]
    t0 = time.time(); allrows = []
    with Pool(2) as pool:
        for n, rows in enumerate(pool.imap_unordered(one, tasks)):
            allrows += rows
            if n % 20 == 0:
                print(n, len(tasks), f"{time.time()-t0:.0f}s", flush=True)
    with open(f"{OUT}/runs_plus.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    print("done", time.time() - t0)


def show(picks=(("α=30", 3), ("α=0.5", 5), ("二群", 2)), regimes=None):
    """代表の構成で、全配置の分布と、各決め方の 100 局の到達点"""
    regimes = regimes or s0.REGIMES
    for pattern, cseed in picks:
        W = composition(pattern, cseed)
        Ws, Sm = EN.enumerate_all(W)
        flat = Ws.ravel(); smin = Sm.ravel()
        # 2 次元の度数分布（W, minS）
        H2, xe, ye = np.histogram2d(flat, smin, bins=(160, 120))
        h1, x1 = np.histogram(flat, bins=300)
        # 合計と最小の満足の境界：W の各区間で最小の満足の最大
        bins = np.linspace(flat.min(), flat.max(), 161)
        idx = np.clip(np.digitize(flat, bins) - 1, 0, 159)
        front = np.full(160, -np.inf); np.maximum.at(front, idx, smin)
        e0, k0, o0 = s0.f_design(); SF = s0.S_all(W, e0, k0, o0)
        am = int(np.argmax(flat)); eS, kS, oS = EN.decode(am)
        out = dict(pattern=pattern, cseed=cseed, W=W.tolist(), WF=float(SF.sum()), minSF=float(SF.min()),
                   wstar=float(flat.max()), star=[eS.tolist(), kS.tolist(), oS.tolist()],
                   h1=h1.tolist(), x1=x1.tolist(), H2=H2.tolist(), xe=xe.tolist(), ye=ye.tolist(),
                   front=front.tolist(), fbins=bins.tolist(), runs={})
        with Pool(2) as pool:
            for reg in regimes:
                res = pool.starmap(s0.run, [(W, reg, 10_000 + sd) for sd in range(100)])
                out["runs"][reg] = [dict(W=float(r["S"].sum()), minS=float(r["S"].min()), minU=float(r["U"].min()),
                                         bal=float(np.abs(r["b"]).sum() / 9), e=r["e"].tolist(), k=r["k"].tolist(), o=r["o"].tolist(),
                                         types=dict(Counter(x["type"] for x in r["events"] if x["ok"])),
                                         Wt=[float(x) for x in r["Wt"]]) for r in res]
        with open(f"{OUT}/show_{pattern}_{cseed}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        print("show", pattern, flush=True)


if __name__ == "__main__":
    {"main": main, "show": show, "plus": main_plus, "showplus": show_plus}[sys.argv[1]]()
