"""独立の再実装（s2_indep_vstar.py）に入れる局の記録を作る。

s2_mine_for_indep.py と同じ 10 条件 × 2 パターン × cseed 1000〜1099 の局を模型 s1 で実行し、
各局の記録（s1.run の返り値）を out/s2/runs.pkl に書き出す。形は {(条件, パターン): [記録, ...]}。
"""
import os
import pickle
from multiprocessing import Pool

from trade import s0, s1
from experiments.explore import composition
from experiments.s2_mine_for_indep import SP, IMP

PATS = ["α=3", "二群"]
LABEL = {"改良版top3ν0.1": "改良版top3"}      # 独立の再実装の出力での条件名


def one(a):
    pat, cs, nm = a
    rr, kw, _ = SP[nm]
    W = composition(pat, cs)
    if kw == "N2":
        _, e, k, o = s0.wstar_linear(W)
        r = s1.run(W, rr, cs, start=(e, k, o), ref=s0.f_design(), protocol="topk", ask=3)
    elif kw == "IMP":
        r = s1.run(W, rr, cs, nu=0.1, newW=lambda rng, i: composition(pat, int(rng.integers(10**6, 10**7)))[i], **IMP)
    else:
        r = s1.run(W, rr, cs, **kw)
    return pat, nm, r


if __name__ == "__main__":
    jobs = [(pat, cs, nm) for pat in PATS for nm in SP for cs in range(1000, 1100)]
    with Pool(2) as p:
        res = p.map(one, jobs, chunksize=10)
    runs = {(LABEL.get(nm, nm), pat): [] for nm in SP for pat in PATS}
    for pat, nm, r in res:
        runs[(LABEL.get(nm, nm), pat)].append(r)
    os.makedirs("out/s2", exist_ok=True)
    with open("out/s2/runs.pkl", "wb") as f:
        pickle.dump(runs, f)
    print(f"{len(res)} 局 → out/s2/runs.pkl")
