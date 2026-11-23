"""trade/vstar.py が、名付けの前の試算（cseed 1000〜1099、α＝3・ν＝0）の値を再現するか"""
import numpy as np
from multiprocessing import Pool
from trade import s0, s1, vstar
from experiments.explore import composition

SPECS = {"F": ("F", {}, None), "N2": ("T1+", "wstar", "planner"), "N3": ("P", {}, None), "N4": ("R", {}, None),
         "BR": ("BR", {}, None), "U": ("U", {}, None), "TP+": ("T1+", {}, None), "TP+top3": ("T1+", dict(protocol="topk", ask=3), None)}


def one(a):
    cs, nm = a
    rr, kw, sa = SPECS[nm]
    W = composition("α=3", cs)
    if kw == "wstar":
        _, e, k, o = s0.wstar_linear(W); r = s1.run(W, rr, cs, start=(e, k, o), ref=s0.f_design())
    else:
        r = s1.run(W, rr, cs, **kw)
    return nm, vstar.record(r, W, sa)


if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(one, [(cs, nm) for cs in range(1000, 1100) for nm in SPECS], chunksize=8)
    for nm in SPECS:
        recs = [r for n, r in res if n == nm]
        print(f"{nm:8s} η={np.nanmean([r['eta'] for r in recs]):5.2f}  V_H={vstar.population(recs,'X')['V']:.2f}  "
              f"V*={vstar.population(recs,'Xc')['V']:.2f}  V*n={vstar.population(recs,'Xn')['V']:.2f}  Λc={np.mean([r['Lc'] for r in recs]):.2f}  Λn={np.mean([r['Ln'] for r in recs]):.2f}")
