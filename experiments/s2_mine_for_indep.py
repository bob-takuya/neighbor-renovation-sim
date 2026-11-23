"""別の実装 experiments/s2_indep_vstar.py と突き合わせるため、同じ 10 条件（cseed 1000〜1099）で trade/vstar.py の値を出す"""
import json, numpy as np
from multiprocessing import Pool
from trade import s0, s1, vstar
from experiments.explore import composition
IMP = dict(credit=20.0, settle=True, reconsent="all", protocol="topk", ask=3)
SP = {"F": ("F", {}, None), "N2": ("T1+", "N2", "planner"), "N3": ("P", {}, None), "BR": ("BR", {}, None), "AI": ("AI+", {}, None),
      "U": ("U", {}, None), "TP+": ("T1+", {}, None), "TP+top3": ("T1+", dict(protocol="topk", ask=3), None),
      "TP+near": ("T1+", dict(protocol="near"), None), "改良版top3ν0.1": ("T1+", "IMP", None)}
def one(a):
    pat, cs, nm = a; rr, kw, sa = SP[nm]; W = composition(pat, cs)
    if kw == "N2":
        _, e, k, o = s0.wstar_linear(W); r = s1.run(W, rr, cs, start=(e, k, o), ref=s0.f_design(), protocol="topk", ask=3)
    elif kw == "IMP":
        r = s1.run(W, rr, cs, nu=0.1, newW=lambda rng, i: composition(pat, int(rng.integers(10**6, 10**7)))[i], **IMP)
    else:
        r = s1.run(W, rr, cs, **kw)
    return pat, nm, vstar.record(r, W, sa)
if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(one, [(pat, cs, nm) for pat in ["α=3", "二群"] for nm in SP for cs in range(1000, 1100)], chunksize=10)
    out = []
    for pat in ["α=3", "二群"]:
        for nm in SP:
            R = [r for (p_, n_, r) in res if p_ == pat and n_ == nm]
            row = dict(pat=pat, cond=nm)
            for lab, key in [("V_H", "X"), ("V*", "Xc"), ("V*n", "Xn")]:
                q = vstar.population(R, key); row[lab] = q["V"]; row[lab + "_I"] = q["I"]; row[lab + "_Imax"] = q["Imax"]
            out.append(row); print(pat, nm, {k: round(v, 3) for k, v in row.items() if k in ("V_H", "V*", "V*n", "V_H_I", "V_H_Imax")})
    json.dump(out, open("out/s2/mine_for_indep.json", "w"), ensure_ascii=False)
