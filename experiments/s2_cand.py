"""段 2・候補を出す段：評価の候補を、帰無・偽装・住民主体の決め方で一度に記録する。
構成は開発用 cseed 0〜39 と、本走行で使っていない 100〜159 の計 100 構成（各パターン）。種 200〜249 は名付けの確かめ、300〜399 は確証に取っておく。
PYTHONPATH=. python3 -m experiments.s2_cand            → out/s2/cand.jsonl
"""
import sys, json, os
import numpy as np
from multiprocessing import Pool
from trade import s0, s1, vstar
from experiments.explore import composition

CS = list(range(0, 40)) + list(range(100, 160))
PATS = ["α=3", "二群"]
NUS = [0.0, 0.1]
COSTS = {"なし": (None, None), "C1": (dict(e=1.0, k=0.25, o=0.25), dict(e=0.5, k=0.1, o=0.1))}
IMP = dict(credit=20.0, settle=True, reconsent="all")
REGS = {
    # 住民主体の決め方
    "U": ("U", {}, None), "V50": ("V½", {}, None), "V75": ("V¾", {}, None), "TR+": ("T0+", {}, None),
    "TP+": ("T1+", {}, None), "TP+top1": ("T1+", dict(protocol="topk", ask=1), None),
    "TP+top3": ("T1+", dict(protocol="topk", ask=3), None), "TP+near": ("T1+", dict(protocol="near"), None),
    "改良版": ("T1+", dict(IMP), None), "改良版top3": ("T1+", dict(IMP, protocol="topk", ask=3), None),
    "改良版near": ("T1+", dict(IMP, protocol="near"), None),
    # 帰無と偽装
    "F": ("F", {}, None), "N2": ("T1+", dict(protocol="topk", ask=3, _start="wstar"), "planner"),
    "N3": ("P", {}, None), "N4": ("R", {}, None), "N4c": ("Rc", {}, None), "BR": ("BR", {}, None),
    "BRdyn": ("BRdyn", {}, None), "AI": ("AI+", {}, None),
    "N6": ("T1+", dict(protocol="topk", ask=3, _start="worst"), None),
}
RESIDENT = ["U", "V50", "V75", "TR+", "TP+", "TP+top1", "TP+top3", "TP+near", "改良版", "改良版top3", "改良版near"]
F0 = s0.f_design()
WORST = s1.best_net(-np.tile(s0.W_BAR, (9, 1)), *[x.copy() for x in F0], F0)[:3]   # 平均の住み手にとって最悪の盤（性向を知らずに建てる）


def one(a):
    pat, cs, nu, cname, reg, rep = a
    rr, kw, sa = REGS[reg]
    kw = dict(kw)
    W = composition(pat, cs)
    seed = cs + 50000 * rep
    st = kw.pop("_start", None)
    if st == "wstar":
        _, e, k, o = s0.wstar_linear(W); kw.update(start=(e, k, o), ref=F0)
    elif st == "worst":
        kw.update(start=tuple(x.copy() for x in WORST))
    cost, rcost = COSTS[cname]
    newW = (lambda rng, i: composition(pat, int(rng.integers(10**6, 10**7)))[i]) if nu > 0 else None
    r = s1.run(W, rr, seed, nu=nu, newW=newW, cost=cost, rcost=rcost, **kw)
    rec = vstar.record(r, W, sa, F0)
    rec.update(pat=pat, cs=cs, nu=nu, cost=cname, reg=reg, rep=rep)
    return rec


def tasks():
    T = [(p, cs, nu, c, rg, 0) for p in PATS for nu in NUS for c in COSTS for rg in REGS for cs in CS]
    T += [(p, cs, 0.0, "なし", rg, 1) for p in PATS for rg in RESIDENT for cs in CS]      # 同じ構成の 2 局目（ICC 用）
    return T


if __name__ == "__main__":
    os.makedirs("out/s2", exist_ok=True)
    T = tasks()
    print(len(T), "局", flush=True)
    with Pool(2) as pool, open("out/s2/cand.jsonl", "w") as f:
        for i, rec in enumerate(pool.imap_unordered(one, T, chunksize=10)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 2000 == 0:
                print(i, flush=True)
    print("done")
