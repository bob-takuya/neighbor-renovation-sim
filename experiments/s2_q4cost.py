"""段 2 の準備：問 4 を費用のある模型で確かめ直す。
前半 15 年は住み手 A、14 年目の終わりに全戸が B に入れ替わる。A を二通り（A1・A2）にして、B にとっての結果を比べる。
費用があると、全知の組み替え（P）も今の盤から動くのに費用がかかるので、全知も前の住み手を「覚える」。
問い：結び直しは、全知と違う「良さ」の帰結を、今の住み手 B にもたらすか。
比べる量（B にとって、後半 15 年）：
  net_B ＝ ΣS_B(X_T) − μ・（後半に払われた費用）     … 全知の組み替えはこれを最大にするよう動く
  dHist ＝ 前の住み手 A1・A2 で最後の形が違う部位の数
  gap   ＝ net_B(全知) − net_B(決め方)              … 負なら全知より良い
PYTHONPATH=. python3 -m experiments.s2_q4cost N
"""
import sys, json
import numpy as np
from multiprocessing import Pool
from trade import s0, s1
from experiments.explore import composition

COSTS = {"なし": (None, None),
         "C1": (dict(e=1.0, k=0.25, o=0.25), dict(e=0.5, k=0.1, o=0.1)),
         "C3": (dict(e=3.0, k=0.75, o=0.75), dict(e=1.5, k=0.3, o=0.3))}
REGS = {"全知": ("P", {}), "TP+": ("T1+", {}), "TP+清再": ("T1+", dict(settle=True, reconsent="all")),
        "改良版": ("T1+", dict(credit=20.0, settle=True, reconsent="all")),
        "改良版top3": ("T1+", dict(credit=20.0, settle=True, reconsent="all", protocol="topk", ask=3))}
F0 = s0.f_design()


def cfg(r):
    return np.concatenate([r["e"], r["k"], r["o"]]).astype(int)


def half_cost(r):
    """後半（15 年目以降）に払われた費用"""
    c = sum(ev.get("cost", 0.0) for ev in r["events"] if ev.get("ok") and ev["t"] >= 15)
    return c


def go(W_first, W_B, reg, seed, cname):
    rr, kw = REGS[reg]
    cost, rcost = COSTS[cname]
    r = s1.run(W_first, rr, seed, newW=lambda rng, i: W_B[i].copy(), turn_at={14: set(range(9))},
               cost=cost, rcost=rcost, ref=F0, **kw)
    if rr == "P":   # 全知の組み替えの費用は、入れ替わりの年の分を spent から取り出す
        r["half"] = r["spent"] - _first_half_planner(W_first, cost, rcost)
    else:
        r["half"] = half_cost(r)
        # 再同意で戻した費用（戻したい新しい世帯が払う）
        if rcost is not None:
            r["half"] += sum(rcost[p[0]] for (t, n, p, how) in r["recon"] if how == "revert")
    return r


def _first_half_planner(W, cost, rcost):
    e, k, o, cst = s1.best_net(W, *[x.copy() for x in F0], F0, cost, rcost)
    return cst


def one(args):
    pat, cs, cname = args
    W_B = composition(pat, cs + 5000); W_A1 = composition(pat, cs); W_A2 = composition(pat, cs + 10000)
    out = []
    WF = s0.S_all(W_B, *F0).sum(); Ws = s0.wstar_linear(W_B)[0]
    for reg in REGS:
        r1 = go(W_A1, W_B, reg, cs, cname); r2 = go(W_A2, W_B, reg, cs, cname)
        X1, X2 = cfg(r1), cfg(r2)
        for nm, r in (("A1", r1), ("A2", r2)):
            S = s0.S_all(W_B, r["e"], r["k"], r["o"]).sum()
            out.append(dict(pat=pat, cs=cs, cost=cname, reg=reg, hist=nm, S=float(S), half=float(r["half"]),
                            net=float(S - s0.MU * r["half"]), eta=float((S - WF) / (Ws - WF)) if Ws - WF > 1e-6 else None,
                            dHist=int((X1 != X2).sum()), U_B=float(np.mean(r["U"]))))
    return out


if __name__ == "__main__":
    N = int(sys.argv[1])
    tasks = [(p, cs, c) for p in ["α=3", "二群"] for cs in range(2000, 2000 + N) for c in COSTS]
    with Pool(2) as pool:
        res = [x for xs in pool.map(one, tasks, chunksize=4) for x in xs]
    json.dump(res, open("out/s2/q4cost.json", "w"), ensure_ascii=False)
    print("done", len(res))
