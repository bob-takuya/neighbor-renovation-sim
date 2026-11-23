"""選ぶ用（cseed 0〜39）での腕と基準の組ごとの差（事前登録の前に、予言の向きと閾値を決めるために見る）→ out/s5a/dev_diffs.json
PYTHONPATH=. python3 -m experiments.s5_devdiff"""
import json
import numpy as np
from multiprocessing import Pool
from trade import vstar, vstar3
from experiments import s5_prof1 as P


def job(a):
    kind, A, Bc, cA, gA, zk = a
    if kind == "eps":
        return vstar.perm_floor(A, "Xn", n=100, seed=1)
    return vstar3.diff3(A, Bc, "Xn", n=200, seed=0, countA=cA, GA_key=gA, Z_key=zk)


if __name__ == "__main__":
    cells = P.load("out/s5a/dev.jsonl")
    c2 = P.load("out/s5a/dev2.jsonl")
    jobs = []
    for b in P.BASES:
        for (p, nu) in [(p, nu) for p in P.PATS for nu in P.NUS]:
            jobs.append(((b, "ε", p, nu, ""), ("eps", cells[(b, "基準", p, nu)], None, (), None, "Z")))
            for arm in P.ARMS[1:]:
                jobs.append(((b, arm, p, nu, "外"), ("d", cells[(b, arm, p, nu)], cells[(b, "基準", p, nu)], (), None, "Z")))
                if arm == "額まで記録":
                    jobs.append(((b, arm, p, nu, "数"), ("d", cells[(b, arm, p, nu)], cells[(b, "基準", p, nu)], ("recorder",), None, "Z")))
            jobs.append(((b, "代わりの手−一律に禁止", p, nu, "外"), ("d", c2[(b, "代わりの手", p, nu)], c2[(b, "一律に禁止", p, nu)], (), None, "Z")))
            jobs.append(((b, "代わりの手−一律に禁止", p, nu, "外・制約"), ("d", c2[(b, "代わりの手", p, nu)], c2[(b, "一律に禁止", p, nu)], (), None, "Zc")))
            for arm in ("一律に禁止", "代わりの手"):
                jobs.append(((b, arm, p, nu, "外・制約"), ("d", c2[(b, arm, p, nu)], c2[(b, "基準", p, nu)], (), None, "Zc")))
    with Pool(2) as pool:
        res = pool.map(job, [j[1] for j in jobs], chunksize=2)
    out = {}
    for (k, _), r in zip(jobs, res):
        out["|".join(map(str, k))] = r
    # η の組ごとの差
    for b in P.BASES:
        for p in P.PATS:
            for nu in P.NUS:
                for arm in P.ARMS[1:]:
                    out[f"{b}|{arm}|{p}|{nu}|η"] = P.eta_diff(cells[(b, arm, p, nu)], cells[(b, "基準", p, nu)], 2000, 1)
                out[f"{b}|代わりの手−一律に禁止|{p}|{nu}|η"] = P.eta_diff(c2[(b, "代わりの手", p, nu)], c2[(b, "一律に禁止", p, nu)], 2000, 1)
                for arm in ("一律に禁止", "代わりの手"):
                    out[f"{b}|{arm}|{p}|{nu}|η制約"] = P.eta_diff(c2[(b, arm, p, nu)], c2[(b, "基準", p, nu)], 2000, 1, "eta_cap")
                out[f"{b}|代わりの手−一律に禁止|{p}|{nu}|η制約"] = P.eta_diff(c2[(b, "代わりの手", p, nu)], c2[(b, "一律に禁止", p, nu)], 2000, 1, "eta_cap")
    json.dump(out, open("out/s5a/dev_diffs.json", "w"), ensure_ascii=False, indent=1)
    print("ε（選ぶ用）", max(v for k, v in out.items() if "|ε|" in k))
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"{k:44s} {v['d']:+.3f}［{v['lo']:+.3f}, {v['hi']:+.3f}］")
