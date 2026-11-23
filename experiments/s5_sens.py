"""判定の後の補い（凍結したスクリプトは変えない）：事前登録 v7a の感度の分析のうち、判定のスクリプトが計算していなかった
「組の手で一つの額に二つの係数が混ざるとき、どちらの係数も分からないとする読み方」の V*n を、確かめ用の記録から求める。
判定には使わない。→ out/s5a/sens_swap.json
PYTHONPATH=. python3 -m experiments.s5_sens"""
import json
from multiprocessing import Pool
from trade import vstar3
from experiments import s5_prof1 as P


def job(a):
    kind, A, B, gA, gB = a
    if kind == "val":
        v = vstar3.population3(A, "Xn", None, (), gA)["V"]
        lo, hi = vstar3.boot3(A, "Xn", n=1000, seed=0, G_key=gA)
        return dict(v=v, lo=lo, hi=hi)
    return vstar3.diff3(A, B, "Xn", n=1000, seed=0, GA_key=gA, GB_key=gB)


if __name__ == "__main__":
    C = P.load("out/s5a/main.jsonl")
    jobs = []
    for b in P.BASES:
        for p in P.PATS:
            for nu in (0.0, 0.05, 0.1):
                alt, ban = C[(b, "代わりの手", p, nu)], C[(b, "一律に禁止", p, nu)]
                jobs.append((f"{b}|代わりの手|{p}|{nu}|組の手を読まない", ("val", alt, None, "swap_none", None)))
                jobs.append((f"{b}|代わりの手−一律に禁止|{p}|{nu}|組の手を読まない", ("diff", alt, ban, "swap_none", None)))
    with Pool(2) as pool:
        res = pool.map(job, [j[1] for j in jobs], chunksize=1)
    out = {k: r for (k, _), r in zip(jobs, res)}
    json.dump(out, open("out/s5a/sens_swap.json", "w"), ensure_ascii=False, indent=1)
    for k, r in out.items():
        print(k, {a: round(x, 3) for a, x in r.items()})
