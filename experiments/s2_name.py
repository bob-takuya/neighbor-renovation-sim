"""段 2・名付けの確かめ：凍結した事前登録 v4（experiments/prereg_s2.json）を、新しい種 cseed 200〜249 で確かめる。
PYTHONPATH=. python3 -m experiments.s2_name run     → out/s2/name.jsonl
PYTHONPATH=. python3 -m experiments.s2_name judge   → out/s2/name_judge.json
"""
import sys, json, hashlib
import numpy as np
from multiprocessing import Pool
from trade import vstar
from experiments import s2_cand

PRE = "experiments/prereg_s2.json"


def run():
    pre = json.load(open(PRE))
    CS = list(range(pre["seeds"][0], pre["seeds"][1] + 1))
    T = [(p, cs, nu, c, rg, 0) for p in pre["patterns"] for nu in pre["nus"] for c in pre["costs"] for rg in pre["regimes"] for cs in CS]
    print(len(T), "局", flush=True)
    with Pool(2) as pool, open("out/s2/name.jsonl", "w") as f:
        for i, rec in enumerate(pool.imap_unordered(s2_cand.one, T, chunksize=10)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 1000 == 0:
                print(i, flush=True)
    print("done")


def judge():
    from experiments.s2_tests import n7_records
    pre = json.load(open(PRE))
    key = pre["quantity"]["key"]
    cells = {}
    for line in open("out/s2/name.jsonl"):
        r = json.loads(line)
        cells.setdefault((r["reg"], r["pat"], r["nu"], r["cost"]), []).append(r)
    CS = list(range(pre["seeds"][0], pre["seeds"][1] + 1))
    for pat in pre["patterns"]:
        for nu in pre["nus"]:
            for c in pre["costs"]:
                cells[("N7", pat, nu, c)] = n7_records(pat, CS)
    for k in cells:
        cells[k].sort(key=lambda r: r.get("cs", 0))
    # ε（事前登録の規則どおり、このデータで求める）
    eps = max(vstar.perm_floor(cells[(rg, pat, nu, c)], key, n=100)
              for rg in ["TP+", "TP+top3", "U"] for pat in pre["patterns"] for nu in pre["nus"] for c in pre["costs"])
    val = {}
    for k, recs in cells.items():
        v = vstar.population(recs, key)["V"]; lo, hi = vstar.boot(recs, key, n=1000)
        val[k] = (v, lo, hi, float(np.nanmean([r["eta"] for r in recs])))
    out = dict(epsilon=eps)
    nullset = [x for x in pre["nulls"] if "@" not in x]
    f1 = [f"{k}: {v[0]:.3f}［{v[1]:.3f}, {v[2]:.3f}］" for k, v in val.items()
          if (k[0] in nullset or (k[0] == "N2" and k[2] == 0.0)) and v[2] > eps]
    out["P1"] = dict(ok=not f1, fails=f1)
    res = []
    for pairs, nus in ((pre["order_pairs"], pre["order_nus"]), (pre["order_pairs_turnover"], [x for x in pre["nus"] if x > 0])):
        for a, b in pairs:
            for pat in pre["patterns"]:
                for nu in nus:
                    for c in pre["costs"]:
                        A, Bv = val[(a, pat, nu, c)], val[(b, pat, nu, c)]
                        res.append(dict(pair=f"{a} ＞ {b}", cell=f"{pat}・ν{nu}・費用{c}", a=A[0], b=Bv[0], ok=bool(A[0] > Bv[0])))
    share = float(np.mean([r["ok"] for r in res]))
    out["P2"] = dict(ok=share >= pre["order_share"], share=share, rows=res)
    f3 = [f"{k}: {v[0]:.3f}［{v[1]:.3f}, {v[2]:.3f}］" for k, v in val.items() if k[0] in pre["no_consent"] and v[2] > eps]
    out["P3"] = dict(ok=not f3, fails=f3)
    f4 = []
    for pat in pre["patterns"]:
        for c in pre["costs"]:
            A, Bv = val[("TP+top3", pat, 0.0, c)], val[("TP+", pat, 0.0, c)]
            f4.append(dict(cell=f"{pat}・費用{c}", top3=A[:3], full=Bv[:3], eta=(A[3], Bv[3]), ok=bool(A[1] > Bv[2])))
    out["P4"] = dict(ok=all(x["ok"] for x in f4), rows=f4)
    f5 = []
    for pat in pre["patterns"]:
        for nu in pre["nus"]:
            for c in pre["costs"]:
                A, Bv = val[("N6", pat, nu, c)], val[("TP+top3", pat, nu, c)]
                f5.append(dict(cell=f"{pat}・ν{nu}・費用{c}", n6=A[:3], top3=Bv[:3], ok=bool(A[1] <= Bv[2] + eps)))
    out["P5"] = dict(ok=all(x["ok"] for x in f5), rows=f5)
    out["values"] = {"|".join(map(str, k)): v for k, v in val.items()}
    out["prereg_sha256"] = hashlib.sha256(open(PRE, "rb").read()).hexdigest()
    json.dump(out, open("out/s2/name_judge.json", "w"), ensure_ascii=False, indent=1)
    print("ε =", round(eps, 4))
    for p in ["P1", "P2", "P3", "P4", "P5"]:
        print(p, "支持" if out[p]["ok"] else "不支持", {k: v for k, v in out[p].items() if k in ("fails", "share")})


if __name__ == "__main__":
    {"run": run, "judge": judge}[sys.argv[1]]()
