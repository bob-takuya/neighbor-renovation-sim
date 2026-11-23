"""選ぶ用での確かめの補い（事前登録の前）：基準と検証者の腕を、制約のもとの物差し（Zc・eta_cap）と、組の手の言い値の読み方の感度つきで
走らせ直す（記録の作り方を足しただけで、局は dev と同じ）。→ out/s5a/dev2.jsonl・dev2_values.json
PYTHONPATH=. python3 -m experiments.s5_dev2"""
import json
import numpy as np
from multiprocessing import Pool
from trade import vstar3
from experiments import s5_prof1 as P

if __name__ == "__main__":
    T = [(p, cs, nu, b, arm) for p in P.PATS for nu in P.NUS for b in P.BASES for arm in ("基準", "一律に禁止", "代わりの手") for cs in P.DEV_CS]
    with Pool(2) as pool:
        R = pool.map(P.one, T, chunksize=8)
    with open("out/s5a/dev2.jsonl", "w") as f:
        for r in R:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cells = {}
    for r in R:
        cells.setdefault((r["base"], r["arm"], r["pat"], r["nu"]), []).append(r)
    out = {}
    for k, recs in sorted(cells.items()):
        recs.sort(key=lambda r: r["cs"])
        row = dict(eta=float(np.mean([r["eta"] for r in recs])), eta_cap=float(np.mean([r["eta_cap"] for r in recs])),
                   Vn=vstar3.population3(recs)["V"], Vn_cap=vstar3.population3(recs, Z_key="Zc")["V"])
        if k[1] == "代わりの手":
            row["Vn_swap_none"] = vstar3.population3(recs, G_key="swap_none")["V"]
        out["|".join(map(str, k))] = row
        print(f"{'|'.join(map(str, k)):30s} η={row['eta']:.3f} η制約={row['eta_cap']:.3f} V*n={row['Vn']:.3f} V*n制約={row['Vn_cap']:.3f}"
              + (f" 組の手を読まない={row['Vn_swap_none']:.3f}" if "Vn_swap_none" in row else ""))
    json.dump(out, open("out/s5a/dev2_values.json", "w"), ensure_ascii=False, indent=1)
