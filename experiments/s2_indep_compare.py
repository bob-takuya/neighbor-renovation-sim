"""別の実装の推定器の値（out/s2/indep_values.csv）と、trade/vstar.py の値（out/s2/mine_for_indep.json）を照合する。

条件ごと・パターンごとに V_H・V*・V*n の差を求め、out/s2/indep_compare.json に書き出す。
"""
import csv
import json

LABEL = {"改良版top3ν0.1": "改良版top3"}      # mine_for_indep.json と indep_values.csv で条件名が違うもの

if __name__ == "__main__":
    mine = {(LABEL.get(r["cond"], r["cond"]), r["pat"]): r for r in json.load(open("out/s2/mine_for_indep.json"))}
    rows = []
    with open("out/s2/indep_values.csv") as f:
        for r in csv.DictReader(f):
            m = mine[(r["cond"], r["pattern"])]
            a = [float(r["VH"]), float(r["Vs"]), float(r["Vn"])]
            b = [m["V_H"], m["V*"], m["V*n"]]
            rows.append(dict(cond=r["cond"], pat=r["pattern"], indep=a, mine=b, maxdiff=max(abs(x - y) for x, y in zip(a, b))))
    out = dict(rows=rows, max_abs_diff=max(r["maxdiff"] for r in rows))
    json.dump(out, open("out/s2/indep_compare.json", "w"), ensure_ascii=False)
    print(f"{len(rows)} 行、差の最大 {out['max_abs_diff']:.2e}")
