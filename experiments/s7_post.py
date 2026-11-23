"""判定の後の確かめ（凍結したファイルは変えない。判定には使わない）。
K2（全知の計画者の η＝1）は、凍結した判定のスクリプトが平均に nan を含めたため、二群・年 10 % のセルで False を返した。
30 年目の住み手にとって全知の最適が始まりの形と同じ建物（ΔΣS*＝0）では、η は定義されない（0 で割る）。
その建物を除いて η を確かめ直す → out/s7/post_K2.json
PYTHONPATH=. python3 -m experiments.s7_post"""
import json
import numpy as np

rows = []; undefined = []
by = {}
for line in open("out/s7/main.jsonl"):
    r = json.loads(line)
    if r["arm"] == "計画者":
        by.setdefault((r["pat"], r["nu"]), []).append(r)
for (p, nu), R in sorted(by.items()):
    e = np.array([r["eta"] for r in R], float)
    nan = [r["cs"] for r in R if r["eta"] != r["eta"]]
    undefined += [dict(pat=p, nu=nu, cs=c, dSstar=[r["dSstar"] for r in R if r["cs"] == c][0]) for c in nan]
    rows.append(dict(pat=p, nu=nu, n=len(R), n_undefined=len(nan), eta_nanmean=float(np.nanmean(e)),
                     all_one=bool(np.all(np.abs(e[~np.isnan(e)] - 1.0) < 1e-9))))
out = dict(K2_rows=rows, undefined=undefined, K2_excluding_undefined=all(x["all_one"] for x in rows))
json.dump(out, open("out/s7/post_K2.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
