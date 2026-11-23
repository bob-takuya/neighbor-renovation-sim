"""図 229：段 2 の候補を出す段の総覧。(a) 条件 × 候補の値（住民主体の最大で割った値）、(b) 試験 A〜C の合否、(c) V*n の決め方ごとの値
PYTHONPATH=. python3 -m experiments.fig229"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from experiments.style import SEQ_B, INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, GREY, DEEP, hair, ink_on

T = json.load(open("out/s2/tests.json"))
C = T["cells"]
CAND = ["V_H", "V*", "V*n", "Lc", "Ln", "Vp", "gohai", "ent", "rebind", "quotes"]
CL = {"V_H": "$V_H$", "V*": "$V^{*}$", "V*n": "$V^{*}_{n}$", "Lc": "$\\Lambda_c$", "Ln": "$\\Lambda_n$", "Vp": "$V_p$",
      "gohai": "誤配", "ent": "記録の\n乱雑さ", "rebind": "結び\n直し", "quotes": "問い\n合わせ"}
NULLS = [("F", "N1 固定 F"), ("N2", "N2 全知の最適から"), ("N3", "N3 全知が組み替え続ける"), ("N4", "N4 ランダム"), ("N4c", "N4c ランダム＋形だけの補償"),
         ("BR", "偽装：全知の仲介者"), ("BRdyn", "偽装：入居のたびに仲介"), ("AI", "偽装：起案 AI"), ("N7", "N7 外部性のない盤")]
WIN = [("N6", "N6 悪い盤から（割れ窓）")]
RES = [("U", "U 単独"), ("V50", "V50 投票・過半"), ("V75", "V75 投票・3/4"), ("TR+", "TR+ 査定額で補償"), ("TP+", "TP+ 全員に聞く"),
       ("TP+top1", "TP+ 上位 1 手だけ聞く"), ("TP+top3", "TP+ 上位 3 手だけ聞く"), ("TP+near", "TP+ 近い戸だけに聞く"),
       ("改良版", "改良版 全員に聞く"), ("改良版top3", "改良版 上位 3 手"), ("改良版near", "改良版 近い戸だけ")]
CELLS = [(p, nu, c) for p in ["α=3", "二群"] for nu in ["0.0", "0.1"] for c in ["なし", "C1"]]


def v(reg, cand, cell):
    k = "|".join([reg, cell[0], cell[1], cell[2]])
    return C[k][cand][0] if k in C else np.nan


fig = plt.figure(figsize=(7.2, 16.0))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.22, 0.42], hspace=0.30)

# (a) 条件 × 候補：8 セルのうち最大の値を、住民主体の決め方の最大で割る（1 を超えるものは 1）
ax = fig.add_subplot(gs[0])
rows = [(None, "― 帰無と偽装（0 になるべき）")] + NULLS + [(None, "― 割れ窓（TP+ 上位 3 手を超えないべき）")] + WIN + [(None, "― 住民主体の決め方")] + RES
M = np.full((len(rows), len(CAND)), np.nan)
for j, c in enumerate(CAND):
    mx = max(abs(v(r, c, cl)) for r, _ in RES for cl in CELLS if not np.isnan(v(r, c, cl)))
    for i, (r, _) in enumerate(rows):
        if r is None:
            continue
        vals = [v(r, c, cl) for cl in CELLS if not (r == "N2" and cl[1] != "0.0")]
        vals = [x for x in vals if not np.isnan(x)]
        if vals:
            M[i, j] = min(1.0, max(vals) / mx) if mx > 0 else 0
for i, (r, lab) in enumerate(rows):
    for j in range(len(CAND)):
        if r is None:
            continue
        x = M[i, j]
        col = SEQ_B(x) if not np.isnan(x) else (1, 1, 1, 1)
        ax.add_patch(Rectangle((j, i), 1, 1, fc=col, ec="white", lw=1.2))
        if not np.isnan(x):
            ax.text(j + 0.5, i + 0.5, f"{x:.2f}" if x >= 0.005 else "0", ha="center", va="center", fontsize=9.5, color=ink_on(col))
ax.set_xlim(0, len(CAND)); ax.set_ylim(len(rows), 0)
ax.set_xticks(np.arange(len(CAND)) + 0.5); ax.set_xticklabels([CL[c] for c in CAND], fontsize=10)
ax.xaxis.tick_top()
ax.set_yticks(np.arange(len(rows)) + 0.5); ax.set_yticklabels(["" if r is None else lab for r, lab in rows], fontsize=10)
for i, (r, lab) in enumerate(rows):
    if r is None:
        ax.text(0.02, i + 0.62, lab, fontsize=10, color=MUTED, ha="left", va="center")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.add_patch(Rectangle((2, 1), 1, len(rows) - 1, fill=False, ec=INK, lw=2.0))
ax.set_title("(a) 条件ごとの候補の値（8 セルの最大を、\n住民主体の決め方の最大で割った値）", fontsize=11.5, pad=44, loc="left", x=-0.62)

# (b) 試験の合否
ax = fig.add_subplot(gs[1])
TESTS = [("A", "A 帰無と偽装"), ("B", "B 全知への抵抗"), ("C", "C 自明さと弁別")]
for j, c in enumerate(CAND):
    for i, (t, lab) in enumerate(TESTS):
        okv = T["tests"][t][c]["ok"]
        col = "#2f7a4f" if okv else "#f0efec"
        if t == "B" and c in ("V_H", "V*", "V*n"):
            col = "#d9a441"
        ax.add_patch(Rectangle((j, i), 1, 1, fc=col, ec="white", lw=1.5))
        txt = "○" if okv else ("△" if (t == "B" and c in ("V_H", "V*", "V*n")) else "×")
        ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=12.5, color="white" if okv or txt == "△" else INK2)
ax.set_xlim(0, len(CAND)); ax.set_ylim(len(TESTS), 0)
ax.set_xticks(np.arange(len(CAND)) + 0.5); ax.set_xticklabels([CL[c] for c in CAND], fontsize=10)
ax.set_yticks(np.arange(len(TESTS)) + 0.5); ax.set_yticklabels([l for _, l in TESTS], fontsize=10)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_title("(b) 試験の合否（△＝全知は手続きを選ぶことでしか上げられない）", fontsize=11.5, loc="left", x=-0.3)

# (c) V*n の決め方ごとの値（α＝3・二群 × 入れ替わり）
ax = fig.add_subplot(gs[2])
regs = ["U", "TP+", "TP+top1", "TP+top3", "TP+near", "改良版", "改良版top3", "改良版near", "N6"]
x = np.arange(len(regs))
w = 0.2
for k, (cl, colr, lab) in enumerate([(("α=3", "0.0", "なし"), BLUE, "α=3・入れ替わりなし"), (("α=3", "0.1", "なし"), "#9bbfea", "α=3・毎年 10 %"),
                                     (("二群", "0.0", "なし"), AQUA, "二群・入れ替わりなし"), (("二群", "0.1", "なし"), "#a2d7cb", "二群・毎年 10 %")]):
    y = [C["|".join([r, *cl])]["V*n"] for r in regs]
    ax.bar(x + (k - 1.5) * w, [a[0] for a in y], w, color=colr, label=lab,
           yerr=[[a[0] - a[1] for a in y], [a[2] - a[0] for a in y]], error_kw=dict(lw=0.8, ecolor=INK2))
ax.set_xticks(x); ax.set_xticklabels(["U", "TP+\n全員", "TP+\n上位1", "TP+\n上位3", "TP+\n近い戸", "改良\n全員", "改良\n上位3", "改良\n近い戸", "N6\n悪い盤"], fontsize=10)
ax.set_ylabel("$V^{*}_{n}$", fontsize=11); ax.tick_params(axis="y", labelsize=10); hair(ax); ax.set_ylim(0, 0.85); ax.legend(fontsize=10, frameon=False, ncol=2, loc="upper left")
ax.set_title("(c) 決め方ごとの $V^{*}_{n}$（費用なし、95 % 区間）", fontsize=11.5, loc="left")
fig.savefig("out/figs/fig229_S2_候補の試験.png", dpi=160, bbox_inches="tight")
print("ok")
