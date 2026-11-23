"""図 230：名付けの確かめ（事前登録 v4、cseed 200〜249）。決め方 × セルの V*n と 95 % 区間、ε の線。
幅 7.2 in・文字 10 pt 以上。パターンごとに、帰無と偽装・割れ窓の段と、住民主体の段に分けて縦に積む。
PYTHONPATH=. python3 -m experiments.fig230"""
import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.style import INK2, MUTED, BLUE, AQUA, hair

J = json.load(open("out/s2/name_judge.json"))
V = J["values"]; eps = J["epsilon"]
NULL = [("F", "N1\n固定"), ("N2", "N2\n全知\nから"), ("N3", "N3\n組み\n替え"), ("N4", "N4\nラン\nダム"), ("N4c", "N4c"), ("BR", "仲介\n者"),
        ("BRdyn", "仲介\n入居\nごと"), ("AI", "起案\nAI"), ("N7", "N7\n外部\n性なし"), ("N6", "N6\n悪い\n盤")]
RES = [("U", "U"), ("V50", "V50"), ("V75", "V75"), ("TR+", "TR+"), ("TP+", "TP+\n全員"), ("TP+top1", "TP+\n上位1"),
       ("TP+top3", "TP+\n上位3"), ("TP+near", "TP+\n近い\n戸"), ("改良版", "改良\n全員"), ("改良版top3", "改良\n上位3"), ("改良版near", "改良\n近い\n戸")]
SER = [("0.0", "なし", BLUE, "入れ替わりなし・費用なし"), ("0.0", "C1", "#9bbfea", "入れ替わりなし・費用あり"),
       ("0.1", "なし", AQUA, "毎年 10 %・費用なし"), ("0.1", "C1", "#a2d7cb", "毎年 10 %・費用あり")]
fig, axs = plt.subplots(4, 1, figsize=(7.2, 13.5), sharey=True, gridspec_kw=dict(height_ratios=[0.7, 1, 0.7, 1]))
panels = [("α=3", NULL, "(a) ばらつきやや大（α＝3）：帰無と偽装（P1）・割れ窓（P5）"), ("α=3", RES, "(b) ばらつきやや大（α＝3）：住民主体の決め方（P2〜P4）"),
          ("二群", NULL, "(c) 二群（広さ派 5・静けさ派 4）：帰無と偽装・割れ窓"), ("二群", RES, "(d) 二群：住民主体の決め方")]
for ax, (pat, REGS, title) in zip(axs, panels):
    w = 0.2
    for k, (nu, c, col, lab) in enumerate(SER):
        for i, (r, _) in enumerate(REGS):
            if r == "N2" and nu != "0.0":
                continue
            v, lo, hi, _ = V[f"{r}|{pat}|{nu}|{c}"]
            ax.bar(i + (k - 1.5) * w, v, w, color=col, label=lab if i == 0 else None)
            ax.plot([i + (k - 1.5) * w] * 2, [lo, hi], color=INK2, lw=0.8)
    ax.axhline(eps, color="#a32d2d", lw=1, ls="--")
    ax.text(-0.5, eps + 0.015, f"ε ＝ {eps:.3f}", color="#a32d2d", fontsize=10, ha="left")
    ax.set_xticks(range(len(REGS))); ax.set_xticklabels([l for _, l in REGS], fontsize=10, linespacing=1.0)
    ax.set_xlim(-0.6, len(REGS) - 0.4); ax.set_ylim(0, 0.66)
    ax.set_ylabel("$V^{*}_{n}$", fontsize=11); ax.tick_params(axis="y", labelsize=10); hair(ax)
    ax.set_title(title, fontsize=11.5, loc="left")
axs[1].legend(fontsize=10, frameon=False, ncol=2, loc="upper left")
fig.tight_layout(h_pad=1.2)
fig.savefig("out/figs/fig230_S2_名付けの確かめ.png", dpi=160)
print("ok")
