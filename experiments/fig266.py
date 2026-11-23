"""図 266：職能 1〜3 を通して、運用役を知ることの順に並べた〔外〕と〔内〕（TP+top3・入れ替わりなし）。
職能 2 の広く聞く仲介者は、職能 2 の判定 out/s6/judge.json（凍結した出力を読むだけ）から、ほかは out/s7/judge.json から。
職能 1 の三つの運用役は個々の係数を知らないので、定義から〔内〕＝〔外〕。値は運用役ごとに違うので、行だけ示す。
幅 7.2 in・文字 10 pt 以上・太字なし。
PYTHONPATH=. python3 -m experiments.fig266   → out/figs/fig266_S7_職能1から3を通して.png（S7_DRY=1 なら dry_figs へ）
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiments.style import INK, INK2, MUTED, BLUE, ORANGE, AQUA, GREY, hair

plt.rcParams["axes.titleweight"] = "normal"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["font.size"] = 10
DRY = os.environ.get("S7_DRY") == "1"
J = json.load(open("out/s7/dry_judge.json" if DRY else "out/s7/judge.json"))
J6 = json.load(open("out/s6/judge.json"))
OUTD = "out/s7/dry_figs" if DRY else "out/figs"
B = "TP+top3"
PATS = ["α=3", "二群"]


def row7(arm, cond, p):
    k = f"参照|{arm}|基準|{p}|0.0" if arm in ("計画者", "BR") else f"{B}|{arm}|{cond}|{p}|0.0"
    nf = 27.0 if arm in ("計画者", "BR") else J["means"][k]["nfacts"]
    return nf, J["values"][k + "|Vn外"], J["values"][k + "|Vn数"]


def row6(arm, cond, p):
    k = f"{B}|{arm}|{cond}|{p}|0.0"
    return J6["means"][k]["nfacts"], J6["values"][k + "|Vn外"], J6["values"][k + "|Vn数"]


ROWS = [("職能 1 の三つの運用役", None, GREY),
        ("職能 3 範囲を選ぶ者・分布だけ", ("7", "選ぶ者:分布", "基準"), AQUA),
        ("職能 3 道具・物理の量（近い戸だけ）", ("7", "道具:物理", "近い戸だけ"), AQUA),
        ("職能 2 仲介者（近い戸だけ）", ("7", "仲介者", "近い戸だけ"), MUTED),
        ("職能 2 広く聞く仲介者（近い戸だけ）", ("6", "広く聞く仲介者", "近い戸だけ"), MUTED),
        ("職能 3 範囲を選ぶ者・性向を知る", ("7", "選ぶ者:性向", "基準"), BLUE),
        ("職能 3 道具・満足の量（近い戸だけ）", ("7", "道具:満足", "近い戸だけ"), BLUE),
        ("全知の仲介者 BR", ("7", "BR", "基準"), ORANGE),
        ("職能 3 全知の計画者", ("7", "計画者", "基準"), ORANGE)]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.4), sharey=True, gridspec_kw=dict(width_ratios=[0.8, 1, 1]))
ys = np.arange(len(ROWS))[::-1]
for y, (lab, src, col) in zip(ys, ROWS):
    if src is None:
        axes[0].barh(y, 0.0, color=col)
        axes[0].text(0.5, y, "0", va="center", fontsize=10, color=INK2)
        for ax in axes[1:]:
            ax.text(0.02, y, "〔内〕＝〔外〕", va="center", fontsize=10, color=MUTED)
        continue
    f = row7 if src[0] == "7" else row6
    nf = np.mean([f(src[1], src[2], p)[0] for p in PATS])
    axes[0].barh(y, nf, color=col, height=0.6)
    axes[0].text(nf + 0.5, y, f"{nf:.0f}", va="center", fontsize=10, color=INK2)
    for ax, p in zip(axes[1:], PATS):
        _, o, c = f(src[1], src[2], p)
        ax.plot([o["v"], c["v"]], [y, y], color=col, lw=1.6, zorder=1)
        ax.scatter([c["v"]], [y], s=70, facecolor="white", edgecolor=col, lw=1.4, zorder=3)
        ax.scatter([o["v"]], [y], s=26, color=col, zorder=4)
axes[0].set_xlim(0, 33); axes[0].set_xlabel("30 年目に知っている係数\n（27 のうち）", fontsize=10)
axes[0].set_title("(a) 運用役が知ること", fontsize=10, color=INK)
for ax, p in zip(axes[1:], PATS):
    ax.set_xlim(-0.03, 0.95); ax.set_xlabel("$V^{*}_{n}$", fontsize=10)
    ax.set_title(f"({'b' if p == 'α=3' else 'c'}) {'α＝3' if p == 'α=3' else '二群'}", fontsize=10, color=INK)
for ax in axes:
    hair(ax, "x"); ax.tick_params(labelsize=10)
axes[0].set_yticks(ys); axes[0].set_yticklabels([r[0] for r in ROWS], fontsize=10)
hs = [Line2D([], [], color=INK2, marker="o", ls="", ms=5), Line2D([], [], color=INK2, marker="o", mfc="white", ls="", ms=8)]
fig.legend(hs, ["〔外〕運用役を作者に含めない", "〔内〕運用役を作者に含める"], loc="upper center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.62, 1.06))
fig.tight_layout()
os.makedirs(OUTD, exist_ok=True)
fn = f"{OUTD}/fig266_S7_職能1から3を通して.png"
fig.savefig(fn, dpi=200, bbox_inches="tight"); print(fn)
