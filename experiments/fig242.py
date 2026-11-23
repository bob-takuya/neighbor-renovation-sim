"""図 242：運用役が住み手について知ることの段（職能 1 → 2 → 3）と、職能 1 の位置。幅 7.2 in・文字 10 pt 以上・太字なし。
PYTHONPATH=. python3 -m experiments.fig242   → out/figs/fig242_S5a_役者が知ること.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from experiments.style import INK, INK2, MUTED, BLUE, GREY

fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.0); ax.axis("off")
cols = [("職能 1", "性向を知らない役者", "盤の設計者・\n記録者・\n検証者", "平均の性向と分布、記録\nに書かれたこと、制約と\n物理の影響", True),
        ("職能 2", "聞いた言い値だけを\n知る役者", "仲介者（全知でない）", "住み手から聞いた\n言い値", False),
        ("職能 3", "性向を知りうる役者", "聞く範囲を選ぶ者・\nAI を道具とする建築家・\n全知の計画者", "分布だけ、または\n全世帯の性向", False)]
for n, (no, head, who, know, here) in enumerate(cols):
    x = 0.15 + n * 3.3
    col = BLUE if here else "#c9c8c3"
    ax.add_patch(FancyBboxPatch((x, 0.15), 3.0, 4.05, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#f0f6fc" if here else "#f4f3f0", ec=col, lw=2.2 if here else 1.0))
    ax.text(x + 0.15, 4.0, no, fontsize=11, color=BLUE if here else INK, va="top")
    ax.text(x + 0.15, 3.55, head, fontsize=10, color=INK, va="top", linespacing=1.25)
    ax.text(x + 0.15, 2.75, who, fontsize=10, color=INK2, va="top", linespacing=1.3)
    ax.text(x + 0.15, 1.62, "知ること：\n" + know, fontsize=10, color=MUTED, va="top", linespacing=1.25)
    # 知ることの多さを階段で示す
    ax.add_patch(FancyBboxPatch((x, 4.45), 3.0, 0.25 + 0.35 * n, boxstyle="square,pad=0", fc=col, ec="none"))
ax.text(0.15 + 1.5, 5.55, "本章で比べる役者", fontsize=10, color=BLUE, ha="center")
ax.annotate("", xy=(9.85, 5.75), xytext=(3.5, 5.75), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))
ax.text(6.7, 5.85, "住み手の性向について知ることが多くなる", fontsize=10, color=INK2, ha="center", va="bottom")
fig.savefig("out/figs/fig242_S5a_役者が知ること.png", dpi=200, bbox_inches="tight")
print("out/figs/fig242_S5a_役者が知ること.png")
