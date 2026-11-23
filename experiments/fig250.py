"""図 250：運用役が住み手について知ること（職能 1 → 2 → 3）と、職能 2 の位置（図 242 を職能 2 の位置に印を付けて描き直したもの）。
幅 7.2 in・文字 10 pt 以上・太字なし。
PYTHONPATH=. python3 -m experiments.fig250   → out/figs/fig250_S6_役者が知ること_職能2.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from experiments.style import INK, INK2, MUTED, BLUE, GREY, ORANGE

plt.rcParams["axes.titleweight"] = "normal"
plt.rcParams["font.weight"] = "normal"
fig, ax = plt.subplots(figsize=(7.2, 4.5))
ax.set_xlim(0, 10); ax.set_ylim(-0.75, 6.0); ax.axis("off")
cols = [("職能 1", "性向を知らない役者", "盤の設計者・\n記録者・\n検証者", "平均の性向と分布、記録\nに書かれたこと、制約と\n物理の影響", False),
        ("職能 2", "聞いた言い値だけを\n知る役者", "仲介者（全知でない）", "住み手から聞いた\n言い値（補償の額と\n出してよい額）", True),
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
    ax.add_patch(FancyBboxPatch((x, 4.45), 3.0, 0.25 + 0.35 * n, boxstyle="square,pad=0", fc=col, ec="none"))
ax.text(0.15 + 3.3 + 1.5, 5.55, "本章で比べる役者", fontsize=10, color=BLUE, ha="center")
ax.annotate("", xy=(9.85, 5.75), xytext=(6.75, 5.75), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))
ax.text(8.3, 5.85, "知ることが多くなる", fontsize=10, color=INK2, ha="center", va="bottom")
# 比較の参照：全知の仲介者 BR は、知ることの端（全世帯の性向）に置かれる
ax.add_patch(FancyBboxPatch((6.45, -0.68), 3.5, 0.62, boxstyle="round,pad=0.02,rounding_size=0.06", fc="#fcede9", ec=ORANGE, lw=1.0))
ax.text(6.6, -0.37, "比べる参照：全知の仲介者 BR\n（全世帯の性向を知る）", fontsize=10, color=INK, va="center", linespacing=1.2)
ax.annotate("", xy=(6.4, -0.37), xytext=(5.0, 0.1), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.0,
                                                                      connectionstyle="arc3,rad=0.25"))
ax.text(3.2, -0.45, "違いは「誰が何を知るか」だけ", fontsize=10, color=ORANGE, ha="center", va="center")
fig.savefig("out/figs/fig250_S6_役者が知ること_職能2.png", dpi=200, bbox_inches="tight")
print("out/figs/fig250_S6_役者が知ること_職能2.png")
