"""図 260：運用役が住み手について知ること（職能 1 → 2 → 3）と、職能 3 の位置（図 250 に合わせて、職能 3 の運用役を知ることの順に並べ直したもの）。
幅 7.2 in・文字 10 pt 以上・太字なし。
PYTHONPATH=. python3 -m experiments.fig260   → out/figs/fig260_S7_役者が知ること_職能3.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from experiments.style import INK, INK2, MUTED, BLUE, GREY, ORANGE

plt.rcParams["axes.titleweight"] = "normal"
plt.rcParams["font.weight"] = "normal"
fig, ax = plt.subplots(figsize=(7.2, 5.6))
ax.set_xlim(0, 10); ax.set_ylim(-0.95, 7.1); ax.axis("off")
cols = [(0.1, 2.2, "職能 1", "性向を知らない\n役者", "盤の設計者・\n記録者・検証者", "平均の性向と\n分布、記録、\n制約と物理の\n影響", False),
        (2.45, 2.2, "職能 2", "聞いた言い値\nだけを知る役者", "仲介者\n（全知でない）", "住み手から\n聞いた言い値", False),
        (4.8, 5.1, "職能 3", "性向を知りうる役者", None, None, True)]
for n, (x, w, no, head, who, know, here) in enumerate(cols):
    col = BLUE if here else "#c9c8c3"
    ax.add_patch(FancyBboxPatch((x, 0.1), w, 5.2, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#f0f6fc" if here else "#f4f3f0", ec=col, lw=2.2 if here else 1.0))
    ax.add_patch(FancyBboxPatch((x, 5.5), w, 0.12 + 0.18 * n, boxstyle="square,pad=0", fc=col, ec="none"))
    ax.text(x + 0.12, 5.12, no, fontsize=11, color=BLUE if here else INK, va="top")
    ax.text(x + 0.12, 4.65, head, fontsize=10, color=INK, va="top", linespacing=1.25)
    if who:
        ax.text(x + 0.12, 3.55, who, fontsize=10, color=INK2, va="top", linespacing=1.3)
        ax.text(x + 0.12, 2.35, "知ること：\n" + know, fontsize=10, color=MUTED, va="top", linespacing=1.25)
# 職能 3 の中：運用役を知ることの少ない順に、二通りの置き方とともに
rows = [("聞く範囲を選ぶ者", "分布だけ（個々の性向は知らない）", "竣工の時の全世帯の性向"),
        ("AI を道具とする建築家", "物理の量（性向を知らない）", "満足の量（毎年、全世帯の性向を読む）"),
        ("全知の計画者", None, "全世帯の性向（入れ替わりのたびに）")]
y = 4.05
for n, (who, lo, hi) in enumerate(rows):
    ax.text(4.95, y, f"{n + 1}. {who}", fontsize=10, color=INK, va="top")
    if lo:
        ax.text(5.15, y - 0.42, "・" + lo, fontsize=10, color=INK2, va="top")
    ax.text(5.15, y - (0.84 if lo else 0.42), "・" + hi, fontsize=10, color=INK2, va="top")
    y -= 1.35 if lo else 0.95
ax.text(4.8 + 2.55, 6.05, "職能 3 の章で比べる役者", fontsize=10, color=BLUE, ha="center")
ax.annotate("", xy=(9.9, 6.6), xytext=(0.1, 6.6), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))
ax.text(1.6, 6.67, "知ることが多くなる", fontsize=10, color=INK2, ha="center", va="bottom")
# 比べる参照：職能 2 の仲介者と、全知の仲介者 BR
ax.add_patch(FancyBboxPatch((2.45, -0.9), 7.45, 0.72, boxstyle="round,pad=0.02,rounding_size=0.06", fc="#fcede9", ec=ORANGE, lw=1.0))
ax.text(2.6, -0.54, "比べる参照：職能 2 の仲介者（届かなかった手だけを聞く）、\n全知の仲介者 BR（全世帯の性向を知り、同意を得て組み替える）",
        fontsize=10, color=INK, va="center", linespacing=1.2)
fig.savefig("out/figs/fig260_S7_役者が知ること_職能3.png", dpi=200, bbox_inches="tight")
print("out/figs/fig260_S7_役者が知ること_職能3.png")
