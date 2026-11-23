"""図（段 1）：模型の盤・手・一年の手順。本文と同じ字の大きさで読めるよう、幅 7.2 in・文字 10 pt 以上で縦に積む"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from experiments.style import SURF, INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, GREY, DEEP

RED, GREEN, PURPLE = "#d0533f", "#5a8a3a", "#7a5aa8"
fig = plt.figure(figsize=(7.2, 12.0))
# (a) 盤
ax = fig.add_axes([0.03, 0.665, 0.94, 0.325]); ax.set_xlim(-0.1, 6.6); ax.set_ylim(-0.1, 2.95); ax.axis("off")
ax.text(-0.1, 2.93, "(a) 盤（南から見た立面）と三種の手", fontsize=12, fontweight="bold", va="top", color=INK)
sun = {3: "6.0 h", 2: "5.0 h", 1: "3.5 h"}
for f in (3, 2, 1):
    for c, cn in enumerate("WME"):
        y = f - 1
        ax.add_patch(Rectangle((c * 0.9, y * 0.9), 0.9, 0.9, fc="#fdf6e3" if f == 3 else "#fdfaf2", ec=INK, lw=1.2))
        ax.text(c * 0.9 + 0.06, y * 0.9 + 0.84, f"{f}{cn}", fontsize=10.5, fontweight="bold", va="top", color=INK)
    ax.text(2.78, (f - 1) * 0.9 + 0.45, f"{f}F　日照 {sun[f]}", fontsize=10, color=INK2, va="center")
ax.add_patch(Rectangle((0.1, 1.8), 0.7, 0.11, fc=RED, ec="none"))
ax.add_patch(Rectangle((1.75, 1.0), 0.1, 0.5, fc=GREEN, ec="none"))
ax.plot([0.9, 0.9], [0.25, 0.5], color=PURPLE, lw=4, solid_capstyle="butt")
ax.text(4.25, 2.62, "各戸 50 ㎡ から始める。\n手は次の三種で、どれも戻せる。", fontsize=10, color=INK2, va="top")
items = [(RED, "張り出し：自戸 ＋6 ㎡。\n真下 −1.5 h・二つ下 −0.5 h の日照"),
         (GREEN, "戸境収納：3 ㎡ を隣へ回す\n（面積はゼロサム）"),
         (PURPLE, "開口：両側の近さ ＋、\n静けさ −")]
for n, (col, t) in enumerate(items):
    yy = 1.72 - n * 0.68
    ax.add_patch(Rectangle((4.25, yy - 0.05), 0.16, 0.12, fc=col, ec="none"))
    ax.text(4.5, yy, t, fontsize=10, color=INK, va="center")
# (b) 世帯と満足
bx = fig.add_axes([0.03, 0.36, 0.94, 0.28]); bx.axis("off"); bx.set_xlim(0, 1); bx.set_ylim(0, 1)
bx.text(0, 0.995, "(b) 世帯の性向と満足", fontsize=12, fontweight="bold", va="top", color=INK)
lines = [("性向 w", "四つの量への重み（和 1）：広さ・日照・静けさ・近さ"),
         ("", "平均の性向 w̄ ＝ (0.35, 0.30, 0.20, 0.15)"),
         ("満足 S", "S ＝ Σ w × φ(量)。φ は量を 0〜1 にそろえる関数"),
         ("", "（主条件は線形。前提を崩す条件で逓減・日照の下限つき）"),
         ("お金 m", "償いとして受け渡した額（㎡ 換算、全戸の和は 0）"),
         ("暮らし向き", "S ＋ μ·m。μ は 1 ㎡ 分の満足（＝0.35／24）"),
         ("帳簿 b", "他戸に与えた損得を、平均の性向 w̄ で測った貸し借り"),
         ("", "（全戸の和は 0。全員 0 なら「ゼロサムで見て平等」）")]
y = 0.87
for k, v in lines:
    if k:
        bx.text(0, y, k, fontsize=10.5, fontweight="bold", color=DEEP, va="top")
    bx.text(0.2, y, v, fontsize=10, color=INK, va="top")
    y -= 0.112 if k else 0.098
# (c) 一年の手順
cx = fig.add_axes([0.03, 0.01, 0.94, 0.34]); cx.axis("off"); cx.set_xlim(0, 1); cx.set_ylim(0, 1)
cx.text(0, 0.995, "(c) 一年の手順（30 年くり返す）", fontsize=12, fontweight="bold", va="top", color=INK)
steps = [("要求", "各世帯が確率 p ＝ 0.3 で「変えたい」と思う\n（順番は毎年ランダム）"),
         ("手を選ぶ", "打てる手のうち、自分の性向で得が最大の手。\n＋の付く決め方では、他戸の手も頼める"),
         ("決める", "決め方の規則で、実行するか見送るかを決め、\n償いを払う"),
         ("記帳", "盤を更新し、お金 m と帳簿 b を動かす"),
         ("入れ替わり", "入れ替わりのある条件では、世帯が転出し、\n別の性向の世帯が入る")]
y = 0.9
for n, (k, v) in enumerate(steps):
    cx.add_patch(FancyBboxPatch((0.0, y - 0.15), 0.99, 0.14, boxstyle="round,pad=0.005,rounding_size=0.015",
                                fc="#f0f6fc" if n != 2 else "#fcede9", ec="#9bbfea" if n != 2 else "#eea18a", lw=1))
    cx.text(0.02, y - 0.025, f"{n+1}  {k}", fontsize=10.5, fontweight="bold", color=INK, va="top")
    cx.text(0.25, y - 0.025, v, fontsize=10, color=INK, va="top", linespacing=1.3)
    y -= 0.175
fig.savefig("out/figs/fig222_S1_模型の盤と手順.png", dpi=170)
print("ok")
