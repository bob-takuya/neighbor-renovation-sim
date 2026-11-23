"""探索 E0 の図（改訂版）。seaborn と matplotlib。色は決め方の「系統」で四つに限る。
python3 -m experiments.figs_e0_v2
"""
import json, glob, os
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib import font_manager as fm
from trade import s0

for f in fm.findSystemFonts():
    if "NotoSansCJK-Regular" in f or "NotoSansCJK-Bold" in f:
        fm.fontManager.addfont(f)
SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8c8b86"; GRID = "#e6e5e1"
sns.set_theme(style="white", rc={"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False,
                                 "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                                 "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF,
                                 "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})

# ---------------------------------------------------------------- 色：系統で 4 色
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
FAMILY = {"F": ("基準", GREY), "U": ("償わない", ORANGE), "V½": ("償わない", ORANGE), "V¾": ("償わない", ORANGE),
          "T0": ("共通の物差しで償う", AQUA), "T0+": ("共通の物差しで償う", AQUA),
          "T1": ("言い値で償う", BLUE), "T1h": ("言い値で償う", BLUE), "T1+": ("言い値で償う", BLUE)}
ORDER = ["F", "U", "V½", "V¾", "T0", "T0+", "T1", "T1h", "T1+"]
DESC = {"F": "固定（変えない）", "U": "単独・償わない", "V½": "過半の投票", "V¾": "3/4 の投票", "T0": "帳簿どおり・自分の手",
        "T0+": "帳簿どおり・他戸の手も", "T1": "言い値・自分の手", "T1h": "言い値＋余りの半分", "T1+": "言い値・他戸の手も"}
PATS = ["同質", "α=30", "α=3", "α=0.5", "二群", "少数の極端", "階の不一致"]
PAT_DESC = {"同質": "全員が平均の性向", "α=30": "ばらつき小", "α=3": "ばらつき中", "α=0.5": "ばらつき大", "二群": "広さ派 5・静けさ派 4",
            "少数の極端": "1 戸だけ近さ派", "階の不一致": "1F が日照派"}
DIV = LinearSegmentedColormap.from_list("div", ["#a32d2d", "#e34948", "#f0efec", "#2a78d6", "#104281"])
SEQ_O = LinearSegmentedColormap.from_list("seqo", ["#fbeee6", "#f3b08f", "#eb6834", "#a8401a"])
SEQ_B = LinearSegmentedColormap.from_list("seqb", ["#eef4fc", "#9ec5f4", "#2a78d6", "#104281"])
OUT = "out/e0/"; FIG = "out/figs/"
os.makedirs(FIG, exist_ok=True)


def hair(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def ink_on(rgb):
    r, g, b = rgb[:3]
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else INK


df = pd.read_csv(OUT + "runs.csv")
if os.path.exists(OUT + "runs_plus.csv"):
    df = pd.concat([df, pd.read_csv(OUT + "runs_plus.csv")], ignore_index=True)
g = pd.read_csv(OUT + "by_composition.csv") if False else None
g = df.groupby(["pattern", "cseed", "regime"]).agg(H=("H", "first"), eta=("eta", "mean"), dW=("dW", "mean"),
                                                   worse=("worse", "mean"), bal=("bal", "mean"), WF=("WF", "first"),
                                                   Wstar=("Wstar", "first")).reset_index()
summ = g.groupby(["pattern", "regime"]).agg(eta=("eta", "mean"), worse=("worse", "mean"), bal=("bal", "mean")).reset_index()

# ================================================================ 図 1
fig = plt.figure(figsize=(18, 13.2))
fig.text(0.012, 0.988, "図 1　決め方 × 住み手の構成：全知の最適にどこまで近づき、誰が損をし、帳簿はどれだけ釣り合うか",
         fontsize=14, fontweight="bold", va="top", color=INK)
fig.text(0.012, 0.965, "3 階 × 3 戸。7 パターン × 40 構成 × 5 局、30 年。列は決め方の系統ごとに並べた（灰＝基準、橙＝償わない、水色＝共通の物差しで償う、青＝言い値で償う）",
         fontsize=9.5, color=INK2, va="top")


def heat(ax, cax, col, cmap, vmin, vmax, fmt, title, cbar_label, ylabels=True):
    M = summ.pivot(index="pattern", columns="regime", values=col).reindex(index=PATS, columns=ORDER)
    sns.heatmap(M, ax=ax, cbar_ax=cax, cmap=cmap, vmin=vmin, vmax=vmax, annot=False, linewidths=2, linecolor=SURF,
                cbar_kws=dict(label=cbar_label))
    cax.yaxis.label.set_color(INK2); cax.tick_params(labelsize=8, colors=INK2)
    for a, p in enumerate(PATS):
        for b, r in enumerate(ORDER):
            v = M.loc[p, r]
            if pd.isna(v):
                ax.text(b + 0.5, a + 0.5, "—", ha="center", va="center", fontsize=8.5, color=MUTED)
            else:
                ax.text(b + 0.5, a + 0.5, fmt.format(v), ha="center", va="center", fontsize=8.5,
                        color=ink_on(cmap(np.clip((v - vmin) / (vmax - vmin), 0, 1))))
    ax.set_xlabel(""); ax.set_ylabel("")
    if ylabels:
        ax.set_yticklabels([f"{p}（{PAT_DESC[p]}）" for p in PATS], rotation=0, fontsize=8.8, color=INK2)
    else:
        ax.set_yticklabels([])
    ax.set_xticklabels(ORDER, rotation=0, fontsize=9, color=INK2)
    groups = [("基準", 0, 1, GREY), ("償わない", 1, 4, ORANGE), ("共通の物差し", 4, 6, AQUA), ("言い値", 6, 9, BLUE)]
    for name, a, b, c in groups:
        ax.plot([a + 0.08, b - 0.08], [-0.22, -0.22], color=c, lw=4, solid_capstyle="butt", clip_on=False)
        ax.text((a + b) / 2, -0.42, name, ha="center", va="bottom", fontsize=8.3, color=INK2)
    ax.set_title(title, pad=40)


ax = fig.add_axes([0.115, 0.56, 0.235, 0.33]); cax = fig.add_axes([0.355, 0.56, 0.008, 0.33])
heat(ax, cax, "eta", DIV, -1, 1, "{:.2f}", "(a) 効率 η", "η（0＝F と同じ、1＝全知の最適）")
ax = fig.add_axes([0.425, 0.56, 0.235, 0.33]); cax = fig.add_axes([0.665, 0.56, 0.008, 0.33])
heat(ax, cax, "worse", SEQ_O, 0, 0.7, "{:.2f}", "(b) F より損をした世帯の割合", "割合（0＝誰も損をしない）", ylabels=False)
ax = fig.add_axes([0.735, 0.56, 0.235, 0.33]); cax = fig.add_axes([0.975, 0.56, 0.008, 0.33])
heat(ax, cax, "bal", SEQ_B, 0, 10, "{:.1f}", "(c) 帳簿の残高 Σ|b|／9", "㎡ 換算（0＝ゼロサムで平等）", ylabels=False)

# (d) 小さな多重図：系統ごと
gg = g[g.pattern.str.startswith("α=")]
star = gg.groupby(["pattern", "cseed"]).agg(H=("H", "first"), d=("Wstar", "first"), WF=("WF", "first")).reset_index()
star["dstar"] = star.d - star.WF
bins = np.quantile(star.H, np.linspace(0, 1, 7)); mid = (bins[1:] + bins[:-1]) / 2


def med_line(s, col):
    return np.array([s[(s.H >= bins[q]) & (s.H <= bins[q + 1])][col].median() for q in range(len(bins) - 1)])


fams = [("償わない", ["U", "V½", "V¾"], ORANGE), ("共通の物差しで償う", ["T0", "T0+"], AQUA), ("言い値で償う", ["T1", "T1+"], BLUE)]
styles = {"U": "-", "V½": "--", "V¾": ":", "T0": "--", "T0+": "-", "T1": "--", "T1+": "-"}
ylo = min(-1.0, gg[gg.regime.isin(sum([f[1] for f in fams], []))].dW.quantile(0.01)); yhi = star.dstar.max() * 1.08
for fi, (fname, regs, c) in enumerate(fams):
    ax = fig.add_axes([0.06 + fi * 0.205, 0.07, 0.18, 0.36])
    ax.scatter(star.H, star.dstar, s=10, color=GREY, alpha=0.45, lw=0, zorder=1)
    wl = med_line(star.rename(columns={"dstar": "v"}), "v")
    ax.plot(mid, wl, color=INK2, lw=2, zorder=3)
    ends = [(wl[-1], "W*")]
    for r in regs:
        s_ = gg[gg.regime == r]
        ax.scatter(s_.H, s_.dW, s=10, color=c, alpha=0.35, lw=0, zorder=2)
        ml = med_line(s_, "dW")
        ax.plot(mid, ml, color=c, lw=2, ls=styles[r], zorder=4)
        ends.append((ml[-1], r))
    ends.sort(); last = -1e9; span = yhi - ylo
    for yv, lab in ends:
        yl = max(yv, last + 0.06 * span); last = yl
        ax.plot([mid[-1], mid[-1] + 0.035], [yv, yl], color=MUTED, lw=0.6)
        ax.text(mid[-1] + 0.04, yl, lab, fontsize=9, color=INK, va="center")
    ax.axhline(0, color=INK2, lw=1)
    ax.text(0.02, 0.02, "F", fontsize=9, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_ylim(ylo, yhi); ax.set_xlim(0.03, 0.72)
    ax.set_xlabel("性向のばらつき H"); hair(ax)
    if fi == 0:
        ax.set_ylabel("ΔW（F に対する満足の合計の差）")
    else:
        ax.set_yticklabels([])
    ax.set_title(f"{'(d) ' if fi == 0 else ''}{fname}", color=INK)
fig.text(0.06, 0.475, "(d) ばらつきと、固定方式に対する差。点＝構成（5 局の平均）、線＝ばらつきの区間ごとの中央値。灰の点と線＝その構成の全知の最適 W*（天井）",
         fontsize=9, color=INK2)

# (e) 打たれた手の内訳（系統ごとに並べた横棒、6 色、2px の隙間）
ax = fig.add_axes([0.70, 0.07, 0.27, 0.36])
types_order = ["張り出す3F", "張り出す2F", "外す2F", "外す1F", "収納", "開口"]
TC = dict(zip(types_order, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
TL = {"張り出す3F": "3F が張り出す", "張り出す2F": "2F が張り出す", "外す2F": "2F が外す", "外す1F": "1F が外す", "収納": "戸境収納を回す", "開口": "開口"}
regs = [r for r in ORDER if r != "F"]
agg = {}
for r in regs:
    s = df[(df.regime == r) & (df.pattern != "同質")]
    cnt = Counter()
    for js in s.types:
        cnt.update(json.loads(js))
    agg[r] = {t: cnt[t] / max(len(s), 1) for t in types_order}
left = np.zeros(len(regs))
for t in types_order:
    v = np.array([agg[r][t] for r in regs])
    ax.barh(range(len(regs)), v, left=left, color=TC[t], height=0.55, edgecolor=SURF, linewidth=2, label=TL[t])
    left += v
fails = [df[(df.regime == r) & (df.pattern != "同質")].n_fail.mean() for r in regs]
for y, (tot, fl) in enumerate(zip(left, fails)):
    ax.text(tot + 0.8, y, f"{tot:.1f}（却下 {fl:.0f}）", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(regs))); ax.set_yticklabels(regs, color=INK2); ax.invert_yaxis()
ax.tick_params(axis="y", pad=12, length=0)
for y, r in enumerate(regs):
    ax.add_patch(Rectangle((-0.028, y - 0.28), 0.018, 0.56, color=FAMILY[r][1], transform=ax.get_yaxis_transform(), clip_on=False))
ax.set_xlim(0, max(left) * 1.35); ax.set_xlabel("30 年に実行された手の数（一局あたり）")
hair(ax, "x")
ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(-0.02, 1.0), handlelength=1.2)
ax.set_title("(e) 打たれた手の内訳（棒の右：合計と却下の数）", pad=34)
fig.savefig(FIG + "fig217_E0_決め方と住み手の構成.png", dpi=150)
plt.close(fig)

# ================================================================ 図 2
shows = [p for p in sorted(glob.glob(OUT + "show_*.json"))]
order_show = ["α=0.5", "α=30", "二群", "階の不一致"]
shows.sort(key=lambda p: [i for i, k in enumerate(order_show) if f"show_{k}_" in p][0])
fig = plt.figure(figsize=(18, 6.4 * len(shows) + 1.6))
H = 1.0 / (len(shows) + 0.25)
fig.text(0.012, 0.997, "図 2　全配置（23,887,872 通り）の中で、各決め方の 100 局がたどり着いた所（同じ住み手で、要求の順序の乱数だけを変えた 100 局）",
         fontsize=14, fontweight="bold", va="top", color=INK)
# 共通の凡例（上の帯に一度だけ）
lg = fig.add_axes([0.012, 1 - 0.245 * H, 0.976, 0.165 * H]); lg.set_xlim(0, 100); lg.set_ylim(0, 10); lg.axis("off")
lg.text(0, 9.5, "左：上は全配置の満足の合計の分布（灰、縦は対数）、下は各決め方の 100 局の到達点（1 行＝1 つの決め方。行の中の上下のばらつきは重なりを避けるためだけで意味はない。黒い縦の短線＝中央値）。縦の線＝F と W*。行の右＝中央値の効率 η と、100 局に現れた形の種類",
        fontsize=8.8, color=INK2, va="top")
lg.text(0, 7.4, "中：満足の合計と、最も不遇な世帯の満足（灰の濃淡＝全配置、黒い線＝合計ごとの min S の上限）。点は 100 局の到達点", fontsize=8.8, color=INK2, va="top")
for xi, (lab, c, mk) in enumerate((("U", ORANGE, "o"), ("T0+", AQUA, "o"), ("T1+", BLUE, "o"), ("F", INK, "x"))):
    lg.scatter([60 + xi * 6], [6.9], s=28 if mk == "o" else 40, color=c, marker=mk, edgecolor=SURF if mk == "o" else None, lw=1)
    lg.text(60.9 + xi * 6, 6.9, lab, fontsize=8.8, va="center", color=INK)
lg.text(0, 5.2, "右の立面（上が 3F、左から W・M・E）：", fontsize=8.8, color=INK2, va="top")
x0 = 17
lg.add_patch(Rectangle((x0, 3.4), 1.6, 1.6, fc="#fff8e8", ec=INK, lw=0.7)); lg.add_patch(Rectangle((x0 + 0.2, 3.4), 1.2, 0.35, fc="#c8462d"))
lg.text(x0 + 2.0, 4.2, "赤い帯＝張り出し", fontsize=8.8, va="center", color=INK2)
x0 = 29
lg.add_patch(Rectangle((x0 + 0.6, 3.3), 0.3, 1.8, fc="#4f7a2f")); lg.text(x0 + 1.5, 4.2, "←", fontsize=9, color="#4f7a2f", va="center")
lg.text(x0 + 2.6, 4.2, "戸境の緑の棒＝収納が片側に回された（矢印の側の住戸が全部使う）", fontsize=8.8, va="center", color=INK2)
x0 = 63
lg.plot([x0, x0 + 1.6], [4.2, 4.2], color="#6b4c9a", lw=2.4); lg.text(x0 + 2.0, 4.2, "紫の短い線＝開口", fontsize=8.8, va="center", color=INK2)
x0 = 75
for xi, cc in enumerate(["#4f7a2f", "#e0a526", "#5b7fb5", "#c8462d"]):
    lg.add_patch(Rectangle((x0 + xi * 0.9, 3.7), 0.6, 1.0, fc=cc))
lg.text(x0 + 4.0, 4.2, "左上の四角＝一番重く見る量\n（緑 広さ・黄 日照・青 静けさ・赤 近さ）", fontsize=8.8, va="center", color=INK2)
lg.plot([0, 100], [1.2, 1.2], color=GRID, lw=1)

for si, path in enumerate(shows):
    D = json.load(open(path, encoding="utf-8"))
    top = 1 - 0.25 * H - si * H
    regs_here = [r for r in ORDER if r in D["runs"]]
    x1 = np.array(D["x1"]); h1 = np.array(D["h1"])
    allw = np.concatenate([[x["W"] for x in D["runs"][r]] for r in regs_here])
    xlo = min(D["WF"], allw.min()) - 0.06 * (D["wstar"] - min(D["WF"], allw.min()) + 0.2); xhi = D["wstar"] + 0.04 * (D["wstar"] - xlo)
    # 上：分布（全配置）
    a0 = fig.add_axes([0.15, top - 0.20 * H, 0.38, 0.15 * H])
    cen = (x1[1:] + x1[:-1]) / 2
    a0.fill_between(cen, h1, step="mid", color="#d9d8d3", lw=0)
    a0.set_yscale("log"); a0.set_xlim(x1[0], x1[-1]); a0.set_yticks([1e2, 1e4, 1e6])
    a0.axvspan(xlo, xhi, color=BLUE, alpha=0.06, lw=0)
    for x, lab in ((D["WF"], "F"), (D["wstar"], "W*")):
        a0.axvline(x, color=INK2, lw=1); a0.text(x, 0.97, f" {lab}", transform=a0.get_xaxis_transform(), ha="left", va="top", fontsize=9, color=INK)
    a0.set_ylabel("配置の数", fontsize=8.5); hair(a0); a0.tick_params(labelsize=8)
    a0.set_title(f"({'abcd'[si]}) {D['pattern']}（{PAT_DESC.get(D['pattern'], '')}、構成 {D['cseed']}）", pad=8)
    a0.text(1.0, 1.04, "青い帯の範囲を下に拡大", transform=a0.transAxes, ha="right", va="bottom", fontsize=8.3, color=INK2)
    # 下：決め方ごとの帯状の点（行＝決め方、横＝満足の合計）
    a1 = fig.add_axes([0.15, top - 0.88 * H, 0.38, 0.62 * H])
    rows = []
    for r in regs_here:
        for x in D["runs"][r]:
            rows.append(dict(regime=r, W=x["W"], fam=FAMILY[r][0]))
    rdf = pd.DataFrame(rows)
    pal = {r: FAMILY[r][1] for r in regs_here}
    sns.stripplot(data=rdf, x="W", y="regime", order=regs_here, hue="regime", palette=pal, legend=False, ax=a1,
                  size=4.2, jitter=0.28, alpha=0.65, linewidth=0)
    med = rdf.groupby("regime").W.median()
    for yi, r in enumerate(regs_here):
        a1.plot([med[r]] * 2, [yi - 0.38, yi + 0.38], color=INK, lw=1.6)
        cnt = Counter((tuple(x["e"]), tuple(x["k"]), tuple(x["o"])) for x in D["runs"][r])
        eta = (med[r] - D["WF"]) / (D["wstar"] - D["WF"])
        a1.text(1.01, yi, f"η {eta:+.2f}　形 {len(cnt)} 種", transform=a1.get_yaxis_transform(), va="center", fontsize=8.5, color=INK2)
    for x, lab in ((D["WF"], "F"), (D["wstar"], "W*")):
        a1.axvline(x, color=INK2, lw=1, zorder=0)
    a1.set_xlim(xlo, xhi); a1.set_ylabel(""); a1.set_xlabel("満足の合計 W（30 年後）")
    a1.set_yticks(range(len(regs_here))); a1.set_yticklabels([f"{DESC[r]}  {r}" for r in regs_here], fontsize=8.5, color=INK2)
    hair(a1, "x")
    for yi, r in enumerate(regs_here):
        a1.add_patch(Rectangle((-0.012, yi - 0.35), 0.008, 0.7, color=FAMILY[r][1], transform=a1.get_yaxis_transform(), clip_on=False))
    # 中：合計と最も不遇な世帯（要の 4 つだけ）
    b = fig.add_axes([0.635, top - 0.88 * H, 0.13, 0.83 * H])
    H2 = np.array(D["H2"]); xe = np.array(D["xe"]); ye = np.array(D["ye"])
    b.imshow(np.log1p(H2.T), origin="lower", extent=(xe[0], xe[-1], ye[0], ye[-1]), aspect="auto", cmap="Greys", alpha=0.45)
    fb = np.array(D["fbins"]); fr = np.array(D["front"]); ok_ = np.isfinite(fr)
    b.plot(((fb[1:] + fb[:-1]) / 2)[ok_], fr[ok_], color=INK2, lw=1.2)
    for r, mk in (("U", "o"), ("T0+", "o"), ("T1+", "o")):
        if r in D["runs"]:
            b.scatter([x["W"] for x in D["runs"][r]], [x["minS"] for x in D["runs"][r]], s=22, color=FAMILY[r][1], edgecolor=SURF, lw=1.2, label=r, zorder=3)
    b.scatter([D["WF"]], [D["minSF"]], marker="x", s=55, color=INK, label="F", zorder=4)
    b.set_xlim(xlo, xhi)
    b.set_xlabel("満足の合計 W"); b.set_ylabel("最も不遇な世帯の満足 min S"); hair(b, "both"); b.tick_params(labelsize=8)
    b.set_title("合計と、最も不遇な世帯", fontsize=10, pad=8)
    # 右：立面
    c = fig.add_axes([0.78, top - 0.85 * H, 0.215, 0.62 * H]); c.set_xlim(0, 30); c.set_ylim(0, 10.5); c.axis("off")

    def board(x0, y0, e, k, o, title, W=None):
        cw, ch = 2.6, 2.4
        for i in range(9):
            f = s0.FLOOR[i]; col = s0.COL[i]
            x = x0 + col * cw; y = y0 + (f - 1) * ch
            c.add_patch(Rectangle((x, y), cw, ch, fc="#fff8e8", ec=INK, lw=0.7))
            if W is not None:
                dom = int(np.argmax(np.array(W[i])))
                c.add_patch(Rectangle((x + 0.15, y + ch - 0.55), 0.4, 0.4, fc=["#4f7a2f", "#e0a526", "#5b7fb5", "#c8462d"][dom]))
            if e[i]:
                c.add_patch(Rectangle((x + 0.3, y), cw - 0.6, 0.35, fc="#c8462d"))
        for bb in range(6):
            l, rr = s0.b_units(bb)
            f = s0.FLOOR[l]; col = s0.COL[l]
            x = x0 + (col + 1) * cw; y = y0 + (f - 1) * ch
            if k[bb] != 0:
                c.add_patch(Rectangle((x - 0.18, y + 0.5), 0.36, 1.2, fc="#4f7a2f"))
                c.text(x + (0.45 if k[bb] > 0 else -0.45), y + 1.1, "←" if k[bb] > 0 else "→", fontsize=7, ha="center", va="center", color="#4f7a2f")
            if o[bb]:
                c.plot([x - 0.35, x + 0.35], [y + 1.9, y + 1.9], color="#6b4c9a", lw=2.2)
        c.text(x0, y0 - 0.5, title, fontsize=8.5, va="top", color=INK2)

    board(0.5, 2.0, *D["star"], "W*\n全知の最適", D["W"])
    for xi, r in enumerate(["T1+" if "T1+" in D["runs"] else "T1", "U"]):
        cnt = Counter((tuple(x["e"]), tuple(x["k"]), tuple(x["o"])) for x in D["runs"][r])
        (cfg, n) = cnt.most_common(1)[0]
        board(10.5 + 10 * xi, 2.0, np.array(cfg[0]), np.array(cfg[1]), np.array(cfg[2]), f"{r} の最頻の形\n{n}/100 局、全 {len(cnt)} 種", D["W"])
fig.savefig(FIG + "fig218_E0_全配置の分布と到達点.png", dpi=130)
plt.close(fig)

# ================================================================ 図 3
fig, axs = plt.subplots(1, 3, figsize=(17, 5.4), sharey=False)
fig.subplots_adjust(left=0.05, right=0.95, top=0.80, bottom=0.17, wspace=0.28)
fig.text(0.012, 0.97, "図 3　F に対する差 ΔW の 30 年の推移（線＝中央値、帯＝中央の半分の局）", fontsize=14, fontweight="bold", va="top", color=INK)
show_regs = [("U", "-"), ("V¾", ":"), ("T0+", "-"), ("T1", "--"), ("T1+", "-")]
handles = [Line2D([0], [0], color=FAMILY[r][1], ls=ls, lw=2, label=f"{r}（{DESC[r]}）") for r, ls in show_regs]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.05, 0.92), ncol=5, frameon=False, fontsize=9)
for ax, p in zip(axs, ["α=30", "α=0.5", "二群"]):
    s = df[df.pattern == p]
    ends = []
    for r, ls in show_regs:
        t = s[s.regime == r]
        M = np.array([json.loads(x) for x in t.Wt]) - t.WF.values[:, None]
        med = np.median(M, 0); lo = np.percentile(M, 25, 0); hi = np.percentile(M, 75, 0)
        ax.fill_between(range(M.shape[1]), lo, hi, color=FAMILY[r][1], alpha=0.10, lw=0)
        ax.plot(range(M.shape[1]), med, color=FAMILY[r][1], lw=2, ls=ls)
        ends.append((med[-1], r))
    # 端の名前（重なれば少しずらし、細い引き出し線で結ぶ）
    ends.sort()
    span = ax.get_ylim()[1] - ax.get_ylim()[0]; last = -1e9
    for yv, r in ends:
        ylab = max(yv, last + 0.07 * span); last = ylab
        ax.plot([30, 31.2], [yv, ylab], color=MUTED, lw=0.6, clip_on=False)
        ax.text(31.4, ylab, r, va="center", fontsize=8.8, color=INK, clip_on=False)
    ax.axhline(0, color=INK2, lw=1); ax.text(0.3, 0, " F", va="bottom", fontsize=9, color=INK2)
    ax.set_xlim(0, 30); ax.set_xlabel("年"); ax.set_ylabel("ΔW"); hair(ax)
    ax.set_title(f"{p}（{PAT_DESC[p]}）", color=INK)
fig.text(0.05, 0.015, "各パネルの縦の目盛は違う（ばらつきが大きいほど、取引で稼げる幅が大きい）。T1h・T0・V½ は図 1 を参照", fontsize=8.5, color=INK2)
fig.savefig(FIG + "fig219_E0_適応の推移.png", dpi=150)
plt.close(fig)
print("ok")
