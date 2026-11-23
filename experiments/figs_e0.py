"""探索 E0 の図。python3 -m experiments.figs_e0"""
import json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager as fm
from trade import s0

for f in fm.findSystemFonts():
    if "NotoSansCJK-Regular" in f or "NotoSansCJK-Bold" in f:
        fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
INK = "#222"; GREY = "#8a8a8a"; LG = "#dddddd"
RC = {"F": "#8a8a8a", "U": "#c8462d", "V½": "#e3a14a", "V¾": "#a8641a", "T0": "#7fa7d4", "T1": "#2f6aa3", "T1h": "#163d6b", "T0+": "#6fbf9a", "T1+": "#1b7a4f"}
REGS = ["F", "U", "V½", "V¾", "T0", "T1", "T1h", "T0+", "T1+"]
PATS = ["同質", "α=30", "α=3", "α=0.5", "二群", "少数の極端", "階の不一致"]
OUT = "out/e0/"
FIG = "out/figs/"
import os; os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(OUT + "runs.csv")
if os.path.exists(OUT + "runs_plus.csv"):
    df = pd.concat([df, pd.read_csv(OUT + "runs_plus.csv")], ignore_index=True)
REGS = [r for r in REGS if r in set(df.regime)]
# 構成ごと（5 局の平均）
g = df.groupby(["pattern", "cseed", "regime"]).agg(H=("H", "first"), eta=("eta", "mean"), dW=("dW", "mean"), pct=("pct", "mean"),
                                                   near1=("near1", "mean"), bal=("bal", "mean"), worse=("worse", "mean"),
                                                   minS=("minS", "mean"), minS_F=("minS_F", "first"), maxmin=("maxmin", "first"),
                                                   minU=("minU", "mean"), n_moves=("n_moves", "mean"), n_fail=("n_fail", "mean"),
                                                   ham=("ham", "mean"), WF=("WF", "first"), Wstar=("Wstar", "first"),
                                                   ndist=("cfg", "nunique"), n_near1=("n_near1", "first")).reset_index()
g.to_csv(OUT + "by_composition.csv", index=False)


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    bs = np.random.default_rng(0).choice(x, (2000, len(x))).mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


summ = []
for p in PATS:
    for r in REGS:
        s = g[(g.pattern == p) & (g.regime == r)]
        row = dict(pattern=p, regime=r)
        for col in ["eta", "dW", "near1", "bal", "worse", "minS", "n_moves", "n_fail", "ham", "ndist"]:
            m, lo, hi = ci(s[col]); row[col] = m; row[col + "_lo"] = lo; row[col + "_hi"] = hi
        row["dminS"] = float((s["minS"] - s["minS_F"]).mean())
        summ.append(row)
summ = pd.DataFrame(summ); summ.to_csv(OUT + "summary.csv", index=False)

# ---------------------------------------------------------------- 図 1：決め方 × 住み手の構成
fig = plt.figure(figsize=(18, 11), dpi=150)
fig.text(0.012, 0.985, "探索 E0：住み手の構成 × 決め方で、全知の最適 W* にどこまで近づき、誰が損をするか（3×3、7 パターン × 40 構成 × 5 局、30 年）",
         fontsize=13.5, fontweight="bold", va="top")


def heat(ax, col, title, fmt, cmap, vmin, vmax, note=""):
    M = np.array([[summ[(summ.pattern == p) & (summ.regime == r)][col].values[0] for r in REGS] for p in PATS])
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for a in range(len(PATS)):
        for b in range(len(REGS)):
            v = M[a, b]
            if np.isnan(v):
                t = "—"
            else:
                t = fmt.format(v)
            ax.text(b, a, t, ha="center", va="center", fontsize=8.6, color="white" if (not np.isnan(v) and (v - vmin) / (vmax - vmin + 1e-12) > 0.62) else INK)
    ax.set_xticks(range(len(REGS))); ax.set_xticklabels(REGS, fontsize=9.5)
    ax.set_yticks(range(len(PATS))); ax.set_yticklabels(PATS, fontsize=9.5)
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    if note:
        ax.text(0, -0.13, note, transform=ax.transAxes, fontsize=8, color=GREY, va="top")
    for s in ax.spines.values():
        s.set_visible(False)
    return im


ax1 = fig.add_axes([0.06, 0.53, 0.26, 0.38])
heat(ax1, "eta", "(a) 効率 η（F＝0、全知の最適＝1）", "{:.2f}", "RdBu", -1.0, 1.0, "同質では W* ＝ W_F のため定義しない")
ax2 = fig.add_axes([0.39, 0.53, 0.26, 0.38])
heat(ax2, "worse", "(b) F より損をした世帯の割合（持ち点込み）", "{:.2f}", "Reds", 0, 0.6)
ax3 = fig.add_axes([0.72, 0.53, 0.26, 0.38])
heat(ax3, "bal", "(c) 帳簿の残高 Σ|b|／9（㎡ 換算、0＝ゼロサムで平等）", "{:.1f}", "Purples", 0, 15)

# (d) η と H
ax4 = fig.add_axes([0.06, 0.07, 0.40, 0.36])
gg = g[g.pattern.str.startswith("α=")]
for r in [x for x in ["U", "V½", "V¾", "T0", "T1", "T1+"] if x in REGS]:
    s = gg[gg.regime == r]
    ax4.scatter(s.H, s.dW, s=10, color=RC[r], alpha=0.55, label=r)
    # 区間ごとの中央値
    bins = np.quantile(gg.H.unique(), np.linspace(0, 1, 9)); mid = (bins[1:] + bins[:-1]) / 2
    med = [s[(s.H >= bins[q]) & (s.H <= bins[q + 1])].dW.median() for q in range(8)]
    ax4.plot(mid, med, color=RC[r], lw=2)
ws = g[g.regime == "F"]
s_star = g.groupby(["pattern", "cseed"]).agg(H=("H", "first"), d=("Wstar", "first"), WF=("WF", "first")).reset_index()
s_star = s_star[s_star.pattern.str.startswith("α=")]
ax4.scatter(s_star.H, s_star.d - s_star.WF, s=14, marker="_", color="#6b4c9a", label="W*（全知の最適）")
ax4.axhline(0, color=INK, lw=0.8)
ax4.set_xlabel("住み手の性向のばらつき H"); ax4.set_ylabel("ΔW（F に対する満足の合計の差）")
ax4.set_ylim(min(-1.0, gg.dW.min() - 0.05), max(1.2, (s_star.d - s_star.WF).max() + 0.05))
ax4.legend(fontsize=8.5, frameon=False, ncol=3, loc="lower left")
for s_ in ("top", "right"):
    ax4.spines[s_].set_visible(False)
ax4.set_title("(d) ばらつきと、固定方式に対する差（ディリクレの 3 パターン 120 構成。線＝区間の中央値）", fontsize=11, loc="left", fontweight="bold")

# (e) 打たれた手の内訳
ax5 = fig.add_axes([0.53, 0.07, 0.40, 0.33])
types_order = ["張り出す3F", "張り出す2F", "外す2F", "外す1F", "収納", "開口"]
tc = {"張り出す3F": "#c8462d", "張り出す2F": "#e58a6f", "外す2F": "#9fc3e6", "外す1F": "#5b8fc7", "収納": "#4f7a2f", "開口": "#6b4c9a"}
agg = {r: {t: 0.0 for t in types_order} for r in REGS}
for r in REGS:
    s = df[(df.regime == r) & (df.pattern != "同質")]
    for js in s.types:
        for t, n in json.loads(js).items():
            if t in agg[r]:
                agg[r][t] += n
    for t in types_order:
        agg[r][t] /= max(len(s), 1)
left = np.zeros(len(REGS))
for t in types_order:
    v = np.array([agg[r][t] for r in REGS])
    ax5.barh(range(len(REGS)), v, left=left, color=tc[t], label=t, height=0.62)
    left += v
fails = [df[(df.regime == r) & (df.pattern != "同質")].n_fail.mean() for r in REGS]
for y, (tot, fl) in enumerate(zip(left, fails)):
    ax5.text(tot + 0.6, y, f"{tot:.1f} 手（却下 {fl:.1f}）", va="center", fontsize=8.6)
ax5.set_yticks(range(len(REGS))); ax5.set_yticklabels(REGS); ax5.invert_yaxis()
ax5.set_xlabel("30 年に実行された手の数（一局あたり、同質を除く平均）")
ax5.legend(fontsize=8.5, frameon=False, ncol=6, loc="lower left", bbox_to_anchor=(0.0, 1.0))
ax5.set_xlim(0, max(left) * 1.25)
for s_ in ("top", "right"):
    ax5.spines[s_].set_visible(False)
ax5.set_title("(e) 打たれた手の内訳", fontsize=11, loc="left", fontweight="bold", pad=24)
fig.savefig(FIG + "fig217_E0_決め方と住み手の構成.png", dpi=150, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 図 2：代表の構成での分布と到達点
shows = sorted(glob.glob(OUT + "show_*.json"))
if shows:
    fig = plt.figure(figsize=(18, 5.6 * len(shows) + 1.0), dpi=130)
    fig.text(0.012, 0.995, "全配置（23,887,872 通り）の分布の中で、各決め方の 100 局がたどり着いた所（順序の乱数だけを変えた 100 局）",
             fontsize=13.5, fontweight="bold", va="top")
    H = 1.0 / len(shows)
    for si, path in enumerate(shows):
        D = json.load(open(path, encoding="utf-8"))
        top = 1 - 0.03 - si * (0.97 * H)
        # 左：W の分布
        a = fig.add_axes([0.05, top - 0.97 * H + 0.06, 0.40, 0.97 * H - 0.11])
        x1 = np.array(D["x1"]); h1 = np.array(D["h1"])
        a.bar((x1[1:] + x1[:-1]) / 2, h1, width=np.diff(x1), color=LG, edgecolor="none")
        a.set_yscale("log"); a.set_ylim(0.8, h1.max() * 30)
        ymax = h1.max() * 8
        for yi, r in enumerate([x for x in REGS if x in D["runs"]]):
            ws = np.array([x["W"] for x in D["runs"][r]])
            yy = ymax / (2.2 ** yi)
            a.scatter(ws, np.full(len(ws), yy) * np.exp(np.random.default_rng(yi).uniform(-0.25, 0.25, len(ws))), s=8, color=RC[r], alpha=0.6)
            a.text(x1[0], yy, r, fontsize=8.5, color=RC[r], va="center", ha="left", fontweight="bold")
        a.axvline(D["WF"], color=GREY, ls="--", lw=1); a.axvline(D["wstar"], color="#6b4c9a", ls="--", lw=1)
        a.text(D["WF"], 1.2, " F", color=GREY, fontsize=9); a.text(D["wstar"], 1.2, " W*", color="#6b4c9a", fontsize=9, ha="right")
        a.set_xlabel("満足の合計 W"); a.set_ylabel("配置の数（対数）")
        a.set_title(f"({'abcd'[si]}) {D['pattern']}（構成 {D['cseed']}）：W の分布と到達点", fontsize=11, loc="left", fontweight="bold")
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
        # 中：W × 最小の満足
        b = fig.add_axes([0.50, top - 0.97 * H + 0.06, 0.20, 0.97 * H - 0.11])
        H2 = np.array(D["H2"]); xe = np.array(D["xe"]); ye = np.array(D["ye"])
        b.imshow(np.log1p(H2.T), origin="lower", extent=(xe[0], xe[-1], ye[0], ye[-1]), aspect="auto", cmap="Greys", alpha=0.55)
        fb = np.array(D["fbins"]); fr = np.array(D["front"]); ok_ = np.isfinite(fr)
        b.plot(((fb[1:] + fb[:-1]) / 2)[ok_], fr[ok_], color="#6b4c9a", lw=1.2, label="境界（W ごとの最小の満足の最大）")
        for r in [x for x in REGS if x in D["runs"]]:
            ws = np.array([x["W"] for x in D["runs"][r]]); ms = np.array([x["minS"] for x in D["runs"][r]])
            b.scatter(ws, ms, s=10, color=RC[r], alpha=0.6, label=r)
        b.scatter([D["WF"]], [D["minSF"]], marker="x", color=GREY, s=50)
        b.set_xlabel("満足の合計 W"); b.set_ylabel("最も不遇な世帯の満足 min S")
        lo = min(np.percentile(xe, 50), D["WF"] - 0.6)
        b.set_xlim(lo, D["wstar"] + 0.08)
        b.set_title("満足の合計と、最も不遇な世帯", fontsize=10.5, loc="left", fontweight="bold")
        if si == 0:
            b.legend(fontsize=7.2, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), ncol=1)
        for s_ in ("top", "right"):
            b.spines[s_].set_visible(False)
        # 右：最終の形（W* と、T1・U の最頻の形）
        c = fig.add_axes([0.765, top - 0.97 * H + 0.04, 0.23, 0.97 * H - 0.09]); c.set_xlim(0, 30); c.set_ylim(0, 12); c.axis("off")

        def board(x0, y0, e, k, o, title, W=None):
            cw, ch = 2.6, 2.4
            for i in range(9):
                f = s0.FLOOR[i]; col = s0.COL[i]
                x = x0 + col * cw; y = y0 + (f - 1) * ch
                c.add_patch(Rectangle((x, y), cw, ch, fc="#fff8e8", ec=INK, lw=0.7))
                if W is not None:
                    wi = np.array(W[i]); dom = int(np.argmax(wi))
                    colr = ["#4f7a2f", "#e0a526", "#5b7fb5", "#c8462d"][dom]
                    c.add_patch(Rectangle((x + 0.15, y + ch - 0.55), 0.4, 0.4, fc=colr, ec="none"))
                if e[i]:
                    c.add_patch(Rectangle((x + 0.3, y), cw - 0.6, 0.35, fc="#c8462d"))
            for bb in range(6):
                l, rr = s0.b_units(bb)
                f = s0.FLOOR[l]; col = s0.COL[l]
                x = x0 + (col + 1) * cw; y = y0 + (f - 1) * ch
                if k[bb] != 0:
                    c.add_patch(Rectangle((x - 0.18 + (-0.25 if k[bb] < 0 else 0.25) * 0, y + 0.5), 0.36, 1.2, fc="#4f7a2f"))
                    c.text(x + (0.45 if k[bb] > 0 else -0.45), y + 1.1, "←" if k[bb] > 0 else "→", fontsize=7, ha="center", va="center", color="#4f7a2f")
                if o[bb]:
                    c.plot([x - 0.35, x + 0.35], [y + 1.9, y + 1.9], color="#6b4c9a", lw=2.2)
            c.text(x0, y0 - 0.8, title, fontsize=8.5, va="top")

        from collections import Counter
        board(0.5, 4.0, *D["star"], "W*\n全知の最適", D["W"])
        for xi, r in enumerate(["T1+" if "T1+" in D["runs"] else "T1", "U"]):
            cnt = Counter((tuple(x["e"]), tuple(x["k"]), tuple(x["o"])) for x in D["runs"][r])
            (cfg, n) = cnt.most_common(1)[0]
            board(10.5 + 10 * xi, 4.0, np.array(cfg[0]), np.array(cfg[1]), np.array(cfg[2]), f"{r} の最頻の形\n{n}/100 局、全 {len(cnt)} 種", D["W"])
        c.text(0.5, 12.0, "左上の色＝その世帯が一番重く見る量（緑 広さ・黄 日照・青 静けさ・赤 近さ）", fontsize=7.5, color=GREY, va="top")
    fig.savefig(FIG + "fig218_E0_全配置の分布と到達点.png", dpi=130, facecolor="white")
    plt.close(fig)

# ---------------------------------------------------------------- 図 3：時間（適応の回数ごと）
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), dpi=150)
for ax, p in zip(axs, ["α=30", "α=0.5", "二群"]):
    s = df[df.pattern == p]
    for r in REGS:
        t = s[s.regime == r]
        M = np.array([json.loads(x) for x in t.Wt]) - t.WF.values[:, None]
        med = np.median(M, 0); lo = np.percentile(M, 25, 0); hi = np.percentile(M, 75, 0)
        ax.plot(range(M.shape[1]), med, color=RC[r], lw=2, label=r)
        ax.fill_between(range(M.shape[1]), lo, hi, color=RC[r], alpha=0.12, lw=0)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_title(f"{p}：F に対する差 ΔW の推移（中央値と四分位）", fontsize=10.5, loc="left", fontweight="bold")
    ax.set_xlabel("年"); ax.set_ylabel("ΔW")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
axs[0].legend(fontsize=8.5, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(FIG + "fig219_E0_適応の推移.png", dpi=150, facecolor="white")
plt.close(fig)
print(summ[["pattern", "regime", "eta", "worse", "bal", "dminS", "n_moves", "n_fail", "near1", "ndist"]].round(3).to_string())
