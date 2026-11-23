"""第一段階・本走行の集計：事前登録 prereg_s1.json の H1〜H5 の判定と、図 4〜6
python3 -m experiments.analyze_s1
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from experiments.style import (SURF, INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, GREY, DEEP, SYM_FAM, hair, ink_on, SEQ_B, SEQ_O)
from matplotlib.colors import LinearSegmentedColormap

import os
OUT = os.environ.get("S1OUT", "out/s1/"); FIG = os.environ.get("S1FIG", "out/figs/")
PRE = json.load(open("experiments/prereg_s1.json", encoding="utf-8"))
IMP = PRE["improved"]["name"]
PATS = PRE["patterns"]; HET = [p for p in PATS if p != "同質"]
RULES = ["F", "U", "V50", "V75", "TR", "TR+", "TP", "TP+", IMP]
TPS = ["TP", "TP+", IMP]; UV = ["U", "V50", "V75"]
rng = np.random.default_rng(20261031)

M = pd.read_csv(OUT + "main.csv")
R = pd.read_csv(OUT + "robust.csv") if __import__("os").path.exists(OUT + "robust.csv") else None
PAT_DESC = {"同質": "全員が平均の性向", "α=300": "ばらつき ごく小", "α=100": "ばらつき 小", "α=30": "ばらつき やや小", "α=10": "ばらつき 中",
            "α=3": "ばらつき やや大", "α=1": "ばらつき 大", "α=0.3": "ばらつき ごく大", "二群": "広さ派 5・静けさ派 4",
            "二群7:3": "広さ派 6・静けさ派 3", "二群9:1": "広さ派 8・静けさ派 1", "日照派と近さ派": "日照派 5・近さ派 4",
            "少数の極端": "1 戸だけ近さ派", "少数の極端2": "2 戸だけ近さ派", "階の不一致": "1F が日照派", "階の一致": "3F が日照派"}


def comp_mean(df, col):
    """構成ごとに局で平均"""
    return df.groupby(["cond", "pattern", "cseed", "rule"])[col].mean().unstack("rule")


def boot(x, n=2000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.nan
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    sd = x.std(ddof=1)
    return x.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975), (x.mean() / sd if sd > 0 else np.inf * np.sign(x.mean()))


def judge(diff, sign=+1):
    """sign=+1：差が正であることを主張したい"""
    m, lo, hi, d = boot(diff)
    ok = (lo > 0) if sign > 0 else (hi < 0)
    size = "小さい" if abs(d) < 0.2 else ""
    return dict(mean=m, lo=lo, hi=hi, d=d, ok=bool(ok), small=size)


rows = []
Mh = M[M.pattern.isin(HET)]
# ---------------------------------------------------------------- H1
dW = comp_mean(Mh, "dW")
H = Mh.groupby(["cond", "pattern", "cseed"]).H.first()
for r in ["TP+", IMP]:
    j = judge(dW[r].values, +1)
    rho, p = spearmanr(H.loc[dW.index].values, dW[r].values)
    rows.append(dict(仮説="H1", 比べるもの=f"{r} の ΔΣS ＞ 0", **j, 補足=f"H と ΔΣS の順位相関 ρ＝{rho:.2f}（p＝{p:.1e}）", 判定=j["ok"] and rho > 0))
# ---------------------------------------------------------------- H2・H4
for h, col in [("H2", "B"), ("H4", "L")]:
    X = comp_mean(Mh, col)
    for a in TPS:
        for b in UV:
            j = judge(X[a].values - X[b].values, -1)
            rows.append(dict(仮説=h, 比べるもの=f"{col}：{a} − {b} ＜ 0", **j, 補足=f"{a} {X[a].mean():.3f}、{b} {X[b].mean():.3f}", 判定=j["ok"]))
# ---------------------------------------------------------------- 順位（記述）
eta = comp_mean(Mh, "eta")
order = eta.mean().sort_values(ascending=False)

# ---------------------------------------------------------------- H3・H5（前提を崩した条件）
if R is not None:
    Rh = R[R.pattern.isin(HET)]
    G = Rh.groupby(["cond", "pattern", "rule"])[["dW", "L", "eta"]].mean().reset_index()
    n_row = n_ok = 0
    for (cond, pat), q in G.groupby(["cond", "pattern"]):
        q = q.set_index("rule")
        if IMP not in q.index:
            continue
        n_row += 1
        ok1 = q.loc[IMP, "dW"] > 0
        ok4 = q.loc[IMP, "L"] <= min(q.loc["U", "L"], q.loc["V75", "L"]) + 1e-9
        n_ok += int(ok1 and ok4)
    rows.append(dict(仮説="H3", 比べるもの=f"{IMP}：ΔΣS ＞ 0 かつ L ≤ U・V75", mean=n_ok / n_row, lo=np.nan, hi=np.nan, d=np.nan, ok=n_ok / n_row >= 0.8,
                     small="", 補足=f"{n_ok}／{n_row} 行", 判定=n_ok / n_row >= 0.8))
    Re = comp_mean(Rh, "eta"); Rl = comp_mean(Rh, "L")
    for cond in ["C_半数払えない", "D_入替10%"]:
        sub = Re.xs(cond, level="cond")
        j = judge(sub[IMP].values - sub["TP+"].values, +1)
        rows.append(dict(仮説="H5", 比べるもの=f"{cond}：η の差 {IMP} − TP+ ＞ 0", **j, 補足=f"{IMP} {sub[IMP].mean():.3f}、TP+ {sub['TP+'].mean():.3f}", 判定=j["ok"]))
    sub = Rl.xs("D_入替10%", level="cond")
    j = judge(sub[IMP].values - sub["TP+"].values, -1)
    rows.append(dict(仮説="H5", 比べるもの=f"D_入替10%：L の差 {IMP} − TP+ ＜ 0", **j, 補足=f"{IMP} {sub[IMP].mean():.3f}、TP+ {sub['TP+'].mean():.3f}", 判定=j["ok"]))
    worst = G.groupby(["rule"]).apply(lambda q: q.eta.min())
    rows.append(dict(仮説="H5", 比べるもの=f"最悪の行の η：{IMP} ＞ TP+", mean=worst[IMP] - worst["TP+"], lo=np.nan, hi=np.nan, d=np.nan,
                     ok=worst[IMP] > worst["TP+"], small="", 補足=f"{IMP} {worst[IMP]:.3f}、TP+ {worst['TP+']:.3f}", 判定=worst[IMP] > worst["TP+"]))
T = pd.DataFrame(rows)
T.to_csv(OUT + "hypotheses.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
print(T[["仮説", "比べるもの", "mean", "lo", "hi", "d", "small", "判定", "補足"]].round(3).to_string())
print("\n到達度の順位（主条件・ばらつきのある 15 パターン）\n", order.round(3).to_string())

# ---------------------------------------------------------------- 表：主条件の要約（パターン × 決め方）
summ = {}
for col in ["eta", "L", "B"]:
    summ[col] = M.groupby(["pattern", "rule"])[col].mean().unstack("rule").reindex(PATS)[RULES]
    summ[col].to_csv(OUT + f"main_{col}.csv")
allrow = {col: comp_mean(Mh, col).mean() for col in ["eta", "L", "B", "dW"]}
pd.DataFrame(allrow).reindex(RULES).to_csv(OUT + "main_overall.csv")
print("\n全体（ばらつきのある 15 パターン）\n", pd.DataFrame(allrow).reindex(RULES).round(3).to_string())
# 形の多様さ：同じ構成の 10 局のあいだの、最終の形の種類の数
div = M.groupby(["pattern", "cseed", "rule"]).cfg.nunique().groupby("rule").mean().reindex(RULES)
div.to_csv(OUT + "main_diversity.csv"); print("\n同じ構成の 10 局での形の種類（平均）\n", div.round(2).to_string())

# ---------------------------------------------------------------- 図 4：主条件のパターン × 決め方
LBL = {r: r for r in RULES}; LBL[IMP] = "改良版\nTP+貸清再"
fig = plt.figure(figsize=(7.2, 19.5))
DIVE = LinearSegmentedColormap.from_list("dive", ["#a32d2d", "#e34948", "#f6dcdb", "#f0efec", "#cee0f5", "#186cce", "#24438f"])
specs = [("eta", "(a) 到達度 η（青＝最適に近い、赤＝F より悪い）", DIVE, -1, 1, "{:.2f}"),
         ("L", "(b) 取り残し率 L（橙が濃いほど多い）", SEQ_O, 0, 0.5, "{:.2f}"),
         ("B", "(c) 残高の偏り B（㎡ 換算、青が濃いほど大きい）", SEQ_B, 0, 8, "{:.1f}")]
for n, (col, title, cmap, vmin, vmax, fmt) in enumerate(specs):
    ax = fig.add_axes([0.385, 0.685 - n * 0.33, 0.605, 0.255])
    A = summ[col].values
    for a in range(A.shape[0]):
        for b in range(A.shape[1]):
            v = A[a, b]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((b, a), 1, 1, fc=SURF, ec=SURF, lw=1.5))
                ax.text(b + 0.5, a + 0.5, "—", ha="center", va="center", fontsize=9.5, color=MUTED)
                continue
            rgb = cmap(np.clip((v - vmin) / (vmax - vmin), 0, 1))
            ax.add_patch(plt.Rectangle((b, a), 1, 1, fc=rgb, ec=SURF, lw=1.5))
            ax.text(b + 0.5, a + 0.5, fmt.format(v), ha="center", va="center", fontsize=9.5, color=ink_on(rgb))
    ax.set_xlim(0, len(RULES)); ax.set_ylim(len(PATS), 0)
    ax.set_xticks(np.arange(len(RULES)) + 0.5); ax.set_xticklabels([LBL[r].replace("改良版\nTP+貸清再", "改良\n版") for r in RULES], fontsize=10, color=INK2, rotation=0)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(PATS)) + 0.5)
    ax.set_yticklabels([f"{p}（{PAT_DESC[p]}）" for p in PATS], fontsize=10, color=INK2)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=11.5, pad=36, x=-0.62, ha="left")
fig.savefig(FIG + "fig225_S1_本走行_主条件.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 図 5：ばらつき H と F との差 ΔΣS（H1）
fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.0), sharey=True, sharex=True)
fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.09, wspace=0.08, hspace=0.25)
axs = axs.ravel()
Hh = Mh.groupby(["pattern", "cseed"]).H.first()
dWa = Mh.groupby(["pattern", "cseed", "rule"]).dW.mean().unstack("rule")
dWs = (Mh.groupby(["pattern", "cseed"]).Wstar.mean() - Mh.groupby(["pattern", "cseed"]).WF.mean())
bins = np.linspace(0, Hh.max() + 1e-9, 9)
for ax, r in zip(axs, ["U", "V75", "TP+", IMP]):
    x = Hh.loc[dWa.index].values; y = dWa[r].values
    idx = np.digitize(x, bins)
    xs, lo, hi = [], [], []
    for k in range(1, len(bins)):
        sel = idx == k
        if sel.sum() > 3:
            xs.append(x[sel].mean()); lo.append(np.quantile(dWs.loc[dWa.index].values[sel], 0.1)); hi.append(np.quantile(dWs.loc[dWa.index].values[sel], 0.9))
    ax.fill_between(xs, lo, hi, color=GREY, alpha=0.18, lw=0)
    ax.scatter(x, y, s=8, color=SYM_FAM[r], alpha=0.5, lw=0)
    ax.axhline(0, color=INK2, lw=0.8)
    rho, _ = spearmanr(x, y)
    ax.set_title(f"{'改良版' if r == IMP else r}", fontsize=11.5)
    ax.text(0.03, 0.97, f"順位相関 ρ ＝ {rho:.2f}", transform=ax.transAxes, fontsize=10, color=INK2, va="top")
    ax.tick_params(labelsize=10); hair(ax, "y")
    if ax in (axs[2], axs[3]):
        ax.set_xlabel("ばらつき H", fontsize=10.5)
    if ax in (axs[0], axs[2]):
        ax.set_ylabel("ΔΣS（F との差）", fontsize=10.5)
fig.savefig(FIG + "fig226_S1_本走行_ばらつきと差.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 図 6：前提を崩した条件（H3・H5）
if R is not None:
    CONDS = ["主条件", "A_逓減", "A_下限", "B_盛り1.5", "C_半数払えない", "D_入替10%", "E_白紙から", "G_換算2"]
    CDESC = {"主条件": "主条件（崩す前）", "A_逓減": "A 満足が逓減", "A_下限": "A 日照に下限", "B_盛り1.5": "B 損を 1.5 倍に言う",
             "C_半数払えない": "C 4 戸が払えない", "D_入替10%": "D 毎年 10 % 入れ替わる", "E_白紙から": "E 何もない状態から", "G_換算2": "G 物差しを 2 倍に誤る"}
    RR = ["U", "V75", "TR+", "TP+", IMP]
    Mm = M[(M.cseed < 110) & M.rule.isin(RR) & (M.seed < 5)]
    both = pd.concat([Mm, R[R.rule.isin(RR)]], ignore_index=True)
    both = both[both.pattern.isin(HET)]
    C = both.groupby(["cond", "pattern", "cseed", "rule"])[["eta", "L"]].mean().reset_index()
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 11.5))
    fig.subplots_adjust(left=0.33, right=0.97, top=0.94, bottom=0.06, hspace=0.2)
    off = {"U": -0.28, "V75": -0.14, "TR+": 0.0, "TP+": 0.14, IMP: 0.28}
    for ax, (col, xlab, xlim) in zip(axs, [("eta", "到達度 η", (-0.6, 1.08)), ("L", "取り残し率 L", (-0.02, 0.6))]):
        for y, cond in enumerate(CONDS):
            for r in RR:
                v = C[(C.cond == cond) & (C.rule == r)][col]
                if v.empty:
                    continue
                ax.plot([v.quantile(0.1), v.quantile(0.9)], [y + off[r]] * 2, color=SYM_FAM[r], lw=1.6, alpha=0.4)
                ax.scatter(v.mean(), y + off[r], s=34, color=SYM_FAM[r], edgecolor=SURF, lw=0.7, zorder=3,
                           marker={"V75": "s", IMP: "D"}.get(r, "o"))
        ax.set_yticks(range(len(CONDS))); ax.set_yticklabels([CDESC[c] for c in CONDS], fontsize=10, color=INK2)
        ax.tick_params(axis="x", labelsize=10)
        ax.set_ylim(len(CONDS) - 0.5, -0.5); ax.set_xlim(*xlim); ax.set_xlabel(xlab, fontsize=10.5)
        ax.axhline(0.5, color=INK2, lw=0.8)
        for y in range(1, len(CONDS)):
            ax.axhline(y + 0.5, color=GRID, lw=0.6)
        hair(ax, "x")
    handles = [Line2D([0], [0], marker={"V75": "s", IMP: "D"}.get(r, "o"), ls="", color=SYM_FAM[r], markersize=7,
                      label="改良版" if r == IMP else r) for r in RR]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.995), frameon=False, ncol=5, fontsize=10.5)
    fig.savefig(FIG + "fig227_S1_本走行_前提を崩す.png", dpi=170)
    plt.close(fig)
    rob = C.groupby(["cond", "rule"])[["eta", "L"]].mean().unstack("rule")
    rob.to_csv(OUT + "robust_summary.csv"); print("\n前提を崩した条件\n", rob.round(3).to_string())
print("figs ok")
