"""探索 E1 の集計と図。E0 の同じ構成（cseed 0〜19）・同じ乱数（seed 0〜2）を基準に、前提を崩した条件と対にして比べる。
python3 -m experiments.figs_e1
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiments.style import (SURF, INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, GREY, FAMILY, DESC, hair, ink_on, SEQ_B)
from matplotlib.colors import LinearSegmentedColormap

OUT = "out/e1/"; FIG = "out/figs/"
e1 = pd.read_csv(OUT + "runs.csv")
e0 = pd.concat([pd.read_csv("out/e0/runs.csv"), pd.read_csv("out/e0/runs_plus.csv")], ignore_index=True)
e0 = e0[(e0.cseed < 20) & (e0.seed < 3) & (e0.pattern != "同質")]

COND_ORDER = ["A_逓減", "A_下限", "B_盛り1.2", "B_盛り1.5", "C_下限20", "C_下限5", "C_半数払えない", "D_入替3%", "D_入替10%", "E_白紙から", "G_換算0.5", "G_換算2"]
COND_DESC = {"A_逓減": "満足の関数を逓減（平方根）に", "A_下限": "日照に下限（2 h 未満で急に下がる）", "B_盛り1.2": "損を 1.2 倍に言う", "B_盛り1.5": "損を 1.5 倍に言う",
             "C_下限20": "持ち点の下限 20 ㎡ 換算", "C_下限5": "持ち点の下限 5 ㎡ 換算", "C_半数払えない": "4 戸が払えない（下限 0）",
             "D_入替3%": "毎年 3 % が入れ替わる", "D_入替10%": "毎年 10 % が入れ替わる", "E_白紙から": "F ではなく何もない状態から",
             "G_換算0.5": "共通の物差しで、日照などを 0.5 倍に見る", "G_換算2": "共通の物差しで、日照などを 2 倍に見る"}
GROUP = {"A": "満足の関数の形（線形だから届いたのか）", "B": "駆け引き（言い値は正直か）", "C": "払う力（持ち点は無制限か）",
         "D": "入れ替わり（住み手は変わらないか）", "E": "始まりの状態", "G": "設計者の物差しの誤り"}
REGS = ["F", "U", "V½", "V¾", "T0", "T0+", "T1", "T1+"]

rows = []
for cond in COND_ORDER:
    c = e1[e1.cond == cond]
    if c.empty:
        continue
    pats = c.pattern.unique()
    for r in [x for x in REGS if x in set(c.regime)]:
        a = c[c.regime == r]
        b = e0[(e0.regime == r) & (e0.pattern.isin(pats))]
        m = a.merge(b, on=["pattern", "cseed", "seed"], suffixes=("", "_0"))
        mm = m.dropna(subset=["eta", "eta_0"])      # η が両方で定義できる対だけ（F がすでに最適な構成は除く）
        rows.append(dict(cond=cond, regime=r, n=len(m), n_eta=len(mm),
                         eta=mm.eta.mean(), eta0=mm.eta_0.mean(), d_eta=(mm.eta - mm.eta_0).mean(),
                         reach=(mm.eta > 0.999).mean(), reach0=(mm.eta_0 > 0.999).mean(),
                         worse=m.worse.mean(), worse0=m.worse_0.mean(),
                         bal=m.bal.mean(), bal0=m.bal_0.mean(), n_fail=m.n_fail.mean(), exit_bal=m.exit_bal.mean()))
S = pd.DataFrame(rows)
S.to_csv(OUT + "summary_vs_e0.csv", index=False)
print(S.round(3).to_string())

# ---------------------------------------------------------------- 図：前提を崩す前（E0）→ 崩した後（E1）
conds = [c for c in COND_ORDER if c in set(S.cond)]
ncol = 4; nrow = int(np.ceil(len(conds) / ncol))
fig = plt.figure(figsize=(18, 3.3 * nrow + 1.8))
fig.text(0.012, 0.995, "図 4　前提を一つずつ崩すと、効率 η はどう変わるか（E0 の同じ構成・同じ乱数と対にして比べた）",
         fontsize=14, fontweight="bold", va="top", color=INK)
fig.text(0.012, 0.975 - 0.002 * nrow, "各段：1 行＝1 つの決め方。白抜きの丸＝E0（線形・正直・払える・入れ替わりなし・F から）、色の丸＝その前提を崩した後、間の線＝変化。"
         "横軸は効率 η（0＝F と同じ、1＝全知の最適）。色は償い方の系統（灰 基準・橙 償わない・水色 共通の物差し・青 言い値）",
         fontsize=9, color=INK2, va="top")
H = 0.86 / nrow
for ci, cond in enumerate(conds):
    rr, cc = divmod(ci, ncol)
    ax = fig.add_axes([0.07 + cc * 0.24, 0.93 - 0.02 - (rr + 1) * H + 0.06 * H, 0.165, H * 0.70])
    d = S[S.cond == cond].set_index("regime")
    regs = [r for r in REGS if r in d.index]
    for y, r in enumerate(regs):
        col = FAMILY[r][1]
        x0, x1 = d.loc[r, "eta0"], d.loc[r, "eta"]
        ax.plot([x0, x1], [y, y], color=MUTED, lw=1.2, zorder=1)
        ax.scatter([x0], [y], s=46, facecolor=SURF, edgecolor=INK2, lw=1.2, zorder=2)
        ax.scatter([x1], [y], s=50, color=col, edgecolor=SURF, lw=1.5, zorder=3)
        ax.text(1.32, y, f"{x1:+.2f}", va="center", ha="right", fontsize=8.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8); ax.axvline(1, color=INK2, lw=0.8)
    ax.set_yticks(range(len(regs))); ax.set_yticklabels(regs, fontsize=8.5, color=INK2); ax.invert_yaxis()
    ax.set_xlim(-0.75, 1.15); ax.set_xticks([-0.5, 0, 0.5, 1.0]); ax.tick_params(axis="x", labelsize=8)
    hair(ax, "x")
    grp = GROUP[cond[0]]
    ax.set_title(f"{cond.split('_')[1]}：{COND_DESC[cond]}", fontsize=9.5, loc="left", pad=6)
    ax.text(0, 1.20, grp, transform=ax.transAxes, fontsize=8.3, color=MUTED, va="bottom")
    if rr == nrow - 1:
        ax.set_xlabel("効率 η", fontsize=9)
handles = [Line2D([0], [0], marker="o", ls="", markerfacecolor=SURF, markeredgecolor=INK2, markersize=7, label="E0（前提どおり）"),
           Line2D([0], [0], marker="o", ls="", color=BLUE, markersize=7, label="前提を崩した後（色は系統）")]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.995), frameon=False, fontsize=9, ncol=2)
fig.savefig(FIG + "fig220_E1_前提を崩すと.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- 図：損をした世帯と、帳簿の残高（取引の系統だけ）
T = ["T0", "T0+", "T1", "T1+"]
fig, axs = plt.subplots(1, 2, figsize=(17, 6.6))
fig.subplots_adjust(left=0.18, right=0.97, top=0.74, bottom=0.04, wspace=0.08)
fig.text(0.012, 0.97, "図 5　前提を崩したとき、取引の決め方で損をする世帯が出るか、帳簿はどれだけ釣り合わなくなるか",
         fontsize=14, fontweight="bold", va="top", color=INK)
fig.text(0.012, 0.925, "行＝崩した前提、列＝取引の決め方。数字は 20 構成 × 3 局の平均。括弧内は E0（前提どおり）の値。", fontsize=9, color=INK2, va="top")
SEQ_O = LinearSegmentedColormap.from_list("seqo", ["#fbeee6", "#f3b08f", "#eb6834", "#a8401a"])
for ax, (col, col0, cmap, vmax, fmt, title) in zip(axs, [("worse", "worse0", SEQ_O, 0.4, "{:.2f}", "F より損をした世帯の割合"),
                                                          ("bal", "bal0", SEQ_B, 8, "{:.1f}", "帳簿の残高 Σ|b|／9（㎡ 換算）")]):
    M = np.full((len(conds), len(T)), np.nan); M0 = M.copy()
    for a, cnd in enumerate(conds):
        for b, r in enumerate(T):
            q = S[(S.cond == cnd) & (S.regime == r)]
            if len(q):
                M[a, b] = q[col].values[0]; M0[a, b] = q[col0].values[0]
    for a in range(len(conds)):
        for b in range(len(T)):
            v = M[a, b]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((b, a), 1, 1, fc=SURF, ec=SURF, lw=2))
                ax.text(b + 0.5, a + 0.5, "—", ha="center", va="center", fontsize=8.5, color=MUTED)
                continue
            rgb = cmap(min(v / vmax, 1))
            ax.add_patch(plt.Rectangle((b, a), 1, 1, fc=rgb, ec=SURF, lw=2))
            ax.text(b + 0.5, a + 0.5, f"{fmt.format(v)}（{fmt.format(M0[a, b])}）", ha="center", va="center", fontsize=8.5, color=ink_on(rgb))
    ax.set_xlim(0, len(T)); ax.set_ylim(len(conds), 0)
    ax.set_xticks(np.arange(len(T)) + 0.5); ax.set_xticklabels([f"{r}\n{DESC[r]}" for r in T], fontsize=8.5, color=INK2)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(conds)) + 0.5)
    ax.set_yticklabels([f"{c.split('_')[1]}：{COND_DESC[c]}" for c in conds] if ax is axs[0] else [], fontsize=8.5, color=INK2)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, pad=40)
fig.savefig(FIG + "fig221_E1_損と残高.png", dpi=140)
plt.close(fig)
print("figs ok")
