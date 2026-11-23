"""図 2・3：開発の段（E1 の前提崩し → 改良の部品）を、新しい記号でまとめる
python3 -m experiments.figs_s1_dev
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiments.style import SURF, INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, GREY, DEEP, hair

REN = {"T0": "TR", "T0+": "TR+", "T1": "TP", "T1+": "TP+"}
COL = {"TR": AQUA, "TR+": AQUA, "TP": BLUE, "TP+": BLUE}
MK = {"TR": "o", "TR+": "D", "TP": "o", "TP+": "D"}

# ---------------------------------------------------------------- 図 2：E1（前提を一つずつ崩す）
S = pd.read_csv("out/e1/summary_vs_e0.csv")
S = S[S.regime.isin(REN)].copy(); S["rule"] = S.regime.map(REN)
conds = [("前提どおり（E0）", None), ("A 満足が逓減", "A_逓減"), ("A 日照に下限", "A_下限"), ("B 損を 1.2 倍に言う", "B_盛り1.2"),
         ("B 損を 1.5 倍に言う", "B_盛り1.5"), ("C 借りの下限 20", "C_下限20"), ("C 借りの下限 5", "C_下限5"),
         ("C 4 戸が払えない", "C_半数払えない"), ("D 毎年 3 % 入れ替わる", "D_入替3%"), ("D 毎年 10 % 入れ替わる", "D_入替10%"),
         ("E 何もない状態から", "E_白紙から"), ("G 物差しを 0.5 倍に誤る", "G_換算0.5"), ("G 物差しを 2 倍に誤る", "G_換算2")]
base = S[S.cond == "A_下限"].set_index("rule").eta0   # E0 の値（下限の条件と同じ 120 構成で対にしたもの）
fig, axs = plt.subplots(2, 1, figsize=(7.2, 11.0))
fig.subplots_adjust(left=0.36, right=0.97, top=0.95, bottom=0.06, hspace=0.22)
for ax, (col, xlab, xlim) in zip(axs, [("eta", "到達度 η（0 ＝ F と同じ、1 ＝ 全知の最適）", (-0.05, 1.08)),
                                       ("worse", "取り残し率 L（F より損をした世帯の割合）", (-0.02, 0.42))]):
    for y, (lab, c) in enumerate(conds):
        for rule in ["TR", "TR+", "TP", "TP+"]:
            if c is None:
                q = S[(S.cond == "A_下限") & (S.rule == rule)]
                v = q.eta0.values[0] if col == "eta" else q.worse0.values[0]
            else:
                q = S[(S.cond == c) & (S.rule == rule)]
                if q.empty:
                    continue
                v = q[col].values[0]
            ax.scatter(v, y + {"TR": -0.18, "TR+": -0.06, "TP": 0.06, "TP+": 0.18}[rule], s=56, marker=MK[rule],
                       color=COL[rule], edgecolor=SURF, lw=0.8, zorder=3)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([l for l, _ in conds], fontsize=10, color=INK2)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylim(len(conds) - 0.5, -0.5); ax.set_xlim(*xlim); ax.set_xlabel(xlab, fontsize=10.5)
    ax.axhline(0.5, color=INK2, lw=0.8)
    for y in [2.5, 4.5, 7.5, 9.5, 10.5]:
        ax.axhline(y, color=GRID, lw=0.8)
    hair(ax, "x")
# 読みの注記
axs[0].annotate("払えない・入れ替わると\nTP+ も大きく下がる", xy=(0.64, 7.2), xytext=(0.05, 8.2), fontsize=10, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
axs[1].annotate("入れ替わると、取引に加わって\nいない新しい世帯が取り残される", xy=(0.35, 9.2), xytext=(0.13, 11.9), fontsize=10, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
handles = [Line2D([0], [0], marker=MK[r], ls="", color=COL[r], markersize=7, label=r) for r in ["TR", "TR+", "TP", "TP+"]]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.995), frameon=False, ncol=4, fontsize=10.5)
fig.savefig("out/figs/fig223_S1_開発E1_前提を崩す.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 図 3：改良の部品
d1 = pd.read_csv("out/s1dev/runs.csv"); d2 = pd.read_csv("out/s1dev/runs2.csv")
d = pd.concat([d2, d1], ignore_index=True).drop_duplicates(subset=["cond", "pattern", "cseed", "cand", "seed"], keep="first")
g = d.groupby(["cond", "cand", "pattern", "cseed"]).mean(numeric_only=True).reset_index()
NAME = {"TP+": "TP+", "TP+割": "TP+割", "TP+貸5": "TP+貸（D＝5）", "TP+貸10": "TP+貸（D＝10）", "TP+貸20": "TP+貸（D＝20）",
        "TP+割貸10": "TP+割貸（D＝10）", "TP+清": "TP+清", "TP+清再": "TP+清再（戻せる造作だけ）", "TP+清再全": "TP+清再（張り出しも）",
        "TP+割貸10清再全": "TP+割貸清再（D＝10）", "TP+貸20清再": "TP+貸清再（戻せる造作だけ）", "TP+貸20清再全": "TP+貸清再 ← 改良版"}
fig = plt.figure(figsize=(7.2, 11.2))
panels = [("(a) 4 戸が払えない（C）", "C_半数払えない", ["TP+", "TP+割", "TP+貸5", "TP+貸10", "TP+貸20", "TP+貸20清再全"], "eta", "到達度 η"),
          ("(b) 毎年 10 % 入れ替わる（D）", "D_入替10%", ["TP+", "TP+清", "TP+清再", "TP+清再全", "TP+貸20清再全"], "eta", "到達度 η"),
          ("(c) 毎年 10 % 入れ替わる（D）：誰が取り残されるか", "D_入替10%", ["TP+", "TP+清再全", "TP+貸20清再全"], "L", "取り残し率 L")]
for n, (title, cond, cands, col, xlab) in enumerate(panels):
    ax = fig.add_axes([0.40, [0.70, 0.37, 0.06][n], 0.50, [0.25, 0.22, 0.17][n]])
    for y, c in enumerate(cands):
        q = g[(g.cond == cond) & (g.cand == c)]
        colr = DEEP if c == "TP+貸20清再全" else BLUE
        if col == "L":
            a, b_ = q.L_new.mean(), q.L_stay.mean()
            ax.barh(y - 0.17, a, height=0.32, color=colr, alpha=0.9)
            ax.barh(y + 0.17, b_, height=0.32, color=colr, alpha=0.35)
            ax.text(a + 0.01, y - 0.17, f"{a:.2f}", va="center", fontsize=10, color=INK2)
            ax.text(b_ + 0.01, y + 0.17, f"{b_:.2f}", va="center", fontsize=10, color=INK2)
        else:
            v = q[col]; lo, hi = v.quantile(0.1), v.quantile(0.9)
            ax.plot([lo, hi], [y, y], color=colr, lw=2, alpha=0.45, solid_capstyle="round")
            ax.scatter(v.mean(), y, s=60, color=colr, edgecolor=SURF, lw=1, zorder=3)
            ax.text(1.17, y, f"{v.mean():.2f}", va="center", ha="left", fontsize=10, color=INK2)
    ax.set_yticks(range(len(cands))); ax.set_yticklabels([NAME[c] for c in cands], fontsize=10, color=INK2)
    ax.set_ylim(len(cands) - 0.5, -0.6); ax.set_title(title, fontsize=11.5, pad=8, x=-0.75, ha="left")
    ax.set_xlabel(xlab, fontsize=10.5); ax.tick_params(axis="x", labelsize=10)
    if col == "L":
        ax.set_xlim(0, 0.75)
        ax.text(1.0, -0.42, "濃い棒＝入れ替わった住戸の世帯　淡い棒＝30 年住み続けた世帯", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color=INK2)
    else:
        ax.set_xlim(-0.3, 1.12); ax.axvline(0, color=INK2, lw=0.8); ax.axvline(1, color=INK2, lw=0.8)
    hair(ax, "x")
fig.savefig("out/figs/fig224_S1_開発_改良の部品.png", dpi=170)
print("ok")
