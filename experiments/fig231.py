"""図 231〜234：段 1 第二報（賃貸の制約）。幅 7.2 in・文字 10 pt 以上、パネルは縦に積む。
PYTHONPATH=. python3 -m experiments.fig231 [main|dev]
  231 入れ替わり：ν ごとの η・V*n・L
  232 原状回復：R の型 × ν の η・V*n・L（費用と見通しあり）
  233 所有と決める場：借家／所有 × G × ν の η・V*n
  234 所有者の型：η・V*n（ν＝0.10・0.30）
"""
import sys, json
import numpy as np
import matplotlib.pyplot as plt
from experiments.style import INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, GREY, DEEP, hair

MODE = sys.argv[1] if len(sys.argv) > 1 else "main"
D = json.load(open(f"out/s3/{MODE}_cells.json")); C = D["cells"]; EPS = D["eps"]
DEEPB = "#24438f"


def get(g, c, rg, col):
    r = C.get(f"{g}|{c}|{rg}")
    if r is None:
        return np.nan
    if col == "Vn":
        return np.mean([r["Vn_α=3"][0], r["Vn_二群"][0]])
    return r[col]


def vn_rng(g, c, rg):
    r = C[f"{g}|{c}|{rg}"]
    lo = np.mean([r["Vn_α=3"][1], r["Vn_二群"][1]]); hi = np.mean([r["Vn_α=3"][2], r["Vn_二群"][2]])
    return lo, hi


def style(ax, ylabel, xlabel=None):
    ax.set_ylabel(ylabel, fontsize=10.5); ax.tick_params(labelsize=10); hair(ax)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5)


# ---------------------------------------------------------------- 231 入れ替わり
NUS = [0.0, 0.05, 0.10, 0.18, 0.30]
REG = [("U", ORANGE, "o", "-", "U（単独）"), ("V50", ORANGE, "s", "--", "V50（投票）"), ("TP+", BLUE, "o", "--", "TP+ 全員に聞く"),
       ("TP+top3", BLUE, "D", "-", "TP+ 上位 3 手"), ("改良版", DEEPB, "o", "--", "改良版 全員に聞く"), ("改良版top3", DEEPB, "D", "-", "改良版 上位 3 手")]
fig, axs = plt.subplots(3, 1, figsize=(7.2, 12.5), sharex=True)
for ax, col, lab in zip(axs, ["eta", "Vn", "L"], ["到達度 η", "$V^{*}_{n}$（2 パターンの平均）", "取り残し率 L"]):
    for rg, c, mk, ls, name in REG:
        y = [get("T", f"幾何 ν={x:.2f}", rg, col) for x in NUS]
        ax.plot(NUS, y, color=c, marker=mk, ls=ls, lw=1.6, ms=6, label=name)
    if col == "Vn":
        for rg, mk in [("N3", "x"), ("BRdyn", "+"), ("N4", "1")]:
            ax.plot(NUS, [get("T", f"幾何 ν={x:.2f}", rg, "Vn") for x in NUS], color=GREY, marker=mk, ls=":", lw=1, ms=8,
                    label={"N3": "N3 全知が組み替え続ける", "BRdyn": "偽装：入居のたびに仲介", "N4": "N4 ランダム"}[rg])
        ax.axhline(EPS, color="#a32d2d", lw=1, ls="--"); ax.text(0.0, EPS + 0.012, f"ε＝{EPS:.3f}", color="#a32d2d", fontsize=10, ha="left")
    for rg, c in [("TP+top3", BLUE), ("改良版top3", DEEPB)]:
        ax.scatter([0.18], [get("T", "更新の山 ν=0.18", rg, col)], marker="^", s=70, color=c, edgecolor="white", zorder=4)
        ax.scatter([0.195], [get("T", "二型の混合", rg, col)], marker="v", s=70, color=c, edgecolor="white", zorder=4)
        ax.scatter([0.10, 0.30], [get("T", f"性向を引き継ぐ ν={x:.2f}", rg, col) for x in (0.10, 0.30)], marker="*", s=110, color=c,
                   edgecolor="white", zorder=4)
    style(ax, lab)
axs[0].set_title("(a) 到達度 η", fontsize=11.5, loc="left"); axs[1].set_title("(b) 話し合いの分散知 $V^{*}_{n}$", fontsize=11.5, loc="left")
axs[2].set_title("(c) 取り残し率 L（30 年目の世帯）", fontsize=11.5, loc="left")
axs[2].set_xlabel("年あたりの退去率 ν（幾何）", fontsize=10.5); axs[2].set_xticks(NUS)
axs[0].legend(fontsize=9.5, frameon=False, ncol=2, loc="lower left")
axs[1].legend(fontsize=9.5, frameon=False, ncol=1, loc="center right", handles=axs[1].get_legend_handles_labels()[0][6:],
              labels=axs[1].get_legend_handles_labels()[1][6:])
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.text(0.02, 0.008, "▲ 更新の山（平均 ν＝0.18）　▼ 二型の混合　★ 性向を引き継ぐ対照（ν＝0.10・0.30）", fontsize=10, color=INK2, va="bottom", ha="left")
fig.savefig("out/figs/fig231_S1b_入れ替わり.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 232 原状回復
NU4 = [0.05, 0.10, 0.18, 0.30]
RT = [("R∅ 費用あり", GREY, "o", "-", "R∅（ゼロ精算）"), ("R0 費用あり", ORANGE, "s", "-", "R0（日本の既定）"),
      ("Rdk10 費用あり", AQUA, "D", "-", "Rdk10（10 年償却の補償）"), ("Rdk20 費用あり", "#156d59", "D", "--", "Rdk20"),
      ("R∅ 費用なし", GREY, "o", ":", "R∅・費用なし"), ("R0 費用なし", ORANGE, "s", ":", "R0・費用なし")]
fig, axs = plt.subplots(3, 2, figsize=(7.2, 11.5), sharex=True, sharey="row")
for j, rg in enumerate(["TP+top3", "改良版top3"]):
    for i, (col, lab) in enumerate([("eta", "到達度 η"), ("Vn", "$V^{*}_{n}$"), ("L", "取り残し率 L")]):
        ax = axs[i, j]
        for cond, c, mk, ls, name in RT:
            ax.plot(NU4, [get("R", f"{cond} ν={x:.2f}", rg, col) for x in NU4], color=c, marker=mk, ls=ls, lw=1.5, ms=5, label=name)
        if col == "Vn":
            ax.axhline(EPS, color="#a32d2d", lw=1, ls="--")
        style(ax, lab if j == 0 else "")
        if i == 0:
            ax.set_title({"TP+top3": "TP+ 上位 3 手", "改良版top3": "改良版 上位 3 手"}[rg], fontsize=11.5)
        if i == 2:
            ax.set_xlabel("退去率 ν", fontsize=10.5); ax.set_xticks(NU4)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig("out/figs/fig232_S1b_原状回復.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 233 所有と決める場
NU3 = [0.04, 0.18, 0.30]
OG = [("所有・G1", BLUE, "所有・寄り合い"), ("所有・G0", "#9bbfea", "所有・場なし"), ("借家・G1", DEEPB, "借家・寄り合い"),
      ("借家・G½", AQUA, "借家・44 条型"), ("借家・G0", ORANGE, "借家・場なし")]
fig, axs = plt.subplots(3, 1, figsize=(7.2, 11.0), sharex=True)
for ax, (col, rg, lab) in zip(axs, [("eta", "TP+top3", "η（TP+ 上位 3 手）"), ("eta", "改良版top3", "η（改良版 上位 3 手）"), ("Vn", "改良版top3", "$V^{*}_{n}$（改良版 上位 3 手）")]):
    w = 0.16
    for k, (cond, c, name) in enumerate(OG):
        ys = [get("O", f"{cond} ν={x:.2f}", rg, col) for x in NU3]
        ax.bar(np.arange(3) + (k - 2) * w, ys, w, color=c, label=name)
    if col == "Vn":
        ax.axhline(EPS, color="#a32d2d", lw=1, ls="--")
    style(ax, lab); ax.axhline(0, color=INK2, lw=0.8)
axs[2].set_xticks(range(3)); axs[2].set_xticklabels([f"ν＝{x:.2f}" for x in NU3], fontsize=10.5)
axs[0].legend(fontsize=10, frameon=False, ncol=3, loc="upper right")
axs[0].set_title("所有の形 × 決める場 × 退去率（費用と見通しあり。借家は個人の所有者・拒否権のみ・R0）", fontsize=10.5, loc="left")
fig.tight_layout()
fig.savefig("out/figs/fig233_S1b_所有と決める場.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 234 所有者の型
OW = ["なし", "個人・拒否", "個人・取引", "個人・D5", "管理∅・拒否", "管理∅・取引", "管理∅・D4", "管理∅・D5", "管理開口・拒否", "サブ・拒否", "サブ・取引", "サブ・D5"]
LAB = {"なし": "所有者なし", "個人・拒否": "個人・拒否権のみ", "個人・取引": "個人・取引の相手", "個人・D5": "個人・目的限定の承諾請求権（D5）",
       "管理∅・拒否": "管理会社・一覧なし", "管理∅・取引": "管理会社・一覧なし・取引", "管理∅・D4": "管理会社・一覧を住み手が決める（D4）",
       "管理∅・D5": "管理会社・D5", "管理開口・拒否": "管理会社・一覧に開口", "サブ・拒否": "サブリース・拒否権のみ", "サブ・取引": "サブリース・取引の相手",
       "サブ・D5": "サブリース・D5"}
fig, axs = plt.subplots(3, 1, figsize=(7.2, 15.0), sharey=True)
for ax, (col, rg, lab) in zip(axs, [("eta", "TP+top3", "到達度 η（TP+ 上位 3 手）"), ("eta", "改良版top3", "到達度 η（改良版 上位 3 手）"),
                                    ("Vn", "改良版top3", "$V^{*}_{n}$（改良版 上位 3 手）")]):
    y = np.arange(len(OW))
    for k, (x, c) in enumerate([(0.10, "#9bbfea"), (0.30, BLUE)]):
        ax.barh(y + (k - 0.5) * 0.38, [get("W", f"{o} ν={x:.2f}", rg, col) for o in OW], 0.36, color=c, label=f"ν＝{x:.2f}")
    if col == "Vn":
        ax.axvline(EPS, color="#a32d2d", lw=1, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([LAB[o] for o in OW], fontsize=10); ax.set_ylim(len(OW) - 0.5, -0.6)
    ax.set_xlabel(lab, fontsize=10.5); ax.tick_params(axis="x", labelsize=10); hair(ax, "x"); ax.axvline(0, color=INK2, lw=0.8)
axs[0].legend(fontsize=10, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("out/figs/fig234_S1b_所有者の型.png", dpi=170); plt.close(fig)
print("ok")

# ---------------------------------------------------------------- 235 総覧
from experiments.style import SEQ_B, ink_on
ROWS = [("基準", [("T", "幾何 ν=0.00")], "入れ替わりなし（第一報と同じ）"),
        ("入れ替わり", [("T", "幾何 ν=0.05")], "毎年 5 %"), (None, [("T", "幾何 ν=0.10")], "毎年 10 %"), (None, [("T", "幾何 ν=0.18")], "毎年 18 %（民営賃貸の全体）"),
        (None, [("T", "幾何 ν=0.30")], "毎年 30 %（単身向け）"), (None, [("T", "性向を引き継ぐ ν=0.30")], "30 %・性向を引き継ぐ対照"),
        ("原状回復", [("R", "R∅ 費用あり ν=0.30")], "30 %・費用あり・R∅"), (None, [("R", "R0 費用あり ν=0.30")], "30 %・費用あり・R0（日本の既定）"),
        (None, [("R", "Rdk20 費用あり ν=0.30")], "30 %・費用あり・Rdk20"),
        ("所有と場", [("O", "所有・G1 ν=0.04")], "所有・寄り合い（4 %）"), (None, [("O", "所有・G0 ν=0.04")], "所有・場なし（4 %）"),
        (None, [("O", "借家・G½ ν=0.04")], "借家・44 条型（4 %）"), (None, [("O", "借家・G1 ν=0.04")], "借家・寄り合い（4 %）"),
        ("所有者", [("W", "なし ν=0.10")], "所有者なし（10 %・R0）"), (None, [("W", "個人・拒否 ν=0.10")], "個人・拒否権のみ"),
        (None, [("W", "個人・取引 ν=0.10")], "個人・取引の相手"), (None, [("W", "管理∅・D4 ν=0.10")], "管理会社・一覧を住み手が決める"),
        (None, [("W", "管理∅・D5 ν=0.10")], "管理会社・承諾請求権（D5）"), (None, [("W", "管理∅・拒否 ν=0.10")], "管理会社・一覧なし")]
COLS = [("eta", "TP+top3", "η\nTP+"), ("eta", "改良版top3", "η\n改良版"), ("Vn", "TP+top3", "$V^{*}_{n}$\nTP+"), ("Vn", "改良版top3", "$V^{*}_{n}$\n改良版")]
fig, ax = plt.subplots(figsize=(7.2, 9.6))
fig.subplots_adjust(left=0.47, right=0.98, top=0.9, bottom=0.02)
for i, (grp, keys, lab) in enumerate(ROWS):
    g, c = keys[0]
    for j, (col, rg, _) in enumerate(COLS):
        v = get(g, c, rg, col)
        vmax = 1.0 if col == "eta" else 0.6
        rgb = SEQ_B(np.clip(max(v, 0) / vmax, 0, 1))
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=rgb, ec="white", lw=1.5))
        ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=10, color=ink_on(rgb))
for i, (grp, keys, lab) in enumerate(ROWS):
    if grp:
        ax.axhline(i, color=INK2, lw=0.9)
        ax.text(-3.55, i + 0.5, grp, fontsize=10.5, fontweight="bold", color=INK, va="center", ha="left")
ax.set_xlim(0, 4); ax.set_ylim(len(ROWS), 0)
ax.set_xticks(np.arange(4) + 0.5); ax.set_xticklabels([c[2] for c in COLS], fontsize=10.5); ax.xaxis.tick_top()
ax.set_yticks(np.arange(len(ROWS)) + 0.5); ax.set_yticklabels([r[2] for r in ROWS], fontsize=10)
ax.axvline(2, color=INK, lw=1.4)
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.tick_params(length=0)
fig.savefig("out/figs/fig235_S1b_総覧.png", dpi=170); plt.close(fig)
print("ok 235")
