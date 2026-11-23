"""図 240・241：確証の実験（事前登録 v6、cseed 300〜399）。幅 7.2 in・文字 10 pt 以上。体裁は style.py・fig230・fig231 に合わせる。
  240 囮を含む V*n の比較：パターンごとに、住民の手続きと囮の V*n と 95 % 区間（4 セル）、M の V* と V*n（清算所を含める／数えない）
  241 予言 v4-P1〜P5 の再判定と、囮の判定・再現の基準の表
PYTHONPATH=. python3 -m experiments.fig240          （S4_DRY=1 なら試しの判定 out/s4/dry_judge.json から描く）
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from experiments.style import INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, GREY, hair

DRY = os.environ.get("S4_DRY") == "1"
J = json.load(open("out/s4/dry_judge.json" if DRY else "out/s4/judge.json"))
OLD = json.load(open("out/s2/name_judge.json"))
V = J["values"]; eps = J["epsilon"]
SUF = "_試し" if DRY else ""
RED = "#a32d2d"; DEEPB = "#24438f"


def val(r, pat, nu, c, q="Vn"):
    return V[f"{r}|{pat}|{nu}|{c}"][q]


# ---------------------------------------------------------------- 240
REGS = [("TP+", "TP+\n全員"), ("TP+top3", "TP+\n上位3"), ("改良版top3", "改良版\n上位3"),
        ("Mtop3", "M 上位3\n清算所を\n含めない"), ("Mtop3ch", "M 上位3\n清算所を\n含める"), ("Reptop3", "Rep\n上位3"),
        ("M", "M 全員\n含めない"), ("Rep", "Rep\n全員")]
SER = [(0.0, "なし", BLUE, "入れ替わりなし・費用なし"), (0.0, "C1", "#9bbfea", "入れ替わりなし・費用あり"),
       (0.1, "なし", AQUA, "毎年 10 %・費用なし"), (0.1, "C1", "#a2d7cb", "毎年 10 %・費用あり")]
fig, axs = plt.subplots(4, 1, figsize=(7.2, 14.0), gridspec_kw=dict(height_ratios=[1, 1, 0.8, 0.8]))
for ax, pat, title in [(axs[0], "α=3", "(a) ばらつきやや大（α＝3）：住民の手続きと二つの囮の $V^{*}_{n}$"),
                       (axs[1], "二群", "(b) 二群（広さ派 5・静けさ派 4）：住民の手続きと二つの囮の $V^{*}_{n}$")]:
    w = 0.2
    for k, (nu, c, col, lab) in enumerate(SER):
        for i, (r, _) in enumerate(REGS):
            v, lo, hi = val(r, pat, nu, c)
            ax.bar(i + (k - 1.5) * w, v, w, color=col, label=lab if i == 0 else None)
            ax.plot([i + (k - 1.5) * w] * 2, [lo, hi], color=INK2, lw=0.8)
    ax.axvspan(2.5, 5.5, color="#f3efe6", zorder=-1)
    ax.axvline(5.5, color=MUTED, lw=0.8, ls=":")
    ax.text(4.0, 0.69, "主な判定に使う囮", fontsize=10, ha="center", color=INK2)
    ax.text(6.5, 0.69, "あわせて報告する囮", fontsize=10, ha="center", color=INK2)
    ax.axhline(eps, color=RED, lw=1, ls="--")
    ax.text(-0.5, eps + 0.015, f"ε ＝ {eps:.3f}", color=RED, fontsize=10, ha="left")
    ax.set_xticks(range(len(REGS))); ax.set_xticklabels([l for _, l in REGS], fontsize=10, linespacing=1.0)
    ax.set_xlim(-0.6, len(REGS) - 0.4); ax.set_ylim(0, 0.75)
    ax.set_ylabel("$V^{*}_{n}$", fontsize=11); ax.tick_params(axis="y", labelsize=10); hair(ax)
    ax.set_title(title, fontsize=11.5, loc="left")
# (c)(d) M の V* と V*n：清算所を含める／数えない（入れ替わりなし・費用なし）
MREG = [("TP+top3", "TP+\n上位3"), ("Mtop3", "M 上位3\n含めない"), ("Mtop3ch", "M 上位3\n含める"),
        ("TP+", "TP+\n全員"), ("M", "M 全員\n含めない"), ("Mch", "M 全員\n含める")]
for ax, pat, lab in [(axs[2], "α=3", "(c) α＝3"), (axs[3], "二群", "(d) 二群")]:
    w = 0.36
    for k, (q, col, nm) in enumerate([("Vs", ORANGE, "$V^{*}$（同意を得た部位）"), ("Vn", BLUE, "$V^{*}_{n}$（損をする側の同意を得た部位）")]):
        for i, (r, _) in enumerate(MREG):
            v, lo, hi = val(r, pat, 0.0, "なし", q)
            ax.bar(i + (k - 0.5) * w, v, w, color=col, label=nm if i == 0 else None)
            ax.plot([i + (k - 0.5) * w] * 2, [lo, hi], color=INK2, lw=0.8)
    ax.axhline(eps, color=RED, lw=1, ls="--")
    ax.axvline(2.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(range(len(MREG))); ax.set_xticklabels([l for _, l in MREG], fontsize=10, linespacing=1.0)
    ax.set_xlim(-0.6, len(MREG) - 0.4); ax.set_ylim(0, 0.85)
    ax.set_ylabel("値", fontsize=11); ax.tick_params(axis="y", labelsize=10); hair(ax)
    ax.set_title(f"{lab}：市場型 M の $V^{{*}}$ と $V^{{*}}_{{n}}$（入れ替わりなし・費用なし）", fontsize=11.5, loc="left")
axs[2].legend(fontsize=10, frameon=False, ncol=2, loc="upper right")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0),
           title="(a)(b) の棒の色", title_fontsize=10)
fig.tight_layout(h_pad=1.3, rect=(0, 0.045, 1, 1))
fig.savefig(f"out/figs/fig240_S4_囮を含むVnの比較{SUF}.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- 241
YES = "支持"; NO = "不支持"
def jud(d):
    return YES if d["ok"] else NO


def p_detail(J_, p):
    if p == "P1":
        return f"不合格 {len(J_['P1']['fails'])}"
    if p == "P2":
        rows = J_["P2"]["rows"]
        return f"{sum(r['ok'] for r in rows)}/{len(rows)}"
    if p == "P3":
        return f"不合格 {len(J_['P3']['fails'])}"
    if p == "P4":
        return f"{sum(r['ok'] for r in J_['P4']['rows'])}/{len(J_['P4']['rows'])}"
    if p == "P5":
        return f"{sum(r['ok'] for r in J_['P5']['rows'])}/{len(J_['P5']['rows'])}"


ROWS = [("v4-P1", "帰無と偽装は\nすべてのセルで ε 以下"), ("v4-P2", "予言した順位の組の\n8 割以上で前者が上"),
        ("v4-P3", "合意のない決め方\n（U・V50・V75）は ε 以下"), ("v4-P4", "η がほぼ同じ TP+top3 と\nTP+ を区別する"),
        ("v4-P5", "悪い盤から始めても\n水増しされない")]
cells = []
for (p, txt) in ROWS:
    k = p.split("-")[1]
    cells.append([p, txt, f"{jud(OLD[k])}（{p_detail(OLD, k)}）", f"{jud(J[k])}（{p_detail(J, k)}）"])
D1 = J["D1"]; D2 = J["D2"]; R = J["replication"]
n_ref = D1["n_ref_cells"]
cells.append(["囮 D1", "市場型 M（上位 3 手・清算所を\n含めない）と TP+top3 を比べる", "—",
              f"{D1['verdict']}\n（上回る {len(D1['cells_hit'])}/8・下回る {D1['n_M_lower']}/{n_ref}）"])
cells.append(["囮 D2", "代表型 Rep（上位 3 手）は\n8 セルすべてで ε 以下か", "—",
              ("ε 以下 8/8\n名を直接取引に限定" if D2["all_below_eps"] else f"ε 以下 {sum(r['below_eps'] for r in D2['rows'])}/8\n限定しない")])
cells.append(["再現", "向きの一致の割合・\n順位相関の中央値（各 0.8 以上）", "基準",
              f"{R['direction_share']:.2f}・{R['rho_median']:.2f}\n（{'関係は一致' if R['direction_ok'] and R['rank_ok'] else '不一致'}）"])
fig, ax = plt.subplots(figsize=(7.2, 6.4))
ax.axis("off")
head = ["予言", "中身", "名付けの確かめ\ncseed 200〜249", "確証\ncseed 300〜399"]
tb = ax.table(cellText=cells, colLabels=head, cellLoc="left", colLoc="left", loc="upper left",
              colWidths=[0.11, 0.37, 0.24, 0.28])
tb.auto_set_font_size(False); tb.set_fontsize(10)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor(GRID); cell.set_linewidth(0.8)
    cell.set_height(0.105)
    if r == 0:
        cell.set_facecolor("#efeee9"); cell.get_text().set_color(INK); cell.get_text().set_fontweight("bold")
    else:
        cell.set_facecolor("#ffffff" if r <= 5 else "#f6f4ee")
        t = cell.get_text().get_text()
        if c == 3 and (t.startswith(NO) or t.startswith("不適切") or "不一致" in t):
            cell.get_text().set_color(RED)
ax.set_title(f"予言 v4-P1〜P5 の再判定と囮の判定\n（確証の ε ＝ {eps:.3f}、ブートストラップ {J['bootstrap']:,} 回）", fontsize=11.5, loc="left")
fig.tight_layout()
fig.savefig(f"out/figs/fig241_S4_再判定の表{SUF}.png", dpi=160); plt.close(fig)
print("ok")
