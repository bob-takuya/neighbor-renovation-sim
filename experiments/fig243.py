"""図 243〜245：職能 1 の結果（事前登録 v7a、cseed 500〜599）。幅 7.2 in・文字 10 pt 以上・太字なし。体裁は style.py に合わせる。
  243 総覧：三つの運用役ごとに、η と V*n〔外〕を退去率に対して示す（改良版top3、両パターン）
  244 同じ図を TP+top3 で
  245 物差しの約束を替えた値：記録者を作者に含める〔内〕と記録の読み方、検証者の制約のもとの物差し、戻せる手の多い盤の割合
PYTHONPATH=. python3 -m experiments.fig243          （S5_DRY=1 なら試しの判定 out/s5a/dry_judge.json から描く）
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from experiments.style import INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, GREY, hair

DRY = os.environ.get("S5_DRY") == "1"
J = json.load(open("out/s5a/dry_judge.json" if DRY else "out/s5a/judge.json"))
V, M = J["values"], J["means"]
eps = J["epsilon"]
SUF = "_試し" if DRY else ""
NUS = [0.0, 0.05, 0.1, 0.3]
XT = ["なし", "5 %", "10 %", "30 %"]
RED = "#a32d2d"; DEEP = "#24438f"
GROUPS = [("盤の設計者", [("基準", "今の盤（基準）", GREY), ("狭い盤", "狭い盤", ORANGE), ("戻せる手の多い盤", "戻せる手の多い盤", BLUE)]),
          ("記録者", [("基準", "同意だけ記録（基準と同じ）", GREY), ("記録なし", "記録なし", ORANGE), ("額まで記録", "額まで記録", BLUE)]),
          ("検証者", [("基準", "制約なし（基準）", GREY), ("一律に禁止", "一律に禁止", ORANGE), ("代わりの手", "代わりの手", BLUE)])]


def overview(base, fn, title):
    fig, axs = plt.subplots(3, 2, figsize=(7.2, 9.6))
    for r, (actor, arms) in enumerate(GROUPS):
        for c, q in enumerate(["eta", "Vn"]):
            ax = axs[r, c]
            for arm, lab, col in arms:
                if arm not in [k.split("|")[1] for k in M if k.startswith(base + "|")]:
                    continue
                for pat, ls, mk in (("α=3", "-", "o"), ("二群", "--", "s")):
                    if q == "eta":
                        y = [M[f"{base}|{arm}|{pat}|{nu}"]["eta"] for nu in NUS]
                        ax.plot(NUS, y, ls=ls, marker=mk, color=col, lw=1.4, ms=4)
                    else:
                        v = [V[f"{base}|{arm}|{pat}|{nu}|外"] for nu in NUS]
                        y = [x["v"] for x in v]
                        ax.plot(NUS, y, ls=ls, marker=mk, color=col, lw=1.4, ms=4)
                        ax.vlines(NUS, [x["lo"] for x in v], [x["hi"] for x in v], color=col, lw=0.8, alpha=0.6)
            if q == "Vn":
                ax.axhline(eps, color=RED, lw=0.8, ls=":")
            ax.set_xticks(NUS); ax.set_xticklabels(XT, fontsize=10); ax.set_xlim(-0.015, 0.315)
            ax.set_ylim((0.0, 1.65) if q == "eta" else (0.0, 0.85))
            ax.tick_params(axis="y", labelsize=10); hair(ax)
            ax.set_title(f"({'abcdef'[2 * r + c]}) {actor}：{'η' if q == 'eta' else '$V^{*}_{n}$〔外〕'}", fontsize=11, loc="left", fontweight="normal")
            if r == 2:
                ax.set_xlabel("毎年の退去率", fontsize=10)
            if c == 1:
                hs = [plt.Line2D([], [], color=col, ls=ls, marker=mk, ms=4, lw=1.4) for _, _, col in arms for ls, mk in (("-", "o"), ("--", "s"))]
                ls_ = [f"{lab}・{pt}" for _, lab, _ in arms for pt in ("α=3", "二群")]
                ok_ = [k for k, (a, _, _) in enumerate([x for x in arms for _ in (0, 1)]) if a in [kk.split("|")[1] for kk in M if kk.startswith(base + "|")]]
                ax.legend([hs[k] for k in ok_], [ls_[k] for k in ok_], fontsize=9, frameon=False, loc="upper right", ncol=1)
    fig.suptitle(title, fontsize=11, x=0.01, ha="left", fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(fn, dpi=200)
    print(fn)


overview("改良版top3", f"out/figs/fig243_S5a_職能1の総覧_改良版{SUF}.png",
         "改良版top3 を基準にした三つの役者の η と $V^{*}_{n}$\n（実線 α＝3、破線 二群。縦線は 95 % 区間、点線は ε）")
overview("TP+top3", f"out/figs/fig244_S5a_職能1の総覧_TP{SUF}.png",
         "TP+top3 を基準にした三つの役者の η と $V^{*}_{n}$\n（実線 α＝3、破線 二群。記録なしは基準と同じなので描かない）")

# ---------------------------------------------------------------- 245 物差しの約束を替えた値
fig, axs = plt.subplots(3, 1, figsize=(7.2, 10.0))
w = 0.18
# (a) 記録者：額まで記録の〔外〕・〔内〕・読み方（改良版top3）
ax = axs[0]
SER = [("基準|外", "基準", GREY), ("額まで記録|外", "額まで記録：入居した世帯が読む〔外〕", BLUE), ("額まで記録|数", "同：記録者を作者に含める〔内〕", DEEP),
       ("額まで記録|外・全員が読む", "同：住み手全員が読む〔外〕", AQUA)]
xl = []
for i, (pat, nu) in enumerate([(p, nu) for p in ("α=3", "二群") for nu in NUS]):
    xl.append(f"{'α=3' if pat == 'α=3' else '二群'}\n{XT[NUS.index(nu)]}")
    for k, (key, lab, col) in enumerate(SER):
        arm, nm = key.split("|")
        x = V[f"改良版top3|{arm}|{pat}|{nu}|{nm}"]
        ax.bar(i + (k - 1.5) * w, x["v"], w, color=col, label=lab if i == 0 else None)
        if "lo" in x:
            ax.plot([i + (k - 1.5) * w] * 2, [x["lo"], x["hi"]], color=INK2, lw=0.7)
ax.set_xticks(range(8)); ax.set_xticklabels(xl, fontsize=10); ax.tick_params(axis="y", labelsize=10); hair(ax)
ax.legend(fontsize=9, frameon=False, ncol=2, loc="upper right")
ax.set_title("(a) 記録者：額まで記録したときの $V^{*}_{n}$（改良版top3）。下は退去率", fontsize=11, loc="left", fontweight="normal")
# (b) 検証者：主な物差しと制約のもとの物差し（入れ替わりなし）
ax = axs[1]
SER = [("基準", GREY), ("一律に禁止", ORANGE), ("代わりの手", BLUE)]
xl = []
cols = [(b, p, z) for b in ("TP+top3", "改良版top3") for p in ("α=3", "二群") for z in ("外", "外・制約のもとの最適")]
for i, (b, p, z) in enumerate(cols):
    xl.append(f"{'TP+' if b == 'TP+top3' else '改良版'}\n{p}\n{'全知の最適' if z == '外' else '制約のもと'}")
    for k, (arm, col) in enumerate(SER):
        x = V[f"{b}|{arm}|{p}|0.0|{z}"]
        ax.bar(i + (k - 1) * 0.25, x["v"], 0.25, color=col, label=arm if i == 0 else None)
        ax.plot([i + (k - 1) * 0.25] * 2, [x["lo"], x["hi"]], color=INK2, lw=0.7)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(xl, fontsize=9.5, linespacing=1.0); ax.tick_params(axis="y", labelsize=10); hair(ax)
ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper right")
ax.set_title("(b) 検証者：$V^{*}_{n}$ を全知の最適と制約のもとの最適で測る（入れ替わりなし）", fontsize=11, loc="left", fontweight="normal")
# (c) 戻せる手の多い盤：量（21 部位を分母）と割合（31 部位を分母）
ax = axs[2]
xl = []
cells = [("TP+top3", p, 0.0) for p in ("α=3", "二群")] + [("改良版top3", p, nu) for p in ("α=3", "二群") for nu in (0.0, 0.05, 0.1)]
for i, (b, p, nu) in enumerate(cells):
    xl.append(f"{'TP+' if b == 'TP+top3' else '改良版'}\n{p}\n{XT[NUS.index(nu)]}")
    base = V[f"{b}|基準|{p}|{nu}|外"]; rich = V[f"{b}|戻せる手の多い盤|{p}|{nu}|外"]
    for k, (val, lab, col) in enumerate([(base["v"], "今の盤", GREY), (rich["v"], "戻せる手の多い盤：量", BLUE), (rich["v_own"], "戻せる手の多い盤：割合", AQUA)]):
        ax.bar(i + (k - 1) * 0.25, val, 0.25, color=col, label=lab if i == 0 else None)
ax.set_xticks(range(len(cells))); ax.set_xticklabels(xl, fontsize=9.5, linespacing=1.0); ax.tick_params(axis="y", labelsize=10); hair(ax)
ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper right")
ax.set_title("(c) 盤の設計者：$V^{*}_{n}$ の量と割合（下は退去率）", fontsize=11, loc="left", fontweight="normal")
for ax in axs:
    ax.set_ylabel("$V^{*}_{n}$", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
fig.tight_layout()
fn = f"out/figs/fig245_S5a_物差しの約束を替えた値{SUF}.png"
fig.savefig(fn, dpi=200); print(fn)
