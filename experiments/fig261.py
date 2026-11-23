"""図 261〜266：職能 3（性向を知りうる運用役）の結果。幅 7.2 in・文字 10 pt 以上・太字なし。
PYTHONPATH=. python3 -m experiments.fig261            → out/figs/fig261〜fig266（判定 out/s7/judge.json から）
S7_DRY=1 なら out/s7/dry_judge.json から out/s7/dry_figs/ へ（試し）。S7_DEV=1 なら選ぶ用の out/s7/dev_values.json から out/s7/dev_figs/ へ
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiments.style import INK, INK2, MUTED, BLUE, ORANGE, AQUA, GREY, DEEP, hair

plt.rcParams["axes.titleweight"] = "normal"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["font.size"] = 10
DRY = os.environ.get("S7_DRY") == "1"
DEV = os.environ.get("S7_DEV") == "1"
SRC = "out/s7/dev_values.json" if DEV else ("out/s7/dry_judge.json" if DRY else "out/s7/judge.json")
J = json.load(open(SRC))
OUTD = "out/s7/dev_figs" if DEV else ("out/s7/dry_figs" if DRY else "out/figs")
os.makedirs(OUTD, exist_ok=True)
V, M, D = J["values"], J["means"], J["diffs"]
EPS = J.get("epsilon", J.get("epsilon_dev"))
PATS = ["α=3", "二群"]
PLAB = {"α=3": "α＝3", "二群": "二群（広さ派と静けさ派）"}
NUS = [0.0, 0.05, 0.1, 0.3]
SCOPES = ["全員", "上位1手", "上位3手", "近い戸だけ"]
REFS = ("計画者", "BR", "BRdyn")
ACTOR = ("選ぶ者:分布", "選ぶ者:性向", "道具:物理", "道具:満足", "仲介者")


def key(b, arm, cond, p, nu):
    if arm in REFS:
        if arm == "BRdyn" and nu == 0:
            arm = "BR"
        return f"参照|{arm}|基準|{p}|{nu}"
    if arm == "範囲:上位3手":
        arm = "基準"
    return f"{b}|{arm}|{cond}|{p}|{nu}"


def vn(b, arm, cond, p, nu, q):
    k = key(b, arm, cond, p, nu)
    if not (arm in ACTOR or arm in REFS):
        q = "Vn外"
    return V[f"{k}|{q}"]


def mean(b, arm, cond, p, nu, c="eta"):
    return M[key(b, arm, cond, p, nu)][c]


def save(fig, name):
    fn = f"{OUTD}/{name}"
    fig.savefig(fn, dpi=200, bbox_inches="tight"); plt.close(fig); print(fn)


# ---------------------------------------------------------------- 図 261：総覧（TP+top3・入れ替わりなし）
ROWS = [("基準", "基準", "運用役なし（全員に届く）", GREY),
        ("基準", "近い戸だけ", "運用役なし（近い戸だけ）", GREY),
        ("選ぶ者:分布", "基準", "範囲を選ぶ者・分布だけ", AQUA),
        ("道具:物理", "近い戸だけ", "道具・物理の量（近い戸だけ）", AQUA),
        ("仲介者", "近い戸だけ", "職能 2 の仲介者（近い戸だけ）", MUTED),
        ("選ぶ者:性向", "基準", "範囲を選ぶ者・性向を知る", BLUE),
        ("道具:満足", "基準", "道具・満足の量（全員に届く）", BLUE),
        ("道具:満足", "近い戸だけ", "道具・満足の量（近い戸だけ）", BLUE),
        ("BR", "基準", "全知の仲介者 BR", ORANGE),
        ("計画者", "基準", "全知の計画者", ORANGE)]


def overview(b, name, nu=0.0):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True)
    ys = np.arange(len(ROWS))[::-1]
    for ax, p in zip(axes, PATS):
        for y, (arm, cond, lab, col) in zip(ys, ROWS):
            o = vn(b, arm, cond, p, nu, "Vn外"); c = vn(b, arm, cond, p, nu, "Vn数")
            ax.plot([o["v"], c["v"]], [y, y], color=col, lw=1.6, zorder=1)
            ax.hlines(y + 0.17, o["lo"], o["hi"], color=col, lw=0.8, alpha=0.7)
            ax.scatter([c["v"]], [y], s=70, facecolor="white", edgecolor=col, lw=1.4, zorder=3)
            ax.scatter([o["v"]], [y], s=26, color=col, zorder=4)
            ax.text(1.0, y, f"{mean(b, arm, cond, p, nu):.2f}", fontsize=10, color=INK2, va="center", ha="right")
        ax.set_xlim(-0.03, 1.02); ax.set_title(PLAB[p], fontsize=10, color=INK)
        ax.text(1.0, len(ROWS) - 0.3, "η", fontsize=10, color=INK2, ha="right")
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8]); ax.set_xlabel("$V^{*}_{n}$", fontsize=10)
        hair(ax, "x"); ax.tick_params(labelsize=10)
    axes[0].set_yticks(ys); axes[0].set_yticklabels([r[2] for r in ROWS], fontsize=10)
    hs = [Line2D([], [], color=INK2, marker="o", ls="", ms=5), Line2D([], [], color=INK2, marker="o", mfc="white", ls="", ms=8)]
    fig.legend(hs, ["〔外〕運用役を作者に含めない", "〔内〕運用役を作者に含める"], loc="upper center", ncol=2, fontsize=10, frameon=False,
               bbox_to_anchor=(0.6, 1.06))
    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- 図 262：聞く範囲を選ぶ者（全員に届く）
def scope_fig(name, nu=0.0):
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.15, 1.15], hspace=0.75, wspace=0.08)
    cols = {"全員": GREY, "上位1手": AQUA, "上位3手": BLUE, "近い戸だけ": ORANGE}
    b = "改良版top3"
    for c_, p in enumerate(PATS):
        ax = fig.add_subplot(gs[0, c_])
        for r_, nu_ in enumerate(NUS):
            cnt = M[key(b, "選ぶ者:性向", "基準", p, nu_)]["choice_counts"]
            n = sum(cnt.values()); left = 0
            for s in SCOPES:
                w = cnt[s] / n
                ax.barh(r_, w, left=left, color=cols[s], height=0.62, label=s if (c_ == 0 and r_ == 0) else None)
                if w > 0.1:
                    ax.text(left + w / 2, r_, f"{cnt[s]}", ha="center", va="center", fontsize=10, color="white")
                left += w
        ax.set_yticks(range(4)); ax.set_yticklabels(["入れ替わりなし", "毎年 5 %", "毎年 10 %", "毎年 30 %"] if c_ == 0 else [], fontsize=10)
        ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("選んだ範囲の割合（100 棟）", fontsize=10)
        ax.set_title(f"({'a' if c_ == 0 else 'b'}) 選んだ範囲・{'α＝3' if p == 'α=3' else '二群'}", fontsize=10, color=INK); ax.tick_params(labelsize=10)
    arms = [("範囲:全員", "全員", GREY), ("範囲:上位1手", "上位 1 手", AQUA), ("範囲:上位3手", "上位 3 手\n（基準）", BLUE),
            ("範囲:近い戸だけ", "近い戸だけ", ORANGE), ("選ぶ者:分布", "選ぶ者\n（分布だけ）", MUTED), ("選ぶ者:性向", "選ぶ者\n（性向を知る）", DEEP)]
    bb = "TP+top3"
    for r_, p in enumerate(PATS):
        ax = fig.add_subplot(gs[1 + r_, :])
        for x, (arm, lab, col) in enumerate(arms):
            o = vn(bb, arm, "基準", p, nu, "Vn外")
            ax.bar(x - 0.18, o["v"], 0.34, color=col)
            ax.vlines(x - 0.18, o["lo"], o["hi"], color=INK2, lw=0.8)
            if arm.startswith("選ぶ者"):
                c = vn(bb, arm, "基準", p, nu, "Vn数")
                ax.bar(x + 0.18, c["v"], 0.34, facecolor="white", edgecolor=col, hatch="///", lw=1.0)
                if c["v"] < 0.005:
                    ax.text(x + 0.18, 0.01, "0", ha="center", va="bottom", fontsize=10, color=col)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([a[1] for a in arms], fontsize=10)
        ax.set_ylim(0, 0.75); ax.set_ylabel("$V^{*}_{n}$", fontsize=10)
        ax.set_title(f"({'c' if r_ == 0 else 'd'}) 固定した範囲と選ぶ者・TP+top3・{PLAB[p]}・入れ替わりなし", fontsize=10, color=INK)
        hair(ax); ax.tick_params(labelsize=10)
        if r_ == 0:
            ax.legend(handles=[Patch(facecolor=INK2), Patch(facecolor="white", edgecolor=INK2, hatch="///")],
                      labels=["〔外〕", "〔内〕（選ぶ者を作者に含める）"], fontsize=10, frameon=False, loc="upper left", ncol=2)
    h = [Patch(facecolor=cols[s]) for s in SCOPES]
    fig.legend(h, ["全員", "上位 1 手", "上位 3 手", "近い戸だけ"], loc="upper center", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.95))
    save(fig, name)


# ---------------------------------------------------------------- 図 263：入れ替わりのもとで（改良版top3）
def turnover_fig(name):
    b = "改良版top3"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    xs = np.array(NUS)
    lines = [("基準", "基準", "運用役なし", GREY, "-"), ("選ぶ者:性向", "基準", "範囲を選ぶ者・性向を知る", BLUE, "-"),
             ("道具:満足", "近い戸だけ", "道具・満足の量（近い戸だけ。〔外〕は運用役なしと重なる）", ORANGE, "-")]
    for c_, p in enumerate(PATS):
        for r_, q in enumerate(("Vn外", "Vn数")):
            ax = axes[r_, c_]
            for arm, cond, lab, col, ls in lines:
                xx = [vn(b, arm, cond, p, nu, q) for nu in NUS]
                ax.plot(xs, [x["v"] for x in xx], color=col, lw=1.4, marker="o", ms=3.5, label=lab)
                ax.vlines(xs, [x["lo"] for x in xx], [x["hi"] for x in xx], color=col, lw=0.7, alpha=0.6)
            ax.set_ylim(0, 0.7); hair(ax); ax.tick_params(labelsize=10)
            ax.set_title(f"{'〔外〕' if q == 'Vn外' else '〔内〕'}・{PLAB[p]}", fontsize=10, color=INK)
            if c_ == 0:
                ax.set_ylabel("$V^{*}_{n}$", fontsize=10)
            if r_ == 1:
                ax.set_xticks(xs); ax.set_xticklabels(["なし", "5 %", "10 %", "30 %"], fontsize=10); ax.set_xlabel("毎年の退去率", fontsize=10)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=1, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.11))
    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- 図 264：運用役が知ったことと、作者に含めると下がる分
def knowledge_fig(name):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    sty = {"選ぶ者:分布": (AQUA, "s", "選ぶ者・分布"), "道具:物理": (AQUA, "^", "道具・物理"), "仲介者": (MUTED, "D", "仲介者"),
           "選ぶ者:性向": (BLUE, "s", "選ぶ者・性向"), "道具:満足": (BLUE, "^", "道具・満足")}
    seen = set()
    for b in ("TP+top3", "改良版top3"):
        for arm, (col, mk, lab) in sty.items():
            for cond in ("基準", "近い戸だけ"):
                for p in PATS:
                    for nu in NUS:
                        k = key(b, arm, cond, p, nu)
                        if k not in M or f"{k}|Vn外" not in V:
                            continue
                        o = V[f"{k}|Vn外"]["v"]; c = V[f"{k}|Vn数"]["v"]
                        if o <= EPS:
                            continue
                        m = M[k]
                        imax_res = V[f"{k}|Vn外"]["Imax"]
                        for ax, x in ((axes[0], m["nfacts"]), (axes[1], (m.get("I_actor") or 0.0) - imax_res)):
                            ax.scatter(x, o - c, s=16, color=col, marker=mk, alpha=0.75, lw=0, label=lab if (lab not in seen and ax is axes[0]) else None)
                        seen.add(lab)
    axes[0].set_xlabel("運用役が 30 年目に知っている係数の数\n（1 棟あたり、27 のうち）", fontsize=10)
    axes[1].set_xlabel("運用役の知識 − 最も多く知る住み手の知識\n（全知の最適を言い当てる度合い）", fontsize=10)
    axes[1].axvline(0, color=GREY, lw=0.8, ls="--")
    for ax, t in zip(axes, ("(a) 知った係数の数", "(b) 知識の量の差")):
        ax.set_ylabel("〔外〕−〔内〕", fontsize=10); ax.set_title(t, fontsize=10, color=INK); hair(ax, "both"); ax.tick_params(labelsize=10)
    axes[0].legend(fontsize=10, frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- 図 265：予言の判定
SHORT = {"PS1": "性向を知る選ぶ者：二群・入れ替わりなしで〔外〕が上がる", "PS2": "性向を知る選ぶ者：入れ替わりなしで η が下がる",
         "PS3": "性向を知る選ぶ者：入れ替わりのもとで〔内〕＝〔外〕", "PS4": "性向を知る選ぶ者：年 5 % で η が下がる",
         "PD1": "分布だけの選ぶ者：入れ替わりなしで〔外〕は基準と同じ", "PD1b": "分布だけの選ぶ者：年 5・10 % で〔外〕は基準と同じ",
         "PT1": "物理の量の道具：近い戸だけの建物で η が上がる", "PT2": "物理の量の道具：近い戸だけで〔外〕が上がる（TP+top3）",
         "PT2b": "同上（改良版top3）", "PT3": "満足の量の道具：入れ替わりのもとで〔内〕が下がる",
         "PT4": "満足の量の道具の〔内〕＜物理の量（TP+top3）", "PT4b": "同上（改良版top3）"}


def verdicts(name):
    P = J["predictions"]
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(P) + 0.6))
    ax.axis("off")
    col = {"支持": BLUE, "一部支持": AQUA, "不支持": ORANGE}
    for n, (k, v) in enumerate(P.items()):
        y = len(P) - n
        ax.text(0.0, y, k, fontsize=10, color=INK, va="center")
        ax.text(0.08, y, SHORT.get(k, v["text"][:36]), fontsize=10, color=INK2, va="center")
        ax.text(0.81, y, f"{v['verdict']}（{v['n_ok']}/{v['n_cells']}）", fontsize=10, color=col[v["verdict"]], va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0.3, len(P) + 0.7)
    save(fig, name)


if __name__ == "__main__":
    overview("TP+top3", "fig261_S7_職能3の総覧_TP.png")
    scope_fig("fig262_S7_聞く範囲を選ぶ者.png")
    turnover_fig("fig263_S7_入れ替わりと知識の古び_改良版.png")
    knowledge_fig("fig264_S7_運用役が知ったことと数え方.png")
    if "predictions" in J:
        verdicts("fig265_S7_予言の判定.png")
