"""図 251〜254：職能 2（聞いた言い値だけを知る仲介者）の結果。幅 7.2 in・文字 10 pt 以上・太字なし。
PYTHONPATH=. python3 -m experiments.fig251            → out/figs/fig251〜fig254（判定 out/s6/judge.json から）
S6_DRY=1 なら out/s6/dry_judge.json から out/s6/dry_figs/ へ（試し）
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from experiments.style import INK, INK2, MUTED, BLUE, ORANGE, AQUA, GREY, DEEP, hair

plt.rcParams["axes.titleweight"] = "normal"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["font.size"] = 10
DRY = os.environ.get("S6_DRY") == "1"
J = json.load(open("out/s6/dry_judge.json" if DRY else "out/s6/judge.json"))
OUTD = "out/s6/dry_figs" if DRY else "out/figs"
os.makedirs(OUTD, exist_ok=True)
V, M = J["values"], J["means"]
PATS = ["α=3", "二群"]
NUS = [0.0, 0.05, 0.1, 0.3]
CONDS = ["基準", "費用1倍", "近い戸だけ"]
CLAB = {"基準": "費用なし・\n全員に届く", "費用1倍": "費用 1 倍", "近い戸だけ": "近い戸に\nしか聞けない"}
ARMS = [("基準", "仲介者なし", GREY), ("仲介者", "仲介者（届かなかった手だけ）", BLUE),
        ("広く聞く仲介者", "広く聞く仲介者", AQUA), ("BR", "全知の仲介者 BR", ORANGE)]


def key(b, arm, cond, p, nu):
    """値の鍵。届かなかった手だけの仲介者は費用なし・全員に届く条件では基準と同じ局（走らせない）。BR は近い戸だけでも基準の条件の局"""
    if arm in ("BR", "BRdyn"):
        return f"参照|{arm if not (arm == 'BRdyn' and nu == 0) else 'BR'}|{'基準' if cond == '近い戸だけ' else cond}|{p}|{nu}"
    if arm == "仲介者" and cond == "基準":
        arm = "基準"
    return f"{b}|{arm}|{cond}|{p}|{nu}"


def vn(b, arm, cond, p, nu, q):
    k = key(b, arm, cond, p, nu)
    if arm == "基準" or (arm == "仲介者" and cond == "基準"):
        q = "Vn外"
    return V[f"{k}|{q}"]


def eta(b, arm, cond, p, nu):
    return M[key(b, arm, cond, p, nu)]["eta"]


def save(fig, name):
    fn = f"{OUTD}/{name}"
    fig.savefig(fn, dpi=200, bbox_inches="tight"); plt.close(fig); print(fn)


# ---------------------------------------------------------------- 図 251：総覧（TP+top3・入れ替わりなし）
def overview(b, name, nu=0.0):
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.6), sharex=True)
    w = 0.2
    for c_, p in enumerate(PATS):
        for r_, (q, lab) in enumerate((("eta", "η（到達度）"), ("Vn外", "$V^{*}_{n}$〔外〕"), ("Vn数", "$V^{*}_{n}$〔内〕"))):
            ax = axes[r_, c_]
            for a_, (arm, alab, col) in enumerate(ARMS):
                xs = np.arange(3) + (a_ - 1.5) * w
                if q == "eta":
                    ys = [eta(b, arm, cd, p, nu) for cd in CONDS]
                    ax.bar(xs, ys, w * 0.92, color=col, label=alab)
                else:
                    xx = [vn(b, arm, cd, p, nu, q) for cd in CONDS]
                    ys = [x["v"] for x in xx]
                    ax.bar(xs, ys, w * 0.92, color=col, label=alab)
                    ax.errorbar(xs, ys, yerr=[[max(0.0, y - x["lo"]) for y, x in zip(ys, xx)], [max(0.0, x["hi"] - y) for y, x in zip(ys, xx)]],
                                fmt="none", ecolor=INK2, elinewidth=0.8, capsize=0)
                    for x_, y_ in zip(xs, ys):
                        if y_ < 0.005:
                            ax.text(x_, 0.012, "0", ha="center", va="bottom", fontsize=10, color=col)
            ax.set_ylim(0, 1.05 if q == "eta" else 0.9)
            hair(ax)
            if c_ == 0:
                ax.set_ylabel(lab, fontsize=10)
            if r_ == 0:
                ax.set_title(("α＝3" if p == "α=3" else "二群（広さ派と静けさ派）"), fontsize=10, color=INK)
            ax.set_xticks(range(3)); ax.set_xticklabels([CLAB[c] for c in CONDS], fontsize=10)
            ax.tick_params(labelsize=10)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.075))
    fig.tight_layout()
    save(fig, name)


overview("TP+top3", "fig251_S6_職能2の総覧_TP.png")


# ---------------------------------------------------------------- 図 252：改良版top3・近い戸にしか聞けない条件、退去率に対して
def by_turnover(cond, name):
    b = "改良版top3"
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2))
    xs = np.array(NUS)
    for ax, (q, lab) in zip(axes, (("eta", "η"), ("Vn外", "$V^{*}_{n}$〔外〕"), ("Vn数", "$V^{*}_{n}$〔内〕"))):
        for arm, alab, col in ARMS[:3]:
            for p, ls, mk in (("α=3", "-", "o"), ("二群", "--", "s")):
                if q == "eta":
                    ys = [eta(b, arm, cond, p, nu) for nu in NUS]
                    ax.plot(xs, ys, ls=ls, marker=mk, ms=3.5, color=col, lw=1.3)
                else:
                    xx = [vn(b, arm, cond, p, nu, q) for nu in NUS]
                    ys = [x["v"] for x in xx]
                    ax.plot(xs, ys, ls=ls, marker=mk, ms=3.5, color=col, lw=1.3)
                    ax.vlines(xs, [x["lo"] for x in xx], [x["hi"] for x in xx], color=col, lw=0.7, alpha=0.6)
        ax.set_xticks(xs); ax.set_xticklabels(["なし", "5 %", "10 %", "30 %"], fontsize=10)
        ax.set_xlabel("毎年の退去率", fontsize=10); ax.set_title(lab, fontsize=10, color=INK)
        ax.set_ylim(0 if q != "eta" else 0.2, 1.02 if q == "eta" else 0.65)
        hair(ax); ax.tick_params(labelsize=10)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=c, lw=1.6) for _, _, c in ARMS[:3]] + [Line2D([], [], color=INK2, ls="-", marker="o", ms=3.5),
                                                                       Line2D([], [], color=INK2, ls="--", marker="s", ms=3.5)]
    fig.legend(hs, [a[1] for a in ARMS[:3]] + ["α＝3", "二群"], loc="upper center", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    save(fig, name)


by_turnover("近い戸だけ", "fig252_S6_近い戸だけ_改良版_退去率.png")
by_turnover("費用1倍", "fig253_S6_費用1倍_改良版_退去率.png")


# ---------------------------------------------------------------- 図 254：仲介者が知ったことと〔内〕（TP+top3・改良版top3、入れ替わりなし）
def knowledge(name):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    ax = axes[0]
    pts = []
    for b, mk in (("TP+top3", "o"), ("改良版top3", "s")):
        for arm, alab, col in ARMS[1:3]:
            for cond in CONDS:
                if arm == "仲介者" and cond == "基準":
                    continue
                for p in PATS:
                    for nu in NUS:
                        m = M[key(b, arm, cond, p, nu)]
                        out = vn(b, arm, cond, p, nu, "Vn外")["v"]; cnt = vn(b, arm, cond, p, nu, "Vn数")["v"]
                        if out <= J["epsilon"]:
                            continue                     # 〔外〕がもともと 0 に近いセルは、数えても下がりようがないので除く
                        pts.append((m["nfacts"], out - cnt))
                        ax.scatter(m["nfacts"], out - cnt, s=14, color=col, marker=mk, alpha=0.75, lw=0)
    ax.set_xlabel("仲介者が 30 年目に知っている係数の数\n（1 棟あたり、27 のうち）", fontsize=10)
    ax.set_ylabel("〔外〕−〔内〕", fontsize=10)
    ax.set_title("(a) 作者に含めると下がる分", fontsize=10, color=INK)
    hair(ax, "both"); ax.tick_params(labelsize=10)
    ax = axes[1]
    w = 0.38
    for a_, (arm, alab, col) in enumerate(ARMS[1:3]):
        ys = []
        for cond in CONDS:
            ys.append(np.mean([M[key("TP+top3", arm, cond, p, 0.0)]["nfacts"] for p in PATS]) if not (arm == "仲介者" and cond == "基準") else 0.0)
        ax.bar(np.arange(3) + (a_ - 0.5) * w, ys, w * 0.92, color=col, label=alab)
    ax.set_xticks(range(3)); ax.set_xticklabels([CLAB[c] for c in CONDS], fontsize=10)
    ax.set_ylabel("知っている係数の数（27 のうち）", fontsize=10)
    ax.set_title("(b) TP+top3・入れ替わりなし", fontsize=10, color=INK)
    ax.set_ylim(0, 27); hair(ax); ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, name)


knowledge("fig254_S6_仲介者が知ったことと数え方.png")


# ---------------------------------------------------------------- 図 255：予言の判定
def verdicts(name):
    P = J["predictions"]
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(P) + 0.6))
    ax.axis("off")
    col = {"支持": BLUE, "一部支持": AQUA, "不支持": ORANGE}
    for n, (k, v) in enumerate(P.items()):
        y = len(P) - n
        ax.text(0.0, y, k, fontsize=10, color=INK, va="center")
        ax.text(0.07, y, v["text"][:58] + ("…" if len(v["text"]) > 58 else ""), fontsize=10, color=INK2, va="center")
        ax.text(0.86, y, f"{v['verdict']}（{v['n_ok']}/{v['n_cells']}）", fontsize=10, color=col[v["verdict"]], va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0.3, len(P) + 0.7)
    save(fig, name)


verdicts("fig255_S6_予言の判定.png")
