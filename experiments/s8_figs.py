"""観察 3 の収支で承諾を決める所有者の図（fig411〜fig415）と表。PYTHONPATH=. python3 -m experiments.s8_figs [main|dev]
入力 out/s8/{main|dev}.jsonl。出力 out/s8/fig41*.png（dev は out/s8/dev_fig41*.png）・out/s8/{mode}_table.json
"""
import sys, json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from experiments.style import INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, GREY, DEEP, hair

MODE = sys.argv[1] if len(sys.argv) > 1 else "main"
FN = {"main": "out/s8/main.jsonl", "dev": "out/s8/dev.jsonl"}[MODE]
PRE = "out/s8/" + ("" if MODE == "main" else "dev_")
FS = 10.5
plt.rcParams.update({"font.size": FS, "axes.titlesize": 11.5, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                     "legend.fontsize": FS, "axes.titleweight": "normal", "mathtext.fontset": "stix"})  # ν は数式の字形（stix の斜体）で描き、ラテン文字の v と区別する  # 図の題は本文の側で付けるので、図の中には題を置かず、太字も使わない
R = defaultdict(dict)
for line in open(FN):
    r = json.loads(line)
    R[(r["cond"], r["reg"])][(r["pat"], r["cs"])] = r
OWNS = ["なし", "評判・拒否", "収支・拒否", "収支・取引"]
OCOL = {"なし": GREY, "評判・拒否": ORANGE, "収支・拒否": BLUE, "収支・取引": AQUA}
WORLDS = ["E0", "E1", "E2"]
OLAB = {"なし": "所有者なし", "評判・拒否": "評判で判断・拒否権", "収支・拒否": "収支で判断・拒否権", "収支・取引": "収支で判断・取引の相手"}
WLAB = {"E0": "退去の効果なし", "E1": "小（φ＝0.05）", "E2": "大（φ＝0.15）"}
rng = np.random.default_rng(1)


def m(cond, reg, col):
    v = [r[col] for r in R[(cond, reg)].values() if r[col] == r[col]]
    return float(np.mean(v)) if v else np.nan


def dpair(c1, r1, c2, r2, col, n=2000):
    A = R[(c1, r1)]; B = R[(c2, r2)]
    d = np.array([A[k][col] - B[k][col] for k in A if k in B and A[k][col] == A[k][col] and B[k][col] == B[k][col]])
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def grouped(col, ylabel, title, fname, vsF=False, regs=("TP+top3", "改良版top3")):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.6), sharey=True)
    for a, nu in enumerate((0.10, 0.30)):
        for b, rg in enumerate(regs):
            ax = axs[a, b]
            w = 0.2
            for k, own in enumerate(OWNS):
                ys, lo, hi = [], [], []
                for wd in WORLDS:
                    c = f"{wd}・{own} ν={nu:.2f}"
                    if vsF:
                        mu, l_, h_ = dpair(c, rg, f"F ν={nu:.2f}", "F", col)
                    else:
                        mu = m(c, rg, col); l_ = h_ = mu
                    ys.append(mu); lo.append(mu - l_); hi.append(h_ - mu)
                xx = np.arange(3) + (k - 1.5) * w
                ax.bar(xx, ys, width=w - 0.02, color=OCOL[own], label=OLAB[own])
                if vsF:
                    ax.errorbar(xx, ys, yerr=[lo, hi], fmt="none", ecolor=INK, lw=1, capsize=2)
            ax.axhline(0, color=MUTED, lw=1)
            ax.set_xticks(range(3)); ax.set_xticklabels(["効果なし", "小", "大"])
            ax.set_title(f"{rg}・$\\nu$＝{nu:.2f}", fontsize=FS)
            hair(ax)
            if b == 0:
                ax.set_ylabel(ylabel)
            if a == 1:
                ax.set_xlabel("退去の効果（住み心地が良いほど出ていかない）")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(f"{PRE}{fname}.png", dpi=200); plt.close(fig)


def sens():
    names = ["基準", "λ=0.25", "λ=1", "v1=0", "v1=0.06", "c_turn=0.4", "c_turn=1", "v0=0.35", "30年読み", "R∅", "費用1倍", "E2φ", "E2ψ"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 5.6), sharey=True)
    out = {}
    for ax, rg in zip(axs, ("TP+top3", "改良版top3")):
        for k, nu in enumerate((0.10, 0.30)):
            ys = []
            for nm in names:
                if nm == "基準":
                    c = f"E2・収支・拒否 ν={nu:.2f}"
                elif nm in ("E2φ", "E2ψ"):
                    c = f"{nm}・収支・拒否 ν={nu:.2f}"
                else:
                    c = f"E2・収支・拒否・{nm} ν={nu:.2f}"
                ys.append(m(c, rg, "eta"))
            out[f"{rg} ν={nu:.2f}"] = dict(zip(names, ys))
            yy = np.arange(len(names))[::-1]
            ax.barh(yy + (0.2 if k == 0 else -0.2), ys, height=0.38, color=BLUE if k == 0 else DEEP, label=f"$\\nu$＝{nu:.2f}")
        LAB = {"基準": "基準", "λ=0.25": "家賃に移る割合 0.25", "λ=1": "家賃に移る割合 1", "v1=0": "散らばりの空室 0",
               "v1=0.06": "散らばりの空室 0.06", "c_turn=0.4": "募集の費用 0.4", "c_turn=1": "募集の費用 1", "v0=0.35": "基準の空室 0.35 年",
               "30年読み": "1 単位を 30 年分と読む", "R∅": "原状回復なし（R∅）", "費用1倍": "工事費 1 倍", "E2φ": "定着の効果だけ",
               "E2ψ": "拒まれた世帯の退去だけ"}
        ax.set_yticks(np.arange(len(names))[::-1]); ax.set_yticklabels([LAB[n] for n in names])
        ax.set_title(rg, fontsize=FS); ax.set_xlabel("到達度 η"); hair(ax, "x")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"{PRE}fig415.png", dpi=200); plt.close(fig)
    return out


def table():
    T = {}
    for (c, rg), d in R.items():
        T[f"{c}|{rg}"] = {col: m(c, rg, col) for col in ["eta", "L", "L_rent", "exit_rate", "moves", "vetoed", "owner_in", "o_rent", "o_vac", "o_turn", "o_net", "changed"]}
    for nu in (0.10, 0.30):
        for wd in WORLDS:
            for own in OWNS:
                for rg in ("U", "TP+top3", "改良版top3"):
                    c = f"{wd}・{own} ν={nu:.2f}"
                    if (c, rg) in R:
                        T[f"{c}|{rg}"]["dO_vsF"] = dpair(c, rg, f"F ν={nu:.2f}", "F", "o_net")
                        T[f"{c}|{rg}"]["dExit_vsF"] = dpair(c, rg, f"F ν={nu:.2f}", "F", "exit_rate")
    return T


if __name__ == "__main__":
    grouped("eta", "到達度 η", "所有者の型と退去の効果ごとの到達度 η", "fig411")
    grouped("o_net", "所有者の 30 年の収支の差（㎡ 換算）", "所有者の実現した収支：F の建物との差", "fig412", vsF=True)
    grouped("L", "取り残し率 L", "取り残し率 L（30 年目の世帯）", "fig413")
    grouped("exit_rate", "退去率の差（年あたり）", "実現した退去率：F の建物との差", "fig414", vsF=True)
    S = sens()
    json.dump(dict(table=table(), sens=S), open(f"{PRE}table.json", "w"), ensure_ascii=False, indent=0)
    print("done")
