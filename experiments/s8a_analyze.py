"""観察 3 の世帯の帳簿の集計と図（fig401〜fig405）。入力は experiments/s8a_ledger.py の出力。
PYTHONPATH=. python3 -m experiments.s8a_analyze  → out/s8a/summary.json・tables.md、図 out/s8a/fig40*.png
金額はすべて ㎡ 換算（1 単位＝平均的な世帯が広さ 1 ㎡ を 10 年使う価値。円への換算は試算の側で行う）。
正味＝満足の差（F に住んだ場合との差、㎡ 換算）− 払った補償 ＋ 受け取った補償 − 工事費・戻す費用・手間・所有者への支払い ＋ 所有者からの補償。
"""
import json, gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiments.style import INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, GREY, DEEP, hair
from trade import s0, s8
from experiments.explore import composition
import experiments.s1_main as S1
import experiments.s3_rent as S3

OUT = "out/s8a"
FS = 10.5
plt.rcParams.update({"font.size": FS, "axes.titlesize": 11.5, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                     "legend.fontsize": FS, "axes.titleweight": "normal", "mathtext.fontset": "stix"})  # ν は数式の字形（stix の斜体）で描き、ラテン文字の v と区別する  # 図の題は本文の側で付けるので、図の中には題を置かず、太字も使わない
EPS = 1e-9


def load():
    h1 = pd.read_csv(f"{OUT}/households_obs1.csv.gz")
    h2 = pd.read_csv(f"{OUT}/households_obs2.csv.gz")
    h = pd.concat([h1, h2], ignore_index=True)
    h["paid"] = h.paid_comp + h.paid_recon
    h["recv"] = h.recv_comp + h.recv_recon
    h["works"] = h.cost + h.restore + h.tcost
    h["cell"] = h.src + "|" + h.cond + "|" + h.reg
    return h


def summarize(h):
    out = {}
    for cell, x in h.groupby("cell", sort=False):
        q = x.net.quantile([0.25, 0.5, 0.75])
        neg = x[x.net < -EPS]
        row = dict(n=len(x), n_runs=x.groupby(["pat", "cs", "seed"]).ngroups,
                   net_mean=x.net.mean(), net_q1=q[0.25], net_med=q[0.5], net_q3=q[0.75], net_min=x.net.min(), net_max=x.net.max(),
                   share_paid=(x.paid > EPS).mean(), share_recv=(x.recv > EPS).mean(), share_neg=(x.net < -EPS).mean(),
                   share_dSneg=(x.dS < -EPS).mean(), share_zero=(x.net.abs() <= 1e-6).mean(),
                   share_prop=(x.n_prop > 0).mean(),
                   paid_mean=x.paid.mean(), recv_mean=x.recv.mean(), works_mean=x.works.mean(), dS_mean=x.dS.mean(),
                   opay_bldg=x.groupby(["pat", "cs", "seed"]).opay.sum().mean(), refund_bldg=x.groupby(["pat", "cs", "seed"]).refund.sum().mean(),
                   floor={int(f): float(v) for f, v in x.groupby("floor").net.mean().items()},
                   floor_neg={int(f): float((v < -EPS).mean()) for f, v in x.groupby("floor").net},
                   type={t: float(v) for t, v in x.groupby("type").net.mean().items()},
                   stay={int(s): float(v) for s, v in x.groupby("stay").net.mean().items()},
                   stay_neg={int(s): float((v < -EPS).mean()) for s, v in x.groupby("stay").net},
                   years_med=float(x.years.median()))
        if len(neg):
            row["neg_typical"] = dict(n=len(neg), dS=neg.dS.median(), paid=neg.paid.median(), recv=neg.recv.median(), works=neg.works.median(),
                                      net=neg.net.median(), proposer=(neg.n_prop > 0).mean(), stay=neg.stay.mean(),
                                      floor={int(f): int(c) for f, c in neg.floor.value_counts().items()},
                                      cause_dS=float(((neg.dS < -EPS) & (neg.recv <= EPS)).mean()),
                                      cause_works=float((neg.works > EPS).mean()), years_med=float(neg.years.median()))
        out[cell] = row
    return out


NAME1 = {"U": "U（単独）", "V50": "V50（過半の投票）", "V75": "V75（3/4 の投票）", "TR+": "TR+（物差しで補償）",
         "TP+": "TP+（言い値で補償）", "TP+貸清再": "改良版（貸し借り・精算・再同意）"}
COL1 = {"U": ORANGE, "V50": ORANGE, "V75": ORANGE, "TR+": AQUA, "TP+": BLUE, "TP+貸清再": DEEP}


def strip(ax, data, labels, colors, title, xlim=None):
    """世帯の正味の分布：点（間引いて）、箱（四分位）、中央値の線。負の世帯の割合を右に書く"""
    rng = np.random.default_rng(0)
    for k, (v, lab, c) in enumerate(zip(data, labels, colors)):
        y = len(data) - 1 - k
        vv = v if len(v) <= 1500 else rng.choice(v, 1500, replace=False)
        ax.scatter(vv, y + rng.uniform(-0.28, 0.28, len(vv)), s=4, color=c, alpha=0.25, lw=0)
        q1, md, q3 = np.quantile(v, [0.25, 0.5, 0.75])
        ax.plot([q1, q3], [y, y], color=INK, lw=5, solid_capstyle="butt", alpha=0.55)
        ax.plot([md, md], [y - 0.35, y + 0.35], color=INK, lw=2)
        neg = np.mean(v < -EPS)
        ax.text(1.01, y, f"{neg*100:.0f} %", transform=ax.get_yaxis_transform(), va="center", fontsize=FS, color=INK2)
    ax.set_yticks(range(len(data))); ax.set_yticklabels(labels[::-1])
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("世帯の正味（㎡ 換算、F の建物に住んだ場合との差）")
    ax.text(1.01, 1.0, "正味が負", transform=ax.transAxes, fontsize=FS, color=INK2, va="bottom")
    # 図の題は本文の側で付ける（図の中に置かない）
    if xlim:
        ax.set_xlim(*xlim)
    hair(ax, "x")


def fig401(h):
    x = h[h.src == "obs1"]
    regs = ["U", "V50", "V75", "TR+", "TP+", "TP+貸清再"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    strip(ax, [x[x.reg == r].net.values for r in regs], [NAME1[r] for r in regs], [COL1[r] for r in regs],
          "世帯ごとの 30 年の正味（観察 1 の主条件、5,400 世帯ずつ）", (-40, 40))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig401.png", dpi=200); plt.close(fig)


def fig402(h):
    """階ごとの平均の正味と、満足の差・補償・工事費の内訳（観察 1）"""
    x = h[h.src == "obs1"]
    regs = ["U", "V75", "TR+", "TP+"]
    fig, axs = plt.subplots(1, len(regs), figsize=(7.2, 4.2), sharey=True, sharex=True)
    for ax, r in zip(axs, regs):
        g = x[x.reg == r].groupby("floor")
        fl = [3, 2, 1]
        dS = [g.dS.mean()[f] for f in fl]; rc = [g.recv.mean()[f] for f in fl]; pd_ = [-g.paid.mean()[f] for f in fl]
        wk = [-g.works.mean()[f] for f in fl]; net = [g.net.mean()[f] for f in fl]
        yy = np.arange(3)[::-1]
        ax.barh(yy + 0.2, dS, height=0.36, color=GREY, label="満足の差")
        ax.barh(yy - 0.2, rc, height=0.36, color=AQUA, label="受け取った補償")
        ax.barh(yy - 0.2, pd_, height=0.36, color=ORANGE, label="払った補償")
        ax.scatter(net, yy, marker="D", s=40, color=INK, zorder=4, label="正味")
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(yy); ax.set_yticklabels([f"{f} 階" for f in fl])
        ax.set_title(r if r != "TP+貸清再" else "改良版", fontsize=FS)
        hair(ax, "x")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(f"{OUT}/fig402.png", dpi=200); plt.close(fig)


def fig403(h):
    x = h[h.src == "obs2"]
    rows = [("T|幾何 ν=0.00", "TP+top3", "TP+top3・入れ替わりなし"), ("T|幾何 ν=0.10", "TP+top3", "TP+top3・$\\nu$＝0.10"),
            ("T|幾何 ν=0.30", "TP+top3", "TP+top3・$\\nu$＝0.30"), ("T|幾何 ν=0.10", "改良版top3", "改良版top3・$\\nu$＝0.10"),
            ("T|幾何 ν=0.30", "改良版top3", "改良版top3・$\\nu$＝0.30"),
            ("R|R∅ 費用あり ν=0.30", "TP+top3", "TP+top3・R∅・費用"), ("R|R0 費用あり ν=0.30", "TP+top3", "TP+top3・R0・費用"),
            ("R|R0 費用あり ν=0.30", "改良版top3", "改良版top3・R0・費用"), ("R|Rdk10 費用あり ν=0.30", "改良版top3", "改良版top3・Rdk10・費用"),
            ("R|R0 費用あり ν=0.30", "U", "U・R0・費用")]
    data = [x[(x.cond == c) & (x.reg == r)].net.values for c, r, _ in rows]
    cols = [BLUE if r == "TP+top3" else (DEEP if r == "改良版top3" else ORANGE) for _, r, _ in rows]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    strip(ax, data, [lab for _, _, lab in rows], cols, "入れ替わり・原状回復のもとでの世帯の正味（観察 2 の条件）", (-25, 30))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig403.png", dpi=200); plt.close(fig)


def rep_building(h, src, cond, reg):
    """代表の 1 棟：α＝3 の局のうち、棟の正味の合計が、その条件の局の中央値に最も近いもの"""
    x = h[(h.src == src) & (h.cond == cond) & (h.reg == reg) & (h.pat == "α=3")]
    tot = x.groupby(["cs", "seed"]).net.sum()
    med = tot.median()
    cs, sd = (tot - med).abs().idxmin()
    return int(cs), int(sd), float(tot[(cs, sd)]), float(med)


def rerun(src, cond, reg, cs, sd):
    W = composition("α=3", cs)
    if src == "obs1":
        rr, kw = S1.RULES[reg]
        return s8.run(W, rr, 1000 * cs + sd, **kw), W
    g, cname = cond.split("|", 1)
    spec = [sp for (gg, c, rg, sp) in S3.conds() if gg == g and c == cname and rg == reg][0]
    rr, kw0 = S3.REG[reg]
    kw = dict(kw0); kw.update(S3.kwargs_of(spec, W, "α=3"))
    return s8.run(W, rr, cs, **kw), W


def ledger_table(r):
    """年ごとの帳簿の行：手（提案した世帯、部位、補償の授受、工事費）と入れ替わり"""
    names = {h: f"{s0.UNITS[v['unit']]}-{k}" for k, (h, v) in enumerate(sorted(r["hh"].items()))}
    nth = {}
    for h, v in sorted(r["hh"].items()):
        u = v["unit"]; nth[u] = nth.get(u, 0) + 1
        names[h] = f"{s0.UNITS[u]}" + ("" if nth[u] == 1 else f"（{nth[u]} 人目）")
    rows = []
    by_t = {}
    for (t, kd, f, to, amt, part, med) in r["ledger"]:
        by_t.setdefault((t, part), []).append((kd, f, to, amt))
    for ev in r["events"]:
        if not ev.get("ok") or ev.get("part") is None or ev.get("drafter") not in ("self", "ai"):
            continue
        t = ev["t"]; p = ev["part"]
        hid = [h for h, v in r["hh"].items() if v["unit"] == ev["i"] and v["t_in"] <= t and (v["t_out"] is None or t <= v["t_out"])][0]
        flows = by_t.get((t, (p[0], int(p[1]))), [])
        comp = [f"{names[to]} へ {amt:.2f}" if f == hid else f"{names[f]} から {amt:.2f}" for kd, f, to, amt in flows if kd == "comp"]
        cost = sum(amt for kd, f, to, amt in flows if kd == "cost" and f == hid)
        harmed = [f"{s0.UNITS[j]} {ev['dS'][j]/s0.MU:+.2f}" for j in range(9) if j != ev["i"] and abs(ev["dS"][j]) > 1e-12]
        rows.append(dict(t=t + 1, who=names[hid], move=ev["move"], gain=ev["dS"][ev["i"]] / s0.MU, others="、".join(harmed) or "—",
                         comp="、".join(comp) or "—", cost=cost))
    return rows, names


def fig404(r, title, fn):
    """代表の棟の世帯ごとの帳簿（積み上げの横棒）"""
    hh = sorted(r["hh"].items(), key=lambda kv: (kv[1]["unit"], kv[1]["t_in"]))
    acc = {h: dict(recv=0.0, paid=0.0, works=0.0) for h, _ in hh}
    for (t, kd, f, to, amt, part, med) in r["ledger"]:
        if kd in ("comp", "recon"):
            acc[f]["paid"] += amt; acc[to]["recv"] += amt
        elif kd in ("cost", "restore", "recon_cost", "tcost", "opay"):
            acc[f]["works"] += amt
        elif kd == "refund":
            acc[to]["recv"] += amt
    labs, dS, rc, pdv, wk, net = [], [], [], [], [], []
    nth = {}
    for h, v in hh:
        u = v["unit"]; nth[u] = nth.get(u, 0) + 1
        S_end = float(r["S"][u]) if v["t_out"] is None else v["S_exit"]
        d = (S_end - v["SF"]) / s0.MU
        yrs = f"{v['t_in']+1}〜{(v['t_out']+1) if v['t_out'] is not None else 30} 年"
        labs.append(f"{s0.UNITS[u]}" + ("" if nth[u] == 1 else f" {nth[u]} 人目") + f"（{yrs}）")
        dS.append(d); rc.append(acc[h]["recv"]); pdv.append(-acc[h]["paid"]); wk.append(-acc[h]["works"])
        net.append(d + acc[h]["recv"] - acc[h]["paid"] - acc[h]["works"])
    n = len(labs)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * n + 1.6))
    yy = np.arange(n)[::-1]
    ax.barh(yy + 0.2, dS, height=0.36, color=GREY, label="満足の差")
    ax.barh(yy - 0.2, rc, height=0.36, color=AQUA, label="受け取った補償")
    ax.barh(yy - 0.2, pdv, height=0.36, color=ORANGE, label="払った補償")
    left = np.minimum(pdv, 0)
    ax.barh(yy - 0.2, wk, left=left, height=0.36, color="#a32d2d", label="工事費など")
    ax.scatter(net, yy, marker="D", s=36, color=INK, zorder=4, label="正味")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(yy); ax.set_yticklabels(labs)
    ax.set_xlabel("㎡ 換算")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12 if n < 12 else -0.06), ncol=5, frameon=False, handletextpad=0.3, columnspacing=0.8)
    hair(ax, "x")
    fig.tight_layout(); fig.savefig(f"{OUT}/{fn}.png", dpi=200); plt.close(fig)
    return [dict(lab=l, dS=a, recv=b, paid=-c, works=-w, net=z) for l, a, b, c, w, z in zip(labs, dS, rc, pdv, wk, net)]


def fig405(S):
    """払った世帯・受け取った世帯・正味が負の世帯の割合（決め方と条件ごと）"""
    cells = [("obs1|主条件|U", "U"), ("obs1|主条件|V75", "V75"), ("obs1|主条件|TR+", "TR+"), ("obs1|主条件|TP+", "TP+"),
             ("obs2|T|幾何 ν=0.00|TP+top3", "TP+top3・入替なし"), ("obs2|T|幾何 ν=0.10|TP+top3", "TP+top3・$\\nu$＝0.10"),
             ("obs2|T|幾何 ν=0.30|TP+top3", "TP+top3・$\\nu$＝0.30"), ("obs2|T|幾何 ν=0.30|改良版top3", "改良版top3・$\\nu$＝0.30"),
             ("obs2|R|R0 費用あり ν=0.30|改良版top3", "改良版top3・R0・費用"), ("obs2|R|Rdk10 費用あり ν=0.30|改良版top3", "改良版top3・Rdk10・費用")]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    xx = np.arange(len(cells)); w = 0.27
    for k, (key, lab, c) in enumerate([("share_paid", "補償を払った", ORANGE), ("share_recv", "補償を受け取った", AQUA), ("share_neg", "正味が負", "#a32d2d")]):
        ax.bar(xx + (k - 1) * w, [S[c_][key] * 100 for c_, _ in cells], width=w - 0.03, color=c, label=lab)
    ax.set_xticks(xx); ax.set_xticklabels([l for _, l in cells], rotation=35, ha="right")
    ax.set_ylabel("世帯の割合（%）"); ax.set_ylim(0, 75)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    hair(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig405.png", dpi=200); plt.close(fig)


def main():
    h = load()
    S = summarize(h)
    reps = {}
    for key, (src, cond, reg, fn, title) in {
            "obs1_TP+": ("obs1", "主条件", "TP+", "fig404a", "代表の 1 棟の帳簿（TP+、入れ替わりなし）"),
            "obs2_改良版top3_ν0.10": ("obs2", "T|幾何 ν=0.10", "改良版top3", "fig404b", "代表の 1 棟の帳簿（改良版・上位 3 手、$\\nu$＝0.10）"),
            "obs2_TP+top3_ν0.10": ("obs2", "T|幾何 ν=0.10", "TP+top3", "fig404c", "代表の 1 棟の帳簿（TP+・上位 3 手、$\\nu$＝0.10）")}.items():
        cs, sd, tot, med = rep_building(h, src, cond, reg)
        r, W = rerun(src, cond, reg, cs, sd)
        rows, names = ledger_table(r)
        bars = fig404(r, title + f"　cseed {cs}", fn)
        reps[key] = dict(cs=cs, seed=sd, total=tot, median_total=med, moves=rows, households=bars,
                         W=[[round(float(v), 3) for v in w] for w in W], n_exit=r["n_exit"])
    fig401(h); fig402(h); fig403(h); fig405(S)
    json.dump(dict(cells=S, reps=reps), open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("done")


if __name__ == "__main__":
    main()
