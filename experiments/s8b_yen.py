"""観察 3：世帯の 30 年の帳簿を円に換算した試算と、住み替え（6 ㎡ 広い部屋への引っ越し）との比較。
入力：out/s8a/households_obs1.csv.gz・households_obs2.csv.gz・summary.json（s8a_ledger・s8a_analyze の出力）。手ごとの補償は、
      s8a_ledger と同じ引数・同じ種で s8.run を再実行して取り出す（模型の値は s8a_ledger の記録と一致することを確認する）。
出力：out/s8b/yen.json（表の値）、図 out/s8b/fig421〜fig425.png
PYTHONPATH=. python3 -m experiments.s8b_yen

換算（10 年の読み、主）：1 単位 ＝ 民営借家の 1 ㎡ あたり月額家賃 p × 120 か月（割引なし）
  根拠：s8 の所有者の単価（50 ㎡ の年の家賃 r0＝5 単位、満足の差を年額に直す Hv＝10 年）と s3 の見通し H＝10 年。
  感度：30 年の読み（× 360）。住んだ年で按分する読み（満足は 1 ㎡ を 1 年 ＝ p × 12、お金は 10 年の読み）。
金額はすべて税抜・割引なし。
単価の幅：
  総務省『令和5年住宅・土地統計調査』住宅及び世帯に関する基本集計 表122-2-2（e-Stat 統計表ID 0004021522）の
  民営借家（専用住宅）の延べ面積 1 ㎡ 当たり家賃（家賃 0 円を含む）で、全国 1,485 円を中央、東京都 2,529 円を上側とする。
  埼玉県の民営借家は同表で 1,507 円と全国より高いので、郊外の値は幅の下側に使わない。
  感度：東京 23 区の分譲マンションの募集賃料 4,866 円（東京カンテイ 2025 年 10 月、参考の上側）と、
        限界の 1 ㎡（全国 × 0.45、藤澤・乾・廣松 2021 のヘドニック推定、参考の下側）。
"""
import json, os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from experiments.style import INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, GREY, DEEP, hair
from trade import s0, s8
from experiments.explore import composition
import experiments.s1_main as S1
import experiments.s3_rent as S3
import experiments.s8a_analyze as A

IN = "out/s8a"
OUT = "out/s8b"
FS = 10.5
plt.rcParams.update({"font.size": FS, "axes.titlesize": 11.5, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                     "legend.fontsize": FS, "axes.titleweight": "normal"})  # 図の題は本文の側で付けるので、図の中には題を置かず、太字も使わない
EPS = 1e-9
MAN = 1e4   # 万円

# ------------------------------------------------------------------ 単価
P = {"全国": 1485, "東京都": 2529, "23区募集": 4866}     # 円/㎡・月（表122-2-2 の民営借家：全国が中央、東京都が上側。23 区の募集賃料は参考）
PRICE_KEYS = ["全国", "東京都", "23区募集"]
P_MARG = round(1485 * 0.45)                                              # 限界の 1 ㎡（ヘドニックの係数 −0.011、50 ㎡）
P_MAIN = "全国"
EXT_M2 = 6.0


def unit_yen(p, years=10):
    return p * 12 * years


def lump(p, case="中"):
    """住み替え一回の一時金（敷金は返還されうるので入れない）：礼金＋仲介手数料（56 ㎡ の家賃の月数）＋引っ越し代"""
    rent56 = 56 * p
    if case == "低":
        return 0.55 * rent56 + 50_000          # 礼金 0、仲介 0.55 か月、近距離の引っ越し
    if case == "高":
        return (1.19 + 1.1) * rent56 + 130_000  # 礼金 1.19、仲介 1.1、3 人の繁忙期に近い額
    return (1.0 + 1.1) * rent56 + 100_000


def drent_year(p):
    return EXT_M2 * p * 12


# 工事費（円/件）：推奨値と幅
REAL = {"既製サンルーム": (1_000_000, 500_000, 1_500_000), "居室の増築": (2_500_000, 1_500_000, 4_000_000),
        "戸境収納": (200_000, 100_000, 300_000), "開口（乾式）": (300_000, 150_000, 500_000), "開口（壁式 RC）": (1_500_000, 500_000, 3_000_000),
        "戻す：張り出し": (200_000, 100_000, 400_000), "戻す：収納": (100_000, 50_000, 100_000),
        "戻す：開口（乾式）": (200_000, 50_000, 300_000), "戻す：開口（壁式 RC）": (500_000, 300_000, 1_000_000)}
MODEL_COST = {"張り出し": 1.0, "収納": 0.25, "開口": 0.25}
MODEL_RCOST = {"張り出し": 0.5, "収納": 0.1, "開口": 0.1}


# ------------------------------------------------------------------ 手ごとの補償（再実行）
def run_one(a):
    src, cond, reg, pat, cs, sd = a
    W = composition(pat, cs)
    if src == "obs1":
        rr, kw = S1.RULES[reg]
        r = s8.run(W, rr, 1000 * cs + sd, **kw)
    else:
        g, cname = cond.split("|", 1)
        spec = [sp for (gg, c, rg, sp) in S3.conds() if gg == g and c == cname and rg == reg][0]
        rr, kw0 = S3.REG[reg]
        kw = dict(kw0); kw.update(S3.kwargs_of(spec, W, pat))
        r = s8.run(W, rr, cs, **kw)
    hid_at = lambda u, t: [h for h, v in r["hh"].items() if v["unit"] == u and v["t_in"] <= t and (v["t_out"] is None or t <= v["t_out"])][0]
    moves = []
    for ev in r["events"]:
        if not ev.get("ok") or ev.get("part") is None or ev.get("drafter") not in ("self", "ai"):
            continue
        p = ev["part"]; t = ev["t"]; i = ev["i"]
        hid = hid_at(i, t)
        comp = sum(amt for (tt, kd, f, to, amt, part, med) in r["ledger"] if tt == t and kd == "comp" and f == hid and part == (p[0], int(p[1])))
        kind = {"e": "張り出し", "k": "収納", "o": "開口"}[p[0]]
        mv = dict(src=src, cond=cond, reg=reg, pat=pat, cs=cs, seed=sd, t=t, unit=s0.UNITS[i], kind=kind, type=ev["type"],
                  nv=ev.get("nv"), comp=comp, cost=float(ev.get("cost", 0.0)), self_part=(p[0] == "e" and int(p[1]) == i),
                  n_paid=int(sum(1 for x in np.atleast_1d(ev.get("pay", [])) if x > EPS)))
        # 張り出しを付けた手：同じ世帯が、その後に同じ部位を残すために再同意で払った額と、戻す費用
        if p[0] == "e" and ev.get("nv") == 1:
            mv["recon_keep"] = sum(amt for (tt, kd, f, to, amt, part, med) in r["ledger"]
                                   if tt > t and kd == "recon" and f == hid and part == (p[0], int(p[1])))
            mv["restore_paid"] = sum(amt for (tt, kd, f, to, amt, part, med) in r["ledger"]
                                     if tt >= t and kd in ("restore", "recon_cost") and f == hid and part == (p[0], int(p[1])))
            v = r["hh"][hid]
            mv["years_after"] = (30 if v["t_out"] is None else v["t_out"] + 1) - t
        moves.append(mv)
    return moves


MOVE_CELLS = [("obs1", "主条件", "TP+"), ("obs1", "主条件", "TR+"),
              ("obs2", "T|幾何 ν=0.30", "TP+top3"), ("obs2", "T|幾何 ν=0.30", "改良版top3"),
              ("obs2", "R|R0 費用あり ν=0.30", "TP+top3"), ("obs2", "R|R0 費用あり ν=0.30", "改良版top3"),
              ("obs2", "W|個人・取引 ν=0.10", "TP+top3")]


def collect_moves():
    tasks = []
    for src, cond, reg in MOVE_CELLS:
        for pat in ["α=3", "二群"]:
            if src == "obs1":
                tasks += [(src, cond, reg, pat, cs, sd) for cs in S1.C_MAIN for sd in range(S1.NS_MAIN)]
            else:
                tasks += [(src, cond, reg, pat, cs, cs) for cs in range(400, 460)]
    with Pool(2) as pool:
        res = pool.map(run_one, tasks, chunksize=8)
    return pd.DataFrame([m for ms in res for m in ms])


# ------------------------------------------------------------------ 世帯の表
def load_hh():
    h = A.load()
    h["recv_all"] = h.recv + h.refund
    h["works_all"] = h.works
    # 住んだ年で按分する読み：満足は Σ_年 (S_y − S_F)/μ を 10 で割る（1 ㎡ を 1 年 ＝ 0.1 単位）。お金はそのまま
    h["dS_exp"] = h.dS_avg * h.years / 10.0
    h["net_exp"] = h.dS_exp + h.money
    return h


DIST = [("obs1|主条件|U", "U（単独・補償なし）"), ("obs1|主条件|V75", "V75（3/4 の投票）"), ("obs1|主条件|TR+", "TR+（物差しで補償）"),
        ("obs1|主条件|TP+", "TP+（言い値で補償）"),
        ("obs2|T|幾何 ν=0.30|TP+top3", "TP+top3・入れ替わり 0.30"), ("obs2|T|幾何 ν=0.30|改良版top3", "改良版top3・入れ替わり 0.30"),
        ("obs2|R|R0 費用あり ν=0.30|U", "U・原状回復・工事費"), ("obs2|R|R0 費用あり ν=0.30|TP+top3", "TP+top3・原状回復・工事費"),
        ("obs2|R|R0 費用あり ν=0.30|改良版top3", "改良版top3・原状回復・工事費"),
        ("obs2|W|個人・取引 ν=0.10|TP+top3", "TP+top3・所有者と取引 0.10")]


def dist_table(h):
    out = {}
    for key, lab in DIST:
        x = h[h.cell == key]
        nb = x.groupby(["pat", "cs", "seed"]).ngroups
        row = dict(label=lab, n=len(x), n_bldg=nb)
        for col, tag in [("net", "R10"), ("net_exp", "按分")]:
            q = x[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
            row[tag] = dict(q10=q[0], q1=q[1], med=q[2], q3=q[3], q90=q[4], min=x[col].min(), max=x[col].max(), mean=x[col].mean(),
                            neg=float((x[col] < -EPS).mean()), neg_big=float((x[col] < -1.0).mean()))
        row["paid_mean"] = x.paid.mean(); row["recv_mean"] = x.recv_all.mean(); row["works_mean"] = x.works.mean()
        row["opay_mean"] = x.opay.mean()
        pp = x[x.paid > EPS].paid; rr = x[x.recv > EPS].recv
        row["paid_payers"] = dict(share=float((x.paid > EPS).mean()), q1=pp.quantile(0.25) if len(pp) else 0, med=pp.median() if len(pp) else 0,
                                  q3=pp.quantile(0.75) if len(pp) else 0)
        row["recv_receivers"] = dict(share=float((x.recv > EPS).mean()), q1=rr.quantile(0.25) if len(rr) else 0, med=rr.median() if len(rr) else 0,
                                     q3=rr.quantile(0.75) if len(rr) else 0)
        row["bldg_comp_total"] = float(x.groupby(["pat", "cs", "seed"]).paid.sum().mean())
        row["bldg_works_total"] = float(x.groupby(["pat", "cs", "seed"]).works.sum().mean())
        row["years_med"] = float(x.years.median())
        out[key] = row
    return out


# ------------------------------------------------------------------ 代表の 1 棟
REPS = [("a", "obs1", "主条件", "TP+", "観察 1・TP+（入れ替わりなし、工事費なし）"),
        ("b", "obs2", "T|幾何 ν=0.10", "改良版top3", "観察 2・改良版top3（入れ替わり 0.10、工事費なし）"),
        ("c", "obs2", "R|R0 費用あり ν=0.30", "改良版top3", "観察 2・改良版top3（入れ替わり 0.30、原状回復 R0、工事費あり）")]


def rep_table(h, src, cond, reg):
    cs, sd, tot, med = A.rep_building(h, src, cond, reg)
    x = h[(h.src == src) & (h.cond == cond) & (h.reg == reg) & (h.pat == "α=3") & (h.cs == cs) & (h.seed == sd)].copy()
    order = {u: k for k, u in enumerate(s0.UNITS)}
    x["o"] = x.unit.map(order)
    x = x.sort_values(["o", "t_in"])
    rows = []
    for _, v in x.iterrows():
        lab = v.unit + ("" if v.nth == 1 else f"（{v.nth} 人目）")
        t_out = 30 if pd.isna(v.t_out) else int(v.t_out) + 1
        rows.append(dict(lab=lab, yrs=f"{int(v.t_in)+1}〜{t_out}", years=int(v.years), stay=int(v.stay), type=v.type,
                         paid=v.paid, recv=v.recv_all, works=v.works, opay=v.opay, dS=v.dS, net=v.net, dS_exp=v.dS_exp, net_exp=v.net_exp,
                         n_prop=int(v.n_prop)))
    r, W = A.rerun(src, cond, reg, cs, sd)
    mv, names = A.ledger_table(r)
    return dict(cs=cs, seed=sd, total=tot, median_total=med, rows=rows, moves=mv, n_exit=int(r["n_exit"]))


# ------------------------------------------------------------------ 損益分岐
def breakeven(comp_units, K_yen, p, years=10, lumpcase="中", mode=None, extra_yen=0.0):
    """張り出しの費用と、住み替え（一時金＋6 ㎡ 分の家賃の差×年数）が等しくなる年数（0 未満は 0、成り立たなければ inf）。
    mode=None（一回払い）：費用＝補償 comp_units×1 単位（years の読み）＋工事費 K＋extra。
    mode="annual"（毎年払う）：補償を、隣人の損が続くあいだ毎年払う（年額＝comp_units ㎡ 分の家賃 1 年分）。
       張り出しの得は 6 ㎡ 分なので、年あたりの差し引きは (6 − comp_units) ㎡ 分の家賃。読みの年数によらない。"""
    M = lump(p, lumpcase)
    if mode == "annual":
        margin = (EXT_M2 - comp_units) * p * 12
        if margin <= 0:
            return float("inf")
        return max(0.0, (K_yen + extra_yen - M) / margin)
    C = comp_units * unit_yen(p, years) + K_yen + extra_yen
    return max(0.0, (C - M) / drent_year(p))


def K_cases(p, years=10):
    u = unit_yen(p, years)
    return [("模型の基準（1 単位）", 1.0 * u, 0.5 * u), ("模型の 3 倍（3 単位）", 3.0 * u, 1.5 * u),
            ("既製サンルーム 100 万円", 1_000_000, 200_000), ("居室の増築 250 万円", 2_500_000, 200_000)]   # (名, 工事費, 戻す費用)


VARIANTS = [("一回払い・10 年の読み", "obs1|主条件|TP+", 10, None, False),
            ("毎年払う（損が続くあいだ）", "obs1|主条件|TP+", 10, "annual", False),
            ("一回払い・30 年の読み", "obs1|主条件|TP+", 30, None, False),
            ("入れ替わり 0.30・見通し 10 年・原状回復", "obs2|R|R0 費用あり ν=0.30|改良版top3", 10, None, True)]


# ------------------------------------------------------------------ 図
def fmt_man(v):
    return f"{v/MAN:,.0f}" if abs(v) >= 10 * MAN else f"{v/MAN:,.1f}"


def fig421(Y, mv):
    """総覧：桁の地図。模型の金額（全国の単価の点、全国〜東京都の幅）と、現実の工事費・相場・住み替えの費用を一本の円の目盛に"""
    pl, pm, ph = P["全国"], P["全国"], P["東京都"]
    rows = []   # (群, ラベル, 中央, 低, 高, 色, 四分位(低,高) or None)
    ext = Y["moves"]["obs1|主条件|TP+"]["張り出し（付ける）"]
    ks = Y["moves"]["obs1|主条件|TP+"]["収納・開口"]
    rmd = Y["dist"]["obs1|主条件|TP+"]["recv_receivers"]; pmd = Y["dist"]["obs1|主条件|TP+"]["paid_payers"]
    G1, G2, G3, G4 = "模型の工事費", "現実の工事費", "模型の補償と承諾の対価", "相場と住み替え"
    for lab, u in [("張り出し・基準（1 単位）", 1.0), ("張り出し・3 倍（3 単位）", 3.0), ("収納・開口・基準（0.25 単位）", 0.25),
                   ("収納・開口・3 倍（0.75 単位）", 0.75)]:
        rows.append((G1, lab, u * unit_yen(pm), u * unit_yen(pl), u * unit_yen(ph), GREY, None))
    for lab, key in [("既製サンルーム（6〜7 ㎡）", "既製サンルーム"), ("居室の増築（6 ㎡）", "居室の増築"), ("戸境収納の付け替え", "戸境収納"),
                     ("戸境の開口（乾式の壁）", "開口（乾式）"), ("戸境の開口（壁式 RC）", "開口（壁式 RC）")]:
        c, lo, hi = REAL[key]
        rows.append((G2, lab, c, lo, hi, "#a32d2d", None))
    for lab, d in [("張り出し 1 件の補償", ext), ("収納・開口 1 件の補償", ks)]:
        rows.append((G3, lab, d["med"] * unit_yen(pm), d["med"] * unit_yen(pl), d["med"] * unit_yen(ph), BLUE,
                     (d["q1"] * unit_yen(pm), d["q3"] * unit_yen(pm))))
    rows.append((G3, "1 世帯が 30 年に受け取った補償", rmd["med"] * unit_yen(pm), rmd["med"] * unit_yen(pl), rmd["med"] * unit_yen(ph), BLUE,
                 (rmd["q1"] * unit_yen(pm), rmd["q3"] * unit_yen(pm))))
    rows.append((G3, "1 世帯が 30 年に払った補償", pmd["med"] * unit_yen(pm), pmd["med"] * unit_yen(pl), pmd["med"] * unit_yen(ph), BLUE,
                 (pmd["q1"] * unit_yen(pm), pmd["q3"] * unit_yen(pm))))
    e2 = Y["moves"]["obs2|R|R0 費用あり ν=0.30|改良版top3"]["張り出し（付ける）"]
    rows.append((G3, "張り出し 1 件の補償（入れ替わり・見通し）", e2["med"] * unit_yen(pm), e2["med"] * unit_yen(pl), e2["med"] * unit_yen(ph), BLUE,
                 (e2["q1"] * unit_yen(pm), e2["q3"] * unit_yen(pm))))
    op = Y["owner_pay"]["W|個人・取引 ν=0.10|TP+top3"]["per_move"]
    rows.append((G3, "所有者への承諾の対価（1 手）", op * unit_yen(pm), op * unit_yen(pl), op * unit_yen(ph), BLUE, None))
    rows.append((G4, "日照の慰謝料（借家人、判例）", 350_000, 350_000, 500_000, AQUA, None))
    rows.append((G4, "日照による家屋の価値の低下（所有者）", 500_000, 500_000, 500_000, AQUA, None))
    rows.append((G4, "借地の一部改築の承諾料（6/50 ㎡）", 72_000, 36_000, 144_000, AQUA, None))
    rows.append((G4, "住み替えの一時金", lump(pm), lump(pl, "低"), lump(ph, "高"), ORANGE, None))
    rows.append((G4, "6 ㎡ 広い部屋の家賃の差・10 年", drent_year(pm) * 10, drent_year(pl) * 10, drent_year(ph) * 10, ORANGE, None))
    rows.append((G4, "同・30 年", drent_year(pm) * 30, drent_year(pl) * 30, drent_year(ph) * 30, ORANGE, None))
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * n + 1.9))
    yy = np.arange(n)[::-1] * 1.0
    gi = 0; ys = []; prev = None
    for k, r in enumerate(rows):
        if r[0] != prev and k > 0:
            gi += 1
        prev = r[0]; ys.append(-(k + 0.9 * gi))
    yy = np.array(ys) - min(ys)
    prev = None
    for y, (g, lab, c, lo, hi, col, iq) in zip(yy, rows):
        if iq is not None:
            ax.plot(iq, [y, y], color=col, lw=1.2, alpha=0.6)
        ax.plot([lo, hi], [y, y], color=col, lw=6, alpha=0.35, solid_capstyle="butt")
        ax.scatter([c], [y], color=col, s=34, zorder=4)
        ax.text(min(hi, 2.4e7) * 1.15, y, fmt_man(c), va="center", fontsize=FS, color=INK2)
        if g != prev:
            ax.axhline(y + 1.2, color=GRID, lw=0.8)
            ax.text(0.005, y + 0.55, g, transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=FS, color=INK)
            prev = g
    ax.set_yticks(yy); ax.set_yticklabels([r[1] for r in rows])
    ax.set_xscale("log"); ax.set_xlim(2e4, 6e6)
    ticks = [3e4, 1e5, 3e5, 1e6, 3e6]
    ax.set_xticks(ticks); ax.set_xticklabels(["3 万", "10 万", "30 万", "100 万", "300 万"])
    ax.set_xlabel("円（対数の目盛。右の数字は点の値、万円）")
    leg = [Line2D([], [], marker="o", color=INK2, lw=0, label="点：全国の家賃の単価で換算／推奨値"),
           Line2D([], [], color=INK2, lw=6, alpha=0.35, label="太い帯：全国〜東京都の単価／現実の幅"),
           Line2D([], [], color=INK2, lw=1.2, alpha=0.6, label="細い線：模型の値の四分位（全国）")]
    ax.legend(handles=leg, loc="upper right", bbox_to_anchor=(1.0, -0.06), ncol=1, frameon=False)
    hair(ax, "x")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig421.png", dpi=200); plt.close(fig)


def fig422(h, Y):
    """世帯の 30 年の正味（万円、全国の単価、10 年の読み）。右に正味が負の割合（模型の読み／住んだ年で按分）。
    目盛の外の世帯は端に寄せて描く（−250 万円未満・450 万円超）"""
    u = unit_yen(P[P_MAIN]) / MAN
    LO, HI = -250, 450
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    n = len(DIST)
    for k, (key, lab) in enumerate(DIST):
        y = n - 1 - k
        v = h[h.cell == key].net.values * u
        col = ORANGE if (lab.startswith("U") or lab.startswith("V75")) else (AQUA if lab.startswith("TR") else (DEEP if "改良版" in lab else BLUE))
        vv = v if len(v) <= 1500 else rng.choice(v, 1500, replace=False)
        ax.scatter(np.clip(vv, LO, HI), y + rng.uniform(-0.28, 0.28, len(vv)), s=4, color=col, alpha=0.25, lw=0)
        q1, md, q3 = np.quantile(v, [0.25, 0.5, 0.75])
        ax.plot([q1, q3], [y, y], color=INK, lw=6, solid_capstyle="butt", alpha=0.6)
        ax.plot([md, md], [y - 0.38, y + 0.38], color=INK, lw=2.4)
        d = Y["dist"][key]
        ax.text(1.03, y, f"{d['R10']['neg']*100:.0f} %", transform=ax.get_yaxis_transform(), va="center", fontsize=FS, color=INK2)
        ax.text(1.17, y, f"{d['按分']['neg']*100:.0f} %", transform=ax.get_yaxis_transform(), va="center", fontsize=FS, color=INK2)
    ax.set_yticks(range(n)); ax.set_yticklabels([lab for _, lab in DIST][::-1])
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(LO - 10, HI + 10)
    ax.set_xlabel("世帯の 30 年の正味（万円。F の建物に住んだ場合との差）\n黒い帯：四分位、縦の線：中央値。目盛の外の世帯は端に描く")
    ax.text(1.03, 1.0, "正味が負の割合\n模型の  住んだ年\n読み    で按分", transform=ax.transAxes, fontsize=FS, color=INK2, va="bottom", linespacing=1.1)
    hair(ax, "x")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig422.png", dpi=200); plt.close(fig)


def fig423(Y):
    """損益分岐の総覧：何年住めば、張り出し（補償＋工事費）が住み替え（一時金＋家賃の差）より安くなるか。
    行＝補償の払い方と読み × 工事費、点の形＝家賃の単価の水準。値は補償の中央値、横の細い線は補償の四分位（全国）"""
    BE = pd.DataFrame(Y["breakeven"])
    vnames = [v[0] for v in VARIANTS] + ["平均の性向の隣人・毎年払う"]
    klabs = [k[0] for k in K_cases(P["全国"])]
    MK = {"全国": ("o", BLUE), "東京都": ("s", DEEP), "23区募集": ("D", INK)}
    XMAX = 40
    rows = [(v, k) for v in vnames for k in klabs]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.2, 0.4 * n + 3.0))
    ys, gi, pv = [], 0, None
    for k_, (v, _) in enumerate(rows):
        if v != pv and k_ > 0:
            gi += 1
        pv = v; ys.append(-(k_ + 1.0 * gi))
    yy = np.array(ys, float) - min(ys)
    prev = None
    for y, (v, k) in zip(yy, rows):
        z = BE[(BE.variant == v) & (BE.K == k)]
        zz = z[(z.price == "全国") & (z.q.isin(["q1", "q3"]))]
        if len(zz):
            lo, hi = [min(XMAX + 1.5, t) for t in (zz["T"].min(), zz["T"].max())]
            ax.plot([lo, hi], [y, y], color=BLUE, lw=1.2, alpha=0.6)
        for pk, (m, c) in MK.items():
            t = z[(z.price == pk) & (z.q.isin(["med", "—"]))]["T"].values[0]
            ax.scatter([min(t, XMAX + 1.5)], [y], marker=m, s=34, color=c, zorder=4, clip_on=False)
        tz = z[(z.price == "全国") & (z.q.isin(["med", "—"]))]["T"].values[0]
        ax.text(XMAX + 3.2, y, ("40 年超" if tz > XMAX else f"{tz:.0f} 年"), va="center", fontsize=FS, color=INK2)
        if v != prev:
            ax.axhline(y + 1.0, color=GRID, lw=0.8)
            vl = v.replace("平均の性向の隣人・毎年払う", "隣人が平均の性向・毎年払う")
            ax.text(0.3, y + 0.55, vl, fontsize=FS, color=INK, va="center")
            prev = v
    top = yy.max() + 1.0
    # 平均居住期間：日管協短観 第 28 回（2023 年度）p.18 の全国の値（観察 2 と同じ版）
    for x, lab, ha in [(3.25, "単身 3 年 3 か月", "right"), (5.25, "ファミリー 5 年 3 か月", "left")]:
        ax.axvline(x, color=ORANGE, lw=1, ls=":")
        ax.text(x + (0.3 if ha == "left" else -0.3), top + 0.25, lab, color=ORANGE, fontsize=FS, ha=ha, va="bottom")
    ax.axvline(30, color=MUTED, lw=1, ls="--"); ax.text(30, top + 0.25, "模型の 30 年", color=MUTED, fontsize=FS, ha="center", va="bottom")
    ax.text(XMAX + 3.2, top + 0.25, "全国", color=INK2, fontsize=FS, va="bottom")
    ax.text(0, top + 1.3, "縦の点線：平均居住期間（日管協短観 第 28 回、2023 年度）", color=ORANGE, fontsize=FS, va="bottom")
    ax.set_yticks(yy); ax.set_yticklabels([k for _, k in rows])
    ax.set_xlim(0, XMAX + 2); ax.set_ylim(-0.7, top + 2.4)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("損益分岐の年数（張り出しの後にこれより長く住むなら、張り出しの方が安い）")
    leg = [Line2D([], [], marker=m, color=c, lw=0, label=f"{pk}（月 {P[pk]:,} 円/㎡）") for pk, (m, c) in MK.items()]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.4, -0.07), ncol=2, frameon=False)
    hair(ax, "x")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig423.png", dpi=200); plt.close(fig)


def fig424(Y):
    """損益分岐の例：住む年数ごとの累積の費用（全国の単価、補償は TP+ の中央値の一回払い）"""
    comp = Y["moves"]["obs1|主条件|TP+"]["張り出し（付ける）"]["med"]
    p = P["全国"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    T = np.linspace(0, 30, 301)
    ax.plot(T, (lump(p) + drent_year(p) * T) / MAN, color=ORANGE, lw=2.4, label="住み替え：一時金＋6 ㎡ 分の家賃の差")
    ax.fill_between(T, (lump(p, "低") + drent_year(p) * T) / MAN, (lump(p, "高") + drent_year(p) * T) / MAN, color=ORANGE, alpha=0.15, lw=0)
    cols = [GREY, INK2, BLUE, DEEP]
    for (lab, K, R), c in zip(K_cases(p), cols):
        C = comp * unit_yen(p) + K
        ax.axhline(C / MAN, color=c, lw=1.8, label=f"張り出し：補償 {comp*unit_yen(p)/MAN:.0f} 万円＋{lab}")
        tb = breakeven(comp, K, p)
        if tb <= 30:
            ax.scatter([tb], [C / MAN], color=c, s=36, zorder=5)
            ax.annotate(f"{tb:.0f} 年", (tb, C / MAN), xytext=(4, 6), textcoords="offset points", fontsize=FS, color=c)
    ax.set_xlim(0, 30); ax.set_ylim(0, 360)
    ax.set_xlabel("張り出しの後に住む年数"); ax.set_ylabel("累積の費用（万円）")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    hair(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig424.png", dpi=200); plt.close(fig)


def fig_rep(rep, title, fn):
    """代表の 1 棟の世帯ごとの収支（万円、全国の単価）"""
    u = unit_yen(P[P_MAIN]) / MAN
    rows = rep["rows"]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * n + 1.9))
    yy = np.arange(n)[::-1]
    dS = np.array([r["dS"] for r in rows]) * u; rc = np.array([r["recv"] for r in rows]) * u
    pdv = -np.array([r["paid"] for r in rows]) * u; wk = -np.array([r["works"] + r["opay"] for r in rows]) * u
    net = np.array([r["net"] for r in rows]) * u; nexp = np.array([r["net_exp"] for r in rows]) * u
    ax.barh(yy + 0.2, dS, height=0.36, color=GREY, label="満足の得")
    ax.barh(yy - 0.2, rc, height=0.36, color=AQUA, label="受け取った補償")
    ax.barh(yy - 0.2, pdv, height=0.36, color=ORANGE, label="払った補償")
    ax.barh(yy - 0.2, wk, left=np.minimum(pdv, 0), height=0.36, color="#a32d2d", label="工事費など")
    ax.scatter(net, yy, marker="D", s=36, color=INK, zorder=4, label="正味")
    ax.scatter(nexp, yy, marker="D", s=36, facecolor="white", edgecolor=INK, zorder=4, label="正味（住んだ年で按分）")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(yy); ax.set_yticklabels([f"{r['lab']}（{r['yrs']} 年）" for r in rows])
    ax.set_xlabel("万円（1 単位＝17.8 万円）")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1 if n < 12 else -0.05), ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.8)
    hair(ax, "x")
    fig.tight_layout(); fig.savefig(f"{OUT}/{fn}.png", dpi=200); plt.close(fig)


def fig_rep_unit(rep, title, fn):
    """入れ替わりのある代表の 1 棟：住戸ごとに、そこに住んだ世帯の合計（横棒）と、世帯ひとりずつの正味（小さな点）"""
    u = unit_yen(P[P_MAIN]) / MAN
    rows = rep["rows"]
    units = s0.UNITS
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * 9 + 2.2))
    yy = np.arange(9)[::-1]
    for y, un in zip(yy, units):
        rs = [r for r in rows if r["lab"].startswith(un)]
        dS = sum(r["dS"] for r in rs) * u; rc = sum(r["recv"] for r in rs) * u
        pdv = -sum(r["paid"] for r in rs) * u; wk = -sum(r["works"] + r["opay"] for r in rs) * u
        net = sum(r["net"] for r in rs) * u
        ax.barh(y + 0.22, dS, height=0.3, color=GREY)
        ax.barh(y - 0.12, rc, height=0.3, color=AQUA)
        ax.barh(y - 0.12, pdv, height=0.3, color=ORANGE)
        ax.barh(y - 0.12, wk, left=min(pdv, 0), height=0.3, color="#a32d2d")
        ax.scatter([net], [y + 0.05], marker="D", s=40, color=INK, zorder=5)
        hn = np.array([r["net"] for r in rs]) * u
        ax.scatter(hn, np.full(len(hn), y - 0.4), marker="o", s=16, facecolor="white", edgecolor=INK2, lw=0.8, zorder=4)
    ax.axvline(0, color=MUTED, lw=1)
    lab = []
    for un in units:
        rs = [r for r in rows if r["lab"].startswith(un)]
        lab.append(f"{un}（{len(rs)} 世帯）")
    ax.set_yticks(yy); ax.set_yticklabels(lab)
    ax.set_xlabel("万円（1 単位＝17.8 万円）。棒と黒の菱形は住戸ごとの全世帯の合計")
    leg = [Patch(color=GREY, label="満足の得"), Patch(color=AQUA, label="受け取った補償"), Patch(color=ORANGE, label="払った補償"),
           Patch(color="#a32d2d", label="工事費・戻す費用"), Line2D([], [], marker="D", color=INK, lw=0, label="正味（住戸の合計）"),
           Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=INK2, color=INK2, lw=0, label="正味（世帯ひとりずつ）")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.8)
    hair(ax, "x")
    fig.tight_layout(); fig.savefig(f"{OUT}/{fn}.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------ 本体
def move_summary(mv):
    out = {}
    for (src, cond, reg), x in mv.groupby(["src", "cond", "reg"], sort=False):
        key = f"{src}|{cond}|{reg}"
        d = {}
        groups = {"張り出し（付ける）": x[(x.kind == "張り出し") & (x.nv == 1) & x.self_part],
                  "張り出しを外してもらう": x[(x.kind == "張り出し") & (x.nv == 0)],
                  "収納・開口": x[x.kind.isin(["収納", "開口"])], "収納": x[x.kind == "収納"], "開口": x[x.kind == "開口"]}
        for g, y in groups.items():
            if not len(y):
                continue
            c = y.comp
            e = dict(n=len(y), per_bldg=len(y) / x.groupby(["pat", "cs", "seed"]).ngroups if len(x) else 0,
                     q1=c.quantile(0.25), med=c.median(), q3=c.quantile(0.75), mean=c.mean(), zero=float((c <= EPS).mean()),
                     cost_med=y.cost.median())
            if g == "張り出し（付ける）":
                e["by_floor"] = {str(f): dict(n=int(len(z)), med=float(z.comp.median()), q1=float(z.comp.quantile(0.25)), q3=float(z.comp.quantile(0.75)))
                                 for f, z in y.groupby(y.unit.str[0])}
                if "recon_keep" in y:
                    e["recon_keep_mean"] = float(y.recon_keep.mean()); e["recon_keep_med"] = float(y.recon_keep.median())
                    e["recon_keep_share"] = float((y.recon_keep > EPS).mean())
                    e["restore_mean"] = float(y.restore_paid.mean())
                    tot = y.comp + y.recon_keep + y.cost + y.restore_paid
                    e["total_q1"], e["total_med"], e["total_q3"] = float(tot.quantile(0.25)), float(tot.median()), float(tot.quantile(0.75))
                    e["years_after_med"] = float(y.years_after.median())
            d[g] = e
        out[key] = d
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    h = load_hh()
    if os.path.exists(f"{OUT}/moves.csv.gz") and os.environ.get("S8B_RERUN") != "1":
        mv = pd.read_csv(f"{OUT}/moves.csv.gz")   # 再実行の結果の写し（S8B_RERUN=1 で作り直す）
    else:
        mv = collect_moves()
        mv.to_csv(f"{OUT}/moves.csv.gz", index=False)
    # 再実行の照合：手ごとの補償の合計と、世帯の表の払った補償の合計（観察 1・TP+ は再同意がないので一致するはず）
    chk = {}
    for src, cond, reg in MOVE_CELLS:
        a = mv[(mv.src == src) & (mv.cond == cond) & (mv.reg == reg)].comp.sum()
        b = h[(h.src == src) & (h.cond == cond) & (h.reg == reg)].paid_comp.sum()
        chk[f"{src}|{cond}|{reg}"] = dict(moves_comp=a, hh_paid_comp=b, diff=abs(a - b))
    Y = dict(prices=P, p_marg=P_MARG, unit10={k: unit_yen(v) for k, v in P.items()}, unit30={k: unit_yen(v, 30) for k, v in P.items()},
             unit_marg=unit_yen(P_MARG), lump={k: dict(低=lump(v, "低"), 中=lump(v), 高=lump(v, "高")) for k, v in P.items()},
             drent_month={k: EXT_M2 * v for k, v in P.items()}, check=chk,
             ratio_to_main={**{k: v / P[P_MAIN] for k, v in P.items()}, "限界の 1 ㎡": P_MARG / P[P_MAIN]})
    Y["moves"] = move_summary(mv)
    Y["dist"] = dist_table(h)
    # 平均的な性向の隣人に対する 3 階の張り出しの言い値（解析的な値）
    W = np.tile(s0.W_BAR, (9, 1)); e, k, o = s0.f_design()
    e2 = e.copy(); e2[2] = 1
    dS = (s0.S_all(W, e2, k, o) - s0.S_all(W, e, k, o)) / s0.MU
    Y["avg_ext3"] = dict(self=float(dS[2]), loss_2F=float(-dS[5]), loss_1F=float(-dS[8]), comp=float(-dS[5] - dS[8]))
    # 損益分岐の表
    BE = []
    for vname, src_key, yrs, mode, with_restore in VARIANTS:
        ext = Y["moves"][src_key]["張り出し（付ける）"]
        for qn in ["q1", "med", "q3"]:
            for pk in PRICE_KEYS:
                for lab, K, R in K_cases(P[pk], yrs):
                    ex = R if with_restore else 0.0
                    c = ext[qn]
                    BE.append(dict(variant=vname, src=src_key, q=qn, comp=c, price=pk, read=yrs, K=lab, K_yen=K, extra=ex,
                                   C=(c * unit_yen(P[pk], yrs) + K + ex) if mode is None else None,
                                   T=breakeven(c, K, P[pk], yrs, "中", mode, ex), T_early=breakeven(c, K, P[pk], yrs, "高", mode, ex),
                                   T_late=breakeven(c, K, P[pk], yrs, "低", mode, ex)))
    for mode in [None, "annual"]:
        for pk in PRICE_KEYS:
            for lab, K, R in K_cases(P[pk]):
                c = Y["avg_ext3"]["comp"]
                BE.append(dict(variant="平均の性向の隣人・" + ("一回払い・10 年の読み" if mode is None else "毎年払う"), src="平均の性向", q="—",
                               comp=c, price=pk, read=10, K=lab, K_yen=K, extra=0.0,
                               C=(c * unit_yen(P[pk]) + K) if mode is None else None, T=breakeven(c, K, P[pk], 10, "中", mode),
                               T_early=breakeven(c, K, P[pk], 10, "高", mode), T_late=breakeven(c, K, P[pk], 10, "低", mode)))
    Y["breakeven"] = BE
    # 代表の棟
    Y["reps"] = {}
    for tag, src, cond, reg, title in REPS:
        Y["reps"][tag] = rep_table(h, src, cond, reg)
        Y["reps"][tag]["title"] = title
    # 所有者への支払い（1 手あたり、取引の相手の条件）
    ro = pd.read_csv(f"{IN}/runs_obs2.csv.gz")
    w = ro[ro.cond.str.startswith("W|個人・取引")]
    Y["owner_pay"] = {f"{c}|{r}": dict(owner_in_bldg=float(z.owner_in.mean()), moves_bldg=float(z.n_moves.mean()),
                                       per_move=float(z.owner_in.sum() / max(1, z.n_moves.sum())))
                      for (c, r), z in w.groupby(["cond", "reg"])}
    json.dump(Y, open(f"{OUT}/yen.json", "w"), ensure_ascii=False, indent=1, default=float)
    fig421(Y, mv); fig422(h, Y); fig423(Y); fig424(Y)
    fig_rep(Y["reps"]["a"], f"代表の 1 棟の 30 年の収支（観察 1・TP+、cseed {Y['reps']['a']['cs']}）", "fig425")
    fig_rep_unit(Y["reps"]["c"], f"代表の 1 棟の 30 年の収支（改良版top3・入れ替わり 0.30・\n原状回復・工事費、cseed {Y['reps']['c']['cs']}、{len(Y['reps']['c']['rows'])} 世帯）", "fig426")
    fig_rep_unit(Y["reps"]["b"], f"代表の 1 棟の 30 年の収支（改良版top3・入れ替わり 0.10・\n工事費なし、cseed {Y['reps']['b']['cs']}、{len(Y['reps']['b']['rows'])} 世帯）", "fig427")
    print(json.dumps(chk, ensure_ascii=False, indent=1, default=float))
    print("done")


if __name__ == "__main__":
    main()
