"""段 2・候補を絞る試験 A〜C。out/s2/cand.jsonl を読んで、候補ごとの合否を out/s2/tests.json・tests.md に書く。
A 帰無と偽装：帰無・偽装の条件で、値の 95 % 区間の上端が ε 以下。N6（わざと悪い盤から）で値が水増しされない。N7（外部性のない盤）で ε 以下。
B 全知への抵抗：性向を知る者が、聞く範囲を構成ごとに選んで値を最大にしても、最良の固定した手続きを ε より超えない。
C 自明さと弁別：手の数・η・ΔΣS*・H・L・B で説明できない分が 25 % 以上、η・手の数との順位相関の絶対値が 0.8 未満、
  TP+ と TP+top3（η がほぼ同じ）を区別できる、同じ構成の 2 局の一致（級内相関）が 0.5 以上。
PYTHONPATH=. python3 -m experiments.s2_tests
"""
import json
from collections import defaultdict
import numpy as np
from scipy import stats
from trade import s0, vstar

RES_REGS = ["U", "V50", "V75", "TR+", "TP+", "TP+top1", "TP+top3", "TP+near", "改良版", "改良版top3", "改良版near"]
NULLS = ["F", "N2", "N3", "N4", "N4c", "BR", "BRdyn", "AI"]
POP = {"V_H": "X", "V*": "Xc", "V*n": "Xn"}
RUN = ["Lc", "Ln", "Vp", "gohai", "ent", "rebind", "quotes"]
NAMES = {"V_H": "分散知（同意の限定なし）V_H", "V*": "分散知 V*（同意で支えられた部位）", "V*n": "分散知 V*n（損をする側の同意で支えられた部位）",
         "Lc": "合意の支え Λc", "Ln": "損をする側の合意の支え Λn", "Vp": "権利の量 Vp（同意を経た損の割合）", "gohai": "誤配（第三者の得、㎡）",
         "ent": "記録のエントロピー", "rebind": "結び直し（再同意で払って残した部位）", "quotes": "言い値の問い合わせの数"}
CTRL = ["moves", "eta", "dSstar", "H", "L", "B"]


def load():
    cells = defaultdict(list)
    for line in open("out/s2/cand.jsonl"):
        r = json.loads(line)
        cells[(r["reg"], r["pat"], r["nu"], r["cost"], r["rep"])].append(r)
    for k in cells:
        cells[k].sort(key=lambda r: r["cs"])
    return cells


def n7_records(pat, cs_list, rep=0):
    """N7：外部性のない盤。張り出しは自戸の日照だけを 1.5 時間減らす（下の戸を陰にしない）。収納・開口は動かせない。
    満足が線形なので、世帯 i の最良は「広さの重み ＞ 日照の重み」なら張り出す。各世帯は 30 年のうち一度でも要求すれば自分の最良に置く。
    損をする者がいないので、変えた部位はすべて自分ひとりの同意で支えられる（Λc）が、損をする側の同意（Λn）はない。"""
    from experiments.explore import composition
    out = []
    for cs in cs_list:
        W = composition(pat, cs)
        rng = np.random.default_rng(np.random.SeedSequence([cs + 50000 * rep, 7]).spawn(3)[0])
        REQ = rng.random((30, 9))
        Z = np.zeros(21, int); Z[:9] = (W[:, 0] > W[:, 1]).astype(int)
        X = np.zeros(21, int); req = (REQ < 0.3).any(0)
        X[:9] = np.where(req, Z[:9], 0)
        G = []
        for a in range(9):
            g = np.zeros(21, int); g[:9] = int(s0.W_BAR[0] > s0.W_BAR[1]); g[a] = Z[a]; G.append(g.tolist())
        G.append(np.zeros(21, int).tolist()); G.append(np.full(21, 0).tolist())
        changed = (X != 0)
        out.append(dict(Z=Z.tolist(), X=X.tolist(), Xc=X.tolist(), Xn=np.zeros(21, int).tolist(), G=G,
                        Lc=float(changed.sum() / 21), Ln=0.0, Vp=float("nan"), gohai=0.0, ent=0.0, rebind=0, quotes=0,
                        moves=int(changed.sum()), eta=1.0))
    return out


def popv(recs, key):
    v = vstar.population(recs, key)["V"]
    lo, hi = vstar.boot(recs, key, n=300)
    return v, lo, hi


def runv(recs, key):
    x = np.array([r[key] for r in recs], float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    bs = [np.mean(np.random.default_rng(i).choice(x, len(x))) for i in range(300)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    cells = load()
    SET = sorted({(k[1], k[2], k[3]) for k in cells if k[4] == 0})       # (パターン, ν, 費用)
    out = dict(cells={}, tests={})
    val = {}
    # ---------------- 値の表（条件ごと）
    for (pat, nu, cost) in SET:
        for reg in RES_REGS + NULLS + ["N6"]:
            recs = cells.get((reg, pat, nu, cost, 0))
            if not recs:
                continue
            row = {}
            for c, key in POP.items():
                row[c] = popv(recs, key)
            for c in RUN:
                row[c] = runv(recs, c)
            row["eta"] = float(np.nanmean([r["eta"] for r in recs])); row["moves"] = float(np.mean([r["moves"] for r in recs]))
            val[(reg, pat, nu, cost)] = row
        n7 = n7_records(pat, sorted({r["cs"] for r in cells[("F", pat, nu, cost, 0)]}))
        row = {c: popv(n7, key) for c, key in POP.items()}
        for c in RUN:
            row[c] = runv(n7, c)
        row["eta"] = 1.0; row["moves"] = float(np.mean([r["moves"] for r in n7]))
        val[("N7", pat, nu, cost)] = row
    # ε：住民主体の決め方の各セルで、Z を入れ替えたときの値の 95 パーセンタイルの最大（集団の量）
    eps = {}
    for c, key in POP.items():
        fl = []
        for (pat, nu, cost) in SET:
            for reg in ["TP+", "TP+top3", "U"]:
                recs = cells.get((reg, pat, nu, cost, 0))
                if recs:
                    fl.append(vstar.perm_floor(recs, key, n=100))
        eps[c] = float(max(fl))
    for c in RUN:   # 一局の量：住民主体の決め方の最大値の 5 %
        mx = max(abs(val[(reg, *s)][c][0]) for reg in RES_REGS for s in SET if (reg, *s) in val)
        eps[c] = 0.05 * mx
    out["eps"] = eps
    # ---------------- A
    A = {}
    for c in list(POP) + RUN:
        fails = []
        for (pat, nu, cost) in SET:
            for reg in NULLS + ["N7"]:
                if reg == "N2" and nu > 0:      # 入れ替わりがあると、全知の最適から始めても新しい住み手が手を打つ。帰無に数えない
                    continue
                v = val.get((reg, pat, nu, cost))
                if v is None:
                    continue
                if v[c][2] > eps[c] + 1e-9:
                    fails.append(f"{reg}（{pat}・ν{nu}・費用{cost}）{v[c][0]:.2f}")
            a, b = val.get(("N6", pat, nu, cost)), val.get(("TP+top3", pat, nu, cost))
            if a and b and a[c][1] > b[c][2] + eps[c]:
                fails.append(f"N6 で水増し（{pat}・ν{nu}・費用{cost}）{a[c][0]:.2f} 対 {b[c][0]:.2f}")
        A[c] = dict(ok=len(fails) == 0, fails=fails)
    # ---------------- B（聞く範囲を構成ごとに選ぶ全知）
    B = {}
    PROTOS = ["TP+", "TP+top1", "TP+top3", "TP+near"]
    for c in list(POP) + RUN:
        worst = []
        for (pat, nu, cost) in SET:
            R = [cells.get((p, pat, nu, cost, 0)) for p in PROTOS]
            if any(x is None for x in R):
                continue
            if c in POP:
                pw = [vstar.pointwise(x, POP[c]) for x in R]
            else:
                pw = [np.nan_to_num(np.array([r[c] for r in x], float)) for x in R]
            pick = np.argmax(np.array(pw), axis=0)
            sel = [R[pick[i]][i] for i in range(len(R[0]))]
            v_sel = vstar.population(sel, POP[c])["V"] if c in POP else float(np.nanmean([r[c] for r in sel]))
            v_best = max(val[(p, pat, nu, cost)][c][0] for p in PROTOS)
            worst.append((v_sel - v_best, f"{pat}・ν{nu}・費用{cost}：選んだ {v_sel:.2f} 対 最良の固定 {v_best:.2f}"))
        mx = max(worst)
        B[c] = dict(ok=mx[0] <= eps[c] + 1e-9, worst=mx[1], gain=float(mx[0]))
    # ---------------- C
    C = {}
    for c in list(POP) + RUN:
        ys, Xs, et, mv, pr = [], [], [], [], []
        for (pat, nu, cost) in SET:
            for reg in RES_REGS:
                recs = cells.get((reg, pat, nu, cost, 0))
                if not recs:
                    continue
                y = vstar.pointwise(recs, POP[c]) if c in POP else np.array([r[c] for r in recs], float)
                ys += list(y); Xs += [[np.nan_to_num(r[k]) for k in CTRL] + [float(nu), float(cost != "なし"), float(pat == "二群")] for r in recs]
        y = np.array(ys, float); Xm = np.array(Xs, float); ok_ = ~np.isnan(y)
        y, Xm = y[ok_], Xm[ok_]
        Xd = np.column_stack([np.ones(len(y)), Xm])
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        res = y - Xd @ beta
        unexpl = float(res.var() / y.var()) if y.var() > 1e-12 else 0.0
        r_eta = float(stats.spearmanr(y, Xm[:, 1]).correlation); r_mv = float(stats.spearmanr(y, Xm[:, 0]).correlation)
        # TP+ 対 TP+top3（η がほぼ同じ）
        disc = []
        for (pat, nu, cost) in SET:
            a, b = val.get(("TP+", pat, nu, cost)), val.get(("TP+top3", pat, nu, cost))
            if a and b and abs(a["eta"] - b["eta"]) < 0.03:
                disc.append(b[c][1] > a[c][2] or a[c][1] > b[c][2])
        # 級内相関（同じ構成の 2 局）
        icc = []
        for pat in sorted({s[0] for s in SET}):
            for reg in RES_REGS:
                r0, r1 = cells.get((reg, pat, 0.0, "なし", 0)), cells.get((reg, pat, 0.0, "なし", 1))
                if not r0 or not r1:
                    continue
                if c in POP:
                    both = vstar.pointwise(r0 + r1, POP[c]); a0, a1 = both[:len(r0)], both[len(r0):]
                else:
                    a0 = np.nan_to_num(np.array([r[c] for r in r0], float)); a1 = np.nan_to_num(np.array([r[c] for r in r1], float))
                if np.std(a0) > 1e-9 and np.std(a1) > 1e-9:
                    icc.append(np.corrcoef(a0, a1)[0, 1])
        iccm = float(np.nanmean(icc)) if icc else float("nan")
        okC = unexpl >= 0.25 and abs(r_eta) < 0.8 and abs(r_mv) < 0.8 and (any(disc) if disc else False) and iccm >= 0.5
        C[c] = dict(ok=bool(okC), unexplained=unexpl, rho_eta=r_eta, rho_moves=r_mv, discriminates=int(sum(disc)), n_pairs=len(disc), icc=iccm)
    out["tests"] = dict(A=A, B=B, C=C)
    out["cells"] = {"|".join(map(str, k)): {c: list(v) if isinstance(v, tuple) else v for c, v in row.items()} for k, row in val.items()}
    json.dump(out, open("out/s2/tests.json", "w"), ensure_ascii=False, indent=1)
    # 要約
    L = ["| 候補 | A 帰無と偽装 | B 全知への抵抗 | C 自明さと弁別 | 通過 |", "|---|---|---|---|---|"]
    for c in list(POP) + RUN:
        a, b, cc = A[c], B[c], C[c]
        L.append(f"| {NAMES[c]} | {'○' if a['ok'] else '×（' + str(len(a['fails'])) + ' 件）'} | {'○' if b['ok'] else '×'}（{b['gain']:+.2f}） | "
                 f"{'○' if cc['ok'] else '×'}（説明できない分 {cc['unexplained']:.2f}、ρη {cc['rho_eta']:.2f}、ρ手 {cc['rho_moves']:.2f}、弁別 {cc['discriminates']}/{cc['n_pairs']}、ICC {cc['icc']:.2f}） | "
                 f"{'**通過**' if a['ok'] and b['ok'] and cc['ok'] else ''} |")
    open("out/s2/tests.md", "w").write("\n".join(L) + "\n\nε：" + json.dumps({k: round(v, 3) for k, v in eps.items()}, ensure_ascii=False) + "\n")
    print("\n".join(L)); print(eps)
    for c in A:
        if A[c]["fails"]:
            print(c, "A の不合格の例：", A[c]["fails"][:4])


if __name__ == "__main__":
    main()
