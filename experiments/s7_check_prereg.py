"""凍結の前の照合：事前登録 v7c（experiments/prereg_s7.json）の文と値が、実行と判定のスクリプト（experiments/s7_prof3.py）、
模型（trade/s7.py）、推定器（trade/vstar5.py ほか）と一致することを確かめる（v7a・v7b の照合と同じ考え方）。
PYTHONPATH=. python3 -m experiments.s7_check_prereg  → out/s7/prereg_check.txt"""
import json, inspect
from experiments import s7_prof3 as S
from trade import s7, vstar5, vstar4, vstar

pre = json.load(open("experiments/prereg_s7.json"))
src_c = inspect.getsource(S); src_m = inspect.getsource(s7); src_v5 = inspect.getsource(vstar5)
src_j = src_c.split("def judge")[1]
lines = []; ok = True


def chk(name, cond):
    global ok
    lines.append(("一致 " if cond else "不一致 ") + name); ok &= bool(cond)


chk("種：確かめ用は cseed 500〜599", pre["seeds"] == [500, 599] and 'range(pre["seeds"][0], pre["seeds"][1] + 1)' in src_c)
chk("凍結のときは試しの設定が切れている（S7_DRY が 1 でない）", S.DRY is False and S.MAIN == "out/s7/main.jsonl" and S.JUDGE == "out/s7/judge.json")
chk("パターン・退去率はスクリプトと同じ", pre["patterns"] == S.PATS and pre["nus"] == S.NUS)
chk("入れ替わりは幾何（観察 2 の入れ替わりの群と同じ）", 'turnover=dict(kind="geo", nu=nu)' in src_c and "geo" in pre["conditions"]["turnover"])
chk("近い戸だけは G0・q_far＝0・tau＝0（職能 2 と同じ）", S.CONDS["近い戸だけ"] == dict(venue=dict(G=0, q_far=0.0, tau=0.0)) and S.CONDS["基準"] == {}
    and "q_far＝0" in pre["conditions"]["近い戸だけ"])
chk("基準：TP+top3 と改良版top3（〔貸〕20・〔清〕・〔再〕all）", set(pre["bases"]) == set(S.BASES) and S.BASES["TP+top3"] == dict(protocol="topk", ask=3)
    and S.BASES["改良版top3"] == dict(credit=20.0, settle=True, reconsent="all", protocol="topk", ask=3))
chk("走らせる腕は事前登録とスクリプトの確かめ用の組で同じ", pre["arms"] == json.loads(json.dumps(S.MAIN_SPEC)) and pre["selector_arms"] == json.loads(json.dumps(S.MAIN_SEL)))
G0, GS = S.pre_groups(pre)
chk("局の数（BRdyn は入れ替わりのあるセルだけ）", (len(G0) + len(GS)) * 100 == pre["n_games"] and not any(g[1] == "BRdyn" and g[4] == 0 for g in G0))
chk("四つの範囲：全員 full・上位 1 手 topk 1・上位 3 手 topk 3・近い戸だけ near", s7.PROTOCOLS == {"全員": dict(protocol="full", ask=3), "上位1手": dict(protocol="topk", ask=1),
    "上位3手": dict(protocol="topk", ask=3), "近い戸だけ": dict(protocol="near", ask=3)} and s7.ORDER == ["全員", "上位1手", "上位3手", "近い戸だけ"]
    and all(x in pre["actors"]["選ぶ者"]["does"] for x in ("full", "topk 1", "topk 3", "near")))
chk("1 局目の局の種は cseed＋50000、2 局目は cseed", S.G1_SHIFT == 50000 and "seed = cs + G1_SHIFT if arm in G1 else cs" in src_c and "cseed＋50000" in pre["actors"]["選ぶ者"]["two_games"])
chk("選ぶ者は手を打たず、init で範囲を返すだけ", "def init(self, ctx):" in inspect.getsource(s7.Scope) and "def evals" not in inspect.getsource(s7.Scope)
    and "def trade" not in inspect.getsource(s7.Scope))
chk("性向を知る選ぶ者：点ごとの値（vstar.pointwise）で建物ごとに選び、know_all(−1) で知識を残す", 'vstar.pointwise(by_scope[s], "Xn")' in src_v5
    and "self.know_all(ctx, -1)" in inspect.getsource(s7.Scope) and "pointwise" in pre["actors"]["選ぶ者"]["knowing"])
chk("分布だけを知る選ぶ者：1 局目の集団の値で一つの範囲を選び、何も知らない", 'vstar3.population3(by_scope[s], "Xn")["V"]' in src_v5
    and 'out[f"{b}|選ぶ者:分布|{cond}|{p}|{nu}"] = {str(cs): dist for cs in kn}' in src_c and "facts は空" in pre["actors"]["選ぶ者"]["distribution"])
chk("同じ値のときは基準の上位 3 手、なければ前のもの", 'return prefer if prefer in top else top[0]' in src_v5 and 'prefer="上位3手"' in src_v5
    and "上位 3 手" in pre["actors"]["選ぶ者"]["tie"])
chk("選ぶ者の腕は 1 局目の後に、選んだ範囲で走らせる", "ch = choose(G1g, cells, choice_path)" in src_c and "run_groups(S, CS, parts, ch)" in src_c)
chk("道具：見せるだけで決めず、近い戸だけの制約を外す（tau＝0 のときだけ）", "kw = dict(kw, venue=None)" in src_m and 'float(v.get("tau", 0.5)) == 0.0' in src_m
    and "return evals" in inspect.getsource(s7.Tool) and "venue を外して" in pre["actors"]["道具"]["effect"])
chk("満足の量の道具は使われた年ごとに know_all(t)、物理の量の道具は何も知らない", "self.know_all(ctx, t)" in inspect.getsource(s7.Tool)
    and 'if self.shows == "satisfaction" and self._read != t:' in inspect.getsource(s7.Tool))
chk("計画者は regime P（帰無 N3）、参照は近い戸だけでも基準の条件の局", S.REFS["計画者"] == "P" and 'kw.pop("venue", None)' in src_c
    and 'if cond == "近い戸だけ":\n                cond = "基準"' in src_c)
chk("役者は役の枠（scope・tool・broker）で〔数〕、計画者・BR は中央の枠を外して〔外〕", S.KIND == {"選ぶ者:分布": "scope", "選ぶ者:性向": "scope", "道具:物理": "tool", "道具:満足": "tool", "仲介者": "broker"}
    and "return recs, (KIND[arm],)" in src_c and "vstar4.drop_central(recs)" in src_c and "CENTRAL = 9" in inspect.getsource(vstar4))
chk("〔数〕−〔外〕は同じ腕・同じ構成の組", 'if q == "Vn数対外":' in src_c and 'RA, cA = prep(C(ab, arm, cond, p, nu), arm, "Vn数")' in src_c)
chk("ε：基準の腕の 16 セル × 2、並べ替え 100 回・種 1・95 パーセンタイルの最大", pre["epsilon"]["bases"] == ["TP+top3", "改良版top3"] and pre["epsilon"]["conds"] == ["基準", "近い戸だけ"]
    and pre["epsilon"]["perm"] == 100 and pre["epsilon"]["seed"] == 1 and "eps = max(epsd.values())" in src_j and "np.percentile(out, 95)" in inspect.getsource(vstar.perm_floor))
chk("ブートストラップ：V*n 1,000 回・種 0、η 2,000 回・種 20261111", pre["bootstrap"]["n"] == 1000 and pre["bootstrap"]["seed"] == 0 and pre["bootstrap"]["eta_n"] == 2000
    and pre["bootstrap"]["eta_seed"] == 20261111 and 'B = bs["n"]' in src_j and 'BE = bs["eta_n"]' in src_j)
chk("差は同じ構成の組（cs をそろえる）", 'assert [x["cs"] for x in A] == [x["cs"] for x in B]' in inspect.getsource(S.eta_diff))
chk("規則 lower・higher・same（±margin）", 'return x["hi"] < 0' in inspect.getsource(S.passes) and 'return x["lo"] > 0' in inspect.getsource(S.passes)
    and 'return x["lo"] >= -margin and x["hi"] <= margin' in inspect.getsource(S.passes) and "上端が 0 より小さい" in pre["rules"]["lower"])
chk("判定：すべてのセルで支持、一部なら一部支持", '"支持" if n_ok == len(rows)' in src_j and "一部支持" in pre["rules"]["verdict"])
chk("same の予言は margin 0.02 を持つ", all(p.get("margin") == 0.02 for p in pre["predictions"] if p["rule"] == "same"))
chk("予言の量は eta・Vn外・Vn数・Vn数対外、規則は lower・higher・same だけ", {p["quantity"] for p in pre["predictions"]} <= {"eta", "Vn外", "Vn数", "Vn数対外"}
    and {p["rule"] for p in pre["predictions"]} <= {"lower", "higher", "same"})
ran = {(g[0], g[1], g[2]) for g in G0 + GS}


def ran_cell(b, arm, cond):
    if arm in S.REFS:
        return ("参照", arm, "基準") in ran
    if arm == "範囲:上位3手":
        arm = "基準"
    return (b, arm, cond) in ran


chk("予言の腕と比べる腕は、走らせる組", all(ran_cell(b, p["arm"], c[0]) and ran_cell(b, p["ref"], c[0]) for p in pre["predictions"] for b in p["bases"] for c in p["cells"]))
chk("予言のセルは事前登録のパターンと退去率の中", all(c[1] in pre["patterns"] and c[2] in pre["nus"] for p in pre["predictions"] for c in p["cells"]))
vc = {("TP+top3", 0.0)} | {("改良版top3", x) for x in (0.0, 0.05, 0.1)}
chk("V*n の予言は Vn_cells の範囲（TP+top3 は入れ替わりなし、改良版top3 は 0・5・10 %）",
    all((b, c[2]) in vc for p in pre["predictions"] if p["quantity"].startswith("Vn") for b in p["bases"] for c in p["cells"]))
chk("予言の文のセルの数と cells の数が合う", all(f"{len(p['cells']) * len(p['bases'])} セル" in p["text"] for p in pre["predictions"]))
chk("予言は選ぶ用でどのセルも同じ向き（dev に添えた値）", all(
    (p["rule"] == "higher" and r["lo"] > 0) or (p["rule"] == "lower" and r["hi"] < 0) or (p["rule"] == "same" and r["lo"] >= -p["margin"] and r["hi"] <= p["margin"])
    for p in pre["predictions"] for r in p["dev"]) and all(len(p["dev"]) == len(p["cells"]) * len(p["bases"]) for p in pre["predictions"]))
chk("定義からの確かめ K1〜K6 を判定のスクリプトで計算する", all(f"K{i}=" in src_j for i in range(1, 7)))
chk("予言の dev の値は選ぶ用の値と同じ（out/s7/dev_values.json）", all(
    abs(r["d"] - (json.load(open("out/s7/dev_values.json"))["eta_diffs" if p["quantity"] == "eta" else "diffs"]
                  [f"{r['base']}|{p['arm']}|{p['ref']}|{r['cond']}|{r['pat']}|{r['nu']}|{p['quantity']}"]["d"])) < 1e-4
    for p in pre["predictions"] for r in p["dev"]))
chk("disclosure に、選ぶ用を見た後に決めたことを書いている", "見た後に決めたこと" in pre["disclosure"] and "(6)" in pre["disclosure"])
open("out/s7/prereg_check.txt", "w").write("\n".join(lines) + ("\nすべて一致\n" if ok else "\n不一致あり\n"))
print("\n".join(lines)); print(len(lines), "項目"); print("すべて一致" if ok else "不一致あり")
