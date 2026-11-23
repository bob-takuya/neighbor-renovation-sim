"""凍結の前の照合：事前登録 v7b（experiments/prereg_s6.json）の文と値が、実行と判定のスクリプト（experiments/s6_prof2.py）、
模型（trade/s6.py）、推定器（trade/vstar3.py・vstar4.py）と一致することを確かめる（v7a の s5_check_prereg と同じ考え方）。
PYTHONPATH=. python3 -m experiments.s6_check_prereg  → out/s6/prereg_check.txt"""
import json, inspect, re
from experiments import s6_prof2 as S
from trade import s6, vstar3, vstar4, vstar

pre = json.load(open("experiments/prereg_s6.json"))
src_c = inspect.getsource(S); src_m = inspect.getsource(s6); src_med = inspect.getsource(s6.Mediator)
src_j = src_c.split("def judge")[1]
lines = []; ok = True


def chk(name, cond):
    global ok
    lines.append(("一致 " if cond else "不一致 ") + name); ok &= bool(cond)


chk("種：確かめ用は cseed 500〜599", pre["seeds"] == [500, 599] and 'range(pre["seeds"][0], pre["seeds"][1] + 1)' in src_c)
chk("凍結のときは試しの設定が切れている（S6_DRY が 1 でない）", S.DRY is False and S.MAIN == "out/s6/main.jsonl" and S.JUDGE == "out/s6/judge.json")
chk("パターン・退去率はスクリプトと同じ", pre["patterns"] == S.PATS and pre["nus"] == S.NUS)
chk("入れ替わりは幾何（観察 2 の入れ替わりの群と同じ）", 'turnover=dict(kind="geo", nu=nu)' in src_c)
chk("費用 1 倍は確証の C1（1.0・0.25・0.25、戻す 0.5・0.1・0.1）", S.C1 == (dict(e=1.0, k=0.25, o=0.25), dict(e=0.5, k=0.1, o=0.1))
    and "1.0・収納 0.25・開口 0.25" in pre["conditions"]["費用1倍"] and "0.5・0.1・0.1" in pre["conditions"]["費用1倍"])
chk("近い戸だけは G0・q_far＝0・tau＝0", S.CONDS["近い戸だけ"] == dict(venue=dict(G=0, q_far=0.0, tau=0.0)) and "q_far＝0" in pre["conditions"]["近い戸だけ"]
    and S.CONDS["基準"] == {})
chk("基準：TP+top3 と改良版top3（〔貸〕20・〔清〕・〔再〕all）", set(pre["bases"]) == set(S.BASES) and S.BASES["TP+top3"] == dict(protocol="topk", ask=3)
    and S.BASES["改良版top3"] == dict(credit=20.0, settle=True, reconsent="all", protocol="topk", ask=3))
chk("腕：仲介者（stuck）と広く聞く仲介者（all）、参照は BR・BRdyn", S.BROKERS == {"仲介者": "stuck", "広く聞く仲介者": "all"} and S.REFS == ["BR", "BRdyn"]
    and set(pre["arms"]["TP+top3"]) == {"基準", "仲介者", "広く聞く仲介者"} and set(pre["arms"]["参照"]) == {"BR", "BRdyn"})
chk("届かなかった手だけの仲介者は、費用なし・全員に届く条件を走らせない（定義から基準と同じ）",
    all("基準" not in pre["arms"][b]["仲介者"] for b in ("TP+top3", "改良版top3")) and any("相談がなく" in x for x in pre["not_run_definitional"]))
chk("BR・BRdyn は近い戸だけを走らせず、基準の条件の局を使う", all("近い戸だけ" not in v for v in pre["arms"]["参照"].values())
    and 'if cond == "近い戸だけ":\n                cond = "基準"' in src_c)
G = S.pre_groups(pre)
chk("局の数 15,600（BRdyn は入れ替わりのあるセルだけ）", len(G) * 100 == pre["n_games"] == 15600 and not any(g[1] == "BRdyn" and g[4] == 0 for g in G))
chk("仲介者が相談を受けるのは、自分の手続きで打てなかったときだけ", "if best is None or best[0] <= 1e-12:" in src_m and "med.consult(" in src_m
    and "自分の手続き" in pre["mediator"]["when"])
chk("届かなかった手：届かない戸がいる手と、費用を足すと赤字になる手", 'far = reach is not None and any(not reach[j] for j in harmed)' in src_med
    and "net - cst <= 1e-12" in src_med and "(a)" in pre["mediator"]["scope"]["仲介者"] and "(b)" in pre["mediator"]["scope"]["仲介者"])
chk("満足の動く世帯すべてに聞き、learn で記録する", "for j in parts:" in src_med and "self.learn(ctx, t, j, c_)" in src_med and "満足の動く世帯すべて" in pre["mediator"]["asks"])
chk("得の大きさに比べた割り勘、損をする世帯は言い値ちょうど", "s = C * G / G.sum()" in src_med and "pay[j] = L[j]" in src_med and "割り勘" in pre["mediator"]["proposes"])
chk("束の同意の世帯は満足の動く世帯すべて", 'supp={int(i)} | harmed | set(bundle["co"])' in src_m and "満足の動く世帯すべて" in pre["mediator"]["decides"])
chk("住み手は二戸の束でだけ相手の係数を知る", "if len(parts) == 2:" in src_med and "二戸だけ" in pre["mediator"]["residents_learn"])
chk("仲介者は役の枠（種類 broker）で〔数〕、BR は中央の枠を外して〔外〕", 'return recs, ("broker",)' in src_c and "vstar4.drop_central(recs)" in src_c
    and s6.Mediator().kind == "broker" and "CENTRAL = 9" in inspect.getsource(vstar4))
chk("ε：基準の腕の 24 セル × 2、並べ替え 100 回・種 1・95 パーセンタイルの最大", pre["epsilon"]["bases"] == ["TP+top3", "改良版top3"] and pre["epsilon"]["perm"] == 100
    and pre["epsilon"]["seed"] == 1 and len(pre["epsilon"]["conds"]) == 3 and "eps = max(epsd.values())" in src_j and "np.percentile(out, 95)" in inspect.getsource(vstar.perm_floor))
chk("ブートストラップ：V*n 1,000 回・種 0、η 2,000 回・種 20261111", pre["bootstrap"]["n"] == 1000 and pre["bootstrap"]["seed"] == 0 and pre["bootstrap"]["eta_n"] == 2000
    and pre["bootstrap"]["eta_seed"] == 20261111 and 'B = bs["n"]' in src_j and 'BE = bs["eta_n"]' in src_j)
chk("差は同じ構成の組（cs をそろえる）", 'assert [x["cs"] for x in recsA] == [x["cs"] for x in recsB]' in inspect.getsource(vstar3.diff3)
    and 'assert [x["cs"] for x in A] == [x["cs"] for x in B]' in inspect.getsource(S.eta_diff))
chk("規則 lower・higher・same（±margin）", 'return x["hi"] < 0' in inspect.getsource(S.passes) and 'return x["lo"] > 0' in inspect.getsource(S.passes)
    and 'return x["lo"] >= -margin and x["hi"] <= margin' in inspect.getsource(S.passes) and "上端が 0 より小さい" in pre["rules"]["lower"])
chk("判定：すべてのセルで支持、一部なら一部支持", '"支持" if n_ok == len(rows)' in src_j and "一部支持" in pre["rules"]["verdict"])
chk("same の予言は margin 0.02 を持つ", all(p.get("margin") == 0.02 for p in pre["predictions"] if p["rule"] == "same"))
chk("予言の量は eta・Vn外・Vn数、規則は lower・higher・same だけ", {p["quantity"] for p in pre["predictions"]} <= {"eta", "Vn外", "Vn数"}
    and {p["rule"] for p in pre["predictions"]} <= {"lower", "higher", "same"})
ran = {(g[0], g[1], g[2]) for g in G}


def ran_cell(b, arm, cond):
    if arm in S.REFS:
        return ("参照", arm, "基準" if cond == "近い戸だけ" else cond) in ran
    return (b, arm, cond) in ran


chk("予言の腕と比べる腕は、走らせる組（定義から同じ局を含む）", all(ran_cell(b, p["arm"], c[0]) and ran_cell(b, p["ref"], c[0])
                                                for p in pre["predictions"] for b in p["bases"] for c in p["cells"]))
chk("予言のセルは事前登録のパターンと退去率の中", all(c[1] in pre["patterns"] and c[2] in pre["nus"] for p in pre["predictions"] for c in p["cells"]))
vc = {("TP+top3", 0.0)} | {("改良版top3", x) for x in (0.0, 0.05, 0.1)}
chk("V*n の予言は Vn_cells の範囲（TP+top3 は入れ替わりなし、改良版top3 は 0・5・10 %）",
    all((b, c[2]) in vc for p in pre["predictions"] if p["quantity"].startswith("Vn") for b in p["bases"] for c in p["cells"]))
chk("予言の文のセルの数と cells の数が合う", all(f"{len(p['cells']) * len(p['bases'])} セル" in p["text"] for p in pre["predictions"]))
chk("予言は選ぶ用でどのセルも同じ向き（dev に添えた値）", all(
    (p["rule"] == "higher" and r["lo"] > 0) or (p["rule"] == "lower" and r["hi"] < 0) or (p["rule"] == "same" and r["lo"] >= -p["margin"] and r["hi"] <= p["margin"])
    for p in pre["predictions"] for r in p["dev"]))
chk("定義からの確かめ KB1・KB2・K3 を判定のスクリプトで計算する", "KB1=" in src_j and "KB2=" in src_j and "K3=" in src_j)
chk("disclosure に、選ぶ用を見た後に決めたことを書いている", "見た後に決めたこと" in pre["disclosure"] and "(6)" in pre["disclosure"])
open("out/s6/prereg_check.txt", "w").write("\n".join(lines) + ("\nすべて一致\n" if ok else "\n不一致あり\n"))
print("\n".join(lines)); print(len(lines), "項目"); print("すべて一致" if ok else "不一致あり")
