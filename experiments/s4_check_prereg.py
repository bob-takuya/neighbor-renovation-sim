"""凍結の前の照合：事前登録 v6（experiments/prereg_s4.json）の文と値が、実行用・判定用のスクリプト（experiments/s4_confirm.py）と
模型（trade/s4.py）・推定器（trade/vstar2.py）と一致することを確かめる。v5 で文とスクリプトが食い違ったことへの対策。
PYTHONPATH=. python3 -m experiments.s4_check_prereg  → out/s4/prereg_check.txt"""
import json, inspect, re
from experiments import s4_confirm as S
from experiments import s2_name
from trade import s4, vstar2

pre = json.load(open("experiments/prereg_s4.json")); v4 = json.load(open("experiments/prereg_s2.json"))
src_c = inspect.getsource(S); src_m = inspect.getsource(s4); src_v = inspect.getsource(vstar2); src_n = inspect.getsource(s2_name)
lines = []; ok = True


def chk(name, cond):
    global ok
    lines.append(("一致 " if cond else "不一致 ") + name); ok &= bool(cond)


chk("種：本走行は cseed 300〜399", pre["seeds"] == [300, 399] and "range(pre[\"seeds\"][0], pre[\"seeds\"][1] + 1)" in src_c)
chk("凍結のときは試しの設定が切れている（S4_DRY が 1 でない）", S.DRY is False and S.MAIN == "out/s4/main.jsonl")
chk("パターン・退去率・費用は v4 と同じ", pre["patterns"] == v4["patterns"] and pre["nus"] == v4["nus"] and pre["costs"] == v4["costs"])
chk("費用の値は v4 の局と同じ（s2_cand.COSTS）", S.COSTS["C1"] == (dict(e=1.0, k=0.25, o=0.25), dict(e=0.5, k=0.1, o=0.1)))
chk("決め方：v4 の 20 に囮 4 つの局（Mtop3・Reptop3・M・Rep）", pre["regimes"]["v4"] == v4["regimes"] and len(pre["regimes"]["v4"]) == 20
    and pre["regimes"]["decoy_runs"] == ["Mtop3", "Reptop3", "M", "Rep"] and all(r in S.REGS for r in pre["regimes"]["v4"] + pre["regimes"]["decoy_runs"]))
chk("囮の聞く範囲：Mtop3・Reptop3 は上位 3 手、M・Rep は全員", S.REGS["Mtop3"] == ("M+", dict(protocol="topk", ask=3), None)
    and S.REGS["Reptop3"] == ("Rep+", dict(protocol="topk", ask=3), None) and S.REGS["M"] == ("M+", {}, None) and S.REGS["Rep"] == ("Rep+", {}, None))
chk("清算所を数える値は Mtop3ch・Mch（推定器の切り替えだけ）", set(pre["regimes"]["decoy_estimator_variants"]) == {"Mtop3ch", "Mch"} and '[x + "ch" for x in pre["regimes"]["decoy_runs"] if x.startswith("M")]' in src_c)
chk("代表を選ぶ乱数は SeedSequence([seed, 7, 321])", "SeedSequence([seed, 7, 321])" in src_m and "[seed, 7, 321]" in pre["decoys"]["Rep"])
chk("代表の役の枠をつねに足す（Grep）", 'if "Grep" in x:' in src_v and "Grep" in pre["quantity"]["role_slots"])
chk("M の提案者は損をする戸が一戸だけの手でだけ係数を知る", "if len(harmed) == 1:" in src_m and "一戸だけ" in pre["decoys"]["M"])
chk("ブートストラップ 1,000 回・種 0", pre["bootstrap"] == dict(n=1000, seed=0, method=pre["bootstrap"]["method"]) and 'B = pre["bootstrap"]["n"]' in src_c
    and "1000" not in re.sub(r"#.*", "", src_c.split("def judge")[1]).replace("1000 回", ""))
chk("ε：TP+・TP+top3・U の 8 セル、並べ替え 100 回、種 1、95 パーセンタイルの最大（v4 と同じ）",
    pre["epsilon"]["regimes"] == ["TP+", "TP+top3", "U"] and pre["epsilon"]["perm"] == 100 and pre["epsilon"]["seed"] == 1
    and "100 回" in pre["epsilon"]["rule"] and "for rg in er[\"regimes\"] for (pat, nu, c) in SET" in src_c
    and "np.percentile(out, 95)" in inspect.getsource(__import__("trade.vstar", fromlist=["x"]).perm_floor))
chk("v4-P1〜P5 の文は v4 のまま", all(pre["predictions_v4"][p] == v4["predictions"][p] for p in ["P1", "P2", "P3", "P4", "P5"]))
# 判定の式が s2_name.judge と同じ形か（主要な不等式）
chk("P1 の式（区間の上端 ＞ ε で不合格）", "V(k[0], k[1:])[2] > eps" in src_c and "v[2] > eps" in src_n)
chk("P2 の式（点推定の大小、割合 ≥ v4 の order_share＝0.8）", "ok=bool(A[0] > Bv[0])" in src_c and 'share >= v4["order_share"]' in src_c and v4["order_share"] == 0.8)
chk("P4 の式（TP+top3 の下端 ＞ TP+ の上端、退去率 0）", "ok=bool(A[1] > Bv[2])" in src_c)
chk("P5 の式（N6 の下端 ≤ TP+top3 の上端 ＋ ε）", "ok=bool(A[1] <= Bv[2] + eps)" in src_c)
dr = pre["decoy_rules"]
chk("D1：主は Mtop3、比べる相手は TP+top3、住民主体は 11", dr["M_main"] == "Mtop3" and dr["M_reference"] == "TP+top3" and len(dr["resident"]) == 11
    and set(dr["resident"]) == set(v4["regimes"][:11]))
chk("D1 (a)：M の下端 ＞ TP+top3 の上端", "a_ = bool(Mv[1] > T3[2])" in src_c and "M の区間の下端が TP+top3 の区間の上端を上回る" in dr["D1"]["text"])
chk("D1 (b)：M の点推定 ＞ 住民主体の最大 ＋ ε", dr["D1"]["b_margin"] == "epsilon" and "b_ = bool(Mv[0] > V(mx, cell)[0] + margin)" in src_c and "ε より大きく上回る" in dr["D1"]["text"])
chk("D1：一つのセルで (a) か (b) なら不適切", "if hit:" in src_c and "どれか一つ" in dr["D1"]["text"])
chk("D1：反証されないは、TP+top3 の下端 ＞ ε のすべてのセルで M の上端 ＜ TP+top3 の下端",
    "lower = bool(Mv[2] < T3[1])" in src_c and "counts = bool(T3[1] > eps)" in src_c and "all(x[\"M_lower\"] for x in sup_cells)" in src_c)
chk("D2：Reptop3 の 8 セルすべてで上端 ≤ ε", dr["Rep_main"] == "Reptop3" and "below_eps=bool(V(Rm, cell)[2] <= eps)" in src_c and "8 セルすべて" in dr["D2"]["text"])
chk("再現の基準：向き 0.8・順位相関の中央値 0.8", pre["replication"]["direction_share"] == 0.8 and pre["replication"]["rank_rho_median"] == 0.8
    and 'rc["direction_share"]' in src_c and 'rc["rank_rho_median"]' in src_c and "0.8 以上" in pre["replication"]["direction_text"]
    and "0.8 以上" in pre["replication"]["rank_text"])
chk("感度の分析の対象", dr["sensitivity"]["resident_role_slot"] == ["TP+top3", "改良版top3", "Mtop3"] and dr["sensitivity"]["rep_without_role_slot"] == ["Reptop3", "Rep"])
open("out/s4/prereg_check.txt", "w").write("\n".join(lines) + ("\nすべて一致\n" if ok else "\n不一致あり\n"))
print("\n".join(lines)); print("すべて一致" if ok else "不一致あり")
