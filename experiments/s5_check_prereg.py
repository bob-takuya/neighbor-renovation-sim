"""凍結の前の照合：事前登録 v7a（experiments/prereg_s5a.json）の文と値が、実行と判定のスクリプト（experiments/s5_prof1.py）、
模型（trade/s5.py）、推定器（trade/vstar3.py）と一致することを確かめる（確証の s4_check_prereg と同じ考え方）。
PYTHONPATH=. python3 -m experiments.s5_check_prereg  → out/s5a/prereg_check.txt"""
import json, inspect, re
from experiments import s5_prof1 as S
from trade import s5, vstar3, vstar

pre = json.load(open("experiments/prereg_s5a.json"))
src_c = inspect.getsource(S); src_m = inspect.getsource(s5); src_v = inspect.getsource(vstar3)
src_j = src_c.split("def judge")[1]
lines = []; ok = True


def chk(name, cond):
    global ok
    lines.append(("一致 " if cond else "不一致 ") + name); ok &= bool(cond)


chk("種：確かめ用は cseed 500〜599", pre["seeds"] == [500, 599] and 'range(pre["seeds"][0], pre["seeds"][1] + 1)' in src_c)
chk("凍結のときは試しの設定が切れている（S5_DRY が 1 でない）", S.DRY is False and S.MAIN == "out/s5a/main.jsonl" and S.JUDGE == "out/s5a/judge.json")
chk("パターン・退去率はスクリプトと同じ", pre["patterns"] == S.PATS and pre["nus"] == S.NUS)
chk("入れ替わりは幾何（観察 2 の入れ替わりの群と同じ）、費用なし", 'turnover=dict(kind="geo", nu=nu)' in src_c and "cost" not in inspect.getsource(S.kw_of))
chk("基準：TP+top3 と改良版top3（〔貸〕20・〔清〕・〔再〕all）", set(pre["bases"]) == set(S.BASES) and S.BASES["TP+top3"] == dict(protocol="topk", ask=3)
    and S.BASES["改良版top3"] == dict(credit=20.0, settle=True, reconsent="all", protocol="topk", ask=3))
chk("腕：TP+top3 は 6、改良版top3 は 7（同意だけ記録と TP+top3 の記録なしは走らせない）",
    len(pre["arms"]["TP+top3"]) == 6 and len(pre["arms"]["改良版top3"]) == 7 and "同意だけ記録" not in sum(pre["arms"].values(), [])
    and "記録なし" not in pre["arms"]["TP+top3"] and all(a in S.ARMS for a in sum(pre["arms"].values(), [])))
n = sum(len(pre["arms"][b]) for b in pre["bases"]) * len(pre["patterns"]) * len(pre["nus"]) * (pre["seeds"][1] - pre["seeds"][0] + 1)
chk("局の数 10,400", n == pre["n_games"] == 10400 and len(S.tasks_main(pre, list(range(500, 600)))) == 10400)
chk("役者の対応：腕 → 役者（盤の設計者・記録者・検証者）", [type(a).__name__ for a in S.actors_of("狭い盤")] == ["Designer"]
    and S.actors_of("狭い盤")[0].board is s5.NARROW and S.actors_of("戻せる手の多い盤")[0].board is s5.RICH
    and S.actors_of("記録なし")[0].mode == "none" and S.actors_of("額まで記録")[0].mode == "amount"
    and S.actors_of("一律に禁止")[0].mode == "ban" and S.actors_of("代わりの手")[0].mode == "alt")
chk("狭い盤は張り出しの手を外す", 'mv[0] != "ext"' in inspect.getsource(s5.NarrowBoard) and "張り出しの付け外しの手を一覧から外す" in pre["actors"]["designer"]["boards"]["狭い盤"])
chk("戻せる手の多い盤：上下の開口 6・上下の収納 4（1.5 ㎡、W・E の列）・部位 31", len(s5.V_OPEN) == 6 and len(s5.V_STORE) == 4 and s5.CLO_V == 1.5
    and len(s5.RICH.keys()) == 31 and "1.5 ㎡" in pre["actors"]["designer"]["boards"]["戻せる手の多い盤"] and "部位は 31" in pre["actors"]["designer"]["boards"]["戻せる手の多い盤"])
chk("記録なしは〔再〕〔清〕を外し〔貸〕を残す", 'self.allows_reconsent = False' in src_m and 'self.allows_settle = False' in src_m and "credit" not in inspect.getsource(s5.Recorder.__init__)
    and "〔貸〕" in pre["actors"]["recorder"]["modes"]["記録なし"])
chk("額の記録：損をする戸の受け取った額と、再同意で払われた額", "pay[j] > 1e-12" in inspect.getsource(s5.Recorder.trade) and 'kind == "paid"' in inspect.getsource(s5.Recorder.recon))
chk("記録を読む住み手：主は入居した世帯（new）、感度は全員（all）と読まない（None）", 'ledger_mode="new"' in inspect.getsource(vstar3.record3)
    and 'ledger_sens=("all", None)' in inspect.getsource(vstar3.record3) and '"all"' in src_j and '"None"' in src_j)
chk("制約：同じ列で二層まで", s5.Verifier.LIMIT == 2 and "二層まで" in pre["actors"]["verifier"]["constraint"])
chk("代わりの手：日照の合計が大きい方を外す（同じなら下の階）", "sun.sum()" in inspect.getsource(s5.Verifier.evals) and "日照の合計が大きい方" in pre["actors"]["verifier"]["modes"]["代わりの手"])
chk("一律に禁止は一覧から外し、代わりを返さない", 'if self.mode == "ban" or mv[0] != "ext":' in inspect.getsource(s5.Verifier.evals))
chk("組の手の額は二つの係数をともに数える（主）", "多めに数える" in inspect.getsource(s5.rev_of) and "両方を数える" in pre["actors"]["recorder"]["modes"]["額まで記録"])
chk("役者は役の枠一つで数える（GA）。〔数〕は役者の種類を count に入れる", 'rec["GA"]' in src_v and "(act,)" in src_j)
chk("ε：基準の腕の 8 セル × 2、並べ替え 100 回・種 1・95 パーセンタイルの最大", pre["epsilon"]["bases"] == ["TP+top3", "改良版top3"] and pre["epsilon"]["perm"] == 100
    and pre["epsilon"]["seed"] == 1 and 'vstar.perm_floor(C(b, "基準", p, nu)' in src_j and "np.percentile(out, 95)" in inspect.getsource(vstar.perm_floor))
chk("ブートストラップ：V*n 1,000 回・種 0、η 2,000 回・種 20261111", pre["bootstrap"]["n"] == 1000 and pre["bootstrap"]["seed"] == 0 and pre["bootstrap"]["eta_n"] == 2000
    and pre["bootstrap"]["eta_seed"] == 20261111 and 'B = bs["n"]' in src_j and 'BE = bs["eta_n"]' in src_j)
chk("差は同じ構成の組（cs をそろえる）", 'assert [x["cs"] for x in recsA] == [x["cs"] for x in recsB]' in inspect.getsource(vstar3.diff3)
    and 'assert [x["cs"] for x in A] == [x["cs"] for x in B]' in inspect.getsource(S.eta_diff))
chk("規則 lower：区間の上端 ＜ 0、higher：下端 ＞ 0", 'return x["hi"] < 0' in inspect.getsource(S.passes) and 'return x["lo"] > 0' in inspect.getsource(S.passes)
    and "上端が 0 より小さい" in pre["rules"]["lower"] and "下端が 0 より大きい" in pre["rules"]["higher"])
chk("判定：すべてのセルで支持、一部なら一部支持", '"支持" if n_ok == len(rows)' in src_j and "一部支持" in pre["rules"]["verdict"])
qs = {p["quantity"] for p in pre["predictions"]}
chk("予言の量は eta・Vn外・Vn数 だけ、規則は lower・higher だけ", qs <= {"eta", "Vn外", "Vn数"} and {p["rule"] for p in pre["predictions"]} <= {"lower", "higher"})
chk("予言の腕と比べる腕は、その基準で走らせる腕", all(p["arm"] in pre["arms"][b] and p["ref"] in pre["arms"][b] for p in pre["predictions"] for b in p["bases"]))
chk("予言のセルは事前登録のパターンと退去率の中", all(c[0] in pre["patterns"] and c[1] in pre["nus"] for p in pre["predictions"] for c in p["cells"]))
vc = {("TP+top3", 0.0)} | {("改良版top3", x) for x in (0.0, 0.05, 0.1)}
chk("V*n の予言は Vn_cells の範囲（TP+top3 は入れ替わりなし、改良版top3 は 0・5・10 %）",
    all((b, c[1]) in vc for p in pre["predictions"] if p["quantity"].startswith("Vn") for b in p["bases"] for c in p["cells"]))
chk("予言の文の数値と cells の数が合う（文に書いたセルの数）", all(str(len(p["cells"]) * len(p["bases"])) + " セル" in p["text"] or "セル）" not in p["text"]
    for p in pre["predictions"]))
chk("定義からの確かめ K1〜K3 を判定のスクリプトで計算する", "K1=" in src_j and "K2=" in src_j and "K3=" in src_j)
open("out/s5a/prereg_check.txt", "w").write("\n".join(lines) + ("\nすべて一致\n" if ok else "\n不一致あり\n"))
print("\n".join(lines)); print("すべて一致" if ok else "不一致あり")
