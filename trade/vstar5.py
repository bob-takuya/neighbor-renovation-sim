"""職能 3 のための推定器の小さな足し。trade/vstar.py・vstar3.py・vstar4.py（凍結）は変えず、import して次だけを足した。

1. 運用役の枠だけの知識の量 actor_I：運用役（名か種類）の役の枠 GA だけで全知の最適を言い当てる度合い
   Σ_p I(Z_p; GA_p) ／ Σ_p H(Z_p)。population3 の Imax（作者のうち最も多く知る者の値）と並べて、運用役の知識が
   最も多く知る住み手の知識を超えるかを示す。
2. 聞く範囲を選ぶ者の選び方（1 局目の記録から、建物ごとの範囲を決める）。
   choose_knowing：建物ごとに、各範囲の 1 局目の V*n〔外〕の点ごとの値（vstar.pointwise。名付けの抵抗の試験と同じ規則）が
     最も大きい範囲。点ごとの値は、建物の 9 世帯の性向（全知の最適 Z）を知って初めて計算できる。
   choose_distribution：1 局目の 100 棟をまとめた集団の V*n〔外〕（vstar3.population3）が最も大きい範囲を、どの建物にも。
     棟を入れ替えても値は変わらないので、どの建物のどの世帯の性向にも依らない。
   同じ値のときは、基準の範囲（上位 3 手、prefer）が最大の中にあればそれを変えず、なければ order の前のものを選ぶ。
"""
from __future__ import annotations
import numpy as np
from trade import vstar, vstar3
from trade.vstar import MI, H_


def actor_I(recs, who):
    """運用役 who（名か種類）の役の枠だけの Σ_p I(Z_p; GA_p) ／ Σ_p H(Z_p)。運用役のない局が混じれば None"""
    G = []
    for x in recs:
        g = [v for nm, v in x.get("GA", {}).items() if nm == who or x["actor_kind"][nm] == who]
        if len(g) != 1:
            return None
        G.append(g[0])
    Z = np.array([x["Z"] for x in recs]); G = np.array(G)
    HZ = sum(H_(list(Z[:, p])) for p in range(Z.shape[1]))
    if HZ < 1e-9:
        return 0.0
    return float(sum(MI(list(Z[:, p]), list(G[:, p])) for p in range(Z.shape[1])) / HZ)


def _pick(vals, order, prefer):
    """vals：{範囲: 値}。最大のうち prefer があれば prefer、なければ order の前のもの"""
    m = max(vals.values())
    top = [s for s in order if vals[s] >= m - 1e-12]
    return prefer if prefer in top else top[0]


def choose_knowing(by_scope, order, prefer="上位3手"):
    """by_scope：{範囲: 1 局目の記録の一覧（cs の順、同じ構成）}。返り値：({cs: 範囲}, 点ごとの値)"""
    css = [x["cs"] for x in by_scope[order[0]]]
    for s in order:
        assert [x["cs"] for x in by_scope[s]] == css
    pw = np.array([vstar.pointwise(by_scope[s], "Xn") for s in order])       # (範囲, 建物)
    return {int(cs): _pick({s: pw[a, n] for a, s in enumerate(order)}, order, prefer) for n, cs in enumerate(css)}, pw


def choose_distribution(by_scope, order, prefer="上位3手"):
    """1 局目の集団の V*n〔外〕が最も大きい範囲を一つ。返り値：(範囲, {範囲: 値})"""
    v = {s: vstar3.population3(by_scope[s], "Xn")["V"] for s in order}
    return _pick(v, order, prefer), v
