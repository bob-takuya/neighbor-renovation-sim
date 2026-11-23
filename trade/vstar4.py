"""職能 2 のための推定器の小さな足し。trade/vstar.py・vstar2.py・vstar3.py（凍結）は変えず、import して次だけを足した。

全知の仲介者 BR（名付けの偽装）の局では、vstar.record が仲介者を「中央の作者」の枠（G の 10 番目）として必ず作者に数える。
BR を作者に数えない値〔外〕を求めるため、その枠を、性向を知らない者の推定（始まりの形 F）に置き換えた写しを返す。
聞いた言い値だけを知る仲介者（s6.Mediator）は、vstar3 の役の枠（GA）で数える（count=("broker",) で〔数〕、count=() で〔外〕）。
"""
from __future__ import annotations
from trade import s0
from trade.vstar import cfgvec

F_VEC = cfgvec(*s0.f_design()).tolist()
CENTRAL = 9          # vstar.record の G の並び：今の住み手 9 戸、中央の作者、去った作者のうち最も多く知る者


def drop_central(recs):
    """中央の作者（BR・計画者・起案 AI）を作者に数えない写し。運用役のない局でも中央の枠はもともと F なので値は変わらない"""
    return [dict(x, G=x["G"][:CENTRAL] + [F_VEC] + x["G"][CENTRAL + 1:]) for x in recs]
