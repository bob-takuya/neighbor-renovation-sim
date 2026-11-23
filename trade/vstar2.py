"""確証の実験のための推定器の拡張。trade/vstar.py（名付けで凍結）は変えず、import して次を足した。

1. 清算所の作者（市場型 M）。s4 の記録 ch_quotes＝(年, 戸, 係数) から、清算所が 30 年目に有効な知識を作る。
   入れ替わった住戸についての事実は、住み手の作者と同じ規則で無効になる。
   record2 は vstar.record の値をそのまま返し、M の局では Gch（清算所の知識だけで解いた最適、長さ 21）を足す。
2. 代表の役の作者（代表型 Rep）。代表は今の住み手（または去った住み手）として住戸の枠で作者に入るが、
   代表の住戸は建物ごとにランダムなので、構成の集団で数える相互情報量では、住戸の枠に代表の知識が散り、
   集団の中で「代表の知識」として数えられない。vstar が中央の作者と「去った住み手のうち最も多く知る者」を
   役の枠で数えるのと同じく、代表を務めた住み手（今の代表と去った代表）のうち最も多く知る者を、役の枠 Grep として足す。
3. 作者の数え方の切り替え。with_ch(recs) は、Gch を作者 G に足した写しを返す。
   vstar.population(recs, key) は清算所を作者に数えない値、vstar.population(with_ch(recs), key) は数える値になる。
   作者を一人足すと max の項は減らないので、数える値は数えない値以下になる（定義から）。
囮でない局では record2 は vstar.record と同じものを返す（tests/test_s4.py）。
"""
from __future__ import annotations
from trade import vstar
from trade.vstar import guess


def ch_knowledge(r):
    """清算所が 30 年目に持っている、有効な事実 (戸, 係数) の集合"""
    tenant = list(range(9)); nxt = 9
    tl = [(t, 0, (j, p)) for (t, j, p) in r["ch_quotes"]] + [(t, 1, u) for (t, u) in r["turns"]]
    tl.sort(key=lambda x: (x[0], x[1]))
    facts = set()
    for t, typ, x in tl:
        if typ == 0:
            j, p = x
            facts.add((j, p, tenant[j]))
        else:
            tenant[x] = nxt; nxt += 1
    return {(j, p) for (j, p, a) in facts if tenant[j] == a}


def rep_knowledge(r):
    """代表を務めた住み手のうち、30 年目に有効な事実が最も多い者の知識 (戸, 係数) の集合。
    事実は vstar.knowledge と同じ規則で数える（自分の係数、聞いた言い値と要求で明かされた係数。入れ替わった住戸の事実は無効）"""
    tenant = list(range(9)); nxt = 9
    facts = {a: {(a, p, a) for p in "asq"} for a in range(9)}
    reps = set()
    tl = [(q[0], 0, q) for q in r["quotes"]] + [(t, 1, u) for (t, u) in r["turns"]] + [(t, 2, u) for (t, u) in r["reps"]]
    tl.sort(key=lambda x: (x[0], x[1]))
    for t, typ, x in tl:
        if typ == 0:
            _, i, j, p = x
            facts.setdefault(tenant[i], set()).add((j, p, tenant[j]))
        elif typ == 1:
            tenant[x] = nxt; facts[nxt] = {(x, p, nxt) for p in "asq"}; nxt += 1
        else:
            reps.add(tenant[x])
    valid = lambda F: {(j, p) for (j, p, a) in F if tenant[j] == a}
    return max((valid(facts[a]) for a in reps), key=len)


def record2(r, W0, start_author=None, F=None):
    rec = vstar.record(r, W0, start_author, F)
    if r.get("decoy") == "M+":
        rec["Gch"] = guess(r["W_final"], ch_knowledge(r)).tolist()
        rec["ch_n"] = len(r["ch_quotes"])
    if r.get("decoy") == "Rep+":
        rec["reps"] = [list(x) for x in r["reps"]]
        rec["Grep"] = guess(r["W_final"], rep_knowledge(r)).tolist()
    return rec


def with_ch(recs):
    """清算所を作者に数える写し"""
    return [dict(x, G=x["G"] + [x["Gch"]]) for x in recs]


def with_roles(recs, count_ch=False):
    """役の枠を作者に足した写し。代表型の局では代表の役の枠をつねに足し、M の局では count_ch のとき清算所を足す"""
    out = []
    for x in recs:
        G = x["G"]
        if "Grep" in x:
            G = G + [x["Grep"]]
        if count_ch:
            G = G + [x["Gch"]]
        out.append(dict(x, G=G) if G is not x["G"] else x)
    return out


def value(recs, key, count_ch=False):
    return vstar.population(with_roles(recs, count_ch), key)


def boot(recs, key, n=1000, seed=0, count_ch=False):
    return vstar.boot(with_roles(recs, count_ch), key, n=n, seed=seed)
