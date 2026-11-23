"""職能 3 のための模型 s7。s6（職能 2 で凍結）を変えず、import して、性向を知りうる運用役を s5 の差し込み口で足した。
新しい局の本文は書かない。s7.run は s6.run をそのまま呼び、運用役が「影響を見せる道具」のときだけ、届かない戸の制約を外す
（下の Tool を参照）。運用役を置かない局と、道具を置かない局では、s7.run は s6.run と同じ乱数を同じ順に使い、同じ記録を返す
（tests/test_s7.py）。

運用役（知ることの少ない順）
  Scope(protocol, ask, knows=False)   聞く範囲を選ぶ者（s5.ScopeChooser の子、種類 "scope"）
             建物ごとに、言い値を誰にどこまで聞くか（全員 full・上位 1 手 topk 1・上位 3 手 topk 3・近い戸だけ near）を選ぶ。
             自分では手を打たない。選んだ範囲を init で返すだけで、局の中では何もしない。
             選ぶのは局の外で行う（experiments/s7_prof3.py）。選ぶのに 1 局目（局の種 cseed＋50000）を、測るのに 2 局目
             （局の種 cseed、運用役を置かない基準と同じ乱数）を使う。
             knows=True（性向を知る）：建物の 9 世帯の性向を知り、1 局目の各範囲の結果を、建物ごとの V*n〔外〕の点ごとの値
                 （vstar.pointwise、名付けの抵抗の試験と同じ規則）で比べて選ぶ。選ぶのに使った知識（竣工の時の全世帯の係数）を
                 know_all(−1) で残す。入れ替わった世帯についての知識は、住み手の知識と同じ規則で無効になる。
             knows=False（分布だけを知る）：建物ごとの世帯を見分けられないので、1 局目の 100 棟をまとめた集団の値
                 （パターン・退去率・基準ごとの V*n〔外〕）が最も高い範囲を、そのセルのどの建物にも選ぶ。facts は空のまま。
  Tool(shows)                 AI を道具として使う建築家（s5.ToolAI の子、種類 "tool"）
             提案者が手を検討するとき、その手の影響を計算して住民に見せる。決めない（決め方の規則は変えない）。
             見せることの効き目（最小の形）：手で満足の動く世帯は、近い戸でなくても、見せられた影響を見て言い値を言いに来る
                 （満足の量で見せる場合は、言い値を聞かなくても損の額が見える）。模型では、近い戸にしか聞けない条件の
                 「損をする世帯に届かない戸がいる手は打てない」という制約を外すことと同じになる。s6.run の venue の働きは、
                 届く戸（reach）を決めることと、相手一人あたりの手間 tau だけなので、tau＝0 の近い戸だけの条件では、
                 venue を外した局と同じになる（s7.run がそうする。tests/test_s7.py で確かめる）。
                 全員に届く条件では、提案者はもともと誰が損をするかを見て言い値を聞くので、見せても局は変わらない（定義から）。
             shows="physical"（物理の量で見せる）：誰の日照・眺望・静けさ（開口の向き）・広さが何単位変わるか。性向は知らない
                 （facts は空のまま）。
             shows="satisfaction"（満足の量で見せる。盲目的な使い方）：道具が、データから住み手の性向を読み、各世帯の満足が何 ㎡
                 分変わるかを見せる。道具が使われた年ごとに、その年の全員の係数を know_all(t) で残す。
             提案者がどの手を検討し、誰に補償し、誰の同意を得るかは、どちらでも運用役なしと同じ。住み手が知ることも同じ
                 （提案者は、損をする世帯の額を、聞いても見せられても、同じ係数として知る）。
  全知の計画者は、名付けの帰無の条件 N3 と同じ置き方（s6.run の regime "P"）で走らせる。運用役の型は使わない。
"""
from __future__ import annotations
from trade import s5, s6

PROTOCOLS = {"全員": dict(protocol="full", ask=3), "上位1手": dict(protocol="topk", ask=1),
             "上位3手": dict(protocol="topk", ask=3), "近い戸だけ": dict(protocol="near", ask=3)}
ORDER = ["全員", "上位1手", "上位3手", "近い戸だけ"]          # 同じ値のときは前のものを選ぶ（名付けの抵抗の試験と同じ順）


class Scope(s5.ScopeChooser):
    """聞く範囲を選ぶ者。選んだ範囲 scope（PROTOCOLS の鍵）を init で返す。knows なら竣工の時の全世帯の係数を知る"""

    def __init__(self, scope, knows, name="scope"):
        super().__init__(name)
        assert scope in PROTOCOLS
        self.scope = scope
        self.knows = bool(knows)

    def init(self, ctx):
        if self.knows:
            self.know_all(ctx, -1)
        return dict(PROTOCOLS[self.scope])


class Tool(s5.ToolAI):
    """AI を道具として使う建築家。shows は "physical"（物理の量）か "satisfaction"（満足の量）"""
    opens_reach = True

    def __init__(self, shows, name="tool"):
        super().__init__(name)
        assert shows in ("physical", "satisfaction")
        self.shows = shows
        self.n_shown = 0            # 手の影響を見せた回数（提案者が手を検討した回数）
        self._read = None           # 満足の量で見せる場合に、性向を最後に読んだ年

    def evals(self, t, i, evals, ctx):
        self.n_shown += 1
        if self.shows == "satisfaction" and self._read != t:
            self.know_all(ctx, t)   # その年の住み手の性向を読む（読んだ年の全員の係数を知る）
            self._read = t
        return evals


def run(W, regime, seed, actors=None, **kw):
    """s6.run を呼ぶ。影響を見せる道具を置いた局では、近い戸にしか聞けない制約（venue の G＝0）を外す"""
    actors = list(actors or [])
    if any(getattr(a, "opens_reach", False) for a in actors) and kw.get("venue"):
        v = kw["venue"]
        if not (v.get("G") == 0 and float(v.get("tau", 0.5)) == 0.0 and "alpha" not in v):
            raise NotImplementedError("見せる道具を置けるのは、手間のない近い戸だけの条件（G＝0・tau＝0）だけ")
        kw = dict(kw, venue=None)
    return s6.run(W, regime, seed, actors=actors, **kw)
