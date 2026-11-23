"""職能の実験のための模型 s5。s3（観察 2 で凍結）を写し、変えずに、運用役（建築家の職能）の部品の差し込み口を足した。
既定の設定（運用役なし）では s3.run と同じ乱数を同じ順に使い、同じ記録を返す（tests/test_s5.py）。

運用役は actors＝[運用役, …] で渡す。運用役はどれも Actor の子で、次の差し込み口のうち必要なものだけを持つ。
  board                 盤（Board）。盤の設計者だけが持つ。部位の一覧・手の一覧・満足の計算・始まりの形を決める
  init(ctx)             局の始めに呼ぶ。聞く範囲などを上書きする dict を返してよい（職能 3 の「聞く範囲を選ぶ者」のため）
  evals(t, i, evals, ctx)
                        提案者 i が検討する手の一覧（手, 各戸の満足の変化 dS, 共通の物差しの変化 dC, 次の形）を受け取り、
                        変えた一覧を返す。聞く範囲で絞る前に呼ぶ（検証者・道具の AI・仲介者のため）
  trade(t, i, mv, dS, pay, ctx)
                        手が打たれた直後に呼ぶ（記録者・仲介者のため）
  recon(t, n, p, kind, G, ctx)
                        入居時の再同意〔再〕で部位 p を見直した直後に呼ぶ。kind は "accept"・"paid"・"revert"
  revert_ok(p, cfg, ctx)
                        再同意で部位 p を始まりの値へ戻してよいか（検証者のため）。False なら戻さない
  allows_reconsent / allows_settle
                        False なら改良版の〔再〕・〔清〕を使えない（記録者が記録を残さない場合）
  facts                 運用役が知った事実 (年, 戸, 係数, 順) の一覧（learn で足す）。係数は a＝広さ・s＝日照・q＝静けさ−近さ（s1.revealed と同じ）
  full                  運用役が全員の係数を知った時 (年, 順) の一覧（know_all で足す。全知の運用役のため。年 −1 は竣工の前）
記録には、運用役ごとに actor_facts[名]＝facts、actor_full[名]＝full、actor_kind[名]＝種類 を返す。
推定器 trade/vstar3.py は、これを「役」の枠（住戸ではなく運用役ごとに一つ）として作者に足すか足さないかを切り替える。

この版で実装した運用役（職能 1：性向を知らない運用役）
  Designer(board)       盤の設計者。board は BASE（今の盤）・NARROW（張り出しを打てない盤）・RICH（戻せる手の多い盤）
  Recorder(mode)        記録者。mode は "none"（記録なし：〔再〕〔清〕を使えない）・"consent"（同意と起案者だけ）・
                        "amount"（言い値の額まで）。"amount" では、記録した額から分かる事実を ledger に (年, 戸, 係数, 順) で残す
  Verifier(mode)        検証者。制約は「同じ列で張り出せるのは二層まで」。mode は "ban"（破る手を一覧から外し、代わりを返さない）・
                        "alt"（破る手の代わりに、同じ列のほかの張り出しを一つ外す組の手を返す）
差し込み口だけの型（中身はまだない）：Broker（職能 2）、ScopeChooser・ToolAI・Planner（職能 3）。
"""
from __future__ import annotations
import numpy as np
from trade.s0 import (S_all, MU, W_BAR, legal_moves, all_moves, apply, move_label, move_type, decide, vote, f_design,
                      quantities, phi, UNITS, FLOOR, COL, A0, EXT_A, CLO_A, SUN0, A_LO, A_HI, SUN_HI, SHADE, b_units,
                      wstar_linear)
from trade.s1 import KEYS, NEAR, part_of, revealed, getp, move_cost, best_net
from trade.s3 import TYPE, market, g_of, refund_frac, renew_base

F0 = f_design()


# ============================================================ 盤
class Board:
    """今の盤（基準）。s0・s1 の関数をそのまま使う"""
    name = "base"
    nk, no = 6, 6

    def S(self, W, e, k, o, form="linear"):
        return S_all(W, e, k, o, form)

    def moves(self, e, k, o):
        return all_moves(e, k, o)

    def legal(self, i, e, k, o):
        return legal_moves(i, e, k, o)

    def start(self, form="linear"):
        return f_design(form)

    def keys(self):
        return KEYS

    def recon_parts(self, scope):
        parts = [("o", x) for x in range(6)] + [("k", x) for x in range(6)]
        if scope == "all":
            parts += [("e", x) for x in range(9)]
        return parts

    def wstar(self, W):
        return wstar_linear(W)

    def label(self, mv, i, e):
        if mv[0] == "swap":
            return f"張り出す {UNITS[mv[1]]}・張り出しを外す {UNITS[mv[2]]}"
        return move_label(mv, i, e)

    def mtype(self, mv, e):
        if mv[0] == "swap":
            return f"張り出す{FLOOR[mv[1]]}F・外す{FLOOR[mv[2]]}F"
        return move_type(mv, e)


class NarrowBoard(Board):
    """狭い盤：張り出しを打てない（張り出しの付け外しの手を一覧から外す。張り出しは始まりの形のまま）"""
    name = "narrow"

    def moves(self, e, k, o):
        return [mv for mv in all_moves(e, k, o) if mv[0] != "ext"]

    def legal(self, i, e, k, o):
        return [mv for mv in legal_moves(i, e, k, o) if mv[0] != "ext"]


# 戻せる手の多い盤：上下の住戸のあいだに、戻せる造作（収納と開口）を足す。
#   上下の開口 6（3 列 × 2 つの床）：o[6..11]。番号 6＋3j＋c（j＝0 は 3F|2F、j＝1 は 2F|1F、c は列）。上下の二戸の開口の数が 1 増える
#   上下の収納 4（W・E の列 × 2 つの床）：k[6..9]。番号 6＋2j＋(0 if W else 1)。＋1 で上の戸が 1.5 ㎡ 使う、−1 で下の戸が使う
#   広さは 44〜62 ㎡ に収まるので（中の列には上下の収納を置かない）、満足は線形のまま部位ごとに分かれる
CLO_V = 1.5
V_OPEN = [(6 + 3 * j + c, 3 * j + c, 3 * j + c + 3) for j in range(2) for c in range(3)]          # (番号, 上の戸, 下の戸)
V_STORE = [(6 + 2 * j + cc, 3 * j + c, 3 * j + c + 3) for j in range(2) for cc, c in enumerate((0, 2))]


class RichBoard(Board):
    name = "rich"
    nk, no = 10, 12

    def _q(self, e, k, o):
        area, sun, op = quantities(e, k[:6], o[:6])
        for x, u, d in V_STORE:
            area[u] += CLO_V * k[x]; area[d] -= CLO_V * k[x]
        for x, u, d in V_OPEN:
            op[u] += o[x]; op[d] += o[x]
        return area, sun, op

    def S(self, W, e, k, o, form="linear"):
        if form != "linear":
            raise NotImplementedError("戻せる手の多い盤は線形の満足だけ")
        a, s, op = self._q(e, k, o)
        assert a.min() >= A_LO - 1e-9 and a.max() <= A_HI + 1e-9
        return (phi(a, s, op, form) * W).sum(-1)

    def moves(self, e, k, o):
        mv = all_moves(e, k[:6], o[:6])
        for x, _, _ in V_STORE:
            if k[x] < 1:
                mv.append(("shift", x, 1))
            if k[x] > -1:
                mv.append(("shift", x, -1))
        for x, _, _ in V_OPEN:
            mv.append(("open", x))
        return mv

    def legal(self, i, e, k, o):
        raise NotImplementedError("戻せる手の多い盤は他戸の手も打てる決め方（+）だけ")

    def start(self, form="linear"):
        e, k, o = f_design(form)
        return e, np.concatenate([k, np.zeros(4, int)]), np.concatenate([o, np.zeros(6, int)])

    def keys(self):
        return [("e", x) for x in range(9)] + [("k", x) for x in range(10)] + [("o", x) for x in range(12)]

    def recon_parts(self, scope):
        parts = [("o", x) for x in range(12)] + [("k", x) for x in range(10)]
        if scope == "all":
            parts += [("e", x) for x in range(9)]
        return parts

    def wstar(self, W):
        """線形の満足での厳密な最適（部位ごとに分かれる）"""
        W = np.asarray(W, float)
        _, e, k6, o6 = wstar_linear(W)
        da = 1.0 / (A_HI - A_LO)
        k = np.concatenate([k6, np.zeros(4, int)]); o = np.concatenate([o6, np.zeros(6, int)])
        for x, u, d in V_STORE:
            c = (W[u, 0] - W[d, 0]) * CLO_V * da
            k[x] = 1 if c > 1e-12 else (-1 if c < -1e-12 else 0)
        for x, u, d in V_OPEN:
            c2 = 0.5 * ((W[u, 3] - W[u, 2]) + (W[d, 3] - W[d, 2]))
            o[x] = 1 if c2 > 1e-12 else 0
        return float(self.S(W, e, k, o).sum()), e, k, o

    def label(self, mv, i, e):
        kind, x = mv[0], mv[1]
        if kind == "shift" and x >= 6:
            _, u, d = [v for v in V_STORE if v[0] == x][0]
            return f"上下の収納を {UNITS[d]}→{UNITS[u]}" if mv[2] > 0 else f"上下の収納を {UNITS[u]}→{UNITS[d]}"
        if kind == "open" and x >= 6:
            _, u, d = [v for v in V_OPEN if v[0] == x][0]
            return f"上下の開口 {UNITS[u]}|{UNITS[d]}"
        return Board.label(self, mv, i, e)

    def mtype(self, mv, e):
        kind, x = mv[0], mv[1]
        if kind in ("shift", "open") and x >= 6:
            return "上下の収納" if kind == "shift" else "上下の開口"
        return Board.mtype(self, mv, e)


BASE, NARROW, RICH = Board(), NarrowBoard(), RichBoard()


def coef_of(p, j):
    """部位 p を変えて戸 j の満足が動くとき、その額から分かる係数（s1.revealed と同じ規則を部位で書いたもの）"""
    if p[0] == "e":
        return "a" if j == p[1] else "s"
    if p[0] == "o":
        return "q"
    return "a"


def part_of5(mv):
    """手の部位。組の手（swap）は張り出す部位を返す（外す部位は二つ目の記録に書く）。ほかは s1.part_of と同じ"""
    if mv[0] == "swap":
        return ("e", mv[1])
    return part_of(mv)


def rev_of(mv, j):
    """手 mv の言い値から分かる戸 j の係数の一覧。組の手（swap）では、戸 j の満足を動かす係数をすべて数える（多めに数える）"""
    if mv[0] != "swap":
        return [revealed(mv, j)]
    out = []
    xa, xr = mv[1], mv[2]
    if j in (xa, xr):
        out.append("a")
    if any(j == x + 3 * d for x in (xa, xr) for d in (1, 2)):
        out.append("s")
    return out


# ============================================================ 運用役
class Actor:
    """運用役の差し込み口。何もしない既定の振る舞いを持つ"""
    kind = "actor"
    board = None
    allows_reconsent = True
    allows_settle = True

    def __init__(self, name=None):
        self.name = name or self.kind
        self.facts = []          # (年, 戸, 係数, 順)
        self.full = []           # (年, 順)：全員の係数を知った時

    def learn(self, ctx, t, j, c):
        """戸 j の係数 c を知る。順の番号を付けて facts に残し、番号を返す（推定器が入れ替わりとの前後を決めるのに使う）"""
        sq = ctx["seq"]()
        self.facts.append((t, j, c, sq))
        return sq

    def know_all(self, ctx, t):
        """その時点の全員の係数を知る（全知の運用役のため）"""
        self.full.append((t, ctx["seq"]()))

    def init(self, ctx):
        return {}

    def evals(self, t, i, evals, ctx):
        return evals

    def trade(self, t, i, mv, dS, pay, ctx):
        pass

    def recon(self, t, n, p, kind, G, ctx):
        pass

    def revert_ok(self, p, cfg, ctx):
        return True


class Designer(Actor):
    """盤の設計者。知るのは平均の性向と母集団の分布だけで、個々の世帯の係数は知らない（facts は空のまま）"""
    kind = "designer"

    def __init__(self, board, name=None):
        super().__init__(name or f"designer:{board.name}")
        self.board = board


class Recorder(Actor):
    """記録者。棋譜に、誰がいつ誰との合意で何を変えたかを残す。
    none：記録を残さない。改良版の〔再〕（入居時の再同意）と〔清〕（転出時の精算）は記録があって初めて使えるので、使えない。
          〔貸〕（建物の中の貸し借り）は、貸した世帯と借りた世帯の二者のあいだの勘定として残す（棋譜を要しない）。
    consent：同意した世帯と起案者だけを記録する。額は書かないので、記録から係数は分からない。
    amount：言い値の額まで記録する。打たれた手で損をする戸が受け取った額（その戸の言い値）と、再同意で戻すのをやめてもらう
          ために払われた額（新しい世帯の言い値）を書く。額から分かる (戸, 係数) を facts と ledger に残す。"""
    kind = "recorder"

    def __init__(self, mode, name=None):
        super().__init__(name or f"recorder:{mode}")
        self.mode = mode
        if mode == "none":
            self.allows_reconsent = False
            self.allows_settle = False

    def trade(self, t, i, mv, dS, pay, ctx):
        if self.mode != "amount":
            return
        for j in range(9):
            if j != i and pay[j] > 1e-12:
                for c in rev_of(mv, j):
                    ctx["ledger"].append((t, j, c, self.learn(ctx, t, j, c)))

    def recon(self, t, n, p, kind, G, ctx):
        if self.mode == "amount" and kind == "paid":
            c = coef_of(p, n)
            ctx["ledger"].append((t, n, c, self.learn(ctx, t, n, c)))


class Verifier(Actor):
    """検証者。取引で決めてはならない制約（ここでは構造：同じ列で張り出せるのは二層まで）を確かめる。
    知るのは制約と、手が建物に及ぼす物理的な影響（広さ・日照の時間）だけで、個々の世帯の係数は知らない。
    ban：制約を破る手を、提案者が検討する手の一覧から外す（代わりの手は返さない）。
    alt：制約を破る張り出しの手を、同じ列のほかの二つの張り出しのうち一つを外す組の手に置き換えて返す。
         どちらを外すかは物理の量だけで決める：外したあとの建物の日照の合計が大きい方（同じなら下の階）。
    入居時の再同意で、戻すと制約を破る部位は、どちらでも戻さない。"""
    kind = "verifier"
    LIMIT = 2

    def __init__(self, mode, name=None):
        super().__init__(name or f"verifier:{mode}")
        self.mode = mode
        self.n_banned = 0
        self.n_alt = 0

    @staticmethod
    def violates(e):
        return any(int(e[c] + e[c + 3] + e[c + 6]) > Verifier.LIMIT for c in range(3))

    def evals(self, t, i, evals, ctx):
        out = []
        SA, W, Wbar, form = ctx["S"], ctx["W"], ctx["Wbar"], ctx["form"]
        e, k, o = ctx["cfg"]()
        S0_ = None
        for x in evals:
            mv, dS, dC, nxt = x
            if not self.violates(nxt[0]):
                out.append(x); continue
            self.n_banned += 1
            if self.mode == "ban" or mv[0] != "ext":
                continue
            xa = mv[1]; col = COL[xa]
            others = [u for u in (col, col + 3, col + 6) if u != xa and e[u] == 1]
            best = None
            for xr in others:
                e1 = e.copy(); e1[xa] = 1; e1[xr] = 0
                _, sun, _ = quantities(e1, k[:6], o[:6])
                key = (float(sun.sum()), FLOOR[xr] * -1)        # 日照の合計が大きい方、同じなら下の階（FLOOR が小さい方）
                if best is None or key > best[0]:
                    best = (key, xr, e1)
            if best is None:
                continue
            _, xr, e1 = best
            if S0_ is None:
                S0_ = SA(W, e, k, o, form); C0 = SA(Wbar, e, k, o, form)
            dS1 = SA(W, e1, k, o, form) - S0_
            dC1 = SA(Wbar, e1, k, o, form) - C0
            self.n_alt += 1
            out.append((("swap", xa, xr), dS1, dC1, (e1, k.copy(), o.copy())))
        return out

    def revert_ok(self, p, cfg, ctx):
        if p[0] != "e":
            return True
        e = cfg[0].copy(); e[p[1]] = ctx["start"][0][p[1]]
        return not self.violates(e)


# ---- 職能 2・3 の型（差し込み口と記録の形だけ）
class Broker(Actor):
    """職能 2：全知でない仲介者。住み手から聞いた言い値だけを知る。
    使い方：evals で多者の取引の束（組の手）を足し、聞いた言い値を learn で facts に残す。決めるのは住民。"""
    kind = "broker"


class ScopeChooser(Actor):
    """職能 3：聞く範囲を選ぶ者。init で dict(protocol=..., ask=...) を返して、その建物の聞く範囲を決める。手は打たない。
    性向を知る場合は full に −1（竣工の前）と入れ替わりの年を入れ、分布だけを知る場合は何も入れない。"""
    kind = "scope"


class ToolAI(Actor):
    """職能 3：AI を道具として使う建築家。evals で手の影響を住民に見せる（物理の量なら facts は空、満足の量なら
    性向を読むので、読んだ年の全員の係数を full に入れる）。決めない。"""
    kind = "tool"


class Planner(Actor):
    """職能 3：全知の計画者（帰無の条件 N3 と同じ置き方）。full に −1 と入れ替わりの年を入れる。"""
    kind = "planner"


# ============================================================ 一局
def run(W, regime, seed, years=30, p_req=0.3, form="linear", pi=0.5, start=None,
        theta=0.0, budget=None, conv_scale=1.0, nu=0.0, newW=None,
        split=False, credit=None, settle=False, reconsent=None,
        protocol="full", ask=3, cost=None, rcost=None, turn_at=None, ref=None,
        turnover=None, restore=None, amort=10, horizon=None, owner=None, venue=None, actors=None):
    actors = list(actors or [])
    boards = [a.board for a in actors if a.board is not None]
    if len(boards) > 1:
        raise ValueError("盤の設計者は一人まで")
    B = boards[0] if boards else BASE
    SA = B.S
    if B is not BASE and (owner or venue or restore or regime in ("P", "BR", "BRdyn", "R", "Rc") or form != "linear"):
        if B is RICH:
            raise NotImplementedError("戻せる手の多い盤で使えるのは、言い値の取引・入れ替わり・改良版の部品だけ")
    W = np.array(W, float).copy()
    ss = np.random.SeedSequence([seed, 7])
    r_req, r_ord, r_vote = [np.random.default_rng(s) for s in ss.spawn(3)]
    REQ = r_req.random((years, 9))
    ORD = np.array([r_ord.permutation(9) for _ in range(years)])
    VOT = r_vote.random((years, 9, 9))
    r_turn = np.random.default_rng(np.random.SeedSequence([seed, 7, 99]))
    TURN = r_turn.random((years, 9))
    r_null = np.random.default_rng(np.random.SeedSequence([seed, 7, 123]))
    r_type = np.random.default_rng(np.random.SeedSequence([seed, 7, 77]))
    r_ven = np.random.default_rng(np.random.SeedSequence([seed, 7, 55]))
    e, k, o = start if start is not None else B.start(form)
    e, k, o = e.copy(), k.copy(), o.copy()
    X0 = (e.copy(), k.copy(), o.copy())
    st = (e.copy(), k.copy(), o.copy()) if ref is None else tuple(a.copy() for a in ref)
    c = np.zeros(9); m = np.zeros(9); b = np.zeros(9)
    n_recon = n_revert = 0; n_exit = 0; exit_bal = 0.0
    moved = np.zeros(9, bool)
    Wbar = np.tile(W_BAR * np.array([1.0, conv_scale, conv_scale, conv_scale]), (9, 1))
    events, quotes, turns, recon = [], [], [], []
    spent = 0.0
    extra = dict(owner_in=0.0, owner_out=0.0, tcost=0.0, vetoed=0, unreached=0, restored=0, refund=0.0)
    Wt = [SA(W, e, k, o, form).sum()]
    acc = c if credit is not None else m
    lim = np.full(9, credit) if credit is not None else budget

    # ---- 運用役の文脈
    ledger, seqlog = [], []          # ledger：(年, 戸, 係数, 順)、seqlog：入れ替わりと記録の順（推定器が読む順を決める）
    _seq = [0]

    def nextseq():
        _seq[0] += 1
        return _seq[0]
    ctx = dict(S=SA, form=form, Wbar=Wbar, start=st, ledger=ledger, seq=nextseq, board=B, blocked=[], mixed=[])
    ctx["cfg"] = lambda: (e, k, o)
    if actors:
        for a in actors:
            ov = a.init(ctx) or {}
            protocol = ov.get("protocol", protocol); ask = ov.get("ask", ask)
        if not all(a.allows_reconsent for a in actors):
            reconsent = None
        if not all(a.allows_settle for a in actors):
            settle = False

    kind = turnover["kind"] if turnover else None
    if kind == "geo":
        nu = turnover["nu"]
    if kind == "mix":
        sig = np.where(r_type.random(9) < turnover.get("piS", 0.72), 0, 1)
        nus = (turnover.get("nuS", 0.32), turnover.get("nuL", 0.087))
    if kind == "renew":
        h0 = renew_base(turnover["nu"], turnover.get("rho", 2.0)); rho = turnover.get("rho", 2.0)
        nu = turnover["nu"]
    tenure = np.zeros(9, int)
    tenant = np.arange(9); nxt_tenant = 9

    def mean_nu(i):
        if kind == "mix":
            return nus[sig[i]]
        return nu

    def hazard(i):
        if kind == "mix":
            return nus[sig[i]]
        if kind == "renew":
            return min(1.0, h0 * (rho if (tenure[i] + 1) % 2 == 0 else 1.0))
        return nu
    turning = (nu > 0 or turn_at or kind is not None) and newW is not None

    author = {}
    own = owner or None
    Lam = set(own.get("Lam", set())) if own else set()

    def owner_value(e0, k0, o0, e1, k1, o1):
        D0, X0_ = market(e0, k0, o0); D1, X1_ = market(e1, k1, o1)
        gam = own.get("gamma", 1.0) * (own.get("M", 10) if own["kind"] == "sub" else 1.0)
        return own.get("beta", 0.5) * float((D1 - D0).sum()) / MU - gam * float((X1_ - X0_).sum())

    def d4_list():
        L = set()
        for typ in ("ext", "store", "open"):
            cnt = 0
            S0_ = SA(W, e, k, o, form)
            for i in range(9):
                for mv in B.legal(i, e, k, o):
                    if TYPE[mv[0]] != typ:
                        continue
                    e1, k1, o1 = apply(mv, i, e, k, o)
                    if SA(W, e1, k1, o1, form)[i] - S0_[i] > 1e-12:
                        cnt += 1; break
            if cnt >= 5:
                L.add(typ)
        return L

    def central(t, W_now):
        nonlocal e, k, o, spent
        if regime == "P":
            e2, k2, o2, cst = best_net(W_now, e, k, o, st, cost, rcost, form)
            events.append(dict(t=t, i=None, part=None, ok=True, planner=True, cons=False, supp=set(), drafter="planner",
                               changed=bool(any((a != b_).any() for a, b_ in zip((e2, k2, o2), (e, k, o))))))
            e, k, o = e2, k2, o2; spent += cst
        else:
            ez, kz, oz, _ = best_net(W_now, e, k, o, st, cost, rcost, form)
            tgt = (ez, kz, oz)
            events.append(dict(t=t, i=None, part=None, ok=False, cons=False, supp=set(), drafter="broker", look=True))
            for p in KEYS:
                v = getp(tgt, p)
                if getp((e, k, o), p) == v:
                    continue
                S0_ = SA(W_now, e, k, o, form)
                nx = [e.copy(), k.copy(), o.copy()]; {"e": nx[0], "k": nx[1], "o": nx[2]}[p[0]][p[1]] = v
                dS = SA(W_now, *nx, form) - S0_
                harmed = {j for j in range(9) if dS[j] < -1e-12}; gain = {j for j in range(9) if dS[j] > 1e-12}
                spent += move_cost(p, v, st, cost, rcost)
                events.append(dict(t=t, i=None, part=p, nv=int(v), ok=True, dS=dS, pay=np.zeros(9), cons=True,
                                   supp=harmed | gain, drafter="broker", broker=True))
                author.pop(p, None)
                e, k, o = nx

    if regime in ("P", "BR", "BRdyn"):
        central(-1, W)
    for t in range(years):
        if own and own.get("fix") == "D4":
            Lam = d4_list()
        if venue and venue.get("G") == 1:
            fee = venue.get("tauG", 0.5) / 9
            m -= fee; extra["tcost"] += fee * 9
        if regime in ("R", "Rc"):
            for i in ORD[t]:
                if REQ[t, i] >= p_req:
                    continue
                cands = B.moves(e, k, o)
                mv = cands[int(r_null.integers(len(cands)))]
                S0_ = SA(W, e, k, o, form)
                e1, k1, o1 = apply(mv, i, e, k, o)
                dS = SA(W, e1, k1, o1, form) - S0_
                pay = np.zeros(9)
                if regime == "Rc":
                    for j in range(9):
                        if j != i:
                            pay[j] = max(0.0, -dS[j] / MU)
                    m[i] -= pay.sum(); m += pay
                p = part_of(mv); nv = int(getp((e1, k1, o1), p))
                cst = move_cost(p, nv, st, cost, rcost); spent += cst; m[i] -= cst
                events.append(dict(t=t, i=int(i), part=p, nv=nv, ok=True, dS=dS, pay=pay, cons=False, supp=set(),
                                   drafter="random", cost=cst))
                author.pop(p, None)
                e, k, o = e1, k1, o1
        elif regime not in ("F", "P", "BR", "BRdyn"):
            plus = regime.endswith("+")
            base = regime.rstrip("+")
            if base == "AI":
                base = "T1"
            gv = np.array([g_of(mean_nu(j), horizon) for j in range(9)])
            for i in ORD[t]:
                if REQ[t, i] >= p_req:
                    continue
                S0_ = SA(W, e, k, o, form); C0 = SA(Wbar, e, k, o, form)
                cands = B.moves(e, k, o) if plus else B.legal(i, e, k, o)
                tr = base in ("T0", "T1", "T1h")
                reach = None
                if venue and venue.get("G") == 0 and tr:
                    u = r_ven.random(9)
                    reach = np.array([j == i or j in NEAR[i] or u[j] < venue.get("q_far", 0.4) for j in range(9)])
                evals = []
                for mv in cands:
                    e1, k1, o1 = apply(mv, i, e, k, o)
                    dS = SA(W, e1, k1, o1, form) - S0_
                    dC = SA(Wbar, e1, k1, o1, form) - C0
                    evals.append((mv, dS, dC, (e1, k1, o1)))
                if actors:
                    ctx["W"] = W
                    for a in actors:
                        evals = a.evals(t, int(i), evals, ctx)
                drafter = "self"
                if regime == "AI+":
                    good = []
                    for x in evals:
                        p = part_of5(x[0]); nv = int(getp(x[3], p))
                        val, ok, pay = decide("T1", i, x[1], x[2], None, pi, theta, False)
                        val -= move_cost(p, nv, st, cost, rcost)
                        if ok and val > 1e-12:
                            good.append((x[1].sum() - MU * move_cost(p, nv, st, cost, rcost), x))
                    evals = [max(good, key=lambda z: z[0])[1]] if good else []
                    drafter = "ai"
                elif protocol == "topk" and tr:
                    evals = [x for x in evals if x[1][i] * gv[i] > 1e-12]
                    evals.sort(key=lambda x: -x[1][i])
                    evals = evals[:ask]
                elif protocol == "near" and tr:
                    evals = [x for x in evals if {j for j in range(9) if j != i and abs(x[1][j]) > 1e-12} <= NEAR[i]]
                best = None
                for mv, dS, dC, nxt in evals:
                    dSe = dS * gv if horizon is not None else dS
                    harmed_set = [j for j in range(9) if j != i and dS[j] < -1e-12]
                    if reach is not None and any(not reach[j] for j in harmed_set):
                        extra["unreached"] += 1
                        continue
                    val, ok, pay = decide(base, i, dSe, dC, None, pi, theta, split)
                    p = part_of5(mv); nv = int(getp(nxt, p))
                    cst = move_cost(p, nv, st, cost, rcost)
                    if mv[0] == "swap":
                        cst += move_cost(("e", mv[2]), 0, st, cost, rcost)
                    ant = 0.0
                    if horizon is not None and nv != getp(st, p):
                        if restore == "R0" and rcost is not None:
                            ant += float(rcost.get(p[0], 0.0)) * (1 - (1 - mean_nu(i)) ** horizon)
                        if restore == "Rdk" and cost is not None:
                            ant -= float(cost.get(p[0], 0.0)) * refund_frac(mean_nu(i), amort)
                    tc = venue.get("tau", 0.5) * len(harmed_set) if (reach is not None) else 0.0
                    opay = 0.0
                    if own is not None:
                        typ = TYPE[mv[0]]
                        if own["kind"] == "mgmt":
                            okd = typ in Lam
                            if not okd and own.get("fix") == "D5":
                                okd = owner_value(e, k, o, *nxt) + (val - cst - ant - tc) >= -1e-12
                            if not okd:
                                ok = False; val = -1.0
                        else:
                            Vo = owner_value(e, k, o, *nxt)
                            net = val - cst - ant - tc
                            if venue and venue.get("G") == 0.5:
                                okd = Vo + venue.get("alpha", 0.5) * net >= -1e-12
                            elif own.get("fix") == "D5":
                                okd = Vo >= -1e-12 or Vo + net >= -1e-12
                            else:
                                okd = Vo >= -1e-12
                            if not okd and own.get("role") == "trade":
                                opay = -Vo; okd = True
                            if not okd:
                                ok = False; val = -1.0
                    val = val - cst - ant - tc - opay
                    if tr:
                        for j in range(9):
                            if j != i and dS[j] < -1e-12:
                                if mv[0] == "swap":
                                    cs_ = rev_of(mv, j)
                                    for c_ in cs_:
                                        if len(cs_) > 1:
                                            ctx["mixed"].append(len(quotes))
                                        quotes.append((t, int(i), j, c_))
                                else:
                                    quotes.append((t, int(i), j, revealed(mv, j)))
                    if ok and lim is not None:
                        out_i = max(0.0, pay.sum()) + cst + tc + opay
                        if acc[i] - out_i < -lim[i] - 1e-9:
                            ok = False
                        for j in range(9):
                            if pay[j] < 0 and acc[j] + pay[j] < -lim[j] - 1e-9:
                                ok = False
                        if not ok:
                            val = -1.0
                    if best is None or val > best[0]:
                        best = (val, ok, pay, mv, dS, dC, nxt, cst, tc, opay)
                    if protocol == "topk" and tr and ok and val > 1e-12:
                        break
                if best is None:
                    continue
                val, ok, pay, mv, dS, dC, nxt, cst, tc, opay = best
                if val <= 1e-12:
                    if own is not None and ok is False:
                        extra["vetoed"] += 1
                    continue
                nyes = None
                if base in ("V½", "V¾"):
                    ok, nyes = vote(base, i, dS, VOT[t, i], pi)
                harmed = {j for j in range(9) if j != i and dS[j] < -1e-12}
                cons = tr
                if base in ("U", "V½", "V¾"):
                    cons = len(harmed) == 0
                p = part_of5(mv)
                lab = B.label(mv, i, e) if B is not BASE or mv[0] == "swap" else move_label(mv, i, e)
                mty = B.mtype(mv, e) if B is not BASE or mv[0] == "swap" else move_type(mv, e)
                ev = dict(t=t, i=int(i), move=lab, type=mty, ok=bool(ok), val=float(val),
                          yes=nyes, part=p, nv=int(getp(nxt, p)), dS=dS, pay=pay.copy(), cons=cons,
                          supp=({int(i)} | harmed) if cons else set(), drafter=drafter, cost=cst if ok else 0.0)
                ev2 = None
                if mv[0] == "swap":
                    # 組の手は、外す部位の記録を二つ目の手として並べる（推定器は部位ごとに同意を数える）
                    ev2 = dict(t=t, i=int(i), move=lab, type="外す（組の手）", ok=bool(ok), val=0.0, yes=nyes,
                               part=("e", mv[2]), nv=0, dS=np.zeros(9), pay=np.zeros(9), cons=cons,
                               supp=set(ev["supp"]), drafter=drafter, cost=0.0, pair=True)
                if ok:
                    for j in range(9):
                        if j != i:
                            b[j] -= dC[j] / MU
                            b[i] += dC[j] / MU
                    for j in range(9):
                        if pay[j] != 0:
                            b[i] += pay[j]; b[j] -= pay[j]
                            acc[i] -= pay[j]; acc[j] += pay[j]
                    acc[i] -= cst; spent += cst
                    if tc or opay:
                        acc[i] -= tc + opay; extra["tcost"] += tc; extra["owner_in"] += opay
                    for pp, evx in ((p, ev), (("e", mv[2]) if ev2 else None, ev2)):
                        if pp is None:
                            continue
                        if evx["nv"] == getp(st, pp):
                            author.pop(pp, None)
                        else:
                            author[pp] = (int(i), int(tenant[i]), t)
                    e, k, o = nxt
                    if actors:
                        for a in actors:
                            a.trade(t, int(i), mv, dS, pay, ctx)
                events.append(ev)
                if ev2 is not None:
                    events.append(ev2)
        changed = False
        if turning:
            for i in range(9):
                if turn_at is not None:
                    out = i in turn_at.get(t, ())
                elif kind is not None:
                    out = TURN[t, i] < hazard(i)
                else:
                    out = TURN[t, i] < nu
                if out:
                    mine = [p for p, (u_, a_, _) in author.items() if u_ == i and a_ == tenant[i]]
                    for p in mine:
                        if restore == "R0":
                            s_ = getp(st, p)
                            if getp((e, k, o), p) != s_:
                                nx = [e.copy(), k.copy(), o.copy()]; {"e": nx[0], "k": nx[1], "o": nx[2]}[p[0]][p[1]] = s_
                                dS = SA(W, *nx, form) - SA(W, e, k, o, form)
                                rc = 0.0 if rcost is None else float(rcost.get(p[0], 0.0))
                                acc[i] -= rc; spent += rc
                                events.append(dict(t=t, i=int(i), part=p, nv=int(s_), ok=True, dS=dS, pay=np.zeros(9),
                                                   cons=False, supp=set(), drafter="restore", cost=rc))
                                e, k, o = nx; extra["restored"] += 1
                            author.pop(p, None)
                        elif restore == "Rdk" and cost is not None:
                            age = t - author[p][2]
                            rf = float(cost.get(p[0], 0.0)) * max(0.0, 1 - age / amort)
                            acc[i] += rf; extra["refund"] += rf; extra["owner_out"] += rf
                    n_exit += 1; exit_bal += abs(b[i]); moved[i] = True
                    turns.append((t, i)); changed = True
                    if actors:
                        seqlog.append(("turn", t, i, nextseq()))
                    W[i] = newW(r_turn, i)
                    tenant[i] = nxt_tenant; nxt_tenant += 1; tenure[i] = 0
                    if kind == "mix":
                        sig[i] = 0 if r_type.random() < turnover.get("piS", 0.72) else 1
                    if settle:
                        m[i] = 0.0; c[i] = 0.0; b[i] = 0.0
                    if reconsent:
                        gn = g_of(mean_nu(i), horizon) if horizon is not None else 1.0
                        e, k, o, nr, nv_, sp = _reconsent(i, t, W, e, k, o, st, form, reconsent, acc, lim, recon, rcost, gn,
                                                          author, events, B, actors, ctx)
                        n_recon += nr; n_revert += nv_; spent += sp
                else:
                    tenure[i] += 1
        if changed and regime in ("P", "BRdyn"):
            central(t, W)
        Wt.append(SA(W, e, k, o, form).sum())
    S_fin = SA(W, e, k, o, form)
    out = dict(e=e, k=k, o=o, S=S_fin, U=S_fin + MU * (m + c), b=b, m=m, c=c, Wt=np.array(Wt), events=events,
               W_final=W, n_exit=n_exit, exit_bal=exit_bal, n_recon=n_recon, n_revert=n_revert, moved=moved,
               quotes=quotes, turns=turns, recon=recon, spent=spent, start=X0, ref=st, extra=extra)
    if actors:
        out.update(board=B.name, ledger=ledger, seqlog=seqlog, blocked=ctx["blocked"], mixed_quotes=ctx["mixed"],
                   actor_facts={a.name: list(a.facts) for a in actors},
                   actor_full={a.name: list(a.full) for a in actors},
                   actor_kind={a.name: a.kind for a in actors},
                   actor_stats={a.name: {k_: v for k_, v in vars(a).items() if k_.startswith("n_")} for a in actors})
    return out


def _reconsent(n, t, W, e, k, o, st, form, scope, acc, lim, recon, rcost, gn=1.0, author=None, events=None,
               B=BASE, actors=(), ctx=None):
    """〔再〕入居した世帯 n の再同意（s3._reconsent と同じ規則。盤の部位の一覧と、運用役の差し込み口を足した）"""
    parts = B.recon_parts(scope)
    SA = B.S
    nr = nv = 0; sp = 0.0
    for p in parts:
        cur = getp((e, k, o), p); s_ = getp(st, p)
        if cur == s_:
            continue
        nx = [e.copy(), k.copy(), o.copy()]; {"e": nx[0], "k": nx[1], "o": nx[2]}[p[0]][p[1]] = s_
        dS = SA(W, *nx, form) - SA(W, e, k, o, form)
        rc = 0.0 if rcost is None else float(rcost.get(p[0], 0.0))
        G = gn * dS[n] / MU - rc
        if G <= 1e-12:
            recon.append((t, n, p, "accept"))
            for a in actors:
                a.recon(t, n, p, "accept", G, ctx)
            continue
        if actors and not all(a.revert_ok(p, (e, k, o), ctx) for a in actors):
            # 戻すと制約を破るので戻さない。新しい世帯は同意していないので、recon には入れない（推定器は支える世帯に数えない）
            ctx["blocked"].append((t, n, p))
            continue
        nr += 1
        L = np.maximum(0.0, -gn * dS / MU) if gn != 1.0 else np.maximum(0.0, -dS / MU); L[n] = 0.0
        keep = False
        if L.sum() > G + 1e-12:
            s = G * L / L.sum()
            if lim is None or all(acc[j] - s[j] >= -lim[j] - 1e-9 for j in range(9) if s[j] > 0):
                keep = True
                for j in range(9):
                    acc[j] -= s[j]
                acc[n] += G
        recon.append((t, n, p, "paid" if keep else "revert"))
        for a in actors:
            a.recon(t, n, p, "paid" if keep else "revert", G, ctx)
        if not keep:
            e, k, o = nx; nv += 1; acc[n] -= rc; sp += rc
            if author is not None:
                author.pop(p, None)
    return e, k, o, nr, nv, sp
