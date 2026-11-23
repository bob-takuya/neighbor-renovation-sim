"""S0 最小模型（3 階 × 3 戸）。

住戸 i = (3 − 階)·3 + 列（列 0=W, 1=M, 2=E）。並びは 3W 3M 3E 2W 2M 2E 1W 1M 1E。
戸境 b = (3 − 階)·2 + j（j=0：W|M、j=1：M|E）。
状態：e[9]（張り出し 0/1）、k[6]（戸境収納 −1/0/+1。+1＝左の戸が全部使う）、o[6]（開口 0/1）。
"""
from __future__ import annotations
import numpy as np

UNITS = ["3W", "3M", "3E", "2W", "2M", "2E", "1W", "1M", "1E"]
K = ["広さ", "日照", "静けさ", "近さ"]
W_BAR = np.array([0.35, 0.30, 0.20, 0.15])
A0, EXT_A, CLO_A = 50.0, 6.0, 3.0
SHADE = (1.5, 0.5)
SUN0 = np.array([6.0] * 3 + [5.0] * 3 + [3.5] * 3)
A_LO, A_HI, SUN_HI = 38.0, 62.0, 6.0
MU = W_BAR[0] / (A_HI - A_LO)          # 1 ㎡（持ち点 1）が満足でいくつ

FLOOR = np.array([3, 3, 3, 2, 2, 2, 1, 1, 1])
COL = np.array([0, 1, 2] * 3)


def above(i, d):
    j = i - 3 * d
    return j if j >= 0 else None


def lb(i):  # 左の戸境（この戸は右側）
    return (3 - FLOOR[i]) * 2 + COL[i] - 1 if COL[i] > 0 else None


def rb(i):  # 右の戸境（この戸は左側）
    return (3 - FLOOR[i]) * 2 + COL[i] if COL[i] < 2 else None


def b_units(b):
    f = 3 - b // 2; j = b % 2
    return (3 - f) * 3 + j, (3 - f) * 3 + j + 1


# ------------------------------------------------------------ 量と満足
def quantities(e, k, o):
    """各戸の (広さ, 日照, 開口数)。e,k,o は長さ 9,6,6 の配列"""
    area = np.full(9, A0) + EXT_A * np.asarray(e, float)
    sun = SUN0.copy()
    op = np.zeros(9)
    for i in range(9):
        r, l = rb(i), lb(i)
        if r is not None:
            area[i] += CLO_A * k[r]; op[i] += o[r]
        if l is not None:
            area[i] -= CLO_A * k[l]; op[i] += o[l]
        for d, s in zip((1, 2), SHADE):
            a = above(i, d)
            if a is not None and e[a]:
                sun[i] -= s
    return area, sun, op


def phi(area, sun, op, form="linear"):
    za = np.clip((area - A_LO) / (A_HI - A_LO), 0, 1)
    zs = np.clip(sun / SUN_HI, 0, 1)
    if form == "concave":
        za, zs = np.sqrt(za), np.sqrt(zs)
    if form == "floor":
        zs = np.where(sun < 2.0, zs - 0.6 * (2.0 - sun) / 2.0, zs)
    return np.stack([za, zs, 1 - op / 2, op / 2], axis=-1)   # (..., 4)


def S_all(W, e, k, o, form="linear"):
    """各戸の満足（長さ 9）。W は (9, 4)"""
    a, s, op = quantities(e, k, o)
    return (phi(a, s, op, form) * W).sum(-1)


# ------------------------------------------------------------ 状態の符号化（総当たりの添字と一致させる）
def encode(e, k, o):
    ie = int(sum(int(e[i]) << (8 - i) for i in range(9)))
    ik = 0
    for b in range(6):
        ik = ik * 3 + (int(k[b]) + 1)
    io = int(sum(int(o[b]) << (5 - b) for b in range(6)))
    return io, ie, ik


# ------------------------------------------------------------ 手
def legal_moves(i, e, k, o):
    """世帯 i が打てる手：('ext', i) 張り出しの付け外し、('take', b) 戸境収納を自分の側へ、('open', b) 開口の付け外し"""
    mv = [("ext", i)]
    l, r = lb(i), rb(i)
    if r is not None:
        if k[r] < 1:
            mv.append(("take", r))
        mv.append(("open", r))
    if l is not None:
        if k[l] > -1:
            mv.append(("take", l))
        mv.append(("open", l))
    return mv


def all_moves(e, k, o):
    """盤のすべての手（他戸の手を「買う」場合）：各戸の張り出しの付け外し、各戸境の収納を左右に一段、各開口の付け外し"""
    mv = [("ext", j) for j in range(9)]
    for b in range(6):
        if k[b] < 1:
            mv.append(("shift", b, 1))
        if k[b] > -1:
            mv.append(("shift", b, -1))
        mv.append(("open", b))
    return mv


def apply(move, i, e, k, o):
    e, k, o = e.copy(), k.copy(), o.copy()
    kind, x = move[0], move[1]
    if kind == "ext":
        e[x] = 1 - e[x]
    elif kind == "take":
        left, right = b_units(x)
        k[x] += 1 if i == left else -1
    elif kind == "open":
        o[x] = 1 - o[x]
    elif kind == "shift":
        k[x] += move[2]
    return e, k, o


def move_label(move, i, e):
    kind, x = move[0], move[1]
    if kind == "shift":
        l, r = b_units(x)
        return f"収納を {UNITS[r]}→{UNITS[l]}" if move[2] > 0 else f"収納を {UNITS[l]}→{UNITS[r]}"
    if kind == "ext":
        return f"張り出す {UNITS[x]}" if e[x] == 0 else f"張り出しを外す {UNITS[x]}"
    if kind == "take":
        l, r = b_units(x)
        other = r if i == l else l
        return f"収納を回してもらう {UNITS[other]}→{UNITS[i]}"
    l, r = b_units(x)
    return f"開口 {UNITS[l]}|{UNITS[r]}"


def move_type(move, e_before):
    kind, x = move[0], move[1]
    if kind == "shift":
        return "収納"
    if kind == "ext":
        return f"張り出す{FLOOR[x]}F" if e_before[x] == 0 else f"外す{FLOOR[x]}F"
    if kind == "take":
        return "収納"
    return "開口"


# ------------------------------------------------------------ 決め方
REGIMES = ["F", "U", "V½", "V¾", "T0", "T1", "T1h"]
REGIMES_BUY = ["T0+", "T1+"]   # 他戸の手も、影響を受ける全員に償って「買える」


def decide(regime, i, dS, dC, rng_vote, pi=0.5, theta=0.0, split=False):
    """一つの手について、提案者 i の見込みの値（持ち点の単位）と、実行するか、償いを返す。
    dS：各戸の満足の変化（自分の性向）、dC：各戸の満足の変化（平均的な性向＝共通の物差し）。
    theta：損をする側が、自分の損を (1+theta) 倍に言う（駆け引き）。T0 では断る基準、T1 では言い値に効く。
    返り値 (value, ok, pay)。pay[j] は i から j への支払い（負なら j から i へ）。"""
    others = [j for j in range(9) if j != i]
    own = dS[i] / MU
    pay = np.zeros(9)
    if regime == "U":
        return own, own > 1e-12, pay
    if regime in ("V½", "V¾"):
        return own, None, pay        # 可否は投票で別に決める
    if regime == "T0":
        for j in others:
            pay[j] = -dC[j] / MU     # 損をした相手には払い、得をした相手からは受け取る
        val = own - pay.sum()
        claim = [dS[j] / MU * (1 + theta) if dS[j] < 0 else dS[j] / MU for j in range(9)]
        ok = val > 1e-12 and all(claim[j] + pay[j] >= -1e-12 for j in others)
        return val, ok, pay
    if regime in ("T1", "T1h"):
        for j in others:
            pay[j] = max(0.0, -dS[j] / MU) * (1 + theta)   # 言い値（自分の性向での損。theta だけ盛る）
        if split and regime == "T1":
            # 〔割〕償いの総額を、得をする戸（提案者を含む）が得の大きさに比例して割り勘にする
            G = np.maximum(0.0, dS / MU)
            C = pay.sum()
            if C > 0 and G.sum() > C + 1e-12:
                s = C * G / G.sum()
                pay = pay - s; pay[i] = 0.0          # pay[j] ＝ 償い L_j − 割り当て s_j（負なら j が払う）
                return G[i] - s[i], G[i] - s[i] > 1e-12, pay
            if C > 0:
                return -1.0, False, pay
        val = own - pay.sum()
        if regime == "T1h" and val > 0:
            harmed = [j for j in others if pay[j] > 0]
            if harmed:
                for j in harmed:
                    pay[j] += 0.5 * val / len(harmed)
                val = own - pay.sum()
        return val, val > 1e-12, pay
    raise ValueError(regime)


def vote(regime, i, dS, u, pi=0.5):
    """u：その手の投票の乱数（長さ 9）"""
    yes = 0
    for j in range(9):
        if j == i or dS[j] > 1e-12:
            yes += 1
        elif abs(dS[j]) <= 1e-12 and u[j] < pi:
            yes += 1
    need = 5 if regime == "V½" else 7
    return yes >= need, yes


# ------------------------------------------------------------ 固定方式 F の形
def f_design(form="linear"):
    """平均的な性向 w̄ に最適な形（張り出しの 2^9 通りを数え上げ。収納と開口は平均では得がないので初期のまま）。
    同点なら張り出しの少ない方。"""
    W = np.tile(W_BAR, (9, 1))
    best, be = -1e9, None
    k0 = np.zeros(6, int); o0 = np.zeros(6, int)
    for bits in range(512):
        e = np.array([(bits >> (8 - i)) & 1 for i in range(9)])
        v = S_all(W, e, k0, o0, form).sum() - 1e-9 * e.sum()
        if v > best + 1e-12:
            best, be = v, e
    return be, k0.copy(), o0.copy()


# ------------------------------------------------------------ 一局
def run(W, regime, seed, years=30, p_req=0.3, form="linear", pi=0.5, start=None, log=False,
        theta=0.0, budget=None, conv_scale=1.0, nu=0.0, newW=None,
        split=False, credit=None, settle=False, reconsent=None):
    """W：(9,4) 世帯の性向。共通の乱数：同じ seed なら決め方が違っても要求・順序・投票の乱数は同じ。
    theta：損を盛って言う割合。budget：各戸の持ち点の下限の大きさ（長さ 9、None なら無制限。m_i ≥ −budget_i）。
    conv_scale：共通の換算で、日照・静けさ・近さを広さに対して何倍に見るか（設計者の物差しの誤り）。
    nu：毎年の入れ替わりの確率。newW(rng, i) が新しい世帯の性向を返す。
    改良の部品（言い値の取引 T1 系に効く）
      split：〔割〕償いを得をする戸で割り勘にする
      credit：〔貸〕お金の代わりに建物の中の貸し借り c で払う。全戸に同じ借りの上限 D（＝credit、㎡ 換算）。外のお金の下限 budget は効かない
      settle：〔清〕転出時に、その世帯のお金・貸し借り・帳簿を精算して持ち出す（新しい世帯は 0 から）。False なら住戸に残る（E1 の既定）
      reconsent：〔再〕入居時の再同意。"rev"＝戻せる造作（収納・開口）、"all"＝張り出しも含む。
                 新しい世帯が始まりの状態へ戻したい部位は、残したい戸がその世帯の言い値を払えば残り、払わなければ戻る"""
    W = np.array(W, float).copy()
    ss = np.random.SeedSequence([seed, 7])
    r_req, r_ord, r_vote = [np.random.default_rng(s) for s in ss.spawn(3)]
    REQ = r_req.random((years, 9))
    ORD = np.array([r_ord.permutation(9) for _ in range(years)])
    VOT = r_vote.random((years, 9, 9))
    r_turn = np.random.default_rng(np.random.SeedSequence([seed, 7, 99]))
    TURN = r_turn.random((years, 9))
    n_exit = 0; exit_bal = 0.0
    e, k, o = start if start is not None else f_design(form)
    e, k, o = e.copy(), k.copy(), o.copy()
    e_s, k_s, o_s = e.copy(), k.copy(), o.copy()       # 始まりの状態（再同意で戻す先）
    c = np.zeros(9)                                    # 〔貸〕建物の中の貸し借り
    n_recon = 0; n_revert = 0
    moved = np.zeros(9, bool)                          # 30 年のあいだに一度でも入れ替わった住戸
    Wbar = np.tile(W_BAR * np.array([1.0, conv_scale, conv_scale, conv_scale]), (9, 1))
    b = np.zeros(9); m = np.zeros(9)
    events = []
    Wt = [S_all(W, e, k, o, form).sum()]
    for t in range(years):
        if regime != "F":
            for i in ORD[t]:
                if REQ[t, i] >= p_req:
                    continue
                S0_ = S_all(W, e, k, o, form); C0 = S_all(Wbar, e, k, o, form)
                best = None
                base = regime.rstrip("+")
                cands = all_moves(e, k, o) if regime.endswith("+") else legal_moves(i, e, k, o)
                for mv in cands:
                    e1, k1, o1 = apply(mv, i, e, k, o)
                    dS = S_all(W, e1, k1, o1, form) - S0_
                    dC = S_all(Wbar, e1, k1, o1, form) - C0
                    val, ok, pay = decide(base, i, dS, dC, None, pi, theta, split)
                    acc, lim = (c, np.full(9, credit)) if credit is not None else (m, budget)
                    if ok and lim is not None:
                        # 払う側（提案者と、割り勘・T0 で払う相手）が下限を割らないか
                        out_i = max(0.0, pay.sum())
                        if acc[i] - out_i < -lim[i] - 1e-9:
                            ok = False
                        for j in range(9):
                            if pay[j] < 0 and acc[j] + pay[j] < -lim[j] - 1e-9:
                                ok = False
                        if not ok:
                            val = -1.0
                    if best is None or val > best[0]:
                        best = (val, ok, pay, mv, dS, dC, (e1, k1, o1))
                val, ok, pay, mv, dS, dC, nxt = best
                if val <= 1e-12:
                    continue
                nyes = None
                if regime in ("V½", "V¾"):
                    ok, nyes = vote(regime, i, dS, VOT[t, i], pi)
                ev = dict(t=t, i=UNITS[i], move=move_label(mv, i, e), type=move_type(mv, e), ok=bool(ok), val=float(val), yes=nyes)
                if ok:
                    # 帳簿：他戸への影響を共通の物差しで記帳
                    for j in range(9):
                        if j != i:
                            b[j] -= dC[j] / MU
                            b[i] += dC[j] / MU
                    acc = c if credit is not None else m
                    for j in range(9):
                        if pay[j] != 0:
                            b[i] += pay[j]; b[j] -= pay[j]
                            acc[i] -= pay[j]; acc[j] += pay[j]
                    e, k, o = nxt
                events.append(ev)
        if nu > 0 and newW is not None:
            for i in range(9):
                if TURN[t, i] < nu:
                    n_exit += 1; exit_bal += abs(b[i]); moved[i] = True
                    W[i] = newW(r_turn, i)
                    if settle:
                        m[i] = 0.0; c[i] = 0.0; b[i] = 0.0
                    if reconsent:
                        e, k, o, nr, nv = _reconsent(i, W, e, k, o, e_s, k_s, o_s, form, reconsent,
                                                     c if credit is not None else m,
                                                     np.full(9, credit) if credit is not None else budget)
                        n_recon += nr; n_revert += nv
        Wt.append(S_all(W, e, k, o, form).sum())
    S_fin = S_all(W, e, k, o, form)
    return dict(e=e, k=k, o=o, S=S_fin, U=S_fin + MU * (m + c), b=b, m=m, c=c, Wt=np.array(Wt), events=events,
                W_final=W, n_exit=n_exit, exit_bal=exit_bal, n_recon=n_recon, n_revert=n_revert, moved=moved)


def _reconsent(n, W, e, k, o, e_s, k_s, o_s, form, scope, acc, lim):
    """〔再〕入居した世帯 n の再同意。始まりの状態と違う部位を一つずつ見て、n が戻したいなら、
    残したい戸（戻すと損をする戸）が n の言い値を損の大きさに比例して割り勘で払えば残し、払えなければ戻す。
    返り値 (e, k, o, 見直した部位の数, 戻した部位の数)"""
    parts = [("o", x) for x in range(6)] + [("k", x) for x in range(6)]
    if scope == "all":
        parts += [("e", x) for x in range(9)]
    nr = nv = 0
    for kind, x in parts:
        cur = {"e": e, "k": k, "o": o}[kind]; st = {"e": e_s, "k": k_s, "o": o_s}[kind]
        if cur[x] == st[x]:
            continue
        e1, k1, o1 = e.copy(), k.copy(), o.copy()
        {"e": e1, "k": k1, "o": o1}[kind][x] = st[x]
        dS = S_all(W, e1, k1, o1, form) - S_all(W, e, k, o, form)
        G = dS[n] / MU
        if G <= 1e-12:
            continue
        nr += 1
        L = np.maximum(0.0, -dS / MU); L[n] = 0.0
        keep = False
        if L.sum() > G + 1e-12:
            s = G * L / L.sum()
            if lim is None or all(acc[j] - s[j] >= -lim[j] - 1e-9 for j in range(9) if s[j] > 0):
                keep = True
                for j in range(9):
                    acc[j] -= s[j]
                acc[n] += G
        if not keep:
            e, k, o = e1, k1, o1; nv += 1
    return e, k, o, nr, nv


# ------------------------------------------------------------ 線形の満足での厳密な最適（分離できる）
def wstar_linear(W):
    """線形の φ では、満足の合計は各部位（張り出し 9・収納 6・開口 6）の寄与の和に分かれる
    （広さ 44〜62 ㎡、日照 1.5〜6 h は φ の折れ目に掛からない）。各部位を独立に最良にすれば最適。
    返り値 (W*, e, k, o)"""
    W = np.asarray(W, float)
    e = np.zeros(9, int); k = np.zeros(6, int); o = np.zeros(6, int)
    da = 1.0 / (A_HI - A_LO); ds = 1.0 / SUN_HI
    for i in range(9):
        c = W[i, 0] * EXT_A * da
        for d, sh in zip((1, 2), SHADE):
            j = i + 3 * d
            if j < 9:
                c -= W[j, 1] * sh * ds
        e[i] = 1 if c > 1e-12 else 0
    for b_ in range(6):
        l, r = b_units(b_)
        c = (W[l, 0] - W[r, 0]) * CLO_A * da
        k[b_] = 1 if c > 1e-12 else (-1 if c < -1e-12 else 0)
        c2 = 0.5 * ((W[l, 3] - W[l, 2]) + (W[r, 3] - W[r, 2]))
        o[b_] = 1 if c2 > 1e-12 else 0
    return float(S_all(W, e, k, o).sum()), e, k, o
