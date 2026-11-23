"""観察 3（近隣改修の収支）のための模型 s8。s3（観察 2 で凍結）は変えず、写して次の三つを足した。
どれも既定では切れていて（帳簿は記録を足すだけ）、既定の引数では s3.run と同じ乱数を同じ順に使い、同じ記録を返す（tests/test_s8.py）。

A. 世帯の帳簿（ledger、常に記録。模型の動きは変えない）
   世帯は (住戸, その住戸の何人目の住み手か＝tenant の番号) で呼ぶ。お金と貸し借りの授受を一件ずつ、
   (年, 種類, 払う側, 受け取る側, 額, 部位, 媒体) で残す。払う側・受け取る側は世帯の番号か "owner"（所有者）か "out"（工事業者など建物の外）。
   種類：comp（補償。手の提案者と損をする戸）、cost（施工費）、restore（退去時に戻す費用）、refund（Rdk の補償。所有者から）、
   opay（所有者への支払い）、tcost（交渉の手間）、fee（寄り合いの手間）、recon（再同意で残したい戸が新しい世帯に払う）、
   recon_cost（再同意で戻すときの費用）、rent（家賃の上乗せ。所有者の収支 landlord があるときだけ）。
   世帯の表 hh：入居と退去の年、性向、入居時と退去時（戻す前）の満足、F に住んだ場合の満足、毎年の満足。
B. 退去の効果 exitfx＝dict(phi=φ, psi=ψ)（既定 None）
   退去の確率を、世帯の住み心地が F より良いほど下げる：h_i ＝ min(1, h⁰_i · exp(−φ·x_i)) · (1 + ψ·拒まれた_i)。
   x_i ＝ (S_i − S_i^F)/μ（㎡ 換算）、h⁰_i は入れ替わりの型が与える基準の確率、拒まれた_i はその年に所有者に手を拒まれて何も打てなかったこと。
   乱数 TURN はそのまま使う（共通乱数）。φ＝ψ＝0 なら s3 と同じ。
C. 自分の収支で判断する所有者 owner＝dict(kind="econ", role="veto"|"trade", ...)（landlord の単価を使う）
   所有者は、承諾したときと拒んだときの、見通しの年数 Ho（既定 10 年、割引なし）の収支を比べる。住戸 j ごとに
     Π_j(Xa, Xr) ＝ −P_j(Xa)·C_j(Xr) ＋ (Ho − Y_j(Xa))·[ λ·d_j(Xr)/μ/Hv − ν'_j(Xr)·C_j(Xr) ]
   Y_j・P_j は今の世帯がこの先 Ho 年に住む年数の期待値と、Ho 年のうちに退去する確率（退去の確率 h_j(Xa) から）、
   d_j は平均的な新しい住み手が今の形を F と比べて好む度合い（s3.market と同じ）、λ は家賃に移る割合、Hv＝10 は満足の差を年あたりに直す年数、
   ν'_j ＝ ν·exp(−φ·d_j/μ) は平均的な次の世帯の退去の確率、C_j(X) ＝ (v0 ＋ v1·σ_j(X))·r0 ＋ c_turn は一回の退去の費用
   （空室の期間 v0＋v1·σ_j 年の家賃と、募集の費用）、σ_j は参照の母集団での好みの散らばり（㎡）。
   承諾の値 ＝ Σ_j [w_j Π_j(X1, X1) ＋ (1−w_j) Π_j(X1, X0)] − Rdk の補償の見込み、
   拒否の値 ＝ Σ_j Π_j(X0, X0) − ψ·h_i(X0)·C_i(X0)（拒まれた提案者の退去が増える分）。
   w_j は手が残る割合：R0 では提案者の住戸 0、ほかの住戸は提案者が住み続ける年の割合 Y_i/Ho。R∅・Rdk では 1。
   承諾の値 ≥ 拒否の値なら承諾。role="trade" なら、損のとき提案者がその差を所有者に払えば承諾する。
   landlord＝dict(r0, lam, v0, v1, cturn, Ho, Hv) を渡すと、所有者の実現した収支（家賃の上乗せ、空室の損、募集の費用、補償、受け取った額）も記録する。

以下は s3 の注（そのまま）。
段 1 の第二報のための模型 s3。s1（段 2 で凍結）は変えず、写して賃貸の制約を四つの部品として足した。
どの部品も既定では切れていて、既定の引数では s1.run と同じ乱数を同じ順に使い、同じ記録を返す（tests/test_s3.py）。

1. 入れ替わりの型 turnover（既定 None ＝ s1 の nu をそのまま使う）
     dict(kind="geo", nu=ν)                     毎年、各戸が確率 ν で出る（s1 と同じ）
     dict(kind="renew", nu=ν, rho=2)            2 年ごとの更新の年だけ退去の確率が rho 倍。年あたりの平均の退去率が ν になるように基準を合わせる
     dict(kind="mix", nuS=0.32, nuL=0.087, piS=0.72)
                                                短く住む型（年 0.32）と長く住む型（年 0.087）の混合。入居する世帯が短く住む型である確率 piS
   性向を引き継ぐ対照は、newW に前の住み手の性向を返す関数を渡して作る（模型は変えない）。
2. 原状回復の型 restore（既定 None ＝ 何も起きない。ゼロ精算と同じ）
     "R0"   退去する世帯が自分の起こした手（最後に変えたのがその世帯である部位）を始まりの状態に戻し、戻す費用を払う
     "Rdk"  改変は残る。退去する世帯は、自分の起こした手の施工費の未償却分（amort 年の直線）を所有者から受け取る
3. 見通し horizon（既定 None）
     H 年を見通す。各世帯は、自分の満足の変化を「H 年のうち住んでいると見込む年の割合」g＝(1−(1−ν)^H)/(Hν) で割り引いて評価する。
     R0 なら戻す費用を H 年のうちに出る確率で、Rdk なら受け取る見込みの未償却分を、提案者の費用の見込みに入れる。
4. 所有者 owner（既定 None ＝ 所有者なし）
     dict(kind="ind"|"sub"|"mgmt", role="veto"|"trade", beta=, gamma=, M=10, Lam=set(), fix=None|"D4"|"D5")
     ind   個人の所有者。手が建物の市場での評判を上げるかで承諾する：
           V_o ＝ beta·Σ_j ΔΔ_j/μ − gamma·Σ_j Δξ_j
           Δ_j は市場の平均的な新しい住み手（参照の母集団）が住戸 j の今の形を F と比べてどれだけ好むか、ξ_j はそのうち F を好む割合（癖）。
     sub   サブリース。ind と同じ式で、癖の損を M 棟分（gamma×M）に見る
     mgmt  管理会社に委託。一覧 Lam に載る種類の手（"ext"・"store"・"open"）だけを承諾する。価格には反応しない
     role="trade" なら、V_o＜0 のとき提案者が −V_o を所有者に払って承諾を買える（mgmt では効かない）
     fix="D4"  承諾の一覧を住み手が決める（毎年、今の住み手の過半が得をする手を一つでも持つ種類を一覧に載せる）。mgmt に使う
     fix="D5"  目的限定の承諾請求権。所有者は、自分の損が住み手の正味の得を上回るときだけ拒める（V_o＋val ≥ 0 なら承諾）。
               mgmt では、一覧の外の手を、ind と同じ式の V_o で裁く
5. 決める場 venue（既定 None ＝ s1 と同じ。全員に届き、手間はない）
     dict(G=0, q_far=0.4, tau=0.5)   場なし。近い戸（同じ階の隣・同じ列の上下一つ）以外の相手には確率 q_far でしか届かない。
                                     損をする相手が一人でも届かない手は打てない。相手一人あたり tau（㎡）の手間を提案者が払う
     dict(G=1, tauG=0.5)            年に一度の寄り合い。全員に届く。手間は年に tauG を 9 戸で割って払う（判断には効かない）
     dict(G=0.5, alpha=0.5)         区分所有法 44 条型。全員に届くが、決めるのは所有者で、所有者は住み手の正味の得に重み alpha をかけて足す
                                     （V_o＋alpha·val ≥ 0 なら承諾）
"""
from __future__ import annotations
import numpy as np
from trade.s0 import (S_all, MU, W_BAR, legal_moves, all_moves, apply, move_label, move_type, decide, vote, f_design,
                      quantities, phi)
from trade.s1 import KEYS, NEAR, part_of, revealed, getp, move_cost, best_net

TYPE = {"ext": "ext", "take": "store", "shift": "store", "open": "open"}
F0 = f_design()


def _ref_pop(n=300):
    """市場の参照の母集団：平均の性向のまわりのディリクレ（集中度 3）。所有者は個々の住み手の性向を知らず、この分布だけを知る"""
    rng = np.random.default_rng(424242)
    P = rng.dirichlet(W_BAR * 3.0, n)
    return np.clip(P, 1e-9, None) / np.clip(P, 1e-9, None).sum(1, keepdims=True)


POP = _ref_pop()
POPM = POP.mean(0)
PHI_F = phi(*quantities(*F0))


def market(e, k, o):
    """各住戸について (Δ_j, ξ_j)。Δ_j は平均的な新しい住み手が今の形を F と比べて好む度合い（満足）、ξ_j は F を好む割合"""
    D = phi(*quantities(e, k, o)) - PHI_F          # (9, 4)
    V = POP @ D.T                                   # (n, 9)
    return D @ POPM, (V < -1e-12).mean(0)


def market2(e, k, o):
    """〔s8〕各住戸について (d_j, σ_j)（どちらも ㎡ 換算）。d_j は平均的な新しい住み手が今の形を F と比べて好む度合い、
    σ_j は参照の母集団でのその好みの散らばり（標準偏差）。F の盤では両方 0"""
    D = phi(*quantities(e, k, o)) - PHI_F
    V = POP @ D.T / MU
    return V.mean(0), V.std(0)


def stay_years(h, H):
    """退去の確率 h（年）の世帯が、この先 H 年のうちに住む年数の期待値 Y と、H 年のうちに退去する確率 P"""
    if h <= 0:
        return float(H), 0.0
    q = 1.0 - h
    return float((1 - q ** H) / h), float(1 - q ** H)


def g_of(nu, H):
    if H is None:
        return 1.0
    if nu <= 0:
        return 1.0
    return (1 - (1 - nu) ** H) / (H * nu)


def refund_frac(nu, A):
    """幾何分布の居住で、施工費の未償却分を受け取る見込みの割合 E[(1−T/A)⁺]（T は施工から退去までの年）"""
    if nu <= 0:
        return 0.0
    return float(sum(nu * (1 - nu) ** (t - 1) * max(0.0, 1 - t / A) for t in range(1, int(A) + 1)))


def renew_base(nu, rho, horizon=400):
    """更新の山つきの居住で、平均の居住年数が 1/ν になる基準の退去率 h0 を二分法で求める"""
    def mean_T(h0):
        s, surv = 0.0, 1.0
        for a in range(1, horizon + 1):
            h = min(1.0, h0 * (rho if a % 2 == 0 else 1.0))
            s += a * surv * h; surv *= (1 - h)
        return s + horizon * surv
    lo, hi = 1e-6, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if mean_T(mid) > 1 / nu:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run(W, regime, seed, years=30, p_req=0.3, form="linear", pi=0.5, start=None,
        theta=0.0, budget=None, conv_scale=1.0, nu=0.0, newW=None,
        split=False, credit=None, settle=False, reconsent=None,
        protocol="full", ask=3, cost=None, rcost=None, turn_at=None, ref=None,
        turnover=None, restore=None, amort=10, horizon=None, owner=None, venue=None,
        exitfx=None, landlord=None):
    W = np.array(W, float).copy()
    ss = np.random.SeedSequence([seed, 7])
    r_req, r_ord, r_vote = [np.random.default_rng(s) for s in ss.spawn(3)]
    REQ = r_req.random((years, 9))
    ORD = np.array([r_ord.permutation(9) for _ in range(years)])
    VOT = r_vote.random((years, 9, 9))
    r_turn = np.random.default_rng(np.random.SeedSequence([seed, 7, 99]))
    TURN = r_turn.random((years, 9))
    r_null = np.random.default_rng(np.random.SeedSequence([seed, 7, 123]))
    r_type = np.random.default_rng(np.random.SeedSequence([seed, 7, 77]))     # 入れ替わりの型（s1 の乱数とは別）
    r_ven = np.random.default_rng(np.random.SeedSequence([seed, 7, 55]))      # 決める場で相手に届くか
    e, k, o = start if start is not None else f_design(form)
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
    Wt = [S_all(W, e, k, o, form).sum()]
    acc = c if credit is not None else m
    lim = np.full(9, credit) if credit is not None else budget

    # ---- 入れ替わりの型
    kind = turnover["kind"] if turnover else None
    if kind == "geo":
        nu = turnover["nu"]
    if kind == "mix":
        sig = np.where(r_type.random(9) < turnover.get("piS", 0.72), 0, 1)     # 0＝短く住む型, 1＝長く住む型
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

    # ---- 〔s8〕世帯の帳簿
    ledger = []
    Fd = F0 if form == "linear" else f_design(form)
    PHI_Fd = phi(*quantities(*Fd), form)
    SF_all = lambda W_: (PHI_Fd * W_).sum(-1)          # S_all(W_, *Fd, form) と同じ計算（F の量は変わらないので先に求める）
    hh = {}                                  # 世帯の番号 → 表

    def hh_new(i, t_in):
        a = int(tenant[i])
        hh[a] = dict(unit=int(i), hid=a, t_in=int(t_in), t_out=None, W=W[i].tolist(),
                     S_in=float(S_all(W, e, k, o, form)[i]), SF=float(SF_all(W)[i]), S_exit=None, Sy=[])

    def hid(i):
        return int(tenant[i])

    def led(t, kind_, frm, to, amt, part=None, med="m"):
        if abs(amt) > 0:
            ledger.append((int(t), kind_, frm, to, float(amt), None if part is None else (part[0], int(part[1])), med))
    medium = "c" if credit is not None else "m"
    for i in range(9):
        hh_new(i, 0)

    # ---- 〔s8〕退去の効果
    phi_x = float(exitfx.get("phi", 0.0)) if exitfx else 0.0
    psi_x = float(exitfx.get("psi", 0.0)) if exitfx else 0.0
    refused = np.zeros(9, bool)

    def xfit(i, X=None, W_=None):
        W_ = W if W_ is None else W_
        X = (e, k, o) if X is None else X
        return float((S_all(W_, *X, form)[i] - SF_all(W_)[i]) / MU)

    def hazard_fx(i):
        h = hazard(i)
        if exitfx:
            h = min(1.0, h * np.exp(-phi_x * xfit(i)) * (1.0 + psi_x * float(refused[i])))
        return h


    # ---- 作者の記録（原状回復のため）：部位 → (戸, 住み手の番号, 年)
    author = {}

    # ---- 所有者
    own = owner or None
    Lam = set(own.get("Lam", set())) if own else set()

    # ---- 〔s8〕所有者の収支（landlord の単価）
    LL = dict(r0=5.0, lam=0.5, v0=0.25, v1=0.02, cturn=1.0, Ho=10, Hv=10)
    if landlord is not None:
        LL.update(landlord)
    if own and own.get("kind") == "econ":
        LL.update({kk: own[kk] for kk in ("r0", "lam", "v0", "v1", "cturn", "Ho", "Hv") if kk in own})
    opl = dict(rent=0.0, vacancy=0.0, turn=0.0, refund=0.0, paid_in=0.0, n_exit=0)
    rp = np.zeros(9)                          # 今の世帯の家賃の上乗せ（年あたり、入居時に決まる）

    def owner_value(e0, k0, o0, e1, k1, o1):
        D0, X0_ = market(e0, k0, o0); D1, X1_ = market(e1, k1, o1)
        gam = own.get("gamma", 1.0) * (own.get("M", 10) if own["kind"] == "sub" else 1.0)
        return own.get("beta", 0.5) * float((D1 - D0).sum()) / MU - gam * float((X1_ - X0_).sum())

    def econ_cexit(sig):
        return (LL["v0"] + LL["v1"] * sig) * LL["r0"] + LL["cturn"]

    def econ_pi(Xa, Xr):
        """住戸ごとの Π_j(Xa, Xr)（長さ 9）：今の世帯の退去は Xa の形での住み心地で、次の入居は Xr の形で決まる"""
        Ho = LL["Ho"]
        dR, sR = market2(*Xr)
        out = np.zeros(9)
        SF_ = SF_all(W); Sa = S_all(W, *Xa, form)
        for j in range(9):
            h0 = hazard(j)
            hj = min(1.0, h0 * np.exp(-phi_x * (Sa[j] - SF_[j]) / MU))
            Y, P = stay_years(hj, Ho)
            Cj = econ_cexit(sR[j])
            nun = min(1.0, mean_nu(j) * np.exp(-phi_x * dR[j]))
            out[j] = -P * Cj + (Ho - Y) * (LL["lam"] * dR[j] / LL["Hv"] - nun * Cj)
        return out

    def econ_delta(i, X0, X1, p):
        """承諾の値 − 拒否の値（所有者の見通しの収支の差、㎡ 換算）"""
        Ho = LL["Ho"]
        if restore == "R0":
            hi_ = min(1.0, hazard(i) * np.exp(-phi_x * (S_all(W, *X1, form)[i] - SF_all(W)[i]) / MU))
            Yi, _ = stay_years(hi_, Ho)
            w = np.full(9, Yi / Ho); w[i] = 0.0
        else:
            w = np.ones(9)
        A = (w * econ_pi(X1, X1) + (1 - w) * econ_pi(X1, X0)).sum()
        if restore == "Rdk" and cost is not None and getp(X1, p) != getp(st, p):
            A -= float(cost.get(p[0], 0.0)) * refund_frac(mean_nu(i), amort)
        _, s0_ = market2(*X0)
        h_i = min(1.0, hazard(i) * np.exp(-phi_x * xfit(i, X0)))
        R = econ_pi(X0, X0).sum() - psi_x * h_i * econ_cexit(s0_[i])
        return float(A - R)

    def d4_list():
        L = set()
        for typ in ("ext", "store", "open"):
            cnt = 0
            S0_ = S_all(W, e, k, o, form)
            for i in range(9):
                for mv in legal_moves(i, e, k, o):
                    if TYPE[mv[0]] != typ:
                        continue
                    e1, k1, o1 = apply(mv, i, e, k, o)
                    if S_all(W, e1, k1, o1, form)[i] - S0_[i] > 1e-12:
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
                S0_ = S_all(W_now, e, k, o, form)
                nx = [e.copy(), k.copy(), o.copy()]; {"e": nx[0], "k": nx[1], "o": nx[2]}[p[0]][p[1]] = v
                dS = S_all(W_now, *nx, form) - S0_
                harmed = {j for j in range(9) if dS[j] < -1e-12}; gain = {j for j in range(9) if dS[j] > 1e-12}
                spent += move_cost(p, v, st, cost, rcost)
                events.append(dict(t=t, i=None, part=p, nv=int(v), ok=True, dS=dS, pay=np.zeros(9), cons=True,
                                   supp=harmed | gain, drafter="broker", broker=True))
                author.pop(p, None)
                e, k, o = nx

    if regime in ("P", "BR", "BRdyn"):
        central(-1, W)
    acct = (landlord is not None) or bool(own and own.get("kind") == "econ")
    for t in range(years):
        refused[:] = False
        if own and own.get("fix") == "D4":
            Lam = d4_list()
        if venue and venue.get("G") == 1:
            fee = venue.get("tauG", 0.5) / 9
            m -= fee; extra["tcost"] += fee * 9
            for i in range(9):
                led(t, "fee", hid(i), "out", fee, None, "m")
        if regime in ("R", "Rc"):
            for i in ORD[t]:
                if REQ[t, i] >= p_req:
                    continue
                cands = all_moves(e, k, o)
                mv = cands[int(r_null.integers(len(cands)))]
                S0_ = S_all(W, e, k, o, form)
                e1, k1, o1 = apply(mv, i, e, k, o)
                dS = S_all(W, e1, k1, o1, form) - S0_
                pay = np.zeros(9)
                if regime == "Rc":
                    for j in range(9):
                        if j != i:
                            pay[j] = max(0.0, -dS[j] / MU)
                    m[i] -= pay.sum(); m += pay
                    for j in range(9):
                        if pay[j] > 0:
                            led(t, "comp", hid(i), hid(j), pay[j], part_of(mv), "m")
                p = part_of(mv); nv = int(getp((e1, k1, o1), p))
                cst = move_cost(p, nv, st, cost, rcost); spent += cst; m[i] -= cst
                led(t, "cost", hid(i), "out", cst, p, "m")
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
                S0_ = S_all(W, e, k, o, form); C0 = S_all(Wbar, e, k, o, form)
                cands = all_moves(e, k, o) if plus else legal_moves(i, e, k, o)
                tr = base in ("T0", "T1", "T1h")
                reach = None
                if venue and venue.get("G") == 0 and tr:
                    u = r_ven.random(9)
                    reach = np.array([j == i or j in NEAR[i] or u[j] < venue.get("q_far", 0.4) for j in range(9)])
                evals = []
                for mv in cands:
                    e1, k1, o1 = apply(mv, i, e, k, o)
                    dS = S_all(W, e1, k1, o1, form) - S0_
                    dC = S_all(Wbar, e1, k1, o1, form) - C0
                    evals.append((mv, dS, dC, (e1, k1, o1)))
                drafter = "self"
                if regime == "AI+":
                    good = []
                    for x in evals:
                        p = part_of(x[0]); nv = int(getp(x[3], p))
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
                vetoed_good = False
                for mv, dS, dC, nxt in evals:
                    dSe = dS * gv if horizon is not None else dS
                    harmed_set = [j for j in range(9) if j != i and dS[j] < -1e-12]
                    if reach is not None and any(not reach[j] for j in harmed_set):
                        extra["unreached"] += 1
                        continue
                    val, ok, pay = decide(base, i, dSe, dC, None, pi, theta, split)
                    p = part_of(mv); nv = int(getp(nxt, p))
                    cst = move_cost(p, nv, st, cost, rcost)
                    ant = 0.0                                   # 見通しに入る、後で払う・受け取る見込み
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
                            pre_ok = bool(ok) and (val - cst - ant - tc) > 1e-12
                            okd = typ in Lam
                            if not okd and own.get("fix") == "D5":
                                # 一覧の外でも、所有者の市場での損（ind と同じ式）が住み手の正味の得を上回らなければ拒めない
                                okd = owner_value(e, k, o, *nxt) + (val - cst - ant - tc) >= -1e-12
                            if not okd:
                                ok = False; val = -1.0
                                if pre_ok:
                                    vetoed_good = True
                        else:
                            pre_ok = bool(ok) and (val - cst - ant - tc) > 1e-12
                            Vo = econ_delta(i, (e, k, o), nxt, p) if own["kind"] == "econ" else owner_value(e, k, o, *nxt)
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
                                if pre_ok:
                                    vetoed_good = True
                    val = val - cst - ant - tc - opay
                    if tr:
                        for j in range(9):
                            if j != i and dS[j] < -1e-12:
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
                    if vetoed_good:
                        refused[i] = True
                    continue
                val, ok, pay, mv, dS, dC, nxt, cst, tc, opay = best
                if val <= 1e-12:
                    if vetoed_good:
                        refused[i] = True
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
                p = part_of(mv)
                ev = dict(t=t, i=int(i), move=move_label(mv, i, e), type=move_type(mv, e), ok=bool(ok), val=float(val),
                          yes=nyes, part=p, nv=int(getp(nxt, p)), dS=dS, pay=pay.copy(), cons=cons,
                          supp=({int(i)} | harmed) if cons else set(), drafter=drafter, cost=cst if ok else 0.0)
                if ok:
                    for j in range(9):
                        if j != i:
                            b[j] -= dC[j] / MU
                            b[i] += dC[j] / MU
                    for j in range(9):
                        if pay[j] != 0:
                            b[i] += pay[j]; b[j] -= pay[j]
                            acc[i] -= pay[j]; acc[j] += pay[j]
                            if pay[j] > 0:
                                led(t, "comp", hid(i), hid(j), pay[j], p, medium)
                            else:
                                led(t, "comp", hid(j), hid(i), -pay[j], p, medium)
                    acc[i] -= cst; spent += cst
                    led(t, "cost", hid(i), "out", cst, p, medium)
                    if tc or opay:
                        acc[i] -= tc + opay; extra["tcost"] += tc; extra["owner_in"] += opay
                        led(t, "tcost", hid(i), "out", tc, p, medium)
                        led(t, "opay", hid(i), "owner", opay, p, medium)
                        opl["paid_in"] += opay
                    if ev["nv"] == getp(st, p):
                        author.pop(p, None)
                    else:
                        author[p] = (int(i), int(tenant[i]), t)
                    e, k, o = nxt
                events.append(ev)
        changed = False
        Snow = S_all(W, e, k, o, form)
        for i in range(9):
            hh[hid(i)]["Sy"].append(float(Snow[i]))
            if acct and rp[i] != 0:
                led(t, "rent", hid(i), "owner", rp[i], None, "m"); opl["rent"] += rp[i]
        if turning:
            for i in range(9):
                if turn_at is not None:
                    out = i in turn_at.get(t, ())
                elif kind is not None:
                    out = TURN[t, i] < hazard_fx(i)
                else:
                    out = TURN[t, i] < hazard_fx(i)
                if out:
                    hh[hid(i)]["S_exit"] = float(S_all(W, e, k, o, form)[i]); hh[hid(i)]["t_out"] = int(t)
                    # 原状回復：退去する世帯が起こした手
                    mine = [p for p, (u_, a_, _) in author.items() if u_ == i and a_ == tenant[i]]
                    for p in mine:
                        if restore == "R0":
                            s_ = getp(st, p)
                            if getp((e, k, o), p) != s_:
                                nx = [e.copy(), k.copy(), o.copy()]; {"e": nx[0], "k": nx[1], "o": nx[2]}[p[0]][p[1]] = s_
                                dS = S_all(W, *nx, form) - S_all(W, e, k, o, form)
                                rc = 0.0 if rcost is None else float(rcost.get(p[0], 0.0))
                                acc[i] -= rc; spent += rc
                                led(t, "restore", hid(i), "out", rc, p, medium)
                                events.append(dict(t=t, i=int(i), part=p, nv=int(s_), ok=True, dS=dS, pay=np.zeros(9),
                                                   cons=False, supp=set(), drafter="restore", cost=rc))
                                e, k, o = nx; extra["restored"] += 1
                            author.pop(p, None)
                        elif restore == "Rdk" and cost is not None:
                            age = t - author[p][2]
                            rf = float(cost.get(p[0], 0.0)) * max(0.0, 1 - age / amort)
                            acc[i] += rf; extra["refund"] += rf; extra["owner_out"] += rf
                            led(t, "refund", "owner", hid(i), rf, p, medium); opl["refund"] += rf
                    n_exit += 1; exit_bal += abs(b[i]); moved[i] = True
                    turns.append((t, i)); changed = True
                    if acct:
                        d_, s_g = market2(e, k, o)
                        rp[i] = LL["lam"] * d_[i] / LL["Hv"]
                        opl["vacancy"] += (LL["v0"] + LL["v1"] * s_g[i]) * (LL["r0"] + rp[i])
                        opl["turn"] += LL["cturn"]; opl["n_exit"] += 1
                    W[i] = newW(r_turn, i)
                    tenant[i] = nxt_tenant; nxt_tenant += 1; tenure[i] = 0
                    hh_new(i, t + 1)
                    if kind == "mix":
                        sig[i] = 0 if r_type.random() < turnover.get("piS", 0.72) else 1
                    if settle:
                        m[i] = 0.0; c[i] = 0.0; b[i] = 0.0
                    if reconsent:
                        gn = g_of(mean_nu(i), horizon) if horizon is not None else 1.0
                        e, k, o, nr, nv_, sp = _reconsent(i, t, W, e, k, o, st, form, reconsent, acc, lim, recon, rcost, gn,
                                                          author, events, logf=lambda *a: led(*a, medium), hidf=hid)
                        n_recon += nr; n_revert += nv_; spent += sp
                else:
                    tenure[i] += 1
        if changed and regime in ("P", "BRdyn"):
            central(t, W)
        Wt.append(S_all(W, e, k, o, form).sum())
    S_fin = S_all(W, e, k, o, form)
    return dict(e=e, k=k, o=o, S=S_fin, U=S_fin + MU * (m + c), b=b, m=m, c=c, Wt=np.array(Wt), events=events,
                W_final=W, n_exit=n_exit, exit_bal=exit_bal, n_recon=n_recon, n_revert=n_revert, moved=moved,
                quotes=quotes, turns=turns, recon=recon, spent=spent, start=X0, ref=st, extra=extra,
                ledger=ledger, hh=hh, owner_pl=opl, tenant=tenant.copy())


def _reconsent(n, t, W, e, k, o, st, form, scope, acc, lim, recon, rcost, gn=1.0, author=None, events=None,
               logf=None, hidf=None):
    """〔再〕入居した世帯 n の再同意（s1._reconsent と同じ規則。見通しがあれば満足の変化を g で割り引く）"""
    parts = [("o", x) for x in range(6)] + [("k", x) for x in range(6)]
    if scope == "all":
        parts += [("e", x) for x in range(9)]
    nr = nv = 0; sp = 0.0
    for p in parts:
        cur = getp((e, k, o), p); s_ = getp(st, p)
        if cur == s_:
            continue
        nx = [e.copy(), k.copy(), o.copy()]; {"e": nx[0], "k": nx[1], "o": nx[2]}[p[0]][p[1]] = s_
        dS = S_all(W, *nx, form) - S_all(W, e, k, o, form)
        rc = 0.0 if rcost is None else float(rcost.get(p[0], 0.0))
        G = gn * dS[n] / MU - rc
        if G <= 1e-12:
            recon.append((t, n, p, "accept"))
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
                    if logf is not None and s[j] > 0:
                        logf(t, "recon", hidf(j), hidf(n), s[j], p)
                acc[n] += G
        recon.append((t, n, p, "paid" if keep else "revert"))
        if not keep:
            e, k, o = nx; nv += 1; acc[n] -= rc; sp += rc
            if logf is not None:
                logf(t, "recon_cost", hidf(n), "out", rc, p)
            if author is not None:
                author.pop(p, None)
    return e, k, o, nr, nv, sp
