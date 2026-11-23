"""職能の実験のための推定器 vstar3。trade/vstar.py（名付けで凍結）と trade/vstar2.py（確証で凍結）は変えず、import して次を足した。

1. 運用役の「役」の枠。s5 の記録 actor_facts[名]＝[(年, 戸, 係数, 順)]、actor_full[名]＝[(年, 順)] から、運用役ごとに 30 年目に有効な知識を作り、
   その知識だけで解いた最適 GA[名] を記録に足す。入れ替わった住戸についての事実は、住み手の作者と同じ規則で無効になる。
   運用役は住戸ではなく役ごとに一つの枠で数える（確証の代表の役の枠と同じ考え方）。
2. 作者の集合の切り替え。population3(recs, key, count=…) の count は作者に数える運用役の名（または種類）の集合。
   count が空なら運用役を作者に数えない値 V*n〔外〕、運用役を入れれば数える値 V*n〔数〕。作者を足すと max の項は減らないので〔数〕≤〔外〕。
3. 記録を読む住み手。記録者が額まで記録した局（ledger）では、住み手が記録を読んで係数を知る。読み方 ledger_mode：
     "new"   入居した世帯が、入居の時点までの記録をすべて読む（主な読み方）
     "all"   "new" に加えて、記録が書かれるたびに、そのときの住み手 9 戸が読む（感度の分析）
     None    読まない（記録の額が住み手の知識に入らない場合。比べるため）
4. 盤。戻せる手の多い盤（s5.RICH、部位 31）では、30 年目の全知の最適 Z・形 X・推定 G を 31 部位で作る。
   η は今の盤の到達できる幅を 1 とする値（eta、盤のあいだで同じ物差し）と、その盤の到達できる幅を 1 とする値（eta_own）を返す。
   V*n は、分母を今の盤の 21 部位のエントロピーの合計にした値（V、同じ物差し）と、31 部位の合計にした値（V_own）を返す。
5. 検証者の制約のもとでの物差し（感度の分析）。運用役のある局の記録に、張り出しを「同じ列で二層まで」に制限した最適 Zc と、
   その最適を到達できる幅とする η（eta_cap）を足す。population3(…, Z_key="Zc") で、制約のもとの最適に対する V*n になる。
今の盤と狭い盤（部位は今の盤と同じ 21）で、記録の額を読まない局では、record3 は vstar.record と同じものに役の枠を足して返し、
population3 は運用役を数えなければ vstar.population と同じ値を返す（tests/test_s5.py）。
"""
from __future__ import annotations
import numpy as np
from trade import s0, vstar
from trade.s1 import getp
from trade.vstar import cfgvec, guess, H_, MI, entropy_tokens, PB, ALLF
from trade import s5

RICH_KEYS = s5.RICH.keys()
BASE_IDX = [n for n, p in enumerate(RICH_KEYS) if (p[0] == "e") or (p[0] in "ko" and p[1] < 6)]     # 今の盤の 21 部位の位置


def guess3(W, known, board):
    """known の係数だけ本当の値、ほかは平均の性向で解いた最適（vstar.guess と同じ規則を、盤の最適で解く）"""
    if board.name != "rich":
        return guess(W, known)
    G = np.zeros((9, 4))
    for j in range(9):
        a = W[j, 0] if (j, "a") in known else PB["a"]
        s = W[j, 1] if (j, "s") in known else PB["s"]
        q = W[j, 2] - W[j, 3] if (j, "q") in known else PB["q"]
        r = max(1e-9, 1 - a - s)
        G[j] = [a, s, (r + q) / 2, (r - q) / 2]
    _, e, k, o = board.wstar(G)
    return cfgvec(e, k, o)


def _board(r):
    return {"base": s5.BASE, "narrow": s5.NARROW, "rich": s5.RICH}[r.get("board", "base")]


# ------------------------------------------------------------ 知識
def knowledge3(r, start_author=None, ledger_mode="new"):
    """vstar.knowledge と同じ規則に、記録を読む住み手を足したもの。
    返り値 (今の住み手 9 戸の知識, 中央の作者の知識 or None, 去った作者のうち最大の知識)"""
    led = r.get("ledger") or []
    if not led or ledger_mode is None:
        return vstar.knowledge(r, start_author)
    tenant = list(range(9)); nxt = 9
    facts = {a: {(j, p, a) for p in "asq"} for a in range(9) for j in [a]}
    central = None
    if start_author == "planner":
        central = {(j, p, j) for j, p in ALLF}
    tl = [((q[0], 0, 0), "q", q) for q in r["quotes"]]
    tl += [((t, 1, sq), "L", (j, c)) for (t, j, c, sq) in led]
    tl += [((t, 1, sq), "T", u) for (kind, t, u, sq) in r["seqlog"] if kind == "turn"]
    cent_ai = sorted({ev["t"] for ev in r["events"] if ev.get("drafter") == "ai"})
    cent_pl = sorted({ev["t"] for ev in r["events"] if ev.get("drafter") in ("planner", "broker")})
    tl += [((t, 0.5, 0), "C", None) for t in cent_ai] + [((t, 2, 0), "C", None) for t in cent_pl]
    tl.sort(key=lambda x: x[0])
    hist = []
    for _, typ, x in tl:
        if typ == "C":
            central = (central or set()) | {(j, p, tenant[j]) for j, p in ALLF}
        elif typ == "q":
            _, i, j, p = x
            facts.setdefault(tenant[i], set()).add((j, p, tenant[j]))
        elif typ == "L":
            j, c = x
            f = (j, c, tenant[j]); hist.append(f)
            if ledger_mode == "all":
                for u in range(9):
                    facts[tenant[u]].add(f)
        else:
            tenant[x] = nxt; facts[nxt] = {(x, p, nxt) for p in "asq"} | set(hist); nxt += 1
    valid = lambda F: {(j, p) for (j, p, a) in F if tenant[j] == a}
    cur = [valid(facts[tenant[u]]) for u in range(9)]
    former = [valid(F) for a, F in facts.items() if a not in tenant]
    best_former = max(former, key=len) if former else set()
    return cur, (valid(central) if central is not None else None), best_former


def actor_knowledge(r, name):
    """運用役 name が 30 年目に持っている有効な事実 (戸, 係数) の集合"""
    tenant = list(range(9)); nxt = 9
    tl = [((t, sq), "F", (j, c)) for (t, j, c, sq) in r["actor_facts"].get(name, [])]
    tl += [((t, sq), "A", None) for (t, sq) in r["actor_full"].get(name, [])]
    tl += [((t, sq), "T", u) for (kind, t, u, sq) in r.get("seqlog", []) if kind == "turn"]
    tl.sort(key=lambda x: x[0])
    facts = set()
    for _, typ, x in tl:
        if typ == "F":
            facts.add((x[0], x[1], tenant[x[0]]))
        elif typ == "A":
            facts |= {(j, p, tenant[j]) for j, p in ALLF}
        else:
            tenant[x] = nxt; nxt += 1
    return {(j, p) for (j, p, a) in facts if tenant[j] == a}


# ------------------------------------------------------------ 同意で支えられた部位（盤の満足で）
def supported3(r, W_T, start, board):
    """vstar.supported と同じ規則。満足を盤の式で計算する（戻せる手の多い盤のため）"""
    if board.name != "rich":
        return vstar.supported(r, W_T, start)
    X0d = {"e": start[0], "k": start[1], "o": start[2]}
    st = {}; byt = {}
    for ev in r["events"]:
        byt.setdefault(ev["t"], []).append(("ev", ev))
    for (t, u) in r["turns"]:
        byt.setdefault(t, []).append(("turn", u))
    rc = {}
    for (t, n, p, kind) in r["recon"]:
        rc.setdefault((t, n), []).append((p, kind))
    for t in sorted(byt):
        for kind, x in byt[t]:
            if kind == "ev":
                ev = x
                if ev.get("planner"):
                    if ev.get("changed", True):
                        st = {}
                    continue
                if not ev["ok"] or ev.get("part") is None:
                    continue
                p = ev["part"]
                if ev["nv"] == X0d[p[0]][p[1]]:
                    st.pop(p, None); continue
                st[p] = [ev["nv"], set(ev["supp"]) if ev["cons"] else None]
            else:
                for p, v in st.items():
                    if v[1] is not None:
                        v[1].discard(x)
                for p, how in rc.get((t, x), []):
                    if how == "revert":
                        st.pop(p, None)
                    elif p in st and st[p][1] is not None:
                        st[p][1].add(x)
    e, k, o = r["e"], r["k"], r["o"]
    sup, neg = set(), set()
    for p, (v, C) in st.items():
        if getp((e, k, o), p) != v or C is None:
            continue
        nx = [e.copy(), k.copy(), o.copy()]; {"e": nx[0], "k": nx[1], "o": nx[2]}[p[0]][p[1]] = X0d[p[0]][p[1]]
        dS = board.S(W_T, *nx) - board.S(W_T, e, k, o)
        hn = {j for j in range(9) if dS[j] > 1e-12}
        if hn <= C:
            sup.add(p)
            if hn:
                neg.add(p)
    return sup, neg


# ------------------------------------------------------------ 検証者の制約のもとでの最適（物差しの感度のため）
def ext_cap(W, e_opt, limit=2):
    """張り出しを「同じ列で limit 層まで」に制限した最適の張り出し。線形の満足では張り出しの寄与は戸ごとに分かれ、
    制約は列ごとにしか効かないので、列ごとに寄与の大きい順に limit 個まで残せば最適になる"""
    da = 1.0 / (s0.A_HI - s0.A_LO); ds = 1.0 / s0.SUN_HI
    c = np.zeros(9)
    for i in range(9):
        c[i] = W[i, 0] * s0.EXT_A * da
        for d, sh in zip((1, 2), s0.SHADE):
            j = i + 3 * d
            if j < 9:
                c[i] -= W[j, 1] * sh * ds
    e = np.array(e_opt, int).copy()
    for col in range(3):
        on = [u for u in (col, col + 3, col + 6) if e[u] == 1]
        if len(on) > limit:
            for u in sorted(on, key=lambda u: c[u])[:len(on) - limit]:
                e[u] = 0
    return e


def cap_fields(rec, r, board):
    """制約のもとの最適 Zc と、その物差しの η（eta_cap）を足す。張り出しの部位は Z の先頭 9"""
    WT = r["W_final"]
    Z = np.array(rec["Z"]); ec = ext_cap(WT, Z[:9])
    Zc = Z.copy(); Zc[:9] = ec
    F = board.start()
    WF = board.S(WT, *F).sum(); Wx = board.S(WT, r["e"], r["k"], r["o"]).sum()
    _, e21, k21, o21 = s0.wstar_linear(WT)
    Wc = s0.S_all(WT, ext_cap(WT, e21), k21, o21).sum()
    rec["Zc"] = Zc.tolist()
    rec["eta_cap"] = float((Wx - WF) / (Wc - WF)) if Wc - WF > 1e-6 else float("nan")
    return rec


# ------------------------------------------------------------ 一局の記録
def _G(WT, kn3, board, F):
    cur, cen, former = kn3
    G = [guess3(WT, kn, board) for kn in cur]
    G.append(guess3(WT, cen, board) if cen is not None else cfgvec(*F))
    G.append(guess3(WT, former, board))
    return [g.tolist() for g in G]


def _record_rich(r, W0, start_author, board):
    F = board.start()
    WT = r["W_final"]
    _, ez, kz, oz = board.wstar(WT)
    Z = cfgvec(ez, kz, oz)
    X = cfgvec(r["e"], r["k"], r["o"])
    start = r["ref"]
    S0v = cfgvec(*start)
    KEYS = board.keys()
    sup, neg = supported3(r, WT, start, board)
    Xc = np.array([X[n] if KEYS[n] in sup else S0v[n] for n in range(len(KEYS))])
    Xn = np.array([X[n] if KEYS[n] in neg else S0v[n] for n in range(len(KEYS))])
    WF = board.S(WT, *F).sum(); Ws = board.wstar(WT)[0]; Wx = board.S(WT, r["e"], r["k"], r["o"]).sum()
    Ws21 = s0.wstar_linear(WT)[0]
    oks = [ev for ev in r["events"] if ev.get("ok") and not ev.get("planner") and ev.get("part") is not None]
    loss_all = loss_cons = 0.0; third = 0.0
    for ev in oks:
        if "dS" not in ev:
            continue
        dS = ev["dS"]; harmed = [j for j in range(9) if j != ev.get("i") and dS[j] < -1e-12]
        L = -dS[harmed].sum() if harmed else 0.0
        loss_all += L
        if ev["cons"]:
            loss_cons += L
        inv = set(harmed) | ({ev["i"]} if ev.get("i") is not None else set())
        third += sum(max(0.0, dS[j]) for j in range(9) if j not in inv)
    U = r["U"]; SF_i = board.S(WT, *F)
    return dict(Z=Z.tolist(), X=X.tolist(), Xc=Xc.tolist(), Xn=Xn.tolist(), G=None,
                eta=float((Wx - WF) / (Ws21 - WF)) if Ws21 - WF > 1e-6 else float("nan"),
                dSstar=float(Ws21 - WF), L=float(np.mean(U < SF_i - 1e-9)), B=float(np.abs(r["b"]).sum() / 9),
                moves=len(oks), quotes=len(r["quotes"]), Lc=len(sup) / len(KEYS), Ln=len(neg) / len(KEYS),
                Vp=float(loss_cons / loss_all) if loss_all > 1e-12 else float("nan"),
                gohai=float(third / s0.MU), ent=entropy_tokens([str(ev.get("part")) for ev in oks]),
                rebind=sum(1 for x in r["recon"] if x[3] == "paid"), spent=float(r.get("spent", 0.0)),
                H=float(np.mean(np.abs(W0 - vstar.WB).sum(1) / 2)),
                eta_own=float((Wx - WF) / (Ws - WF)) if Ws - WF > 1e-6 else float("nan"), dSstar_own=float(Ws - WF))


def record3(r, W0, start_author=None, F=None, ledger_mode="new", ledger_sens=("all", None)):
    """一局の記録。運用役のない局では vstar.record と同じ（キーも値も）。運用役のある局では次を足す：
    GA（運用役ごとの役の枠の推定）、actor_kind、board、eta_own、Gsens（記録の読み方を替えた住み手の推定。感度の分析）"""
    board = _board(r)
    F = F if F is not None else s0.f_design()
    if board.name == "rich":
        rec = _record_rich(r, W0, start_author, board)
        rec["G"] = _G(r["W_final"], knowledge3(r, start_author, ledger_mode), board, board.start())
    else:
        rec = vstar.record(r, W0, start_author, F)
        if r.get("ledger") and ledger_mode is not None:
            rec["G"] = _G(r["W_final"], knowledge3(r, start_author, ledger_mode), board, F)
    if "actor_facts" not in r:
        return rec
    WT = r["W_final"]
    rec["board"] = board.name
    rec.setdefault("eta_own", rec["eta"])
    rec["GA"] = {nm: guess3(WT, actor_knowledge(r, nm), board).tolist() for nm in r["actor_facts"]}
    rec["actor_kind"] = dict(r["actor_kind"])
    rec["actor_nfacts"] = {nm: len(actor_knowledge(r, nm)) for nm in r["actor_facts"]}
    Fb = board.start() if board.name == "rich" else F
    if r.get("ledger"):
        rec["Gsens"] = {str(m): _G(WT, knowledge3(r, start_author, m), board, Fb) for m in ledger_sens}
    if r.get("mixed_quotes"):
        # 組の手で、一つの言い値に二つの係数が混ざる場合は、どちらの係数も分からないとする読み方（感度の分析）
        mx = set(r["mixed_quotes"])
        r2 = dict(r, quotes=[q for n_, q in enumerate(r["quotes"]) if n_ not in mx])
        rec.setdefault("Gsens", {})["swap_none"] = _G(WT, knowledge3(r2, start_author, ledger_mode), board, Fb)
    rec["actor_stats"] = r.get("actor_stats", {})
    rec["n_blocked"] = len(r.get("blocked", []))
    return cap_fields(rec, r, board)


# ------------------------------------------------------------ 集団の量
def with_authors(recs, count=(), G_key=None):
    """count に入る運用役（名か種類）の役の枠を作者に足した写し。G_key を与えると住み手の推定をその読み方の推定に替える"""
    count = set(count)
    out = []
    for x in recs:
        G = x["G"] if (G_key is None or "Gsens" not in x) else x["Gsens"][G_key]     # 記録が空の局は読み方で変わらない
        add = [g for nm, g in x.get("GA", {}).items() if nm in count or x["actor_kind"][nm] in count]
        if add or G_key is not None:
            out.append(dict(x, G=G + add))
        else:
            out.append(x)
    return out


def population3(recs, key="Xn", idx=None, count=(), G_key=None, Z_key="Z"):
    """V*n（count が空なら〔外〕、運用役を入れれば〔数〕）。部位が 21 なら vstar.population と同じ値。
    部位が 31（戻せる手の多い盤）なら、V は分母を今の盤の 21 部位にした値、V_own は 31 部位にした値"""
    R = with_authors(recs, count, G_key)
    if Z_key != "Z":
        R = [dict(x, Z=x[Z_key]) for x in R]
    if len(R[0]["Z"]) == 21:
        out = vstar.population(R, key, idx)
        out["V_own"] = out["V"]
        return out
    if idx is None:
        idx = np.arange(len(R))
    Z = np.array([R[i]["Z"] for i in idx]); X = np.array([R[i][key] for i in idx])
    G = np.array([R[i]["G"] for i in idx])
    P = Z.shape[1]
    Hp = [H_(list(Z[:, p])) for p in range(P)]
    HZ = sum(Hp[p] for p in BASE_IDX); HZo = sum(Hp)
    if HZ < 1e-9:
        return dict(V=0.0, I=0.0, Imax=0.0, HZ=0.0, V_own=0.0)
    I = sum(MI(list(Z[:, p]), list(X[:, p])) for p in range(P))
    Ia = [sum(MI(list(Z[:, p]), list(G[:, a, p])) for p in range(P)) for a in range(G.shape[1])]
    return dict(V=max(0.0, I - max(Ia)) / HZ, I=I / HZ, Imax=max(Ia) / HZ, HZ=HZ, V_own=max(0.0, I - max(Ia)) / HZo, HZ_own=HZo)


def boot3(recs, key="Xn", n=1000, seed=0, count=(), G_key=None, field="V", Z_key="Z"):
    rng = np.random.default_rng(seed)
    N = len(recs)
    v = [population3(recs, key, rng.integers(0, N, N), count, G_key, Z_key)[field] for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def diff3(recsA, recsB, key="Xn", n=1000, seed=0, countA=(), countB=(), GA_key=None, GB_key=None, field="V", Z_key="Z"):
    """V*n(A) − V*n(B)：同じ構成の組（cs で並べた同じ位置）を一緒に引き直して、差の 95 % 区間を出す"""
    assert [x["cs"] for x in recsA] == [x["cs"] for x in recsB]
    d0 = population3(recsA, key, None, countA, GA_key, Z_key)[field] - population3(recsB, key, None, countB, GB_key, Z_key)[field]
    rng = np.random.default_rng(seed)
    N = len(recsA); out = []
    for _ in range(n):
        ix = rng.integers(0, N, N)
        out.append(population3(recsA, key, ix, countA, GA_key, Z_key)[field] - population3(recsB, key, ix, countB, GB_key, Z_key)[field])
    return dict(d=float(d0), lo=float(np.percentile(out, 2.5)), hi=float(np.percentile(out, 97.5)))
