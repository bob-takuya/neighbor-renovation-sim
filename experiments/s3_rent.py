"""段 1 第二報：賃貸の制約の四つの群を、模型 s3 で走らせる。
  T 入れ替わり：平均の退去率 ν の掃引と、居住年数の分布の形（幾何・更新の山・二型の混合）、性向を引き継ぐ対照、帰無の対照
  R 原状回復：R∅（何も起きない＝ゼロ精算）・R0（日本の既定）・Rdk10・Rdk20（償却つきの補償）。施工費 C1 と 10 年の見通しのもとで
  O 所有と決める場：借家（個人の所有者・拒否権のみ＋R0）と所有（所有者なし＋R∅）× 決める場 G（0・½・1）× ν
  W 所有者の型：なし・個人・管理会社・サブリース × 拒否権のみ／取引の相手 ＋ 直し D4（一覧を住み手が決める）・D5（目的限定の承諾請求権）
構成：開発用 cseed 0〜19（規則と値を決める）と、検証用 cseed 400〜459（事前登録の後に一度だけ走らせる）。パターンは α＝3 と二群。
PYTHONPATH=. python3 -m experiments.s3_rent dev|main   → out/s3/dev.jsonl | out/s3/main.jsonl
"""
import sys, json, os
import numpy as np
from multiprocessing import Pool
from trade import s0, s3, vstar
from experiments.explore import composition

PATS = ["α=3", "二群"]
C1 = (dict(e=1.0, k=0.25, o=0.25), dict(e=0.5, k=0.1, o=0.1))
IMP = dict(credit=20.0, settle=True, reconsent="all")
REG = {"U": ("U", {}), "V50": ("V½", {}), "TP+": ("T1+", {}), "TP+top3": ("T1+", dict(protocol="topk", ask=3)),
       "改良版": ("T1+", dict(IMP)), "改良版top3": ("T1+", dict(IMP, protocol="topk", ask=3)),
       "N3": ("P", {}), "BRdyn": ("BRdyn", {}), "N4": ("R", {})}
F0 = s0.f_design()
HZ = 10
OWN = {
    "なし": None,
    "個人・拒否": dict(kind="ind", role="veto", beta=0.5, gamma=1.0),
    "個人・取引": dict(kind="ind", role="trade", beta=0.5, gamma=1.0),
    "個人・D5": dict(kind="ind", role="veto", beta=0.5, gamma=1.0, fix="D5"),
    "管理∅・拒否": dict(kind="mgmt", role="veto", Lam=set(), beta=0.5, gamma=1.0),
    "管理∅・取引": dict(kind="mgmt", role="trade", Lam=set(), beta=0.5, gamma=1.0),
    "管理∅・D4": dict(kind="mgmt", role="veto", Lam=set(), beta=0.5, gamma=1.0, fix="D4"),
    "管理∅・D5": dict(kind="mgmt", role="veto", Lam=set(), beta=0.5, gamma=1.0, fix="D5"),
    "管理開口・拒否": dict(kind="mgmt", role="veto", Lam={"open"}, beta=0.5, gamma=1.0),
    "サブ・拒否": dict(kind="sub", role="veto", beta=0.5, gamma=1.0, M=10),
    "サブ・取引": dict(kind="sub", role="trade", beta=0.5, gamma=1.0, M=10),
    "サブ・D5": dict(kind="sub", role="veto", beta=0.5, gamma=1.0, M=10, fix="D5"),
}
VEN = {"G0": dict(G=0, q_far=0.4, tau=0.5), "G½": dict(G=0.5, alpha=0.5), "G1": dict(G=1, tauG=0.0)}


def conds():
    """(群, 条件名, 決め方, s3.run への引数（JSON にできる形）)"""
    out = []
    RT = ["U", "V50", "TP+", "TP+top3", "改良版", "改良版top3"]
    for nu in [0.0, 0.05, 0.10, 0.18, 0.30]:
        for rg in RT:
            out.append(("T", f"幾何 ν={nu:.2f}", rg, dict(tv=["geo", nu])))
        for rg in ["N3", "BRdyn", "N4"]:
            out.append(("T", f"幾何 ν={nu:.2f}", rg, dict(tv=["geo", nu])))
    for rg in RT:
        out.append(("T", "更新の山 ν=0.18", rg, dict(tv=["renew", 0.18])))
        out.append(("T", "二型の混合", rg, dict(tv=["mix", None])))
    for nu in [0.10, 0.30]:
        for rg in ["TP+top3", "改良版top3"]:
            out.append(("T", f"性向を引き継ぐ ν={nu:.2f}", rg, dict(tv=["geo", nu], inherit=True)))
    for nu in [0.05, 0.10, 0.18, 0.30]:
        for rg in ["U", "TP+top3", "改良版top3"]:
            for R in ["R∅", "R0", "Rdk10", "Rdk20"]:
                out.append(("R", f"{R} 費用あり ν={nu:.2f}", rg, dict(tv=["geo", nu], R=R, cost=1)))
            for R in ["R∅", "R0"]:
                out.append(("R", f"{R} 費用なし ν={nu:.2f}", rg, dict(tv=["geo", nu], R=R, cost=0)))
    for nu in [0.04, 0.18, 0.30]:
        for rg in ["U", "TP+top3", "改良版top3"]:
            for own, gs in [("借家", ["G0", "G½", "G1"]), ("所有", ["G0", "G1"])]:
                for G in gs:
                    out.append(("O", f"{own}・{G} ν={nu:.2f}", rg, dict(tv=["geo", nu], R="R0" if own == "借家" else "R∅", cost=1,
                                                                    own="個人・拒否" if own == "借家" else "なし", ven=G)))
    for nu in [0.10, 0.30]:
        for rg in ["U", "TP+top3", "改良版top3"]:
            for o in OWN:
                out.append(("W", f"{o} ν={nu:.2f}", rg, dict(tv=["geo", nu], R="R0", cost=1, own=o)))
    return out


def kwargs_of(spec, W, pat):
    kw = {}
    kind, nu = spec["tv"]
    if kind == "geo":
        if nu > 0:
            kw["turnover"] = dict(kind="geo", nu=nu)
    elif kind == "renew":
        kw["turnover"] = dict(kind="renew", nu=nu, rho=2.0)
    else:
        kw["turnover"] = dict(kind="mix")
    if "turnover" in kw:
        if spec.get("inherit"):
            kw["newW"] = lambda rng, i, W0=W.copy(): W0[i]
        else:
            kw["newW"] = lambda rng, i: composition(pat, int(rng.integers(10**6, 10**7)))[i]
    R = spec.get("R", "R∅")
    if R == "R0":
        kw["restore"] = "R0"
    elif R.startswith("Rdk"):
        kw["restore"] = "Rdk"; kw["amort"] = int(R[3:])
    if spec.get("cost"):
        kw["cost"], kw["rcost"] = C1; kw["horizon"] = HZ
    if spec.get("own") and OWN[spec["own"]] is not None:
        kw["owner"] = OWN[spec["own"]]
    if spec.get("ven"):
        kw["venue"] = VEN[spec["ven"]]
    return kw


def one(a):
    grp, cname, reg, spec, pat, cs = a
    W = composition(pat, cs)
    rr, kw0 = REG[reg]
    kw = dict(kw0); kw.update(kwargs_of(spec, W, pat))
    r = s3.run(W, rr, cs, **kw)
    rec = vstar.record(r, W, None, F0)
    ex = r["extra"]
    U = r["U"]; SF = s0.S_all(r["W_final"], *F0)
    stay = ~r["moved"]
    rec.update(grp=grp, cond=cname, reg=reg, pat=pat, cs=cs, exit_rate=r["n_exit"] / 270,
               L_stay=float(np.mean(U[stay] < SF[stay] - 1e-9)) if stay.any() else float("nan"),
               L_new=float(np.mean(U[~stay] < SF[~stay] - 1e-9)) if (~stay).any() else float("nan"),
               changed=int(sum(1 for p in range(21) if rec["X"][p] != vstar.cfgvec(*r["ref"])[p])),
               owner_in=ex["owner_in"], refund=ex["refund"], restored=ex["restored"], tcost=ex["tcost"],
               unreached=ex["unreached"], recon_paid=sum(1 for x in r["recon"] if x[3] == "paid"),
               recon_revert=sum(1 for x in r["recon"] if x[3] == "revert"))
    return rec


def tasks(cs_list, groups="TROW"):
    return [(g, c, rg, sp, p, cs) for (g, c, rg, sp) in conds() if g in groups for p in PATS for cs in cs_list]


if __name__ == "__main__":
    mode = sys.argv[1]
    os.makedirs("out/s3", exist_ok=True)
    if mode == "dev":
        T = tasks(list(range(0, 20))); fn = "out/s3/dev.jsonl"
    else:
        T = tasks(list(range(400, 460))); fn = "out/s3/main.jsonl"
    print(len(T), "局", flush=True)
    with Pool(2) as pool, open(fn, "w") as f:
        for i, rec in enumerate(pool.imap_unordered(one, T, chunksize=8)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 2000 == 0:
                print(i, flush=True)
    print("done")
