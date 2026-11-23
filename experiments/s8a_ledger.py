"""観察 3：既存の条件を同じ種で再実行し、世帯ごとの 30 年の帳簿を作る（新しい実験ではない）。

対象
  観察 1 の主条件：決め方 U・V50・V75・TR+・TP+・TP+貸清再、パターン α=3・二群、検証用の構成 cseed 100〜129、各 10 局
                   （局の種 1000·cseed＋局番号。experiments/s1_main.py と同じ）
  観察 2：入れ替わりの群  TP+top3・改良版top3 × 幾何 ν＝0・0.10・0.30（費用なし）
          原状回復の群    U・TP+top3・改良版top3 × R∅・R0・Rdk10（ν＝0.30、費用 1 倍と見通し 10 年）
          所有者の型の群  TP+top3・改良版top3 × 個人・取引の相手（ν＝0.10・0.30、R0、費用あり）  ← 所有者に払った額を見るため
          検証用の構成 cseed 400〜459、局の種＝cseed（experiments/s3_rent.py と同じ引数・同じ種）
再実行の確認（二重）
  (1) 凍結したパイプライン（s1_main.one_run・s3_rent.one）をそのまま実行し、凍結した記録（out/s1/main.csv・out/s3/main.jsonl.gz）と全項目を照合
  (2) 帳簿を取るための s8.run（既定の設定＝s3 と同じ）からも同じ量を計算し、同じ記録と照合
出力：out/s8a/households_obs1.csv.gz・households_obs2.csv.gz（1 行が 1 世帯）、out/s8a/runs.csv.gz（1 行が 1 局。照合の結果）、out/s8a/verify.json、
      out/s8a/ledger_examples.json.gz（代表の棟の候補の帳簿の全件）
PYTHONPATH=. python3 -m experiments.s8a_ledger run [obs1|obs2|all]
"""
import sys, os, json, gzip, csv, time
from multiprocessing import Pool
import numpy as np
from trade import s0, s8, vstar
from experiments.explore import composition, hetero
import experiments.s1_main as S1
import experiments.s3_rent as S3

OUT = "out/s8a"
OBS1_RULES = ["U", "V50", "V75", "TR+", "TP+", "TP+貸清再"]
PATS = ["α=3", "二群"]
TOL = 1e-9


def ptype(pat, w):
    w = np.asarray(w)
    if pat == "二群":
        return "広さ派" if w[0] > 0.5 else "静けさ派"
    return ["広さ重視", "日照重視", "静けさ重視", "近さ重視"][int(np.argmax(w / s0.W_BAR))]


def households(r, pat, meta, years=30):
    """s8.run の結果から世帯ごとの帳簿の行を作る（金額は ㎡ 換算）"""
    acc = {h: dict(paid_comp=0.0, recv_comp=0.0, paid_recon=0.0, recv_recon=0.0, cost=0.0, restore=0.0, tcost=0.0,
                   opay=0.0, refund=0.0, rent=0.0, fee=0.0) for h in r["hh"]}
    for (t, kd, f, to, amt, part, med) in r["ledger"]:
        if kd == "comp":
            acc[f]["paid_comp"] += amt; acc[to]["recv_comp"] += amt
        elif kd == "recon":
            acc[f]["paid_recon"] += amt; acc[to]["recv_recon"] += amt
        elif kd == "cost":
            acc[f]["cost"] += amt
        elif kd in ("restore", "recon_cost"):
            acc[f]["restore"] += amt
        elif kd in ("tcost", "opay", "fee", "rent"):
            acc[f][kd] += amt
        elif kd == "refund":
            acc[to]["refund"] += amt
    props = {}
    for ev in r["events"]:
        if ev.get("ok") and ev.get("i") is not None and ev.get("drafter") in ("self", "ai"):
            props[(ev["i"], ev["t"])] = props.get((ev["i"], ev["t"]), 0) + 1
    SF_fin = s0.S_all(r["W_final"], *s0.f_design())
    rows = []
    nth = {}
    for h in sorted(r["hh"]):
        v = r["hh"][h]; u = v["unit"]
        nth[u] = nth.get(u, 0) + 1
        a = acc[h]
        stay = v["t_out"] is None
        S_end = float(r["S"][u]) if stay else v["S_exit"]
        dS = (S_end - v["SF"]) / s0.MU
        dSavg = float(np.mean(np.array(v["Sy"]) - v["SF"]) / s0.MU) if v["Sy"] else 0.0
        money = a["recv_comp"] - a["paid_comp"] + a["recv_recon"] - a["paid_recon"] - a["cost"] - a["restore"] - a["tcost"] - a["opay"] + a["refund"] - a["fee"]
        t_out = years - 1 if stay else v["t_out"]
        n_prop = sum(c for (i, t), c in props.items() if i == u and v["t_in"] <= t <= t_out)
        rows.append(dict(meta, unit=s0.UNITS[u], floor=int(s0.FLOOR[u]), col="WME"[s0.COL[u]], hid=h, nth=nth[u],
                         t_in=v["t_in"], t_out=(None if stay else v["t_out"]), years=t_out - v["t_in"] + 1, stay=int(stay),
                         type=ptype(pat, v["W"]), w=";".join(f"{x:.3f}" for x in v["W"]),
                         dS=dS, dS_avg=dSavg, money=money, net=dS + money, n_prop=n_prop, **a))
    return rows


# ------------------------------------------------------------------ 観察 1
def obs1_task(args):
    pat, cs = args
    W = composition(pat, cs)
    out_runs, out_hh = [], []
    for rule in OBS1_RULES:
        reg, kw = S1.RULES[rule]
        for sd in range(S1.NS_MAIN):
            seed = 1000 * cs + sd
            d1 = S1.one_run(W, rule, seed, {}, pat, cs)                     # 凍結したパイプライン
            r = s8.run(W, reg, seed, **kw)                                  # 帳簿を取る s8（既定の設定）
            eF, kF, oF = s0.f_design()
            SF = s0.S_all(r["W_final"], eF, kF, oF)
            wstar = s0.wstar_linear(r["W_final"])[0]
            Wfin = float(r["S"].sum()); WF = float(SF.sum())
            d8 = dict(W=Wfin, WF=WF, Wstar=wstar, eta=(Wfin - WF) / (wstar - WF) if wstar - WF > 1e-6 else np.nan,
                      L=float((r["U"] < SF - 1e-9).mean()), B=float(np.abs(r["b"]).sum() / 9),
                      minS=float(r["S"].min()), n_moves=sum(1 for x in r["events"] if x["ok"]),
                      n_fail=sum(1 for x in r["events"] if not x["ok"]),
                      cfg="".join(map(str, r["e"])) + "|" + "".join(str(x + 1) for x in r["k"]) + "|" + "".join(map(str, r["o"])))
            meta = dict(src="obs1", cond="主条件", reg=rule, pat=pat, cs=cs, seed=sd)
            hhrows = households(r, pat, meta)
            # 整合：入れ替わりがないので、世帯の正味 × μ ＝ 暮らし向き − F の満足
            cons = max(abs(h["net"] * s0.MU - (r["U"][s0.UNITS.index(h["unit"])] - SF[s0.UNITS.index(h["unit"])])) for h in hhrows)
            out_runs.append(dict(meta, frozen=d1, s8=d8, cons=cons, n_moves=d8["n_moves"]))
            out_hh += hhrows
    return out_runs, out_hh


# ------------------------------------------------------------------ 観察 2
def obs2_conds():
    want = {("T", "幾何 ν=0.00"), ("T", "幾何 ν=0.10"), ("T", "幾何 ν=0.30"),
            ("R", "R∅ 費用あり ν=0.30"), ("R", "R0 費用あり ν=0.30"), ("R", "Rdk10 費用あり ν=0.30"),
            ("W", "個人・取引 ν=0.10"), ("W", "個人・取引 ν=0.30")}
    regs = {"T": ("TP+top3", "改良版top3"), "R": ("U", "TP+top3", "改良版top3"), "W": ("TP+top3", "改良版top3")}
    return [(g, c, rg, sp) for (g, c, rg, sp) in S3.conds() if (g, c) in want and rg in regs[g]]


def s8_rec(a):
    """s3_rent.one と同じ計算を、s8.run で行う"""
    grp, cname, reg, spec, pat, cs = a
    W = composition(pat, cs)
    rr, kw0 = S3.REG[reg]
    kw = dict(kw0); kw.update(S3.kwargs_of(spec, W, pat))
    r = s8.run(W, rr, cs, **kw)
    rec = vstar.record(r, W, None, S3.F0)
    ex = r["extra"]
    U = r["U"]; SF = s0.S_all(r["W_final"], *S3.F0)
    stay = ~r["moved"]
    rec.update(grp=grp, cond=cname, reg=reg, pat=pat, cs=cs, exit_rate=r["n_exit"] / 270,
               L_stay=float(np.mean(U[stay] < SF[stay] - 1e-9)) if stay.any() else float("nan"),
               L_new=float(np.mean(U[~stay] < SF[~stay] - 1e-9)) if (~stay).any() else float("nan"),
               changed=int(sum(1 for p in range(21) if rec["X"][p] != vstar.cfgvec(*r["ref"])[p])),
               owner_in=ex["owner_in"], refund=ex["refund"], restored=ex["restored"], tcost=ex["tcost"],
               unreached=ex["unreached"], recon_paid=sum(1 for x in r["recon"] if x[3] == "paid"),
               recon_revert=sum(1 for x in r["recon"] if x[3] == "revert"))
    return rec, r


def obs2_task(a):
    grp, cname, reg, spec, pat, cs = a
    rec_frozen = S3.one(a)
    rec8, r = s8_rec(a)
    meta = dict(src="obs2", cond=f"{grp}|{cname}", reg=reg, pat=pat, cs=cs, seed=cs)
    hhrows = households(r, pat, meta)
    ex = dict(meta, ledger=r["ledger"], hh={str(h): {k_: v_ for k_, v_ in v.items()} for h, v in r["hh"].items()},
              events=[dict(t=ev["t"], i=ev.get("i"), part=ev.get("part"), nv=ev.get("nv"), ok=ev.get("ok"), drafter=ev.get("drafter"),
                           dS=[round(float(x) / s0.MU, 4) for x in ev["dS"]] if "dS" in ev else None) for ev in r["events"]],
              turns=r["turns"], recon=r["recon"], owner_pl=r["owner_pl"]) if pat == "α=3" else None
    return dict(meta, frozen=rec_frozen, s8=rec8, n_moves=rec8["moves"]), hhrows, ex


def cmp(a, b):
    """二つの記録の全項目が一致するか（数値は 1e-9、nan は nan と一致）"""
    bad = []
    for k in a:
        if k not in b:
            bad.append(k); continue
        x, y = a[k], b[k]
        if isinstance(x, (list, tuple)):
            if np.asarray(x, float).shape != np.asarray(y, float).shape or not np.allclose(np.asarray(x, float), np.asarray(y, float), equal_nan=True, atol=TOL):
                bad.append(k)
        elif isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool):
            if not ((x != x and y != y) or abs(float(x) - float(y)) <= TOL * max(1.0, abs(float(x)))):
                bad.append(k)
        elif x != y:
            bad.append(k)
    return bad


def load_frozen():
    f1 = {}
    with open("out/s1/main.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["cond"] == "主条件" and row["pattern"] in PATS and row["rule"] in OBS1_RULES:
                f1[(row["pattern"], int(row["cseed"]), row["rule"], int(row["seed"]))] = row
    f2 = {}
    with gzip.open("out/s3/main.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            f2[(r["grp"], r["cond"], r["reg"], r["pat"], r["cs"])] = r
    return f1, f2


def check1(row, d):
    """観察 1：main.csv の行と、再実行の値を照合"""
    bad = []
    for k in ["W", "WF", "Wstar", "eta", "L", "B", "minS", "n_moves", "n_fail"]:
        x = float(row[k]); y = float(d[k])
        if not ((x != x and y != y) or abs(x - y) <= 1e-9 * max(1, abs(x)) + 1e-12):
            bad.append(k)
    if row["cfg"] != d["cfg"]:
        bad.append("cfg")
    return bad


def run(which):
    os.makedirs(OUT, exist_ok=True)
    f1, f2 = load_frozen()
    t0 = time.time()
    runs_out = gzip.open(f"{OUT}/runs_{which}.csv.gz", "wt", encoding="utf-8", newline="")
    hh_out = gzip.open(f"{OUT}/households_{which}.csv.gz", "wt", encoding="utf-8", newline="")
    rw = hw = None
    V = dict(n=0, frozen_ok=0, s8_ok=0, bad_frozen=[], bad_s8=[], cons_max=0.0)
    exs = []
    with Pool(2) as pool:
        if which == "obs1":
            it = pool.imap_unordered(obs1_task, [(p, c) for p in PATS for c in S1.C_MAIN])
            for n, (runs, hh) in enumerate(it):
                for x in runs:
                    key = (x["pat"], x["cs"], x["reg"], x["seed"])
                    b1 = check1(f1[key], x["frozen"]); b2 = check1(f1[key], x["s8"])
                    V["n"] += 1; V["frozen_ok"] += not b1; V["s8_ok"] += not b2
                    if b1: V["bad_frozen"].append([list(key), b1])
                    if b2: V["bad_s8"].append([list(key), b2])
                    V["cons_max"] = max(V["cons_max"], x["cons"])
                    row = dict(src="obs1", cond="主条件", reg=x["reg"], pat=x["pat"], cs=x["cs"], seed=x["seed"],
                               eta=x["s8"]["eta"], L=x["s8"]["L"], B=x["s8"]["B"], n_moves=x["n_moves"], ok_frozen=int(not b1), ok_s8=int(not b2))
                    if rw is None:
                        rw = csv.DictWriter(runs_out, fieldnames=list(row)); rw.writeheader()
                    rw.writerow(row)
                for h in hh:
                    if hw is None:
                        hw = csv.DictWriter(hh_out, fieldnames=list(h)); hw.writeheader()
                    hw.writerow(h)
                if n % 10 == 0:
                    print("obs1", n, 60, f"{time.time()-t0:.0f}s", flush=True)
        else:
            T = [(g, c, rg, sp, p, cs) for (g, c, rg, sp) in obs2_conds() for p in PATS for cs in range(400, 460)]
            print(len(T), "局", flush=True)
            for n, (x, hh, ex) in enumerate(pool.imap_unordered(obs2_task, T, chunksize=4)):
                key = (x["cond"].split("|")[0], x["cond"].split("|", 1)[1], x["reg"], x["pat"], x["cs"])
                fr = f2[key]
                b1 = cmp(fr, x["frozen"]); b2 = cmp(fr, x["s8"])
                V["n"] += 1; V["frozen_ok"] += not b1; V["s8_ok"] += not b2
                if b1: V["bad_frozen"].append([list(key), b1])
                if b2: V["bad_s8"].append([list(key), b2])
                row = dict(src="obs2", cond=x["cond"], reg=x["reg"], pat=x["pat"], cs=x["cs"], seed=x["cs"],
                           eta=x["s8"]["eta"], L=x["s8"]["L"], B=x["s8"]["B"], n_moves=x["n_moves"], ok_frozen=int(not b1), ok_s8=int(not b2),
                           exit_rate=x["s8"]["exit_rate"], spent=x["s8"]["spent"], owner_in=x["s8"]["owner_in"], refund=x["s8"]["refund"])
                if rw is None:
                    rw = csv.DictWriter(runs_out, fieldnames=list(row)); rw.writeheader()
                rw.writerow(row)
                for h in hh:
                    if hw is None:
                        hw = csv.DictWriter(hh_out, fieldnames=list(h)); hw.writeheader()
                    hw.writerow(h)
                if ex is not None:
                    exs.append(ex)
                if n % 200 == 0:
                    print("obs2", n, len(T), f"{time.time()-t0:.0f}s", flush=True)
    runs_out.close(); hh_out.close()
    if exs:
        with gzip.open(f"{OUT}/ledger_examples_{which}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(exs, f, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    V["sec"] = time.time() - t0
    json.dump(V, open(f"{OUT}/verify_{which}.json", "w"), ensure_ascii=False, indent=1)
    print(which, "照合", V["n"], "局：凍結のパイプライン一致", V["frozen_ok"], "／ s8 一致", V["s8_ok"], "整合の最大の差", V["cons_max"], flush=True)


if __name__ == "__main__":
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    for w in (["obs1", "obs2"] if which == "all" else [which]):
        run(w)
