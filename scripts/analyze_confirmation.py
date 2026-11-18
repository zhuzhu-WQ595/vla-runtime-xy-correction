"""analyze_confirmation.py — 用冻结参数评价全部规则（不重新拟合任何参数）。

用法：
  python analyze_confirmation.py dev       # 开发集自检：复现登记的开发集干预数
  python analyze_confirmation.py confirm   # init30-49 独立确认
参数来源：runs/predictability/confirm_config.json + confirm_addendum_2026-11-15.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import audit_predictability as au  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
R = BASE / "results"
PR = BASE / "runs/predictability"
CFG = json.load(open(PR / "confirm_config.json"))
ADD = json.load(open(PR / "confirm_addendum_2026-11-15.json"))
CONDS = ["orig", "trans", "rot", "trans_y"]

PC = CFG["primary_rule"]
M_FIT, B_FIT, TH_C = np.array(PC["M_m_per_unit"]), np.array(PC["b_m"]), PC["theta_C_m"]
CMP = CFG["comparators"]
M_NOM, B_NOM = 0.0128 * np.eye(2), np.zeros(2)
TH_NOM, TH_NOM_M = CMP["C_nom"]["theta_C_m"], CMP["C_nom_matched"]["theta_C_m"]
BF = CMP["B_fixed"]
BM = CMP["B_matched"]
TH_BF = (BF["theta_B1_m"], BF["theta_B2"], BF["theta_B3_m"])
TH_BM = (BM["theta_B1_m"], BM["theta_B2"], BM["theta_B3_m"])
TH_S = 0.010196
assert "0.010196" in ADD["secondary_rule_short_axis"]["rule"]

RULES = ["always_a0", "always_a1", "C_fit", "C_nom", "C_nom_matched",
         "B_fixed", "B_matched", "S_short"]


def arr(d, k):
    return np.asarray(d[k].tolist(), dtype=float)


def load(tag, arm):
    p = R / f"traj2_{tag}_{arm}.npz"
    if not p.exists():
        return None, None
    d = np.load(p, allow_pickle=True)
    return d, json.loads(str(d["meta"]))


def build(inits, scan_file):
    scan = {r["tag"]: r for r in csv.DictReader(open(R / scan_file))}
    S, issues = [], []
    for i in inits:
        for c in CONDS:
            tag = f"init{i:02d}_{c}"
            dg, mg = load(tag, "gate")
            s = {"tag": tag, "init": i, "cond": c, "gate_steps": mg["steps"]}
            ts = scan[tag]["t_star"]
            s["triggered"] = ts not in (None, "", "None")
            if not s["triggered"]:
                s.update(a0_g=mg["grasp_success"], a0_t=mg["task_success"],
                         a1_g=mg["grasp_success"], a1_t=mg["task_success"],
                         label_g="no_trigger", label_t="no_trigger")
                S.append(s)
                continue
            ts = int(float(ts))
            ip, ii = ts - 2, ts - 1
            d0, m0 = load(tag, "branch_a0_single")
            d1, m1 = load(tag, "branch_a1_single")
            if d0 is None or d1 is None:
                issues.append((tag, "missing branch run"))
                s["label_g"] = "MISSING"
                S.append(s)
                continue
            eg = arr(dg, "exec_action")
            e0, e1 = arr(d0, "exec_action"), arr(d1, "exec_action")
            s["chk_a0_identical"] = bool(len(eg) == len(e0) and np.allclose(eg, e0, atol=1e-8))
            s["chk_a1_prefix"] = bool(np.allclose(e1[:ii], eg[:ii], atol=1e-8))
            s["chk_a1_tug_at_tstar"] = bool(
                np.allclose(e1[ii, :2], np.array(eval(scan[tag]["prop_xy"])), atol=1e-6))
            if not (s["chk_a0_identical"] and s["chk_a1_prefix"] and s["chk_a1_tug_at_tstar"]):
                issues.append((tag, f"check failed a0={s['chk_a0_identical']} "
                                    f"prefix={s['chk_a1_prefix']} tug={s['chk_a1_tug_at_tstar']}"))
            eef, tgt = arr(dg, "eef_pos"), arr(dg, "tgt_pos")
            raw = arr(dg, "raw_action")
            tq = arr(d0, "tgt_quat")
            bx, by = au.body_axes(tq[ip])
            e = eef[ip, :2] - tgt[ip, :2]
            prop = np.array(eval(scan[tag]["prop_xy"]))
            ehist = eef[:, :2] - tgt[:, :2]
            ebx = np.abs(ehist @ bx)
            d = prop - raw[ii, :2]
            s.update({
                "t_star": ts, "e_xy": e, "raw_xy": raw[ii, :2], "prop_xy": prop,
                "bx": bx, "by": by,
                "e_bx_mm": float(e @ bx) * 1e3, "e_by_mm": float(e @ by) * 1e3,
                "abs_e_s": abs(float(e @ bx)),
                "trend4": float(ebx[ip] - ebx[ip - 4]) if ip >= 4 else 0.0,
                "cos_d": float(d @ -e / (np.linalg.norm(d) * np.linalg.norm(e) + 1e-12)),
                "a0_g": m0["grasp_success"], "a0_t": m0["task_success"],
                "a1_g": m1["grasp_success"], "a1_t": m1["task_success"]})
            s["s_fit"] = au.pred_imp(s, M_FIT, B_FIT, "eef")
            s["s_nom"] = au.pred_imp(s, M_NOM, B_NOM, "eef")
            for k, (a, b) in (("label_g", (s["a0_g"], s["a1_g"])),
                              ("label_t", (s["a0_t"], s["a1_t"]))):
                s[k] = ("rescue" if (a, b) == (False, True) else
                        "break" if (a, b) == (True, False) else
                        "both_success" if a else "both_fail")
            S.append(s)
    return S, issues


def decide(s):
    if not s["triggered"] or s.get("label_g") == "MISSING":
        return {r: False for r in RULES}
    return {"always_a0": False, "always_a1": True,
            "C_fit": s["s_fit"] > TH_C,
            "C_nom": s["s_nom"] > TH_NOM,
            "C_nom_matched": s["s_nom"] > TH_NOM_M,
            "B_fixed": au.dec_B(s, TH_BF),
            "B_matched": au.dec_B(s, TH_BM),
            "S_short": abs(s["e_by_mm"]) / 1e3 > TH_S}


def outcome(s, tug):
    if tug:
        return s["a1_g"], s["a1_t"]
    return s["a0_g"], s["a0_t"]


def tally(S, rule, cond=None):
    rows = [s for s in S if cond is None or s["cond"] == cond]
    t = {"N": len(rows), "trig": sum(s["triggered"] for s in rows),
         "int": 0, "rescue_g": 0, "break_g": 0, "rescue_t": 0, "break_t": 0,
         "grasp": 0, "task": 0}
    for s in rows:
        tug = s["dec"][rule]
        t["int"] += tug
        if tug:
            t["rescue_g"] += s["label_g"] == "rescue"
            t["break_g"] += s["label_g"] == "break"
            t["rescue_t"] += s["label_t"] == "rescue"
            t["break_t"] += s["label_t"] == "break"
        g, k = outcome(s, tug)
        t["grasp"] += g
        t["task"] += k
    return t


def cluster_boot(S, ra, rb, key, n=10000, seed=20261115):
    inits = sorted({s["init"] for s in S})
    per = {i: 0 for i in inits}
    for s in S:
        ga, ta = outcome(s, s["dec"][ra])
        gb, tb = outcome(s, s["dec"][rb])
        per[s["init"]] += (ga - gb) if key == "grasp" else (ta - tb)
    v = np.array([per[i] for i in inits], dtype=float)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n, len(v)))].sum(1)
    return {"diff": float(v.sum()), "ci95": [float(np.percentile(bs, 2.5)),
                                             float(np.percentile(bs, 97.5))],
            "nonzero_clusters": int((v != 0).sum())}


def main():
    mode = sys.argv[1]
    if mode == "dev":
        S, issues = build(range(10, 30), "trigger_scan.csv")
    else:
        S, issues = build(range(30, 50), "confirm_trigger_scan.csv")
    for s in S:
        s["dec"] = decide(s)
    print(f"[{mode}] states={len(S)} triggered={sum(s['triggered'] for s in S)} issues={issues}")
    summ = {"mode": mode, "issues": issues, "rules": {}, "by_cond": {}}
    print(f"{'rule':14s} trig int | g:res brk | t:res brk | grasp task")
    for r in RULES:
        t = tally(S, r)
        summ["rules"][r] = t
        summ["by_cond"][r] = {c: tally(S, r, c) for c in CONDS}
        print(f"{r:14s} {t['trig']:4d} {t['int']:3d} | {t['rescue_g']:3d} {t['break_g']:3d} | "
              f"{t['rescue_t']:3d} {t['break_t']:3d} | {t['grasp']:3d}/{t['N']} {t['task']:3d}/{t['N']}")
    if mode == "dev":
        return

    print("\n按条件（干预/救回g/破坏g/抓取/入篮）")
    for r in RULES:
        print(f"{r:14s}", "  ".join(
            f"{c}:{summ['by_cond'][r][c]['int']}/{summ['by_cond'][r][c]['rescue_g']}/"
            f"{summ['by_cond'][r][c]['break_g']}/{summ['by_cond'][r][c]['grasp']}/"
            f"{summ['by_cond'][r][c]['task']}" for c in CONDS))
    T = [s for s in S if s["triggered"]]
    print("触发覆盖率:", {c: f"{sum(s['triggered'] for s in S if s['cond']==c)}/20" for c in CONDS})

    # C_fit vs 全牵
    av = [s["tag"] for s in T if s["label_g"] == "break" and not s["dec"]["C_fit"]]
    ms = [s["tag"] for s in T if s["label_g"] == "rescue" and not s["dec"]["C_fit"]]
    avt = [s["tag"] for s in T if s["label_t"] == "break" and not s["dec"]["C_fit"]]
    mst = [s["tag"] for s in T if s["label_t"] == "rescue" and not s["dec"]["C_fit"]]
    print(f"\nC_fit vs 全牵: 抓取 避免破坏={av} 错过救回={ms}")
    print(f"               入篮 避免破坏={avt} 错过救回={mst}")
    summ["C_fit_vs_a1"] = {"avoided_break_grasp": av, "missed_rescue_grasp": ms,
                           "avoided_break_task": avt, "missed_rescue_task": mst}
    # 分歧
    summ["disagree"] = {}
    for other in ("S_short", "B_fixed", "B_matched", "C_nom", "C_nom_matched", "always_a1"):
        dis = [s for s in T if s["dec"]["C_fit"] != s["dec"][other]]
        eg = [(s["tag"], s["label_g"], "C_fit牵" if s["dec"]["C_fit"] else f"{other}牵")
              for s in dis if s["label_g"] in ("rescue", "break")]
        et = [(s["tag"], s["label_t"], "C_fit牵" if s["dec"]["C_fit"] else f"{other}牵")
              for s in dis if s["label_t"] in ("rescue", "break")]
        summ["disagree"][other] = {"n": len(dis), "grasp_changing": eg, "task_changing": et}
        print(f"C_fit vs {other:13s}: 决策不同 {len(dis):2d}；改变抓取 {eg}；改变入篮 {et}")
    # bootstrap
    summ["boot"] = {}
    for a, b in (("C_fit", "always_a1"), ("C_fit", "always_a0"), ("always_a1", "always_a0"),
                 ("C_fit", "S_short"), ("C_fit", "B_fixed"), ("C_fit", "B_matched"),
                 ("C_fit", "C_nom"), ("C_fit", "C_nom_matched")):
        for key in ("grasp", "task"):
            bres = cluster_boot(S, a, b, key)
            summ["boot"][f"{a}-{b}:{key}"] = bres
            print(f"boot {a}-{b} {key}: diff={bres['diff']:+.0f} CI95={bres['ci95']} "
                  f"nonzero_inits={bres['nonzero_clusters']}")
    # H1
    h1 = summ["rules"]["C_fit"]["break_g"] <= summ["rules"]["always_a1"]["break_g"]
    print("H1（设计性质检查）:", h1)
    summ["H1_design_check"] = bool(h1)
    # 写表
    cols = ["tag", "init", "cond", "triggered", "t_star", "gate_steps", "a0_g", "a1_g", "a0_t",
            "a1_t", "label_g", "label_t", "e_bx_mm", "e_by_mm", "abs_e_s", "trend4", "cos_d",
            "s_fit", "s_nom", "chk_a0_identical", "chk_a1_prefix", "chk_a1_tug_at_tstar"]
    with open(R / "confirm_states.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + RULES)
        w.writeheader()
        for s in S:
            row = {c: s.get(c, "") for c in cols}
            for c in ("abs_e_s", "trend4", "s_fit", "s_nom"):
                if row[c] != "":
                    row[c] = round(row[c] * 1e3, 3)
            for c in ("e_bx_mm", "e_by_mm", "cos_d"):
                if row[c] != "":
                    row[c] = round(row[c], 3)
            for r in RULES:
                row[r] = s["dec"][r] if s["triggered"] else ""
            w.writerow(row)
    json.dump(summ, open(R / "confirm_summary.json", "w"), ensure_ascii=False, indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else
              (bool(o) if isinstance(o, np.bool_) else str(o)))
    print("wrote results/confirm_states.csv, confirm_summary.json")


if __name__ == "__main__":
    main()
