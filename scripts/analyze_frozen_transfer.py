"""analyze_frozen_transfer.py — 阶段三 B 冻结三臂配对分析（规则全部取自冻结清单）。

输出：
  results/frozen_transfer_{butter,milk}.csv   逐状态配对
  results/frozen_transfer_summary.json        分物体/条件/终点汇总 + init 簇 bootstrap
用法：python analyze_frozen_transfer.py [--partial]   (--partial 允许缺回合，仅用于进度检查)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
R = BASE / "results"
MAN = json.load(open(BASE / "runs/predictability/frozen_visual_transfer_2026-11-16.json"))
CONDS = MAN["conditions"]
ARMS = {"v": "vanilla", "g": "gate", "x": "visgate"}
EPS = {"g": "grasp_success", "t": "task_success"}
N_BOOT, SEED = 2000, 0


def load(pre, tag, arm):
    p = R / f"{pre}_{tag}_{arm}.npz"
    if not p.exists():
        return None, None
    d = np.load(p, allow_pickle=True)
    a = {}
    for k in d.files:
        if k == "meta":
            continue
        try:
            a[k] = np.asarray(d[k], dtype=float)
        except (TypeError, ValueError):
            a[k] = np.asarray(d[k])
    return a, json.loads(str(d["meta"]))


def lab(a, b):
    return ("rescue" if (not a and b) else "break" if (a and not b)
            else "both_success" if a else "both_fail")


def vis_stats(ax):
    """visgate 抬起前（目标 z 未离开静置高度 5mm）的漏检率与 XY 误差中位数。"""
    gt, est = ax["tgt_pos"], ax["est_tgt"]
    miss = ax["vis_miss"].astype(int)
    pre = gt[:, 2] < gt[0, 2] + 0.005
    ok = pre & (miss == 0)
    exy = np.linalg.norm(est[ok, :2] - gt[ok, :2], axis=1)
    return (round(float(miss[pre].mean()), 3) if pre.any() else None,
            round(1000 * float(np.median(exy)), 2) if ok.any() else None)


def collect(sname, cfg, partial):
    rows, missing = [], []
    for i in cfg["inits"]:
        for c in CONDS:
            tag = f"init{i:02d}_{c}"
            got = {k: load(cfg["out_prefix"], tag, a) for k, a in ARMS.items()}
            if any(m is None for _, m in got.values()):
                missing.append(tag)
                continue
            r = dict(tag=tag, init=i, cond=c,
                     pos_override=json.dumps(got["v"][1].get("pos_override")),
                     settled_xy=json.dumps([round(float(z), 4) for z in
                                            got["v"][1]["settled_pos"][:2]]))
            for k in ARMS:
                for e, key in EPS.items():
                    r[f"{k}_{e}"] = bool(got[k][1][key])
            for e in EPS:
                r[f"xv_{e}"] = lab(r[f"v_{e}"], r[f"x_{e}"])
                r[f"gv_{e}"] = lab(r[f"v_{e}"], r[f"g_{e}"])
                r[f"xg_{e}"] = lab(r[f"g_{e}"], r[f"x_{e}"])
            r["g_trig"] = bool((got["g"][0]["corr_active"] > 0).any())
            r["x_trig"] = bool((got["x"][0]["corr_active"] > 0).any())
            r["x_miss_prelift"], r["x_exy_mm_prelift"] = vis_stats(got["x"][0])
            rows.append(r)
    if missing and not partial:
        raise SystemExit(f"{sname}: 缺 {len(missing)} 个状态 {missing[:6]}…；用 --partial 仅查进度")
    return rows, missing


def tally(rs, e):
    n = len(rs)
    v = sum(r[f"v_{e}"] for r in rs); g = sum(r[f"g_{e}"] for r in rs)
    x = sum(r[f"x_{e}"] for r in rs)
    o_res = [r["tag"] for r in rs if r[f"gv_{e}"] == "rescue"]
    kept = [t for t in o_res if next(r for r in rs if r["tag"] == t)[f"x_{e}"]]
    return dict(
        n=n, vanilla=v, oracle=g, visual=x,
        x_rescue=[r["tag"] for r in rs if r[f"xv_{e}"] == "rescue"],
        x_break=[r["tag"] for r in rs if r[f"xv_{e}"] == "break"],
        g_rescue=o_res,
        g_break=[r["tag"] for r in rs if r[f"gv_{e}"] == "break"],
        rescue_denom=n - v, break_denom=v,
        oracle_kept=kept,
        oracle_lost=[t for t in o_res if t not in kept],
        visual_only_rescue=[r["tag"] for r in rs
                            if r[f"xv_{e}"] == "rescue" and r[f"gv_{e}"] != "rescue"],
        oracle_retention=(round(len(kept) / len(o_res), 3) if o_res else None))


def cluster_ci(rs, a, b, e, rng):
    by = {}
    for r in rs:
        by.setdefault(r["init"], []).append(int(r[f"{a}_{e}"]) - int(r[f"{b}_{e}"]))
    keys = sorted(by)
    if not keys:
        return None
    cm = np.array([np.mean(by[k]) for k in keys])
    pt = float(cm.mean())
    bs = [cm[rng.integers(0, len(cm), len(cm))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [round(pt, 3), round(float(lo), 3), round(float(hi), 3)]


def main():
    partial = "--partial" in sys.argv
    summary = {"manifest_frozen_at": MAN["frozen_at"]}
    for sname, cfg in MAN["sets"].items():
        rows, missing = collect(sname, cfg, partial)
        rng = np.random.default_rng(SEED)
        out = dict(label=cfg["label"], n_states=len(rows), missing=missing, by=dict())
        for scope in CONDS + ["trans+trans_y", "ALL"]:
            rs = (rows if scope == "ALL" else
                  [r for r in rows if r["cond"] in ("trans", "trans_y")]
                  if scope == "trans+trans_y" else
                  [r for r in rows if r["cond"] == scope])
            out["by"][scope] = {}
            for e in EPS:
                t = tally(rs, e)
                t["ci_x_minus_v"] = cluster_ci(rs, "x", "v", e, rng)
                t["ci_x_minus_g"] = cluster_ci(rs, "x", "g", e, rng)
                t["ci_g_minus_v"] = cluster_ci(rs, "g", "v", e, rng)
                out["by"][scope][e] = t
        ms = [r["x_miss_prelift"] for r in rows if r["x_miss_prelift"] is not None]
        es = [r["x_exy_mm_prelift"] for r in rows if r["x_exy_mm_prelift"] is not None]
        out["vision"] = dict(miss_prelift_mean=round(float(np.mean(ms)), 3) if ms else None,
                             exy_mm_prelift_median=round(float(np.median(es)), 2) if es else None)
        summary[sname] = out
        if rows:
            with open(R / f"frozen_transfer_{sname}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

        print(f"\n===== {sname}  states={len(rows)} missing={len(missing)}")
        print(f"{'scope':14s} ep  N | v  g  x | x救 x破 | g救 g破 | oracle保留 | Δx-v CI | Δx-g CI")
        for scope, d in out["by"].items():
            for e, t in d.items():
                print(f"{scope:14s} {e}  {t['n']:2d} | {t['vanilla']:2d} {t['oracle']:2d} "
                      f"{t['visual']:2d} | {len(t['x_rescue']):2d}/{t['rescue_denom']:<2d} "
                      f"{len(t['x_break']):2d}/{t['break_denom']:<2d} | "
                      f"{len(t['g_rescue']):2d} {len(t['g_break']):2d} | "
                      f"{len(t['oracle_kept'])}/{len(t['g_rescue'])} | "
                      f"{t['ci_x_minus_v']} | {t['ci_x_minus_g']}")
        print("vision:", out["vision"])
    if not partial:
        json.dump(summary, open(R / "frozen_transfer_summary.json", "w"),
                  ensure_ascii=False, indent=1)
        print("\nwrote results/frozen_transfer_summary.json")


if __name__ == "__main__":
    main()
