"""analyze_online_vs_lock.py — 阶段三 C：visgate(逐帧) vs vislock(初始锁定) 机制对照。

配对规则与诊断类别均按 runs/predictability/vislock_spec_2026-11-17.json。
主分析：两物体各自 trans+trans_y；orig 单列；两臂按同一 (init,cond) 配对。
翻转病例用离线真值归因：tracked / drift / corrected / undetermined。

输出：results/online_vs_lock_{set}.csv、results/online_vs_lock_summary.json
用法：python analyze_online_vs_lock.py [--partial]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
R = BASE / "results"
SETS = {"butter": dict(pre="fzb", inits=range(30, 50)),
        "milk": dict(pre="fzm", inits=range(20, 50))}
CONDS = ["orig", "trans", "trans_y"]
MAIN_CONDS = ("trans", "trans_y")
EPS = {"g": "grasp_success", "t": "task_success"}
N_BOOT, SEED = 2000, 0
# 诊断阈值（spec 预写）
MOVE_MM, DRIFT_MM, BIAS_MM = 5.0, 5.0, 8.0


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


def cluster_ci(rs, a, b, e, rng):
    by = {}
    for r in rs:
        by.setdefault(r["init"], []).append(int(r[f"{a}_{e}"]) - int(r[f"{b}_{e}"]))
    if not by:
        return None
    cm = np.array([np.mean(v) for _, v in sorted(by.items())])
    pt = float(cm.mean())
    bs = [cm[rng.integers(0, len(cm), len(cm))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [round(pt, 3), round(float(lo), 3), round(float(hi), 3)]


def diagnose(al, ml, ax, mx):
    """翻转病例离线归因（真值仅诊断用）。

    al=vislock 轨迹, ax=visgate 轨迹。返回 (类别, 证据dict)。"""
    gt = al["tgt_pos"]                      # 真值（两臂同一状态，真值相同）
    lock = np.asarray(ml["locked_est"]) if ml.get("locked_est") else None
    ls = ml.get("lock_step")
    live = ax["est_tgt"]                    # visgate 喂给 corr 的逐帧目标
    live_raw = al["est_live"]               # vislock 记录的逐帧原始估计（诊断）
    n = min(len(gt), len(live))
    if n == 0:
        return "undetermined", dict(note="empty traj")
    obj_move = float(np.nanmax(
        np.linalg.norm(gt[:n, :2] - gt[0, :2], axis=1))) * 1000
    ok = ~np.isnan(live_raw[:n, 0]) if "est_live" in al else np.zeros(n, bool)
    # lock 与逐帧估计的偏差
    live_err = np.full(n, np.nan)         # 全步长度，ok 步填误差
    if lock is not None and ok.any():
        dv = np.linalg.norm(live_raw[:n][ok, :2] - lock[:2], axis=1)
        lock_drift = float(np.max(dv)) * 1000
        lock_err = float(np.linalg.norm(lock[:2] - gt[int(ls or 0), :2])) * 1000
        live_err[ok] = np.linalg.norm(live_raw[:n][ok, :2] - gt[:n][ok, :2], axis=1)
        e_ok = live_err[ok]
        late_err = float(np.median(e_ok[max(0, len(e_ok) // 2):])) * 1000
    else:
        lock_drift = lock_err = late_err = float("nan")
    ev = dict(lock_step=ls, obj_move_mm=round(obj_move, 1),
              lock_err_mm=None if np.isnan(lock_err) else round(lock_err, 1),
              est_vs_lock_mm=None if np.isnan(lock_drift) else round(lock_drift, 1),
              live_late_err_mm=None if np.isnan(late_err) else round(late_err, 1))
    moved = obj_move > MOVE_MM
    drifting = not np.isnan(lock_drift) and lock_drift > DRIFT_MM
    biased = not np.isnan(lock_err) and lock_err > BIAS_MM
    corrected = biased and not np.isnan(late_err) and late_err < BIAS_MM
    if lock is None:
        return "undetermined", {**ev, "note": "no lock (miss all episode)"}
    if moved and drifting:
        # 物体动且估计跟随动：看移动后 est_live 误差是否仍小
        dgt = np.linalg.norm(np.diff(gt[:n, :2], axis=0), axis=1)
        t_mv = int(np.argmax(dgt)) + 1 if len(dgt) else 0
        post_err = live_err[t_mv:]
        if np.isfinite(post_err).any():
            follow = float(np.nanmedian(post_err)) * 1000
            ev["post_move_err_mm"] = round(follow, 1)
            if follow <= 2 * MOVE_MM:
                return "tracked", ev
        return "undetermined", {**ev, "note": "moved+drift, follow unclear"}
    if moved:
        return "tracked", {**ev, "note": "obj moved, live tracked"}
    if drifting:
        return "corrected" if corrected else "drift", ev
    if corrected:
        return "corrected", ev
    return "undetermined", {**ev, "note": "no dominant signal"}


def main():
    partial = "--partial" in sys.argv
    rng = np.random.default_rng(SEED)
    summary = {}
    for sname, cfg in SETS.items():
        rows, missing = [], []
        for i in cfg["inits"]:
            for c in CONDS:
                tag = f"init{i:02d}_{c}"
                ax, mx = load(cfg["pre"], tag, "visgate")
                al, ml = load(cfg["pre"], tag, "vislock")
                if mx is None or ml is None:
                    missing.append(tag)
                    continue
                r = dict(tag=tag, init=i, cond=c,
                         lock_step=ml.get("lock_step"),
                         lock_miss=ml.get("lock_step") is None,
                         diag="", diag_ev="")
                for e, key in EPS.items():
                    r[f"x_{e}"], r[f"l_{e}"] = bool(mx[key]), bool(ml[key])
                    r[f"flip_{e}"] = lab(r[f"l_{e}"], r[f"x_{e}"])
                r["x_trig"] = bool((ax["corr_active"] > 0).any())
                r["l_trig"] = bool((al["corr_active"] > 0).any())
                rows.append(r)
        if missing and not partial:
            raise SystemExit(f"{sname}: 缺 {len(missing)} 状态 {missing[:6]}…；--partial 查进度")

        # 逐状态表 + 翻转归因
        diag = {}
        for r in rows:
            flips = {e for e in EPS if r[f"flip_{e}"] in ("rescue", "break")}
            if not flips:
                continue
            ax, mx = load(cfg["pre"], r["tag"], "visgate")
            al, ml = load(cfg["pre"], r["tag"], "vislock")
            cat, ev = diagnose(al, ml, ax, mx)
            r["diag"], r["diag_ev"] = cat, json.dumps(ev, ensure_ascii=False)
            diag[r["tag"]] = (cat, ev, {e: r[f"flip_{e}"] for e in flips})

        out = dict(n=len(rows), missing=missing, by={}, diag={})
        for scope in ["orig", "trans", "trans_y", "trans+trans_y", "ALL"]:
            rs = (rows if scope == "ALL" else
                  [r for r in rows if r["cond"] in MAIN_CONDS]
                  if scope == "trans+trans_y" else
                  [r for r in rows if r["cond"] == scope])
            for e in EPS:
                x = sum(r[f"x_{e}"] for r in rs)
                l = sum(r[f"l_{e}"] for r in rs)
                out["by"].setdefault(scope, {})[e] = dict(
                    n=len(rs), visgate=x, vislock=l,
                    online_wins=[r["tag"] for r in rs
                                 if r[f"flip_{e}"] == "rescue"],
                    lock_wins=[r["tag"] for r in rs
                               if r[f"flip_{e}"] == "break"],
                    ci_x_minus_l=cluster_ci(rs, "x", "l", e, rng))
        n_lock = sum(not r["lock_miss"] for r in rows)
        ls0 = sum(r["lock_step"] == 0 for r in rows if not r["lock_miss"])
        out["lock"] = dict(acquired=n_lock, at_t0=ls0, never=sum(r["lock_miss"] for r in rows),
                           delayed=[r["tag"] for r in rows
                                    if not r["lock_miss"] and r["lock_step"] > 0])
        out["diag"] = {t: dict(cat=c, **v, flips=f) for t, (c, v, f) in diag.items()}
        summary[sname] = out
        if rows:
            with open(R / f"online_vs_lock_{sname}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

        print(f"\n===== {sname}  states={len(rows)} missing={len(missing)}")
        print(f"lock: 首帧锁定 {out['lock']['at_t0']}, 延迟锁定 "
              f"{len(out['lock']['delayed'])}, 全程未锁定 {out['lock']['never']}")
        print(f"{'scope':14s} ep  N | x  l | 逐帧赢 锁定赢 | Δx-l CI")
        for scope, dd in out["by"].items():
            for e, t in dd.items():
                print(f"{scope:14s} {e}  {t['n']:2d} | {t['visgate']:2d} "
                      f"{t['vislock']:2d} | {len(t['online_wins'])} {len(t['lock_wins'])} "
                      f"| {t['ci_x_minus_l']}")
        print("翻转归因:")
        for t, d in out["diag"].items():
            print(f"  {t:20s} {d['cat']:13s} {d['flips']} ev={d}")
    if not partial:
        json.dump(summary, open(R / "online_vs_lock_summary.json", "w"),
                  ensure_ascii=False, indent=1)
        print("\nwrote results/online_vs_lock_summary.json")


if __name__ == "__main__":
    main()
