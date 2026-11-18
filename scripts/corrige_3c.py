"""corrige_3c.py — 三 C 离线勘误（只用已有 npz，不跑仿真）。

A) 按条件×终点重算唯一翻转状态数（双向翻转合并去重）。
B) 对翻转病例做同一轨迹同步比较：
   t_div = visgate vs vislock exec_action XY 首个非零差步（raw）；
   t_con = vislock 轨迹首个 ncon_tgt>0 步；
   t_mv  = vislock 轨迹首个 |tgt_xy(t)-tgt_xy(0)|>5mm 步；
   并在 t_div 处比较 |lock_est-gt| vs |live_est-gt|（同步、同轨迹）。

重分类（覆盖 online_vs_lock 的粗略四类）：
 - tracked_confirmed：t_mv 存在且 live 在 t_div 处的误差比 lock 小 ≥3mm
   （逐帧更新在分歧时刻确实更准）
 - knock_following：t_con<t_mv<t_div 模式且物体被夹爪碰移后才分歧
   （逐帧跟的是被撞动的物体，结局差可能发生在抓取成形后）
 - no_quality_edge：t_div 处 live 与 lock 误差差 <3mm（分歧非由估计质量）
 - miss_driven：t_div±5 步内 est_live 出现 NaN/held 切换
 - undetermined：其余
"""
import csv
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
R = BASE / "results"
SETS = {"butter": "fzb", "milk": "fzm"}
RAW2M = 0.0128
TOL = 1e-6
EDGE_MM = 3.0
MOVE_MM = 5.0


def load(pre, tag, arm):
    d = np.load(R / f"{pre}_{tag}_{arm}.npz", allow_pickle=True)
    a = {}
    for k in d.files:
        if k == "meta":
            continue
        try:
            a[k] = np.asarray(d[k], dtype=float)
        except (TypeError, ValueError):
            a[k] = np.asarray(d[k])
    return a, json.loads(str(d["meta"]))


def analyze(sname, tag):
    ax, _ = load(SETS[sname], tag, "visgate")
    al, ml = load(SETS[sname], tag, "vislock")
    n = min(len(ax["exec_action"]), len(al["exec_action"]))
    dxy = np.linalg.norm(ax["exec_action"][:n, :2] - al["exec_action"][:n, :2],
                         axis=1)
    t_div = next((t for t in range(n) if dxy[t] > TOL), None)
    gt = al["tgt_pos"]
    # ncon_tgt 含桌面接触（物体静置桌上恒>0），不能作"首次接触"；
    # 用指尖中点与目标的 3D 距离 <25mm 作夹爪-物体接触代理
    fm = al["finger_mid"]
    d_fg = np.linalg.norm(fm - gt, axis=1)
    t_con = next((t for t in range(len(d_fg)) if d_fg[t] < 0.025), None)
    d0 = np.linalg.norm(gt[:, :2] - gt[0, :2], axis=1)
    t_mv = next((t for t in range(len(d0)) if d0[t] * 1000 > MOVE_MM), None)
    lock = np.asarray(ml["locked_est"]) if ml.get("locked_est") else None
    live = al["est_live"]                    # 同轨迹逐帧原始估计
    # 同步误差序列（est_live 有 NaN 的步跳过）
    lock_err = (np.linalg.norm(gt[:, :2] - lock[:2], axis=1) * 1000
                if lock is not None else np.full(len(gt), np.nan))
    live_ok = ~np.isnan(live[:, 0])
    live_err = np.full(len(gt), np.nan)
    live_err[live_ok] = np.linalg.norm(
        live[live_ok, :2] - gt[live_ok, :2], axis=1) * 1000
    w = slice(max(0, (t_div or 0) - 5), min(n, (t_div or 0) + 6))
    edge = None
    if t_div is not None and np.isfinite(live_err[w]).any():
        edge = float(np.nanmedian(live_err[w]) - np.nanmedian(lock_err[w]))
    # est_live 在分歧窗口有 NaN（漏检切换）
    miss_near = bool((~live_ok[w]).any())
    ev = dict(t_div=t_div, t_con=t_con, t_mv=t_mv,
              obj_max_mm=round(float(d0.max()) * 1000, 1),
              lock_err_at_div=None if t_div is None or not np.isfinite(lock_err[t_div])
              else round(float(lock_err[t_div]), 1),
              live_err_at_div=None if (t_div is None or t_div >= len(live_err)
                                       or not np.isfinite(live_err[t_div]))
              else round(float(live_err[t_div]), 1),
              edge_mm=None if edge is None else round(edge, 1),
              miss_near_div=miss_near,
              order=("div<con<mv" if (t_div is not None and t_con is not None
                     and t_mv is not None and t_div < t_con < t_mv) else
                     "con<mv<div" if (t_con is not None and t_mv is not None
                     and t_div is not None and t_con < t_mv < t_div) else
                     "other"))
    # 重分类
    if t_div is None:
        cat = "undetermined"
        ev["note"] = "exec identical? unexpected"
    elif miss_near:
        cat = "miss_driven"
    elif edge is not None and edge <= -EDGE_MM:
        if t_con is not None and t_mv is not None and t_con < t_mv <= t_div:
            cat = "knock_following"
        else:
            cat = "tracked_confirmed"
    elif edge is not None and abs(edge) < EDGE_MM:
        cat = "no_quality_edge"
    elif t_con is not None and t_mv is not None and t_con <= t_mv < t_div:
        cat = "knock_following"
    else:
        cat = "undetermined"
        ev["note"] = f"edge={edge}"
    return cat, ev


def main():
    out = {}
    print("===== A) 按条件×终点唯一翻转状态数（双向合并去重）=====")
    for s in SETS:
        rows = list(csv.DictReader(open(R / f"online_vs_lock_{s}.csv")))
        out[s] = dict(by={}, cases={})
        for c in ("orig", "trans", "trans_y"):
            for e in ("g", "t"):
                fs = sorted({r["tag"] for r in rows
                             if r["cond"] == c and r[f"flip_{e}"] in ("rescue", "break")})
                union = sorted({r["tag"] for r in rows if r["cond"] == c
                                and (r["flip_g"] in ("rescue", "break")
                                     or r["flip_t"] in ("rescue", "break"))})
                out[s]["by"][f"{c}:{e}"] = len(fs)
                print(f"  {s} {c:8s} {e}: {len(fs)} 状态 {fs}")
            print(f"    {c} 合并去重: {len(union)} 状态 {union}")
        print(f"\n===== B) {s} 同步误差/时序重分类 =====")
        flips = sorted({r["tag"] for r in rows
                        if r["flip_g"] in ("rescue", "break")
                        or r["flip_t"] in ("rescue", "break")})
        for tag in flips:
            cat, ev = analyze(s, tag)
            r0 = next(r for r in rows if r["tag"] == tag)
            out[s]["cases"][tag] = dict(cat=cat, flip_g=r0["flip_g"],
                                        flip_t=r0["flip_t"], **ev)
            print(f"  {tag:20s} {cat:16s} div={ev['t_div']} con={ev['t_con']} "
                  f"mv={ev['t_mv']} max={ev['obj_max_mm']}mm "
                  f"lock@{ev['lock_err_at_div']} live@{ev['live_err_at_div']} "
                  f"edge={ev['edge_mm']} {ev['order']}")
    json.dump(out, open(R / "corrige_3c.json", "w"), ensure_ascii=False, indent=1)
    print("\nwrote results/corrige_3c.json")


if __name__ == "__main__":
    main()
