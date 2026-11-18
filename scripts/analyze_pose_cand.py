"""analyze_pose_cand.py — 阶段四 A：局部抓取位姿候选配对分析。

按 pose_cand_spec_2026-11-17.json：
- 每状态 5 候选（c0 基线 + px/nx/ry/ly）矩阵：稳定抓取、完整入篮。
- achieved 位姿差：候选与 c0 在同一绝对步的 eef_pos/eef_quat 差
  （在候选 run 的 first_close_t 及偏置期最大值，剔除策略自身运动）。
- Q1 基线失败且≥1候选成功；Q2 基线成功且某候选失败；
  Q3 单一固定候选收益 + 事后择优上限。

输出：results/pose_cand_{set}.csv、results/pose_cand_summary.json
用法：python analyze_pose_cand.py [--partial]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
R = BASE / "results"
CANDS = ["c0", "px", "nx", "ry", "ly"]
SETS = {"butter": dict(pre="f4b",
                     inits=[30, 34, 37, 40, 43, 44, 46]),
        "milk": dict(pre="f4m",
                     inits=[0, 1, 4, 6, 9, 12, 15, 17])}
CONDS = ["trans", "trans_y"]
EPS = {"g": "grasp_success", "t": "task_success"}


def yaw(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def load(pre, tag, cand):
    p = R / f"{pre}_{tag}_pb_{cand}.npz"
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


def achieved(base, cand_a, meta):
    """候选 vs c0 的同绝对步 eef 位姿差：闭合时刻 + 偏置期最大。"""
    if base is None or cand_a is None or meta.get("cand_step") is None:
        return None
    t0 = int(meta["cand_step"])
    eb, ec = base["eef_pos"], cand_a["eef_pos"]
    n = min(len(eb), len(ec))
    d = (ec[:n] - eb[:n]) * 1000               # mm
    u = np.asarray(meta["u_vec"]) if meta.get("u_vec") else None
    du = d[:n, :2] @ u if u is not None else d[:n, :2]
    fc = meta.get("first_close_t")             # 1-based
    tc = min(fc, n - 1) if fc else n - 1
    # 偏置期内（t0..t0+30）沿 u 的最大有符号位移
    seg = du[t0:min(n, t0 + 30)]
    du_seg = (seg[np.argmax(np.abs(seg))] if seg.ndim == 1 and len(seg)
              else None)
    dyaw = None
    if fc:
        qb, qc = base["eef_quat"], cand_a["eef_quat"]
        dyaw = float(np.degrees(wrap(yaw(qc[tc]) - yaw(qb[tc]))))
    return dict(du_at_close_mm=(round(float(du[tc]), 1)
                                if du.ndim == 1 and np.ndim(du[tc]) == 0
                                else None),
                du_max_mm=(round(float(du_seg), 1) if du_seg is not None
                           else None),
                dxy_at_close_mm=round(float(np.linalg.norm(d[tc, :2])), 1),
                dyaw_at_close_deg=(round(dyaw, 1) if dyaw is not None else None))


def main():
    partial = "--partial" in sys.argv
    summ = {}
    for sname, cfg in SETS.items():
        rows, missing = [], []
        for i in cfg["inits"]:
            for c in CONDS:
                tag = f"init{i:02d}_{c}"
                trajs, metas = {}, {}
                for cd in CANDS:
                    trajs[cd], metas[cd] = load(cfg["pre"], tag, cd)
                if any(m is None for m in metas.values()):
                    missing.append(tag)
                    continue
                r = dict(tag=tag, init=i, cond=c,
                         cand_step=metas["c0"].get("cand_step"),
                         no_trigger=metas["c0"].get("cand_step") is None,
                         lock_step=metas["c0"].get("lock_step"))
                for e, key in EPS.items():
                    for cd in CANDS:
                        r[f"{e}_{cd}"] = bool(metas[cd][key])
                # achieved：各候选对 c0
                for cd in CANDS[1:]:
                    ach = achieved(trajs["c0"], trajs[cd], metas[cd])
                    for k, v in (ach or {}).items():
                        r[f"{cd}_{k}"] = v
                rows.append(r)
        if missing and not partial:
            raise SystemExit(f"{sname}: 缺 {missing}; --partial 查进度")

        # Q1/Q2/Q3
        q = {}
        for e in EPS:
            bf = [r for r in rows if not r[f"{e}_c0"]]
            bs = [r for r in rows if r[f"{e}_c0"]]
            q1 = [r["tag"] for r in bf
                  if any(r[f"{e}_{cd}"] for cd in CANDS[1:])]
            q2 = [r["tag"] for r in bs
                  if any(not r[f"{e}_{cd}"] for cd in CANDS[1:])]
            per_c = {cd: [r["tag"] for r in bf if r[f"{e}_{cd}"]]
                     for cd in CANDS[1:]}
            q[e] = dict(n_base_fail=len(bf), n_base_ok=len(bs),
                        Q1_any_cand_saves=q1,
                        Q2_any_cand_breaks=q2,
                        per_cand_saves=per_c,
                        best_fixed=max(per_c, key=lambda k: len(per_c[k])),
                        oracle_pick=q1)
        no_trig = [r["tag"] for r in rows if r["no_trigger"]]
        summ[sname] = dict(n=len(rows), missing=missing,
                           no_trigger=no_trig, q=q)

        if rows:
            with open(R / f"pose_cand_{sname}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

        print(f"\n===== {sname}  states={len(rows)} missing={len(missing)} "
              f"no_trigger={no_trig}")
        for e in EPS:
            qq = q[e]
            print(f"[{e}] 基线失败 {qq['n_base_fail']} → ≥1候选救回 "
                  f"{len(qq['Q1_any_cand_saves'])}: {qq['Q1_any_cand_saves']}")
            print(f"     基线成功 {qq['n_base_ok']} → 候选破坏 "
                  f"{len(qq['Q2_any_cand_breaks'])}: {qq['Q2_any_cand_breaks']}")
            for cd in CANDS[1:]:
                print(f"     {cd}: 救 {len(qq['per_cand_saves'][cd])} "
                      f"{qq['per_cand_saves'][cd]}")
            print(f"     best_fixed={qq['best_fixed']}  oracle覆盖 "
                  f"{len(qq['oracle_pick'])}")
    if not partial:
        json.dump(summ, open(R / "pose_cand_summary.json", "w"),
                  ensure_ascii=False, indent=1)
        print("\nwrote results/pose_cand_summary.json")


if __name__ == "__main__":
    main()
