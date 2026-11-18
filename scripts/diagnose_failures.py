"""diagnose_failures.py — 逐条失败轨迹的量化诊断

对每个 task_success=False 或 grasp_success=False 的 episode 输出：
- min_d:        eef↔目标 全程最小 xy 距离（接近质量）
- d_at_close:   首次闭合瞬间 eef↔目标 xy 距离（闭合时机/位置）
- z_at_close:   首次闭合时 eef 高度 - 目标高度（闭合高度是否对）
- max_ncon:     目标最大同时接触数（有无碰到）
- knock_cm:     首次闭合后目标 xy 位移（被撞开？）
- lift:         是否曾抬离 z_rest+4cm
- closest:      eef 全程最接近的物体
"""

import csv
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"


def diagnose(tag):
    d = np.load(RESULTS / f"traj_{tag}.npz", allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    eef = np.asarray(d["eef_pos"], dtype=float)          # (T,3)
    tgt = np.asarray(d["tgt_pos"], dtype=float)          # (T,3)
    gcmd = np.asarray(d["gripper_cmd"], dtype=float)     # (T,)
    ncon = np.asarray(d["ncon_tgt"], dtype=float)        # (T,)

    d_xy = np.linalg.norm(eef[:, :2] - tgt[:, :2], axis=1)
    min_d = float(d_xy.min())
    i_min = int(d_xy.argmin())

    close_idx = np.where(gcmd > 0)[0]
    i_close = int(close_idx[0]) if len(close_idx) else None

    out = {
        "tag": tag,
        "task_ok": meta["task_success"], "grasp_ok": meta["grasp_success"],
        "steps": meta["steps"],
        "min_d_cm": round(min_d * 100, 1), "t_min": i_min,
        "eef_z_at_min": round(float(eef[i_min, 2]), 3),
        "tgt_z_rest": round(meta["z_rest"], 3),
        "max_ncon": int(ncon.max()),
        "first_lift": meta.get("first_lift_step"),
        "lift_eef_d_cm": None if meta.get("lift_eef_dist") is None
                         else round(meta["lift_eef_dist"] * 100, 1),
        "closest": meta.get("closest_obj"),
        "min_dists_cm": {k: round(v * 100, 1)
                         for k, v in (meta.get("min_dists") or {}).items()},
    }
    if i_close is not None:
        out["i_close"] = i_close
        out["d_at_close_cm"] = round(float(d_xy[i_close]) * 100, 1)
        out["dz_at_close_cm"] = round(
            float(eef[i_close, 2] - tgt[i_close, 2]) * 100, 1)
        # 闭合后目标被撞位移：close 之后 tgt xy 相对 close 时的最大偏移
        moved = np.linalg.norm(tgt[i_close:, :2] - tgt[i_close, :2], axis=1)
        out["knock_cm"] = round(float(moved.max()) * 100, 1)
    else:
        out["i_close"] = None
    # 后期目标 z：是否最终仍静置
    out["tgt_final_z"] = round(float(tgt[-1, 2]), 3)
    out["tgt_zmax"] = round(float(tgt[:, 2].max()), 3)
    return out


def main():
    rows = list(csv.DictReader(open(RESULTS / "episodes.csv")))
    fails = [r["tag"] for r in rows if r["task_success"] != "True"]
    print(f"{len(fails)} failed episodes\n")
    for tag in fails:
        o = diagnose(tag)
        print(f"== {o['tag']}  steps={o['steps']}  grasp_ok={o['grasp_ok']}")
        print(f"   min_d={o['min_d_cm']}cm@t{o['t_min']}  "
              f"eef_z@min={o['eef_z_at_min']}  tgt_z_rest={o['tgt_z_rest']}")
        if o["i_close"] is not None:
            print(f"   close@t{o['i_close']}  d_at_close={o['d_at_close_cm']}cm "
                  f" dz_at_close={o['dz_at_close_cm']}cm  knock={o['knock_cm']}cm")
        else:
            print("   全程无闭合动作")
        print(f"   max_ncon_tgt={o['max_ncon']}  lift@={o['first_lift']} "
              f" lift_eef_d={o['lift_eef_d_cm']}cm")
        print(f"   tgt_zmax={o['tgt_zmax']}  tgt_final_z={o['tgt_final_z']}  "
              f"closest={o['closest']}")
        print(f"   min_dists={o['min_dists_cm']}\n")


if __name__ == "__main__":
    main()
