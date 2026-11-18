"""verify_trans_cases.py — 阶段二A 任务1

- 俯视对比图：每例 trans 空抓 vs 同 init 的 orig
  （原目标位姿、移动后目标位姿、orig 首闭点、trans 首闭点、闭合时 eef 轨迹尾段）
- 数值：夹爪沿实际平移方向跟随距离、垂直方向偏移
- 各条件 vs orig 的 grasp 成功↔失败翻转统计
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
HALF = np.array([0.009, 0.020, 0.039])  # butter 碰撞盒半尺寸


def load(tag):
    d = np.load(RESULTS / f"traj_{tag}.npz", allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    return meta, d


def first_close(d):
    g = np.asarray(d["exec_action"], dtype=float)[:, -1]
    idx = np.where(g > 0)[0]
    return int(idx[0]) if len(idx) else None


def yaw_of(q):  # quat (x,y,z,w) from obs *_quat — 提取 yaw
    x, y, z, w = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def box_corners(center_xy, yaw, hx, hy):
    c = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy], [hx, hy]])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return center_xy + c @ R.T


def case_panel(ax, init_idx):
    mo, do_ = load(f"init{init_idx:02d}_orig")
    mt, dt = load(f"init{init_idx:02d}_trans")
    co, ct = first_close(do_), first_close(dt)
    eo = np.asarray(do_["eef_pos"], float)
    et = np.asarray(dt["eef_pos"], float)
    tq_o = np.asarray(do_["tgt_quat"], float)[co]
    tq_t = np.asarray(dt["tgt_quat"], float)[ct]
    p_orig = np.asarray(do_["tgt_pos"], float)[co, :2]   # orig 目标（闭合时）
    p_move = np.asarray(dt["tgt_pos"], float)[ct, :2]   # 移动后目标（闭合时）
    e_orig = eo[co, :2]                                  # orig 闭合点
    e_move = et[ct, :2]                                  # trans 闭合点

    ax.plot(*np.c_[box_corners(p_orig, yaw_of(tq_o), HALF[0], HALF[1])].T,
            "C0-", lw=2, label="tgt(orig)")
    ax.plot(*np.c_[box_corners(p_move, yaw_of(tq_t), HALF[0], HALF[1])].T,
            "C3-", lw=2, label="tgt(trans)")
    ax.plot(eo[max(0, co - 15):co + 1, 0], eo[max(0, co - 15):co + 1, 1],
            "C0--", lw=1)
    ax.plot(et[max(0, ct - 15):ct + 1, 0], et[max(0, ct - 15):ct + 1, 1],
            "C3--", lw=1)
    ax.plot(*e_orig, "C0o", ms=10, mfc="none", mew=2, label="close(orig)")
    ax.plot(*e_move, "C3s", ms=10, mfc="none", mew=2, label="close(trans)")
    ax.annotate("", xy=p_move, xytext=p_orig,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("", xy=e_move, xytext=e_orig,
                arrowprops=dict(arrowstyle="->", color="k", ls="--"))
    ax.set_aspect("equal"); ax.legend(fontsize=7)
    ax.set_title(f"init{init_idx:02d} first-close (orig t{co}, trans t{ct})")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    # 数值
    trans_vec = p_move - p_orig          # 实际平移向量（以各自闭合时刻目标为准）
    # 注：扰动本身是 +5cm x；settled pos 已核实。用 perturb 位移方向做投影更严谨：
    req = mt["perturb_req"]["pos"][:2]
    settled = np.array(mt["settled_pos"][:2])
    disp_dir = settled - np.array(req)   # ≈ [+0.05,0]
    L = np.linalg.norm(disp_dir)
    disp_dir = disp_dir / L
    follow = float((e_move - e_orig) @ disp_dir)
    perp = float((e_move - e_orig) @ np.array([-disp_dir[1], disp_dir[0]]))
    return dict(init=init_idx, close_t=[co, ct], follow_cm=follow * 100,
                perp_cm=perp * 100, tgt_shift_cm=L * 100,
                e_orig=e_orig.tolist(), e_move=e_move.tolist(),
                p_orig=p_orig.tolist(), p_move=p_move.tolist())


def flip_stats():
    rows = list(csv.DictReader(open(RESULTS / "episodes.csv")))
    g = {(int(r["init_idx"]), r["condition"]): r for r in rows}
    out = {}
    for cond in ["trans", "rot"]:
        s2f = [i for i in range(10)
               if g[(i, "orig")]["grasp_success"] == "True"
               and g[(i, cond)]["grasp_success"] != "True"]
        f2s = [i for i in range(10)
               if g[(i, "orig")]["grasp_success"] != "True"
               and g[(i, cond)]["grasp_success"] == "True"]
        t_s2f = [i for i in range(10)
                 if g[(i, "orig")]["task_success"] == "True"
                 and g[(i, cond)]["task_success"] != "True"]
        t_f2s = [i for i in range(10)
                 if g[(i, "orig")]["task_success"] != "True"
                 and g[(i, cond)]["task_success"] == "True"]
        out[cond] = dict(grasp_s2f=s2f, grasp_f2s=f2s,
                         task_s2f=t_s2f, task_f2s=t_f2s)
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    res = [case_panel(ax, i) for ax, i in zip(axes, [2, 7])]
    fig.suptitle("First-close top-down: orig vs trans (target +5cm in x)")
    fig.tight_layout()
    fig.savefig(BASE / "results" / "trans_empty_grasp_topdown.png", dpi=140)
    print("saved results/trans_empty_grasp_topdown.png")
    for r in res:
        print(f"init{r['init']:02d}: 目标平移 {r['tgt_shift_cm']:.1f}cm | "
              f"闭合点沿平移方向跟随 {r['follow_cm']:+.1f}cm | "
              f"垂直方向偏移 {r['perp_cm']:+.1f}cm")
        print(f"   orig close={np.round(r['e_orig'],4)} trans close={np.round(r['e_move'],4)}")
        print(f"   orig tgt={np.round(r['p_orig'],4)} moved tgt={np.round(r['p_move'],4)}")
    print("\n== 翻转统计（相对 orig）==")
    for cond, s in flip_stats().items():
        print(f"{cond}: grasp 成功→失败 {len(s['grasp_s2f'])} 例 {s['grasp_s2f']} | "
              f"失败→成功 {len(s['grasp_f2s'])} 例 {s['grasp_f2s']}")
        print(f"      task  成功→失败 {len(s['task_s2f'])} 例 {s['task_s2f']} | "
              f"失败→成功 {len(s['task_f2s'])} 例 {s['task_f2s']}")


if __name__ == "__main__":
    main()
