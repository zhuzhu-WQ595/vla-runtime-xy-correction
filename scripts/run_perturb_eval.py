"""run_perturb_eval.py — GraspField 阶段一：配对位姿偏移诊断实验

用法（在 <repo-root> 下）：
  MUJOCO_GL=egl python scripts/run_perturb_eval.py --smoke
  MUJOCO_GL=egl python scripts/run_perturb_eval.py --n_init_states 10

三条件（同一 init_state 配对）：orig / trans(+5cm x) / rot(+30° yaw，绕物体自身竖直轴)。
每 episode 记录：逐步轨迹 npz（eef/物体位姿/原始动作/执行动作/夹爪/接触数）、
policy 视角 mp4、episodes.csv 汇总行。
仿真真值仅用于场景设置、轨迹记录、结果判定，不进入模型输入。
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# openvla 仓库的 experiments.robot 包（与官方评估相同的导入路径）
sys.path.insert(0, os.environ.get("OPENVLA_DIR", str(Path(__file__).resolve().parents[2] / "openvla")))

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
import imageio
from libero.libero import benchmark

# ---------------- 配置 ----------------
BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
VIDEOS = BASE / "videos"

TASK_ID = 6  # pick up the butter and place it in the basket
TARGET = "butter_1"
TARGET_JOINT = "butter_1_joint0"
TARGET_BODY = "butter_1_main"

# libero_object 任务 id -> (obs 前缀, free joint, 主 body)；仅换目标标识用
TASK_OBJECTS = {
    0: ("alphabet_soup_1", "alphabet_soup_1_joint0", "alphabet_soup_1_main"),
    1: ("cream_cheese_1", "cream_cheese_1_joint0", "cream_cheese_1_main"),
    2: ("salad_dressing_1", "salad_dressing_1_joint0", "salad_dressing_1_main"),
    3: ("bbq_sauce_1", "bbq_sauce_1_joint0", "bbq_sauce_1_main"),
    4: ("ketchup_1", "ketchup_1_joint0", "ketchup_1_main"),
    5: ("tomato_sauce_1", "tomato_sauce_1_joint0", "tomato_sauce_1_main"),
    6: ("butter_1", "butter_1_joint0", "butter_1_main"),
    7: ("milk_1", "milk_1_joint0", "milk_1_main"),
    8: ("chocolate_pudding_1", "chocolate_pudding_1_joint0", "chocolate_pudding_1_main"),
    9: ("orange_juice_1", "orange_juice_1_joint0", "orange_juice_1_main"),
}


def set_task(task_id):
    """切换目标任务：只改模块级目标标识，其他配置保持不变。"""
    global TASK_ID, TARGET, TARGET_JOINT, TARGET_BODY
    TASK_ID = task_id
    TARGET, TARGET_JOINT, TARGET_BODY = TASK_OBJECTS[task_id]
    return TARGET, TARGET_JOINT, TARGET_BODY
CHECKPOINT = "openvla/openvla-7b-finetuned-libero-object"
TASK_SUITE = "libero_object"
MAX_STEPS = 280          # 官方 libero_object 值
NUM_WAIT = 10            # 官方 num_steps_wait
SEED = 7

TRANS_XY = np.array([0.05, 0.0])   # 仅水平平移 +5cm（世界 x 方向）
ROT_YAW_DEG = 30.0                 # 绕物体自身竖直轴（世界 z，过物体质心）

LIFT_H = 0.04            # 抬升阈值：相对静置 z 抬高 4cm
LIFT_HOLD_STEPS = 15     # 稳定保持步数（~1.5s@10Hz 控制频）
NEAR_DIST = 0.08         # “已接近正确物体”：eef 与目标水平距离阈值

CONDITIONS = ["orig", "trans", "rot"]


def quat_mul(q1, q2):
    """(w,x,y,z) 四元数乘法 — MuJoCo free-joint qpos 用此顺序。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def apply_perturb(env, cond):
    """set_init_state 之后调用。返回 (requested, actual) 位姿描述。"""
    q = env.sim.data.get_joint_qpos(TARGET_JOINT).copy()
    req = {"pos": q[:3].tolist(), "quat_wxyz": q[3:].tolist()}
    if cond == "trans":
        q[:2] += TRANS_XY
    elif cond == "rot":
        a = np.deg2rad(ROT_YAW_DEG) / 2.0
        yaw_q = np.array([np.cos(a), 0.0, 0.0, np.sin(a)])  # w,x,y,z
        q[3:] = quat_mul(yaw_q, q[3:])
        q[3:] /= np.linalg.norm(q[3:])
    env.sim.data.set_joint_qpos(TARGET_JOINT, q)
    env.sim.forward()
    act = env.sim.data.get_joint_qpos(TARGET_JOINT).copy()
    return req, {"pos": act[:3].tolist(), "quat_wxyz": act[3:].tolist()}


def target_geom_ids(env):
    bid = env.sim.model.body_name2id(TARGET_BODY)
    return set(np.where(env.sim.model.geom_bodyid == bid)[0].tolist())


def count_target_contacts(env, tgt_geoms):
    """兼容两种 contact 访问方式：结构化数组 (con.geom1[i]) 或对象列表 (con[i].geom1)。"""
    n = 0
    con = env.sim.data.contact
    try:
        g1s, g2s = con.geom1, con.geom2
        for i in range(env.sim.data.ncon):
            if int(g1s[i]) in tgt_geoms or int(g2s[i]) in tgt_geoms:
                n += 1
    except (TypeError, AttributeError):
        for i in range(env.sim.data.ncon):
            c = con[i]
            if int(c.geom1) in tgt_geoms or int(c.geom2) in tgt_geoms:
                n += 1
    return n


class Cfg:
    """get_action/get_model 需要的 cfg 字段（对齐 run_libero_eval.py）。"""
    model_family = "openvla"
    pretrained_checkpoint = CHECKPOINT
    load_in_8bit = False
    load_in_4bit = True
    center_crop = True
    task_suite_name = TASK_SUITE
    unnorm_key = TASK_SUITE
    seed = SEED


def run_episode(env, model, processor, resize_size, task_description,
                init_state, init_idx, cond, ep_tag):
    """跑 1 个 episode，返回 (summary_dict, traj_dict, replay_images)。"""
    tgt_geoms = target_geom_ids(env)
    env.reset()
    obs = env.set_init_state(init_state)

    # 注入位姿偏移（原始条件不加）
    req, act = None, None
    if cond != "orig":
        req, act = apply_perturb(env, cond)

    z_rest = float(obs[f"{TARGET}_pos"][2])

    # 记录逐步数据
    log = {k: [] for k in [
        "t", "sim_time", "eef_pos", "eef_quat", "gripper_qpos",
        "tgt_pos", "tgt_quat", "all_obj_pos", "raw_action", "exec_action",
        "gripper_cmd", "ncon_tgt", "reward", "done",
    ]}
    replay_images = []

    # ---- 沉降阶段（也记录，便于核对偏移是否稳定）----
    for _ in range(NUM_WAIT):
        obs, r, d, info = env.step(get_libero_dummy_action("openvla"))

    # 沉降后的实际目标位姿（偏移有效性核对）
    settled = obs[f"{TARGET}_pos"].tolist()
    settled_quat = obs[f"{TARGET}_quat"].tolist()

    done = False
    t = 0
    exc = None
    first_lift_step = None   # 目标 z 首次超过 z_rest+LIFT_H
    lifted_since = None      # 连续抬升开始步
    hold_ok = False          # 是否达成稳定保持
    lift_eef_dist = None     # 抬升瞬间 eef-目标水平距

    try:
        while t < MAX_STEPS:
            img = get_libero_image(obs, resize_size)
            replay_images.append(img)

            observation = {
                "full_image": img,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                     obs["robot0_gripper_qpos"])
                ),
            }
            raw_action = get_action(Cfg, model, observation, task_description,
                                    processor=processor)
            action = normalize_gripper_action(np.asarray(raw_action).copy(), binarize=True)
            action = invert_gripper_action(action)

            obs, reward, done, info = env.step(action.tolist())
            t += 1

            # ---- 记录 ----
            eef_pos = obs["robot0_eef_pos"]
            tgt_pos = obs[f"{TARGET}_pos"]
            all_pos = {k[:-4]: obs[k].tolist() for k in obs
                       if k.endswith("_pos") and "_to_" not in k
                       and not k.startswith("robot0")}
            log["t"].append(t)
            log["sim_time"].append(float(env.sim.data.time))
            log["eef_pos"].append(eef_pos.tolist())
            log["eef_quat"].append(obs["robot0_eef_quat"].tolist())
            log["gripper_qpos"].append(obs["robot0_gripper_qpos"].tolist())
            log["tgt_pos"].append(tgt_pos.tolist())
            log["tgt_quat"].append(obs[f"{TARGET}_quat"].tolist())
            log["all_obj_pos"].append(all_pos)
            log["raw_action"].append(np.asarray(raw_action).tolist())
            log["exec_action"].append(action.tolist())
            log["gripper_cmd"].append(float(action[-1]))
            log["ncon_tgt"].append(count_target_contacts(env, tgt_geoms))
            log["reward"].append(float(reward))
            log["done"].append(bool(done))

            # ---- 抓取成功检测（独立于任务成功）----
            dz = tgt_pos[2] - z_rest
            if dz >= LIFT_H:
                if first_lift_step is None:
                    first_lift_step = t
                    lifted_since = t
                    lift_eef_dist = float(np.linalg.norm(eef_pos[:2] - tgt_pos[:2]))
                if lifted_since is not None and t - lifted_since >= LIFT_HOLD_STEPS:
                    hold_ok = True
            else:
                lifted_since = None

            if done:
                break
    except Exception as e:
        exc = f"{type(e).__name__}: {e}"

    grasp_success = hold_ok and (lift_eef_dist is not None and lift_eef_dist <= NEAR_DIST)

    # 最接近哪个物体（用于 wrong-object 判断，事后分析用）
    eef = np.array(log["eef_pos"])           # (T,3)
    last_obj = log["all_obj_pos"][-1] if log["all_obj_pos"] else {}
    names = list(last_obj.keys())
    if names:
        tgt_arr = np.array([[o[k] for o in log["all_obj_pos"]] for k in names])  # (nobj,T,3)
        tgt_arr = np.transpose(tgt_arr, (1, 0, 2))                               # (T,nobj,3)
        dists = np.linalg.norm(eef[:, None, :] - tgt_arr, axis=2)                # (T,nobj)
        min_each = dists.min(axis=0)
        closest_name = names[int(np.argmin(min_each))]
        min_dists = {n: float(d) for n, d in zip(names, min_each)}
    else:
        closest_name, min_dists = None, {}

    summary = {
        "tag": ep_tag, "task_id": TASK_ID, "task": task_description,
        "init_idx": int(init_idx), "condition": cond, "seed": SEED,
        "steps": t, "task_success": bool(done), "grasp_success": bool(grasp_success),
        "exception": exc,
        "first_lift_step": first_lift_step, "lift_eef_dist": lift_eef_dist,
        "closest_obj": closest_name, "min_dists": min_dists,
        "z_rest": z_rest,
        "perturb_req": req, "perturb_actual_after_inject": act,
        "settled_pos": settled, "settled_quat": settled_quat,
        "final_tgt_pos": obs[f"{TARGET}_pos"].tolist(),
    }
    traj = {k: np.array(v, dtype=object) if k == "all_obj_pos" else np.asarray(v)
            for k, v in log.items()}
    return summary, traj, replay_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 条件各 1 episode 冒烟")
    ap.add_argument("--n_init_states", type=int, default=0)
    ap.add_argument("--init_start", type=int, default=0)
    ap.add_argument("--conditions", type=str, default=",".join(CONDITIONS))
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)

    set_seed_everywhere(SEED)
    model = get_model(Cfg)
    assert Cfg.unnorm_key in model.norm_stats
    processor = get_processor(Cfg)
    resize_size = get_image_resize_size(Cfg)

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    initial_states = suite.get_task_init_states(TASK_ID)
    env, task_description = get_libero_env(task, "openvla", resolution=256)

    conds = [c for c in args.conditions.split(",") if c in CONDITIONS]
    idxs = [args.init_start] if args.smoke else \
        list(range(args.init_start, args.init_start + args.n_init_states))

    csv_path = RESULTS / "episodes.csv"
    write_header = not csv_path.exists()
    csv_f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=[
        "tag", "task_id", "init_idx", "condition", "seed", "steps",
        "task_success", "grasp_success", "first_lift_step", "lift_eef_dist",
        "closest_obj", "z_rest", "settled_pos", "final_tgt_pos", "video", "traj",
    ])
    if write_header:
        writer.writeheader()

    for init_idx in idxs:
        for cond in conds:
            ep_tag = f"init{init_idx:02d}_{cond}"
            t0 = time.time()
            summary, traj, replay = run_episode(
                env, model, processor, resize_size, task_description,
                initial_states[init_idx], init_idx, cond, ep_tag)
            ok = "G" if summary["grasp_success"] else ("T" if summary["task_success"] else "F")
            vpath = VIDEOS / f"{ep_tag}_{ok}.mp4"
            w = imageio.get_writer(str(vpath), fps=30)
            for im in replay:
                w.append_data(im)
            w.close()
            tpath = RESULTS / f"traj_{ep_tag}.npz"
            np.savez_compressed(tpath, **traj,
                                meta=json.dumps(summary))
            summary["video"] = str(vpath)
            summary["traj"] = str(tpath)
            writer.writerow({k: summary.get(k) for k in writer.fieldnames})
            csv_f.flush()
            print(f"[{ep_tag}] steps={summary['steps']} task={summary['task_success']} "
                  f"grasp={summary['grasp_success']} lift@{summary['first_lift_step']} "
                  f"closest={summary['closest_obj']} ({time.time()-t0:.0f}s)", flush=True)

    csv_f.close()
    env.close()
    print("DONE ->", csv_path)


if __name__ == "__main__":
    main()
