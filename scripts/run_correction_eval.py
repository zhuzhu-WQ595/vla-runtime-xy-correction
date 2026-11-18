"""run_correction_eval.py — 阶段二A：原策略 vs 位置纠偏策略 配对运行

arm=vanilla：与阶段一完全相同的原策略回放（检验可复现性 + 配对基线）
arm=corr   ：仅横向位置纠偏。触发后用目标相对位置误差覆盖 action[:2]，
             其余维度（dz/旋转/夹爪）保留策略原始输出。

纠偏参数（开发集 init02/init07 trans 上固定，全批不再调）：
- 触发进入：夹爪开(上一步 cmd<0) AND z_gap∈[Z_LO,Z_HI]
            AND (近4步净下降≥DESC_NET OR z_gap<Z_LOW_OK) AND lat_err>R_ON
- 触发保持(滞回)：lat_err>R_OFF AND z_gap∈[Z_HY_LO,Z_HY_HI] AND 夹爪仍开
- 抓取范围 R_ON=2.5cm（orig 成功闭合偏移 1.4–2.4cm 均在范围内，不误触发）
- 参考点：eef xy vs 目标中心 xy（butter_1_pos，当前观测）
- 修正：action[:2]=clip(K*err_xy, ±C)，K=15，C=0.8（≈1cm/步实测上限）

因果性：仅用当前及历史信息；first_close 只做事后分析字段。
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import imageio
import numpy as np

sys.path.insert(0, os.environ.get("OPENVLA_DIR", str(Path(__file__).resolve().parents[2] / "openvla")))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action, get_libero_env, get_libero_image, quat2axisangle)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action, get_image_resize_size, get_model,
    invert_gripper_action, normalize_gripper_action)
from libero.libero import benchmark
import run_perturb_eval as rpe
from run_perturb_eval import (
    Cfg, CONDITIONS, LIFT_H, LIFT_HOLD_STEPS, MAX_STEPS, NEAR_DIST, NUM_WAIT,
    SEED, TARGET, TARGET_JOINT, TASK_ID, apply_perturb, count_target_contacts,
    target_geom_ids)

BASE = Path(__file__).resolve().parents[1]
RESULTS, VIDEOS = BASE / "results", BASE / "videos2"

# 阶段四 A 候选臂参数（pose_cand_spec_2026-11-17.json 冻结）
CAND_W_MAX = 30                   # 候选偏置最长持续（控制步）
CAND_XY_MM = 5.0                  # 平移候选目标位移（名义）
CAND_RAW2M = 0.0128               # 实测 1 raw ≈ 0.0128 m
# 偏置持续至闭合+2 步（策略伺服会回拉，瞬时偏置达不到抓取位姿差）
CAND_XY_PER_STEP = 0.039          # ≈0.5mm/步
CAND_YAW_PER_STEP = 0.2           # yaw 候选 raw/步；实测 achieved 另报
CAND_YAW_DIM = 3                  # 世界系 yaw 通道（实测 exec[3]~world-z r=-0.71）
CAND_DEC_D = 0.07                 # 决策距离阈值 3D|eef−锁定目标|<70mm
CAND_HOLD_AFTER_CLOSE = 2
# 阶段四 C 冻结选择器（action_sel_spec_2026-11-17.json）
SEL_S = -1                        # pick px iff SEL_S*e > SEL_TE ∧ approach > SEL_TA
SEL_TE_MM = 2.0
SEL_TA_MM = 0.0
SEL_K = 5                         # approach 回看控制步数
SEL_OUT = BASE / "results" / "selector_dev_picks.json"  # dev 复现验证用
# 类别固定参考点（fix_xy_ref_spec_2026-11-17.json；dev 平移状态 locked_est XY 中位数）
FIX_XY = {"butter": (-0.10640, -0.21480), "milk": (-0.09335, -0.21605)}

# 扩展条件（阶段二B 应力测试；冻结参数下仅加新扰动类型）
CONDITIONS_EXT = list(CONDITIONS) + ["trans_negx", "trans_y"]


def apply_perturb_ext(env, cond):
    """原三条件委托给阶段一函数；新增 -5cm x / +5cm y。"""
    if cond in CONDITIONS:
        return apply_perturb(env, cond)
    q = env.sim.data.get_joint_qpos(TARGET_JOINT).copy()
    req = {"pos": q[:3].tolist(), "quat_wxyz": q[3:].tolist()}
    if cond == "trans_negx":
        q[:2] += np.array([-0.05, 0.0])
    elif cond == "trans_y":
        q[:2] += np.array([0.0, 0.05])
    else:
        raise ValueError(cond)
    env.sim.data.set_joint_qpos(TARGET_JOINT, q)
    env.sim.forward()
    act = env.sim.data.get_joint_qpos(TARGET_JOINT).copy()
    return req, {"pos": act[:3].tolist(), "quat_wxyz": act[3:].tolist()}

# ---------------- 纠偏参数（开发集冻结）----------------
Z_LO, Z_HI = 0.0, 0.055      # 预抓取高度区间：eef_z - tgt_z
Z_LOW_OK = 0.02              # 低带豁免：低于此高度无需下降条件
Z_HY_LO, Z_HY_HI = -0.005, 0.075   # 滞回保持区间
DESC_NET = 0.002             # 近 4 步净下降 ≥2mm
R_ON, R_OFF = 0.025, 0.015   # 抓取范围：>2.5cm 触发，<1.5cm 释放
K_CORR, C_CORR = 15.0, 0.8   # 比例增益、单步限幅（raw 单位；0.8≈1cm/步）
WIN = 4                      # 下降窗


class CorrState:
    """纠偏状态机：因果触发 + 滞回；gated=True 时闭合屏蔽（阶段二C）。"""

    def __init__(self, gated=False):
        self.gated = gated
        self.masked = False        # 闭合屏蔽：闭合当步起、夹爪重开解除
        self.active = False
        self.z_hist = []          # eef_z 历史
        self.first_trigger = None  # 首次触发步（只记最早，不被重入覆盖）
        self.activations = []      # 所有激活区间 [(start,end),...]
        self.n_steps = 0
        self.n_masked = 0          # 屏蔽步数
        self.mask_mismatch = 0     # 屏蔽期 exec!=raw 步数（应恒 0）
        self.err_at_trigger = None
        self.corr_integral = np.zeros(2)   # 活动期修正位移累计（m）

    def update(self, action, eef_z, eef_xy, tgt_z, tgt_xy, last_cmd_open, t):
        """返回修正后的 action 与诊断 dict。"""
        self.z_hist.append(float(eef_z))
        z_gap = eef_z - tgt_z
        err = tgt_xy - eef_xy
        lat = float(np.linalg.norm(err))

        descent = (len(self.z_hist) > WIN
                   and self.z_hist[-WIN - 1] - self.z_hist[-1] >= DESC_NET)
        in_band = Z_LO <= z_gap <= Z_HI
        low_ok = z_gap < Z_LOW_OK

        if not self.active:
            if (last_cmd_open and in_band and (descent or low_ok)
                    and lat > R_ON):
                self.active = True
                if self.first_trigger is None:
                    self.first_trigger = t
                    self.err_at_trigger = lat
                self.activations.append([t, t])
        else:
            if (not last_cmd_open or lat <= R_OFF
                    or not (Z_HY_LO <= z_gap <= Z_HY_HI)):
                self.active = False
                self.activations[-1][1] = t

        # 阶段二C：本步闭合命令 -> 屏蔽横向覆盖并结束激活；重开步解除。
        # 重开后是否再介入，仍由上面的因果触发/滞回条件决定。
        cmd_close = action[-1] > 0
        if self.gated:
            if cmd_close:
                if self.active:
                    self.active = False
                    self.activations[-1][1] = t
                self.masked = True
            elif self.masked:
                self.masked = False   # 本步为重开命令

        if self.active:
            new_a = action.copy()
            new_a[:2] = np.clip(K_CORR * err, -C_CORR, C_CORR)
            self.n_steps += 1
            self.corr_integral += (new_a[:2] - action[:2]) * 0.0128  # 实测映射
            return new_a, {"corr_active": 1, "z_gap": z_gap, "lat_err": lat,
                           "gate_masked": int(self.masked)}
        if self.masked:
            self.n_masked += 1
        return action, {"corr_active": 0, "z_gap": z_gap, "lat_err": lat,
                        "gate_masked": int(self.masked)}


def run_episode(env, model, processor, obs, z_rest, tgt_geoms,
                task_description, resize_size, arm,
                branch_alpha=0.0, branch_t=-1, branch_mode="single",
                cand="c0"):
    """arm='vanilla' 原样执行；'corr' 触发后覆盖；'gate' 加闭合屏蔽；
    'diag' 消融臂：内部纯 2B 状态机（不因屏蔽截断激活区间），只在输出端
    按 2C 闭合/重开判定屏蔽横向覆盖，屏蔽期执行 raw_action；
    'branch' 单步因果分支臂：同 diag 内部+屏蔽语义，但在牵引步执行
    raw+alpha*(prop-raw)。branch_mode='single' 仅 branch_t 一步牵引；
    'close' 每次屏蔽开启步（=每次闭合命令上升沿）都牵引。"""
    corr = (CorrState(gated=(arm in ("gate", "visgate", "vislock", "poseb",
                                    "fixlock")))
            if arm in ("corr", "gate", "visgate", "vislock", "poseb",
                       "fixlock", "diag", "branch") else None)
    # visgate/vislock/poseb/fixlock：视觉目标中心估计（类名由 TARGET 派生）
    est = None
    if arm in ("visgate", "vislock", "poseb", "fixlock"):
        from vision_target import TargetEstimator
        est = TargetEstimator(TARGET.split("_")[0], res=256)
    locked_est = None     # vislock/poseb：首个非空估计锁定为整回合目标
    lock_step = None
    cand_step = None      # poseb：决策步（None=未触发→同基线）
    u_vec = None          # 决策时闭合轴单位向量（XY）
    pose_at_dec = pose_at_end = None
    cand_exec = close_overlap = 0
    fm_hist = []          # 每迭代步前夹爪中点（sel 特征用）
    sel_pick = sel_e = sel_ap = None   # sel 臂：冻结选择结果与特征
    replay, log = [], {k: [] for k in
                     ["t", "eef_pos", "tgt_pos", "tgt_quat", "gripper_cmd",
                      "gripper_qpos", "finger_mid", "exec_action",
                      "raw_action", "prop_action",
                      "corr_active", "gate_masked", "branch_tug", "z_gap",
                      "lat_err", "ncon_tgt", "done",
                      "est_tgt", "vis_miss", "vis_score", "est_live",
                      "eef_quat", "cand_bias"]}
    last_cmd = -1.0
    diag_masked = False      # diag/branch 臂输出端屏蔽标志（2C 闭合/重开判定）
    tug_steps = []           # branch 臂：实际施加牵引的控制步(1-based)
    # 两指垫 geom → 指尖中点（Panda 夹爪碰撞垫）
    _fids = None
    for _nm in ("gripper0_finger1_pad_collision",
                "gripper0_finger2_pad_collision"):
        try:
            _fid = env.sim.model.geom_name2id(_nm)
        except Exception:
            _fid = None
        _fids = (_fids or []) + [_fid]
    t, done = 0, False
    exc = None
    first_lift_step = lifted_since = None
    hold_ok = False
    lift_eef_dist = None
    first_close_t = None
    close_ts = []                # 所有闭合命令步

    try:
        while t < MAX_STEPS:
            img = get_libero_image(obs, resize_size)
            replay.append(img)
            observation = {"full_image": img,
                           "state": np.concatenate(
                               (obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"]))}
            raw = get_action(Cfg, model, observation, task_description,
                             processor=processor)
            action = normalize_gripper_action(np.asarray(raw).copy(),
                                              binarize=True)
            action = invert_gripper_action(action)
            raw_action = action.copy()   # 本分支本步未经纠偏的动作

            diag = {"corr_active": 0, "gate_masked": 0, "branch_tug": 0,
                    "z_gap": None, "lat_err": None}
            # 喂给 corr 的目标点：visgate/vislock 用视觉估计，其余臂用真值
            e_est, e_live, vis_miss, vis_score = (None, None,
                (1 if arm in ("visgate", "vislock", "fixlock") else 0),
                float("nan"))
            if est is not None:
                e_live, vdbg = est.estimate(obs["agentview_image"])
                vis_score = float(vdbg.get("score", float("nan")))
                if arm in ("vislock", "poseb", "fixlock"):
                    if locked_est is None:
                        e_est = e_live
                        if e_live is not None:
                            locked_est = np.asarray(e_live,
                                                    dtype=float).copy()
                            if arm == "fixlock":
                                # 唯一改动：锁定 XY 替换为类固定点（z 保持）
                                locked_est[:2] = FIX_XY[TARGET.split("_")[0]]
                            lock_step = t          # 锁定步（0-based 控制步）
                    else:
                        e_est = locked_est       # 已锁定：忽略逐帧估计
                else:
                    e_est = e_live
                vis_miss = int(e_est is None)
            if est is not None and e_est is None:
                # 漏检：tgt_xy=eef_xy（lat=0 天然抑制触发/关断），z 用 eef_z
                t_z, t_xy = (float(obs["robot0_eef_pos"][2]),
                             obs["robot0_eef_pos"][:2])
            elif est is not None:
                t_z, t_xy = float(e_est[2]), e_est[:2]
            else:
                t_z = obs[f"{TARGET}_pos"][2]
                t_xy = obs[f"{TARGET}_pos"][:2]
            if corr is not None:
                action, diag = corr.update(
                    action, obs["robot0_eef_pos"][2], obs["robot0_eef_pos"][:2],
                    t_z, t_xy,
                    last_cmd_open=(last_cmd < 0), t=t)
            prop_action = action.copy()  # 本分支状态下纠偏规则提议的动作
            if arm == "poseb":   # 步前夹爪中点（迭代 t ↔ post-step 日志索引 t-1）
                if _fids and _fids[0] is not None and _fids[1] is not None:
                    fm_hist.append((np.asarray(env.sim.data.geom_xpos[_fids[0]][:2])
                                    + np.asarray(env.sim.data.geom_xpos[_fids[1]][:2])) / 2)
                else:
                    fm_hist.append(obs["robot0_eef_pos"][:2].copy())
            # poseb 候选臂：决策检测（冻结规则：已锁定+未闭合+3D 接近）
            bias = [0.0, 0.0, 0.0]
            if arm == "poseb":
                if (cand_step is None and locked_est is not None
                        and not close_ts):
                    _t3 = np.array([t_xy[0], t_xy[1], t_z])
                    if np.linalg.norm(obs["robot0_eef_pos"] - _t3) \
                            < CAND_DEC_D:
                        cand_step = t
                        if _fids and _fids[0] is not None \
                                and _fids[1] is not None:
                            _u = (env.sim.data.geom_xpos[_fids[0]][:2]
                                  - env.sim.data.geom_xpos[_fids[1]][:2])
                            _nu = np.linalg.norm(_u)
                            u_vec = (_u / _nu if _nu > 1e-6
                                     else np.array([1.0, 0.0]))
                        else:
                            u_vec = np.array([1.0, 0.0])
                        pose_at_dec = (obs["robot0_eef_pos"].copy(),
                                       obs["robot0_eef_quat"].copy())
                        if cand == "sel" and u_vec is not None:
                            # 冻结选择器（决策状态=当前步前 obs）
                            _fm0 = fm_hist[t][:2]
                            _fmk = fm_hist[max(0, t - SEL_K)][:2]
                            sel_e = float((np.asarray(locked_est[:2])
                                           - _fm0) @ u_vec) * 1000
                            sel_ap = float((_fm0 - _fmk) @ u_vec) * 1000
                            sel_pick = ("px" if (SEL_S * sel_e > SEL_TE_MM
                                        and sel_ap > SEL_TA_MM) else "c0")
                # 候选窗口：偏置持续至首次闭合+HOLD 或 W_MAX（施加于 corr/屏蔽后）
                _eff = sel_pick if cand == "sel" else cand
                _stop = (close_ts and t >= close_ts[0] + CAND_HOLD_AFTER_CLOSE)
                if (cand_step is not None and _eff != "c0"
                        and cand_step <= t < cand_step + CAND_W_MAX
                        and not _stop):
                    action = action.copy()
                    if _eff in ("px", "nx"):
                        s = 1.0 if _eff == "px" else -1.0
                        b = s * u_vec * CAND_XY_PER_STEP
                        action[:2] += b
                        bias = [float(b[0]), float(b[1]), 0.0]
                    else:
                        s = 1.0 if cand == "ry" else -1.0
                        action[CAND_YAW_DIM] += s * CAND_YAW_PER_STEP
                        bias = [0.0, 0.0, s * CAND_YAW_PER_STEP]
                    cand_exec += 1
                    if action[-1] > 0:
                        close_overlap += 1
            if arm == "diag":
                if action[-1] > 0:
                    diag_masked = True        # 本步闭合命令
                elif diag_masked:
                    diag_masked = False       # 屏蔽后首个重开命令
                if diag_masked:
                    action = raw_action.copy()  # 仅输出端屏蔽，内部区间不截断
                    diag["gate_masked"] = 1
                    corr.n_masked += 1
            elif arm == "branch":
                cmd_close = action[-1] > 0
                prev_masked = diag_masked
                if cmd_close:
                    diag_masked = True        # 本步闭合命令 → 屏蔽开
                elif diag_masked:
                    diag_masked = False       # 屏蔽后首个重开命令
                mask_on = cmd_close and not prev_masked   # 屏蔽开启步(闭合上升沿)
                if diag_masked:
                    action = raw_action.copy()
                    diag["gate_masked"] = 1
                    corr.n_masked += 1
                # 牵引步：single=仅 branch_t；close=每个屏蔽开启步
                tug = ((branch_mode == "single" and t + 1 == branch_t)
                       or (branch_mode == "close" and mask_on))
                if tug:
                    action = action.copy()
                    action[:2] = (raw_action[:2] + branch_alpha
                                  * (prop_action[:2] - raw_action[:2]))
                    diag["branch_tug"] = 1
                    tug_steps.append(t + 1)
            # 屏蔽期验证：executed 必须等于 raw_action（同一动作处理层）
            # branch 牵引步为设计内的有意偏离，不计违例
            if (diag["gate_masked"] and corr is not None
                    and not diag.get("branch_tug") and not np.allclose(
                    action[:2], raw_action[:2], atol=1e-8)):
                corr.mask_mismatch += 1

            obs, reward, done, info = env.step(action.tolist())
            t += 1
            if (arm == "poseb" and cand_step is not None and pose_at_end is None
                    and (t == cand_step + CAND_W_MAX
                         or (close_ts and t == close_ts[0]
                             + CAND_HOLD_AFTER_CLOSE))):
                pose_at_end = (obs["robot0_eef_pos"].copy(),
                               obs["robot0_eef_quat"].copy())
            if action[-1] > 0:
                if first_close_t is None:
                    first_close_t = t
                close_ts.append(t)
            last_cmd = action[-1]

            eef_pos, tgt_pos = obs["robot0_eef_pos"], obs[f"{TARGET}_pos"]
            log["t"].append(t)
            log["eef_pos"].append(eef_pos.tolist())
            log["tgt_pos"].append(tgt_pos.tolist())
            log["gripper_cmd"].append(float(action[-1]))
            log["exec_action"].append(action.tolist())
            log["raw_action"].append(raw_action.tolist())
            log["prop_action"].append(prop_action.tolist())
            log["tgt_quat"].append(obs[f"{TARGET}_quat"].tolist())
            log["gripper_qpos"].append(obs["robot0_gripper_qpos"].tolist())
            if _fids and _fids[0] is not None and _fids[1] is not None:
                _fm = (np.asarray(env.sim.data.geom_xpos[_fids[0]])
                       + np.asarray(env.sim.data.geom_xpos[_fids[1]])) / 2
            else:
                _fm = eef_pos
            log["finger_mid"].append(_fm.tolist())
            log["corr_active"].append(diag["corr_active"])
            log["gate_masked"].append(diag["gate_masked"])
            log["branch_tug"].append(diag.get("branch_tug", 0))
            log["z_gap"].append(diag["z_gap"])
            log["lat_err"].append(diag["lat_err"])
            log["ncon_tgt"].append(count_target_contacts(env, tgt_geoms))
            log["done"].append(bool(done))
            log["est_tgt"].append([float(t_xy[0]), float(t_xy[1]), float(t_z)])
            log["vis_miss"].append(vis_miss)
            log["vis_score"].append(vis_score)
            log["est_live"].append(
                e_live.tolist() if e_live is not None else [np.nan] * 3)
            log["eef_quat"].append(eef_pos and obs["robot0_eef_quat"].tolist()
                                   if False else obs["robot0_eef_quat"].tolist())
            log["cand_bias"].append(bias)

            dz = tgt_pos[2] - z_rest
            if dz >= LIFT_H:
                if first_lift_step is None:
                    first_lift_step = t
                    lifted_since = t
                    lift_eef_dist = float(
                        np.linalg.norm(eef_pos[:2] - tgt_pos[:2]))
                if lifted_since is not None and t - lifted_since >= LIFT_HOLD_STEPS:
                    hold_ok = True
            else:
                lifted_since = None
            if done:
                break
    except Exception as e:
        exc = f"{type(e).__name__}: {e}"

    grasp_ok = hold_ok and (lift_eef_dist is not None
                            and lift_eef_dist <= NEAR_DIST)
    eef = np.asarray(log["eef_pos"], float)
    tgt = np.asarray(log["tgt_pos"], float)
    lat = np.asarray([x if x is not None else np.nan for x in log["lat_err"]],
                     dtype=float)
    # 事后指标（仅用记录，不影响控制）
    d_xy = np.linalg.norm(eef[:, :2] - tgt[:, :2], axis=1)
    summary = {
        "steps": t, "task_success": bool(done),
        "grasp_success": bool(grasp_ok), "exception": exc,
        "first_close_t": first_close_t,
        "first_lift_step": first_lift_step,
        "lift_eef_dist_cm": None if lift_eef_dist is None
        else round(lift_eef_dist * 100, 1),
        "min_d_cm": round(float(d_xy.min()) * 100, 1),
        "d_at_close_cm": None if first_close_t is None
        else round(float(d_xy[first_close_t - 1]) * 100, 1),
    }
    if corr is not None:
        # 提前量：首次触发 → 其后最近一个闭合命令步（触发后无闭合则 None）
        lead = None
        if corr.first_trigger is not None:
            after = [c for c in close_ts if c > corr.first_trigger]
            if after:
                lead = after[0] - corr.first_trigger
        summary.update({
            "first_trigger_t": corr.first_trigger,
            "trigger_lead_steps": lead,
            "activations": corr.activations,
            "n_corr_steps": corr.n_steps,
            "err_at_trigger_cm": None if corr.err_at_trigger is None
            else round(corr.err_at_trigger * 100, 1),
            "err_at_close_cm": (None if first_close_t is None or np.isnan(
                lat[first_close_t - 1] if first_close_t - 1 < len(lat) else np.nan)
                else round(float(lat[first_close_t - 1]) * 100, 1)),
            "corr_path_cm": round(float(np.linalg.norm(corr.corr_integral)) * 100, 1),
            "n_masked_steps": corr.n_masked,
            "mask_mismatch": corr.mask_mismatch,
        })
    try:
        _hs = env.sim.model.geom_size[next(iter(tgt_geoms))]
        summary["box_half_xy_mm"] = [round(float(x) * 1000, 1) for x in _hs[:2]]
        summary["box_half_size_mm"] = [round(float(x) * 1000, 1) for x in _hs]
    except Exception:
        pass
    if arm in ("vislock", "poseb", "fixlock"):
        summary.update({"lock_step": lock_step,
                        "locked_est": (None if locked_est is None
                                       else [round(float(x), 4)
                                             for x in locked_est])})
    if arm == "poseb":
        def _yaw(q):
            w, x, y, z = q
            return float(np.arctan2(2 * (w * z + x * y),
                                    1 - 2 * (y * y + z * z)))
        ach = dict(du_mm=None, dxy_mm=None, dyaw_deg=None)
        if pose_at_dec is not None and pose_at_end is not None:
            dxy = pose_at_end[0][:2] - pose_at_dec[0][:2]
            ach["dxy_mm"] = round(float(np.linalg.norm(dxy)) * 1000, 1)
            ach["du_mm"] = (round(float(dxy @ u_vec) * 1000, 1)
                            if u_vec is not None else None)
            dyaw = _yaw(pose_at_end[1]) - _yaw(pose_at_dec[1])
            ach["dyaw_deg"] = round(float(
                (dyaw + np.pi) % (2 * np.pi) - np.pi) * 180 / np.pi, 1)
        summary.update({"cand": cand, "cand_step": cand_step,
                        "sel_pick": sel_pick,
                        "sel_e_mm": (None if sel_e is None
                                     else round(sel_e, 1)),
                        "sel_ap_mm": (None if sel_ap is None
                                      else round(sel_ap, 1)),
                        "u_vec": (None if u_vec is None
                                  else [round(float(x), 4) for x in u_vec]),
                        "cand_w_exec": cand_exec,
                        "close_overlap": close_overlap,
                        "achieved_du_mm": ach["du_mm"],
                        "achieved_dxy_mm": ach["dxy_mm"],
                        "achieved_dyaw_deg": ach["dyaw_deg"]})
    if arm == "branch":
        summary.update({"branch_alpha": branch_alpha,
                        "branch_t": branch_t,
                        "branch_mode": branch_mode,
                        "n_tug_steps": len(tug_steps),
                        "tug_steps": tug_steps,
                        "prop_at_branch": True})
    return summary, {k: np.asarray(v, dtype=object) for k, v in log.items()}, replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["vanilla", "corr", "gate", "diag",
                                      "branch", "visgate", "vislock", "poseb",
                                      "fixlock"],
                    required=True)
    ap.add_argument("--cand", choices=["c0", "px", "nx", "ry", "ly",
                                         "sel"],
                    default="c0",
                    help="poseb 候选：c0 基线/px,nx 沿闭合轴±5mm/ry,ly ±yaw")
    ap.add_argument("--branch_alpha", default="0",
                    help="逗号分隔 α 列表；branch 臂逐 α 回放")
    ap.add_argument("--branch_t", default="-1",
                    help="逗号分隔分叉步 t*(1-based)，与 (init×cond) 对位")
    ap.add_argument("--branch_mode", choices=["single", "close"],
                    default="single")
    ap.add_argument("--episodes", default="",
                    help="如 '2,7' 只跑这些 init 的 trans 条件（开发用）")
    ap.add_argument("--n_init_states", type=int, default=10)
    ap.add_argument("--init_list", default="",
                    help="如 '10-29' 或 '10,12,15'，与 --conditions 组合")
    ap.add_argument("--conditions", default="",
                    help="逗号分隔，默认 orig,trans,rot")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--task_id", type=int, default=None,
                    help="libero_object 任务 id；默认沿用 TASK_ID=6(butter)")
    ap.add_argument("--out_prefix", default="traj2",
                    help="npz 文件名前缀，第二物体实验用 traj3 等区分")
    ap.add_argument("--vid_prefix", default="",
                    help="视频文件名前缀（默认不加，沿用原命名）")
    ap.add_argument("--pos_override", default="",
                    help="JSON: {init_idx: [x,y]} 新位置验证——set_init_state 后覆盖目标 free-joint XY")
    args = ap.parse_args()

    # 第二物体迁移：仅替换目标标识与任务本身，其余冻结配置不变
    global TARGET, TARGET_JOINT, TASK_ID
    if args.task_id is not None and args.task_id != TASK_ID:
        TARGET, TARGET_JOINT, _ = rpe.set_task(args.task_id)
        TASK_ID = args.task_id

    out_csv = args.out_csv or str(RESULTS / f"episodes_{args.arm}.csv")
    RESULTS.mkdir(exist_ok=True); VIDEOS.mkdir(exist_ok=True)

    model = get_model(Cfg)
    processor = get_processor(Cfg)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(TASK_ID)
    task_description = task.language
    initial_states = suite.get_task_init_states(TASK_ID)
    pos_override = None
    if args.pos_override:
        pos_override = json.load(open(args.pos_override)).get("positions",
                            json.load(open(args.pos_override)))
        print(f"[pos_override] {len(pos_override)} 个 init 有新位置覆盖")
    resize_size = get_image_resize_size(Cfg)

    conds = (args.conditions.split(",") if args.conditions
             else CONDITIONS)
    if args.episodes:
        dev = {int(x) for x in args.episodes.split(",")}
        ep_list = [(i, "trans") for i in sorted(dev)]
    elif args.init_list:
        inits = []
        for part in args.init_list.split(","):
            if "-" in part:
                a, b = part.split("-")
                inits += list(range(int(a), int(b) + 1))
            else:
                inits.append(int(part))
        ep_list = [(i, c) for i in sorted(set(inits)) for c in conds]
    else:
        ep_list = [(i, c) for i in range(args.n_init_states)
                   for c in conds]

    write_header = not Path(out_csv).exists()
    # branch 臂：alpha 逐值回放；branch_t 与 ep_list 逐例对位
    alphas = ([float(x) for x in args.branch_alpha.split(",")]
              if args.arm == "branch" else [0.0])
    bts = [int(x) for x in args.branch_t.split(",")]
    if args.arm == "branch" and args.branch_mode == "single":
        if len(bts) == 1 and len(ep_list) > 1:
            bts *= len(ep_list)
        assert len(bts) == len(ep_list), \
            f"branch_t 个数({len(bts)})需与 episodes({len(ep_list)})对齐"
    with open(out_csv, "a", newline="") as f:
        w = None
        for ep_i, (init_idx, cond) in enumerate(ep_list):
            tag = f"init{init_idx:02d}_{cond}"
            for alpha in alphas:
                # 每个分支从相同初始状态全新重建（策略/仿真无前缀缓存）
                env, _ = get_libero_env(task, "openvla", resolution=256)
                env.reset()
                obs = env.set_init_state(initial_states[init_idx])
                pos_ovr = None
                if pos_override and str(init_idx) in pos_override:
                    pos_ovr = pos_override[str(init_idx)]
                    q = env.sim.data.get_joint_qpos(TARGET_JOINT).copy()
                    q[0], q[1] = pos_ovr
                    env.sim.data.set_joint_qpos(TARGET_JOINT, q)
                    env.sim.forward()
                    obs = env.regenerate_obs_from_state(env.get_sim_state())
                perturb_req = perturb_act = None
                if cond != "orig":
                    perturb_req, perturb_act = apply_perturb_ext(env, cond)
                z_rest = obs[f"{TARGET}_pos"][2]
                tgt_geoms = target_geom_ids(env)
                for _ in range(NUM_WAIT):
                    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                arm_label = (f"branch_a{alpha:g}_{args.branch_mode}"
                             if args.arm == "branch" else
                             f"pb_{args.cand}" if args.arm == "poseb"
                             else args.arm)
                summary, traj, replay = run_episode(
                    env, model, processor, obs, z_rest, tgt_geoms,
                    task_description, resize_size, args.arm,
                    branch_alpha=alpha,
                    branch_t=(bts[ep_i] if args.arm == "branch" else -1),
                    branch_mode=args.branch_mode, cand=args.cand)
                summary.update({"tag": tag, "arm": arm_label,
                                "init_idx": init_idx,
                                "condition": cond, "seed": SEED,
                                "perturb_req": perturb_req,
                                "perturb_act": perturb_act,
                                "pos_override": pos_ovr,
                                "settled_pos": obs[f"{TARGET}_pos"].tolist()})
                np.savez_compressed(RESULTS / f"{args.out_prefix}_{tag}_{arm_label}.npz",
                                    meta=json.dumps(summary), **traj)
                vf = VIDEOS / (f"{args.vid_prefix}{tag}_{arm_label}_"
                               f"{'G' if summary['grasp_success'] else 'F'}.mp4")
                vw = imageio.get_writer(str(vf), fps=30)
                for fr in replay:
                    vw.append_data(fr)
                vw.close()
                summary["video"] = str(vf)
                print(f"[{arm_label}:{tag}] steps={summary['steps']} "
                      f"task={summary['task_success']} "
                      f"grasp={summary['grasp_success']}", flush=True)
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(summary.keys()))
                    if write_header:
                        w.writeheader()
                w.writerow(summary)
                f.flush()
                env.close()


if __name__ == "__main__":
    main()
