"""gen_butter_newpos.py — 生成 butter「新位置验证」的 20 个冻结物体位置。

要求（冻结前规则，不看结局）：
- 在 WORKSPACE_RECT 内、远离全部 50 个已有 init 位置（≥1.5cm），
- 两两间距 ≥2.0cm，且在扩展区内均匀分布，
- env 内校验：放置后物体静置高度正常（|z-z_rest|<5mm）、与其他 body 无穿插
  （ncon 中目标与外部 body 无接触对）。
输出 runs/predictability/butter_newpos.json: {"init_idx": [x,y], ...}
键 init_idx = 复用的基底 init 状态（init30-49，机器人/场景配置来源）。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ.get("OPENVLA_DIR", str(Path(__file__).resolve().parents[2] / "openvla")))
from experiments.robot.libero.libero_utils import get_libero_env  # noqa: E402
from libero.libero.benchmark import get_benchmark  # noqa: E402
from vision_target import WORKSPACE_RECT  # noqa: E402

BASE_INITS = list(range(30, 50))          # 基底 init（位置被覆盖）
RNG = np.random.default_rng(20261116)
REGION = (-0.175, -0.065, -0.295, -0.185)  # 新位置采样区（workspace 内,绕开篮筐侧）
MIN_INIT_DIST = 0.015
MIN_PAIR_DIST = 0.020

suite = get_benchmark("libero_object")()
task = suite.get_task(6)
init_states = suite.get_task_init_states(6)

env, _ = get_libero_env(task, "openvla", resolution=128)
env.reset()
obs = env.set_init_state(init_states[30])
TGT_KEY = [k for k in obs if k.endswith("_pos") and "butter" in k][0]
J = "butter_1_joint0"
z0 = float(obs[TGT_KEY][2])

init_pos = np.array(json.load(open("runs/predictability/init_positions.json"))["butter"])

bid = env.sim.model.body_name2id("butter_1_main")
tgt_gid = set(np.where(env.sim.model.geom_bodyid == bid)[0].tolist())
print("joint:", J, "z0:", z0, "n_geom:", len(tgt_gid))


def try_pos(x, y):
    env.set_init_state(init_states[30])
    q = env.sim.data.get_joint_qpos(J).copy()
    q[0], q[1] = x, y
    env.sim.data.set_joint_qpos(J, q)
    env.sim.forward()
    p = env.sim.data.get_joint_qpos(J)[:3]
    if abs(float(p[2]) - z0) > 0.005:
        return False
    con = env.sim.data.contact
    try:
        g1s, g2s = con.geom1, con.geom2
        pairs = [(int(g1s[i]), int(g2s[i])) for i in range(env.sim.data.ncon)]
    except (TypeError, AttributeError):
        pairs = [(int(con[i].geom1), int(con[i].geom2))
                 for i in range(env.sim.data.ncon)]
    for g1, g2 in pairs:
        if (g1 in tgt_gid) ^ (g2 in tgt_gid):
            other = g2 if g1 in tgt_gid else g1
            obid = env.sim.model.geom_bodyid[other]
            bname = env.sim.model.body_id2name(int(obid)) or ""
            if "table" not in bname and "floor" not in bname:
                return False      # 与其他物体接触
    return True


cands = []
tries = 0
while len(cands) < 20 and tries < 20000:
    tries += 1
    x = RNG.uniform(REGION[0], REGION[1])
    y = RNG.uniform(REGION[2], REGION[3])
    if not (WORKSPACE_RECT[0] <= x <= WORKSPACE_RECT[1]
            and WORKSPACE_RECT[2] <= y <= WORKSPACE_RECT[3]):
        continue
    if np.linalg.norm(init_pos - [x, y], axis=1).min() < MIN_INIT_DIST:
        continue
    if cands and min(np.hypot(x - c[0], y - c[1]) for c in cands) < MIN_PAIR_DIST:
        continue
    if not try_pos(x, y):
        continue
    cands.append((round(float(x), 5), round(float(y), 5)))
    print(len(cands), cands[-1])

assert len(cands) == 20, f"只生成 {len(cands)} 个有效位置 (tries={tries})"
out = {str(BASE_INITS[i]): [cands[i][0], cands[i][1]] for i in range(20)}
json.dump({"note": "butter 新位置验证: base=init30-49, xy 覆盖到采样区,"
                   "z/yaw 保留基底 init 值; REGION=%s; tries=%d" % (REGION, tries),
           "positions": out},
          open("runs/predictability/butter_newpos.json", "w"), indent=1)
print("wrote runs/predictability/butter_newpos.json, tries=", tries)
env.close()
