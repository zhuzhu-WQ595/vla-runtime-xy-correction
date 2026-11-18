"""plot_gate_cases.py — 阶段二C 关键病例时序图。

每例一行：lat_err(t) 双线（2B 灰 / 2C 蓝）、eef_z-tgt_z 高度差、
gripper_cmd、2B corr_active（灰底）与 2C gate_masked（蓝底）区间、
闭合命令竖线。输出 results/gate_cases.png。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
R = BASE / "results"

CASES = [
    ("init11_orig", "2B break -> 2C save"),
    ("init22_orig", "2B break -> 2C save"),
    ("init02_trans", "reopen->re-engage (val)"),
    ("init10_trans_y", "2B save -> 2C lost"),
    ("init26_rot", "residual break"),
]


def load(tag, arm):
    d = np.load(R / f"traj2_{tag}_{arm}.npz", allow_pickle=True)
    m = json.loads(str(d["meta"]))
    lat = np.asarray([x if x is not None else np.nan
                      for x in d["lat_err"].tolist()], float)
    zg = np.asarray([x if x is not None else np.nan
                     for x in d["z_gap"].tolist()], float)
    g = np.asarray(d["gripper_cmd"].tolist(), float)
    ca = np.asarray(d["corr_active"].tolist(), float)
    gm = (np.asarray(d["gate_masked"].tolist(), float)
          if "gate_masked" in d else np.zeros_like(ca))
    return (np.asarray(d["t"].tolist(), int), lat, zg, g, ca, gm, m)


fig, axes = plt.subplots(len(CASES), 1, figsize=(11, 2.5 * len(CASES)),
                         sharex=False)
for ax, (tag, note) in zip(axes, CASES):
    t2, lat2, zg2, g2, ca2, _, m2 = load(tag, "corr")
    tg, latg, zgg, gg, cag, gmg, mg = load(tag, "gate")
    ax.plot(t2, lat2 * 100, color="0.5", lw=1.4, label="2B lat_err")
    ax.plot(tg, latg * 100, color="tab:blue", lw=1.6, label="2C lat_err")
    ax.axhline(2.5, color="r", ls=":", lw=0.8, alpha=0.6)
    # 2B 激活区间（灰底）
    for s, e in m2.get("activations", []):
        ax.axvspan(s, e, color="0.5", alpha=0.15)
    # 2C 激活区间（蓝底）与屏蔽区间（青底）
    for s, e in mg.get("activations", []):
        ax.axvspan(s, e, color="tab:blue", alpha=0.2)
    mstart = tg[np.concatenate([[gmg[0] == 1], np.diff(gmg) == 1])]
    mend = tg[np.concatenate([np.diff(gmg) == -1, [gmg[-1] == 1]])]
    for s, e in zip(mstart, mend):
        ax.axvspan(s, e, color="tab:cyan", alpha=0.25)
    # 闭合命令竖线
    for t in tg[1:][np.diff((gg > 0).astype(int)) == 1]:
        ax.axvline(t, color="tab:blue", ls="--", lw=0.7, alpha=0.5)
    ax.set_ylabel("lat err (cm)")
    ax.set_title(f"{tag}  {note}   [2B grasp={m2['grasp_success']} "
                 f"task={m2['task_success']} | 2C grasp={mg['grasp_success']} "
                 f"task={mg['task_success']} masked={mg.get('n_masked_steps')}]",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, max(np.nanmax(lat2 * 100), np.nanmax(latg * 100)) * 1.05)
axes[-1].set_xlabel("step")
fig.suptitle("gray=2B corr active  blue=2C corr active  "
             "cyan=2C masked  blue dash=close cmd",
             y=1.0)
fig.tight_layout()
fig.savefig(R / "gate_cases.png", dpi=140, bbox_inches="tight")
print("wrote", R / "gate_cases.png")
