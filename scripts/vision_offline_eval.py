"""vision_offline_eval.py — 视觉估计离线核验（阶段三 A 第 3 步）。

两段：
A) 标定：vanilla 各回合 t<lift 步跑 NCC 检测，收 (底中点 px224, 真值 XY) 对
   → 最小二乘仿射 → runs/predictability/pix2w_affine.json
B) 验证：vanilla+gate 全回合逐步估计 →
   接近段(t<lift) XY 误差 / 闭合前窗口(首闭合前10步) XY 误差 /
   Z 有符号误差 / 漏检率 / 用估计真值回放的触发翻转步数
真值仅用于评分与 oracle 对照，不进入估计器。
"""
import cv2
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_target import TargetEstimator, PIX2W_PATH  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
R, V = BASE / "results", BASE / "videos2"
SETS = {
    "butter": dict(cls="butter", prefix="traj2", vpre="", inits=range(30, 50),
                   conds=["orig", "trans", "rot", "trans_y"]),
    "milk": dict(cls="milk", prefix="traj3", vpre="milk_", inits=range(20),
                 conds=["orig", "trans", "trans_y", "rot"]),
}
DESC_NET = 0.002; WIN = 4
Z_LO, Z_HI, Z_LOW_OK = 0.0, 0.055, 0.02
Z_HY_LO, Z_HY_HI = -0.005, 0.075
R_ON, R_OFF = 0.025, 0.015


class ReplayEst:
    """ReplayCorr 视觉版：est 为 None 的步直接抑制触发（冻结漏检策略）。"""

    def __init__(self):
        self.active = False
        self.z_hist = []

    def update(self, eef_z, eef_xy, est, last_cmd_open):
        """est: None 或 (x,y,z)。返回 (active, lat, z_gap)。"""
        self.z_hist.append(float(eef_z))
        if est is None:
            self.active = False
            return 0, None, None
        tx, ty, tz = est
        z_gap = eef_z - tz
        lat = float(np.hypot(tx - eef_xy[0], ty - eef_xy[1]))
        descent = (len(self.z_hist) > WIN
                   and self.z_hist[-WIN - 1] - self.z_hist[-1] >= DESC_NET)
        in_band = Z_LO <= z_gap <= Z_HI
        low_ok = z_gap < Z_LOW_OK
        if not self.active:
            if last_cmd_open and in_band and (descent or low_ok) and lat > R_ON:
                self.active = True
        else:
            if (not last_cmd_open or lat <= R_OFF
                    or not (Z_HY_LO <= z_gap <= Z_HY_HI)):
                self.active = False
        return int(self.active), lat, z_gap


def vid_path(cfg, tag, arm, success_g):
    suf = "G" if success_g else "F"
    return V / f"{cfg['vpre']}{tag}_{arm}_{suf}.mp4"


def frames(vpath):
    cap = cv2.VideoCapture(str(vpath))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)[::-1, ::-1])
    cap.release()
    return out


def lift_t(gt):
    z0 = np.median(gt[:10, 2])
    m = gt[:, 2] > z0 + 0.03
    c = np.convolve(m.astype(int), np.ones(5, int), "valid")
    ix = np.where(c == 5)[0]
    return int(ix[0]) if len(ix) else len(gt)


def first_close(gc):
    ix = np.where(np.asarray(gc) > 0)[0]
    return int(ix[0]) if len(ix) else len(gc)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    only = sys.argv[2] if len(sys.argv) > 2 else None

    # ---------- A: 仿射标定 ----------
    if mode in ("all", "calib"):
        pairs = {s: ([], []) for s in SETS}
        for sname, cfg in SETS.items():
            est = TargetEstimator(cfg["cls"], res=224)
            est.pix2w = None
            for i in cfg["inits"]:
                for c in cfg["conds"]:
                    tag = f"init{i:02d}_{c}"
                    tp = R / f"{cfg['prefix']}_{tag}_vanilla.npz"
                    if not tp.exists():
                        continue
                    d = np.load(tp, allow_pickle=True)
                    gt = np.asarray(d["tgt_pos"], dtype=float)
                    meta = json.loads(str(d["meta"]))
                    vp = vid_path(cfg, tag, "vanilla", meta["grasp_success"])
                    if not vp.exists():
                        continue
                    fs = frames(vp)
                    lt = lift_t(gt)
                    for t in range(0, min(lt, len(fs))):
                        e, dbg = est.estimate(fs[t])
                        if not dbg["miss"] and dbg.get("px") is not None:
                            pairs[sname][0].append(dbg["px"])
                            pairs[sname][1].append(gt[t, :2])
            P = np.asarray(pairs[sname][0], dtype=float)
            A = np.column_stack([P, np.ones(len(P))])
            W = np.asarray(pairs[sname][1], dtype=float)
            sol, res, *_ = np.linalg.lstsq(A, W, rcond=None)
            pred = A @ sol
            err = np.linalg.norm(pred - W, axis=1)
            json.dump({"A": sol.T.tolist(),
                       "n": len(err), "rmse": float(np.sqrt((err ** 2).mean())),
                       "p95": float(np.percentile(err, 95)),
                       "note": "px224底中点 -> 世界XY 仿射，标定于 vanilla 未抬起段"},
                      open(PIX2W_PATH.with_name(f"pix2w_{sname}.json"), "w"))
            print(f"[calib {sname}] n={len(err)} rmse={np.sqrt((err**2).mean()):.4f} "
                  f"p95={np.percentile(err,95):.4f} max={err.max():.4f}")
        return

    # ---------- B: 验证 ----------
    for sname, cfg in SETS.items():
        if only and sname != only:
            continue
        pj = json.load(open(PIX2W_PATH.with_name(f"pix2w_{sname}.json")))
        rows = []
        for arm in ("vanilla", "gate"):
            for i in cfg["inits"]:
                for c in cfg["conds"]:
                    tag = f"init{i:02d}_{c}"
                    tp = R / f"{cfg['prefix']}_{tag}_{arm}.npz"
                    if not tp.exists():
                        continue
                    d = np.load(tp, allow_pickle=True)
                    meta = json.loads(str(d["meta"]))
                    vp = vid_path(cfg, tag, arm, meta["grasp_success"])
                    if not vp.exists():
                        continue
                    a = {k: np.asarray(d[k], dtype=float)
                         for k in d.files if k != "meta"}
                    fs = frames(vp)
                    est = TargetEstimator(cfg["cls"], res=224,
                                          affine_path=PIX2W_PATH.with_name(
                                              f"pix2w_{sname}.json"))
                    gt = a["tgt_pos"]; lt = lift_t(gt); fc = first_close(a["gripper_cmd"])
                    errs_ap, errs_pc, zerr, miss, held = [], [], [], 0, 0
                    est_seq = []
                    for t in range(min(len(fs), len(gt))):
                        e, dbg = est.estimate(fs[t])
                        est_seq.append(None if e is None else e.copy())
                        if dbg.get("held"):
                            held += 1
                        if t < lt:
                            if dbg["miss"] and not dbg.get("held"):
                                miss += 1
                            if e is not None:
                                errs_ap.append(np.linalg.norm(e[:2] - gt[t, :2]))
                                zerr.append(e[2] - gt[t, 2])
                            if fc - 10 <= t < fc and e is not None:
                                errs_pc.append(np.linalg.norm(e[:2] - gt[t, :2]))
                    # 触发回放（gate 轨迹，eef/raw/cmd 用日志，tgt 用估计）
                    flips = 0
                    if arm == "gate" and "raw_action" in a:
                        raw = a["raw_action"]; ex = a["exec_action"]
                        eef = a["eef_pos"]; ca = a["corr_active"]
                        st = ReplayEst(); last = -1.0
                        for t in range(min(len(raw), len(est_seq))):
                            ip = max(0, t - 1)
                            act, _, _ = st.update(
                                eef[ip, 2], eef[ip, :2],
                                est_seq[ip] if ip < len(est_seq) else None,
                                last < 0)
                            if t < len(ca) and act != int(ca[t]):
                                flips += 1
                            last = float(ex[t, -1])
                    rows.append(dict(set=sname, tag=tag, arm=arm,
                                     lift=lt, close=fc, n=min(len(fs), len(gt)),
                                     ap_med=float(np.median(errs_ap)) if errs_ap else np.nan,
                                     ap_p95=float(np.percentile(errs_ap, 95)) if errs_ap else np.nan,
                                     pc_med=float(np.median(errs_pc)) if errs_pc else np.nan,
                                     z_bias=float(np.median(zerr)) if zerr else np.nan,
                                     miss=miss, held=held, flips=flips))
        vn = [r for r in rows if r["arm"] == "vanilla"]
        print(f"\n===== {sname} =====")
        print(f"vanilla n={len(vn)} 接近段XY中位 {np.nanmedian([r['ap_med'] for r in vn]):.4f} "
              f"p95 {np.nanmedian([r['ap_p95'] for r in vn]):.4f} "
              f"闭合前XY中位 {np.nanmedian([r['pc_med'] for r in vn]):.4f} "
              f"z_bias {np.nanmedian([r['z_bias'] for r in vn]):+.4f}")
        tot_miss = sum(r["miss"] for r in vn); tot_held = sum(r["held"] for r in vn)
        tot_steps = sum(r["lift"] for r in vn)
        print(f"漏检(接近段) {tot_miss} 步/总 {tot_steps} 步 = {tot_miss/max(tot_steps,1):.3f}; 沿用 {tot_held}")
        g = [r for r in rows if r["arm"] == "gate"]
        print(f"gate n={len(g)} 触发翻转步: 总 {sum(r['flips'] for r in g)}; "
              f"回合非零 {sum(r['flips']>0 for r in g)}/{len(g)}")
        worst = sorted(g, key=lambda r: -r["flips"])[:10]
        for r in worst:
            if r["flips"]:
                print(f"  {r['tag']:20s} flips={r['flips']} miss={r['miss']} held={r['held']} "
                      f"ap_med={r['ap_med']:.3f}")
        json.dump(rows, open(R / f"vision_eval_{sname}.json", "w"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
