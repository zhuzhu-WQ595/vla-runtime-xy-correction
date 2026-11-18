"""vision_target.py — 视觉目标中心估计（阶段三 A，冻结版）。

部署输入（无真值、无深度）：
- agentview RGB 帧（raw 朝向，任意分辨率，内部统一 224）
- 静态相机位姿 + FOV（一次读取）
- 任务生成区先验 WORKSPACE_RECT（目标物体初始放置区，任务知识）
- 类模板 PNG + 类参数（z_prior 静止质心高、分数阈值）——物体专属信息
- 像素→世界仿射 PIX2W（桌面共面标定，数据对一次拟合）

估计口径（冻结）：
- RGB NCC top-K 候选 → 每个底中点经理论反投影进 WORKSPACE_RECT 过滤
- 存活候选取分数最高；底中点经 PIX2W 仿射 → est XY；est Z = z_prior
- 无存活候选 → miss=None（调用方持用最近估计 ≤LAST_HOLD 步后抑制触发）
- 每步独立估计，无时间滤波
"""
import json
from pathlib import Path

import cv2
import numpy as np

CAM_POS = np.array([8.96577372e-01, 5.21618273e-07, 6.50000000e-01])
CAM_R = np.array([[-1.72339050e-06, -5.28769744e-01, 8.48765314e-01],
                  [1.00000000e+00, -7.82314965e-07, 1.54309560e-06],
                  [-1.51940455e-07, 8.48765314e-01, 5.28769744e-01]])
FOVY_DEG = 45.0
CALIB_RES = 256.0
EST_RES = 224
WORKSPACE_RECT = (-0.20, 0.10, -0.32, -0.10)

CLASSES = {
    "butter": dict(template="tmpl_butter.png", z_prior=0.0087,
                   score_min=0.55, k=4),
    "milk": dict(template="tmpl_milk.png", z_prior=0.0697,
                 score_min=0.55, k=4),
}
Z_CONTACT = 0.003
LAST_HOLD = 5
PIX2W_PATH = Path(__file__).parent.parent / "runs" / "predictability" / "pix2w_affine.json"


def _focal(res):
    return 0.5 * res / np.tan(np.deg2rad(FOVY_DEG / 2))


def unproject(px, py, res, plane_z=Z_CONTACT):
    f = _focal(res)
    d_cam = np.array([(px - res / 2) / f, -(py - res / 2) / f, -1.0])
    d_w = CAM_R @ d_cam
    if d_w[2] > -1e-4:
        return None
    s = (plane_z - CAM_POS[2]) / d_w[2]
    if s <= 0:
        return None
    return CAM_POS[:2] + s * d_w[:2]


class TargetEstimator:
    def __init__(self, cls, res=224, affine_path=None):
        self.p = CLASSES[cls]
        self.res = res
        tp = Path(__file__).parent.parent / "runs" / "predictability" / self.p["template"]
        self.tmpl = cv2.imread(str(tp), cv2.IMREAD_COLOR)
        if self.tmpl is None:
            raise FileNotFoundError(tp)
        self.th, self.tw = self.tmpl.shape[:2]
        if affine_path is None:
            affine_path = PIX2W_PATH.with_name(f"pix2w_{cls}.json")
        if not Path(affine_path).exists():
            raise FileNotFoundError(
                f"pix2w affine missing: {affine_path} "
                "(run scripts/vision_offline_eval.py calib)")
        self.pix2w = np.array(json.load(open(affine_path))["A"])
        self.last_est = None
        self.last_age = 10 ** 9

    def estimate(self, rgb_raw):
        """返回 (est_xyz or None, dbg)。rgb_raw 为 raw 朝向 RGB。"""
        img = cv2.resize(rgb_raw, (EST_RES, EST_RES),
                         interpolation=cv2.INTER_AREA) \
            if rgb_raw.shape[0] != EST_RES else rgb_raw
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.float32)
        res = cv2.matchTemplate(img, self.tmpl.astype(np.float32),
                                cv2.TM_CCOEFF_NORMED)
        r = res.mean(axis=2) if res.ndim == 3 else res
        r = r.copy()
        cands = []
        for _ in range(self.p["k"]):
            _, mx, _, ml = cv2.minMaxLoc(r)
            if mx < self.p["score_min"]:
                break
            cx = ml[0] + self.tw / 2
            bx, by = cx, ml[1] + self.th - 1          # 底中点
            xy = unproject(bx / EST_RES * CALIB_RES,
                           by / EST_RES * CALIB_RES, CALIB_RES)
            inside = (xy is not None
                      and WORKSPACE_RECT[0] <= xy[0] <= WORKSPACE_RECT[1]
                      and WORKSPACE_RECT[2] <= xy[1] <= WORKSPACE_RECT[3])
            cands.append(dict(score=float(mx), px=(bx, by),
                              xy=None if xy is None else xy.tolist(),
                              inside=bool(inside)))
            r[max(0, ml[1] - 8):ml[1] + 8, max(0, ml[0] - 8):ml[0] + 8] = -1
        good = [c for c in cands if c["inside"]]
        if not good:
            self.last_age += 1
            if self.last_est is not None and self.last_age <= LAST_HOLD:
                return self.last_est, dict(miss=True, held=True, cands=cands)
            return None, dict(miss=True, held=False, cands=cands)
        best = max(good, key=lambda c: c["score"])
        bx, by = best["px"]
        if self.pix2w is not None:
            est_xy = self.pix2w @ np.array([bx, by, 1.0])
        else:
            est_xy = np.array(best["xy"])
        est = np.array([est_xy[0], est_xy[1], self.p["z_prior"]])
        self.last_est, self.last_age = est, 0
        return est, dict(miss=False, held=False, score=best["score"],
                         px=(bx, by), cands=cands)
