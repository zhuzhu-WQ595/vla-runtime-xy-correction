"""frozen_review.py — 阶段三 B 第一步：离线复核（只看已有数据，不跑仿真）。

1) 从 perception_transfer_{set}.csv 重生成平移主表、oracle/visual 救回交差集、
   各终点破坏率；以 init 为簇的 bootstrap 配对区间（抓取与入篮分列）。
2) butter trans_y 丢失救回 7 例 + milk 漏检关联 6 例：
   gate↔visgate 首次实质动作分歧步（XY 差 > TOL=5mm，另报 2mm 灵敏度），
   共同前缀上的估计误差（fresh/held/miss-fallback 分列）、corr_active、exec XY；
   分歧归因分类；前缀分叉后的误差另列。

visgate 日志口径：vis_miss=1 → est_tgt 为 eef 回退；vis_miss=0 且 vis_score=NaN
→ held 沿用估计；vis_miss=0 且 vis_score 有限 → 当步原始估计。
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
R = BASE / "results"
SETS = {"butter": "traj2", "milk": "traj3"}
TRANS = ("trans", "trans_y")
# exec_action 为 raw 动作单位，实测映射 1 raw ≈ 0.0128 m（runner 注释）
RAW2M = 0.0128
TOL_MM = 5.0 / 12.8      # 实质动作分歧阈值 ≈5mm（raw 0.39）
TOL_SENS = 2.0 / 12.8    # 灵敏度 ≈2mm（raw 0.156）
RNG = np.random.default_rng(0)

BUTTER_TY_LOST = [f"init{i:02d}_trans_y" for i in (36, 37, 41, 43, 45, 46, 47)]
BUTTER_T_LOST = ["init34_trans", "init37_trans"]   # 平移丢失救回补齐（trans 方向）
MILK_MISS_CASES = ["init02_rot", "init05_rot", "init06_trans_y",
                   "init09_trans_y", "init11_trans_y", "init12_trans_y"]


def load_csv(s):
    return list(csv.DictReader(open(R / f"perception_transfer_{s}.csv")))


def b(x):
    return x in ("True", "1", "true")


# ---------------- 1) 主表 + 配对区间 ----------------

def main_table(rows):
    out = {}
    for s, rr in rows.items():
        for scope, conds in (("ALL", ("orig", "trans", "trans_y", "rot")),
                             ("TRANS", TRANS)):
            sub = [r for r in rr if r["cond"] in conds]
            key = lambda r: tuple(b(r[k]) for k in ("v_g", "g_g", "x_g"))
            o_rescue = [r for r in sub if not b(r["v_g"]) and b(r["g_g"])]
            kept = [r["tag"] for r in o_rescue if b(r["x_g"])]
            lost = [r["tag"] for r in o_rescue if not b(r["x_g"])]
            new = [r["tag"] for r in sub if not b(r["v_g"]) and not b(r["g_g"])
                   and b(r["x_g"])]
            xbrk = [r["tag"] for r in sub if b(r["v_g"]) and not b(r["x_g"])]
            gbrk = [r["tag"] for r in sub if b(r["v_g"]) and not b(r["g_g"])]
            t_or = [r for r in sub if not b(r["v_t"]) and b(r["g_t"])]
            t_kept = [r["tag"] for r in t_or if b(r["x_t"])]
            t_lost = [r["tag"] for r in t_or if not b(r["x_t"])]
            t_new = [r["tag"] for r in sub if not b(r["v_t"]) and not b(r["g_t"])
                     and b(r["x_t"])]
            t_xbrk = [r["tag"] for r in sub if b(r["v_t"]) and not b(r["x_t"])]
            out[f"{s}:{scope}"] = dict(
                n=len(sub),
                grasp=dict(v=sum(b(r["v_g"]) for r in sub),
                           g=sum(b(r["g_g"]) for r in sub),
                           x=sum(b(r["x_g"]) for r in sub),
                           oracle_rescued=len(o_rescue), kept=len(kept),
                           lost=lost, new=new, x_broke=xbrk, g_broke=gbrk),
                task=dict(v=sum(b(r["v_t"]) for r in sub),
                          g=sum(b(r["g_t"]) for r in sub),
                          x=sum(b(r["x_t"]) for r in sub),
                          oracle_rescued=len(t_or), kept=len(t_kept),
                          lost=t_lost, new=t_new, x_broke=t_xbrk),
                # 终点破坏率：破坏数 / 原成功数
                break_rate_grasp=len(xbrk) / max(1, sum(b(r["v_g"]) for r in sub)),
                break_rate_task=len(t_xbrk) / max(1, sum(b(r["v_t"]) for r in sub)),
            )
    return out


def paired_ci(rows, conds=None):
    """以 init 为簇的 bootstrap 95% CI：Δx-v 与 Δx-g，抓取/入篮分列。"""
    res = {}
    for s, rr0 in rows.items():
        rr = [r for r in rr0 if conds is None or r["cond"] in conds]
        by_init = {}
        for r in rr:
            by_init.setdefault(r["tag"].split("_")[0], []).append(r)
        inits = sorted(by_init)
        out = {}
        for ep in ("g", "t"):        # g=抓取, t=入篮
            for arm in ("x_v", "x_g"):
                a, bb = arm.split("_")
                ds = np.array([
                    [b(r[f"{a}_{ep}"]) - b(r[f"{bb}_{ep}"])
                     for r in by_init[i]] for i in inits], dtype=object)
                pt = float(np.mean([np.mean(d) for d in ds]))
                boots = []
                for _ in range(2000):
                    samp = RNG.choice(len(inits), len(inits), replace=True)
                    boots.append(np.mean([np.mean(ds[j]) for j in samp]))
                lo, hi = np.percentile(boots, [2.5, 97.5])
                out[f"{a}-{bb}_{ep}"] = (round(pt, 3), round(float(lo), 3),
                                         round(float(hi), 3))
        res[s] = out
    return res


# ---------------- 2) 分歧分析 ----------------

def load_npz(prefix, tag, arm):
    d = np.load(R / f"{prefix}_{tag}_{arm}.npz", allow_pickle=True)
    out = {}
    for k in d.files:
        if k == "meta":
            continue
        a = np.asarray(d[k])
        try:
            a = a.astype(float)
        except (TypeError, ValueError):
            pass
        out[k] = a
    out["meta"] = json.loads(str(d["meta"]))
    return out


def est_kind(v):
    """返回逐步估计类别数组：fresh/held/miss_fallback。"""
    miss = v["vis_miss"].astype(int)
    score = v["vis_score"].astype(float)
    k = np.where(miss == 1, "miss_fb",
                 np.where(np.isnan(score), "held", "fresh"))
    return k


def divergence_case(prefix, tag):
    g = load_npz(prefix, tag, "gate")
    v = load_npz(prefix, tag, "visgate")
    n = min(len(g["exec_action"]), len(v["exec_action"]))
    dxy = np.linalg.norm(g["exec_action"][:n, :2] - v["exec_action"][:n, :2],
                         axis=1)
    t0 = next((t for t in range(n) if dxy[t] > 1e-6), None)   # 首个执行差=因果锚点
    t5 = next((t for t in range(n) if dxy[t] > TOL_MM), None)
    t2 = next((t for t in range(n) if dxy[t] > TOL_SENS), None)
    tdiv = t0 if t0 is not None else (t5 if t5 is not None else n)
    # 前缀统计
    e = v["est_tgt"][:tdiv]; gt = v["tgt_pos"][:tdiv]
    kind = est_kind(v)[:tdiv]
    exy = np.linalg.norm(e[:, :2] - gt[:, :2], axis=1)
    ez = e[:, 2] - gt[:, 2]
    miss_first = next((t for t in range(len(v["vis_miss"]))
                       if v["vis_miss"][t] == 1), None)
    pref = dict(
        fresh_exy_med=float(np.median(exy[kind == "fresh"])) if (kind == "fresh").any() else None,
        held_exy_med=float(np.median(exy[kind == "held"])) if (kind == "held").any() else None,
        miss_steps=int((kind == "miss_fb").sum()),
        held_steps=int((kind == "held").sum()),
        ez_med=float(np.median(ez[kind != "miss_fb"])) if (kind != "miss_fb").any() else None,
        g_active=int(g["corr_active"][:tdiv].sum()),
        v_active=int(v["corr_active"][:tdiv].sum()),
    )
    # 分叉后误差（另列）
    post = None
    if tdiv < len(v["est_tgt"]):
        e2 = v["est_tgt"][tdiv:]; gt2 = v["tgt_pos"][tdiv:]
        k2 = est_kind(v)[tdiv:]
        ok = k2 != "miss_fb"
        if ok.any():
            post = dict(exy_med=float(np.median(np.linalg.norm(
                e2[ok, :2] - gt2[ok, :2], axis=1))),
                steps=int(ok.sum()))
    # 归因分类（分歧步±5步窗口证据）
    w0, w1 = max(0, tdiv - 5), min(n, tdiv + 6)
    kw = est_kind(v)[w0:w1]
    g_on = g["corr_active"][w0:w1].astype(bool)
    v_on = v["corr_active"][w0:w1].astype(bool)
    t_vfirst = next((t for t in range(len(v["corr_active"]))
                     if v["corr_active"][t] > 0), None)
    t_gfirst = next((t for t in range(len(g["corr_active"]))
                     if g["corr_active"][t] > 0), None)
    if (kw == "miss_fb").any():
        cause = "漏检/回退"
    elif (kw == "held").any():
        cause = "沿用旧估计"
    elif g_on.any() and not v_on.any():
        cause = "触发时机不同"
    elif g_on.any() and v_on.any():
        cause = "均触发但修正向量不同"
    elif t_gfirst is not None and (t_vfirst is None or t_vfirst > t_gfirst):
        cause = "触发时机不同"
    else:
        cause = "未定"
    return dict(tag=tag, t0=t0, tdiv_5mm=t5, tdiv_2mm=t2, prefix=pref,
                first_miss=miss_first, miss_before_div=(
                    miss_first is not None and (t0 is None or miss_first <= t0)),
                cause=cause, post=post,
                t_gfirst=t_gfirst, t_vfirst=t_vfirst,
                dxy_at_div=float(dxy[t5]) if t5 is not None else float(dxy.max()))


def main():
    rows = {s: load_csv(s) for s in SETS}
    out = {"main": main_table(rows), "ci": paired_ci(rows)}

    print("===== 平移主表（trans+trans_y 合并） =====")
    for s in SETS:
        m = out["main"][f"{s}:TRANS"]
        print(f"{s} n={m['n']} grasp v/g/x={m['grasp']['v']}/{m['grasp']['g']}/{m['grasp']['x']} "
              f"oracle救回{m['grasp']['oracle_rescued']} 保留{m['grasp']['kept']} "
              f"丢失{m['grasp']['lost']} 新增{m['grasp']['new']} "
              f"x破坏{m['grasp']['x_broke']}(率{m['break_rate_grasp']:.3f}) "
              f"g破坏{m['grasp']['g_broke']}")
        print(f"  task v/g/x={m['task']['v']}/{m['task']['g']}/{m['task']['x']} "
              f"oracle救回{m['task']['oracle_rescued']} 保留{m['task']['kept']} "
              f"丢失{m['task']['lost']} 新增{m['task']['new']} "
              f"x破坏{m['task']['x_broke']}(率{m['break_rate_task']:.3f})")

    print("\n===== 配对区间（init 簇 bootstrap, 全条件） =====")
    for s in SETS:
        for k, (pt, lo, hi) in out["ci"][s].items():
            print(f"  {s} Δ{k}: {pt:+.3f} [{lo:+.3f},{hi:+.3f}]")

    print("\n===== butter trans_y 丢失救回分歧分析 =====")
    cases_b = [divergence_case(SETS["butter"], t)
               for t in BUTTER_TY_LOST + BUTTER_T_LOST]
    for c in cases_b:
        print(f"  {c['tag']}: t0={c['t0']} div5={c['tdiv_5mm']} div2={c['tdiv_2mm']} "
              f"cause={c['cause']} g_on@{c['t_gfirst']} v_on@{c['t_vfirst']} "
              f"prefix={c['prefix']} post={c['post']}")

    print("\n===== milk 漏检关联病例分歧分析 =====")
    cases_m = [divergence_case(SETS["milk"], t) for t in MILK_MISS_CASES]
    for c in cases_m:
        print(f"  {c['tag']}: t0={c['t0']} div5={c['tdiv_5mm']} div2={c['tdiv_2mm']} "
              f"cause={c['cause']} miss_first={c['first_miss']} "
              f"miss_before_div={c['miss_before_div']} "
              f"prefix={c['prefix']} post={c['post']}")

    out["cases"] = {"butter_trans_y_lost": cases_b[:7],
                    "butter_trans_lost": cases_b[7:],
                    "milk_miss": cases_m}
    out["ci_trans"] = paired_ci(rows, conds=set(TRANS))
    print("\n===== 配对区间（init 簇 bootstrap, 仅 trans+trans_y） =====")
    for s in SETS:
        for k, v in out["ci_trans"][s].items():
            print(f"  {s} Δ{k}: {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]")
    json.dump(out, open(R / "frozen_review.json", "w"),
              ensure_ascii=False, indent=1, default=str)
    print("\nwrote results/frozen_review.json")


if __name__ == "__main__":
    sys.exit(main())
