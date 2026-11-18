"""analyze_action_select.py — 阶段四 B：离线评估决策步执行前能否选择局部抓取动作。

全部用四 A 已跑分支（不新增仿真）：
P1 从 npz 重生成固定候选/按物体候选成绩、{c0,px}/{c0,px,ry}/五候选逐状态
   择优上限；120 对（30状态×4候选）共同前缀审计（含无触发全程等价），
   解释 60/60 分母；每候选在 闭合前一步/闭合当步/闭合+2 的配对位姿差。
P2 c0 vs px：四配对标签；决策步前特征 = 锁定视觉目标相对夹爪中点在
   闭合轴 u 上的有符号误差 e，及近 K 步沿 u 的接近运动。简单规则 R1：
   e>θ→px else c0（θ 仅训练折选）。真值/扰动标签/闭合后信息仅诊断。
P3 按 物体-init 15 组 LOGO CV：θ、全局固定、按物体固定均训练折选；
   留出折逐例计数（抓取/入篮/救回/破坏/有效分歧）。
P4 ry 探索：相对 c0/px 新增覆盖的抓取病例 + ry 破坏病例的执行前差异。

输出 results/action_select_summary.json + action_select_states.csv
"""
import csv
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
R = BASE / "results"
CANDS = ["c0", "px", "nx", "ry", "ly"]
NONC0 = CANDS[1:]
SETS = {"butter": dict(pre="f4b", inits=[30, 34, 37, 40, 43, 44, 46]),
        "milk": dict(pre="f4m", inits=[0, 1, 4, 6, 9, 12, 15, 17])}
CONDS = ["trans", "trans_y"]
EPS = ["g", "t"]
K = 5                    # 接近运动回看步数
THETA_GRID = np.arange(0, 0.021, 0.001)   # 0..20mm
TE_GRID = np.arange(0, 0.041, 0.002)      # |e| 阈值 0..40mm
TA_GRID = [-np.inf] + list(np.arange(-0.02, 0.021, 0.005))  # 接近守卫 mm


def yaw(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load(pre, tag, cand):
    p = R / f"{pre}_{tag}_pb_{cand}.npz"
    d = np.load(p, allow_pickle=True)
    a = {k: np.asarray(d[k]) for k in d.files if k != "meta"}
    for k in a:
        try:
            a[k] = a[k].astype(float)
        except (TypeError, ValueError):
            pass
    return a, json.loads(str(d["meta"]))


def rel_delta(base, cand_a, u, t):
    """候选 vs c0 在绝对步 t 的配对位姿差：du(mm 沿u), dxy(mm), dyaw(deg)。"""
    n = min(len(base["eef_pos"]), len(cand_a["eef_pos"]))
    if t is None or t >= n or t < 0:
        return None
    d = cand_a["eef_pos"][t] - base["eef_pos"][t]
    dy = np.degrees(wrap(yaw(cand_a["eef_quat"][t]) - yaw(base["eef_quat"][t])))
    return dict(du=round(float(d[:2] @ u) * 1000, 1),
                dxy=round(float(np.linalg.norm(d[:2])) * 1000, 1),
                dyaw=round(float(dy), 1))


def main():
    states = []        # 每状态：标签/特征/各候选结局
    audit = []         # 配对审计 120 对
    for sname, cfg in SETS.items():
        for i in cfg["inits"]:
            for cond in CONDS:
                tag = f"init{i:02d}_{cond}"
                trajs, metas = {}, {}
                for cd in CANDS:
                    trajs[cd], metas[cd] = load(cfg["pre"], tag, cd)
                m0 = metas["c0"]
                t0 = m0["cand_step"]          # 决策步（0-based）
                u = np.asarray(m0["u_vec"]) if m0.get("u_vec") else None
                fc = m0.get("first_close_t")   # 1-based
                row = dict(obj=sname, tag=tag, init=i, cond=cond,
                           cand_step=t0, first_close=fc,
                           no_trigger=t0 is None)
                for cd in CANDS:
                    for e in EPS:
                        key = "grasp_success" if e == "g" else "task_success"
                        row[f"{e}_{cd}"] = int(bool(metas[cd][key]))
                # ---- P1 配对审计：首个 exec_action 分歧 vs cand_step ----
                e0 = trajs["c0"]["exec_action"].astype(float)
                for cd in NONC0:
                    ec = trajs[cd]["exec_action"].astype(float)
                    n = min(len(e0), len(ec))
                    diff = np.abs(e0[:n] - ec[:n]).max(1)
                    tdiv = (int(np.argmax(diff > 1e-9))
                            if (diff > 1e-9).any() else None)
                    cs = metas[cd]["cand_step"]
                    audit.append(dict(
                        tag=tag, cand=cd, cand_step=cs, tdiv=tdiv,
                        ok=(tdiv == cs)
                        or (tdiv is None and cs is None)
                        or (tdiv is None and cs is not None),  # 无分歧=等价
                        identical=tdiv is None))
                    # ---- 三时点 achieved（候选 run 的 first_close）----
                    fcc = metas[cd].get("first_close_t")
                    for lbl, tt in (("pre", (fcc or 0) - 1),
                                    ("at", fcc), ("post", (fcc or 0) + 2)):
                        dlt = rel_delta(trajs["c0"], trajs[cd],
                                        u if u is not None else
                                        np.array([1.0, 0.0]), tt)
                        for kk, vv in (dlt or {}).items():
                            row[f"{cd}_{lbl}_{kk}"] = vv
                # ---- P2 决策步前特征（c0 轨迹，= 各候选共同前缀）----
                if t0 is not None:
                    fm = trajs["c0"]["finger_mid"].astype(float)
                    est = trajs["c0"]["est_tgt"].astype(float)
                    # e(t) = (locked_xy - finger_mid_xy)·u，t0 为决策步
                    e_ts = (est[:, :2] - fm[:, :2]) @ u
                    row["e_at_dec_mm"] = round(float(e_ts[t0]) * 1000, 1)
                    tp = max(0, t0 - K)
                    row["approach_mm"] = round(
                        float(e_ts[t0] - e_ts[tp]) * -1000, 1)  # >0 沿u接近
                    row["e_series"] = None
                    # 诊断用真值（不用于规则）
                    gt = trajs["c0"]["tgt_pos"].astype(float)
                    row["e_gt_mm"] = round(
                        float((gt[t0, :2] - fm[t0, :2]) @ u) * 1000, 1)
                states.append(row)

    # ============ P1 汇总 ============
    p1 = {}
    # 固定候选成绩（按物体、合计）
    for e in EPS:
        p1[e] = {}
        for sname in SETS:
            rr = [r for r in states if r["obj"] == sname]
            p1[e][sname] = {cd: sum(r[f"{e}_{cd}"] for r in rr) for cd in CANDS}
        p1[e]["all"] = {cd: sum(r[f"{e}_{cd}"] for r in states) for cd in CANDS}
        # 逐状态择优上限
        for cset_name, cset in (("c0_px", ["c0", "px"]),
                                ("c0_px_ry", ["c0", "px", "ry"]),
                                ("all5", CANDS)):
            p1[e][cset_name] = {
                sname: sum(max(r[f"{e}_{cd}"] for cd in cset)
                           for r in states if r["obj"] == sname)
                for sname in SETS}
            p1[e][cset_name]["all"] = sum(
                max(r[f"{e}_{cd}"] for cd in cset) for r in states)
    # 配对审计
    n_ok = sum(a["ok"] for a in audit)
    n_id = sum(a["identical"] for a in audit)
    audit_bad = [a for a in audit if not a["ok"]]
    p1["audit"] = dict(n_pairs=len(audit), n_ok=n_ok, n_identical=n_id,
                       bad=audit_bad,
                       note="原报告 60/60 分母=30状态×{px,nx}两候选；"
                            "本次 120=30×4，含无触发等价对")

    # ============ P2 c0 vs px 标签 ============
    for r in states:
        for e in EPS:
            b, p = r[f"{e}_c0"], r[f"{e}_px"]
            r[f"{e}_label"] = ("px_saves" if (not b and p) else
                               "px_breaks" if (b and not p) else
                               "both_ok" if b else "both_fail")

    # ============ P3 LOGO CV ============
    groups = [(s, i) for s, cfg in SETS.items() for i in cfg["inits"]]
    assert len(groups) == 15
    cv_rows, cv_sum = [], []
    for gs, gi in groups:
        train = [r for r in states if not (r["obj"] == gs and r["init"] == gi)]
        held = [r for r in states if r["obj"] == gs and r["init"] == gi]
        for e in EPS:
            # 规则族：px iff s·e>θe ∧ approach>θa（方向/阈值均训练折选）
            def pick_px(r, s, te, ta):
                if r["no_trigger"] or r.get("e_at_dec_mm") is None:
                    return False
                return (s * r["e_at_dec_mm"] > te * 1000
                        and r["approach_mm"] > ta * 1000)

            def rule_outcome(rr, s, te, ta):
                return sum(r[f"{e}_px"] if pick_px(r, s, te, ta)
                           else r[f"{e}_c0"] for r in rr)
            cands = []
            for s in (1, -1):
                for te in TE_GRID:
                    for ta in TA_GRID:
                        sc = rule_outcome(train, s, te, ta)
                        npx = sum(pick_px(r, s, te, ta) for r in train)
                        cands.append((sc, -npx, ta == -np.inf, s, te, ta))
            cands.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            _, _, _, s_sel, te_sel, ta_sel = cands[0]
            # 基线：全局固定 / 按物体固定（训练折）
            gf = max(CANDS, key=lambda cd: sum(r[f"{e}_{cd}"] for r in train))
            tr_obj = [r for r in train if r["obj"] == gs]
            po = max(CANDS, key=lambda cd: sum(r[f"{e}_{cd}"] for r in tr_obj))
            for r in held:
                pick = "px" if pick_px(r, s_sel, te_sel, ta_sel) else "c0"
                cv_rows.append(dict(
                    obj=gs, tag=r["tag"], endpoint=e,
                    rule=f"s={s_sel},te={te_sel*1000:.0f}mm,ta={ta_sel*1000:.0f}mm",
                    rule_pick=pick, rule_out=r[f"{e}_{pick}"],
                    e_at_dec_mm=r.get("e_at_dec_mm"), c0=r[f"{e}_c0"],
                    global_fixed=gf, gf_out=r[f"{e}_{gf}"],
                    per_object=po, po_out=r[f"{e}_{po}"],
                    label=r[f"{e}_label"]))
    # 留出折计数
    for e in EPS:
        rr = [r for r in cv_rows if r["endpoint"] == e]
        tot = dict(rule=sum(r["rule_out"] for r in rr),
                   c0=sum(r["c0"] for r in rr),
                   global_fixed=sum(r["gf_out"] for r in rr),
                   per_object=sum(r["po_out"] for r in rr))
        saves = [r["tag"] for r in rr if r["c0"] == 0 and r["rule_out"] == 1]
        breaks = [r["tag"] for r in rr if r["c0"] == 1 and r["rule_out"] == 0]
        vs_gf = [r["tag"] for r in rr if r["rule_out"] != r["gf_out"]]
        vs_po = [r["tag"] for r in rr if r["rule_out"] != r["po_out"]]
        eff = [r["tag"] for r in rr
               if r["rule_pick"] != "c0" and r["rule_out"] != r["c0"]]
        cv_sum.append(dict(endpoint=e, n=len(rr), totals=tot,
                           saves_vs_c0=saves, breaks_vs_c0=breaks,
                           rule_neq_gf=vs_gf, rule_neq_po=vs_po,
                           effective_divergence=eff))

    # ============ P4 ry 探索 ============
    ry_new, ry_break = [], []
    for r in states:
        if r["g_c0"] == 0 and r["g_px"] == 0 and r["g_ry"] == 1:
            ry_new.append(r["tag"])
        if r["g_c0"] == 1 and r["g_ry"] == 0:
            ry_break.append(r["tag"])
    feat_cols = ["e_at_dec_mm", "approach_mm", "e_gt_mm"]
    ry_feat = {c: dict(new=[{k: r.get(k) for k in feat_cols}
                            for r in states if r["tag"] in ry_new],
                       brk=[{k: r.get(k) for k in feat_cols}
                            for r in states if r["tag"] in ry_break])
               for c in ["g"]}
    # 闭合前已形成 vs 闭合窗内持续旋转：ry achieved 三时点
    ry_delta = [{ "tag": r["tag"],
                  "pre": r.get("ry_pre_dyaw"), "at": r.get("ry_at_dyaw"),
                  "post": r.get("ry_post_dyaw"),
                  "new_cover": r["tag"] in ry_new,
                  "broken": r["tag"] in ry_break}
                for r in states if r.get("ry_at_dyaw") is not None]

    out = dict(p1=p1,
               pair_labels={e: {l: [r["tag"] for r in states
                                    if r[f"{e}_label"] == l]
                                for l in ("px_saves", "px_breaks",
                                          "both_ok", "both_fail")}
                            for e in EPS},
               cv=cv_sum, cv_rows=cv_rows,
               ry=dict(new_over_c0px=ry_new, breaks=ry_break,
                       feat=ry_feat, delta3=ry_delta),
               no_trigger=[r["tag"] for r in states if r["no_trigger"]])
    json.dump(out, open(R / "action_select_summary.json", "w"),
              ensure_ascii=False, indent=1, default=str)

    # 逐状态 CSV（含特征/标签/achieved 三时点）
    cols = (["obj", "tag", "init", "cond", "cand_step", "first_close",
             "no_trigger", "e_at_dec_mm", "approach_mm", "e_gt_mm",
             "g_label", "t_label"]
            + [f"{e}_{cd}" for e in EPS for cd in CANDS]
            + [f"{cd}_{p}_{k}" for cd in NONC0 for p in ("pre", "at", "post")
               for k in ("du", "dxy", "dyaw")])
    with open(R / "action_select_states.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(states)
    # 留出决策表
    with open(R / "action_select_cv.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cv_rows[0].keys()))
        w.writeheader()
        w.writerows(cv_rows)

    # ---- 打印摘要 ----
    print("=== P1 audit:", len(audit), "pairs | ok", n_ok,
          "| identical", n_id, "| bad", len(audit_bad))
    for e in EPS:
        print(f"[{e}] 固定候选成绩 all:", p1[e]["all"],
              "| butter:", p1[e]["butter"], "| milk:", p1[e]["milk"])
        print(f"    oracle c0_px={p1[e]['c0_px']['all']} "
              f"c0_px_ry={p1[e]['c0_px_ry']['all']} all5={p1[e]['all5']['all']}")
    for e in EPS:
        lbl = out["pair_labels"][e]
        print(f"[{e}] c0/px 标签: saves={lbl['px_saves']} "
              f"breaks={lbl['px_breaks']} both_ok={len(lbl['both_ok'])} "
              f"both_fail={len(lbl['both_fail'])}")
    for s in cv_sum:
        print(f"[{s['endpoint']}] LOGO留出 totals:", s["totals"],
              "| saves:", s["saves_vs_c0"], "| breaks:", s["breaks_vs_c0"],
              "| 有效分歧:", s["effective_divergence"])
    print("ry 新增覆盖:", ry_new)
    print("ry 破坏:", ry_break)
    print("无触发:", out["no_trigger"])


if __name__ == "__main__":
    main()
