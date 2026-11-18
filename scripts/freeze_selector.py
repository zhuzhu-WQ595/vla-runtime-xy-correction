"""freeze_selector.py — 阶段四 C：冻结 c0/px 选择器。

1) 从 action_select_states.csv 的特征+结局重算：逐状态 pick（抓取主终点）、
   按该 pick 的重算 g/t 结局、相对 c0/always-px/按物体固定 的逐例得失、干预数。
2) 在全部 30 开发状态上按抓取终点+平局规则（max g → 少干预 → ta=-inf）选
   最终 (s,θe,θa)；写 runs/predictability/action_sel_spec_*.json（含代码哈希）。
3) 导出 dev 逐状态 pick 表供 runner sel 臂在线复现验证。
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
R = BASE / "results"
CANDS = ["c0", "px"]
TE_GRID = np.arange(0, 0.041, 0.002)
TA_GRID = [-np.inf] + list(np.arange(-0.02, 0.021, 0.005))
SETS = {"butter": [30, 34, 37, 40, 43, 44, 46],
        "milk": [0, 1, 4, 6, 9, 12, 15, 17]}


def pick_px(r, s, te, ta):
    if r["no_trigger"] == "True" or not r["e_at_dec_mm"]:
        return False
    e = float(r["e_at_dec_mm"])
    ap = float(r["approach_mm"])
    return s * e > te * 1000 and ap > ta * 1000


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]


def load_features():
    """决策状态口径特征：决策检测在迭代 t0 步前发生，对应 post-step 日志
    索引 t0-1（t0=0 用 0）；approach 回看 5 步（索引 t0-6）。"""
    import glob
    pre = {"butter": "f4b", "milk": "f4m"}
    rows = []
    for obj, inits in SETS.items():
        for i in inits:
            for cond in ("trans", "trans_y"):
                tag = f"init{i:02d}_{cond}"
                r = {"obj": obj, "tag": tag, "init": i, "cond": cond}
                d = np.load(R / f"{pre[obj]}_{tag}_pb_c0.npz",
                            allow_pickle=True)
                m0 = json.loads(str(d["meta"]))
                r["cand_step"] = m0["cand_step"]
                r["no_trigger"] = str(m0["cand_step"] is None)
                if m0["cand_step"] is not None:
                    u = np.asarray(m0["u_vec"])
                    fm = np.asarray(d["finger_mid"], float)
                    est = np.asarray(d["est_tgt"], float)
                    t0 = m0["cand_step"]
                    i_dec, i_prev = max(0, t0 - 1), max(0, t0 - 6)
                    r["e_at_dec_mm"] = (est[i_dec, :2] - fm[i_dec, :2]) @ u * 1000
                    r["approach_mm"] = (fm[i_dec, :2] - fm[i_prev, :2]) @ u * 1000
                for e in ("g", "t"):
                    for cd in CANDS:
                        dm = json.loads(str(np.load(
                            R / f"{pre[obj]}_{tag}_pb_{cd}.npz",
                            allow_pickle=True)["meta"]))
                        key = ("grasp_success" if e == "g"
                               else "task_success")
                        r[f"{e}_{cd}"] = str(int(bool(dm[key])))
                rows.append(r)
    return rows


def main():
    rows = load_features()
    assert len(rows) == 30
    # ---------- 1. 全开发集冻结（g 主终点） ----------
    best = None
    for s in (1, -1):
        for te in TE_GRID:
            for ta in TA_GRID:
                sc = sum(int(r["g_px"]) if pick_px(r, s, te, ta)
                         else int(r["g_c0"]) for r in rows)
                npx = sum(pick_px(r, s, te, ta) for r in rows)
                key = (sc, -npx, ta == -np.inf)
                if best is None or key > best[0]:
                    best = (key, s, te, ta)
    _, S, TE, TA = best
    print(f"冻结规则: s={S} θe={TE*1000:.1f}mm θa={TA*1000:.1f}mm"
          f" ({'px iff e<'+str(-TE*1000)+'mm' if S<0 else 'px iff e>'+str(TE*1000)+'mm'}"
          f" ∧ approach>{TA*1000}mm)")

    # ---------- 2. 逐状态 pick + 重算 ----------
    # 按物体固定（四 B 训练折稳定值）：butter→c0，milk→px
    POBJ = {"butter": "c0", "milk": "px"}
    tab = []
    for r in rows:
        px_ = pick_px(r, S, TE, TA)
        pick = "px" if px_ else "c0"
        r["pick"] = pick
        po = POBJ[r["obj"]]
        t = dict(obj=r["obj"], tag=r["tag"], pick=pick,
                 g_out=int(r[f"g_{pick}"]), t_out=int(r[f"t_{pick}"]),
                 g_c0=int(r["g_c0"]), g_px=int(r["g_px"]), g_po=int(r[f"g_{po}"]),
                 t_c0=int(r["t_c0"]), t_px=int(r["t_px"]), t_po=int(r[f"t_{po}"]),
                 e_at_dec_mm=r.get("e_at_dec_mm"),
                 approach_mm=r.get("approach_mm"),
                 cand_step=r["cand_step"], no_trigger=r["no_trigger"])
        tab.append(t)
    n_int = sum(t["pick"] == "px" for t in tab)
    print("干预次数(px pick):", n_int, "/30")
    for e in ("g", "t"):
        tot = dict(sel=sum(t[f"{e}_out"] for t in tab),
                   c0=sum(t[f"{e}_c0"] for t in tab),
                   px=sum(t[f"{e}_px"] for t in tab),
                   pobj=sum(t[f"{e}_po"] for t in tab))
        saves = [t["tag"] for t in tab if t[f"{e}_c0"] == 0 and t[f"{e}_out"] == 1]
        brk = [t["tag"] for t in tab if t[f"{e}_c0"] == 1 and t[f"{e}_out"] == 0]
        vs_px = [t["tag"] for t in tab if t[f"{e}_out"] != t[f"{e}_px"]]
        vs_po = [t["tag"] for t in tab if t[f"{e}_out"] != t[f"{e}_po"]]
        print(f"[{e}] 全开发集 sel={tot['sel']} c0={tot['c0']} px={tot['px']} pobj={tot['pobj']}")
        print(f"    救回:{saves} 破坏:{brk} |vs always-px:{vs_px} |vs 按物体:{vs_po}")

    with open(R / "selector_dev_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tab[0].keys()))
        w.writeheader(); w.writerows(tab)

    # ---------- 3. 冻结 spec ----------
    spec = {
        "frozen": "2026-11-17", "stage": "4C",
        "selector": {
            "rule": "pick px iff s*e > θe AND approach > θa else c0",
            "s": S, "theta_e_mm": round(TE * 1000, 1),
            "theta_a_mm": (None if np.isinf(TA) else round(TA * 1000, 1)),
            "meaning": ("e=(locked_est_xy−finger_mid_xy)·u；"
                        "approach=(fm(t0)−fm(t0−5))·u（K=5 控制步）"),
            "u": "normalize(geom_xpos[finger1_pad]−geom_xpos[finger2_pad])[:2]，"
                 "geom 名 gripper0_finger{1,2}_pad_collision；u 在决策步计算",
            "e": "锁定视觉目标(vislock locked_est) xy 相对夹爪中点的闭合轴有符号误差",
            "no_trigger": "无 cand_step（未锁定/未接近/已闭合）→ c0 回退",
            "decision_step": "同四 A：首个 已锁定∧未闭合∧3D|eef−锁定目标|<70mm 步",
            "window": "pick=px 时偏置自决策步持续至首次闭合+2 或 30 步；px=u*0.039 raw/步加 action[:2]",
        },
        "dev_states": {"butter": [f"init{i}_{c}" for i in SETS["butter"]
                                  for c in ("trans", "trans_y")],
                       "milk": [f"init{i:02d}_{c}" for i in SETS["milk"]
                                for c in ("trans", "trans_y")]},
        "tie_break": "max g successes → fewer px picks → θa=-inf(更简单)",
        "code_hash": {"run_correction_eval": sha(BASE / "scripts/run_correction_eval.py"),
                       "analyze_action_select": sha(BASE / "scripts/analyze_action_select.py"),
                       "freeze_selector": sha(BASE / "scripts/freeze_selector.py")},
        "selection_basis": "全部 30 开发状态、仅抓取主终点",
    }
    out = BASE / "runs/predictability/action_sel_spec_2026-11-17.json"
    json.dump(spec, open(out, "w"), ensure_ascii=False, indent=1)
    print("spec ->", out)
    # dev picks 供 sel 臂复现验证
    picks = {t["tag"]: t["pick"] for t in tab}
    json.dump(picks, open(R / "selector_dev_picks.json", "w"))
    print("dev picks:", picks)


if __name__ == "__main__":
    main()
