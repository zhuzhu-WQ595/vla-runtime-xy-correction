"""analyze_selector_confirm.py — 阶段四 C 确认分析（32 全新状态 × {c0,px,sel}）。

- 每状态核对：sel 轨迹与 sel_pick 所指臂轨迹逐比特一致（在线选择器正确性）。
- 冻结选择器结局 = sel 臂结局；主比较 = 按物体固定(butter→c0,milk→px)；
  并列 always-c0 / always-px。按物体×扰动方向报干预数、救回、破坏、
  有效分歧；抓取/入篮双终点；按 init 成簇 paired bootstrap CI。
输出 results/action_sel_confirm_summary.json + action_sel_confirm_table.csv
"""
import csv
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
R = BASE / "results"
PRE = {"butter": "f4cb", "milk": "f4cm"}
INITS = {"butter": [31, 33, 36, 39, 42, 45, 48, 49],
         "milk": [2, 5, 8, 11, 14, 16, 18, 19]}
CONDS = ("trans", "trans_y")
POBJ = {"butter": "c0", "milk": "px"}
RNG = np.random.default_rng(0)


def load(pre, tag, cand):
    d = np.load(R / f"{pre}_{tag}_pb_{cand}.npz", allow_pickle=True)
    m = json.loads(str(d["meta"]))
    return d, m


def paired_ci(rows, key_a, key_b):
    """按 init 簇 bootstrap：簇内差均值 → 簇重采样。"""
    by = {}
    for r in rows:
        by.setdefault(r["init"], []).append(r[key_a] - r[key_b])
    keys = sorted(by)
    cm = np.array([np.mean(by[k]) for k in keys])
    n = 10000
    idx = RNG.choice(len(cm), (n, len(cm)), replace=True)
    samp = cm[idx].mean(1)
    lo, hi = np.percentile(samp, [2.5, 97.5])
    return round(float(cm.mean()), 3), round(float(lo), 3), round(float(hi), 3)


def main():
    rows, mism = [], []
    for obj, inits in INITS.items():
        for i in inits:
            for cond in CONDS:
                tag = f"init{i:02d}_{cond}"
                try:
                    dsel, msel = load(PRE[obj], tag, "sel")
                except FileNotFoundError:
                    print("MISSING", obj, tag)
                    continue
                pick = msel.get("sel_pick") or "c0"
                r = dict(obj=obj, init=i, cond=cond, tag=tag,
                         pick=pick, sel_e=msel.get("sel_e_mm"),
                         sel_ap=msel.get("sel_ap_mm"),
                         cand_step=msel.get("cand_step"))
                # sel 轨迹 vs 所选臂轨迹（配对一致性核对）
                ref, mref = load(PRE[obj], tag, pick)
                ea = np.asarray(dsel["exec_action"], float)
                er = np.asarray(ref["exec_action"], float)
                n = min(len(ea), len(er))
                ok = (len(ea) == len(er)) and np.allclose(ea[:n], er[:n])
                if not ok:
                    mism.append(tag)
                r["sel_eq_pick"] = bool(ok)
                for e in ("g", "t"):
                    k = "grasp_success" if e == "g" else "task_success"
                    r[f"{e}_sel"] = int(bool(msel[k]))
                    for cd in ("c0", "px"):
                        _, mc = load(PRE[obj], tag, cd)
                        r[f"{e}_{cd}"] = int(bool(mc[k]))
                rows.append(r)

    # ---- 汇总 ----
    out = {"n_states": len(rows), "sel_mismatch": mism, "endpoints": {}}
    for e in ("g", "t"):
        ep = {}
        for obj in INITS:
            rr = [r for r in rows if r["obj"] == obj]
            ep[obj] = dict(
                n=len(rr),
                interventions=sum(r["pick"] == "px" for r in rr),
                sel=sum(r[f"{e}_sel"] for r in rr),
                c0=sum(r[f"{e}_c0"] for r in rr),
                px=sum(r[f"{e}_px"] for r in rr),
                pobj=sum(r[f"{e}_{POBJ[obj]}"] for r in rr))
            for cond in CONDS:
                rc = [r for r in rr if r["cond"] == cond]
                ep[obj][cond] = dict(
                    interventions=sum(r["pick"] == "px" for r in rc),
                    sel=sum(r[f"{e}_sel"] for r in rc),
                    pobj=sum(r[f"{e}_{POBJ[obj]}"] for r in rc))
        ep["all"] = dict(
            sel=sum(r[f"{e}_sel"] for r in rows),
            c0=sum(r[f"{e}_c0"] for r in rows),
            px=sum(r[f"{e}_px"] for r in rows),
            pobj=sum(r[f"{e}_{POBJ[r['obj']]}"] for r in rows),
            interventions=sum(r["pick"] == "px" for r in rows))
        # 逐例得失 vs 三基线
        for bl, kk in (("c0", f"{e}_c0"), ("px", f"{e}_px"),
                       ("pobj", None)):
            gain, loss = [], []
            for r in rows:
                b = (r[kk] if kk else r[f"{e}_{POBJ[r['obj']]}"])
                if r[f"{e}_sel"] > b:
                    gain.append(r["tag"])
                elif r[f"{e}_sel"] < b:
                    loss.append(r["tag"])
            ep[f"gain_vs_{bl}"] = gain
            ep[f"loss_vs_{bl}"] = loss
        # 有效分歧：pick≠c0 且结局≠c0
        ep["effective_div"] = [r["tag"] for r in rows
                               if r["pick"] == "px"
                               and r[f"{e}_sel"] != r[f"{e}_c0"]]
        # init 簇配对 CI：sel−pobj、sel−c0、sel−px
        ep["ci_sel_minus_pobj"] = paired_ci(
            [{"init": r["init"], "a": r[f"{e}_sel"],
              "b": r[f"{e}_{POBJ[r['obj']]}"]} for r in rows], "a", "b")
        ep["ci_sel_minus_c0"] = paired_ci(
            [{"init": r["init"], "a": r[f"{e}_sel"], "b": r[f"{e}_c0"]}
             for r in rows], "a", "b")
        ep["ci_sel_minus_px"] = paired_ci(
            [{"init": r["init"], "a": r[f"{e}_sel"], "b": r[f"{e}_px"]}
             for r in rows], "a", "b")
        out["endpoints"][e] = ep

    json.dump(out, open(R / "action_sel_confirm_summary.json", "w"),
              ensure_ascii=False, indent=1)
    cols = (["obj", "tag", "pick", "sel_e", "sel_ap", "cand_step",
             "sel_eq_pick"]
            + [f"{e}_{a}" for e in ("g", "t") for a in ("sel", "c0", "px")])
    with open(R / "action_sel_confirm_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print("sel==pick臂 轨迹不符:", mism)
    for e in ("g", "t"):
        ep = out["endpoints"][e]
        print(f"[{e}] all:", ep["all"])
        for obj in INITS:
            print(f"  {obj}:", ep[obj])
        print("  gain/loss vs pobj:", ep["gain_vs_pobj"], ep["loss_vs_pobj"])
        print("  gain/loss vs c0:", ep["gain_vs_c0"], ep["loss_vs_c0"])
        print("  gain/loss vs px:", ep["gain_vs_px"], ep["loss_vs_px"])
        print("  有效分歧:", ep["effective_div"])
        print("  CI sel−pobj:", ep["ci_sel_minus_pobj"],
              "| sel−c0:", ep["ci_sel_minus_c0"],
              "| sel−px:", ep["ci_sel_minus_px"])


if __name__ == "__main__":
    main()
