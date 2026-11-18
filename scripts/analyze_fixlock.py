"""analyze_fixlock.py — 类别固定XY vs 首有效视觉XY 机制对照（3B 平移状态）
配对：f{o}fx_init{ii}_{cond}_fixlock.npz vs f{o}_init{ii}_{cond}_vislock.npz
输出：results/fixlock_ref_summary.json + fixlock_ref_table.csv
"""
import json, glob, os
import numpy as np

R = "results"
FIX_XY = {"butter": np.array([-0.10640, -0.21480]),
          "milk": np.array([-0.09335, -0.21605])}
INITS = {"butter": range(30, 50), "milk": range(20, 50)}
PRE = {"butter": ("fzb", "fzbfx"), "milk": ("fzm", "fzmfx")}
NEAR_MM = 3.0
EPS = ("g", "t")
RNG = np.random.default_rng(0)


def meta(p):
    return json.loads(np.load(p, allow_pickle=True)["meta"].tobytes())


def paired_ci(rows, key_a, key_b):
    """init 簇 bootstrap：簇内差均值 → 簇重采样。"""
    by = {}
    for r in rows:
        by.setdefault(r["init"], []).append(r[key_a] - r[key_b])
    keys = sorted(by)
    cm = np.array([np.mean(by[k]) for k in keys])
    idx = RNG.choice(len(cm), (10000, len(cm)), replace=True)
    samp = cm[idx].mean(1)
    lo, hi = np.percentile(samp, [2.5, 97.5])
    return round(float(cm.mean()), 3), round(float(lo), 3), round(float(hi), 3)


def pooled_ci(rows):
    """合并 CI：簇键 (obj,init)（各簇恰 2 状态→簇等权=状态等权）。"""
    out = {}
    for e in EPS:
        by = {}
        for r in rows:
            by.setdefault((r["obj"], r["init"]), []).append(
                r[f"{e}_f"] - r[f"{e}_v"])
        keys = sorted(by)
        cm = np.array([np.mean(by[k]) for k in keys])
        samp = cm[RNG.choice(len(cm), (10000, len(cm)), replace=True)].mean(1)
        lo, hi = np.percentile(samp, [2.5, 97.5])
        out[e] = dict(diff=round(float(cm.mean()), 3),
                      ci=[round(float(lo), 3), round(float(hi), 3)],
                      n_clusters=len(keys))
    return out


def cross_tab(rows, obj, cond):
    """抓取四类 × 入篮翻转交叉账。g_cell/t_cell: 2*v+f。"""
    GN = {0: "both_fail", 1: "save", 2: "break", 3: "both_ok"}
    TN = GN
    tab = {}
    for r in rows:
        if r["obj"] != obj or r["cond"] != cond:
            continue
        gk = GN[r["g_v"] * 2 + r["g_f"]]
        tk = TN[r["t_v"] * 2 + r["t_f"]]
        tab.setdefault(gk, {}).setdefault(tk, []).append(r["tag"])
    return tab


def main():
    rows = []
    for obj, (pv, pf) in PRE.items():
        for init in INITS[obj]:
            for cond in ("trans", "trans_y"):
                tag = f"init{init:02d}_{cond}"
                pv_ = f"{R}/{pv}_{tag}_vislock.npz"
                pf_ = f"{R}/{pf}_{tag}_fixlock.npz"
                if not (os.path.exists(pv_) and os.path.exists(pf_)):
                    continue
                mv, mf = meta(pv_), meta(pf_)
                le_v = mv.get("locked_est")
                le_f = mf.get("locked_est")
                d = (None if le_v is None else
                     float(np.linalg.norm(np.asarray(le_v[:2]) - FIX_XY[obj]))
                     * 1000)
                rows.append(dict(
                    obj=obj, init=init, cond=cond, tag=tag,
                    lock_v=mv.get("lock_step"), lock_f=mf.get("lock_step"),
                    le_v=le_v, le_f=le_f, ref_dist_mm=d,
                    g_v=int(bool(mv["grasp_success"])),
                    g_f=int(bool(mf["grasp_success"])),
                    t_v=int(bool(mv["task_success"])),
                    t_f=int(bool(mf["task_success"]))))
    miss = [t for t in ("trans", "trans_y") for o in INITS
            for i in INITS[o]
            if not os.path.exists(f"{R}/{PRE[o][1]}_init{i:02d}_{t}_fixlock.npz")]
    out = {"n_pairs": len(rows), "missing_fixlock": miss, "groups": {}}
    for obj in INITS:
        for cond in ("trans", "trans_y"):
            rr = [r for r in rows if r["obj"] == obj and r["cond"] == cond]
            g = {}
            for e in EPS:
                kv, kf = f"{e}_v", f"{e}_f"
                saves = [r["tag"] for r in rr if not r[kv] and r[kf]]
                brks = [r["tag"] for r in rr if r[kv] and not r[kf]]
                g[e] = dict(v=sum(r[kv] for r in rr), f=sum(r[kf] for r in rr),
                            saves=saves, breaks=brks, ci=paired_ci(rr, kf, kv))
            near = [dict(tag=r["tag"], d_mm=round(r["ref_dist_mm"], 1))
                    for r in rr if r["ref_dist_mm"] is not None
                    and r["ref_dist_mm"] <= NEAR_MM]
            ds = [r["ref_dist_mm"] for r in rr if r["ref_dist_mm"] is not None]
            g["ref_dist_mm"] = dict(median=round(float(np.median(ds)), 1),
                                    min=round(float(min(ds)), 1),
                                    max=round(float(max(ds)), 1),
                                    n_no_lock=sum(1 for r in rr
                                                  if r["ref_dist_mm"] is None),
                                    near_coincident=near)
            out["groups"][f"{obj}_{cond}"] = g
    out["pooled"] = pooled_ci(rows)
    out["milk_trans_crosstab"] = cross_tab(rows, "milk", "trans")
    json.dump(out, open(f"{R}/fixlock_ref_summary.json", "w"),
              ensure_ascii=False, indent=1)
    import csv
    with open(f"{R}/fixlock_ref_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for k, g in out["groups"].items():
        print(f"== {k} ==")
        for e in EPS:
            x = g[e]
            print(f"  [{e}] vislock={x['v']} fixlock={x['f']} "
                  f"救={x['saves']} 破={x['breaks']} CI={x['ci']}")
        print("  ref dist:", g["ref_dist_mm"])


if __name__ == "__main__":
    main()
