"""plot_fixlock_main.py — 三B平移状态主结果图（四臂 × g/t × 物体分面）
臂序：vanilla / 类别固定XY(fixlock) / 首次视觉XY(vislock) / 逐帧视觉XY(visgate)
输出：results/fixlock_main.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

R = "results"
INITS = {"butter": range(30, 50), "milk": range(20, 50)}
PRE = {"vanilla": ("fzb", "fzm"), "fixlock": ("fzbfx", "fzmfx"),
       "vislock": ("fzb", "fzm"), "visgate": ("fzb", "fzm")}
ARM_TAG = {"vanilla": "vanilla", "fixlock": "fixlock",
           "vislock": "vislock", "visgate": "visgate"}
ARMS = ["vanilla", "fixlock", "vislock", "visgate"]
ARM_LBL = ["vanilla\n（无纠偏）", "类别固定 XY\nfixlock",
           "首次视觉 XY\nvislock", "逐帧视觉 XY\nvisgate"]
CONDS = ["trans", "trans_y"]
C_LBL = {"trans": "trans (x)", "trans_y": "trans_y (y)"}
C_COL = {"trans": "#4C9BE0", "trans_y": "#E8894C"}


def meta(p):
    return json.loads(np.load(p, allow_pickle=True)["meta"].tobytes())


def get(arm, obj, init, cond, ep):
    pre = PRE[arm][0 if obj == "butter" else 1]
    p = f"{R}/{pre}_init{init:02d}_{cond}_{ARM_TAG[arm]}.npz"
    if not os.path.exists(p):
        return None
    m = meta(p)
    return int(bool(m["task_success"])) if ep == "t" \
        else int(bool(m["grasp_success"]))


def main():
    summ = json.load(open(f"{R}/fixlock_ref_summary.json"))["groups"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6),
                             sharey=True)
    EPS = [("g", "稳定抓取 g"), ("t", "完整入篮 t")]
    for ri, obj in enumerate(("butter", "milk")):
        for ci_, (ep, ep_lbl) in enumerate(EPS):
            ax = axes[ri, ci_]
            x = np.arange(len(ARMS))
            w = 0.36
            for arm_i, arm in enumerate(ARMS):
                for j, cond in enumerate(CONDS):
                    vals = [get(arm, obj, i, cond, ep)
                            for i in INITS[obj]]
                    vals = [v for v in vals if v is not None]
                    rate = np.mean(vals) if vals else np.nan
                    bar = ax.bar(arm_i + (j - 0.5) * w, rate, w,
                                 color=C_COL[cond],
                                 alpha=1.0 if arm != "fixlock" else 0.75,
                                 edgecolor="k", linewidth=0.4)
                    ax.text(arm_i + (j - 0.5) * w, rate + 0.015,
                            f"{int(round(rate * len(vals)))}/{len(vals)}",
                            ha="center", fontsize=7.5)
            # fixlock vs vislock 胜负+区间（fixlock−vislock）
            grp = summ[f"{obj}_{'trans'}"][ep]
            grp2 = summ[f"{obj}_{'trans_y'}"][ep]
            ax.annotate("固定XY vs 首次视觉XY（配对 Δ=f−v）",
                        xy=(1.5, 1.22), ha="center", fontsize=8.2,
                        fontweight="bold")
            for j, (grp_, lbl) in enumerate(
                    ((grp, "trans"), (grp2, "trans_y"))):
                ns, nb = len(grp_["saves"]), len(grp_["breaks"])
                d, lo, hi = grp_["ci"]
                ax.annotate(
                    f"{lbl}: 救{ns} 破{nb} Δ{d:+.2f} [{lo:+.2f},{hi:+.2f}]",
                    xy=(1.5, 1.11 - j * 0.09), ha="center",
                    fontsize=7.8, color=C_COL[lbl])
            ax.set_xticks(x)
            if ci_ == 0:
                ax.set_ylabel("成功率")
            ax.set_xticklabels(ARM_LBL if ri == 1 else [""] * 4,
                               fontsize=8)
            ax.set_ylim(0, 1.30)
            ax.set_title(f"{obj} · {ep_lbl}", fontsize=10)
            ax.axhline(0.5, ls=":", lw=0.6, c="gray")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if ri == 0 and ci_ == 0:
                for j, c in enumerate(CONDS):
                    ax.bar([], [], color=C_COL[c], label=C_LBL[c],
                           edgecolor="k", linewidth=0.4)
                ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("三 B 平移状态 · 四臂成功率（同一批 100 状态）", fontsize=12)
    fig.text(0.5, 0.005,
             "注：butter=新位置验证集(init30–49, n=40)；milk=新init验证集"
             "(init20–49, n=60)。固定点对照复用已分析状态，属机制对照。",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(f"{R}/fixlock_main.png", dpi=200)
    fig.savefig(f"{R}/fixlock_main.pdf", bbox_inches="tight")  # 论文用矢量版
    print("saved results/fixlock_main.png")


if __name__ == "__main__":
    main()
