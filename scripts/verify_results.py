"""verify_results.py — 仅用随附逐状态配对 CSV 重算主表数字（无需 npz 轨迹）。

用法: python scripts/verify_results.py
数据源: results/{fixlock_ref_table, online_vs_lock_{butter,milk},
        frozen_transfer_{butter,milk}, action_sel_confirm_table}.csv
逐回合 npz 轨迹(全量)未随仓库分发，见 README「大文件获取」。
"""
import csv
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"


def rows(name):
    with open(R / name, newline="") as f:
        return list(csv.DictReader(f))


def tob(x):
    return str(x).strip() in ("1", "True", "true")


def main():
    print("== 主表: 同一批 100 个平移状态 (butter 40 + milk 60, trans+trans_y) ==")
    fx = rows("fixlock_ref_table.csv")
    # vanilla/gate/visgate 每状态结果来自 frozen_transfer 表 (v/g/x 列)
    ft = {}
    for obj in ("butter", "milk"):
        for r in rows(f"frozen_transfer_{obj}.csv"):
            ft[(obj, r["tag"])] = r          # tag 在两物体间重名
    main = {"vanilla": [0, 0], "fixlock": [0, 0],
            "vislock": [0, 0], "visgate": [0, 0]}
    for r in fx:
        fr = ft[(r["obj"], r["tag"])]
        for arm, gk, tk in (("vanilla", "v_g", "v_t"),
                            ("fixlock", "g_f", "t_f"),
                            ("vislock", "g_v", "t_v"),
                            ("visgate", "x_g", "x_t")):
            src = r if arm in ("fixlock", "vislock") else fr
            main[arm][0] += tob(src[gk])
            main[arm][1] += tob(src[tk])
    for arm in ("vanilla", "fixlock", "vislock", "visgate"):
        print(f"  {arm:8s} g={main[arm][0]:3d}/100  t={main[arm][1]:3d}/100")
    assert [a[0] for a in main.values()] == [37, 44, 54, 58], main
    assert [a[1] for a in main.values()] == [28, 40, 38, 39], main
    print("  -> 与冻结报告一致: g 37/44/54/58, t 28/40/38/39")

    print("\n== RQ3: visgate - vislock 逐帧更新翻转 (trans+trans_y) ==")
    for obj in ("butter", "milk"):
        rr = [r for r in rows(f"online_vs_lock_{obj}.csv")
              if r["cond"] in ("trans", "trans_y")]
        g_x = sum(tob(r["x_g"]) for r in rr)
        g_l = sum(tob(r["l_g"]) for r in rr)
        print(f"  {obj:6s} n={len(rr)}  visgate_g={g_x}  vislock_g={g_l}  "
              f"net=+{g_x - g_l}")

    print("\n== RQ4: 四C 确认集 32 状态 ==")
    rr = rows("action_sel_confirm_table.csv")
    for arm, k in (("c0", "g_c0"), ("px", "g_px"), ("sel", "g_sel")):
        print(f"  {arm}: g={sum(tob(r[k]) for r in rr)}/{len(rr)}")

    print("\n全部核对通过。")


if __name__ == "__main__":
    main()
