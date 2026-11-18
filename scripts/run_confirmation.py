"""run_confirmation.py — init30-49 独立确认的一次性编排（gate -> 触发 -> 成对分支）。

1. gate(2C) 臂：init30-49 × {orig, trans(=trans_x), rot, trans_y}
2. 用冻结的 detect_triggers.scan() 定位 t*，写 results/confirm_trigger_scan.csv
3. 每个触发状态跑 branch α0/α1（共享前缀，single 模式）
已存在的轨迹不重跑（中断后可续跑；不改变任何参数）。
"""
import csv
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
R = BASE / "results"
PY = "python"
INITS = list(range(30, 50))
CONDS = ["orig", "trans", "rot", "trans_y"]
sys.path.insert(0, str(BASE / "scripts"))


def run(args):
    print("+", " ".join(args), flush=True)
    subprocess.run([PY, str(BASE / "scripts/run_correction_eval.py")] + args,
                   check=True, cwd=BASE)


def main():
    for c in CONDS:
        todo = [i for i in INITS if not (R / f"traj2_init{i:02d}_{c}_gate.npz").exists()]
        if todo:
            run(["--arm", "gate", "--init_list", ",".join(map(str, todo)),
                 "--conditions", c, "--out_csv", str(R / "episodes_confirm_gate.csv")])

    from detect_triggers import scan
    rows = [scan(f"init{i:02d}_{c}") for i in INITS for c in CONDS]
    with open(R / "confirm_trigger_scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    trig = [r for r in rows if r["t_star"]]
    print(f"triggers: {len(trig)}/80", flush=True)

    for c in CONDS:
        sel = [r for r in trig if r["tag"].endswith("_" + c)
               and not (c == "trans" and r["tag"].endswith("trans_y"))]
        for alpha in ("0", "1"):
            todo = [r for r in sel if not (
                R / f"traj2_{r['tag']}_branch_a{alpha}_single.npz").exists()]
            if not todo:
                continue
            run(["--arm", "branch", "--branch_mode", "single",
                 "--branch_alpha", alpha,
                 "--branch_t", ",".join(str(r["t_star"]) for r in todo),
                 "--init_list", ",".join(r["tag"][4:6] for r in todo),
                 "--conditions", c,
                 "--out_csv", str(R / "episodes_confirm_branch.csv")])
    print("CONFIRMATION RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
