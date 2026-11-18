#!/usr/bin/env bash
# 阶段三 B 冻结三臂跑批（串行；单卡只能容纳一个 OpenVLA 实例）
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
MAN=runs/predictability/frozen_visual_transfer_2026-11-16.json
LOG=results/frozen_batch_logs
mkdir -p "$LOG"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

verify() {
  $PY - <<EOF || { echo "[ABORT] frozen file hash mismatch"; exit 2; }
import hashlib, json, sys
m = json.load(open("$MAN"))
bad = [p for p, h in m["file_sha256"].items()
       if hashlib.sha256(open(p, "rb").read()).hexdigest() != h]
if bad:
    print("mismatch:", bad); sys.exit(1)
EOF
}

run() {  # set task_id inits prefix extra...
  local set=$1 tid=$2 inits=$3 pre=$4; shift 4
  for arm in vanilla gate visgate; do
    verify
    echo "[$(date +%T)] start $set $arm"
    $PY scripts/run_correction_eval.py --arm "$arm" --task_id "$tid" \
      --init_list "$inits" --conditions orig,trans,trans_y \
      --out_prefix "$pre" --vid_prefix "${pre}_" \
      --out_csv "results/${pre}_episodes_${arm}.csv" "$@" \
      > "$LOG/${pre}_${arm}.log" 2>&1
    echo "[$(date +%T)] done  $set $arm rc=$?"
  done
}

run butter 6 30-49 fzb --pos_override runs/predictability/butter_newpos.json
run milk   7 20-49 fzm
echo "[$(date +%T)] ALL DONE"
