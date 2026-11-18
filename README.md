# vla-runtime-xy-correction

**Runtime XY correction for a frozen VLA policy — what does the target reference point need to be?**

When the target object is displaced, a frozen Vision-Language-Action policy (OpenVLA,
LIBERO `libero_object`) reaches to the remembered location and closes on empty space.
This project wraps the policy output with a small causal controller that overrides only
the XY channels inside a gated window, and studies — via paired ablations — what target
reference the controller actually needs.

Paper draft (Chinese, 初稿): `docs/PAPER_DRAFT_v2_1.md`

## Pipeline

```
LIBERO env (robosuite, control_freq=20 Hz, 280 steps max)
  ├─ obs["agentview_image"] ──► get_libero_image: 180° rotate + lanczos resize
  │                                └─► OpenVLA 7B (4-bit, frozen) ──► raw 7-DoF action
  ├─ TargetEstimator (template NCC + workspace filter + pix2w affine)
  │                                └─► per-frame est target (visgate) or first-lock (vislock)
  └─ CorrState controller (only when triggered, XY channels only)
       trigger : gripper-open(prev step) ∧ z_gap∈[0,55]mm ∧ (descent≥2mm/4steps ∨ z_gap<20mm)
                  ∧ |err_xy|>25mm
       act     : action[:2] = clip(15 · err_xy[m], ±0.8)   # normalized action units
       release : lat≤15mm | z_gap∉[-5,75]mm | gripper closes
       mask    : first close command → raw action passes through until gripper reopens
```

Five arms on the **same 100 held-out translation states** (butter n=40, milk n=60;
perturbation = +5 cm x or +5 cm y applied at episode start):

| arm | target XY source | update |
|---|---|---|
| vanilla | — (no controller) | — |
| gate | simulator GT (privileged oracle reference) | every step |
| visgate | per-frame visual estimate | every step |
| vislock | first non-empty visual estimate, locked for the episode | once |
| fixlock | identical to vislock, but locked XY replaced by a class-fixed point | once |

## Main results (100 paired states, stable grasp g / complete task t)

| arm | stable grasp | complete task |
|---|---|---|
| vanilla | 37 | 28 |
| fixlock | 44 | 40 |
| vislock | 54 | 38 |
| visgate | 58 | 39 |

- **Per-episode reference matters**: vislock − fixlock = **+10 pp** on grasp,
  cluster-bootstrap 95% CI [+1, +20].
- **Per-frame update**: visgate nets +2 grasps on each object over vislock;
  per-object intervals cross zero — evidence boundary is reported, not overstated.
- **Task≠grasp**: the milk trans subset shows a reverse task-side effect
  (fixlock 18 vs vislock 12 tasks); reported per paired accounting only.
- **Action selection (supplementary, exploratory/negative)**: a frozen
  pre-close selector on 32 unseen states does not beat keeping the baseline
  action (sel 24 vs c0 25 / 32).

butter = **new object positions**; milk = **held-out init states**. The
fixed-point and update-mode comparisons reuse already-analyzed states
(mechanism controls, not an independent validation set).

## Verify the numbers (no GPU, no npz needed)

```bash
python scripts/verify_results.py   # recomputes main table + RQ3/RQ4 counts from shipped per-state CSVs
```

Expected: `g 37/44/54/58, t 28/40/38/39`, visgate−vislock `+2` grasps per object,
4C `c0 25, px 23, sel 24 / 32`.

## Full reproduction (needs GPU + LIBERO + OpenVLA)

```bash
# 1. Install upstream deps (see requirements.txt): LIBERO, robosuite,
#    transformers, torch; clone OpenVLA repo beside this one or set OPENVLA_DIR
export OPENVLA_DIR=/path/to/openvla
#    checkpoint "openvla/openvla-7b-finetuned-libero-object" auto-downloads from HF

# 2. Run representative episodes (example: butter vislock arm)
MUJOCO_GL=egl python scripts/run_correction_eval.py \
    --task_id 6 --init_list 30-49 --conditions trans,trans_y \
    --pos_override runs/predictability/butter_newpos.json \
    --arm vislock --out_prefix fzb
# milk task_id=7, --init_list 20-49; arms: vanilla|gate|visgate|vislock|fixlock

# 3. Re-run analyses on the shipped per-state CSVs
python scripts/verify_results.py
python scripts/analyze_fixlock.py          # needs npz trajectories for some fields
python scripts/plot_fixlock_main.py        # main figure → results/fixlock_main.{png,pdf}
```

Frozen protocols (values fixed *before* outcomes, with code hashes):
`runs/predictability/*_spec_2026-11-{16,17}.json`

## Repo layout

- `scripts/` — runner (`run_perturb_eval.py`, `run_correction_eval.py`), estimator
  (`vision_target.py`), frozen-config batch scripts, paired-CI analyses
- `runs/predictability/` — frozen protocols, class assets (`tmpl_*.png`,
  `pix2w_*.json`), state lists
- `results/` — per-state paired CSVs, summary JSONs, `corrige_3c.json` case audit,
  main figure
- `docs/` — reports (frozen transfer, fixed-XY control, action selection,
  selector confirmation) + paper draft (Chinese 初稿)
- `media/` — side-by-side rollout clips (vanilla vs corrected)

## Not shipped (kept local)

- OpenVLA weights (HF auto-download), conda env
- Per-episode `.npz` trajectories (~80 MB) and full video sets (~115 MB)
- Dev-stage logs and superseded CSVs

Analysis scripts that read `.npz` fields will degrade gracefully or print a
"missing states" notice; the headline numbers verify without them.

## Limitations

Single policy/checkpoint, sim-only, two objects; perturbations are translation
only; template-matching estimator (affine + z_prior calibrated offline on GT);
selector conclusions limited to the c0/px two-candidate, this perturbation family.

## Third-party

Built on OpenVLA (Stanford), LIBERO (Liu et al.), robosuite/MuJoCo, GroundingDINO-style
template matching implemented locally. See `NOTICE.md`. License for this project's
own code: TBD (private repository; contact before reuse).

## Paper status

Manuscript in preparation — Chinese 初稿 (`docs/PAPER_DRAFT_v2_1.md`), not published.
