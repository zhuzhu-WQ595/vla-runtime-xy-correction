# FROZEN_VISUAL_TRANSFER_REPORT — 阶段三 B：冻结视觉迁移验证

日期：2026-11-17 ｜ 冻结时间：**2026-11-16 19:20**（早于任何结局产生）

> **结论先行：在从未进入感知开发的冻结状态上，冻结视觉 2C 相对 vanilla 仍取得显著的正向收益（两物体、两个终点的 Δx−v 区间均不含 0）。**
> butter 抓取终点上 visual 低于 oracle（Δx−g = −0.117 [−0.25, −0.017]）；milk 上 visual 与 oracle 差异不显著（Δx−g 区间跨 0，不能下非劣结论，见 §5.3）。

## 0. 与阶段三 A 的区别

| | 阶段三 A | 阶段三 B（本报告） |
|---|---|---|
| 状态 | butter init30–49 / milk init00–19，与标定、模板、z_prior 来源重叠 | **冻结前未参与任何感知开发的状态** |
| 性质 | 固定状态比较 | 冻结迁移验证 |
| 位置 | 原始 init 物体位置 | butter：20 个新物体 XY；milk：init20–49 原位置（与标定集位置分布重叠 ≤1cm，见 §2 警告） |

冻结清单：`runs/predictability/frozen_visual_transfer_2026-11-16.json`，含 9 个代码/资产文件的 SHA-256；跑批脚本在每臂启动前校验 hash，不符即中止。

## 1. 实验设计（冻结内容）

- **臂**：`vanilla`（无纠偏）、`gate`（oracle-2C，仿真真值目标）、`visgate`（visual-2C，冻结 RGB 估计器）。
- **条件**：`orig`（无扰动）、`trans`（+5cm x）、`trans_y`（+5cm y）。
- **配对**：同一 (init, cond) 状态三臂配对；区间以 init 为簇 bootstrap 95%（2000 次，seed 0）。
- **主对比**：visgate − vanilla。**次对比**：visgate − gate 及 oracle 救回保留率。
- **终点**：稳定抓取（主）、完整入篮（次）。救回分母=vanilla 失败数；破坏分母=vanilla 成功数。
- **状态集**：
  - **butter（新位置验证）**：init30–49 机器人/场景配置 + `butter_newpos.json` 的 20 个新 XY（距全部 50 个原始位置 ≥1.5cm、相互 ≥2cm、静置高度与无穿插校验通过；20 init × 3 条件 = 60 状态）。
  - **milk（新 init）**：init20–49（30 init × 3 条件 = 90 状态）。

## 2. 感知资产来源与重叠核对

来源表：`results/perception_provenance.csv`。**所有感知资产均来自阶段三 A 评估集**，这是本实验必须冻结新状态的原因：

| 资产 | 来源 | 与本阶段状态的关系 |
|---|---|---|
| `tmpl_butter.png` | `init43_trans` vanilla 视频第 0 帧（NCC 自匹配 score=1.0） | init43 仅作 butter 基底状态；物体已移到新 XY，模板仍匹配目标外观 |
| `tmpl_milk.png` | `milk_init00_orig` vanilla 视频第 0 帧 | milk init00 不在本阶段状态集 |
| `pix2w_butter/milk.json` | butter init30–49 / milk init00–19 vanilla 轨迹像素-真值拟合 | **butter 基底 init 与仿射集同名但物体位置全新；milk 物体 XY 与仿射集位置分布重叠 ≤1cm** |
| `z_prior` | 同上真值统计 | 同上 |

**警告**：milk init20–49 的物体 XY 分布与标定集在约 1cm 内重叠，属"新 init 状态"而非"新位置验证"；只有 butter 是真正的新位置验证。

## 3. 主结果

### 3.1 butter（新位置验证，60 状态）

| 终点 | vanilla | oracle | visual | visual 救/破 | oracle 救/破 | oracle 保留 | Δx−v [CI] | Δx−g [CI] |
|---|---|---|---|---|---|---|---|---|
| 抓取 ALL | 21 | 44 | 37 | 17/39, 1/21 | 24, 1 | 15/24 (62.5%) | +0.267 [+0.167, +0.383] | −0.117 [−0.250, −0.017] |
| 入篮 ALL | 18 | 27 | 26 | 10/42, 2/18 | 13, 4 | 8/13 (61.5%) | +0.133 [+0.017, +0.267] | −0.017 [−0.133, +0.083] |

按条件（抓取）：

| 条件 | vanilla | oracle | visual | oracle 保留 | Δx−v [CI] | Δx−g [CI] |
|---|---|---|---|---|---|---|
| orig | 8 | 15 | 14 | 6/8 | +0.30 [+0.05, +0.55] | −0.05 [−0.20, +0.10] |
| trans | 6 | 17 | 13 | 7/11 | +0.35 [+0.15, +0.55] | −0.20 [−0.40, −0.05] |
| trans_y | 7 | 12 | 10 | 2/5 | +0.15 [+0.00, +0.35] | −0.10 [−0.30, +0.10] |

按条件（入篮）：orig 7/10/9；trans 6/10/10；trans_y 5/7/7。

- visual 抓取破坏仅 1 例（`init32_orig`，oracle 同例也破坏）；入篮破坏 2 例（`init44_trans`、`init48_orig`，oracle 均同破坏）。
- visual 在 oracle 之外新增救回：抓取 `init39_orig`、`init45_trans_y`；入篮 `init33_trans_y`、`init37_trans`。

### 3.2 milk（新 init，90 状态）

| 终点 | vanilla | oracle | visual | visual 救/破 | oracle 救/破 | oracle 保留 | Δx−v [CI] | Δx−g [CI] |
|---|---|---|---|---|---|---|---|---|
| 抓取 ALL | 52 | 59 | **64** | 13/38, 1/52 | 9, 2 | 6/9 (66.7%) | +0.133 [+0.067, +0.211] | +0.056 [−0.022, +0.133] |
| 入篮 ALL | 45 | 49 | **51** | 6/45, 0/45 | 6, 2 | 2/6 (33.3%) | +0.067 [+0.022, +0.111] | +0.022 [−0.033, +0.078] |

按条件（抓取）：

| 条件 | vanilla | oracle | visual | oracle 保留 | Δx−v [CI] | Δx−g [CI] |
|---|---|---|---|---|---|---|
| orig | 28 | 29 | 29 | 1/1 | +0.03 [0, +0.10] | 0 |
| trans | 13 | 18 | 20 | 3/6 | +0.23 [+0.07, +0.40] | +0.07 [−0.13, +0.27] |
| trans_y | 11 | 12 | **15** | 2/2 | +0.13 [+0.03, +0.27] | +0.10 [0, +0.20] |

按条件（入篮）：orig 28/29/29；trans 6/9/11；trans_y 11/11/11。

- milk `orig` vanilla 已达 28/30，天花板效应明显，扰动条件才是有效区分面。
- **visual 抓取总数反超 oracle**（64 vs 59）：trans_y 上 visual 有 2 例 oracle 未救回的"visual-only"救回（`init26_trans_y`、`init43_trans_y`），trans 上 5 例。Δx−g 区间均跨 0，不能断言 visual 优于 oracle，只能确认**不低于**。
- oracle 本身有破坏（抓取 `init30_trans_y`、`init32_trans`；入篮 `init30_trans_y`、`init36_trans_y`），说明纠偏并非单调有益。

## 4. 冻结状态下的视觉质量

| | miss 率（抬起前） | XY 误差中位数（抬起前） |
|---|---|---|
| butter 新位置 | 8.5% | **12.9mm** |
| milk 新 init | 15.6% | 4.6mm |

- butter 新位置的 XY 误差（12.9mm）明显大于三 A 期水平（y 方向系统偏差相关），与"oracle 保留率仅 62.5%、Δx−g 为负"一致：视觉定位是 butter 上的主要差距来源，方向以 y 平移最弱（保留 2/5）。
- milk 误差仍小（4.6mm），visual 基本复现 oracle 收益，且有少量 oracle 未覆盖的救回。
- **触发覆盖**：visgate 全程至少一次纠偏激活的回合 butter 53/60、milk 52/90；milk 约四成回合 2C 从未激活（多数 vanilla 本已成功），这些回合三臂结局相同，稀释了全条件差值但不影响方向结论。

## 5. 与阶段三 A 的对比（统一口径）

**口径**：两阶段均取 trans+trans_y 平移子集；分母=该子集状态数；救回交集=oracle 救回 ∩ visual 救回（即 oracle 保留数）；区间=init 簇 bootstrap 95%（同一实现 `frozen_review.paired_ci` / `analyze_frozen_transfer.cluster_ci`，均按 init 成簇重采样簇内均值）。

### 5.1 抓取终点（平移子集）

| | n | v / g / x | oracle 救回 → visual 保住（交集） | Δx−v [CI] | Δx−g [CI] |
|---|---|---|---|---|---|
| 三 A butter（init30–49 原位置） | 40 | 21 / 35 / 26 | 15 → 6（40%） | +0.125 [0.000, +0.275] | −0.225 [−0.350, −0.100] |
| 三 B butter（init30–49 新位置） | 40 | 13 / 29 / 23 | 16 → 9（56%） | +0.250 [+0.125, +0.375] | −0.150 [−0.325, 0.000] |
| 三 A milk（init00–19） | 40 | 14 / 24 / 24 | 10 → 5（50%） | +0.250 [+0.075, +0.425] | 0.000 [−0.175, +0.175] |
| 三 B milk（init20–49） | 60 | 24 / 30 / 35 | 8 → 5（62%）+7 新增 | +0.183 [+0.083, +0.283] | +0.083 [−0.033, +0.200] |

### 5.2 入篮终点（平移子集）

| | n | v / g / x | oracle 救回 → visual 保住 | Δx−v [CI] | Δx−g [CI] |
|---|---|---|---|---|---|
| 三 A butter | 40 | 18 / 22 / 21 | 10 → 3 | +0.075 [−0.050, +0.225] | −0.025 [−0.200, +0.175] |
| 三 B butter | 40 | 11 / 17 / 17 | 8 → 5 | +0.150 [+0.025, +0.300] | 0.000 [−0.100, +0.125] |
| 三 A milk | 40 | 11 / 13 / 14 | 4 → 2 | +0.075 [−0.050, +0.200] | +0.025 [−0.100, +0.175] |
| 三 B milk | 60 | 17 / 20 / 22 | 5 → 1 | +0.083 [+0.017, +0.150] | +0.033 [−0.050, +0.133] |

**判读**：两阶段方向一致——visual 对 vanilla 为正、与 oracle 差距随物体/终点不同。三 B butter 抓取 Δx−v 幅度不小于三 A；milk Δx−g 两阶段均跨 0。注意两阶段分母不同（三 A 每物体 40 状态、三 B butter 40 / milk 60），绝对数不可直接相减，比较以比例与区间为准。

### 5.3 milk 表述修正

三 B milk 抓取 visual 64 > oracle 59，但 Δx−g = +0.056 [−0.022, +0.133]（全条件）/ +0.083 [−0.033, +0.200]（平移）区间跨 0。**不能下非劣或"不低于 oracle"结论**，只能表述为：未观察到显著劣势，点估计略偏正；差异不显著。oracle 保留率低分母（9 例抓取救回）进一步限制了推断力。

### 5.4 butter 丢失的 oracle 平移救回：首次执行动作分歧（三 A，抓取终点 9 例）

分歧锚点 = 两臂 exec_action XY 首个非零差步 t0（raw 单位；1 raw ≈ 0.0128m）；另报 5mm/2mm 越界步。共同前缀统计 fresh 估计误差与 miss/held。

| 病例 | t0 | 5mm 步 | 归因 | gate/visual 首触发 | 前缀 fresh exy 中位 | 前缀 miss/held |
|---|---|---|---|---|---|---|
| init36_trans_y | 52 | 64 | 均触发但修正向量不同 | 52 / 52 | 8.6mm | 0/0 |
| init37_trans_y | 46 | 63 | 均触发但修正向量不同 | 46 / 46 | 6.3mm | 0/0 |
| init41_trans_y | 46 | 56 | 均触发但修正向量不同 | 46 / 46 | 8.0mm | 0/0 |
| init43_trans_y | 50 | 56 | 均触发但修正向量不同 | 50 / 50 | 7.4mm | 0/0 |
| init45_trans_y | 44 | 55 | 均触发但修正向量不同 | 44 / 44 | 7.6mm | 0/0 |
| init46_trans_y | 49 | 59 | 均触发但修正向量不同 | 49 / 49 | 7.1mm | 0/0 |
| init47_trans_y | 55 | 63 | 均触发但修正向量不同 | 55 / 55 | 6.5mm | 0/0 |
| init34_trans | 54 | 54 | **触发时机不同（gate 触发，visual 未触发）** | 54 / 无 | 16.5mm | 0/0 |
| init37_trans | 49 | 49 | **触发时机不同（gate 触发，visual 未触发）** | 49 / 无 | 15.1mm | 0/0 |

- trans_y 丢失 7 例全部是"同步触发、修正向量不同"：视觉 fresh 估计误差 6–9mm（y 向系统偏差）使修正方向/幅度偏离，未检/held 均无。
- trans 丢失 2 例（init34/37_trans）模式相反：前缀误差更大（15–17mm），gate 于 t0 步触发而 visual 全程未触发——估计偏差使 CorrState 触发条件从未满足。
- 入篮终点另有 7 例丢失（init31/36/37/41/43/45/47_trans_y，与抓取丢失高度重叠），均与上述同因。

## 6. 局限

- milk 验证集物体位置与标定集分布重叠 ≤1cm，独立性弱于 butter；milk 的高分不能外推到全新空间位置。
- butter 基底 init（30–49）虽位置全新，但曾参与仿射拟合（机器人/相机布局相同），验证的是"位置迁移"而非"场景迁移"。
- oracle 救回保留率分母小（butter 24、milk 9），比例估计粗糙。
- 逐状态表见 `results/frozen_transfer_{butter,milk}.csv`；汇总见 `results/frozen_transfer_summary.json`。

## 7. 可复现

```bash
# 冻结清单与 hash
cat runs/predictability/frozen_visual_transfer_2026-11-16.json
# 跑批（已实现 hash 自检；约 5h，450 回合）
scripts/run_frozen_batch.sh
# 分析
python scripts/analyze_frozen_transfer.py
```
