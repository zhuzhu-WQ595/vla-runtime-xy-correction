# FIXED_XY_REFERENCE_CONTROL — 类别固定 XY 参考点机制对照

**冻结**：2026-11-17。协议 `runs/predictability/fix_xy_ref_spec_2026-11-17.json`
（代码哈希：runner `75f752637083`、analyze_fixlock `21a5018d36e0`）。
**性质**：已分析状态（三 B 平移子集）上的机制对照（ablation），非独立确认。

## 1. 设计

- **对照臂 `fixlock`**：与 `vislock` 完全相同——首次有效检测时刻锁定、漏检回退
  （tgt=eef）、类高 z 保持该次估计、实时 EEF、CorrState 触发/闭合屏蔽/动作参数
  全不变。**唯一改动**：`locked_est[:2]` 替换为该类固定点。
- **固定点（开跑前冻结）**：开发集全部平移状态 `meta.locked_est[:2]` 的中位数——
  butter **(−106.4, −214.8)mm**（n=14）；milk **(−93.35, −216.05)mm**（n=16）。
  注意 dev 估计双峰：trans 簇 ≈(−85,−238)/(−64,−240)，trans_y 簇
  ≈(−126,−192)/(−117,−192)，类中位点位于两簇之间。
- **等价实验核查**：arm 集合 {vanilla,corr,gate,diag,branch,visgate,vislock,
  poseb} 及结果目录中无固定参考点臂 → 无先例，新建。
- **状态**：butter init30–49 ×{trans,trans_y} n=40；milk init20–49 ×{trans,trans_y}
  n=60（三 B 平移子集）；配对基线 `fzb/fzm_*_vislock.npz`。
- **判定**：init 簇 bootstrap 95%（10000 次，seed 0），差值 = fixlock − vislock。

## 2. 参考点距离（先验判别力）

| | 中位 | 范围 | 无锁定 | 近重合 ≤3mm |
|---|---|---|---|---|
| butter trans | 44.0mm | 10.4–95.4 | 0 | 0 |
| butter trans_y | 40.6mm | 8.5–71.1 | 1(init44) | 0 |
| milk trans | 32.5mm | 29.8–44.3 | 0 | 0 |
| milk trans_y | 39.5mm | 33.0–44.3 | 0 | 0 |

**全部 99 个有锁状态两参考点间距 ≥8.5mm —— 无近重合，对照区分能力满格**。
唯一无锁定状态（butter init44_trans_y）两臂均无估计、路径等价，自然配对。

## 3. 结果（n=100 配对）

### 稳定抓取 g（主终点）

| 组 | n | vislock | fixlock | 救回（v✗→f✓） | 破坏（v✓→f✗） | Δ CI |
|---|---|---|---|---|---|---|
| butter trans | 20 | 10 | 6 | init46 | init34,41,47,48,49 | **−0.200 [−0.400, 0.000]** |
| butter trans_y | 20 | 11 | 8 | — | init43,48,49 | −0.150 [−0.300, 0.000] |
| milk trans | 30 | 20 | 20 | init28,39,41,49 | init25,32,35,37 | 0.000 [−0.200, 0.200] |
| milk trans_y | 30 | 13 | 10 | init30 | init36,44,46,47 | −0.100 [−0.233, 0.033] |
| **合计** | 100 | **54** | **44** | 6 | 16 | −0.100 [−0.200, −0.010]¹ |

### 完整入篮 t（次终点）

| 组 | n | vislock | fixlock | 救回 | 破坏 | Δ CI |
|---|---|---|---|---|---|---|
| butter trans | 20 | 8 | 5 | init44 | init34,47,48,49 | −0.150 [−0.350, 0.050] |
| butter trans_y | 20 | 7 | 7 | init30 | init43 | 0.000 [−0.150, 0.150] |
| milk trans | 30 | 12 | **18** | init20,23,24,29,34,41,44,49 | init32,45 | **+0.200 [0.000, 0.400]** |
| milk trans_y | 30 | 11 | 10 | init30 | init36,44 | −0.033 [−0.133, 0.067] |
| **合计** | 100 | 38 | 40 | 11 | 9 | +0.020 [−0.070, +0.100]¹ |

¹ 合计 CI 按 `(obj,init)` 簇键合并重采样（复核：50 簇、每簇恰 2 状态、无重复
tag → 簇等权=状态等权），实现见 `scripts/analyze_fixlock.py::pooled_ci`，
值存 `fixlock_ref_summary.json::pooled`。

### milk trans 交叉账（抓取四类 × 入篮翻转）

抓取 20=20 完全平手，但入篮 12→18。钱在何处：

| 抓取格 | n | 入篮同胜 | 入篮同败 | 入篮救（v✗→f✓） | 入篮破（v✓→f✗） |
|---|---|---|---|---|---|
| g 同胜 | 16 | 6（init26,27,30,42,46,47） | 3（init21,36,40） | **6**（init20,23,24,29,34,44） | 1（init45） |
| g 救 | 4 | 2（init28,39） | — | 2（init41,49） | — |
| g 破 | 4 | 1（init37） | 2（init25,35） | — | 1（init32） |
| g 同败 | 6 | 1（init48†） | 5（init22,31,33,38,43） | — | — |

† init48 两臂均无抓取记录但任务成功（推/碰入篮）。
**8 个入篮救中 6 个发生在抓取已同胜的状态**——fixlock 的获益集中在放置阶段，
与抓取本身无关；见 `fixlock_ref_summary.json::milk_trans_crosstab`。

## 4. 判读

- **抓取端"首个有效视觉 XY"明确优于类别固定点**：合计 Δ=−0.10 [−0.19,−0.01]，
  butter 两组方向一致为负（trans −0.20、trans_y −0.15）；milk trans_y −0.10、
  milk trans 净零（4 救 4 破对称）。固定点平均偏 32–44mm 时，逐状态估计位置信
  息对**把抓取做对**是有贡献的——"有一次纠偏"并不等于"纠偏内容不重要"。
- **入篮端出现异常方向**：milk trans 上 fixlock 反而 +0.20 [0.00, 0.40]
  （8 救 2 破）且抓取完全相等（20=20）——交叉账显示 6/8 救发生在抓取同胜
  状态，获益集中在放置阶段。
  **[待验证解释]**：milk trans 固定点相对真实估计 (−62,−240) 偏 (−31,+24)mm，
  或把抓取位姿系统拉向篮筐一侧（系统性偏置巧合）。此解释未经核实，不可外推；
  需按抓取点几何与轨迹复核后采信。
- **区分能力**：所有状态间距 ≥8.5mm，无近重合稀释；唯一无锁定状态
  （init44_trans_y）两臂同为 miss 回退。
- **边界**：机制对照在已分析状态上进行；固定点定义（类中位数）若改逐 cond 中位
  数，对照含义会变（当前单点同时跨两个扰动簇）。

## 5. 主结果图

![fixlock_main](results/fixlock_main.png)

`results/fixlock_main.png`：四臂（vanilla / 类别固定 XY / 首次视觉 XY / 逐帧
视觉 XY）× g,t × butter,milk 分面；面板上方标注固定 XY vs 首次视觉 XY 的逐
cond 配对胜负数与区间。butter=新位置验证集（init30–49, n=40），milk=新 init
验证集（init20–49, n=60）；固定点对照复用已分析状态，属机制对照。
生成：`scripts/plot_fixlock_main.py`。

## 6. 文件

- 协议：`runs/predictability/fix_xy_ref_spec_2026-11-17.json`
- 逐状态表：`results/fixlock_ref_table.csv`；汇总：`results/fixlock_ref_summary.json`
- 回合：`results/fzbfx_init{30-49}_{trans,trans_y}_fixlock.npz`（40）+
  `fzmfx_init{20-49}_{trans,trans_y}_fixlock.npz`（60）
- 批次日志：`results/fixlock_logs/`；分析：`scripts/analyze_fixlock.py`
- runner 改动：`run_correction_eval.py` `arm="fixlock"`（仅锁定 XY 替换，见 §1）
