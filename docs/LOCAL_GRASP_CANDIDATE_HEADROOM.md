# 阶段四 A：局部抓取位姿候选余量（Local Grasp Candidate Headroom）

日期：2026-11-17 · 冻结协议：`runs/predictability/pose_cand_spec_2026-11-17.json`
基线臂：**vislock**（visual_initial_lock，三 C 冻结臂）· 开发状态 30 个 · 每状态 5 候选 = 150 回合
数据：`results/f4b_*/f4m_*_pb_{c0,px,nx,ry,ly}.npz` · 矩阵：`results/pose_cand_{butter,milk}.csv` · 汇总：`results/pose_cand_summary.json` · 分析：`scripts/analyze_pose_cand.py` · runner：`run_correction_eval.py --arm poseb`

## 1. 协议（执行时冻结项）

- **决策时刻**：首个满足「已锁定视觉目标 ∧ 尚无闭合命令 ∧ 3D|eef−锁定目标|<70mm」的控制步。三项均在线可得；发生在闭合前。冒烟发现接近段仅 ~10 步，30mm 阈值过晚已上调至 70mm；偏置若瞬时仅施加一窗会被策略伺服回拉，故**偏置持续至首次闭合+2 步或上限 30 步**。
- **候选**（固定幅度，施加于 corr/屏蔽之后的最终动作）：
  - `c0`：基线（无干预，配对校验列）
  - `px`/`nx`：沿夹爪闭合轴 u=normalize(finger1−finger2 的 XY 投影) ±偏置，0.039 raw/步（≈0.5mm/步）
  - `ry`/`ly`：世界系 yaw ±0.2 raw/步，偏置 `action[3]`（实测世界-z 通道，exec[3]~world-z r=-0.71；首版误用 action[5] 实为绕 X 轴，该批结果已作废重跑）
- **配对**：每候选从相同 init+扰动全新重建，确定性保证前缀逐比特一致。**验证：60/60 对执行动作序列首个非零分歧恰在 cand_step**。
- **实际达到位姿差**（achieved，候选 vs c0 同绝对步）：
  - 平移 `du@close` 中位 **2–3mm**（指令名义 5mm，政策回拉打折）
  - yaw `dyaw@close` 中位 butter **ry≈13° / ly≈30°**，milk **ry≈8° / ly≈11°**——ly 不对称偏大，0.2/步在 butter 上已属较大扰动
- **无触发病例**：milk `init09_trans`（锁定太晚/从未接近，cand_step=空，按基线保留）
- 状态集（分层自 3A visgate 结局）：butter init{30,34,37,40,43,44,46}×{trans,trans_y}=14；milk init{00,01,04,06,09,12,15,17}×{trans,trans_y}=16

## 2. 逐状态候选矩阵（c0 px nx ry ly；✓成功 ·失败）

### butter · 稳定抓取 g / 完整入篮 t

| 状态 | g | t |
|---|---|---|
| init30_trans | ✓✓✓·✓ | ✓✓✓·✓ |
| init30_trans_y | ✓✓✓·· | ✓✓✓·· |
| init34_trans | ·✓✓·· | ·✓✓·· |
| init34_trans_y | ···✓· | ···✓· |
| init37_trans | ·✓✓·· | ·✓✓·· |
| init37_trans_y | ✓·✓·✓ | ✓···✓ |
| init40_trans | ✓✓✓·· | ✓✓✓·· |
| init40_trans_y | ✓··✓· | ✓···· |
| init43_trans | ✓✓✓·· | ✓✓✓·· |
| init43_trans_y | ···✓✓ | ···✓✓ |
| init44_trans | ✓✓✓·· | ✓✓✓·· |
| init44_trans_y | ✓✓··✓ | ✓···✓ |
| init46_trans | ✓✓✓·· | ✓✓✓·· |
| init46_trans_y | ✓···· | ····· |

### milk · 稳定抓取 g / 完整入篮 t

| 状态 | g | t |
|---|---|---|
| init00_trans | ···✓· | ···✓· |
| init00_trans_y | ✓✓✓✓· | ✓✓✓·· |
| init01_trans | ✓✓✓✓· | ·✓✓·· |
| init01_trans_y | ✓✓✓✓✓ | ✓✓✓·✓ |
| init04_trans | ····· | ····· |
| init04_trans_y | ····· | ····· |
| init06_trans | ✓✓✓✓✓ | ✓✓✓✓· |
| init06_trans_y | ·✓✓·· | ····· |
| init09_trans | ·····（无触发） | ····· |
| init09_trans_y | ·✓·✓· | ····· |
| init12_trans | ·✓✓✓✓ | ·✓✓✓✓ |
| init12_trans_y | ✓✓··· | ····· |
| init15_trans | ✓✓✓✓✓ | ✓✓✓✓· |
| init15_trans_y | ✓✓✓✓✓ | ✓✓✓✓✓ |
| init17_trans | ··✓✓· | ····· |
| init17_trans_y | ·✓··· | ····· |

## 3. 三问回答

### 稳定抓取（主终点）

| | butter(14) | milk(16) | 合计(30) |
|---|---|---|---|
| Q1 基线失败→≥1候选救回 | **4/4** | **6/9** | **10/13** |
| Q2 基线成功→某候选破坏 | 10/10 | 3/7 | 13/17 |
| Q3 最优固定候选 | px（救2 破3） | px（救4 破0） | px（救6 破3） |
| Q3 事后择优上限 | 4 | 6 | 10 |

### 完整入篮（次终点）

| | butter | milk | 合计 |
|---|---|---|---|
| Q1 | 4/5 | 3/11 | 7/16 |
| Q2 | 9/9 | 4/5 | 13/14 |
| Q3 最优固定 | px（救2 破3） | px（救2 破0） | px（救4 破3） |
| Q3 择优上限 | 4 | 3 | 7 |

逐候选明细（g）：butter 救回 px2/nx2/ry2/ly1，破坏 px3/nx3/ry9/ly7；milk 救回 px4/nx3/ry4/ly1，破坏 px0/nx1/ry1/ly3。

## 4. 判读与限制

- **有余量但不对称**：13 个抓取失败状态中 10 个能被某个候选救回（77%），平移 px 单独即可救 6 个且破坏最少（milk 0、butter 3）。说明基线抓取位姿确实留了局部余量，**沿闭合轴的平移是收益/代价比最好的固定候选**。
- **yaw 扰动破坏性大**：achieved yaw 中位 8–30°，大于"小 yaw"本意（0.2 raw/步已偏大；ly 不对称达 30°）。butter 上 ry/ly 破坏 9/7 个基线成功状态，说明现有姿态对 yaw 扰动敏感；但也救回 milk init00_trans、init17_trans 等，方向依赖强。
- **事后择优 ≠ 可部署策略**：oracle 10/13 只是这组 5 候选在已跑分支上的观察上限；选择器本身需要在线可判定依据，本阶段不评估。
- **achieved ≠ 指令**：平移指令名义 5mm 但因策略回拉只达 ~2–3mm；候选实际探索的邻域比名义小。个别状态 du@close 离群（px 196mm）是分歧后整段轨迹漂移，不视为位姿差。
- **无触发病例**：milk init09_trans 保留（锁定过晚）；决策规则本身在无锁定时退化为基线。
- 小样本（30 状态、开发集、两条件），不用于泛化结论；不入冻结验证集、不训练模型（按约定）。

## 5. 文件清单

- `runs/predictability/pose_cand_spec_2026-11-17.json` 冻结协议（含冒烟修订记录）
- `results/pose_cand_{butter,milk}.csv` 逐状态矩阵 + achieved 字段
- `results/pose_cand_summary.json` Q1/Q2/Q3 汇总
- `results/f4b_*/f4m_*_pb_*.npz` 全轨迹（eef_quat/cand_bias 含其中）
- `scripts/run_correction_eval.py`（`--arm poseb --cand`）、`scripts/analyze_pose_cand.py`
- `results/frozen_batch_logs/f4{b,m}_pb_*{,2}.log` 批次日志；`posecand{,_yaw}_master.log`
