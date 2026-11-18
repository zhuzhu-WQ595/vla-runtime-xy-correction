# 冻结视觉语言动作策略的运行时横向纠偏：目标参考点来源与更新方式的配对消融

**PAPER_DRAFT_v2.1 · 2026-11-17 · 中文初稿**
数字均沿用冻结报告最终复核版本，可回溯到仓库 CSV/JSON/脚本（各表脚注）。

## 摘要

目标位置偏移会使冻结视觉语言动作策略在接近和闭合阶段出现错位。我们在 LIBERO 的 butter 和 milk 抓取入篮任务中，固定同一 OpenVLA 检查点，并研究运行时横向纠偏对目标参考点来源和更新方式的依赖。在 100 个平移状态上，原策略、类别固定参考点、首次有效视觉定位和逐帧视觉定位分别取得 37、44、54 和 58 次稳定抓取。首次有效视觉定位相对类别固定点的配对增益为 10 个百分点，95% 簇 bootstrap 区间为 [1,20]。逐帧更新在两物体上各净增加两次抓取，分物体区间均跨零。四种策略分别完成 28、40、38 和 39 次完整入篮，表明抓取改善与任务完成需要分别评价。局部动作候选在开发状态中显示可恢复空间，但冻结选择器在 32 个确认状态上抓取 24 次，保持基线动作为 25 次。该研究在限定的单策略、两物体设置中，量化了本回合位置信息的作用，并呈现进一步跟踪与动作选择的证据边界。

**关键词**：视觉语言动作模型；运行时纠偏；目标参考点；消融对照；仿真验证

---

## 1. 研究问题与主张

研究问题：**当冻结 VLA 策略因目标物体位移而抓空时，一个只改写动作输出 XY 通道的运行时纠偏层，需要什么样的目标参考点来改善稳定抓取及完整任务表现？**

- **RQ1 有效性**：不依赖真值的视觉纠偏，在留出验证状态上是否仍有正收益？
- **RQ2 逐回合参考点的价值**：纠偏需要的是"本回合这一次的目标位置"，还是类别级固定参考点已足够？
- **RQ3 更新方式**：首次有效定位与逐帧更新差多少？
- **RQ4 动作选择（进一步分析）**：抓取位姿是否存在局部候选余量，且能否在执行前几何上被在线选择？

**主张**：①视觉纠偏在留出验证状态上呈正收益（RQ1）；②抓取端收益依赖逐回合的目标参考点（RQ2）；③首次有效定位已捕获较大部分观察净增益（RQ3）；④局部动作余量存在，但执行前几何未提供超过"最优每物体固定动作"的证据（RQ4）。

## 2. 方法（复现规格）

### 2.1 任务、策略与控制循环

LIBERO `libero_object` pick-and-place：Franka Panda 7-DoF；回合上限 280 步（官方值）。环境创建沿用 `get_libero_env`（bddl 文件 + 256² 相机），未显式覆盖控制频率——**本机安装版 `bddl_base_domain` 默认 `control_freq=20`（读取确认，50ms/控制步）**；本文所有"步"指控制步，稳定保持判据 15 步 ≈0.75s。模型前向推理墙钟时间与视频回放帧率（30fps MP4）均与控制周期无关，文中不混用。环境/代码版本以 `runs/predictability/` 各冻结协议内的代码哈希为准。

策略：OpenVLA `openvla/openvla-7b-finetuned-libero-object`（已针对 LIBERO-Object 微调的官方检查点 [1,2]），`load_in_4bit=True` 加载；本实验全程权重冻结，"不微调"仅指本研究干预过程。动作反归一化键 `unnorm_key="libero_object"`（`model.norm_stats` 内断言）。图像管线按官方 `get_libero_image`：`agentview_image` 先 180° 旋转匹配训练预处理，再 lanczos resize 至 OpenVLA 输入尺寸（224）；夹爪动作经 `normalize_gripper_action` 二值化与 `invert_gripper_action` 后执行。任务：`butter`（task_id=6，新物体位置需 `--pos_override`）与 `milk`（task_id=7）。扰动在回合初始施加于目标物体 XY：`orig`（不动）、`trans`（+5cm x）、`trans_y`（+5cm y）。

### 2.2 纠偏控制器 CorrState（所有纠偏臂共用）

**运行时输入**（每步、因果）：`robot0_eef_pos`（z 与 XY）、上一步与本步 gripper 闭合命令、`eef_z` 近 4 步历史、按臂而异的**目标点** `tgt=(t_xy, t_z)`。**离线评分真值**：`obs[f"{TARGET}_pos"]` 的仿真真值位置、接触计数、任务成功判定——**视觉臂控制器运行时永不读取**；真值只进入 (a) `gate` 特权参照臂的目标供给，(b) 事后打分，(c) 视觉资产的离线标定（§2.3）。上一步闭合命令参与触发进入判定；本步闭合命令参与闭合屏蔽，两者分别列出。

- **触发进入**：上一步夹爪开命令 `∧ z_gap=eef_z−t_z ∈ [0,55]mm ∧`（近 4 步净下降 ≥2mm `∨ z_gap<20mm`）`∧ |t_xy−eef_xy|>25mm`。
- **保持与释放**：保持需 `|err|>15mm ∧ z_gap∈[−5,75]mm ∧` 夹爪开；任一破坏即释放。
- **修正律**：活动期 `action[:2]=clip(15·err_xy, ±0.8)`——**`err_xy` 以米输入**，`±0.8` 为策略归一化动作量限幅（非位移单位）；实测一步横向位移状态依赖、量级 ~1cm，不作精确换算。其余通道（z、姿态、夹爪）保留策略原输出。
- **覆盖何时结束**（与最终代码一致的清单）：(i) 滞回释放（lat≤15mm 或 z_gap 出窗或夹爪闭合）；(ii) **闭合屏蔽**：首次闭合命令当步起横向覆盖被屏蔽、raw 直通，夹爪重开解除；(iii) 回合结束。
- **漏检回退**（视觉臂）：估计器内部持用最近估计 ≤`LAST_HOLD`=5 步（见 §2.3）；持用耗尽返回空后，`t_xy←eef_xy`（lat=0 抑制触发/保持）、`t_z←eef_z`。

### 2.3 各臂定义与比较的实际干预

五臂间不存在"唯一变量"的统一定义；下表按实际干预列出每个比较改变的量。

| 臂 | 控制器 | 目标 XY | t_z | 漏检行为 | 更新 |
|---|---|---|---|---|---|
| **vanilla** | 无 | — | — | — | — |
| **gate**（特权参照） | CorrState | 仿真真值每步 | 真值 z | 无 | 每步 |
| **visgate** | CorrState | 逐帧视觉估计（含 ≤5 步持用） | 类 `z_prior` | 持用耗尽→EEF | 每步 |
| **vislock** | CorrState | 首个非空估计锁定为整回合目标 | 锁定帧 `z_prior` | 锁定前同 visgate；**锁定后持续用锁定值，实时漏检不覆盖** | 仅锁定前 |
| **fixlock** | CorrState | 同 vislock 锁定逻辑，锁定瞬间 XY 替换为类固定点 | 锁定帧 `z_prior` | 同 vislock | 仅锁定前 |

| 比较 | 实际回答的问题 | 共同项与残留差异 |
|---|---|---|
| vanilla↔visgate | 整套"感知+触发+控制"系统对原策略的总效应 | 同时引入感知、触发窗与覆盖控制 |
| gate↔visgate | 真值目标供给与当前视觉目标实现的差距 | 可能含 XY 估计误差、z_prior 误差与漏检行为共同作用 |
| vislock↔fixlock | 首次有效视觉 XY 相对类别固定 XY 的贡献 | 共享锁定时刻、t_z 来源、漏检路径与控制器，**唯一差异为锁定 XY 的取值** |
| visgate↔vislock | 目标坐标持续更新的净效应 | 两者均持续读实时 EEF；锁定后对漏检/估计漂移的响应路径不同 |

**视觉估计器**（`vision_target.py`，算法详见附录 B）：类模板 `tmpl_{cls}.png` 的 NCC 匹配（`score_min=0.55`、k=4 候选）→ 候选底中点先经理论反投影、以 `WORKSPACE_RECT` 工作区矩形过滤，仅区内候选参与评分取最高 → `pix2w_{cls}.json` 类仿射映到世界 XY，`est_z=z_prior`；无区内候选→miss，持用 ≤5 步后返回空。**类别资产**（换物体所需）：模板 PNG、`pix2w_{cls}.json`（类轨迹像素-真值离线拟合）、`z_prior`（butter 0.0087m、milk 0.0697m）、fixlock 的 `FIX_XY`；**仿射与 z_prior 的标定使用离线真值**，运行时视觉臂不读真值。
**fixlock 类固定点**（开跑前冻结）：开发集全部平移状态 `locked_est[:2]` 中位数——butter (−106.4,−214.8)mm、milk (−93.35,−216.05)mm；其余与 vislock 逐行相同。

### 2.4 终点、配对与统计口径

- **终点**：稳定抓取 g（抬升 4cm 且保持 15 控制步，主终点）、完整入篮 t（任务成功，次终点）——抓取改善与任务完成分别评价。
- **配对**：同一 `(init,cond)` 状态各臂独立回放，差值为臂间配对差。
- **区间**：init 簇 bootstrap 95%（3B/3C 2000 次、fixlock 10000 次，seed 0）。**差值方向全文统一 `ΔA−B = A − B`**，百分点记 pp。
- **状态来源标签**：`[新位置]`=butter init30–49 换 20 个新物体 XY（距原位 ≥1.5cm）；`[留出 init]`=milk init20–49（位置分布与标定集重叠 ≤1cm）；`[确认集]`=四 C 32 个未参与候选/阈值选择的状态；`[机制]`=复用已分析状态的消融。

## 3. 结果

### 3.1 现象：平移空抓病例（问题动机）

![trans 空抓俯视](results/trans_empty_grasp_topdown.png)

`results/trans_empty_grasp_topdown.png`（`scripts/verify_trans_cases.py`）：目标 +5cm x 平移后，vanilla 沿原接近路径下落，闭合时指尖中点落在物体旁侧。此为"位置失配→空抓"的典型表现，是纠偏层要消除的失败模式。

### 3.2 RQ1：留出验证状态上的有效性 `[新位置]/[留出 init]`

三 B：butter init30–49 全部换**新物体位置**；milk init20–49 为**留出 init**（独立性弱于 butter）。主报告 trans+trans_y 平移子集：

| 物体 | n | 终点 | vanilla | gate(参照) | visgate | Δvisgate−vanilla [CI] | Δvisgate−gate [CI] |
|---|---|---|---|---|---|---|---|
| butter | 40 | g | 13 | 29 | 23 | **+0.250 [0.125,0.375]** | −0.150 [−0.325,0.000] |
| | | t | 11 | 17 | 17 | +0.150 [0.025,0.300] | 0.000 [−0.100,0.125] |
| milk | 60 | g | 24 | 30 | 35 | **+0.183 [0.083,0.283]** | +0.083 [−0.033,0.200] |
| | | t | 17 | 20 | 22 | +0.083 [0.017,0.150] | +0.033 [−0.050,0.133] |

- 两物体、两终点的 Δvisgate−vanilla 区间均不含 0——视觉纠偏在留出状态上呈正收益。
- butter 抓取上视觉目标供给与真值目标供给存在差距（Δ−0.150，区间上界触 0；butter 新位置估计误差中位 12.9mm），该差距的 XY/Z/漏检构成未作分离；milk 上未观察到显著差距，不构成非劣结论。
- 来源：`results/frozen_transfer_{butter,milk}.csv`、`frozen_transfer_summary.json`；`scripts/analyze_frozen_transfer.py`；`FROZEN_VISUAL_TRANSFER_REPORT.md` §5。

### 3.3 RQ2+RQ3：同一批 100 个平移状态上的参考点来源与更新方式消融（主结果）

![fixlock_main](results/fixlock_main.pdf)

四臂共用同一批三 B 平移状态（butter n=40、milk n=60）。**100 状态主表**：稳定抓取 vanilla 37、fixlock 44、vislock 54、visgate 58；完整入篮 28、40、38、39。

**RQ2 逐回合参考点的价值**（vislock−fixlock，`[机制]`）：

| 物体×条件 | n | 终点 | vislock | fixlock | 配对翻转 | Δvislock−fixlock [CI] |
|---|---|---|---|---|---|---|
| butter trans | 20 | g / t | 10 / 8 | 6 / 5 | +5 −1 / +4 −1 | **+0.200 [0.000,0.400]** / +0.150 [−0.050,0.350] |
| butter trans_y | 20 | g / t | 11 / 7 | 8 / 7 | +3 −0 / +1 −1 | +0.150 [0.000,0.300] / 0.000 [−0.150,0.150] |
| milk trans | 30 | g / t | 20 / 12 | 20 / 18 | +4 −4 / +2 −8 | 0.000 [−0.200,0.200] / **−0.200 [−0.400,0.000]** |
| milk trans_y | 30 | g / t | 13 / 11 | 10 / 10 | +4 −1 / +2 −1 | +0.100 [−0.033,0.233] / +0.033 [−0.067,0.133] |
| **合计** | 100 | **g / t** | **54 / 38** | **44 / 40** | +16 −6 / +9 −11 | **g +0.100 [0.010,0.200]** / t −0.020 [−0.100,0.070] |

- 抓取端：首次有效视觉定位相对类别固定点 +10pp，合并区间不含 0——**逐回合参考点有贡献**。逐回合参考与类固定点间距全 ≥8.5mm（中位 32–44mm）；相对**真值**的误差为另一量（butter 新位置估计误差中位 12.9mm），两类距离不混用。
- 入篮端存在反向子效应：milk trans 上 fixlock 比 vislock 多完成 6 例任务（Δvislock−fixlock −0.200）。**仅按配对账表报告**：8 例 task 救回中 6 例发生在两臂均达稳定抓取判据的状态，即分歧定位于稳定抬起之后的下游阶段；抓取时形成的夹持姿态仍可能参与下游失败，不能据此判定为放置阶段的独立获益。
- 来源：`results/fixlock_ref_table.csv`、`fixlock_ref_summary.json`；`scripts/analyze_fixlock.py`、`plot_fixlock_main.py`；`FIXED_XY_REFERENCE_CONTROL.md`。

**RQ3 更新方式**（visgate−vislock，`[机制]`，三 C）：

| 物体 | n | g：visgate/vislock | Δg [CI] | t：visgate/vislock | Δt [CI] |
|---|---|---|---|---|---|
| butter | 40 | 23 / 21 | +0.050 [−0.075,0.175] | 17 / 15 | +0.050 [−0.050,0.175] |
| milk | 60 | 35 / 33 | +0.033 [−0.050,0.117] | 22 / 23 | −0.017 [−0.067,0.033] |

- 逐帧更新在两物体上各净增加两次抓取（+5.0pp、+3.3pp），分物体区间均跨零；本样本不支持两种更新方式等效或增量被紧密限定的结论。**正文结论依赖上述成对成功率**。
- **病例归因（仅已核实实例）**：19 个翻转状态中，2 例在分歧时刻实时估计显著优于锁定估计（同一轨迹、同一观测索引下的 `lock_err_at_div`/`live_err_at_div`），符合"逐帧供给更准参考点"的解释：
  - milk `init26_trans_y`：分歧索引 t_div=54 前物体自初始位置已超阈位移（t_mv=53；整回合最大 44.6mm）；同索引误差 lock 10.2 / live 1.7mm——**符合接触前位移被逐帧跟踪**。
  - butter `init40_trans`：分歧索引 t_div=53；同索引误差 lock 19.0 / live 13.7mm；但**分歧前物体位移未超阈**（t_mv=81 在分歧之后，整回合最大 42.0mm）——收益来源不唯一指向"跟踪已移物体"，分歧时点的估计质量差异同样可解释。夹爪中点-目标距离首次 <25mm（几何距离代理 `t_con`；`ncon_tgt` 含桌面静置接触不可用）发生于 t=156，晚于分歧。
  - 另有 9/19 例在分歧时刻两臂估计误差相当；其余未定，**未定病例可能包含其他收益机制，本文不作机制主张**。
- 归因口径变更说明：早期审计按初筛列为 5/19 候选、`init40_trans` 记 undetermined；`results/corrige_3c.json` 复核以同一分歧索引下的锁定/实时误差、目标位移时序（`t_mv`）与夹爪-目标几何距离重新判定，收敛为 2/19"实时估计显著更准"，其中仅 milk 例满足"分歧前已位移"。
- 来源：`results/online_vs_lock_{butter,milk}.csv`、`corrige_3c.json`；`scripts/analyze_online_vs_lock.py`、`corrige_3c.py`；`ONLINE_VS_INITIAL_TARGET.md`。

### 3.4 RQ4：局部动作余量与在线选择（进一步分析）

**余量存在**（四 A，`[机制]` 开发集 30 状态 ×5 候选，事后择优非可部署）：13 个基线抓取失败中 10 个能被某候选救回；沿闭合轴平移 `px` 为最优固定候选（救 6 破 3）；px 相对 c0 在**闭合邻近同一控制步的累计达到位姿差**中位 ~2–3mm（名义指令位移 5mm，非单步位移）；5 候选逐状态上限 g 27/30、t 21/30。候选偏置在 CorrState 门控之后施加于 `action[:2]`（0.039 归一化动作量/步），自决策步起最长 30 步或至**首次闭合+2 步**——作用时段与横向纠偏窗口不同，单独描述。来源：`results/pose_cand_{butter,milk}.csv`、`pose_cand_summary.json`；`scripts/analyze_pose_cand.py`。

**在线选择未超过保持基线动作**（四 C `[确认集]` 32 个未参与候选/阈值选择的状态；`c0` 定义为**保持 vislock 基线的原动作**，非 vanilla）：冻结规则 `pick px iff e<−2.0mm ∧ approach>0mm`（e=决策时刻闭合轴上有符号误差、approach=近 5 步沿轴运动；决策步=首个已锁定∧未闭合∧|eef−锁|<70mm）：

| 臂 | 抓取 /32 | 入篮 /32 |
|---|---|---|
| c0（保持基线动作） | **25** | 17 |
| px（固定平移候选） | 23 | 17 |
| sel（冻结选择器） | 24 | **18** |
| 按物体固定（开发集规则） | 21 | 16 |
| oracle(c0,px) 逐状态上限 | 27 | 21 |

- sel−c0 = −0.031 [−0.156,0.094]；sel−按物体基线 = +0.094 [0.000,0.188]（方向性提示非显著）。
- **归因**：选择器修复了冻结类别基线中的三个错误选择——butter 两例（init33_trans、init36_trans_y）、milk 一例（init11_trans）。在本确认集、c0/px 二选一、稳定抓取终点的限定下，事后按物体固定已取尽逐状态余量（27/32=上限）；**入篮端仍存在类内观察余量**（上限 21/32，事后类固定 18/32），当前规则未捕获。冻结类基线的失效主要表现为 milk→px 在新 init 上 8/16（c0 12/16），属类规则跨 init 泛化问题。
- 来源：`results/action_sel_confirm_table.csv`、`action_sel_confirm_summary.json`；`scripts/freeze_selector.py`、`analyze_selector_confirm.py`；`FROZEN_ACTION_SELECTOR_CONFIRMATION.md`。

## 4. 相关工作（含原文核对）

### 4.1 近邻方法比较表

| 方法 | 冻结策略 | 运行时感知输入 | 干预权限 | 目标时序 | 实验问题 |
|---|---|---|---|---|---|
| **ActionGround** [3] | OpenVLA/-OFT/Force-VLA/Generalist-VLA | 仿真场景状态：相位判定用碗/盘水平距离与抬升阈值（瞬时几何，无记忆/无接触传感）；动力学 oracle 供混合权重 | Branch A 相位规则修正（接近否决/抓取混合/运输提升平滑/放置减速）；Branch B 惯量加权动作混合（EL 残差仅作诊断记录） | 每步真值状态 | 训练免费纠偏对成功率/稳定性的总效应 |
| **ReGuide** [4] | π0.5 | RGB-D+GroundingDINO 点云配准到声明实例；**指令发出时一次登记，之后不跟踪** | 几何重绑定修正 WHERE、保留策略 HOW | 登记一次（≈本文首次有效定位） | oracle vs RGB-D grounding 的配对差（−7.0pp [−9.8,−4.4]） |
| **FIT-MAC** [5] | 冻结 VLA | GroundingDINO+VLM 解歧，目标实例绑定；验证器监测绑定置信度与执行进度 | 目标重绑定 或 有界局部控制器（逐轴尺度、阶段 waypoint） | 子任务内绑定+失效时重绑 | 闭环多智体纠偏的效应与模块消融 |
| **RTPC** [6] | 冻结 OpenVLA | 同族深度检索压力代理事件 | 命中时按检索方向做平移投影修正，旋转/夹爪不变 | 每步检索判定 | 独立复现与失败归因 |
| **CorrectVLA** [7] | 冻结 VLA | 人给任务级语言修正 | LLM 将语言转为动作维度/方向/幅度/时间窗，对策略输出施加**短时加性偏置** | 按修正窗 | 单句修正可修复的失败分类 |
| **PhysVLA** [8] | 即插即用推理时框架 | 物理约束状态 | 条件化物理约束修正 | 每步 | 物理一致性修正的效应 |
| **本文** | OpenVLA（固定检查点） | 分级消融：真值(gate)/逐帧视觉(visgate)/首帧锁定(vislock)/类固定点(fixlock) | 仅 `action[:2]` 横向覆盖，因果触发窗内，闭合屏蔽 | 按臂分级 | 同一受限控制器下参考点来源与更新方式的配对消融 |

### 4.2 逐条区别与定位

- **ActionGround** [3]：原文 §III-C、§IV 已核——跨四骨干主表采用 **Branch A**；Branch B 另在 OpenVLA 上单独评估，其混合权重来自惯量，EL 残差仅作逐步诊断记录、不控制混合或触发。相位判定只用瞬时几何状态（物体/目的地水平距离与抬升阈值），无记忆且无接触传感。OpenVLA 上成功率 36%→38%、稳定性 20.1%→36.8%（另有 T3/T6 两任务回退）；35%→95% 的注入偏置结果来自 **matched-seed Robosuite 仿真对照**，真实 Agilex Piper 仅为定性展示。与本文差异：其目标几何直接取仿真真值距离，**感知源未做消融**；本文在同一受限 XY 控制器内做参考点来源与更新方式的配对消融。
- **ReGuide** [4]：C.6 已核——在固定执行框架中比较 oracle 与 RGB-D grounding，**指令发出时一次登记、之后不视觉跟踪**（87.5% vs 94.5%，配对 −7.0pp [−9.8,−4.4]）。该节比较的是视觉与 oracle grounding，没有逐帧跟踪对照，其结果支持一次登记方案在所测系统中的有效性。本文增量：在同一控制器内进一步区分"首次有效定位 vs 逐帧更新 vs 类固定参考点"。
- **FIT-MAC** [5]：§3.3–3.5、§4.4、A.5 已核——目标实例**锁定**指语义实例绑定（供 grounded prompt 与恢复 waypoint），与本文锁定**世界 XY 参考点**含义不同；其执行失败用有界局部控制器恢复（验证器诊断触发、含放置阶段），本文候选仅在决策步施加固定偏置。其绑定置信度+失效重绑是本文未做的方向。
- **RTPC** [6]：§3.3 已核——检索触发式平移修正，含独立复现批次与失败归因；修正方向取自历史压力事件而非当前几何，与本文四 A–C 的观察余量/选择失败可对话。
- **CorrectVLA** [7]（*What Can One Sentence Fix?*，v2）：语言是人提供修正的**接口**；LLM 将其转成动作维度、方向、幅度与时间窗，最终对冻结策略输出施加短时加性偏置——与本文同样修改执行动作；区别在于修正信息来源（人的语言反馈 vs 几何估计）、任务级复用与位置反馈方式。
- **PhysVLA** [8]：即插即用推理时物理约束修正，与 ActionGround 同列"推理时注入"家族；未见目标参考点来源消融。

### 4.3 定位声明与缺口判断

**定位**：已有工作覆盖了一次登记定位（ReGuide）、实例绑定+有界恢复（FIT-MAC）、检索触发平移修正（RTPC）、特权真值相位修正（ActionGround）与语言修正通道（CorrectVLA）。**本文的可主张增量**：在同一受限 XY 控制器内，对目标参考点来源（类固定 vs 逐回合）与更新方式（首次有效定位 vs 逐帧）做配对消融，并用冻结选择器检验局部动作余量的在线可选择性——该配对口径在现有文献中未见同口径报告。

**与外部结果的关系**：ReGuide 在其固定执行框架中量化了视觉定位替代真值定位后的任务成功率变化，并展示了一次登记方案的可行性。本文进一步在一个受限横向纠偏器内，比较类别固定参考点、首次有效视觉参考点和逐帧视觉参考点。两项研究的模型、任务、感知输入、控制权限和终点不同，百分点差值应在各自实验内解释。若要主张本文控制器相对其他方法的性能优势，需要在共享检查点、状态、目标输入和执行预算下进行明确版本的配对评估。

**缺口判断**：可能改变主张的缺口是——当前"抓取端需逐回合参考点"基于类中位点与锁定估计相距 32–44mm（中位）的强分离设定；中等失配档结果未知。
## 5. 局限

- 仿真（LIBERO+OpenVLA 固定检查点）单策略单纠偏层；本主实验扰动仅 +5cm x 或 +5cm y 平移（早期旋转开发试验见附录 A 阶段记录），不含旋转/高度/动态位移。
- 视觉估计器为模板匹配基线；估计误差的 XY/Z/漏检构成未分离，结论对更高精度估计器的迁移未验证。
- milk 验证集位置分布与标定集重叠 ≤1cm；butter 基底 init 曾参与仿射拟合（验证位置迁移非场景迁移）。
- fixlock、milk 入篮子效应、逐帧翻转归因均系已分析状态上的机制对照；sel 结论限于 c0/px 二选一、本扰动族与本状态分布。

## 参考文献

1. Kim, M. J., et al. *OpenVLA: An Open-Source Vision-Language-Action Model*. arXiv:2406.09246, CoRL 2024.
2. Liu, B., et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*. arXiv:2306.03310, NeurIPS 2023 Datasets and Benchmarks.
3. Affiliation: Chandra, N., Thareja, M., Damodaran, S., Lin Wang, A. *ActionGround: Training-Free Runtime Refinement of Frozen VLA Policies*. arXiv:2609.33256.
4. *Correcting WHERE, Preserving HOW: Compositional Generalization for Vision-Language-Action Models via Referential Guidance* (ReGuide). arXiv:2609.38616v1.
5. *FIT-MAC: A Closed-Loop Multi-Agent Framework for Robust Vision-Language-Action Models*. arXiv:2610.28109v1.
6. *Retrieval-triggered Action Projection Adjustment for Frozen VLA Policies: An Empirical Study and Failure Analysis* (RTPC). arXiv:2610.24573v1.
7. *What Can One Sentence Fix? A Failure Taxonomy and a Minimal Language Correction for Frozen VLA Policies* (CorrectVLA). arXiv:2608.29967v2.
8. *PhysVLA: Towards Physically-Grounded VLA for Embodied Robotic Manipulation*. arXiv:2606.13886.

## 附录 A. 开发阶段编号 ↔ 本文映射

| 阶段 | 本文位置 | 产物 |
|---|---|---|
| 一/二 | §2.2 CorrState 开发、闭合屏蔽、病例 | `run_correction_eval.py`、`results/gate_cases.png` |
| 三 A | 感知资产开发（模板/仿射/z_prior） | `results/perception_provenance.csv` |
| 三 B | §3.2 RQ1 留出验证 | `FROZEN_VISUAL_TRANSFER_REPORT.md` |
| 三 C | §3.3 vislock vs visgate | `ONLINE_VS_INITIAL_TARGET.md`、`results/corrige_3c.json` |
| 四 A/B/C | §3.4 候选余量/离线选择/确认 | `LOCAL_GRASP_CANDIDATE_HEADROOM.md`、`LOCAL_ACTION_SELECTABILITY.md`、`FROZEN_ACTION_SELECTOR_CONFIRMATION.md` |
| 固定点对照 | §3.3 fixlock | `FIXED_XY_REFERENCE_CONTROL.md`、`results/fixlock_main.{png,pdf}` |

## 附录 B. 视觉估计器算法（`vision_target.py`）

```
estimate(rgb_raw):
  img = resize(rgb_raw, 224) → BGR
  r = matchTemplate(img, tmpl_cls)                 # NCC
  repeat k=4: 取 argmax(score≥0.55) → 底中点 px
      px → 理论反投影 unproject → world XY
      若 XY ∉ WORKSPACE_RECT(x∈[-0.20,0.10], y∈[-0.32,-0.10]): 标 outside，不参与
  区内候选 score 最高 → pix2w 仿射 → est_xy；est_z = z_prior
  无区内候选 → miss：持用最近估计（last_age≤LAST_HOLD=5 步），否则返回空
控制器侧（视觉臂）：本步空 → t_xy←eef_xy, t_z←eef_z（vislock/fixlock 锁定后恒用 locked_est）
```

## 附录 C. 复现入口（具体可执行命令）

```bash
# butter（task_id=6，需新位置文件）
python scripts/run_correction_eval.py --task_id 6 --init_list 30-49 \
  --conditions trans,trans_y --pos_override runs/predictability/butter_newpos.json \
  --arm vislock --out_prefix fzb
# milk（task_id=7）
python scripts/run_correction_eval.py --task_id 7 --init_list 20-49 \
  --conditions trans,trans_y --arm vislock --out_prefix fzm
# 其余臂替换 --arm {vanilla|gate|visgate|vislock|fixlock}
python scripts/analyze_frozen_transfer.py    # RQ1 区间
python scripts/analyze_online_vs_lock.py     # RQ3 逐帧对照
python scripts/analyze_fixlock.py            # RQ2 配对+合并CI+交叉账
python scripts/plot_fixlock_main.py          # 主结果图（PNG+PDF）
python scripts/freeze_selector.py && python scripts/analyze_selector_confirm.py  # RQ4
```

冻结协议（取值/来源/哈希均先于结局）：`runs/predictability/{frozen_visual_transfer_2026-11-16,vislock_spec_2026-11-17,fix_xy_ref_spec_2026-11-17,pose_cand_spec_2026-11-17,action_sel_spec_2026-11-17}.json`
