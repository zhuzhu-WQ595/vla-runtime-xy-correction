# 阶段四 B：决策步执行前能否选择局部抓取动作（离线，零新增仿真）

日期：2026-11-17 · 数据：四 A 全部 150+60 分支 `results/f4{b,m}_*_pb_*.npz`（未新增回合）
分析：`scripts/analyze_action_select.py` · 输出：`results/action_select_summary.json`、
逐状态表 `action_select_states.csv`、**留出折决策表 `action_select_cv.csv`**
分组：15 个「物体–init」组（butter 7 + milk 8），同 init 两扰动条件始终同折。

## 1. 数据重生成与配对审计

### 固定候选成绩（成功数/30，| butter /14，milk /16）

| 终点 | c0 | px | nx | ry | ly |
|---|---|---|---|---|---|
| 稳定抓取 g（合计） | 17 | **20** | 18 | 13 | 9 |
| g（butter） | **10** | 9 | 9 | 3 | 4 |
| g（milk） | 7 | **11** | 9 | 10 | 5 |
| 完整入篮 t（合计） | 14 | 15 | 15 | 7 | 7 |
| t（butter） | **9** | 8 | 8 | 2 | 4 |
| t（milk） | 5 | 7 | 7 | 5 | 3 |

### 逐状态事后择优上限（观察上限，非可部署）

| 终点 | {c0,px} | {c0,px,ry} | 全部 5 候选 |
|---|---|---|---|
| 抓取 | 23/30 | **27/30** | 27/30 |
| 入篮 | 18/30 | **21/30** | 21/30 |

ry 加入把上限从 23→27（+4，即 §5 的四个新增覆盖）；nx/ly 不再增加。

### 配对审计（120/120 通过）

- 本次审计分母 **120 = 30 状态 × 4 候选**（px/nx/ry/ly 对 c0）。每对从相同 init+扰动重建，首个 `exec_action` 非零分歧应恰在 cand_step：**120 对全部满足，其中 4 对完全等价**。
- 4 对完全等价 = 唯一无触发状态 milk `init09_trans`（cand_step 为空）的 4 个候选——决策条件全程未满足，轨迹与 c0 逐比特相同。
- **原报告 "60/60" 分母**当时只核对了 px、nx 两个候选（30×2=60）；本次补齐 ry/ly 后为 120。
- 每候选「闭合前一步 / 闭合当步 / 闭合+2 步」的配对位姿差已逐状态写入 CSV（`*_{pre,at,post}_{du,dxy,dyaw}` 列）：平移 achieved 中位 ~2–3mm；yaw achieved 在**闭合前一步已基本形成**（pre≈at≈−10~−20°），闭合后仅小幅增长——见 §5。

## 2. c0 vs px：配对标签与决策步前特征

### 四类配对标签（稳定抓取 g）

- **px 救回（6）**：init34_trans、init37_trans、init06_trans_y、init09_trans_y、init12_trans、init17_trans_y
- **px 破坏（3）**：init37_trans_y、init40_trans_y、init46_trans_y（全部 butter trans_y）
- **双成功（14）** / **双失败（7）**——合计 21/30 选哪臂结局相同，准确率不足以评价本任务

入篮 t：救回 4（init34/37_trans、init01_trans、init12_trans）、破坏 3、双成功 11、双失败 12。

### 决策步执行前特征（在线可得）

- **e** = (锁定视觉目标 xy − 夹爪中点 xy)·u，闭合轴上的有符号误差。u=normalize(finger1−finger2)。
- **approach** = (fm(t0)−fm(t0−5))·u，近 5 步夹爪中点沿 u 的运动。注意：主域 e<0 下 **approach<0 才是沿 −u 接近偏置目标**、>0 为远离（此稿前一版符号写反，已校正；e=(locked−fm)·u，fm 沿 −u 运动使 e 趋 0=接近）。
- 诊断用真值 e_gt 与 e 几乎重合（差 <几 mm）：**e 的系统性负偏是真实几何**——决策时刻物体普遍位于夹爪中点的 −u 侧（靠 finger2 垫一侧），不是估计偏差。

分布（g）：6 例救回 e 全部 ≤−2.4mm；3 例破坏 e≤−28mm **且 approach 全部 <−10mm**（夹爪正沿 −u 快速滑向偏置目标——px 的 +u 推与此运动相抵，破坏居中）；7 例双失败 e≤−9.5 居多但 approach 不定。仅靠 e 无法分开救回与破坏（重叠区 −28~−41mm），**approach 守卫是关键**。

## 3. 规则与留出折验证（LOGO CV，15 组）

**规则族**：`px iff s·e > θe ∧ approach > θa`，s∈{±1}、θe∈[0,40mm]、θa∈{无守卫,−20..+20mm}——方向、阈值、守卫全部只在训练折选择（最大化训练折该终点成功数，tie→更少干预→更简单规则）。

**各折稳定选出的规则**（14/15 折相同）：`s=−1, θe≈2mm, θa≈−10mm`，即
> **e < −2mm 且 approach > −10mm → 选 px，否则 c0**
> （物体偏 −u 侧、且夹爪未在沿 −u 快速滑向目标时，给 +u 平移候选；milk init06 折选 θa=0、init01 折 θe=0，同族变体）

### 留出折合计（30 状态，每例由其未见过的训练折规则判定）

| 终点 | 规则 | c0 | 全局固定 | 按物体固定 |
|---|---|---|---|---|
| **抓取 g** | **21** | 17 | 20(px) | 21 |
| **入篮 t** | **18** | 14 | 12 | 16 |

- **救回（相对 c0）**：g 4 例 init34_trans、init37_trans、init09_trans_y、init12_trans；t 4 例 init34_trans、init37_trans、init01_trans、init12_trans
- **破坏（相对 c0）**：**0 例**（两终点）——approach 守卫挡掉了全部 3 例潜在 px 破坏
- **有效分歧**：上述 4+4 例，即规则选 px 且结局不同于 c0 的全部病例；规则未救到 init06_trans_y（该折守卫更严 θa=0）、init17_trans_y（approach=−12<−10）
- 规则 ≠ 全局固定的状态（g）：init34_trans、init37_trans_y、init40_trans_y、init46_trans_y、init09_trans_y、init12_trans、init06_trans_y、init17_trans_y（规则在这些状态放弃 px 或接手 px）
- 规则 ≠ 按物体固定的状态（g）：同上加 butter 全体（butter 按物体固定为 c0，规则在多例选 px）

## 4. 首轮问题回答

> **在没有结局信息时，一个简单规则能否超越固定动作基线？**

**能，但有条件。** 双条件规则 `e<−2mm ∧ approach>−10mm → px`：
- 抓取端 **21/30 = 按物体固定 21 > 全局固定 px 20 > c0 17**；与按物体固定打平（但后者按物体分治，规则是单一全局规则）
- 入篮端 **18 > 16（按物体）> 14（c0）> 12（全局）**，明确最优
- **增益来源**：4 例救回（butter init34/37_trans、milk init09_trans_y/init12_trans；入篮端换 init09_trans_y→init01_trans），全部满足「物体偏 −u 侧 + 未快速远离」
- **代价**：零——留出折上没有任何基线成功被规则破坏；单条件规则（无 approach 守卫）会付出 3 个 butter 成功回合

## 5. ry 旋转探索（探索性）

- **新增覆盖（c0 与 px 均失败的抓取病例，4 例）**：butter init34_trans_y、init43_trans_y；milk init00_trans、init17_trans。共同特征：e≤−9.5mm 深偏、approach 非负或温和（+1~+14，仅一例 −7.5）。
- **破坏的基线成功（10 例）**：butter 9（init30×2、init37_ty、init40_t、init43_t、init44×2、init46×2）+ milk init12_trans_y。e 分布从 −28 到 +10 混杂，approach 也混杂——**执行前未见一致可分差异**：4 例新覆盖与若干破坏例同处深负 e 区，无法用现有两特征区分。
- **闭合前形成 vs 闭合窗内旋转**：三时点 achieved 显示 dyaw 在闭合前一步已 −10~−20°（≈闭合当步值），闭合+2 仅再增 2–5°——ry 的实际作用主要是**闭合前已形成的姿态差**，而非闭合窗口内持续旋转；即 ry 类候选若要在线选择，依据应在决策步姿态，而非闭合期行为。
- 结论：ry 收益真实存在但方向/时机依赖强，当前特征不足以安全选择；不建议首轮纳入规则。

## 6. 限制

- 30 状态 / 15 组，小样本；LOGO 方差大（换一折可动 ±1–2 例）。
- 规则只在已跑分支上反事实成立；在线部署时 px 分支的 cand_step/前缀与 c0 相同（已审计），触发条件在线可算（锁定目标+夹爪中点+u 均需已锁定；无触发退化 c0）。
- achieved≠指令：px 名义 +5mm 实达 ~2–3mm，规则对应的实际位姿差更小；机制上是"扰动接近轨迹"而非纯位姿平移，解读需保守。
- e 的负偏符号依赖 u 定义（u=f1−f2）；换抓取物体/夹爪方向需重新标定方向。
- 入篮端全局固定成绩（12）低于 c0（14）：px 在 butter 入篮上净负，全局最优折间不稳定。

## 7. 文件清单

- `scripts/analyze_action_select.py` 全部离线分析（无新仿真）
- `results/action_select_summary.json` 汇总（审计/成绩/标签/CV/ry）
- `results/action_select_states.csv` 逐状态特征+配对标签+三时点 achieved
- `results/action_select_cv.csv` **留出折决策表**：每状态每终点一行（所选规则参数、pick、结局、双基线结局）
- 规则定义：见 §3 框内公式（`s=−1, θe≈2mm, θa≈−10mm`）
