# ONLINE_VS_INITIAL_TARGET — 阶段三 C：逐帧更新 vs 初始锁定的增量价值

日期：2026-11-17 ｜ 预注册规则：`runs/predictability/vislock_spec_2026-11-17.json`（早于任何 vislock 结局）

> **结论先行：在三 B 已分析状态上，逐帧更新目标相对"首帧估计整回合锁定"的增量很小且不显著——抓取端 Δ(visgate−vislock) butter +0.05、milk +0.033，区间均跨 0；翻转病例双向对称。视觉 2C 的收益主要来自"拿到一次带纠偏的估计"，而非持续跟踪。**

## 0. 对照臂定义（预注册）

**vislock = visual_initial_lock**：与 visgate 完全相同的相机、模板、仿射、z_prior、阈值、工作区、LAST_HOLD 及 CorrState 触发/闭合屏蔽/重开/控制参数。**唯一有意改变的输入逻辑**：回合内首个非空估计一旦获得即锁定该世界坐标，整回合固定喂给 CorrState；锁定前与 visgate 逐行同路径（含 miss→EEF 回退）。逐帧估计仍在后台记录（`est_live`），仅作离线诊断，不影响任何输入。

## 1. 数据与配对

- 状态：复用三 B 冻结集——butter init30–49 新位置 60 状态、milk init20–49 共 90 状态；conds orig/trans/trans_y。
- 配对：同一 (init,cond) 的 visgate vs vislock；主分析 trans+trans_y；orig 单列。
- 区间：init 簇 bootstrap 95%（2000 次，seed 0）。
- **标签：已分析状态上的机制对照（ablation），非新状态验证。**

## 2. 锁定达成

| | 首帧锁定 t=0 | 延迟锁定 | 全程未锁定 |
|---|---|---|---|
| butter | 59 | 0 | 1（`init44_trans_y`，整回合估计器无输出 → 全程 EEF 回退） |
| milk | 88 | 2（`init21_trans` step129、`init26_trans` step60） | 0 |

初始漏检病例全部保留并标记（`lock_step` / `lock_miss` 见逐状态表）。

## 3. 主结果（trans+trans_y）

| | N | visgate / vislock 抓取 | 逐帧赢 / 锁定赢 | Δx−l [CI] | visgate / vislock 入篮 | Δx−l [CI] |
|---|---|---|---|---|---|---|
| butter | 40 | 23 / 21 | 4 / 2 | +0.050 [−0.075, +0.175] | 17 / 15 | +0.050 [−0.050, +0.175] |
| milk | 60 | 35 / 33 | 4 / 2 | +0.033 [−0.050, +0.117] | 22 / 23 | −0.017 [−0.067, +0.033] |

分方向（抓取）：

| | trans x/l | Δx−l [CI] | trans_y x/l | Δx−l [CI] |
|---|---|---|---|---|
| butter | 13 / 10 | +0.150 [0.000, +0.300] | 10 / 11 | −0.050 [−0.250, +0.100] |
| milk | 20 / 20 | 0.000 [0, 0] | 15 / 13 | +0.067 [−0.100, +0.233] |

orig 单列：butter 抓取 14/12（逐帧赢 3、锁定赢 1，CI 跨 0）；milk 29/29 完全相同。milk orig 两臂无任何翻转——初始锁定已足够。

全条件（含 orig）供参考：butter 抓取 +0.067 [0.000, +0.134]；milk 抓取 +0.022 [−0.033, +0.078]。

## 4. 翻转病例归因（勘误版，2026-11-17）

**勘误方法**（`scripts/corrige_3c.py`，纯离线）：同一轨迹同一时刻比较 |lock_est−真值| 与 |live_est−真值|；标注三个事件的先后——**t_div**（两臂 exec_action 首个非零差步）、**t_con**（指尖中点与目标 3D 距离 <25mm，夹爪-物体接触代理；`ncon_tgt` 含桌面接触不可用）、**t_mv**（物体水平位移 >5mm）。

### 4.0 唯一翻转状态数（双向合并去重，勘误重算）

| | orig g/t | trans g/t | trans_y g/t | trans+trans_y 唯一 |
|---|---|---|---|---|
| butter | 4 / 2（共5唯一） | 3 / 2（3唯一） | 3 / 2（4唯一） | **7** |
| milk | 0 / 0 | 0 / 1（1唯一） | 6 / 2（6唯一） | **7** |

### 4.1 重分类（取代初版粗分类）

关键证据列：`div/con/mv` = 事件步；`lock@div / live@div` = 分歧时刻两估计各自误差；`edge` = live−lock 误差差（负=逐帧更准）。

| 病例 | 翻转 | 类别 | div / con / mv | lock@ / live@ (mm) | 判读 |
|---|---|---|---|---|---|
| **butter** | | | | | |
| init33_trans | res/res | no_quality_edge | 41 / 42 / 44 | 13.1 / 13.6 | 分歧恰在接触前，两估计误差相同，结局差非估计质量 |
| init40_trans | res/— | **tracked_confirmed** | 53 / 156 / 81 | 19.0 / 13.7 | 物体位移 42mm，分歧时逐帧准 5mm |
| init45_trans_y | res/— | no_quality_edge | 45 / — / 219 | 9.4 / 4.1 | edge=−3.0 恰界内，位移远晚于分歧 |
| init37_trans | res/res | no_quality_edge | 61 / — / 64 | 32.3 / 31.9 | 两估计都差 32mm，误差相等 |
| init39/42/48_orig | res | no_quality_edge / undet | 50/41/48 | 见 corrige_3c.json | orig 条件翻转，分歧早（41–50）且无接触 |
| init33_orig | brk/brk | undetermined | 41 / — / 53 | 28.1 / 38.1 | lock 反而准 10mm，破坏原因不明 |
| init33/43/48_trans_y | res/brk | undetermined | 44/48/44 | lock 更准 14–22mm | 物体被撞飞 >450mm，两臂估计均失效 |
| **milk** | | | | | |
| init26_trans_y | res/— | **tracked_confirmed** | 54 / — / 53 | 10.2 / 1.7 | 物体扰动后位移 44.6mm（无接触），逐帧跟上；lock 停在原值 |
| init28/43/44_trans_y | res/brk | no_quality_edge | 55–107 | edge −2.8~0.9 | 分歧时两估计误差几乎相同 |
| init30/47_trans_y + init48_trans | res/brk | **miss_driven** | 75/50/90 | live=NaN@div | 分歧时刻 live 估计缺失（漏检/held 切换驱动） |

### 4.2 勘误后机制结论（修正 §5 第 2、3 条）

1. **可确认 tracked 的只有 2 例**（butter init40_trans、milk init26_trans_y）——均为物体在接触前发生位移且逐帧估计跟上。原判的 milk init28/30_trans_y 经同步比较后分别降为 no_quality_edge 与 miss_driven。
2. **"跟错被撞飞物体"证据不足**：物体位移 >400mm 的病例（milk init44/47/48、butter init33/43/48_trans_y）中 lock 在分歧时刻反而更准或误差相当，不能归因于逐帧跟错；更可能是接触后轨迹差异的下游噪声。**保留为未定。**
3. **多数翻转（9/19）在分歧时刻两臂估计误差几乎相同（no_quality_edge）**——结局差来自同质量输入下的轨迹分叉，非估计精度差异。
4. 净效应结论不变（区间跨 0），但机制图景修正为：**逐帧更新的可证实收益仅限于"物体接触前位移跟踪"（2/19 翻转），其余翻转无证据表明与估计更新有关。**

## 5. 机制判读（以 §4 勘误版为准）

1. **初始一次估计已捕获绝大部分收益**：vislock 在两物体抓取上仅比逐帧低 2 个状态（butter 21 vs 23、milk 33 vs 35），Δ 区间跨 0。
2. **可证实的逐帧收益 = 物体接触前位移跟踪，仅 2/19 翻转**（butter init40_trans、milk init26_trans_y）。
3. **多数翻转与估计质量无关**（9/19 在分歧时刻两臂估计误差相当），miss_driven 3 例，undetermined 5 例。
4. 净效应 ≈ 0；butter trans 上 +0.15 方向性优势样本小，只能作线索。

## 6. 局限

- 机制对照在已分析状态上进行，非独立验证。
- 勘误重分类阈值（edge 3mm、MOVE 5mm、接触代理 25mm）与初版预写阈值并存说明：`corrige_3c.py` 内注明；证据不足一律 undetermined。
- 接触事件用指尖中点-目标距离代理（`ncon_tgt` 含桌面接触不可用）；未达 25mm 记 con=None。
- 锁定规则只保证"首个非空估计"，butter `init44_trans_y` 整回合无估计 → 与 visgate miss 回退路径等价，该例两臂结局天然接近。

## 7. 文件

- 逐状态表：`results/online_vs_lock_{butter,milk}.csv`（含 lock_step、lock_miss、翻转标签、初版归因）
- 勘误：`results/corrige_3c.json`、`scripts/corrige_3c.py`（唯一翻转数 + 同步误差/时序重分类，本节数值以此为准）
- 汇总：`results/online_vs_lock_summary.json`
- 分析脚本：`scripts/analyze_online_vs_lock.py`（`--partial` 可查进度）
- 预注册：`runs/predictability/vislock_spec_2026-11-17.json`
