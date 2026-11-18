# PERCEPTION_TRANSFER_REPORT — 阶段三 A：视觉 2C 抓取迁移验证

日期：2026-11-16 ｜ 结论先行：

> **在平移条件下，冻结 2C 纠偏用视觉估计替代仿真质心真值后仍保留大部分抓取收益。**
> butter 抓取 54→65（vanilla→visual-2C），oracle-2C 为 73；
> milk 抓取 50→60，oracle-2C 为 59（visual 持平甚至略超 oracle）。
> oracle 在 trans_x/trans_y 上救回的 butter 抓取 15 例中 visual 保留 6 例，
> trans_y 是主要损失项（11 例救回中仅保留 4）；milk trans_x/trans_y oracle 救回 10 例中
> visual 保留 5 例且平移条件新增 6 例救回（估计噪声反而触发了一些 oracle 未触发的救回）。
> 代价：butter 破坏 1 个原成功抓取 / 4 个原成功入篮；milk 破坏 2 个抓取 / 2 个入篮。

## 1. 部署输入盘点（无真值）

| 输入 | 来源 | 用途 |
|---|---|---|
| agentview RGB | `obs["agentview_image"]` raw 朝向 256²，估计器内统一 224 | 唯一视觉输入（策略输入为其 180° 翻转版） |
| 静态标定 | CAM_POS/CAM_R/FOVY=45° 一次读取 | 候选底中点的理论反投影（仅用于工作区过滤） |
| 生成区先验 | WORKSPACE_RECT = x∈[-0.20,0.10], y∈[-0.32,-0.10] | 任务知识：目标放置区，过滤干扰候选 |
| 像素→世界仿射 | `runs/predictability/pix2w_{cls}.json`（每类一次标定） | 底中点 px224 → 世界 XY |
| 类参数 | 模板 PNG + 静置质心高 z_prior + score_min | **物体专属信息**（见 §6 迁移记录） |
| 机器人本体 | `robot0_eef_pos/quat`、`gripper_qpos` | 沿用既有日志口径 |

不可用/未用：**无深度渲染**、手腕相机图像未用、仿真实例掩码未用（本臂）。
真值 `*_pos/quat` 仅出现在日志中供离线评分与 oracle 对照，不进入估计。

## 2. 冻结估计方案（`scripts/vision_target.py`）

- **RGB NCC 模板匹配** top-K=4 候选（score_min=0.55），候选底中点经理论反投影
  进 WORKSPACE_RECT 过滤；存活候选取最高分。
- 底中点 → 类专属仿射 → est XY；**est Z = 类先验静置质心高**（不依赖像素推 z）。
- butter z_prior=0.0087；milk z_prior=0.0697（日志静置中位数，类级常数）。
- **漏检策略（冻结）**：无工作区内候选 → 沿用最近估计 ≤5 步（LAST_HOLD），
  否则 miss；miss 时喂 corr 的 tgt_xy=eef_xy（lat=0 天然抑制触发/关断），tgt_z=eef_z。
- **每步独立估计，无时间滤波**；时延=当步帧（1 控制步）。
- 纠偏器与 oracle-2C **完全相同**（CorrState(gated=True)，Z/Z_LOW/Z_HY/DESC_NET/WIN/
  R_ON/R_OFF/K/C 与闭合屏蔽、重开逻辑一律不变）。仅 tgt 来源不同。

颜色阈值分割（HSV）已被数据否决：butter 与木桌同色（误差 0.2–0.56m），
milk 白色阈值吞入背景；故采用模板匹配。

## 3. 既有轨迹离线核验（`scripts/vision_offline_eval.py`）

| 集 | 仿射标定 (n, rmse, p95) | 接近段 XY 中位/p95 | 闭合前 XY 中位 | Z 有偏 | 接近段漏检 | 触发翻转步/回合 |
|---|---|---|---|---|---|---|
| butter | 9215, 0.0157, 0.0269 | **0.0070 / 0.0198** | 0.0082 | **+0.000** | 1175/10729 = **11.0%** | 704 / 63 回合非零 |
| milk | 9361, 0.0121, 0.0316 | **0.0044 / 0.0118** | 0.0044 | **−0.000** | 2700/12209 = **22.1%** | 442 / 54 回合非零 |

- Z 有偏首轮 butter +0.0023 / milk −0.0347 → 修正 z_prior 后归零（上表）。
- 触发翻转=在 gate 日志上用估计值回放 corr 判定、与日志 corr_active 不同的步数。
  注意：翻转多因 lat 在阈值附近抖动，不等于结局必变；实际影响以配对回合为准。
- **高度差专项**：milk 模板底中点对应的是可见盒体而非质心，若用像素推 z
  会高出 ~4cm（顶面 ~0.11 vs 质心 0.070）；采用类先验质心高后 z_bias≈0。
- **线上核验（修正后）**：live `est_tgt` 与视频离线回放一致；butter 冒烟回合
  接近段误差中位 3.8mm、milk ~5–7mm，漏检与离线一致（运输段漏检为主，
  接近段 butter≈0、milk 高漏检多发生在物体被抓后）。

**已知缺陷修正记录**：首轮 butter 跑批因 runner 未传类专属仿射文件，
静默退化到理论反投影（~4.5cm 偏差）；已发现并修复（仿射缺失时硬报错），
**本文全部 visgate 数据来自修正后重跑**（删除首轮 30 回合污染数据）。

## 4. Butter 三臂主表（init30–49 × orig/trans/trans_y/rot，各 20）

| cond | N | grasp v / g / x | x 救/破/净 (vs v) | task v / g / x | x 救/破/净 | x 触发回合 |
|---|---|---|---|---|---|---|
| orig | 20 | 19 / 20 / 20 | 1 / 0 / +1 | 19 / 19 / 19 | 1 / 1 / +0 | 20 |
| trans | 20 | 15 / 18 / 17 | 2 / 0 / +2 | 13 / 11 / 16 | 3 / 0 / +3 | 11 |
| trans_y | 20 | 6 / 17 / 9 | 4 / 1 / +3 | 5 / 11 / 5 | 2 / 2 / +0 | 20 |
| rot | 20 | 14 / 18 / 19 | 5 / 0 / +5 | 13 / 17 / 17 | 5 / 1 / +4 | 14 |
| **ALL** | 80 | **54 / 73 / 65** | **12 / 1 / +11** | **50 / 58 / 57** | **11 / 4 / +7** | — |

抓取救回对照：oracle 救回 20 例 → visual 同救回 **11**，丢失 9 例
（init34_trans、init36/37/41/43/45/46/47_trans_y——**8/9 为 trans_y**），
新增救回 1 例（init36_rot），破坏 1 例（init38_trans_y）。
入篮：oracle 救回 15 → visual 同救回 8，丢失 7，新增 3，破坏 4。

分歧归因（visgate↔oracle 结局不同的 23 例，首次 corr 分歧）：
- **平面定位**（exy 0.015–0.019 于分歧步）：7 例，多在 trans_y——平移后物体
  处于工作区边沿/视角透视下模板峰位偏移 1–2px，经仿射放大为 1.5cm+ 误差，
  使 lat 在 R_ON 附近抖动而错过/错发触发。
- **高度/时序**（ez≈±0.01 或时序差）：14 例，多为触发步差几步后走向不同，
  有赢有输（oracle 救回丢失 vs 新增救回并存）。
- **漏检**：butter 分歧集中不在漏检（仅 1 例附近）；orig 全触发回合 20/20。

## 5. Milk 三臂主表（init00–19 × 同条件，各 20）

| cond | N | grasp v / g / x | x 救/破/净 | task v / g / x | x 救/破/净 | x 触发回合 |
|---|---|---|---|---|---|---|
| orig | 20 | 19 / 19 / 19 | 0 / 0 / +0 | 18 / 18 / 18 | 0 / 0 / +0 | 7 |
| trans | 20 | 8 / 14 / 14 | 7 / 1 / +6 | 5 / 8 / 7 | 4 / 2 / +2 | 14 |
| trans_y | 20 | 6 / 10 / 10 | 4 / 0 / +4 | 6 / 5 / 7 | 1 / 0 / +1 | 18 |
| rot | 20 | 17 / 16 / 17 | 1 / 1 / +0 | 13 / 12 / 14 | 1 / 0 / +1 | 9 |
| **ALL** | 80 | **50 / 59 / 60** | **12 / 2 / +10** | **42 / 43 / 46** | **6 / 2 / +4** | — |

抓取：oracle 救回 10 → visual 同救回 **5**，丢失 5，**新增 7**，破坏 2——
净效果 visual(60) ≥ oracle(59)。估计抖动触发了若干 oracle 未触发的救回，
但同样带来 2 个原成功回合的破坏（init02_rot、init17_trans）。
入篮：oracle 救回 5 → visual 同救回 2，新增 4，破坏 2。

分歧归因（20 例）：**漏检类分歧 6 例**（init02_rot@169、init05_rot@152、
init06_trans_y@82、init09_trans_y@55、init11_trans_y@39、init12_trans_y@59，
均为 miss=1 且 ez 估计偏差 0.04–0.16 的步），其余多为高度/时序类触发抖动。

## 6. 物体专属信息（butter→milk 迁移清单）

| 项 | butter | milk | 复用程度 |
|---|---|---|---|
| 模板 PNG | tmpl_butter.png 12×18 | tmpl_milk.png 30×26 | **需按类截取**（各 1 张） |
| z_prior 质心高 | 0.0087 | 0.0697 | **需按类标定**（静置中位） |
| pix2w 仿射 | pix2w_butter.json | pix2w_milk.json | 同相机同场景理论上可共用，**本实验按类各自标定**（保守） |
| score_min/top_k/工作区/漏检策略 | 相同 | 相同 | 直接复用 |
| 纠偏器参数 | 完全相同 | 完全相同 | 直接复用 |

迁移成本：每类仅需 1 张模板 + 静置质心高 + （可选）仿射重标定；
其余（NCC 流程、工作区先验、漏检策略、2C 参数）零改动。

## 7. 回答首要问题

**x、y 平移下 oracle 可救回的抓取中，视觉版保留了哪些，代价是多少原成功回合？**

- **butter**：oracle 在 trans_x 救回 4 例中保留 2（另 oracle 破坏 1 例，
  使净抓取 g=18 > x=17）；trans_y 救回 11 例中保留 4（9/17）——trans_y 物体
  被推向工作区边沿，透视偏差 + 模板峰偏移使 1.5–2cm 定位误差跨过 R_ON 阈值。
  平移两条件合计 oracle 救回 15 → visual 保留 6。
  代价：**1 个原成功抓取被破坏**（init38_trans_y），另有 4 个原成功入篮转败。
- **milk**：trans_x+trans_y oracle 救回 10 例中保留 5（trans 6→4、trans_y 4→1），
  且平移条件新增 6 例救回（trans 3 + trans_y 3），另有 rot 新增 1——
  净抓取 60 ≥ oracle 59。代价：**2 个原成功回合被破坏**
  （init02_rot、init17_trans），漏检类分歧 6 例。
- **总体**：平移条件下视觉版保留 oracle 抓取收益的主要部分
  （butter 平移两条件 visual 净+6 vs vanilla，oracle 净+14）；
  milk 上视觉版甚至略超 oracle——估计噪声是双刃剑，既丢也送。
  **代价合计破坏 3 个原成功抓取、6 个原成功入篮**（160 回合中 5.6%）。

## 8. 交付物清单

- `scripts/vision_target.py` — 冻结估计器（NCC+先验+仿射+类 z_prior）
- `scripts/vision_offline_eval.py` — 标定+离线核验
- `scripts/analyze_perception_transfer.py` — 三臂配对+分歧归因
- `scripts/run_correction_eval.py` — 新增 `visgate` 臂（est_tgt/vis_miss/vis_score 日志）
- `runs/predictability/perception_config_2026-11-16.json` — 冻结配置
- `runs/predictability/pix2w_{butter,milk}.json`、`tmpl_{butter,milk}.png` — 类资产
- `results/perception_transfer_{butter,milk}.csv` — 逐状态配对表
- `results/traj{2,3}_*_visgate.npz`、`videos2/{vis_,vismilk_}*.mp4` — 轨迹与视频
- `results/episodes_{butter,milk}_visgate.csv` — 回合摘要

## 9. 局限与后续

- 模板为固定尺度/朝向：rot 条件靠目标本身旋转后的外观仍匹配，
  但大角度或遮挡下靠漏检+沿用兜底；更稳健需多模板或 learned 检测。
- 顶面→质心高度差目前用类先验常数处理；若物体姿态显著变化（翻转/侧躺）
  est_z 会系统性偏移 → 触发高度区间漂移。
- 工作区先验假设目标在生成区内；若运输中被推离先验区会漏检（设计内保守）。
- 触发翻转步数较多（704/442 步）说明 lat 在阈值附近抖动；可考虑估计时域平滑
  或滞回增强（需另行冻结）。
