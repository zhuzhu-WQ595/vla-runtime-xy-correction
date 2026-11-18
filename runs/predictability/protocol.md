# 干预可预测性实验协议（冻结于 2026-11-14）

## 范围

- 开发集：`init10–29 × {orig, trans, rot, trans_y}` = 80 条件状态，全部保留
  （含无触发回合），不按结局挑病例。
- 封存：`init30–49` 不动。
- 沿用冻结 2B（`results/freeze_2B.json`）、2C gate、已验证 `branch` 臂。

## 决策步定义（在线、只用执行前信息）

逐控制步回放内部 2B 状态机（`CorrState(gated=False)`，diag 语义）：
决策步 t* = 首个同时满足以下条件的控制步：

1. 策略发出闭合命令 `raw[-1] > 0`；
2. 内部机此时激活（即完整 2B 会覆盖 `action[:2]`）；
3. `|prop[:2] − raw[:2]| > TOL`，**TOL = 0.05**（raw 动作单位，冻结）。

实现细节（已核对）：日志索引 j = 第 j+1 步执行后状态；update 在第 j+1
步读入的是执行前观测 = 索引 j−1 的 `eef_pos/tgt_pos`；`last_cmd =
exec[j−1, −1]`（j=0 为 −1）。`raw_action` 直接取索引 j。

**核对结果**：6 例已知 t*=[47,42,48,55,85,78] 全部复现；回放
`prop_action` 与既有 diag npz 逐位一致（7/7，max|diff|=0.00e+00）；
负对照 `init12_orig` 无触发。80 状态共 **48/80 触发**
（orig 8、trans 17、rot 7、trans_y 16）。

## 配对分支

- α=0：t* 步 `exec = raw`，其余步冻结 2C（已验证与 gate 逐位一致 6/6）。
- α=1：仅 t* 步 `exec[:2] = prop[:2]`，夹爪与其余维度保持各自 raw，
  之后两侧均按冻结 2C 闭环。
- 无触发状态沿用 2C 结果计入分母。
- 记录：本次闭合尝试是否抓起（段内 tgt_z−rest ≥ 0.04）、整回合稳定
  抓起、入篮、后续重试数；配对标签 rescue / break / both_success /
  both_fail（按稳定抓取）。

## 特征（全部只用 t* 执行前信息）

- `e` = 两指中点 − 盒心，世界 xy。两指中点 XY ≡ `eef_xy`
  （probe 实测 `fmid − eef = [0,0,+4mm]` 常值，XY 精确相等）。
- `e_L, e_S` = e 在盒体长轴(19.9 mm 半长)/短轴(9 mm 半长)上的投影；
  盒系由 `tgt_quat` 的 yaw 定。指轴 = 世界 Y（夹爪沿 Y 闭合）。
- `trend4` = `|e_S|(t*−1) − |e_S|(t*−5)`（近 4 步短轴误差变化）。
- `d = prop[:2] − raw[:2]`，`|d|` 与 `cos(d, −e_xy)`。
- 接触输入：**不使用**（gate npz 无指垫−目标专属接触计数；B/C 信息相同）。

## 决策规则（公式冻结）

- **A**：`tug iff |e_S| > θ_A`。
- **B**：`tug iff |e_S| > θ_B1 AND cos(d, −e_xy) > θ_B2 AND trend4 > θ_B3`。
- **C**：`tug iff resid(e + M·prop + b) < resid(e + M·raw + b) − θ_C`；
  `e` 为盒系 (e_L, e_S)；`resid` = 盒系误差到可行抓取区域
  `R = {|e_L| ≤ 20 mm, |e_S| ≤ 10 mm}`（几何冻结值，不拟合）的距离；
  `M, b` 为 `exec[:2] → Δe` 的最小二乘映射，**仅用训练折**的 gate
  轨迹逐步拟合；C 的预测误差用留出折下一步实测位移评价（实测未来量
  只用于评价）。

## 阈值选择与折划分

- 折：按 init 分组 5 折（{10–13},{14–17},{18–21},{22–25},{26–29}），
  同 init 四条件始终同折；逐折留出评估，其余 16 init 训练。
- 网格：`θ_A, θ_B1 ∈ {2,…,20} mm`；`θ_B2 ∈ {0,…,0.9}`；
  `θ_B3 ∈ {−10,…,5} mm`；`θ_C ∈ {0,…,5} mm`。
- 目标：`max(#rescue − #break)`，平局取干预次数更少者。

## 交付

`INTERVENTION_PREDICTABILITY.md`、`results/branch80.csv`（80 行逐状态
分支表）、本目录 `protocol.json`（SHA + 公式 + 折清单）。
