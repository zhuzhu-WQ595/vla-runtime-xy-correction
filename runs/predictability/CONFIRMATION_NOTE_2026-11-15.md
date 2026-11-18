# 独立确认说明（开封前写成，2026-11-15）

写成时 `results/` 中没有任何 init30–49 产物（计数为 0）。

## 冻结文件 SHA-256

| 文件 | SHA-256 |
|---|---|
| runs/predictability/confirm_config.json（原配置，不修改） | 508cb37831c353ad473673d9b527fe5040162363cde931aba68aa398e6ffe55d |
| runs/predictability/confirm_addendum_2026-11-15.json | f435bd4ac5c26091b86fc29d88d962dc8c8687d5915a86ba272f891dc5c33821 |
| scripts/run_confirmation.py | 091dd5ce6243ce29ec5c41f867c52c43084be695476082a0ef3cf237e7c24f76 |
| scripts/analyze_confirmation.py | d6910755bad7fce1634e7286b1c1614bccc7a9ca34e4fa5afa11273ec175031e |
| scripts/run_correction_eval.py | fc1b30eba2d099f2c971de692b1f34090938f3a9ee02a54b9b62e0c1e0da7a03 |
| scripts/detect_triggers.py | c794c39c2e1fff6ac36cb2a3d9fcb6620f3121efe3ed10af956598c346609c9d |
| scripts/audit_predictability.py | d772f41d7db07d337372fe60079d1eec8152d653d91857c23ed4b98efca09a5d |
| scripts/analyze_predictability.py | 255c5cdd357dfc75621d02a2a946e07ac3a99ef7ce862cb972b54ba5bb4134dd |

## 冻结内容

- **主规则 C_fit**：M、b、θ_C = 2 mm、触发定义、eef 基点和容差矩形数值都与
  `confirm_config.json` 相同。容差矩形（|e·body_y| ≤ 20 mm 且
  |e·body_x| ≤ 10 mm）是**算法内部使用的容差**，不是经过验证的物理可抓取
  区域。
- **参数来源**：全开发集重拟合，开发集干预 24/48。开发集留出折中的 23/48
  是五折各自参数下的结果，确认集不使用。
- **对照**：始终 α0（2C）、始终 α1、C_nom（θ = 4 mm）、C_nom 同预算版
  （θ = 5.821 mm）、修正平局后的 B (2 mm, 0.7, −2 mm)、B 同预算版
  (2 mm, 0.3, −4 mm)。同预算版本的阈值按开发集 24/48 冻结，确认集上的
  实际干预次数允许不同。
- **次级规则 S_short**（事后提出，确认前冻结）：牵引当且仅当
  |e·body_y| > 10.196 mm。
  - 称为"物体短轴误差"，四个条件统一使用这一名称。日志里没有末端姿态，
    无法核对夹爪闭合方向，所以不称其为"夹爪闭合方向"。
  - 阈值只用开发集 48 个触发状态的误差分布校准：降序第 24 个值为
    10.323 mm，第 25 个为 10.069 mm，取中点。开发集干预恰为 24/48，
    分界处没有并列值；确认集上用严格大于判定。
  - 没有使用救回例与破坏例之间的误差间隔来定阈值。

## 开发集自检

开封前用 `analyze_confirmation.py dev` 检查，冻结参数下的开发集干预数与
登记值完全一致：

| 规则 | 开发集干预 |
|---|---|
| C_fit | 24 |
| C_nom | 28 |
| C_nom 同预算 | 24 |
| B（修正平局） | 14 |
| B 同预算 | 24 |
| S_short | 24 |

48 个触发状态的一致性检查全部通过。这些都是**样本内**结果（例如 C_fit
抓取 73/80、入篮 58/80），不能当作对确认集的预期。

## 执行与异常处理

- 执行：`python scripts/run_confirmation.py` 一次性完成 gate → 触发定位
  （冻结的 `detect_triggers.scan`）→ 每个触发状态跑 α0/α1 分支，然后运行
  `python scripts/analyze_confirmation.py confirm`。
- 确认期间不改任何参数。
- 若出现实现错误，在 `INDEPENDENT_CONFIRMATION.md` 中记录受影响的状态、
  原因和处理过程；不得因结局而重跑。
