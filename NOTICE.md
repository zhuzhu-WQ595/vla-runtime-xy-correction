# 第三方出处与许可说明

本项目代码为原创研究代码（与论文草稿配套），当前未对外声明统一许可证；
私有仓库阶段不授权复用，转公开前将补 LICENSE。上游组件按其各自许可证：

- **OpenVLA**（`openvla/openvla-7b-finetuned-libero-object`，MIT）
  https://github.com/openvla/openvla — 策略推理与图像/反归一化管线复用其
  `experiments/robot/libero` 工具函数（`get_libero_image` 等）。
- **LIBERO**（BSD-3-Clause / CC-BY-4.0 数据集）
  https://github.com/Lifelong-Robot-Learning/LIBERO — `libero_object` 任务与
  bddl 环境（`control_freq=20` 默认）。
- **robosuite / MuJoCo** — LIBERO 依赖的仿真栈。
- 引用论文见 `docs/PAPER_DRAFT_v2_1.md` 参考文献节（ActionGround、ReGuide、
  FIT-MAC、RTPC、CorrectVLA、PhysVLA 等 arXiv 预印本）。

`runs/predictability/tmpl_*.png` 与 `pix2w_*.json` 为本项目自行标定的类资产。
