# multiflow 分支 —— MF-18 PBSTF 倒奶 demo（修改版 Genesis + 说明文档）

本分支根目录即 multiflow 项目实际使用的**修改版 Genesis 引擎**（v1.3.3 基线，taichi 已替换为
quadrants==1.3.0），相对上游的主要改动：

- **PBSTF 求解器独立路径**（`genesis/engine/solvers/pbstf_solver.py` 等）：cubic 核 support=3ps、
  build 质量标定含自项 m·W(0)、表面密度目标 0.7、表面/内部距离约束 40/180、分区 XSPH 粘性、
  面积表面张力约束；
- **浓度扩散**：PBD 液体粒子浓度场 c + XSPH 风格平滑（`diffusion_coeff`，0=关闭），双液体
  （咖啡/牛奶）混合变色；
- **PBSTF recon 渲染分支**（`genesis/vis/rasterizer_context.py`）：多实体合并单次 splashsurf
  表面重建 + 浓度顶点色；
- mesh collider（杯/缸）+ frozen 球墙边界、摩擦 0.1。

## 尺度与参数（必看）

PBSTF/PBF 的 compliance 参数随长度尺度变化（density ∝ s⁻²、面积 ST ∝ s²、距离/粘性不变），
跨粒子尺寸照搬是错的。本分支 demo 用真实尺度：ps=0.002、support=3ps=0.006、rho0=1000、
h=1/480（dt 1/240×2 substeps）、20 迭代。完整规则、参数表与对照见 **[doc/SCALE.md](doc/SCALE.md)**。

## 目录

- `genesis/`、`examples/`、`tests/` 等 —— 引擎本体（含上述改动）
- `doc/` —— 全部说明文档：
  - `SCALE.md` 尺度问题与参数缩放规则；`PBSTF_UNIFORM_SCALE_DERIVATION.md`、
    `PBSTF_ITERATION_SCALING_DERIVATION.md` 两份推导；
  - `MF18_PBSTF_ALIGNMENT_PLAN.md` 对齐方案；`MF18_PBSTF_PERFORMANCE_AUDIT.md` 性能审计；
  - `MF18_FULL1_JITTER_DIAGNOSIS.md` 抖动诊断；`MF18_FULL3_RIM_MIX_REVIEW.md` 杯口/混合审阅；
  - `P2_SHELL_DESIGN.md` 球墙壳层设计
- `scripts/` —— 生产/测试脚本（主脚本 `mf18_pbstf_pour.py`；注：脚本内 sys.path 断言
  "multiflow" 路径，供本地 multiflow/ 布局使用，从本分支直接运行需自行调整引擎路径）
- `assets/` —— 杯/缸 OBJ + settled 初态 npz（sidecar json 校验）

视频与仿真产物（本地 `videos/`，约 4.5G）不入库。

## 运行

conda env `ipbf`（py3.12，torch cu128，`pip install -e <本分支根>`）。
