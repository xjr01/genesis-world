# SCALE.md —— 尺度问题与参数尺度说明（multiflow / MF-18）

> 2026-09-23。本文说明：① 为什么 PBSTF/PBF 的 compliance 参数**不能跨粒子尺度直接照搬**；
> ② 本分支（MF-18 倒奶 demo）实际使用的尺度与参数；③ 换尺度时各参数该怎么乘。
> 严格推导见 `PBSTF_UNIFORM_SCALE_DERIVATION.md`（齐次相似）与
> `PBSTF_ITERATION_SCALING_DERIVATION.md`（迭代数/dt 联合）。

## 1. 尺度问题是什么

PBSTF 的约束投影按如下结构求解（密度、面积、距离、壁黏附同构）：

```
lambda   = -C / (c/m0 + Σ_j |grad_j C|² / m_j)
Δx_j     = lambda * grad_j C / m_j
```

代码参数 `c`（compliance）**不是无量纲数**：若约束满足 `C(s·x) = s^p · C(x)`
（密度 p=0、面积 p=2、距离 p=1），粒子质量随体积标定 `m ∝ s³`，则梯度分母项缩放为
`s^(2p-5)`，要保证每一步位移修正严格按几何缩放（Δx_new = s·Δx），必须：

```
c_new = s^(2p-2) · c_old
```

**踩过的坑（实证）**：MF-18 早期 PBF 阶段曾把 `st_comp=0.7` 从 ps=8mm 场景原样搬到
ps=2mm 场景，表面张力实际弱了约 5 倍——面积约束 p=2，compliance 应按长度平方缩放。
同事 PBSTF 的材质文档也明确写着"场景缩放后必须重新标定"：其参考场景细化 15 倍时

| 参数 | 处理 |
|---|---|
| 面积表面张力 compliance | ÷ 15² = ÷225（如 teapot 3.0 → 0.01333） |
| 密度 compliance | × 225（150 → 33750） |
| 表面/内部距离 compliance | 不变（40 / 180） |
| 壁黏附 compliance | 不变（20） |
| 粘性系数 | 数值不变 |

## 2. 缩放规则速查表（整场景等比例缩放，重力固定）

长度缩放比 `s = ps_new/ps_old`，时间缩放比 `τ = √s`（保持重力相似时）：

| 量 / 参数 | 乘以 |
|---|---:|
| 所有长度：ps、support、hash cell、几何、接触偏移 | s |
| 单粒质量（标定自动完成） | s³ |
| 子步 dt、动作关键帧时刻、总时长 | τ = √s |
| 重力及其他加速度 | s / τ²（重力固定则不变） |
| **density_compliance**（p=0） | **s⁻²** |
| **surface_tension_compliance**（面积，p=2） | **s²** |
| surface / interior_distance_compliance（p=1） | 1（不变） |
| collider_adhesion_compliance（法向距离） | 1（不变） |
| XSPH 表面/内部粘性（每步滤波系数） | 1（不变） |
| 每子步迭代数、topology 重建间隔 | 1（不变） |
| rho0、表面密度目标系数 0.7 | 1（不变） |
| 浓度扩散系数（同 XSPH 结构的每步平滑系数） | 1（不变） |

注意：上表保证的是**同一实现、同一迭代数**下的严格相似。换实现（不同核、不同迭代数、
不同 dt）时 compliance 数值不能直接比较——例如本分支 `surface_tension_compliance=0.0048`
与同事 teapot 的 `0.01333` 差异主要来自迭代数/dt/密度公式不同，不能按"每秒投影次数"
简单换算（该旧结论已勘误）。

## 3. 本分支实际尺度与参数（MF-18 生产配方）

### 3.1 尺度

| 量 | 值 | 说明 |
|---|---:|---|
| 粒子直径 ps | **0.002 m（2mm）** | 真实世界尺度 |
| support（核半径） | **3ps = 0.006 m** | cubic 核，密度/梯度同一核 |
| 静息质量标定 | 7.985668e-6 kg | build 时按静息晶格 ΣW 标定，含自项 m·W(0) |
| rho0 | 1000 kg/m³ | 表面目标系数 0.7 |
| 场景 | 杯内径 7cm、高 10cm | domain [-0.10,-0.12,0]→[0.34,0.12,0.30] m |
| 重力 | 9.81 m/s² | 真实尺度 |
| dt / substeps | 1/240 s × **2** → h = 1/480 s | "B 档"生产档 |
| 约束迭代 | **20 次/子步**（=40 次/帧、19200 次/s） | topology 每 2 轮重建 |

### 3.2 配方（`scripts/mf18_pbstf_pour.py` RECIPE）

| 参数 | 值 | 备注 |
|---|---:|---|
| density_compliance | **375000** | 零重力方块扫描选定（rho50≈1.000）；同事 teapot 33750 对应其 6.67mm 粒子 |
| surface_tension_compliance | **0.0048** | 面积约束；方块扫描软档，趋球且最静止（KE 1.6e-7） |
| surface_distance_compliance | 40 | 防粒子间距小于一粒径的单边斥力 |
| interior_distance_compliance | 180 | 内部距离 |
| surface / interior_viscosity | **0.2 / 0.05** | 同事 teapot 值，40s 全场景验证有效（RECIPE 默认 0/0，由 `--surface-visc/--interior-visc` 传入） |
| diffusion_coeff | 0.005 | 浓度 XSPH 平滑（`--diffusion` 可调，0=关闭） |
| collider_adhesion_compliance | 1e12 | 实质关闭（不粘墙规则） |
| collider_friction | 0.1 | 球墙/桌面摩擦 |
| sampler | regular | 规则晶格采样 |

### 3.3 与同事 teapot 场景对照（为什么数值长得不一样）

| 项目 | 同事 PBSTF teapot | 本分支 MF-18 |
|---|---:|---:|
| 粒子直径 | 6.67 mm（ps=2/300） | **2 mm** |
| support | 3ps = 20 mm | 3ps = **6 mm** |
| 时间步 | 0.01 s × 1 substep | 1/240 s × 2 substeps |
| 约束迭代 | 5 次/step（500 次/s） | 20 次/substep（19200 次/s） |
| 面积 ST compliance | 0.01333 | **0.0048** |
| 密度 compliance | 33750 | **375000** |
| 距离 compliance | 40 / 180 | 40 / 180（相同） |
| 粘性 | 表面 0.2 / 内部 0.05 | 表面 0.2 / 内部 0.05（相同数值，不同执行频率与邻域） |
| 边界 | mesh SDF + 黏附/摩擦 | mesh collider + frozen 球墙 + 摩擦 0.1 |

要点：**数值相同不代表作用相同**（粘性 .2/.05 两边执行频率差 ~38 倍，实测恰好都合适属于
各自标定结果）；**数值不同也不代表强度不同**（ST compliance 受迭代数/dt/密度公式联合影响）。

## 4. 换尺度操作清单

1. 改 ps 与全部几何（含 support=3ps、hash cell、杯壁厚度），rho0 不动。
2. 质量不用手算——build 标定自动按 s³ 调整。
3. 按 §2 表乘 compliance：density ×s⁻²，面积 ST ×s²，距离/黏附/粘性不动。
4. dt 与动作时间线按 τ=√s 缩放（重力固定时）；迭代数不动。
5. **标定后必须重跑零重力方块 + 静置验证**（趋球、rho50≈1、KE 地板），并重新生成
   settled 初态（旧 settled 跨配方不可用）。
