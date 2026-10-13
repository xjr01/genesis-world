# examples/sand_water_coupling 沙水耦合示例运行指南

## 概述

本目录是沙水耦合（DEM 沙 + FLIP 水 + 吸收耦合）系列示例的入库镜像，来源于仓库外 `experiments/` 的 phase5–phase9 系列脚本：从滴水吸水（phase5）到 Franka 机械臂夹持猫砂铲对深床湿沙完成「沉降 → 滴水 → 下压 → 斜插 → 转起 → 抬升 → 抖铲」的完整铲掘（phase9 深床）。主流程现为**粗沙 A4 吸收档**（2026-10-11：grid_res 64、沙粒半径 3.125 mm、max_ratio 0.5 + VISCOSITY 1.0 + droplet r 0.020；表面张力默认 ×10 粘力档 ST 0.2（2026-10-13 用户拍板，见「参数说明」ST 挡位；该档必配 safety ≤ 0.5，默认组合自动满足），DEM 子步 safety 默认 0.5，沙杨氏模量默认软档 1e5，抬升后默认接 120 步抖铲相位（SHAKE g2 挡，干湿分离，见「参数说明」SHAKE 挡位），全量约 19.5 分钟（1309 步实测）；快档组合 ST 0.02 + safety 1.0 为 10.6 分钟（1335 步）、存档组合 1e6 + 0.5（ST 0.02，关抖铲）为 54.9 分钟，见「参数说明」ST / SAFETY / YOUNG / SHAKE 挡位；轨迹与夹持为 2026-09-03 验收的 v2 版 + 2026-10-12/13 验收的抖铲相位）。18 个脚本入库时已全部通过 py_compile；除体素化 / matplotlib 回放 / 采样模块外，全部脚本 headless 运行（`gs.init(backend=gs.gpu)`），需要 NVIDIA GPU。

## 环境要求

- **NVIDIA GPU + CUDA 版 PyTorch**。仓库根 `pyproject.toml` 的依赖列表不含 torch（其 `[tool.uv]` 段注明 PyTorch 须在 Genesis 之前单独安装），须先装 CUDA 版 torch，再装 genesis。
- Python `>=3.10,<3.14`（`pyproject.toml` `requires-python`；本目录验证时用 3.11）。
- 按项目约定用独立 conda 环境（不污染 base）：

```bash
conda create -n genesis-sandwater python=3.11 -y
conda run -n genesis-sandwater pip install torch --index-url https://download.pytorch.org/whl/cu126
cd D:\workspace\python-workspace\genesis\genesis-world
conda run -n genesis-sandwater pip install -e ".[dev]"
```

- 装 `.[dev]` 而非裸 `-e .` 的原因：核心依赖（trimesh、libigl、moviepy 等）由后者满足，但体素化脚本还需要 `scipy`、matplotlib 回放还需要 `matplotlib`，这两个只在 dev 附加依赖里（仓库 README 对 examples 同样建议 `pip install -e ".[dev]"`）。
- 本目录实测组合：torch 2.11.0+cu128 + genesis-world 1.3.1，RTX 4060 8GB。

## 一次性资产准备

库内只带两个对齐后的网格：`assets/litter_scoop_aligned.glb`、`assets/shovel_vintage_aligned.glb`。**主流程唯一缺的资产是 `assets/litter_scoop_sdf_slots_s0.5.npz`（约 42MB，不入库）**——猫砂铲的 SDF 体素化网格。

主路径：本地再生（纯 CPU，<1 分钟），在本目录下执行：

```bash
conda run -n genesis-sandwater python litter_scoop_voxelize.py noplate 0.5
```

`noplate` 保留铲斗真实开孔底板（输出文件名带 `slots`），`0.5` 是 phase9 半尺寸场景的网格缩放（输出文件名带 `_s0.5`）。

备选：从仓库外的 experiments 目录直接复制（若存在）：

```bash
copy D:\workspace\python-workspace\genesis\experiments\assets\litter_scoop_sdf_slots_s0.5.npz assets\
```

验证文件在位：

```bash
conda run -n genesis-sandwater python -c "import os; print(os.path.getsize('assets/litter_scoop_sdf_slots_s0.5.npz')/1e6, 'MB')"
```

另外：`shovel_wet_sand_sdf.py`（phase5 SDF 变体）默认用全尺寸的 `assets/litter_scoop_sdf_slots.npz`，同样不入库，再生命令为 `conda run -n genesis-sandwater python litter_scoop_voxelize.py noplate`（不带 scale）。

## 快速开始

先 `cd` 到本目录（下述命令均在此目录执行；脚本输出目录基于脚本自身位置，与工作目录无关）：

```bash
cd /d D:\workspace\python-workspace\genesis\genesis-world\examples\sand_water_coupling
```

（PowerShell 下去掉 `/d`。）

**第 1 步：smoke 自检（约 3-4 分钟）**——建场景 + 30 步沉降 + 一次全重启 IK 求解后退出，验证资产、建床、GPU、IK 全链路：

```bash
conda run -n genesis-sandwater --no-capture-output python franka_shovel_deepbed.py smoke
```

看控制台末尾的 `SMOKE done` 行（打印 n_sand / n_water / first-IK grasp_err / gpu mem），过程中无 NaN 报错即通过。注意 smoke 产物在 `recordings/phase9_deepbed_full_smoke/` 且**不含 meta.json，不能喂给渲染脚本**（渲染验证只对 full 记录有效）。

**第 2 步：full 全量仿真（默认组合 young 1e5 + ST 0.2 + safety 0.5 + shake g2 约 20 分钟：实测 1309 步 / 19.5 min，零 NaN，抖铲干沙 -37.6%（468→294）、湿粒留存 298→299 = 100% 零误抖；同族（ST0.2 + s0.5）无抖对照 `off`：1148 步 / 18.6 min / 湿留存 291/291 = 100%；快档组合 ST 0.02 + safety 1.0：1335 步 / 10.6 min / 抖掉 44.8%；存档组合 1e6 + 0.5（ST 0.02）约 55 分钟：1105 步 / 54.9 min，A4 归档数字；1e6 + 1.0：1110 步 / 25.8 min）**：

```bash
conda run -n genesis-sandwater --no-capture-output python franka_shovel_deepbed.py full > deepbed_full.log 2>&1
```

其他组合显式写法（空 tag 占位）：快档 `python franka_shovel_deepbed.py full "" 1.0 1e5 g2 0.02`、存档档 `python franka_shovel_deepbed.py full "" 0.5 1e6 off`（挡位名或数值均可：safety `0.5`/`archive` 或 `1.0`/`fast`、young `1e5`/`soft` 或 `1e6`/`archive`、shake `g2`/`g1`/`off`、ST `0.2`/`high` 或 `0.02`/`base`；ST ≥ 0.2 必配 safety ≤ 0.5）；挡位值写入 meta.json 的 `ddt_safety`、`young_modulus`、`surface_tension` 与 `shake`（挡位 + 参数 + census 结果）字段供溯源。

建议放独立终端或后台任务跑；日志重定向到 `.log` 文件天然被 .gitignore 忽略（见「仓库卫生」）。

**第 3 步：渲染（约 10 分钟；全量帧回放 + 稀疏段补帧 + ffmpeg 合成）**：

```bash
conda run -n genesis-sandwater --no-capture-output python franka_shovel_deepbed_render.py > deepbed_render.log 2>&1
```

产物：`videos/phase9_deepbed_full.mp4`（1 视频帧 = 1 仿真步 = 1/60s；帧数随挡位组合与判停浮动：默认档（ST0.2 + s0.5）含抖铲 1309 帧 = 21.82 s、快档（ST0.02 + s1.0 + g2）1335 帧 = 22.25 s、`off` 挡软档 1159 帧 = 19.32 s、1e6 + 1.0 档 1110 帧 = 18.50 s、A4 存档档 1105 帧 = 18.42 s）+ `frames/` 关键帧 PNG（默认档 9 张，含抖铲末一帧）。

**tag 规则**：sim 的第 2 位置参数与 render 的第 1 位置参数必须一致（默认都是空 tag）。对照跑示例：`franka_shovel_deepbed.py full _v2` → `franka_shovel_deepbed_render.py _v2`。

## 脚本清单

| 脚本 | 功能 | 位置参数 | 依赖资产 | 主要输出 |
|---|---|---|---|---|
| `franka_shovel_deepbed.py` | **主流程（phase9 深床，粗沙 A4 吸收档 + safety / young / shake / ST 四挡位）**：1.8× Poisson 深床，沉降判稳后第 4 秒滴水、吸水，Franka 跟随猫砂铲完成完整铲掘，抬升后抖铲干湿分离 | `[smoke\|full] [tag] [safety] [young] [shake] [st]`（默认 full / 空 / 0.5 / 1e5 / g2 / 0.2；safety 接受 `0.5`/`archive` 或 `1.0`/`fast`，young 接受 `1e5`/`soft` 或 `1e6`/`archive`，shake 接受 `g2`/`g1`/`off`，st 接受 `0.2`/`high` 或 `0.02`/`base`） | `litter_scoop_sdf_slots_s0.5.npz`（需再生） | `recordings/phase9_deepbed_full{tag}/`（frame_*.npz、meta.json、ik_log.csv；跑抖铲挡位时另有 metrics_shake.json） |
| `franka_shovel_deepbed_render.py` | 主流程渲染：湿色 4 桶 + 逐帧重驱 Franka IK + 时间轴对齐 | `[tag] [max_frames]` | `litter_scoop_aligned.glb` + 对应 recordings | `videos/phase9_deepbed_full{tag}.mp4`、`frames/` 关键帧、`recordings/…/ik_render_log{tag}.csv` |
| `franka_shovel_sand_sdf.py` | phase9 半尺寸 Franka+猫砂铲铲湿沙（v4 轨迹版，浅床） | `[max_steps] [sdf文件名] [tag]` | `litter_scoop_sdf_slots_s0.5.npz`（需再生）、glb | `recordings/phase9_franka_shovel_sdf{tag}/`、`videos/` |
| `franka_shovel_sdf_render_wet_native.py` | 上者的原生湿色渲染 | `[max_frames] [tag]` | glb + 对应 recordings | `videos/phase9_franka_shovel_sdf{tag}.mp4`、`frames/` |
| `gripper_shovel_sand.py` | phase7：Franka 夹铲柄复刻 phase5 最优铲掘轨迹 | `[max_steps] [tag]` | glb | `recordings/phase7_gripper_shovel{tag}/`、`videos/` |
| `gripper_shovel_render.py` | phase7 离线重渲（连续 IK 平滑手臂动作） | `[max_frames]` | glb + phase7 recordings | `videos/phase7_gripper_shovel.mp4`、`frames/` |
| `gripper_shovel_render_wet_native.py` | phase7 原生湿色渲染 | `[max_frames]` | glb + phase7 recordings | `videos/phase7_gripper_shovel_sandwet_native.mp4`、`frames/` |
| `shovel_wet_sand.py` | phase5 基线：滴水 + 解析 tilt-box 障碍铲起湿沙 | 无 | glb | `recordings/phase5_shovel_wet/`、`videos/` |
| `shovel_wet_sand_sdf.py` | phase5 SDF 障碍变体（真实猫砂铲网格参与物理） | `[max_steps] [sdf文件名]` | `litter_scoop_sdf_slots.npz`（需再生）、glb | `recordings/phase5_shovel_wet_sdf/`、`videos/` |
| `shovel_wet_sand_replay.py` | phase5 记录的 matplotlib 回放（纯 CPU，不建 genesis 场景） | `[stride] [max_frames]` | `recordings/phase5_shovel_wet/` | `videos/phase5_shovel_wet_replay.mp4` |
| `litter_scoop_render_wet_native.py` | phase8d 无臂版湿色渲染 | `[max_frames]` | glb + `recordings/phase5_shovel_wet_sdf/` | `videos/phase5_shovel_wet_sdf.mp4`、`frames/` |
| `wet_sand_absorption.py` | 吸水测试（无铲，资产依赖最轻的物理脚本） | `[drag_coeff] [tag] [droplet_z]`（默认 0.3 / 空 / 0.205） | 无 | `videos/phase5_absorb_test{tag}.mp4`、`frames/` |
| `wet_sand_absorption_stab.py` | 吸收 max_ratio 稳定性探针（粗场景） | `<max_ratio> [n_steps] [radius] [grid_res]` | 无 | 控制台输出 |
| `litter_scoop_voxelize.py` | 猫砂铲 glb → SDF 体素化（资产再生工具） | `[noplate] [scale]` | `litter_scoop_aligned.glb` | `assets/litter_scoop_sdf[_slots][_s{scale}].npz` |
| `litter_scoop_align.py` | 猫砂铲 objaverse 资产对齐（一次性，产物已入库） | 无 | 原始 objaverse 资产（不在库） | `assets/litter_scoop_aligned.glb`（已入库） |
| `shovel_asset_align.py` | 铁锹资产对齐（一次性，产物已入库） | 无 | 原始 objaverse 资产（不在库） | `assets/shovel_vintage_aligned.glb`（已入库） |
| `franka_gripper_smoke.py` | phase7-1 零物理冒烟：重力/碰撞关闭，IK 追 phase5 记录的铲柄位姿 | 无 | `recordings/phase5_shovel_wet/` | 控制台（零物理自检） |
| `fast_poisson.py` | 向量化 Poisson 盘采样模块，加速建床（被 import 使用，不直接运行） | — | — | monkeypatch，无文件输出 |

所有脚本均无 argparse，裸 `sys.argv` 位置参数，顺序固定、不能带名传参。

## 参数说明

### 常调的顶部常量（`franka_shovel_deepbed.py`）

| 常量 | 当前值 | 位置（约） | 作用 |
|---|---|---|---|
| `grid_res` | 64 | :431 | FLIP 网格分辨率（dx 5.47 mm）；改它必须连 `PARTICLE_RADIUS` 一起（见下） |
| `PARTICLE_RADIUS` | 3.125e-3 | :97 | 沙粒半径（直径 6.25 mm ≈ res64 dx）；与 grid_res 同比例缩放，只改一边会破坏粒径:dx 比 |
| `MAX_RATIO` | 0.5 | :106 | 沙粒吸水上限（A4 档主旋钮：引擎吸收分数下限 1−0.74·(1+max_ratio) 在 ~0.35 以上截断为负、0.5 已到收益终点；0.3→0.5 单独把探针吸干 661→255 步） |
| `SURFACE_TENSION`（即 ST 挡位） | 0.2 | :121 | DEM 表面张力（液桥粘聚）挡位表，argv[6] 选 `0.2`/`high`（默认）或 `0.02`/`base`。**0.2 = ×10 粘力档**（2026-10-13 用户拍板转正默认）：湿团更刚（lift 段内部位移 std 0.020 vs 0.034 mm/帧），抖铲分离依旧单向（干沙 -37.6%、湿粒 100% 留存 0 误抖、SHAKE 段蠕变 std 0.068 mm/帧），**必配 safety ≤ 0.5**（见下守则）；**0.02 = 原始档**（约为库默认 0.007 的 ×3，用户早前选定）：2026-10-13 前所有归档数字之档，也是唯一与快档 safety 1.0 组合验证过的 ST 档。挡位值写入 meta.json 的 `surface_tension` 字段供溯源；探针全数据见仓库外 `experiments/st10_s05_probe/` 与 `experiments/shake_probe/RESULT_SHAKE_ST10.md` |
| `VISCOSITY` | 1.0 | :122 | 沙↔水二次拖曳系数（FLIP `viscosity_coeff`；沿革 0.01→2.0→0.4→0.8→4.0→1.0，A4 档值即 C++ 参考默认 m_ViscosityCoeff=1） |
| `DROPLET_RADIUS` | 0.020 | :182 | 水滴半径（A4 档回到 0.020：体积 ×1.95、水粒 540→1408，是同时提升铺开与深度的唯一旋钮；写入 meta 供渲染端读取） |
| `ABSORB_MAX` | 700 | :270 | 吸水段步数上限（软档 154 步、A4 档 111 步即自然判停，cap 仅兜底；visc4 档尾段慢曾顶满 699） |
| `SAFETY`（即 `ddt_safety` 挡位） | 0.5 | :159 | DEM 子步安全系数挡位表，argv[3] 选 `0.5`/`archive`（默认）或 `1.0`/`fast`。**0.5 = 存档档（默认，2026-10-13 起随 ST 0.2 转正）**：C++ 保守裕度（子步长为公式稳定极限的一半），A4 归档数字之档（1105 步 / 54.9 min / 2.98 s/步，ST0.02 + 1e6 口径），且为默认 ST 0.2 档的必配（见上 ST 行与下方守则）；默认组合（ST0.2 + 1e5 + g2）实测 1309 步 / 19.5 min / 0.90 s/步、湿粒 298→299 = 100%。**1.0 = 快档**：子步长顶满 C++ 公式稳定极限（ddt = r·π·√(ρ/E)/2），本场景全相位实测零 NaN 无 popcorn（ST0.02 口径）：1e6 档 1110 步 / 25.8 min / 1.394 s/步、湿留存 297/297 = 100%，1e5 档 1335 步 / 10.6 min；探针全数据见仓库外 `experiments/safety10_probe/`。两档轨迹自沉降段即分岔、互相不可比；挡位值写入 meta.json 的 `ddt_safety` 字段供渲染/分析溯源 |
| `YOUNG`（即 `young_modulus` 挡位） | 1e5 | :177 | 沙杨氏模量挡位表，argv[4] 选 `1e5`/`soft`（默认）或 `1e6`/`archive`，传入 Sand 材质（接触刚度 K_norm = E·r 与子步公式 ddt ∝ √(1/E) 均随之自动适配）。**1e5 = 软档**：比 C++ 参考值软 10 倍，子步/控制步 2148→680（√10 精确），本场景全相位实测零 NaN 无 popcorn、**湿粒留存 299/299 = 100%**：1159 步 / 9.65 min / 中位 0.460 s/步，探针全数据见仓库外 `experiments/e1e5_gear_verify/`；实测软床物理（存档档口径为基准）：床面低 ~13 mm、渗透 0.0242（−23%）、吸干 154 步（+39%）、铲内装载 583（−15%）、铲尖末触底 −4.1 mm（S2 锚定轨迹几何连带）、wincombo 窗口内 E = 2.5e4。**1e6 = 存档档**：C++ 参考值，A4 归档数字之档。两档轨迹自沉降段即分岔、互相不可比；挡位值写入 meta.json 的 `young_modulus` 字段供溯源 |
| `SHAKE`（即 `SHAKE_GEARS` 挡位） | g2 | :228 | 抖铲相位挡位表，argv[5] 选 `g2`（默认）/`g1`/`off`。**g2 = 3.5 cm / 3.5 Hz / 120 步**（1.73 g 抛掷式）：默认 ST0.2 档干沙抖掉 **37.6%**（468→294）、湿粒 298→299 = **100% 留存 0 误抖**；ST0.02 快档口径 43.8%（主线复跑 44.8%）；**g1 = 2.5 cm / 2.5 Hz / 120 步**（0.63 g 滑落式，ST0.02 档实测）：干沙 -34.2%、湿粒 100%；**off**：跳过相位（抬升直接接停驻，即 2026-10-12 前的无抖铲行为）。挡位 + 参数 + census 结果写入 meta.json 的 `shake` 字段。详见下方「SHAKE 抖铲相位」小节 |

当前默认即 **A4 吸收档**（MAX_RATIO 0.5 / VISCOSITY 1.0 / DROPLET_RADIUS 0.020）+ **ST ×10 粘力档 0.2** + **safety 档 0.5** + **young 软档 1e5** + **shake g2 挡**（2026-10-13 用户拍板的新默认四件套），出自 res64 吸收调参矩阵（探针依据见仓库外 `experiments/res64_absorb_tuning/TUNING.md`）、safety10 / e1e5 / st10 / st10_s05 探针与 shake_probe 四轮抖铲实验（含 ST0.2 复测）。**改任何常量前先跑 smoke**；每个参数的完整沿革与每轮实测数字见仓库外 `experiments/record.md`。

**换挡提示**：safety 1.0 与 young 1e5 均仅在本深床场景全相位验证（1e5 探针在 safety 1.0 下跑出），ST 0.2 仅在 safety 0.5 下验证——高冲击稠密工况（当年 342 m/s popcorn 属远更烈的工况，未覆盖）不在保证范围内；**换场景先用 smoke / 短探针确认无 NaN、无爆粒（popcorn）再上快/软/高粘档**，交付与归档数字可比性要求高时用 0.5 + 1e6 + ST0.02。**软档（1e5）额外注意**：床面比存档档低 ~1 cm（S2 锚定的铲取轨迹因此铲尖末贴底/微没入地面 ~4 mm，视觉连带）；湿团 100% 留存在本场景实测保持，但更软的沙在其他铲形/速度下兜载能力未验证。

**挡位组合守则（ST × safety）**：`SURFACE_TENSION`（ST 挡位，:121）**调到 ≥ 0.2（×10 液桥）时必须搭配 safety ≤ 0.5**——s1.0 大子步下 ×10 粘力触发数值蠕变-剪切断裂（湿团留存跌至 29.2%），s0.5 下同粘力全程稳定（留存 100%、蠕变 std 0.014 mm/帧）；st10 / st10_s05 实验确立，全数据见仓库外 `experiments/st10_s05_probe/RESULT_ST10S05.md`。**默认组合（ST 0.2 + safety 0.5）已自动满足该守则**。

### SHAKE 抖铲相位（干湿分离，`SHAKE_GEARS`，argv[5]，默认 `g2`）

G 抬升完成后、H 停驻前，铲子保持抬升高度沿世界 **+x**（相机屏幕左右向；铲唇朝 +x，即插入方向前缘，−x 侧是封闭铲背 + 铲柄）正弦平移 **120 步 = 2.0 s**，首尾各 **15 步 raised-cosine ramp**（C¹ 平滑起停、无冲击），每步以差分速度经 `set_sdf_obstacle_vel` 驱动（与其余相位同一接口）——往复晃动把铲内松散干沙抖出铲唇，湿团（液桥粘聚、刚体骑行）整体留存，实现干湿分离（用户需求 2026-10-12「铲起来之后左右晃一晃 把干沙子抖下去一些看看」）。挡位表：

| 挡位 | 幅度 / 频率 / 步数 | 峰值速度 / 加速度 | 分离方式 | 实测（默认 ST0.2 档；括号内 ST0.02 快档组合口径） |
|---|---|---|---|---|
| `g2`（默认） | 3.5 cm / 3.5 Hz / 120 步 | 0.77 m/s / 1.73 g | 抛掷式（干沙被抛离铲唇） | 干沙 **468→294（抖掉 37.6%）**、湿粒 298→299 = **100% 留存 0 误抖**；主线复跑 447→273 = 39.6%、湿粒 303→303（ST0.02：272→153 = 43.8%，主线复跑 279→154 = 44.8%、湿粒 300→299） |
| `g1` | 2.5 cm / 2.5 Hz / 120 步 | 0.39 m/s / 0.63 g | 滑落式（干沙滑动越唇） | ST0.02 档实测：干沙 263→175（34.2%）、湿粒 307/307 = 100%（ST0.2 档未复测） |
| `off` | 跳过相位（抬升直接接停驻，即 2026-10-12 前的无抖铲行为） | | | ST0.2 同族无抖对照 1148 步 / 18.6 min（ST0.02 快档 1159 步 / 9.65 min） |

- 计数口径：a4 scoop 分析（铲斗局部框内且 z > S2+2 cm；湿 = ratio > 0.02）。抖掉物全部落回床面、0 粒空中乱飞；0 NaN / 0 popcorn（vmax 峰 ≤2.0 m/s，远低于 20 熔断线）。
- **×10 粘力组合已转正为默认档**（young 1e5 + **ST 0.2** + safety 0.5 + g2，2026-10-13 用户拍板；数字即上表 g2 行）：干沙 468→294（**37.6%**）、湿粒 298→299（**100%**）——×10 液桥未把更多干沙咬在湿团上（铲载差主要是 s0.5 床几何、贴团比例两版同 67%），高粘力档下依旧单向分离。
- **饱和特性**：前 30 步（含 ramp）完成 **~90%**（84-92%）的分离后饱和——剩余干沙为与湿团底座咬合的压实层，纯平移晃动撼不动；要进一步分离需侧倾倒出或高频点抖，继续加大平移幅度会把湿团连同干壳一起送出铲唇。
- **IK 说明**：run 侧 SHAKE 段 grasp_err max ~45 mm / mean ~38-39 mm——warm-start 连续 IK 的每步收敛进度瓶颈（SHAKE 段 DQ_MAX 已放宽到 0.06 rad/步也不变，非限幅问题），follower 是无碰撞运动学重放、**物理零影响**（铲为 SDF 权威运动）；**渲染侧 re-drive 自带漂移恢复**（超 2 mm 放宽重试 + 全重启），视频里手臂跟踪恢复到 max ~19 mm / mean ~1.7 mm。
- 探针全数据见仓库外 `experiments/shake_probe/`（RESULT_SHAKE.md 两轮对照 + RESULT_SHAKE_ST10.md ×10 粘力复测）。

两个易漏点：

- `DROPLET_RADIUS` 由渲染脚本从 meta.json 的 `droplet_radius` 字段读取（粗沙档起 sim 写入；旧记录无该字段时回退 0.016）；`grid_res`（渲染端 :286）与 `PARTICLE_RADIUS`（:58）仍是**硬编码镜像**，与 sim 侧同步改。渲染脚本另有 `n_particles == meta n_water` 断言兜底，不一致会在渲染启动时被拦下。
- 夹爪常量 `FINGER_GRIP` / `HANDLE_OFFSET_LOCAL` / `HANDLE_ANGLE`（约 :244–263）：改动后必须先过 `experiments/phase9_gripcheck.py`（仓库外，import 同一份常量对 6 个记录位姿做 IK + 特写渲染，抽帧确认两指捏在铲柄杆上）再上 full——历史上正是 blade→handle 偏移错导致两指捏空。

### C++ 对齐锁定项（勿动）

以下算法项与 C++ 参考实现（sand-water-coupling-PIC-DEM-3d）逐条对齐，是物理正确性的锚点，改动即偏离参考：

- DEM 子步长公式 `ddt = radius·π·√(DEMDensity/max_Young)/2`，及其速度自适应子循环 `ddt = min(2r/(maxVel+√(9.8·r·2)), ddt)`；`SAFETY`（`ddt_safety`）只整体缩放该 ddt——1.0 即公式原值，0.5 为 Genesis 侧额外保守裕度（公式本身两档一致）；
- 接触力模型：线性弹性法向力 `min(K_norm)·penetration·n`、速度相关切向力 `-min(K_tang)·v_tangential`、库仑摩擦限幅 `|shear| ≤ |normal|·tan(FricAngle)`，其中 `K_norm = Young·radius`、`K_tang = K_norm·Poisson`；
- `poisson_ratio` 保持库默认 0.3（`franka_shovel_deepbed.py` :178 注释明确 never touched）；
- **`YOUNG`（`young_modulus`）挡位与 C++ 对齐口径**：存档档 1e6 = C++ 参考值（对齐锚点）；软档 1e5 为授权偏差（10 倍软化，探针留档 `experiments/e1e5_gear_verify/`）。接触力与子步公式本身两档逐字一致，只是 E 取值不同——软档下所有含 E 的量（K_norm、ddt、wincombo 窗口刚度）同比例缩放；
- 初始采样间距 ~2.001×radius（防初始重叠爆炸）。

## 输出位置与仓库卫生

- 所有输出写在本目录下：`recordings/`（frame_*.npz、meta.json、ik_log.csv、ik_render_log*.csv）、`frames/`（raw 帧 + 关键帧 PNG）、`videos/`（mp4）。
- 仓库 `.gitignore` 只忽略 `*.mp4`、`*.log` 和 `logs/`；**`recordings/`、`frames/`、`*.csv` 都不被忽略**——跑完 `git status` 会出现大量未跟踪文件。
- 建议：长任务日志一律重定向到 `.log`（天然被忽略）；跑完先 `git -C <仓库根> status` 检查，不需要留产物时用 `git -C <仓库根> clean -nd examples/sand_water_coupling` 预览、确认后 `git clean -fd` 删除；或把 `recordings/`、`frames/` 写入 `.git/info/exclude` 一劳永逸。

## 常见坑

1. **改参数先 smoke**——full 默认组合（young 1e5 + ST 0.2 + safety 0.5 + shake g2）约 20 分钟（快档组合 ST 0.02 + safety 1.0 约 11 分钟；存档组合 1e6 + 0.5 约 55 分钟）；smoke 约 4 分钟即可暴露资产缺失、建床、IK 大部分问题。
2. **smoke 不能渲染**：smoke 提前退出、不写 meta.json，渲染脚本启动即读 meta.json 会直接报错。渲染验证只对 full 记录有效。
3. **DROPLET_RADIUS 两文件硬编码须同步**（sim :182 与 render 约 :350，见上）。
4. **夹爪常量改动必须过 `experiments/phase9_gripcheck.py`** 再上 full。
5. **同一块 GPU 不要并行跑两个本目录脚本**（含渲染）——双进程抢 GPU 会导致 CUDA 崩溃。
6. **不带 tag 重跑 full 会清掉旧记录**：sim 在记录开始前自动删除输出目录下全部旧 `frame_*.npz`（:484 附近）；要留档先给旧目录换 tag 或备份。
7. **sim 与 render 的 tag 必须一致**（默认都为空 tag），否则渲染找不到记录目录。

## 性能与资源

**A4 吸收档 + safety / young / shake / ST 四挡位（2026-10-11/13 实测）**：吸收三旋钮与挡位无关——grid_res 64 + 3.125 mm 沙粒 + max_ratio 0.5 / viscosity 1.0 / droplet r 0.020 → n_sand 47 543（res128 的 1/8）、n_water 1408；B 吸收段自然判停（1e6+0.5 档 111 步、1e6+1.0 档 107 步、1e5+1.0 档 154 步、ST0.2+1e5+0.5 档 156 步）。抖铲相位（g2）在抬升与停驻之间加 120 步（每步与全流程同速），且带每 10 步一次的铲内 census 插桩（开销可忽略）。**组合并列**（E 指 young 档；全量总步数含 120 步抖铲，off 挡见括号）：

- **默认组合 1e5 + ST0.2 + 0.5 + shake g2**（2026-10-13 用户拍板转正）：全量 **1309 步 / 19.5 min**（1172 s，均 0.90 s/步），零 NaN 无 popcorn（vmax 峰 2.21）；**抖铲干沙 -37.6%（468→294）、湿粒 298→299 = 100% 留存 0 误抖、SHAKE 段蠕变 std 0.068 mm/帧**；S1 0.0974 / S2 0.0949 / 吸干 156 步 / n_wet 305；同族无抖对照（st10_s05，ST0.2+s0.5 无抖）1148 步 / 18.6 min / 湿留存 291/291 = 100%。全数据见仓库外 `experiments/shake_probe/RESULT_SHAKE_ST10.md`。主线复跑（2026-10-13，挡位转正当日）：1317 步 / 21.2 min、干沙 447→273 = 39.6%（a4 口径，在 37.6% ±5% 带内）、湿粒 303→303 = 100% 0 误抖、蠕变 std 0.090 mm/帧（std 峰 0.194 / p95 峰 12.9 与实验轮逐位一致）、0 NaN。
- **快档组合 1e5 + ST0.02 + 1.0 + shake g2**：全量 **1335 步 / 10.6 min**（633 s；off 挡 1159 步 / 9.65 min / 中位 0.460 s/步，子步/控制步 680 = 2148/√10 精确），零 NaN 无 popcorn；**抖铲干沙 -44.8%（279→154）、湿粒留存 300→299（hold 末稳定 299/299）**；其余物理（1e6+0.5 存档口径为基准）：S1 −13.8 mm（软 10 倍 + 大子步双压实）、渗透 0.0242 m（−23%）、湿斑直径 0.0649 m、铲尖末 −4.1 mm 贴底（S2 锚定几何连带）。
- **1e6 + 1.0（快档 safety、存档 young，均 ST0.02）**：全量 **1110 步 / 25.8 min / 1.394 s/步**（中位 1.361；未跑抖铲），零 NaN 无 popcorn；湿粒留存 297/297 = **100%**；S1 −8.6 mm、渗透 0.0285 m（−9.5%）、湿斑直径 0.0623 m、铲内总装载 629 粒。
- **存档组合 1e6 + 0.5（ST0.02）**：全量 **1105 步 / 54.9 min / 2.98 s/步**（未跑抖铲），零 NaN；渗透 0.0315 m、湿斑直径 0.063 m、湿粒 297；铲载 686 粒、湿粒留存 297/297 = **100%**。A4 归档数字即此组合。

**不同挡位组合的轨迹自沉降段即分岔（颗粒级不可比、不可逐帧混剪）**：safety 切换只改数值积分精度，young 切换改接触刚度（物理常数）；探针全数据见仓库外 `experiments/e1e5_gear_verify/RESULT_E1E5.md` 与 `experiments/safety10_probe/RESULT_SAFETY10.md`。

提速四层：引擎侧三项零物理内核优化（commit `dba2cf7f`；20 控制步稳态中位 39.86→19.75 s/步）合计 **×2.02**；网格粗化 res64 相对 res128（v2 的 40.35 s/步 → visc4 档 2.66 s/步）**×15.2**（理论 16×；残差为 FLIP 网格固定开销 + Franka IK）；safety 挡位 0.5→1.0（A4 口径 2.98→1.394 s/步）**×2.13**；young 挡位 1e6→1e5（safety 1.0 口径 1.394→0.460 s/步中位）**×2.96 ≈ √10**（端到端 25.8→9.65 min ×2.67）。默认组合自 2026-10-13 起取 ST 0.2 + s0.5（×10 粘力必配保守子步，全量 19.5 min）；追求最快可切 ST 0.02 + s1.0（抖铲版 10.6 min）。

| 项目 | 数值 |
|---|---|
| smoke | 约 3-4 分钟（build ~100 s 热缓存 + 30 步沉降 ~0.5 s/步 + 1 次 IK） |
| full（默认 young 1e5 + ST 0.2 + safety 0.5 + shake g2） | 约 20 分钟（1309 步 1172 s = 19.5 min，0.90 s/步，零 NaN；同族 off 挡无抖对照 1148 步 / 18.6 min） |
| full（快档 young 1e5 + ST 0.02 + safety 1.0 + shake g2） | 约 11 分钟（1335 步 633 s = 10.6 min，零 NaN；off 挡 1159 步 579 s = 9.65 min） |
| full（young 1e6 + safety 1.0，ST 0.02） | 约 26 分钟（1110 步 1548 s = 1.394 s/步，零 NaN） |
| full（存档组合 1e6 + 0.5，ST 0.02） | 约 55 分钟（1105 步 3291 s = 2.98 s/步，零 NaN；A4 归档数字） |
| 渲染 | 约 5-10 分钟（全量帧回放 + 补帧 + ffmpeg：默认档（ST0.2 + s0.5）1309-1317 帧 ≈ 21.8-22.0 s、快档（ST0.02 + s1.0 + g2）1335 帧 = 22.25 s、off 挡软档 1159 帧 = 19.32 s、1e6 + 1.0 档 1110 帧 = 18.50 s、A4 档 1105 帧 = 18.42 s，均按 1 帧=1 步推算；软档渲染墙钟未实测，同量级 e10x 软沙渲染 753 帧 3.4 min） |
| 显存 | 约 3.1-3.9 GB（RTX 4060 8GB 实测：软档 3146 MB、A4 档 ~3.9 GB） |

visc4 对照档（旧，2026-10-10）：VISCOSITY 4.0 + droplet r 0.016 → n_water 540、2.66 s/步、全量 1696 步 75.1 min（吸水 699 步顶满 700 cap）；旧版视频见 `videos/phase9_deepbed_full_visc4.mp4`。

细粒参考档（res128 + 1.5625 mm，2026-09-03 v2）：单个控制步约 40 s——子步公式在 r=1.5625 mm 下一个 1/60 s 控制步需 **8589 个 DEM 子步 × 约 4.7 ms/子步 ≈ 40 s**（v2 全量实测 1419 步 / 57263 s = 40.4 s/步；smoke 约 35 分钟、full 约 16 h、渲染约 1.5–2 h、显存 38 万粒约 2.2 GB）。
