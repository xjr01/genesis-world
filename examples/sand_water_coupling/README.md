# examples/sand_water_coupling 沙水耦合示例运行指南

## 概述

本目录是沙水耦合（DEM 沙 + FLIP 水 + 吸收耦合）系列示例的入库镜像，来源于仓库外 `experiments/` 的 phase5–phase9 系列脚本：从滴水吸水（phase5）到 Franka 机械臂夹持猫砂铲对深床湿沙完成「沉降 → 滴水 → 下压 → 斜插 → 转起 → 抬升」的完整铲掘（phase9 深床）。主流程参数为 2026-09-03 验收的 v2 版（夹持修复 + 水滴第 4 秒延迟释放）。18 个脚本入库时已全部通过 py_compile；除体素化 / matplotlib 回放 / 采样模块外，全部脚本 headless 运行（`gs.init(backend=gs.gpu)`），需要 NVIDIA GPU。

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

**第 1 步：smoke 自检（约 35 分钟）**——建场景 + 30 步沉降 + 一次全重启 IK 求解后退出，验证资产、建床、GPU、IK 全链路：

```bash
conda run -n genesis-sandwater --no-capture-output python franka_shovel_deepbed.py smoke
```

看控制台末尾的 `SMOKE done` 行（打印 n_sand / n_water / first-IK grasp_err / gpu mem），过程中无 NaN 报错即通过。注意 smoke 产物在 `recordings/phase9_deepbed_full_smoke/` 且**不含 meta.json，不能喂给渲染脚本**（渲染验证只对 full 记录有效）。

**第 2 步：full 全量仿真（约 16 小时；v2 实测 1419 步 15.9h，零 NaN）**：

```bash
conda run -n genesis-sandwater --no-capture-output python franka_shovel_deepbed.py full > deepbed_full.log 2>&1
```

建议放独立终端或后台任务跑；日志重定向到 `.log` 文件天然被 .gitignore 忽略（见「仓库卫生」）。

**第 3 步：渲染（约 1.5–2 小时；1227 帧约 1–1.5h 渲染 + 稀疏段补帧 + ffmpeg 合成）**：

```bash
conda run -n genesis-sandwater --no-capture-output python franka_shovel_deepbed_render.py > deepbed_render.log 2>&1
```

产物：`videos/phase9_deepbed_full.mp4`（1 视频帧 = 1 仿真步 = 1/60s）+ `frames/` 关键帧 PNG。

**tag 规则**：sim 的第 2 位置参数与 render 的第 1 位置参数必须一致（默认都是空 tag）。对照跑示例：`franka_shovel_deepbed.py full _v2` → `franka_shovel_deepbed_render.py _v2`。

## 脚本清单

| 脚本 | 功能 | 位置参数 | 依赖资产 | 主要输出 |
|---|---|---|---|---|
| `franka_shovel_deepbed.py` | **主流程（phase9 深床 v2）**：1.8× Poisson 深床，沉降判稳后第 4 秒滴水、吸水，Franka 跟随猫砂铲完成完整铲掘 | `[smoke\|full] [tag]`（默认 full；tag 默认 smoke→`_smoke`、full→空） | `litter_scoop_sdf_slots_s0.5.npz`（需再生） | `recordings/phase9_deepbed_full{tag}/`（frame_*.npz、meta.json、ik_log.csv） |
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
| `VISCOSITY` | 0.8 | :65 | 沙↔水二次拖曳系数（FLIP `viscosity_coeff`；沿革 0.01→2.0→0.4→0.8） |
| `DROPLET_RADIUS` | 0.016 | :82 | 水滴半径（原始 0.02 的 0.8 倍） |
| `ABSORB_MAX` | 700 | :140 | 吸水段步数上限（吸水自然判停则提前结束） |
| `SAFETY`（即 `ddt_safety`） | 0.5 | :77 | DEM 子步安全系数 |

**改任何常量前先跑 smoke**；每个参数的完整沿革与每轮实测数字见仓库外 `experiments/record.md`。

两个易漏点：

- `DROPLET_RADIUS` 在渲染脚本里是**独立硬编码的镜像**（`franka_shovel_deepbed_render.py` 构建水球处，约 :344；meta.json 不记录该值），两处必须同步改。渲染脚本另有 `n_particles == meta n_water` 断言兜底，漏改会在渲染启动时被拦下。
- 夹爪常量 `FINGER_GRIP` / `HANDLE_OFFSET_LOCAL` / `HANDLE_ANGLE`（约 :121–133）：改动后必须先过 `experiments/phase9_gripcheck.py`（仓库外，import 同一份常量对 6 个记录位姿做 IK + 特写渲染，抽帧确认两指捏在铲柄杆上）再上 full——历史上正是 blade→handle 偏移错导致两指捏空。

### C++ 对齐锁定项（勿动）

以下算法项与 C++ 参考实现（sand-water-coupling-PIC-DEM-3d）逐条对齐，是物理正确性的锚点，改动即偏离参考：

- DEM 子步长公式 `ddt = radius·π·√(DEMDensity/max_Young)/2`，及其速度自适应子循环 `ddt = min(2r/(maxVel+√(9.8·r·2)), ddt)`；
- 接触力模型：线性弹性法向力 `min(K_norm)·penetration·n`、速度相关切向力 `-min(K_tang)·v_tangential`、库仑摩擦限幅 `|shear| ≤ |normal|·tan(FricAngle)`，其中 `K_norm = Young·radius`、`K_tang = K_norm·Poisson`；
- `poisson_ratio` 保持库默认 0.3（`franka_shovel_deepbed.py` :78 注释明确 never touched）；
- 初始采样间距 ~2.001×radius（防初始重叠爆炸）。

## 输出位置与仓库卫生

- 所有输出写在本目录下：`recordings/`（frame_*.npz、meta.json、ik_log.csv、ik_render_log*.csv）、`frames/`（raw 帧 + 关键帧 PNG）、`videos/`（mp4）。
- 仓库 `.gitignore` 只忽略 `*.mp4`、`*.log` 和 `logs/`；**`recordings/`、`frames/`、`*.csv` 都不被忽略**——跑完 `git status` 会出现大量未跟踪文件。
- 建议：长任务日志一律重定向到 `.log`（天然被忽略）；跑完先 `git -C <仓库根> status` 检查，不需要留产物时用 `git -C <仓库根> clean -nd examples/sand_water_coupling` 预览、确认后 `git clean -fd` 删除；或把 `recordings/`、`frames/` 写入 `.git/info/exclude` 一劳永逸。

## 常见坑

1. **改参数先 smoke**——full 一次约 16h；smoke 约 35 分钟即可暴露资产缺失、建床、IK 大部分问题。
2. **smoke 不能渲染**：smoke 提前退出、不写 meta.json，渲染脚本启动即读 meta.json 会直接报错。渲染验证只对 full 记录有效。
3. **DROPLET_RADIUS 两文件硬编码须同步**（sim :82 与 render 约 :344，见上）。
4. **夹爪常量改动必须过 `experiments/phase9_gripcheck.py`** 再上 full。
5. **同一块 GPU 不要并行跑两个本目录脚本**（含渲染）——双进程抢 GPU 会导致 CUDA 崩溃。
6. **不带 tag 重跑 full 会清掉旧记录**：sim 在记录开始前自动删除输出目录下全部旧 `frame_*.npz`（:314–324）；要留档先给旧目录换 tag 或备份。
7. **sim 与 render 的 tag 必须一致**（默认都为空 tag），否则渲染找不到记录目录。

## 性能与资源

单个控制步约 40s 的构成：子步公式在 r=1.5625mm 下给出一个 1/60s 控制步需 **8589 个 DEM 子步 × 约 4.7ms/子步 ≈ 40s**（v2 全量实测 1419 步 / 57263s = 40.4s/步）。

| 项目 | 数值 |
|---|---|
| smoke | 约 35 分钟（build 约 13 分钟 + 30 步沉降 + 1 次 IK） |
| full | 约 16 小时（v2 实测 1419 步 15.9h，零 NaN） |
| 渲染 | 约 1.5–2 小时（1227 帧约 1–1.5h 渲染 + 补帧 + 合成） |
| 显存 | 沙粒约 38 万时约 2.2GB（RTX 4060 8GB 实测可跑） |
