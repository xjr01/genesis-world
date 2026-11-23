# 双臂咖啡与水操作示例使用指南

## 场景与环境

[pbstf_coffee_water.py](pbstf_coffee_water.py) 使用 Sim1 双臂机器人完成倒水、接住倾倒的水杯、扶正放杯、海绵擦水和搅拌咖啡。PBSTF 模拟液体、表面张力与浓度扩散，软海绵通过 PBD 变形并吸收洒出的液体。杯子、搅拌棒和夹爪通过刚体接触交互。

本指南只介绍这个操作示例；本目录及子目录中的其他 example 仅作测试用。

在仓库根目录（包含 `pyproject.toml` 的目录）运行。先安装 CUDA 版 PyTorch，再执行 `python -m pip install -e ".[dev]"`。仿真和回放入口均使用 NVIDIA CUDA；机器人、杯子和搅拌棒模型来自仓库的 `genesis/assets/`。

## 快速运行

```powershell
# 短测：使用刚体海绵预览机器人初始运动。
python -m examples.multiflow.pbstf_coffee_water --check-motion --steps 100 --vis
# 保留软海绵，关闭液体，预览完整操作。
python -m examples.multiflow.pbstf_coffee_water --no-liquid --vis
# 完整液体仿真，保存逐帧 checkpoint，回放时使用液体表面。
python -m examples.multiflow.pbstf_coffee_water --surface --record --output out/coffee-water
```

每个控制步为 0.002 秒，内部推进两个 0.001 秒的物理步，因此 `--steps 100` 只覆盖 0.2 秒初始动作。省略 `--steps` 执行完整流程，默认上限为 14,000 个控制步、28 秒仿真时间；双臂都完成动作时提前结束。追加 `--vis` 可实时观察，追加 `--surface` 可显示重建后的液体表面。

`--surface` 将两种液体一起重建为表面，并把粒子浓度 `c` 插值到表面顶点：`c=1` 为深咖啡色，`c=0` 为乳白色，中间浓度使用混合色。表面重建在主进程内通过 CPU 多线程完成。实时窗口按刷新率选取控制步更新表面，每次在两个物理步完成后更新一次；物理推进和 checkpoint 保存仍按上述间隔执行。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--steps N` | 完整流程 | 限制控制步数，必须为正；100 步为 0.2 秒 |
| `--scale N` | `1500` | 液体粒子间距为 `2/N` 米；增大值提高空间分辨率，也增加粒子数量、显存和计算开销 |
| `--vis` | 关闭 | 打开实时窗口 |
| `--record` | 关闭 | 保存初始状态及每个控制步的视觉回放 checkpoint，增加磁盘占用和写入开销；可与两种运动调试模式组合 |
| `--surface` | 关闭 | 将粒子显示改为液体表面重建，便于观察液面形状，增加渲染开销 |
| `--check-motion` | 关闭 | 关闭液体并使用刚体海绵，较快预览抓取和轨迹；软海绵变形与吸水效果需在完整模式验证 |
| `--check-trajectory` | 关闭 | 启用仿真轨迹验收，增加检查开销，失败时停止；默认跳过验收并继续推进仿真，详见“调机器人动作” |
| `--no-liquid` | 关闭 | 关闭液体，保留软海绵及刚体接触，便于调试形变和动作；吸水效果需在完整模式验证 |
| `--output PATH` | `out/pbstf_coffee_water` | 指标和 checkpoint 的输出目录 |

## 调场景与材料

修改 [pbstf_coffee_water.py](pbstf_coffee_water.py) 顶部常量和 `build_scene()`：

| 内容 | 入口 |
| --- | --- |
| 杯子与液量 | `COFFEE_CUP_POS`、`WATER_CUP_POS`、`COFFEE_FILL_FRACTION`、`WATER_FILL_FRACTION` |
| 机器人与工具 | `ROBOT_POS`、`ROBOT_QUAT`、`SPONGE_START`、`SPONGE_SIZE`、`ROD_PARK` 及抓取姿态常量 |
| 液体材料与混合 | `PBSTF.Liquid` 中的密度、表面张力、黏性和初始浓度；`PBSTFOptions.diffusion_coeff` 控制浓度扩散 |
| 海绵与吸水 | `PBD.Elastic` 材料，以及吸收碰撞体的 `absorption_rate`、`absorption_capacity_fraction` |
| 时间离散 | `CONTROL_DT` 控制动作更新间隔，`CONTACT_SUBSTEPS` 控制每个控制步的物理步数 |
| 相机 | `build_scene()` 中的 `camera_pos`、`camera_lookat` 和 `ViewerOptions` |

本场景 Y 轴向上；从机器人视角，世界 X 向右、负 Z 朝桌面前方。位置用米，四元数顺序为 W-X-Y-Z。默认液体粒子间距约 1.33 mm；降低 `--scale` 可减少计算量，但较粗采样可能让液体滞留在倾斜杯中，改变分辨率后需重新检查倒水和擦水效果。

## 调机器人动作

`water_cup_pose()` 定义倒水阶段的杯子参考轨迹，`motion_target()` 决定双臂动作阶段与目标，`update_motion()` 将目标转换为关节控制。先用 `--check-motion` 预览抓取、倒水和接杯路径，再用 `--no-liquid` 预览软海绵交互，最后运行完整液体场景。

仿真命令 `pbstf_coffee_water` 的轨迹验收默认关闭。追加 `--check-trajectory` 会检查逆运动学误差、抓取、接杯时限、跟踪误差、穿透、液体与夹爪间距和最终动作结果，增加检查开销，失败时停止。它可与 `--check-motion` 或 `--no-liquid` 组合使用，例如：

```powershell
python -m examples.multiflow.pbstf_coffee_water --no-liquid --check-trajectory
```

默认运行跳过上述验收并继续记录指标，动作阶段仍按抓取、倾角等反馈条件切换；液体有限值、质量和求解器错误检查始终保留。液体与夹爪间距仅在开启轨迹验收时计算，关闭时 `metrics.csv` 对应列留空。

修改目标位置或阶段时间时，保持相邻阶段的位姿连续，并检查实际手部与目标的误差。右臂应完成接杯、扶正、放杯和擦水，左臂应完成搅拌并放回搅拌棒；完整流程检查搅拌三圈以及海绵吸收到洒出的水。

## 回放与导出视频

仿真保存的 checkpoint 由 [replay_checkpoints.py](../genesis_origin_example/rendering/replay_checkpoints.py) 加载。沿用快速运行中的输出目录：

```powershell
python -m examples.genesis_origin_example.rendering.replay_checkpoints out/coffee-water/checkpoints
```

回放默认沿用录制时保存的粒子或表面显示模式。追加 `--surface` 强制重建液体表面，追加 `--particles` 强制显示液体粒子，两者互斥；杯子、机器人和海绵保留原显示方式。两种录制模式的 checkpoint 都保存粒子位置、有效标记和浓度，因此已有录制可以直接切换显示方式。显示表面时执行重建，或读取下面的表面缓存。

```powershell
# 将已有粒子录制显示为液体表面。
python -m examples.genesis_origin_example.rendering.replay_checkpoints out/coffee-water/checkpoints --surface
# 将已有表面模式录制显示为粒子，便于快速检查。
python -m examples.genesis_origin_example.rendering.replay_checkpoints out/coffee-water/checkpoints --particles
```

默认按仿真时间以 50 帧/秒选择待显示的 checkpoint。对于本示例的 0.002 秒间隔，通常每 10 个 checkpoint 显示一次，并包含末帧；导出视频保持原始仿真时长。实际播放速度取决于重建和绘制耗时，`--fps` 表示采样目标。需要检查中间状态时，可用方向键逐 checkpoint 浏览，或启用 `--sequential`。

| 回放参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--surface` | 沿用录制配置 | 强制显示重建液体表面并按浓度着色，便于观察液面和混合，增加重建开销；与 `--particles` 互斥 |
| `--particles` | 沿用录制配置 | 强制显示液体粒子，省去表面重建开销，便于检查粒子分布；与 `--surface` 互斥 |
| `--fps N` | `50` | 预览采样和导出视频的帧率，必须为正整数；提高值保留更多运动细节，也增加待处理帧数 |
| `--sequential` | 关闭 | 依次显示全部 checkpoint，便于逐帧检查，播放耗时更长；视频仍按 `--fps` 采样 |
| `--cache-surfaces PATH` | 关闭 | 将重建表面保存到指定目录供重复回放使用，以磁盘空间换取更低的重建开销 |
| `--precompute` | 关闭 | 按当前采样设置预计算表面后退出，省去预览绘制；须同时指定 `--cache-surfaces`，并通过 `--surface` 或录制配置启用液体表面，准备耗时取决于待处理帧数 |
| `--output PATH` | `<checkpoint目录>/recordings` | 指定导出视频的目录 |

反复查看液体表面时，可先预计算，再使用同一显示模式和缓存目录回放：

```powershell
python -m examples.genesis_origin_example.rendering.replay_checkpoints out/coffee-water/checkpoints --surface --cache-surfaces out/coffee-water/surface-cache --precompute
python -m examples.genesis_origin_example.rendering.replay_checkpoints out/coffee-water/checkpoints --surface --cache-surfaces out/coffee-water/surface-cache
```

也可以直接执行第二条命令，边回放边生成缓存。预计算和回放使用相同的 `--fps` 可直接复用所选帧；改变帧率或手动浏览其他 checkpoint 时，会补建缺失的缓存。缓存按实际显示模式区分，包含网格和顶点浓度，因此保留混合颜色，也支持改变观察视角。`--particles` 直接显示粒子；表面缓存仅用于启用重建的液体。缓存可单独删除，需要时重新生成。

| 操作 | 效果 |
| --- | --- |
| 空格 | 播放或暂停 |
| 左 / 右方向键 | 前后移动一个 checkpoint，长按连续浏览 |
| 鼠标 | 旋转、平移和缩放视角，暂停时也可调整 |
| `R` | 开始或停止录制当前视角的视频 |
| `I` | 展开窗口帮助 |
| `Esc` | 退出并保存正在录制的视频 |

先调整视角和窗口大小，按 `R` 开始录制，再按空格播放。视频按 `--fps` 和记录的仿真时间采样；暂停期间不增加视频帧，每次手动跳帧计入一个 checkpoint 间隔。视频时长向上取整到一个输出帧，checkpoint 间隔较大时重复视频帧。默认保存到 `<checkpoint目录>/recordings/recording-0001.mp4`，后续录像按编号递增；回放命令追加 `--output out/coffee-water/videos` 可更改视频目录。

2026-10-12 在 Windows、Ryzen 9 7950X、RTX 4090 上，用约 21.3 万液体粒子的录制抽测第 1、1000、5000、10000 个 checkpoint，960×720 隐藏窗口的完整显示耗时如下，包含文件读取、状态恢复、表面处理和绘制同步，场景初始化与视频编码另计：

| 模式 | 平均每帧耗时 |
| --- | --- |
| 优化前每帧创建重建子进程 | 7.33 秒 |
| 主进程重建 | 0.65 秒 |
| 命中表面缓存 | 0.22 秒 |

该录制的 11,144 个 checkpoint 默认选取 1,116 个显示帧。四个缓存样本各占约 14–26 MB，按样本均值估算完整选帧缓存约需 24 GB；实际占用随表面复杂度变化。这些测量用于判断耗时和磁盘取舍，该场景仍低于实时 50 帧/秒。

回放需使用录制时的 Genesis 版本和模型资产。`scene.pkl` 保存场景描述，加载来源可信的本地录制数据。

## 输出与验收

| 文件 | 内容与用途 |
| --- | --- |
| `<output>/metrics.csv` | 每 100 个控制步、阶段切换和指定观察时刻记录指标，包括双臂阶段、手部误差、杯子倾角、搅拌圈数、液体质量及海绵吸水量 |
| `<output>/checkpoints/scene.pkl` | 开启 `--record` 后保存的场景构建描述 |
| `<output>/checkpoints/frame-00000000.npz` 等 | 初始帧及逐控制步的姿态、粒子位置、有效标记和液体浓度，用于视觉回放 |
| `<output>/checkpoints/recordings/recording-0001.mp4` 等 | 回放窗口中按 `R` 导出的视频，目录可由回放入口的 `--output` 指定 |
| `<缓存目录>/<帧标识>/*.npz` | 指定 `--cache-surfaces` 后保存的表面网格、法线和顶点浓度，供后续回放复用 |

每次实验使用独立的输出目录：`metrics.csv` 会覆盖同名文件，录制要求 `checkpoints/` 为空。完整录制按控制步保存数据，磁盘占用随粒子数和步数增加。

观察杯子是否稳定随手运动、液体是否从杯口流出、接杯后是否保留部分水、海绵是否接触并吸收洒出的液体，以及搅拌棒是否完成三圈并回到放置点。结合回放和 `metrics.csv` 定位异常出现的阶段。
