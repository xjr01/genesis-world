# 双臂咖啡与水操作示例使用指南

## 场景与环境

[pbstf_coffee_water.py](pbstf_coffee_water.py) 使用 Sim1 双臂机器人完成倒水、接住倾倒的水杯、扶正放杯、海绵擦水和搅拌咖啡。PBSTF 模拟液体、表面张力与浓度扩散，软海绵通过 PBD 变形并吸收洒出的液体。杯子、搅拌棒和夹爪通过刚体接触交互。

本指南只介绍这个操作示例；本目录及子目录中的其他 example 仅作测试用。

在仓库根目录（包含 `pyproject.toml` 的目录）运行。先安装 CUDA 版 PyTorch，再执行 `python -m pip install -e ".[dev]"`。仿真和回放入口均使用 NVIDIA CUDA；机器人、杯子和搅拌棒模型来自仓库的 `genesis/assets/`。

## 快速运行

```powershell
# 短测：使用刚体海绵检查机器人初始运动。
python -m examples.multiflow.pbstf_coffee_water --check-motion --steps 100 --vis
# 保留软海绵，关闭液体，检查完整操作。
python -m examples.multiflow.pbstf_coffee_water --no-liquid --vis
# 完整液体仿真，保存逐帧 checkpoint。
python -m examples.multiflow.pbstf_coffee_water --record --output out/coffee-water
```

每个控制步为 0.002 秒，内部推进两个 0.001 秒的物理步，因此 `--steps 100` 只覆盖 0.2 秒初始动作。省略 `--steps` 执行完整流程，默认上限为 14,000 个控制步、28 秒仿真时间；双臂都完成动作时提前结束。追加 `--vis` 可实时观察，追加 `--surface` 可显示重建后的液体表面。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--steps N` | 完整流程 | 限制控制步数，必须为正；100 步为 0.2 秒 |
| `--scale N` | `1500` | 液体粒子间距为 `2/N` 米；增大值提高空间分辨率，也增加粒子数量、显存和计算开销 |
| `--vis` | 关闭 | 打开实时窗口 |
| `--record` | 关闭 | 保存初始状态及每个控制步的视觉回放 checkpoint，增加磁盘占用和写入开销；可与两种运动调试模式组合 |
| `--surface` | 关闭 | 将粒子显示改为液体表面重建，便于观察液面形状，增加渲染开销 |
| `--check-motion` | 关闭 | 关闭液体并使用刚体海绵，较快检查抓取和轨迹；软海绵变形与吸水效果需在完整模式验证 |
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

`water_cup_pose()` 定义倒水阶段的杯子参考轨迹，`motion_target()` 决定双臂动作阶段与目标，`update_motion()` 将目标转换为关节控制。先用 `--check-motion` 检查抓取、倒水和接杯路径，再用 `--no-liquid` 检查软海绵交互，最后运行完整液体场景。

修改目标位置或阶段时间时，保持相邻阶段的位姿连续，并检查实际手部与目标的误差。右臂应完成接杯、扶正、放杯和擦水，左臂应完成搅拌并放回搅拌棒；完整流程检查搅拌三圈以及海绵吸收到洒出的水。

## 回放与导出视频

仿真保存的 checkpoint 由 [replay_checkpoints.py](../genesis_origin_example/rendering/replay_checkpoints.py) 加载。沿用快速运行中的输出目录：

```powershell
python -m examples.genesis_origin_example.rendering.replay_checkpoints out/coffee-water/checkpoints
```

| 操作 | 效果 |
| --- | --- |
| 空格 | 播放或暂停 |
| 左 / 右方向键 | 前后移动一帧，长按连续浏览 |
| 鼠标 | 旋转、平移和缩放视角，暂停时也可调整 |
| `R` | 开始或停止录制当前视角的视频 |
| `I` | 展开窗口帮助 |
| `Esc` | 退出并保存正在录制的视频 |

先调整视角和窗口大小，按 `R` 开始录制，再按空格播放。视频以 50 帧/秒按记录的仿真时间采样；暂停期间不增加视频帧，手动逐帧浏览也会计入录像。默认保存到 `<checkpoint目录>/recordings/recording-0001.mp4`，后续录像按编号递增；回放命令追加 `--output out/coffee-water/videos` 可更改视频目录。

回放需使用录制时的 Genesis 版本和模型资产。`scene.pkl` 保存场景描述，加载来源可信的本地录制数据。

## 输出与验收

| 文件 | 内容与用途 |
| --- | --- |
| `<output>/metrics.csv` | 每 100 个控制步、阶段切换和指定观察时刻记录指标，包括双臂阶段、手部误差、杯子倾角、搅拌圈数、液体质量及海绵吸水量 |
| `<output>/checkpoints/scene.pkl` | 开启 `--record` 后保存的场景构建描述 |
| `<output>/checkpoints/frame-00000000.npz` 等 | 初始帧及逐控制步的姿态、粒子位置、有效标记和液体浓度，用于视觉回放 |
| `<output>/checkpoints/recordings/recording-0001.mp4` 等 | 回放窗口中按 `R` 导出的视频，目录可由回放入口的 `--output` 指定 |

每次实验使用独立的输出目录：`metrics.csv` 会覆盖同名文件，录制要求 `checkpoints/` 为空。完整录制按控制步保存数据，磁盘占用随粒子数和步数增加。

观察杯子是否稳定随手运动、液体是否从杯口流出、接杯后是否保留部分水、海绵是否接触并吸收洒出的液体，以及搅拌棒是否完成三圈并回到放置点。运行器会对手部跟踪、刚体穿透、液体有限值及质量等进行检查；结合回放和 `metrics.csv` 定位异常出现的阶段。
