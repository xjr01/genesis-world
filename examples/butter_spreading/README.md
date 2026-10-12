# 黄油涂抹仿真

使用 Genesis 的物质点法（MPM）模拟刀具下压、涂抹和抬起，以及黄油流动和面包压实。刀具按规定轨迹运动，黄油采用 Herschel–Bulkley 本构，面包采用可压实弹塑性材料。

本示例的场景、求解、粒子导出和后处理都在本仓库内。几何使用程序生成的基本体。录像使用 Genesis 相机，不包含 Isaac 场景、渲染器或资产。

调节材料、接触、动作和批量数据参数，请阅读[调参指南](TUNING.md)。

## 安装与运行环境

使用 Python `>=3.10,<3.14`，先安装适合硬件的 PyTorch，再从本仓库根目录安装：

```shell
python -m pip install -e ".[dev]"
```

仿真使用当前仓库的 Genesis 和项目固定的 Quadrants。审计和重建使用 NumPy、SciPy、Numba、scikit-image，随上述安装提供。所有命令均在包含 `pyproject.toml` 的仓库根目录执行，`python` 应指向已安装本仓库的环境。无需其他项目 checkout、独立后处理 Python 或本机专有资产。

完整场景包含约 31 万至 43 万粒子，默认使用 GPU；`--backend cpu` 可用于小规模调试。录像需要可用的图形上下文及驱动，即使不打开实时窗口。首次执行包含编译，耗时明显高于复用缓存。

## 快速开始

```shell
# 短测：只检查构建与初始接触，约 0.035 秒仿真时间。
python -m examples.butter_spreading.demo --config examples/butter_spreading/configs/dataset.json --steps 1000 --output-dir out/butter-smoke

# 完整四秒，保存初末状态和 97 帧全部粒子。
python -m examples.butter_spreading.demo --config examples/butter_spreading/configs/dataset.json --save-state --save-trajectory --output-dir out/butter-trajectory

# 数据流水线：仿真、物理审计、黄油表面重建、波纹诊断。
python -m examples.butter_spreading.pipeline --config examples/butter_spreading/configs/dataset.json --output-dir out/butter-episode-000

# 同一流水线增加 Genesis 相机录像。
python -m examples.butter_spreading.pipeline --config examples/butter_spreading/configs/dataset.json --record --output-dir out/butter-episode-video
```

每次使用新的输出目录。入口拒绝覆盖已有运行；流水线要求空目录。失败日志保留在该目录，重跑使用新的目录。

`demo --vis` 打开实时窗口；`demo --record` 保存相机录像。`pipeline --steps 1000` 可短测完整工具链，预览会跳过最终铺展要求和波纹诊断，并在完成记录中标记为非完整、不可入库的样本。

## 配置与入口

| 配置 | 面包 / 黄油粒子间距 | 粒子总数 | 用途 |
| --- | --- | ---: | --- |
| [configs/dataset.json](configs/dataset.json) | 1.06666666666667 / 0.6 mm | 310,755 | 较低成本的批量生成起点 |
| 代码默认配置，省略 `--config` | 1 / 0.5 mm | 428,160 | 更密集的粒子采样 |

两种配置使用相同材料和接触参数。更改分辨率会改变数值离散，应分别验证薄层、边缘和面包形变。

`--config` 支持部分 JSON，省略字段采用默认值；显式的 `--seed`、`--bread-particle-size`、`--butter-particle-size` 优先。未知字段和不合法参数会报错。完整配置写入 `run-config.json`。

| 参数 | 适用入口 | 含义 |
| --- | --- | --- |
| `--config`、`--seed` | 两者 | 配置文件与初始采样种子 |
| `--backend gpu/cpu` | 两者 | 计算后端，默认 GPU |
| `--steps` | 两者 | 限制步数；省略则完整四秒 |
| `--record` | 两者 | 保存原生相机 MP4，增加绘制耗时 |
| `--output-dir` | 两者 | 本次输出目录 |
| `--bread-particle-size`、`--butter-particle-size` | 两者 | 覆盖粒子间距，单位米 |
| `--vis` | demo | 实时显示 |
| `--save-state`、`--save-trajectory` | demo | 保存初末状态、全程采样；pipeline 自动启用 |
| `--progress-every` | demo | 日志步数间隔，0 表示关闭 |

五段运动为 0–0.9 s 下压、0.9–1.65 s 加速、1.65–3.15 s 铺展、3.15–3.5 s 减速、3.5–4 s 抬起。默认时间步 `3.5e-5 s`，共 114,286 步，离散终点为 4.00001 s。坐标单位米，Z 向上。

## 输出与数据验收

| 文件 | 内容 |
| --- | --- |
| `run-config.json` | 展开的实际参数 |
| `timing.json` | 初始化、五段步进、采样、检查、导出与录像封装耗时 |
| `particle-state.npz` | 初末面包/黄油位置及最终速度 |
| `mpm-state.npz`、`mpm-state.json` | 全程粒子采样与格式、几何、离散参数 |
| `completion.json` | 仿真完成状态、配置与关键源码哈希、版本、命令 |
| `physics-audit.json` | 物理检查布尔值及沉积、铺展、面包漂移等指标 |
| `mpm-surface.npz`、`mpm-surface.json` | 各帧黄油密度等值面与拓扑、体积、计时指标 |
| `surface-ripples.json` | 最终沉积区域的几何波纹诊断 |
| `full-pipeline-timing.json`、各阶段 `.log` | 子进程耗时、退出码及日志 |
| `pipeline-completion.json` | 全流水线完成、完整样本标记、入库标记及产物哈希 |
| `butter-spreading.mp4` | 开启 `--record` 后的 Genesis 相机录像 |

`mpm-state.npz` 中，`particles`、`particle_velocities`、`bread_particles` 的形状为 `(帧数, 粒子数, 3)`；`knife_positions` 为 `(帧数, 3)` 的工具参考位置；`time_s` 为实际离散采样时刻。还包含 `fps=24`、面包中心和尺寸、刀具尺寸。粒子编号在帧间保持一致。完整运行保存 97 帧，积分步中间状态未逐步存盘；这些导出也不是完整求解器恢复 checkpoint。

相机录像约为 30 fps，具体值受时间步离散影响；粒子导出为 24 fps。制作逐帧视觉标签时应按实际采样时间匹配，不能直接按两个文件的帧编号对应。

`mpm-surface.npz` 按帧拼接顶点、法线与三角形，使用 `vertex_offsets`、`face_offsets` 切分。每帧三角形索引相对于该帧顶点，拓扑可随帧变化。密度重建保留断开的分量和孔洞并报告其指标，不改变物理粒子。面包形变保存在粒子轨迹中；本示例不附带外部高精度面包外观网格。

完成运行不等于物理合格。入库前必须确认 `pipeline-completion.json` 的 `is_complete`、`is_full_trajectory`、`is_accepted` 均为 true，`physics-audit.json` 的 `is_passed` 为 true，且文件哈希匹配。审计检查时长、粒子数、有限状态、配置、刀具轨迹、飞散、沉积、铺展和面包漂移。失败阶段以非零状态退出，留下日志，不生成流水线完成标记。

`is_accepted` 表示通过当前涂抹任务的物理判据且完成后处理。波纹指标为诊断量；录像画质、标签和其他任务的质量要求仍需另行验收。更改任务目标或几何时，需同步修改审计判据及波纹区域。

## 独立后处理

```shell
python -m examples.butter_spreading.audit --state out/butter-trajectory/mpm-state.npz --output out/butter-trajectory/physics-audit.json
python -m examples.butter_spreading.surface --state out/butter-trajectory/mpm-state.npz --output out/butter-trajectory/mpm-surface.npz
python -m examples.butter_spreading.ripples --state out/butter-trajectory/mpm-state.npz --surface out/butter-trajectory/mpm-surface.npz --output out/butter-trajectory/surface-ripples.json
```

审计读取轨迹旁的 `mpm-state.json`、`run-config.json` 和 `completion.json`。短测审计需添加 `--preview`。表面重建可通过 `--kernel-width` 指定正的核宽度，单位米；改变它只影响导出表面，应记录为不同重建版本。

## 计时与维护

比较性能使用 `timing.json` 的 `stepping_seconds`，统一粒子配置、轨迹保存、相机开关、硬件和软件环境。它包含刀具更新、接触、求解及轨迹读回；初始化、导出和后处理单独统计。首次编译和缓存状态会影响墙钟时间。

| 文件 | 职责 |
| --- | --- |
| `config.py`、`configs/` | 参数定义与可复用配置 |
| `motion.py` | 连续刀具轨迹 |
| `demo.py`、`contact.py`、`runtime.py` | 场景、步进、接触、局部显示通知管理和导出 |
| `pipeline.py` | 独立数据流水线与失败传播 |
| `audit.py`、`surface.py`、`ripples.py` | 粒子验收、表面重建与诊断 |

修改后在仓库根目录运行相关测试：

```shell
python -m pytest -n 0 --backend gpu tests/particles/test_butter_spreading.py tests/particles/test_butter_pipeline.py tests/particles/test_butter_runtime.py
```

纯物理步进期间，黄油入口仅临时暂停当前场景的显示通知，退出循环（包括异常退出）后恢复并刷新显示缓存。有窗口或相机时保持显示通知。相关处理位于 `runtime.py`，公共引擎和其他示例的订阅行为保持原样。使用该上下文的循环内应仅进行物理操作。

`motion.knife_pose()` 返回世界坐标轨迹；`demo.knife_pose()` 为面包顶面坐标系的便捷入口，其 Z 原点位于未变形的面包上表面。配置文件和导出的 `knife_positions` 均使用世界坐标。
