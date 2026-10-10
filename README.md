# 具身操作与多物理场景使用指南

本仓库基于 Genesis World，汇集机器人与液体、沙、布料、黄油和可破碎物体交互的场景，供具身操作任务的场景搭建、轨迹调试、接触调参与结果检查使用。

根目录说明如何开始和选择示例；具体命令、参数、轨迹入口与产物放在各示例目录的中文指南中。

## 选择场景

| 任务 | 使用指南 | 主要调试内容 |
| --- | --- | --- |
| 铲猫砂、滴水与湿沙搬运 | [沙水耦合](examples/sand_water_coupling/README.md) | 铲子资产与 SDF、Franka 抓取、分阶段铲掘轨迹、吸水 |
| 倒水、接杯、擦水、搅拌 | [液体交互](examples/multiflow/README.md) | 双臂运动、杯子抓取、液体与软海绵 |
| 茶壶倒水、扫水、海绵吸水 | [茶壶与 PBSTF](examples/multiflow/teapot/README.md) | 初始抓取、工具轨迹、壁面黏附、吸收 |
| 双臂叠衣服 | [Scene527 叠衣服](examples/garment_folding/README.md) | 初始沉降、关节轨迹、布料接触、断点恢复 |
| 涂抹黄油 | [黄油涂抹](examples/butter_spreading/README.md) | 刀具压入、涂抹与抬起、材料及接触 |
| 花瓶破碎 | [破碎示例](examples/fracture/README.md) | 初始冲击、分块、破坏阈值及碎片接触 |

部分示例通过直接设置工具位姿或机器人关节位置规定运动。它们适合先验证几何、轨迹与接触；评估控制策略时还需接入执行器控制，检查实际运动与目标的偏差。

## 安装

沿用 Genesis 的源码安装方式：先按 [PyTorch 安装页](https://pytorch.org/get-started/locally/)安装适合硬件的 PyTorch，再在本仓库根目录（包含 `pyproject.toml`）运行：

```shell
python -m pip install -e ".[dev]"
```

使用 Python `>=3.10,<3.14`。该命令安装当前仓库及示例所需的开发依赖；只安装官方 `genesis-world` PyPI 包不会包含本仓库的扩展。Quadrants 随项目依赖安装，无需单独安装。

叠衣服的 IPC 后端、铲猫砂的 SDF 资产等额外要求，以及各任务支持的计算后端，见对应示例指南。

## 录像与耗时

各入口的选项并不统一，请以示例指南和 `--help` 为准。黄油与叠衣服可以关闭实时窗口而录像；PBSTF 茶壶的 `--record` 会开启窗口并在结束时询问输出路径。

仿真时间与墙钟时间不同。首次构建可能包含采样、资产预处理和编译。剩余耗时可按剩余步数除以最近步速估计，并单独考虑录像封装。粒子、网格、接触复杂度、显示方式及其他 GPU 任务都会影响速度。

## 目录与参考

| 目录 | 内容 |
| --- | --- |
| `examples/` | 场景、轨迹、任务资产、交互工具与使用指南 |
| `genesis/engine/` | 求解器、材料和耦合实现；普通场景与动作调试优先修改示例 |
| `genesis/assets/` | 共享资产；任务资产也可能位于各示例的 `assets/` |
| `tests/` | 接口、状态恢复及物理行为的回归检查 |
| `out/` | 本地录像、日志、指标和 checkpoint；部分示例另有输出目录 |

通用 API 参见 [Genesis 官方文档](https://genesis-world.readthedocs.io/en/latest/)。本仓库扩展场景的实际入口和默认值以各中文指南及源码为准。

本项目基于 [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world)，源码使用 [Apache 2.0 许可证](LICENSE)。Quadrants、libuipc、PyRender 等组件遵循各自许可证。
