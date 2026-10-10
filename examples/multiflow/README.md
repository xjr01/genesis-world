# 液体交互示例使用指南

本目录包含 PBSTF 显式液体场景、IPBSTF 隐式液体对照，以及倒水、接杯、擦水、搅拌咖啡的操作示例。所有命令在仓库根目录运行。先安装 CUDA 版 PyTorch，再执行 `python -m pip install -e ".[dev]"`；以下入口使用 NVIDIA CUDA。

## 选择与运行

| 目标 | 入口与说明 |
| --- | --- |
| 茶壶、扫水、海绵和初始抓取编辑 | [teapot 使用指南](teapot/README.md) |
| 液体隐式求解对照 | [ipbstf.py](ipbstf.py)；复用 PBSTF case 定义，另增加 `dam_break` |
| 双臂倒水、接杯、擦水与搅拌 | [pbstf_coffee_water.py](pbstf_coffee_water.py) |

```powershell
# 隐式液体短测。
python -m examples.multiflow.ipbstf --case dam_break --steps 100 --vis
# 先检查机器人运动，使用刚体海绵，不模拟液体。
python -m examples.multiflow.pbstf_coffee_water --check-motion --steps 100 --vis
# 保留软海绵但不模拟液体。
python -m examples.multiflow.pbstf_coffee_water --no-liquid --steps 100 --vis
# 完整操作录像。
python -m examples.multiflow.pbstf_coffee_water --record --output out/coffee-water
```

短测用于检查初始状态，可能尚未进入倒水、接杯或擦水阶段。完整场景省略 `--steps`。这些场景以 Y 轴向上；从机器人视角世界 X 向右、负 Z 朝桌面前方，四元数顺序为 W-X-Y-Z。

## 调 IPBSTF

`--case` 选择场景；`--scale` 决定粒子半径 `1/scale`；`--dt` 和 `--steps` 控制离散与时长；`--alpha` 是惯性能量权重；`--iterations` 是每步局部 Newton 迭代次数。几何和工具运动复用 [PBSTF 场景与参数说明](teapot/pbstf_surface_tension.md)，材料入口在 `ipbstf.py` 的 `_liquid_material()`。

IPBSTF 的 `--record` 会开启窗口，并在结束时询问输出路径。对比显式和隐式求解时，先保持初始几何、采样、轨迹和观察时刻相同；不要仅对比步速。

## 调咖啡与水操作

场景、抓取、杯子和液体参数在 [pbstf_coffee_water.py](pbstf_coffee_water.py) 的常量、数据结构及场景构建代码中。运动调试入口包括 `motion_target()`、`update_motion()` 和 `water_cup_pose()`；修改阶段时间与目标时检查相邻阶段连续性，以及真实手部和目标位姿的误差。

| 参数 | 用途 |
| --- | --- |
| `--check-motion` | 无液体、刚体海绵，先验证机器人运动 |
| `--no-liquid` | 无液体，保留软海绵 |
| `--scale` | 粒子精度，默认 1500；更粗的采样可能让液体留在倾斜杯中 |
| `--steps` | 限制物理步数 |
| `--vis` | 实时窗口 |
| `--record` | 保存录像和阶段图像 |
| `--surface` | 重建液体表面，增加显示与重建开销 |
| `--output` | 输出目录，默认 `out/pbstf_coffee_water` |

先在运动检查模式确认杯子抓取、倒水和接杯路径，再加入软海绵与液体。观察手指与碰撞几何是否对齐、杯子是否实际随手运动、液体是否从杯口流出、海绵是否接触液体以及另一只手的搅拌是否持续。改变控制方式或粒子分辨率后重新验收完整操作。
