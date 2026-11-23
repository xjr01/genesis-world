# 茶壶、扫水和海绵示例使用指南

## 环境与入口

在仓库根目录运行。使用 CUDA 版 PyTorch 和当前仓库的 editable 安装：`python -m pip install -e ".[dev]"`。本目录还包含 PBD 和 SPH 茶壶脚本；以下操作与参数针对 [pbstf_surface_tension.py](pbstf_surface_tension.py)，不同求解器的脚本不能直接共用全部参数。

## 快速运行

```powershell
python examples/multiflow/teapot/pbstf_surface_tension.py --case teapot --steps 100 --vis
python examples/multiflow/teapot/pbstf_surface_tension.py --case sweep --steps 100 --vis
python examples/multiflow/teapot/pbstf_surface_tension.py --case mop --steps 100 --vis
```

去掉 `--steps` 运行默认完整过程：茶壶默认 10,000 步，扫水和海绵默认 1,000 步；默认 `dt=0.01` 秒。短测可能只覆盖初始状态。

| 参数 | 作用 |
| --- | --- |
| `--case` | 选择茶壶、扫水、海绵等 case；完整 choices 见 `--help` |
| `--scale` | 粒子半径为 `1/scale`、直径为 `2/scale`；茶壶要求不低于 300 |
| `--dt`、`--steps` | 时间步长与物理步数 |
| `--sponge-grid-resolution NX NY NZ` | 海绵网格分辨率 |
| `--sponge-render-style skeleton/solid` | 海绵显示为四面体骨架或实体表面 |
| `--vis` | 实时窗口 |
| `--record` | 开启窗口录像，在结束时询问输出路径 |

## 调场景、接触和轨迹

本组 PBSTF 场景以 **Y 轴向上**，四元数采用 **W-X-Y-Z**。世界几何应用长度缩放；不要直接搬用 Z 向上场景的坐标或缩放前的速度。

| 需要修改 | 代码入口 |
| --- | --- |
| 液体、桌面、工具、海绵和阶段设置 | `pbstf_surface_tension.py` 的 `case_settings()` |
| 液体黏度、密度和表面张力参数 | `_case_liquid_material()` |
| 扫水与海绵运动 | `wipe_pose(time, settings)` 和运行器中的工具更新 |
| 茶壶模型、缩放、抓取、末端偏移 | [fluid_helper.py](fluid_helper.py) 的 `create_teapot_settings()` |
| 茶壶运动和机器人跟随 | `teapot_pose()`、`update_teapot_manipulator()` |
| 壁面黏附、摩擦、海绵吸收 | 创建 collider 时的参数 |

运动计划用仿真秒组织，改变 `dt` 会改变每阶段的步数。改 `--steps` 仅改变总时长，不会自动压缩轨迹。先调抓取与工具路径，再固定路径调接触和材料。更低 compliance 对应更强约束；增强壁面黏附也会使液体更难脱离。

详细默认值、缩放规则、茶壶抓取与吸收参数见 [PBSTF 参数说明](pbstf_surface_tension.md)。

## 编辑初始抓取

```powershell
python examples/multiflow/teapot/pbstf_teapot_grasp_editor.py --output out/teapot-grasp.json
```

Tk 控制窗口可调整手的位置与旋转，通过 KUKA IK 更新手臂，并编辑 Shadow Hand 关节。编辑器始终停留在第 0 帧，不模拟倒水。点击 Save Grasp Pose 保存完整 JSON。

`--headless` 保存初始第 0 帧状态并退出。运行器没有自动加载任意 JSON 的命令行参数；应用编辑结果时，将保存的抓取与关节设置核对到 `fluid_helper.py` 的对应配置。不要只复制世界位置而忽略茶壶局部抓取坐标。

## 结果检查

确认机器人跟随、手指与茶壶位置、茶壶碰撞壁面与液体采样对齐；扫水与海绵要看工具是否真实接触液体、吸收是否发生以及移开后是否留下残液。粒子动画用于快速调参，完整效果需要覆盖实际操作阶段。

`--record` 包含窗口，不等同于黄油入口的纯离屏录像。窗口与高精度液体都会增加开销；对比计算速度时单独测量不带窗口、不录制的运行。
