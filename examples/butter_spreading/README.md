# 黄油涂抹示例使用指南

## 场景与环境

本示例用 MPM 模拟刀具压入、涂抹、抬起黄油，以及可压实的面包。刀具位姿由轨迹直接规定，尚未通过机械臂执行器驱动。

在仓库根目录（包含 `pyproject.toml` 的目录）运行。先安装与 GPU 匹配的 PyTorch，再执行 `python -m pip install -e ".[dev]"`。支持 `--backend gpu` 或 `cpu`；默认规模的完整运行宜使用 GPU。使用当前 checkout 的环境，避免加载另一个已安装的 Genesis。

## 快速运行

```powershell
# 短测：约 0.035 秒仿真时间，仅检查构建和初始状态。
python -m examples.butter_spreading.demo --backend gpu --steps 1000 --progress-every 1000
# 完整流程，离屏录像。
python -m examples.butter_spreading.demo --backend gpu --record --output out/butter-full
# 实时观察；去掉 --steps 运行完整轨迹。
python -m examples.butter_spreading.demo --backend gpu --steps 1000 --vis
```

完整默认运行 114,286 步，约 4 秒仿真时间。短测不会覆盖涂抹和抬起阶段。

| 参数 | 含义 |
| --- | --- |
| `--steps` | 限制物理步数；省略时使用完整轨迹 |
| `--backend gpu/cpu` | 选择后端 |
| `--vis` | 开启实时窗口 |
| `--record` | 保存相机录像，可以不带 `--vis` |
| `--output` | 录像输出目录，默认 `out/butter_spreading` |
| `--progress-every` | 每 N 步报告进度及平均步速，0 关闭报告 |

## 调场景与材料

修改 [demo.py](demo.py) 的 `build_scene()`：

| 内容 | 入口与默认值 |
| --- | --- |
| 面包、黄油、刀具、支撑台 | 尺寸、位置和刀具 XY 偏移 |
| 相机 | `camera_pos`、`camera_lookat`、FOV 和相机分辨率 |
| 时间离散 | 文件顶部 `DT=3.5e-5` 秒 |
| 空间离散 | 网格密度 512；面包间距 1 mm、黄油间距 0.5 mm |
| 本构 | 创建 `PorousBread` 与 `HerschelBulkleyButter` 时传入的材料参数 |
| 接触 | 创建 `ButterContact` 时的黏附应力、距离、剪切拖曳和刀具分离响应 |

默认面包 255,360 粒子，黄油 172,800 粒子。修改粒子间距会改变分辨率、数量及离散结果，应在目标分辨率下重新验收。本构实现位于 [engine 材料模块](../../genesis/engine/materials/MPM/butter.py)，[contact.py](contact.py) 用 Quadrants kernel 执行接触并直接更新 MPM 粒子字段。

## 调工具轨迹

修改 `knife_pose(time_s)`。输入是仿真秒，返回相对面包上表面任务原点的 `(x, y, z)`；主循环再加上任务世界原点和刀具 XY 偏移。此场景 Z 轴向上，位置单位为米。

优先调压入高度、水平起终点、各阶段时长和抬起高度。压入、加速涂抹、减速、抬起阶段要保持位置连续。运行器从相邻两步位置计算速度，位置跳变会造成很大的接触速度。先固定时间步长，仅改变轨迹，便于区分动作与离散精度的影响。

## 输出与验收

`--record` 输出 `<output>/butter-spreading.mp4`；终端报告构建耗时、粒子数、步数、仿真时间和平均步速。运行完成后检查粒子位置及速度是否有限。入口当前没有自动保存完整粒子轨迹或恢复 checkpoint。

观察黄油是否铺展、刀具是否穿透、面包是否被合理压实，以及抬起时黄油是否脱离。有限值检查和运行完成不等于操作效果合格。

本地在 2026-10-10 完成完整离屏录像，仿真约 24 分钟；该耗时不是其他硬件的保证。本构函数与 integration 分支一致，但接触实现已由 PyTorch 改成 Quadrants，完整粒子轨迹等价性尚未验证。

## 常见问题

- 首次构建需要加载和编译；估计剩余耗时用剩余步数除以最近步速，并单独考虑构建时间。
- 1000 步只有 0.035 秒，看不到完整涂抹是正常的。
- 实时窗口和录像同时运行曾触发渲染缓冲并发错误；完整录像优先不带 `--vis`。
- 对比修改前后效果时保持初始采样、随机种子、相机和观察时刻一致，每轮使用独立输出目录。
