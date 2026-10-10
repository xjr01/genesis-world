# 花瓶破碎示例使用指南

## 场景与环境

[pbd_vase_fracture.py](pbd_vase_fracture.py) 模拟桌边花瓶受到初始推力和旋转后跌落、破碎。PBD 负责材料及接缝破坏，完全脱离的碎片通过桥接交给刚体求解器处理接触。此示例没有机器人或轨迹文件。

在仓库根目录运行；安装与 GPU 匹配的 PyTorch 和当前仓库 `python -m pip install -e ".[dev]"`。入口使用 `gs.gpu`。花瓶使用仓库共享资产 `genesis/assets/meshes/vase.obj`，碎片碰撞代理在运行时生成。

## 快速运行

```powershell
python examples/fracture/pbd_vase_fracture.py --duration 1
python examples/fracture/pbd_vase_fracture.py --duration 8 --vis
python examples/fracture/pbd_vase_fracture.py --record --output out/vase-fracture.mp4
```

| 参数 | 含义 |
| --- | --- |
| `--duration` | 仿真秒数，默认 8，必须为正 |
| `--vis` | 实时窗口 |
| `--record` | 相机录像，可不显示窗口 |
| `--output` | MP4 文件路径，默认 `pbd_vase_fracture.mp4` |

## 调场景与破碎

修改 `main()` 中的桌子范围、花瓶初始位置和材料。此场景 Z 轴向上，长度用米。默认外层步长 2 ms，10 个子步，粒子间距 5 mm。

| 参数入口 | 作用 |
| --- | --- |
| `sector_seeds()` | 周向扇区和高度层，默认 6 扇区 × 2 层，形成 12 个预定义碎片区域 |
| `fracture_threshold`、`shear_threshold`、`rotation_threshold` | 拉伸、剪切与旋转破坏阈值 |
| `seam_failure_threshold`、`bond_radius_factor` | 接缝失效和邻接范围 |
| `static_friction`、`kinetic_friction` | 花瓶材料接触摩擦 |
| 相机与 `realtime_factor` | 观察角度和慢动作呈现；默认时间因子 0.24 |

分块更细会增加碎片数量和计算开销，也可能让小碎片在凸包表面上持续摇摆。默认碎片接触没有滚动摩擦，不能仅凭最终仍有轻微摇摆判断破碎失效。

## 调运动与冲击

修改文件末尾设置初始速度的表达式：

```python
vel = np.cross((0.0, -4.0, 0.0), pos0 - pos0.mean(axis=0)) + (-0.5, 0.0, 0.0)
vase.set_particles_vel(vel)
```

叉积项提供绕 Y 轴旋转，后一个向量提供整体平移，设置后花瓶自由演化。先固定材料，只调初始速度、花瓶摆放和桌沿位置；确定冲击路径后再调破碎阈值。位置用米，平移速度用米/秒，角速度用弧度/秒。

## 输出与验收

`--record` 输出指定 MP4。录像按配置的慢动作时间因子采样，不应把视频时长直接当作物理时长。检查裂缝是否随冲击形成、碎片是否脱离、是否穿地或持续爆炸；增加 `--duration` 用于观察后续运动。每个参数组合用独立录像路径保存。
