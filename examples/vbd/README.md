# Genesis VBD Solver

Profiling: [original bottleneck analysis](results/profile/README.md) and
[graph scheduling and cooperative vertex results](results/scheduling/README.md).

CUDA runs replay each color sweep through a Quadrants graph and use eight cooperating threads per vertex.
Coloring is computed during scene construction. CPU runs use one thread per vertex.

Local energy line search is disabled by default. Add `--line-search` to `vbd_elastic.py`, `vbd_shell.py`, or
`vbd_profile.py` to enable backtracking for difficult deformations at extra computational cost. The corresponding
solver option is `VBDOptions(is_line_search_enabled=True)`; `line_search_iterations` limits its backtracking attempts.
Contact displacement limits and invalid-state checks remain active in both modes. This flag applies to the VBD solver.

## 运行示例

在仓库根目录、安装了当前 checkout 的 Python 环境中运行：

```powershell
python examples/vbd/vbd_shell.py -g -v
python examples/vbd/vbd_elastic.py -g -v
python examples/vbd/vbd_elastic.py -g -v --solver ipc
python examples/vbd/vbd_shell.py -g --scenario fold --plastic
python examples/vbd/vbd_elastic.py -g --scenario load_unload --plastic --yield-strain 1
```

`-g` 使用 GPU，`-v` 打开查看器，`-s` 设置步数（默认 200）。省略 `-v` 可无窗口运行；省略 `-g` 使用 CPU。两个文件可以独立运行。所有 jelly 仿真统一读取 `genesis/assets/meshes/jelly/remeshed.obj`，包括薄壳、VBD 弹性体、加载卸载和 FEM/IPC 基线。资产已采用米制尺寸和 Z-up，脚本仅将底部抬高到 0.015 m。气泡装饰不参与仿真。

时间步和重力由 `SimOptions` 统一控制。示例采用 `dt=0.005`、2 个子步、每子步最多 20 次迭代；`--iterations` 和 `--precision 32|64` 可用于收敛与精度对照。`--tolerance` 默认 `1e-7` 米，当所有顶点的局部校正低于此值时提前结束；设为 `0` 执行完整迭代次数。

弹性体下落示例默认使用 `E=10000 Pa`，便于观察果冻落地时的压缩和回振。加载卸载示例及 `VBD.Elastic` 材料本身仍默认使用 `100000 Pa`。可用 `--youngs-modulus` 显式覆盖；更高刚度缩小形变量，并提高有限迭代下的求解难度。弹性体示例还支持 `--substeps`：增加子步可减少隐式积分对振动的耗散，同时增加每帧计算量。

体积加载卸载场景固定底部顶点，在前半段向顶部施加总计 `--load-force` N 的水平力（默认 20 N），后半段清除。上面的塑性演示显式降低屈服阈值以产生残余形变；可省略 `--plastic` 对照弹性恢复。JSON 记录卸载前位移、末态残余位移和累积塑性应变。

## 共享仿真资产

`remeshed.obj` 从原始 `jelly.glb` 的 `Jelly_Fluted_12Lobes` 节点生成，采用 2 mm 目标边长进行各向同性重网格化。原始 GLB 保留为来源。新表面包含 12,740 个顶点和 25,476 个三角形；最小三角形角从 0.819 度提高到 17.297 度，最小角低于 5 度的三角形从 2,160 个降到 0。双向顶点及面心采样的最大表面偏差为 0.180 mm，体积变化为 +0.00422%。采样偏差不是严格 Hausdorff 距离上界。

资产通过闭合性、朝向、自相交和初始四面体正体积检查。弹性体脚本从同一个保存的表面生成四面体，保留 `--tet-max-volume` 控制；薄壳直接使用该表面。每份仿真 JSON 记录资产路径和 SHA-256，便于区分网格版本。具体质量指标和上下表面网格对比见 [资产说明](../../genesis/assets/meshes/jelly/README.md)。

重新生成资产及质量报告：

```powershell
python examples/vbd/vbd_remesh.py
```

重网格化后的下落验收在 RTX 4090 上使用 GPU fp32、每帧 5 ms 和两个物理子步，每组运行 200 帧。每帧读取公开顶点，以 float64 计算相对于初始网格的有向体积比：

| 网格 / 求解器 | 刚度 Pa | 全程采样最小体积比 | 采样中最多负体积单元 |
| --- | ---: | ---: | ---: |
| 原始网格 / IPC | 10,000 | -37.0624 | 294 |
| 重网格 / IPC | 10,000 | 0.6510 | 0 |
| 重网格 / VBD | 10,000 | 0.5373 | 0 |
| 重网格 / IPC | 100,000 | 0.9530 | 0 |
| 重网格 / VBD | 100,000 | 0.5505 | 0 |

原始网格 IPC 对照末态仍有 293 个负体积单元，并呈现顶部尖刺；重网格化后该次对照中的尖刺消失。此验证检查采样时刻的体积，覆盖当前下落设置。原网格较差的单元形状是这次问题的重要因素；这些数据并不能把所有视觉伪影都归为四面体翻转。结果及逐帧 CSV 见 [重网格验收](results/remesh_validation.json)，图像见 [IPC 新旧网格对比](results/remesh_ipc_comparison.png)。

同一新资产还完成了 200 帧 [薄壳下落](results/remesh/vbd_shell_jelly.json) 和 [20 N 塑性加载卸载](results/remesh/vbd_elastic_load_unload.json)。薄壳末态最低点为 0.170 mm，高于 0.125 mm 半厚度；加载卸载末态最小体积比为 0.0203，记录到正的累积塑性应变。加载卸载的位移同时包含重力和残余运动。

## 接口与参数映射

`gs.materials.VBD.Cloth` 创建三角形薄壳，`gs.materials.VBD.Elastic` 创建四面体弹性体。两者都返回 `VBDEntity`。`gs.morphs.TetrahedralMesh` 可直接提供体积网格；封闭表面网格也可复用 Genesis 的四面体化流程。薄壳保留表面三角形拓扑。

| 需求参数 | Genesis 接口 |
| --- | --- |
| `youngs_modulus_pa`, `poisson_ratio`, `density_kg_m3` | 材料 `E`, `nu`, `rho` |
| `self_friction` | 材料 `friction_mu`，同时用于地面和自接触 |
| `shear_stiffness_n_m`, `thickness_m` | Cloth 的 `shear_stiffness`, `thickness` |
| `bending_model`, `bending_rigidity_nm` | Cloth 的 `bending_model`, `bending_rigidity` |
| `bending_plasticity` | Cloth 的 `is_bending_plasticity_enabled` |
| `bending_yield_angle_deg`, `bending_plastic_hardening` | Cloth 的同名字段 |
| `volume_plasticity` | Elastic 的 `is_volume_plasticity_enabled` |
| `volume_yield_strain_percent`, `volume_plastic_hardening` | Elastic 的同名字段 |
| `contact_d_hat_m` | `gs.options.VBDOptions(contact_d_hat=...)` |
| `dt`, `gravity` | `gs.options.SimOptions(dt=..., gravity=...)` |
| `tet_max_volume_m3` | 弹性体示例的 `--tet-max-volume`，实际网格规模记录在输出中 |

```python
entity.set_vertex_constraints([0, 1])
entity.set_vertex_forces([[1, 0, 0], [2, 0, 0]], [2, 2])
entity.clear_vertex_forces()
state = entity.get_state()
scene.reset()
```

点力单位为 N，采用世界坐标。每次 `set_vertex_forces` 替换所选环境中该实体的完整力集合，同一顶点的重复项相加，其余顶点清零。力持续到下一次替换或清除，更新力保留位置、速度和塑性历史。控制接口支持 `envs_idx`；顶点索引均为实体局部索引。`get_state().pos`、`vel` 和 `forces` 始终为 `[B, N, 3]`。

构建前设置的控制进入初始快照。`scene.reset()` 恢复初始位置、速度、固定点、力和塑性历史；也可通过 `scene.get_state()` 保存快照，再传给 `scene.reset(state)`。实体的 `surface_triangles` 和 `vgeoms[*].sim_verts_idx` 提供表面映射。

## 材料与接触约定

求解器最小化隐式 Euler 的惯性能量和材料、接触能量。静态图着色使同色顶点可并行计算局部梯度和 3×3 Hessian，逐色更新位置，并用局部回溯控制能量下降。

体积材料使用可压缩 Neo-Hookean 能量：`mu=E/(2*(1+nu))`，`lambda=E*nu/((1+nu)*(1-2*nu))`，单位参考体积能量为 `mu/2*(||F||^2-3)-mu*log(J)+lambda/2*log(J)^2`。`J` 接近零时能量发散，以抵抗体积塌缩。塑性采用弹性奇异值的对数偏应变返回映射，屈服阈值为 `log(1+percent/100)`，累积塑性应变提高后续阈值。塑性变形保持体积。

薄壳使用平面应力膜能量；独立剪切刚度添加到 Green 应变的偏量能量中。完整厚度 `t` 决定面密度 `rho*t`、膜刚度和两侧各 `t/2` 的接触半径。离散弯曲使用静止二面角和 `3*D*edge_length^2/(area_left+area_right)` 的铰链系数。折痕通过更新参考二面角产生，硬化按累积塑性转角增加屈服阈值。`bending_model=None` 使弯曲能量及其历史更新全部关闭。

接触覆盖顶点—三角形、边—边及静态平面，包含对数障碍能量和正则化摩擦。`contact_d_hat` 是物理表面之外的激活间隙，与薄壳厚度分别计算。候选对使用 GPU 包围盒层次结构生成；运动受保守距离界约束。自由顶点每轮搜索的相对运动预算上限为 `max(2*contact_d_hat, 最短关联边长/4)`，兼顾接触间隙和局部网格尺度。预算耗尽时重新搜索后继续下降；较大力会增加所需迭代数。公共平移单独处理，固定点按其目标位置分配预算。塑性历史在每个已接受子步更新一次。

同一实体的自接触激活距离以参考形状中的物理间隙为上限，避免密集网格在静止形状中产生额外排斥力。惯性预测先逐顶点处理地面，再局部修复单元可行性；剩余限步按实体及其接触关联传播，保持独立实体的运动独立。惯性能量始终使用完整预测目标。

当前集成范围为 VBD 实体之间的接触和固定刚体 Plane；平面使用构建时的姿态，运行期间保持固定。查看器使用 Rasterizer。自动微分、其他求解器之间的双向接触和其他渲染器会明确报错。数值无效或接触交叉会在同一次 `scene.step()` 返回前报错，此后必须重置所有环境才能继续读取或推进状态。

## 性能复现

```powershell
python examples/vbd/vbd_shell.py -g --benchmark
python examples/vbd/vbd_elastic.py -g --youngs-modulus 100000 --benchmark
python examples/vbd/vbd_elastic.py -g --solver ipc --youngs-modulus 100000 --benchmark
```

`--benchmark` 预热 20 个采样帧，然后重置并重复 5 次。默认每次推进 1 秒仿真时间（200 帧），每帧同步后计时，输出平均耗时、P95、总墙钟时间及独立的初始化时间。初始化从 `gs.init()` 返回后开始，包含网格预处理、场景构建和首次内核编译；总墙钟时间包含五轮推进和轮间重置，预热单独排除。报告写入 `out/vbd/`，可用 `-o` 更改目录。

弹性体对照采用相同资产提取、四面体化输入、材料参数、物理子步长度和接触距离。每个采样帧是 5 ms、两个 2.5 ms 物理子步。当前 IPC 接入每次调用使用 `SimOptions.dt`，因此脚本为 IPC 构建 `dt=0.0025, substeps=1` 的场景，并在一帧内调用两次 `scene.step()`；VBD 使用 `dt=0.005, substeps=2` 并调用一次。JSON 同时记录 `dt/substeps`（采样帧）和 `scene_dt/scene_substeps`（实际场景设置）。

VBD 使用对数 Neo-Hookean，FEM/IPC 使用当前接入提供的 stable Neo-Hookean，接触刚度的定义也不同。JSON 的 `precision` 是 Genesis 状态精度；当前 IPC 原生交换接口使用 float64。对照衡量当前集成在相同物理时长下的耗时和变形效果，两者的能量与收敛准则分别记录。FEM/IPC 基线限于弹性下落场景；塑性和加载卸载单独验收。

IPC 使用 libuipc 0.0.25 的原生默认求解配置：Newton 最多 1024 次、速度容差 0.05、线搜索最多 8 次、`fused_pcg` 线性求解容差 0.001。软件版本、完整采样设置及相关默认值见 [environment.json](results/environment.json)。

以下为 2026-10-23 **原始 GLB 网格、回弹修复前的初版性能记录**，用于保留初版对照；当前求解器及仿真网格均已变化，需用上述命令重新计时。设备为 RTX 4090、Ryzen 9 7950X、Windows，关闭查看器。每轮 200 帧、5 轮，共推进 5 秒物理时间；表中耗时对应一个 5 ms 采样帧，含两个物理子步。弹性体使用 `E=100000 Pa`。

| 场景 | 网格规模 | 平均 ms/帧 | P95 ms/帧 | 五轮墙钟 s | 初始化 s |
| --- | --- | ---: | ---: | ---: | ---: |
| [VBD jelly 薄壳](results/vbd_shell_jelly.json) | 13,442 顶点 / 26,880 三角形 | 357.13 | 420.18 | 357.14 | 14.49 |
| [VBD jelly 弹性体](results/vbd_elastic_drop.json) | 22,212 顶点 / 96,745 四面体 | 406.61 | 438.59 | 406.63 | 18.83 |
| [FEM/IPC jelly 弹性体](results/ipc_elastic_drop.json) | 同上 | 110.35 | 193.71 | 110.49 | 12.89 |

初版 VBD 弹性体在这组设置下耗时约为 FEM/IPC 的 3.68 倍。该次 VBD 弹性体末态最小体积比为 0.426，最低点为 1.14e-6 m；薄壳末态最低点为 0.164 mm，高于半厚度 0.125 mm。所有计时运行均启用自接触、地面接触和摩擦。

原始网格的单次 1 秒 IPC 下落诊断发现 232 / 96,745 个四面体的有向体积为负；直接检查原生 float64 顶点得到相同数量，排除了公开 fp32 状态转换作为原因。最小体积比约为 -22.11，原生与公开顶点最大差约为 3.73e-9 m，详见 [ipc_quality.json](results/ipc_quality.json)。该诊断独立于五轮计时运行，属于原始网格的历史记录。

## 自动验收

```powershell
python -m pytest -n 0 tests/deformable/test_vbd.py --backend cpu
python -m pytest -n 0 tests/deformable/test_vbd.py --backend gpu
```

测试覆盖单环境和两个环境：自由落体、固定点、重复力求和、运行中力替换与持续作用、分环境重置、地面与自接触、弹性回弹、密集表面的静止形状、独立实体的运动、摩擦、材料参数响应、两类塑性卸载、屈服与硬化、关闭弯曲，以及失败后重置。接触测试采用 4 个子步以分辨小弹性体的回振。CPU 使用双精度，GPU 使用单精度。渲染可用上述 `-v` 命令检查。

重网格化前的求解器验证（2026-10-23）：CPU fp64 与 GPU fp32 各 8 项通过。本地相机检查确认原始 jelly 的 VBD 网格随仿真更新，原 FEM 网格的更新也通过。已有 FEM 渲染快照测试因远程参考图片下载失败而未完成。

原始网格在回弹修复后另运行 200 帧，采用下落示例默认的 `E=10000 Pa`、2 个子步和 20 次迭代。首次接触约在 0.055 s，首次压缩极值在 0.075 s，随后质心回升 2.11 mm，上升速度峰值为 0.105 m/s；最低顶点全程高于地面 2.44e-6 m，末态最小体积比为 0.00387。数据按顶点质量加权，见 [回弹验收](results/vbd_elastic_rebound.json) 和 [逐帧运动记录](results/vbd_elastic_rebound.csv)。这次运行用于原始网格的形变与回弹验收，前面的五轮性能表保留初版计时。

原始网格在回弹修复后的 [20 N 加载卸载](results/vbd_elastic_load_unload.json) 已复跑 200 帧，刚度为 `100000 Pa`，屈服应变设为 1%，末态最小体积比为 0.104。顶部平均位移在卸载前为 48.2 mm，末态为 53.3 mm；该位移包含重力和未消散的运动，塑性残留由上面的无重力单元测试单独验证。[薄片折叠](results/vbd_shell_fold.json) 也复跑 200 帧，含塑性折痕和自接触，末态最小高度 0.432 mm，大于半厚度 0.125 mm。

算法参考：[Gaia](https://github.com/AnkaChan/Gaia)。性能对照使用当前 Genesis FEM/IPC。

## 实现与验收要求

在 Genesis 中实现 GPU VBD solver，支持本仓库的薄壳与体积软体，以及下列材料参数。目标是在保留功能和合理变形效果的前提下提高仿真速度。

只做 solver 及必要的 Genesis 接入，不涉及前端、HTTP/WebSocket、文件导入导出或渲染。参数需要实际生效、调节方向合理；不要求精确标定真实材料，也不要求数值复现现有 IPC。材料模型和参数转换方式可自行选择，但应简要说明映射关系。

## 1. 材料参数

以下字段和默认值保留需求文档中的参数语义；Genesis 对应接口见前面的映射表。

| 参数 | 适用类型 | 默认值 | 要求 |
| --- | --- | --- | --- |
| `youngs_modulus_pa` | 两者 | 100000 | 弹性刚度，Pa；越大越难变形。 |
| `poisson_ratio` | 两者 | 0.25 | 参与材料响应调节；体积模式控制横向收缩和可压缩性，薄壳可沿用系数换算语义。 |
| `density_kg_m3` | 两者 | 360 | 密度，kg/m³；决定质量和惯性，薄壳结合厚度计算质量。 |
| `self_friction` | 两者 | 0.4 | 自接触与地面接触的摩擦系数；0 表示无摩擦。 |
| `shear_stiffness_n_m` | 薄壳 | 8 | 面内剪切刚度，N/m；独立于杨氏模量、泊松比和厚度调整。 |
| `thickness_m` | 薄壳 | 0.00025 | 完整厚度，m；影响质量、面内拉伸刚度和接触厚度，不能误作半厚度。 |
| `bending_model` | 薄壳 | `DiscreteShellBending` | 支持离散壳弯曲；`null` 表示关闭弯曲能量。 |
| `bending_rigidity_nm` | 薄壳 | 5e-8 | 独立弯曲刚度，N·m；越大越难弯曲，允许为 0。 |
| `bending_plasticity` | 薄壳 | false | 开启后支持永久折痕；不要求面内永久拉伸或剪切。 |
| `bending_yield_angle_deg` | 薄壳 | 10 | 参考屈服折角，度；越小越容易留下折痕。 |
| `bending_plastic_hardening` | 薄壳 | 0 | 折痕积累后提高后续屈服门槛；0 表示不硬化。 |
| `volume_plasticity` | 体积 | false | 开启后支持永久变形；塑性保持体积，不要求永久体积压实。 |
| `volume_yield_strain_percent` | 体积 | 10 | 参考屈服形变百分比；越小越容易产生永久变形，不是残留变形上限。 |
| `volume_plastic_hardening` | 体积 | 0 | 塑性积累后提高后续屈服门槛；0 表示不硬化。 |

数值必须有限。杨氏模量、密度、厚度和剪切刚度为正；泊松比范围为 `(0, 0.5)`；屈服折角范围为 `[0, 180]`。摩擦、弯曲刚度、屈服形变和两种硬化参数允许为 0，不能为负。

## 2. Solver 必须支持的行为

- **几何与变形：**三角形薄壳、四面体体积软体；支持大转动、拉伸、压缩、剪切及薄壳弯曲。网格生成可复用 Genesis，`tet_max_volume_m3` 属于预处理参数。
- **推进：**三维重力 `gravity`，默认 `(0, 0, -9.81)`；可调时间步 `dt`，默认 0.005 s。允许内部子步，一次推进的总时间仍为 `dt`。
- **固定点：**支持指定顶点保持位置不变。变更固定点允许重建并重置，不要求无重置切换。
- **点力：**按顶点施加世界坐标三维恒定力，单位 N；同一点多个力相加。支持运行中替换或清除力，不重置形状、速度和塑性历史。固定点不因外力移动。
- **接触：**支持静态地面、自碰撞和摩擦，覆盖薄壳折叠接触与体积软体大变形。支持可调接触作用距离，对应 `contact_d_hat_m`，默认 0.0005 m；说明其在新算法中的含义，与材料厚度区分。无需照搬 IPC，但不能靠关闭接触提速，也不能忽略明显穿透。
- **塑性：**关闭时保持弹性；开启后卸载仍有残留形变。塑性弯曲不能重复叠加两份弯曲刚度；`bending_model=null` 时全部弯曲及折痕参数不参与。历史不能随每次 VBD 迭代重复累计，重置时清空。
- **状态与稳定性：**可读取变形顶点，保留拓扑和表面顶点映射；支持重置初始位置、速度与塑性历史。无效状态应报错并停止，失败步不能作为成功结果发布；允许要求重置，不强制实现整步回滚。
- **实现与性能：**实际使用 VBD 求解，接入 Genesis 的场景、材料和状态接口。主要推进在 GPU 上完成，避免逐顶点 Python 循环和不必要的 CPU/GPU 同步。数值精度、迭代数和加速方式由实现选择并记录。

## 3. 简单验收与交付

1. 薄壳、体积两种模式都能构建、推进、读取状态和重置；不依赖本仓库前端。
2. 对材料参数做有针对性的对照，确认软硬、剪切、弯曲、质量和摩擦调节实际生效；塑性用加载—卸载检查残留形变、屈服和硬化效果。
3. 检查固定点、动态增删点力、地面接触、薄壳折叠自接触；检查关闭塑性、关闭弯曲和重置后的行为。
4. 提供可复现性能结果：记录网格规模、材料、时间步、子步、迭代数、精度、接触设置，以及预热后的平均耗时和 P95。初始化单独计时；按相同仿真时长和相近效果比较，不能只比较迭代次数或不同时间步的单步耗时。暂不设未经确认的 FPS 或加速倍数门槛。
5. 交付 Genesis 内的实现、最小运行示例、必要测试和简短的使用/性能说明。两类塑性、自碰撞等完整要求不能仅以 TODO 代替。

重点性能资产由 `genesis/assets/meshes/jelly/jelly.glb` 提供形状，目标机器为 RTX 4090。仿真统一使用其重网格化版本 `genesis/assets/meshes/jelly/remeshed.obj`，含 12,740 个表面顶点、25,476 个三角形。默认四面体化保留这些表面顶点，并生成内部顶点；实际规模记录在 JSON 中。折叠验收使用独立薄片场景，体积加载卸载使用同一个重网格化 jelly。
