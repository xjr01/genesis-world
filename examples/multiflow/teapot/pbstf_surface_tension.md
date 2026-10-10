# PBSTF 表面张力与吸收参数指南

本文说明 [pbstf_surface_tension.py](pbstf_surface_tension.py) 的 `teapot`、`sweep` 和 `mop` 场景。三者以 Y 轴向上，重力为 `(0.0, -9.8, 0.0)`；未特别注明的位置均为世界坐标，四元数顺序为 W-X-Y-Z。快速运行和抓取编辑另见 [使用指南](README.md)。

## 运行与配置入口

在仓库根目录运行：

```powershell
python examples/multiflow/teapot/pbstf_surface_tension.py --case teapot --vis
python examples/multiflow/teapot/pbstf_surface_tension.py --case sweep --vis
python examples/multiflow/teapot/pbstf_surface_tension.py --case mop --vis
```

| 参数 | 含义 |
| --- | --- |
| `--case` | 选择场景 |
| `--scale` | 粒子半径 `1/scale`、直径 `2/scale` |
| `--dt` | 覆盖仿真步长，单位秒 |
| `--steps` | 覆盖物理步数 |
| `--sponge-render-style` | `skeleton` 显示四面体边，`solid` 显示海绵表面 |
| `--sponge-grid-resolution NX NY NZ` | 海绵网格单元数量，越细越耗时 |
| `--vis` | 开启窗口 |
| `--record` | 开启窗口录像，结束时询问输出路径 |

`scale` 增大时粒子更细，数量、内存与运行时间通常近似按立方增长。茶壶要求 `scale >= 300`。运动计划采用仿真秒；改 `dt` 会改变各阶段步数，改 `steps` 只改变总时长。

几何、材料、轨迹和吸收值在 `case_settings()`、`_case_liquid_material()` 中配置；茶壶几何与操纵器设置在 [fluid_helper.py](fluid_helper.py)。

## 共享离散与材料

`build_scene()` 设置 `particle_size=2/scale`，每两步重建拓扑。世界几何和规定线速度应用长度系数 `s=1/15`，以原始缩放为 1 的 Franka 为基准；粒子精度参数相应增大 15 倍，保持粒子数量和相对分辨率。

固定步长下，密度柔度按 `1/s²` 缩放，表面张力柔度按 `s²` 缩放，距离和壁面黏附柔度不变。静止密度、黏度、摩擦、时间、重力、迭代数与离散数量保留配置值。

| 设置 | 茶壶 | 扫水与海绵 |
| --- | ---: | ---: |
| `scale` | 300 | 300 |
| 粒子直径 | `1/150` | `1/150` |
| `dt` | 0.01 | 0.01 |
| 重力 | `(0.0, -9.8, 0.0)` | `(0.0, -9.8, 0.0)` |
| 下界 | `(-4/3, -6.04186/15, -4/3)` | `(-0.4, -1/15, -4/15)` |
| 上界 | `(4/3, 1.0, 4/3)` | `(0.4, 4/15, 4/15)` |
| 求解迭代数 | 5 | 30 |
| 表面邻居容量 | 128 | 128 |
| 局部网格邻居容量 | 64 | 64 |
| 主成分分析法线 | 关闭 | 关闭 |
| 默认步数 | 10000 | 1000 |
| 默认仿真时长 | 100 秒 | 10 秒 |

| 液体材料参数 | 茶壶 | 扫水与海绵 |
| --- | ---: | ---: |
| 采样方式 | `regular` | `regular` |
| 静止密度 | 1000.0 | 1000.0 |
| 密度柔度 | 33750.0 | 33750.0 |
| 表面张力柔度 | `3/225` | `1/2000` |
| 表面距离柔度 | 40.0 | 40.0 |
| 内部距离柔度 | 180.0 | 180.0 |
| 表面黏度 | 0.2 | 0.5 |
| 内部黏度 | 0.05 | 0.5 |

壁面交互通过 `PBSTFStaticColliderOptions` 配置，同一碰撞体的设置在环境间共享。

| 碰撞体设置 | 茶壶 | 扫水与海绵 |
| --- | ---: | ---: |
| 黏附与摩擦 | 开启 | 开启 |
| 黏附柔度 | 20.0 | 50.0 |
| 摩擦 | 0.01 | 0.5 |

柔度越小，约束越强；黏度越大，相对运动阻尼越强。摩擦仅影响未吸收粒子，已吸收粒子跟随碰撞体运动。设置 `is_collider_adhesion_friction_enabled=True` 开启壁面效应。较低 `collider_adhesion_compliance` 增强润湿，也使脱离更困难；较高 `collider_friction` 消耗更多切向动能，0 保持滑动，1 消除相对切向运动。上述三项接口默认值为 `False`、10.0、0.1；关闭壁面效应仍保留几何碰撞。

## 茶壶几何、采样与轨迹

茶壶场景在变换后的 Utah 茶壶中填充液体，用移动网格静态碰撞体表示壶壁，并更新 KUKA 与 Shadow Hand 跟随规定抓取姿态。

`create_teapot_settings()` 提供以下值：

| 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `asset` | `meshes/utah_teapot_modified.obj` | 显示、空腔采样和碰撞网格来源 |
| `mesh_scale` | 0.15 | 同比例缩放网格和局部抓取位置 |
| `offset` | `(0.0, -3.79/15, 0.0)` | 初始世界位置 |
| `quat` | `(sqrt(0.5), 0.0, -sqrt(0.5), 0.0)` | 初始世界朝向 |
| `sdf_res` | 150 | 壶壁 SDF 分辨率 |
| `particles_seed` | `(0.0, -0.21, 0.0)` | 搜索液体空腔的种子点 |
| `particles_max_height` | `0.7/15` | 液体填充最高世界 Y 坐标 |
| `particles_vel` | `(0.0, 0.0, 0.0)` | 初始液体速度 |

空腔采样与壁面保持半个粒子直径的间隙。提高最高填充高度会增加液体，但种子必须连通到目标空腔；将种子移到空腔外可能选错区域或无法得到有效采样。

茶壶碰撞体索引为 0。`update_teapot_case()` 用 `teapot_pose()` 同时移动碰撞体和透明显示网格，再根据同一茶壶局部抓取姿态重新求解机械臂 IK。

茶壶绕穿过 `turning_axis_pos` 的世界 X 轴旋转；该位置由局部抓取点、网格缩放、初始位置和四元数计算。

| 仿真时间 | 运动 |
| --- | --- |
| 0–18 秒 | 以 1.5 度/秒转至 27 度 |
| 18–28 秒 | 保持 27 度 |
| 28–29.6 秒 | 以 5 度/秒回转至 19 度 |
| 29.6 秒后 | 保持 19 度 |

在 `teapot_pose()` 修改 `turning_rate_initial`、`stop_angle_initial`、`hold_end`、`turning_rate_final` 和 `stop_angle_final`。返回的线速度、角速度应与位姿一致，保证壁面向液体提供正确速度。

`TeapotManipulatorSettings` 管理 KUKA 资产、缩放、基座、末端 link、初始关节位置，Shadow Hand 资产、缩放、安装变换和关节位置，局部 `grasp_pos/grasp_quat`、IK 工具中心 `tool_center_point` 及相机位置。KUKA 缩放为 0.8，Shadow Hand 为 `14/15`，以保持统一缩放后的原比例。

用 [抓取编辑器](pbstf_teapot_grasp_editor.py) 编写或检查抓取变换。此 demo 关闭机械臂和手的碰撞，液体边界由 PBSTF 网格静态碰撞体负责。

## 扫水与海绵

两者共用桌面、液体区域、静止尺寸和擦拭轨迹。`sweep` 用解析 `PBSTFBoxStaticColliderOptions` 推水；`mop` 用 PBD 海绵，将吸收碰撞体绑定到变形四面体表面。液体将海绵视为单向边界。

| `WipeSettings` 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `collider_idx` | 1 | 运动工具索引；桌面为 0 |
| `collider_entity_name` | `sponge` / `sweep_collider` | 工具显示实体 |
| `collider_lower` | `(-0.04, 0.02/15, -0.08)` | 静止局部下角 |
| `collider_upper` | `(0.04, 0.85/15, 0.08)` | 静止局部上角 |
| `table_entity_name` | `wipe_table` | 显示桌面，也是 PBD 碰撞桌面 |
| `table_pos` | `(0.0, -1/60, 0.0)` | 桌面实体世界位置 |
| `table_size` | `(0.8, 1/30, 8/15)` | 桌面尺寸，顶面 Y=0 |
| `liquid_lower` | `(-1/6, 1/300, -7/150)` | 液体初始下角 |
| `liquid_upper` | `(-1/30, 7/300, 7/150)` | 液体初始上角 |
| `start_pos` | `(-7/30, 0.0, 0.0)` | 行程开始位置 |
| `end_pos` | `(7/30, 0.0, 0.0)` | 行程结束位置 |
| `quat` | `(1.0, 0.0, 0.0, 0.0)` | 工具与桌面朝向 |
| `settle_time` | 1 秒 | 初始液体沉降时长 |
| `wipe_time` | 5 秒 | 线性擦拭时长 |

`wipe_pose()` 先保持起点，沉降后线性运动到终点，之后保持。默认步长下沉降 100 步，第 600 步结束行程。`update_wipe_case()` 同步解析工具和显示实体；`update_mop_case()` 协调 Franka、海绵动力学、变形碰撞几何和同一轨迹。

### 夹爪与海绵阶段

Franka 使用 `urdf/panda_bullet/panda.urdf`，原始缩放为 1。七个手臂关节由 IK 计算，两个手指关节采用规定位置。工具中心指向海绵顶部中心，手指开口沿世界 X，夹爪位于液体区域上方。

| 仿真时间 | 行为 |
| --- | --- |
| 0–1 秒 | 手指从 0.04 闭合至 `0.4/15`，海绵因重力与接触变形 |
| 1–6 秒 | IK 带动手沿擦拭轨迹移动，海绵持续参与动力学 |
| 6 秒后 | 手保持终点，海绵继续弹性和接触响应 |

海绵使用规则四面体网格，材料为 `PBD.Elastic(rho=30, stretch_relaxation=0.1, volume_relaxation=0.15)`，拉伸与体积柔度都为 0。`PBDUnifiedOptions` 每子步迭代 30 次，`constraint_acceleration=0.85`。连续位置迭代的外推加速形状恢复；更高加速度可能减少迭代需求，也可能在接触变化时过冲。每子步从预测位置重置迭代历史，粒子速度保留常规 PBD 更新。

每轮从一个位置状态读取，累加拉伸和体积修正到 `dpos`，应用总修正，再将顶点和表面三角形投影到刚体外。布料在同一循环使用拉伸和弯曲。规定刚体边界不接收反作用；海绵通过接触和弹性响应运动。零柔度时，统一密度缩放在按质量归一化的弹性修正中抵消，因此该规定边界场景主要由约束参数和迭代预算控制重力下变形。

每次硬接触最多 100 轮，所有环境解除穿插后退出。初始夹爪重叠比后续小幅运动更费时。三角形接触向分离半空间投影，已满足条件的顶点保持；有符号体积约束可恢复翻转的自由四面体。全部顶点固定而体积无效、非有限状态和未解开的接触会报错。液体保留 30 次迭代、`dt=0.01` 及上述材料参数。

Panda 加载所有 link 的碰撞几何；固定显示桌面也参与 PBD 碰撞查询。PBD 管理单向碰撞投影，液体通过两个 PBSTF 碰撞体接触解析桌面与吸收海绵。

海绵显示三角形与 PBD 体积边界相同。PBSTF 订阅 PBD 几何变化，在液体子步前同步边界顶点与嵌入吸收体素；运行查询直接使用当前三角形。局部 `collider_lower/upper` 定义静止材料格点，变形四面体边界定义实际接触与吸收表面。

## 吸收参数

默认海绵碰撞体在 `case_settings()` 中配置：

```python
wipe_collider = gs.options.PBSTFAbsorbentBoxStaticColliderOptions(
    pos=wipe.start_pos,
    quat=wipe.quat,
    is_collider_adhesion_friction_enabled=True,
    collider_adhesion_compliance=50.0,
    collider_friction=0.5,
    lower=wipe.collider_lower,
    upper=wipe.collider_upper,
    absorption_rate=4000.0,
    absorption_capacity_fraction=1.0,
    pbd_entity_name=wipe.collider_entity_name,
)
```

`lower/upper` 定义局部静止材料网格，`pos/quat` 将其放到世界；`pbd_entity_name` 绑定使用 `PBDUnifiedOptions` 的体积 PBD 实体。`sdf_res` 不设置时，对持续变形的形状保留精确三角形查询。

### 吸收速率 `absorption_rate`

同时控制捕获吞吐与向内移动速度。每个碰撞体、每个环境、每子步增加 `absorption_rate * dt` 捕获额度，每绑定一个新粒子消耗一个额度。有持续接触且容量充足时，每仿真秒最多约捕获 `absorption_rate` 个粒子。空闲额度最多保留约一个额外粒子，避免干海绵积攒大批瞬时吸收。

目标体素的曼哈顿图距离为 `d` 时：

```text
beta = 1 - exp(-absorption_rate * dt / (d + 1))
progress_new = progress + beta * (1 - progress)
local_pos_new = local_pos + beta * (target_local_pos - local_pos)
```

最近体素时间常数为 `1/absorption_rate`，距离 `d` 时为 `(d+1)/absorption_rate`。例如速率 2000 时，持续接触每秒约捕获 2000 个粒子；距离 0、1、3 的移动时间常数分别为 0.0005、0.001、0.002 秒。上面的海绵 case 使用速率 4000。

提高速率会增加吞吐并加快、加剧向内移动；降低速率使移动更平滑但吸收更慢。值必须大于 0。

### 容量比例 `absorption_capacity_fraction`

表示静止盒体积中可用于容纳液体的比例，范围为 `(0,1]`。整数粒子容量计算为：

```text
particle_rest_volume = particle_mass / liquid_rest_density
total_capacity = floor(absorption_capacity_fraction * box_volume / particle_rest_volume)
```

默认盒体积为 `0.08 * (0.83/15) * 0.16 = 0.0007082667`，比例 1 使用完整体积；粒子槽数取决于质量校准与分辨率。提高比例增加容量、延迟饱和；降低比例提早饱和，随后工具会推开更多水。不需要吸收时使用普通 `PBSTFBoxStaticColliderOptions`，如 `sweep`。

### 体素、饱和与状态恢复

```text
grid_res = ceil((upper - lower) / support_radius)
```

默认 `scale=300`、粒子直径 `1/150`、支持半径 0.02，海绵使用 `(4,3,8)` 网格。整数槽位均匀分配到 96 个体素，保留精确总容量。

每个静止体素中心通过重心坐标嵌入 PBD 四面体，同步时由当前顶点求出位置，使目标跟随变形。容量仍由静止体积计算，变形不改变单体素或总槽数。

粒子接触海绵时，选择最近变形体素中心，再在不变的六邻域材料图中广度优先搜索：接触体素距离 0，相邻面体素距离 1，再逐层扩展。同层按变形中心物理距离及稳定体素索引排序，首个有空槽的候选捕获粒子。

已捕获粒子保持活跃和可见，随海绵运动并收敛到当前嵌入目标，不再参加密度、表面、距离、黏度、黏附和摩擦处理。每个粒子继续占用保留的材料体素；显式设置其位置、速度或活跃状态会解除吸收绑定。

场景保存恢复包含绑定、捕获额度、体素距离、进度、局部目标、碰撞体动态位姿、变形表面、嵌入体素位置、搜索顺序和 SDF 激活状态；SDF 由保存表面恢复，湿度由恢复的绑定进度重建。

改变 `scale` 也改变粒子尺寸、支持半径、体素数量、图距离、校准粒子体积和整数容量分布。体积吞吐为 `absorption_rate * particle_rest_volume`；分辨率改变后如需保持相同每秒体积吞吐，速率应按粒子静止体积反比调整。改变 `dt` 时连续时间含义保持，但较小步长能更细地解析接触与移动。

## 查询湿度

场景构建后：

```python
from examples.multiflow.teapot.pbstf_surface_tension import CASE_MOP, build_scene, case_settings, get_wipe_settings

settings = case_settings(CASE_MOP)
wipe = get_wipe_settings(settings)
scene, _ = build_scene(case=CASE_MOP)

wetness = scene.pbstf_solver.get_static_collider_wetness(collider_idx=wipe.collider_idx)
```

每个值在 `[0,1]`，为该体素捕获进度之和除以槽容量后截断；坐标轴按局部 X、Y、Z 从 `lower` 到 `upper` 排列。单环境返回 `[nx,ny,nz]`，批量环境返回 `[B,nx,ny,nz]`；传 `envs_idx` 可选择环境。桌面或普通扫水盒没有湿度，查询会报错。

实时检查时在 `scene.step()` 后查询。返回张量是新值，可用于显示或日志，不会修改内部湿度字段。
