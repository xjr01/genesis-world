# UniRoboSim Genesis 多物理仿真引擎

这个仓库是面向具身任务的 Genesis 多物理仿真后端，供 UniRoboSim-Genesis 调用。

当前整体调用关系如下：

```text
FastSim
  -> UniRoboSim
      -> UniRoboSim-Genesis
          -> 本仓库
              -> Genesis / Quadrants
```

仓库目前采用“Genesis 核心扩展 + 通用物理接入工具 + 标准化任务接入”的三层结构。

```text
UniRoboSim / UniRoboSim-Genesis
              |
              v
genesis.integrations
或 examples.multiphysics
              |
              v
Genesis Scene 公共接口
              |
              v
Simulator / Solver / Entity / Material
              |
              v
Quadrants 计算后端
```

## 1. Genesis 核心扩展层

新增物理能力目前直接集成在 Genesis 内部：

```text
genesis/
├── options/
│   └── solvers.py
└── engine/
    ├── scene.py
    ├── simulator.py
    ├── solvers/
    │   ├── ipbf_solver.py
    │   ├── ipbstf/
    │   ├── dem_solver.py
    │   ├── flip_solver.py
    │   └── mpm_solver.py
    ├── entities/
    │   ├── dem_entity.py
    │   ├── flip_entity.py
    │   └── mpm_entity.py
    └── materials/
        ├── DEM/
        ├── FLIP/
        └── MPM/
            └── butter.py
```

当前核心层包含：

- IPBF、IPBSTF、PBSTF 粒子流体；
- DEM 颗粒；
- FLIP 流体；
- DEM/FLIP 沙水耦合；
- 支持不同材料使用不同粒径的 MPM；
- 多孔面包材料和 Herschel-Bulkley 黄油材料；
- 新增 solver 对应的状态保存和 reset 支持。

这些 solver 当前仍然由 `Simulator` 显式创建，还没有拆成独立的外部插件。

## 2. 通用物理接入工具层

```text
genesis/integrations/
├── particle_fluid.py
└── granular_fluid.py
```

这一层按可复用的 solver 家族划分，不按任务数量划分。两个模块覆盖五个新增 solver：

- `particle_fluid.py`：IPBF、IPBSTF、PBSTF；
- `granular_fluid.py`：DEM、FLIP 以及二者的耦合配置。

这一层供 UniRoboSim-Genesis 使用，主要负责：

- 把通用物理参数转换为 Genesis solver options 和 material；
- 校验粒径、密度、仿真区域等参数；
- 提供移动流体边界同步器；
- 向上层提供明确、类型化的构建结果；
- 避免 adapter 访问 solver 列表、Quadrants 字段或内部粒子索引。

粒子流体的典型调用方式如下：

```python
import genesis as gs

from genesis.integrations import (
    ParticleFluidProperties,
    ParticleFluidSolver,
    create_particle_fluid_setup,
)

fluid = create_particle_fluid_setup(
    ParticleFluidProperties(
        density=1000.0,
        particle_size=0.008,
    ),
    ParticleFluidSolver.PBSTF,
)

scene = gs.Scene(
    pbstf_options=fluid.solver_options,
)
liquid = scene.add_entity(
    morph=particle_morph,
    material=fluid.material,
)
```

UniRoboSim-Genesis 只需要保存跨后端含义相同的通用参数。Genesis 特有的数值参数继续使用 Genesis 自己的
typed options。

## 3. 标准化任务样例层

```text
examples/multiphysics/
├── coffee_water/
├── table_wiping/
├── litter_scoop/
├── garment_folding/
└── butter_spreading/
```

目前包括：

- `coffee_water`：PBSTF 倒咖啡、搅拌和清理；
- `table_wiping`：PBSTF 液体吸收和可变形海绵擦拭；
- `litter_scoop`：DEM 猫砂和 FLIP 水的耦合铲取；
- `garment_folding`：FEM/IPC 衣物抓取和折叠；
- `butter_spreading`：MPM 面包、黄油和刀具接触。

每个标准化任务按照下面的形式组织：

```text
config.py   参数定义
scene.py    创建场景并返回命名实体
task.py     逐步执行任务控制
run.py      命令行运行入口
assets/     任务资产
```

每个任务的参数分为：

- `SolverConfig`：时间步、粒径、仿真边界、网格密度和迭代次数；
- `MaterialConfig`：密度、刚度、摩擦、黏度、屈服应力等材料属性；
- `Assets`：模型路径、尺寸、位置、缩放和语义点；
- `TaskConfig`：任务轨迹、阶段时间、抓取间距和任务目标；
- `ContactConfig`：任务需要的额外接触模型参数。

标准任务的调用方式如下：

```python
import genesis as gs

from examples.multiphysics.butter_spreading import (
    ButterSpreadingController,
    ButterSpreadingScenarioConfig,
    build_scene,
)

gs.init(backend=gs.gpu, precision="32")

config = ButterSpreadingScenarioConfig()
runtime = build_scene(config, show_viewer=False)
controller = ButterSpreadingController(config.task, config.solver.dt)

controller.step(runtime)
```

`build_scene()` 返回命名 runtime。runtime 中保存 `scene` 和任务需要的实体句柄，例如 `bread`、`butter`、
`blade`、`garment` 和 `jaws`。上层不需要猜测实体索引或 solver 顺序。

资产相关的任务轨迹使用资产局部坐标。修改杯子、面包或衣物的位置、尺寸和缩放后，依赖这些资产的任务目标
会根据资产配置重新计算，不需要同时修改 solver 参数。

## 4. Scene 回调机制

需要在每次仿真步自动执行的行为通过下面的公共接口注册：

```python
scene.register_pre_step_callback(callback)
```

目前主要用于：

- 同步移动刚体和流体边界；
- 执行黄油与面包、刀具之间的额外接触计算。

UniRoboSim-Genesis 正常调用：

```python
scene.step()
```

即可得到完整行为，不需要额外调用场景内部的接触函数。

## 5. 研究参考代码

```text
examples/multiflow/
examples/sand_water_coupling/
```

这些目录保留完整实验、参数研究、渲染和资产校准脚本。它们用于开发、复现和参数比较，不是稳定的 adapter
接口。正式对接应使用 `genesis.integrations` 或 `examples.multiphysics`。

## 6. UniRoboSim-Genesis 的接入边界

UniRoboSim-Genesis 应使用：

- `genesis.integrations` 中的构建和校验函数；
- `examples.multiphysics` 中的标准化场景；
- `scene.pbstf_solver`、`scene.dem_solver`、`scene.flip_solver` 等公开 solver 属性；
- `Scene`、`Entity` 的公开状态和控制接口；
- `scene.get_state()`、`scene.reset()` 和 `scene.step()`。

UniRoboSim-Genesis 不应依赖：

- `scene.sim._solvers`；
- solver 在内部列表中的顺序；
- Quadrants 内部字段；
- 私有粒子起始索引；
- 示例内部的接触实现细节。

FLIP 当前每个 Scene 支持一个独立环境。需要独立 reset 的多个 FLIP 环境，应分别创建 Scene。

### 五个场景的能力来源和对接要求

这里的“Genesis 原有公共能力”是指不依赖本 fork 自研 solver 就已经存在的公开接口；“本 fork 新增能力”
包括自研 solver、材料、耦合、场景接口和为稳定对接增加的扩展。

| 场景 | Genesis 原有公共能力 | 本 fork 新增能力 | UniRoboSim-Genesis 需要对接 |
|---|---|---|---|
| 倒咖啡、搅拌和清理 | `Scene`、Rigid、PBD、机器人关节控制、Mesh/URDF、Entity 状态接口 | PBSTF solver/material/options、浓度与混合、静态流体边界、移动边界同步、咖啡任务配置 | PBSTF 能力探测；杯体与 cavity 资产绑定；移动杯和搅拌棒边界同步；液体和吸收状态保存/reset；任务控制器 |
| 擦桌子 | `Scene`、Rigid、PBD 可变形实体、公开实体控制接口 | PBSTF 表面张力、液体吸收/湿润状态、擦拭场景配置和控制器 | PBSTF 能力探测；桌面、液体和海绵资产映射；海绵控制；吸收状态保存/reset |
| 铲猫砂 | `Scene`、Rigid 工具、公开实体控制和状态接口 | DEM、FLIP、DEM/FLIP 双向耦合、吸水比例、颗粒和网格状态、颗粒流体配置工具 | DEM/FLIP 联合能力探测；统一 domain 和粒径校验；铲子控制；完整耦合状态保存/reset；每个独立环境一个 Scene |
| 叠衣服 | `Scene`、FEM Cloth、IPC coupler、Rigid 夹爪和公开位姿控制 | 标准化衣物资产、语义点、局部坐标轨迹、任务 runtime 和 controller | IPC 依赖检查；衣物 mesh/texture 映射；资产位置和缩放传递；夹爪控制；FEM/IPC 状态保存/reset |
| 抹黄油 | `Scene`、MPM、Rigid 刀具、粒子实体公开 getter/setter | MPM 每材料独立粒径、多孔面包、Herschel-Bulkley 黄油、黄油接触 callback、任务 runtime 和 controller | MPM 扩展能力探测；面包/黄油/刀具资产映射；材料参数传递；callback 自动注册；MPM 状态保存/reset |

五个场景对应五个任务入口，但不需要五个 `genesis.integrations` 文件。`genesis.integrations` 只承载多个任务
可以共享的 solver 构建逻辑；具体资产、轨迹和任务接触逻辑由 `examples.multiphysics` 中的场景包承载。

### 所有场景共同的 adapter 要求

1. 启动时探测所需 options、materials、solver 属性和状态类型，只发布完整可用的能力。
2. 固定本仓库 commit，避免相同 backend 名称对应不同物理实现。
3. 将 solver、material、asset、task 和 contact 参数分别映射，禁止把资产尺寸写进 solver 配置。
4. 通过命名 runtime 保存实体句柄，不保存 solver 列表下标或私有粒子索引。
5. 使用公开批量 getter/setter，保留环境维度。
6. 通过 `scene.step()` 推进；移动边界和额外接触由已注册 callback 自动执行。
7. solver 类型、粒径、边界几何和资产缩放变化时重建 Scene；运行时位姿通过 Entity API 更新。
8. checkpoint/reset 必须覆盖场景依赖的完整 solver 状态和耦合状态。

### 当前接口完整度

- 五个场景都在各自的 `examples.multiphysics.<scenario>` 包内提供 `build_scene()`、命名 runtime 和
  `Controller.step()`。
- 咖啡场景的稳定入口是 `examples.multiphysics.coffee_water`；原来的 `examples.pbstf_coffee_water` 仅保留为
  命令行和旧导入路径的薄入口。

## 7. 当前状态

当前架构已经能够支持 UniRoboSim-Genesis 以较小改动接入新增流体、颗粒、多物理耦合和标准任务。

当前主要架构债务是新增 solver 仍然直接放在 Genesis core 中。把 solver 完全拆成外部插件属于后续阶段，
不影响目前通过公开 Scene 接口和稳定接入层完成 UniRoboSim-Genesis 对接。

进一步说明：

- [UniRoboSim-Genesis 详细对接指南](./docs/integration/unirobosim_genesis.md)
- [多物理场景接口契约](./docs/architecture/multiphysics_scenario_contract.md)
- [粒子流体最小接入说明](./docs/architecture/minimal_particle_fluid_integration.md)
- [标准化场景目录](./examples/multiphysics/README.md)
- [后续样例迁移规则](./examples/multiphysics/AGENTS.md)
