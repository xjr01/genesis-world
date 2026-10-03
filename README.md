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

仓库目前采用“Genesis 核心扩展 + 稳定接入层 + 标准化任务样例”的三层结构。

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

## 2. 稳定接入层

```text
genesis/integrations/
├── particle_fluid.py
└── granular_fluid.py
```

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
