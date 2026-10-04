# UniRoboSim-Genesis integration guide

## Supported integration surface

UniRoboSim-Genesis should treat this repository as one pinned engine distribution. Its provider, session and world
lifecycle can follow the standard Genesis backend. Fork capabilities are discovered at startup and compiled through
the public interfaces below.

- Particle fluid: call `genesis.integrations.create_particle_fluid_setup`, select the solver and retain its typed
  options.
- Granular fluid: call `genesis.integrations.create_granular_fluid_setup` to validate the domain, particle sizes and
  densities.
- Solver access: use `scene.pbstf_solver`, `scene.dem_solver` and `scene.flip_solver` after capability discovery.
- Moving fluid boundary: register `StaticColliderLinkSynchronizer` with `scene.register_pre_step_callback` after
  `scene.build()`.
- Normalized task: convert task and asset specifications into a config for
  `examples.multiphysics.<scenario>.build_scene`.
- Task control: advance a named runtime with its scenario controller and public entity methods.
- State: store and restore the complete engine state through `scene.get_state()` and `scene.reset()`.

Adapter code must not inspect `scene.sim._solvers`, assume solver-list ordering, access Quadrants fields, or reproduce a
scenario's contact law. These details belong to this engine profile.

## Five-scenario adapter requirements

The two modules under `genesis.integrations` are reusable solver-family helpers. They do not correspond one-to-one to
the five task scenarios. The stable adapter surface is the combination of `genesis.integrations` and
`examples.multiphysics`.

### Coffee pouring, stirring and cleanup

- Genesis public capabilities: `Scene`, rigid entities, PBD entities, URDF/mesh loading, robot joint control and public
  entity state APIs.
- Fork capabilities: PBSTF options/material/solver, concentration transport, static fluid colliders, moving-boundary
  synchronization and typed coffee-task configuration.
- Adapter work: probe PBSTF; bind each cup mesh and cavity representation to the same asset revision and transform;
  register moving cup/rod boundary synchronization; preserve liquid, concentration and absorption state; expose the
  stepwise task controller.
- Stable task API: construct `CoffeeWaterScenarioConfig`, call
  `examples.multiphysics.coffee_water.build_scene()`, then advance the returned `CoffeeWaterRuntime` with
  `CoffeeWaterController.step()`. The executable `examples.pbstf_coffee_water` forwards to this package API.

### Table wiping

- Genesis public capabilities: `Scene`, rigid entities, PBD deformable entities and public entity controls.
- Fork capabilities: PBSTF surface tension, liquid absorption/wetness state and the normalized wiping scenario.
- Adapter work: probe PBSTF; map table, liquid and sponge assets; drive the sponge controller; preserve absorption and
  liquid state across checkpoint/reset.

### Coupled litter scooping

- Genesis public capabilities: `Scene`, rigid tools and public entity state/control APIs.
- Fork capabilities: DEM, FLIP, two-way sand-water coupling, absorbed-water ratio, full particle/grid state and
  `create_granular_fluid_setup`.
- Adapter work: probe DEM and FLIP together; validate a shared domain and compatible resolution; drive the shovel;
  preserve all coupled state. FLIP currently requires one Scene per independently reset environment.

### Garment folding

- Genesis public capabilities: FEM cloth, IPC coupling, rigid jaws and public rigid pose control.
- Fork capabilities: normalized garment assets, semantic landmarks, asset-local trajectories, named runtime and
  stepwise controller.
- Adapter work: verify the IPC dependency; resolve garment mesh/texture; pass asset pose and scale; drive jaw targets;
  preserve FEM/IPC state.

### Butter spreading

- Genesis public capabilities: MPM, rigid blade entities and public particle getters/setters.
- Fork capabilities: per-material MPM particle size, porous bread, Herschel-Bulkley butter, the butter contact callback,
  named runtime and controller.
- Adapter work: probe the MPM extensions; map bread, butter and blade assets; pass material/contact parameters; retain
  the callback registered by `build_scene()`; preserve MPM state.

### Requirements shared by every scenario

1. Probe every required option, material, solver property and state type before advertising the capability.
2. Pin the engine commit in the backend profile.
3. Keep solver, material, asset, task and contact parameter ownership separate.
4. Store named runtime handles instead of solver-list positions or private particle offsets.
5. Use public batched getters/setters and preserve the environment dimension.
6. Advance through `scene.step()` so registered boundary/contact callbacks run automatically.
7. Rebuild after solver, particle-size, boundary-geometry or asset-scale changes.
8. Include every solver and coupling field required by the scenario in checkpoint/reset.

## Capability discovery

Probe the required symbols once when the backend starts. A profile should only advertise a capability when all of its
construction and state symbols are available. Pin the engine Git commit in the UniRoboSim-Genesis environment metadata
so two installations cannot silently expose different physics under the same backend name.

The current capability groups are:

- particle fluid: IPBF, IPBSTF and PBSTF;
- granular fluid: DEM and FLIP with sand-water coupling;
- deformable manipulation: FEM/IPC garment folding;
- heterogeneous MPM resolution: per-material particle size;
- MPM task materials: porous bread and Herschel-Bulkley butter.

## Direct solver construction

Use a typed integration helper when UniRoboSim owns scene construction:

```python
import genesis as gs

from genesis.integrations import ParticleFluidProperties, ParticleFluidSolver, create_particle_fluid_setup

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

The portable layer owns values with the same physical meaning across backends. Genesis-specific numerical controls
remain typed Genesis options. Asset mesh paths, scale, poses, fill regions and task trajectories remain asset or task
data.

## Scenario construction

Use a normalized scenario when the task needs calibrated assets, controls or a scenario-specific contact model:

```python
from dataclasses import replace

import genesis as gs

from examples.multiphysics.butter_spreading import (
    ButterSpreadingController,
    ButterSpreadingScenarioConfig,
    build_scene,
)

gs.init(backend=gs.gpu, precision="32")
config = ButterSpreadingScenarioConfig()
config = replace(
    config,
    assets=replace(
        config.assets,
        bread_center=(0.05, 0.0, 0.018),
        bread_size=(0.16, 0.11, 0.018),
    ),
)
runtime = build_scene(config, show_viewer=False)
controller = ButterSpreadingController(config.task, config.solver.dt)
controller.step(runtime)
```

`build_scene()` returns named handles instead of solver indices. Asset-relative task origins and semantic landmarks
move with the configured asset. The controller issues task commands; registered scene callbacks apply boundary and
contact updates whenever the adapter calls `scene.step()`.

## Parameter ownership

- `SolverConfig` owns global discretization, time step, bounds, iterations and numerical capacities.
- `MaterialConfig` owns density, stiffness, friction, viscosity, yield and other constitutive properties.
- `Assets` owns files, primitive dimensions, initial transforms, semantic landmarks and visualization geometry.
- `TaskConfig` owns controller phases, targets, clearances, timing and termination conditions.
- A dedicated contact config owns calibrated contact-law values when the scenario needs a law beyond material contact.

Computed values stay computed. For example, an MPM particle reference volume is derived from particle size inside the
engine. A task may express knife height as clearance above an asset surface, with its world-space pose derived during
scene construction.

## Rebuild and runtime updates

Rebuild the scene after changes to solver type, particle size, boundary geometry, asset scale or build-time coupling.
Entity pose commands and controller targets may be updated at runtime through public entity methods. A moving fluid
boundary uses a registered synchronizer so the adapter still advances the scene through the standard `scene.step()`
call.

FLIP currently supports one environment per scene. Use one scene per independently reset FLIP environment. Other
capabilities must preserve the leading environment dimension exposed by Genesis public getters and setters.

## Integration acceptance

Before updating the commit pinned by UniRoboSim-Genesis:

1. Run the configuration and adapter tests for the changed capability.
2. Run a one-step headless smoke test for each changed normalized scenario.
3. Run the complete default task trajectory on its supported GPU backend and record the command and result.
4. Verify `scene.get_state()` and `scene.reset()` for any solver whose state layout changed.
5. Confirm asset pose and scale changes propagate to task targets without editing solver configuration.

A scenario is structurally integrated after steps 1, 2 and 5. Mark it physically validated only after its complete
trajectory and relevant state/reset checks pass.
