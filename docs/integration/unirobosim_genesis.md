# UniRoboSim-Genesis integration guide

## Supported integration surface

### Source checkout deployment

The adapter-facing scenarios live in `examples.multiphysics`. The current setuptools configuration packages only
`genesis` and `genesis.*`; these scenarios and their scenario-local assets are available through the source checkout,
not through an ordinary engine wheel. Keep the pinned checkout available to the adapter, install the engine with
`python -m pip install -e /path/to/genesis-world`, and add that checkout to the adapter process's `PYTHONPATH` when
launching outside it. Verify both `genesis.__file__` and `examples.multiphysics.__file__` resolve into that checkout.
This is especially important with several editable checkouts on one machine.

The engine and scenario checkout must use the same commit. Use the fork's dependency declaration, a compatible GPU
PyTorch installation, and the optional `pyuipc` dependency for garment folding. The coffee client also imports SciPy.
Scene527 requires its separate portable asset bundle; a lightweight garment scene uses the bundled assets instead.
Engine capability availability and normalized scenario batching are distinct: the current five builders create one
unbatched scene and expose no `n_envs` construction argument. Use one built runtime per task environment until a
scenario-specific batched construction path is implemented and validated. FLIP additionally has a solver-level
single-environment restriction.

### Exact scenario calls

Import from the scenario package, rather than its implementation, research script or command-line module. All five
packages export their scenario config, named runtime, controller, controller-state type and `build_scene`. The API
differences below are deliberate parts of the current callable surface.

| Scenario | Controller construction | Camera construction | Controller snapshot | Native `control_dt` |
|---|---|---|---|---:|
| Coffee | `CoffeeWaterController(config.task)` | `build_scene(config, is_recording=True)` | `controller.get_state(runtime)` | 0.002 s |
| Table | `TableWipingController(config.task)` | `build_scene(config, add_camera=True)` | `controller.get_state()` | 0.002 s |
| Litter | `LitterScoopController(config.task)` | `build_scene(config, add_camera=True)` | `controller.get_state()` | 1/60 s |
| Garment | `GarmentFoldingController(config.task)` | `build_scene(config, add_camera=True)` | `controller.get_state()` | 0.02 s lightweight; 1/120 s Scene527 |
| Butter | `ButterSpreadingController(config.task, config.solver.dt)` | `build_scene(config, add_camera=True)` | `controller.get_state()` | 0.000035 s |

Coffee uses `is_viewer_shown` for its viewer keyword; the other builders use `show_viewer`. Creating a camera leaves
recording under the client's control. Omit camera/viewer keywords for headless adapter construction.

The calibrated world frames also differ: coffee and table are Y-up, while litter, garment and butter are Z-up. Read
the configured gravity and asset/task frame together; preserve that frame when constructing the calibrated scene.
An adapter publishing a common world frame must consistently transform observations and commands at its boundary,
including poses, velocities, gravity and task targets. Changing only a mesh rotation leaves physics and task frames
inconsistent. Positions and durations use metres and seconds; public Genesis quaternions use `(w, x, y, z)`.

`controller.step(runtime)` issues the task command AND advances physics by `runtime.control_dt`. Do not call an
additional `scene.step()` after it. `ScenarioFrameRunner.step()` likewise performs all of that frame's controller
steps and returns their count. Coffee's native controller call contains two scene steps at 0.001 s each; schedule
with `runtime.control_dt`, rather than assuming it equals `scene.dt`. Coffee returns a `MotionTarget` from its
controller step; the other controllers return
`None`. An adapter should read observations through public runtime entity/solver handles and keep step return values
separate from its own observation schema. None of the controllers implement an RL `step(action)`/reward/done API.

Public observation types are explicit at the getter boundary: particle/robot getters used here return Torch tensors,
while DEM's scripted obstacle pose getters return NumPy arrays. Normalize once in the adapter's observation layer;
the lifecycle checker uses `torch.as_tensor` and cloned snapshots to support both and avoid aliasing saved observations.

The adapter owns its episode termination. The calibrated clients use these native-step limits or task predicates:

| Scenario | Calibrated termination |
|---|---|
| Coffee | Both returned target phases have `.name == "REST"`; `config.task.motion_end` is the upper time bound |
| Table | `controller.step_index >= config.task.steps` |
| Litter | `controller.step_index >= config.task.total_steps` |
| Garment | `controller.step_index >= runtime.default_steps`; Scene527 rejects further native trajectory steps |
| Butter | `controller.step_index >= round(config.task.lift_end_time / runtime.control_dt) + 1` |

Use restored controller indices for resumed tasks, not the Scene's restore-local clock. Check termination after each
native controller step when an exact horizon is required. A fixed-rate frame can contain several native steps;
termination checked only after a full frame is quantized to that frame. Stop invoking a completed Scene527 controller.

Two operating modes are supported: use the supplied controller to reproduce the calibrated example, or own task
commands in UniRoboSim and call `scene.step()` directly. In the latter mode, retain registered contact/boundary
callbacks and implement the task's tool commands in the adapter. A bare scene step applies physics and callbacks;
it does not execute the example's scripted controller. At 60 FPS, the optional runner distributes native calls;
for the lightweight garment some frames execute zero calls, and butter executes roughly 476 calls per frame.
The frame runner provides timing, rather than a real-time performance guarantee.

### Paired checkpoint lifecycle

For example, after constructing a table runtime and controller:

```python
from examples.multiphysics import ScenarioFrameRunner

controller.reset(runtime)
runner = ScenarioFrameRunner(controller, runtime, fps=60.0)
runner.step()
scene_state = runtime.scene.get_state()
controller_state = controller.get_state()  # Coffee uses get_state(runtime).
frame_state = runner.get_state()

runner.step()
runtime.scene.restore(scene_state)
controller.set_state(runtime, controller_state)
runner.set_state(frame_state)
runner.step()

runtime.scene.reset()
controller.reset(runtime)
runner.reset()
```

Restore only into a runtime built from the same config, asset revisions and engine/backend profile. Capture a paired
checkpoint after a completed controller/frame step. Serialized checkpoints also depend on backend state support;
changing the solver layout or shovel representation requires regeneration. The coffee snapshot includes its runtime
motion state and joint target; the other snapshot signatures remain unchanged.

Run `python scripts/validate_multiphysics_checkpoint.py <scenario> --record` from the checkout to check public particle
position/velocity, available tool/robot state, controller phase restoration, continuation and bare initial reset.
Its seven rendered diagnostic frames use a 5 FPS display clock and cover initial, checkpoint, advance, restore,
continuation, reset and restart; their playback duration is separate from simulated time. Detailed absorption,
concentration, grid and native IPC state requirements also have solver-specific tests. Acceptance evidence and limits
are tracked in [the validation record](full_trajectory_validation.md).

### Engine boundary

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
- Optional frame timing: an adapter that needs a uniform 60 FPS clock may wrap the unchanged controller API in
  `examples.multiphysics.ScenarioFrameRunner` while retaining each scenario's calibrated native step.
- State: store checkpoints with `scene.get_state()`, resume them with `scene.restore(state)`, and restart with
  `scene.reset()`.
- Task reset: call `runtime.scene.reset()` first, then `controller.reset(runtime)` so controller phase state and scripted
  tool handles are rewound together with the physics state.

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
  preserve all coupled state. FLIP currently requires one Scene per independently reset environment. DEM checkpoints
  include the scripted shovel obstacle's position, orientation, linear velocity and angular velocity;
  `LitterScoopController.reset(runtime)` separately restores its configured task-origin pose on restart.

### Garment folding

- Genesis public capabilities: FEM cloth, IPC coupling, independent cloth self-friction and entity-pair friction,
  rigid jaws, articulated robots and public rigid pose control.
- Fork capabilities: normalized garment assets, semantic landmarks, portable Scene527 asset roots, asset-local robot
  trajectories, named runtime and a stepwise controller shared by the lightweight and high-fidelity profiles.
- Adapter work: verify the IPC dependency; select `GarmentFoldingScenarioConfig()` or
  `create_scene527_config(asset_root)`; call the same `build_scene` and `GarmentFoldingController.step` entry points;
  preserve FEM/IPC and controller state.

The Scene527 profile consumes only the portable physical inputs: the 55k cloth, dual-X5 URDF and referenced meshes,
and the 4,373-frame trajectory. The controller interpolates every 60 Hz action over two 120 Hz physics steps. Reference
replays, rendering assets and research diagnostics remain outside the runtime contract. Stage boundary archives require
the matching Scene checkpoint, which includes native IPC finite-element and affine-body state.

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
9. If the backend elects to publish a uniform frame clock, treat `ScenarioFrameRunner.step()`'s return value as that
   frame's native substep count rather than replacing the scenario solver time step.

For a task restart, use the same lifecycle for all five scenarios:

```python
runtime.scene.reset()
controller.reset(runtime)
```

The Scene restores solver and entity state. The controller reset clears task phase, IK and trajectory caches and
resynchronizes scripted tool handles. A serialized mid-trajectory checkpoint must store `scene.get_state()`,
`controller.get_state(...)` and, when used, the optional frame runner state if execution is expected to resume at the
same phase rather than restart from phase zero. Reset the optional runner with `runner.reset()`.
Restore the Scene state with `runtime.scene.restore(scene_state)` first and then call `controller.set_state(...)`;
`restore` preserves the task's registered initial state, so a later bare `scene.reset()` still restarts from the task
origin. Coffee passes `runtime` to `get_state` because
its coordinated robot trajectory keeps an internal joint target, while the other controllers need no runtime argument
when taking their snapshot.

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
contact updates whenever the adapter calls `scene.step()`. Existing integrations continue calling
`controller.step(runtime)`. If an adapter needs a 60 FPS presentation clock, it may construct
`ScenarioFrameRunner(controller, runtime, fps=60.0)`; ratios that are not integers alternate between adjacent native
substep counts without accumulating simulated-time drift.

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

### Handoff scope

The five normalized packages are the source-checkout integration surface. Config groups and runtimes are named
dataclasses; task state lives in explicit controller snapshots; scenario physics updates use public engine APIs and
registered scene callbacks. CLI recording and acceptance measurements stay in `run.py`, including the coffee client.
The coffee implementation still contains its substantial coordinated manipulation and construction logic; the package
exports isolate that implementation from adapter callers. A future internal split can preserve those exports.

These are calibrated task reproductions rather than a generic robot/task compiler. Each config exposes its declared
knobs; coffee still has detailed phase timings and grasp thresholds inside its controller implementation. Robot link
names, joint layouts and semantic tool points are scenario-specific. Adapting an arbitrary replacement robot or a
different manipulation sequence requires a matching scene/controller implementation, beyond changing material
parameters. Preserve the documented asset representation and task frame when porting the five current examples.

Integration readiness here means a caller can construct, advance, observe and restore a task through the documented
API. It does not certify real-time throughput, batched task builders, an installable scenario wheel, an RL task schema
or visual task success. Those are separate requirements. The remote UniRoboSim implementation is maintained separately;
this repository's public API tests and recorded engine runs are evidence for this side of the boundary, not a joint
end-to-end acceptance of an uninspected adapter revision.

The current table calibration adds an optional `absorption_motion_rate` in inverse seconds. Existing absorbent
collider callers that omit it keep their configured `absorption_rate` for inward motion as well as capture throughput.
The table config supplies its calibrated value explicitly. Controller signatures, native `control_dt` values and
the optional 60 FPS frame-runner contract stay unchanged. The open-slot litter asset correction requires matching
visual/SDF assets and freshly generated checkpoints; check its asset notes in the scenario README before handoff.

Before updating the commit pinned by UniRoboSim-Genesis:

1. Run the configuration and adapter tests for the changed capability.
2. Run a one-step headless smoke test for each changed normalized scenario.
3. Run the complete default task trajectory on its supported GPU backend and record the command and result.
4. Verify `scene.get_state()`, `scene.restore(state)` and `scene.reset()` for any solver whose state layout changed.
5. Confirm asset pose and scale changes propagate to task targets without editing solver configuration.

A scenario is structurally integrated after steps 1, 2 and 5. Mark it physically validated only after its complete
trajectory and relevant state/reset checks pass.
