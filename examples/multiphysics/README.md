# Normalized multiphysics scenarios

These modules are the stable construction examples for backend adapters:

- `coffee_water`: PBSTF pouring, stirring and cleanup;
- `table_wiping`: PBSTF liquid absorption with a deformable sponge;
- `litter_scoop`: coupled DEM sand and FLIP water with a moving shovel;
- `garment_folding`: FEM/IPC cloth folding with public rigid-jaw controls;
- `butter_spreading`: MPM bread and butter with a continuous moving blade.

Each scenario separates `SolverConfig`, `MaterialConfig`, `Assets` and `TaskConfig`. The adapter-facing entry points
are:

| Scenario | Construction | Control |
|---|---|---|
| Coffee pouring and cleanup | `examples.multiphysics.coffee_water.build_scene` | `CoffeeWaterController.step` |
| Table wiping | `examples.multiphysics.table_wiping.build_scene` | `TableWipingController.step` |
| Litter scooping | `examples.multiphysics.litter_scoop.build_scene` | `LitterScoopController.step` |
| Garment folding | `examples.multiphysics.garment_folding.build_scene` | `GarmentFoldingController.step` |
| Butter spreading | `examples.multiphysics.butter_spreading.build_scene` | `ButterSpreadingController.step` |

Construction returns named runtime handles, and every runtime exposes the duration of one native controller step as
`control_dt`. Direct controller calls retain the existing interface and advance one calibrated native step. An adapter
that chooses to publish a uniform clock may use `ScenarioFrameRunner(controller, runtime)`; its default 60 FPS frames
distribute the required native steps through `substeps_for_next_frame()` without changing the scenario's physical
time step. Rendering and offline recording are
optional clients of the runtime. See the
[UniRoboSim-Genesis integration guide](../../docs/integration/unirobosim_genesis.md) for capability discovery,
state/reset requirements and adapter examples. The
[full-trajectory validation record](../../docs/integration/full_trajectory_validation.md) distinguishes complete
physical task runs from construction smoke tests.

Omit `--steps` to run the complete default task trajectory:

```bash
python -m examples.multiphysics.coffee_water.run
python -m examples.multiphysics.table_wiping.run
python -m examples.multiphysics.litter_scoop.run
python -m examples.multiphysics.garment_folding.run --task half
python -m examples.multiphysics.butter_spreading.run
```

Pass `--steps 1` for a construction-and-step smoke test. Coffee also accepts `--check-motion` to validate the robot
trajectory without creating liquid particles; that mode is useful for controller debugging but is not a physical
coffee-task validation.

The long butter and table commands print progress every 1,000 steps, and litter scooping prints every 50 steps. Pass
`--progress-every 0` to suppress those updates or another positive interval for a different validation-log cadence.

The garment scenario uses the optional IPC backend. Install `pyuipc` before constructing that scene:

```bash
pip install pyuipc
```

`GarmentFoldingMaterialConfig` keeps the inherited contact mixing by default. Set `cloth_self_friction` or
`cloth_table_friction` when folded-layer sliding and table contact require independent calibration. The lightweight
profile remains the default. A portable Scene527 bundle can be supplied without changing the adapter contract:

```bash
python -m examples.multiphysics.garment_folding.run \
  --profile scene527 \
  --asset-root /path/to/reproduction/scene527_55k_73s
```

`create_scene527_config(asset_root)` exposes the same setup to Python adapters. The asset root must contain the 55k
cloth, dual-X5 URDF and mesh dependencies, and `controls/trajectory.npz`. Reference videos, Isaac files, replay dumps
and stage checkpoints are optional for a continuous physics run.

The butter scenario uses the public `gs.materials.MPM.HerschelBulkleyButter` and
`gs.materials.MPM.PorousBread` materials. Its stratified sampling and finite-range `ButterContact` law remain scenario
construction details with calibrated values in `ButterSpreadingMaterialConfig` and
`ButterSpreadingContactConfig`. `build_scene()` registers the contact law as a public scene pre-step callback, so any
client that calls `scene.step()` receives the same contact behavior.

The larger scripts under `examples/multiflow` and `examples/sand_water_coupling` provide research references,
parameter studies, asset tools and renderers. UniRoboSim-Genesis should import the normalized scenarios above.
