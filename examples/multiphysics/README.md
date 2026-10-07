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

For adapter handoff, use the [exact-call table](../../docs/integration/unirobosim_genesis.md#exact-scenario-calls)
and paired checkpoint example. Controllers advance physics themselves; call either
the controller/frame runner or your own commands plus `scene.step()`. The current builders create unbatched scenes.
Deploy the pinned source checkout: ordinary engine wheels package `genesis`, while these scenarios remain under
`examples`. The guide documents camera keywords, coffee's snapshot signature and the butter controller's dt argument.

All five default scenarios passed real GPU paired restore, finite continuation, bare scene reset and restart checks
on 2026-10-07. Seven-phase diagnostic recordings and logs are under `out/handoff_20261007`; these short lifecycle
checks do not replace complete task trajectories or visual acceptance. See the validation record for both scopes.

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

To inspect Scene527 before executing the folding trajectory, record only its 60 native settling steps:

```bash
python -m examples.multiphysics.garment_folding.run \
  --profile scene527 --asset-root /path/to/reproduction/scene527_55k_73s \
  --settle-only --settle-steps 60 --output out/garment_folding/settle
```

This writes a video, per-step height/velocity/centroid/extent CSV, summary JSON and a serialized scene checkpoint.
The 60 native steps cover 0.5 seconds at 120 Hz; they are not 60 control frames. Inspect the result before using
the checkpoint for trajectory continuation. Saving a checkpoint alone does not certify that settling succeeded.

The source Scene527 calibration used libuipc 0.0.25. Newer versions preserve much of the Python API but change
shell energy semantics: 0.0.26 replaces membrane volume weighting with area weighting, and 0.0.28 also removes
the outer bending-hinge area multiplier
([0.0.28 reference metric](https://github.com/spiriMirror/libuipc/blob/v0.0.28/src/backends/cuda/finite_element/constitutions/discrete_shell_bending_reference.h)).
The IPC adapter preserves the legacy effective membrane coefficients
and per-edge bending weights rather than passing the old numbers directly. The Scene527 profile explicitly
requires at least one Newton iteration and filters same-articulation IPC rigid pairs while retaining robot/table
contact. These are solver/material corrections; they do not change the 60 Hz control interface.

On Windows, point both `QD_OFFLINE_CACHE_FILE_PATH` and `GS_CACHE_FILE_PATH` to writable local directories.
An unwritable Quadrants cache can stall construction in temporary-file retries. Scene527 uses IPC rigid contact
and disables the separate Genesis rigid collider; disabled rigid collision now also skips hollow-mesh SDF probes.

On 2026-10-06, the Windows/libuipc 0.0.28 settling regression completed 60 steps with a finalized 60-frame video
and finite FEM checkpoint state. Final median/P90/maximum table-relative heights were 3.284/11.993/25.823 mm,
versus 3.280/11.186/25.652 mm in the source 0.0.25 run. Warm scene construction took 38 seconds and settling
took 401 seconds. This validates the initial settling correction, not the later folding trajectory or checkpoint
continuation. Results are locally stored under `out/garment_folding/fixed_hinge_area_settle60_20261006`.

Resume the folding trajectory from the inspected settling checkpoint:

```bash
python -m examples.multiphysics.garment_folding.run \
  --profile scene527 --asset-root /path/to/reproduction/scene527_55k_73s \
  --checkpoint out/garment_folding/settle/settled-scene-state.pt \
  --segment-steps 120 --output out/garment_folding/fold
```

Each segment saves a video and a paired Scene/controller checkpoint. With a checkpoint, omitted `--steps` runs
the remaining trajectory; explicit `--steps` specifies additional native steps. CSV distinguishes controller
trajectory time from the Scene's restore-local clock. The recorded trajectory remains 60 Hz with two native
120 Hz steps per action. Scene527 continuation reached controller step 8,746 with finite state and 73 finalized video
segments. Its 8,738-step recorded continuation starts at controller step 8; initial settling is a separate video.
Folding shape quality remains a visual acceptance item. The lightweight half-fold is a separate profile requiring
its own full rerun after the IPC material correction.

Coffee records finalized one-second video segments by default (`--record-segment-seconds`), and reports build
and simulation wall times. Use `--profile-build` for compilation diagnostics and `--progress-every` for progress.
On 2026-10-07, the 500-step/one-second liquid run finalized a 50-frame video: all sampled states were finite,
mass remained 0.5043063 kg, all 142,312 water particles stayed in their cup, and built-in tracking/clearance checks
passed. Cold build took 636 seconds; the subsequent full run's warm build took 103 seconds. The full manipulation
sequence then completed both arms' REST phase at step 11,140 / 22.28 seconds, with built-in terminal checks passing
and 23 finalized video segments. The CLI's older completion message reports the 14,000-step upper bound rather than
the actual early-termination count; use the final CSV time/step for that run. The current CLI reports the executed
count and lives in `run.py`, with the original executable forwarding to the same entry point. Windows recording uses
`PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1`.

The litter shovel uses a mesh signed distance field (SDF) with open slots, paired with its visual GLB by SHA-256.
The bundled `litter_scoop_sdf_slots.npz` has 1.5 mm cells in the GLB's converted Z-up local frame. Regenerate it with:

```bash
python examples/sand_water_coupling/litter_scoop_voxelize.py noplate
python -m examples.multiphysics.litter_scoop.run --record --output out/litter_scoop/mesh_slots
```

The `noplate` argument is required: the research generator's other variant fills the slots. Custom shovel meshes
require matching open-slot SDF data via `LitterScoopAssets.shovel_sdf`. The former `blade_half_extents` and `handle_*`
box-proxy fields are removed. Controller and frame-runner entry points and the 60 Hz interface remain unchanged.
Mesh-obstacle pose and velocity now participate in DEM Scene snapshots. Old box-proxy litter checkpoints must be
regenerated with the mesh-obstacle scene. Litter videos finalize every 60 native steps and metrics report finite
particle state and height/speed statistics. The mesh-slot default trajectory completed all 766 steps / 12.766667
seconds with finite sampled state and 13 finalized video segments. Sand retention and slot-throughput quality remain
visual acceptance items.

The butter scenario uses the public `gs.materials.MPM.HerschelBulkleyButter` and
`gs.materials.MPM.PorousBread` materials. Its stratified sampling and finite-range `ButterContact` law remain scenario
construction details with calibrated values in `ButterSpreadingMaterialConfig` and
`ButterSpreadingContactConfig`. `build_scene()` registers the contact law as a public scene pre-step callback, so any
client that calls `scene.step()` receives the same contact behavior.

The larger scripts under `examples/multiflow` and `examples/sand_water_coupling` provide research references,
parameter studies, asset tools and renderers. UniRoboSim-Genesis should import the normalized scenarios above.

## Table-wiping calibration

The wiping liquid uses surface-tension compliance `100/225`, capture rate `800` particles per second and inward-motion
rate `10` inverse seconds. Larger surface-tension compliance weakens the surface-area constraint. Capture throughput
and inward motion are independent: the nearest absorption target has a 0.1-second time constant, increasing with
material-voxel distance. Other scenarios that omit `absorption_motion_rate` retain their configured rate for both
effects. Solver dt and the UniRoboSim 60 Hz stepping interface stay unchanged.

```bash
python -m examples.multiphysics.table_wiping.run --record --metrics-every 25 \
  --output out/table_wiping/calibrated
```

Recording finalizes one-second video segments; `--record-segment-seconds` adjusts their length. The CSV tracks free and
captured particle counts, table-relative water height, horizontal extent, free-liquid speed and captured-liquid
absorption progress. Particle conservation and finite state are checked while sampling. Particle-mode rendering and
tetrahedral sponge visualization remain available for diagnosis; the calibration controls physical behavior.

On 2026-10-07, the full 5,000-step / 10-second trajectory completed with 201 finite CSV samples, conserved 840-particle
count and ten decoded video segments. Pre-contact water maximum at 0.2 seconds decreased from 52.42 to 17.73 mm.
The captured count rose from 96 at 0.4 seconds to 576 at 1 second, reaching 840 by 1.4 seconds. Results are in
`out/table_wiping/calibrated_full_v2_20261007`; isolation, API and absorption regression evidence is recorded in
`docs/integration/full_trajectory_validation.md`. Particle spacing remains 6.67 mm, limiting thin-film resolution.

## Recorded results and readiness

All five packages expose the normalized adapter API. Full numerical execution and desired visual task outcomes are
separate gates: Scene527, coffee, open-slot litter and calibrated table have recorded completed runs; butter's full
trajectory and the corrected lightweight garment's full run remain pending. See the validation record for lifecycle
checks and the source-deployment limitations in the integration guide.

The following combined videos are local artifacts, alongside their original segments and metrics. `out/` is excluded
from Git, so share these files separately when handing off to another machine.

| Run | Combined video relative to repository root |
|---|---|
| Scene527 checkpoint continuation | `out/garment_folding/scene527_full_fold_20261006/garment-folding-full.mp4` |
| Coffee full task | `out/coffee_water/full_default_20261007/coffee-water-full.mp4` |
| Open-slot litter full task | `out/litter_scoop/mesh_slots_default_20261007/litter-scoop-full.mp4` |
| Calibrated table full task | `out/table_wiping/calibrated_full_v2_20261007/table-wiping-full.mp4` |
