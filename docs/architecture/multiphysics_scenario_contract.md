# Multiphysics scenario contract

The normalized construction inputs under `examples.multiphysics` use the Genesis public scene lifecycle.
UniRoboSim-Genesis should call these modules; scripts under `examples.multiflow` and
`examples.sand_water_coupling` provide research and rendering references.

## Configuration ownership

Every normalized scenario separates four kinds of input:

- `SolverConfig` contains scene-wide numerical resolution, time step, bounds and iteration controls.
- `MaterialConfig` contains physical properties such as density, stiffness, friction, viscosity and absorption.
- `Assets` contains mesh paths, entity dimensions, initial poses and visualization geometry.
- `TaskConfig` contains robot or tool motion, phase duration and task termination parameters.

Changing an asset pose must not change solver resolution. Changing the particle size must not silently resize a cup,
table, sand bed or shovel. A scenario groups these objects only for convenient construction.

## Canonical entry points

### Coffee pouring and cleanup

`examples.pbstf_coffee_water` is the validated executable. Its PBSTF, liquid material, task assets and collider
configuration live under `examples.multiphysics.coffee_water`. `examples.multiflow.pbstf_coffee_water` forwards to
the same implementation.

### Table wiping

`examples.multiphysics.table_wiping.build_scene(config)` accepts `TableWipingScenarioConfig` and returns named scene
handles. `TableWipingController.step(runtime)` applies the task command and advances one scene step. The PBSTF
surface-tension script remains available for the other exploratory cases.

### Coupled litter scooping

`examples.multiphysics.litter_scoop.build_scene(config)` accepts `LitterScoopScenarioConfig` and returns the scene,
entities and validated DEM/FLIP setup. `LitterScoopController.step(runtime)` moves the shovel and advances one scene
step. Recording, plotting and asset-alignment scripts remain under `examples.sand_water_coupling` and do not form
part of the adapter contract.

### Garment folding

`examples.multiphysics.garment_folding.build_scene(config)` accepts `GarmentFoldingScenarioConfig` and returns named
garment, table, jaw and landmark handles. `GarmentFoldingController.step(runtime)` drives the public rigid-entity API.
Garment placement, scale and semantic grasp landmarks belong to `GarmentFoldingAssets`; task trajectories are evaluated
in that asset-local frame. The validated default is the half-fold task; the other task variants remain experimental.

### Butter spreading

`examples.multiphysics.butter_spreading.build_scene(config)` accepts `ButterSpreadingScenarioConfig` and returns named
bread, butter, blade and contact handles. `ButterSpreadingController.step(runtime)` advances the continuous
press-spread-lift trajectory. The controller only commands the blade; the scene invokes `ButterContact` through its
public pre-step callback for every `scene.step()` call. Bread and butter use the public
`gs.materials.MPM.PorousBread` and `gs.materials.MPM.HerschelBulkleyButter` materials. Asset placement defines a task
origin at the bread surface, while trajectory heights remain task-relative clearances.

## Solver integration

`genesis.integrations.create_particle_fluid_setup` validates common density and particle-size inputs for IPBF,
IPBSTF and PBSTF while preserving their solver-specific options.

`genesis.integrations.create_granular_fluid_setup` validates the DEM/FLIP domain, particle size and densities. It also
rejects a coupled grid that is too coarse to absorb liquid into one grain. The returned setup contains ordinary
Genesis options and materials and is passed to `Scene` without exposing solver-list indices.

After `scene.build()`, use the public `scene.dem_solver`, `scene.flip_solver` and `scene.pbstf_solver` properties.
Adapter code must not access `scene.sim._solvers` or rely on solver ordering.

MPM materials may set `particle_size` to override the solver-wide sampling spacing. Particle mass and stress transfer
use each material point's own reference volume, which permits coarse bread and fine butter in one MPM solve without
changing total mass when sampling resolution changes.

## State and reset

DEM snapshots include particle position, velocity, active state and absorbed-water ratio. FLIP snapshots include
particle position, velocity, active state, MAC-grid face velocities and the previous fluid step duration. These fields
are required so `scene.get_state()` and `scene.reset()` restore coupled motion instead of restoring only rendered
particle positions.

FLIP currently supports one environment per scene. UniRoboSim-Genesis must create one scene per independently reset
FLIP environment until batched FLIP is implemented.

## Reference scripts

The reference directories contain parameter studies, offline renderers, asset-alignment tools and alternate shovel
models. They document solver behavior and reproduce comparison results. New adapter work should begin with a
normalized scenario and use a reference script only for a tested parameter set or visualization workflow.
