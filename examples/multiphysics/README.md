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
| Coffee pouring and cleanup | `examples.pbstf_coffee_water.build_scene` | `step_scene` |
| Table wiping | `examples.multiphysics.table_wiping.build_scene` | `TableWipingController.step` |
| Litter scooping | `examples.multiphysics.litter_scoop.build_scene` | `LitterScoopController.step` |
| Garment folding | `examples.multiphysics.garment_folding.build_scene` | `GarmentFoldingController.step` |
| Butter spreading | `examples.multiphysics.butter_spreading.build_scene` | `ButterSpreadingController.step` |

Construction returns named runtime handles and control advances one step at a time. Rendering and offline recording are
optional clients of the runtime. See the
[UniRoboSim-Genesis integration guide](../../docs/integration/unirobosim_genesis.md) for capability discovery,
state/reset requirements and adapter examples.

Run the scenarios with:

```bash
python -m examples.multiphysics.garment_folding.run --task half --steps 1
python -m examples.multiphysics.butter_spreading.run --steps 1
```

The butter scenario uses the public `gs.materials.MPM.HerschelBulkleyButter` and
`gs.materials.MPM.PorousBread` materials. Its stratified sampling and finite-range `ButterContact` law remain scenario
construction details with calibrated values in `ButterSpreadingMaterialConfig` and
`ButterSpreadingContactConfig`. `build_scene()` registers the contact law as a public scene pre-step callback, so any
client that calls `scene.step()` receives the same contact behavior.

The larger scripts under `examples/multiflow` and `examples/sand_water_coupling` provide research references,
parameter studies, asset tools and renderers. UniRoboSim-Genesis should import the normalized scenarios above.
