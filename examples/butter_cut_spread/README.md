# Continuous butter cutting and spreading

One Genesis scene simulates cutting an end slice from a butter bar, carrying it on
a blade, and spreading it left to right on deformable bread. The same material
points remain active throughout; there is no particle replacement or state handoff.
The material point method (MPM) uses the engine's enabled CPIC cutting capability,
the repository's Herschel-Bulkley butter material, and its porous-bread material.
Finite cohesive contact and a unilateral blade correction supplement native coupling.

## Install and run

Use Python 3.10-3.13 and a supported GPU. Install a hardware-compatible PyTorch
build, then install this repository from its root:

```shell
python -m pip install -e ".[dev]"
python -m examples.butter_cut_spread.pipeline --run out/cut-spread
```

The full example contains 720,000 butter and 325,404 bread particles. At a 25 us
time step, it takes 230,000 steps to simulate 5.75 seconds. Outputs include a
139-frame particle cross-section video at 24 fps, covering the complete operation.
Allow several GB of RAM and disk space for full trajectories. First-run compilation
adds initialization time.

The default primitive example uses only this repository and its declared Python
dependencies. Its geometry is generated in code. The optional authored-asset
workflow below consumes user-supplied meshes; its Blender export utility runs
separately from Genesis. Neither simulation workflow requires Isaac or another
project checkout. Particle video uses Matplotlib's noninteractive Agg backend and PyAV; no
window, graphics context, separately installed video encoder, or surface mesh is needed.
The existing `examples.butter_spreading` modules provide shared material parameters,
bread contact and physics-only display notification management. They are used without
modification. Do not copy only this example into an unrelated upstream installation.

Stages can also be run individually:

```shell
python -m examples.butter_cut_spread.demo --run out/cut-spread-physics
python -m examples.butter_cut_spread.particles --run out/cut-spread-physics
python -m examples.butter_cut_spread.audit --run out/cut-spread-physics
```

`demo --steps 100 --run out/cut-spread-smoke` only checks construction and initial
contact. It is not a complete sample and cannot pass the full-trajectory audit.
Use a new output directory for each simulation. Failed stages retain their logs;
the pipeline propagates the failure instead of reporting completion.

## Scene and motion

| Quantity | Default |
| --- | --- |
| Blade face, thickness | 48 x 20 mm, 0.8 mm |
| Symmetric cutting bevel | 1.5 mm |
| Butter bar, cut-normal x blade-length x height | 80 x 32 x 18 mm |
| Cut-plane offset from butter center | +35.4 mm |
| Nominal slice thickness outside the blade | 4.2 mm |
| Nominal slice volume | 2.42 cm3 |
| Bread | 110 x 105 x 14 mm |
| Butter / bread particle spacing | 0.4 / 0.8 mm |
| MPM grid density | 768 cells per meter |
| Spreading travel | 17.4 mm |

These are illustrative dimensions, not measurements of a calibrated real specimen.
The blade is a convex beveled thin plate; handle, neck and rounded tip are omitted.
World coordinates use meters with Z upward. Each board's top is at Z=0. The blade's
local negative-X edge cuts, and its negative-Z broad face carries the end slice.
The blade remains upright during transport, then pitches down to place that loaded
face against bread; it does not make a horizontal half-turn.

| Phase | Simulation time (s) |
| --- | --- |
| Settle | 0-0.15 |
| Cut | 0.15-0.95 |
| Separate | 0.95-1.35 |
| Lift | 1.35-1.95 |
| Transfer | 1.95-3.25 |
| Approach | 3.25-4.00 |
| Spread | 4.00-5.00 |
| Withdraw | 5.00-5.75 |

The defaults also appear as a partial override in `configs/end_slice.json`:

```shell
python -m examples.butter_cut_spread.pipeline --config examples/butter_cut_spread/configs/end_slice.json --run out/end-slice
```

## Parameters and external integration

`config.py` defines dimensions, particle spacing, time step, contact strengths and
the prescribed trajectory. `--config` accepts a partial JSON object; omitted fields
use defaults. The expanded configuration is saved with each run.

`demo`, `asset_demo`, and `pipeline` accept `--material` with a partial
`ButterParameters` JSON. The effective physical parameters are saved separately in
`material.json`. See [material calibration](MATERIAL_CALIBRATION.md) for the
self-support and flow checks; changing a Blender surface shader does not change
the mechanical material.

- `butter_size`, `blade_size` and `cut_offset` control slice geometry. For the
  upright cut, slice thickness is `butter_size[0]/2 - cut_offset - blade_size[2]/2`.
  Keep it several particle spacings thick and fit the cut face within the blade.
- `normal_stress` and `shear_stress` govern finite pickup adhesion. Increasing
  them can retain more butter but also distort it or prevent release.
- `spread_normal_stress`, `spread_shear_stress` and `spread_adhesion_range` govern
  wet contact on both broad faces during spreading. Visual steel roughness has
  no role in these forces. Excess range can stretch deposited butter into strands.
- `spread_start_gap`, `spread_end_gap` and the start/end offsets control compression
  and stroke length. Match travel to available slice volume rather than stretching
  it into a layer thinner than the numerical resolution can represent.
- `board_normal_stress` and `board_shear_stress` control adhesion to the cutting
  board. Excess adhesion can leave the bottom of the slice behind.
- Changes to time step, resolution, motion or adhesion require another complete
  run and physical review. No inactive-particle residue or permanently pinned
  butter is used in this example.

The integration seam is the loop in `demo.py`: prescribe tool pose and velocity,
apply bread and tool contact, call `scene.step`, then apply the unilateral tool
correction. External trajectories must provide consistent position, quaternion,
linear velocity and angular velocity at each integration step. The helper
`angular_velocities` supplies body-frame joint rates and world-frame contact rates.

This is a complete reproducible example, not an arbitrary-asset loading interface.
The contact frame supports pitch, yaw and an optional world-X roll. Replacing the blade requires
matching the convex collision mesh, contact halfspaces and broad-face bounds.
Board contact and box initialization use the configured board bounds and top height.
Curved supports require the sampled height map used by the asset example below.
The contact recipe changes from anchored pickup to wet spreading
during 3.6-4.0 s, with a cutting shear cap before 0.95 s. When changing phase timing
or importing an external trajectory, adapt these controls together; changing only
the pose samples is insufficient. Material strengths are fitted demonstration
settings, not measured traction labels for a specific butter product.

## Authored Blender assets

`export_blend.py` runs inside Blender and exports explicitly named evaluated mesh
objects. It applies object transforms and the scene unit scale exactly once. Select
the finished assets; exclude hidden scan references, preview floors and cameras.
The Genesis runtime consumes exported geometry and has no Blender dependency.

```shell
blender --background --python examples/butter_cut_spread/export_blend.py -- --file BUTTER.blend --output out/assets/butter --names BUTTER_OBJECT
blender --background --python examples/butter_cut_spread/export_blend.py -- --file DISH.blend --output out/assets/dish --names DISH_OBJECT
blender --background --python examples/butter_cut_spread/export_blend.py -- --file KNIFE.blend --output out/assets/knife --names HANDLE_OBJECT CAP_OBJECT SEAM_OBJECT FERRULE_OBJECT STEEL_OBJECT
python -m examples.butter_cut_spread.asset_init --exports out/assets --output out/initialized
python -m examples.butter_cut_spread.prepare_runtime --assets out/initialized --output out/prepared --blender PATH_TO_BLENDER
python -m examples.butter_cut_spread.asset_demo --assets out/prepared --run out/asset-cut
python -m examples.butter_cut_spread.asset_particles --run out/asset-cut
```

The initialization adapter is for a butter asset with long X, middle Y and short Z
dimensions, a dish with Z up, and the offset-neck palette knife whose steel is the
last exported mesh. It rotates the butter onto its long/short footprint, fits the
knife's broad-face plane, and places the actual butter bottom against the measured
dish surface. Dimensions are preserved. `assets.json` records the transforms and
`support.npz` records the support height field. Other asset conventions need a
corresponding adapter, not an implicit axis swap or arbitrary scale correction.

The asset demo uses the same constitutive model and a 25 microsecond physical step.
Its default [physical material preset](asset_material.json) uses yield stress
160 Pa and consistency 15 Pa s^0.5 for the 39.17 mm tall asset. Moduli, density,
and flow exponent are unchanged. This is an engineering fit, not a measured
universal butter material; see [material calibration](MATERIAL_CALIBRATION.md).
Use `--material` to select another physical preset. The primitive example and
shared engine defaults are unaffected.
The authored knife uses its shallow 15-degree cutting posture. It settles for
0.6 s, lowers the blade through 1.5 s,
and advances once across the short side through 2.9 s. It then rises through
3.7 s and holds through 4.0 s. During the first 0.2 s of ascent, a monotone 3 mm
outward peel opens the cut. There is no fore/aft reversal, oscillation, or pause.
Stroke limits combine the downstroke with the single bottom draw using the actual
steel cross-section. This covers butter taller than the horizontal blade before
lifting, while retaining the shallow knife posture. The standalone initialization retains a
nominal 5 mm slice; the annotated cut-and-spread initializer defaults to 10 mm
measured beyond the outer blade face. Different tool shapes require checking
the complete swept section and tool clearance against the support.
Native
ordinary contact uses the actual nonconvex steel and dish meshes. The example's
`CuttingScene` restricts CPIC separation to the steel entity selected through
`scene.build(cutting_tools=(blade,))`. The dish and countertop retain native
normal contact and friction, but do not participate in cutting separation.
The same selection is used in the primitive cut-and-spread example. Other scenes
using `gs.Scene` are unaffected; no engine files or global classes are patched.
The handle is rendered outside the contact region; this example does not simulate
a robot grip. See [validation notes](VALIDATION.md) for the concave-support
failure diagnosed before this separation and the remaining material-size limits.

When integrating external assets, import `CuttingScene` from
`examples.butter_cut_spread.cutting_scene`, keep `enable_CPIC=True`, and pass the
rigid cutting entities explicitly to `build(cutting_tools=(blade,))`. Every geom
of a selected entity participates in CPIC, so keep non-cutting supports in separate
entities. Selection is fixed at build time and survives scene reset. The adapter
uses the repository's legacy coupler; it does not support other coupler types.
`scene.get_cpic_transfer_counts()` reports separated transfers by rigid geom in
the most recent substep. This diagnostic is not intended for every-step polling.

Broad-face adhesion is limited to the fitted planar blade patch. Post-step
nonpenetration correction uses the actual steel mesh's native signed distance
field, replacing the rectangular correction in the primitive demo. It removes
inward relative normal velocity and tracks the previous side within the interior
of the broad face. Flow around the blade edge remains allowed. This adapter
assumes the constant tool orientation of this trajectory; rotating trajectories
require recording the previous and current contact frames separately. The
correction is a numerical contact constraint, not an additional material model.
The prescribed tool uses `align=False`: automatic COM/principal-axis alignment
would change the frame addressed by absolute pose setters. Native geometry poses
are checked against the authored trajectory and recorded for rendering. Do not
replace this with a link-pose-only check, which can miss a collision-frame offset.
The MPM domain covers the dish footprint with an additional 25 mm margin.

For an isolated support check, add `--stationary --duration 0.6` and use a new run
directory. `particle-metrics.json` reports height, width and center-of-mass changes,
including height of fixed initial central particle IDs (whole-block height alone
can hide central settling);
a larger asset may still yield under its own weight with unchanged material settings.
The first high-resolution collision-distance-field build can be expensive; its
cached cost is distinct from steady simulation time.

The asset example exports `asset-cut-particles.mp4` and `asset-audit.json`
(`python -m examples.butter_cut_spread.asset_audit --run out/asset-cut`). The mesh
audit tests all particles within each saved blade AABB and a fixed particle-ID
sample against the dish. These are saved-time checks, not proof that no crossing
occurs between samples. The pickup/deposition acceptance criteria below apply to
the primitive cut-and-spread example, not this separate asset cut-and-lift case.

## Outputs and validation

- `config.json`, `source/`: effective settings and source snapshot.
- `state.npz`: butter and bread positions with persistent particle IDs, blade
  poses, cutting-contact side assignments and actual sample times.
- `simulation.json`: particle accounting, completion, per-phase timing and maximum
  post-step contact correction.
- `cut-spread-particles.mp4`, `particles-video.json`: particle cross-section video,
  encoding time and source/video hashes. Orange marks the initial end-slice IDs;
  particles leaving the central section disappear from this view, not from simulation.
- `audit.json`: pickup, coating-zone deposition, bread deformation, prescribed-pose
  error, interior flat-face penetration/crossing checks and decoded video checks.
- `pipeline.json`: elapsed time for each successfully completed stage.

`state.npz` uses `(frame, particle, xyz)` arrays for `particles` and `bread_particles`;
`knife_positions` is `(frame, 3)`, `knife_quaternions` is `(frame, 4)` in WXYZ order,
and `time_s` is the physical sampling time. All saved frames are shown in the video.
The state archive and phase checkpoints omit internal stress/deformation fields and
are not restart checkpoints.

Acceptance requires more than 95% pickup and more than 50% of the initial slice
inside the bread footprint within +/-4 mm of its nominal top, as well as bounded
tool-pose and contact-correction errors. Sampled interior crossings exclude flow
around edges and do not prove continuous-time nonpenetration. The particle audit
does not make claims about reconstructed surfaces or photorealistic appearance.

To compare two independently computed runs with identical configurations:

```shell
python -m examples.butter_cut_spread.compare --run out/reproduction --reference out/reference
```

This optional check uses every saved particle ID and frame, reports RMS, per-frame
99th-percentile and maximum deviations, and separately compares bread and tool
motion. GPU reductions may differ at floating-point precision; a visually similar
movie alone is not treated as a numerical reproduction. A reference is not required
to build or run the example.

Near the cut plane, small numerical differences can change which side retains an
individual particle. Its later transport then creates a large per-ID position
difference even if bulk pickup and deposition remain close. The strict comparison
can therefore fail on an independent repeat with the same configuration. Preserve
that result, inspect the outliers and compare independent repeats rather than
claiming bitwise reproducibility. See [validation notes](VALIDATION.md) for the
measured scope and limitations of the end-slice example.

Run the focused tests from the repository root:

```shell
python -m pytest -n 0 --backend gpu tests/particles/test_butter_cut_spread.py
```

## Annotated board, bread and handled-knife assets

For a continuous end cut, pickup and left-to-right spreading stroke using annotated
ZIP assets at their actual dimensions, see [aligned asset setup](ALIGNED_ASSETS.md).
This entry places deformable bread on the annotated second cutting board and uses
the authored handled knife throughout. It preserves the butter material state
across the cutting and spreading phases.
