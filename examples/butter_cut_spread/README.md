# Continuous butter cutting and spreading

**最新版真实资产案例：** [独立安装、完整模拟与 Isaac 渲染](STANDALONE.md)。
运行所需资产和渲染代码均包含在本仓库；默认使用全程固定的 firm-smooth 材质、
0.6 mm 黄油采样和 50 μs 时间步。入口为 `examples.butter_cut_spread.asset_pipeline`。
下文的 `pipeline` 是保留的基础几何案例。

One Genesis scene simulates cutting an end slice from a butter bar, carrying it on
a blade, and spreading it left to right on deformable bread. The same material
points remain active throughout; there is no particle replacement or state handoff.
The material point method (MPM) uses the engine's enabled CPIC cutting capability,
the repository's Herschel-Bulkley butter material, and its porous-bread material.
Finite cohesive contact and a unilateral blade correction supplement native coupling.

## Install and run

Use Python 3.11 and a supported GPU. Install a hardware-compatible PyTorch
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
dependencies. Its geometry is generated in code. The optional source-asset export
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

The authored-asset default uses 240 Pa yield stress and 15 Pa s^0.5 consistency
to better retain the cut slice's shape. The earlier 160 Pa and intermediate
200 Pa settings remain available as `configs/asset_yield_160.json` and
`configs/asset_firmer.json`, respectively, through `--material`.
External callers must load a preset into the material constructor; copying its
JSON file or importing the scene does not replace the engine's 80 / 25 defaults.
See [material calibration](MATERIAL_CALIBRATION.md#slightly-firmer-slices-and-external-trajectories)
for effective-parameter logging and the required comparison on the actual motion.

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
240 Pa and consistency 15 Pa s^0.5 for the 39.17 mm tall asset. Moduli, density,
and flow exponent are unchanged. This is an engineering fit, not a measured
universal butter material; see [material calibration](MATERIAL_CALIBRATION.md).
Use `--material` to select another physical preset. The primitive example and
shared engine defaults are unaffected.
The complete 240 Pa authored-asset result is recorded in
[default material validation](DEFAULT_MATERIAL_VALIDATION.md), including pickup,
spreading, contact limits and the increased bread deformation.
The [performance guide](PERFORMANCE.md) explains the full-resolution execution,
uncompressed trajectory storage and reconstruction tradeoffs.
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

## Lower particle count and a firmer cut

`asset_demo --prepare-only --particle-size 0.0006` rebuilds the butter samples
while retaining the authored grid, timestep, bread sampling and calibrated
contact shell. The original 0.4 mm setup remains available. The CLI requires at
least two particles across a grid cell; reducing quadrature density trades fine
material detail and boundary-volume accuracy for less particle work and storage.
Preparation must target a fresh directory to retain the reference assets.

The optional `configs/asset_hard.json` uses an 80 kPa shear modulus and 1200 Pa
yield stress. This fitted demonstration material reduces recoverable strain and
plastic flow during cutting; it also requires greater force during spreading.
It remains a deformable MPM material with finite tool adhesion.

```shell
python -m examples.butter_cut_spread.asset_demo --assets out/horizontal-cut-spread-prepared --run out/hard-prepared --prepare-only --particle-size 0.0006 --material examples/butter_cut_spread/configs/asset_hard.json
python -m examples.butter_cut_spread.asset_demo --assets out/hard-prepared --run out/hard-run --material examples/butter_cut_spread/configs/asset_hard.json
python -m examples.butter_cut_spread.shape_audit --run out/hard-run
python -m examples.butter_cut_spread.asset_particles --run out/hard-run
python -m examples.butter_cut_spread.aligned_audit --run out/hard-run
python -m examples.butter_cut_spread.asset_audit --run out/hard-run
```

Use `--material` in both preparation and execution. Omission retains the 240 Pa
asset default. `shape_audit` measures all initial slice particles at phase
checkpoints after removing a proper rigid transform, reporting RMS and
99th-percentile deformation in meters. Compare reduced sampling with a fine run
of the **same material** to isolate discretization changes.

See [measured performance, shape retention and tradeoffs](PERFORMANCE.md#reduced-sampling-with-a-firm-material-preset)
for the complete local validation and the same-material fine-sampling comparison.

## Faster integration and visual review

For a firmer cut with easier spreading using **one material throughout**, use
`configs/asset_firm_smooth.json` in place of `asset_hard.json`. Its fixed parameters
are G=80 kPa, K=150 kPa, yield=600 Pa, consistency=25 and flow exponent=0.5.
There is no time-dependent material interpolation or phase switch. The measured
run is `out/firm-smooth-full`, using the existing 50 microsecond prepared assets:

```shell
python -m examples.butter_cut_spread.asset_demo --assets out/firm-50us-prepared --run out/firm-smooth-new --material examples/butter_cut_spread/configs/asset_firm_smooth.json
```

Compared with the fixed 1200 Pa preset, cut slice deformation RMS changes from
1.167 to 1.241 mm and lifted slice RMS from 1.597 to 1.860 mm. Spreading becomes
longer (59.81 to 76.50 mm), with 80.96% deposition and 99.95% pickup. Full physics
takes 558.31 seconds. These are demonstration material fits, not measurements of
a particular butter product.

The supported-film density reconstruction uses one particle
spacing of additional tangential kernel width, with normalized particle volume
and sufficient grid padding. This smooths submillimeter sampling gaps in thin
coatings. It preserves the bulk reconstruction away from bread and still passes
the test that retains a real 4 mm gap. Full physics and the 203-frame particle
video pass the aligned audit; Isaac review covers five frames.

### Thin coating holes

At 0.6 mm particle spacing, spreading stretches particles into a thin, unevenly
sampled film. The reconstructed density can fall below the 0.5 surface threshold
even where neighboring particles support a continuous-looking coating. In the
final checkpoint, the 13 enclosed gaps in the horizontal density projection had
nearest-particle distances of 0.21--1.19 mm. This indicates sampling/reconstruction
contributes to the visible holes; it does not establish that every gap is artificial.

`film.py` repairs enclosed gaps on the bread only when every
gap pixel is within two particle spacings of supported particles. It redistributes
surplus nearby density while keeping donor voxels above the surface threshold.
It preserves the density integral, leaves open edges and unsupported larger gaps,
and makes no changes to particle positions, mass, material or solver steps.
This is a visual regularization: the extracted mesh volume is not exactly conserved.

For `out/firm-smooth-full`, final projected enclosed gaps decrease from 13
(28.66 square mm) to zero. The integrated density changes by less than 1e-11
relative, while final combined mesh volume increases by about 0.0165%.
Three-dimensional mesh topology still has 12 handles/tunnels, and irregular open
edges remain. These metrics must not be interpreted as a completely hole-free
solid. The five reviewed Isaac frames use 128 samples each; the full 203-frame
Isaac animation has not been regenerated. The material is fixed throughout.

Focused tests cover density conservation, absent particle support, insufficient
density budget, retained large holes and open edges. Run these from this repository:

```shell
python -m pytest -n 0 tests/particles/test_butter_film_gaps.py tests/particles/test_butter_cut_spread_surface.py -k "sampling_gap or density or supported_single_layer" -q
```

`asset_demo --prepare-only --dt SECONDS` rebuilds the prescribed trajectory at a
different integration interval. The material integration and contact impulses use
that same interval. This changes numerical integration accuracy, so compare the
complete pickup and spreading sequence before adopting a larger interval for a
different material. The default remains 25 microseconds.

For the firm asset preset, the tested faster configuration uses 0.6 mm butter
sampling and a 50 microsecond step:

```shell
python -m examples.butter_cut_spread.asset_demo --assets out/firm-balanced-prepared --run out/fast-prepared --prepare-only --particle-size 0.0006 --dt 0.00005 --material examples/butter_cut_spread/configs/asset_hard.json
python -m examples.butter_cut_spread.asset_demo --assets out/fast-prepared --run out/fast-run --material examples/butter_cut_spread/configs/asset_hard.json
```

The prepared `out/firm-50us-prepared` directory is also available on the measured
workstation. Use a fresh `--run` directory for each simulation. See
[fast-profile measurements and numerical differences](PERFORMANCE.md#larger-timestep-with-the-same-firm-particles).

Tool velocity and position writes now defer their geometry update until the
quaternion write, publishing the complete prescribed state once per step.

The bundled Isaac replay supports a matching five-frame reconstruction/capture
preview. Run the following from this repository,
using its existing reconstruction and Isaac Python environments respectively:

```shell
python -m examples.butter_cut_spread.asset_prepare --run RUN_DIRECTORY --preview --workers 5 --coating-voxel 0.0001
python -m examples.butter_cut_spread.asset_render --run RUN_DIRECTORY --preview --samples 128
```

This preview keeps the original spatial reconstruction and render sample budget;
it covers the initial, quarter, middle, three-quarter and final recorded frames.
It is a visual review of five states, not a full animation. Remove `--preview`
from **both** commands to reconstruct and render every frame. The renderer rejects
a full capture when the reconstruction manifest only covers the preview frames.
