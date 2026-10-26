# Full-resolution cutting and spreading performance

## Larger timestep with the same firm particles

The `firm-50us-full` run completes the full 8.4-second operation with 168,000
steps and 203 recorded frames. It uses the same initial 549,158 butter and
272,258 bread particles, 768 cells/m grid, contact law and `asset_hard.json`
material as `firm-balanced-full`. Prepared tool poses match exactly at shared
timestamps. The integration interval increases from 25 to 50 microseconds.
The tool setters also publish geometry once after writing velocity, position and
orientation, eliminating intermediate forward-kinematics calls.

| Measure | Previous firm profile | Faster firm profile |
| --- | ---: | ---: |
| Physics stepping | 1348.44 s | **570.11 s** |
| Physics steps | 336,000 | 168,000 |
| Slice pickup | 100% | 100% |
| Final deposition | 78.99% | 79.98% |
| Final deposited X span, p05–p95 | 55.58 mm | 59.81 mm |
| Deposited span growth | 20.51 mm | 24.96 mm |
| Cut slice deformation RMS | 1.097 mm | 1.167 mm |
| Lift slice deformation RMS | 1.563 mm | 1.597 mm |

Stepping is 2.37 times faster, or 57.7% less time, on the same workstation.
Initialization, integration, export and hashing together take 593.37 seconds.
All recorded particles remain active and finite; the existing complete aligned
audit passes without changing its thresholds. The audit finds zero sampled
interior blade side changes and 0.288 mm maximum sampled bread/board intrusion.
The asset audit finds no particles reaching the simulation domain boundary.

This is an approximate integration preset, accepted for the requested tolerance
of modest visual changes. It does **not** satisfy strict particle-identity
reproduction tolerances: at the final checkpoint butter position RMS difference
is 1.548 mm and p99 is 3.683 mm; the maximum is 183.216 mm because rare particles
take different attachment/release paths. Bread position RMS difference is
1.098 mm. Individual particle agreement is therefore distinct from the similar
rendered shape and maintained deposition. The same-camera final images retain
a continuous butter strip, with a slightly longer, differently textured end.
The cut slice remains deformable, with approximately the previous firmness.

`--dt` is a preparation-time option so motion samples, material integration and
contact impulses use the same interval. The conservative default stays at 25
microseconds. Genesis emits its grid-based timestep advisory at 50 microseconds;
the complete validation here applies to this specific firm configuration, not
arbitrary stiffer materials. See the README for the exact preparation/run commands.

The Isaac replay checkout also supports `asset_prepare --preview`, sharing its
frame selection with `asset_render --preview`. For this run it reconstructs and
captures indices 0, 50, 101, 152 and 202. Reconstruction takes 25.76 seconds and
capture 35.36 seconds, retaining the 0.3 mm bulk / 0.1 mm coating voxels and
verified 128 render samples. This is a five-state review, **not** a complete
203-frame Isaac video or an equivalent-work rendering speedup. Full replay uses
both commands without `--preview`; requesting all frames from a preview-only
surface manifest raises an error before capture.

Validation artifacts are in `out/firm-50us-full`: `input-comparison.json`,
`timestep-comparison.json`, `shape-audit.json`, `aligned-audit.json`,
`asset-audit.json`, and `fast-validation.json`. The complete particle video has
203 decoded frames; the Isaac preview has five. Twelve focused tests pass,
including a GPU check that combined tool updates publish the same pose and
velocity as separate updates for translated and rotated free tools.

## Original full-resolution baseline

The asset case integrates 8.4 seconds with 336,000 steps at 25 microseconds,
1,883,802 butter particles and 272,258 bread particles. Its precision settings
remain 32-bit solver arithmetic, 0.4 mm butter spacing, 0.8 mm bread spacing and
768 grid cells per meter. Material, contact, trajectory, geometry and all 203
recorded frames remain part of the validation.

## Simulation and trajectory storage

The complete integration loop and all physical kernels retain the validated
implementation. A short GPU profile attributes approximately 56% of kernel time
to particle-to-grid transfers, 10% to deformation/SVD, 9% to grid-to-particle
transfers and 7% to state copying. These measurements characterize the stationary
start of this scene; they do not apportion the entire trajectory.

A candidate that skipped redundant non-tool compatible particle-in-cell (CPIC)
normal queries passed single/batched support and separation checks and reduced
short-run stepping by about 9%. Its long-run particle-identity comparison exceeded
the existing tolerance during carrying: a small number of particles followed
different attachment/release branches. At the transfer checkpoint butter RMS
difference was 0.252 mm, above the existing 0.1 mm acceptance limit. The candidate
was rejected and restored before release. Acceptance tolerances remain unchanged.
Physical stepping therefore retains its measured reference cost of 2,065.75 s.
The accepted changes accelerate data handling and rendering around this calculation.

`asset_demo` writes uncompressed `state.npz`. All consumers continue to read it
with NumPy. This avoids compression during export and repeated decompression
during reconstruction and auditing. On the same complete trajectory:

| Storage | Write time | Read and exact comparison | File size |
| --- | ---: | ---: | ---: |
| Compressed NPZ | 172.93 s | 20.37 s | 4.27 GB |
| Uncompressed NPZ | 3.67 s | 4.21 s | 6.78 GB |

Every array, shape and dtype matches exactly. Faster access uses about 59% more
disk space. Dataset archival can apply lossless compression after processing;
compression need not occupy the simulation/export critical path. SHA-256 hashing
streams the file instead of allocating an additional archive-sized byte buffer.
Simulation reports expose export and hash durations separately.

## Surface reconstruction

The local density kernel computes each one-dimensional quadratic kernel weight
once per particle and retains it across the other two axes. Scalar bounds avoid
temporary coordinate arrays. Particle accumulation order, multiplication order,
kernel widths, volume normalization, isovalue selection and Boolean clipping
remain unchanged.

The Isaac-side `asset_prepare` defaults to up to six independent frame workers,
bounded by the available CPU count. Use fewer workers to reduce working memory.
Workers share a read-only trajectory and own their individual frame buffers.
The measured workstation has a Core i9-14900KF, 64 GB RAM and an RTX 5090 D v2.

The complete 203-frame reconstruction decreased from 747.79 s to 319.71 s,
including trajectory loading. The comparison uses the same particle arrays,
0.3 mm bulk voxels and 0.1 mm supported-coating voxels. All 203 exported PLY files
match the reference byte for byte. Unit checks also cover clipped kernels,
single/double input precision, particle volume and anisotropic kernel moments.

```shell
python -m examples.butter_cut_spread.asset_prepare --run RUN_DIRECTORY --workers 6 --coating-voxel 0.0001
```

This command belongs to the separate local Isaac replay environment. The solver
release continues to run independently of Isaac and its rendering dependencies.

## Capture settings and redundant viewport work

The cut-and-spread mesh uploader flattens the already contiguous triangle index
buffer in one operation. The former upload helper concatenated individual rows of
a tracked mesh array, creating millions of temporary Python objects. For a late
frame with 2,768,492 triangles, index conversion decreased from 8.36 s to 0.016 s
in an isolated comparison, with identical integer buffers. Point positions,
vertex normals, triangle order/winding, extents and USD float precision are
unchanged. This adapter is local to the cut-and-spread renderer; other demos and
the shared upload helper are unchanged.

Headless Isaac capture disables background viewport updates while retaining the
explicit 1280 x 720 render product. Each recorded state is captured at the requested
128 samples per pixel. The renderer applies its settings after stage initialization
and checks the effective render mode and sample count for every captured frame.
The original initialization path was observed resetting to 64 samples after an
earlier 128-sample request, so historical reports storing only the request do not
prove the effective budget. The current budget is verified rather than reduced.

Capture reads the first completed valid RGB image. Empty readbacks retry the same
frozen state, up to the existing bounded retry limit. A probe spanning initial,
cut, pickup, transfer, spread and final states compared the first and subsequent
valid images; paired valid captures were identical. Redundant unconditional
discarded captures are omitted. Per-frame reports separate geometry handling from
capture time and include the effective sample count. Materials, lights, camera,
bounces, coating reconstruction and recorded motion retain their settings.

The complete 203-frame render comparison at **verified 128 samples on both sides**
measured 1,066.10 s before bulk upload and 523.93 s afterwards (50.9% less time).
Geometry handling decreased from 662.24 s to 136.80 s; image capture measured
377.97 s and 363.91 s respectively. The A/B retains the same capture loop and
disabled hidden viewport, isolating the bulk uploader. Final exported USD mesh
positions, normals, triangle buffers and extents match byte for byte. The ten
saved PNG comparisons have approximately 0.11-0.12 mean absolute difference on
the 0-255 scale and a 99th-percentile difference of 2; separate path-tracing
processes are not claimed to produce bitwise-identical pixels.

| Stage | Before | After | Measurement |
| --- | ---: | ---: | --- |
| Physics integration | 2,065.75 s | Unchanged | Original complete physical run; kernels unchanged |
| Trajectory archive write | 172.93 s | 3.67 s | Same assembled arrays, compressed versus uncompressed |
| Surface reconstruction | 747.79 s | 319.71 s | All 203 frames, identical exported meshes |
| Isaac rendering | 1,066.10 s | 523.93 s | All 203 frames, verified 128 samples in both runs |

The historical 928.60 s video used a requested-only sampling report and recovery
across two processes. It is not the same-sampling baseline used in the table.

## Validation scope

The complete postprocessing comparison replays the same previously validated
240 Pa physical trajectory. It does not claim a new complete physics execution.
The repacked archive includes provenance identifying this reuse. All arrays are
bitwise identical, all 203 surface files are byte-identical, and final validation
checks video frame counts, trajectory hashes, saved-frame contact metrics and
actual render sampling. A separate native-solver smoke run exercises the updated
export entry point. The unmodified integration loop retains the complete earlier
physical validation.

Validation artifacts are `performance-summary.json`, `repack-validation.json`,
`surface-equivalence.json`, `render-equivalence.json` and `final-validation.json`
in the output directory. Final verification passes for both complete 203-frame,
24 fps, 1280 x 720 videos. The regression checks cover density accumulation,
anisotropic volume/moments and the uploaded USD buffers. The native-solver smoke
run completes 2,000 steps with every particle active and writes the new archive.

The 172.93/3.67 s export comparison isolates writing the same already assembled
arrays. It excludes stacking recorded frames in memory. The historical full-run
simulation/export duration uses a different timing boundary, so these figures
must not be added into a purported newly measured end-to-end total. Rendering
and reconstruction totals are complete wall-clock stage measurements; profiling,
failed experiments and verification are additional development costs.

## Running after an update

Refresh the prepared manifest after updating the entry point, then use a fresh
run directory. Existing geometry and native distance-field caches are reused.

```shell
python -m examples.butter_cut_spread.asset_demo --assets out/aligned-prepared --run out/aligned-prepared --prepare-only
python -m examples.butter_cut_spread.asset_demo --assets out/aligned-prepared --run out/aligned-case
```

## Reduced sampling with a firm material preset

`asset_demo --prepare-only --particle-size 0.0006` provides a separate speed/quality
tradeoff. It rebuilds the butter samples and manifest, retaining the 768 cells/m
grid, 25 microsecond integration step, 0.8 mm bread samples, 24 fps recording and
the original calibrated 0.4 mm contact shell. The complete prescribed motion
matches the reference exactly. The CLI limits butter spacing to half a grid cell.
The original-resolution workflow above remains available.

The optional `configs/asset_hard.json` selects shear modulus 80 kPa, bulk modulus
150 kPa and yield stress 1200 Pa, retaining density 900 kg/m^3, consistency 15 and
flow exponent 0.5. These are fitted demonstration parameters. This material keeps
the stock and cut slice firmer while allowing plastic spreading under pressure.
All particles remain in the same MPM material throughout the trajectory.

The complete local run is `out/firm-balanced-full`; its prepared assets are
`out/firm-balanced-prepared`. Against the historical 240 Pa full-resolution run:

| Measurement | 240 Pa / 0.4 mm | 1200 Pa / 0.6 mm |
| --- | ---: | ---: |
| Butter particles | 1,883,802 | 549,158 |
| Total particles including bread | 2,156,060 | 821,416 |
| Full physical stepping | 2,065.75 s | 1,348.44 s |
| Downstroke slice deformation RMS | 2.247 mm | 0.447 mm |
| Cut-end slice deformation RMS | 2.350 mm | 1.097 mm |
| Lift slice deformation RMS | 2.384 mm | 1.563 mm |
| Cut-end slice world-Z height retained, p01–p99 | 85.67% | 97.19% |
| Slice pickup | 99.987% | 100% |
| Final deposited slice fraction | 83.68% | 78.99% |
| Deposited slice X-span growth | 45.80 mm | 20.51 mm |
| Bread displacement median / p95 | 1.115 / 2.960 mm | 2.666 / 4.113 mm |
| Maximum bread displacement | 5.628 mm | 16.084 mm |

Butter particle work decreases by 70.85%; total particle count decreases by
61.90%. The measured integration time decreases by 34.72%. These are single runs
with different material parameters and a historical timing reference, rather than
an isolated particle-count benchmark. The integration still executes all 336,000
steps. `shape_audit` uses every initial slice particle and removes a proper rigid
transform before measuring deformation; the world-Z height metric includes any
slice rotation.

The complete 203-frame physics run passes the existing aligned audit thresholds.
Every particle remains active and finite, only the knife separates CPIC transfers,
the sampled blade-interior side-change count is zero, and no butter particle reaches the
domain boundary. The sampled bread/board intrusion is 0.287 mm, within the existing
0.8 mm bread-sampling threshold. The complete particle video retains all 203 frames.

Reducing samples changes boundary quadrature: sampled butter volume is 1.61% lower.
This is an approximate optimization. Increasing material firmness also deliberately
changes the physical result: the coating spreads less and the bread deforms more.
The slice retains finite deformation; this preset targets a firmer rectangular
appearance. A 3000 Pa screening case failed the original 10 mm spread-growth
criterion (6.95 mm), so that stronger setting is excluded from the offered preset.

Use the commands in [the README](README.md#lower-particle-count-and-a-firmer-cut),
including `--material` during both preparation and execution. The 240 Pa asset
default and original sampling remain selectable for reference comparisons.

### Same-material sampling check

`out/firm-balanced-fine-cut` repeats the first 1.5 seconds with the same 80 kPa /
1200 Pa material and 0.4 mm particles. The complete prepared tool motion and
effective material JSON match exactly. Both resolutions retain the same grid,
timestep, contact-shell thickness and bread sampling. Downstroke deformation RMS
is 0.385 mm at 0.4 mm sampling and 0.447 mm at 0.6 mm sampling, a 0.062 mm
difference. Slice p01–p99 extents differ by +0.270, -0.131 and +0.033 mm in X/Y/Z.
These quantify a cut-stage quadrature comparison; they provide no full-trajectory
particle-position error bound or proof of resolution independence.

`optimization-validation.json` in the complete run records these measurements and
their scope. The uncompressed 203-frame trajectory occupies 2.447 GB. Focused
alignment/preparation checks and rigid-fit measurement checks pass (11 tests);
the touched Python files pass Ruff. The rejected stronger-material run and the
partial material screenings remain separate local artifacts.

### Complete Isaac replay

The 203 reconstructed surfaces use the original 0.3 mm bulk and 0.1 mm supported
coating voxels. Reconstruction took 299.05 s. Full Isaac replay took 560.88 s at
verified 128 samples, 1280 x 720 and 24 fps. Both particle and Isaac videos decode
to 203 nonempty frames; simulation, surface and rendering reports agree on the
trajectory hash. `final-validation.json` passes. Rendering time varies with the
new deformed geometry; the measured speed benefit here comes from physics and
smaller trajectory storage.

Visual inspection of cutting, carrying and final coating confirms that the stock
retains its block shape and the slice remains attached through transport. The
slice still has visible edge bending, so the reference's perfectly rigid cuboid
appearance is **not achieved** by this constant-material preset. The final film
has a thicker starting ridge and less spread than the softer reference. This
limitation remains explicit alongside the numerical improvements.

## Fixed firm-smooth material and bounded film gap repair

The fixed `asset_firm_smooth.json` preset retains G=80 kPa and K=150 kPa, with
yield=600 Pa and consistency=25 throughout. The measured full run took 558.31 s of
physics stepping, with 99.95% pickup, 80.96% deposition and 76.50 mm spread span.
Cut deformation RMS was 1.241 mm and lifted-slice RMS was 1.860 mm. This trades a
small amount of slice shape retention for easier spreading; it never switches
the material during the trajectory.

The final supported coating reconstruction additionally redistributes surplus
display density into enclosed sampling gaps with nearby particles. In the
recorded final density projection, 13 enclosed gaps (28.66 square mm) become zero;
the integrated density is preserved to less than 1e-11 relative change. The
combined final mesh volume increases by 0.0165%. Three-dimensional channels and
open boundary gaps remain. This is display regularization, not a change to
physical mass or a proof that all visible gaps are artificial.

These firm-smooth optical checks cover five Isaac frames at 128 samples. The
203-frame Isaac measurements in the preceding section belong to the earlier
1200 Pa, 25 microsecond preset. Do not attribute those full-video timings to the
latest material or reconstruction. See [standalone instructions](STANDALONE.md)
for reproducing the current bundled configuration.
