# Butter cutting and spreading validation

## Scope and reproduction

The example contains a procedural cut-and-spread scene and an optional authored-
asset cut-and-lift scene. The authored asset measurements below use supplied
123.24 x 25.94 x 39.17 mm butter, porcelain support and palette-knife geometry.
These meshes are user inputs and are not distributed in this repository.
The simulation and particle-video tools require the repository dependencies;
Blender is used only for export. Isaac rendering is a separate local validation
step whose code and dependencies are outside this solver submission.

The asset preset is yield stress 160 Pa and consistency 15 Pa s^0.5. The primitive
scene retains the standard material unless a preset is supplied. See
[MATERIAL_CALIBRATION.md](MATERIAL_CALIBRATION.md) for self-support, flow fitting
and the distinction between an engineering fit and measured material properties.

```shell
python -m pytest -n 0 --backend gpu tests/particles/test_butter_cut_spread.py -x -v
python -m examples.butter_cut_spread.pipeline --run out/spread-check --material examples/butter_cut_spread/asset_material.json
python -m examples.butter_cut_spread.asset_demo --assets out/initialized --run out/asset-cut
python -m examples.butter_cut_spread.asset_particles --run out/asset-cut
python -m examples.butter_cut_spread.asset_audit --run out/asset-cut
python -m examples.butter_cut_spread.material_audit --run out/asset-cut
```

Prepare `out/initialized` using the authored-asset commands in the README. Use a
new output directory for each run. First-time high-resolution mesh distance-field
construction can take much longer than a cached build.

## Focused regression checks

Seven cases pass on the clean integration branch based on `merge` commit
`1f97655a`, on Windows with Python 3.11.15, Quadrants 1.2.0 and an RTX 5090 D v2.
They cover:

- Short-side swept coverage with an asymmetric knife and offset butter center.
- Ordinary concave-support free fall, support contact and friction, together with
  blade-only CPIC separation, in unbatched and two-environment scenes and after reset.
- Prescribed motion, blade geometry and angular-velocity consistency.
- Two-sided wet contact, bounded adhesion, release and unilateral correction.
- Authored asymmetric tool poses matching native collision geometry.
- Detection of intermediate-frame particle-ID changes in trajectory comparisons.

The support regression fails its free-fall assertion when all rigid geometries
are allowed to participate in CPIC separation. The trajectory coverage regression
rejects the off-side entry and incomplete sweep that left the upper section untouched.
The checks assert physical or geometric observables, not only successful execution.

## Complete authored-asset cut and lift

The tool-only CPIC adapter retains ordinary dish and countertop contact. Imported
steel uses its authored frame (`align=False`), and the post-step correction reads
that steel mesh's native signed distance field. The fixed tool orientation is an
assumption of this correction; rotating trajectories need separate previous and
current contact frames.

Entry is positioned over the full short side. The lowest point of the actual
steel section starts 1 mm beyond the far boundary and traverses to 1 mm beyond
the near boundary, then the blade returns to the short-side center for pickup.
For the measured geometry, entry and draw-end Y are +0.0183 and -27.9201 mm.
The purely vertical press-and-lift alternative has not been validated.

The geometric sweep covers all 6,135 sampled cutting-section points at 0.4 mm
spacing. The unswept continuous-mesh area is 0.0207%, confined below the first
particle-center layer where support clearance is retained. All 71,350 vertices
of the five tool meshes were checked at 25 near-support poses; minimum vertical
clearance above the dish is 0.2000 mm. Replacement assets need their own coverage
and clearance checks.

The complete run uses 160,000 steps at 25 microseconds, 1,883,802 butter particles
and 97 saved frames over four seconds. Initial particles, assets, material, grid,
step size, X/Z tool motion and orientation match the material-calibration control;
only the short-side tool path changes.

| Measurement | Result |
| --- | ---: |
| Final core height loss | 0.5061% |
| Initial slice particles above the original butter top | 97.7955% |
| Corresponding material-control pickup | 88.1880% |
| Active and finite particles | All |
| Particles reaching domain boundary | 0 |
| Dish penetration in 5,011 fixed sampled IDs | 0 mm |
| Maximum blade penetration among all saved-frame AABB candidates | 0.02872 mm |
| Countertop/dish CPIC-separated transfers at checked phase boundaries | 0 |

A small near-side residue remains on the end face. Geometric cutting coverage
and adhesive pickup are different measures. These checks at saved times do not
prove continuous-time nonpenetration or complete removal of every slice particle.

| Stage | Wall time (s) |
| --- | ---: |
| Initialization/build | 21.03 |
| Settle | 126.53 |
| Press | 191.46 |
| Draw cut | 237.34 |
| Separate | 64.72 |
| Lift | 170.85 |
| Hold | 63.06 |
| All physical steps | 853.95 |
| Simulation including initialization and export | 946.83 |
| Offline exact-mesh audit | 279.63 |
| Surface reconstruction | 361.56 |
| Optional Isaac rendering, 128 samples, including startup | 532.03 |

Audit and reconstruction ran concurrently; their times are not a serial pipeline
benchmark. The particle and Isaac videos each decoded to 97 frames at 1280 x 720,
24 fps. Their simulation, audit, reconstruction and rendering reports identify
this particle archive SHA-256:
`b1d8c11f18f9fa7ede63d832b1276af4400ac35d816e439cd2a989fcdde22483`.
The renderer's appearance profile was fixed and uploaded meshes matched the
recorded surfaces. Entry, cutting, separation and lifted-slice frames were inspected.

## Complete spreading compatibility check

With the fitted asset preset explicitly selected, the procedural example completed
230,000 steps over 5.75 seconds, with 720,000 butter and 325,404 bread particles
and 139 recorded frames. Its existing audit passed:

| Measurement | Result |
| --- | ---: |
| Initial slice pickup | 99.857% |
| Initial slice in the final bread coating zone | 97.726% |
| Maximum bread displacement | 1.069 mm |
| Sampled interior blade-side switches | 0 |
| Maximum sampled interior blade penetration | 0.00000813 mm |

The coating zone is the bread footprint within +/-4 mm of its nominal top;
its population does not establish uniform film coverage. Edge flow is excluded
from the interior side-switch check. The authored-asset scene itself ends after
lifting and does not include bread.

The full trajectories were generated before integration onto `1f97655a` using
these example physics sources. The intervening remote changes affect the DEM
solver and its options; the MPM and rigid coupling sources used here are unchanged.
Clean-branch regression tests and entry-point smoke checks supplement those full
runs; they are not presented as a second full-trajectory benchmark.
