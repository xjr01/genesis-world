# Default 240 Pa material: full cutting and spreading

Validated on 2026-10-25 with the authored-asset horizontal-knife workflow.
`asset_material.json` now selects yield stress 240 Pa and consistency 15 Pa s^0.5.
This is a 20% yield increase over the intermediate 200 Pa screening preset.
Shear modulus 20 kPa, bulk modulus 150 kPa, density 900 kg/m^3 and flow exponent
0.5 are unchanged. These remain fitted demonstration parameters.

The full run uses the default material without a `--material` argument. The earlier
160 Pa and intermediate 200 Pa presets remain available as
`configs/asset_yield_160.json` and `configs/asset_firmer.json`. An explicit material
override takes precedence over the default. External callers must load the JSON
into the material constructor; shared engine and primitive-demo defaults are unchanged.

## Controlled full-scene comparison

The 160 Pa reference is `out/horizontal-cut-spread`; the new run is
`out/firmer-yield-240-full`. Initial butter/bread particle files, complete tool
motion, scene configuration and asset metadata match byte for byte. Only the
effective yield stress differs. Contact strengths, trajectory, time step, sampling,
collision geometry and rendering settings are held fixed.

Both runs simulate 8.4 s using 336,000 steps at 25 us, with 1,883,802 butter and
272,258 bread particles, recording 203 frames at 24 fps. The real butter dimensions
are about 123.24 x 25.94 x 39.17 mm, with a 10 mm end slice. All material points
remain in the same MPM scene through cutting, carrying and spreading.

| Measurement | 160 Pa reference | 240 Pa default |
| --- | ---: | ---: |
| Core height retained after 0.6 s settling | 99.4703% | 99.6481% |
| Slice deformation RMS after downstroke, 1.5 s | 3.328 mm | 2.244 mm |
| Slice deformation RMS after draw, 2.9 s | 3.973 mm | 2.346 mm |
| Slice pickup | 96.4040% | 99.9870% |
| Slice retained over bread work area after transfer | 96.4019% | 99.9870% |
| Slice deposited near bread after withdrawal | 79.7761% | 83.6808% |
| Deposited-particle X-span growth | 46.282 mm | 45.797 mm |
| Deposition beyond approach front + 5 mm | 27.3878% | 29.8697% |
| Sampled interior blade-side changes | 0 | 0 |
| Maximum saved-frame blade intrusion | 0.028034 mm | 0.027456 mm |
| Maximum sampled dish intrusion | 0 mm | 0 mm |
| Maximum sampled bread/board intrusion | 0.162365 mm | 0.202522 mm |
| Particles reaching domain boundary | 0 | 0 |

`aligned-audit.json` passes the existing thresholds without changes. All particles
remain active and finite, CPIC separates transfers only at the knife, and the
particle video contains all 203 frames. Shape comparison follows fixed initial
slice IDs, sampled every 20th ID, and removes a fitted proper rigid transform;
the remaining RMS measures deformation rather than surface rectangularity.
The cut-end value falls by about 41%, and the particle views show straighter sides
and less compression and trailing residue. The butter remains deformable.

Deposition means initial slice particles within the bread XY footprint and within
6 mm of final bread particles. It is not an exact surface-contact volume or a
uniform-coverage measurement. Span growth uses the same finally deposited IDs
at approach and completion. Saved-frame contact checks cannot exclude transient
penetrations between frames. This is one matched comparison, not a statistical
repeatability study or validation of an unknown external trajectory.

The firmer butter loads the bread somewhat more: bread displacement median / 95th
percentile / maximum changes from 0.959 / 2.595 / 4.218 mm to
1.115 / 2.960 / 5.628 mm. Mean downward displacement changes from 0.617 to 0.642 mm.
Bread parameters and the knife's prescribed penetration are unchanged.

## Timing

| Stage | Seconds |
| --- | ---: |
| Initialization/build | 21.51 |
| Settle | 143.67 |
| Downstroke | 221.04 |
| Bottom draw | 350.47 |
| Lift start | 37.20 |
| Lift | 158.15 |
| Transfer | 364.28 |
| Approach | 197.59 |
| Spread | 373.79 |
| Withdraw | 146.35 |
| Hold | 72.97 |
| All physical stepping | 2,065.75 |
| State assembly/export | 168.88 |
| Initialization through export | 2,256.13 |

These are local elapsed measurements, not a performance guarantee. Offline
preparation, particle-video generation, independent audits, reconstruction and
Isaac rendering are outside the initialization-through-export total.

## Reproduction and outputs

Prepare assets on the target machine after changing the default material; the
preparation manifest intentionally rejects stale source hashes. Existing compact
collision geometry can be retained while refreshing preparation:

```shell
python -m examples.butter_cut_spread.asset_demo --assets out/aligned-prepared --run out/aligned-prepared --prepare-only
python -m examples.butter_cut_spread.asset_demo --assets out/aligned-prepared --run out/default-cut-spread
python -m examples.butter_cut_spread.asset_particles --run out/default-cut-spread
python -m examples.butter_cut_spread.aligned_audit --run out/default-cut-spread
python -m examples.butter_cut_spread.asset_audit --run out/default-cut-spread
```

The validated local prepared directory is `out/firmer-yield-240-prepared`.
The completed physical trajectory has SHA-256
`e9e410ca4e5dbb538d1309996c38216657efb892d1854f9d838fe3f6cce4a76c`.
`cut-material-comparison.png` compares downstroke, cut completion and lift with
the reference. `comparison-*.json` and `bread-comparison.json` retain the numerical
checkpoint comparisons. Generated trajectories, assets and media are local outputs.

## Complete rendering and final verification

`final-validation.json` passes. Both `aligned-cut-spread-particles.mp4` and
`aligned-cut-spread-isaac.mp4` contain all 203 frames at 24 fps and 1280 x 720.
The simulation, physical audits, reconstruction and rendering report the same
trajectory SHA-256 above. Appearance profile SHA-256 remains
`11103a5c30a3472fa17f6d1e75b4a9ac7bbaf88e585996a46ab676b26ac83a75`;
The requested 128 samples, camera, four lights and coating voxel size match the reference.
Reconstruction took 747.79 s; rendering process time totaled 928.60 s.
These are additional to the 2,256.13 s simulation/export time.

The first Isaac process returned an empty RGB image at index 186 after eight
acquisition attempts. Its first 186 encoded frames were complete and nonempty.
A fresh process rendered recorded states 186 through 202 with the same renderer,
geometry, camera and appearance. FFmpeg joined both segments by stream copy;
all 203 resulting frames were decoded and checked. No physics, surface geometry,
frame poses or appearance parameters were altered for recovery. Per-frame timings
from the failed process were not persisted and are recorded as null, not zero.
The rendering total includes the failed acquisition and fresh-process startup,
but excludes investigation and concatenation. Logs and segment files remain in
the output directory; the renderer metadata records this recovery explicitly.

Visual inspection of initial, pickup, transfer, approach, spread and final frames
shows the slice staying together during transport and forming a continuous butter
strip on the bread after withdrawal. The strip retains thicker start/end ridges;
this is not a claim of perfectly uniform coverage. Frames around the segment join
were also inspected without an obvious pose or color discontinuity.
`validation-contact-sheet.jpg` and `render-recovery-seam.jpg` retain these views.

The historical renderer recorded the requested sample count, without reading it
back after stage initialization. A subsequent reproduction of that initialization
path observed a reset to 64 effective samples. The current rendering entry reapplies
and verifies the requested count after initialization; see [performance](PERFORMANCE.md).
The effective count was not recorded for this historical video.
