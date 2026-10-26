# Firmer material screening

The optional `configs/asset_firmer.json` raises yield stress from 160 to 200 Pa.
Consistency stays at 15 Pa s^0.5; all other material parameters, contact settings,
initial particles and prescribed tool poses are unchanged. This is a fitted
candidate for retaining cut shape, not a measured material or a change to defaults.

On 2026-10-25, a 3.7 s cut/lift screening run used 606,114 particles and 148,000
steps at 25 us. It retains the real slice dimensions, tool, support and 0.4 mm
particle spacing, but shortens the distant butter stock to about 40 mm and omits
bread. The earlier 160 Pa run used the same initial particles and poses, verified
by exact array comparison. Only effective yield stress differs between the runs.

| Measurement | 160 Pa | 200 Pa |
| --- | ---: | ---: |
| Slice deformation RMS at end of downstroke (1.5 s) | 3.565 mm | 2.144 mm |
| Slice deformation RMS near end of draw (2.8333 s) | 4.006 mm | 2.653 mm |
| Slice pickup at lift completion (3.7 s) | 98.223% | 98.988% |

Deformation is the residual after fitting a proper rigid transform from the same
initial slice particle IDs to their current positions, sampled every 20th ID.
It removes translation/rotation, but measures material-point deformation rather
than surface rectangularity. Pickup counts initial slice particles more than 5 mm
above the initial butter top. Both runs have finite coordinates; all candidate
particles remain active. The side-by-side particle views show less compression of
the candidate slice, while its shape still deforms under the knife.

Candidate initialization took 56.33 s, stepping 561.90 s, and total runtime with
state export 623.48 s. These single-run timings are not a performance benchmark.

Local artifacts in `out/firmer-yield-200-screen`:

- `firmer-comparison.mp4`: 160 / 200 Pa side-by-side, 24 frames at 6 fps, 1280x720.
- `firmer-comparison-lift.png`: both runs at the end of lift.
- `firmer-comparison.json`: per-stage measurements and their exact sample times.
- `material.json` and `simulation.json`: effective parameters and run status.

This screening does not validate full-size stock, spreading, an external tool
trajectory or the user's screenshot. The external run's effective material is
unknown; first check whether it used the engine's 80 / 25 default or the asset
preset's 160 / 15. See `MATERIAL_CALIBRATION.md` for explicit loading and logging.
No Isaac render or full cut-and-spread validation was performed for the 200 Pa
preset. This is the historical intermediate screening; the current authored-asset
default is 240 Pa; see [complete default-material validation](DEFAULT_MATERIAL_VALIDATION.md).
