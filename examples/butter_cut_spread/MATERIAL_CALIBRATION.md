# Calibrating an imported butter asset

Keep the authored meter-scale geometry, placement, contact representation, grid,
particle sampling, time step, and tool trajectory fixed while comparing material
parameters. Use the Herschel-Bulkley material already supplied by this repository.
Blender surface shaders do not specify its mechanical properties.

Pass a partial `ButterParameters` JSON through `--material` to either `asset_demo`
or `demo`. Each output directory stores the full effective `material.json`.
The real-asset `asset_demo` defaults to `asset_material.json` (240 Pa / 15 Pa s^0.5).
The primitive `demo` and `pipeline` retain the standard example material unless a
preset is selected explicitly. Shared material and engine defaults are unchanged.
For `asset_demo`, a partial custom file overrides the asset preset, so changing
only consistency retains yield stress 240 Pa. For the primitive `demo` and
`pipeline`, omitted fields use the standard `ButterParameters` defaults. Pass the
full `asset_material.json` to use the fitted material there.

External trajectory integrations must also pass the selected parameters when
constructing `gs.materials.MPM.HerschelBulkleyButter`. Loading an asset mesh alone
does not select this preset, and the engine's default yield stress remains 80 Pa.
Read the supplied JSON with `ButterParameters.model_validate_json`, then expand
`parameters.model_dump()` into the material constructor alongside the simulation
`dt`, sampler and particle size. Save the effective parameters with each dataset
trajectory so that the same mesh cannot silently acquire different mechanics.

```json
{
  "yield_stress": 240.0,
  "consistency": 15.0
}
```

`yield_stress` is in Pa. `consistency` is in Pa s^n, with the unchanged exponent
`n = 0.5`; it is not a Newtonian viscosity in Pa s. Density remains 900 kg/m^3,
shear modulus 20,000 Pa, and bulk modulus 150,000 Pa. The selected preset is an
engineering fit for the asset described below, not an experimentally measured
material. Other dimensions, temperatures and trajectories need their own checks.

## Slightly firmer slices and external trajectories

First inspect the material actually passed to `HerschelBulkleyButter`. The engine
constructor defaults to yield stress 80 Pa and consistency 25 Pa s^0.5; the asset
demo explicitly loads 240 / 15. Importing the scene adapter, loading a mesh, or
copying the preset beside an external program does not load that preset. An image
alone cannot distinguish a missing preset from a material that is too soft.

`configs/asset_firmer.json` is an explicit, complete six-parameter preset with
yield stress 200 Pa and consistency 15 Pa s^0.5, retained for intermediate comparisons.
The current 240 Pa default is 20% higher than this intermediate setting and 50%
higher than the earlier 160 Pa preset (`configs/asset_yield_160.json`). Only yield
stress changes between these presets. This raises the threshold for
permanent shape change; elastic moduli, density, flow exponent, contact and
trajectory settings stay fixed. It is a fitted option, not a measured butter
property. Higher yield can preserve cut edges but also hinder spreading and
release; validate it on the actual external motion before dataset production.

Use the default without a material override. For comparison, select a lower-yield
preset explicitly with the material flag, not the scene-configuration flag:

```shell
python -m examples.butter_cut_spread.asset_demo --assets out/aligned-prepared --run out/firmer-cut-spread --material examples/butter_cut_spread/configs/asset_firmer.json
```

For external integration, load the complete preset and save the effective material
from the constructed instance. `dt` and `particle_size` below must be the actual
simulation values; the example values match the calibrated asset workflow.

```python
from pathlib import Path

import genesis as gs
from examples.butter_spreading.config import ButterParameters

parameters = ButterParameters.model_validate_json(Path("examples/butter_cut_spread/asset_material.json").read_text())
material = gs.materials.MPM.HerschelBulkleyButter(
    **parameters.model_dump(),
    dt=0.000025,
    particle_size=0.0004,
)
effective = ButterParameters.model_validate(
    {name: material.model_dump()[name] for name in ButterParameters.model_fields}
)
Path("effective-butter-material.json").write_text(effective.model_dump_json(indent=2))
print(effective.model_dump_json(indent=2))
# Pass this same material instance to scene.add_entity(material=material, ...)
```

Log the effective values before comparing external runs. An 80 / 25 run has not
loaded any asset preset; compare it explicitly with the selected 160 / 15,
200 / 15 or default 240 / 15 material, rather than assuming that copying a JSON
changed the mechanics. Keep adhesion, tool motion, asset size and numerical
resolution fixed for that comparison.

The local cut/lift comparison is recorded in [firmer material screening](FIRMER_VALIDATION.md).
It supports reduced cut deformation and retained pickup on that trajectory, but
does not establish acceptable spreading on an external trajectory.
The current 240 Pa default has a separate
[full authored-asset validation](DEFAULT_MATERIAL_VALIDATION.md). Do not substitute
the 200 Pa reduced-stock screening for that full-scene result.

## Self-support before flow

Use the actual dish with ordinary contact and a stationary blade clear of the butter.
Only the blade is selected for CPIC cutting. Preserve gravity and support friction.
Do not lock the butter, remove gravity, scale down its height, or increase viscosity
just to hide a support failure.

```shell
python -m examples.butter_cut_spread.asset_demo --assets out/assets --run out/static-check --material material.json --stationary --duration 3 --fps 4
python -m examples.butter_cut_spread.material_audit --run out/static-check
```

`--fps` changes recording frequency only. Integration still uses the asset's `dt`.
The audit follows the same initial central particle IDs, measures the 1st-to-99th
percentile height, and fits the height-loss slope over the second half of the hold.
The engineering screen is at most 1% height loss and an absolute late slope at most
0.1 percentage points/s. This allows a small elastic compression but rejects visible
ongoing settling. Start with a short sweep, then repeat a passing candidate for at
least the intended motion's waiting period. A few seconds cannot establish stability
over minutes or at another temperature.

Increase yield stress first, keeping consistency fixed. Select a value that passes
with some margin; a value exactly on the threshold may fail after changing the
initial orientation, height, support, or grid. Repeat the hold after choosing the
final consistency to check that viscosity was not merely delaying creep.

For the evaluated 123.24 x 25.94 x 39.17 mm asset on its actual porcelain support,
the following independent holds used the same 0.4 mm particles, 768/m grid,
25 microsecond step, and unchanged contact settings:

| Yield (Pa) | Consistency (Pa s^0.5) | Hold (s) | Core height loss | Late drift (percentage points/s) | Screen |
| --- | --- | --- | --- | --- | --- |
| 120 | 25 | 0.6 | 1.960% | +1.325 | Fail |
| 160 | 25 | 0.6 | 0.472% | +0.045 | Pass, short hold |
| 160 | 15 | 3.0 | 0.496% | -0.014 | Pass |

The negative final slope is a small recovery rather than continuing settling.
These results bracket a useful support setting; they do not identify the minimum
possible yield stress or a universal butter material. The softer 80 Pa reference
lost about 12% of central height in its first 0.583 seconds before cutting.

## Flow after yielding

At fixed yield stress, compare consistency values with identical cutting and
spreading trajectories and unchanged adhesion. Larger consistency slows plastic
flow under a given excess stress; smaller consistency speeds it up. It does not
determine the velocity of a prescribed knife.

The steady simple-shear interpretation is
`tau = yield_stress + consistency * shear_rate^n` after yielding. Thus doubling
consistency at `n=0.5` reduces the rate at a given excess stress by a factor of four.
This is a constitutive interpretation, not a prediction of the full cutting motion:
normal compression, changing contact area and adhesion also influence that motion.

For example, at an illustrative shear rate of 50 /s, the model gives about 257 Pa
for yield 80 Pa / consistency 25, 337 Pa for 160 / 25, and 266 Pa for 160 / 15.
The lower-consistency candidate can therefore raise the static threshold while
retaining a similar resistance at faster deformation. The 50 /s rate is an
illustration, not a measured rate from the reference video. Full motion comparisons
are still required because the stress field is neither uniform nor prescribed.

Two complete 4-second asset cuts used identical initial particles, geometry,
contact settings, prescribed poses and recording times. At the end of the hold:

| Yield (Pa) | Consistency (Pa s^0.5) | Core height loss | Initial slice IDs above original top |
| --- | --- | --- | --- |
| 80 | 25 | 17.638% | 78.066% |
| 160 | 25 | 0.550% | 86.220% |
| 160 | 15 | 0.592% | 88.188% |

These earlier experiments selected 160 / 15 as an engineering preset: it passed the longer support
hold, remains more compliant after yielding, and gave slightly more pickup in this
single comparison. This is not a claim of a unique optimum or statistically proven
pickup improvement. Neither candidate picks up the whole initial end slice.
Changing viscosity alone does not guarantee complete cutting, adhesion or release.

These material comparisons use one fixed control trajectory. A later asset
trajectory separately corrected short-side cutting coverage and reached 97.7955%
pickup with the same 160 / 15 preset; see [validation](VALIDATION.md). This change
in pickup is a trajectory comparison, not another material calibration result.

For the selected run, all particles remained finite and active, and none reached
the domain boundary. An exact-mesh audit at 97 saved times found zero dish intrusion
in a fixed 5,011-particle sample and maximum blade-center intrusion of 0.0307 mm
among all particles inside the blade bounding box. These are saved-time checks,
not a proof of continuous-time nonpenetration.

Inspect slice pickup, deformation relative to the tool, retained bulk shape,
spreading footprint and deposition. `material_audit` reports finite-difference
slice speeds in the moving tool frame; they include elastic motion and contact
slip, so they are not a direct measurement of plastic shear rate. The complete
primitive cut-and-spread example provides a separate spreading compatibility check:

```shell
python -m examples.butter_cut_spread.demo --run out/spread-check --material material.json
python -m examples.butter_cut_spread.particles --run out/spread-check
python -m examples.butter_cut_spread.audit --run out/spread-check
```

To reproduce the earlier primitive compatibility run with its 160 Pa preset:

```shell
python -m examples.butter_cut_spread.pipeline --run out/spread-fitted --material examples/butter_cut_spread/configs/asset_yield_160.json
```

The earlier 160 / 15 preset also completed the full 5.75-second primitive
cut-and-spread operation: 230,000 steps, 720,000 butter particles, 325,404 bread
particles, and 139 recorded frames. Its existing acceptance checks passed:

| Check | Measured result |
| --- | --- |
| Initial slice pickup | 99.857% |
| Initial slice deposited in the bread coating zone | 97.726% |
| Maximum bread displacement | 1.069 mm |
| Sampled interior blade-side switches | 0 |
| Maximum particle-center blade intrusion | 0.00000813 mm |

The coating zone is the bread footprint within +/-4 mm of its nominal top; this
fraction is not a measurement of uniform film coverage. This is a compatibility
check in the separate primitive scene, not a claim that the imported knife/dish
scene includes bread or attains the same pickup fraction. The real-asset dimensions
and trajectory remained unchanged throughout their own calibration.

Keep visual material and lighting fixed when comparing movies. Inspect particle
motion before judging reconstructed surfaces. A visually good result does not by
itself identify material parameters: quantitative fitting needs known temperature,
physical motion timing, dimensions and preferably force/displacement measurements.
Without those, label the result an engineering fit, not measured room-temperature
butter properties. Spatial convergence and continuous-time collision checks remain
separate from this material calibration.
