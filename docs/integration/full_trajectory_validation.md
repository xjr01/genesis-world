# Five-scenario full-trajectory validation

This record covers the normalized scenarios exposed to UniRoboSim-Genesis. Current status was reconciled against
the completed run logs and local video files on 2026-10-07. A **full trajectory** means running the
scenario's default CLI without `--steps`; the one-step runs listed separately only verify construction and one physics
advance.

## Environment

- Date: 2026-10-05
- OS: Windows
- GPU: NVIDIA Quadro RTX 8000, 48 GB
- Python: 3.11
- Genesis: 1.3.1 from this branch
- PyTorch: 2.6.0+cu124
- Quadrants: 1.2.0
- pyuipc: 0.0.28

PyTorch prints a compatibility warning because current Genesis recommends PyTorch 2.8 or newer. The runs below still
use CUDA successfully. Garment folding additionally requires the optional `pyuipc` package.

On a restricted Windows workspace, Quadrants and Genesis caches and the temporary directory must point to writable
locations. Otherwise Quadrants can repeatedly retry `tempfile.mkstemp()` after `PermissionError`, which looks like an
indefinitely slow compile. The validation used:

```powershell
$cacheRoot = (Resolve-Path '.').Path + '\.runtime-cache'
$env:PYTHONUTF8 = '1'
$env:QD_OFFLINE_CACHE_FILE_PATH = $cacheRoot + '\all-five-q12\quadrants'
$env:GS_CACHE_FILE_PATH = $cacheRoot + '\genesis'
$env:TEMP = $cacheRoot + '\tmp'
$env:TMP = $cacheRoot + '\tmp'
```

## Full default trajectories

| Scenario | Command | Steps | Simulated time | Result |
|---|---|---:|---:|---|
| Table wiping | `python -m examples.multiphysics.table_wiping.run --record` | 5,000 | 10.000000 s | COMPLETE (CALIBRATED, NUMERICAL CHECKS PASS) |
| Garment folding (half fold) | `python -m examples.multiphysics.garment_folding.run --task half` | 801 | 16.020000 s | REVALIDATION REQUIRED |
| Garment folding (Scene527 checkpoint continuation) | Scene527 `--checkpoint`, omitted `--steps` | 8,738 additional; controller ends at 8,746 | 72.816667 s additional | COMPLETE (FINITE STATE; VISUAL ACCEPTANCE OPEN) |
| Coupled litter scooping | `python -m examples.multiphysics.litter_scoop.run --record` | 766 | 12.766667 s | COMPLETE (OPEN-SLOT MESH; VISUAL ACCEPTANCE OPEN) |
| Coffee pouring, stirring and cleanup | `python -m examples.multiphysics.coffee_water.run --record` | 11,140 actual; 14,000 upper bound | 22.280000 s actual | COMPLETE (BUILT-IN TERMINAL CHECKS PASS) |
| Butter spreading | `python -m examples.multiphysics.butter_spreading.run` | 114,287 | 4.000045 s | PENDING |

The calibrated table run completed on 2026-10-07 with its full 5,000-step default horizon, ten finalized video
segments and conserved particle count. See the table-wiping isolation record below. The final open-slot litter run
completed with 373,219 DEM particles and 19,064 FLIP particles. Completion and finite state certify numerical
execution; garment fold shape, litter sieving quality and physical fidelity require separate visual acceptance.
The lightweight garment profile and butter retain their separate pending validation status.

## 2026-10-07 correction batch

### Table-wiping parameter isolation

The original mop client and the normalized client both formed a thick drop under the inherited parameters on this
Windows environment. Before sponge contact, their water maxima at 0.2 seconds were 50.00 and 52.42 mm. At 1 second,
both had captured all 840 particles. Their videos and measurements are in `out/table_wiping/source_parity_20261007`.
The PBSTF solver file at the pre-calibration integration commit, remote `merge` and remote `mopping-simplified` has
the same Git blob hash `b89cc26da03e7e274f33133897ed1cf5c7e49bd9`. This checks solver provenance; it does not recreate
the historical operating system, compiler or the rest of that engine revision.

With the sponge held stationary away from the water, increasing only surface-tension compliance from `1/225` to
`100/225` produced a spread with median/P90/maximum heights 7.91/12.78/16.27 mm and horizontal extent 150.57 by
122.19 mm at 0.4 seconds. Density, viscosity, resolution and liquid volume stayed fixed. Reducing density compliance
to one hundredth while retaining the strong surface constraint was rejected: particles hit the domain boundaries,
went below the table, and reached 8.47 m/s RMS speed. Both isolation runs retained videos under
`out/table_wiping/calibration_20261007`.

The normalized material now uses `100/225` surface compliance, `800` captures per second and `10 s^-1` inward-motion
rate. The two absorption rates have independent controls. Omitting `absorption_motion_rate` resolves it to the
capture rate, preserving the behavior of coffee and research clients. Time discretization and the frame-runner
contract remain unchanged. This is a scenario calibration rather than a physical surface-tension coefficient in
SI units. The 6.67 mm particle spacing also limits the thickness of a represented water film.

Four static/deformable, unbatched/two-environment absorption tests passed in 316 seconds. The first capture's
progress matches `1-exp(-5*dt)` while throughput remains 100 particles per second; moving-collider, wetness and
snapshot checks also passed. A separate-process fault probe deliberately tied inward motion to the capture rate:
the first-progress assertion failed at 0.095163 versus expected 0.004988. The ordinary analytic geometry and source
case-settings checks passed, as did 17 integration/controller/API checks. No fault injection modifies shipped code.

The first full-run attempt stopped before trajectory advancement because the recording client tried to stop an
unstarted camera. Its log remains in `out/table_wiping/calibrated_full_20261007`; the corrected recording client
tracks recording state and reruns in `out/table_wiping/calibrated_full_v2_20261007`.

The corrected full run completed 5,000 steps / 10 seconds. Build and simulation times were 118.604 and 300.847 seconds.
All 201 CSV samples were finite, with free plus captured particle count fixed at 840. Captures at 0.4/0.6/0.8/1.0/1.2
seconds were 96/256/416/576/736; all particles were captured by the 1.4-second sample. Mean inward absorption progress
was 0.723 at 1 second and 0.999997 at termination. Maximum water height before contact at 0.2 seconds was 17.73 mm,
compared with 52.42 mm in the inherited normalized configuration. Ten videos decoded successfully, each with 29 frames
at 29.4118 FPS: integer 17-step camera sampling approximates the requested 30 FPS at native dt 0.002 seconds. The
physics horizon is exactly 10 seconds; segment videos each cover approximately 0.986 seconds of displayed frames.
Inspected frames show a spreading water layer, progressive uptake and a clear table at the end. Visualization still
uses diagnostic particles and a tetrahedral sponge; this record validates the calibration behavior and numerical
checks rather than a measured physical water-film or porous-media model.

The earlier lightweight half-fold PASS is historical: shell energy conventions now match the inherited material
calibration, so that default needs a fresh full run. Scene527 completed its separate 60-step settling regression
and an eight-step checkpoint continuation, with videos and finite FEM state. Its full 8,746-native-step folding
trajectory completed from controller step 8 through step 8,746 with 73 segments. Its final median/P90/maximum heights
were 31.36/61.20/80.53 mm. This recorded continuation excludes the separate settling video and first eight trajectory
steps; a neat folded shape has not been certified.

Coffee completed 500 control steps (1 second), with 50 decoded video frames at 50 FPS. Active particle count,
mass and water-in-cup count remained 213,162, 0.5043063 kg and 142,312 respectively. Cold construction took
636.213 seconds and simulation took 1,997.870 seconds. The 28-second run restarted from zero with the warm
cache and took 102.585 seconds to construct. It completed at 22.28 seconds / 11,140 executed steps, when both arms
reached REST. All original terminal checks passed, including cup recovery, three stirring turns and nonzero water
absorption. The 23 videos contain 1,114 decoded frames. The historical completion message prints the 14,000-step
upper bound; metrics and scene time identify the actual early termination. The handoff cleanup makes the current CLI
report its executed count and moves that CLI into `coffee_water/run.py` without changing its physical update loop.

Litter now uses the matching open-slot mesh SDF rather than the analytic box proxy. Its default recorded run
constructed in 162.399 seconds and completed all 766 native steps / 12.766667 seconds with finite sampled sand/water
state. Its 13 videos contain 383 decoded frames under `out/litter_scoop/mesh_slots_default_20261007`. Numerical
completion covers insertion, lifting and hold; visual sand retention and sieve throughput remain open.

Focused configuration/controller/boundary/shell tests passed 32 cases. The CPU mesh-obstacle regression passed
both unbatched and two-environment cases: grains remain supported by a solid strip while grains above a slot fall
through, and moving obstacle pose/velocity survive Scene restoration. Disabling obstacle restoration deliberately
caused the position assertion to fail (100 micrometers versus the saved 50 micrometers); restoring the implementation
passed both cases again. An inactive FLIP solver now permits the standalone DEM multi-environment test; active
FLIP remains single-environment.

Scenario/test Ruff checks, syntax compilation and `git diff --check` passed. Whole-engine lint retains existing
diagnostics verified against HEAD: 30 in the IPC coupler, three in DEM and five in FLIP. No cache directories,
recordings or checkpoints belong in the correction commit; the paired 42 MB SDF asset does.

## Construction and one-step smoke tests

All five normalized packages built their real Genesis scenes and completed one physics step with their default
physical assets:

```text
table_wiping       1 step, 0.002000 s
coffee_water       1 step, 0.002000 s
litter_scoop       1 step, 0.016667 s
garment_folding    1 step, 0.020000 s
butter_spreading   1 step, 0.000035 s
```

The coffee smoke used liquid particles; it was not the motion-only mode. The default coffee scene contains 70,850
coffee particles and 142,312 water particles. The default butter scene contains 255,360 bread particles and 172,800
butter particles.

## Interface and controller regression

The focused public-scenario suite passed on the same checkout after adding the adapter clock and IPC reset dispatch:

```text
35 passed, 1 warning in 174.77 s
```

It covers controller checkpoint snapshots for all five scenarios plus the coffee, granular-fluid, garment and butter
scenario contracts. It also verifies 60 FPS native-substep distribution for all five time steps, frame-clock
checkpoint continuation and IPC checkpoint dispatch. `python -m compileall` passed for the normalized scenarios and
touched engine modules. Ruff passed for the normalized scenarios, focused tests and new typed state containers;
`git diff --check` also passed. A source search found no runtime use of `scene.sim`, solver-list indexing, private
particle offsets or `examples.multiflow` inside the five normalized implementations. The only search match was the
policy text in `examples/multiphysics/AGENTS.md` that explicitly forbids those dependencies.

An additional five-case public API contract test passed for the normalized packages. It pins each package's exported
scenario config, named runtime, controller, controller state and `build_scene(config, ...)` entry point, and requires
every controller to retain `reset`, `get_state`, `set_state` and `step` methods.

A six-case adapter-boundary regression also passed. It checks every normalized package for forbidden access to
`scene.sim`, `_solvers` and private particle ranges, and separately enforces that table wiping has no runtime import
from `examples.multiflow`.

The public `Scene.restore(state)` checkpoint operation has a focused core regression test. It verifies that checkpoint
restoration uses `keep_init=True` and resets recorders, so a later bare `Scene.reset()` still returns to the scenario's
registered task origin.

The IPC checkpoint path now stores and restores native FEM position/velocity, affine-body transform/velocity and
external-articulation continuation state through the public libuipc state-accessor features. A continuation regression
was added to `tests/ipc/test_deformable.py`. On this checkout its collection is currently blocked by the test utility's
missing `httpcore` dependency; the exact command fails before collecting the IPC test with
`ModuleNotFoundError: No module named 'httpcore'`.

Time-driven table and butter controllers store their task step in ControllerState rather than relying on the Scene
clock. DEM solver state also stores the scripted tilt-box position, orientation, linear velocity and angular velocity,
so a litter checkpoint resumes both particle/grid physics and shovel motion.

A minimal CPU DEM scene compiled the new checkpoint kernels and passed a real tilt-obstacle round trip: it saved the
obstacle pose and velocity, moved it for one step, restored the checkpoint, and recovered position, orientation,
linear velocity and angular velocity within `1e-6`.

Paired runtime checkpoint/reset is exercised in separate Genesis processes (initialization is process-global):

```powershell
python scripts/validate_multiphysics_checkpoint.py coffee_water --record
python scripts/validate_multiphysics_checkpoint.py table_wiping --record
python scripts/validate_multiphysics_checkpoint.py litter_scoop --record
python scripts/validate_multiphysics_checkpoint.py garment_folding --record
python scripts/validate_multiphysics_checkpoint.py butter_spreading --record
```

Each command advances the real scene, snapshots both Scene and controller state, advances again, restores the pair
with `scene.restore()` plus `controller.set_state()`,
continues, then restarts using bare `scene.reset()` plus `controller.reset(runtime)`. The handoff checker compares
public particle position/velocity and available tool/robot observables at the restored checkpoint and registered
initial state (absolute/relative tolerance `1e-6`), checks controller phase, and verifies finite continuation. Its seven
diagnostic video frames use 5 FPS for display, independently of the simulated interval.

## Adapter handoff cleanup

The 2026-10-07 handoff regression passed 41 configuration, exported API, controller-state, frame-timing, private-boundary
and Scene restoration tests in 182.06 seconds (`out/handoff_20261007/adapter-regression.log`). The exported API test
also checks frozen config groups, each documented controller snapshot signature and named runtime clock property.
Coffee CLI/recording/measurements now live in `run.py`; the original executable forwards to it. The moved CLI body
matches its pre-move body, with the executed-step reporting correction retained. Adapter callers keep their package
imports and existing controller signatures.

After the CLI move, seven coffee-entry/public-package checks passed again in 45.51 seconds
(`out/handoff_20261007/cli-final-regression.log`), including identity of the original executable's forwarded `main`.
Both coffee entry points also completed `--help`. Ruff check and format check passed across all normalized scenario
modules, the lifecycle checker and changed adapter tests; syntax compilation and `git diff --check` passed. Existing
engine-only lint diagnostics remain separately tracked rather than hidden by the scenario checks.

A separate-process API fault probe removed coffee's required runtime snapshot argument. The strengthened exported
API test failed at its snapshot-signature assertion (`out/handoff_20261007/public-api-fault.log`). Shipped controller
code was unchanged by the probe; the ordinary seven-case rerun passed with the documented signature.

The integration guide documents source-checkout deployment, single-environment builders, the exact call differences,
double-stepping prevention, native versus frame clocks, paired snapshots and limits of engine-side acceptance. The
scenario README records completed trajectories and the four combined local videos. Video/frame counts were checked
against all source segments and complete combined files decoded successfully. Runtime checkpoints and media stay in
ignored `out/`; share media separately from the engine commit.

### Real-runtime paired lifecycle checks

All runs below use unchanged default physical resolution with recording, seed 1 and the GPU backend. Each successful
video contains seven decoded phase frames. These are early-trajectory lifecycle checks, separate from full task
trajectories and late-stage task/absorption acceptance.

| Runtime | Paired restore, finite continuation, bare reset and restart | Log under `out/handoff_20261007` |
|---|---|---|
| Calibrated table | PASS | `table_wiping-checkpoint.log` |
| Lightweight garment | PASS | `garment_folding-checkpoint.log` |
| Coffee with liquid and deformable sponge | PASS | `coffee_water-checkpoint.log` |
| Open-slot litter | PASS | `litter_scoop-checkpoint-v2.log` |
| Butter | PASS | `butter_spreading-checkpoint-v2.log` |

The initial litter checker attempt stopped before native trajectory advancement because a DEM obstacle pose getter
returns a NumPy array and the checker assumed Torch. The checker now normalizes public observations with
`torch.as_tensor` before finite checks and clones the resulting snapshots. The original log is preserved in
`litter_scoop-checkpoint.log`; the corrected run passes without changing the engine getter's return type.
