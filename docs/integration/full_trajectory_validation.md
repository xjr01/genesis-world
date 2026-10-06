# Five-scenario full-trajectory validation

This record covers the normalized scenarios exposed to UniRoboSim-Genesis. A **full trajectory** means running the
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
| Table wiping | `python -m examples.multiphysics.table_wiping.run` | 5,000 | 10.000000 s | FINAL RERUN QUEUED |
| Garment folding (half fold) | `python -m examples.multiphysics.garment_folding.run --task half` | 801 | 16.020000 s | PASS |
| Coupled litter scooping | `python -m examples.multiphysics.litter_scoop.run` | 766 | 12.766667 s | FINAL RERUN QUEUED |
| Coffee pouring, stirring and cleanup | `python -m examples.multiphysics.coffee_water.run` | pending | pending | RUNNING |
| Butter spreading | `python -m examples.multiphysics.butter_spreading.run` | 114,287 | 4.000045 s | PENDING |

The earlier table run completed successfully in about five minutes. Its controller was subsequently changed to keep
task time in ControllerState for checkpoint continuation; the equivalent final-code 5,000-step rerun is queued after
the checkpoint validations and must finish before this row returns to PASS. The earlier litter run took about 104
minutes and simulated 373,219 DEM particles with 19,064 FLIP particles. Its final-code rerun is queued because DEM
checkpoint fields participate in the scenario's initial-state registration. Coffee, butter, final table and final
litter must not be marked PASS until their default commands reach their built-in terminal assertions and exit
successfully.

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

After the full-trajectory GPU queue is empty, paired runtime checkpoint/reset is exercised in separate Genesis
processes (Genesis initialization is process-global):

```powershell
python scripts/validate_multiphysics_checkpoint.py coffee_water
python scripts/validate_multiphysics_checkpoint.py table_wiping
python scripts/validate_multiphysics_checkpoint.py litter_scoop
python scripts/validate_multiphysics_checkpoint.py garment_folding
python scripts/validate_multiphysics_checkpoint.py butter_spreading
```

Each command advances the real scene, snapshots both Scene and controller state, advances again, restores the pair
with `scene.restore()` plus `controller.set_state()`,
continues, then performs a separate restart from the initial Scene state plus `controller.reset(runtime)`. Results will
be recorded here after those commands actually complete.

After the final-code table and litter reruns, the validation queue runs the complete focused regression list, Ruff,
`compileall` and `git diff --check`; `out/full_trajectory_validation/final_regression.exitcode` is the terminal
machine-readable gate for this validation batch.
