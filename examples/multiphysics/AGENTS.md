# Multiphysics scenario migration rules

These rules apply to every file under `examples/multiphysics`. Follow the repository root `AGENTS.md` as well.

## Required module shape

Each scenario package exposes these public pieces from `__init__.py`:

- frozen dataclasses for `SolverConfig`, `MaterialConfig`, `Assets` and `TaskConfig`;
- one frozen `ScenarioConfig` that owns those groups through `default_factory`;
- `build_scene(config, ...)` returning a frozen, named runtime dataclass;
- a controller with `step(runtime)` when the task prescribes motion;
- optional typed boundary or contact config when those values form a distinct physical model.

Use `config.py`, `scene.py`, `task.py` and `run.py`. Add another module only for a cohesive implementation such as a
contact law or asset-boundary construction. `run.py` is a thin command-line client of the same API an adapter uses.

## Configuration ownership

Put a value in exactly one group:

- Solver: global discretization, time step, bounds, iteration counts and numerical capacities.
- Material: per-entity constitutive and contact properties such as density, stiffness, friction and viscosity.
- Assets: file paths, dimensions, initial transforms, semantic points and rendering geometry.
- Task: controller timing, targets, clearances, phases and termination.

Keep derived values as properties or compute them during construction. Never paste a calculated particle radius,
reference volume, world-space landmark or task height into configuration when its source parameters already exist.

Task motion uses an asset-local frame. Convert it to world coordinates through the configured asset pose and scale.
Changing an asset pose, size or scale must update dependent task targets without changing solver parameters.

## Adapter boundary

The scenario and controller use public Genesis APIs. Store named entity and solver handles during construction. Never
index `scene.sim._solvers`, read solver Quadrants fields, use private particle offsets, assume environment zero, or
require an adapter to call a scenario-internal physics function.

Register moving-boundary synchronization and scenario contact laws with `scene.register_pre_step_callback`. After
`build_scene()`, an adapter calling ordinary `scene.step()` must receive all required physics updates.

Public getter and setter calls operate on all environments together. Preserve the leading batch dimension and test
both unbatched and multiple-environment inputs for any new runtime computation.

Reusable material models belong under `genesis.engine.materials` and are exported through `gs.materials`. Reusable
solver construction and validation belongs under `genesis.integrations`. Calibrated task trajectories, assets and
scenario-only contact setup stay in the scenario package.

## Migration workflow

1. Read the complete source example and list every numerical value by owner: solver, material, asset, task or contact.
2. Identify engine changes the example requires. Add the smallest typed public engine capability before normalizing
   the scenario.
3. Implement the config groups and named runtime.
4. Build the scene without starting a control loop, viewer, recording or offline rendering.
5. Move task motion into a stepwise controller. Keep rendering and recording as optional clients.
6. Make asset transforms drive semantic points, boundaries and controller targets from one source of truth.
7. Export the scenario API, add it to `examples/multiphysics/README.md`, and update the integration contract when the
   adapter surface changes.
8. Validate configuration ownership, public batched state access, asset transform propagation and physical behavior.

Research scripts may remain in their subject directory for calibration and comparison. The normalized package is the
only adapter-facing construction path and contains no imports from a sibling checkout or runtime path manipulation.

## Validation record

Run the smallest useful checks while developing, then record all applicable results in the task report:

- syntax compilation for the scenario and touched engine modules;
- focused unit tests asserting configuration and observable behavior;
- a headless one-step scenario smoke test;
- the complete default trajectory on supported hardware;
- state and reset checks for solver-state changes;
- `git diff --check` and the repository lint checks.

Do not call a scenario physically validated when only imports, configuration tests or one-step execution have passed.
If hardware or an optional backend blocks the complete run, report the exact command and missing requirement.
