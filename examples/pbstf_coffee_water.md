# Multi-fluid integration and coffee pouring

This merge integrates reviewed multiflow commit `2b2a890ce1833d9b1c7f45b389153f86347b9baf` into
`mopping-simplified`, starting at `846de604`. Both parent histories are retained. The integration brings PBD,
PBSTF and implicit position-based fluid (IPBF) concentration transport, container boundaries and liquid rendering
into the current engine. IPBSTF, unified PBD, absorbent/deformable PBSTF colliders and existing examples remain.

Solver `diffusion_coeff` defaults to zero. Materials accept `c_init`; PBD and PBSTF also accept `c_init_z_mid`.
Entities expose `get_particles_concentration` and `set_particles_concentration`. Concentration follows particle
reordering, environment selection, snapshots, resets and emitter slot reuse. Surface reconstruction interpolates
concentration over the combined liquid particles in each solver and environment. Repeated rendering at one
simulation time replaces the reconstructed surface so updated colors are visible.

The integration uses the current engine as its base. Upstream example/test deletions, unrelated engine replacements,
generated assets and experiment history counters are excluded. New option booleans use `is_` / `has_` names.

## Running the standalone example

From the repository root, with the `genesis-world` environment active:

```powershell
$env:PROCESSOR_ARCHITECTURE = 'AMD64'
$env:PYTHONPATH = (Get-Location).Path
$env:TEMP = Join-Path (Get-Location) 'tmp/multiflow/temp'
$env:TMP = $env:TEMP
$env:GS_CACHE_FILE_PATH = Join-Path (Get-Location) 'tmp/multiflow/cache/genesis'
$env:QD_OFFLINE_CACHE_FILE_PATH = Join-Path (Get-Location) 'tmp/multiflow/cache/quadrants'
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
python examples/pbstf_coffee_water.py --record
```

The environment settings resolve this Windows session's empty CPU architecture and restricted temporary/cache
directories. `PYTHONPATH` also ensures that worktree tests import the selected checkout.

The example defaults to CUDA, scale 750, particle diameter `2/750` m and 5000 steps. `--vis` opens the viewer;
`--surface` reconstructs the liquid surface at additional rendering cost; `--scale` changes spatial resolution;
`--steps` extends the final resting phase. Running without `--vis` is headless. `--record` writes a video, stage
images and state arrays under `out/pbstf_coffee_water/`; the metrics CSV is always written there.

The script generates two watertight open cup shells in a temporary asset directory. Their walls and gravity move
the liquid. The motion settles for 1 s, lifts and translates the water cup, tilts it to 115 degrees, then returns
it by 8.5 s. Each cup has space for the combined liquid volume. Scene construction, physical parameters, mesh
generation and cup motion are contained in the example file.

The timestep (0.002 s), gravity, domain, 10 solver iterations, topology rebuild interval 10, neighbor capacities
(128/64), and liquid density/compliance/viscosity/adhesion/friction settings match the current mop example. Initial
concentrations are coffee=1 and water=0. The example selects dimensionless diffusion strength 0.005; increasing the
step count allows further natural mixing and settling.

## Validation

The isolated pre-merge checkout imported its own `genesis` package. Its 42 particle tests produced 36 passes and
6 failures. Four failures stopped at stale mop/sweep configuration assertions. The current values are absorption
rate 4000, sponge grid (10, 7, 20), adhesion compliance 50 and closed finger position 0.036. Configuration assertions
were updated to these values; physical assertions and tolerances are retained. The remaining two baseline failures
are `test_dam_break_spreads_under_gravity[0-cuda]` and `[2-cuda]`: liquid y coordinates fall below the asserted floor.

Final coverage combines the full particle regression with focused reruns after the configuration updates:

| Capability | Result |
| --- | --- |
| PBSTF, absorption, deformable collision, adhesion, emitters, concentration and reset | 26 passed |
| PBD, unified PBD, attachment, rigid contact and analytic cup contact | 18 passed |
| IPBSTF | 2 passed; the same 2 baseline failures |
| Concentration color, merged reconstruction and PBD reconstruction radius | 5 passed |
| FEM/PBD textured rendering | 1 passed |

The full particle run is `regression-final.log`. The updated configuration, concentration and serialization checks
are in `final-focused.log`; the analytic cup checks are in `shell-exact.log`. A two-environment CPU comparison of
one default PBD floor-contact step gives exactly equal positions and velocities before and after the merge.

Analytic shell contact uses distance to the exposed cup cross-section, including the floor axis and the inner
floor-wall corner. Tests check projected positions and stopped inward velocity at the inner wall, outer wall,
floor and underside, before and after tilting the container. The incoming shell implementation fails these
velocity assertions. Exact binary contact coordinates and timestep keep the checks robust to finite differencing.

The default recorded pour completed all 5000 steps:

| Observable | Result |
| --- | --- |
| Active particles | 5056 throughout |
| Liquid mass | 0.10451473 kg throughout |
| Concentration sum | 2528, within 0.001 |
| Concentration variance | 0.25 initially, 0.023943026 finally |
| Initial water particles in coffee cup | 82.87% finally |
| State values | Finite; sampled concentrations remain in [0, 1] |
| Cup-wall intersections | Zero particle centers inside either shell at seven checkpoints |
| Minimum sampled wall distance | 1.296 mm; particle radius is 1.333 mm |

The settled, pouring and returned-cup images were visually checked. Residual liquid remains attached to the water
cup under the mop adhesion parameters. The final mixture continues evolving when `--steps` exceeds 5000.

Full logs and numerical wall checks are under `tmp/multiflow/logs/`. The baseline log is `baseline-isolated.log`,
the full pour log is `coffee-final.log`, and its independent mesh-distance check is `pour-validation.log`.

Concentration transport tests cover all three liquid solvers, diffusion strengths 0 and 0.005, and `n_envs=0,2`.
They check initialization, exact label retention with diffusion off, scalar conservation, reduced variance,
reordering, serialized state restoration and selective environment reset. The emitter test checks concentration
on reused slots. Disabling that update in an isolated test process makes the assertion fail with concentration
0.1 instead of 0.75. The correct implementation passes both environment counts.

The reconstructed-surface and point-color tests check each liquid solver separately for both environment counts. The PBD reconstruction
radius regression passes, and the FEM/PBD texture test matches its downloaded reference image. RayTracer validation
is skipped because this environment lacks LuisaRenderPy. A default-resolution `--surface --record --steps 2` run also
passes, with its image inspected under `tmp/multiflow/surface-smoke/out/pbstf_coffee_water/`.

Reproduce the particle and rendering checks with the environment above:

```powershell
python -m pytest -n 0 --backend=gpu --basetemp=tmp/multiflow/pytest-particles `
    tests/particles/test_pbstf.py tests/particles/test_pbd.py tests/particles/test_ipbstf.py
python -m pytest -n 0 --backend=gpu --basetemp=tmp/multiflow/pytest-render `
    tests/rendering/test_dynamic_meshes.py -k 'concentration or pbd_reconstruction'
```
