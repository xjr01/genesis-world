# VBD graph scheduling and cooperative vertex updates

The vertex block descent (VBD) solver replays an entire color sweep through a Quadrants graph. CUDA uses eight
threads per vertex to accumulate incident element, shell bending and contact responses. CPU uses one thread per
vertex. Vertex coloring runs during scene construction.

## Execution

The default convergence tolerance is zero: every substep runs its configured iteration count, with the convergence
reduction and maximum-displacement scalar compiled out. Pure-volume scenes submit that budget without reading
iteration statistics back to the host. Shell scenes retain status checks to refresh contact candidates as needed.

Each active graph replay executes the ordered color passes. Contact geometry is evaluated at each vertex's current
trial position. Each pass solves its vertices and applies their positions with a global barrier before the next
color reads them. The accepted
correction is reduced during position application, with one atomic maximum per active subgroup. Padded lanes use
zero correction and participate in the subgroup reduction. A zero convergence tolerance compiles out this reduction
and its scalar storage.

During the local solve, four vertices share a CUDA warp. Each group reduces energy, gradient and Hessian and broadcasts
the leader's sums.
Fixed vertices and padded lanes participate in the collectives. With line search enabled, a warp-wide vote keeps
the loop active until all groups finish backtracking. This permits independent acceptance decisions per vertex.

For shells or a positive convergence tolerance, the host submits batches of 1, 2, 4 and up to 8 sweeps, capped by the
remaining iteration budget. Device-side checks
stop updates on the first converged sweep, contact rebuild request or fatal state error. The host reads the completed
count and control flags once per batch. Rebuilding resets the batch size to one, keeping checks responsive during frequent
contact changes. The device gate preserves the per-sweep stopping decision while later queued sweeps remain idle.

Contact candidate rebuilding, exact-sized buffer allocation and invalid-state reporting remain in the host loop.
The graph receives the current contact buffers on every call. Line search is disabled by default and can be enabled
with `--line-search` in the examples.

The [Quadrants graph guide](https://genesis-embodied-ai.github.io/quadrants/user_guide/graph.html) describes graph
replay and conditional loops. The RTX 4090 supports graph replay; its conditional-loop implementation uses a host
fallback. This solver captures each color sweep and batches its submissions with device-side update gating. The
[subgroup guide](https://genesis-embodied-ai.github.io/quadrants/user_guide/subgroup.html) describes the cooperative
operations used for vertex reductions.

## Optional diagnostics

The elastic and shell examples measure frame times, synchronize CUDA for timing, inspect the final state and write
JSON reports only with `--benchmark`. The elastic viewer retains its FPS display during ordinary runs. The separate
`vbd_profile.py` command explicitly enables performance measurement; temporary investigation scripts in `out/` run
only when invoked directly.

Penetration diagnostics default to disabled. Set `VBDOptions.is_contact_diagnostics_enabled=True` or pass
`--contact-diagnostics` to an example to enable geometric penetration checks and warnings. Collision detection and
contact forces remain active in either mode. Invalid deformation and non-finite state retain their error checks.

The current defaults pass all 18 CUDA float32 and all 18 CPU float64 cases in `tests/deformable/test_vbd.py`.
Vertex controls cover pure-volume and mixed shell/volume scenes; contact tests explicitly enable penetration
diagnostics and verify continued motion. Material tests cover fixed iterations and optional early convergence.
Restoring the default tolerance to 1e-7 in an isolated source copy fails the contact force-balance assertion:
0.011334 N support versus 0.011301 N weight, exceeding its 2e-5 N tolerance.

A 200-frame GPU audit of the ordinary elastic example records zero iteration-statistics readbacks, zero explicit
CUDA timing synchronizations, no maximum-displacement buffer and no frame-timing samples. Each substep executes
20 iterations. Final minimum height is 9.971e-6 m and minimum volume ratio is 0.945321.

## Volume collision alignment

The algorithm reference is [Gaia](https://github.com/AnkaChan/Gaia/tree/c229692045465a76233f9fba9197fb22bbfb3694),
revision `c229692045465a76233f9fba9197fb22bbfb3694`, published under the Apache 2.0 license. The implementation uses
Genesis data structures and Quadrants kernels for the following stages:

1. Discrete collision detection (DCD) tests each surface vertex against nonincident tetrahedra at the beginning of
   each substep. A contained vertex searches for the closest admissible surface feature. A walk through adjacent
   tetrahedra validates the segment connecting that feature to the containing tetrahedron.
2. Continuous collision detection (CCD) tests the swept vertex against swept triangles during inertia prediction.
   Cubic coplanarity roots in `[0, 1]` are checked against triangle barycentric coordinates. The earliest hit supplies
   the contact face for a vertex that has no DCD contact. The same traversal expands its query bounds by
   `contact_d_hat` and checks predicted positions for nearby admissible surface features. A swept crossing takes
   priority over a proximity contact.
3. Each querying vertex contributes one face contact, with reverse incidence for the triangle's three vertices.
   The vertex solve recomputes the contact point, weights and oriented normal from current positions. A quadratic
   penalty uses `E = k * max(contact_d_hat - gap, 0)^2 / 2`; its active normal Hessian is `k * n * n.T`. Fixed planes use
   the same penalty. The regularized friction force uses the current normal load and a lagged tangent Hessian.

Volume contacts are detected once per substep, following the reference's default intermediate-detection setting.
The former per-sweep contact refresh and volume displacement bounds are removed. Contact incidence uses exact-sized
buffers. The accepted subgroup displacement reduction and batched iteration-status readback remain in use.

`contact_stiffness` denotes physical stiffness in N/m and defaults to 100,000. `contact_d_hat` defaults to 1e-5 m
and controls the activation gap for solid, plane and shell contact. Shell material thickness is accounted for
before applying this gap. The activation margin extends the reference's zero-gap solid and plane penalties.
Finite penalty contact permits load-dependent penetration when the margin is compressed under sufficient load.
Friction is evaluated during each vertex solve from the current normal load and relative tangential displacement;
a detected candidate contributes friction only while its normal penalty is active.

With contact diagnostics enabled, geometric penetration sets a bit reported once per state restoration, suggesting a larger
`VBDOptions.contact_d_hat`. The solver continues updating after this warning. Accepted positions are checked against
planes and the selected contact pairs, sharing contact geometry with the force calculation. The diagnostic uses the
existing status readback. Invalid deformation and non-finite state retain their fatal error handling.

The runtime implementation retains a grouped Morton-order bounding volume hierarchy (BVH) on the device. Bounds
are refitted for each detection phase and the tree is spatially reordered every 16 frames. Gaia uses separate
Embree trees with different rebuild periods. Cubic roots use bisection on monotone intervals, and feature tests use
local geometric scales for tolerance. These implementation choices preserve the detection stages while allowing
the solver to use the existing Genesis backends.

Shell contact uses radius queries for vertex-face and edge-edge pairs, incident-feature exclusion, closest-feature
filtering and conservative bounds of `0.45 * (distance - thickness)`. The local response combines the reference's
quadratic/logarithmic two-stage penalty with regularized friction; near-parallel edge pairs suppress their unstable
normal response. `shell_contact_stiffness` defaults to 10,000 N/m, including contact with fixed planes.

Shell initialization uses the previous position when nearby eligible contacts exist. Elsewhere, bounded inertia
prediction permits a common entity translation, capped by external clearance, to preserve dense-mesh free fall.
Shell corrections use a relaxation factor of 0.5 because contact can connect equal-colored vertices. These are
Genesis integration choices: the reference exposes initialization and step-size parameters, with previous-position
initialization and unit step size as its shell defaults. Edge-edge queries retain pairs when either closest point
projects onto the opposite segment. Line search permits relative energy roundoff near stationary points.

### Collision performance before the activation margin

The accepted iteration-control snapshot and the collision replacement use the same remeshed jelly, RTX 4090,
20-iteration limit, 1e-7 m convergence tolerance, 30 warmup frames and three reset trials of 60 measured frames.
Compilation and final-state inspection are outside synchronized frame timers. The contact law changes from the
previous activation-gap penalty to a zero-gap physical-surface penalty.

| Implementation | Mean ms/frame | P95 ms/frame | Speedup |
| --- | ---: | ---: | ---: |
| [Accepted iteration control](collision_before_20.json) | 32.15 | 85.95 | 1.00x |
| [Volume collision replacement](collision_after_20.json) | 21.21 | 26.39 | 1.52x |

These measurements compare Genesis implementations. The official Gaia executable has not been benchmarked on
this scene, so they establish the gain from replacing collision rather than parity with Gaia's total frame time.
The three final trial means are 20.649, 21.854 and 21.133 ms/frame. The physical-surface penalty gives a final
minimum height of -2.71e-8 m and a minimum volume ratio of 0.941873 over this measurement horizon.

The [200-frame elastic run](collision_elastic_200.json) completes one simulated second with minimum volume ratio
0.945334 and minimum height -2.87e-8 m. Its timings cover a longer trajectory and ordinary example measurements.

### Contact validation status

The activation-margin snapshot passes all 16 CUDA float32 cases and all 16 CPU float64 cases in `tests/deformable/test_vbd.py`,
covering single and batched environments, both line-search settings and fixed or early-converged iteration budgets.

The contact test includes a thin fixed tetrahedron, a fast incident tetrahedron and an initially penetrating
tetrahedron. Disabling continuous collision detection makes the fast-crossing assertion fail; disabling discrete
collision detection makes the initial-penetration assertion fail. Removing the line-search roundoff allowance makes
the static-support assertion fail on CUDA float32 (0.011622 N support versus 0.011301 N weight).

The contact test uses the full iteration budget to check total plane support against the two shells' weight within
2e-5 N, including compression of the activation margin. Gravity-compensated tetrahedra inside and outside the margin
distinguish proximity response from surface crossing. After deliberately overlapping two shells, an independent falling entity must
continue advancing. Settling velocity, rebound, friction, invalid deformation and state reset remain covered.

Removing the proximity query fails the inside-margin displacement check. Restoring a fatal penetration error fails
the initially intersecting scene. These mutation runs use isolated source copies and the same contact test.

### Activation margin measurements

These measurements use convergence tolerance 1e-7 m, enabled penetration diagnostics and sequential runs with zero
and 1e-5 m activation gaps on the same RTX 4090 jelly scene, with 30 warmup frames and three reset trials of 60 measured frames.
Compilation, resets and final-state inspection are outside synchronized frame timers.

| Activation gap | Mean ms/frame | P95 ms/frame | Final minimum height, m | Final minimum volume ratio |
| --- | ---: | ---: | ---: | ---: |
| [Zero](margin_zero_20.json) | 20.86 | 24.28 | -2.71e-8 | 0.941873 |
| [1e-5 m](margin_active_20.json) | 21.03 | 25.28 | 9.97e-6 | 0.941877 |

The mean difference is 0.16 ms/frame (0.8%). Active-margin trial means are 21.632, 20.877 and 20.575 ms/frame;
the difference is smaller than the variation across these trials. Changing the activation gap also changes the
trajectory and convergence work, so this comparison measures the overall effect on the scene.

The [200-frame elastic run](margin_elastic_200.json) completes one simulated second with final minimum height
9.971e-6 m and minimum volume ratio 0.945322. It produces no penetration warning over this trajectory. The zero-gap
profile emits a penetration warning after each reset and continues through every measured frame.

## Iteration control measurements

These measurements compare revision `2776a1fc189ce85b528195f97383058c1993ec30` with the subgroup correction reduction
and batched statistics readback described above. They use the same RTX 4090 jelly scene and physical parameters
listed below, with sequential runs and synchronization around each frame timer. Compilation and warmup are outside
the timers. The revision field in the after JSON identifies the base commit of the measured working tree.

| Iterations per substep | Before, ms/frame | After, ms/frame | Speedup |
| --- | ---: | ---: | ---: |
| Up to 20, tolerance 1e-7 m | [50.19](control_before_20.json) | [32.98](control_after_20.json) | **1.52x** |
| Fixed 100, tolerance zero | [210.20](control_before_100.json) | [92.44](control_after_100.json) | **2.27x** |

The 20-iteration comparison uses 30 warmup frames and three reset trials of 60 frames. The optimized trial means
are 33.004, 32.992 and 32.939 ms/frame. The 100-iteration comparison uses 15 warmup frames and one reset trial of
30 frames. Each frame advances 0.005 s through two physical substeps.

The [before](control_before_lines.json) and [after](control_after_lines.json) line traces each record 2,001 completed
sweeps across 120 physical substeps: 21 substeps stop after one sweep and 99 use the full 20. Statistics readbacks
decrease from 2,001 to 682, a 65.9% reduction. The optimized run submits 2,122 graph replays, including 121 idle
replays after a device-side stop. Host line times include queued device execution and synchronization.

| Final-state comparison | Up to 20 iterations | Fixed 100 iterations |
| --- | ---: | ---: |
| Maximum vertex position difference, m | 1.53e-6 | 3.38e-6 |
| Maximum vertex velocity difference, m/s | 5.36e-5 | 1.39e-4 |
| Minimum volume ratio, before / after | 0.958540 / 0.958531 | 0.981666 / 0.981676 |
| Minimum height, before / after, mm | 0.202461 / 0.202438 | 0.279074 / 0.279048 |

The iteration-control snapshot passed 16 CUDA float32 cases and 16 CPU float64 cases. Material-response coverage exercises
both fixed-iteration and early-convergence settings, in single and batched environments, with line search enabled
and disabled. Existing contact, friction, rebound, fixed-vertex, reset, invalid-state and plasticity assertions pass.

## Earlier graph scheduling measurements

Measurements use the working tree based on `8c75b61aa24dd6c372be4596e69e3ecbdb033f1b`, including the line-search
option and predicted-velocity update described in the [previous report](../profile/README.md). The before/after
comparison keeps these changes in both variants and measures the graph and cooperative updates separately.
The revision in each JSON identifies that base commit; it includes uncommitted changes described here.

- RTX 4090, Ryzen 9 7950X, Windows WDDM, Python 3.11.15, Quadrants 1.2.0, Torch 2.12.1+cu126, float32.
- Asset: `meshes/jelly/remeshed.obj`, SHA-256
  `d5b2a11cdc686a4af8922879b2303c0b19bbb3ec2ced7eca82ca2b97cb0a0827`.
- 20,119 vertices, 82,373 tetrahedra, 25,476 surface triangles, 12 color groups, fixed ground, viewer disabled.
- Log Neo-Hookean elasticity, E=10,000 Pa, nu=0.25, density=360 kg/m3, friction=0.4, contact distance=0.0005 m.
- One frame advances 0.005 s through two physical substeps of 0.0025 s; line search is disabled.
- Runs execute sequentially with GPU synchronization around frame timing. The desktop remains active.
- Compilation, warmup, resets and final state inspection are outside the wall timers. Graph compilation increases
  first-run startup cost; the reported gains apply after warmup.

## Jelly results

With a limit of 20 iterations per substep and convergence tolerance 1e-7 m, each variant runs three reset trials
of 60 frames after 30 warmup frames:

| Implementation | Mean ms/frame | Median ms/frame | P95 ms/frame | Speedup over before |
| --- | ---: | ---: | ---: | ---: |
| [Before](before_20.json) | 247.57 | 281.60 | 357.00 | 1.00x |
| [Graph, one thread per vertex](graph_20.json) | 69.02 | 72.88 | 122.77 | 3.59x |
| [Graph, eight threads per vertex](optimized_20.json) | **51.79** | **54.09** | **102.10** | **4.78x** |

The cooperative update reduces frame time by another 25.0% relative to graph-only execution. The final run means
are 51.65, 51.14 and 52.57 ms/frame. A [32-thread variant](full_warp_20.json) measured 82.44 ms/frame, so the default
uses smaller groups. These measurements support eight threads for this workload; other meshes can have different
incidence counts and hardware utilization.

Final minimum tetrahedral volume ratios are 0.958531 before and 0.958527 after optimization. Final minimum heights
are 0.202452 and 0.202449 mm. Parallel summation changes floating-point rounding while retaining the physical
checks described below.

The fixed-iteration comparison uses **100 iterations per substep**, convergence tolerance zero, 15 warmup frames
and one reset trial of 30 measured frames:

| Implementation | Mean ms/frame | Mean ms/physical substep | Speedup |
| --- | ---: | ---: | ---: |
| [Before](before_100.json) | 1295.80 | 647.90 | 1.00x |
| [Graph and cooperative updates](optimized_100.json) | **242.19** | **121.09** | **5.35x** |

This comparison keeps the iteration count fixed and includes the ground and surface-contact paths. Its shorter
measurement window differs from the 60-frame table. Paper timings use different scenes, energies, contact settings
and hardware; these measurements establish the gain within this Genesis workload.

## Performance test and physical validation

`tests/benchmarks/test_vbd.py` measures a stationary tetrahedral grid with 20 full iterations, one warmup frame and
20 timed steps. It checks unchanged positions and zero velocity and covers single and batched environments.
The same benchmark runs against both source variants:

| Environments | Before, aggregate FPS | After, aggregate FPS | Speedup |
| --- | ---: | ---: | ---: |
| 1 (`n_envs=0`) | 11.14 | 70.86 | 6.36x |
| 2 (`n_envs=2`) | 22.96 | 84.62 | 3.69x |

Raw measurements are in [before XML](benchmark_before.xml) and [after XML](benchmark_after.xml). Aggregate FPS
counts environments; the two-environment row counts two simulation advances per `scene.step()`.

The existing physics suite passes all 12 CUDA float32 cases and all 12 CPU float64 cases, covering single/batched
environments, both line-search settings, fixed vertices, reset, force updates, material response, ground contact,
self-contact, friction and plasticity. One CPU case initially failed during Windows display initialization
(`GetDpiForMonitor`, invalid handle); that case passed when rerun alone. Physical assertions were retained.

Both jelly examples complete 200 frames (1 s of simulated time) with the optimized solver and default settings.
The [elastic run](elastic_200.json) finishes with minimum volume ratio 0.956302 and minimum height 0.187298 mm.
The [shell run](shell_200.json) finishes with minimum height 0.160262 mm, above its half-thickness of 0.125 mm.
These example runs include their ordinary state measurements and use a different horizon from the timing tables.
Ruff checks, formatting checks and `git diff --check` pass for the changed code.

## Reproduction

Run from the repository root with the Genesis Python environment:

```powershell
$env:PYTHONPATH = "$PWD"
$env:QD_OFFLINE_CACHE_FILE_PATH = "$PWD/.cache/qd_vbd_schedule"
python examples/vbd/vbd_profile.py --mode baseline --tolerance 1e-7 --steps 60 --warmup 30 --repeats 3 --output-dir out/vbd/scheduling/final
python examples/vbd/vbd_profile.py --mode baseline --iterations 100 --tolerance 0 --steps 30 --warmup 15 --output-dir out/vbd/scheduling/final
python -m pytest tests/benchmarks/test_vbd.py --backend gpu -n 0 -m benchmarks
python -m pytest tests/deformable/test_vbd.py --backend gpu -n 0
python -m pytest tests/deformable/test_vbd.py --backend cpu -n 0
python examples/vbd/vbd_elastic.py -g --benchmark --steps 200 --output-dir out/vbd/scheduling/elastic
python examples/vbd/vbd_shell.py -g --benchmark --steps 200 --output-dir out/vbd/scheduling/shell
```

Reproduce the activation-gap comparison with:

```powershell
python examples/vbd/vbd_profile.py --mode baseline --tolerance 1e-7 --contact-diagnostics --contact-distance 0 --steps 60 --warmup 30 --repeats 3 --output-dir out/vbd/contact_margin/profile
python examples/vbd/vbd_profile.py --mode baseline --tolerance 1e-7 --contact-diagnostics --contact-distance 0.00001 --steps 60 --warmup 30 --repeats 3 --output-dir out/vbd/contact_margin/profile
```

The managed Windows shell also required `PROCESSOR_ARCHITECTURE=AMD64` for py-cpuinfo.
Historical variants were measured using local source snapshots before restoring the optimized implementation.
