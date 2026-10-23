# VBD graph scheduling and cooperative vertex updates

The vertex block descent (VBD) solver replays an entire color sweep through a Quadrants graph. CUDA uses eight
threads per vertex to accumulate incident element, shell bending and contact responses. CPU uses one thread per
vertex. Vertex coloring runs during scene construction.

## Execution

Each graph replay refreshes contact geometry, clears iteration statistics and executes the ordered color passes.
Each pass solves its vertices and then applies their positions with a global barrier before the next color reads
them. This reduces the jelly scene's 24 separate color kernel calls per iteration to one graph call, also including
the contact refresh and statistics reset.

Four vertices share a CUDA warp. Each group reduces energy, gradient and Hessian and broadcasts the leader's sums.
Fixed vertices and padded lanes participate in the collectives. With line search enabled, a warp-wide vote keeps
the loop active until all groups finish backtracking. This permits independent acceptance decisions per vertex.

Contact candidate rebuilding, exact-sized buffer allocation, convergence decisions and invalid-state reporting
remain in the outer iteration loop. The graph receives the current contact buffers on every call. Line search is
disabled by default and can be enabled with `--line-search` in the examples.

The [Quadrants graph guide](https://genesis-embodied-ai.github.io/quadrants/user_guide/graph.html) describes graph
replay and conditional loops. The RTX 4090 supports graph replay; its conditional-loop implementation uses a host
fallback. This solver captures each color sweep and retains host control over iterations. The
[subgroup guide](https://genesis-embodied-ai.github.io/quadrants/user_guide/subgroup.html) describes the cooperative
operations used for vertex reductions.

## Measurement scope

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
python examples/vbd/vbd_profile.py --mode baseline --steps 60 --warmup 30 --repeats 3 --output-dir out/vbd/scheduling/final
python examples/vbd/vbd_profile.py --mode baseline --iterations 100 --tolerance 0 --steps 30 --warmup 15 --output-dir out/vbd/scheduling/final
python -m pytest tests/benchmarks/test_vbd.py --backend gpu -n 0 -m benchmarks
python -m pytest tests/deformable/test_vbd.py --backend gpu -n 0
python -m pytest tests/deformable/test_vbd.py --backend cpu -n 0
python examples/vbd/vbd_elastic.py -g --steps 200 --output-dir out/vbd/scheduling/elastic
python examples/vbd/vbd_shell.py -g --steps 200 --output-dir out/vbd/scheduling/shell
```

The managed Windows shell also required `PROCESSOR_ARCHITECTURE=AMD64` for py-cpuinfo.
Historical variants were measured using local source snapshots before restoring the optimized implementation.
