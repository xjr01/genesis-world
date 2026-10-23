# VBD jelly profiling, 2026-10-23

This report records the implementation before graph scheduling and cooperative vertex updates. See the
[scheduling comparison](../scheduling/README.md) for the current implementation and its measured speedup.

The main bottleneck is the colored solve loop and its host launch path. Collision search is a smaller cost in this jelly drop. The local solve includes elasticity, ground contact, self-contact contributions and backtracking; its device timing measures those terms together.

## Reproduction and scope

Run from the repository root using the Genesis Python environment:

```powershell
$env:PYTHONPATH = "$PWD"
$env:QD_OFFLINE_CACHE_FILE_PATH = "$PWD/.cache/qd_vbd_profile"
python examples/vbd/vbd_profile.py --mode baseline --warmup 30 --repeats 3 --line-search
python examples/vbd/vbd_profile.py --mode kernels --warmup 30 --line-search
python examples/vbd/vbd_profile.py --mode lines --warmup 30 --line-search
python examples/vbd/vbd_profile.py --mode python --warmup 30 --line-search
python examples/vbd/vbd_profile.py --mode baseline --warmup 15 --steps 30 --iterations 20 --tolerance 0 --line-search
python examples/vbd/vbd_profile.py --mode baseline --warmup 15 --steps 30 --iterations 100 --tolerance 0 --line-search
```

Outputs go to `out/vbd/profile`; the JSON and text results from this run are archived beside this document. Python binary profiles remain in `out/vbd/profile`. The managed Windows shell required `PROCESSOR_ARCHITECTURE=AMD64` for py-cpuinfo. Kernel caches use a writable workspace directory. Build, compilation, warmup, resets and final state inspection are outside the wall timers.

The measurements below use enabled line search. The current CLI defaults to disabled line search and appends
`_lson` or `_lsoff` to output names so that comparison runs retain both sets of results.

- Engine revision: `8c75b61aa24dd6c372be4596e69e3ecbdb033f1b`.
- RTX 4090, Windows WDDM, Python 3.11.15, Quadrants 1.2.0, Torch 2.12.1+cu126, float32.
- Asset: `meshes/jelly/remeshed.obj`, SHA-256 `d5b2a11cdc686a4af8922879b2303c0b19bbb3ec2ced7eca82ca2b97cb0a0827`.
- 20,119 vertices, 82,373 tetrahedra, 25,476 surface triangles; fixed ground plane, viewer disabled.
- Log Neo-Hookean elasticity, E=10,000 Pa, nu=0.25, density=360 kg/m3, friction=0.4, contact distance=0.0005 m, up to 12 local backtracking attempts.
- One output frame advances 0.005 s through two physical substeps of 0.0025 s each. The default limit is 20 iterations per substep with tolerance 1e-7 m.
- Another IPC visualization process had exited before the archived measurements. The desktop remained active; these are workstation timings.

## Wall time

The baseline uses three reset runs of 60 frames after 30 warmup frames. Mean frame time is **233.10 ms**, median **269.55 ms**, P95 **295.28 ms**. Mean physical-substep cost is **116.55 ms**, obtained by dividing frame cost by two. The three run means are 231.27, 232.96 and 235.09 ms/frame.

| Frames in each reset run | Mean ms/frame |
| --- | ---: |
| 1-10, before ground contact | 29.53 |
| 11-30, initial impact and rebound | 276.14 |
| 31-60, subsequent motion | 272.27 |

The line trace recorded 2,002 solver iterations across 120 physical substeps: 20 substeps used one iteration, one used two, and 99 reached the limit of 20. Much of the impact-time increase is the extra iterations. The final height is 0.00020239 m and the final minimum tetrahedral volume ratio is 0.95855.

For an iteration-count comparison, two additional uninstrumented runs force every iteration with tolerance zero. Both use 15 warmup frames followed by a reset and the same 30-frame measurement window. Their physical trajectories can differ as convergence changes.

| Fixed iterations per physical substep | Mean ms/frame | Mean ms/physical substep |
| --- | ---: | ---: |
| 20 | 272.09 | 136.05 |
| 100 | 1,419.77 | 709.89 |

Five times the iteration count costs 5.22 times as much wall time. This comparison separates the iteration limit from the early exits used in the default run.

## Device attribution

Quadrants device timers record 113.70 ms/frame of device tasks. The instrumented run takes 413.17 ms/frame in wall time; enabling these timers substantially increases overhead. The percentages below refer to recorded device-task time. They cannot be applied directly to the uninstrumented 233.10 ms/frame baseline.

| Device stage | Recorded ms/frame | Share of device time |
| --- | ---: | ---: |
| Local color solve: elasticity, contacts, backtracking | 87.85 | 77.26% |
| Apply color positions | 13.52 | 11.89% |
| Surface contact search | 6.39 | 5.62% |
| Refresh contact geometry | 2.06 | 1.81% |
| Bounding-volume hierarchy refit | 0.54 | 0.47% |
| Remaining prediction, validation and framework tasks | 3.35 | 2.95% |

The color passes account for 89.15%. Search plus bound updates and hierarchy refits accounts for about 6.24%. The raw kernel table separates serial and parallel device tasks. JSON groups are summed from individual timing records so that a multi-task kernel contributes each task exactly once.

## Host attribution

The separate line-tracing run costs 319.36 ms/frame, including tracing overhead. These are host elapsed times attributed to Python statements; asynchronous device execution is charged at whichever call waits for completion.

| Statement / stage | Host ms/frame |
| --- | ---: |
| `solve_color` calls | 194.02 |
| `apply_color` calls | 76.47 |
| `refresh_contacts` calls | 13.54 |
| Error readback immediately after contact search | 5.95 |
| Iteration-statistics readback | 5.40 |
| Contact search launches, count and fill passes | 1.89 |
| Bounding-volume hierarchy refit launches | 0.75 |

The color solve and position application account for 84.7% of the traced frame. The error readback after search includes waiting for the search kernel. It would be misleading to interpret its entire duration as data-copy bandwidth cost.

The independent cProfile run costs 295.55 ms/frame. It records 52,798 Quadrants kernel calls in 60 frames. Kernel launch processing accumulates 15.274 s of 17.723 s total. Within that launch path, recursive argument binding accumulates 4.833 s and structure-array argument binding 1.521 s: about 105.9 ms/frame together under cProfile. These cumulative timings overlap their parent launch timings and must not be added to them. The three profiles have different overheads and must not be combined into one additive wall-time budget.

There are 12 color groups with sizes `[3819, 3622, 3522, 2854, 1865, 1238, 1108, 1035, 730, 286, 39, 1]`. A full frame invokes 960 color kernels: 2 substeps * 20 iterations * 12 colors * 2 calls. Each color call in this run emits both a serial device task and a parallel device task, giving 1,920 device tasks before prediction, contacts and framework work. Even the singleton group pays the dispatch cost.

Contact-list maintenance also launches refresh kernels every iteration, including iterations with empty candidate lists. The line trace records 172 search rebuilds over 120 substeps, with one fill pass; final candidate counts are zero at the end of each traced substep. Ground-plane forces are evaluated inside the local solve and are separate from these surface-pair counts. Dense folded-shell contact can have a different breakdown.

## Relation to the reference

The paper reports 3.3 ms for 100 iterations on a 15K-vertex/50K-tetrahedron flattening case, with contact parameters marked NA. Its twisting-beam case uses 97K vertices, 266K tetrahedra and 100 iterations at 60 ms/step. These are different workloads and physical time steps. Its GPU implementation assigns 16 threads to each vertex; experiments use fixed iteration counts, omit line search, use Neo-Hookean energy without the logarithmic term, and usually perform continuous collision detection (CCD) once per step. See [the paper, Sections 4-5 and Table 1](https://arxiv.org/html/2403.06321v3).

Gaia captures color passes in a CUDA graph and launches the graph from its iteration loop. See [the reference implementation](https://github.com/AnkaChan/Gaia/blob/main/Simulator/Modules/VBD/VBDPhysics.cpp). Our loop issues separate Python/Quadrants calls, serially accumulates adjacent-element responses within each vertex thread and reads iteration statistics onto the host each iteration. The profiled configuration enables local backtracking. It also uses barrier contact and conservative displacement limits. Implementing the vertex-update algorithm alone does not reproduce the reference implementation's execution costs.

## Optimization order suggested by the measurements

1. Reduce repeated host argument binding and launch overhead by executing color sweeps through a reusable device graph or equivalent compiled execution path, while preserving contact-buffer lifetimes and exact allocations.
2. Parallelize adjacent-element response evaluation within each vertex and improve color balance. The final two groups together contain only 40 vertices.
3. Profile local energy evaluation and backtracking separately before changing their mathematics. The current kernel timing combines elasticity, contact and line search; it cannot establish their individual shares.
4. Reduce unnecessary empty-contact refresh work and per-iteration host readbacks while retaining invalid-state reporting and contact-rebuild decisions.

These priorities are inferred from the enabled-line-search baseline above. The following measurements evaluate the line-search option separately.

## Execution capabilities

Vertex coloring runs once in `VBDSolver.build()`. Every substep reuses the stored color offsets and indices.

Quadrants supports `@qd.kernel(graph=True)` and `qd.graph.do_while`; the repository's PBSTF solver already uses them. The installed Quadrants 1.2.0 supports graph replay on the RTX 4090, while its conditional `do_while` loop uses a host fallback on this GPU. Native conditional loops require SM 9.0+. A compiled color sweep can still use graph replay to reduce launch overhead. See the [Graph guide](https://genesis-embodied-ai.github.io/quadrants/user_guide/graph.html).

Cooperative work within one vertex is also available through shared memory, block synchronization and subgroup operations. The implementation measured in this report serially accumulates incident elements in one thread. Parallelizing that accumulation requires a cooperative reduction of the force and Hessian. See the [subgroup guide](https://genesis-embodied-ai.github.io/quadrants/user_guide/subgroup.html). Both optimizations are implemented in the [scheduling comparison](../scheduling/README.md).

## Line-search option comparison

These runs use the working tree on top of the revision above, including `is_line_search_enabled` and the velocity update from predicted velocity plus solve correction. The latter reduces position-rounding amplification in single precision. All original physical assertions remain in place; 12 CPU float64 and 12 GPU float32 cases pass, covering both line-search modes and single/batched environments.

```powershell
python examples/vbd/vbd_profile.py --mode baseline --steps 60 --warmup 30 --repeats 3 --output-dir out/vbd/line_search/profile
python examples/vbd/vbd_profile.py --mode baseline --steps 60 --warmup 30 --repeats 3 --line-search --output-dir out/vbd/line_search/profile
```

Both runs use the same asset, material, timestep and 20-iteration limit listed above. They run sequentially after the tests finish. The desktop remains active. The JSON files with `_lsoff` and `_lson` suffixes archive these measurements.

| Line search | Mean ms/frame | Median ms/frame | P95 ms/frame | Final minimum volume ratio |
| --- | ---: | ---: | ---: | ---: |
| Disabled (default) | 243.40 | 277.88 | 337.52 | 0.958535 |
| Enabled | 242.76 | 279.84 | 310.08 | 0.958508 |

The difference in mean wall time is 0.27%, smaller than the variation among reset runs. This experiment shows no appreciable frame-time speedup from disabling line search under the current launch architecture. Disabled run means are 245.13, 241.52 and 243.56 ms/frame; enabled run means are 243.29, 241.92 and 243.07 ms/frame.

A separate 200-frame elastic example run with line search disabled completes with a final minimum volume ratio of 0.956308 and minimum height of 0.00018726 m. The 200-frame jelly shell run also completes, with minimum height 0.00018098 m, above its half-thickness of 0.000125 m. Each substep retains contact displacement limits and invalid-state checks. Results are archived as `elastic_200_lsoff.json` and `shell_200_lsoff.json`.

```powershell
python examples/vbd/vbd_elastic.py -g --steps 200 --output-dir out/vbd/line_search/elastic
python examples/vbd/vbd_shell.py -g --steps 200 --output-dir out/vbd/line_search/shell
```

The separate device-timer comparison uses 15 warmup frames and one reset run of 30 measured frames:

```powershell
python examples/vbd/vbd_profile.py --mode kernels --steps 30 --warmup 15 --output-dir out/vbd/line_search/profile
python examples/vbd/vbd_profile.py --mode kernels --steps 30 --warmup 15 --line-search --output-dir out/vbd/line_search/profile
```

| Device-timer stage | Disabled ms/frame | Enabled ms/frame |
| --- | ---: | ---: |
| Local color solve | 42.36 | 68.05 |
| Apply color positions | 10.54 | 9.77 |
| Surface contact search | 11.50 | 7.75 |
| Refresh contact geometry | 1.60 | 1.70 |
| Bounding-volume hierarchy refit | 1.28 | 0.97 |
| All recorded device tasks | 71.30 | 91.63 |

Disabling line search reduces recorded local-solve time by 37.7% and total recorded device-task time by 22.2%. Both runs record 19,224 local-solve device tasks, so the local reduction occurs at equal solve work counts. Surface-search device tasks increase from 218 to 316, showing that the larger accepted updates also change contact-search work. The instrumented wall times are 325.44 and 334.57 ms/frame; these belong to a shorter measurement window with profiler overhead and must be kept separate from the uninstrumented 60-frame comparison.

The local GPU work falls while overall frame time remains dominated by the existing launch architecture and associated synchronization. The measurements support prioritizing graph execution and cooperative vertex updates ahead of further line-search tuning.
