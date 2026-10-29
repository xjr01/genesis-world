# Shell profiling and Gaia comparison

This is the baseline analysis. The subsequent [implementation and measured optimization results](../shell_optimization/README.md)
document the completed engine changes and validation.

The jelly shell takes **75.56 ms/frame** on the RTX 4090. Its dominant cost is the contact search **count pass and
conservative-bound updates: 46.62 ms/frame**, or **85.2% of captured GPU kernel time**. The corresponding fill pass
takes 1.82 ms/frame. All local material, bending, plane and self-contact solves together take 3.13 ms/frame on the GPU.
The first optimization should target how contact queries accumulate conservative bounds.

A parameter experiment reduces the search radius from 2 mm to 1 mm and reaches **30.04 ms/frame (2.52x faster)**.
This is a useful scene-specific starting point; its changed deformation requires accuracy validation before a default
change. Disabling bending gives 75.76 ms/frame, within baseline variation.

These measurements were taken on October 25, 2026. They compare Genesis execution costs with Gaia's implementation
structure. Gaia's cloth implementation uses CPU parallelism; the CUDA graph implementation in its separate volume
solver is a different comparison target. An end-to-end Gaia speed ratio requires a matched executable benchmark.

## Scope and reproducibility

- Genesis engine revision: `fd4f64e84f7e7b00ad9effb1bd069a50b6adcfa4`. Engine files remain unchanged during these runs.
- Gaia source revision: `c229692045465a76233f9fba9197fb22bbfb3694`, available locally at `.cache/gaia-reference`.
- Windows WDDM, NVIDIA RTX 4090, driver 610.88, Python 3.11.15, Quadrants 1.2.0, Torch 2.12.1+cu126, float32.
- Asset: `genesis/assets/meshes/jelly/remeshed.obj`; SHA-256
  `d5b2a11cdc686a4af8922879b2303c0b19bbb3ec2ced7eca82ca2b97cb0a0827`.
- 12,740 vertices, 25,476 triangles, 38,214 edges; one closed shell falling onto a fixed plane. Viewer disabled.
- Frame duration 0.005 s, two physical substeps of 0.0025 s, 20 iterations per substep, zero convergence tolerance.
- Young's modulus 100,000 Pa, Poisson ratio 0.25, density 360 kg/m3, thickness 0.00025 m, shear stiffness 8 N/m,
  bending rigidity 5e-8 N m, friction coefficient 0.4. Elastic bending; line search and penetration diagnostics disabled.
- Contact activation gap 1e-5 m, search radius 0.002 m, shell contact stiffness 10,000 N/m, plane stiffness 100,000 N/m.
- 60 warmup frames, reset, then 200 measured frames. Baselines use three reset trials. Every measured frame ends with
  CUDA synchronization. Initialization, compilation, warmup, resets and final-state extraction are outside frame timers.
- Profiling processes run sequentially on an active desktop. These are workstation measurements.

The [profiling entry point](../../vbd_profile.py) supports shell and volume scenes, uninstrumented wall timing,
Python call/line tracing, Quadrants kernel timing, and a CUDA profiler capture range. It records material settings,
per-frame times, color sizes, contact/rebuild counts, and final geometric measurements.

```powershell
$env:PYTHONPATH = "$PWD"
$env:PROCESSOR_ARCHITECTURE = "AMD64"
$env:QD_OFFLINE_CACHE_FILE_PATH = "$PWD/.cache/qd_shell_profile"
$python = "C:/Users/admin/miniconda3/envs/genesis-world/python.exe"
$nsys = "C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"

& $python examples/vbd/vbd_profile.py --model shell --mode baseline --warmup 60 --steps 200 --repeats 3 --output-dir out/vbd/shell_profile
& $python examples/vbd/vbd_profile.py --model shell --mode kernels --warmup 60 --steps 200 --output-dir out/vbd/shell_profile
& $python examples/vbd/vbd_profile.py --model shell --mode lines --warmup 60 --steps 200 --output-dir out/vbd/shell_profile
& $python examples/vbd/vbd_profile.py --model shell --mode python --warmup 60 --steps 200 --output-dir out/vbd/shell_profile
& $python examples/vbd/vbd_profile.py --model shell --mode baseline --warmup 60 --steps 200 --repeats 3 --no-bending --output-dir out/vbd/shell_profile
& $python examples/vbd/vbd_profile.py --model shell --mode baseline --warmup 60 --steps 200 --repeats 3 --search-radius 0.001 --output-dir out/vbd/shell_profile
& $python examples/vbd/vbd_profile.py --model shell --mode lines --warmup 60 --steps 200 --search-radius 0.001 --output-dir out/vbd/shell_profile
& $nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --output=out/vbd/shell_profile/shell_nodes $python examples/vbd/vbd_profile.py --model shell --mode nsys --warmup 60 --steps 200 --output-dir out/vbd/shell_profile
& $nsys export --type sqlite --output out/vbd/shell_profile/shell_nodes.sqlite out/vbd/shell_profile/shell_nodes.nsys-rep
```

Use a fresh output prefix for subsequent Nsight captures. JSON and text results are archived beside this report.
The full [Nsight trace](../../../../out/vbd/shell_profile/shell_nodes.nsys-rep), SQLite export and Python binary profile
remain in `out/vbd/shell_profile`.

## Wall time

The [baseline](shell_baseline_i20_tol0_lsoff_gap1e-05_radius0.002_bend1.json) averages 75.56 ms/frame, with median
71.53 ms and P95 81.71 ms. Trial means are 74.75, 76.12 and 75.81 ms/frame. Dividing by two gives an average physical
substep cost of 37.78 ms. Final minimum height is 0.134895 mm, above the 0.125 mm half-thickness; minimum triangle
area relative to rest is 0.91335. These are final-state measurements.

| Frame window in each reset trial | Mean ms/frame |
| --- | ---: |
| 1-10 | 89.46 |
| 11-30 | 95.26 |
| 31-60 | 73.34 |
| 61-100 | 72.22 |
| 101-200 | 72.23 |

## GPU attribution including graph nodes

Nsight captures 393,712 kernel executions, summing to **54.708 ms/frame**, while the instrumented wall time is
**79.20 ms/frame**. The percentages below use captured kernel duration as their denominator. Host and device work
overlap, so this table and the host table describe separate views of the same execution.

| Stage | GPU ms/frame | Share of captured kernel duration |
| --- | ---: | ---: |
| Shell search count pass, including bound atomics | 46.620 | 85.22% |
| Shell search fill pass | 1.817 | 3.32% |
| Local solves across all colors | 3.131 | 5.72% |
| Apply color positions | 1.466 | 2.68% |
| Serial tasks inside color graphs | 0.834 | 1.52% |
| Bounding volume hierarchy refit | 0.172 | 0.31% |
| Initialize shell bounds and translation | 0.153 | 0.28% |
| Prediction and its limits | 0.127 | 0.23% |
| Finish and validate | 0.047 | 0.09% |
| Volume query kernels in the shell scene | 0.019 | 0.03% |
| Update hierarchy input bounds | 0.016 | 0.03% |
| Remaining kernels and search serial tasks | 0.307 | 0.56% |

See [the machine-readable summary](nsys_summary.json). Copy operations total 0.059 ms/frame across all recorded
directions, with 15.66 MB transferred over the entire capture. The scalar-readback wait is therefore dominated by
queued computation rather than copy bandwidth.

There are **427 count calls and 427 fill calls**. Mean count-call time is **21.836 ms**, versus **0.851 ms** for fill:
the count pass is **25.65x** as expensive. Both traverse the same hierarchy and evaluate the same geometric filters.
The two passes share a kernel name; ordering their range-kernel events by GPU start time gives alternating count
and fill calls, matching the host call order and independent rebuild count.
Count additionally updates conservative bounds, entity contact flags and translation bounds; fill writes accepted
pairs. The four vertex-bound atomic minima occur before feasible-contact filtering in
[the query kernel](../../../../genesis/engine/solvers/vbd/collision.py#L469).
This asymmetry strongly implicates the bound-update path. The precise share attributable to atomic contention,
instruction scheduling and divergence still requires a controlled kernel experiment or hardware-counter analysis.

The colors contain `[2191, 2170, 2020, 2022, 1705, 1311, 828, 384, 101, 8]` vertices. CUDA uses eight threads per
vertex. Each graph has 41 kernel nodes: ten local solves, ten applications and 21 serial tasks. Nsight records
8,012 graph executions, matching the independent line trace: 8,000 completed iterations plus 12 gated extra replays.
The entire graph accounts for 5.43 GPU ms/frame. Idle replays contribute only 0.15% of graph submissions.

Quadrants' own profiler reports 52.01 ms/frame and attributes 93.7% of its recorded time to shell search. Its records
omit `solve_colors` graph nodes and Torch kernels, so that percentage has a narrower denominator. Nsight supplies
the graph-inclusive table above. Nsight 2025.5.2 reports using CUDA 13.1 tracing libraries with this CUDA 13.3 driver
and warns that some CUDA events might be missing. The complete expected graph-node counts and both search-pass
counts match independent traces; the 48.44 ms/frame Nsight search total also agrees with Quadrants' 48.81 ms/frame.
Future acceptance runs should use a profiler release supporting the installed driver.

## Host waits and contact workload

The [line trace](shell_lines_i20_tol0_lsoff_gap1e-05_radius0.002_bend1.json) takes 82.86 ms/frame. Values below are
elapsed time attributed to Python statements, including asynchronous GPU waits and tracing overhead.
Raw line hits count trace events; multiline call expressions can emit two events. Dedicated substep counters record
graph submissions using the first argument's single event, verified against Nsight's graph-node counts.

| Statement | Host ms/frame | Share of traced frame |
| --- | ---: | ---: |
| Read exact shell candidate count via `item()` | 46.86 | 56.55% |
| Submit color graphs | 14.72 | 17.77% |
| Read iteration status | 1.90 | 2.30% |
| Submit shell count and fill queries | 2.79 | 3.36% |
| Build vertex incidence counts via `bincount` | 0.96 | 1.16% |
| Sort contact incidences | about 0.14 | about 0.17% |

The separate [Python call profile](shell_python_i20_tol0_lsoff_gap1e-05_radius0.002_bend1.txt) takes 80.91 ms/frame.
It records 19,373 Quadrants calls; recursive argument binding and structure-array argument binding accumulate
8.19 ms/frame together, within the broader launch path. The fixed plane also participates in the rigid solver,
whose substep path accumulates 4.50 ms/frame under this instrumentation. These cumulative times overlap their parents.

Across 400 substeps, all 8,000 requested iterations execute. There are 427 shell rebuilds, 2,035 status reads and
at most 195 retained pairs. Mean per-substep maximum pair count is 9.99; late substeps retain at most 12 pairs.
Only 27 rebuilds exceed the mandatory one per substep. Search operates over 50,954 vertex/edge queries per pass,
despite the small retained list. Consequently:

- Reducing extra rebuilds has limited headroom in this scene.
- Moving or removing `item()` moves the wait unless the count computation changes.
- Candidate sorting is a lower priority than conservative-bound accumulation.
- Membrane and bending arithmetic are contained within the 3.13 ms/frame local-solve total.

## Comparison with Gaia cloth

The source links below pin the reference revision. The default cloth step dispatches to its colored CPU solver;
`cpu_parallel_for` uses TBB in release builds. Its separate volume CUDA implementation has a different workload.

| Component | Genesis shell | Gaia cloth | Performance implication |
| --- | --- | --- | --- |
| Execution | Quadrants CUDA graph per color sweep, eight threads per vertex | CPU TBB parallel loops per color, followed by position application | Graph scheduling is already implemented here; adopting the volume graph architecture again offers little guidance for the present hotspot. |
| Membrane | Plane-stress Saint Venant-Kirchhoff energy, 3-by-3 deformation storage, clamped compressive scalar Hessian term | 3-by-2 deformation gradient, explicit membrane Hessian, material damping terms | Algebra specialization is possible, but its measured budget is part of the small local-solve total. |
| Bending | Rest dihedral angles, angle gradients, outer-product Hessian approximation; supports plastic rest angles | Precomputed cotangent quadratic form `Q`, force proportional to `Q X`, diagonal Hessian blocks proportional to `Q_ii I` | The models represent different bending behavior. A quadratic option needs explicit material semantics and validation on curved rest shapes and plastic creases. |
| Search | GPU hierarchy traversal twice, for exact count and fill | Embree vertex-face and edge-edge queries populate per-query contact lists | Duplicate traversal contributes fill cost; count-side bound accumulation is much more expensive in this measurement. |
| Bounds | Each nearby pair atomically minimizes bounds of all four vertices before feasible-region filtering | Tracks query/face/edge minimum distances, then gathers neighboring primitive minima to each vertex | This is the highest-priority structural difference to investigate. |
| Contact incidence | Exact-sized pair arrays; sort, histogram and prefix sum | Per-primitive lists and reverse face-contact links, including fixed-capacity inline storage | Preserve Genesis' exact-allocation requirement when adopting the primitive-level accumulation idea. |
| Requery rule | First iteration plus exhausted displacement bounds | First iteration plus conservative-bound truncation or collider updates | The 0.45 clearance bound and requery mechanism are shared design choices. |
| Local correction | Shell Newton correction multiplied by 0.5 to relax equal-color contact interactions | Configurable step size, default 1.0 | Compare convergence at matched physical error before changing relaxation. |

Sources: [Gaia cloth scheduling and bounds](https://github.com/AnkaChan/Gaia/blob/c229692045465a76233f9fba9197fb22bbfb3694/Simulator/Modules/VBDCloth/VBDClothPhysics.cpp),
[Gaia membrane and bending](https://github.com/AnkaChan/Gaia/blob/c229692045465a76233f9fba9197fb22bbfb3694/Simulator/Modules/VBDCloth/VBDTriMeshStVK.cpp),
[Gaia query implementation](https://github.com/AnkaChan/Gaia/blob/c229692045465a76233f9fba9197fb22bbfb3694/Simulator/Modules/CollisionDetector/ClothContactDetector.cpp),
[Gaia contact storage](https://github.com/AnkaChan/Gaia/blob/c229692045465a76233f9fba9197fb22bbfb3694/Simulator/Modules/CollisionDetector/ClothContactDetector.h),
[Genesis solve scheduling](../../../../genesis/engine/solvers/vbd/solver.py#L444),
[Genesis material responses](../../../../genesis/engine/solvers/vbd/physics.py#L241).

## Sensitivity experiments

Disabling bending gives **75.76 ms/frame**, versus 75.56 with bending: a 0.27% increase, within trial variation.
Trial means are 75.01, 74.71 and 77.57 ms/frame. The final minimum area ratio changes from 0.91335 to 0.97774, and
the color count changes from ten to seven. This experiment changes both the material and scheduling, so it measures
overall sensitivity rather than isolating the dihedral formula. It provides no frame-time case for removing bending.
See [the no-bending baseline](shell_baseline_i20_tol0_lsoff_gap1e-05_radius0.002_bend0.json).

Reducing search radius to 1 mm, while retaining thickness, activation gap, stiffness and bending, gives the following
[baseline](shell_baseline_i20_tol0_lsoff_gap1e-05_radius0.001_bend1.json) and
[line trace](shell_lines_i20_tol0_lsoff_gap1e-05_radius0.001_bend1.json):

| Quantity | 2 mm search radius | 1 mm search radius |
| --- | ---: | ---: |
| Uninstrumented mean ms/frame | 75.56 | 30.04 |
| Uninstrumented P95 ms/frame | 81.71 | 38.56 |
| Three trial means, ms/frame | 74.75 / 76.12 / 75.81 | 29.97 / 30.19 / 29.96 |
| Shell rebuilds over 400 substeps | 427 | 548 |
| Completed iterations | 8,000 | 8,000 |
| Graph submissions | 8,012 | 8,185 |
| Maximum retained pairs | 195 | 28 |
| Candidate-count readback, host ms/frame under tracing | 46.86 | 2.31 |
| Graph submission, host ms/frame under tracing | 14.72 | 15.12 |
| Final minimum height, mm | 0.134895 | 0.134900 |
| Final minimum area ratio | 0.91335 | 0.93688 |

The smaller radius is still greater than thickness plus activation gap. Rebuilds increase 28.3%, while overall
frame time falls 60.2%. This supports reducing the cost per query ahead of reducing the number of queries.
The radius changes conservative motion bounds and the finite-iteration trajectory, as the area ratios show.
Final height alone establishes neither equal deformation accuracy nor clearance throughout the trajectory.
The 1 mm configuration makes host graph submission a much more prominent optimization target.

## Optimization sequence and acceptance

1. **Use the measured radius setting as a scene-level candidate.** Try
   `VBDOptions(contact_search_radius=0.001)` for this remeshed jelly, retaining the same material and iteration budget.
   Validate full-trajectory clearance, folding and convergence before changing shared defaults. Sweep radius jointly
   with the iteration budget at equal physical error; the measured 2.52x gain includes a changed trajectory.

2. **Change conservative-bound accumulation first among engine changes.** Keep per-query minima in registers,
   accumulate minima on faces and edges, and gather these into vertex bounds once. Allocate arrays using the exact vertex/face/edge counts.
   Preserve the current weighted thickness, both participants of edge-edge pairs, cross-entity translation bounds,
   pre-feasibility bound updates and fatal-state reporting. Compare a unique-pair scatter implementation with symmetric
   edge queries that reduce locally; extra traversal may be cheaper than contended floating-point atomics. Measure
   count and fill separately on identical frozen geometry before accepting an implementation. This targets the
   measured 46.62 ms/frame stage; attributing all of it to atomics would overstate the evidence.

3. **Then reduce remaining search work.** Separate vertex-face and edge-edge query kernels, benchmark tighter
   distance-based hierarchy pruning, and avoid repeating expensive narrow-phase geometry during fill when an
   exact-sized intermediate can retain it. Preserve all conservative-distance information, including pairs rejected
   by the contact feasible-region filter. Extra intermediates trade bandwidth and memory for recomputation. Keep
   allocation proportional to actual geometry/candidates; Gaia's fixed-capacity contact storage conflicts with the
   repository's allocation rule.

4. **Reduce host scheduling after the query bottleneck falls.** Capture several gated sweeps per graph submission,
   keeping the current per-sweep error/rebuild decisions. A submission batch currently still calls the one-sweep graph
   repeatedly. Profile argument binding and buffer updates before changing the graph API. Add a shell-only path to
   skip volume queries, their empty `nonzero`, and the previous/swept hierarchy work used solely for volume contact.
   Preserve global barriers between solve and apply: equal-colored vertices can interact through dynamic contacts.

5. **Optimize material arithmetic only against the remaining budget.** Specialize shell deformation to 3-by-2,
   benchmark 4/8/16 threads per vertex, and improve color balance while retaining hinge adjacency. Consider quadratic
   bending as an explicit alternate material model if its behavior is wanted. Evaluate relaxation and acceleration
   using equal residual/contact accuracy rather than equal iteration counts. The current graph, local-solve and
   no-bending measurements place this work below contact accumulation.

Using the single Nsight run as an Amdahl estimate, halving the 46.62 ms count stage changes 79.20 to **55.89 ms/frame
(1.42x)**; reducing it to one quarter gives **44.24 ms/frame (1.79x)**. These are conditional estimates holding every
other cost fixed. Eliminating all local-solve GPU arithmetic saves only 3.13 ms in that accounting. Reprofile after
each change because launch overhead becomes more visible as the count pass shrinks.

Acceptance should retain the current 200-frame jelly baseline and add a folded/contact-dense benchmark plus
inter-entity contact. Repeat synchronized timings at least three times, record count/fill duration, rebuilds, candidate
counts and graph submissions, and compare physical error at the same simulated duration. Extend existing
`tests/deformable/test_vbd.py` capabilities with CPU float64 and CUDA float32, single and two environments. Preserve
clearance under motion, static support, friction, curved rest shapes, plastic creases and reset behavior. Exercise
fast approach and tangential motion through the complete assertion horizon; retain all existing thresholds.

