# Shell performance optimization results

The accepted engine changes reduce the jelly shell's frame time at the original 2 mm search radius from
**75.11 to 37.17 ms**, a **2.02x speedup**. The jelly example now defaults to a
1 mm search radius and takes **29.69 ms/frame**. The original engine already takes
**30.34 ms/frame** at 1 mm: the benefit from changing the scene parameter and the benefit from the
engine changes overlap substantially.

This implements the follow-up to the [Gaia comparison and baseline profile](../shell_profile/README.md).
Measurements use the RTX 4090 on October 25, 2026. The support-force precision issue is tracked below separately,
as requested. Its assertion and threshold remain intact.

## Matched wall time measurements

| Engine | Radius | Mean ms/frame | P95 ms/frame | Three trial means |
| --- | ---: | ---: | ---: | --- |
| [Original](original_2mm.json) | 2 mm | 75.11 | 84.10 | 75.16 / 75.22 / 74.94 |
| [Original](original_1mm.json) | 1 mm | 30.34 | 39.47 | 30.07 / 29.82 / 31.12 |
| [Optimized](optimized_2mm.json) | 2 mm | 37.17 | 43.18 | 37.06 / 37.11 / 37.34 |
| [Optimized](optimized_1mm.json) | 1 mm | 29.69 | 41.14 | 29.98 / 29.65 / 29.44 |

Each cell uses 60 warmup frames, a reset, and three trials of 200 frames. CUDA synchronization ends every frame.
The four configurations run sequentially. Initialization, compilation and warmup are outside frame timings.
Earlier exploratory measurements experienced different workstation load; this table uses the final verification
series exclusively. Raw per-frame measurements are linked in the table.

The scene has 12,740 vertices, 25,476 triangles and 38,214 edges. Both engine versions use float32, a 0.005 s frame,
two substeps, 20 iterations per substep, zero convergence tolerance, eight threads per vertex, and disabled line
search and contact diagnostics. Material parameters are E = 100,000 Pa, nu = 0.25, rho = 360 kg/m3, thickness
0.00025 m, shear stiffness 8 N/m, bending rigidity 5e-8 N m and friction 0.4. The contact activation gap is
1e-5 m, shell stiffness is 10,000 N/m and plane stiffness is 100,000 N/m.

Environment: Windows WDDM, NVIDIA driver 610.88, Python 3.11.15, Quadrants 1.2.0 and Torch 2.12.1+cu126.
Original engine revision: `fd4f64e84f7e7b00ad9effb1bd069a50b6adcfa4`.
The [engine hashes](engine_sha256.json) identify the original and final source variants.
The asset is `genesis/assets/meshes/jelly/remeshed.obj`, SHA-256
`d5b2a11cdc686a4af8922879b2303c0b19bbb3ec2ced7eca82ca2b97cb0a0827`.

## Accepted changes

**Accumulate conservative movement limits on primitives.** A vertex-face candidate contains the query vertex P and
the three corners A, B and C of the queried triangle. An edge-edge candidate contains the two endpoints A and B of
one edge and C and D of the other edge. These are the four vertices discussed in the original comparison.

The count pass keeps the query's minimum in local variables and updates the minimum stored on the target face or
edge. A subsequent pass visits each vertex's incident edges and faces and gathers their minima. Edge-edge candidates
contribute to both edges. Separate components preserve all-pair clearance and cross-entity clearance, including the
translation limit of each entity. Bounds still include nearby pairs rejected by the contact feasibility filter and
use the weighted thickness and the original factor 0.45. A monotone minimum check skips redundant atomic writes.

The additional array payload is exact-sized: two floating-point values per surface vertex, face and edge per environment,
plus one edge index per directed adjacency entry. The jelly adds 917,152 bytes, about 0.875 MiB, at float32 in one
environment. The gather costs 0.014 ms/frame in the GPU capture.

**Use a 1 mm search radius for the jelly example.** `vbd_shell.py` accepts `--search-radius`; its jelly default is
0.001 m and its fold default remains 0.002 m. `VBDOptions` retains its shared 0.002 m default. For jelly, the smaller
radius reduces query work and increases contact-list rebuilds. It changes conservative movement limits and therefore the
finite-iteration trajectory, as the area measurements below show. Thickness and contact activation distance retain
their physical values.

**Remove unnecessary search and host work.** Pure shell scenes skip volume contact queries, their previous/swept
bounding-volume hierarchy (BVH) updates and the empty volume-contact `nonzero` synchronization. Periodic BVH
rebuilding happens at the first shell query of the corresponding substep. Vertex-face queries compute triangle
weights directly; edge weights are computed only for edge-edge queries. Contact arrays remain allocated to their
exact measured sizes.

**Reduce material arithmetic.** The membrane response uses a 3-by-2 deformation gradient and 2-by-2 strain/stress
matrices. Thread-group reduction sums the six independent entries of the symmetric 3-by-3 Hessian and mirrors them.
This preserves the plane-stress Saint Venant-Kirchhoff model and the existing dihedral bending model. Registers per
thread in the local solve fall from 140 to 128; local-solve GPU time remains approximately 3.07 ms/frame.

## GPU time breakdown

| GPU stage | Original 2 mm ms/frame | Optimized 2 mm ms/frame | Optimized 1 mm ms/frame |
| --- | ---: | ---: | ---: |
| Contact count and bounds | 46.620 (85.2%) | 12.504 (61.3%) | 4.266 (35.2%) |
| Contact fill | 1.817 (3.3%) | 1.820 (8.9%) | 1.170 (9.7%) |
| Bound gather | 0.000 (0.0%) | 0.014 (0.1%) | 0.019 (0.2%) |
| Bound initialization | 0.153 (0.3%) | 0.172 (0.8%) | 0.233 (1.9%) |
| Local solve | 3.131 (5.7%) | 3.074 (15.1%) | 3.347 (27.6%) |
| Position update | 1.466 (2.7%) | 1.455 (7.1%) | 1.581 (13.0%) |
| Graph control | 0.834 (1.5%) | 0.829 (4.1%) | 0.921 (7.6%) |
| Other | 0.688 (1.3%) | 0.515 (2.5%) | 0.578 (4.8%) |
| Total | 54.708 | 20.383 | 12.116 |

Times are summed device kernel durations divided by 200 captured frames. Percentages use the sum of captured GPU
kernel durations as their denominator. The count row contains both geometric query work and conservative-bound
updates. Count and fill are separated by the alternating range-kernel launches; their serial launch kernels belong
to Other. Graph solve, position application and graph-control kernels are counted individually. Every captured
color sweep has 41 graph kernels across ten colors. See [the raw breakdown](gpu_breakdown.json).

The original trace is the baseline archive's `shell_nodes` capture; final captures use the final engine source.
Nsight Systems 2025.5.2 reports a CUDA tracing-library version older than the installed driver. Kernel names and
timestamps are available; graph counts were cross-checked against the expected kernels per sweep.

At 1 mm, the final line trace records 543 rebuilds, 2227 status reads and 8158 solver
submissions over 400 substeps and 8,000 applied iterations. The host line containing graph submission accounts for
14.65 ms/frame under tracing; the candidate-count readback accounts for 3.67 ms/frame and status
reads for 2.22 ms/frame. Host timings include waits for queued GPU work and tracing overhead, so they cannot
be added to the device table. Captures and line traces are separate runs and can differ slightly in rebuild counts.

## Experiments excluded from the final implementation

| Experiment | Observation and decision |
| --- | --- |
| Separate vertex-face and edge-edge kernels; Euclidean BVH pruning | The combined experiment increased frame time. The original combined traversal and box-overlap test remain. |
| Multiple sweeps statically unrolled in one graph | Warmup had still not finished after several minutes. The original one-sweep graph remains. |
| Dynamic graph loop | The installed Quadrants uses a host fallback on this SM 8.9 GPU; measured frame time increased. |
| `fastcache=True` | Current VBD structures are unsupported cache parameters, and line search triggered a purity error involving `gs.EPS`. The flags were removed and all affected tests rerun. |
| Four threads per vertex | The small wall-time difference was insufficient to establish a stable improvement. Eight remains the CUDA default; the profiler exposes `--threads-per-vertex` for further measurement. |

The remaining 1 mm workload warrants backend work on graph argument binding and submission overhead. A native
multi-sweep graph needs preserved solve/apply barriers and per-sweep rebuild/error gating. Query count still takes
35.2% of GPU kernel time, making it the largest remaining device category. The material change chiefly
reduces register use in this single-environment measurement.

## Physical validation and the deferred precision issue

| Engine | Radius | Final minimum height mm | Final minimum area ratio |
| --- | ---: | ---: | ---: |
| Original | 2 mm | 0.134894653 | 0.913349676 |
| Original | 1 mm | 0.134900154 | 0.936876004 |
| Optimized | 2 mm | 0.134894653 | 0.913348112 |
| Optimized | 1 mm | 0.134900169 | 0.936875374 |

At each fixed radius the original and optimized engines have closely matching final height and area ratio.
Changing radius changes the deformation reached at the fixed iteration budget. Minimum area is an aggregate
deformation observation; the contact tests provide the direct clearance assertions.

| Optimized fold | Mean ms/frame | P95 ms/frame | Minimum final height mm |
| --- | ---: | ---: | ---: |
| [2 mm](fold_2mm.json) | 33.53 | 90.94 | 0.134991 |
| [1 mm](fold_1mm.json) | 63.36 | 129.73 | 0.134890 |

The fold benchmark uses the existing constrained fold, release and settling sequence, 200 frames per trial and
five reset trials at each radius. These measurements exercise a second contact configuration and its final height.
The smaller radius increases fold frame time by 89%, supporting the scene-specific defaults.

The existing VBD tests cover CPU float64 and CUDA float32, `n_envs=0` and `n_envs=2`, line search on/off, material
response, friction, contact support, plastic unloading, bending, reset and invalid-state handling. The contact test
now also contains two approaching triangle pairs: one for vertex-face clearance and one for crossing edges.
Their signed separation after motion must remain at least the shell thickness minus 2e-7 m.

Intentional mutation verified the new assertion: omitting edge-bound gathering produced approximately -2.1 mm
separation, and omitting the entire gather produced approximately -2.4 mm separation in the vertex-face case.
Both fail at the new clearance assertion. Omitting face-only gathering still passes this scene. All mutations were reverted.

The full suite followed by the six-case line-search rerun covers 18 passing CPU combinations and 14 passing GPU
combinations. Four GPU contact combinations retain the support-force assertion failure below. The full first run
also exposed six cache-related compile errors per backend; removing the unsupported cache flag resolved these
in the rerun. [Validation excerpts](validation.txt) retain the failures and rerun summaries.

| Line search | Original and optimized measured support | Expected support | Absolute error | Existing tolerance |
| --- | ---: | ---: | ---: | ---: |
| Off | 0.011280499398708344 N | 0.011301120 N | 2.06206013e-5 N | 2e-5 N |
| On | 0.011335650458931923 N | 0.011301120 N | 3.45304589e-5 N | 2e-5 N |

Both engine versions reproduce these values on the extended contact test. The optimized test reproduces the same
values in single and two-environment runs. The support assertion runs at the end of the test, using the original
measurement and unchanged threshold, so friction, reset and containment checks also execute. This precision issue
is deferred independently from the performance work, per the user's instruction.

Ruff checking with the repository's manual import grouping (`--ignore I001`), formatting checks and `git diff --check`
pass for the changed files.

## Reproduction

```powershell
$env:PYTHONPATH = "$PWD"
$env:PROCESSOR_ARCHITECTURE = "AMD64"
$env:PYTHONIOENCODING = "utf-8"
$env:QD_OFFLINE_CACHE_FILE_PATH = "$PWD/.cache/qd_shell_optimization"
$python = "C:/Users/admin/miniconda3/envs/genesis-world/python.exe"
$nsys = "C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"

& $python examples/vbd/vbd_profile.py --model shell --mode baseline --warmup 60 --steps 200 --repeats 3 --search-radius 0.002 --output-dir out/vbd/recheck_2mm
& $python examples/vbd/vbd_profile.py --model shell --mode baseline --warmup 60 --steps 200 --repeats 3 --search-radius 0.001 --output-dir out/vbd/recheck_1mm
& $python examples/vbd/vbd_profile.py --model shell --mode lines --warmup 60 --steps 200 --search-radius 0.001 --output-dir out/vbd/recheck_lines
& $nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --output=out/vbd/recheck_capture $python examples/vbd/vbd_profile.py --model shell --mode nsys --warmup 60 --steps 200 --search-radius 0.002 --output-dir out/vbd/recheck_capture_data
& $python examples/vbd/vbd_shell.py --gpu --scenario fold --benchmark --steps 200 --search-radius 0.001 --output-dir out/vbd/recheck_fold
& $python -m pytest -n 4 --backend cpu tests/deformable/test_vbd.py -q --tb=short *> out/vbd/recheck_cpu.log
& $python -m pytest -n 4 --backend gpu tests/deformable/test_vbd.py -q --tb=short *> out/vbd/recheck_gpu.log
```

Full local test logs, exploratory source snapshots and Nsight SQLite captures are under
`out/vbd/shell_optimization/`. Portable measurements, kernel summaries and validation excerpts accompany this report.
