# End-slice validation

The example was checked on the `merge` base `d57c09f4`, using an independent
worktree and virtual environment. Runtime imports resolved to that worktree's
Genesis and example sources. The package dependency check passed. No source files
from another project, external asset files or proprietary renderer were required.

The environment used Windows, Python 3.11.15, an NVIDIA RTX 5090 D v2,
PyTorch 2.11.0+cu128, Quadrants 1.2.0, NumPy 2.4.6, SciPy 1.17.1,
Matplotlib 3.11.0, PyAV 18.0.0 and trimesh 4.12.2. Hardware and compilation-cache
state affect timings; these are one-run measurements, not performance guarantees.

## Complete operation

The end-slice configuration completed 230,000 integration steps, 5.75 seconds of
physical motion, and 139 output frames. All 720,000 butter particles and 325,404
bread particles remained active. The decoded particle video passed frame-count,
frame-rate, dimensions, nonempty-frame and changing-frame checks.

| Measurement | Result |
| --- | ---: |
| Slice pickup | 99.8546% |
| Slice particles in the final coating zone | 93.3042% |
| Maximum bread particle displacement | 0.965 mm |
| Maximum prescribed blade pose error | 0.0000103 mm |
| Maximum sampled interior blade penetration | 0.00000813 mm |
| Sampled interior side switches | 0 |
| Maximum post-step contact correction | 0.00366 mm |
| Initialization | 36.74 s |
| Integration, including sampling | 722.26 s |
| Complete simulation, including export | 822.04 s |
| Particle-video encoding and input checks | 10.47 s |

The coating zone is the bread footprint within +/-4 mm of its nominal top.
Interior crossing checks exclude flow around blade edges. Sampled-frame results
do not prove continuous-time nonpenetration. No surface-reconstruction or
photorealistic-rendering acceptance is claimed by this solver-only example.

The three focused tests passed: prescribed motion/geometry and angular-velocity
consistency; two-sided wet adhesion, release and unilateral correction on the GPU;
and detection of intermediate-frame particle-ID changes by the comparison tool.
The repository's pre-commit checks also passed. The complete repository test suite
was not run; changes are confined to this new example and its dedicated test.

## Cross-run numerical comparison

The independently generated particle trajectory was compared against a prior
end-slice run with identical effective settings. Initial particles, knife geometry,
sample times and prescribed tool poses matched. The contact executable body was
unchanged when moved into the self-contained example.

| Comparison over all 139 frames | Difference |
| --- | ---: |
| Butter aggregate RMS position error | 0.221 mm |
| Largest per-frame butter RMS | 0.338 mm |
| Largest per-frame butter 99th-percentile error | 0.0386 mm |
| Largest individual butter error | 110.17 mm |
| Largest individual bread error | 0.105 mm |
| Blade position / quaternion difference | 0 / 0 |
| Mean decoded-video pixel absolute difference, 0-255 | 0.137 |

The strict particle-ID comparison **did not pass** its 0.1 mm per-frame butter
RMS limit. Its other position limits passed. At the final frame, 11 of 720,000
butter particles differed by more than 10 mm; their initial positions lay near
the cut plane. Different retention decisions send these few particles toward
different boards. The threshold and failed result were retained.

Pickup differed by 0.00265 percentage points and coating-zone deposition by
0.01058 percentage points. Frame-by-frame video review showed matching bulk
cutting, transport and spreading behavior. This supports reproduction of the
demonstrated effect, **not exact reproduction of every individual particle path**.
Particle-ID-sensitive datasets must use the actual exported state of each run.

An additional full run of the unchanged reference code and configuration also
failed the same strict comparison. Its aggregate butter RMS was 0.256 mm, maximum
per-frame RMS 0.370 mm, maximum per-frame p99 0.0399 mm, and maximum individual
error 110.17 mm. Its maximum bread error was 0.145 mm. These measured repeat
differences were larger than the corresponding clean-worktree differences above.
This control supports ordinary repeat-run sensitivity near attachment boundaries
as the explanation in this case; it does not establish a universal tolerance or
deterministic behavior on other hardware. No comparison limit was loosened.
