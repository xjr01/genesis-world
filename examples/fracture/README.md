# PBD vase fracture demo

A ceramic vase is perched on the edge of a table, nudged sideways so it topples off and slams into
the floor. The vase is modeled as a PBD solid with bonded particles: bonds break progressively under
stretch, shear, and relative rotation, producing realistic crack propagation along pre-seeded fragment
boundaries. Separated fragments are automatically handed to the native rigid-body solver through the
`PBDRigidFragmentBridge`, so the shattered pieces bounce and scatter under rigid-body contact dynamics
while the PBD render mesh remains the single visual representation.

A second demo in this directory, `pbstf_vase_water_fracture.py`, fills a half-scale copy of the
same vase with PBSTF water (see the "PBSTF water-filled vase" section below).

## Running

Requires a CUDA GPU with a matching PyTorch build and Genesis installed from this repository
(`python -m pip install -e ".[dev]"`). Run from the repository root.

```bash
python examples/fracture/pbd_vase_fracture.py --record
```

With an interactive viewer instead of headless recording:

```bash
python examples/fracture/pbd_vase_fracture.py --vis
```

Shorter smoke test (writes no video):

```bash
python examples/fracture/pbd_vase_fracture.py --duration 3
```

## Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `-v`, `--vis` | off | Show the interactive Genesis viewer. |
| `-r`, `--record` | off | Record an MP4 video of the shatter to `--output`. |
| `--duration` | 8.0 | Simulation duration in seconds. |
| `--output` | `pbd_vase_fracture.mp4` | Recording output path (created only with `--record`). |
| `--fragments` | 12 | Number of pre-seeded fragments (4–64); with `regular` the actual count equals sectors × bands. |
| `--seed-layout` | `regular` | Fragment seed layout: `regular` (ring × sector grid) or `random` (volumetric variable-density). |
| `--layout-seed` | 3 | RNG seed for the `random` layout; deterministic across runs. |

## Key simulation parameters (set in source)

These are hardcoded in `main()` of `pbd_vase_fracture.py` and control the fracture behavior. The
scene uses Z-up, lengths in metres, and a 5 mm PBD particle spacing. To change the table dimensions,
initial pose, or material, edit `main()` directly.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `fracture_threshold` | 0.11 | Bond distance strain threshold; lower = fragments separate more easily. |
| `shear_threshold` | 0.2 | Bond shear strain threshold; controls diagonal cracking. |
| `rotation_threshold` | 0.25 | Relative rotation threshold (rad) for bond failure. |
| `seam_failure_threshold` | 1.0 | Seam damage threshold at fragment interfaces. |
| `bond_radius_factor` | 1.15 | Bond search radius multiplier; higher = more neighbor bonds. |
| `fragment_seeds` | set by `--seed-layout` + `--fragments` | Voronoi seed layout; `regular` uses a ring × sector grid, `random` samples the wall volume with variable-density Poisson-disk (see `genesis/utils/fracture_seeds.py`). |
| `dt` | 2e-3 s | Simulation timestep per outer step. |
| `substeps` | 10 | PBD solver iterations per outer step. |

## Outputs

With `--record`, an MP4 (960x720, 60 fps) is written to the `--output` path, showing the vase tipping
over the table edge, the progressive fracture, and the shards scattering across the floor in slow
motion (`realtime_factor=0.24`). The simulation runs for `--duration` seconds of physical time.

## Performance

Measured on an RTX 4060 with warm JIT caches and no recording: roughly 100-130 ms per outer step
(10 substeps x 2 ms dt), or 9-10 FPS. The 8 s default duration (4000 outer steps) takes about
7 minutes wall-clock.

## Known limitations

- Fracture behavior is timestep-dependent: at coarser timesteps or fewer substeps, bond breaking and
  contact response shift.
- Fragment collision proxies are convex hulls; concave features (the vase mouth, thin-wall curvature)
  are approximated, so resting pieces may visually sink slightly into the floor.
- The contact time constant is clamped to twice the substep timestep by the solver; slammed fragments
  may briefly penetrate the floor before the contact force ramps.

## Impulse-triggered shatter demo

`impulse_shatter.py` demonstrates the alternative fracture path: a pure rigid-body scene where a
pre-fragmented object (vase, bowl, or plate) topples off the table and shatters on the calibrated floor
impact. A throwaway PBD pre-pass partitions the source mesh (`genesis/utils/fracture_seeds.py`), builds
the mass model, and exports per-fragment assets: the cut render meshes and a VHACD convex decomposition
of each closed fragment as the collision source geometry (fused by the loader into one watertight SDF
collision geom per fragment). The delivered scene is rigid-only: a compound parent body
carries surface-hull collision proxies until `PBDRigidFragmentBridge.arm_impact_trigger()` fires, then
all fragment links are released at once with the parent's exact state.

Each preset pins the md5 of its calibrated seed partition, so the fragment layout reproduces the
validated one bit-for-bit (`--skip-seed-gate` lifts the pin for exploring new layouts). Impact
thresholds are per-object impulse sums in N*s from the calibration ladder (gentle placements stay
event-free, the floor slam fires); `--threshold` overrides.

Requires the `fracture` extra for the seed sampler and the convex decomposition backend:

```bash
python -m pip install -e ".[fracture]"
python examples/fracture/impulse_shatter.py --object vase --video
python examples/fracture/impulse_shatter.py --object bowl --video
python examples/fracture/impulse_shatter.py --object plate --video
```

First run for an object takes several minutes (pre-pass + VHACD + mesh build); assets are cached under
`examples/fracture/.cache/<object>_seed<seed>/` and reused afterwards. Outputs land in `--out-dir`
(default `<work-dir>/run/`): `summary.json` (fire step, switch continuity errors, per-fragment rest
penetration, visual rest gate), `metrics.npz` (per-step traces), frames and `shatter_720p.mp4` with
`--video`. `--steps 300` gives a quick smoke test that reaches the trigger and release.

## Related engine capabilities

- **Seam damage time model**: `PBD.Solid` accepts `seam_damage_time_constant` and related parameters
  that model gradual seam degradation under cyclic loading. Not exposed as a CLI flag; set in source.

## PBSTF water-filled vase (`pbstf_vase_water_fracture.py`)

The same vase at half its asset scale (`VASE_SCALE = 0.5` in the source; the table, floor, and fall
height stay world-sized) holds a low fill of PBSTF liquid and topples off the table; the shards
scatter and the water escapes through the broken wall and spreads across the floor. The fragment
layout is selectable: `--seed-layout regular` keeps the fixed 24-fragment Voronoi wall grid shared
with `pbd_vase_fracture.py`, while `--seed-layout random` samples variable-density volumetric seeds
through `genesis/utils/fracture_seeds.py` and accepts any resulting fragment count in 18-30
(`--layout-seed 3` yields 28). The liquid boundary is a single hidden FEM tetrahedral proxy: each
fragment's
convex hull fans into tetrahedra over an interior center, the 24 fans merge into one mesh with
per-fragment independent vertices, and that entity drives one
`PBSTFAbsorbentBoxStaticColliderOptions` deformable collider. The collider box itself shrinks to a
single voxel with zero total capture capacity, so it absorbs nothing and only carries the moving
boundary. The runtime checks this invariant and reports the collider capacity and per-fragment proxy
deviation in its metrics. The coupling stays strictly one-way: fragments push the water, and the water
exerts no reaction on vase or shards. Before the first step and before every step the fragment links'
world poses (one batched public readback) rewrite the proxy vertices through `FEMEntity.set_position`,
and the PBSTF solver refreshes the collider surface from the proxy at the next substep.

### Running

The bundled assets in `pbstf_vase_water_assets/` (per-fragment proxy OBJs plus `water_particles.npz`
holding the filtered water spawn and the merged tetra proxy) match the source-fixed fragment layout,
so the demo runs directly. The regular layout lives in the directory root (24 OBJs); every random
layout seed keeps its own `random_seed_<n>/` subdirectory (seed 3 bundles 28 OBJs). At the 2 mm
liquid particle size the half-scale fill is 14842 particles for the regular layout and 17525 for
random seed 3 (the full-size vase sampled 115283). Outputs land in the workspace-root
`experiments/vase_pbstf_fracture/` by default; the asset directory only ever holds the OBJs and the
NPZ (mesh and particle caches go to the user cache directory):

```bash
python examples/fracture/pbstf_vase_water_fracture.py --record --metrics-output metrics.json --keyframe-dir keyframes
```

Inspect the floor-release case (no push, water co-translated with the vase). In the current local
CUDA verification, all 24 fragment handoffs occurred within the first 0.02 s and the water spread as
the pieces separated. This mode is useful for inspecting that release behavior, but it does not
demonstrate a long-lasting intact filled vase:

```bash
python examples/fracture/pbstf_vase_water_fracture.py --settle-only --duration 3
```

Regenerate the bundled assets after changing the vase layout, fill height, or particle size in the
source; random layouts regenerate into their own subdirectory. The runtime verifies asset/scene
consistency and refuses mismatches:

```bash
python examples/fracture/pbstf_vase_water_fracture.py --prepare-proxies
python examples/fracture/pbstf_vase_water_fracture.py --prepare-proxies --seed-layout random --layout-seed 3
```

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `-v`, `--vis` | off | Show the interactive Genesis viewer. |
| `-r`, `--record` | off | Record an MP4 to `--output`. |
| `--duration` | 8.0 | Simulation duration in seconds; must be positive. |
| `--output` | `experiments/vase_pbstf_fracture/pbstf_vase_water_fracture.mp4` | Recording path. |
| `--prepare-proxies` | off | Regenerate the 24 OBJs and water NPZ, then exit. |
| `--settle-only` | off | Place the filled vase on the floor without the initial push. |
| `--water-render` | `recon` | Render the liquid as `recon` (reconstructed surface) or `particle`. |
| `--seed-layout` | `regular` | Fragment seed layout: `regular` (fixed 24-fragment wall grid) or `random` (volumetric variable-density, 18-30 fragments). |
| `--layout-seed` | 3 | RNG seed for the `random` layout; also names its asset subdirectory. |
| `--metrics-output` | none | Write final metrics and sampled wall-clearance diagnostics to JSON. |
| `--keyframe-dir` | none | Save rendered PNG keyframes in this directory. |
| `--keyframe-times` | mode-specific | Optional list of keyframe times in seconds. Drop defaults are 0.2, 0.45, 0.7, 1.2, and 3.0 s, plus the final state. |

### Recording and collision structure

The scene advances with `dt=0.002` s and 10 substeps. The default 8 s run therefore executes 4000
outer steps. Recording uses `realtime_factor=1.0` and `fps=50`, sampling one frame every 10 outer
steps; an 8 s recording contains 400 frames at 1280x720. Keyframes are separate point samples, not a
record of every substep.

PBSTF receives two static collider configurations. One is the zero-capacity deformable collider
attached to the single hidden FEM fragment boundary described above. The other is an analytic
`PBSTFBoxStaticColliderOptions` matching the rigid table volume to provide water-table collision.
The final metrics report both the total collider count and the deformable collider count,
the absorption capacity, and the number of water particles inside the table volume.

Measured on an RTX 4060 at half scale with 2 mm liquid particles and recon recording: 1.5-1.8
outer steps per second; the default 8 s run (4000 steps) spends 41-43 minutes in the stepping
loop and 42-50 minutes end to end including kernel build and video finalization.

### Notes and limitations

- The liquid boundary is the convex hull of each fragment, so the curved wall is approximated by
  chord facets bulging a few millimetres into the cavity; the water spawn keeps a matching clearance
  (0.65 x PBSTF particle size) to every hull, and small water-shard gaps remain visible after the
  break. Use `--water-render particle` to diagnose the raw particle state if the reconstructed
  surface looks off. The reconstructed surface can appear as blobs or connected clumps rather than
  a resolved thin sheet.
- The single deformable collider tracks all 24 fragment hulls through one triangle BVH, refreshed
  from the proxy every substep; the proxy is rigidly posed from the fragment links each frame, so no
  elastic deformation enters the boundary.
- Coupling is one-way: vase fragments push the water, with no water reaction on the vase or shards.
- Each layout's bundled proxies, water spawn, tetra proxy, and runtime registration stay in
  lockstep; switching layouts or layout seeds requires regenerating that layout's assets explicitly.
- The hidden FEM proxy is re-posed from the fragment links every step, so its Young's modulus only
  bounds the explicit-integrator stiffness of the thinnest fan tetrahedra; it scales with
  `VASE_SCALE^2` to stay inside the substep stability limit, which the flat shards of a random
  layout otherwise sit on top of.
- Sampled hull diagnostics are not a per-substep non-penetration guarantee. In the half-scale
  random-layout 8 s run, no particle sat inside a fragment hull through 0.05 s (minimum signed
  distance +0.13 mm in a dedicated 0.1 s diagnostic); during the 0.08-0.45 s fracture sweep the
  separating hulls transiently dipped up to 122 of 17525 particles inside (minimum about -2.7 mm
  at the 0.2 s sample), and every sample from 0.45 s on reported zero inside particles.
