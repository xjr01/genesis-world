# PBD vase fracture demo

A ceramic vase is perched on the edge of a table, nudged sideways so it topples off and slams into
the floor. The vase is modeled as a PBD solid with bonded particles: bonds break progressively under
stretch, shear, and relative rotation, producing realistic crack propagation along pre-seeded fragment
boundaries. Separated fragments are automatically handed to the native rigid-body solver through the
`PBDRigidFragmentBridge`, so the shattered pieces bounce and scatter under rigid-body contact dynamics
while the PBD render mesh remains the single visual representation.

A second demo in this directory, `pbstf_vase_water_fracture.py`, fills the same vase with PBSTF water
(see the "PBSTF water-filled vase" section below).

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
| `--fragments` | 12 | Number of pre-seeded fragments (4-64); the actual count equals sectors x bands. |

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
| `fragment_seeds` | set by `--fragments` | Voronoi seed layout; controlled by the `--fragments` CLI flag. |
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

## Related engine capabilities

- **Impulse-triggered brittle fracture**: `PBDRigidFragmentBridge.arm_impact_trigger()` releases all
  fragments at once when a calibrated impact impulse exceeds a threshold, bypassing progressive PBD
  bond breaking. Not demonstrated in this example; API in
  `genesis/engine/couplers/pbd_rigid_fragments.py`.
- **Seam damage time model**: `PBD.Solid` accepts `seam_damage_time_constant` and related parameters
  that model gradual seam degradation under cyclic loading. Not exposed as a CLI flag; set in source.
- **Non-convex collision proxies**: collision accuracy for resting fragments can be improved by
  replacing the convex hull proxy with a VHACD convex decomposition (multiple convex geoms per
  fragment link). Not demonstrated in this example.

## PBSTF water-filled vase (`pbstf_vase_water_fracture.py`)

The same vase (same fixed 24-fragment Voronoi wall layout) holds a low fill of PBSTF liquid and
topples off the table; the shards scatter and the water escapes through the broken wall and spreads
across the floor. The liquid boundary is a single hidden FEM tetrahedral proxy: each fragment's
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

The bundled assets in `pbstf_vase_water_assets/` (24 proxy OBJs plus `water_particles.npz` holding
the filtered water spawn and the merged tetra proxy) match the source-fixed fragment layout, so the
demo runs directly. Outputs land in the workspace-root `experiments/vase_pbstf_fracture/` by
default; the asset directory only ever holds the OBJs and the NPZ (mesh and particle caches go to
the user cache directory):

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
source. The runtime verifies asset/scene consistency and refuses mismatches:

```bash
python examples/fracture/pbstf_vase_water_fracture.py --prepare-proxies
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
- The fixed 24-fragment layout keeps the bundled proxies, the water spawn, the tetra proxy and the
  runtime registration in lockstep; other layouts require regenerating the assets explicitly.
- Sampled hull diagnostics are not a per-substep non-penetration guarantee. In an independent 1.2 s
  table-collider verification, 8 of 1463 water particles were inside fragment hulls at the 0.5 s
  fracture sample, with a minimum signed distance of about -0.904 mm; the 0.7 s sample returned to
  zero inside particles. The default drop keyframe times do not include 0.5 s, so the 8 s metrics
  do not independently resolve that transient.
