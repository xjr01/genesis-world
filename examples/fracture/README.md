# PBD vase fracture demo

A ceramic vase is perched on the edge of a table, nudged sideways so it topples off and slams into
the floor. The vase is modeled as a PBD solid with bonded particles: bonds break progressively under
stretch, shear, and relative rotation, producing realistic crack propagation along pre-seeded fragment
boundaries. Separated fragments are automatically handed to the native rigid-body solver through the
`PBDRigidFragmentBridge`, so the shattered pieces bounce and scatter under rigid-body contact dynamics
while the PBD render mesh remains the single visual representation.

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

With `--record`, an MP4 (960×720, 60 fps) is written to the `--output` path, showing the vase tipping
over the table edge, the progressive fracture, and the shards scattering across the floor in slow
motion (`realtime_factor=0.24`). The simulation runs for `--duration` seconds of physical time.

## Performance

Measured on an RTX 4060 with warm JIT caches and no recording: roughly 100–130 ms per outer step
(10 substeps × 2 ms dt), or 9–10 FPS. The 8 s default duration (4000 outer steps) takes about
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
