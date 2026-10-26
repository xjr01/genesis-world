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
