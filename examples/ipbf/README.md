# IPBF — Implicit Position-Based Fluids in Genesis

A reproduction of **"Implicit Position-Based Fluids" (IPBF, Diaz et al., SIGGRAPH Asia 2025)**
in Genesis. IPBF rewrites incompressible SPH as an implicit-Euler minimization over particle
positions and solves it with per-particle Newton steps (VBD-style) updated in a relaxed-Jacobi
fashion — no global linear system, unconditionally stable, and converges to low density error
within a few iterations.

The implementation adds a fourth particle-fluid solver alongside SPH/PBD:

- material: `gs.materials.IPBF.Liquid(rho=1000.0, sampler="regular")`
- options: `gs.options.IPBFOptions(...)`
- solver: `genesis/engine/solvers/ipbf_solver.py` (`scene.sim.ipbf_solver`)

## Minimal example

```python
import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1.0 / 60, substeps=8),   # substep h = 1/480 s
    ipbf_options=gs.options.IPBFOptions(
        lower_bound=(-0.5, -0.5, 0.0), upper_bound=(0.5, 0.5, 1.0),
        particle_size=0.01,          # kernel support radius = 2 * particle_size
        ipbf_iterations=2, alpha=0.0,
    ),
    show_viewer=False,
)
scene.add_entity(morph=gs.morphs.Plane())
liquid = scene.add_entity(
    material=gs.materials.IPBF.Liquid(),
    morph=gs.morphs.Box(pos=(-0.25, 0.0, 0.25), size=(0.4, 0.8, 0.5)),
    surface=gs.surfaces.Default(color=(0.4, 0.8, 1.0), vis_mode="particle"),
)
cam = scene.add_camera(res=(1280, 960), pos=(1.8, -1.8, 1.2), lookat=(0, 0, 0.3), fov=35, GUI=False)
scene.build()
cam.start_recording(save_to_filename="ipbf.mp4", fps=60)
for _ in range(300):
    scene.step()
cam.stop_recording()
```

## Examples

All scripts accept `--backend {cpu,gpu}` and write to `examples/ipbf/output/` (created
automatically). When collected by `tests/test_examples.py` (i.e. `PYTEST_VERSION` is set),
each script automatically switches to a tiny CPU smoke run.

| Script | What it shows | Command |
|---|---|---|
| `01_gravity_skeleton.py` | Hello world: liquid block drops and splashes. | `python examples/ipbf/01_gravity_skeleton.py` |
| `02_dam_break.py` | Single-column dam break; `--bigstep` for h=1/30 s, `--iters`, `--damping`. | `python examples/ipbf/02_dam_break.py` |
| `03_density_check.py` | Resting block; logs avg/max density constraint C to CSV. | `python examples/ipbf/03_density_check.py` |
| `04_settling_damping.py` | Damping A/B test (paper Fig. 13): kinetic-energy decay, damped video. | `python examples/ipbf/04_settling_damping.py --alpha-star 1.0` |
| `05_double_dam_break.py` | Double dam break benchmark (paper Table 1): ms/frame + density error + video. | `python examples/ipbf/05_double_dam_break.py` |
| `06_large_scale.py` | 384k-particle block flop benchmark + video. | `python examples/ipbf/06_large_scale.py --iters 1` |
| `07_convergence.py` | Density error vs iteration count (paper Fig. 10), same-state single-step protocol. | `python examples/ipbf/07_convergence.py` |

## Key parameters (`IPBFOptions`)

| Parameter | Default | Meaning |
|---|---|---|
| `particle_size` | 0.02 | Particle diameter (m); kernel support radius = 2 × particle_size. |
| `ipbf_iterations` | 2 | Newton (relaxed-Jacobi) iterations per substep. |
| `alpha` | 0.0 | Compliance of the pressure energy (1/k). 0 = infinite stiffness (paper default). |
| `damping_enabled` | False | Artificial damping (paper §3.6, eqs. 16-18). Off by default: with the current scale-rescaled `damping_alpha_star` it visibly thickens the flow (tuning left to the user, see `04`). |
| `damping_alpha_star` | 3.2e6 | Compliance of the alternative low-stiffness solution. Dimensional: paper's 1e-3 at kernel radius 1, rescaled here for radius 0.02 m. |
| `damping_beta` | 60.0 | Damping proximity threshold, in units of the kernel radius (paper: 60). |
| `boundary_particles` | True | Static Akinci-style boundary particles on the domain walls, included in the density sum. |
| `boundary_layers` | 2 | Boundary particle layers, starting one particle_size outside the wall plane. |

## Implementation notes

- Cubic spline kernel (Koschier et al. 2019), support radius R = 2 × particle_size; analytic
  first/second radial derivatives and kernel Hessian assembled per neighbor pair.
- Per-particle 3×3 Newton step: force and Hessian from eqs. 10/11 of the paper; the indefinite
  geometric-stiffness term is replaced by its column-norm diagonal approximation (Andrews et
  al. 2017) — required for stability (paper Fig. 15). Solved with an explicit
  adjugate/determinant inverse; degenerate Hessians (isolated particle / fully clamped
  neighborhood) yield a zero step.
- Relaxed Jacobi update: all particles compute Δx in parallel from the same positions, then
  x ← x + Δx/2 (fixed relaxation 1/2); neighbor structure is rebuilt once per substep and
  densities/derivatives are recomputed every iteration.
- Density constraint C = ρ/ρ₀ − 1 in volume-normalized form, clamped at 0 (negative-pressure
  clamp, enabled in all paper tests). Velocities are recovered PBD-style: v = (xⁿ⁺¹ − xⁿ)/h.
- Static boundary particles (Akinci et al. 2012 style) contribute only to fluid densities and
  constraint gradients; they carry no constraint and are never updated or rendered. A
  CubeBoundary position clamp remains as a fallback.

## Measured performance & accuracy (RTX 5090 D, this implementation)

| Scene | Particles | Iters | ms/frame | Density error |
|---|---|---|---|---|
| Double dam break | 288k fluid + 102k boundary | 2 | 23.2 | avg_C 6.1e-3 (active sloshing window) |
| Block flop | 384k fluid + 94k boundary | 2 | 67.0 | avg_C 6.7e-3 (final frame) |
| Block flop | 384k fluid + 94k boundary | 1 | 34.0 | avg_C 2.4e-2 (final frame) |
| Convergence (dam break, single step) | — | 1/2/4/8 | — | avg_C 1.7e-2 → 3.2e-3 → 1.2e-3 → 2.9e-4 |

Paper reference numbers (RTX 4090): 70 ms/frame for double dam break (their particle count not
reported), ~50 ms/frame for 250k particles @1 iteration; avg density error 9.2e-4. Our density
error is measured during active sloshing with damping off and boundary particles on, so the
comparison is indicative only. Timings include the per-substep hash-grid rebuild.

## Known limitations

- **h = 1/30 s (`--bigstep`)**: the fluid does not blow up but over-compresses into a thin
  layer on the floor (a degenerate fixed point of the fallback wall clamp combined with only
  2 iterations per step). Use h ≤ 1/240 s for production.
- **Artificial damping is off by default**: the scale-rescaled `damping_alpha_star` needs
  per-scene tuning (see `04_settling_damping.py` for the A/B tooling).
- No rigid-body coupling for IPBF particles yet, no raytracer integration, and no
  differentiable interface (stubs only).
