"""MF-5 validation: IPBF concentration field + XSPH-style diffusion in a box domain.

Two stacked liquid boxes in one IPBF solver: coffee box (z in [0.01, 0.15], c_init=1.0) below,
water box (z in [0.15, 0.30], c_init=0.0) above. Runs under gravity from the start (the stack
sits 1 particle size above the floor, so it settles almost immediately).

Modes:
  --repo multiflow (default): use the multiflow copy, pass diffusion_coeff / material c_init.
  --repo main: use the MAIN repo (no diffusion support) for the zero-regression baseline;
               diffusion_coeff and c_init are NOT passed (main repo IPBF lacks both).

Per-frame CSV: step, t, sum_c, std_c, ke, nan_count (fluid particles only).

Pass criteria (checked by the caller):
  1) epsd=0.1, 10 s: sum_c relative drift < 1%; std_c final < 70% of initial; nan_count = 0.
  2) epsd=0, 5 s: std_c drift < 5%; KE curve vs --repo main run bitwise-identical for the first
     4 frames, later divergence within same-code rerun noise (GPU scheduling noise).
  3) particles_render.c readback matches particles.c (spot-checked once mid-run here).

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 "$PY" \
      multiflow/scripts/mf5_ipbf_diffusion_box.py --epsd 0.1 --seconds 10
"""

import argparse
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
MAIN_REPO = "D:/workspace/python-workspace/ipbf/genesis-world"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsd", type=float, default=0.0, help="diffusion_coeff (0 = off)")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--repo", choices=["multiflow", "main"], default="multiflow")
    parser.add_argument("--out", default=None, help="CSV output path")
    args = parser.parse_args()
    if args.out is None:
        tag = "baseline" if args.repo == "main" else f"epsd{args.epsd:g}"
        args.out = os.path.join(WORKSPACE, "videos", f"mf5_{tag}.csv")
    return args


args = parse_args()

if args.repo == "main":
    sys.path.insert(0, MAIN_REPO)
else:
    sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np

import genesis as gs

print("genesis.__file__ =", gs.__file__)
if args.repo == "main":
    assert os.path.normpath(gs.__file__).startswith(os.path.normpath(MAIN_REPO))
else:
    assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE))

DT = 1.0 / 60.0
SUBSTEPS = 8


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    ipbf_kwargs = dict(
        particle_size=0.01,
        lower_bound=(-0.2, -0.2, 0.0),
        upper_bound=(0.2, 0.2, 0.6),
        boundary_particles=True,
        ipbf_iterations=2,
        alpha=1e-8,
        viscosity_xsph=0.1,
        damping_enabled=False,
    )
    coffee_mat = dict(sampler="regular", rho=1000.0)
    water_mat = dict(sampler="regular", rho=1000.0)
    if args.repo == "multiflow":
        ipbf_kwargs["diffusion_coeff"] = args.epsd
        coffee_mat["c_init"] = 1.0
        water_mat["c_init"] = 0.0
    else:
        assert args.epsd == 0.0, "baseline (main repo) run must use --epsd 0"

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        ipbf_options=gs.options.IPBFOptions(**ipbf_kwargs),
        show_viewer=False,
    )

    coffee = scene.add_entity(
        material=gs.materials.IPBF.Liquid(**coffee_mat),
        morph=gs.morphs.Box(lower=(-0.1, -0.1, 0.01), upper=(0.1, 0.1, 0.15)),
    )
    water = scene.add_entity(
        material=gs.materials.IPBF.Liquid(**water_mat),
        morph=gs.morphs.Box(lower=(-0.1, -0.1, 0.15), upper=(0.1, 0.1, 0.30)),
    )
    scene.build()

    solver = scene.sim.ipbf_solver
    n_fluid = solver._n_fluid_particles
    n_frames = int(round(args.seconds / DT))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    render_check_done = False

    with open(args.out, "w") as fh:
        fh.write("step,t,sum_c,std_c,ke,nan_count\n")
        for frame in range(n_frames + 1):
            if frame > 0:
                scene.step()
            pos = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
            vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
            if args.repo == "multiflow":
                c = solver.particles.c.to_numpy()[:n_fluid, 0]
            else:
                c = np.zeros(n_fluid, dtype=np.float32)  # main repo has no concentration field
            nan_count = int((~np.isfinite(pos)).sum() + (~np.isfinite(vel)).sum() + (~np.isfinite(c)).sum())
            sum_c = float(np.nansum(c))
            std_c = float(np.nanstd(c))
            ke = float(0.5 * np.nansum((vel**2).sum(axis=1)))
            fh.write(f"{frame},{frame * DT:.6f},{sum_c:.8g},{std_c:.8g},{ke:.8g},{nan_count}\n")
            if frame % 60 == 0:
                print(f"frame {frame}/{n_frames}  sum_c={sum_c:.4f} std_c={std_c:.5f} ke={ke:.4g} nan={nan_count}")

            # criterion 3: render-field readback spot check, once mid-run
            if args.repo == "multiflow" and not render_check_done and frame == min(30, n_frames):
                solver.update_render_fields()
                c_render = solver.particles_render.c.to_numpy()[:n_fluid, 0]
                c_state = solver.particles.c.to_numpy()[:n_fluid, 0]
                max_diff = float(np.max(np.abs(c_render - c_state)))
                print(f"render-check @frame {frame}: max |particles_render.c - particles.c| = {max_diff:.3g}")
                assert max_diff == 0.0, "particles_render.c mismatch"
                render_check_done = True

    print("n_fluid_particles:", n_fluid, "(coffee:", coffee.n_particles, " water:", water.n_particles, ")")
    print("csv saved to", args.out)


if __name__ == "__main__":
    main()
