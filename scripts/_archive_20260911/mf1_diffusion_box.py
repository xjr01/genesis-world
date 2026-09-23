"""MF-1 validation: PBF concentration field + XSPH-style diffusion in a box domain.

Liquid box fills the lower half (z in [0.01, 0.5]) of the domain; concentration c=1 below
z_mid=0.25 ("coffee"), 0 above ("water"). Diffusion coeff eps_d is CLI-parameterized.

Modes:
  --repo multiflow (default): use the multiflow copy, pass diffusion_coeff / c_init_z_mid.
  --repo main: use the MAIN repo (no diffusion support) for the zero-regression baseline;
               diffusion_coeff and c_init_z_mid are NOT passed.

Per-step CSV: t, sum_c, std_c, c_zbar (z-centroid of c), ke (0.5*sum|v|^2), nan_count.

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 "$PY" \
      multiflow/scripts/mf1_diffusion_box.py --epsd 0.2 --seconds 10
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
        args.out = os.path.join(WORKSPACE, "videos", f"mf1_{tag}.csv")
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

DT = 2e-3
Z_MID = 0.25


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    pbd_kwargs = dict(
        lower_bound=(-0.3, -0.3, 0.0),
        upper_bound=(0.3, 0.3, 0.8),
        particle_size=0.01,
        max_density_solver_iterations=10,
        max_viscosity_solver_iterations=1,
    )
    mat_kwargs = dict(
        sampler="regular",
        rho=1.0,
        density_relaxation=1.0,
        viscosity_relaxation=0.0,
    )
    if args.repo == "multiflow":
        pbd_kwargs["diffusion_coeff"] = args.epsd
        mat_kwargs["c_init_z_mid"] = Z_MID
    else:
        assert args.epsd == 0.0, "baseline (main repo) run must use --epsd 0"

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        pbd_options=gs.options.PBDOptions(**pbd_kwargs),
        show_viewer=False,
    )

    liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(**mat_kwargs),
        morph=gs.morphs.Box(lower=(-0.15, -0.15, 0.01), upper=(0.15, 0.15, 0.5)),
    )
    scene.build()

    solver = scene.sim.pbd_solver
    n_steps = int(args.seconds / DT)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w") as fh:
        fh.write("step,t,sum_c,std_c,c_zbar,ke,nan_count\n")
        for step in range(n_steps + 1):
            if step > 0:
                scene.step()
            pos = solver.particles.pos.to_numpy()[:, 0, :]
            vel = solver.particles.vel.to_numpy()[:, 0, :]
            if args.repo == "multiflow":
                c = solver.particles.c.to_numpy()[:, 0]
            else:
                c = np.zeros(pos.shape[0], dtype=np.float32)  # main repo has no concentration field
            nan_count = int((~np.isfinite(pos)).sum() + (~np.isfinite(vel)).sum() + (~np.isfinite(c)).sum())
            sum_c = float(np.nansum(c))
            std_c = float(np.nanstd(c))
            c_zbar = float(np.nansum(c * pos[:, 2]) / sum_c) if sum_c > 0 else float("nan")
            ke = float(0.5 * np.nansum((vel**2).sum(axis=1)))
            fh.write(f"{step},{step * DT:.6f},{sum_c:.8g},{std_c:.8g},{c_zbar:.8g},{ke:.8g},{nan_count}\n")
            if step % 500 == 0:
                print(f"step {step}/{n_steps}  sum_c={sum_c:.4f} std_c={std_c:.5f} ke={ke:.4g} nan={nan_count}")

    print("n_particles:", liquid.n_particles)
    print("csv saved to", args.out)


if __name__ == "__main__":
    main()
