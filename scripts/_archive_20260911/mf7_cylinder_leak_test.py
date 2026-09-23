"""MF-7 validation: CylinderBoundary clamp for IPBFSolver (leak / impact test).

Physical constraint is the CylinderBoundary position/velocity clamp (center (0,0), R=0.12,
z_bottom=0.0, open top); no cup mesh and no box-wall boundary particles
(boundary_particles=False).

Tests:
  settle: a liquid column (r=0.10, h=0.30) resting in the cylinder domain, 10 s.
  drop:   a liquid column (r=0.08, h=0.15) free-falling from z=0.50 into the cylinder, 10 s.

Pass criteria (asserted at the end of each run):
  1. r_max  <= R + particle_size (= 0.13) for every frame
  2. z_min  >= z_bottom - particle_size (= -0.01) for every frame
  3. nan_count == 0 for every frame
  4. (drop only) max r_max over the run > 0.10  (wall impact actually happened and was clamped)

Outputs:
  multiflow/videos/mf7_settle_metrics.csv
  multiflow/videos/mf7_drop_metrics.csv

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf7_cylinder_leak_test.py --test both --seconds 10
"""

import argparse
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np  # noqa: E402

import genesis as gs  # noqa: E402

print("genesis.__file__ =", gs.__file__)
assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), "Wrong genesis! Must be the multiflow copy."

DT = 1.0 / 60.0
SUBSTEPS = 8
PS = 0.01  # particle_size
R_CYL = 0.12  # cylinder boundary radius
Z_BOTTOM = 0.0


def run_test(name, morph_kwargs, seconds, require_wall_hit):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        ipbf_options=gs.options.IPBFOptions(
            particle_size=PS,
            lower_bound=(-0.14, -0.14, 0.0),
            upper_bound=(0.14, 0.14, 0.7),
            boundary_particles=False,
            boundary_cylinder=(0.0, 0.0, R_CYL, Z_BOTTOM),
            ipbf_iterations=2,
            alpha=1e-8,
            viscosity_xsph=0.1,
        ),
        show_viewer=False,
    )

    liquid = scene.add_entity(
        material=gs.materials.IPBF.Liquid(sampler="regular", rho=1000.0),
        morph=gs.morphs.Cylinder(**morph_kwargs),
    )
    scene.build()

    solver = scene.sim.ipbf_solver
    assert type(solver.boundary).__name__ == "CylinderBoundary", f"unexpected boundary: {type(solver.boundary)}"
    n_fluid = solver._n_fluid_particles
    n_frames = int(round(seconds / DT))

    csv_path = os.path.join(WORKSPACE, "videos", f"mf7_{name}_metrics.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    r_max_ever = 0.0
    z_min_ever = np.inf
    nan_total = 0

    with open(csv_path, "w") as fh:
        fh.write("step,t,r_max,z_min,ke,nan_count\n")
        for frame in range(n_frames + 1):
            if frame > 0:
                scene.step()
            pos = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
            vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
            nan_count = int((~np.isfinite(pos)).sum() + (~np.isfinite(vel)).sum())
            r_max = float(np.nanmax(np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)))
            z_min = float(np.nanmin(pos[:, 2]))
            ke = float(0.5 * np.nansum((vel**2).sum(axis=1)))
            fh.write(f"{frame},{frame * DT:.6f},{r_max:.8g},{z_min:.8g},{ke:.8g},{nan_count}\n")
            r_max_ever = max(r_max_ever, r_max)
            z_min_ever = min(z_min_ever, z_min)
            nan_total += nan_count
            if frame % 60 == 0:
                print(f"[{name}] frame {frame}/{n_frames}  r_max={r_max:.4f} z_min={z_min:.4f} ke={ke:.4g} nan={nan_count}")

    print(f"[{name}] n_particles={n_fluid}  r_max_ever={r_max_ever:.4f}  z_min_ever={z_min_ever:.4f}  nan_total={nan_total}")
    print(f"[{name}] csv saved to {csv_path}")

    ok = True
    checks = [
        ("r_max <= R + ps", r_max_ever <= R_CYL + PS + 1e-6, f"r_max_ever={r_max_ever:.6f} vs limit {R_CYL + PS}"),
        ("z_min >= z_bottom - ps", z_min_ever >= Z_BOTTOM - PS - 1e-6, f"z_min_ever={z_min_ever:.6f} vs limit {Z_BOTTOM - PS}"),
        ("nan_count == 0", nan_total == 0, f"nan_total={nan_total}"),
    ]
    if require_wall_hit:
        checks.append(("wall hit (r_max_ever > 0.10)", r_max_ever > 0.10, f"r_max_ever={r_max_ever:.6f}"))
    for label, passed, detail in checks:
        print(f"[{name}] {'PASS' if passed else 'FAIL'}: {label}  ({detail})")
        ok = ok and passed
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["settle", "drop", "both"], default="both")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    results = {}
    if args.test in ("settle", "both"):
        # liquid column r=0.10, h=0.30 resting on the cylinder bottom
        results["settle"] = run_test(
            "settle",
            dict(radius=0.10, height=0.30, pos=(0.0, 0.0, 0.15)),
            args.seconds,
            require_wall_hit=False,
        )
    if args.test in ("drop", "both"):
        # liquid column r=0.08, h=0.15 free-falling from z=0.50 (bottom face at z=0.425)
        results["drop"] = run_test(
            "drop",
            dict(radius=0.08, height=0.15, pos=(0.0, 0.0, 0.50)),
            args.seconds,
            require_wall_hit=True,
        )

    for name, ok in results.items():
        print(f"== {name}: {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'} ==")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
