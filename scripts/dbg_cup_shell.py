"""dbg: standalone validation of the second (static upright) shell slot PBDOptions.boundary_cup_shell.

Scene = cup shell + infinite table plane ONLY (no boundary_cylinder, no pitcher): a water blob
(r=0.05, released at z=0.10 inside the cavity) falls, splashes and settles in the cup. Exercises
the three shell code paths added for the second slot: per-substep world-solid impose
(_kernel_solve_boundary_collision), margin-gated per-iteration projection (PBD_APPLY_CLAMP) and
the global nearest-surface adhesion arbitration (_adh_query_nearest).

Pass criteria (3 s):
  [1] nan == 0 the whole run
  [2] no particle leaks through the shell onto the table:
      z < -ps  OR  (r > R_out + 3 ps AND z < z_bot), r measured from the cup axis
  [3] settled water stays inside the cavity: final r_max <= R_in + 2 ps

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \\
    "$PY" multiflow/scripts/dbg_cup_shell.py
"""

import os
import sys
import time

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np  # noqa: E402

import genesis as gs  # noqa: E402

print("genesis loaded from:", gs.__file__)
assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), "Wrong genesis! Must be the multiflow copy."

DT = 1.0 / 60.0
SUBSTEPS = 8
SECONDS = 3.0

# cup shell geometry (option tuple layout: cx, cy, z_bot, r_in, L, t_wall, t_bottom, lip_round)
CX, CY = 0.0, 0.0
Z_BOT = 0.016  # inner cavity floor (outer bottom sits at z_bot - t_bottom = 0.0 = table plane)
R_IN = 0.15
L = 0.30
T_WALL = 0.016
T_BOTTOM = 0.016
LIP_ROUND = 0.008
R_OUT = R_IN + T_WALL
Z_TABLE = 0.0

# water blob: released inside the cavity, falls onto the cup floor
R_DROP = 0.05
Z_DROP = 0.10

PS = 0.008


def main():
    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
        pbd_options=gs.options.PBDOptions(
            particle_size=PS,
            lower_bound=(-0.40, -0.40, 0.0),
            upper_bound=(0.40, 0.40, 0.80),
            boundary_cup_shell=(CX, CY, Z_BOT, R_IN, L, T_WALL, T_BOTTOM, LIP_ROUND),
            boundary_plane=(Z_TABLE,),
            max_density_solver_iterations=20,
            max_viscosity_solver_iterations=1,
            density_lambda_epsilon=0.1,
            surface_tension_enabled=True,
            st_compliance=1.0,
            st_surface_density_factor=1.0,
            st_distance_enabled=True,
            st_distance_compliance=40.0,
            st_max_surface_neighbors=256,
            velocity_damping=1.0,
            wall_adhesion_enabled=True,
            wall_adhesion_compliance=20.0,
            wall_friction=0.0,
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )

    water = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=0.0,
            density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Sphere(radius=R_DROP, pos=(CX, CY, Z_DROP)),
    )
    scene.build()
    print(f"water particles: {water.n_particles}")
    print(
        f"cup shell: r_in={R_IN} L={L} t_wall={T_WALL} t_bottom={T_BOTTOM} z_bot={Z_BOT} "
        f"lip_round={LIP_ROUND} | r_out={R_OUT:.3f} rim z={Z_BOT + L:.3f} | plane z={Z_TABLE}"
    )

    solver = scene.sim.pbd_solver
    assert type(solver.boundary).__name__ == "CubeBoundary", f"unexpected boundary: {type(solver.boundary)}"
    assert type(solver.boundary_cup).__name__ == "TiltedCylinderShellBoundary", (
        f"unexpected boundary_cup: {type(solver.boundary_cup)}"
    )
    assert type(solver.boundary3).__name__ == "PlaneBoundary", f"unexpected boundary3: {type(solver.boundary3)}"
    assert solver.boundary2 is None and solver._has_boundary_shell and not solver._pitcher_is_shell
    n_fluid = solver._n_fluid_particles

    n_frames = int(round(SECONDS / DT))
    print_every = int(round(0.5 / DT))
    n_leak_max = 0
    nan_max = 0
    r_max_final = 0.0
    z_min_final = 0.0
    n_under_final = 0
    wall0 = time.time()

    for i in range(n_frames + 1):
        t = i * DT
        if i > 0:
            scene.step()

        pos = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
        vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
        nan = int((~np.isfinite(pos)).sum() + (~np.isfinite(vel)).sum())
        nan_max = max(nan_max, nan)
        safe = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        r = np.sqrt((safe[:, 0] - CX) ** 2 + (safe[:, 1] - CY) ** 2)
        z = safe[:, 2]
        r_max = float(r.max())
        z_min = float(z.min())
        n_leak = int(((z < -PS) | ((r > R_OUT + 3.0 * PS) & (z < Z_BOT))).sum())
        n_under = int((z < Z_BOT - 0.5 * PS).sum())  # diagnostic: below the cavity floor level
        n_leak_max = max(n_leak_max, n_leak)
        ke = float(0.5 * np.nansum((vel**2).sum(axis=1)))
        if i == n_frames:
            r_max_final, z_min_final, n_under_final = r_max, z_min, n_under
        if i % print_every == 0 or i == n_frames:
            print(
                f"t={t:5.2f}s  n={n_fluid}  z_min={z_min:+.4f}  r_max={r_max:.4f}"
                f"  z_max={float(z.max()):.4f}  leak={n_leak}  n_under={n_under}  KE={ke:.3e}  nan={nan}"
            )

    p1 = nan_max == 0
    p2 = n_leak_max == 0
    p3 = r_max_final <= R_IN + 2.0 * PS
    print("=" * 70)
    print(f"[1] nan       max = {nan_max}                                  -> {'PASS' if p1 else 'FAIL'}")
    print(
        f"[2] leak      max = {n_leak_max}  (z < -{PS} OR r > {R_OUT + 3.0 * PS:.3f} AND z < {Z_BOT})"
        f" -> {'PASS' if p2 else 'FAIL'}"
    )
    print(
        f"[3] r_max   final = {r_max_final:.4f}  (limit R_in + 2 ps = {R_IN + 2.0 * PS:.3f})"
        f" -> {'PASS' if p3 else 'FAIL'}"
    )
    print(
        f"final: z_min={z_min_final:+.4f} (cavity floor {Z_BOT}, table {Z_TABLE}), "
        f"n_below_floor={n_under_final}, wall={time.time() - wall0:.1f}s"
    )
    if not (p1 and p2 and p3):
        sys.exit(1)


if __name__ == "__main__":
    main()
