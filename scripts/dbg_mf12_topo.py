"""dbg for MF-12: measure ST topology sizes (candidate count / final ring / surface fraction)
on the PBF+ST tilt-pour scene, to calibrate st_max_surface_neighbors / st_max_localmesh_neighbors.

Runs the same scene as mf12_pbf_st_tilt_pour.py with oversized capacities (no overflow raise),
prints per-frame diagnostics from solver.dbg_st_read_reset() plus KE/containment.

  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/dbg_mf12_topo.py --seconds 3
"""

import argparse
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np  # noqa: E402

import genesis as gs  # noqa: E402

print("genesis loaded from:", gs.__file__)
assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), "Wrong genesis!"

CUP_OBJ = os.path.join(WORKSPACE, "assets", "cup_big.obj")
PITCHER_OBJ = os.path.join(WORKSPACE, "assets", "pitcher45.obj")

R_IN = 0.20
T_BOTTOM = 0.02
Z_RIM = T_BOTTOM + 1.0
DT = 1.0 / 60.0
SUBSTEPS = 8
R_COFFEE = 0.18
Z_COFFEE0 = T_BOTTOM + 0.01
Z_COFFEE1 = 0.50
R_PITCHER = 0.16
L_PITCHER = 0.45
T_PITCHER_FLOOR = 0.016
R_WATER = R_PITCHER - 0.008
S0_WATER = 0.03
H_WATER = 0.32
PX, PZ = 0.28, 1.08


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--dens-iters", type=int, default=4)
    parser.add_argument("--dens-relax", type=float, default=0.2)
    parser.add_argument("--leps", type=float, default=100.0, help="density_lambda_epsilon")
    parser.add_argument("--substeps", type=int, default=SUBSTEPS)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--st-comp", type=float, default=2.0)
    parser.add_argument("--st-sdf", type=float, default=1.0, help="st_surface_density_factor")
    parser.add_argument("--vdamp", type=float, default=1.0)
    parser.add_argument("--no-st", action="store_true")
    args = parser.parse_args()

    ps = 0.008
    O0 = np.array([PX, 0.0, PZ])
    A0 = np.array([0.0, 0.0, 1.0])

    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt, substeps=args.substeps, gravity=(0.0, 0.0, -9.81)),
        pbd_options=gs.options.PBDOptions(
            particle_size=ps,
            lower_bound=(-0.30, -0.30, 0.0),
            upper_bound=(1.00, 0.45, 2.10),
            boundary_cylinder=(0.0, 0.0, R_IN, T_BOTTOM, Z_RIM),
            boundary_pitcher=(*O0, *A0, R_PITCHER, L_PITCHER),
            max_density_solver_iterations=args.dens_iters,
            max_viscosity_solver_iterations=1,
            density_lambda_epsilon=args.leps,
            diffusion_coeff=0.01,
            surface_tension_enabled=not args.no_st,
            st_compliance=args.st_comp,
            st_surface_density_factor=args.st_sdf,
            st_distance_enabled=True,
            st_distance_compliance=40.0,
            st_max_surface_neighbors=1024,  # oversized: measure, don't raise
            st_max_localmesh_neighbors=512,
            velocity_damping=args.vdamp,
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(file=CUP_OBJ, pos=(0.0, 0.0, 0.0), fixed=True, decimate=False),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.8, 0.8, 0.85, 0.35), opacity=0.35),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(file=PITCHER_OBJ, pos=tuple(O0 - T_PITCHER_FLOOR * A0), euler=(0.0, 0.0, 0.0),
                             fixed=False, decimate=False),
        material=gs.materials.Rigid(gravity_compensation=1.0),
        surface=gs.surfaces.Default(color=(0.85, 0.75, 0.65, 0.5), opacity=0.5),
    )
    coffee = scene.add_entity(
        material=gs.materials.PBD.Liquid(sampler="regular", rho=1000.0, c_init=1.0,
                                         density_relaxation=args.dens_relax),
        morph=gs.morphs.Cylinder(radius=R_COFFEE, height=Z_COFFEE1 - Z_COFFEE0,
                                 pos=(0.0, 0.0, 0.5 * (Z_COFFEE0 + Z_COFFEE1))),
    )
    water_sample_pos = (0.0, 0.0, 0.59 + 0.5 * H_WATER)
    water = scene.add_entity(
        material=gs.materials.PBD.Liquid(sampler="regular", rho=1000.0, c_init=0.0, boundary_group=1,
                                         density_relaxation=args.dens_relax),
        morph=gs.morphs.Cylinder(radius=R_WATER, height=H_WATER, pos=water_sample_pos),
    )
    scene.build()
    print(f"coffee={coffee.n_particles} water={water.n_particles}")

    solver = scene.sim.pbd_solver
    n_fluid = solver._n_fluid_particles
    n_coffee = coffee.n_particles

    O_w = np.array([PX, 0.0, PZ])
    delta = (O_w + np.array([0.0, 0.0, S0_WATER + 0.5 * H_WATER])) - np.asarray(water_sample_pos)
    pos = water.get_particles_pos().cpu().numpy()
    water.set_particles_pos(pos + delta)
    _bg = solver.particles_ng.boundary_group.to_numpy()
    _bg[n_coffee:n_fluid, :] = 1
    solver.particles_ng.boundary_group.from_numpy(_bg)

    if not args.no_st:
        solver.dbg_st_read_reset()  # drop build/warm-up counts
    n_frames = int(round(args.seconds / args.dt))
    for i in range(n_frames + 1):
        t = i * args.dt
        if i > 0:
            scene.step()
        if i % 10 == 0:
            d = solver.dbg_st_read_reset() if not args.no_st else (0, 0, 0, 0)
            pos_all = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
            vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
            nan = int((~np.isfinite(pos_all)).sum() + (~np.isfinite(vel)).sum())
            ke = float(0.5 * np.nansum((vel**2).sum(axis=1)))
            safe = np.nan_to_num(pos_all, nan=0.0)
            r_c = np.sqrt(safe[:n_coffee, 0] ** 2 + safe[:n_coffee, 1] ** 2)
            # sums accumulate over substeps since last reset -> average
            n_sub = 10 * SUBSTEPS if i > 0 else 1
            print(
                f"t={t:5.2f}  max_cand={d[0]:4d}  max_ring={d[1]:4d}  "
                f"surf={d[2] / n_sub:7.0f} ({d[2] / n_sub / n_fluid:5.1%})  valid={d[3] / n_sub:7.0f}  "
                f"r_max_c={r_c.max():.4f}  KE={ke:.3e}  nan={nan}"
            )

    # ---- ground truth: brute-force neighbor counts at the final state ----------------------
    pos_all = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
    safe = np.nan_to_num(pos_all, nan=0.0)
    r_c = np.sqrt(safe[:n_coffee, 0] ** 2 + safe[:n_coffee, 1] ** 2)
    print(f"final: coffee z in [{safe[:n_coffee, 2].min():.4f}, {safe[:n_coffee, 2].max():.4f}]  "
          f"r_max_c={r_c.max():.4f}")
    rng = np.random.default_rng(0)
    sample = rng.choice(n_fluid, size=300, replace=False)
    ring_r = 3.0 * 0.008
    sup_r = 1.25 * 0.008
    cnt_ring = np.zeros(300, dtype=int)
    cnt_sup = np.zeros(300, dtype=int)
    for c0 in range(0, n_fluid, 20000):
        chunk = safe[c0 : c0 + 20000]
        d2 = ((safe[sample][:, None, :] - chunk[None, :, :]) ** 2).sum(axis=-1)
        cnt_ring += (d2 < ring_r**2).sum(axis=1)
        cnt_sup += (d2 < sup_r**2).sum(axis=1)
    cnt_ring -= 1  # exclude self
    cnt_sup -= 1
    print(f"ground truth neighbors within 3ps={ring_r:.3f}: max={cnt_ring.max()} mean={cnt_ring.mean():.1f}")
    print(f"ground truth neighbors within 1.25ps={sup_r:.4f}: max={cnt_sup.max()} mean={cnt_sup.mean():.1f}")


if __name__ == "__main__":
    main()
