"""MF-12 helper: pre-settle the coffee + pitcher water into their emergent-equilibrium
configuration and save the particle positions to an .npz, for `mf12_pbf_st_tilt_pour.py
--settled` to start the narrative from (no t=0 blast in the recorded video).

Why this exists (MF-12 smoke history): the stock PBF lattice starts ~2x over-dense for this
solver's calibration (poly6 self-term C(0)=+0.57, lattice C~+1 at eps=0.1), so every body
flash-expands at t=0. In the open-top upright pitcher the blast fountains the water out of the
mouth, and at dt_sub=1/480 the blast velocity tunnels particles through the clamp bands
(displacement/substep ~= band). A script-side mouth lid alone is NOT enough (smoke2: water
tunneled out the bottom, s=-1.06). Here we run the SAME scene + SAME solver recipe but at
dt=1/240, substeps=4 (dt_sub=1/960 -> per-substep ballistic displacement 4x smaller, no
tunneling) + velocity_damping=0.99 + the mouth lid, let everything relax, and save positions.

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf12_presettle.py --seconds 6 --tag _v1
"""

import argparse
import os
import sys
import time

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np  # noqa: E402

import genesis as gs  # noqa: E402

print("genesis loaded from:", gs.__file__)
assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), "Wrong genesis! Must be the multiflow copy."

CUP_OBJ = os.path.join(WORKSPACE, "assets", "cup_big.obj")
PITCHER_OBJ = os.path.join(WORKSPACE, "assets", "pitcher45.obj")
VIDEOS_DIR = os.path.join(WORKSPACE, "videos")

# geometry identical to mf12_pbf_st_tilt_pour.py
R_IN = 0.20
T_BOTTOM = 0.02
Z_RIM = T_BOTTOM + 1.0
R_COFFEE = 0.18
Z_COFFEE0 = T_BOTTOM + 0.01  # 0.03
Z_COFFEE1 = 0.79
R_PITCHER = 0.16
L_PITCHER = 0.45
T_PITCHER_FLOOR = 0.016
R_WATER = R_PITCHER - 0.008
S0_WATER = 0.03
H_WATER = 0.67  # MF-12 rev3: relaxed ~0.427 = ~95% cavity depth, ~3 layers below lip
PX, PZ = 0.28, 1.08

DT = 1.0 / 240.0
SUBSTEPS = 4  # dt_sub = 1/960


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--dens-iters", type=int, default=20)
    parser.add_argument("--leps", type=float, default=0.1)
    parser.add_argument("--st-comp", type=float, default=2.0)
    parser.add_argument("--vdamp", type=float, default=0.99)
    parser.add_argument("--tag", type=str, default="_v1")
    args = parser.parse_args()

    ps = 0.008
    out_path = os.path.join(VIDEOS_DIR, f"mf12{args.tag}_settled.npz")

    O0 = np.array([PX, 0.0, PZ])
    A0 = np.array([0.0, 0.0, 1.0])

    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
        pbd_options=gs.options.PBDOptions(
            particle_size=ps,
            lower_bound=(-0.30, -0.30, 0.0),
            upper_bound=(1.00, 0.45, 2.10),
            boundary_cylinder=(0.0, 0.0, R_IN, T_BOTTOM, Z_RIM),
            # presettle-only: TALLER analytic pitcher (keep region 0.60 vs physical rim 0.45).
            # The first-frames lattice rearrangement kicks surface particles over the lip
            # (v5/v5b: ~1.7% lost in <0.5s no matter the fill level) — with a taller keep
            # region they stay group-1 and the script mouth lid pushes them back in. The
            # settled state ends well below the real rim, so mf12's L_PITCHER clamp accepts it.
            boundary_pitcher=(*O0, *A0, R_PITCHER, L_PITCHER + 0.15),
            max_density_solver_iterations=args.dens_iters,
            max_viscosity_solver_iterations=1,
            density_lambda_epsilon=args.leps,
            diffusion_coeff=0.01,
            surface_tension_enabled=True,
            st_compliance=args.st_comp,
            st_surface_density_factor=1.0,
            st_distance_enabled=True,
            st_distance_compliance=40.0,
            st_max_surface_neighbors=256,
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
    pitcher = scene.add_entity(
        morph=gs.morphs.Mesh(file=PITCHER_OBJ, pos=tuple(O0 - T_PITCHER_FLOOR * A0), euler=(0.0, 0.0, 0.0),
                             fixed=True, decimate=False),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.85, 0.75, 0.65, 0.5), opacity=0.5),
    )
    coffee = scene.add_entity(
        material=gs.materials.PBD.Liquid(sampler="regular", rho=1000.0, c_init=1.0),
        morph=gs.morphs.Cylinder(radius=R_COFFEE, height=Z_COFFEE1 - Z_COFFEE0,
                                 pos=(0.0, 0.0, 0.5 * (Z_COFFEE0 + Z_COFFEE1))),
    )
    water_sample_pos = (0.0, 0.0, 0.85 + 0.5 * H_WATER)
    water = scene.add_entity(
        material=gs.materials.PBD.Liquid(sampler="regular", rho=1000.0, c_init=0.0, boundary_group=1),
        morph=gs.morphs.Cylinder(radius=R_WATER, height=H_WATER, pos=water_sample_pos),
    )
    scene.build()
    print(f"coffee={coffee.n_particles} water={water.n_particles}")

    solver = scene.sim.pbd_solver
    n_fluid = solver._n_fluid_particles
    n_coffee = coffee.n_particles

    # teleport water into the upright pitcher + restore boundary_group (same as mf12)
    O_w = np.array([PX, 0.0, PZ])
    delta = (O_w + np.array([0.0, 0.0, S0_WATER + 0.5 * H_WATER])) - np.asarray(water_sample_pos)
    pos = water.get_particles_pos().cpu().numpy()
    pos = pos + delta
    # axial pre-compression to the emergent bulk density (same as mf12, K_Z=0.62): slightly
    # UNDER-dense start (initial top 0.445 vs emergent 0.425) — the column contracts downward
    # (self-stabilizing); an over-dense start blasts the surface up and out (v5b evidence).
    # The taller presettle keep region (above) contains the residual rearrangement kicks.
    K_Z = 0.62
    z_base = O_w[2] + S0_WATER
    pos[:, 2] = z_base + (pos[:, 2] - z_base) * K_Z
    water.set_particles_pos(pos)
    _bg = solver.particles_ng.boundary_group.to_numpy()
    _bg[n_coffee:n_fluid, :] = 1
    solver.particles_ng.boundary_group.from_numpy(_bg)

    # pin the pitcher to its nominal pose every frame — identical to mf12's narrative loop.
    # Without this the free (gravity-compensated) pitcher body drifts under the t=0 blast and
    # the fluid settles around the DRIFTED mesh; mf12 then pins the mesh back to nominal and
    # the coupler squeezes the settled fluid every frame (MF-12 iso-A: KE 1.6e4 vs 2.9).
    print(f"pitcher pose @t=0: pos={pitcher.get_pos().cpu().numpy()}, quat={pitcher.get_quat().cpu().numpy()}")
    # axial lid just above the fill line, but NEVER above the rim minus a safety margin:
    # particles pushed past s=L exit the clamp keep region and are permanently transferred to
    # group 0 (MF-12 v4 failure: H=0.52 put the lid at 0.566 > rim 0.45 and the blast owned
    # the mouth). With K_Z pre-compression the blast is gone (v4c: capped=0 throughout), so a
    # thin 0.003 margin suffices — the lid is pure insurance against initial jitter now.
    s_cap = min(S0_WATER + H_WATER + 2.0 * ps, L_PITCHER - 0.003)
    n_frames = int(round(args.seconds / DT))
    wall0 = time.time()
    for i in range(n_frames):
        solver.set_pitcher_pose(O0, A0)
        pitcher.set_pos(O0 - T_PITCHER_FLOOR * A0, relative=True, zero_velocity=True)
        pitcher.set_quat(np.array([1.0, 0.0, 0.0, 0.0]), relative=True, zero_velocity=True)
        scene.step()
        pos_np = solver.particles.pos.to_numpy()
        w = pos_np[n_coffee:n_fluid, 0, :]
        s_w = (w - O0) @ A0
        over = s_w > s_cap
        if over.any():
            w[over] -= ((s_w[over] - s_cap)[:, None] * A0)
            solver.particles.pos.from_numpy(pos_np)
        if (i + 1) % 120 == 0:
            vel_np = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
            ke = float(0.5 * np.nansum((vel_np**2).sum(axis=1)))
            nan = int((~np.isfinite(pos_np)).sum())
            r_w2 = np.linalg.norm((w - O0) - s_w[:, None] * A0, axis=1)
            n_in_p = int(((s_w < L_PITCHER + ps) & (s_w > -ps) & (r_w2 <= R_PITCHER + ps)).sum())
            print(f"t={(i + 1) * DT:5.2f}s / {args.seconds:.1f}s  KE={ke:.3e}  capped={int(over.sum())}"
                  f"  n_in={n_in_p}/{water.n_particles}  s_max={s_w.max():.3f}"
                  f"  nan={nan}  ({(time.time() - wall0) / (i + 1) * 1000:.0f} ms/frame)")

    # ---- containment verification + save ---------------------------------------------------
    print(f"pitcher pose @end: pos={pitcher.get_pos().cpu().numpy()}, quat={pitcher.get_quat().cpu().numpy()}")
    pos_np = solver.particles.pos.to_numpy()
    safe = np.nan_to_num(pos_np[:n_fluid, 0, :], nan=0.0)
    r_c = np.sqrt(safe[:n_coffee, 0] ** 2 + safe[:n_coffee, 1] ** 2)
    w = safe[n_coffee:n_fluid]
    rel = w - O0
    s_w = rel @ A0
    r_w = np.linalg.norm(rel - s_w[:, None] * A0, axis=1)
    n_in = int(((s_w < L_PITCHER + ps) & (s_w > -ps) & (r_w <= R_PITCHER + ps)).sum())
    print(f"final: coffee z in [{safe[:n_coffee, 2].min():.4f}, {safe[:n_coffee, 2].max():.4f}]  "
          f"r_max_c={r_c.max():.4f}")
    print(f"final: water s in [{s_w.min():.3f}, {s_w.max():.3f}]  r_w_max={r_w.max():.4f}  "
          f"n_in_pitcher={n_in}/{water.n_particles}")
    assert r_c.max() <= R_IN + ps + 1e-6, "coffee escaped the cup"
    assert n_in >= 0.99 * water.n_particles, "water escaped the pitcher during presettle"
    # particles resting exactly ON the floor/wall planes are fine — tolerance, not strict <
    assert s_w.min() > -ps and s_w.max() < L_PITCHER + ps and r_w.max() < R_PITCHER + ps

    os.makedirs(VIDEOS_DIR, exist_ok=True)
    np.savez(
        out_path,
        pos=pos_np,
        n_coffee=n_coffee,
        n_fluid=n_fluid,
        ps=ps,
        dens_iters=args.dens_iters,
        leps=args.leps,
        st_comp=args.st_comp,
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
