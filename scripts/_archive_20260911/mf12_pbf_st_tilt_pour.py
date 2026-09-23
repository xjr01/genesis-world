"""MF-12: tilt-animation pour demo on plain PBF + PBSTF-style surface tension (NOT IPBF).

Same narrative as MF-11 (user-specified, unchanged):
  phase A (0-6s)    : everything at rest — coffee settles in the big cup, milk settles in the
                      UPRIGHT pitcher (also validates the vertical pitcher clamp does not leak)
  phase B (6-14s)   : pitcher smoothly tilts theta: 0 -> --tilt-max deg about the pour-side lip
                      pivot (smoothstep); milk spills over the lip into the big cup
  phase C (14-16s)  : hold at max tilt; the stream continues and thins to a trickle
  phase D (16-21s)  : theta back to 0 AND the pitcher lifts up / moves aside (smoothstep);
                      the stream breaks (rev3: early tilt-back, residual milk stays in pitcher)
  phase E (21-45s)  : pitcher parked aside; the big cup settles while milk diffuses into coffee

Solver: PBDSolver (PBF density projection) + surface tension projected inside the density
iteration loop (multiflow MF-12 port of the PBSTF reference: spherical-illumination surface
detection -> color-field normals -> local-mesh area constraint + one-sided surface distance
constraint). Stiffness recipe (MF-12 tuning, see RECORD PMF-5): the stock lambda_epsilon=100
makes the PBF projection too soft to hold a tall column (emergent bulk ~1.7x rho_rest), so
--leps 0.1; PBSTF's 0.7 surface density target is calibrated for a bulk at rho_rest and EJECTS
surface particles in this fluid, so --st-sdf 1.0. Boundary ownership is the MF-11 scheme:
big cup clamp (`boundary_cylinder`) for boundary_group=0 only, dynamic pitcher clamp
(`boundary_pitcher` -> TiltedCylinderBoundary, `solver.set_pitcher_pose`) for boundary_group=1
only, with permanent group transfer once a particle leaves the pitcher keep region.

Damping: simplest possible — global per-substep velocity decay (--vdamp, 1.0 = off).

Per-frame CSV: multiflow/videos/mf12{tag}_metrics.csv
  (frame, t, phase, theta_deg, sum_c, std_c, zc_c, r_max_coffee, z_min, n_in_pitcher, ke, nan)
Video: multiflow/videos/mf12_pbf_st_tilt_pour{tag}.mp4

Run (smoke first!):
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf12_pbf_st_tilt_pour.py --seconds 15 --tag _smoke --no-video
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf12_pbf_st_tilt_pour.py --seconds 45
"""

import argparse
import csv
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

# big cup geometry (assets/gen_cup.py --r-in 0.2 --h-in 1.0): cavity floor z=0.02, rim z=1.02
R_IN = 0.20
T_BOTTOM = 0.02
Z_RIM = T_BOTTOM + 1.0
DT = 1.0 / 60.0
SUBSTEPS = 8
_DT_OVERRIDE = None  # set from --dt/--substeps in main() before scene creation

# coffee column inside the big cup, same as MF-9/10/11 cup but taller: the stock PBF's emergent
# bulk density is ~1.7x the nominal rho_rest (MF-12 finding: lambda_epsilon-soft projection), so
# the column settles to ~55-60% of its sampled height. Sampled to 0.79 -> settles ~0.45 (visual
# parity with the MF-11 half-cup look).
R_COFFEE = 0.18
Z_COFFEE0 = T_BOTTOM + 0.01  # 0.03
Z_COFFEE1 = 0.79

# pitcher geometry (assets/gen_cup.py --r-in 0.16 --h-in 0.45 --t-wall 0.016 --t-bottom 0.016):
# cavity floor at local z=0.016, rim at local z=0.466
R_PITCHER = 0.16
L_PITCHER = 0.45  # cavity floor -> rim, along the axis
T_PITCHER_FLOOR = 0.016
# water pre-fill inside the upright pitcher (surface below the lip)
R_WATER = R_PITCHER - 0.008  # one particle radius margin to the clamp wall
S0_WATER = 0.03  # bottom margin above the pitcher floor plane
H_WATER = 0.67  # axial fill height (sampled). With K_Z=0.62 pre-compression + 0.955 residual
                # settle (MF-12 v4c calibration), relaxes to ~0.427 = ~95% of the 0.45 cavity
                # depth, ~3 particle layers below the lip (user rev3: nearly full pitcher).

# timeline (seconds) — MF-12 rev3: pitcher nearly full (surface ~3 layers below lip), so the
# stream starts at theta~15deg (v4c: 52.7deg with freeboard 0.114 -> spill geometry
# tan(theta)~freeboard/0.087). Ramp stretched to 8s so the pour begins gently at t~8s; tilt
# back after ~8s of pouring (user: early tilt-back, residual milk stays in the pitcher).
T_A = 6.0  # rest
T_B0, T_B1 = 6.0, 14.0  # tilt up
T_C = 16.0  # hold until
T_D0, T_D1 = 16.0, 21.0  # tilt back + move aside
TILT_MAX_DEG = 80.0
D_ASIDE = np.array([0.30, 0.0, 0.25])  # phase-D translation of the whole pitcher

# upright pitcher placement: axis x = PX, cavity-floor bottom center z = PZ; pour-side lip pivot
# P = (PX - R_PITCHER, 0, PZ + L_PITCHER) stays fixed in world during the tilt (phase B/C)
PX, PZ = 0.28, 1.08
PIVOT = np.array([PX - R_PITCHER, 0.0, PZ + L_PITCHER])


def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def pitcher_pose(t):
    """Pitcher clamp pose at time t: (origin O, unit axis a, theta_rad). Rotates about the
    pour-side lip pivot P toward -x (the big cup), then lifts/moves aside in phase D."""
    if t < T_B0:
        th, u_tilt = 0.0, 0.0
    elif t < T_B1:
        u_tilt = smoothstep((t - T_B0) / (T_B1 - T_B0))
        th = np.deg2rad(TILT_MAX_DEG) * u_tilt
    elif t < T_C:
        u_tilt = 1.0
        th = np.deg2rad(TILT_MAX_DEG)
    elif t < T_D1:
        u_tilt = 1.0 - smoothstep((t - T_D0) / (T_D1 - T_D0))
        th = np.deg2rad(TILT_MAX_DEG) * u_tilt
    else:
        u_tilt = 0.0
        th = 0.0
    u_aside = smoothstep((t - T_D0) / (T_D1 - T_D0)) if t >= T_D0 else 0.0
    a = np.array([-np.sin(th), 0.0, np.cos(th)])  # bottom -> mouth
    e1 = np.array([np.cos(th), 0.0, np.sin(th)])  # in-plane, toward the pour side
    O = PIVOT - L_PITCHER * a + R_PITCHER * e1 + u_aside * D_ASIDE
    return O, a, th


def quat_roty_neg(theta):
    """Quaternion (w, x, y, z) for rotY(-theta): maps mesh local +z to the pitcher axis."""
    return np.array([np.cos(theta / 2.0), 0.0, -np.sin(theta / 2.0), 0.0])


def main():
    global TILT_MAX_DEG, DT, SUBSTEPS, T_C, T_D0, T_D1
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--settle", type=float, default=4.0, help="pre-roll settle seconds (not recorded)")
    parser.add_argument("--epsd", type=float, default=0.01, help="diffusion_coeff (0 = off)")
    parser.add_argument("--dens-iters", type=int, default=20, help="max_density_solver_iterations")
    parser.add_argument("--dens-relax", type=float, default=0.2, help="material density_relaxation")
    parser.add_argument("--leps", type=float, default=0.1, help="density_lambda_epsilon (stock 100)")
    parser.add_argument("--visc-relax", type=float, default=0.01, help="material viscosity_relaxation")
    parser.add_argument("--st-comp", type=float, default=2.0, help="st_compliance (smaller = stronger ST)")
    parser.add_argument("--st-sdf", type=float, default=1.0, help="st_surface_density_factor (PBSTF 0.7)")
    parser.add_argument("--st-dist-comp", type=float, default=40.0, help="st_distance_compliance")
    parser.add_argument("--vdamp", type=float, default=1.0, help="velocity_damping per substep (1.0 = off)")
    parser.add_argument("--adh", action="store_true", help="wall adhesion on both cups (PBSTF port)")
    parser.add_argument("--adh-comp", type=float, default=20.0, help="wall_adhesion_compliance (smaller = stickier)")
    parser.add_argument("--wfric", type=float, default=0.0, help="wall_friction: velocity-stage tangential damping for adhering particles (0 = off)")
    parser.add_argument("--tilt-max", type=float, default=TILT_MAX_DEG, help="max tilt angle (deg)")
    parser.add_argument("--hold-until", type=float, default=T_C,
                        help="hold at tilt-max until this time (s); tilt-back runs 5s after it")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-st-dist", action="store_true")
    parser.add_argument("--settled", type=str, default=None,
                        help="npz from mf12_presettle.py: skip built-in settle, load settled state")
    parser.add_argument("--st-max-neigh", type=int, default=256, help="st_max_surface_neighbors")
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--dt", type=float, default=None, help="frame dt override (default 1/60)")
    parser.add_argument("--substeps", type=int, default=None, help="substeps override (default 8)")
    args = parser.parse_args()

    if args.dt is not None:
        DT = args.dt
        if args.substeps is not None:
            SUBSTEPS = args.substeps

    ps = 0.008
    mp4_path = os.path.join(VIDEOS_DIR, f"mf12_pbf_st_tilt_pour{args.tag}.mp4")
    csv_path = os.path.join(VIDEOS_DIR, f"mf12{args.tag}_metrics.csv")

    TILT_MAX_DEG = args.tilt_max
    T_C = T_D0 = args.hold_until
    T_D1 = args.hold_until + 5.0

    O0, A0, _ = pitcher_pose(0.0)

    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
        pbd_options=gs.options.PBDOptions(
            particle_size=ps,
            lower_bound=(-0.30, -0.30, 0.0),
            upper_bound=(1.00, 0.45, 2.10),
            boundary_cylinder=(0.0, 0.0, R_IN, T_BOTTOM, Z_RIM),  # big cup: wall below rim + escape band
            boundary_pitcher=(*O0, *A0, R_PITCHER, L_PITCHER),  # pitcher: dynamic open-mouth clamp
            max_density_solver_iterations=args.dens_iters,
            max_viscosity_solver_iterations=1,
            density_lambda_epsilon=args.leps,
            diffusion_coeff=args.epsd,
            surface_tension_enabled=True,
            st_compliance=args.st_comp,
            st_surface_density_factor=args.st_sdf,
            st_distance_enabled=not args.no_st_dist,
            st_distance_compliance=args.st_dist_comp,
            st_max_surface_neighbors=args.st_max_neigh,
            velocity_damping=args.vdamp,
            wall_adhesion_enabled=args.adh,
            wall_adhesion_compliance=args.adh_comp,
            wall_friction=args.wfric,
        ),
        vis_options=gs.options.VisOptions(
            render_particle_as="points",  # per-particle concentration coloring
        ),
        show_viewer=False,
    )

    # big cup: visualization only (CylinderBoundary does the physics)
    cup = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=CUP_OBJ,
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            decimate=False,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.8, 0.8, 0.85, 0.35), opacity=0.35),
    )

    # pitcher: visualization only (dynamic TiltedCylinderBoundary does the physics). FIXED
    # (kinematic) body, teleported every frame to the clamp pose — a free body drifts under
    # the PBF t=0 lattice blast and each re-pin teleports the mesh through the settled fluid,
    # injecting energy (MF-12 presettle v2 failure). Probe: set_pos/set_quat work on fixed
    # entities (multiflow/scripts/dbg_fixed_setpos.py). Spawn it directly at the initial
    # (upright) pose so it never overlaps the big cup.
    pitcher = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=PITCHER_OBJ,
            pos=tuple(O0 - T_PITCHER_FLOOR * A0),
            euler=(0.0, 0.0, 0.0),
            fixed=True,
            decimate=False,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.85, 0.75, 0.65, 0.5), opacity=0.5),
    )

    coffee = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=1.0,
            density_relaxation=args.dens_relax, viscosity_relaxation=args.visc_relax,
        ),
        morph=gs.morphs.Cylinder(
            radius=R_COFFEE,
            height=Z_COFFEE1 - Z_COFFEE0,
            pos=(0.0, 0.0, 0.5 * (Z_COFFEE0 + Z_COFFEE1)),
        ),
    )
    # water sampled upright INSIDE the big cup above the (taller) coffee column, teleported into
    # the upright pitcher right after build (see MF-11 note on the build warm-up group flip).
    water_sample_pos = (0.0, 0.0, 0.85 + 0.5 * H_WATER)
    water = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=0.0, boundary_group=1,
            density_relaxation=args.dens_relax, viscosity_relaxation=args.visc_relax,
        ),
        morph=gs.morphs.Cylinder(
            radius=R_WATER,
            height=H_WATER,
            pos=water_sample_pos,
        ),
    )

    # wide oblique view: big cup (left) + upright pitcher hovering off to the right + pour arc
    cam = scene.add_camera(res=(960, 1280), pos=(2.35, -2.35, 2.05), lookat=(0.2, 0.0, 1.0), fov=40, GUI=False)
    scene.build()
    print(f"cup geoms: {cup.n_geoms}, pitcher geoms: {pitcher.n_geoms}, "
          f"coffee particles: {coffee.n_particles}, water particles: {water.n_particles}")
    print(
        f"ps={ps} epsd={args.epsd} dens_iters={args.dens_iters} dens_relax={args.dens_relax:g} "
        f"leps={args.leps:g} visc_relax={args.visc_relax:g} st_comp={args.st_comp:g} "
        f"st_sdf={args.st_sdf:g} st_dist_comp={args.st_dist_comp:g} "
        f"vdamp={args.vdamp:g} tilt_max={args.tilt_max:g} adh={args.adh}/{args.adh_comp:g} wfric={args.wfric:g} "
        f"seconds={args.seconds} tag='{args.tag}'"
    )

    if not args.no_video:
        scene.visualizer.rasterizer._camera_targets[cam.uid].point_size = args.point_size

    solver = scene.sim.pbd_solver
    assert type(solver.boundary).__name__ == "CylinderBoundary", f"unexpected boundary: {type(solver.boundary)}"
    assert type(solver.boundary2).__name__ == "TiltedCylinderBoundary", f"unexpected boundary2: {type(solver.boundary2)}"
    n_fluid = solver._n_fluid_particles
    n_coffee = coffee.n_particles
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    # ---- move the water into the upright pitcher (translation + axial pre-compression) -----
    O_w = np.array([PX, 0.0, PZ])  # upright pitcher bottom-center (theta=0, aside=0)
    water_target_center = O_w + np.array([0.0, 0.0, S0_WATER + 0.5 * H_WATER])
    delta = water_target_center - np.asarray(water_sample_pos)
    pos = water.get_particles_pos().cpu().numpy()  # (n, 3); env dim already stripped (n_envs==0)
    pos = pos + delta
    # Axial pre-compression to the emergent bulk density: the PBF lattice samples ~1.6x
    # under-dense (poly6 self-term), and with the radius fixed by the pitcher wall the
    # compression is carried axially. K_Z=0.62 puts the raw top at 0.445 (slightly under-dense,
    # self-stabilizing downward contraction; an over-dense start blasts the surface out — v5b
    # evidence). Must stay below the rim: s>L means instant permanent group transfer.
    K_Z = 0.62
    z_base = O_w[2] + S0_WATER
    pos[:, 2] = z_base + (pos[:, 2] - z_base) * K_Z
    water.set_particles_pos(pos)
    # scene.build() runs one warm-up step while the water is still at its big-cup sample
    # position — outside the pitcher keep region — so _kernel_solve_boundary_collision
    # permanently transferred the whole water entity to group 0. Restore group 1 now that the
    # water is actually inside the pitcher (verified below).
    _bg = solver.particles_ng.boundary_group.to_numpy()
    _bg[n_coffee:n_fluid, :] = 1
    solver.particles_ng.boundary_group.from_numpy(_bg)
    got = water.get_particles_pos().cpu().numpy()
    rel = got - O_w
    s_chk = rel[:, 2]
    r_chk = np.linalg.norm(rel[:, :2], axis=1)
    print(f"water in pitcher: s in [{s_chk.min():.3f}, {s_chk.max():.3f}] (L={L_PITCHER}), "
          f"r in [{r_chk.min():.4f}, {r_chk.max():.4f}] (R={R_PITCHER})")
    assert r_chk.max() < R_PITCHER and s_chk.min() > 0.0 and s_chk.max() < L_PITCHER
    # boundary_group sanity: coffee all 0, water all 1
    gid = solver.particles_ng.boundary_group.to_numpy()
    print(f"boundary_group field shape: {gid.shape}")
    gid = gid[:n_fluid, 0]
    uniq_c, cnt_c = np.unique(gid[:n_coffee], return_counts=True)
    uniq_w, cnt_w = np.unique(gid[n_coffee:n_fluid], return_counts=True)
    print(f"coffee boundary_group: {dict(zip(uniq_c.tolist(), cnt_c.tolist()))}")
    print(f"water  boundary_group: {dict(zip(uniq_w.tolist(), cnt_w.tolist()))}")
    assert (gid[:n_coffee] == 0).all() and (gid[n_coffee:n_fluid] == 1).all(), "boundary_group mismatch"

    # ---- settled-state load / settle pre-roll (NOT recorded) ------------------------------
    # The PBF lattice starts ~2x over-dense for this solver's calibration (poly6 self-term
    # C(0) = +0.57, lattice C ~ +1), so every body flash-expands at t=0. In the open-top
    # upright pitcher that blast fountains the water straight out of the mouth (MF-12 smoke:
    # 45k -> 0 within 9s). Preferred path: --settled npz from mf12_presettle.py (relaxed at
    # dt=1/240 so the blast cannot tunnel through the pitcher clamp band). Fallback: the
    # in-script settle pre-roll below (mouth lid only; known to tunnel at dt_sub=1/480).
    if args.settled:
        dat = np.load(args.settled)
        pos0 = dat["pos"].astype(np.float32)
        n_total = int(solver.particles.pos.to_numpy().shape[0])
        assert pos0.shape == (n_total, 1, 3), f"settled pos shape {pos0.shape} != scene ({n_total}, 1, 3)"
        assert int(dat["n_coffee"]) == n_coffee and int(dat["n_fluid"]) == n_fluid
        solver.particles.pos.from_numpy(pos0)
        solver.particles.vel.from_numpy(np.zeros_like(pos0))
        print(f"loaded settled state from {args.settled} (keys: {sorted(dat.files)})")
        got = water.get_particles_pos().cpu().numpy()
        rel = got - O_w
        s_chk = rel[:, 2]
        r_chk = np.linalg.norm(rel[:, :2], axis=1)
        n_out = int(((s_chk < -ps) | (s_chk > L_PITCHER + ps) | (r_chk > R_PITCHER + ps)).sum())
        print(f"settled water: s in [{s_chk.min():.3f}, {s_chk.max():.3f}] (L={L_PITCHER}), "
              f"r_max={r_chk.max():.4f} (R={R_PITCHER}), outside={n_out}")
        assert n_out == 0, "settled state has water outside the pitcher"
    else:
        n_settle = int(round(args.settle / DT))
        if n_settle > 0:
            s_cap = S0_WATER + H_WATER + 2.0 * ps  # axial lid just above the fill line
            wall_s = time.time()
            for i in range(n_settle):
                solver.set_pitcher_pose(O0, A0)
                pitcher.set_pos(O0 - T_PITCHER_FLOOR * A0, relative=True, zero_velocity=True)
                pitcher.set_quat(quat_roty_neg(0.0), relative=True, zero_velocity=True)
                scene.step()
                pos_np = solver.particles.pos.to_numpy()
                w = pos_np[n_coffee:n_fluid, 0, :]
                s_w = (w - O0) @ A0
                over = s_w > s_cap
                if over.any():
                    w[over] -= ((s_w[over] - s_cap)[:, None] * A0)
                    solver.particles.pos.from_numpy(pos_np)
                if (i + 1) % 60 == 0:
                    vel_np = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
                    ke_s = float(0.5 * np.nansum((vel_np**2).sum(axis=1)))
                    print(f"settle t={(i + 1) * DT:5.2f}s / {args.settle:.1f}s  KE={ke_s:.3e}  "
                          f"capped={int(over.sum())}  ({(time.time() - wall_s) / (i + 1) * 1000:.0f} ms/frame)")
            # verify the water is back inside the pitcher keep region before the narrative starts
            got = water.get_particles_pos().cpu().numpy()
            rel = got - O0
            s_chk = rel[:, 2]
            r_chk = np.linalg.norm(rel[:, :2], axis=1)
            print(f"post-settle water: s in [{s_chk.min():.3f}, {s_chk.max():.3f}] (L={L_PITCHER}), "
                  f"r_max={r_chk.max():.4f} (R={R_PITCHER})")
            assert s_chk.min() > 0.0 and s_chk.max() < L_PITCHER and r_chk.max() < R_PITCHER, (
                "settle pre-roll failed: water escaped the upright pitcher"
            )

    n_frames = int(round(args.seconds / DT))
    rows = []
    wall0 = time.time()
    if not args.no_video:
        cam.start_recording(save_to_filename=mp4_path, fps=args.fps)

    def phase_of(t):
        if t < T_B0:
            return 0
        if t < T_B1:
            return 1
        if t < T_C:
            return 2
        if t < T_D1:
            return 3
        return 4

    for i in range(n_frames + 1):
        # ---- animation: physics clamp pose + visual mesh pose -----------------------------
        t = i * DT
        O, A, th = pitcher_pose(t)
        solver.set_pitcher_pose(O, A)
        mesh_pos = O - T_PITCHER_FLOOR * A
        # relative=True: pos/quat are given in the user (morph) frame — same convention as the
        # morph's own pos/euler. relative=False would target the solver's link frame (center of
        # mass + principal inertia axes), which is offset from the mesh frame.
        pitcher.set_pos(mesh_pos, relative=True, zero_velocity=True)
        pitcher.set_quat(quat_roty_neg(th), relative=True, zero_velocity=True)

        if i > 0:
            scene.step()

        # ---- metrics ----------------------------------------------------------------------
        phase = phase_of(t)
        pos_all = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
        vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
        c = solver.particles.c.to_numpy()[:n_fluid, 0]
        nan_count = int((~np.isfinite(pos_all)).sum() + (~np.isfinite(vel)).sum() + (~np.isfinite(c)).sum())
        safe = np.nan_to_num(pos_all, nan=0.0, posinf=0.0, neginf=0.0)
        safe_c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        sum_c = float(safe_c.sum())
        std_c = float(safe_c.std())
        zc_c = float((safe_c * safe[:, 2]).sum() / sum_c) if sum_c > 0 else float("nan")
        # big-cup containment is judged on the coffee entity only (the pitcher water lives away)
        r_c = np.sqrt(safe[:n_coffee, 0] ** 2 + safe[:n_coffee, 1] ** 2)
        r_max = float(r_c.max())
        if r_max > 0.208:
            viol = np.nonzero(r_c > 0.208)[0]
            pv = safe[viol[:5]]
            print(f"DBG r_max t={t:.2f}: n_viol={len(viol)}, pos={np.round(pv, 4).tolist()}")
        z_min = float(safe[:, 2].min())
        # water still captive in the pitcher (current animated frame O, A)
        rel = safe[n_coffee:n_fluid] - O
        s_w = rel @ A
        r_w = np.linalg.norm(rel - s_w[:, None] * A, axis=1)
        n_in_pitcher = int(((s_w < L_PITCHER + ps) & (s_w > -ps) & (r_w <= R_PITCHER + ps)).sum())
        ke = float(0.5 * np.nansum((vel**2).sum(axis=1)))
        # big-cup mixture surface level (rev3 report): p99 z of fluid inside the cup interior
        # (r < R_IN, below rim — excludes the parked/falling pitcher-side milk)
        r_all = np.sqrt(safe[:, 0] ** 2 + safe[:, 1] ** 2)
        in_cup = (r_all < R_IN) & (safe[:, 2] < Z_RIM)
        z_surf = float(np.percentile(safe[in_cup, 2], 99)) if in_cup.any() else float("nan")
        n_adh = solver.wall_adhesion_stats()
        rows.append((i, t, phase, float(np.rad2deg(th)), sum_c, std_c, zc_c, r_max, z_min, n_in_pitcher, ke, nan_count, z_surf, n_adh))
        if i % 60 == 0:
            wall = time.time() - wall0
            print(
                f"t={t:6.2f}s  ph={phase}  th={np.rad2deg(th):5.1f}  sum_c={sum_c:9.2f}  std_c={std_c:.5f}"
                f"  zc_c={zc_c:.4f}  r_max_c={r_max:.4f}  z_min={z_min:.4f}  n_pitcher={n_in_pitcher}"
                f"  KE={ke:.4e}  nan={nan_count}  ({wall / max(i, 1) * 1000:.0f} ms/frame avg)"
            )

    if not args.no_video:
        cam.stop_recording()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["frame", "t", "phase", "theta_deg", "sum_c", "std_c", "zc_c", "r_max_coffee",
             "z_min", "n_in_pitcher", "ke", "nan_count", "z_surf", "n_adh"]
        )
        w.writerows(rows)

    # ---- pass/fail summary ------------------------------------------------------------------
    cols = ["frame", "t", "phase", "theta", "sum_c", "std_c", "zc_c", "r_max_coffee", "z_min",
            "n_in_pitcher", "ke", "nan", "z_surf", "n_adh"]
    arr = {k: np.array([r[j] for r in rows], dtype=float) for j, k in enumerate(cols)}
    sum_c0 = arr["sum_c"][0]
    drift = float(np.max(np.abs(arr["sum_c"] - sum_c0)) / sum_c0)
    p1 = bool(arr["r_max_coffee"].max() <= R_IN + ps + 1e-6)
    p2 = bool(arr["z_min"].min() >= T_BOTTOM - ps - 1e-6)
    p3 = bool(arr["nan"].max() == 0)
    p4 = bool(drift < 0.01)
    n_w = water.n_particles
    # phase-A leak check: water count in the upright pitcher must hold steady while at rest
    mask_a = (arr["t"] >= 1.0) & (arr["t"] <= min(T_A - 0.5, args.seconds))
    p5 = None
    if mask_a.any():
        n0 = arr["n_in_pitcher"][0]
        leak = float(np.max(np.abs(arr["n_in_pitcher"][mask_a] - n0)) / max(n0, 1))
        p5 = bool(leak < 0.01)
    # pour check: enough milk must have left the pitcher (absolute, ~previous demo's pour volume)
    p6 = None
    if args.seconds >= T_D1:
        poured = arr["n_in_pitcher"][0] - arr["n_in_pitcher"][-1]
        p6 = bool(poured >= 30000)
    # post-tilt-back check: the milk left in the pitcher stays put once parked
    p7 = None
    if args.seconds >= T_D1 + 2.0:
        mask_e = arr["t"] >= T_D1 + 0.5
        if mask_e.any():
            n_end = arr["n_in_pitcher"][-1]
            p7 = bool(np.max(np.abs(arr["n_in_pitcher"][mask_e] - n_end)) <= 0.01 * arr["n_in_pitcher"][0])
    print("=" * 70)
    print(f"[1] r_max_coffee max = {arr['r_max_coffee'].max():.4f}  (limit {R_IN + ps:.2f})  -> {'PASS' if p1 else 'FAIL'}")
    print(f"[1] z_min  min = {arr['z_min'].min():.4f}  (limit {T_BOTTOM - ps:.2f})   -> {'PASS' if p2 else 'FAIL'}")
    print(f"[1] nan    max = {int(arr['nan'].max())}                              -> {'PASS' if p3 else 'FAIL'}")
    print(f"[2] sum_c drift = {drift:.3e}  (limit 1e-2, sum_c0={sum_c0:.2f})   -> {'PASS' if p4 else 'FAIL'}")
    if p5 is not None:
        print(f"[3] phase-A leak = {leak:.4%} of {int(n0)}  (limit 1%)          -> {'PASS' if p5 else 'FAIL'}")
    if p6 is not None:
        print(f"[4] pitcher pour: {int(arr['n_in_pitcher'][0])} -> {int(arr['n_in_pitcher'][-1])} / {n_w} "
              f"(poured {int(poured)}, need >=30000) -> {'PASS' if p6 else 'FAIL'}")
    if p7 is not None:
        print(f"[5] post-park n_in_pitcher stable (limit 1%)                    -> {'PASS' if p7 else 'FAIL'}")
    print(f"[6] big-cup surface z_surf: max {np.nanmax(arr['z_surf']):.3f} / final {arr['z_surf'][-1]:.3f}"
          f"  (rim {Z_RIM:.2f}) -> {'PASS' if np.nanmax(arr['z_surf']) < Z_RIM - ps else 'FAIL'}")
    print(f"KE final = {arr['ke'][-1]:.4e}")
    print(f"csv: {csv_path}")
    if not args.no_video:
        print(f"mp4: {mp4_path}")


if __name__ == "__main__":
    main()
