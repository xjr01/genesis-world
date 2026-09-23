"""MF-14: teapot-effect narrative demo = MF-12 (rev9 recipe) + pitcher SHELL boundary + Phase E
trickle segment (multiflow P2, design doc multiflow/P2_SHELL_DESIGN.md section 6.2).

Same base narrative as MF-12:
  phase A (0-6s)    : rest — coffee settles in the big cup, milk in the UPRIGHT pitcher
  phase B (6-14s)   : pitcher tilts 0 -> --tilt-max deg about the pour-side lip pivot; milk
                      spills over the lip into the big cup
  phase C (14-16s)  : hold at max tilt; the stream thins to a trickle
  phase D (16-21s)  : tilt back to 0 AND the pitcher lifts / moves aside; the stream breaks
  phase E (21-24s)  : parked aside; the big cup settles
  phase E2 (24-29s) : TRICKLE segment — the parked pitcher makes a second small tilt
                      (--tilt2 deg, ramp 1s / hold 3s / back 1s) so the residual milk creeps
                      over the lip, runs DOWN THE OUTER WALL (teapot effect), detaches at the
                      bottom edge and drips onto the TABLE (z=0 plane)
  phase F (29-50s)  : everything settles; milk puddle on the table + milk in the big cup

Difference vs MF-12: the pitcher clamp is the P2 shell model (TiltedCylinderShellBoundary,
finite-thickness wall with an OUTER surface — exact (r,s) cross-section SDF). impose acts
only inside the shell solid; the exterior is free space; wall adhesion (PBSTF port, P1) holds
the outer-wall film. New per-frame metrics: n_outer_wall / n_table / n_milk_cup / n_leak
(mid-wall side flips).

Per-frame CSV: multiflow/videos/mf14{tag}_metrics.csv
Video: multiflow/videos/mf14_shell_teapot{tag}.mp4

Run (smoke first!):
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf14_shell_teapot.py --seconds 30 --tag _smoke --no-video \
    --settled multiflow/videos/mf12_v5_settled.npz
  "$PY" multiflow/scripts/mf14_shell_teapot.py --seconds 50 --settled multiflow/videos/mf12_v5_settled.npz
"""

import argparse
import csv
import os
import sys
import time

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import genesis as gs  # noqa: E402
from shell_sdf import shell_sdf_np, side_masks  # noqa: E402

print("genesis loaded from:", gs.__file__)
assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), "Wrong genesis! Must be the multiflow copy."

CUP_OBJ = os.path.join(WORKSPACE, "assets", "cup_big.obj")
PITCHER_OBJ = os.path.join(WORKSPACE, "assets", "pitcher45.obj")
VIDEOS_DIR = os.path.join(WORKSPACE, "videos")

# big cup geometry (assets/gen_cup.py --r-in 0.2 --h-in 1.0): cavity floor z=0.02, rim z=1.02
R_IN = 0.20
T_BOTTOM = 0.02
Z_RIM = T_BOTTOM + 1.0
Z_TABLE = 0.0  # table plane (milk drips land here; the big cup's outer bottom sits at z=0)
DT = 1.0 / 60.0
SUBSTEPS = 8

# coffee column (MF-12 calibration, unchanged)
R_COFFEE = 0.18
Z_COFFEE0 = T_BOTTOM + 0.01  # 0.03
Z_COFFEE1 = 0.79

# pitcher shell geometry (assets/gen_cup.py --r-in 0.16 --h-in 0.45 --t-wall 0.016)
R_PITCHER = 0.16
L_PITCHER = 0.45
T_PITCHER_WALL = 0.016
T_PITCHER_FLOOR = 0.016
R_PITCHER_OUT = R_PITCHER + T_PITCHER_WALL
R_WATER = R_PITCHER - 0.008
S0_WATER = 0.03
H_WATER = 0.67  # axial fill height (MF-12 v4c calibration; relaxes to ~95% of the cavity)

# timeline (seconds) — MF-12 rev3 timing (rev9 recipe: tilt-max 55, hold until 16)
T_A = 6.0
T_B0, T_B1 = 6.0, 14.0
T_C = 16.0
T_D0, T_D1 = 16.0, 21.0
TILT_MAX_DEG = 80.0
D_ASIDE = np.array([0.50, 0.0, 0.25])  # park far enough that the E2 trickle arc (vx~-0.6 m/s
# at the lip, carrying ~0.2-0.3 m -x during the 0.39 s fall) still lands OUTSIDE the big
# cup's footprint (r=0.23): lip at x=0.62 -> landing x~0.32-0.42 = table (smoke52/55sr lesson:
# parked at 0.42 the whole spill arced back into the cup)
# Phase E2 trickle segment (P2 design 4.3): second small tilt of the PARKED pitcher
T_E0 = 24.0  # second tilt starts
T_E1 = 28.0  # hold end (tilt back runs 1s)
T_E2 = 29.0  # fully back
TILT2_DEG = 25.0  # design range 15-25 deg

# upright pitcher placement: axis x = PX, cavity-floor bottom center z = PZ
PX, PZ = 0.28, 1.08
PIVOT = np.array([PX - R_PITCHER, 0.0, PZ + L_PITCHER])


def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def pitcher_pose(t):
    """Pitcher clamp pose at time t: (origin O, unit axis a, total tilt rad). MF-12 tilt about
    the pour-side lip pivot, lift/aside in phase D, then the P2 second small tilt about the
    PARKED lip pivot (phase E2)."""
    if t < T_B0:
        th1 = 0.0
    elif t < T_B1:
        th1 = np.deg2rad(TILT_MAX_DEG) * smoothstep((t - T_B0) / (T_B1 - T_B0))
    elif t < T_C:
        th1 = np.deg2rad(TILT_MAX_DEG)
    elif t < T_D1:
        th1 = np.deg2rad(TILT_MAX_DEG) * (1.0 - smoothstep((t - T_D0) / (T_D1 - T_D0)))
    else:
        th1 = 0.0
    u_aside = smoothstep((t - T_D0) / (T_D1 - T_D0)) if t >= T_D0 else 0.0
    a1 = np.array([-np.sin(th1), 0.0, np.cos(th1)])
    e1 = np.array([np.cos(th1), 0.0, np.sin(th1)])
    O1 = PIVOT - L_PITCHER * a1 + R_PITCHER * e1 + u_aside * D_ASIDE
    # second tilt about the parked lip pivot (same plane)
    if t < T_E0:
        th2 = 0.0
    elif t < T_E0 + 2.5:
        # slow 2.5 s ramp (P2 smoke52 lesson: a 1 s ramp sloshes the residual milk OVER the
        # lip ballistically — it arcs back into the big cup instead of adhering to the outer
        # wall; the gentle ramp keeps the trickle slow enough for adhesion capture)
        th2 = np.deg2rad(TILT2_DEG) * smoothstep((t - T_E0) / 2.5)
    elif t < T_E1:
        th2 = np.deg2rad(TILT2_DEG)
    elif t < T_E2:
        th2 = np.deg2rad(TILT2_DEG) * (1.0 - smoothstep(t - T_E1))
    else:
        th2 = 0.0
    if th2 > 0.0:
        pivot2 = O1 + L_PITCHER * a1 - R_PITCHER * e1  # pour-side lip of the current pose
        a = -np.sin(th2) * e1 + np.cos(th2) * a1
        e2 = np.cos(th2) * e1 + np.sin(th2) * a1
        # the lip point is O + L*a - R*e2 (pour side), so O = pivot2 - L*a + R*e2
        O = pivot2 - L_PITCHER * a + R_PITCHER * e2
        return O, a, th1 + th2
    return O1, a1, th1


def quat_roty_neg(theta):
    """Quaternion (w, x, y, z) for rotY(-theta): maps mesh local +z to the pitcher axis."""
    return np.array([np.cos(theta / 2.0), 0.0, -np.sin(theta / 2.0), 0.0])


def main():
    global TILT_MAX_DEG, DT, SUBSTEPS, T_C, T_D0, T_D1, TILT2_DEG
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=50.0)
    parser.add_argument("--resettle", type=float, default=1.0,
                        help="quiet re-settle seconds after loading --settled (absorbs the "
                             "lip_round cavity shrink jolt; not recorded)")
    parser.add_argument("--epsd", type=float, default=0.01, help="diffusion_coeff (0 = off)")
    parser.add_argument("--dens-iters", type=int, default=20)
    parser.add_argument("--dens-relax", type=float, default=0.2)
    parser.add_argument("--leps", type=float, default=0.1)
    parser.add_argument("--visc-relax", type=float, default=0.01)
    parser.add_argument("--st-comp", type=float, default=1.0, help="st_compliance (rev9 recipe: 1.0)")
    parser.add_argument("--st-sdf", type=float, default=1.0)
    parser.add_argument("--st-dist-comp", type=float, default=40.0)
    parser.add_argument("--vdamp", type=float, default=1.0)
    parser.add_argument("--adh-comp", type=float, default=20.0)
    parser.add_argument("--wfric", type=float, default=0.0)
    parser.add_argument("--lip-round", type=float, default=0.008)
    parser.add_argument("--tilt-max", type=float, default=55.0, help="rev9 recipe: 55 deg")
    parser.add_argument("--tilt2", type=float, default=TILT2_DEG, help="phase E2 second tilt (deg)")
    parser.add_argument("--hold-until", type=float, default=T_C)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-st-dist", action="store_true")
    parser.add_argument("--settled", type=str, default=None)
    parser.add_argument("--st-max-neigh", type=int, default=256)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--substeps", type=int, default=None)
    args = parser.parse_args()

    if args.dt is not None:
        DT = args.dt
        if args.substeps is not None:
            SUBSTEPS = args.substeps

    ps = 0.008
    r_p = 0.5 * ps
    mp4_path = os.path.join(VIDEOS_DIR, f"mf14_shell_teapot{args.tag}.mp4")
    csv_path = os.path.join(VIDEOS_DIR, f"mf14{args.tag}_metrics.csv")

    TILT_MAX_DEG = args.tilt_max
    T_C = T_D0 = args.hold_until
    T_D1 = args.hold_until + 5.0
    TILT2_DEG = args.tilt2

    O0, A0, _ = pitcher_pose(0.0)

    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
        pbd_options=gs.options.PBDOptions(
            particle_size=ps,
            lower_bound=(-0.30, -0.30, 0.0),
            upper_bound=(1.30, 0.45, 2.10),  # x extended for the farther E2 park (D_ASIDE 0.5)
            # big cup (unchanged, no shell) + escape_band=3ps (MF-14): the radial clamp and
            # bottom plane only act within [R_IN, R_IN+3ps], so milk film clinging to the
            # outer wall stays captured while detached droplets escape and fall to the table
            # (legacy behavior teleported every below-rim particle back onto the wall and
            # lifted table droplets to z>=0.02, which made n_table structurally 0).
            boundary_cylinder=(0.0, 0.0, R_IN, T_BOTTOM, Z_RIM, 3.0 * ps),
            boundary_pitcher_shell=(*O0, *A0, R_PITCHER, L_PITCHER, T_PITCHER_WALL,
                                    T_PITCHER_FLOOR, args.lip_round),
            boundary_plane=(Z_TABLE,),  # table for the trickle drips (all particles)
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
            wall_adhesion_enabled=True,  # MF-14 is the adhesion demo: always on
            wall_adhesion_compliance=args.adh_comp,
            wall_friction=args.wfric,
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )

    cup = scene.add_entity(
        morph=gs.morphs.Mesh(file=CUP_OBJ, pos=(0.0, 0.0, 0.0), fixed=True, decimate=False),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.8, 0.8, 0.85, 0.35), opacity=0.35),
    )
    pitcher = scene.add_entity(
        morph=gs.morphs.Mesh(file=PITCHER_OBJ, pos=tuple(O0 - T_PITCHER_FLOOR * A0),
                             euler=(0.0, 0.0, 0.0), fixed=True, decimate=False),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.85, 0.75, 0.65, 0.5), opacity=0.5),
    )
    # table visual (P2 mf14): the PlaneBoundary does the physics; this fixed box only makes
    # the table surface visible so the trickle puddle reads on screen. Top face exactly at
    # Z_TABLE — same constraint plane as the analytic boundary, so the rigid coupling's
    # contact (if any) is consistent, not double.
    table = scene.add_entity(
        morph=gs.morphs.Box(size=(1.7, 1.0, 0.04), pos=(0.55, 0.0, Z_TABLE - 0.02), fixed=True),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.55, 0.55, 0.58, 1.0), opacity=1.0),
    )
    coffee = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=1.0,
            density_relaxation=args.dens_relax, viscosity_relaxation=args.visc_relax,
        ),
        morph=gs.morphs.Cylinder(
            radius=R_COFFEE, height=Z_COFFEE1 - Z_COFFEE0, pos=(0.0, 0.0, 0.5 * (Z_COFFEE0 + Z_COFFEE1)),
        ),
    )
    water_sample_pos = (0.0, 0.0, 0.85 + 0.5 * H_WATER)
    water = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=0.0, boundary_group=1,
            density_relaxation=args.dens_relax, viscosity_relaxation=args.visc_relax,
        ),
        morph=gs.morphs.Cylinder(radius=R_WATER, height=H_WATER, pos=water_sample_pos),
    )

    cam = scene.add_camera(res=(960, 1280), pos=(2.35, -2.35, 2.05), lookat=(0.35, 0.0, 1.0), fov=40, GUI=False)
    scene.build()
    print(f"cup geoms: {cup.n_geoms}, pitcher geoms: {pitcher.n_geoms}, "
          f"coffee particles: {coffee.n_particles}, water particles: {water.n_particles}")
    print(
        f"ps={ps} epsd={args.epsd} dens_iters={args.dens_iters} dens_relax={args.dens_relax:g} "
        f"leps={args.leps:g} st_comp={args.st_comp:g} st_dist_comp={args.st_dist_comp:g} "
        f"vdamp={args.vdamp:g} tilt_max={args.tilt_max:g} tilt2={args.tilt2:g} "
        f"lip_round={args.lip_round:g} adh={args.adh_comp:g} wfric={args.wfric:g} "
        f"seconds={args.seconds} tag='{args.tag}'"
    )

    if not args.no_video:
        scene.visualizer.rasterizer._camera_targets[cam.uid].point_size = args.point_size

    solver = scene.sim.pbd_solver
    assert type(solver.boundary).__name__ == "CylinderBoundary", f"unexpected boundary: {type(solver.boundary)}"
    assert type(solver.boundary2).__name__ == "TiltedCylinderShellBoundary", (
        f"unexpected boundary2: {type(solver.boundary2)}"
    )
    n_fluid = solver._n_fluid_particles
    n_coffee = coffee.n_particles
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    # ---- move the water into the upright pitcher (MF-12 flow, unchanged) -------------------
    O_w = np.array([PX, 0.0, PZ])
    water_target_center = O_w + np.array([0.0, 0.0, S0_WATER + 0.5 * H_WATER])
    delta = water_target_center - np.asarray(water_sample_pos)
    pos = water.get_particles_pos().cpu().numpy()
    pos = pos + delta
    K_Z = 0.62
    z_base = O_w[2] + S0_WATER
    pos[:, 2] = z_base + (pos[:, 2] - z_base) * K_Z
    water.set_particles_pos(pos)
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
    gid = solver.particles_ng.boundary_group.to_numpy()[:n_fluid, 0]
    assert (gid[:n_coffee] == 0).all() and (gid[n_coffee:n_fluid] == 1).all(), "boundary_group mismatch"

    # ---- settled-state load + quiet re-settle ----------------------------------------------
    if not args.settled:
        raise SystemExit("mf14 requires --settled (MF-12 presettle npz); in-script settle tunnels")
    dat = np.load(args.settled)
    pos0 = dat["pos"].astype(np.float32)
    n_total = int(solver.particles.pos.to_numpy().shape[0])
    assert pos0.shape == (n_total, 1, 3), f"settled pos shape {pos0.shape} != scene ({n_total}, 1, 3)"
    assert int(dat["n_coffee"]) == n_coffee and int(dat["n_fluid"]) == n_fluid
    solver.particles.pos.from_numpy(pos0)
    solver.particles.vel.from_numpy(np.zeros_like(pos0))
    print(f"loaded settled state from {args.settled}")
    # quiet re-settle: the lip_round dilation shrinks the cavity by rho, so the settled milk's
    # outermost layer starts inside the (dilated) shell solid and is projected inward on the
    # first substep; absorb that jolt here (not recorded)
    n_reset = int(round(args.resettle / DT))
    for i in range(n_reset):
        solver.set_pitcher_pose(O0, A0)
        pitcher.set_pos(O0 - T_PITCHER_FLOOR * A0, relative=True, zero_velocity=True)
        pitcher.set_quat(quat_roty_neg(0.0), relative=True, zero_velocity=True)
        scene.step()
    if n_reset:
        vel_np = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
        print(f"re-settle {args.resettle:.1f}s done, KE={0.5 * np.nansum((vel_np**2).sum(axis=1)):.3e}")

    n_frames = int(round(args.seconds / DT))
    rows = []
    wall0 = time.time()
    prev_sides = None
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
        if t < T_E0:
            return 4
        if t < T_E2:
            return 5
        return 6

    for i in range(n_frames + 1):
        t = i * DT
        O, A, th = pitcher_pose(t)
        solver.set_pitcher_pose(O, A)
        mesh_pos = O - T_PITCHER_FLOOR * A
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
        r_c = np.sqrt(safe[:n_coffee, 0] ** 2 + safe[:n_coffee, 1] ** 2)
        r_max = float(r_c.max())
        z_min = float(safe[:, 2].min())
        milk = safe[n_coffee:n_fluid]
        rel = milk - O
        s_w = rel @ A
        r_w = np.linalg.norm(rel - s_w[:, None] * A[None, :], axis=1)
        n_in_pitcher = int(((s_w < L_PITCHER + ps) & (s_w > -ps) & (r_w <= R_PITCHER + ps)).sum())
        ke = float(0.5 * np.nansum((vel**2).sum(axis=1)))
        r_all = np.sqrt(safe[:, 0] ** 2 + safe[:, 1] ** 2)
        in_cup = (r_all < R_IN) & (safe[:, 2] < Z_RIM)
        z_surf = float(np.percentile(safe[in_cup, 2], 99)) if in_cup.any() else float("nan")
        n_adh = solver.wall_adhesion_stats()
        # ---- P2 shell metrics (milk only) --------------------------------------------------
        d_sh, r_sh, s_sh = shell_sdf_np(milk, O, A, R_PITCHER, L_PITCHER, T_PITCHER_WALL,
                                        T_PITCHER_FLOOR, args.lip_round)
        inside_cav, outside_wall = side_masks(milk, O, A, R_PITCHER, L_PITCHER, T_PITCHER_WALL,
                                              T_PITCHER_FLOOR, r_p, args.lip_round)
        n_leak = 0
        if prev_sides is not None:
            prev_in, prev_out = prev_sides
            n_leak = int(((prev_in & outside_wall) | (prev_out & inside_cav)).sum())
        prev_sides = (inside_cav, outside_wall)
        adhering = np.abs(d_sh) <= r_p
        n_outer_wall = int((adhering & ((r_sh > R_PITCHER + 0.5 * T_PITCHER_WALL)
                                        | (s_sh > L_PITCHER) | (s_sh < -0.5 * T_PITCHER_FLOOR))).sum())
        r_milk_xy = np.sqrt(milk[:, 0] ** 2 + milk[:, 1] ** 2)
        n_table = int(((milk[:, 2] <= Z_TABLE + 2.0 * ps) & (r_milk_xy > R_IN + 0.03)).sum())
        n_milk_cup = int(((r_milk_xy < R_IN) & (milk[:, 2] < Z_RIM)).sum())
        # table-droplet shape metric (P2 ST-beading check): p95 height of ALL milk near the
        # table outside the cup footprint — a beaded droplet reads ~2-4 ps tall, a pancake
        # ~1 ps; the wide z gate keeps the bead TOP in the sample (unlike n_table's 2ps gate)
        near_table = (r_milk_xy > R_IN + 0.03) & (milk[:, 2] < Z_TABLE + 0.15)
        z_table_p95 = float(np.percentile(milk[near_table, 2], 95)) if near_table.any() else 0.0
        rows.append((i, t, phase, float(np.rad2deg(th)), sum_c, std_c, zc_c, r_max, z_min,
                     n_in_pitcher, ke, nan_count, z_surf, n_adh, n_outer_wall, n_table,
                     n_milk_cup, n_leak, z_table_p95))
        if i % 60 == 0:
            wall = time.time() - wall0
            print(
                f"t={t:6.2f}s  ph={phase}  th={np.rad2deg(th):5.1f}  n_pitcher={n_in_pitcher}"
                f"  n_outer={n_outer_wall}  n_table={n_table}  n_cup={n_milk_cup}  n_leak={n_leak}"
                f"  zt95={z_table_p95:.3f}"
                f"  KE={ke:.4e}  nan={nan_count}  ({wall / max(i, 1) * 1000:.0f} ms/frame avg)"
            )

    if not args.no_video:
        cam.stop_recording()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["frame", "t", "phase", "theta_deg", "sum_c", "std_c", "zc_c", "r_max_coffee",
             "z_min", "n_in_pitcher", "ke", "nan_count", "z_surf", "n_adh", "n_outer_wall",
             "n_table", "n_milk_cup", "n_leak", "z_table_p95"]
        )
        w.writerows(rows)

    # ---- pass/fail summary ------------------------------------------------------------------
    cols = ["frame", "t", "phase", "theta", "sum_c", "std_c", "zc_c", "r_max_coffee", "z_min",
            "n_in_pitcher", "ke", "nan", "z_surf", "n_adh", "n_outer_wall", "n_table",
            "n_milk_cup", "n_leak", "z_table_p95"]
    arr = {k: np.array([r_[j] for r_ in rows], dtype=float) for j, k in enumerate(cols)}
    sum_c0 = arr["sum_c"][0]
    drift = float(np.max(np.abs(arr["sum_c"] - sum_c0)) / sum_c0)
    p1 = bool(arr["r_max_coffee"].max() <= R_IN + ps + 1e-6)
    p2 = bool(arr["z_min"].min() >= Z_TABLE - ps - 1e-6)  # table plane is the floor now
    p3 = bool(arr["nan"].max() == 0)
    p4 = bool(drift < 0.01)
    mask_a = (arr["t"] >= 1.0) & (arr["t"] <= min(T_A - 0.5, args.seconds))
    p5 = None
    if mask_a.any():
        n0 = arr["n_in_pitcher"][0]
        leak = float(np.max(np.abs(arr["n_in_pitcher"][mask_a] - n0)) / max(n0, 1))
        p5 = bool(leak < 0.01)
    p6 = None
    if args.seconds >= T_D1:
        poured = arr["n_in_pitcher"][0] - arr["n_in_pitcher"][np.nonzero(arr["t"] >= T_D1)[0][0]]
        p6 = bool(poured >= 30000)
    p7 = None
    if args.seconds >= T_E2 + 2.0:
        mask_f = arr["t"] >= T_E2 + 1.0
        if mask_f.any():
            n_end = arr["n_in_pitcher"][-1]
            p7 = bool(np.max(np.abs(arr["n_in_pitcher"][mask_f] - n_end)) <= 0.01 * arr["n_in_pitcher"][0])
    print("=" * 70)
    print(f"[1] r_max_coffee max = {arr['r_max_coffee'].max():.4f}  (limit {R_IN + ps:.2f})  -> {'PASS' if p1 else 'FAIL'}")
    print(f"[1] z_min  min = {arr['z_min'].min():.4f}  (limit {Z_TABLE - ps:.2f})   -> {'PASS' if p2 else 'FAIL'}")
    print(f"[1] nan    max = {int(arr['nan'].max())}                              -> {'PASS' if p3 else 'FAIL'}")
    print(f"[2] sum_c drift = {drift:.3e}  (limit 1e-2, sum_c0={sum_c0:.2f})   -> {'PASS' if p4 else 'FAIL'}")
    if p5 is not None:
        print(f"[3] phase-A leak = {leak:.4%} of {int(n0)}  (limit 1%)          -> {'PASS' if p5 else 'FAIL'}")
    if p6 is not None:
        print(f"[4] pitcher pour: {int(arr['n_in_pitcher'][0])} -> poured {int(poured)} by t={T_D1:.0f}s "
              f"(need >=30000) -> {'PASS' if p6 else 'FAIL'}")
    if p7 is not None:
        print(f"[5] post-trickle n_in_pitcher stable (limit 1%)                   -> {'PASS' if p7 else 'FAIL'}")
    print(f"[6] big-cup surface z_surf: max {np.nanmax(arr['z_surf']):.3f} / final {arr['z_surf'][-1]:.3f}"
          f"  (rim {Z_RIM:.2f}) -> {'PASS' if np.nanmax(arr['z_surf']) < Z_RIM - ps else 'FAIL'}")
    # ---- P2 criteria (design 6.2) -----------------------------------------------------------
    q1 = q2 = q3 = q4 = None
    if args.seconds >= T_E2 + 5.0:
        win_t = (arr["t"] >= T_E0) & (arr["t"] <= T_E2 + 2.0)
        sustained = 0
        best = 0
        for v in arr["n_outer_wall"][win_t]:
            sustained = sustained + 1 if v >= 300 else 0
            best = max(best, sustained)
        q1 = bool(best >= int(1.0 / DT))  # >= 1s sustained at >= 300
        pre = np.nonzero((arr["t"] >= T_D1) & (arr["t"] < T_E0 - 0.5))[0]
        if len(pre):
            k = pre[-1]
            denom = arr["n_milk_cup"][k] + arr["n_table"][k]
            frac_cup = arr["n_milk_cup"][k] / denom if denom > 0 else float("nan")
            q2 = bool(frac_cup >= 0.9)
        denom_end = arr["n_milk_cup"][-1] + arr["n_table"][-1]
        frac_table = arr["n_table"][-1] / denom_end if denom_end > 0 else float("nan")
        q3 = bool(frac_table >= 0.6)
        ke_peak = arr["ke"][win_t].max()
        ke_late = arr["ke"][arr["t"] >= T_E2 + 5.0].max() if (arr["t"] >= T_E2 + 5.0).any() else float("nan")
        q4 = bool(arr["n_leak"].max() == 0 and ke_late < 0.1 * ke_peak)
        print(f"[P2-2] trickle n_outer_wall >=300 sustained {best} frames (need {int(1.0 / DT)}) -> {'PASS' if q1 else 'FAIL'}")
        if len(pre):
            print(f"[P2-3] main-pour frac_cup = {frac_cup:.3f} (>=0.9), end frac_table = {frac_table:.3f} (>=0.6) -> {'PASS' if (q2 and q3) else 'FAIL'}")
        print(f"[P2-4] n_leak max = {int(arr['n_leak'].max())}, KE late/peak = {ke_late:.3e}/{ke_peak:.3e} -> {'PASS' if q4 else 'FAIL'}")
    ms_frame = (time.time() - wall0) / max(n_frames, 1) * 1000
    print(f"KE final = {arr['ke'][-1]:.4e}   frame time = {ms_frame:.0f} ms (mf12 rev9: 755 ms, limit +10%)")
    print(f"csv: {csv_path}")
    if not args.no_video:
        print(f"mp4: {mp4_path}")


if __name__ == "__main__":
    main()
