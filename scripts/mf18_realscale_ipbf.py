"""MF-18 real-scale IPBF probe: the MF-17 pour narrative at REAL-WORLD dimensions.

Everything is scaled to physical reality (user spec): the big cup is a 10 cm tall, 7 cm
diameter mug (R_IN=0.035, L=0.10, walls 4 mm); the jug is a 10 cm, 5.6 cm diameter cup
(R_JUG=0.028, L=0.10); particle size drops 4x to ps=0.002 (2 mm), so the cup spans ~17.5
particles per radius (resolution parity with MF-17's 18.75) while the particles themselves
are 4x smaller in absolute terms.  Same narrative/timeline/tilts as MF-17 (a 52 s schedule
is slow-motion relative to the 1.7x faster real-time dynamics - timescales scale as
sqrt(L)).

Physics expectations at real scale, fixed before the runs:
  * surface tension of milk/water has a capillary length of ~2.7 mm; a metastable dome
    above the rim CANNOT exist physically.  The real-scale finale is "fill to the brim,
    then spill" - the dome gate below is informational only.
  * IPBF ST k~_st is dimensionful (m^-4) and must be recalibrated for the new ps; the
    --kst default is the 1/ps^3 heuristic rescale of the MF-11 value (5e4 * 4^3 = 3.2e6).
    Run --no-st first for a clean containment baseline, then calibrate.
  * damping_alpha_star carries scale (see IPBFOptions docstring); --dastar 8 is the MF-11
    value at ps=0.008 - observe and re-tune if the fluid looks over/under-damped.

Boundaries as in mf17_ipbf_pour.py: big cup = boundary_cylinder (escape band 3 ps), jug =
boundary_pitcher animated from the same jug_pose(t), table = boundary_plane (0,).  Fresh
sampling (no presettle at this scale); milk is sampled stacked above the coffee column for
the sampler guard, then teleported into the upright jug (MF-11 pattern).

Per-frame CSV: multiflow/videos/mf18_ipbf{tag}_metrics.csv
Videos: multiflow/videos/mf18_ipbf{tag}_{main,closeup}.mp4
"""

import argparse
import csv
import os
import sys
import time

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import genesis as gs  # noqa: E402

print("genesis loaded from:", gs.__file__)
assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), "Wrong genesis! Must be the multiflow copy."

VIDEOS_DIR = os.path.join(WORKSPACE, "videos")

DT = 1.0 / 60.0
SUBSTEPS = 8
PS = 0.002

# Real-world mug: 7 cm diameter mouth, 10 cm tall, 4 mm walls, standing on the table.
R_IN = 0.035
L_MUG = 0.10
T_WALL = 0.004
T_BOTTOM = 0.004
R_OUT = R_IN + T_WALL  # 0.039
Z_FLOOR = T_BOTTOM  # 0.004 cavity floor
Z_RIM = Z_FLOOR + L_MUG  # 0.104 rim
Z_TABLE = 0.0
ESCAPE_BAND = 3.0 * PS

# Real-world jug (small cup): 5.6 cm diameter, 10 cm tall; pitcher origin = inner bottom.
R_JUG = 0.028
L_JUG = 0.10
T_JUG_WALL = 0.004
T_JUG_BOTTOM = 0.004
Z_JUG_RIM = T_JUG_BOTTOM + L_JUG  # 0.104 jug-local rim

# fresh-sampled columns (lattice settles/compacts by the 0.8 volume factor into ~0.7-0.8 fill).
# H_MILK must fit the jug interior (S0 + H_MILK < L_JUG) or the sample top leaves the
# pitcher keep region on the first substep.
R_COFFEE = R_IN - 0.5 * PS  # 0.0345
H_COFFEE = 0.100
R_MILK = R_JUG - 0.75 * PS  # 0.0265
H_MILK = 0.095
S0 = 0.5 * PS  # sample margin above the cavity floor

# jug placement / timeline: MF-17 values scaled by 1/3 (same relative geometry).
PX, PZ = 0.14, 0.0
PIVOT_X, PIVOT_Z = 0.030, 0.125
PARK_X = 0.217
T_LIFT0, T_LIFT1 = 5.0, 7.0
T_TILT0, T_TILT1 = 7.0, 11.0
T_TRICKLE = 26.0
TRICKLE_RAMP = 1.5
T_DOME_STOP = 30.0
STOP_RAMP = 1.0
STOP_RETREAT_X = 0.04 / 3.0
STOP_LIFT_Z = 0.02 / 3.0
T_OVERFILL = 34.0
OVERFILL_RAMP = 0.75
T_BACK = 37.0
T_PARK0, T_PARK1 = 42.0, 46.0
TILT_MAX_DEG = 72.0
TILT_TRICKLE_DEG = 86.0
TILT_DOME_HOLD_DEG = 65.0
TILT_OVERFILL_DEG = 86.0
DUR = 52.0

LOWER_BOUND = np.array((-0.20, -0.20, 0.0), dtype=np.float32)
UPPER_BOUND = np.array((0.37, 0.20, 0.40), dtype=np.float32)


def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _ramp(t, t0, t1):
    return smoothstep((t - t0) / max(t1 - t0, 1e-9))


def quat_roty_neg(theta):
    return np.array([np.cos(theta / 2.0), 0.0, -np.sin(theta / 2.0), 0.0])


def jug_origin_p():
    return np.array([PIVOT_X + R_JUG, 0.0, PIVOT_Z - Z_RIM])


def jug_park():
    return np.array([PARK_X, 0.0, PZ])


def jug_theta(t):
    th_max = np.deg2rad(TILT_MAX_DEG)
    th_tr = np.deg2rad(TILT_TRICKLE_DEG)
    th_hold = np.deg2rad(TILT_DOME_HOLD_DEG)
    if t < T_TILT0:
        return 0.0
    if t < T_TILT1:
        return th_max * _ramp(t, T_TILT0, T_TILT1)
    if t < T_TRICKLE:
        return th_max
    if t < T_DOME_STOP:
        return th_max + (th_tr - th_max) * _ramp(t, T_TRICKLE, min(T_TRICKLE + TRICKLE_RAMP, T_DOME_STOP))
    if t < T_OVERFILL:
        angle_start = T_DOME_STOP + 0.5 * STOP_RAMP
        return th_tr + (th_hold - th_tr) * _ramp(t, angle_start, min(T_DOME_STOP + STOP_RAMP, T_OVERFILL))
    th_over = np.deg2rad(TILT_OVERFILL_DEG)
    if t < T_BACK:
        return th_hold + (th_over - th_hold) * _ramp(t, T_OVERFILL, min(T_OVERFILL + OVERFILL_RAMP, T_BACK))
    return th_over * (1.0 - _ramp(t, T_BACK, T_PARK1))


def jug_pose(t):
    """MF-17 jug pose (scale-agnostic): (origin=outer-bottom centre, quat, axis, theta, u)."""
    th = jug_theta(t)
    a = np.array([-np.sin(th), 0.0, np.cos(th)])
    e1 = np.array([np.cos(th), 0.0, np.sin(th)])
    origin_0 = np.array([PX, 0.0, PZ])
    lift_top = np.array([PX, 0.0, PIVOT_Z - T_JUG_BOTTOM])
    origin_p = jug_origin_p()
    pivot = np.array([PIVOT_X, 0.0, PIVOT_Z])
    rot = (pivot - Z_RIM * a + R_JUG * e1) - origin_p
    u_lift = _ramp(t, T_LIFT0, T_LIFT1)
    u_tilt = _ramp(t, T_TILT0, T_TILT1)
    u_park = _ramp(t, T_BACK, T_PARK1)
    stop_mid = T_DOME_STOP + 0.5 * STOP_RAMP
    if t < T_DOME_STOP:
        u_stop = 0.0
    elif t < stop_mid:
        u_stop = _ramp(t, T_DOME_STOP, stop_mid)
    elif t < T_OVERFILL:
        u_stop = 1.0
    elif t < T_OVERFILL + OVERFILL_RAMP:
        u_stop = 1.0 - _ramp(t, T_OVERFILL, T_OVERFILL + OVERFILL_RAMP)
    else:
        u_stop = 0.0
    origin = (origin_0
              + u_lift * (lift_top - origin_0)
              + u_tilt * (origin_p - lift_top)
              + rot
              + u_stop * np.array([STOP_RETREAT_X, 0.0, STOP_LIFT_Z])
              + u_park * (jug_park() - origin_p))
    quat = quat_roty_neg(th).astype(np.float32)
    return origin.astype(np.float32), quat, a.astype(np.float32), th, u_tilt


def coherent_mug_surface(pos, vel, bulk_mask, ps):
    """MF-17 coherent surface height (p75 of column tops) and above-rim dome count."""
    core_r = R_IN - 2.0 * ps
    speed = np.linalg.norm(vel, axis=1)
    r = np.linalg.norm(pos[:, :2], axis=1)
    eligible = bulk_mask & (r <= core_r) & (speed <= 0.2)
    idx = np.flatnonzero(eligible)
    if idx.size < 32:
        return float("nan"), 0

    cell = 1.5 * ps
    nxy = int(np.ceil(2.0 * core_r / cell)) + 1
    ij = np.floor((pos[idx, :2] + core_r) / cell).astype(np.int32)
    keys = ij[:, 0].astype(np.int64) * nxy + ij[:, 1]
    order = np.lexsort((pos[idx, 2], keys))
    sorted_idx, sorted_keys = idx[order], keys[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1])]
    ends = np.r_[starts[1:], sorted_idx.size]
    tops = np.asarray([sorted_idx[e - 1] for s, e in zip(starts, ends) if e - s >= 4], dtype=np.int64)
    if tops.size < 24:
        return float("nan"), 0
    bulk_idx = np.flatnonzero(bulk_mask)
    tree = cKDTree(np.asarray(pos[bulk_idx], dtype=np.float64))
    n_local = tree.query_ball_point(
        np.asarray(pos[tops], dtype=np.float64), r=1.75 * ps, return_length=True, workers=-1
    )
    tops = tops[np.asarray(n_local) >= 6]
    if tops.size < 24:
        return float("nan"), 0
    top_z = pos[tops, 2]
    z_surface = float(np.percentile(top_z, 75.0))
    n_dome = int(np.count_nonzero(top_z >= Z_RIM + 0.5 * ps))
    return z_surface, n_dome


def phase_of(t):
    if t < T_LIFT0:
        return 0
    if t < T_TILT0:
        return 1
    if t < T_TILT1:
        return 2
    if t < T_TRICKLE:
        return 3
    if t < T_BACK:
        return 4
    return 5 if t < T_PARK1 else 6


def longest_true_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def main():
    global T_LIFT0, T_LIFT1, T_TILT0, T_TILT1
    global T_TRICKLE, TRICKLE_RAMP, T_DOME_STOP, STOP_RAMP, STOP_RETREAT_X, STOP_LIFT_Z
    global T_OVERFILL, OVERFILL_RAMP, T_BACK, T_PARK0, T_PARK1
    global TILT_MAX_DEG, TILT_TRICKLE_DEG, TILT_DOME_HOLD_DEG, TILT_OVERFILL_DEG, DUR

    parser = argparse.ArgumentParser()
    parser.add_argument("--dur", type=float, default=DUR)
    parser.add_argument("--smoke", action="store_true", help="20 s compressed MF-17 smoke schedule")
    parser.add_argument("--substeps", type=int, default=SUBSTEPS)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1e-8)
    parser.add_argument("--xsph", type=float, default=0.1)
    parser.add_argument("--svisc", type=float, default=0.0,
                        help="surface_viscosity_xsph; interface-restricted XSPH smoothing (0 = off)")
    parser.add_argument("--epsd", type=float, default=0.01)
    parser.add_argument("--dastar", type=float, default=1024.0,
                        help="damping_alpha_star; calibrated at real scale (ps=0.002): 8..64 inert, "
                             "1024 = 2.5x quieter KE with the pour rate untouched, 8192 over-damped")
    parser.add_argument("--dbeta", type=float, default=60.0)
    parser.add_argument("--no-damp", action="store_true")
    parser.add_argument("--no-st", action="store_true")
    parser.add_argument("--kst", type=float, default=3.2e6,
                        help="st_stiffness; default = MF-11 5e4 * 4^3 (1/ps^3 heuristic for ps 0.008->0.002)")
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--st-max-neigh", type=int, default=256)
    parser.add_argument("--tilt-max", type=float, default=TILT_MAX_DEG)
    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument("--point-size-cup", type=float, default=10.0)
    args = parser.parse_args()

    ps = PS
    csv_path = os.path.join(VIDEOS_DIR, f"mf18_ipbf{args.tag}_metrics.csv")
    mp4_main = os.path.join(VIDEOS_DIR, f"mf18_ipbf{args.tag}_main.mp4")
    mp4_closeup = os.path.join(VIDEOS_DIR, f"mf18_ipbf{args.tag}_closeup.mp4")

    TILT_MAX_DEG = args.tilt_max
    DUR = 20.0 if args.smoke else args.dur
    if args.smoke:
        T_LIFT0, T_LIFT1 = 2.0, 3.5
        T_TILT0, T_TILT1 = 3.5, 6.0
        T_TRICKLE, TRICKLE_RAMP = 8.0, 1.0
        T_DOME_STOP, STOP_RAMP = 11.0, 1.0
        T_OVERFILL, OVERFILL_RAMP = 14.25, 0.75
        T_BACK, T_PARK0, T_PARK1 = 17.5, 17.5, 19.0
        TILT_MAX_DEG, TILT_TRICKLE_DEG = 72.0, 78.0
        TILT_DOME_HOLD_DEG, TILT_OVERFILL_DEG = 65.0, 112.0

    # pitcher pose at t=0: inner bottom centre = outer-bottom origin + T_JUG_BOTTOM * axis
    origin0, quat0, a0, _, _ = jug_pose(0.0)
    O_pitcher0 = origin0 + T_JUG_BOTTOM * a0

    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=args.substeps, gravity=(0.0, 0.0, -9.81)),
        ipbf_options=gs.options.IPBFOptions(
            particle_size=ps,
            lower_bound=tuple(LOWER_BOUND),
            upper_bound=tuple(UPPER_BOUND),
            boundary_particles=False,
            boundary_cylinder=(0.0, 0.0, R_IN, Z_FLOOR, Z_RIM, ESCAPE_BAND),
            boundary_pitcher=(*O_pitcher0, *a0, R_JUG, L_JUG),
            boundary_plane=(Z_TABLE,),
            ipbf_iterations=args.iters,
            alpha=args.alpha,
            viscosity_xsph=args.xsph,
            surface_viscosity_xsph=args.svisc,
            damping_enabled=not args.no_damp,
            damping_alpha_star=args.dastar,
            damping_beta=args.dbeta,
            diffusion_coeff=args.epsd,
            surface_tension_enabled=not args.no_st,
            st_model="quadratic",
            st_stiffness=args.kst,
            st_distance_enabled=True,
            st_distance_stiffness=args.kd,
            st_max_surface_neighbors=args.st_max_neigh,
        ),
        vis_options=gs.options.VisOptions(
            render_particle_as="points",
        ),
        show_viewer=False,
    )

    # real-scale visuals: simple translucent cylinders matching the clamp geometry (probe;
    # the MF-17 OBJ assets have a different aspect and cannot be reused at this scale)
    mug = scene.add_entity(
        morph=gs.morphs.Cylinder(radius=R_OUT, height=Z_RIM, pos=(0.0, 0.0, 0.5 * Z_RIM), fixed=True),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.85, 0.85, 0.9, 0.35), opacity=0.35),
    )
    jug = scene.add_entity(
        morph=gs.morphs.Cylinder(
            radius=R_JUG + T_JUG_WALL, height=Z_JUG_RIM,
            # morph reference = geometric centre; jug_pose's origin is the OUTER-BOTTOM centre,
            # so the centre rides one half-height up the (upright) axis
            pos=tuple(origin0 + np.array([0.0, 0.0, 0.5 * Z_JUG_RIM])), fixed=True,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.9, 0.92, 0.95, 0.45), opacity=0.45),
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(size=(0.87, 0.53, 0.013), pos=(0.117, 0.0, Z_TABLE - 0.0067), fixed=True),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.55, 0.55, 0.58, 1.0), opacity=1.0),
    )

    coffee = scene.add_entity(
        material=gs.materials.IPBF.Liquid(sampler="regular", rho=1000.0, c_init=1.0, boundary_group=0),
        morph=gs.morphs.Cylinder(
            radius=R_COFFEE,
            height=H_COFFEE,
            pos=(0.0, 0.0, Z_FLOOR + S0 + 0.5 * H_COFFEE),
        ),
    )
    # sampler guard: milk stacked ABOVE the coffee column inside the cup mouth (open rim),
    # teleported into the upright jug right after build (MF-11 pattern).
    milk_sample_pos = (0.0, 0.0, Z_FLOOR + S0 + H_COFFEE + 2.5 * PS + 0.5 * H_MILK)
    milk = scene.add_entity(
        material=gs.materials.IPBF.Liquid(sampler="regular", rho=1000.0, c_init=0.0, boundary_group=1),
        morph=gs.morphs.Cylinder(
            radius=R_MILK,
            height=H_MILK,
            pos=milk_sample_pos,
        ),
    )

    camera_res = (480, 640) if args.smoke else (960, 1280)
    cam = scene.add_camera(res=camera_res, pos=(0.093, -0.473, 0.210), lookat=(0.073, 0.0, 0.077), fov=42, GUI=False)
    cam_close = scene.add_camera(res=camera_res, pos=(0.147, -0.133, 0.143), lookat=(0.020, 0.0, 0.107), fov=32, GUI=False)
    scene.build()
    print(f"mug geoms: {mug.n_geoms}, jug geoms: {jug.n_geoms}, coffee: {coffee.n_particles}, "
          f"milk: {milk.n_particles}")
    print(f"REAL SCALE: R_IN={R_IN} rim={Z_RIM} R_JUG={R_JUG} ps={ps} ({R_IN / ps:.1f} particles/radius) "
          f"dt={DT:.5f} substeps={args.substeps} iters={args.iters} alpha={args.alpha:g} "
          f"xsph={args.xsph:g} svisc={args.svisc:g} damp={not args.no_damp} dastar={args.dastar:g} "
          f"st={not args.no_st} kst={args.kst:g} kd={args.kd:g} dur={DUR:g} tag='{args.tag}'")

    scene.visualizer.rasterizer._camera_targets[cam.uid].point_size = args.point_size
    scene.visualizer.rasterizer._camera_targets[cam_close.uid].point_size = args.point_size_cup

    solver = scene.sim.ipbf_solver
    assert type(solver.boundary).__name__ == "CylinderBoundary", f"unexpected boundary: {type(solver.boundary)}"
    assert type(solver.boundary2).__name__ == "TiltedCylinderBoundary", f"unexpected boundary2: {type(solver.boundary2)}"
    assert type(solver.boundary3).__name__ == "PlaneBoundary", f"unexpected boundary3: {type(solver.boundary3)}"
    n_fluid = solver._n_fluid_particles
    n_coffee = coffee.n_particles
    n_milk = milk.n_particles
    assert n_coffee + n_milk == n_fluid
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    # ---- teleport the milk into the upright jug (pure translation; MF-11 pattern) ---------
    O_jug0 = origin0 + T_JUG_BOTTOM * a0
    milk_target_center = O_jug0 + np.array([0.0, 0.0, S0 + 0.5 * H_MILK])
    delta = milk_target_center - np.asarray(milk_sample_pos)
    milk_pos = milk.get_particles_pos().cpu().numpy()  # (n, 3); env dim stripped (n_envs==0)
    milk.set_particles_pos(milk_pos + delta)
    bg = solver.particles_ng.boundary_group.to_numpy()
    bg[n_coffee:n_fluid, :] = 1
    solver.particles_ng.boundary_group.from_numpy(bg)
    got = milk.get_particles_pos().cpu().numpy()
    rel_chk = got - O_jug0
    s_chk = rel_chk[:, 2]
    r_chk = np.linalg.norm(rel_chk[:, :2], axis=1)
    print(f"milk in jug: s in [{s_chk.min():.4f}, {s_chk.max():.4f}] (L={L_JUG}), "
          f"r in [{r_chk.min():.4f}, {r_chk.max():.4f}] (R={R_JUG})")
    assert r_chk.max() < R_JUG and s_chk.min() > 0.0 and s_chk.max() < L_JUG
    gid = solver.particles_ng.boundary_group.to_numpy()[:n_fluid, 0]
    assert (gid[:n_coffee] == 0).all() and (gid[n_coffee:] == 1).all(), "boundary_group mismatch after teleport"

    n_frames = int(round(DUR / DT))
    rows = []
    wall0 = time.time()
    if not args.no_video:
        cam.start_recording(save_to_filename=mp4_main, fps=args.fps)
        cam_close.start_recording(save_to_filename=mp4_closeup, fps=args.fps)
    print_every = int(round(1.0 / DT))

    for i in range(n_frames + 1):
        t = i * DT
        origin, quat, axis, th, _ = jug_pose(t)
        solver.set_pitcher_pose(origin + T_JUG_BOTTOM * axis, axis)
        # visual centre = outer-bottom origin + half-height along the CURRENT axis (rotating
        # the centre-fixed morph about its own centre lands the body exactly on the pose)
        jug.set_pos(origin + 0.5 * Z_JUG_RIM * axis, relative=True, zero_velocity=True)
        jug.set_quat(quat, relative=True, zero_velocity=True)

        frame_t0 = time.time()
        if i > 0:
            scene.step()

        phase = phase_of(t)
        pos_all = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
        vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
        c = solver.particles.c.to_numpy()[:n_fluid, 0]
        nan_count = int((~np.isfinite(pos_all)).sum() + (~np.isfinite(vel)).sum() + (~np.isfinite(c)).sum())
        safe = np.nan_to_num(pos_all, nan=0.0, posinf=0.0, neginf=0.0)
        safe_v = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        safe_c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        r_all = np.linalg.norm(safe[:, :2], axis=1)
        z_all = safe[:, 2]
        sum_c = float(safe_c.sum())
        std_c = float(safe_c.std())
        ke = float(0.5 * np.nansum((safe_v ** 2).sum(axis=1)))
        z_min = float(z_all.min())

        O_p = origin + T_JUG_BOTTOM * axis
        rel_j = safe - O_p
        s_j = rel_j @ axis
        r_j = np.linalg.norm(rel_j - s_j[:, None] * axis[None, :], axis=1)
        n_milk_jug = int(((s_j[n_coffee:] > -ps) & (s_j[n_coffee:] < L_JUG + ps)
                          & (r_j[n_coffee:] <= R_JUG + ps)).sum())
        in_jug_fp = (s_j > Z_FLOOR - 3.0 * ps) & (s_j < Z_RIM + 3.0 * ps) \
            & (r_j <= R_JUG + T_JUG_WALL + 3.0 * ps)
        not_jug = np.ones(n_fluid, dtype=bool)
        not_jug[n_coffee:] = ~in_jug_fp[n_coffee:]
        in_mug = (r_all <= R_IN + ps) & (z_all >= Z_FLOOR - ps) & (z_all <= Z_RIM + ps)
        n_coffee_mug = int((in_mug[:n_coffee]).sum())
        n_milk_mug = int((in_mug & not_jug)[n_coffee:].sum())
        bulk_surface = in_mug & not_jug
        z_surf, n_dome = coherent_mug_surface(safe, safe_v, bulk_surface, ps)
        n_over = int((not_jug[n_coffee:] & (z_all[n_coffee:] > Z_RIM + 0.5 * ps)
                      & (r_all[n_coffee:] > R_IN + ps)).sum())
        n_table = int((not_jug[n_coffee:] & (z_all[n_coffee:] <= 2.5 * ps)
                       & (r_all[n_coffee:] > R_OUT + 3.0 * ps)).sum())
        frame_ms = (time.time() - frame_t0) * 1000.0
        rows.append((i, t, phase, float(np.rad2deg(th)), sum_c, std_c, ke, nan_count,
                     n_coffee_mug, n_milk_jug, n_milk_mug, z_surf, n_dome, n_over, n_table,
                     z_min, frame_ms))
        if i % print_every == 0 or i == n_frames:
            wall = time.time() - wall0
            print(f"t={t:6.2f}s  ph={phase}  th={np.rad2deg(th):5.1f}  jug={n_milk_jug:6d}  "
                  f"mugM={n_milk_mug:5d}  z_surf={z_surf:6.4f}  dome={n_dome:5d}  over={n_over:4d}  "
                  f"table={n_table:5d}  coffeeM={n_coffee_mug:6d}  KE={ke:.3e}  nan={nan_count}  "
                  f"z_min={z_min:+.4f}  ({wall / max(i, 1) * 1000:.0f} ms/frame avg)")

    if not args.no_video:
        cam.stop_recording()
        cam_close.stop_recording()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "phase", "theta_deg", "sum_c", "std_c", "ke", "nan",
                    "n_coffee_mug", "n_milk_jug", "n_milk_mug", "z_surf_coherent_p75", "n_dome",
                    "n_over_rim", "n_table", "z_min", "frame_ms"])
        w.writerows(rows)

    # ---- probe summary ---------------------------------------------------------------------
    cols = ["frame", "t", "phase", "theta", "sum_c", "std_c", "ke", "nan", "n_coffee_mug",
            "n_milk_jug", "n_milk_mug", "z_surf", "n_dome", "n_over", "n_table", "z_min", "frame_ms"]
    arr = {k: np.array([r_[j] for r_ in rows], dtype=float) for j, k in enumerate(cols)}
    t_end = arr["t"][-1]
    sum_c0 = arr["sum_c"][0]
    drift = float(np.max(np.abs(arr["sum_c"] - sum_c0)) / sum_c0) if sum_c0 > 0 else float("nan")
    print("=" * 78)
    p1 = bool(arr["nan"].max() == 0)
    print(f"[1] nan max = {int(arr['nan'].max())}                                              -> {'PASS' if p1 else 'FAIL'}")
    p2 = bool(arr["z_min"].min() >= -ps - 1e-6)
    print(f"[2] z_min min = {arr['z_min'].min():+.5f}  (table {Z_TABLE:.3f}, limit {-ps:.4f})     -> {'PASS' if p2 else 'FAIL'}")
    p3 = bool(drift < 0.01)
    print(f"[3] sum_c drift = {drift:.3e}  (limit 1e-2, sum_c0={sum_c0:.2f})        -> {'PASS' if p3 else 'FAIL'}")
    mask_a = (arr["t"] >= 1.0) & (arr["t"] <= min(T_LIFT0 - 0.5, t_end))
    p4 = True
    if mask_a.any():
        n0 = arr["n_milk_jug"][mask_a][0]
        leak = float(np.max(np.abs(arr["n_milk_jug"][mask_a] - n0)) / max(n0, 1))
        p4 = bool(leak < 0.01)
        print(f"[4] phase-A milk leak = {leak:.4%} of {int(n0)} (limit 1%)       -> {'PASS' if p4 else 'FAIL'}")
    p5 = p6 = p7 = p8 = True
    if t_end >= T_BACK:
        drain = float(arr["n_milk_jug"][-1] / max(arr["n_milk_jug"][0], 1))
        p5 = bool(drain <= 0.5)
        print(f"[5] jug drain: {int(arr['n_milk_jug'][0])} -> {int(arr['n_milk_jug'][-1])} / {n_milk} "
              f"(need <=50% left) -> {'PASS' if p5 else 'FAIL'}")
    if t_end >= T_TRICKLE:
        m2m = float(arr["n_milk_mug"].max()) / n_milk
        p6 = bool(m2m >= 0.25)
        print(f"[6] milk reached mug: max n_milk_mug = {int(arr['n_milk_mug'].max())} "
              f"({m2m:.1%} of milk, need >=25%) -> {'PASS' if p6 else 'FAIL'}")
    if t_end >= T_OVERFILL:
        hold = (arr["t"] >= T_DOME_STOP + STOP_RAMP) & (arr["t"] < T_OVERFILL)
        if np.count_nonzero(hold) >= 2:
            hold_above = hold & (arr["z_surf"] >= Z_RIM + 0.5 * ps) & (arr["n_dome"] > 0)
            hold_run = float(longest_true_run(hold_above)) * DT
            print(f"[7] dome hold (INFORMATIONAL at real scale: capillary length ~2.7 mm makes a "
                  f"physical above-rim dome impossible): longest={hold_run:.2f}s")
        n_tab_end = int(arr["n_table"][-1])
        n_ov_max = int(arr["n_over"].max())
        p8 = bool(n_tab_end > 0)
        print(f"[8] spill: n_table end={n_tab_end}, n_over max={n_ov_max} "
              f"-> {'PASS' if p8 else 'FAIL'}")
    ms_p95 = float(np.percentile(arr["frame_ms"], 95))
    print(f"[9] frame_ms p95 = {ms_p95:.0f} (diagnostic)")
    mandatory = [p1, p2, p3, p4, p5, p6]
    if not all(mandatory):
        print("MF18 REAL-SCALE PROBE FAIL")
        raise SystemExit(1)
    print("MF18 REAL-SCALE PROBE CORE PASS")

    print(f"csv: {csv_path}")
    if not args.no_video:
        print(f"mp4: {mp4_main}")
        print(f"mp4: {mp4_closeup}")


if __name__ == "__main__":
    main()
