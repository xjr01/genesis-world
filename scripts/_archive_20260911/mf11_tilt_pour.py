"""MF-11: tilt-animation pour demo — a small upright pitcher pre-filled with water progressively
tilts about its pour-side lip pivot and pours a continuous stream into the big cup of coffee.

User-specified narrative (supersedes the MF-10 fixed-tilt/deactivated-pop-in design):
  phase A (0-6s)    : everything at rest — coffee settles in the big cup, water settles in the
                      UPRIGHT pitcher (also validates the vertical pitcher clamp does not leak)
  phase B (6-14s)   : pitcher smoothly tilts theta: 0 -> --tilt-max deg about the pour-side lip
                      pivot (smoothstep); water spills over the lip into the big cup
  phase C (14-24s)  : hold at max tilt; the stream continues and thins to a trickle
  phase D (24-30s)  : theta back to 0 AND the pitcher lifts up / moves aside (smoothstep);
                      the stream breaks
  phase E (30-45s)  : pitcher parked aside; the big cup settles while water diffuses into coffee

Physics (MF-11 boundary ownership fix): the big cup clamp (`boundary_cylinder`, wall below rim +
above-rim escape band) applies ONLY to boundary_group=0 particles; the pitcher clamp
(`boundary_pitcher` -> TiltedCylinderBoundary, DYNAMIC pose via `solver.set_pitcher_pose`)
applies ONLY to boundary_group=1 particles (water entity, material `boundary_group=1`). A
group-1 particle that leaves the pitcher's keep region (past the rim, beyond the escape band,
or below the bottom) permanently transfers to group 0. This fixes the MF-10 bug where the
GLOBAL big-cup clamp also grabbed the pitcher water (r_xy > R_cup) and froze it into
wall-hugging chunks.

The pitcher visual mesh is a FREE rigid body with gravity_compensation=1.0, teleported every
frame to the same pose as the physics clamp (visualization only, no fluid coupling).

Per-frame CSV: multiflow/videos/mf11{tag}_metrics.csv
  (frame, t, phase, theta_deg, sum_c, std_c, zc_c, r_max_coffee, z_min, n_in_pitcher, ke, nan)
Video: multiflow/videos/mf11_tilt_pour{tag}.mp4

Run (smoke first!):
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf11_tilt_pour.py --seconds 15 --tag _smoke
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf11_tilt_pour.py --seconds 45 --tag _final
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

# coffee column inside the big cup: r=0.18, z in [0.03, 0.50] (~half the cup), same as MF-9/10
R_COFFEE = 0.18
Z_COFFEE0 = T_BOTTOM + 0.01  # 0.03
Z_COFFEE1 = 0.50

# pitcher geometry (assets/gen_cup.py --r-in 0.16 --h-in 0.45 --t-wall 0.016 --t-bottom 0.016):
# cavity floor at local z=0.016, rim at local z=0.466
R_PITCHER = 0.16
L_PITCHER = 0.45  # cavity floor -> rim, along the axis
T_PITCHER_FLOOR = 0.016
# water pre-fill inside the upright pitcher (surface below the lip)
R_WATER = R_PITCHER - 0.008  # one particle radius margin to the clamp wall
S0_WATER = 0.03  # bottom margin above the pitcher floor plane
H_WATER = 0.32  # axial fill height (lip is at 0.45 -> freeboard 0.10)

# timeline (seconds)
T_A = 6.0  # rest
T_B0, T_B1 = 6.0, 14.0  # tilt up
T_C = 24.0  # hold until
T_D0, T_D1 = 24.0, 30.0  # tilt back + move aside
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
    global TILT_MAX_DEG
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--epsd", type=float, default=0.01, help="diffusion_coeff (0 = off)")
    parser.add_argument("--kst", type=float, default=5e4, help="st_stiffness (damp8_st5e4 recipe)")
    parser.add_argument("--kd", type=float, default=0.5, help="st_distance_stiffness")
    parser.add_argument("--dastar", type=float, default=8.0, help="damping_alpha_star (user spec)")
    parser.add_argument("--dbeta", type=float, default=60.0, help="damping_beta")
    parser.add_argument("--tilt-max", type=float, default=TILT_MAX_DEG, help="max tilt angle (deg)")
    parser.add_argument("--no-damp", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-st-dist", action="store_true")
    parser.add_argument("--st-max-neigh", type=int, default=256, help="st_max_surface_neighbors")
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    ps = 0.008
    mp4_path = os.path.join(VIDEOS_DIR, f"mf11_tilt_pour{args.tag}.mp4")
    csv_path = os.path.join(VIDEOS_DIR, f"mf11{args.tag}_metrics.csv")

    TILT_MAX_DEG = args.tilt_max

    O0, A0, _ = pitcher_pose(0.0)

    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
        ipbf_options=gs.options.IPBFOptions(
            particle_size=ps,
            lower_bound=(-0.30, -0.30, 0.0),
            upper_bound=(1.00, 0.45, 2.10),
            boundary_particles=False,
            boundary_cylinder=(0.0, 0.0, R_IN, T_BOTTOM, Z_RIM),  # big cup: wall below rim + escape band
            boundary_pitcher=(*O0, *A0, R_PITCHER, L_PITCHER),  # pitcher: dynamic open-mouth clamp
            ipbf_iterations=2,
            alpha=1e-8,
            viscosity_xsph=0.1,
            damping_enabled=not args.no_damp,
            damping_alpha_star=args.dastar,
            damping_beta=args.dbeta,
            diffusion_coeff=args.epsd,
            surface_tension_enabled=True,
            st_model="quadratic",
            st_stiffness=args.kst,
            st_distance_enabled=not args.no_st_dist,
            st_distance_stiffness=args.kd,
            st_max_surface_neighbors=args.st_max_neigh,
        ),
        vis_options=gs.options.VisOptions(
            render_particle_as="points",  # per-particle concentration coloring (IPBF points branch)
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

    # pitcher: visualization only (dynamic TiltedCylinderBoundary does the physics). FREE rigid
    # with gravity compensation so it floats; teleported every frame to the clamp pose. Spawn it
    # directly at the initial (upright) pose so it never overlaps the big cup.
    pitcher = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=PITCHER_OBJ,
            pos=tuple(O0 - T_PITCHER_FLOOR * A0),
            euler=(0.0, 0.0, 0.0),
            fixed=False,
            decimate=False,
        ),
        material=gs.materials.Rigid(gravity_compensation=1.0),
        surface=gs.surfaces.Default(color=(0.85, 0.75, 0.65, 0.5), opacity=0.5),
    )

    coffee = scene.add_entity(
        material=gs.materials.IPBF.Liquid(sampler="regular", rho=1000.0, c_init=1.0),
        morph=gs.morphs.Cylinder(
            radius=R_COFFEE,
            height=Z_COFFEE1 - Z_COFFEE0,
            pos=(0.0, 0.0, 0.5 * (Z_COFFEE0 + Z_COFFEE1)),
        ),
    )
    # water sampled upright INSIDE the big cup (teleported into the upright pitcher right after
    # build, before the first real step). Sampling above the coffee column avoids any overlap
    # during scene.build()'s warm-up step; the group flip caused by that warm-up step (water
    # outside the pitcher keep region) is restored right after the teleport below. The pitcher
    # starts vertical, so the transform is a pure translation.
    water_sample_pos = (0.0, 0.0, 0.59 + 0.5 * H_WATER)
    water = scene.add_entity(
        material=gs.materials.IPBF.Liquid(sampler="regular", rho=1000.0, c_init=0.0, boundary_group=1),
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
        f"ps={ps} epsd={args.epsd} kst={args.kst:g} dastar={args.dastar:g} dbeta={args.dbeta:g} "
        f"tilt_max={args.tilt_max:g} seconds={args.seconds} tag='{args.tag}'"
    )

    if not args.no_video:
        scene.visualizer.rasterizer._camera_targets[cam.uid].point_size = args.point_size

    solver = scene.sim.ipbf_solver
    assert type(solver.boundary).__name__ == "CylinderBoundary", f"unexpected boundary: {type(solver.boundary)}"
    assert type(solver.boundary2).__name__ == "TiltedCylinderBoundary", f"unexpected boundary2: {type(solver.boundary2)}"
    n_fluid = solver._n_fluid_particles
    n_coffee = coffee.n_particles
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    # ---- move the water into the upright pitcher (pure translation) ----------------------
    O_w = np.array([PX, 0.0, PZ])  # upright pitcher bottom-center (theta=0, aside=0)
    water_target_center = O_w + np.array([0.0, 0.0, S0_WATER + 0.5 * H_WATER])
    delta = water_target_center - np.asarray(water_sample_pos)
    pos = water.get_particles_pos().cpu().numpy()  # (n, 3); env dim already stripped (n_envs==0)
    water.set_particles_pos(pos + delta)
    # scene.build() runs one warm-up step while the water is still at its big-cup sample
    # position — outside the pitcher keep region — so _kernel_impose_boundary permanently
    # transferred the whole water entity to group 0. Restore group 1 now that the water is
    # actually inside the pitcher (verified below).
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
        # mass + principal inertia axes), which is offset from the mesh frame and left the visual
        # pitcher tilted at theta=0 in the smoke run.
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
        rows.append((i, t, phase, float(np.rad2deg(th)), sum_c, std_c, zc_c, r_max, z_min, n_in_pitcher, ke, nan_count))
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
             "z_min", "n_in_pitcher", "ke", "nan_count"]
        )
        w.writerows(rows)

    # ---- pass/fail summary ------------------------------------------------------------------
    cols = ["frame", "t", "phase", "theta", "sum_c", "std_c", "zc_c", "r_max_coffee", "z_min",
            "n_in_pitcher", "ke", "nan"]
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
    # pour check: by the end most of the water must have left the pitcher
    p6 = None
    if args.seconds >= T_D1:
        p6 = bool(arr["n_in_pitcher"][-1] <= 0.6 * arr["n_in_pitcher"][0])
    print("=" * 70)
    print(f"[1] r_max_coffee max = {arr['r_max_coffee'].max():.4f}  (limit {R_IN + ps:.2f})  -> {'PASS' if p1 else 'FAIL'}")
    print(f"[1] z_min  min = {arr['z_min'].min():.4f}  (limit {T_BOTTOM - ps:.2f})   -> {'PASS' if p2 else 'FAIL'}")
    print(f"[1] nan    max = {int(arr['nan'].max())}                              -> {'PASS' if p3 else 'FAIL'}")
    print(f"[2] sum_c drift = {drift:.3e}  (limit 1e-2, sum_c0={sum_c0:.2f})   -> {'PASS' if p4 else 'FAIL'}")
    if p5 is not None:
        print(f"[3] phase-A leak = {leak:.4%} of {int(n0)}  (limit 1%)          -> {'PASS' if p5 else 'FAIL'}")
    if p6 is not None:
        print(f"[4] pitcher drain: {int(arr['n_in_pitcher'][0])} -> {int(arr['n_in_pitcher'][-1])} / {n_w} "
              f"-> {'PASS' if p6 else 'FAIL'}")
    print(f"KE final = {arr['ke'][-1]:.4e}")
    print(f"csv: {csv_path}")
    if not args.no_video:
        print(f"mp4: {mp4_path}")


if __name__ == "__main__":
    main()
