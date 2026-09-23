"""MF-10: pitcher pour demo — a small tilted pitcher slowly pours water into the big cup of coffee.

Scene: big open-top cup (assets/cup_big.obj, inner R=0.20, cavity floor z=0.02, rim z=1.02;
visualization only, physics = CylinderBoundary clamp with the radial wall limited to z <= rim so
the pitcher hovering above is not sucked onto the cup wall) with a resting coffee column in the lower
half (c_init=1.0, brown). Above the rim and off to one side: a small pitcher (assets/pitcher.obj,
inner R=0.16, depth 0.75, visualization only) tilted --tilt degrees from vertical with its mouth
lowest lip hovering at (--lip-x, 0, --lip-z) above the big-cup opening. The water entity
(c_init=0.0, white) is sampled as an upright cylinder and then transformed into the tilted
pitcher frame with `set_particles_pos`; it starts DEACTIVATED (particles_ng.active=False ->
invisible, excluded from the hash/neighbor loops) and is released by the KE gate like MF-9.

Pitcher physics: `IPBFOptions.boundary_pitcher` -> TiltedCylinderBoundary (open-mouth tilted
cylinder clamp; side wall band-limited so the poured stream falling past the wall extension is
not sucked back in; mouth open past the rim so the pitcher actually pours). The pour rate is
controlled purely by geometry: the tilt angle sets the below-lip capacity, water above it drains
under gravity through the mouth.

Per-frame CSV: multiflow/videos/mf10{tag}_metrics.csv
  (frame, t, phase, sum_c, std_c, zc_c, r_max_coffee, z_min, n_in_pitcher, ke, nan_count)
Video: multiflow/videos/mf10_pitcher_pour{tag}.mp4

Run (smoke first!):
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf10_pitcher_pour.py --seconds-a 8 --seconds-b 4 --tag _smoke
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf10_pitcher_pour.py --seconds-a 15 --seconds-b 25 --tag _pitcher
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
PITCHER_OBJ = os.path.join(WORKSPACE, "assets", "pitcher.obj")
VIDEOS_DIR = os.path.join(WORKSPACE, "videos")

# big cup geometry (assets/gen_cup.py --r-in 0.2 --h-in 1.0): cavity floor z=0.02, rim z=1.02
R_IN = 0.20
T_BOTTOM = 0.02
Z_RIM = T_BOTTOM + 1.0
DT = 1.0 / 60.0
SUBSTEPS = 8

# coffee column inside the big cup: r=0.18, z in [0.03, 0.50] (~half the cup), same as MF-9
R_COFFEE = 0.18
Z_COFFEE0 = T_BOTTOM + 0.01  # 0.03
Z_COFFEE1 = 0.50

# pitcher geometry (assets/gen_cup.py --r-in 0.16 --h-in 0.75 --t-wall 0.016 --t-bottom 0.016):
# cavity floor at local z=0.016, rim at local z=0.766
R_PITCHER = 0.16
L_PITCHER = 0.75  # cavity floor -> rim, along the axis
T_PITCHER_FLOOR = 0.016
# water column inside the pitcher (upright sampling before the tilt transform)
R_WATER = R_PITCHER - 0.008  # one particle radius margin to the clamp wall
H_WATER = 0.62  # axial fill; S0 + H_WATER must stay below L_PITCHER
S0_WATER = 0.03  # bottom margin above the pitcher floor plane

# KE gate for phase A -> B (same calibration as MF-9): KE <= max(--ke-rel * KE_peak,
# KE_abs_floor) sustained for 1 s; KE_abs_floor = n_fluid * 0.006.
KE_REL = 0.01
KE_ABS_PER_PARTICLE = 0.006
KE_SUSTAIN_S = 1.0
KE_GATE_MIN_T = 5.0


def pitcher_frame(tilt_deg, lip_target):
    """Bottom-center origin O, unit axis a (bottom->mouth), and orthonormal in-plane basis e1/e2
    for a pitcher whose rim's lowest lip point sits at lip_target. Tilt is about the y axis,
    mouth pointing toward -x."""
    th = np.deg2rad(tilt_deg)
    a = np.array([-np.sin(th), 0.0, np.cos(th)])  # bottom -> mouth
    e1 = np.array([np.cos(th), 0.0, np.sin(th)])  # in-plane, roughly toward the mouth side
    e2 = np.array([0.0, 1.0, 0.0])
    # downward-most direction within the rim plane (points from rim center to the lowest lip)
    t = np.array([-1.0, 0.0, 0.0])  # horizontal tilt direction (mouth side)
    dvec = t - (t @ a) * a
    d_hat = dvec / np.linalg.norm(dvec)  # has negative z component
    # M = rim center = O + L_PITCHER * a;  lip_low = M + R_PITCHER * d_hat
    M = np.asarray(lip_target, dtype=float) - R_PITCHER * d_hat
    O = M - L_PITCHER * a
    return O, a, e1, e2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsd", type=float, default=0.01, help="diffusion_coeff (0 = off)")
    parser.add_argument("--kst", type=float, default=5e4, help="st_stiffness (damp8_st5e4 recipe)")
    parser.add_argument("--dastar", type=float, default=7.0, help="damping_alpha_star")
    parser.add_argument("--dbeta", type=float, default=60.0, help="damping_beta")
    parser.add_argument("--kd", type=float, default=0.5, help="st_distance_stiffness")
    parser.add_argument("--st-max-neigh", type=int, default=256, help="st_max_surface_neighbors")
    parser.add_argument("--no-st", action="store_true", help="disable surface tension (isolation test)")
    parser.add_argument("--no-st-dist", action="store_true", help="disable the ST one-sided distance constraint")
    parser.add_argument("--no-damp", action="store_true", help="disable artificial damping (isolation test)")
    parser.add_argument("--tilt", type=float, default=65.0, help="pitcher tilt from vertical, degrees")
    parser.add_argument("--lip-x", type=float, default=0.02, help="world x of the mouth's lowest lip point")
    # NOTE: with tilt=65deg the water column dips R_WATER*sin(65deg) below its bottom axis point,
    # so the lowest water point sits at lip_z - 0.295. The big-cup hard wall clamp lives below the
    # rim (z <= 1.02), hence lip_z must keep lip_z - 0.295 > rim + margin (default 1.36 -> 1.065).
    parser.add_argument("--lip-z", type=float, default=1.36, help="world z of the mouth's lowest lip point")
    parser.add_argument("--seconds-a", type=float, default=15.0, help="max phase A duration (settle)")
    parser.add_argument("--seconds-b", type=float, default=25.0, help="phase B duration (pour); 0 = no phase B")
    parser.add_argument("--ps", type=float, default=0.008, help="particle_size")
    parser.add_argument("--no-ke-gate", action="store_true", help="always run the full --seconds-a of phase A")
    parser.add_argument("--no-video", action="store_true", help="skip recording (fast metric-only smoke)")
    parser.add_argument("--tag", default="", help="suffix for output file names, e.g. '_smoke'")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--point-size", type=float, default=6.0, help="GL point size in pixels")
    args = parser.parse_args()

    ps = args.ps
    csv_path = os.path.join(VIDEOS_DIR, f"mf10{args.tag}_metrics.csv")
    mp4_path = os.path.join(VIDEOS_DIR, f"mf10_pitcher_pour{args.tag}.mp4")

    # pitcher placement: axis, origin (physical clamp = inner cavity), mesh pose
    O, A, E1, E2 = pitcher_frame(args.tilt, (args.lip_x, 0.0, args.lip_z))
    mesh_pos = O - T_PITCHER_FLOOR * A  # mesh local z=0 (outer bottom) sits T_PITCHER_FLOOR below the cavity floor
    print(f"pitcher: tilt={args.tilt}deg  origin O={np.round(O, 4)}  axis={np.round(A, 4)}")
    print(f"pitcher: mouth center M={np.round(O + L_PITCHER * A, 4)}  mesh_pos={np.round(mesh_pos, 4)}  euler=(0, {-args.tilt}, 0)")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        ipbf_options=gs.options.IPBFOptions(
            particle_size=ps,
            lower_bound=(-0.30, -0.30, 0.0),
            upper_bound=(1.05, 0.45, 1.95),
            boundary_particles=False,
            boundary_cylinder=(0.0, 0.0, R_IN, T_BOTTOM, Z_RIM),  # big cup: wall clamp below rim only
            boundary_pitcher=(*O, *A, R_PITCHER, L_PITCHER),  # pitcher: tilted open-mouth clamp
            ipbf_iterations=2,
            alpha=1e-8,
            viscosity_xsph=0.1,
            damping_enabled=not args.no_damp,
            damping_alpha_star=args.dastar,
            damping_beta=args.dbeta,
            diffusion_coeff=args.epsd,
            surface_tension_enabled=not args.no_st,
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

    # pitcher: visualization only (TiltedCylinderBoundary does the physics)
    pitcher = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=PITCHER_OBJ,
            pos=tuple(mesh_pos),
            euler=(0.0, -args.tilt, 0.0),
            fixed=True,
            decimate=False,
        ),
        material=gs.materials.Rigid(),
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
    # water sampled upright in the big cup's upper half (must sit inside the solver's primary
    # CylinderBoundary for the add-time boundary check; moved into the pitcher frame right after
    # build, before the first step); z>0.55 also avoids any overlap with the coffee column
    water_morph_pos = (0.0, 0.0, 0.55 + 0.5 * H_WATER)
    water = scene.add_entity(
        material=gs.materials.IPBF.Liquid(sampler="regular", rho=1000.0, c_init=0.0),
        morph=gs.morphs.Cylinder(
            radius=R_WATER,
            height=H_WATER,
            pos=water_morph_pos,
        ),
    )

    # wide oblique view: big cup (left) + tilted pitcher (right) + the pour arc between them
    cam = scene.add_camera(res=(960, 1280), pos=(2.15, -2.15, 1.85), lookat=(0.28, 0.0, 0.95), fov=38, GUI=False)
    scene.build()
    print(f"cup geoms: {cup.n_geoms}, pitcher geoms: {pitcher.n_geoms}, "
          f"coffee particles: {coffee.n_particles}, water particles: {water.n_particles}")
    print(
        f"ps={ps} epsd={args.epsd} kst={args.kst:g} dastar={args.dastar:g} dbeta={args.dbeta:g} "
        f"seconds_a={args.seconds_a} seconds_b={args.seconds_b} tag='{args.tag}'"
    )

    if not args.no_video:
        scene.visualizer.rasterizer._camera_targets[cam.uid].point_size = args.point_size

    solver = scene.sim.ipbf_solver
    assert type(solver.boundary).__name__ == "CylinderBoundary", f"unexpected boundary: {type(solver.boundary)}"
    assert type(solver.boundary2).__name__ == "TiltedCylinderBoundary", f"unexpected boundary2: {type(solver.boundary2)}"
    n_fluid = solver._n_fluid_particles
    n_coffee = coffee.n_particles
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    # ---- move the water into the tilted pitcher frame --------------------------------
    pos = water.get_particles_pos().cpu().numpy()  # (n, 3); env dim already stripped (n_envs==0)
    local = pos - np.asarray(water_morph_pos)  # x, y in disc; z in [-H/2, H/2]
    s = local[:, 2] + 0.5 * H_WATER + S0_WATER
    assert s.min() > 0.0 and s.max() < L_PITCHER - 0.5 * ps, f"water fill [{s.min():.3f}, {s.max():.3f}] out of pitcher"
    world = O + local[:, :1] * E1 + local[:, 1:2] * E2 + s[:, None] * A
    water.set_particles_pos(world)
    got = water.get_particles_pos().cpu().numpy()
    rel = got - O
    s_chk = rel @ A
    r_chk = np.linalg.norm(rel - s_chk[:, None] * A, axis=1)
    print(f"water in pitcher: s in [{s_chk.min():.3f}, {s_chk.max():.3f}] (L={L_PITCHER}), "
          f"r in [{r_chk.min():.4f}, {r_chk.max():.4f}] (R={R_PITCHER})")
    assert r_chk.max() < R_PITCHER and s_chk.min() > 0.0 and s_chk.max() < L_PITCHER

    # ---- delayed activation: hide the water until phase B (MF-9 pattern) --------------
    water_active = np.zeros(water.n_particles, dtype=bool)
    water.set_particles_active(water_active)
    got_active = water.get_particles_active()
    assert not got_active.any(), "water should be fully inactive at start"
    print(f"water deactivated: {int((~got_active).sum())}/{water.n_particles} inactive")

    n_frames_a = int(round(args.seconds_a / DT))
    n_frames_b = int(round(args.seconds_b / DT))
    ke_abs_floor = KE_ABS_PER_PARTICLE * n_fluid
    ke_peak = 0.0
    ke_below_since = None
    phase = 0
    t_switch = None

    rows = []
    wall0 = time.time()
    if not args.no_video:
        cam.start_recording(save_to_filename=mp4_path, fps=args.fps)
    for i in range(n_frames_a + n_frames_b + 1):
        if i > 0:
            scene.step()

        # ---- metrics ------------------------------------------------------------------
        pos_all = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
        vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
        c = solver.particles.c.to_numpy()[:n_fluid, 0]
        nan_count = int((~np.isfinite(pos_all)).sum() + (~np.isfinite(vel)).sum() + (~np.isfinite(c)).sum())
        safe = np.nan_to_num(pos_all, nan=0.0, posinf=0.0, neginf=0.0)
        safe_c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        sum_c = float(safe_c.sum())
        std_c = float(safe_c.std())
        zc_c = float((safe_c * safe[:, 2]).sum() / sum_c) if sum_c > 0 else float("nan")
        # big-cup containment is judged on the coffee entity only (the pitcher water lives at r~0.8)
        r_max = float(np.sqrt(safe[:n_coffee, 0] ** 2 + safe[:n_coffee, 1] ** 2).max())
        z_min = float(safe[:, 2].min())
        # water still captive in the pitcher (s < L and r <= R + ps; the tilted axis's infinite
        # cylinder also passes over the big-cup rim, so water splashing near the rim can be
        # miscounted — accepted metric tolerance, the pour-drain trend is what matters)
        rel = safe[n_coffee:] - O
        s_w = rel @ A
        r_w = np.linalg.norm(rel - s_w[:, None] * A, axis=1)
        n_in_pitcher = int(((s_w < L_PITCHER) & (r_w <= R_PITCHER + ps)).sum())
        ke = float(0.5 * np.nansum((vel**2).sum(axis=1)))
        rows.append((i, i * DT, phase, sum_c, std_c, zc_c, r_max, z_min, n_in_pitcher, ke, nan_count))
        if i % 60 == 0:
            wall = time.time() - wall0
            print(
                f"t={i * DT:6.2f}s  ph={phase}  sum_c={sum_c:9.2f}  std_c={std_c:.5f}  zc_c={zc_c:.4f}"
                f"  r_max_c={r_max:.4f}  z_min={z_min:.4f}  n_pitcher={n_in_pitcher}"
                f"  KE={ke:.4e}  nan={nan_count}  ({wall / max(i, 1) * 1000:.0f} ms/frame avg)"
            )

        # ---- phase A -> B switch -------------------------------------------------------
        if phase == 0:
            ke_peak = max(ke_peak, ke)
            if not args.no_ke_gate and i * DT >= KE_GATE_MIN_T and i < n_frames_a:
                ke_thresh = max(KE_REL * ke_peak, ke_abs_floor)
                if ke <= ke_thresh:
                    if ke_below_since is None:
                        ke_below_since = i
                    elif (i - ke_below_since) * DT >= KE_SUSTAIN_S:
                        print(f"KE gate open at t={i * DT:.2f}s (KE={ke:.3e} <= {ke_thresh:.3e}, peak={ke_peak:.3e})")
                        n_frames_a = i
                else:
                    ke_below_since = None
            if i >= n_frames_a and i < n_frames_a + n_frames_b:
                phase = 1
                t_switch = i * DT
                water_active[:] = True
                water.set_particles_active(water_active)
                got_active = water.get_particles_active()
                assert got_active.all(), "water should be fully active in phase B"
                print(f">>> phase B at t={t_switch:.2f}s: water activated ({int(got_active.sum())} particles)")
    if not args.no_video:
        cam.stop_recording()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "phase", "sum_c", "std_c", "zc_c", "r_max_coffee", "z_min", "n_in_pitcher", "ke", "nan_count"])
        w.writerows(rows)

    # ---- pass/fail summary -------------------------------------------------------------
    cols = ["frame", "t", "phase", "sum_c", "std_c", "zc_c", "r_max_coffee", "z_min", "n_in_pitcher", "ke", "nan"]
    arr = {k: np.array([r[j] for r in rows], dtype=float) for j, k in enumerate(cols)}
    sum_c0 = arr["sum_c"][0]
    drift = float(np.max(np.abs(arr["sum_c"] - sum_c0)) / sum_c0)
    p1 = bool(arr["r_max_coffee"].max() <= R_IN + ps + 1e-6)
    p2 = bool(arr["z_min"].min() >= T_BOTTOM - ps - 1e-6)
    p3 = bool(arr["nan"].max() == 0)
    p4 = bool(drift < 0.01)
    has_b = bool((arr["phase"] == 1).any())
    p5 = bool(arr["std_c"][-1] < 0.6 * arr["std_c"][0]) if has_b else None
    n_w = water.n_particles
    print("=" * 70)
    print(f"[1] r_max_coffee max = {arr['r_max_coffee'].max():.4f}  (limit {R_IN + ps:.2f})  -> {'PASS' if p1 else 'FAIL'}")
    print(f"[1] z_min  min = {arr['z_min'].min():.4f}  (limit {T_BOTTOM - ps:.2f})   -> {'PASS' if p2 else 'FAIL'}")
    print(f"[1] nan    max = {int(arr['nan'].max())}                              -> {'PASS' if p3 else 'FAIL'}")
    print(f"[2] sum_c drift = {drift:.3e}  (limit 1e-2, sum_c0={sum_c0:.2f})   -> {'PASS' if p4 else 'FAIL'}")
    print(f"[gate] KE_peak={ke_peak:.4e}  KE_floor={ke_abs_floor:.4e}  t_switch={t_switch}")
    if has_b:
        print(f"[4] pitcher drain: {int(arr['n_in_pitcher'][0])} -> {int(arr['n_in_pitcher'][-1])} / {n_w} particles left")
        print(
            f"[3] std_c {arr['std_c'][0]:.5f} -> {arr['std_c'][-1]:.5f}"
            f"  (target < {0.6 * arr['std_c'][0]:.5f})       -> {'PASS' if p5 else 'FAIL'}"
        )
    print(f"csv: {csv_path}")
    if not args.no_video:
        print(f"mp4: {mp4_path}")


if __name__ == "__main__":
    main()
