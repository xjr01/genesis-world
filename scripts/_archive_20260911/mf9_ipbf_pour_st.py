"""MF-9: IPBF + surface tension + artificial damping in a big cup; settle, then pour an
equal volume of water into the resting coffee (delayed activation of the water entity).

Scene: big open-top cup (multiflow/assets/cup_big.obj, inner R=0.20, inner H=1.00,
cavity floor z=0.02, rim z=1.02) as a semi-transparent rigid mesh (visualization only;
physics = CylinderBoundary clamp at R=0.20, z_bottom=0.02, open top). Inside: a coffee
column (c_init=1.0, brown) r=0.18, z in [0.03, 0.50]. Above the rim: an equal-volume
water column (c_init=0.0, blue) r=0.18, z in [1.10, 1.57], initially DEACTIVATED
(particles_ng.active=False -> invisible, excluded from the hash/neighbor loops).

Phase A (~--seconds-a): coffee only + surface tension + damping, until the kinetic
energy gate opens (KE <= max(--ke-rel * KE_peak, KE_abs_floor) sustained for 1 s,
KE_abs_floor = n_fluid * 0.0025 ~ rms speed 0.05 m/s) or --seconds-a elapses.
Phase B (~--seconds-b): water.set_particles_active(True); it free-falls into the
coffee; concentration diffusion (diffusion_coeff) mixes the phases.

Per-frame CSV: multiflow/videos/mf9{tag}_metrics.csv
  (frame, t, phase, sum_c, std_c, zc_c, r_max, z_min, ke, nan_count)
Video: multiflow/videos/mf9_ipbf_pour_st{tag}.mp4

Run (smoke first!):
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf9_ipbf_pour_st.py --seconds-a 5 --seconds-b 0 --tag _smokeA
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf9_ipbf_pour_st.py --seconds-a 15 --seconds-b 25 --tag ""
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
VIDEOS_DIR = os.path.join(WORKSPACE, "videos")

# cup geometry (assets/gen_cup.py --r-in 0.2 --h-in 1.0): cavity floor z=0.02, rim z=1.02
R_IN = 0.20
T_BOTTOM = 0.02
Z_RIM = T_BOTTOM + 1.0
DT = 1.0 / 60.0
SUBSTEPS = 8

# coffee column inside the cup: r=0.18 (< R_IN - ps), z in [0.03, 0.50] (~half the cup)
R_COFFEE = 0.18
Z_COFFEE0 = T_BOTTOM + 0.01  # 0.03
Z_COFFEE1 = 0.50
# equal-volume water column above the rim, initially inactive; same radius -> same height
R_WATER = 0.18
H_WATER = R_COFFEE**2 * (Z_COFFEE1 - Z_COFFEE0) / R_WATER**2  # 0.47
Z_WATER0 = 1.10

# KE gate for phase A -> B: KE <= max(--ke-rel * KE_peak, KE_abs_floor) sustained for 1 s.
# KE_abs_floor = n_fluid * KE_ABS_PER_PARTICLE; measured on the 15 s settle run (mf9_settle15):
# peak 2.75e4 at t=0.15 s, sustained wall jitter mean ~290-400 with spikes to ~1.3e3
# (same per-particle level as MF-7). Floor 0.006/particle (~576) sits just above the jitter
# band center, so the gate opens ~t=6 s once the transient has died (min phase-A time 5 s).
KE_REL = 0.01
KE_ABS_PER_PARTICLE = 0.006
KE_SUSTAIN_S = 1.0
KE_GATE_MIN_T = 5.0  # never gate before this time (let the transient develop + show the settle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsd", type=float, default=0.01, help="diffusion_coeff (0 = off)")
    parser.add_argument("--kst", type=float, default=1e5, help="st_stiffness (quadratic area weight k~_st; MF-9 scan at kd=0.5: <=1e5 calm, 1e6 boils, 6e6 crashes)")
    parser.add_argument("--dastar", type=float, default=0.32, help="damping_alpha_star")
    parser.add_argument("--dbeta", type=float, default=60.0, help="damping_beta (gate radius in units of support radius)")
    parser.add_argument("--no-st", action="store_true", help="disable surface tension (isolation test)")
    parser.add_argument("--kd", type=float, default=0.5, help="st_distance_stiffness (main-repo ST runs all used 0.5; the 1e3 option default explodes this scene)")
    parser.add_argument("--st-max-neigh", type=int, default=256, help="st_max_surface_neighbors (128 overflows at ps=0.008 when the water activates; 7-bit ring slot only caps st_max_localmesh_neighbors)")
    parser.add_argument("--no-st-dist", action="store_true", help="disable the ST one-sided distance constraint")
    parser.add_argument("--no-damp", action="store_true", help="disable artificial damping (isolation test)")
    parser.add_argument("--seconds-a", type=float, default=15.0, help="max phase A duration (settle)")
    parser.add_argument("--seconds-b", type=float, default=25.0, help="phase B duration (pour); 0 = no phase B")
    parser.add_argument("--ps", type=float, default=0.01, help="particle_size")
    parser.add_argument("--no-ke-gate", action="store_true", help="always run the full --seconds-a of phase A")
    parser.add_argument("--no-video", action="store_true", help="skip recording (fast metric-only smoke)")
    parser.add_argument("--tag", default="", help="suffix for output file names, e.g. '_smokeA'")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--point-size", type=float, default=6.0, help="GL point size in pixels")
    args = parser.parse_args()

    ps = args.ps
    csv_path = os.path.join(VIDEOS_DIR, f"mf9{args.tag}_metrics.csv")
    mp4_path = os.path.join(VIDEOS_DIR, f"mf9_ipbf_pour_st{args.tag}.mp4")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        ipbf_options=gs.options.IPBFOptions(
            particle_size=ps,
            lower_bound=(-0.26, -0.26, 0.0),
            upper_bound=(0.26, 0.26, 1.65),
            boundary_particles=False,
            boundary_cylinder=(0.0, 0.0, R_IN, T_BOTTOM),  # open-top cylinder clamp
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

    # cup: visualization only (no rigid<->IPBF coupling; CylinderBoundary does the physics)
    cup = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=CUP_OBJ,
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            decimate=False,  # keep the exact 96-segment cylinder
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.8, 0.8, 0.85, 0.35), opacity=0.35),
    )

    coffee = scene.add_entity(
        material=gs.materials.IPBF.Liquid(sampler="regular", rho=1000.0, c_init=1.0),
        morph=gs.morphs.Cylinder(
            radius=R_COFFEE,
            height=Z_COFFEE1 - Z_COFFEE0,
            pos=(0.0, 0.0, 0.5 * (Z_COFFEE0 + Z_COFFEE1)),
        ),
    )
    water = scene.add_entity(
        material=gs.materials.IPBF.Liquid(sampler="regular", rho=1000.0, c_init=0.0),
        morph=gs.morphs.Cylinder(
            radius=R_WATER,
            height=H_WATER,
            pos=(0.0, 0.0, Z_WATER0 + H_WATER / 2.0),
        ),
    )

    # oblique view: free surface, cup side, and the (initially hidden) water column above the rim
    cam = scene.add_camera(res=(960, 1280), pos=(1.62, -1.62, 1.58), lookat=(0.0, 0.0, 0.78), fov=36, GUI=False)
    scene.build()
    print(f"cup geoms: {cup.n_geoms}, coffee particles: {coffee.n_particles}, water particles: {water.n_particles}")
    print(
        f"ps={ps} epsd={args.epsd} kst={args.kst:g} dastar={args.dastar:g} dbeta={args.dbeta:g} "
        f"seconds_a={args.seconds_a} seconds_b={args.seconds_b} tag='{args.tag}'"
    )

    if not args.no_video:
        scene.visualizer.rasterizer._camera_targets[cam.uid].point_size = args.point_size

    solver = scene.sim.ipbf_solver
    assert type(solver.boundary).__name__ == "CylinderBoundary", f"unexpected boundary: {type(solver.boundary)}"
    n_fluid = solver._n_fluid_particles
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    # ---- delayed activation: hide the water until phase B --------------------------
    water_active = np.zeros(water.n_particles, dtype=bool)
    water.set_particles_active(water_active)
    got = water.get_particles_active()
    assert not got.any(), "water should be fully inactive at start"
    print(f"water deactivated: {int((~got).sum())}/{water.n_particles} inactive")

    n_frames_a = int(round(args.seconds_a / DT))
    n_frames_b = int(round(args.seconds_b / DT))
    ke_abs_floor = KE_ABS_PER_PARTICLE * n_fluid
    ke_peak = 0.0
    ke_below_since = None  # frame index since which KE stayed under the gate
    phase = 0
    t_switch = None

    rows = []
    wall0 = time.time()
    if not args.no_video:
        cam.start_recording(save_to_filename=mp4_path, fps=args.fps)
    for i in range(n_frames_a + n_frames_b + 1):
        if i > 0:
            scene.step()

        # ---- metrics over all fluid particles (inactive water included: c=0, vel=0,
        # pos above the rim; it does not violate the r_max / z_min limits) ----
        pos = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
        vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
        c = solver.particles.c.to_numpy()[:n_fluid, 0]
        nan_count = int((~np.isfinite(pos)).sum() + (~np.isfinite(vel)).sum() + (~np.isfinite(c)).sum())
        safe = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        safe_c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        sum_c = float(safe_c.sum())
        std_c = float(safe_c.std())
        zc_c = float((safe_c * safe[:, 2]).sum() / sum_c) if sum_c > 0 else float("nan")
        r_max = float(np.sqrt(safe[:, 0] ** 2 + safe[:, 1] ** 2).max())
        z_min = float(safe[:, 2].min())
        ke = float(0.5 * np.nansum((vel**2).sum(axis=1)))
        rows.append((i, i * DT, phase, sum_c, std_c, zc_c, r_max, z_min, ke, nan_count))
        if i % 60 == 0:
            wall = time.time() - wall0
            print(
                f"t={i * DT:6.2f}s  ph={phase}  sum_c={sum_c:9.2f}  std_c={std_c:.5f}  zc_c={zc_c:.4f}"
                f"  r_max={r_max:.4f}  z_min={z_min:.4f}  KE={ke:.4e}  nan={nan_count}"
                f"  ({wall / max(i, 1) * 1000:.0f} ms/frame avg)"
            )

        # ---- phase A -> B switch ---------------------------------------------------
        if phase == 0:
            ke_peak = max(ke_peak, ke)
            if not args.no_ke_gate and i * DT >= KE_GATE_MIN_T and i < n_frames_a:
                ke_thresh = max(KE_REL * ke_peak, ke_abs_floor)
                if ke <= ke_thresh:
                    if ke_below_since is None:
                        ke_below_since = i
                    elif (i - ke_below_since) * DT >= KE_SUSTAIN_S:
                        print(f"KE gate open at t={i * DT:.2f}s (KE={ke:.3e} <= {ke_thresh:.3e}, peak={ke_peak:.3e})")
                        n_frames_a = i  # truncate phase A at this frame; activation happens below
                else:
                    ke_below_since = None
            if i >= n_frames_a and i < n_frames_a + n_frames_b:
                phase = 1
                t_switch = i * DT
                water_active[:] = True
                water.set_particles_active(water_active)
                got = water.get_particles_active()
                assert got.all(), "water should be fully active in phase B"
                print(f">>> phase B at t={t_switch:.2f}s: water activated ({int(got.sum())} particles)")
    if not args.no_video:
        cam.stop_recording()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "phase", "sum_c", "std_c", "zc_c", "r_max", "z_min", "ke", "nan_count"])
        w.writerows(rows)

    # ---- pass/fail summary ---------------------------------------------------------
    cols = ["frame", "t", "phase", "sum_c", "std_c", "zc_c", "r_max", "z_min", "ke", "nan"]
    arr = {k: np.array([r[j] for r in rows], dtype=float) for j, k in enumerate(cols)}
    sum_c0 = arr["sum_c"][0]
    drift = float(np.max(np.abs(arr["sum_c"] - sum_c0)) / sum_c0)
    p1 = bool(arr["r_max"].max() <= R_IN + ps + 1e-6)
    p2 = bool(arr["z_min"].min() >= T_BOTTOM - ps - 1e-6)
    p3 = bool(arr["nan"].max() == 0)
    p4 = bool(drift < 0.01)
    has_b = bool((arr["phase"] == 1).any())
    p5 = bool(arr["std_c"][-1] < 0.6 * arr["std_c"][0]) if has_b else None
    print("=" * 70)
    print(f"[1] r_max  max = {arr['r_max'].max():.4f}  (limit {R_IN + ps:.2f})     -> {'PASS' if p1 else 'FAIL'}")
    print(f"[1] z_min  min = {arr['z_min'].min():.4f}  (limit {T_BOTTOM - ps:.2f})   -> {'PASS' if p2 else 'FAIL'}")
    print(f"[1] nan    max = {int(arr['nan'].max())}                              -> {'PASS' if p3 else 'FAIL'}")
    print(f"[2] sum_c drift = {drift:.3e}  (limit 1e-2, sum_c0={sum_c0:.2f})   -> {'PASS' if p4 else 'FAIL'}")
    print(f"[gate] KE_peak={ke_peak:.4e}  KE_floor={ke_abs_floor:.4e}  t_switch={t_switch}")
    if has_b:
        print(
            f"[3] std_c {arr['std_c'][0]:.5f} -> {arr['std_c'][-1]:.5f}"
            f"  (target < {0.6 * arr['std_c'][0]:.5f})       -> {'PASS' if p5 else 'FAIL'}"
        )
    print(f"csv: {csv_path}")
    if not args.no_video:
        print(f"mp4: {mp4_path}")


if __name__ == "__main__":
    main()
