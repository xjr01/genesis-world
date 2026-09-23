"""MF-14 unit: TiltedCylinderShellBoundary cell-level validation (multiflow P2, design doc
multiflow/P2_SHELL_DESIGN.md section 6.1).

U1 (static shell trickle): pitcher shell held at a fixed 20 deg tilt over a table plane; milk
blobs (~500 particles each) are dropped from 2 cm above the pour-side lip. Criteria:
  [1] n_leak = 0 (no particle deeper than 0.1*ps inside the shell solid)
  [2] within 1.5s of first wall contact n_outer_wall >= 200, sustained >= 30 frames
  [3] adhering particles |C| p95 <= 1.2 * particle_radius
  [4] established-film tangential velocity mean |v_t| in [0.05, 1.0] m/s, >= 80% downward
  [5] after the film passes the bottom edge n_outer_wall decays and detached particles are
      ballistic (|(dv/dt) - g| / g <= 20%)
U2 (cavity zero-regression): upright pitcher filled with water, 2s settle + 5s rest at
dt=1/240; run once with --shell and once without; n_in_pitcher / KE / leak curves must match
within 1%.
U3 (lip rounding ablation): run U1 with --lip-round 0 and --lip-round 0.008 and compare the
dv_adh_p95 column (per-frame p95 velocity jump of adhering particles) during the lip transit.

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf14_shell_unit.py --mode u1 --seconds 6 --tag _u1
  "$PY" multiflow/scripts/mf14_shell_unit.py --mode u2 --shell --tag _u2shell --no-video
  "$PY" multiflow/scripts/mf14_shell_unit.py --mode u2 --tag _u2legacy --no-video
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

PITCHER_OBJ = os.path.join(WORKSPACE, "assets", "pitcher45.obj")
VIDEOS_DIR = os.path.join(WORKSPACE, "videos")

# pitcher shell geometry (same as assets/gen_cup.py --r-in 0.16 --h-in 0.45 --t-wall 0.016)
R_IN = 0.16
L = 0.45
T_WALL = 0.016
T_BOTTOM = 0.016
R_OUT = R_IN + T_WALL
PS = 0.008
R_P = 0.5 * PS  # particle radius = adhesion shell half-width


def quat_roty_neg(theta):
    """Quaternion (w, x, y, z) for rotY(-theta): maps mesh local +z to the pitcher axis."""
    return np.array([np.cos(theta / 2.0), 0.0, -np.sin(theta / 2.0), 0.0])


def shell_pose_u1():
    th = np.deg2rad(20.0)
    O = np.array([0.0, 0.0, 0.8])
    a = np.array([-np.sin(th), 0.0, np.cos(th)])
    return O, a, th


def build_scene(args, mode):
    gs.init(backend=gs.gpu, precision="32", seed=0)
    if mode == "u1":
        O0, A0, _ = shell_pose_u1()
        dt, substeps = 1.0 / 60.0, 8
        lower, upper = (-1.0, -1.0, 0.0), (1.0, 1.0, 4.0)
    else:  # u2: upright pitcher, fine dt against settle tunneling (mf12 presettle lesson)
        O0 = np.array([0.0, 0.0, 0.05])
        A0 = np.array([0.0, 0.0, 1.0])
        dt, substeps = 1.0 / 240.0, 8
        lower, upper = (-0.6, -0.6, 0.0), (0.6, 0.6, 1.2)

    shell_tuple = (*O0, *A0, R_IN, L, T_WALL, T_BOTTOM, args.lip_round)
    legacy_tuple = (*O0, *A0, R_IN, L)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, gravity=(0.0, 0.0, -9.81)),
        pbd_options=gs.options.PBDOptions(
            particle_size=PS,
            lower_bound=lower,
            upper_bound=upper,
            boundary_pitcher_shell=shell_tuple if (mode == "u1" or args.shell) else None,
            boundary_pitcher=None if (mode == "u1" or args.shell) else legacy_tuple,
            boundary_plane=(0.0,) if mode == "u1" else None,
            max_density_solver_iterations=args.dens_iters,
            max_viscosity_solver_iterations=1,
            density_lambda_epsilon=args.leps,
            surface_tension_enabled=True,
            st_compliance=args.st_comp,
            st_surface_density_factor=1.0,
            st_distance_enabled=True,
            st_distance_compliance=40.0,
            st_max_surface_neighbors=256,
            velocity_damping=args.vdamp,
            wall_adhesion_enabled=True,
            wall_adhesion_compliance=args.adh_comp,
            wall_friction=args.wfric,
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )

    # pitcher: visualization only (the analytic boundary does the physics)
    pitcher = scene.add_entity(
        morph=gs.morphs.Mesh(file=PITCHER_OBJ, pos=tuple(O0 - T_BOTTOM * A0), fixed=True, decimate=False),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.85, 0.75, 0.65, 0.5), opacity=0.5),
    )

    milk_parts = []
    if mode == "u1":
        # milk blobs dropped onto the pour-side outer lip: fall heights 2cm / 15cm / 35cm give
        # three distinct impacts within ~0.3s (kept low so the per-substep impact travel stays
        # below half the wall thickness — faster impacts would tunnel by construction)
        e1 = np.array([np.cos(np.deg2rad(20.0)), 0.0, np.sin(np.deg2rad(20.0))])
        p_lip = O0 + L * A0 + R_OUT * e1  # pour-side lip OUTER edge
        r_blob = 0.037  # ~450 particles per blob
        for k, h in enumerate((0.02 + r_blob, 0.15 + r_blob, 0.35 + r_blob)):
            c = p_lip + e1 * (0.25 * T_WALL) + np.array([0.0, 0.0, h])
            milk_parts.append(
                scene.add_entity(
                    material=gs.materials.PBD.Liquid(
                        sampler="regular", rho=1000.0, c_init=0.0, boundary_group=1,
                        density_relaxation=args.dens_relax, viscosity_relaxation=0.01,
                    ),
                    morph=gs.morphs.Sphere(radius=r_blob, pos=tuple(c)),
                )
            )
    else:
        milk_parts.append(
            scene.add_entity(
                material=gs.materials.PBD.Liquid(
                    sampler="regular", rho=1000.0, c_init=0.0, boundary_group=1,
                    density_relaxation=args.dens_relax, viscosity_relaxation=0.01,
                ),
                morph=gs.morphs.Cylinder(radius=R_IN - 0.008, height=0.30, pos=(0.0, 0.0, 0.05 + 0.16)),
            )
        )

    cam = None
    if mode == "u1" and not args.no_video:
        e1 = np.array([np.cos(np.deg2rad(20.0)), 0.0, np.sin(np.deg2rad(20.0))])
        p_lip = O0 + L * A0 + R_OUT * e1
        cam = scene.add_camera(
            res=(800, 800),
            pos=tuple(p_lip + np.array([0.55, -0.75, 0.30])),
            lookat=tuple(p_lip + np.array([0.0, 0.0, -0.12])),
            fov=35,
            GUI=False,
        )
    scene.build()
    return scene, pitcher, milk_parts, cam, O0, A0, dt


def frame_metrics(pos, vel, prev_vel, prev_sides, O, A, lip_round, dt):
    """numpy-side shell metrics for one frame (milk particles only)."""
    d, r, s = shell_sdf_np(pos, O, A, R_IN, L, T_WALL, T_BOTTOM, lip_round)
    # n_leak = mid-wall side flips since the previous frame (a real tunnel cannot be seen as
    # d < 0 at frame boundaries: the per-substep impose sanitizes solid penetration)
    inside_cav, outside_wall = side_masks(pos, O, A, R_IN, L, T_WALL, T_BOTTOM, R_P, lip_round)
    n_leak = 0
    if prev_sides is not None:
        prev_in, prev_out = prev_sides
        n_leak = int(((prev_in & outside_wall) | (prev_out & inside_cav)).sum())
    sides = (inside_cav, outside_wall)
    adhering = np.abs(d) <= R_P
    n_adh = int(adhering.sum())
    outer = adhering & ((r > R_IN + 0.5 * T_WALL) | (s > L) | (s < -0.5 * T_BOTTOM))
    n_outer = int(outer.sum())
    c_p95 = float(np.percentile(np.abs(d[adhering]), 95)) if n_adh > 0 else 0.0
    # approximate per-particle surface normal for the film velocity split
    vt_mean = 0.0
    vt_down = 0.0
    if outer.any():
        e_r = np.zeros_like(pos)
        rel = pos - O[None, :]
        ss = rel @ A
        radial = rel - ss[:, None] * A[None, :]
        rn = np.linalg.norm(radial, axis=1, keepdims=True)
        e_r = radial / np.maximum(rn, 1e-12)
        n_vec = e_r.copy()
        underside = s < -0.5 * T_BOTTOM
        lip_top = s > L
        n_vec[underside] = -A[None, :]
        n_vec[lip_top] = A[None, :]
        v = vel[outer]
        nv = n_vec[outer]
        vt = v - (v * nv).sum(axis=1, keepdims=True) * nv
        vt_norm = np.linalg.norm(vt, axis=1)
        vt_mean = float(vt_norm.mean())
        vt_down = float((vt[:, 2] < 0.0).mean())
    # U3 metric: p95 per-frame velocity jump of adhering particles (lip-crossing smoothness)
    dv_p95 = 0.0
    if prev_vel is not None and n_adh > 0:
        dv_p95 = float(np.percentile(np.linalg.norm(vel[adhering] - prev_vel[adhering], axis=1), 95))
    # detached = below the cup, no longer adhering, still well above the table (the table
    # clamp would corrupt the ballistic check): ballistic check against g
    detached = (s < -T_BOTTOM - 2.0 * R_P) & (~adhering) & (pos[:, 2] > 0.1)
    n_det = int(detached.sum())
    bal_err = float("nan")
    if prev_vel is not None and n_det >= 20:
        dv = (vel[detached] - prev_vel[detached]) / dt
        g_vec = np.array([0.0, 0.0, -9.81])
        bal_err = float(np.linalg.norm(dv - g_vec[None, :], axis=1).mean() / 9.81)
    return sides, dict(
        n_leak=n_leak, n_adh=n_adh, n_outer=n_outer, c_p95=c_p95,
        vt_mean=vt_mean, vt_down=vt_down, n_det=n_det, bal_err=bal_err, dv_p95=dv_p95,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["u1", "u2"], required=True)
    parser.add_argument("--shell", action="store_true", help="u2: use the shell boundary (default: legacy)")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--lip-round", type=float, default=0.008)
    parser.add_argument("--adh-comp", type=float, default=10.0)
    parser.add_argument("--wfric", type=float, default=0.1)
    parser.add_argument("--st-comp", type=float, default=1.0)
    parser.add_argument("--leps", type=float, default=0.1)
    parser.add_argument("--dens-iters", type=int, default=20)
    parser.add_argument("--dens-relax", type=float, default=0.2)
    parser.add_argument("--vdamp", type=float, default=0.98)
    parser.add_argument("--settle", type=float, default=2.0, help="u2: pre-roll settle seconds")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    seconds = args.seconds if args.seconds is not None else (6.0 if args.mode == "u1" else 5.0)
    mp4_path = os.path.join(VIDEOS_DIR, f"mf14_shell_unit{args.tag}.mp4")
    csv_path = os.path.join(VIDEOS_DIR, f"mf14_shell_unit{args.tag}_metrics.csv")
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    scene, pitcher, milk_parts, cam, O0, A0, DT = build_scene(args, args.mode)
    solver = scene.sim.pbd_solver
    expect = "TiltedCylinderShellBoundary" if (args.mode == "u1" or args.shell) else "TiltedCylinderBoundary"
    assert type(solver.boundary2).__name__ == expect, f"unexpected boundary2: {type(solver.boundary2)}"
    n_fluid = solver._n_fluid_particles
    print(f"mode={args.mode} shell={args.mode == 'u1' or args.shell} lip_round={args.lip_round} "
          f"adh_comp={args.adh_comp} wfric={args.wfric} st_comp={args.st_comp} vdamp={args.vdamp} "
          f"n_fluid={n_fluid} dt={DT}")

    # pose the visual mesh once (static scene)
    th = np.deg2rad(20.0) if args.mode == "u1" else 0.0
    pitcher.set_pos(O0 - T_BOTTOM * A0, relative=True, zero_velocity=True)
    pitcher.set_quat(quat_roty_neg(th), relative=True, zero_velocity=True)

    rows = []
    wall0 = time.time()
    prev_vel = None
    prev_sides = None

    def record(i, t):
        nonlocal prev_vel, prev_sides
        pos = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
        vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
        nan_count = int((~np.isfinite(pos)).sum() + (~np.isfinite(vel)).sum())
        safe = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        ke = float(0.5 * np.nansum((vel**2).sum(axis=1)))
        sides, m = frame_metrics(safe, np.nan_to_num(vel), prev_vel, prev_sides, O0, A0, args.lip_round, DT)
        # captivity (same convention as mf12 n_in_pitcher)
        rel = safe - O0
        s_w = rel @ A0
        r_w = np.linalg.norm(rel - s_w[:, None] * A0[None, :], axis=1)
        n_in = int(((s_w < L + PS) & (s_w > -PS) & (r_w <= R_IN + PS)).sum())
        prev_vel = np.nan_to_num(vel).copy()
        prev_sides = sides
        rows.append((i, t, m["n_leak"], m["n_adh"], m["n_outer"], m["c_p95"], m["vt_mean"],
                     m["vt_down"], m["n_det"], m["bal_err"], m["dv_p95"], n_in, ke, nan_count))
        if i % 60 == 0:
            print(f"t={t:6.3f}  n_leak={m['n_leak']}  n_adh={m['n_adh']}  n_outer={m['n_outer']}  "
                  f"c95={m['c_p95']:.5f}  vt={m['vt_mean']:.3f}  det={m['n_det']}  n_in={n_in}  "
                  f"KE={ke:.3e}  nan={nan_count}  ({(time.time() - wall0) / max(i, 1) * 1000:.0f} ms/frame)")

    if args.mode == "u2":
        # settle pre-roll (upright, not recorded)
        n_settle = int(round(args.settle / DT))
        for i in range(n_settle):
            solver.set_pitcher_pose(O0, A0)
            scene.step()
        prev_vel = None
        prev_sides = None
        wall0 = time.time()

    n_frames = int(round(seconds / DT))
    if cam is not None and not args.no_video:
        cam.start_recording(save_to_filename=mp4_path, fps=args.fps)
    for i in range(n_frames + 1):
        solver.set_pitcher_pose(O0, A0)
        if i > 0:
            scene.step()
        record(i, i * DT)
    if cam is not None and not args.no_video:
        cam.stop_recording()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "n_leak", "n_adh", "n_outer_wall", "c_p95_adh", "film_vt_mean",
                    "film_vt_down", "n_detached", "bal_err", "dv_adh_p95", "n_in_pitcher", "ke", "nan"])
        w.writerows(rows)

    cols = ["frame", "t", "n_leak", "n_adh", "n_outer", "c_p95", "vt_mean", "vt_down", "n_det",
            "bal_err", "dv_p95", "n_in", "ke", "nan"]
    arr = {k: np.array([r_[j] for r_ in rows], dtype=float) for j, k in enumerate(cols)}
    print("=" * 70)
    if args.mode == "u1":
        t_contact = arr["t"][np.nonzero(arr["n_outer"] > 0)[0][0]] if (arr["n_outer"] > 0).any() else float("inf")
        win = (arr["t"] >= t_contact) & (arr["t"] <= t_contact + 1.5)
        sustained = 0
        best_sustained = 0
        for v in arr["n_outer"][win] if win.any() else []:
            sustained = sustained + 1 if v >= 200 else 0
            best_sustained = max(best_sustained, sustained)
        p1 = bool(arr["n_leak"].max() == 0)
        p2 = bool(best_sustained >= 30)
        film = arr["n_outer"] >= 50
        p3 = bool(np.nanmax(arr["c_p95"][arr["n_adh"] > 50]) <= 1.2 * R_P) if (arr["n_adh"] > 50).any() else False
        p4 = bool(film.any() and 0.05 <= np.nanmean(arr["vt_mean"][film]) <= 1.0 and np.nanmean(arr["vt_down"][film]) >= 0.8)
        post = arr["t"] > t_contact + 1.0
        decays = bool(post.any() and arr["n_outer"][post].max() > 0 and arr["n_outer"][-1] < 0.5 * arr["n_outer"][post].max())
        bal = arr["bal_err"][np.isfinite(arr["bal_err"])]
        p5 = bool(decays and (len(bal) == 0 or np.nanmean(bal) <= 0.2))
        print(f"[1] n_leak max = {int(arr['n_leak'].max())}                          -> {'PASS' if p1 else 'FAIL'}")
        print(f"[2] n_outer_wall >= 200 sustained frames = {best_sustained} (need 30) -> {'PASS' if p2 else 'FAIL'}")
        print(f"[3] adhering |C| p95 max = {np.nanmax(arr['c_p95'][arr['n_adh'] > 50]) if (arr['n_adh'] > 50).any() else float('nan'):.5f} (limit {1.2 * R_P:.5f}) -> {'PASS' if p3 else 'FAIL'}")
        if film.any():
            print(f"[4] film |v_t| mean = {np.nanmean(arr['vt_mean'][film]):.3f} m/s, down frac = {np.nanmean(arr['vt_down'][film]):.2f} -> {'PASS' if p4 else 'FAIL'}")
        print(f"[5] film detaches ballistically (decay={decays}, bal_err mean = {np.nanmean(bal) if len(bal) else float('nan'):.3f}) -> {'PASS' if p5 else 'FAIL'}")
    else:
        print(f"u2 done: n_in {int(arr['n_in'][0])} -> {int(arr['n_in'][-1])}, KE final {arr['ke'][-1]:.4e}, "
              f"n_leak max {int(arr['n_leak'].max())}, nan max {int(arr['nan'].max())}")
        print("compare shell vs legacy CSVs: n_in / KE / n_leak curves must match within 1%")
    print(f"csv: {csv_path}")
    if cam is not None and not args.no_video:
        print(f"mp4: {mp4_path}")


if __name__ == "__main__":
    main()
