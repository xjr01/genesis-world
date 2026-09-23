"""MF-13: droplet-on-table wall-adhesion validation (PBF + PBSTF-style ST + PlaneBoundary).

A single milk droplet (~2-3k particles) falls from above onto a horizontal table plane
(PlaneBoundary, PBDOptions.boundary_plane) and either spreads and sticks (hydrophilic,
--adh --adh-comp 20) or beads up (hydrophobic, adhesion off). Standalone scene: no cups —
the analytic table plane is the only adhesion/clamp boundary besides the domain box.

Physics chain under test (multiflow P1):
  - PlaneBoundary.impose_pos inside the density Jacobi apply pass (B1 fix, PBSTF
    _kernel_apply_position_delta isomorph) + per-substep impose_pos_vel fallback
  - PlaneBoundary.adhesion_query in the nearest-surface multi-boundary adhesion dispatch
  - ST area/distance constraints + adhesion competing inside the same Jacobi iteration

Per-frame CSV: multiflow/videos/mf13{tag}_metrics.csv
  (frame, t, spread_r, z_p95, z_min, n_on_table, ke, nan, n_adh)
Video: multiflow/videos/mf13_droplet{tag}.mp4

Run (both groups):
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf13_droplet_table.py --adh --tag _philic --seconds 10
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf13_droplet_table.py --tag _phobic --seconds 10
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

VIDEOS_DIR = os.path.join(WORKSPACE, "videos")

# scene geometry
Z_TABLE = 0.02  # table top plane (matches the MF-12 cup floor height)
R_DROP = 0.07  # droplet radius (~2.8k particles at ps=0.008 regular sampling)
Z_DROP = 0.10  # droplet center height: bottom sits ~1 particle above the table — a near-sessile
               # touchdown (a real fall splatters: this soft PBF has ~no viscosity, so impact
               # kinetic energy pancakes the drop into a monolayer regardless of adhesion)
DT = 1.0 / 60.0
SUBSTEPS = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--hold", type=float, default=1.5,
                        help="kinematic pin seconds: dissipate the t=0 lattice flash in air before release "
                             "(velocity damped x0.3 + centroid re-pinned every frame; MF-12 presettle analog)")
    parser.add_argument("--adh", action="store_true", help="wall adhesion on the table (hydrophilic)")
    parser.add_argument("--adh-comp", type=float, default=20.0, help="wall_adhesion_compliance (smaller = stickier)")
    parser.add_argument("--wfric", type=float, default=0.0, help="wall_friction (0 = off)")
    parser.add_argument("--dens-iters", type=int, default=20, help="max_density_solver_iterations")
    parser.add_argument("--leps", type=float, default=0.1, help="density_lambda_epsilon (stock 100)")
    parser.add_argument("--st-comp", type=float, default=1.0,
                        help="st_compliance (smaller = stronger ST; 0.5 fights adhesion into KE~1 jitter, "
                             "2.0 too weak to hold the drop — 1.0 is the calm spot, MF-13 smoke tuning)")
    parser.add_argument("--st-dist-comp", type=float, default=40.0, help="st_distance_compliance")
    parser.add_argument("--visc-relax", type=float, default=0.05, help="material viscosity_relaxation")
    parser.add_argument("--vdamp", type=float, default=0.98,
                        help="velocity_damping per substep (1.0 = off; the ST-ringing droplet boils at "
                             "v_rms~0.2 undamped and pancakes on contact regardless of adhesion)")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--st-max-neigh", type=int, default=256, help="st_max_surface_neighbors")
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    ps = 0.008
    mp4_path = os.path.join(VIDEOS_DIR, f"mf13_droplet{args.tag}.mp4")
    csv_path = os.path.join(VIDEOS_DIR, f"mf13{args.tag}_metrics.csv")

    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
        pbd_options=gs.options.PBDOptions(
            particle_size=ps,
            lower_bound=(-0.40, -0.40, 0.0),
            upper_bound=(0.40, 0.40, 0.80),
            boundary_plane=(Z_TABLE,),  # infinite table plane (MF-13)
            max_density_solver_iterations=args.dens_iters,
            max_viscosity_solver_iterations=1,
            density_lambda_epsilon=args.leps,
            surface_tension_enabled=True,
            st_compliance=args.st_comp,
            st_surface_density_factor=1.0,
            st_distance_enabled=True,
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

    # table top: visualization only (PlaneBoundary does the physics)
    scene.add_entity(
        morph=gs.morphs.Box(size=(0.60, 0.60, Z_TABLE), pos=(0.0, 0.0, 0.5 * Z_TABLE), fixed=True),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.55, 0.45, 0.35, 1.0)),
    )

    # milk droplet (white: c_init=0.0), released at t=0 above the table
    milk = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=0.0,
            density_relaxation=0.2, viscosity_relaxation=args.visc_relax,
        ),
        morph=gs.morphs.Sphere(radius=R_DROP, pos=(0.0, 0.0, Z_DROP)),
    )

    cam = scene.add_camera(res=(960, 960), pos=(0.52, -0.52, 0.34), lookat=(0.0, 0.0, 0.05), fov=35, GUI=False)
    scene.build()
    print(f"milk particles: {milk.n_particles}")
    print(
        f"ps={ps} dens_iters={args.dens_iters} leps={args.leps:g} st_comp={args.st_comp:g} "
        f"st_dist_comp={args.st_dist_comp:g} adh={args.adh}/{args.adh_comp:g} wfric={args.wfric:g} "
        f"seconds={args.seconds} tag='{args.tag}'"
    )

    if not args.no_video:
        scene.visualizer.rasterizer._camera_targets[cam.uid].point_size = args.point_size

    solver = scene.sim.pbd_solver
    assert type(solver.boundary).__name__ == "CubeBoundary", f"unexpected boundary: {type(solver.boundary)}"
    assert type(solver.boundary3).__name__ == "PlaneBoundary", f"unexpected boundary3: {type(solver.boundary3)}"
    n_fluid = solver._n_fluid_particles
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    n_frames = int(round(args.seconds / DT))
    rows = []
    wall0 = time.time()
    if not args.no_video:
        cam.start_recording(save_to_filename=mp4_path, fps=args.fps)

    for i in range(n_frames + 1):
        t = i * DT
        if i > 0:
            scene.step()
            if t < args.hold:
                # kinematic hold: the PBF lattice samples ~2x over-dense and flash-expands at
                # t=0 (MF-12 finding) — in free air that blast shatters the droplet. Hold the
                # centroid in place (z re-pinned) and heavily damp velocities so the flash can
                # expand-and-relax WITHOUT storing pressure, then release. Not physical, only a
                # presettle device; metrics still recorded.
                pos_np = solver.particles.pos.to_numpy()
                pos_np[:n_fluid, 0, 2] += Z_DROP - pos_np[:n_fluid, 0, 2].mean()
                solver.particles.pos.from_numpy(pos_np)
                vel_np = solver.particles.vel.to_numpy()
                vel_np *= 0.3
                solver.particles.vel.from_numpy(vel_np)

        # ---- metrics ----------------------------------------------------------------------
        pos_all = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
        vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
        nan_count = int((~np.isfinite(pos_all)).sum() + (~np.isfinite(vel)).sum())
        safe = np.nan_to_num(pos_all, nan=0.0, posinf=0.0, neginf=0.0)
        cx, cy = safe[:, 0].mean(), safe[:, 1].mean()
        r_hor = np.sqrt((safe[:, 0] - cx) ** 2 + (safe[:, 1] - cy) ** 2)
        spread_r = float(np.percentile(r_hor, 95))  # p95 radial spread about the centroid
        z_p95 = float(np.percentile(safe[:, 2], 95))  # contact height proxy
        z_min = float(safe[:, 2].min())
        n_on_table = int((safe[:, 2] < Z_TABLE + 2.0 * ps).sum())
        ke = float(0.5 * np.nansum((vel**2).sum(axis=1)))
        n_adh = solver.wall_adhesion_stats()
        rows.append((i, t, spread_r, z_p95, z_min, n_on_table, ke, nan_count, n_adh))
        if i % 60 == 0:
            wall = time.time() - wall0
            print(
                f"t={t:6.2f}s  spread_r={spread_r:.4f}  z_p95={z_p95:.4f}  z_min={z_min:.4f}"
                f"  n_table={n_on_table}  KE={ke:.4e}  nan={nan_count}  n_adh={n_adh}"
                f"  ({wall / max(i, 1) * 1000:.0f} ms/frame avg)"
            )

    if not args.no_video:
        cam.stop_recording()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "spread_r", "z_p95", "z_min", "n_on_table", "ke", "nan", "n_adh"])
        w.writerows(rows)

    # ---- pass/fail summary ------------------------------------------------------------------
    cols = ["frame", "t", "spread_r", "z_p95", "z_min", "n_on_table", "ke", "nan", "n_adh"]
    arr = {k: np.array([r_[j] for r_ in rows], dtype=float) for j, k in enumerate(cols)}
    settle = arr["t"] >= args.seconds - 2.0  # last 2s = settled phase
    p1 = bool(arr["nan"].max() == 0)
    p2 = bool(arr["z_min"].min() >= Z_TABLE - ps - 1e-6)  # no table penetration
    p3 = bool(arr["ke"][settle].max() < 0.05)  # settled: no limit cycle / residual jitter
    p4 = bool(arr["spread_r"].max() < 0.35)  # contained on the table (no explosion/escape)
    p5 = bool(arr["n_on_table"][settle].min() >= 100)  # the drop rests ON the table (a bead
    # keeps most mass ABOVE 2*ps, so this is only a contact check, not a flatness check)
    print("=" * 70)
    print(f"[1] nan      max = {int(arr['nan'].max())}                              -> {'PASS' if p1 else 'FAIL'}")
    print(f"[1] z_min    min = {arr['z_min'].min():.4f}  (table {Z_TABLE}, limit {Z_TABLE - ps:.3f}) -> {'PASS' if p2 else 'FAIL'}")
    print(f"[2] KE settled max = {arr['ke'][settle].max():.4e}  (limit 0.05)     -> {'PASS' if p3 else 'FAIL'}")
    print(f"[2] spread_r max = {arr['spread_r'].max():.4f}  (limit 0.35)          -> {'PASS' if p4 else 'FAIL'}")
    print(f"[3] n_on_table settled min = {int(arr['n_on_table'][settle].min())} / {n_fluid} (>=100) -> {'PASS' if p5 else 'FAIL'}")
    print(
        f"final: spread_r={arr['spread_r'][-1]:.4f} (initial sphere r={R_DROP}), "
        f"z_p95={arr['z_p95'][-1]:.4f}, KE={arr['ke'][-1]:.4e}, n_adh={int(arr['n_adh'][-1])}"
    )
    print(f"csv: {csv_path}")
    if not args.no_video:
        print(f"mp4: {mp4_path}")


if __name__ == "__main__":
    main()
