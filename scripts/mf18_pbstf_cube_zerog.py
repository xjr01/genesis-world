"""MF-18 plan section 6-B: real-scale zero-gravity cube test on the ported PBSTF path.

Scene (fixed by the plan, section 5): 3 cm cube, regular sampling, ps=0.002
(15^3 = 3375 particles), no walls, zero gravity, h=1/960 (sim dt=1/240 x 4
substeps), 20 solver iterations, topology_rebuild_interval=2, PCA normals on.

Per-run records (plan section 6-B):
  * per-frame CSV: nan count, KE, center of mass, momentum, interior density
    ratio p25/p50/p75, on_surface count, topology overflow, ms/frame;
  * t=0 and final npz (positions + on_surface flags, reordered view);
  * closed surface area / volume under ONE reconstruction setting (pysplashsurf,
    same kwargs as genesis.utils.particle._splashsurf_recon_kwargs) for the t=0
    state, the final state, and an equal-volume reference sphere;
  * sphericity: A^3/(36 pi V^2) plus centroid radial distance min/max/mean of
    mesh vertices (all-direction check; no three-axis-equality shortcut);
  * one fixed-camera mp4 per run (30 fps, 960x1280, point particles).

Usage:
  PYTHONIOENCODING=utf-8 "$PY" mf18_pbstf_cube_zerog.py --run area_m \
      --density-compliance 500 --area-compliance 0.0012 --dur 2.0
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
ROOT_REPO = os.path.dirname(WORKSPACE)  # ipbf/

DT = 1.0 / 240.0
SUBSTEPS = 4
PS = 0.002
CUBE_L = 0.030  # 15 particles per side -> 3375 particles
CUBE_CENTER = (0.0, 0.0, 0.05)
RHO0 = 1000.0

VIDEOS_DIR = os.path.join(WORKSPACE, "videos")
OUT_PREFIX = "mf18_pbstf_stcube_"
RECON_CSV = os.path.join(VIDEOS_DIR, OUT_PREFIX + "recon.csv")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=str, required=True, help="run name, used in output file names")
    parser.add_argument("--engine", choices=("multiflow", "root"), default="multiflow")
    parser.add_argument("--dur", type=float, default=2.0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--density-compliance", type=float, default=500.0)
    parser.add_argument("--area-compliance", type=float, default=1e12)
    parser.add_argument("--surface-dist-compliance", type=float, default=40.0)
    parser.add_argument("--interior-dist-compliance", type=float, default=180.0)
    parser.add_argument("--surface-visc", type=float, default=0.0)
    parser.add_argument("--interior-visc", type=float, default=0.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-recon", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def reconstruct(positions):
    """One fixed reconstruction setting for every state (pysplashsurf, in-process).

    Parameters mirror genesis.utils.particle._splashsurf_recon_kwargs defaults so
    the numbers are comparable with any mesh produced through the engine wrapper.
    """
    import pysplashsurf
    import trimesh

    mesh_with_data, _ = pysplashsurf.reconstruction_pipeline(
        np.ascontiguousarray(positions, dtype=np.float64),
        particle_radius=0.5 * PS,
        smoothing_length=2.0,
        cube_size=0.8,
        iso_surface_threshold=0.6,
        mesh_smoothing_weights=True,
        mesh_smoothing_iters=25,
        normals_smoothing_iters=10,
        mesh_cleanup=True,
        compute_normals=True,
        multi_threading=True,
    )
    return trimesh.Trimesh(
        vertices=mesh_with_data.mesh.vertices,
        faces=mesh_with_data.mesh.triangles,
        process=False,
    )


def shape_metrics(tag, phase, pos, mass, recon_rows):
    """Reconstruct `pos`, append area/volume/sphericity row, print summary."""
    mesh = reconstruct(pos)
    area = float(mesh.area)
    vol = float(mesh.volume) if mesh.is_watertight else float("nan")
    verts = np.asarray(mesh.vertices)
    if len(verts):
        ctr = verts.mean(axis=0)
        r = np.linalg.norm(verts - ctr, axis=1)
        r_min, r_mean, r_max = float(r.min()), float(r.mean()), float(r.max())
    else:
        r_min = r_mean = r_max = float("nan")
    sphericity = area**3 / (36.0 * np.pi * vol**2) if np.isfinite(vol) and vol > 0 else float("nan")
    vol_nominal = float(mass * len(pos) / RHO0)
    row = dict(
        run=tag,
        phase=phase,
        n_particles=len(pos),
        n_vertices=len(mesh.vertices),
        n_faces=len(mesh.faces),
        watertight=bool(mesh.is_watertight),
        area=area,
        volume=vol,
        sphericity=sphericity,
        r_min=r_min,
        r_mean=r_mean,
        r_max=r_max,
        vol_nominal=vol_nominal,
        vol_ratio=vol / vol_nominal if np.isfinite(vol) else float("nan"),
    )
    recon_rows.append(row)
    print(
        f"[recon:{phase}] V={len(mesh.vertices)}v/{len(mesh.faces)}f watertight={mesh.is_watertight} "
        f"area={area:.6e} vol={vol:.6e} (nominal {vol_nominal:.6e}, ratio {row['vol_ratio']:.4f}) "
        f"sphericity={sphericity:.4f} r=({r_min:.4f},{r_mean:.4f},{r_max:.4f})"
    )
    return row


def reference_sphere_points():
    """Equal-volume sphere, point-sampled on the same regular lattice as the cube."""
    vol = CUBE_L**3
    r_sphere = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    # regular lattice with spacing ps, one site at the sphere center (same family
    # as the cube sampler: sites at k*ps offsets)
    n_half = int(np.ceil(r_sphere / PS)) + 1
    axes = np.arange(-n_half, n_half + 1) * PS
    gx, gy, gz = np.meshgrid(axes, axes, axes, indexing="ij")
    pts = np.stack((gx.ravel(), gy.ravel(), gz.ravel()), axis=1)
    pts = pts[np.linalg.norm(pts, axis=1) <= r_sphere]
    return pts.astype(np.float32), r_sphere


def append_recon_csv(rows):
    if not rows:
        return
    fields = [
        "run", "phase", "n_particles", "n_vertices", "n_faces", "watertight",
        "area", "volume", "sphericity", "r_min", "r_mean", "r_max",
        "vol_nominal", "vol_ratio",
    ]
    exists = os.path.exists(RECON_CSV)
    with open(RECON_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    args = parse_args()

    if args.engine == "multiflow":
        sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))
    else:
        sys.path.insert(0, os.path.join(ROOT_REPO, "genesis-world"))

    import genesis as gs  # noqa: E402
    from genesis.utils.misc import qd_to_numpy  # noqa: E402

    print("genesis loaded from:", gs.__file__)
    if args.engine == "multiflow":
        assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), "Wrong genesis (want multiflow)."
    else:
        assert not os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), "Wrong genesis (want root)."

    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, 0.0)),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=PS,
            lower_bound=(-0.06, -0.06, -0.01),
            upper_bound=(0.06, 0.06, 0.11),
            max_solver_iterations=args.iters,
            topology_rebuild_interval=2,
        ),
        vis_options=(
            # root repo (v1.3.1) lacks the "points" render mode; root runs are --no-video anyway
            gs.options.VisOptions(render_particle_as="points")
            if args.engine == "multiflow"
            else None
        ),
        show_viewer=False,
    )
    cube = scene.add_entity(
        material=gs.materials.PBSTF.Liquid(
            rho=RHO0,
            density_compliance=args.density_compliance,
            surface_tension_compliance=args.area_compliance,
            surface_distance_compliance=args.surface_dist_compliance,
            interior_distance_compliance=args.interior_dist_compliance,
            surface_viscosity=args.surface_visc,
            interior_viscosity=args.interior_visc,
            sampler="regular",
        ),
        morph=gs.morphs.Box(size=(CUBE_L, CUBE_L, CUBE_L), pos=CUBE_CENTER),
        surface=gs.surfaces.Default(color=(0.35, 0.55, 0.95, 1.0), opacity=1.0),
    )
    cam = scene.add_camera(
        res=(960, 1280), pos=(0.075, -0.075, 0.05), lookat=(0.0, 0.0, 0.05), fov=40, GUI=False,
    )
    scene.build()

    sim = scene.sim
    solver = sim.pbstf_solver
    n_fluid = cube.n_particles
    mass = solver._default_mass
    gravity_now = solver._gravity.to_numpy()
    print(
        f"PBSTF CUBE ZEROG [{args.run}/{args.engine}]: n={n_fluid} L={CUBE_L} ps={PS} "
        f"dt={solver._dt:.6e} substep_dt={solver._substep_dt:.6e} substeps={sim._substeps} "
        f"gravity={gravity_now} support={solver._support_radius:.4f} iters={solver._max_solver_iterations} "
        f"topo_interval={solver._topology_rebuild_interval} mass={mass:.10e}"
    )
    print(
        f"compliances: density={args.density_compliance:g} area={args.area_compliance:g} "
        f"sdist={args.surface_dist_compliance:g} idist={args.interior_dist_compliance:g} "
        f"visc=({args.surface_visc:g},{args.interior_visc:g})"
    )
    assert solver._substep_dt == 1.0 / 960.0, f"substep dt {solver._substep_dt} != 1/960"
    assert np.allclose(gravity_now, 0.0), f"gravity not zero: {gravity_now}"

    def snapshot():
        """Reordered-view state: pos/vel/density/on_surface all in one consistent order."""
        pos = qd_to_numpy(solver.particles_reordered.pos, transpose=True)[0]
        vel = qd_to_numpy(solver.particles_reordered.vel, transpose=True)[0]
        density = qd_to_numpy(solver.particles_reordered.density, transpose=True)[0]
        on_surface = qd_to_numpy(solver.on_surface, transpose=True)[0].astype(bool)
        return pos, vel, density, on_surface

    # t=0 state. NOTE: scene.build() runs one kernel-compilation sim.step() followed by
    # a reset; the reset restores the user-order `particles` fields only, leaving the
    # reordered views stale (verified: garbage spread across the domain bounds). One
    # explicit reorder + topology rebuild re-synchronizes them without moving particles.
    solver._kernel_reorder_particles(0)
    solver._rebuild_topology(0)
    pos0, vel0, rho0_arr, surf0 = snapshot()
    init_npz = os.path.join(VIDEOS_DIR, OUT_PREFIX + "init.npz")
    np.savez(init_npz, pos=pos0, on_surface=surf0)
    print(f"[t0] pos saved -> {init_npz}; on_surface={surf0.sum()}/{len(surf0)}")

    recon_rows = []
    if not args.no_recon:
        shape_metrics(args.run, "t0", pos0, mass, recon_rows)
        sphere_pts, r_sphere = reference_sphere_points()
        print(f"[ref] equal-volume sphere R={r_sphere:.5f} m, sampled {len(sphere_pts)} lattice points")
        shape_metrics("sphere_ref", "t0", sphere_pts, mass, recon_rows)

    mp4 = os.path.join(VIDEOS_DIR, f"{OUT_PREFIX}{args.run}.mp4")
    csv_path = os.path.join(VIDEOS_DIR, f"{OUT_PREFIX}{args.run}_metrics.csv")
    if not args.no_video:
        cam.start_recording(save_to_filename=mp4, fps=args.fps)

    vol_nominal = mass * n_fluid / RHO0
    r_sphere = (3.0 * vol_nominal / (4.0 * np.pi)) ** (1.0 / 3.0)
    print(f"[ref] nominal volume={vol_nominal:.6e} m^3, equal-volume sphere R={r_sphere:.5f} m")

    n_frames = int(round(args.dur / DT))
    print_every = max(1, int(round(0.25 / DT)))
    rows = []
    overflow_msg = ""
    t_start = time.perf_counter()

    def frame_metrics(i, pos, vel, density, on_surface, ms):
        nan_count = int((~np.isfinite(pos).all(axis=1) | ~np.isfinite(vel).all(axis=1)).sum())
        com = pos.mean(axis=0)
        mom = mass * vel.sum(axis=0)
        ke = float(0.5 * mass * np.sum(vel * vel))
        vmax = float(np.linalg.norm(vel, axis=1).max())
        interior = density[~on_surface] / RHO0
        if len(interior):
            p25, p50, p75 = (float(np.percentile(interior, q)) for q in (25, 50, 75))
        else:
            p25 = p50 = p75 = float("nan")
        return (i * DT, nan_count, ke, *com, *mom, p25, p50, p75, vmax, int(on_surface.sum()), ms)

    rows.append(frame_metrics(0, pos0, vel0, rho0_arr, surf0, 0.0))
    for i in range(1, n_frames + 1):
        t0_step = time.perf_counter()
        try:
            scene.step()
        except Exception as e:  # topology overflow / NaN errno abort the run; record and stop
            overflow_msg = f"{type(e).__name__}: {e}"
            print(f"[abort] frame {i}: {overflow_msg}")
            break
        ms = (time.perf_counter() - t0_step) * 1e3
        pos, vel, density, on_surface = snapshot()
        rows.append(frame_metrics(i, pos, vel, density, on_surface, ms))
        if i % print_every == 0 or i == n_frames:
            r = rows[-1]
            print(
                f"t={r[0]:5.2f}s nan={r[1]} KE={r[2]:.3e} com_z={r[5]:.4f} "
                f"rho50={r[10]:.4f} vmax={r[12]:.3f} n_surf={r[13]} ms={ms:.1f}"
            )

    if not args.no_video:
        cam.stop_recording()
    wall = time.perf_counter() - t_start
    n_done = len(rows) - 1
    print(f"[perf] {n_done} frames in {wall:.1f}s -> {1e3*wall/max(n_done,1):.2f} ms/frame (incl. render+IO)")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["t", "nan_count", "ke_J", "com_x", "com_y", "com_z", "mom_x", "mom_y", "mom_z",
             "rho_p25", "rho_p50", "rho_p75", "max_v", "n_surface", "ms_frame"]
        )
        w.writerows(rows)

    final_npz = os.path.join(VIDEOS_DIR, f"{OUT_PREFIX}{args.run}_final.npz")
    if n_done > 0:
        np.savez(final_npz, pos=pos, on_surface=on_surface, overflow=overflow_msg)
        print(f"final npz: {final_npz} (on_surface={on_surface.sum()}/{len(on_surface)})")
        if not args.no_recon:
            shape_metrics(args.run, "final", pos, mass, recon_rows)

    append_recon_csv(recon_rows)
    print(f"csv: {csv_path}")
    if not args.no_video:
        print(f"mp4: {mp4}")
    print(f"recon csv: {RECON_CSV}")


if __name__ == "__main__":
    main()
