"""MF-18 ST sanity probe: zero-gravity cube -> sphere test at the real scale (ps=0.002).

User question (2026-09-21): the pour videos show no obvious surface tension -- is the ST code
actually doing anything at this scale?  This probe isolates ST completely:

  * zero gravity, free space (no containers, no table, no boundary balls);
  * production recipe otherwise: ps=0.002, dt=1/240 x 4 substeps (h=1/960), iters=20,
    leps=0.1, density_clamp_negative=True (production --no-tension), visc=0.005;
  * ST block identical to mf18_realscale_pbd.py (area + one-sided distance, compliance arg).

A/B: default runs with ST on; --no-st runs the same scene with the whole ST block off.
If ST works, the cube must visibly round toward its equal-volume sphere:
  spans L -> ~1.241 L,  r_std collapses toward the lattice-shell floor,  corners vanish.

Outputs: multiflow/videos/mf18_stcube{tag}_metrics.csv + _main.mp4
"""

import argparse
import csv
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np  # noqa: E402

import genesis as gs  # noqa: E402

print("genesis loaded from:", gs.__file__)
assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), "Wrong genesis! Must be the multiflow copy."

VIDEOS_DIR = os.path.join(WORKSPACE, "videos")

DT = 1.0 / 240.0
SUBSTEPS = 4
PS = 0.002
CUBE_L = 0.030  # 15 particles per side -> 3375 particles
CUBE_CENTER = (0.0, 0.0, 0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dur", type=float, default=4.0)
    parser.add_argument("--st-comp", type=float, default=0.05)
    parser.add_argument("--no-st", action="store_true")
    parser.add_argument("--no-st-dist", action="store_true",
                        help="keep the area constraint but disable the one-sided surface distance constraint")
    parser.add_argument("--visc", type=float, default=0.005)
    parser.add_argument("--density-relaxation", type=float, default=0.2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--init-from", type=str, default="",
                        help="optional npz with `pos` (N,3) to teleport the cube particles to after build")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, 0.0)),
        pbd_options=gs.options.PBDOptions(
            particle_size=PS,
            lower_bound=(-0.05, -0.05, 0.0),
            upper_bound=(0.05, 0.05, 0.10),
            max_density_solver_iterations=args.iters,
            max_viscosity_solver_iterations=1,
            density_lambda_epsilon=0.1,
            density_clamp_negative=True,  # production --no-tension: ST is the ONLY cohesion here
            surface_tension_enabled=not args.no_st,
            st_compliance=args.st_comp,
            st_surface_density_factor=1.0,
            st_distance_enabled=(not args.no_st) and (not args.no_st_dist),
            st_distance_compliance=40.0,
            st_max_surface_neighbors=256,
            st_max_localmesh_neighbors=128,
        ),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_pbd=False),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )

    cube = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0,
            density_relaxation=args.density_relaxation, viscosity_relaxation=args.visc,
        ),
        morph=gs.morphs.Box(size=(CUBE_L, CUBE_L, CUBE_L), pos=CUBE_CENTER),
        surface=gs.surfaces.Default(color=(0.35, 0.55, 0.95, 1.0), opacity=1.0),
    )
    cam = scene.add_camera(
        res=(640, 640), pos=(0.075, -0.075, 0.05), lookat=(0.0, 0.0, 0.05), fov=40, GUI=False,
    )
    scene.build()

    solver = scene.sim.pbd_solver
    n_fluid = cube.n_particles
    if args.init_from:
        with np.load(args.init_from) as dat:
            init_pos = np.asarray(dat["pos"], dtype=np.float32)
        assert init_pos.shape == (n_fluid, 3), f"{args.init_from}: pos shape {init_pos.shape} != ({n_fluid}, 3)"
        cube.set_particles_pos(init_pos)
        cube.set_particles_vel(np.zeros((n_fluid, 3), dtype=np.float32))
        print(f"init state loaded from {args.init_from} (velocities zeroed)")
    print(
        f"ST CUBE ZEROG: n={n_fluid} L={CUBE_L} ps={PS} h={DT/SUBSTEPS:.6f} iters={args.iters} "
        f"visc={args.visc} st={not args.no_st} st_comp={args.st_comp:g} dur={args.dur} tag='{args.tag}'"
    )

    # analytic references (same volume)
    vol = CUBE_L**3
    r_sphere = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    print(f"[ref] cube: span={CUBE_L:.4f}  r_face={0.5*CUBE_L:.4f}  r_corner={0.8660254*CUBE_L:.4f}")
    print(f"[ref] equal-volume sphere: R={r_sphere:.4f}  span(2R)={2*r_sphere:.4f} (= {2*r_sphere/CUBE_L:.3f} L)")

    mp4 = os.path.join(VIDEOS_DIR, f"mf18_stcube{args.tag}_main.mp4")
    csv_path = os.path.join(VIDEOS_DIR, f"mf18_stcube{args.tag}_metrics.csv")
    cam.start_recording(save_to_filename=mp4, fps=args.fps)

    n_frames = int(round(args.dur / DT))
    rows = []
    print_every = max(1, int(round(0.5 / DT)))
    for i in range(n_frames + 1):
        if i > 0:
            scene.step()
        pos = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
        vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
        spans = pos.max(axis=0) - pos.min(axis=0)
        r = np.linalg.norm(pos - pos.mean(axis=0), axis=1)
        ke = float(0.5 * np.mean(np.sum(vel * vel, axis=1)))
        n_surf = -1
        if not args.no_st:
            n_surf = int(solver.on_surface.to_numpy()[:n_fluid, 0].sum())
        t = i * DT
        rows.append((t, *spans, r.mean(), r.std(), r.min(), r.max(), ke, n_surf))
        if i % print_every == 0 or i == n_frames:
            print(
                f"t={t:5.2f}s  span=({spans[0]:.4f},{spans[1]:.4f},{spans[2]:.4f})  "
                f"r_mean={r.mean():.4f} r_std={r.std():.5f} r_min={r.min():.4f} r_max={r.max():.4f}  "
                f"KE={ke:.3e}  n_surf={n_surf}"
            )

    cam.stop_recording()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "span_x", "span_y", "span_z", "r_mean", "r_std", "r_min", "r_max", "ke", "n_surface"])
        w.writerows(rows)

    # final-shape sphericity analysis
    pos = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
    ctr = pos.mean(axis=0)
    r = np.linalg.norm(pos - ctr, axis=1)
    shell = r[r > 0.9 * r.max()]
    # inertia tensor eigenvalues: equal => isotropic (sphere); elongated => cube remnant
    rel = pos - ctr
    inertia = np.einsum("ni,nj->ij", rel, rel) / len(pos)
    eig = np.sort(np.linalg.eigvalsh(inertia))[::-1]
    npz_path = os.path.join(VIDEOS_DIR, f"mf18_stcube{args.tag}_final.npz")
    np.savez(npz_path, pos=pos)
    print(f"[shape] r_max={r.max():.4f}  shell(r>0.9r_max): n={len(shell)} mean={shell.mean():.4f} "
          f"std={shell.std():.5f} min={shell.min():.4f} max={shell.max():.4f}")
    print(f"[shape] inertia eig=({eig[0]:.3e},{eig[1]:.3e},{eig[2]:.3e})  ratio max/min={eig[0]/eig[2]:.3f} (sphere=1.0)")
    print(f"[shape] eff ball radius from mean r (uniform ball r_mean=3R/4): R_eff={r.mean()/0.75:.4f} "
          f"vs equal-vol R={r_sphere:.4f} -> volume ratio ~{(r.mean()/0.75/r_sphere)**3:.3f}")
    print(f"npz: {npz_path}")
    print(f"csv: {csv_path}")
    print(f"mp4: {mp4}")


if __name__ == "__main__":
    main()
