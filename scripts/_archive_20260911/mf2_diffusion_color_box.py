"""MF-2 validation: per-particle concentration coloring (point cloud) for PBD diffusion.

Same box scenario as MF-1: liquid box in the lower half of the domain, c=1 ("coffee")
below z_mid=0.25, c=0 ("water") above. Rendered with VisOptions(render_particle_as="points"),
which colors each particle by its concentration field via the new `on_pbd`/`update_pbd`
points branch in rasterizer_context.py.

Outputs:
  - video  multiflow/videos/mf2_diffusion_color.mp4 (default)
  - CSV    per-log-step t, sum_c, std_c (confirms diffusion is running)

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 "$PY" \
      multiflow/scripts/mf2_diffusion_color_box.py --epsd 0.2 --seconds 10
"""

import argparse
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsd", type=float, default=0.2, help="diffusion_coeff (0 = off)")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--point-size", type=float, default=6.0, help="GL point size in pixels")
    parser.add_argument("--video", default=os.path.join(WORKSPACE, "videos", "mf2_diffusion_color.mp4"))
    parser.add_argument("--csv", default=None, help="CSV output path (default: alongside video)")
    args = parser.parse_args()
    if args.csv is None:
        args.csv = os.path.splitext(args.video)[0] + ".csv"
    return args


args = parse_args()

sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np

import genesis as gs

print("genesis.__file__ =", gs.__file__)
assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE))

DT = 2e-3
Z_MID = 0.25


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        pbd_options=gs.options.PBDOptions(
            lower_bound=(-0.3, -0.3, 0.0),
            upper_bound=(0.3, 0.3, 0.8),
            particle_size=0.01,
            max_density_solver_iterations=10,
            max_viscosity_solver_iterations=1,
            diffusion_coeff=args.epsd,
        ),
        vis_options=gs.options.VisOptions(
            render_particle_as="points",
            visualize_pbd_boundary=True,
        ),
        show_viewer=False,
    )

    liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular",
            rho=1.0,
            density_relaxation=1.0,
            viscosity_relaxation=0.0,
            c_init_z_mid=Z_MID,
        ),
        morph=gs.morphs.Box(lower=(-0.15, -0.15, 0.01), upper=(0.15, 0.15, 0.5)),
    )

    cam = scene.add_camera(
        res=(640, 480),
        pos=(0.3, -0.85, 0.55),
        lookat=(0.0, 0.0, 0.28),
        fov=40,
        GUI=False,
    )

    scene.build()
    print("n_particles:", liquid.n_particles)

    # GL point size for the point-cloud rendering (consumed in Rasterizer.render_camera)
    scene.visualizer.rasterizer._camera_targets[cam.uid].point_size = args.point_size

    solver = scene.sim.pbd_solver
    n_steps = int(args.seconds / DT)
    os.makedirs(os.path.dirname(args.video), exist_ok=True)

    cam.start_recording(save_to_filename=args.video, fps=args.fps)
    with open(args.csv, "w") as fh:
        fh.write("step,t,sum_c,std_c\n")
        for step in range(n_steps + 1):
            if step > 0:
                scene.step()
            c = solver.particles.c.to_numpy()[:, 0]
            fh.write(f"{step},{step * DT:.6f},{float(np.sum(c)):.8g},{float(np.std(c)):.8g}\n")
            if step % 500 == 0:
                print(f"step {step}/{n_steps}  sum_c={float(np.sum(c)):.4f} std_c={float(np.std(c)):.5f}")
    cam.stop_recording()

    print("video saved to", args.video)
    print("csv saved to", args.csv)


if __name__ == "__main__":
    main()
