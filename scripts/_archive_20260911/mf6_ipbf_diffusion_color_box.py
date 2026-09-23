"""MF-6 validation: per-particle concentration coloring (point cloud) for IPBF diffusion.

Same stacked-box scenario as MF-5: coffee box (z in [0.01, 0.15], c_init=1.0) below,
water box (z in [0.15, 0.30], c_init=0.0) above, in one IPBF solver with static boundary
particles enabled. Rendered with VisOptions(render_particle_as="points"), which colors
each particle by its concentration field via the new points branch in
`on_ipbf`/`update_ipbf` (rasterizer_context.py), mirroring the PBD MF-2 implementation.

Outputs:
  - video  multiflow/videos/mf6_ipbf_diffusion_color.mp4 (default)
  - CSV    per-frame t, sum_c, std_c (confirms diffusion is running; physics must match
           the mf5 non-points run of the same configuration)

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 "$PY" \
      multiflow/scripts/mf6_ipbf_diffusion_color_box.py --epsd 0.1 --seconds 8
"""

import argparse
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsd", type=float, default=0.1, help="diffusion_coeff (0 = off)")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--point-size", type=float, default=6.0, help="GL point size in pixels")
    parser.add_argument("--video", default=os.path.join(WORKSPACE, "videos", "mf6_ipbf_diffusion_color.mp4"))
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

DT = 1.0 / 60.0
SUBSTEPS = 8


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        ipbf_options=gs.options.IPBFOptions(
            particle_size=0.01,
            lower_bound=(-0.2, -0.2, 0.0),
            upper_bound=(0.2, 0.2, 0.6),
            boundary_particles=True,
            ipbf_iterations=2,
            alpha=1e-8,
            viscosity_xsph=0.1,
            damping_enabled=False,
            diffusion_coeff=args.epsd,
        ),
        vis_options=gs.options.VisOptions(
            render_particle_as="points",
        ),
        show_viewer=False,
    )

    coffee = scene.add_entity(
        material=gs.materials.IPBF.Liquid(sampler="regular", rho=1000.0, c_init=1.0),
        morph=gs.morphs.Box(lower=(-0.1, -0.1, 0.01), upper=(0.1, 0.1, 0.15)),
    )
    water = scene.add_entity(
        material=gs.materials.IPBF.Liquid(sampler="regular", rho=1000.0, c_init=0.0),
        morph=gs.morphs.Box(lower=(-0.1, -0.1, 0.15), upper=(0.1, 0.1, 0.30)),
    )

    cam = scene.add_camera(
        res=(640, 480),
        pos=(0.3, -0.7, 0.28),
        lookat=(0.0, 0.0, 0.16),
        fov=40,
        GUI=False,
    )

    scene.build()
    print("n_particles: coffee", coffee.n_particles, " water", water.n_particles)

    # GL point size for the point-cloud rendering (consumed in Rasterizer.render_camera)
    scene.visualizer.rasterizer._camera_targets[cam.uid].point_size = args.point_size

    solver = scene.sim.ipbf_solver
    n_fluid = solver._n_fluid_particles
    n_frames = int(round(args.seconds / DT))
    os.makedirs(os.path.dirname(args.video), exist_ok=True)

    cam.start_recording(save_to_filename=args.video, fps=args.fps)
    with open(args.csv, "w") as fh:
        fh.write("step,t,sum_c,std_c\n")
        for frame in range(n_frames + 1):
            if frame > 0:
                scene.step()
            c = solver.particles.c.to_numpy()[:n_fluid, 0]
            fh.write(f"{frame},{frame * DT:.6f},{float(np.sum(c)):.8g},{float(np.std(c)):.8g}\n")
            if frame % 60 == 0:
                print(f"frame {frame}/{n_frames}  sum_c={float(np.sum(c)):.4f} std_c={float(np.std(c)):.5f}")
    cam.stop_recording()

    print("video saved to", args.video)
    print("csv saved to", args.csv)


if __name__ == "__main__":
    main()
