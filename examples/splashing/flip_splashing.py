"""Push a metal sphere into a full glass using fluid implicit particle (FLIP) simulation.

Run with ``--gpu --vis`` for particle preview, or add ``--surface --record`` for a reconstructed water surface
and a four-times slow-motion video. ``--ball-speed`` controls the prescribed downward speed in meters per second.
The glass and sphere are one-way fluid boundaries; their visible entities follow the same prescribed poses.
"""

import argparse
import math
from pathlib import Path

import genesis as gs
from genesis.utils import mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-g", "--gpu", dest="is_gpu", action="store_true", help="Use the GPU for the dense fluid grid.")
    parser.add_argument("-v", "--vis", dest="is_vis", action="store_true", help="Open the interactive viewer.")
    parser.add_argument(
        "-r", "--record", dest="is_record", action="store_true", help="Record a four-times slow-motion video."
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("out/flip_splashing"), help="Recording directory."
    )
    parser.add_argument("-t", "--seconds", type=float, default=1.0, help="Simulated duration in seconds.")
    parser.add_argument(
        "--surface", dest="is_surface", action="store_true", help="Reconstruct smooth water at extra render cost."
    )
    parser.add_argument(
        "--ball-speed", type=float, default=0.4, help="Downward speed in m/s; higher speeds eject water faster."
    )
    parser.add_argument(
        "--grid-res", type=int, default=160, help="Cells along the longest axis; finer grids resolve thin walls."
    )
    parser.add_argument("--dt", type=float, default=1.0 / 480.0, help="Maximum scene timestep in seconds.")
    args = parser.parse_args()
    if args.seconds <= 0 or args.ball_speed <= 0 or args.dt <= 0 or args.grid_res < 32:
        parser.error("Duration, speed, and timestep must be positive; grid resolution must be at least 32.")

    gs.init(backend=gs.gpu if args.is_gpu else gs.cpu)
    cup_asset = "meshes/drinking_glass/12-oz-glass.obj"
    cavity_asset = "meshes/drinking_glass/12-oz-glass-cavity.obj"
    cavity = mesh.load_mesh(Path(gs.utils.get_assets_dir()) / cavity_asset)
    floor, rim = cavity.bounds[:, 1]
    radius = 0.9 * 0.070522047730 / 2.0
    rotation = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    start_z = rim + radius + 0.015
    stop_z = floor + radius
    # Align the contact time to a scene step so the prescribed speed stays constant through arrival.
    descent_steps = math.ceil((start_z - stop_z) / (args.ball_speed * args.dt))
    dt = float((start_z - stop_z) / (args.ball_speed * descent_steps))
    hold_steps = math.ceil(0.05 / dt)
    camera_pos = (0.40, -0.55, 0.30)
    camera_lookat = (0.0, 0.0, 0.14)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.81),
        ),
        flip_options=gs.options.FLIPOptions(
            static_colliders=[
                gs.options.FLIPMeshStaticColliderOptions(
                    quat=rotation,
                    file=cup_asset,
                    sdf_res=192,
                ),
                gs.options.FLIPSphereStaticColliderOptions(
                    pos=(0.0, 0.0, start_z),
                    is_kinematic=True,
                    radius=radius,
                ),
            ],
            grid_res=args.grid_res,
            lower_bound=(-0.16, -0.16, 0.0),
            upper_bound=(0.16, 0.16, 0.384),
            seed_sub_factor=2,
            dem_coupling=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40,
            realtime_factor=0.25,
        ),
        show_viewer=args.is_vis,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(color=(0.20, 0.23, 0.27)),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            quat=rotation,
            file=cup_asset,
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(color=(0.70, 0.85, 0.95, 0.18), roughness=0.08),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, start_z),
            radius=radius,
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(color=(0.55, 0.58, 0.62), roughness=0.18, metallic=0.95),
    )
    water = scene.add_entity(
        morph=gs.morphs.Mesh(
            quat=rotation,
            file=cavity_asset,
        ),
        material=gs.materials.FLIP.Liquid(),
        surface=gs.surfaces.Default(
            color=(0.04, 0.40, 0.72), roughness=0.12, vis_mode="recon" if args.is_surface else "particle"
        ),
    )
    camera = None
    if args.is_record:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        camera = scene.add_camera(res=(1280, 960), pos=camera_pos, lookat=camera_lookat, fov=40)
    scene.build()
    if camera is not None:
        camera.start_recording(save_to_filename=str(args.output_dir / "flip_splashing.mp4"), fps=0.125 / dt)
    next_frame = 1.0 / 240.0
    max_height = rim
    try:
        for step in range(math.ceil(args.seconds / dt)):
            progress = min(max(step + 1 - hold_steps, 0), descent_steps)
            height = max(stop_z, start_z - progress * dt * args.ball_speed)
            scene.flip_solver.set_static_colliders_pose([0.0, 0.0, height], [1.0, 0.0, 0.0, 0.0], colliders_idx=[1])
            sphere.set_pos([0.0, 0.0, height])
            scene.step(update_visualizer=args.is_vis)
            if (step + 1) * dt >= next_frame:
                max_height = max(max_height, water.get_particles_pos()[..., 2].max().item())
                next_frame += 1.0 / 240.0
    finally:
        if camera is not None:
            camera.stop_recording()
    positions = water.get_particles_pos()
    outside = (positions[..., :2].norm(dim=-1) > 0.038).sum().item()
    gs.logger.info(
        f"FLIP splash: {water.n_particles} particles, max height {max_height:.4f} m, "
        f"{outside} particles outside the cup, sphere z={height:.6f} m."
    )


if __name__ == "__main__":
    main()
