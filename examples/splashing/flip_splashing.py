"""Push a thin cylindrical metal plate into a full glass using fluid implicit particle (FLIP) simulation.

Run with ``--gpu --vis`` for particle preview, or add ``--surface --record`` for a reconstructed water surface
and a four-times slow-motion video. ``--ball-speed`` controls the prescribed downward speed in meters per second.
``--plate-radius`` controls the plate radius in meters and must stay smaller than the glass opening.
``--plate-clearance`` sets the initial gap above the rim for a longer visible approach before impact.
Use ``--export-particles`` to save every step, including the initial state, for offline surface reconstruction.
The glass and plate are one-way fluid boundaries; their visible entities follow the same prescribed poses.
"""

import argparse
import math
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.utils import mesh
from genesis.utils.misc import tensor_to_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-g", "--gpu", dest="is_gpu", action="store_true", help="Use the GPU for the dense fluid grid.")
    parser.add_argument("-v", "--vis", dest="is_vis", action="store_true", help="Open the interactive viewer.")
    parser.add_argument(
        "-r", "--record", dest="is_record", action="store_true", help="Record a four-times slow-motion video."
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("out/flip_splashing"), help="Video and particle export directory."
    )
    parser.add_argument("-t", "--seconds", type=float, default=1.3, help="Simulated duration in seconds.")
    parser.add_argument(
        "--surface", dest="is_surface", action="store_true", help="Reconstruct smooth water at extra render cost."
    )
    parser.add_argument(
        "--export-particles", dest="is_export", action="store_true", help="Export particle positions and velocities."
    )
    parser.add_argument(
        "--ball-speed", type=float, default=0.8, help="Downward speed in m/s; higher speeds eject water faster."
    )
    parser.add_argument(
        "--plate-radius",
        type=float,
        default=0.029,
        help="Plate radius in meters; wider plates narrow the flow gap and require finer grids.",
    )
    parser.add_argument(
        "--plate-clearance",
        type=float,
        default=0.24,
        help="Initial gap above the rim in meters; a larger gap extends the approach and needs a longer simulation.",
    )
    parser.add_argument(
        "--grid-res", type=int, default=160, help="Cells along the longest axis; finer grids resolve thin walls."
    )
    parser.add_argument("--dt", type=float, default=1.0 / 480.0, help="Maximum scene timestep in seconds.")
    args = parser.parse_args()
    if args.seconds <= 0 or args.ball_speed <= 0 or args.plate_radius <= 0 or args.dt <= 0 or args.grid_res < 32:
        parser.error("Duration, speed, radius, and timestep must be positive; grid resolution must be at least 32.")
    if args.plate_clearance < 0:
        parser.error("Initial plate clearance must be nonnegative.")
    particle_dir = args.output_dir / "particles"
    if args.is_export and particle_dir.exists() and any(particle_dir.iterdir()):
        parser.error(f"Particle export directory must be empty: {particle_dir}")

    gs.init(backend=gs.gpu if args.is_gpu else gs.cpu)
    cup_asset = "meshes/drinking_glass/12-oz-glass.obj"
    cavity_asset = "meshes/drinking_glass/12-oz-glass-cavity.obj"
    cavity = mesh.load_mesh(Path(gs.utils.get_assets_dir()) / cavity_asset)
    floor, rim = cavity.bounds[:, 1]
    mouth_vertices = cavity.vertices[np.isclose(cavity.vertices[:, 1], rim)]
    mouth_radius = 0.5 * np.ptp(mouth_vertices[:, (0, 2)], axis=0).min()
    if args.plate_radius >= mouth_radius:
        parser.error(f"Plate radius must be smaller than the glass opening radius ({mouth_radius:.6f} m).")
    thickness = 0.005
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plate_file = str((args.output_dir / "iron_plate.obj").resolve())
    mesh.create_cylinder(args.plate_radius, thickness, sections=128).export(plate_file)
    rotation = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    start_z = rim + thickness / 2.0 + args.plate_clearance
    stop_z = floor + thickness / 2.0
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
                gs.options.FLIPMeshStaticColliderOptions(
                    pos=(0.0, 0.0, start_z),
                    is_kinematic=True,
                    file=plate_file,
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
    plate = scene.add_entity(
        morph=gs.morphs.Cylinder(
            pos=(0.0, 0.0, start_z),
            height=thickness,
            radius=args.plate_radius,
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
    if args.is_export:
        particle_dir.mkdir(parents=True, exist_ok=True)
        cup = mesh.load_mesh(Path(gs.utils.get_assets_dir()) / cup_asset)
        np.savez_compressed(
            args.output_dir / "scene.npz",
            dt=dt,
            particle_spacing=scene.flip_solver.dx / scene.flip_solver.seed_sub_factor,
            grid_spacing=scene.flip_solver.dx,
            plate_radius=args.plate_radius,
            plate_thickness=thickness,
            plate_start_z=start_z,
            plate_stop_z=stop_z,
            plate_clearance=args.plate_clearance,
            plate_speed=args.ball_speed,
            cup_vertices=cup.vertices,
            cup_faces=cup.faces,
            cup_quat=rotation,
            floor=floor,
            rim=rim,
        )
    if camera is not None:
        camera.start_recording(save_to_filename=str(args.output_dir / "flip_splashing.mp4"), fps=0.125 / dt)
    next_frame = 1.0 / 240.0
    max_height = rim
    steps = math.ceil(args.seconds / dt)
    height = start_z
    try:
        for step in range(steps + 1):
            if args.is_export:
                path = particle_dir / f"frame-{step:06d}.npz"
                temporary = path.with_suffix(".npz.part")
                with temporary.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        time=step * dt,
                        positions=tensor_to_array(water.get_particles_pos())[0],
                        velocities=tensor_to_array(water.get_particles_vel())[0],
                        is_active=tensor_to_array(water.get_particles_active())[0],
                        plate_pos=(0.0, 0.0, height),
                    )
                temporary.replace(path)
            if step == steps:
                break
            progress = min(max(step + 1 - hold_steps, 0), descent_steps)
            height = max(stop_z, start_z - progress * dt * args.ball_speed)
            scene.flip_solver.set_static_colliders_pose([0.0, 0.0, height], [1.0, 0.0, 0.0, 0.0], colliders_idx=[1])
            plate.set_pos([0.0, 0.0, height])
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
        f"{outside} particles outside the cup, plate z={height:.6f} m."
    )


if __name__ == "__main__":
    main()
