"""Stretch a cylindrical liquid membrane between two translating rings (C++ case 5).

Scale 90 reproduces the paper's 53,620 particles. Y is vertical and gravity is zero. Rings separate at speed 0.1,
double their speed at 78/30 seconds, and stop at 200/30 seconds. Physical neck breakup is part of this trajectory.
Use ``--export-particles`` for per-step numerical checkpoints, ``--check`` for sparse particle snapshots,
``--vis`` for the viewer, or ``--record`` for a camera movie.
"""

import argparse
import math
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.utils.checkpoint_replay import CheckpointWriter
from genesis.utils.misc import tensor_to_array


def cylindrical_membrane(radius):
    """Sample the reference's staggered cylindrical layers and return positions and moving rim indices."""
    spacing = 1.4 * radius
    theta_step = 2 * math.asin(spacing / 2)
    layers = []
    rims = []
    offset = 0
    for cylinder_radius, end in ((1.0, 0.5), (1.0 - spacing / 2, 0.5 + spacing / 2)):
        x = -end + spacing * np.arange(math.floor(2 * end / spacing) + 1)
        is_rim = (x <= -end + 0.2 * spacing) | (x + spacing > end)
        counts = np.ceil(2 * math.pi / (theta_step * np.where(is_rim, 0.7, 1.0))).astype(gs.np_int)
        rings_idx = np.repeat(np.arange(len(x)), counts)
        theta = (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)) * (
            2 * math.pi / counts[rings_idx]
        )
        positions = np.column_stack((x[rings_idx], cylinder_radius * np.cos(theta), cylinder_radius * np.sin(theta)))
        rims.append(offset + np.flatnonzero(is_rim[rings_idx]))
        layers.append(positions)
        offset += len(positions)
    return np.concatenate(layers), np.concatenate(rims)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive particle viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record a camera movie")
    parser.add_argument("-t", "--seconds", type=float, default=10.0, help="Simulated duration")
    parser.add_argument("--scale", type=int, default=90, help="Particle radius is 1 / scale in reference world units")
    parser.add_argument("--dt", type=float, default=1 / 30, help="Simulation timestep")
    parser.add_argument("--check", action="store_true", help="Save particle snapshots once per simulated second")
    parser.add_argument(
        "--export-particles", action="store_true", help="Export numerical checkpoints at every timestep"
    )
    parser.add_argument("--surface", action="store_true", help="Reconstruct the liquid surface at extra rendering cost")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("out/thin_film/film_catenoid"), help="Output directory"
    )
    args = parser.parse_args()
    if args.scale < 4 or args.dt <= 0 or args.seconds < 0:
        parser.error("Require scale >= 4, dt > 0 and seconds >= 0")

    # The surface topology implementation requires CUDA.
    gs.init(backend=gs.cuda, precision="32", seed=0)
    radius = 1 / args.scale
    positions, rim_idx = cylindrical_membrane(radius)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            gravity=(0.0, 0.0, 0.0),
        ),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=2 * radius,
            max_solver_iterations=100,
            topology_rebuild_interval=10,
            lower_bound=(-100.0, -100.0, -100.0),
            upper_bound=(100.0, 100.0, 100.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 2.2, 5.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_up=(0.0, 1.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )
    film = scene.add_entity(
        morph=gs.morphs.Particles(
            positions=positions,
        ),
        material=gs.materials.PBSTF.Liquid(
            density_compliance=600.0,
            surface_tension_compliance=0.08,
            surface_distance_compliance=10.0,
            interior_distance_compliance=180.0,
            surface_distance_ratio=0.7,
            initial_density_ratio=0.95,
            surface_viscosity=0.1,
            interior_viscosity=0.1,
        ),
        surface=gs.surfaces.Default(color=(0.25, 0.65, 0.9, 1.0), vis_mode="recon" if args.surface else "particle"),
    )
    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(960, 720), pos=(3.5, 2.2, 5.0), lookat=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), fov=40
        )
    scene.build()
    film.fix_particles(rim_idx)
    gs.logger.info(f"Film: {film.n_particles}; moving rim: {len(rim_idx)}")
    rim_initial = positions[rim_idx]
    rim_velocity = np.zeros_like(rim_initial)
    rim_velocity[:, 0] = np.sign(rim_initial[:, 0]) * 0.1
    if args.record or args.check or args.export_particles:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_writer = None
    if args.export_particles:
        checkpoint_writer = CheckpointWriter(scene, args.output_dir / "checkpoints", args.dt)
        np.savez_compressed(args.output_dir / "rim_indices.npz", rim_idx=rim_idx)
    if camera is not None:
        camera.start_recording(save_to_filename=str(args.output_dir / "film_catenoid.mp4"), fps=30)
    steps = math.ceil(args.seconds / args.dt)
    try:
        for step in range(steps + 1):
            if checkpoint_writer is not None:
                checkpoint_writer.write_frame()
            if args.check and (step % max(1, round(1 / args.dt)) == 0 or step == steps):
                np.savez_compressed(
                    args.output_dir / f"particles_{step:05d}.npz", film=tensor_to_array(film.get_particles_pos())
                )
                gs.logger.info(f"t={step * args.dt:.3f}: saved membrane positions")
            if step == steps:
                break
            time = step * args.dt
            speed_ratio = 0.0 if time >= 200 / 30 else (2.0 if time >= 78 / 30 else 1.0)
            film.set_particles_vel(rim_velocity * speed_ratio, rim_idx)
            scene.step()
    finally:
        if camera is not None:
            camera.stop_recording()
        if args.vis:
            scene.viewer.stop()


if __name__ == "__main__":
    main()
