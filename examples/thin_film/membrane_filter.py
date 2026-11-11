"""A falling droplet merges into a liquid membrane supported by a circular rim.

Run with ``--vis`` for the viewer or ``--record`` for a camera movie. ``--check`` measures projected film coverage
and saves particle snapshots. Dimensions and sampling follow the C++ case 22; Y is vertical. Scale 75 gives 14,046
particles (23,218 with the second membrane enabled in the reference). Gravity and timestep follow the paper's table 1.
The default 64-bit arithmetic favors stability during impact; ``--precision 32`` reduces GPU cost.
Use ``--export-particles`` to save per-step numerical checkpoints for offline rendering.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import genesis as gs
from genesis.utils import particle
from genesis.utils.checkpoint_replay import CheckpointWriter
from genesis.utils.misc import tensor_to_array


def spherical_membrane(radius, circular_radius=1.0, sphere_radius=6.0):
    """Sample two spherical caps and return their positions and fixed rim indices."""
    theta_start = math.acos(circular_radius / sphere_radius)
    theta_step = 2.0 * math.asin(radius / sphere_radius)
    theta = theta_start + theta_step * np.arange(math.floor((math.pi / 2 - theta_start) / theta_step) + 1)
    theta = theta[sphere_radius * np.cos(theta) >= radius]
    phi_step = 2 * np.arcsin(radius / (sphere_radius * np.cos(theta)))
    counts = np.ceil(2 * math.pi / phi_step).astype(gs.np_int)
    rings_idx = np.repeat(np.arange(len(theta)), counts)
    phi = (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)) * phi_step[rings_idx]
    height = math.sqrt(sphere_radius**2 - circular_radius**2)
    positions = np.column_stack(
        (
            sphere_radius * np.cos(phi) * np.cos(theta[rings_idx]),
            sphere_radius * np.sin(theta[rings_idx]) - height,
            sphere_radius * np.sin(phi) * np.cos(theta[rings_idx]),
        )
    )
    positions = np.concatenate((positions, [[0.0, sphere_radius - height, 0.0]]))
    rim_idx = np.concatenate((np.arange(counts[0]), len(positions) + np.arange(counts[0])))
    return np.concatenate((positions, positions + (0.0, 2 * radius, 0.0))), rim_idx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive particle viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record a camera movie")
    parser.add_argument("-t", "--seconds", type=float, default=10.0, help="Simulated duration")
    parser.add_argument("--scale", type=int, default=75, help="Particle radius is 1 / scale in reference world units")
    parser.add_argument("--dt", type=float, default=1 / 30, help="Simulation timestep")
    parser.add_argument("--precision", choices=("32", "64"), default="64", help="64-bit arithmetic costs more GPU time")
    parser.add_argument("--check", action="store_true", help="Measure film coverage and save particle snapshots")
    parser.add_argument(
        "--export-particles", action="store_true", help="Export numerical checkpoints at every timestep"
    )
    parser.add_argument("--surface", action="store_true", help="Reconstruct the liquid surface at extra rendering cost")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("out/thin_film/membrane_filter"), help="Output directory"
    )
    args = parser.parse_args()
    if args.scale < 4 or args.dt <= 0 or args.seconds < 0:
        parser.error("Require scale >= 4, dt > 0 and seconds >= 0")

    # The surface topology implementation requires CUDA.
    gs.init(backend=gs.cuda, precision=args.precision, seed=0)
    radius = 1.0 / args.scale
    positions, rim_idx = spherical_membrane(radius)
    drop_positions = particle.box_to_particles(p_size=2 * radius, size=(0.5, 0.5, 0.5), sampler="staggered")
    drop_positions = drop_positions[np.linalg.norm(drop_positions, axis=1) <= 0.25] + (0.0, 1.5, 0.0)
    material = gs.materials.PBSTF.Liquid(
        density_compliance=600.0,
        surface_tension_compliance=0.1,
        surface_distance_compliance=10.0,
        surface_viscosity=0.5,
        interior_viscosity=0.1,
    )
    surface = gs.surfaces.Default(color=(0.25, 0.65, 0.9, 1.0), vis_mode="recon" if args.surface else "particle")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            gravity=(0.0, -0.5, 0.0),
        ),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=2 * radius,
            max_solver_iterations=40,
            lower_bound=(-2.0, -2.0, -2.0),
            upper_bound=(2.0, 12.0, 2.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.2, 1.5, 4.2),
            camera_lookat=(0.0, -0.1, 0.0),
            camera_up=(0.0, 1.0, 0.0),
            camera_fov=45,
        ),
        show_viewer=args.vis,
    )
    film = scene.add_entity(
        morph=gs.morphs.Particles(
            positions=positions,
        ),
        material=material,
        surface=surface,
    )
    # Inactive storage excludes the droplet from the reference film-only mass calibration.
    emitter = scene.add_emitter(material=material, max_particles=len(drop_positions), surface=surface)
    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(960, 720), pos=(3.2, 1.5, 4.2), lookat=(0.0, -0.1, 0.0), up=(0.0, 1.0, 0.0), fov=45
        )
    scene.build()
    film.fix_particles(rim_idx)
    emitter.entity.set_particles_pos(drop_positions)
    emitter.entity.set_particles_vel(0.0)
    emitter.entity.set_particles_active(True)
    gs.logger.info(f"Film: {film.n_particles}; droplet: {len(drop_positions)}; fixed rim: {len(rim_idx)}")
    if args.record or args.check or args.export_particles:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_writer = None
    if args.export_particles:
        checkpoint_writer = CheckpointWriter(scene, args.output_dir / "checkpoints", args.dt)
        np.savez_compressed(args.output_dir / "rim_indices.npz", rim_idx=rim_idx)
    if camera is not None:
        camera.start_recording(save_to_filename=str(args.output_dir / "membrane_filter.mp4"), fps=30)
    measurements = []
    sample_axis = np.arange(-1.0, 1.0 + radius, radius)
    sample_grid = np.stack(np.meshgrid(sample_axis, sample_axis, indexing="ij"), axis=-1).reshape(-1, 2)
    sample_grid = sample_grid[np.linalg.norm(sample_grid, axis=1) < 1.0 - 2 * radius]
    steps = math.ceil(args.seconds / args.dt)
    try:
        for step in range(steps + 1):
            if checkpoint_writer is not None:
                checkpoint_writer.write_frame()
            if args.check:
                film_positions = tensor_to_array(film.get_particles_pos())
                droplet_positions = tensor_to_array(emitter.entity.get_particles_pos())
                liquid_positions = np.concatenate((film_positions, droplet_positions))
                tree = cKDTree(liquid_positions)
                # Detached droplets must be excluded so their projection cannot hide an opening in the supported film.
                _, components = connected_components(
                    tree.sparse_distance_matrix(tree, 3 * radius, output_type="coo_matrix"), directed=False
                )
                is_supported = components == components[rim_idx[0]]
                supported_positions = liquid_positions[is_supported]
                gap = cKDTree(supported_positions[:, (0, 2)]).query(sample_grid)[0].max() / (2 * radius)
                separation = cKDTree(film_positions).query(droplet_positions)[0].min() / (2 * radius)
                connected_drop = np.count_nonzero(is_supported[film.n_particles :])
                measurements.append((step * args.dt, gap, separation, droplet_positions[:, 1].mean(), connected_drop))
                if step % max(1, round(1 / args.dt)) == 0 or step == steps:
                    np.savez_compressed(
                        args.output_dir / f"particles_{step:05d}.npz", film=film_positions, drop=droplet_positions
                    )
                    gs.logger.info(
                        f"t={step * args.dt:.3f}: coverage gap/diameter={gap:.3f}, drop separation/diameter={separation:.3f}"
                    )
                if gap > 1.0:
                    np.savez_compressed(
                        args.output_dir / "coverage_failure.npz", film=film_positions, drop=droplet_positions
                    )
                    raise RuntimeError(
                        f"The supported film has an opening larger than a particle diameter at t={step * args.dt}."
                    )
            if step == steps:
                break
            scene.step()
    finally:
        if camera is not None:
            camera.stop_recording()
        if args.check:
            with (args.output_dir / "measurements.csv").open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(
                    (
                        "time",
                        "coverage_gap_diameters",
                        "drop_separation_diameters",
                        "drop_mean_y",
                        "connected_drop_particles",
                    )
                )
                writer.writerows(measurements)
        if args.vis:
            scene.viewer.stop()


if __name__ == "__main__":
    main()
