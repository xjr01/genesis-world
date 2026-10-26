"""Reconstruct recorded asset-case particles and clip visual kernels against actual solid meshes."""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from examples.butter_cut_spread.config import Config
from examples.butter_cut_spread.density import splat_density
from examples.butter_cut_spread.replay_frames import replay_indices
from examples.butter_cut_spread.surface import closed_density_surface, coating_density


@dataclass(frozen=True)
class ParticleBuffer:
    name: str
    shape: tuple[int, ...]
    dtype: str


def reconstruct_frames(run, particles, positions, quats, indices, coating_voxel):
    """Reconstruct disjoint frames without changing any spatial or temporal filtering."""
    started = time.perf_counter()
    config = json.loads((run / "config.json").read_text())
    assets = json.loads((run / "assets.json").read_text())
    coating = (
        Config(
            particle_size=config["particle_size"],
            bread_center=tuple(config["bread_center"]),
            bread_size=tuple(config["bread_size"]),
        )
        if assets.get("is_cut_and_spread", False)
        else None
    )
    dish = trimesh.load_mesh(run / assets.get("dish_collision_file", "dish.obj"), process=True)
    blade = trimesh.load_mesh(run / "blade.obj", process=True)
    output = run / "surfaces"
    output.mkdir(exist_ok=True)
    bulk_voxel, width = 0.0003, 0.00085
    shared = None
    if isinstance(particles, ParticleBuffer):
        shared = SharedMemory(name=particles.name)
        particles = np.ndarray(particles.shape, dtype=particles.dtype, buffer=shared.buf)
    volumes = []
    try:
        for index in indices:
            points = particles[index]
            groups = [(points, bulk_voxel, 1.0)]
            if coating is not None and coating_voxel < bulk_voxel:
                bread_left = config["bread_center"][0] - config["bread_size"][0] / 2
                bread_top = config["bread_center"][2] + config["bread_size"][2] / 2
                separator = (bread_left + config["butter_center"][0] + config["butter_size"][0] / 2) / 2
                near = points[:, 0] > separator
                if near.any() and (~near).any():
                    right, left = points[near], points[~near]
                    if (
                        right[:, 0].min() - left[:, 0].max() > 4 * width
                        and np.quantile(right[:, 2], 0.95) < bread_top + 0.025
                    ):
                        groups = [
                            (left, bulk_voxel, 1.0),
                            (right, coating_voxel, coating_voxel / config["particle_size"]),
                        ]
            meshes = []
            for group, voxel, normal_floor in groups:
                padding = max(2 * width, 1.5 * (width + config["particle_size"])) if coating else 2 * width
                origin = np.floor((group.min(0) - padding) / voxel) * voxel
                shape = tuple(np.ceil((group.max(0) + padding - origin) / voxel).astype(int) + 1)
                density = (
                    splat_density(group, origin, shape, voxel, width, config["particle_size"] ** 3)
                    if coating is None
                    else coating_density(group, origin, shape, voxel, width, coating, normal_floor)
                )
                meshes.append(closed_density_surface(density, origin, voxel).mesh)
            surface = trimesh.util.concatenate(meshes)
            tool = blade.copy()
            tool.vertices = Rotation.from_quat(quats[index, [1, 2, 3, 0]]).apply(blade.vertices) + positions[index]
            surface = trimesh.boolean.difference((surface, dish, tool), engine="manifold", check_volume=True)
            surface.export(output / f"{index:04d}.ply")
            volumes.append((index, float(surface.volume)))
            if index % 6 == 0:
                print(f"Surface {index + 1}/{len(particles)} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    finally:
        if shared is not None:
            shared.close()
    return volumes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(6, os.cpu_count() or 1),
        help="Independent frame workers (up to six by default); faster reconstruction uses more RAM.",
    )
    parser.add_argument(
        "--coating-voxel",
        type=float,
        default=0.0003,
        help="Local supported-coating resolution in meters; smaller values resolve thinner films at higher cost.",
    )
    parser.add_argument("--preview", action="store_true", help="Reconstruct only the five Isaac preview frames.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Worker count must be positive.")
    if not 0 < args.coating_voxel <= 0.0003:
        parser.error("Coating voxel must be positive and no larger than the bulk voxel (0.3 mm).")
    run = args.run.resolve()
    (run / "surface-report.json").unlink(missing_ok=True)
    started = time.perf_counter()
    assets = json.loads((run / "assets.json").read_text())
    with np.load(run / "state.npz") as data:
        particles, times, positions, quats = (
            data[k] for k in ("particles", "time_s", "knife_positions", "knife_quaternions")
        )
    indices = replay_indices(len(times), args.preview)
    workers = min(args.workers, len(indices))
    if workers == 1:
        records = reconstruct_frames(run, particles, positions, quats, indices, args.coating_voxel)
    else:
        shared = SharedMemory(create=True, size=particles.nbytes)
        try:
            descriptor = ParticleBuffer(shared.name, particles.shape, particles.dtype.str)
            np.ndarray(particles.shape, dtype=particles.dtype, buffer=shared.buf)[:] = particles
            del particles
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        reconstruct_frames,
                        run,
                        descriptor,
                        positions,
                        quats,
                        indices[i::workers],
                        args.coating_voxel,
                    )
                    for i in range(workers)
                ]
                records = [record for future in futures for record in future.result()]
        finally:
            shared.close()
            shared.unlink()
    volumes = [volume for _, volume in sorted(records)]
    with (run / "state.npz").open("rb") as source:
        state_hash = hashlib.file_digest(source, "sha256").hexdigest()
    (run / "surface-report.json").write_text(
        json.dumps(
            {
                "frames": len(indices),
                "frame_indices": indices,
                "is_complete": len(indices) == len(times),
                "elapsed_seconds": time.perf_counter() - started,
                "state_sha256": state_hash,
                "volumes_m3": volumes,
                "supported_coating_reconstruction": assets.get("is_cut_and_spread", False),
                "workers": workers,
                "coating_voxel_m": args.coating_voxel,
                "dish_clip_mesh": assets.get("dish_collision_file", "dish.obj"),
                "scope": "Particle-derived surfaces; Boolean clipping removes visual kernel overlap with the validated support collider and steel.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
