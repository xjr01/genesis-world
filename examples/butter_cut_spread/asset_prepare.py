"""Reconstruct recorded asset-case particles and clip visual kernels against actual solid meshes."""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from examples.butter_cut_spread.blade_film import BladeFilm
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
    film = BladeFilm(run, config["particle_size"], config["blade_size"][2])
    film_output = run / "blade-film"
    film_output.mkdir(exist_ok=True)
    volumes = []
    try:
        for index in indices:
            points = particles[index]
            partition = film.partition(points, positions[index], quats[index])
            weights = partition.bulk_weights
            np.savez(
                film_output / f"{index:04d}.npz",
                opacity=film.opacity(partition.thickness),
                counts=partition.counts,
                volumes_m3=partition.volumes,
                sheet_volumes_m3=partition.thickness.sum(axis=(1, 2)) * film.settings.pixel**2,
                bulk_volume_m3=weights.sum() * config["particle_size"] ** 3,
            )
            groups = [(points, weights, bulk_voxel, 1.0)]
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
                            (left, weights[~near], bulk_voxel, 1.0),
                            (right, weights[near], coating_voxel, coating_voxel / config["particle_size"]),
                        ]
            separated_groups = []
            for group, group_weights, voxel, normal_floor in groups:
                has_bulk = group_weights > 0
                group, group_weights = group[has_bulk], group_weights[has_bulk]
                if not len(group):
                    continue
                if coating is not None:
                    top = coating.bread_center[2] + coating.bread_size[2] / 2
                    is_high = group[:, 2] > top + 0.015
                    if is_high.any() and (~is_high).any():
                        gap = group[is_high, 2].min() - group[~is_high, 2].max()
                        if gap > 4 * (width + config["particle_size"]):
                            separated_groups.extend(
                                (group[mask], group_weights[mask], voxel, normal_floor) for mask in (is_high, ~is_high)
                            )
                            continue
                separated_groups.append((group, group_weights, voxel, normal_floor))
            meshes = []
            for group, group_weights, voxel, normal_floor in separated_groups:
                padding = max(2 * width, 1.5 * (width + config["particle_size"])) if coating else 2 * width
                origin = np.floor((group.min(0) - padding) / voxel) * voxel
                shape = tuple(np.ceil((group.max(0) + padding - origin) / voxel).astype(int) + 1)
                density = (
                    splat_density(
                        group, origin, shape, voxel, width, config["particle_size"] ** 3, particle_weights=group_weights
                    )
                    if coating is None
                    else coating_density(group, origin, shape, voxel, width, coating, normal_floor, group_weights)
                )
                if density.max() > 0.5:
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
    particle_count = particles.shape[1]
    config = json.loads((run / "config.json").read_text())
    film = BladeFilm(run, config["particle_size"], config["blade_size"][2])
    (run / "blade-film").mkdir(exist_ok=True)
    np.savez(run / "blade-film" / "geometry.npz", vertices=film.vertices, faces=film.faces)
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
    film_files = ["geometry.npz", *(f"{index:04d}.npz" for index in indices)]
    film_hashes = {name: hashlib.sha256((run / "blade-film" / name).read_bytes()).hexdigest() for name in film_files}
    partition_errors, sheet_errors = [], []
    for index in indices:
        with np.load(run / "blade-film" / f"{index:04d}.npz") as data:
            total_volume = particle_count * config["particle_size"] ** 3
            partition_errors.append(abs((data["bulk_volume_m3"] + data["volumes_m3"].sum()) / total_volume - 1))
            sheet_errors.append(float(np.max(abs(data["sheet_volumes_m3"] - data["volumes_m3"]))))
            opacity = data["opacity"]
            if (
                opacity.shape != (2, len(film.vertices[0]))
                or not np.isfinite(opacity).all()
                or np.any((opacity < 0) | (opacity > 1))
            ):
                raise ValueError("Blade film opacity has invalid geometry correspondence or values.")
    if max(partition_errors) > 1e-12 or max(sheet_errors) > 1e-15:
        raise ValueError("Complementary blade-film volume accounting failed.")
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
                "film_files_sha256": film_hashes,
                "film_settings": asdict(film.settings),
                "film_partition_max_relative_error": float(max(partition_errors)),
                "film_sheet_max_volume_error_m3": max(sheet_errors),
                "blade_film": "Complementary particle-volume partition on both local blade faces.",
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
