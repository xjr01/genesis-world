"""Prepare an independently renderable copy of an accepted spreading trajectory."""

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from examples.butter_spreading.render_geometry import bind_bread, deform_bread


@dataclass(frozen=True)
class RenderInputs:
    state_sha256: str
    surface_sha256: str
    bread_sha256: str
    frames: int
    fps: int
    support_top_m: float
    support_size_m: tuple[float, float, float]
    blade_offset_m: tuple[float, float, float]
    blade_size_m: tuple[float, float, float]
    max_butter_mesh_blade_penetration_m: float
    max_bread_surface_displacement_m: float
    blade_film_particle_count: list[int]
    blade_film_status: str
    preparation_seconds: float


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Accepted spreading pipeline output.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, run = args.source.resolve(), args.output_dir.resolve()
    if run.exists() and any(run.iterdir()):
        parser.error("Use an empty output directory; source simulations are never overwritten.")
    started = time.perf_counter()
    completion = json.loads((source / "pipeline-completion.json").read_text(encoding="utf-8"))
    audit = json.loads((source / "physics-audit.json").read_text(encoding="utf-8"))
    if (
        not all(completion[key] for key in ("is_complete", "is_full_trajectory", "is_accepted"))
        or not audit["is_passed"]
    ):
        raise ValueError("Rendering requires a complete accepted physical trajectory.")
    state_hash = hashlib.sha256((source / "mpm-state.npz").read_bytes()).hexdigest()
    surface_hash = hashlib.sha256((source / "mpm-surface.npz").read_bytes()).hexdigest()
    if state_hash != completion["state_sha256"] or state_hash != audit["state_sha256"]:
        raise ValueError("State hashes disagree.")
    if surface_hash != completion["artifact_sha256"]["mpm-surface.npz"]:
        raise ValueError("Surface hash differs from the accepted pipeline.")
    meta = json.loads((source / "mpm-state.json").read_text(encoding="utf-8"))
    with np.load(source / "mpm-state.npz") as state:
        bread, particles, knives, times = (
            state[key] for key in ("bread_particles", "particles", "knife_positions", "time_s")
        )
        fps, knife_size = int(state["fps"]), state["knife_size"]
    with np.load(source / "mpm-surface.npz") as surface:
        if str(surface["state_sha256"]) != state_hash or len(surface["vertex_offsets"]) != len(times) + 1:
            raise ValueError("Surface source or frame count differs from the trajectory.")
        vertices, offsets = surface["vertices"], surface["vertex_offsets"]
    blade_offset = (*meta["knife_collider_xy_offset_m"], 0.0)
    centers = knives + blade_offset
    penetration = 0.0
    film_counts = []
    for index, center in enumerate(centers):
        gap = np.abs(vertices[offsets[index] : offsets[index + 1]] - center) - knife_size / 2
        penetration = max(penetration, float(-gap.max(axis=1).min()))
        particle_gap = np.abs(particles[index] - center) - knife_size / 2
        film_counts.append(
            int(
                np.count_nonzero(
                    (particle_gap[:, :2] < 0).all(axis=1)
                    & (particle_gap[:, 2] >= 0)
                    & (particle_gap[:, 2] < meta["particle_spacing_m"])
                )
            )
        )
    if penetration > 1e-7:
        raise ValueError("The recorded reconstructed surface intersects the physical blade; inspect before rendering.")
    if max(film_counts) > 0:
        raise ValueError(
            "This trajectory has unresolved blade-film candidates; contact tracking must be adapted first."
        )
    binding = bind_bread(bread[0], meta["bread_particle_spacing_m"])
    moved, normals = [], []
    max_displacement = 0.0
    for index, frame in enumerate(bread):
        points, directions = deform_bread(binding, bread[0], frame)
        max_displacement = max(max_displacement, float(np.linalg.norm(points - binding.vertices, axis=1).max()))
        moved.append(points.astype(np.float32))
        normals.append(directions.astype(np.float32))
        if index % 24 == 0:
            print(f"Bread surface: {index + 1}/{len(times)}", flush=True)
    run.mkdir(parents=True, exist_ok=True)
    for name in (
        "mpm-state.npz",
        "mpm-state.json",
        "mpm-surface.npz",
        "mpm-surface.json",
        "run-config.json",
        "completion.json",
        "pipeline-completion.json",
        "physics-audit.json",
    ):
        shutil.copy2(source / name, run / name)
    np.savez_compressed(
        run / "bread-surface.npz",
        vertices=np.stack(moved),
        normals=np.stack(normals),
        faces=binding.faces.astype(np.int32),
        colors=binding.colors.astype(np.float32),
        time_s=times,
        state_sha256=state_hash,
    )
    report = RenderInputs(
        state_hash,
        surface_hash,
        hashlib.sha256((run / "bread-surface.npz").read_bytes()).hexdigest(),
        len(times),
        fps,
        0.01,
        (0.18, 0.14, 0.02),
        blade_offset,
        tuple(knife_size),
        penetration,
        max_displacement,
        film_counts,
        "No blade-contact thin-layer particles; coverage inactive, no synthetic attachment added.",
        time.perf_counter() - started,
    )
    (run / "isaac-inputs.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
