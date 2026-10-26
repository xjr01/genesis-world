"""Measure support settling and sampled mesh penetration without editing the particle trajectory."""

import argparse
import json
import time
from pathlib import Path

import igl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from examples.butter_cut_spread.asset_runtime import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    with np.load(run / "state.npz") as data:
        particles, times, positions, quats = (
            data[k] for k in ("particles", "time_s", "knife_positions", "knife_quaternions")
        )
    dish = trimesh.load_mesh(run / "dish.obj", process=True)
    blade = trimesh.load_mesh(run / "blade.obj", process=True)
    indices = np.arange(0, particles.shape[1], max(1, particles.shape[1] // 5000))
    bulk = particles[0, :, 0] < config["butter_center"][0] + config["cut_offset"] - 0.002
    baseline = np.quantile(particles[0, bulk], (0.01, 0.99), axis=0)
    boundary_lower = np.array(config["lower_bound"]) + 3 / config["grid_density"]
    boundary_upper = np.array(config["upper_bound"]) - 3 / config["grid_density"]
    touched_boundary = np.zeros(particles.shape[1], dtype=bool)
    # Both reference meshes are fixed in their own frames. Query all recorded
    # points together so libigl builds each mesh acceleration structure once.
    sampled = particles[:, indices].reshape(-1, 3).astype(np.float64)
    sd_dish, *_ = igl.signed_distance(sampled, dish.vertices, dish.faces)
    sd_dish = sd_dish.reshape(len(times), len(indices))
    tool_queries = []
    records = []
    for i, points in enumerate(particles):
        local = Rotation.from_quat(quats[i, [1, 2, 3, 0]]).inv().apply(points - positions[i])
        candidates = np.all((local >= blade.bounds[0]) & (local <= blade.bounds[1]), axis=1)
        tool_queries.append(local[candidates])
        low, high = np.quantile(points[bulk], (0.01, 0.99), axis=0)
        at_boundary = np.any((points <= boundary_lower + 1e-7) | (points >= boundary_upper - 1e-7), axis=1)
        touched_boundary |= at_boundary
        records.append(
            {
                "time_s": float(times[i]),
                "bulk_height_ratio": float((high[2] - low[2]) / (baseline[1, 2] - baseline[0, 2])),
                "bulk_width_ratio": float((high[1] - low[1]) / (baseline[1, 1] - baseline[0, 1])),
                "dish_penetration_max_m": float(max(0, -sd_dish[i].min())),
                "dish_inside_fraction": float(np.mean(sd_dish[i] < -config["particle_size"] / 2)),
                "blade_candidate_particles": int(candidates.sum()),
                "domain_boundary_particles": int(at_boundary.sum()),
            }
        )
        if i % 12 == 0:
            print(f"Audited {i + 1}/{len(times)}", flush=True)
    counts = [len(queries) for queries in tool_queries]
    offsets = np.r_[0, np.cumsum(counts)]
    distances = np.empty(0)
    if offsets[-1] > 0:
        distances, *_ = igl.signed_distance(np.concatenate(tool_queries), blade.vertices, blade.faces)
    for i, record in enumerate(records):
        distance = distances[offsets[i] : offsets[i + 1]]
        record["blade_penetration_max_m"] = float(max(0, -distance.min())) if len(distance) else 0.0
        record["blade_inside_fraction"] = float(np.sum(distance < -config["particle_size"] / 2) / particles.shape[1])
    report = {
        "state_sha256": file_digest(run / "state.npz"),
        "sampled_particles": len(indices),
        "all_particles_finite": bool(np.isfinite(particles).all()),
        "domain_boundary_unique_particles": int(touched_boundary.sum()),
        "frames": records,
        "elapsed_seconds": time.perf_counter() - started,
        "mesh_distance_batches": 2 if offsets[-1] else 1,
        "scope": "Fixed-ID dish sample and all particles within the blade bounding box against exact meshes at saved frames; not continuous-time nonpenetration proof.",
    }
    (run / "asset-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"first": records[0], "last": records[-1]}, indent=2))


if __name__ == "__main__":
    main()
