"""Measure support settling and sampled mesh penetration without editing the particle trajectory."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import igl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


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
    records = []
    for i, points in enumerate(particles):
        sampled = points[indices].astype(np.float64)
        sd_dish, *_ = igl.signed_distance(sampled, dish.vertices, dish.faces)
        local = Rotation.from_quat(quats[i, [1, 2, 3, 0]]).inv().apply(points - positions[i])
        candidates = np.all((local >= blade.bounds[0]) & (local <= blade.bounds[1]), axis=1)
        sd_tool = np.ones(1)
        if candidates.any():
            sd_tool, *_ = igl.signed_distance(local[candidates], blade.vertices, blade.faces)
        low, high = np.quantile(points[bulk], (0.01, 0.99), axis=0)
        at_boundary = np.any((points <= boundary_lower + 1e-7) | (points >= boundary_upper - 1e-7), axis=1)
        touched_boundary |= at_boundary
        records.append(
            {
                "time_s": float(times[i]),
                "bulk_height_ratio": float((high[2] - low[2]) / (baseline[1, 2] - baseline[0, 2])),
                "bulk_width_ratio": float((high[1] - low[1]) / (baseline[1, 1] - baseline[0, 1])),
                "dish_penetration_max_m": float(max(0, -sd_dish.min())),
                "blade_penetration_max_m": float(max(0, -sd_tool.min())),
                "dish_inside_fraction": float(np.mean(sd_dish < -config["particle_size"] / 2)),
                "blade_inside_fraction": float(np.sum(sd_tool < -config["particle_size"] / 2) / len(points)),
                "blade_candidate_particles": int(candidates.sum()),
                "domain_boundary_particles": int(at_boundary.sum()),
            }
        )
        if i % 12 == 0:
            print(f"Audited {i + 1}/{len(times)}", flush=True)
    report = {
        "state_sha256": hashlib.sha256((run / "state.npz").read_bytes()).hexdigest(),
        "sampled_particles": len(indices),
        "all_particles_finite": bool(np.isfinite(particles).all()),
        "domain_boundary_unique_particles": int(touched_boundary.sum()),
        "frames": records,
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "Fixed-ID dish sample and all particles within the blade bounding box against exact meshes at saved frames; not continuous-time nonpenetration proof.",
    }
    (run / "asset-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"first": records[0], "last": records[-1]}, indent=2))


if __name__ == "__main__":
    main()
