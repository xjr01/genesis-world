"""Audit continuous particle identity, pickup, deposition and aligned asset contact."""

import argparse
import hashlib
import json
from pathlib import Path

import av
import igl
import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed aligned simulation directory.")
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    assets = json.loads((run / "assets.json").read_text())
    simulation = json.loads((run / "simulation.json").read_text())
    with np.load(run / "state.npz") as archive:
        particles, bread, times, centers, quaternions = (
            archive[key] for key in ("particles", "bread_particles", "time_s", "knife_positions", "knife_quaternions")
        )
    slice_ids = particles[0, :, 0] > config["butter_center"][0] + config["cut_offset"] + config["blade_size"][2] / 2
    lift_index = np.argmin(abs(times - 3.7))
    original_top = np.max(particles[0, :, 2])
    pickup_fraction = float(np.mean(particles[lift_index, slice_ids, 2] > original_top + 0.005))
    bounds = np.array(assets["bread_bounds_m"])
    deposited = particles[-1, slice_ids]
    inside_xy = np.all((deposited[:, :2] >= bounds[0, :2]) & (deposited[:, :2] <= bounds[1, :2]), axis=1)
    distance, _ = cKDTree(bread[-1]).query(deposited)
    deposited_mask = inside_xy & (distance < 0.006)
    deposition_fraction = float(np.mean(deposited_mask))
    approach_index = np.argmin(abs(times - 6.0))
    approach = particles[approach_index, slice_ids][deposited_mask]
    approach_span = np.diff(np.quantile(approach[:, 0], [0.05, 0.95]))[0] if len(approach) else 0.0
    deposited_span = (
        np.diff(np.quantile(deposited[deposited_mask, 0], [0.05, 0.95]))[0] if deposited_mask.any() else 0.0
    )
    spread_gain = float(deposited_span - approach_span)
    approach_front = np.quantile(approach[:, 0], 0.90) if len(approach) else np.inf
    advanced_fraction = float(np.mean(deposited_mask & (deposited[:, 0] > approach_front + 0.005)))
    half = np.array(config["blade_size"]) / 2
    previous_local = None
    face_origin = face_spacing = face_distances = None
    if (run / "blade-face.npz").is_file():
        with np.load(run / "blade-face.npz") as archive:
            face_origin, face_spacing, face_distances = (archive[name] for name in ("origin", "spacing", "distance"))
    crossings = []
    board = trimesh.load_mesh(run / "board-collision.obj", process=False)
    bread_sample_ids = np.arange(0, bread.shape[1], max(1, bread.shape[1] // 3000))
    board_penetrations = []
    for index, points in enumerate(particles):
        local = Rotation.from_quat(quaternions[index, [1, 2, 3, 0]]).inv().apply(points - centers[index])
        if previous_local is not None:
            inside = np.abs(local[:, 2]) < half[2] + 0.001
            inside &= np.abs(previous_local[:, 2]) < half[2] + 0.001
            if face_distances is None:
                inside &= np.all(np.abs(local[:, :2]) < half[:2] - 0.0015, axis=1)
                inside &= np.all(np.abs(previous_local[:, :2]) < half[:2] - 0.0015, axis=1)
            else:
                near_ids = np.flatnonzero(inside)
                current_distance = map_coordinates(
                    face_distances,
                    ((local[near_ids, :2] - face_origin) / face_spacing).T,
                    order=1,
                    mode="constant",
                    cval=-0.001,
                )
                previous_distance = map_coordinates(
                    face_distances,
                    ((previous_local[near_ids, :2] - face_origin) / face_spacing).T,
                    order=1,
                    mode="constant",
                    cval=-0.001,
                )
                inside[near_ids] &= (current_distance > 0.0015) & (previous_distance > 0.0015)
            crossings.append(int(np.sum(inside & (local[:, 2] * previous_local[:, 2] < 0))))
        previous_local = local
        board_distance, *_ = igl.signed_distance(
            bread[index, bread_sample_ids].astype(np.float64), board.vertices, board.faces
        )
        board_penetrations.append(float(max(0, -board_distance.min())))
    with av.open(str(run / "aligned-cut-spread-particles.mp4")) as video:
        video_frames = sum(1 for _ in video.decode(video=0))
    cp_indices = simulation["cpic_cutting_geom_indices"]
    is_tool_only = all(
        sum(row["counts"]) == sum(row["counts"][i] for i in cp_indices) for row in simulation["cpic_separation_checks"]
    )
    is_finite = bool(np.isfinite(particles).all() and np.isfinite(bread).all())
    report = {
        "state_sha256": hashlib.sha256((run / "state.npz").read_bytes()).hexdigest(),
        "is_complete": simulation["is_complete"],
        "all_particles_active": simulation["all_particles_active"],
        "all_particles_finite": is_finite,
        "frames": len(times),
        "particle_video_frames": video_frames,
        "slice_particle_count": int(slice_ids.sum()),
        "pickup_fraction": pickup_fraction,
        "deposition_fraction": deposition_fraction,
        "approach_deposited_slice_span_x_m": float(approach_span),
        "final_deposited_slice_span_x_m": float(deposited_span),
        "spread_span_gain_m": spread_gain,
        "advanced_deposition_fraction": advanced_fraction,
        "cpic_tools_only": is_tool_only,
        "sampled_interior_side_changes": sum(crossings),
        "authored_face_footprint": face_distances is not None,
        "side_changes_per_interval": crossings,
        "bread_max_displacement_m": float(np.linalg.norm(bread[-1] - bread[0], axis=1).max()),
        "bread_minimum_height_m": float(bread[:, :, 2].min()),
        "bread_board_sample_count": len(bread_sample_ids),
        "bread_board_max_penetration_m": max(board_penetrations),
        "scope": "Persistent particle IDs at saved frames. Side-change count is diagnostic; it does not resolve motion between saved frames.",
    }
    report["is_passed"] = bool(
        simulation["is_complete"]
        and simulation["all_particles_active"]
        and is_finite
        and is_tool_only
        and video_frames == len(times)
        and pickup_fraction > 0.95
        and deposition_fraction > 0.5
        and spread_gain > 0.010
        and advanced_fraction > 0.15
        and max(board_penetrations) < config["bread_particle_size"]
    )
    (run / "aligned-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "side_changes_per_interval"}, indent=2))
    if not report["is_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
