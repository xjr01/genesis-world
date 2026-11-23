"""Audit particle trajectories against the configured spreading task."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from examples.butter_spreading.config import load_config
from examples.butter_spreading.motion import knife_pose


@dataclass(frozen=True)
class AuditReport:
    is_passed: bool
    is_full_trajectory: bool
    checks: dict[str, bool]
    state: str
    state_sha256: str
    initial_span_m: list[float]
    final_span_m: list[float]
    spread_area_ratio: float
    centroid_shift_m: float
    max_bread_center_drift_m: float
    early_contact_time_s: float
    early_bottom_butter_shift_m: list[float]
    early_bread_shift_m: list[float]
    early_butter_bread_slip_m: float
    residual_bread_indent_mm: float
    largest_butter_component_fraction: float
    minimum_largest_component_fraction: float
    final_component_count: int
    component_sizes: list[int]
    small_component_limit_particles: int
    spray_particle_count: int
    max_spray_particle_count: int
    deposited_butter_fraction: float
    off_bread_fraction: float
    deposited_x_span_m: float
    late_spread_advance_m: float
    particle_count: int
    bread_particle_count: int
    max_blade_intrusion_m: float
    max_centers_inside_blade: int
    frames: int


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="Recorded particle trajectory NPZ.")
    parser.add_argument("--output", type=Path, required=True, help="Physics audit JSON.")
    parser.add_argument(
        "--preview", action="store_true", help="Audit a short trajectory without final spread requirements."
    )
    args = parser.parse_args()
    with np.load(args.state) as state:
        p = state["particles"]
        bread = state["bread_particles"]
        knife = state["knife_positions"]
        velocities = state["particle_velocities"]
        times = state["time_s"]
        fps = int(state["fps"])
    metadata = json.loads(args.state.with_suffix(".json").read_text(encoding="utf-8"))
    config = load_config(args.state.parent / "run-config.json")
    completion = json.loads((args.state.parent / "completion.json").read_text(encoding="utf-8"))
    if p.ndim != 3 or bread.ndim != 3 or p.shape[2] != 3 or bread.shape[2] != 3:
        raise ValueError("Particle trajectories must have shape (frames, particles, 3).")
    if len(p) < 2 or len(bread) != len(p) or knife.shape != (len(p), 3) or times.shape != (len(p),):
        raise ValueError("Audit requires at least two synchronized trajectory frames.")
    if velocities.shape != p.shape:
        raise ValueError("Particle velocities must match the position trajectory.")
    if not all(np.isfinite(values).all() for values in (p, bread, knife, times, velocities)):
        raise ValueError("Trajectory contains non-finite values.")
    spacing = float(metadata["particle_spacing_m"])
    knife_size = np.asarray(metadata["knife_collider_size_m"], dtype=float)
    knife_xy_offset = np.asarray(metadata["knife_collider_xy_offset_m"], dtype=float)
    relative_to_knife = p - knife[:, None, :] - np.r_[knife_xy_offset, 0.0]
    under_blade = np.all(np.abs(relative_to_knife[:, :, :2]) <= knife_size[:2] / 2, axis=2)
    blade_bottom_intrusion = np.where(under_blade, relative_to_knife[:, :, 2] + knife_size[2] / 2, -np.inf)
    max_blade_intrusion_m = float(max(0.0, np.max(blade_bottom_intrusion)))
    max_centers_inside_blade = int(
        np.max(np.sum(under_blade & (np.abs(relative_to_knife[:, :, 2]) <= knife_size[2] / 2), axis=1))
    )

    initial_span = np.ptp(p[0, :, :2], axis=0)
    final_span = np.ptp(p[-1, :, :2], axis=0)
    centroid_shift = float(np.linalg.norm(p[-1].mean(axis=0) - p[0].mean(axis=0)))
    bread_center_xy = bread[:, :, :2].mean(axis=1)
    max_bread_drift_m = float(np.linalg.norm(bread_center_xy - bread_center_xy[0], axis=1).max())
    early_frame = int(np.argmin(np.abs(times - (0.9 + 0.1))))
    bottom_butter = p[0, :, 2] < p[0, :, 2].min() + spacing * 0.55
    bottom_butter_shift = (p[early_frame, bottom_butter, :2] - p[0, bottom_butter, :2]).mean(axis=0)
    bread_shift = bread_center_xy[early_frame] - bread_center_xy[0]
    early_butter_bread_slip_m = float(np.linalg.norm(bottom_butter_shift - bread_shift))
    top = bread[0, :, 2] >= bread[0, :, 2].max() - spacing * 0.55
    pressed = top & (bread[0, :, 0] > -0.065) & (bread[0, :, 0] < 0.025)
    pressed &= np.abs(bread[0, :, 1]) < 0.032
    residual_indent_mm = float(np.median(bread[0, pressed, 2] - bread[-1, pressed, 2]) * 1000)

    max_spray_particles = 0
    min_largest_component_fraction = 1.0
    small_component_limit = 64
    for frame in p:
        edge_i, edge_j = cKDTree(frame).query_pairs(2.5 * spacing, output_type="ndarray").T
        edges = coo_matrix(
            (np.ones(2 * len(edge_i), dtype=bool), (np.r_[edge_i, edge_j], np.r_[edge_j, edge_i])),
            shape=(len(frame), len(frame)),
        ).tocsr()
        component_count, labels = connected_components(edges, directed=False)
        component_sizes = np.bincount(labels)
        min_largest_component_fraction = min(min_largest_component_fraction, float(component_sizes.max() / len(frame)))
        max_spray_particles = max(
            max_spray_particles, int(component_sizes[component_sizes < small_component_limit].sum())
        )
    largest_fraction = float(component_sizes.max() / len(labels))
    spray_particles = int(component_sizes[component_sizes < small_component_limit].sum())
    final_bread_top = float(np.percentile(bread[-1, :, 2], 99.5))
    deposited = (p[-1, :, 2] < final_bread_top + 0.012) & (np.abs(p[-1, :, 0]) < 0.070) & (np.abs(p[-1, :, 1]) < 0.057)
    deposited_fraction = float(deposited.mean())
    off_bread_fraction = float(((np.abs(p[-1, :, 0]) > 0.070) | (np.abs(p[-1, :, 1]) > 0.057)).mean())
    deposited_x_span = float(np.ptp(np.percentile(p[-1, deposited, 0], [1, 99]))) if deposited.any() else 0.0
    late_begin = int(np.argmin(np.abs(times - (1.65 + 0.45 * (3.5 - 1.65)))))
    late_end = int(np.argmin(np.abs(times - (3.5 - 0.25))))
    late_spread_advance_m = float(np.percentile(p[late_end, :, 0], 99) - np.percentile(p[late_begin, :, 0], 99))

    checks = {
        "simulation_completed": completion["is_complete"] is True,
        "configuration_hash_matches": completion["configuration_sha256"]
        == hashlib.sha256((args.state.parent / "run-config.json").read_bytes()).hexdigest(),
        "expected_frame_count": len(p) >= 2 if args.preview else len(p) == round(4.0 * fps) + 1,
        "constant_particle_count": p.ndim == 3
        and p.shape[1] == metadata["butter_particle_count"]
        and bread.shape[1] == metadata["bread_particle_count"],
        "finite_state": bool(np.isfinite(p).all() and np.isfinite(bread).all() and np.isfinite(knife).all()),
        "full_time_coverage": args.preview or bool(abs(times[0]) < 1e-12 and abs(times[-1] - 4.0) <= config.dt),
        "declared_time_coverage": bool(
            metadata["frames"] == len(p) and abs(times[-1] - metadata["total_steps"] * config.dt) < config.dt * 1e-6
        ),
        "configuration_matches_state": bool(
            metadata["dt_s"] == config.dt
            and metadata["particle_spacing_m"] == config.butter_particle_size
            and metadata["bread_particle_spacing_m"] == config.bread_particle_size
        ),
        "ordered_times": bool(np.all(np.diff(times) > 0) and abs(fps - 24) < 1),
        "press_then_spread_trajectory": bool(
            np.allclose(knife, np.stack([knife_pose(t, config.motion) for t in times]), atol=1e-5)
        ),
        "butter_within_solver_domain": bool(
            (np.abs(p[:, :, 0]) < 0.112).all()
            and (np.abs(p[:, :, 1]) < 0.072).all()
            and (p[:, :, 2] > -0.012).all()
            and (p[:, :, 2] < 0.102).all()
        ),
        "bread_stays_on_support": max_bread_drift_m < 0.002,
        # Multiple cohesive deposits are allowed. Small isolated components at
        # any frame are the particle-spray artifact we need to reject.
        "no_particle_spray": max_spray_particles == 0,
        "butter_stays_cohesive": min_largest_component_fraction > 0.98,
        "mostly_deposited": args.preview or deposited_fraction > 0.75,
        "no_butter_over_bread_edge": args.preview or off_bread_fraction < 0.01,
        "meaningful_spread": args.preview or deposited_x_span > 0.06,
        "butter_advances_during_late_spread": args.preview or late_spread_advance_m > 0.01,
        "no_particle_anchors": metadata["particle_constraints"] is False,
        "knife_does_not_penetrate_butter": max_blade_intrusion_m < 0.0001,
    }
    report = AuditReport(
        is_passed=all(checks.values()),
        is_full_trajectory=not args.preview,
        checks=checks,
        state=str(args.state),
        state_sha256=hashlib.sha256(args.state.read_bytes()).hexdigest(),
        initial_span_m=initial_span.tolist(),
        final_span_m=final_span.tolist(),
        spread_area_ratio=float(np.prod(final_span / initial_span)),
        centroid_shift_m=centroid_shift,
        max_bread_center_drift_m=max_bread_drift_m,
        early_contact_time_s=float(times[early_frame]),
        early_bottom_butter_shift_m=bottom_butter_shift.tolist(),
        early_bread_shift_m=bread_shift.tolist(),
        early_butter_bread_slip_m=early_butter_bread_slip_m,
        residual_bread_indent_mm=residual_indent_mm,
        largest_butter_component_fraction=largest_fraction,
        minimum_largest_component_fraction=min_largest_component_fraction,
        final_component_count=int(component_count),
        component_sizes=sorted(component_sizes.tolist(), reverse=True),
        small_component_limit_particles=small_component_limit,
        spray_particle_count=spray_particles,
        max_spray_particle_count=max_spray_particles,
        deposited_butter_fraction=deposited_fraction,
        off_bread_fraction=off_bread_fraction,
        deposited_x_span_m=deposited_x_span,
        late_spread_advance_m=late_spread_advance_m,
        particle_count=int(p.shape[1]),
        bread_particle_count=int(bread.shape[1]),
        max_blade_intrusion_m=max_blade_intrusion_m,
        max_centers_inside_blade=max_centers_inside_blade,
        frames=len(p),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(json.dumps(asdict(report), indent=2))
    if not report.is_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
