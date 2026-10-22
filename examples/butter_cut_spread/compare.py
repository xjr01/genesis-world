"""Compare all saved material-point frames against an explicitly supplied reference run."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from examples.butter_cut_spread.config import Config


@dataclass(frozen=True)
class TrajectoryDifference:
    is_identical: bool
    rms_m: float
    max_frame_rms_m: float
    max_frame_p99_m: float
    max_m: float


@dataclass(frozen=True)
class ComparisonReport:
    is_accepted: bool
    state_sha256: str
    reference_state_sha256: str
    frames: int
    butter: TrajectoryDifference
    bread: TrajectoryDifference
    max_blade_position_difference_m: float
    max_blade_quaternion_difference: float
    scope: str


def trajectory_difference(actual, reference):
    """Compare persistent particle identities without allocating a second full trajectory."""
    if actual.shape != reference.shape:
        raise ValueError("Reference particle identities or frame counts differ.")
    square_sum, maximum, frame_rms, frame_p99 = 0.0, 0.0, 0.0, 0.0
    for current, original in zip(actual, reference):
        distance = np.linalg.norm(current - original, axis=-1)
        if not np.isfinite(distance).all():
            raise ValueError("Nonfinite reference comparison.")
        square_sum += np.sum(distance**2, dtype=np.float64)
        maximum = max(maximum, distance.max())
        frame_rms = max(frame_rms, np.sqrt(np.mean(distance**2)))
        frame_p99 = max(frame_p99, np.quantile(distance, 0.99))
    return TrajectoryDifference(
        bool(maximum == 0),
        float(np.sqrt(square_sum / (actual.shape[0] * actual.shape[1]))),
        float(frame_rms),
        float(frame_p99),
        float(maximum),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Run to compare.")
    parser.add_argument(
        "--reference", type=Path, required=True, help="Independent reference run with the same configuration."
    )
    args = parser.parse_args()
    run, reference = args.run.resolve(), args.reference.resolve()
    config = Config(**json.loads((run / "config.json").read_text()))
    if asdict(config) != asdict(Config(**json.loads((reference / "config.json").read_text()))):
        raise ValueError("Simulation configurations differ; this is not a reproduction comparison.")
    with np.load(run / "state.npz") as current, np.load(reference / "state.npz") as original:
        if not np.array_equal(current["time_s"], original["time_s"]):
            raise ValueError("Reference sampling times differ.")
        frames = len(current["time_s"])
        butter = trajectory_difference(current["particles"], original["particles"])
        bread = trajectory_difference(current["bread_particles"], original["bread_particles"])
        position = float(np.max(np.linalg.norm(current["knife_positions"] - original["knife_positions"], axis=-1)))
        quaternion = float(np.max(abs(current["knife_quaternions"] - original["knife_quaternions"])))
    is_accepted = bool(
        butter.max_frame_rms_m < config.particle_size / 4
        and butter.max_frame_p99_m < config.particle_size
        and bread.max_m < config.bread_particle_size / 2
        and position < config.particle_size / 20
        and quaternion < 1e-6
    )
    report = ComparisonReport(
        is_accepted,
        hashlib.sha256((run / "state.npz").read_bytes()).hexdigest(),
        hashlib.sha256((reference / "state.npz").read_bytes()).hexdigest(),
        frames,
        butter,
        bread,
        position,
        quaternion,
        "All recorded frames and persistent particle IDs are compared. Reproduction tolerances: maximum per-frame "
        "butter RMS below 0.25 particle spacing, per-frame p99 below one spacing, maximum bread error below half "
        "its spacing. Rare individual butter deviations are reported separately, not hidden by a claim of bitwise identity.",
    )
    (run / "comparison.json").write_text(json.dumps(asdict(report), indent=2))
    print(json.dumps(asdict(report), indent=2), flush=True)
    if not is_accepted:
        raise RuntimeError("Trajectory reproduction failed; inspect comparison.json.")


if __name__ == "__main__":
    main()
