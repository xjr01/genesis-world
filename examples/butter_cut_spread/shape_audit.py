"""Measure slice deformation after removing rigid motion at saved phase checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np


def deformation(source, target):
    """Return RMS and 99th-percentile residual distances after a proper rigid fit."""
    centered_source = source - source.mean(axis=0)
    centered_target = target - target.mean(axis=0)
    left, _, right = np.linalg.svd(centered_source.T @ centered_target)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(left @ right))
    residual = np.linalg.norm(centered_source @ (left @ correction @ right) - centered_target, axis=1)
    return np.sqrt(np.mean(residual**2)), np.quantile(residual, 0.99)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    initial = np.load(run / "butter-particles.npy")
    selected = initial[:, 0] > config["butter_center"][0] + config["cut_offset"] + config["blade_size"][2] / 2
    records = []
    for path in sorted(run.glob("checkpoint-*.npz")):
        with np.load(path) as checkpoint:
            rms, p99 = deformation(initial[selected], checkpoint["particles"][selected])
            records.append(
                {
                    "phase": path.stem,
                    "time_s": float(checkpoint["time_s"]),
                    "slice_rms_m": float(rms),
                    "slice_p99_m": float(p99),
                }
            )
    report = {
        "butter_particles": len(initial),
        "slice_particles": int(selected.sum()),
        "sampled_volume_m3": len(initial) * config["particle_size"] ** 3,
        "phases": sorted(records, key=lambda record: record["time_s"]),
    }
    (run / "shape-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
