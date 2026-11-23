"""Measure fixed-particle support settling and end-slice transport from recorded asset runs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    simulation = json.loads((run / "simulation.json").read_text())
    with np.load(run / "state.npz") as data:
        points, times, centers, quats = (
            data[k] for k in ("particles", "time_s", "knife_positions", "knife_quaternions")
        )
    initial = points[0]
    core = (abs(initial[:, 0] - config["butter_center"][0]) < 0.02) & (
        abs(initial[:, 1] - config["butter_center"][1]) < 0.004
    )
    core_height = np.diff(np.quantile(initial[core, 2], (0.01, 0.99)))[0]
    upper = core & (initial[:, 2] > np.quantile(initial[core, 2], 0.9))
    cut_x = config["butter_center"][0] + config["cut_offset"]
    slice_ids = initial[:, 0] > cut_x + config["blade_size"][2] / 2
    surface = config["butter_center"][2] + config["butter_size"][2] / 2
    records = []
    previous_local = None
    for i, frame in enumerate(points):
        if not np.isfinite(frame).all():
            raise RuntimeError(f"Nonfinite particles at frame {i}")
        height = np.diff(np.quantile(frame[core, 2], (0.01, 0.99)))[0]
        local = Rotation.from_quat(quats[i, [1, 2, 3, 0]]).inv().apply(frame[slice_ids] - centers[i])
        records.append(
            {
                "time_s": float(times[i]),
                "core_height_loss_percent": float(100 * (1 - height / core_height)),
                "upper_core_drop_mm": float(1000 * np.median(initial[upper, 2] - frame[upper, 2])),
                "core_width_mm": float(1000 * np.diff(np.quantile(frame[core, 1], (0.01, 0.99)))[0]),
                "slice_lifted_fraction": float(np.mean(frame[slice_ids, 2] > surface)),
                "slice_relative_speed_p50_p90_mm_s": [0.0, 0.0]
                if i == 0
                else (
                    1000
                    * np.quantile(
                        np.linalg.norm(local - previous_local, axis=1) / (times[i] - times[i - 1]), (0.5, 0.9)
                    )
                ).tolist(),
                "minimum_z_m": float(frame[:, 2].min()),
            }
        )
        previous_local = local
    tail = [record for record in records if record["time_s"] >= times[-1] / 2]
    slope = (
        float(
            np.polyfit(
                [record["time_s"] for record in tail], [record["core_height_loss_percent"] for record in tail], 1
            )[0]
        )
        if len(tail) > 1
        else None
    )
    report = {
        "material": simulation["material"],
        "stationary": simulation["stationary"],
        "frames": records,
        "late_height_loss_percent_per_s": slope,
        "static_screen_pass": bool(records[-1]["core_height_loss_percent"] <= 1.0 and abs(slope) <= 0.1)
        if simulation["stationary"] and slope is not None
        else None,
        "scope": "Fixed initial core/slice IDs. Relative speeds are finite differences between saved frames in the tool frame. "
        "Static screen requires <=1% height loss and <=0.1 percentage points/s late drift; these are engineering targets, "
        "not experimentally measured butter properties. A short hold cannot establish long-term creep stability.",
    }
    (run / "material-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"run": str(run), "last": records[-1], "late_drift": slope, "pass": report["static_screen_pass"]}))


if __name__ == "__main__":
    main()
