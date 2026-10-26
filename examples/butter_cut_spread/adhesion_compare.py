"""Compare complete cutting and spreading runs against explicit macroscopic acceptance limits."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    run, reference = args.run.resolve(), args.reference.resolve()
    reports = []
    reference_bread = None
    bread_difference_p99 = 0.0
    bread_center_difference = 0.0
    for folder in (reference, run):
        audit = json.loads((folder / "aligned-audit.json").read_text())
        shape = json.loads((folder / "shape-audit.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        initial = np.load(folder / "butter-particles.npy")
        with np.load(folder / "checkpoint-hold.npz") as data:
            final, bread = data["particles"], data["bread_particles"]
        if reference_bread is None:
            reference_bread = bread
        else:
            bread_difference_p99 = float(np.quantile(np.linalg.norm(bread - reference_bread, axis=1), 0.99))
            bread_center_difference = float(np.linalg.norm(bread.mean(axis=0) - reference_bread.mean(axis=0)))
        cut = config["butter_center"][0] + config["cut_offset"]
        is_stock = initial[:, 0] < cut - 2 * config["particle_size"]
        displacement = np.linalg.norm(final[is_stock] - initial[is_stock], axis=1)
        is_slice = initial[:, 0] > cut + config["blade_size"][2] / 2
        distance, _ = cKDTree(bread).query(final[is_slice])
        deposited = final[is_slice][distance < 0.006]
        reports.append(
            {
                "state_sha256": audit["state_sha256"],
                "audit_passed": audit["is_passed"],
                "pickup_fraction": audit["pickup_fraction"],
                "deposition_fraction": audit["deposition_fraction"],
                "spread_span_m": audit["final_deposited_slice_span_x_m"],
                "spread_gain_m": audit["spread_span_gain_m"],
                "side_changes": audit["sampled_interior_side_changes"],
                "bread_max_displacement_m": audit["bread_max_displacement_m"],
                "bread_top_m": float(bread[:, 2].max()),
                "cut_rms_m": next(row["slice_rms_m"] for row in shape["phases"] if row["phase"] == "checkpoint-cut"),
                "lift_rms_m": next(row["slice_rms_m"] for row in shape["phases"] if row["phase"] == "checkpoint-lift"),
                "stock_displacement_median_m": float(np.median(displacement)),
                "stock_displacement_p99_m": float(np.quantile(displacement, 0.99)),
                "deposited_center_m": deposited.mean(axis=0).tolist() if len(deposited) else [float("nan")] * 3,
            }
        )
    before, after = reports
    checks = {
        "complete_audit": after["audit_passed"],
        "pickup_within_1_percentage_point": after["pickup_fraction"] >= before["pickup_fraction"] - 0.01,
        "deposition_within_3_percentage_points": abs(after["deposition_fraction"] - before["deposition_fraction"])
        <= 0.03,
        "span_within_10_percent": abs(after["spread_span_m"] / before["spread_span_m"] - 1) <= 0.10,
        "cut_deformation_within_15_percent": after["cut_rms_m"] <= before["cut_rms_m"] * 1.15,
        "lift_deformation_within_15_percent": after["lift_rms_m"] <= before["lift_rms_m"] * 1.15,
        "stock_median_within_1mm": after["stock_displacement_median_m"]
        <= before["stock_displacement_median_m"] + 0.001,
        "stock_p99_within_3mm": after["stock_displacement_p99_m"] <= before["stock_displacement_p99_m"] + 0.003,
        "deposition_center_within_5mm": np.linalg.norm(
            np.array(after["deposited_center_m"]) - before["deposited_center_m"]
        )
        <= 0.005,
        "bread_p99_difference_within_1mm": bread_difference_p99 <= 0.001,
        "bread_center_difference_within_1mm": bread_center_difference <= 0.001,
        "bread_top_rise_within_1mm": after["bread_top_m"] <= before["bread_top_m"] + 0.001,
        "zero_sampled_side_changes": after["side_changes"] == 0,
        "fixed_material_matches_reference": json.loads((run / "material.json").read_text())
        == json.loads((reference / "material.json").read_text()),
    }
    with np.load(run / "motion.npz") as actual, np.load(reference / "motion.npz") as original:
        checks["identical_prescribed_motion"] = all(
            np.array_equal(actual[key], original[key]) for key in original.files
        )
    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "is_passed": all(checks.values()),
        "checks": checks,
        "reference": before,
        "candidate": after,
        "bread_difference_p99_m": bread_difference_p99,
        "bread_center_difference_m": bread_center_difference,
        "bread_scope": "Maximum displacement includes localized tangential slip; bulk positional difference and top height gate macroscopic deformation and lift.",
    }
    (run / "adhesion-comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["is_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
