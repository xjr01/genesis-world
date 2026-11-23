"""Measure both blade-face films from saved particle identities and record volume accounting."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from examples.butter_cut_spread.blade_film import BladeFilm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    film = BladeFilm(run, config["particle_size"], config["blade_size"][2])
    records = []
    for path in sorted(run.glob("checkpoint-*.npz")):
        with np.load(path) as data:
            particles = data["particles"]
            result = film.partition(particles, data["knife_positions"], data["knife_quaternions"])
            particle_volume = len(particles) * film.spacing**3
            represented_volume = result.bulk_weights.sum() * film.spacing**3 + sum(result.volumes)
            sheet_volume = result.thickness.sum(axis=(1, 2)) * film.settings.pixel**2
            records.append(
                {
                    "phase": path.stem.removeprefix("checkpoint-"),
                    "time_s": float(data["time_s"]),
                    "thin_particle_counts_negative_positive": result.counts,
                    "thin_volume_mm3_negative_positive": (np.array(result.volumes) * 1e9).tolist(),
                    "sheet_volume_mm3_negative_positive": (sheet_volume * 1e9).tolist(),
                    "partition_relative_error": abs(represented_volume / particle_volume - 1),
                    "covered_area_mm2_negative_positive": (
                        np.count_nonzero(film.opacity(result.thickness) > 0.02, axis=1) * film.settings.pixel**2 * 1e6
                    ).tolist(),
                }
            )
    report = {
        "film_settings": asdict(film.settings),
        "phase_records": sorted(records, key=lambda row: row["time_s"]),
        "scope": "Geometric thin-layer candidates from existing particles; thick attached slices stay in the bulk partition.",
    }
    (run / "residue-audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
