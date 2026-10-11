"""Measure geometric ripples in the deposited strip, independently of lighting."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from examples.butter_spreading.surface import splat


@dataclass(frozen=True)
class RippleReport:
    state: str
    patch_xy_m: list[list[float]]
    state_sha256: str
    surface_sha256: str
    missing_area_fraction: float
    height_ripple_rms_mm: float
    height_ripple_p95_abs_mm: float
    coherent_cross_strip_ridge_rms_mm: float
    coherent_along_strip_ridge_rms_mm: float
    height_range_mm: list[float]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="Recorded particle trajectory NPZ.")
    parser.add_argument("--output", type=Path, required=True, help="Ripple diagnostic JSON.")
    parser.add_argument("--surface", type=Path, required=True, help="Reconstructed triangle mesh NPZ.")
    args = parser.parse_args()
    with np.load(args.state) as data:
        p = data["particles"]
    meta = json.loads(args.state.with_suffix(".json").read_text(encoding="utf-8"))
    spacing = meta["particle_spacing_m"]
    with np.load(args.surface) as mesh:
        width, voxel = float(mesh["kernel_width_m"]), float(mesh["voxel_size_m"])
        if str(mesh["state_sha256"]) != hashlib.sha256(args.state.read_bytes()).hexdigest():
            raise ValueError("Surface and particle trajectory hashes differ.")
    origin = np.floor((p.min(axis=(0, 1)) - 2 * width) / voxel) * voxel
    shape = tuple((np.ceil((p.max(axis=(0, 1)) + 2 * width - origin) / voxel).astype(int) + 1).tolist())
    field = splat(p[-1].astype(np.float64), origin.astype(np.float64), shape, voxel, width, spacing**3)
    inside = field >= 0.5
    occupied = inside.any(axis=2)
    k = shape[2] - 1 - np.argmax(inside[:, :, ::-1], axis=2)
    k = np.minimum(k, shape[2] - 2)
    d0 = np.take_along_axis(field, k[:, :, None], axis=2)[:, :, 0]
    d1 = np.take_along_axis(field, (k + 1)[:, :, None], axis=2)[:, :, 0]
    fraction = np.divide(0.5 - d0, d1 - d0, out=np.zeros_like(d0), where=np.abs(d1 - d0) > 1e-8)
    height = origin[2] + (k + fraction) * voxel
    x, y = np.meshgrid(origin[0] + np.arange(shape[0]) * voxel, origin[1] + np.arange(shape[1]) * voxel, indexing="ij")
    patch = (x > -0.020) & (x < 0.035) & (np.abs(y) < 0.010)
    missing_fraction = float(np.mean(~occupied[patch]))
    valid = patch & occupied
    if not np.any(valid):
        raise ValueError("No deposited surface in the ripple measurement patch.")
    u, v = x[valid] / 0.03, y[valid] / 0.03
    # Remove only the broad thickness slope/curvature for this diagnostic. The
    # simulation, reconstruction and delivered video never use this fitted field.
    basis = np.stack([u**a * v**b for a in range(4) for b in range(4 - a)], axis=1)
    z = height[valid] * 1000
    residual = z - basis @ np.linalg.lstsq(basis, z, rcond=None)[0]
    # Directional coherence distinguishes cross-strip ridges from small irregular
    # roughness. These statistics are diagnostic only; no field is filtered for
    # reconstruction or rendering. Keep reporting the original total RMS too.
    residual_map = np.full(height.shape, np.nan)
    residual_map[valid] = residual
    rows = np.flatnonzero(np.any(valid, axis=1))
    columns = np.flatnonzero(np.any(valid, axis=0))
    cross_strip_mean = np.nanmean(residual_map[np.ix_(rows, columns)], axis=1)
    along_strip_mean = np.nanmean(residual_map[np.ix_(rows, columns)], axis=0)
    report = RippleReport(
        state=str(args.state),
        patch_xy_m=[[-0.020, 0.035], [-0.010, 0.010]],
        state_sha256=hashlib.sha256(args.state.read_bytes()).hexdigest(),
        surface_sha256=hashlib.sha256(args.surface.read_bytes()).hexdigest(),
        missing_area_fraction=missing_fraction,
        height_ripple_rms_mm=float(np.sqrt(np.mean(residual**2))),
        height_ripple_p95_abs_mm=float(np.percentile(np.abs(residual), 95)),
        coherent_cross_strip_ridge_rms_mm=float(np.sqrt(np.mean(cross_strip_mean**2))),
        coherent_along_strip_ridge_rms_mm=float(np.sqrt(np.mean(along_strip_mean**2))),
        height_range_mm=np.percentile(z, [1, 50, 99]).tolist(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
