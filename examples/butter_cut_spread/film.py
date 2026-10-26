"""Bound particle-scale display gaps while conserving integrated coating density."""

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter, label
from scipy.spatial import cKDTree


def repair_sampling_gaps(field, origin, voxel, spacing, bread_center, bread_size, points=None):
    """Redistribute surplus display density into enclosed, particle-supported film gaps.

    Modify the field in place, preserving its integral rather than the isosurface
    volume. Open edges and gaps farther than two spacings from particles remain.
    Without particle positions, the distance to occupied columns bounds repairs.
    This regularizes reconstruction only; it does not modify simulated particles.
    """
    occupied = field.max(axis=2) >= 0.5
    holes = binary_fill_holes(occupied) & ~occupied
    if not holes.any():
        return field
    labels, count = label(holes)
    distances = distance_transform_edt(~occupied) * voxel
    if points is not None:
        supported_points = points[
            (points[:, 2] > bread_center[2] - bread_size[2] / 2 - spacing)
            & (points[:, 2] < bread_center[2] + bread_size[2] / 2 + 0.004)
        ]
        if not len(supported_points):
            return field
        indices = np.argwhere(holes)
        distances[holes] = cKDTree(supported_points[:, :2]).query(origin[:2] + indices * voxel)[0]
    radii = np.zeros(count + 1)
    np.maximum.at(radii, labels[holes], distances[holes])
    selected = holes & (radii[labels] <= 2 * spacing)
    x = origin[0] + np.arange(field.shape[0]) * voxel
    y = origin[1] + np.arange(field.shape[1]) * voxel
    selected &= abs(x[:, None] - bread_center[0]) < bread_size[0] / 2
    selected &= abs(y[None, :] - bread_center[1]) < bread_size[1] / 2
    if not selected.any():
        return field
    indices = np.argwhere(selected)
    padding = int(np.ceil(3 * spacing / voxel))
    lower = np.maximum(indices.min(axis=0) - padding, 0)
    upper = np.minimum(indices.max(axis=0) + padding + 1, field.shape[:2])
    region = field[lower[0] : upper[0], lower[1] : upper[1], :]
    local = selected[lower[0] : upper[0], lower[1] : upper[1]]
    smoothed = gaussian_filter(region, sigma=(spacing / voxel, spacing / voxel, 0), mode="constant")
    rows, cols = np.where(local)
    profiles = smoothed[rows, cols]
    peaks = profiles.max(axis=1)
    target = profiles * (0.65 / np.maximum(peaks, 1e-12))[:, None]
    z = origin[2] + np.arange(field.shape[2]) * voxel
    centers = (profiles @ z) / np.maximum(profiles.sum(axis=1), 1e-12)
    supported = centers > bread_center[2] - bread_size[2] / 2 - spacing
    supported &= centers < bread_center[2] + bread_size[2] / 2 + 0.004
    added = np.maximum(target - region[rows, cols], 0) * supported[:, None]
    needed = added.sum(dtype=np.float64)
    # Keep donor voxels above the 0.5 isovalue while redistributing their surplus density
    reservoir = np.maximum(region - 0.65, 0)
    available = reservoir.sum(dtype=np.float64)
    fraction = min(1.0, available / max(needed, 1e-12))
    region -= reservoir * (needed * fraction / max(available, 1e-12))
    region[rows, cols] += added * fraction
    return field
