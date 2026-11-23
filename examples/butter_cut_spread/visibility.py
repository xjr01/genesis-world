"""Top-view occlusion audit of the exact exported bread and butter triangles."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.ndimage import binary_fill_holes, label


@dataclass(frozen=True)
class Visibility:
    enclosed_visible_hole_area_mm2: float
    largest_visible_hole_mm2: float
    bread_protrusion_area_mm2: float
    sample_spacing_m: float


@njit(cache=True)
def top_surface(vertices, faces, origin, shape, spacing):
    height = np.full(shape, -np.inf)
    for face in faces:
        a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-18:
            continue
        low = np.maximum(np.ceil((np.minimum(np.minimum(a[:2], b[:2]), c[:2]) - origin) / spacing).astype(np.int32), 0)
        high = np.minimum(
            np.floor((np.maximum(np.maximum(a[:2], b[:2]), c[:2]) - origin) / spacing).astype(np.int32) + 1,
            np.array(shape),
        )
        for i in range(low[0], high[0]):
            x = origin[0] + i * spacing
            for j in range(low[1], high[1]):
                y = origin[1] + j * spacing
                u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
                v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
                w = 1 - u - v
                if min(u, v, w) >= -1e-7:
                    height[i, j] = max(height[i, j], u * a[2] + v * b[2] + w * c[2])
    return height


def visible_holes(butter_vertices, butter_faces, bread_vertices, bread_faces, config):
    spacing = 0.0002
    origin = np.array(config.bread_center[:2]) - np.array(config.bread_size[:2]) / 2
    shape = tuple(np.ceil(np.array(config.bread_size[:2]) / spacing).astype(int) + 1)
    top = config.bread_center[2] + config.bread_size[2] / 2
    is_deposited = np.max(butter_vertices[butter_faces, 2], axis=1) < top + 0.01
    butter = top_surface(butter_vertices, butter_faces[is_deposited], origin, shape, spacing)
    bread = top_surface(bread_vertices, bread_faces, origin, shape, spacing)
    silhouette = np.isfinite(butter)
    protrudes = silhouette & (bread > butter)
    visible = silhouette & ~protrudes
    holes = binary_fill_holes(visible) & ~visible
    regions, count = label(holes)
    areas = np.bincount(regions[holes], minlength=count + 1)[1:] * (spacing * 1000) ** 2
    return Visibility(
        float(areas.sum()),
        float(areas.max()) if count else 0.0,
        float(protrudes.sum() * (spacing * 1000) ** 2),
        spacing,
    )
