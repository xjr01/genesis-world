"""Bake the authored planar blade outline once and share its contact distance field."""

from typing import NamedTuple

import numba
import numpy as np

import genesis as gs

qd = gs.qd


class FaceSamples(NamedTuple):
    origin: np.ndarray
    spacing: float
    distance: np.ndarray


def face_samples(folder, mesh, minimum_y, prepare, spacing=0.0001):
    path = folder / "blade-face.npz"
    if prepare:
        section = mesh.section(plane_origin=(0, 0, 0), plane_normal=(0, 0, 1))
        if section is None or len(section.discrete) != 1:
            raise ValueError("The planar blade face must have one closed outline.")
        polygon = section.discrete[0][:-1, :2]
        following = np.roll(polygon, -1, axis=0)
        inside = polygon[:, 1] >= minimum_y
        crossing = inside != np.roll(inside, -1)
        fraction = (minimum_y - polygon[crossing, 1]) / (following[crossing, 1] - polygon[crossing, 1])
        intersections = polygon[crossing] + fraction[:, None] * (following[crossing] - polygon[crossing])
        order = np.r_[np.flatnonzero(inside), np.flatnonzero(crossing) + fraction]
        polygon = np.concatenate((polygon[inside], intersections))[order.argsort()]
        origin = polygon.min(axis=0) - 2 * spacing
        shape = np.ceil((polygon.max(axis=0) - origin) / spacing).astype(int) + 3
        distance = polygon_distances(polygon, origin, spacing, *shape)
        np.savez(path, origin=origin, spacing=spacing, distance=distance, polygon=polygon)
    with np.load(path) as data:
        return FaceSamples(data["origin"], float(data["spacing"]), data["distance"])


@numba.njit(cache=True)
def polygon_distances(polygon, origin, spacing, nx, ny):
    """Sample positive-inside Euclidean distance to a simple closed polygon."""
    result = np.empty((nx, ny))
    for ix in range(nx):
        for iy in range(ny):
            x, y = origin[0] + ix * spacing, origin[1] + iy * spacing
            squared = np.inf
            inside = False
            for k in range(len(polygon)):
                a, b = polygon[k], polygon[(k + 1) % len(polygon)]
                dx, dy = b[0] - a[0], b[1] - a[1]
                length = dx * dx + dy * dy
                if length > 0:
                    t = min(1.0, max(0.0, ((x - a[0]) * dx + (y - a[1]) * dy) / length))
                    squared = min(squared, (x - a[0] - t * dx) ** 2 + (y - a[1] - t * dy) ** 2)
                if (a[1] > y) != (b[1] > y) and x < a[0] + (y - a[1]) * dx / dy:
                    inside = not inside
            result[ix, iy] = np.sqrt(squared) * (1 if inside else -1)
    return result


class BladeFace:
    def __init__(self, samples):
        self.origin = qd.Vector(samples.origin.tolist())
        self.spacing = samples.spacing
        self.shape = qd.Vector(list(samples.distance.shape))
        self.distance = qd.field(gs.qd_float, shape=samples.distance.shape)
        self.distance.from_numpy(samples.distance.astype(gs.np_float))


@qd.func
def face_distance(x, y, face: qd.template()):
    uv = (qd.Vector([x, y]) - face.origin) / face.spacing
    distance = -face.spacing
    if 0 <= uv[0] < face.shape[0] - 1 and 0 <= uv[1] < face.shape[1] - 1:
        cell = qd.cast(qd.floor(uv), qd.i32)
        fraction = uv - cell
        distance = 0.0
        for a, b in qd.static(qd.ndrange(2, 2)):
            weight = qd.select(a == 0, 1 - fraction[0], fraction[0])
            weight *= qd.select(b == 0, 1 - fraction[1], fraction[1])
            distance += weight * face.distance[cell[0] + a, cell[1] + b]
    return distance
