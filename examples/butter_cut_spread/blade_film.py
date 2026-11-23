"""Partition recorded butter volume between a bulk surface and two rotating blade films."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from examples.butter_cut_spread.film_settings import FilmSettings
from examples.butter_spreading.surface import b2


@dataclass(frozen=True)
class FilmFrame:
    bulk_weights: np.ndarray
    thickness: np.ndarray
    counts: tuple[int, int]
    volumes: tuple[float, float]


@njit(cache=False)
def splat_sheet(points, weights, origin, shape, pixel, width, volume):
    """Integrate normalized quadratic particle kernels along the local face normal."""
    sheet = np.zeros(shape)
    for p, weight in zip(points, weights):
        low = np.maximum(np.ceil((p[:2] - 1.5 * width - origin) / pixel).astype(np.int64), 0)
        high = np.floor((p[:2] + 1.5 * width - origin) / pixel).astype(np.int64) + 1
        high[0] = min(high[0], shape[0])
        high[1] = min(high[1], shape[1])
        sum_x, sum_y = 0.0, 0.0
        for x in range(low[0], high[0]):
            sum_x += b2((origin[0] + x * pixel - p[0]) / width)
        for y in range(low[1], high[1]):
            sum_y += b2((origin[1] + y * pixel - p[1]) / width)
        amplitude = volume * weight / max(sum_x * sum_y * pixel**2, 1e-30)
        for x in range(low[0], high[0]):
            wx = b2((origin[0] + x * pixel - p[0]) / width)
            for y in range(low[1], high[1]):
                wy = b2((origin[1] + y * pixel - p[1]) / width)
                sheet[x, y] += amplitude * wx * wy
    return sheet


class BladeFilm:
    def __init__(self, folder, spacing, blade_thickness, settings=None):
        if settings is None:
            settings = FilmSettings()
        self.settings = settings
        self.spacing = spacing
        self.half = blade_thickness / 2
        with np.load(folder / "blade-face.npz") as data:
            self.face_origin = data["origin"]
            self.face_spacing = float(data["spacing"])
            self.distance = data["distance"]
        padding = 2 * max(settings.kernel, settings.sheet_kernel)
        self.origin = self.face_origin - padding
        extent = (np.array(self.distance.shape) - 1) * self.face_spacing + 2 * padding
        self.shape = tuple(np.ceil(extent / settings.pixel).astype(int) + 1)
        x, y = np.meshgrid(
            self.origin[0] + np.arange(self.shape[0]) * settings.pixel,
            self.origin[1] + np.arange(self.shape[1]) * settings.pixel,
            indexing="ij",
        )
        xy = np.column_stack((np.concatenate(x), np.concatenate(y)))
        self.is_inside = self.edge_distance(xy) > 0
        self.vertices = np.stack(
            [np.column_stack((xy, np.full(len(xy), side * (self.half + settings.clearance)))) for side in (-1, 1)]
        )
        grid = np.arange(x.size).reshape(self.shape)
        a, b = np.concatenate(grid[:-1, :-1]), np.concatenate(grid[1:, :-1])
        c, d = np.concatenate(grid[1:, 1:]), np.concatenate(grid[:-1, 1:])
        positive = np.concatenate((np.column_stack((a, b, c)), np.column_stack((a, c, d))))
        self.faces = np.stack((positive[:, ::-1], positive))

    def edge_distance(self, xy):
        return map_coordinates(
            self.distance, ((xy - self.face_origin) / self.face_spacing).T, order=1, mode="constant", cval=-1
        )

    def partition(self, points, center, quaternion):
        local = (points - center) @ Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        gap = np.abs(local[:, 2]) - self.half
        edge = self.edge_distance(local[:, :2])
        weights = np.ones(len(points))
        sheets, counts, volumes = [], [], []
        s = self.settings
        for side in (-1, 1):
            is_column = (side * local[:, 2] > 0) & (edge > 0) & (gap > -self.spacing / 2)
            is_column &= gap < s.column_distance
            column = local[is_column]
            thickness = splat_sheet(
                column, np.ones(len(column)), self.origin, self.shape, s.pixel, s.kernel, self.spacing**3
            )
            is_near = is_column & (gap < s.capture_distance)
            near = local[is_near]
            layers = map_coordinates(thickness, ((near[:, :2] - self.origin) / s.pixel).T, order=1) / self.spacing
            blend = np.clip((layers - s.thin_layers) / (s.thick_layers - s.thin_layers), 0, 1)
            film_weight = 1 - blend**2 * (3 - 2 * blend)
            weights[is_near] -= film_weight
            sheets.append(
                splat_sheet(near, film_weight, self.origin, self.shape, s.pixel, s.sheet_kernel, self.spacing**3)
            )
            counts.append(int(np.count_nonzero(film_weight > 0)))
            volumes.append(float(film_weight.sum() * self.spacing**3))
        return FilmFrame(weights, np.stack(sheets), tuple(counts), tuple(volumes))

    def opacity(self, thickness):
        opacity = 1 - np.exp(
            -self.settings.optical_depth_per_layer * thickness / self.settings.optical_reference_spacing
        )
        return opacity.reshape(2, -1) * self.is_inside
