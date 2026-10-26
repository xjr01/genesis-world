"""Accumulate particle density with reusable separable quadratic kernel weights."""

import numpy as np
from numba import njit

from examples.butter_spreading.surface import b2


@njit(cache=True)
def splat_density(points, origin, shape, voxel, width, volume, normal_widths=None, tangent_widths=None):
    density = np.zeros(shape, dtype=np.float32)
    for i_p in range(len(points)):
        point = points[i_p]
        if normal_widths is None:
            normal_width = tangent_width = width
            amplitude = volume / width**3
        else:
            normal_width = normal_widths[i_p]
            tangent_width = tangent_widths[i_p]
            amplitude = volume / (tangent_width**2 * normal_width)
        low_x = max(int(np.ceil((point[0] - 1.5 * tangent_width - origin[0]) / voxel)), 0)
        low_y = max(int(np.ceil((point[1] - 1.5 * tangent_width - origin[1]) / voxel)), 0)
        low_z = max(int(np.ceil((point[2] - 1.5 * normal_width - origin[2]) / voxel)), 0)
        high_x = min(int(np.floor((point[0] + 1.5 * tangent_width - origin[0]) / voxel)) + 1, shape[0])
        high_y = min(int(np.floor((point[1] + 1.5 * tangent_width - origin[1]) / voxel)) + 1, shape[1])
        high_z = min(int(np.floor((point[2] + 1.5 * normal_width - origin[2]) / voxel)) + 1, shape[2])
        weights_y = np.empty(max(0, high_y - low_y))
        weights_z = np.empty(max(0, high_z - low_z))
        for i_y in range(low_y, high_y):
            weights_y[i_y - low_y] = b2((origin[1] + i_y * voxel - point[1]) / tangent_width)
        for i_z in range(low_z, high_z):
            weights_z[i_z - low_z] = b2((origin[2] + i_z * voxel - point[2]) / normal_width)
        # Preserve particle order and multiplication order at every density cell.
        for i_x in range(low_x, high_x):
            wx = b2((origin[0] + i_x * voxel - point[0]) / tangent_width)
            for i_y in range(low_y, high_y):
                wxy = wx * weights_y[i_y - low_y]
                for i_z in range(low_z, high_z):
                    density[i_x, i_y, i_z] += amplitude * wxy * weights_z[i_z - low_z]
    return density
