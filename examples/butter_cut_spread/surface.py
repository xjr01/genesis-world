"""Volume-normalized density reconstruction for bulk butter and supported thin coatings."""

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes

from examples.butter_cut_spread.density import splat_density
from examples.butter_cut_spread.film import repair_sampling_gaps


@dataclass(frozen=True)
class DensitySurface:
    mesh: trimesh.Trimesh
    level: float


def closed_density_surface(field, origin, voxel):
    """Resolve numerical isovalue coincidences without filling geometric holes."""
    for level in (0.5, 0.500001, 0.50001):
        vertices, faces, _, _ = marching_cubes(
            field, level, spacing=(voxel,) * 3, gradient_direction="ascent", allow_degenerate=False
        )
        mesh = trimesh.Trimesh(vertices + origin, faces, process=False)
        if mesh.volume < 0:
            mesh.invert()
        if mesh.is_volume:
            return DensitySurface(mesh, level)
    raise ValueError("Density surface remains nonmanifold after bounded numerical isovalue resolution.")


def coating_density(points, origin, shape, voxel, width, config, normal_floor_ratio=1.0):
    """Resolve sub-kernel coating thickness while retaining each particle's rest volume."""
    field = splat_density(points, origin, shape, voxel, width, config.particle_size**3)
    top = config.bread_center[2] + config.bread_size[2] / 2
    is_supported = np.all(
        np.abs(points[:, :2] - np.array(config.bread_center[:2])) < np.array(config.bread_size[:2]) / 2, axis=1
    )
    is_supported &= (points[:, 2] > top - 0.007) & (points[:, 2] < top + 0.004)
    if not is_supported.any():
        return field
    thickness = field.sum(axis=2) * voxel
    local_depth = map_coordinates(thickness, ((points[:, :2] - origin[:2]) / voxel).T, order=1, mode="nearest")
    # Narrow the normal kernel only where the material column is thinner than the bulk kernel.
    normal_width = np.clip(0.65 * local_depth, config.particle_size * normal_floor_ratio, width)
    blend = np.clip((top + 0.004 - points[:, 2]) / 0.002, 0, 1)
    blend = blend**2 * (3 - 2 * blend)
    normal_width = np.where(is_supported, width + blend * (normal_width - width), width)
    # Thin supported films need overlapping tangential footprints at the particle sampling scale
    tangent_width = np.where(is_supported, width + blend * config.particle_size, width)
    del field
    field = splat_density(points, origin, shape, voxel, width, config.particle_size**3, normal_width, tangent_width)
    return repair_sampling_gaps(
        field, origin, voxel, config.particle_size, config.bread_center, config.bread_size, points
    )
