"""Material-coordinate bread display surface for recorded spreading trajectories."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BreadBinding:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray
    particle_ids: np.ndarray
    weights: np.ndarray


def bind_bread(reference, spacing):
    """Bind six boundary grids to the complete reference particle lattice.

    Half-cell boundary extrapolation preserves affine motion, including uniform
    compression. Surface topology and procedural colors stay in material space.
    """
    axes = [np.unique(reference[:, axis]) for axis in range(3)]
    shape = tuple(len(axis) for axis in axes)
    if min(shape) < 2 or np.prod(shape) != len(reference):
        raise ValueError("Bread requires a complete regular reference particle lattice.")
    indices = tuple(np.searchsorted(axes[axis], reference[:, axis]) for axis in range(3))
    if len(np.unique(np.ravel_multi_index(indices, shape))) != len(reference):
        raise ValueError("Duplicate bread material coordinates.")
    grid = np.empty(shape, dtype=np.int32)
    grid[indices] = np.arange(len(reference))
    low, high = reference.min(axis=0) - spacing / 2, reference.max(axis=0) + spacing / 2
    vertices, faces, top_masks = [], [], []
    offset = 0
    for axis in range(3):
        tangent = [index for index in range(3) if index != axis]
        for side in (-1, 1):
            u, v = np.meshgrid(
                np.linspace(low[tangent[0]], high[tangent[0]], shape[tangent[0]] + 1),
                np.linspace(low[tangent[1]], high[tangent[1]], shape[tangent[1]] + 1),
                indexing="ij",
            )
            points = np.empty((u.size, 3))
            points[:, axis] = low[axis] if side < 0 else high[axis]
            points[:, tangent[0]], points[:, tangent[1]] = np.concatenate(u), np.concatenate(v)
            ids = np.arange(u.size).reshape(u.shape)
            a, b = np.concatenate(ids[:-1, :-1]), np.concatenate(ids[1:, :-1])
            c, d = np.concatenate(ids[1:, 1:]), np.concatenate(ids[:-1, 1:])
            triangles = np.concatenate((np.column_stack((a, b, c)), np.column_stack((a, c, d))))
            normal = np.cross(
                points[triangles[0, 1]] - points[triangles[0, 0]], points[triangles[0, 2]] - points[triangles[0, 0]]
            )
            if normal[axis] * side < 0:
                triangles = triangles[:, ::-1]
            vertices.append(points)
            faces.append(triangles + offset)
            top_masks.append(np.full(len(points), axis == 2 and side == 1))
            offset += len(points)
    vertices, faces = np.concatenate(vertices), np.concatenate(faces)
    lower, fraction = [], []
    for axis, coordinates in enumerate(axes):
        index = np.clip(np.searchsorted(coordinates, vertices[:, axis], side="right") - 1, 0, len(coordinates) - 2)
        lower.append(index)
        fraction.append((vertices[:, axis] - coordinates[index]) / (coordinates[index + 1] - coordinates[index]))
    particle_ids, weights = [], []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                particle_ids.append(grid[lower[0] + i, lower[1] + j, lower[2] + k])
                weights.append(
                    (fraction[0] if i else 1 - fraction[0])
                    * (fraction[1] if j else 1 - fraction[1])
                    * (fraction[2] if k else 1 - fraction[2])
                )
    edge = np.minimum(vertices[:, :2] - low[:2], high[:2] - vertices[:, :2]).min(axis=1)
    crumb = np.clip(edge / 0.003, 0, 1) * np.concatenate(top_masks)
    crumb = crumb**2 * (3 - 2 * crumb)
    colors = (1 - crumb[:, None]) * np.array((0.32, 0.125, 0.035)) + crumb[:, None] * np.array((0.72, 0.53, 0.29))
    # Deterministic fine color variation, not simulated pores or displaced geometry.
    grain = np.sin(vertices @ np.array((6700, 4900, 9100))) * np.sin(vertices @ np.array((3700, -8100, 5300)))
    colors *= 0.96 + 0.04 * grain[:, None]
    return BreadBinding(vertices, faces, colors, np.stack(particle_ids, axis=1), np.stack(weights, axis=1))


def deform_bread(binding, reference, current):
    displacement = current - reference
    moved = binding.vertices + np.einsum("ij,ijk->ik", binding.weights, displacement[binding.particle_ids])
    faces = binding.faces
    normals = np.zeros_like(moved)
    face_normals = np.cross(moved[faces[:, 1]] - moved[faces[:, 0]], moved[faces[:, 2]] - moved[faces[:, 0]])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    length = np.linalg.norm(normals, axis=1)
    if not np.isfinite(moved).all() or np.any(length < 1e-15):
        raise ValueError("Invalid deformed bread surface.")
    normals /= length[:, None]
    return moved, normals
