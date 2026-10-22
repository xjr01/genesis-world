"""Closed beveled collision geometry in the spreading tool's local frame."""

import numpy as np
import trimesh


def convex_planes(vertices, faces):
    """Return outward unit halfspaces of an oriented convex triangle mesh."""
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    offsets = np.einsum("ij,ij->i", normals, triangles[:, 0])
    return np.column_stack((normals, -offsets))


def tool_mesh(config):
    """Place the cutting edge at negative local X and the broad faces along Z."""
    length, width, thickness = config.blade_size
    size = (thickness, width, length)
    if config.blade_bevel_height == 0:
        mesh = trimesh.creation.box(extents=size)
    else:
        half = np.array(size) / 2
        section = np.array(
            [
                [-half[0], half[2]],
                [half[0], half[2]],
                [half[0], -half[2] + config.blade_bevel_height],
                [0.0, -half[2]],
                [-half[0], -half[2] + config.blade_bevel_height],
            ]
        )
        points = np.column_stack(
            (np.tile(section[:, 0], 2), np.repeat([-half[1], half[1]], len(section)), np.tile(section[:, 1], 2))
        )
        mesh = trimesh.convex.convex_hull(points)
    mesh.vertices = mesh.vertices @ np.array(((0, 0, -1), (0, 1, 0), (1, 0, 0)))
    return mesh
