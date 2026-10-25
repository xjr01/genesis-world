"""Closed beveled collision geometry in the spreading tool's local frame."""

import numpy as np
import trimesh


def cutting_sweep_limits(mesh, width, height, spacing, clearance=0.00002, direction=1):
    """Fit a full downstroke followed by one draw, including cuts taller than the blade."""
    if direction not in (-1, 1):
        raise ValueError("Cutting direction must be -1 or 1.")
    section = mesh.section(plane_origin=(0, 0, 0), plane_normal=(1, 0, 0))
    if section is None:
        raise ValueError("The blade must intersect its cutting plane.")
    segments = np.concatenate([np.stack((line[:-1, 1:], line[1:, 1:]), axis=1) for line in section.discrete])
    # Solve a negative-Y draw, reflecting the section for a positive-Y draw.
    segments[:, :, 0] *= -direction
    start, end = segments[:, 0], segments[:, 1]
    active = abs(end[:, 1] - start[:, 1]) > 1e-12
    start, end = start[active], end[active]
    bottom = mesh.bounds[0, 2] - clearance
    bounds = bottom + np.array((spacing / 2, height - spacing / 2))
    heights = np.unique(np.r_[bounds, np.clip(segments[:, :, 1].reshape(-1), *bounds)])
    progress = (heights[:, None] - start[None, :, 1]) / (end - start)[None, :, 1]
    y = start[None, :, 0] + progress * (end - start)[None, :, 0]
    valid = (progress >= -1e-10) & (progress <= 1 + 1e-10)
    lower = np.where(valid, y, np.inf).min(axis=1)
    upper = np.where(valid, y, -np.inf).max(axis=1)
    margin = max(0.001, 2 * spacing)
    vertices = segments.reshape(-1, 2)
    below = vertices[None, :, 1] <= heights[:, None] + 1e-12
    down_lower = np.minimum(lower, np.where(below, vertices[None, :, 0], np.inf).min(axis=1))
    down_upper = np.maximum(upper, np.where(below, vertices[None, :, 0], -np.inf).max(axis=1))
    if not np.isfinite(down_upper).all():
        raise ValueError("The blade cannot reach the bottom particle layer.")
    first = np.max(width / 2 - down_upper) + margin
    unfinished = first + down_lower > -width / 2 - margin
    if np.any(unfinished & ~np.isfinite(lower)):
        raise ValueError("The blade downstroke leaves an unreachable upper cutting section.")
    last = min(first, np.min(-width / 2 - lower[unfinished] - margin)) if unfinished.any() else first
    return -direction * first, -direction * last


def rigid_fit(source, target):
    """Fit corresponding authored vertices with a proper rigid transformation."""
    if source.shape != target.shape:
        raise ValueError("Authored mesh topology differs between the asset versions.")
    left, _, right = np.linalg.svd((source - source.mean(0)).T @ (target - target.mean(0)))
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    translation = target.mean(0) - source.mean(0) @ rotation
    error = np.linalg.norm(source @ rotation + translation - target, axis=1).max()
    if error > 1e-6:
        raise ValueError(f"Asset identity check failed: rigid-fit error {error} m.")
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation.T, translation
    return transform, float(error)


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
