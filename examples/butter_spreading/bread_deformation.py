"""Embed the porous display mesh in the bread's reference MPM lattice.

Bindings use material coordinates and are computed once. Each frame transfers
the saved particle displacement, with unit gain, to the original mesh. No
particle positions, surface topology, UVs or material assignments are changed.
"""

import numpy as np
from numba import njit


@njit(inline="always", fastmath=False)
def _deform_impl(rest, normals, lower, fraction, inverse_spacing, grid, displacement):
    points = np.empty_like(rest)
    moved_normals = np.empty_like(normals)
    determinants = np.empty(len(rest), dtype=np.float32)
    for p in range(len(rest)):
        u = np.zeros(3, dtype=np.float64)
        F = np.eye(3, dtype=np.float64)
        for a in range(2):
            wx = fraction[p, 0] if a else 1.0 - fraction[p, 0]
            gx = inverse_spacing[p, 0] * (1.0 if a else -1.0)
            for b in range(2):
                wy = fraction[p, 1] if b else 1.0 - fraction[p, 1]
                gy = inverse_spacing[p, 1] * (1.0 if b else -1.0)
                for c in range(2):
                    wz = fraction[p, 2] if c else 1.0 - fraction[p, 2]
                    gz = inverse_spacing[p, 2] * (1.0 if c else -1.0)
                    index = grid[lower[p, 0] + a, lower[p, 1] + b, lower[p, 2] + c]
                    for d in range(3):
                        value = displacement[index, d]
                        u[d] += wx * wy * wz * value
                        F[d, 0] += gx * wy * wz * value
                        F[d, 1] += wx * gy * wz * value
                        F[d, 2] += wx * wy * gz * value
        # cof(F) n has the direction of F^{-T} n for positive det(F).
        cofactor = np.empty((3, 3), dtype=np.float64)
        for d in range(3):
            r, s = (d + 1) % 3, (d + 2) % 3
            for e in range(3):
                j, k = (e + 1) % 3, (e + 2) % 3
                cofactor[d, e] = F[r, j] * F[s, k] - F[r, k] * F[s, j]
        det = F[0, 0] * cofactor[0, 0] + F[0, 1] * cofactor[0, 1] + F[0, 2] * cofactor[0, 2]
        determinants[p] = det
        normal = cofactor @ normals[p].astype(np.float64)
        length = np.sqrt(np.sum(normal * normal))
        for d in range(3):
            points[p, d] = rest[p, d] + u[d]
            moved_normals[p, d] = normal[d] / max(length, 1e-20)
    return points, moved_normals, determinants


@njit(cache=True, fastmath=False)
def _deform(rest, normals, lower, fraction, inverse_spacing, grid, displacement):
    return _deform_impl(rest, normals, lower, fraction, inverse_spacing, grid, displacement)


class BreadDeformation:
    def __init__(self, reference_particles, reference_vertices, reference_normals):
        self.reference_particles = np.asarray(reference_particles, dtype=np.float32)
        self.rest = np.asarray(reference_vertices, dtype=np.float32)
        self.normals = np.asarray(reference_normals, dtype=np.float32)
        if self.normals.shape != self.rest.shape:
            raise ValueError("Bread requires one reference normal per vertex")
        axes = [np.unique(self.reference_particles[:, d]) for d in range(3)]
        shape = tuple(len(a) for a in axes)
        if min(shape) < 2 or np.prod(shape) != len(self.reference_particles):
            raise ValueError("Bread binding requires the complete regular reference MPM lattice")
        indices = tuple(np.searchsorted(axes[d], self.reference_particles[:, d]) for d in range(3))
        flat = np.ravel_multi_index(indices, shape)
        if len(np.unique(flat)) != len(flat):
            raise ValueError("Duplicate reference particle coordinates")
        self.grid = np.empty(shape, dtype=np.int32)
        self.grid[indices] = np.arange(len(flat), dtype=np.int32)
        self.lower = np.empty_like(self.rest, dtype=np.int32)
        self.fraction = np.empty_like(self.rest)
        self.inverse_spacing = np.empty_like(self.rest)
        for d, axis in enumerate(axes):
            lower = np.clip(np.searchsorted(axis, self.rest[:, d], side="right") - 1, 0, len(axis) - 2)
            spacing = axis[lower + 1] - axis[lower]
            self.lower[:, d] = lower
            self.fraction[:, d] = (self.rest[:, d] - axis[lower]) / spacing
            self.inverse_spacing[:, d] = 1.0 / spacing
        # The geometric boundary lies half a particle cell beyond the outer
        # particle centers. Linear extrapolation there preserves rigid/affine
        # motion. Reject unrelated meshes instead of silently clamping them.
        if self.fraction.min() < -0.6 or self.fraction.max() > 1.6:
            raise ValueError("Bread mesh extends beyond the reference material cell boundary")

    def deform(self, particles):
        particles = np.asarray(particles, dtype=np.float32)
        if particles.shape != self.reference_particles.shape or not np.isfinite(particles).all():
            raise ValueError("Invalid bread particle frame")
        vertices, normals, determinants = _deform(
            self.rest,
            self.normals,
            self.lower,
            self.fraction,
            self.inverse_spacing,
            self.grid,
            particles - self.reference_particles,
        )
        if not np.isfinite(vertices).all() or not np.isfinite(normals).all() or determinants.min() <= 0:
            raise ValueError("Folded or non-finite bread display deformation")
        return vertices, normals, float(determinants.min())
