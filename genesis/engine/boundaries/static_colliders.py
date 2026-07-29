from abc import ABC, abstractmethod

import numpy as np
import quadrants as qd

import genesis as gs


@qd.data_oriented
class StaticCollider(ABC):
    """Base class for analytic, immovable solid obstacles.

    Unlike a :mod:`genesis.engine.boundaries` container, a static collider
    excludes particles from the inside of a solid. Geometry subclasses provide
    one surface query; the shared helpers expose the individual query results
    and position projection to GPU kernels.
    """

    @classmethod
    @abstractmethod
    def from_options(cls, options):
        """Construct a collider from its solver-facing options object."""

    @abstractmethod
    @qd.func
    def _query_surface(self, pos):
        """Return ``(closest_position, outward_normal, is_inside)``."""
        raise NotImplementedError

    @qd.func
    def closest_position(self, pos):
        closest, _, _ = self._query_surface(pos)
        return closest

    @qd.func
    def closest_normal(self, pos):
        _, normal, _ = self._query_surface(pos)
        return normal

    @qd.func
    def is_inside(self, pos):
        _, _, inside = self._query_surface(pos)
        return inside

    @qd.func
    def project_out(self, pos):
        closest, _, inside = self._query_surface(pos)
        if inside:
            pos = closest
        return pos

    @qd.func
    def separates(self, pos_i, pos_j, particle_radius):
        """Whether the collider blocks a particle-particle interaction.

        Most collider types do not need this optional topology filter. A
        geometry that does need it can override this method.
        """
        return False


class ConeStaticCollider(StaticCollider):
    """Finite analytic cone matching the C++ ``ImplicitCone`` semantics.

    ``center`` is the center of the base disk, ``height`` points from the base
    center to the apex, and ``radius`` is the base radius.
    """

    type = "cone"

    def __init__(self, center, height, radius):
        self.center = np.asarray(center, dtype=gs.np_float)
        self.height = np.asarray(height, dtype=gs.np_float)
        self.radius = float(radius)

        if self.center.shape != (3,):
            gs.raise_exception("Cone static collider `center` must have shape (3,).")
        if self.height.shape != (3,):
            gs.raise_exception("Cone static collider `height` must have shape (3,).")
        if np.linalg.norm(self.height) <= gs.EPS:
            gs.raise_exception("Cone static collider `height` must be non-zero.")
        if self.radius <= 0.0:
            gs.raise_exception("Cone static collider `radius` must be positive.")

        self._center_qd = qd.Vector(self.center, dt=gs.qd_float)
        self._height_qd = qd.Vector(self.height, dt=gs.qd_float)

    @classmethod
    def from_options(cls, options):
        return cls(center=options.center, height=options.height, radius=options.radius)

    @qd.func
    def _radial_direction(self, pos):
        axis = self._height_qd.normalized()
        axial_distance = (pos - self._center_qd).dot(axis)
        radial = pos - self._center_qd - axial_distance * axis
        if radial.norm_sqr() <= gs.EPS**2:
            fallback = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
            if axis[0] >= 0.1:
                fallback = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
            radial = axis.cross(fallback).normalized()
        else:
            radial = radial.normalized()
        return radial

    @qd.func
    def _query_surface(self, pos):
        axis = self._height_qd.normalized()
        axial_distance = (pos - self._center_qd).dot(axis)
        radial = self._radial_direction(pos)

        base_position = pos - axial_distance * axis
        inside_base = axial_distance >= 0.0
        if (base_position - self._center_qd).norm() > self.radius:
            base_position = self._center_qd + self.radius * radial
            inside_base = False

        side_direction = (self.radius * radial - self._height_qd).normalized()
        side_parameter = (pos - self._center_qd - self._height_qd).dot(side_direction)
        side_position = self._center_qd + self._height_qd
        side_normal = axis
        inside_side = False
        slant_length_sqr = self._height_qd.norm_sqr() + self.radius * self.radius
        if side_parameter >= 0.0:
            if side_parameter * side_parameter > slant_length_sqr:
                side_position = self._center_qd + self.radius * radial
                side_normal = -axis
            else:
                side_position = self._center_qd + self._height_qd + side_parameter * side_direction
                side_normal = (radial * self._height_qd.norm() + axis * self.radius).normalized()
                inside_side = (pos - side_position).dot(side_normal) <= 0.0

        closest_position = side_position
        closest_normal = side_normal
        inside = inside_side
        if (base_position - pos).norm_sqr() < (side_position - pos).norm_sqr():
            closest_position = base_position
            closest_normal = -axis
            inside = inside_base

        return closest_position, closest_normal, inside

    @qd.func
    def separates(self, pos_i, pos_j, particle_radius):
        closest_i, normal_i, _ = self._query_surface(pos_i)
        closest_j, normal_j, _ = self._query_surface(pos_j)
        return (
            (pos_i - closest_i).norm() <= particle_radius
            and (pos_j - closest_j).norm() <= particle_radius
            and normal_i.dot(normal_j) < 0.0
        )


_STATIC_COLLIDER_TYPES = {
    ConeStaticCollider.type: ConeStaticCollider,
}


def create_static_collider(options) -> StaticCollider:
    """Create the analytic collider selected by ``options.type``."""
    collider_cls = _STATIC_COLLIDER_TYPES.get(options.type)
    if collider_cls is None:
        gs.raise_exception(f"Unsupported static collider type: {options.type!r}.")
    return collider_cls.from_options(options)


__all__ = ["StaticCollider", "ConeStaticCollider", "create_static_collider"]
