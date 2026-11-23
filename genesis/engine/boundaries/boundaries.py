import numpy as np

import quadrants as qd

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.utils.repr import brief


@qd.data_oriented
class CubeBoundary:
    def __init__(self, lower, upper, restitution=0.0):
        self.restitution = restitution

        self.upper = np.array(upper, dtype=gs.np_float)
        self.lower = np.array(lower, dtype=gs.np_float)
        assert (self.upper >= self.lower).all()

        self.upper_qd = qd.Vector(upper, dt=gs.qd_float)
        self.lower_qd = qd.Vector(lower, dt=gs.qd_float)

    @qd.func
    def impose_pos_vel(self, pos, vel):
        for i in qd.static(range(3)):
            if pos[i] >= self.upper_qd[i] and vel[i] >= 0:
                vel[i] *= -self.restitution
            elif pos[i] <= self.lower_qd[i] and vel[i] <= 0:
                vel[i] *= -self.restitution

        pos = qd.max(qd.min(pos, self.upper_qd), self.lower_qd)

        return pos, vel

    @qd.func
    def impose_pos(self, pos):
        pos = qd.max(qd.min(pos, self.upper_qd), self.lower_qd)
        return pos

    def is_inside(self, pos):
        return np.all(pos < self.upper) and np.all(pos > self.lower)

    def __repr__(self):
        return (
            f"{brief(self)}\n"
            f"lower       : {brief(self.lower)}\n"
            f"upper       : {brief(self.upper)}\n"
            f"restitution : {brief(self.restitution)}"
        )


class CylinderBoundary:
    """Upright open cylindrical cavity with a floor at ``z_bottom``.

    ``z_top`` sets the rim height; above it, radial contact acts within ``band`` of the wall. With no rim, the
    wall extends indefinitely. ``escape_band`` limits radial and floor contact to nearby particles so detached
    droplets can fall freely. Restitution reflects outward normal velocity while preserving tangential motion.
    """

    def __init__(self, center_xy, radius, z_bottom, z_top=None, restitution=0.0, band=0.0, escape_band=None):
        self.restitution = restitution
        self.radius = radius
        self.z_bottom = z_bottom
        self.z_top = z_top
        self._has_top = z_top is not None
        # above-rim escape band (only used when z_top is set): particles crossing the wall
        # above the rim are caught only within [radius, radius+band], so distant containers
        # hovering over the cup are not sucked in but slow splash ejecta are kept
        self.band = band
        self._has_escape_band = escape_band is not None
        self.escape_band = escape_band if escape_band is not None else 0.0

        self.center_xy = np.array(center_xy, dtype=gs.np_float)
        self.center_qd = qd.Vector(center_xy, dt=gs.qd_float)

    @qd.func
    def impose_pos_vel(self, pos, vel):
        dx = pos[0] - self.center_qd[0]
        dy = pos[1] - self.center_qd[1]
        r = qd.sqrt(dx * dx + dy * dy)
        is_clamped = r > self.radius
        if qd.static(self._has_top):
            if pos[2] > self.z_top:
                is_clamped = (r > self.radius) and (r < self.radius + self.band)
        if qd.static(self._has_escape_band):
            is_clamped = is_clamped and (r < self.radius + self.escape_band)
        if is_clamped:
            nx = dx / r
            ny = dy / r
            vr = vel[0] * nx + vel[1] * ny
            if vr >= 0:
                # remove/reflect the outward radial component, keep the tangential part
                vel[0] -= (1.0 + self.restitution) * vr * nx
                vel[1] -= (1.0 + self.restitution) * vr * ny
            pos[0] = self.center_qd[0] + nx * self.radius
            pos[1] = self.center_qd[1] + ny * self.radius

        # bottom plane (escape_band mode: r-gated so table droplets outside the band
        # are left to the PlaneBoundary instead of being lifted to z_bottom)
        is_floor_active = True
        if qd.static(self._has_escape_band):
            is_floor_active = r <= self.radius + self.escape_band
        if is_floor_active:
            if pos[2] <= self.z_bottom and vel[2] <= 0:
                vel[2] *= -self.restitution
            pos[2] = qd.max(pos[2], self.z_bottom)

        return pos, vel

    @qd.func
    def impose_pos(self, pos):
        dx = pos[0] - self.center_qd[0]
        dy = pos[1] - self.center_qd[1]
        r = qd.sqrt(dx * dx + dy * dy)
        is_clamped = r > self.radius
        if qd.static(self._has_top):
            if pos[2] > self.z_top:
                is_clamped = (r > self.radius) and (r < self.radius + self.band)
        if qd.static(self._has_escape_band):
            is_clamped = is_clamped and (r < self.radius + self.escape_band)
        if is_clamped:
            pos[0] = self.center_qd[0] + dx / r * self.radius
            pos[1] = self.center_qd[1] + dy / r * self.radius
        is_floor_active = True
        if qd.static(self._has_escape_band):
            is_floor_active = r <= self.radius + self.escape_band
        if is_floor_active:
            pos[2] = qd.max(pos[2], self.z_bottom)
        return pos

    @qd.func
    def adhesion_query(self, pos):
        dx = pos[0] - self.center_qd[0]
        dy = pos[1] - self.center_qd[1]
        r = qd.sqrt(dx * dx + dy * dy)
        ok = r <= self.radius and pos[2] >= self.z_bottom
        if qd.static(self._has_top):
            ok = ok and pos[2] <= self.z_top
        c_side = self.radius - r
        c_bottom = pos[2] - self.z_bottom
        C = c_bottom
        n = qd.Vector([0.0, 0.0, 1.0])
        if c_side < c_bottom:
            C = c_side
            inv_r = 1.0 / qd.max(r, gs.EPS)
            n = qd.Vector([-dx * inv_r, -dy * inv_r, 0.0])
        return ok, C, n

    def is_inside(self, pos):
        # z_top is an open rim (wall height), not a domain lid: the region above the rim with
        # r < radius is still inside the simulation domain (e.g. a falling pour stream).
        pos = tensor_to_array(pos)
        r = np.linalg.norm(pos[..., :2] - self.center_xy, axis=-1)
        z_ok = pos[..., 2] > self.z_bottom
        return bool(np.all(r < self.radius) and np.all(z_ok))

    def __repr__(self):
        return (
            f"{brief(self)}\n"
            f"center_xy   : {brief(self.center_xy)}\n"
            f"radius      : {brief(self.radius)}\n"
            f"z_bottom    : {brief(self.z_bottom)}\n"
            f"z_top       : {brief(self.z_top)}\n"
            f"restitution : {brief(self.restitution)}"
        )


class TiltedCylinderBoundary:
    """Movable open cylindrical cavity whose axis points from the inner floor toward the mouth.

    Side contact acts within ``band`` of the wall between the floor and rim. Particles beyond that region can
    pour freely. ``set_pose`` updates the origin and axis; ``in_keep_region`` determines container membership.
    """

    def __init__(self, origin, axis, radius, length, band, restitution=0.0):
        self.restitution = restitution
        self.radius = radius
        self.length = length
        self.band = band

        self._origin = qd.Vector.field(3, gs.qd_float, shape=())
        self._axis = qd.Vector.field(3, gs.qd_float, shape=())
        self.set_pose(origin, axis)

    def set_pose(self, origin, axis):
        """Update the container origin and normalize its non-zero axis."""
        axis_np = tensor_to_array(axis)
        axis_length = np.linalg.norm(axis_np)
        if axis_length <= gs.EPS or not np.isfinite(axis_length):
            gs.raise_exception("Boundary axis must be finite and non-zero.")
        axis_np = axis_np / axis_length
        self.origin_np = tensor_to_array(origin)
        self.axis_np = axis_np
        self._origin.from_numpy(self.origin_np)
        self._axis.from_numpy(axis_np)

    @qd.func
    def in_keep_region(self, pos, margin):
        rel = pos - self._origin[None]
        s = rel.dot(self._axis[None])
        r = (rel - s * self._axis[None]).norm()
        return s <= self.length + margin and r <= self.radius + self.band + margin and s >= -margin

    @qd.func
    def impose_pos_vel(self, pos, vel):
        rel = pos - self._origin[None]
        s = rel.dot(self._axis[None])
        radial = rel - s * self._axis[None]
        r = radial.norm()

        # bottom plane: hold particles above s=0 (only within the bottom disk footprint)
        if s < 0.0 and r <= self.radius + self.band:
            vs = vel.dot(self._axis[None])
            if vs <= 0.0:
                vel -= (1.0 + self.restitution) * vs * self._axis[None]
            pos -= s * self._axis[None]
            s = 0.0

        # side wall: band-limited radial clamp (see class docstring)
        if 0.0 <= s and s <= self.length and self.radius < r and r < self.radius + self.band:
            n = radial / r
            vr = vel.dot(n)
            if vr >= 0.0:
                vel -= (1.0 + self.restitution) * vr * n
            pos -= (r - self.radius) * n

        return pos, vel

    @qd.func
    def impose_pos(self, pos):
        rel = pos - self._origin[None]
        s = rel.dot(self._axis[None])
        radial = rel - s * self._axis[None]
        r = radial.norm()
        if s < 0.0 and r <= self.radius + self.band:
            pos -= s * self._axis[None]
            s = 0.0
        if 0.0 <= s and s <= self.length and self.radius < r and r < self.radius + self.band:
            pos -= (r - self.radius) * (radial / r)
        return pos

    @qd.func
    def adhesion_query(self, pos):
        # Wall-adhesion query against the CURRENT (possibly animated) pose -- same convention
        # as CylinderBoundary.adhesion_query: (ok, C, n), C = signed distance to the nearest
        # surface (side wall or bottom, positive inside the fluid), n pointing into the fluid.
        # ok=False outside the cavity (past the rim = poured out, beyond the wall, below the
        # bottom).
        rel = pos - self._origin[None]
        s = rel.dot(self._axis[None])
        radial = rel - s * self._axis[None]
        r = radial.norm()
        ok = 0.0 <= s and s <= self.length and r <= self.radius
        c_side = self.radius - r
        c_bottom = s
        C = c_bottom
        n = self._axis[None]
        if c_side < c_bottom:
            C = c_side
            n = -radial / qd.max(r, gs.EPS)
        return ok, C, n

    def is_inside(self, pos):
        pos = tensor_to_array(pos)
        rel = pos - self.origin_np
        s = rel @ self.axis_np
        r = np.linalg.norm(rel - s[..., None] * self.axis_np, axis=-1)
        return bool(np.all(s > 0.0) and np.all(s < self.length) and np.all(r < self.radius))

    def __repr__(self):
        return (
            f"{brief(self)}\n"
            f"origin      : {brief(self.origin_np)}\n"
            f"axis        : {brief(self.axis_np)}\n"
            f"radius      : {brief(self.radius)}\n"
            f"length      : {brief(self.length)}\n"
            f"band        : {brief(self.band)}\n"
            f"restitution : {brief(self.restitution)}"
        )


@qd.data_oriented
class FloorBoundary:
    def __init__(self, height, restitution=0.0):
        self.height = height
        self.restitution = restitution

    @qd.func
    def impose_pos_vel(self, pos, vel):
        if pos[2] <= self.height and vel[2] <= 0:
            vel[2] *= -self.restitution

        pos[2] = qd.max(pos[2], self.height)

        return pos, vel

    @qd.func
    def impose_pos(self, pos):
        pos[2] = qd.max(pos[2], self.height)
        return pos

    def __repr__(self):
        return f"{brief(self)}\nheight      : {brief(self.height)}\nrestitution : {brief(self.restitution)}"


class PlaneBoundary:
    """
    Horizontal table plane z = z0 (multiflow MF-13). With radius=None the plane is infinite;
    otherwise the clamp and the adhesion query apply only within the disk (center_xy, radius).
    Velocity handling follows CubeBoundary.impose_pos_vel (restitution on the downward normal
    component only). `adhesion_query` follows the CylinderBoundary convention: (ok, C, n) with
    C = z - z0 the signed distance to the surface (positive above the table) and n = +z;
    ok=False below the plane or outside the finite extent (nothing to stick to there).
    """

    def __init__(self, z0, center_xy=(0.0, 0.0), radius=None, restitution=0.0):
        self.z0 = z0
        self.restitution = restitution
        self._finite = radius is not None
        self.radius = radius if radius is not None else 0.0

        self.center_xy = np.array(center_xy, dtype=gs.np_float)
        self.center_qd = qd.Vector(center_xy, dt=gs.qd_float)

    @qd.func
    def _in_extent(self, pos):
        ok = True
        if qd.static(self._finite):
            dx = pos[0] - self.center_qd[0]
            dy = pos[1] - self.center_qd[1]
            ok = dx * dx + dy * dy <= self.radius * self.radius
        return ok

    @qd.func
    def impose_pos_vel(self, pos, vel):
        if self._in_extent(pos) and pos[2] < self.z0:
            if vel[2] <= 0:
                vel[2] *= -self.restitution
            pos[2] = self.z0
        return pos, vel

    @qd.func
    def impose_pos(self, pos):
        if self._in_extent(pos):
            pos[2] = qd.max(pos[2], self.z0)
        return pos

    @qd.func
    def adhesion_query(self, pos):
        ok = self._in_extent(pos) and pos[2] >= self.z0
        C = pos[2] - self.z0
        n = qd.Vector([0.0, 0.0, 1.0])
        return ok, C, n

    def is_inside(self, pos):
        pos = tensor_to_array(pos)
        return bool(np.all(pos[..., 2] > self.z0))

    def __repr__(self):
        return (
            f"{brief(self)}\n"
            f"z0          : {brief(self.z0)}\n"
            f"center_xy   : {brief(self.center_xy)}\n"
            f"radius      : {brief(self.radius)}\n"
            f"restitution : {brief(self.restitution)}"
        )


class TiltedCylinderShellBoundary(TiltedCylinderBoundary):
    """Movable cup shell with finite wall and floor thickness.

    The signed distance function (SDF) measures distance to the five exposed edges of the radial/axial cross-section.
    ``lip_round`` dilates the shell to round its edges. Collision projects embedded particles to the
    nearest surface; adhesion acts on the inner wall, outer wall, lip and underside within a particle radius.
    Container membership includes attached exterior films until they detach from the shell.
    """

    def __init__(self, origin, axis, radius, length, t_wall, t_bottom, lip_round=0.0, band=0.0, restitution=0.0):
        super().__init__(origin, axis, radius, length, band, restitution)
        self.t_wall = t_wall
        self.t_bottom = t_bottom
        self.lip_round = lip_round
        self.r_out = radius + t_wall

    @qd.func
    def _sdf(self, pos):
        rel = pos - self._origin[None]
        axis = self._axis[None]
        s = rel.dot(axis)
        radial = rel - s * axis
        r = radial.norm()
        e_r = qd.Vector.zero(gs.qd_float, 3)
        if r > gs.EPS:
            e_r = radial / r
        else:
            ref = qd.Vector([1.0, 0.0, 0.0])
            if qd.abs(axis.dot(ref)) > 0.9:
                ref = qd.Vector([0.0, 1.0, 0.0])
            e_r = (ref - ref.dot(axis) * axis).normalized()

        # The five exposed faces bound the solid; the axis is an interior coordinate seam.
        candidates = qd.Matrix(
            [
                [self.radius, qd.min(qd.max(s, 0.0), self.length)],
                [qd.min(qd.max(r, self.radius), self.r_out), self.length],
                [self.r_out, qd.min(qd.max(s, -self.t_bottom), self.length)],
                [qd.min(r, self.r_out), -self.t_bottom],
                [qd.min(r, self.radius), 0.0],
            ]
        )
        normals = qd.Matrix(
            [
                [-1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, -1.0],
                [0.0, 1.0],
            ]
        )
        distance_squared = gs.qd_float(1.0e30)
        closest_rs = qd.Vector.zero(gs.qd_float, 2)
        normal_rs = qd.Vector.zero(gs.qd_float, 2)
        for i in qd.static(range(5)):
            delta = qd.Vector([r, s]) - candidates[i, :]
            if delta.norm_sqr() < distance_squared:
                distance_squared = delta.norm_sqr()
                closest_rs = candidates[i, :]
                normal_rs = normals[i, :]
        distance = qd.sqrt(distance_squared)
        is_in_solid = r <= self.r_out and -self.t_bottom <= s and s <= self.length
        is_in_solid = is_in_solid and (s <= 0.0 or r >= self.radius)
        if is_in_solid:
            distance = -distance
        if qd.abs(distance) > gs.EPS:
            normal_rs = (qd.Vector([r, s]) - closest_rs) / distance
        normal = normal_rs[0] * e_r + normal_rs[1] * axis
        closest = self._origin[None] + closest_rs[0] * e_r + closest_rs[1] * axis + self.lip_round * normal
        return distance - self.lip_round, normal, closest

    @qd.func
    def impose_pos_vel(self, pos, vel):
        # Solid-interior projection only: d < 0 pushes to the NEAREST surface (inner wall,
        # outer wall, lip, underside -- automatic); outside the solid is free space.
        d, n, closest = self._sdf(pos)
        if d < 0.0:
            vn = vel.dot(n)
            if vn < 0.0:
                vel -= (1.0 + self.restitution) * vn * n
            pos = closest
        return pos, vel

    @qd.func
    def impose_pos(self, pos):
        d, n, closest = self._sdf(pos)
        if d < 0.0:
            pos = closest
        return pos

    @qd.func
    def adhesion_query(self, pos, particle_radius):
        # Two-sided solid-centered query (PBSTF convention): ok within one particle radius of
        # ANY shell surface (inner wall, outer wall, lip top, cup underside).
        d, n, closest = self._sdf(pos)
        ok = qd.abs(d) <= particle_radius
        return ok, d, n

    @qd.func
    def in_keep_region(self, pos, margin, particle_radius):
        # boundary_group ownership: the particle still belongs to this shell while inside the
        # (outer wall + escape band) footprint between underside and rim. A past-rim particle
        # still adhering to the lip / outer wall (|d| <= particle_radius) is NOT transferred.
        rel = pos - self._origin[None]
        s = rel.dot(self._axis[None])
        r = (rel - s * self._axis[None]).norm()
        keep = s <= self.length + margin and r <= self.r_out + self.band + margin and s >= -self.t_bottom - margin
        if not keep and s > self.length + margin:
            d, n, closest = self._sdf(pos)
            keep = qd.abs(d) <= particle_radius
        return keep

    def is_inside(self, pos):
        # inside the SHELL SOLID (numpy-side diagnostics, e.g. the n_leak metric)
        pos = tensor_to_array(pos)
        rel = pos - self.origin_np
        s = rel @ self.axis_np
        r = np.linalg.norm(rel - s[..., None] * self.axis_np, axis=-1)
        is_in_wall = (r >= self.radius) & (r <= self.r_out) & (s >= -self.t_bottom) & (s <= self.length)
        is_in_floor = (r <= self.r_out) & (s >= -self.t_bottom) & (s <= 0.0)
        return bool(np.all(is_in_wall | is_in_floor))

    def __repr__(self):
        return (
            f"{brief(self)}\n"
            f"origin      : {brief(self.origin_np)}\n"
            f"axis        : {brief(self.axis_np)}\n"
            f"radius      : {brief(self.radius)}\n"
            f"length      : {brief(self.length)}\n"
            f"t_wall      : {brief(self.t_wall)}\n"
            f"t_bottom    : {brief(self.t_bottom)}\n"
            f"lip_round   : {brief(self.lip_round)}\n"
            f"band        : {brief(self.band)}\n"
            f"restitution : {brief(self.restitution)}"
        )
