import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.misc import *
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


@qd.data_oriented
class CylinderBoundary:
    """
    Vertical-axis cylindrical boundary: radial clamp on xy plus a bottom z plane (and an
    optional top plane; z_top=None leaves the top open). Velocity handling follows
    CubeBoundary.impose_pos_vel: when a particle sits outside and moves outward, the outward
    (radial / z) velocity component is multiplied by -restitution; tangential components are
    left untouched.
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
        # escape_band (optional, multiflow MF-14): when set, the radial clamp and the bottom
        # plane only act within [radius, radius+escape_band]. Milk film clinging to the outer
        # wall (within the band) is still captured, but droplets that detach beyond the band
        # escape and fall to the table (PlaneBoundary) instead of being teleported back onto
        # the wall / lifted off the table. Default None = legacy behavior (bit-identical).
        self._has_escape_band = escape_band is not None
        self.escape_band = escape_band if escape_band is not None else 0.0

        self.center_xy = np.array(center_xy, dtype=gs.np_float)
        self.center_qd = qd.Vector(center_xy, dt=gs.qd_float)

    @qd.func
    def impose_pos_vel(self, pos, vel):
        # radial clamp: project particles beyond the cylinder wall back onto it.
        # With z_top set the wall is a finite open-top cup wall (multiflow MF-10): below the rim
        # the clamp is hard; above the rim it is band-limited (see `band`). Without z_top the
        # wall is infinitely tall (MF-7..9, bit-identical).
        dx = pos[0] - self.center_qd[0]
        dy = pos[1] - self.center_qd[1]
        r = qd.sqrt(dx * dx + dy * dy)
        clamp = r > self.radius
        if qd.static(self._has_top):
            if pos[2] > self.z_top:
                clamp = (r > self.radius) and (r < self.radius + self.band)
        if qd.static(self._has_escape_band):
            clamp = clamp and (r < self.radius + self.escape_band)
        if clamp:
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
        floor_ok = True
        if qd.static(self._has_escape_band):
            floor_ok = r <= self.radius + self.escape_band
        if floor_ok:
            if pos[2] <= self.z_bottom and vel[2] <= 0:
                vel[2] *= -self.restitution
            pos[2] = qd.max(pos[2], self.z_bottom)

        return pos, vel

    @qd.func
    def impose_pos(self, pos):
        dx = pos[0] - self.center_qd[0]
        dy = pos[1] - self.center_qd[1]
        r = qd.sqrt(dx * dx + dy * dy)
        clamp = r > self.radius
        if qd.static(self._has_top):
            if pos[2] > self.z_top:
                clamp = (r > self.radius) and (r < self.radius + self.band)
        if qd.static(self._has_escape_band):
            clamp = clamp and (r < self.radius + self.escape_band)
        if clamp:
            pos[0] = self.center_qd[0] + dx / r * self.radius
            pos[1] = self.center_qd[1] + dy / r * self.radius
        floor_ok = True
        if qd.static(self._has_escape_band):
            floor_ok = r <= self.radius + self.escape_band
        if floor_ok:
            pos[2] = qd.max(pos[2], self.z_bottom)
        return pos

    @qd.func
    def adhesion_query(self, pos):
        # Wall-adhesion query (multiflow MF-12 rev7, PBSTF adhesion-constraint port): returns
        # (ok, C, n) with C the signed distance to the NEAREST container surface (side wall or
        # bottom; positive inside the fluid) and n that surface's normal pointing INTO the
        # fluid. ok=False outside the adhesion region (beyond the wall, below the bottom, or
        # above the rim — an open rim has no wall to stick to).
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
        pos = np.asarray(pos)
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


@qd.data_oriented
class TiltedCylinderBoundary:
    """
    Open-mouth cylindrical container clamp with an arbitrary (tilted) axis (multiflow MF-10
    pitcher). `origin` is the center of the inner bottom plane, `axis` the unit vector from the
    bottom toward the mouth, `length` the bottom->rim distance along the axis. Particles are
    kept inside `radius` for 0 <= s <= length (side wall) and above the bottom plane (s >= 0).
    Past the rim (s > length) particles are FREE — this is how the container pours.

    The side clamp is band-limited to [radius, radius + band]: escapees from inside move slowly
    and are caught while crossing the band, while particles that have already poured out (the
    stream falling past the wall extension, where s < length but r >> radius) are NOT teleported
    back inside. Velocity handling follows CubeBoundary.impose_pos_vel (restitution on the
    outward normal component only).

    The pose lives in qd fields (NOT compile-time constants) so the container can be ANIMATED
    (multiflow MF-11 progressive tilt): call `set_pose(origin, axis)` from python every frame.
    `in_keep_region` decides per-particle boundary-group ownership (MF-11): a pitcher particle
    outside the keep region has left the container for good and transfers to the primary
    boundary's group.
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
        """Update the container pose (python-side; per-frame animation hook for MF-11)."""
        axis_np = np.asarray(axis, dtype=gs.np_float)
        axis_np = axis_np / np.linalg.norm(axis_np)
        self.origin_np = np.asarray(origin, dtype=gs.np_float)
        self.axis_np = axis_np
        self._origin[None] = qd.Vector(self.origin_np, dt=gs.qd_float)
        self._axis[None] = qd.Vector(axis_np, dt=gs.qd_float)

    @qd.func
    def in_keep_region(self, pos, margin):
        # True while the particle still belongs to this container: past the rim, beyond the
        # escape band, or below the bottom plane means it has left for good (MF-11 transfer).
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
        # Wall-adhesion query against the CURRENT (possibly animated) pose — same convention
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
        pos = np.asarray(pos)
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


@qd.data_oriented
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
        pos = np.asarray(pos)
        return bool(np.all(pos[..., 2] > self.z0))

    def __repr__(self):
        return (
            f"{brief(self)}\n"
            f"z0          : {brief(self.z0)}\n"
            f"center_xy   : {brief(self.center_xy)}\n"
            f"radius      : {brief(self.radius)}\n"
            f"restitution : {brief(self.restitution)}"
        )


@qd.data_oriented
class TiltedCylinderShellBoundary:
    """
    Tilted cylindrical cup SHELL boundary (multiflow P2, teapot-effect demo): the pitcher wall
    has finite thickness and an OUTER surface, unlike TiltedCylinderBoundary's cavity-only
    clamp. Exact analytic SDF: a solid of revolution's 3D signed distance equals the 2D SDF of
    its (r, s) half-plane cross-section, here an L-shape = union of two boxes
      B_wall  = [R_in, R_out] x [-t_bottom, L]   (side wall band)
      B_floor = [0,    R_out] x [-t_bottom, 0]   (bottom disk)
    with R_out = R_in + t_wall, origin = inner-bottom center, axis = bottom->mouth unit vector
    (same dynamic qd-field pose as TiltedCylinderBoundary; animate per frame via set_pose).
    `lip_round` offsets the SDF outward (dilates the solid by rho): the sharp lip corner becomes
    a round one with continuous normals for free.

    impose_pos_vel/impose_pos act ONLY inside the solid (d < 0), pushing the particle to the
    nearest surface (inner wall / outer wall / lip / underside — automatic). Everything outside
    the solid is FREE space: milk sliding down the outer wall is never clamped (its position
    constraint lives in the wall-adhesion query). Crossing more than half the wall thickness
    flips the nearest surface to the OUTER wall, i.e. a tunnel — monitored via the n_leak
    metric (P2 design R1).

    adhesion_query follows the PBSTF solid-centered signed convention: C = signed distance
    (positive OUTSIDE the solid, zero on the surface), n = outward gradient, ok = |C| <=
    particle_radius (two-sided: inner wall, outer wall, lip top and cup underside all adhere).
    This matches the P1 convention of CylinderBoundary/PlaneBoundary (C >= 0 on the fluid side,
    n pointing into the fluid), so the solver's nearest-surface arbitration stays meaningful.

    in_keep_region (boundary_group ownership): the radial upper bound is R_out + band + margin
    (NOT R_in — an outer-wall film particle r in [R_out, R_out+band] still belongs to the
    pitcher), the axial bounds are [-t_bottom - margin, L + margin], and a past-rim particle
    still adhering to the lip/outer wall (|d| <= particle_radius) is exempted from transfer.
    """

    def __init__(
        self, origin, axis, radius, length, t_wall, t_bottom, lip_round=0.0, band=0.0, restitution=0.0
    ):
        self.restitution = restitution
        self.radius = radius  # R_in (cavity radius)
        self.length = length  # L: inner bottom -> rim distance along the axis
        self.t_wall = t_wall
        self.t_bottom = t_bottom
        self.lip_round = lip_round
        self.band = band
        self.r_out = radius + t_wall

        self._origin = qd.Vector.field(3, gs.qd_float, shape=())
        self._axis = qd.Vector.field(3, gs.qd_float, shape=())
        self.set_pose(origin, axis)

    def set_pose(self, origin, axis):
        """Update the shell pose (python-side; per-frame animation hook, same as the pitcher)."""
        axis_np = np.asarray(axis, dtype=gs.np_float)
        axis_np = axis_np / np.linalg.norm(axis_np)
        self.origin_np = np.asarray(origin, dtype=gs.np_float)
        self.axis_np = axis_np
        self._origin[None] = qd.Vector(self.origin_np, dt=gs.qd_float)
        self._axis[None] = qd.Vector(axis_np, dt=gs.qd_float)

    @qd.func
    def _box2(self, r, s, cr, cs, hr, hs):
        # 2D box SDF plus closest point and (r, s)-space gradient. Tie at the wall centerline
        # (dr == 0) snaps to the INNER face (toward the cavity): a particle caught exactly
        # mid-wall is pushed back into the cup, not spat onto the outer wall (P2 design 2.2).
        dr = r - cr
        ds = s - cs
        qr = qd.abs(dr) - hr
        qs = qd.abs(ds) - hs
        out_r = qd.max(qr, 0.0)
        out_s = qd.max(qs, 0.0)
        sd = qd.sqrt(out_r * out_r + out_s * out_s) + qd.min(qd.max(qr, qs), 0.0)
        close_r = qd.min(qd.max(r, cr - hr), cr + hr)
        close_s = qd.min(qd.max(s, cs - hs), cs + hs)
        if qr < 0.0 and qs < 0.0:
            # strictly inside: snap the dominant (nearest-face) component onto that face
            if qr >= qs:
                if dr > 0.0:
                    close_r = cr + hr
                else:
                    close_r = cr - hr
            else:
                if ds > 0.0:
                    close_s = cs + hs
                else:
                    close_s = cs - hs
        g_r = r - close_r
        g_s = s - close_s
        g_len = qd.sqrt(g_r * g_r + g_s * g_s)
        inv = 1.0 / qd.max(g_len, gs.EPS)
        return sd, close_r, close_s, g_r * inv, g_s * inv

    @qd.func
    def _sdf(self, pos):
        # Exact shell SDF (minus lip_round), outward gradient, and closest point on the
        # (rounded) surface. Union of the two cross-section boxes; ties keep the wall box.
        rel = pos - self._origin[None]
        a = self._axis[None]
        s = rel.dot(a)
        radial = rel - s * a
        r = radial.norm()
        e_r = qd.Vector.zero(gs.qd_float, 3)
        if r > gs.EPS:
            e_r = radial / r
        else:
            # on the axis the radial direction is degenerate: any unit vector perpendicular
            # to the axis (deterministic fallback)
            ref = qd.Vector([1.0, 0.0, 0.0])
            if qd.abs(a.dot(ref)) > 0.9:
                ref = qd.Vector([0.0, 1.0, 0.0])
            e_r = ref - ref.dot(a) * a
            e_r = e_r / e_r.norm()
        # B_wall = [R_in, R_out] x [-t_bottom, L]
        sd_w, cr_w, cs_w, gr_w, gs_w = self._box2(
            r,
            s,
            0.5 * (self.radius + self.r_out),
            0.5 * (self.length - self.t_bottom),
            0.5 * self.t_wall,
            0.5 * (self.length + self.t_bottom),
        )
        # B_floor = [0, R_out] x [-t_bottom, 0]
        sd_f, cr_f, cs_f, gr_f, gs_f = self._box2(
            r, s, 0.5 * self.r_out, -0.5 * self.t_bottom, 0.5 * self.r_out, 0.5 * self.t_bottom
        )
        sd = sd_w
        close_r = cr_w
        close_s = cs_w
        g_r = gr_w
        g_s = gs_w
        if sd_f < sd_w:
            sd = sd_f
            close_r = cr_f
            close_s = cs_f
            g_r = gr_f
            g_s = gs_f
        n = g_r * e_r + g_s * a
        closest = self._origin[None] + close_s * a + close_r * e_r + self.lip_round * n
        d = sd - self.lip_round
        return d, n, closest

    @qd.func
    def impose_pos_vel(self, pos, vel):
        # Solid-interior projection only: d < 0 pushes to the NEAREST surface (inner wall,
        # outer wall, lip, underside — automatic); outside the solid is free space.
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
        keep = (
            s <= self.length + margin
            and r <= self.r_out + self.band + margin
            and s >= -self.t_bottom - margin
        )
        if not keep and s > self.length + margin:
            d, n, closest = self._sdf(pos)
            keep = qd.abs(d) <= particle_radius
        return keep

    def is_inside(self, pos):
        # inside the SHELL SOLID (numpy-side diagnostics, e.g. the n_leak metric)
        pos = np.asarray(pos)
        rel = pos - self.origin_np
        s = rel @ self.axis_np
        r = np.linalg.norm(rel - s[..., None] * self.axis_np, axis=-1)
        in_wall = (r >= self.radius) & (r <= self.r_out) & (s >= -self.t_bottom) & (s <= self.length)
        in_floor = (r <= self.r_out) & (s >= -self.t_bottom) & (s <= 0.0)
        return bool(np.all(in_wall | in_floor))

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
