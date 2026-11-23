"""Prescribed solid boundaries for fluid implicit particle (FLIP) flow."""

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import quadrants as qd

import genesis as gs
from genesis.engine.boundaries.static_colliders import MeshStaticCollider, SphereStaticCollider, query_static_collider
from genesis.utils import geom
from genesis.utils.array_class import ErrorCode
from genesis.utils.misc import tensor_to_array


def make_flux_compatible(diagonal, rhs, is_fluid, face_weights, fractions):
    """Remove mean flux and fix the pressure gauge in closed components of a weighted grid Laplacian.

    Arrays share cell-axis order; each face array extends its corresponding axis by one. The diagonal and right-hand
    side are updated in place. Returns whether a closed component requires correction.
    """
    n_cells = np.count_nonzero(is_fluid)
    if n_cells == 0:
        return False
    indices = np.full(is_fluid.shape, -1, dtype=gs.np_int)
    indices[is_fluid] = np.arange(n_cells)
    sources, targets, coefficients = [], [], []
    for axis, weights in enumerate(face_weights):
        lower, upper, interior = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
        lower[axis], upper[axis], interior[axis] = slice(None, -1), slice(1, None), slice(1, -1)
        lower, upper, interior = tuple(lower), tuple(upper), tuple(interior)
        is_connected = is_fluid[lower] & is_fluid[upper] & (weights[interior] > 0.0)
        sources.append(indices[lower][is_connected])
        targets.append(indices[upper][is_connected])
        coefficients.append((weights[interior] * 0.5 * (fractions[lower] + fractions[upper]))[is_connected])
    sources, targets, coefficients = np.concatenate(sources), np.concatenate(targets), np.concatenate(coefficients)
    graph = csr_matrix((coefficients, (sources, targets)), shape=(n_cells, n_cells))
    n_components, labels = connected_components(graph, directed=False)
    row_sum = np.bincount(sources, weights=coefficients, minlength=n_cells)
    row_sum += np.bincount(targets, weights=coefficients, minlength=n_cells)
    fluid_diagonal = diagonal[is_fluid]
    leakage = np.bincount(labels, weights=np.maximum(fluid_diagonal - row_sum, 0.0), minlength=n_components)
    scale = np.bincount(labels, weights=fluid_diagonal, minlength=n_components)
    is_closed = leakage <= 8.0 * gs.EPS * scale
    if not is_closed.any():
        return False
    # A sealed component admits only zero net flux; subcell squeeze flow is resolved by particle-solid projection.
    mean_flux = np.bincount(labels, weights=rhs[is_fluid], minlength=n_components) / np.bincount(labels)
    mean_flux[~is_closed] = 0.0
    rhs[is_fluid] -= mean_flux[labels]
    _, first = np.unique(labels, return_index=True)
    fluid_diagonal[first[is_closed]] += 1.0
    diagonal[is_fluid] = fluid_diagonal
    return True


class FLIPBoundary:
    """Cache fixed geometry and interpolate prescribed poses over one scene step."""

    def __init__(self, solver, options):
        self.geoms = tuple(
            MeshStaticCollider(option.file, option.scale, option.sdf_res, option.pos, option.quat)
            if isinstance(option, gs.options.FLIPMeshStaticColliderOptions)
            else SphereStaticCollider.from_options(option)
            for option in options
        )
        self.is_kinematic = tuple(option.is_kinematic for option in options)
        self.start_pos = np.array([option.pos for option in options], dtype=gs.np_float)
        self.start_quat = np.array([option.quat for option in options], dtype=gs.np_float)
        self.target_pos = self.start_pos.copy()
        self.target_quat = self.start_quat.copy()
        self.velocity = np.zeros_like(self.start_pos)
        self.angular_velocity = np.zeros_like(self.start_pos)
        self.elapsed = 0.0
        self.has_motion = False
        self.pos = qd.field(gs.qd_vec3, shape=(len(options), 1))
        self.quat = qd.field(gs.qd_vec4, shape=(len(options), 1))
        self.vel = qd.field(gs.qd_vec3, shape=(len(options), 1))
        self.ang = qd.field(gs.qd_vec3, shape=(len(options), 1))
        self.pos.from_numpy(self.start_pos[:, None])
        self.quat.from_numpy(self.start_quat[:, None])
        self.vel.fill(0.0)
        self.ang.fill(0.0)
        shape = solver.levelset.shape
        self.fixed_nodes = qd.field(gs.qd_float, shape=tuple(n + 1 for n in shape))
        self.nodes = qd.field(gs.qd_float, shape=self.fixed_nodes.shape)
        self.fixed_cells = qd.field(gs.qd_float, shape=shape)
        self.cells = qd.field(gs.qd_float, shape=shape)
        self.wall_u = qd.field(gs.qd_float, shape=solver.vel_u.shape)
        self.wall_v = qd.field(gs.qd_float, shape=solver.vel_v.shape)
        self.wall_w = qd.field(gs.qd_float, shape=solver.vel_w.shape)
        kernel_distances(solver, self, is_fixed=True)
        kernel_distances(solver, self, is_fixed=False)
        kernel_faces(solver, self)

    def set_pose(self, pos, quat, colliders_idx):
        """Set next-step target poses for selected kinematic boundaries."""
        indices = np.arange(len(self.geoms)) if colliders_idx is None else np.atleast_1d(tensor_to_array(colliders_idx))
        if not np.issubdtype(indices.dtype, np.integer) or (indices < 0).any() or (indices >= len(self.geoms)).any():
            gs.raise_exception("FLIP collider indices must be valid integers.")
        if not all(self.is_kinematic[index] for index in indices):
            gs.raise_exception("Pose targets require is_kinematic=True on the selected FLIP boundaries.")
        positions = np.broadcast_to(tensor_to_array(pos), (len(indices), 3))
        quaternions = np.broadcast_to(tensor_to_array(quat), (len(indices), 4))
        norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
        if not np.isfinite(positions).all() or not np.isfinite(quaternions).all() or (norms <= gs.EPS).any():
            gs.raise_exception("FLIP boundary poses must be finite and quaternions must be nonzero.")
        self.target_pos[indices] = positions
        self.target_quat[indices] = quaternions / norms
        self.has_motion = True

    def begin_step(self, dt):
        """Compute constant boundary velocities from the committed and target poses."""
        self.elapsed = 0.0
        self.velocity[:] = (self.target_pos - self.start_pos) / dt
        rotations = geom.transform_quat_by_quat(geom.inv_quat(self.start_quat), self.target_quat)
        self.angular_velocity[:] = geom.quat_to_rotvec(rotations) / dt
        self.vel.from_numpy(self.velocity[:, None])
        self.ang.from_numpy(self.angular_velocity[:, None])

    def advance(self, solver, dt):
        """Advance geometry and rebuild the moving solid grid before pressure projection."""
        self.elapsed += dt
        if self.has_motion:
            fraction = min(self.elapsed / solver.dt, 1.0)
            self.pos.from_numpy((self.start_pos + fraction * (self.target_pos - self.start_pos))[:, None])
            fractions = np.full(len(self.geoms), fraction, dtype=gs.np_float)
            self.quat.from_numpy(geom.slerp(self.start_quat, self.target_quat, fractions)[:, None])

        kernel_distances(solver, self, is_fixed=False)
        kernel_faces(solver, self)

    def finish_step(self):
        """Commit target poses and consume their prescribed motion."""
        self.start_pos[:] = self.target_pos
        self.start_quat[:] = self.target_quat
        self.has_motion = False


@qd.func
def query(index: qd.template(), pos, boundary: qd.template()):
    _closest, normal, is_inside, distance = query_static_collider(
        collider_idx=index,
        env_idx=0,
        pos=pos,
        colliders_pos=boundary.pos,
        colliders_quat=boundary.quat,
        collider=boundary.geoms[index],
    )
    phi = -distance if is_inside else distance
    velocity = boundary.vel[index, 0] + boundary.ang[index, 0].cross(pos - boundary.pos[index, 0])
    return phi, normal, velocity


@qd.kernel
def kernel_distances(solver: qd.template(), boundary: qd.template(), is_fixed: qd.template()):
    for fixed, current, offset in qd.static(
        ((boundary.fixed_nodes, boundary.nodes, 0.0), (boundary.fixed_cells, boundary.cells, 0.5))
    ):
        for index in qd.grouped(current):
            pos = solver._lower_v + (index + offset) * solver._dx
            phi = gs.qd_float(1e20)
            if qd.static(not is_fixed):
                phi = fixed[index]
            for i in qd.static(range(len(boundary.geoms))):
                if qd.static(boundary.is_kinematic[i] != is_fixed):
                    distance, _normal, _velocity = query(i, pos, boundary)
                    phi = qd.min(phi, distance)
            if qd.static(is_fixed):
                fixed[index] = phi
            else:
                current[index] = phi


@qd.func
def triangle_fraction(a, b, c):
    lo = qd.min(a, qd.min(b, c))
    hi = qd.max(a, qd.max(b, c))
    # Comparisons preserve small distances alongside the far-field sentinel; bounded edge ratios avoid overflow.
    mid = qd.max(qd.min(a, b), qd.min(qd.max(a, b), c))
    fraction = gs.qd_float(0.0)
    if hi <= 0.0:
        fraction = 1.0
    elif mid <= 0.0:
        fraction = 1.0 - (hi / (hi - lo)) * (hi / (hi - mid))
    elif lo < 0.0:
        fraction = (lo / (lo - mid)) * (lo / (lo - hi))
    return fraction


@qd.kernel
def kernel_faces(solver: qd.template(), boundary: qd.template()):
    for axis, face, wall in qd.static(
        ((0, solver.open_u, boundary.wall_u), (1, solver.open_v, boundary.wall_v), (2, solver.open_w, boundary.wall_w))
    ):
        for index in qd.grouped(face):
            weight = gs.qd_float(0.0)
            wall_velocity = gs.qd_float(0.0)
            if 0 < index[axis] < solver._res[axis]:
                offset = qd.Vector([0.5, 0.5, 0.5])
                offset[axis] = 0.0
                pos = solver._lower_v + (index + offset) * solver._dx
                e0 = qd.Vector.unit(3, (axis + 1) % 3, gs.qd_int)
                e1 = qd.Vector.unit(3, (axis + 2) % 3, gs.qd_int)
                a = boundary.nodes[index]
                b = boundary.nodes[index + e0]
                c = boundary.nodes[index + e0 + e1]
                d = boundary.nodes[index + e1]
                center = (a + b + c + d) * 0.25
                weight = 1.0 - 0.25 * (
                    triangle_fraction(center, a, b)
                    + triangle_fraction(center, b, c)
                    + triangle_fraction(center, c, d)
                    + triangle_fraction(center, d, a)
                )
                if boundary.cells[index] <= 0.0 or boundary.cells[index - qd.Vector.unit(3, axis, gs.qd_int)] <= 0.0:
                    weight = 0.0
                nearest = 0.25 * (
                    boundary.fixed_nodes[index]
                    + boundary.fixed_nodes[index + e0]
                    + boundary.fixed_nodes[index + e1]
                    + boundary.fixed_nodes[index + e0 + e1]
                )
                for i in qd.static(range(len(boundary.geoms))):
                    if qd.static(boundary.is_kinematic[i]):
                        phi, _normal, velocity = query(i, pos, boundary)
                        if phi < nearest:
                            nearest = phi
                            wall_velocity = velocity[axis]
                if qd.static(solver._has_cylinder):  # noqa: SIM102
                    if (pos.xy - qd.Vector([solver._center_x, solver._center_y])).norm() >= solver._cylinder_radius:
                        weight = 0.0
                        wall_velocity = 0.0
            face[index] = weight
            wall[index] = wall_velocity


@qd.kernel
def kernel_project_particles(solver: qd.template(), boundary: qd.template()):
    for i, env in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles[i, env].active:
            pos = solver.particles[i, env].pos
            vel = solver.particles[i, env].vel
            for iteration in range(3):
                for index in qd.static(range(len(boundary.geoms))):
                    phi, normal, wall_vel = query(index, pos, boundary)
                    if phi < 0.05 * solver._dx:
                        pos += (0.05 * solver._dx - phi) * normal
                        vel -= qd.min((vel - wall_vel).dot(normal), 0.0) * normal
            solver.particles[i, env].pos = pos
            solver.particles[i, env].vel = vel


@qd.kernel
def kernel_check_finite(solver: qd.template(), errno: qd.template()):
    for i, env in qd.ndrange(solver._n_particles, solver._B):
        for axis in qd.static(range(3)):
            p = solver.particles[i, env].pos[axis]
            v = solver.particles[i, env].vel[axis]
            if qd.math.isnan(p) or qd.math.isinf(p) or qd.math.isnan(v) or qd.math.isinf(v):
                qd.atomic_or(errno[env], int(ErrorCode.INVALID_FLIP_STATE_NAN))
