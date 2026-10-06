import math
import os

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import (
    CubeBoundary,
    CylinderBoundary,
    PlaneBoundary,
    TiltedCylinderBoundary,
    TiltedCylinderShellBoundary,
)
from genesis.engine.entities import PBD2DEntity, PBD3DEntity, PBD3DSolidEntity, PBDFreeParticleEntity, PBDParticleEntity
from genesis.engine.entities.pbd_entity import PBDTetEntity
from genesis.engine.materials.PBD.solid import (
    DEFAULT_ROTATION_THRESHOLD,
    DEFAULT_SEAM_FAILURE_THRESHOLD,
    DEFAULT_SHEAR_THRESHOLD,
)
from genesis.engine.states.solvers import PBDFluidSolverState
from genesis.utils.array_class import ErrorCode, LinksState, V_ANNOTATION
from genesis.utils.geom import SpatialHasher
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .base_solver import Solver, StateChange, mutates


@qd.func
def func_validate_pbd_particle(i_b, pos, vel, errno: qd.Tensor):
    if qd.math.isnan(pos).any() or qd.math.isinf(pos).any() or qd.math.isnan(vel).any() or qd.math.isinf(vel).any():
        qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_STATE)


@qd.func
def func_update_pbd_cluster_roots(solver: V_ANNOTATION):
    # A serial union per environment touches only the cluster/bond graph; particle
    # fitting remains parallel. Rebuild after death/reset, never through host reads.
    if qd.static(solver._n_bonds > 0):
        for i_b in range(solver._B):
            if solver.cluster_roots_dirty[i_b]:
                solver.cohesive.rebuild[i_b] = True
                for i_c in range(solver._n_clusters):
                    solver.cluster_roots[i_c, i_b] = i_c
                for i_l in range(solver._n_bonds):
                    if solver.bonds_state[i_l, i_b].alive:
                        i_p = solver.bonds_info[i_l].i_p
                        j_p = solver.bonds_info[i_l].j_p
                        i_c = solver.particle_cluster_ids[solver.particle_cluster_ptr[i_p].member_start]
                        j_c = solver.particle_cluster_ids[solver.particle_cluster_ptr[j_p].member_start]
                        while solver.cluster_roots[i_c, i_b] != i_c:
                            i_c = solver.cluster_roots[i_c, i_b]
                        while solver.cluster_roots[j_c, i_b] != j_c:
                            j_c = solver.cluster_roots[j_c, i_b]
                        solver.cluster_roots[qd.max(i_c, j_c), i_b] = qd.min(i_c, j_c)
                for i_c in range(solver._n_clusters):
                    root = i_c
                    while solver.cluster_roots[root, i_b] != root:
                        root = solver.cluster_roots[root, i_b]
                    solver.cluster_roots[i_c, i_b] = root
                solver.cluster_roots_dirty[i_b] = False


@qd.func
def func_fit_pbd_clusters(solver: V_ANNOTATION, welded: qd.template(), render_only: qd.template()):
    if qd.static(welded):
        func_update_pbd_cluster_roots(solver)
    for i_c, i_b in qd.ndrange(solver._n_clusters, solver._B):
        if qd.static(not render_only) or solver.is_render_dirty[i_b]:
            solver.clusters_fit[i_c, i_b].weight = 0.0
            solver.clusters_fit[i_c, i_b].p_sum = qd.Vector.zero(qd.f64, 3)
            solver.clusters_fit[i_c, i_b].q_sum = qd.Vector.zero(qd.f64, 3)
            solver.clusters_fit[i_c, i_b].covariance = qd.Matrix.zero(qd.f64, 3, 3)
    for i_m, i_b in qd.ndrange(solver._n_cluster_members, solver._B):
        if qd.static(not render_only) or solver.is_render_dirty[i_b]:
            i_c = solver.cluster_member_clusters[i_m]
            if qd.static(welded and solver._n_bonds > 0):
                i_c = solver.cluster_roots[i_c, i_b]
            i_p = solver.cluster_member_particles[i_m]
            # The same membership weight is used in the fit and distributed correction, so
            # each cluster's mass-weighted translation sums to zero even with uneven overlap.
            weight = qd.f64(solver.particles_info[i_p].mass) / qd.max(solver.particle_cluster_ptr[i_p].member_count, 1)
            origin = solver.clusters_rest[i_c, i_b].cm.cast(qd.f64)
            p = solver.particles[i_p, i_b].pos.cast(qd.f64) - origin
            q = solver.particles_solid_rest[i_p, i_b].cast(qd.f64) - origin
            solver.clusters_fit[i_c, i_b].weight += weight
            solver.clusters_fit[i_c, i_b].p_sum += weight * p
            solver.clusters_fit[i_c, i_b].q_sum += weight * q
            solver.clusters_fit[i_c, i_b].covariance += weight * p.outer_product(q)
    for i_c, i_b in qd.ndrange(solver._n_clusters, solver._B):
        if qd.static(not render_only) or solver.is_render_dirty[i_b]:
            fit = solver.clusters_fit[i_c, i_b]
            if fit.weight > 0.0:
                origin = solver.clusters_rest[i_c, i_b].cm.cast(qd.f64)
                solver.clusters_fit[i_c, i_b].cm = origin + fit.p_sum / fit.weight
                solver.clusters_fit[i_c, i_b].rest_cm = origin + fit.q_sum / fit.weight
                covariance = fit.covariance - fit.p_sum.outer_product(fit.q_sum) / fit.weight
                solver.clusters_state[i_c, i_b].cm = solver.clusters_fit[i_c, i_b].cm.cast(gs.qd_float)
                solver.clusters_fit[i_c, i_b].rot = func_pbd_polar_rotation(covariance)
                solver.clusters_state[i_c, i_b].rot = solver.clusters_fit[i_c, i_b].rot.cast(gs.qd_float)
    if qd.static(welded and solver._n_bonds > 0):
        for i_c, i_b in qd.ndrange(solver._n_clusters, solver._B):
            if qd.static(not render_only) or solver.is_render_dirty[i_b]:
                root = solver.cluster_roots[i_c, i_b]
                R = solver.clusters_fit[root, i_b].rot
                solver.clusters_state[i_c, i_b].rot = R.cast(gs.qd_float)
                solver.clusters_state[i_c, i_b].cm = (
                    solver.clusters_fit[root, i_b].cm + R @ (
                        solver.clusters_rest[i_c, i_b].cm.cast(qd.f64) - solver.clusters_fit[root, i_b].rest_cm
                    )
                ).cast(gs.qd_float)


@qd.func
def func_pbd_polar_rotation(A):
    scale = qd.f64(0.0)
    for i, j in qd.static(qd.ndrange(3, 3)):
        scale = qd.max(scale, qd.abs(A[i, j]))
    R = qd.Matrix.identity(qd.f64, 3)
    if scale > 0.0:
        a = A / scale
        # Davenport matrix for max trace(R.T @ covariance). A maps rest to current.
        K = qd.Matrix.zero(qd.f64, 4, 4)
        K[0, 0] = a.trace()
        K[0, 1] = a[2, 1] - a[1, 2]
        K[0, 2] = a[0, 2] - a[2, 0]
        K[0, 3] = a[1, 0] - a[0, 1]
        for i in qd.static(range(3)):
            K[i + 1, 0] = K[0, i + 1]
            for j in qd.static(range(3)):
                K[i + 1, j + 1] = a[i, j] + a[j, i]
            K[i + 1, i + 1] -= a.trace()
        V = qd.Matrix.identity(qd.f64, 4)
        # Symmetric cyclic Jacobi uses rotations, no inverse or rank regularizer.
        # Eight sweeps diagonalize the normalized 4x4 covariance eigensystem.
        for sweep in range(8):
            for p in qd.static(range(3)):
                for q in qd.static(range(p + 1, 4)):
                    value = K[p, q]
                    if value != 0.0:
                        tau = (K[q, q] - K[p, p]) / (2.0 * value)
                        sign = qd.f64(1.0)
                        if tau < 0.0:
                            sign = -1.0
                        t = sign / (qd.abs(tau) + qd.sqrt(1.0 + tau * tau))
                        c = 1.0 / qd.sqrt(1.0 + t * t)
                        s = t * c
                        K[p, p] -= t * value
                        K[q, q] += t * value
                        K[p, q] = 0.0
                        K[q, p] = 0.0
                        for k in qd.static(range(4)):
                            if qd.static(k != p and k != q):
                                kp, kq = K[k, p], K[k, q]
                                K[k, p] = c * kp - s * kq
                                K[p, k] = K[k, p]
                                K[k, q] = s * kp + c * kq
                                K[q, k] = K[k, q]
                            vp, vq = V[k, p], V[k, q]
                            V[k, p] = c * vp - s * vq
                            V[k, q] = s * vp + c * vq
        largest = 0
        for k in qd.static(range(1, 4)):
            if K[k, k] > K[largest, largest]:
                largest = k
        quat = qd.Vector([V[0, largest], V[1, largest], V[2, largest], V[3, largest]])
        quat /= quat.norm()
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        R = qd.Matrix([
            [1.0 - 2.0 * (y*y + z*z), 2.0*(x*y - w*z), 2.0*(x*z + w*y)],
            [2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)],
            [2.0*(x*z - w*y), 2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y)],
        ])
    return R


@qd.func
def func_pbd_inertia_twist(i_b, inertia, angular, errno: qd.Tensor):
    """Minimum-norm angular velocity of a positive semidefinite inertia tensor."""
    scale = qd.f64(0.0)
    for i, j in qd.static(qd.ndrange(3, 3)):
        scale = qd.max(scale, qd.abs(inertia[i, j]))
    omega = qd.Vector.zero(qd.f64, 3)
    is_finite = not (qd.math.isnan(inertia).any() or qd.math.isinf(inertia).any()
                     or qd.math.isnan(angular).any() or qd.math.isinf(angular).any())
    if not is_finite:
        qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_VELOCITY_PROJECTION)
    if is_finite and scale > 0.0:
        A = inertia / scale
        V = qd.Matrix.identity(qd.f64, 3)
        for sweep in range(8):
            for p in qd.static(range(2)):
                for q in qd.static(range(p + 1, 3)):
                    value = A[p, q]
                    if value != 0.0:
                        tau = (A[q, q] - A[p, p]) / (qd.f64(2.0) * value)
                        sign = qd.f64(1.0)
                        if tau < 0.0:
                            sign = -1.0
                        t = sign / (qd.abs(tau) + qd.sqrt(qd.f64(1.0) + tau * tau))
                        c = qd.f64(1.0) / qd.sqrt(qd.f64(1.0) + t*t)
                        s = t * c
                        A[p, p] -= t * value
                        A[q, q] += t * value
                        A[p, q] = 0.0
                        A[q, p] = 0.0
                        for k in qd.static(range(3)):
                            if qd.static(k != p and k != q):
                                kp, kq = A[k, p], A[k, q]
                                A[k, p] = c*kp - s*kq
                                A[p, k] = A[k, p]
                                A[k, q] = s*kp + c*kq
                                A[q, k] = A[k, q]
                            vp, vq = V[k, p], V[k, q]
                            V[k, p] = c*vp - s*vq
                            V[k, q] = s*vp + c*vq
        # Numerical rank uses the f64 eigensystem's roundoff scale. Null rotations
        # have no point velocity, so their minimum-norm angular component is zero.
        for k in qd.static(range(3)):
            if A[k, k] < -qd.f64(64.0) * qd.f64(2.220446049250313e-16):
                qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_VELOCITY_PROJECTION)
            if A[k, k] > qd.f64(64.0) * qd.f64(2.220446049250313e-16):
                direction = qd.Vector([V[0, k], V[1, k], V[2, k]])
                omega += direction * direction.dot(angular) / (scale * A[k, k])
    return omega


@qd.func
def func_project_pbd_rigid_velocities(solver: V_ANNOTATION):
    """Mass-weighted velocity projection for freely moving unit-stiffness groups."""
    for i_c, i_b in qd.ndrange(solver._n_clusters, solver._B):
        solver.is_velocity_projectable[i_c, i_b] = 1
        solver.clusters_fit[i_c, i_b].weight = 0.0
        solver.clusters_fit[i_c, i_b].p_sum = qd.Vector.zero(qd.f64, 3)
        solver.clusters_fit[i_c, i_b].q_sum = qd.Vector.zero(qd.f64, 3)
        solver.clusters_fit[i_c, i_b].covariance = qd.Matrix.zero(qd.f64, 3, 3)
        solver.clusters_fit[i_c, i_b].rest_cm = qd.Vector.zero(qd.f64, 3)
    for i_m, i_b in qd.ndrange(solver._n_cluster_members, solver._B):
        i_c = solver.cluster_member_clusters[i_m]
        root = i_c
        if qd.static(solver._n_bonds > 0):
            root = solver.cluster_roots[i_c, i_b]
        i_p = solver.cluster_member_particles[i_m]
        if (
            not solver.particles[i_p, i_b].free
            or solver.particles_ng[i_p, i_b].is_native_owned
            or solver.clusters_info[i_c].stiffness != 1.0
            or solver.clusters_info[i_c].plastic_creep > 0.0
        ):
            qd.atomic_min(solver.is_velocity_projectable[root, i_b], 0)
        weight = qd.f64(solver.particles_info[i_p].mass) / solver.particle_cluster_ptr[i_p].member_count
        r = solver.particles[i_p, i_b].pos.cast(qd.f64) - solver.clusters_rest[root, i_b].cm.cast(qd.f64)
        v = solver.particles[i_p, i_b].vel.cast(qd.f64)
        solver.clusters_fit[root, i_b].weight += weight
        solver.clusters_fit[root, i_b].p_sum += weight * r
        solver.clusters_fit[root, i_b].q_sum += weight * v
    for i_c, i_b in qd.ndrange(solver._n_clusters, solver._B):
        weight = solver.clusters_fit[i_c, i_b].weight
        if weight > 0.0:
            solver.clusters_fit[i_c, i_b].p_sum /= weight
            solver.clusters_fit[i_c, i_b].q_sum /= weight
    # Center before outer products so coincident and collinear groups preserve
    # their numerical nullspace even after a large world translation.
    for i_m, i_b in qd.ndrange(solver._n_cluster_members, solver._B):
        i_c = solver.cluster_member_clusters[i_m]
        root = i_c
        if qd.static(solver._n_bonds > 0):
            root = solver.cluster_roots[i_c, i_b]
        if solver.is_velocity_projectable[root, i_b] > 0:
            i_p = solver.cluster_member_particles[i_m]
            weight = qd.f64(solver.particles_info[i_p].mass) / solver.particle_cluster_ptr[i_p].member_count
            r = (solver.particles[i_p, i_b].pos.cast(qd.f64) - solver.clusters_rest[root, i_b].cm.cast(qd.f64)
                 - solver.clusters_fit[root, i_b].p_sum)
            v = solver.particles[i_p, i_b].vel.cast(qd.f64) - solver.clusters_fit[root, i_b].q_sum
            solver.clusters_fit[root, i_b].covariance += weight * r.outer_product(r)
            solver.clusters_fit[root, i_b].rest_cm += weight * r.cross(v)
    for i_c, i_b in qd.ndrange(solver._n_clusters, solver._B):
        if solver.is_velocity_projectable[i_c, i_b] > 0 and solver.clusters_fit[i_c, i_b].weight > 0.0:
            covariance = solver.clusters_fit[i_c, i_b].covariance
            inertia = covariance.trace() * qd.Matrix.identity(qd.f64, 3) - covariance
            solver.clusters_fit[i_c, i_b].rest_cm = func_pbd_inertia_twist(
                i_b, inertia, solver.clusters_fit[i_c, i_b].rest_cm, solver._errno
            )
    for i_m, i_b in qd.ndrange(solver._n_cluster_members, solver._B):
        i_c = solver.cluster_member_clusters[i_m]
        root = i_c
        if qd.static(solver._n_bonds > 0):
            root = solver.cluster_roots[i_c, i_b]
        if solver.is_velocity_projectable[root, i_b] > 0:
            i_p = solver.cluster_member_particles[i_m]
            r = (solver.particles[i_p, i_b].pos.cast(qd.f64) - solver.clusters_rest[root, i_b].cm.cast(qd.f64)
                 - solver.clusters_fit[root, i_b].p_sum)
            goal = solver.clusters_fit[root, i_b].q_sum + solver.clusters_fit[root, i_b].rest_cm.cross(r)
            solver.particles[i_p, i_b].dpos += ((goal - solver.particles[i_p, i_b].vel.cast(qd.f64))
                * qd.f64(solver._substep_dt) / solver.particle_cluster_ptr[i_p].member_count).cast(gs.qd_float)
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles_info[i_p].material_type == solver.MATERIAL.SOLID:
            solver.particles[i_p, i_b].vel += solver.particles[i_p, i_b].dpos / solver._substep_dt
            solver.particles[i_p, i_b].dpos.fill(0.0)


class CohesiveProjection:
    def __init__(self, solver):
        # Build-only bulk reads, exact physical graph sizes; no runtime host traversal.
        i_p = qd_to_numpy(solver.bonds_info.i_p, transpose=True)
        j_p = qd_to_numpy(solver.bonds_info.j_p, transpose=True)
        seams = qd_to_numpy(solver.bonds_info.seam, transpose=True)
        starts = qd_to_numpy(solver.particle_cluster_ptr.member_start, transpose=True)
        owners = qd_to_numpy(solver.particle_cluster_ids, transpose=True)[starts]
        clusters = np.unique(np.concatenate((owners[i_p], owners[j_p])))
        physical_seams, first = np.unique(seams, return_index=True)
        stiffness = qd_to_numpy(solver.clusters_info.stiffness, transpose=True)[clusters]
        self.n_nodes = len(clusters)
        self.n_edges = len(physical_seams)
        self.length = solver._particle_size
        self.relative_tol = 1e-8
        node_map = np.full(solver._n_clusters, -1, dtype=gs.np_int)
        node_map[clusters] = np.arange(self.n_nodes)
        edge_map = np.full(solver._n_seams, -1, dtype=gs.np_int)
        edge_map[physical_seams] = np.arange(self.n_edges)
        assert np.all(stiffness[node_map[owners[i_p[first]]]] == stiffness[node_map[owners[j_p[first]]]]), "Connected fragments require uniform stiffness"
        self.node_map = qd.field(gs.qd_int, shape=(solver._n_clusters,))
        self.node_map.from_numpy(node_map)
        self.edge_map = qd.field(gs.qd_int, shape=(solver._n_seams,))
        self.edge_map.from_numpy(edge_map)
        self.clusters = qd.field(gs.qd_int, shape=(self.n_nodes,))
        self.clusters.from_numpy(clusters)
        rest = qd_to_numpy(solver.particles_solid_rest, transpose=True)[0]
        member_clusters = qd_to_numpy(solver.cluster_member_clusters, transpose=True)
        selected_members = np.flatnonzero(node_map[member_clusters] >= 0)
        member_particles = qd_to_numpy(solver.cluster_member_particles, transpose=True)[selected_members]
        self.n_members = len(selected_members)
        self.origin = qd.Vector(rest[np.unique(member_particles)].mean(0), dt=qd.f64)
        member_type = qd.types.struct(particle=gs.qd_int, cluster=gs.qd_int, node=gs.qd_int)
        self.members = member_type.field(shape=(self.n_members,))
        self.members.particle.from_numpy(member_particles)
        self.members.cluster.from_numpy(member_clusters[selected_members])
        self.members.node.from_numpy(node_map[member_clusters[selected_members]])
        node_type = qd.types.struct(
            rhs=qd.types.vector(6, qd.f64), x=qd.types.vector(6, qd.f64),
            r=qd.types.vector(6, qd.f64), roundoff=qd.f64,
        )
        self.nodes = node_type.field(shape=(self.n_nodes, solver._B))
        edge_info = qd.types.struct(i=gs.qd_int, j=gs.qd_int, seam=gs.qd_int)
        self.edges_info = edge_info.field(shape=(self.n_edges,))
        self.edges_info.i.from_numpy(node_map[owners[i_p[first]]])
        self.edges_info.j.from_numpy(node_map[owners[j_p[first]]])
        self.edges_info.seam.from_numpy(physical_seams)
        self.K = qd.Matrix.field(6, 6, qd.f64, shape=(self.n_edges, solver._B))
        self.roots = qd.field(gs.qd_int, shape=(self.n_nodes, solver._B))
        self.rebuild = qd.field(gs.qd_bool, shape=(solver._B,))
        self.rebuild.fill(True)
        # Each entity owns one exact-size block. Unbreakable/plastic clusters
        # have no incident bonds and allocate no factor storage.
        edge_i = node_map[owners[i_p[first]]]
        edge_j = node_map[owners[j_p[first]]]
        graphs = []
        matrix_size = 0
        for entity in solver.entities:
            if isinstance(entity, PBD3DSolidEntity):
                nodes = np.flatnonzero((clusters >= entity.cluster_start) & (clusters < entity.cluster_end))
                if len(nodes):
                    edges = np.flatnonzero(np.isin(edge_i, nodes))
                    assert np.array_equal(nodes, np.arange(nodes[0], nodes[-1] + 1))
                    assert np.array_equal(edges, np.arange(edges[0], edges[-1] + 1))
                    assert np.isin(edge_j[edges], nodes).all()
                    graphs.append((nodes[0], len(nodes), edges[0], len(edges), matrix_size))
                    matrix_size += (6 * len(nodes)) ** 2
        self.n_graphs = len(graphs)
        self.matrix_size = matrix_size
        graph_type = qd.types.struct(node_start=gs.qd_int, node_count=gs.qd_int,
                                     edge_start=gs.qd_int, edge_count=gs.qd_int, matrix_start=gs.qd_int)
        self.graphs_info = graph_type.field(shape=(self.n_graphs,))
        graph_data = np.array(graphs, dtype=gs.np_int)
        self.graphs_info.node_start.from_numpy(graph_data[:, 0])
        self.graphs_info.node_count.from_numpy(graph_data[:, 1])
        self.graphs_info.edge_start.from_numpy(graph_data[:, 2])
        self.graphs_info.edge_count.from_numpy(graph_data[:, 3])
        self.graphs_info.matrix_start.from_numpy(graph_data[:, 4])
        self.factor = qd.field(qd.f64, shape=(self.matrix_size, solver._B))
        self.is_factor_valid = qd.field(gs.qd_bool, shape=(self.n_graphs, solver._B))
        self.overload = qd.field(qd.f64, shape=(solver._n_bonds, solver._B))
        self.stats = qd.Vector.field(4, qd.f64, shape=(solver._B,))


@qd.func
def func_cohesive_patch(r, reduced_mass, length):
    length = qd.f64(length)
    S = qd.Matrix([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
    inertia = qd.f64(2.0) / 5.0 * reduced_mass * (length / 2.0) ** 2
    moment = -reduced_mass * S @ S + inertia * qd.Matrix.identity(qd.f64, 3)
    K = qd.Matrix.zero(qd.f64, 6, 6)
    for i, j in qd.static(qd.ndrange(3, 3)):
        if qd.static(i == j):
            K[i, j] = reduced_mass * length * length
        K[i, j + 3] = -reduced_mass * S[i, j] * length
        K[i + 3, j] = reduced_mass * S[i, j] * length
        K[i + 3, j + 3] = moment[i, j]
    return K


@qd.func
def func_factor_cohesive_graphs(solver: V_ANNOTATION):
    # Live components each have one six-DOF gauge. Gauge identity rows carry
    # no physical stiffness; their conjugate RHS is zero after closure validation.
    for i_e, i_b in qd.ndrange(solver.cohesive.n_edges, solver._B):
        if qd.math.isnan(solver.cohesive.K[i_e, i_b]).any() or qd.math.isinf(solver.cohesive.K[i_e, i_b]).any():
            qd.atomic_or(solver._errno[i_b], ErrorCode.INVALID_PBD_COHESIVE_REACTION)
    for i_g, i_b in qd.ndrange(solver.cohesive.n_graphs, solver._B):
        if solver.cohesive.rebuild[i_b]:
            graph = solver.cohesive.graphs_info[i_g]
            dofs = 6 * graph.node_count
            start = graph.matrix_start
            solver.cohesive.is_factor_valid[i_g, i_b] = True
            for row in range(dofs):
                i_n = graph.node_start + row // 6
                for col in range(row + 1):
                    j_n = graph.node_start + col // 6
                    value = qd.f64(0.0)
                    if i_n == solver.cohesive.roots[i_n, i_b] or j_n == solver.cohesive.roots[j_n, i_b]:
                        if row == col:
                            value = 1.0
                    else:
                        for i_e in range(graph.edge_start, graph.edge_start + graph.edge_count):
                            edge = solver.cohesive.edges_info[i_e]
                            if i_n == j_n and (edge.i == i_n or edge.j == i_n):
                                value += solver.cohesive.K[i_e, i_b][row % 6, col % 6]
                            elif (edge.i == i_n and edge.j == j_n) or (edge.j == i_n and edge.i == j_n):
                                value -= solver.cohesive.K[i_e, i_b][row % 6, col % 6]
                    for k in range(col):
                        value -= solver.cohesive.factor[start + row * dofs + k, i_b] * solver.cohesive.factor[start + col * dofs + k, i_b]
                    entry = qd.f64(0.0)
                    if qd.math.isnan(value) or qd.math.isinf(value):
                        solver.cohesive.is_factor_valid[i_g, i_b] = False
                    elif row == col:
                        if value > 0.0:
                            entry = qd.sqrt(value)
                        else:
                            solver.cohesive.is_factor_valid[i_g, i_b] = False
                    else:
                        pivot = solver.cohesive.factor[start + col * dofs + col, i_b]
                        if pivot > 0.0:
                            entry = value / pivot
                        else:
                            solver.cohesive.is_factor_valid[i_g, i_b] = False
                    solver.cohesive.factor[start + row * dofs + col, i_b] = entry
            if not solver.cohesive.is_factor_valid[i_g, i_b]:
                qd.atomic_or(solver._errno[i_b], ErrorCode.INVALID_PBD_COHESIVE_REACTION)


@qd.func
def func_solve_cohesive_graphs(solver: V_ANNOTATION):
    for i_g, i_b in qd.ndrange(solver.cohesive.n_graphs, solver._B):
        graph = solver.cohesive.graphs_info[i_g]
        dofs = 6 * graph.node_count
        start = graph.matrix_start
        valid = solver.cohesive.is_factor_valid[i_g, i_b]
        # Forward/back substitutions are performed on the cached Cholesky block.
        for row in range(dofs):
            i_n = graph.node_start + row // 6
            value = qd.f64(0.0)
            if i_n != solver.cohesive.roots[i_n, i_b]:
                value = solver.cohesive.nodes[i_n, i_b].rhs[row % 6]
            for col in range(row):
                j_n = graph.node_start + col // 6
                value -= solver.cohesive.factor[start + row * dofs + col, i_b] * solver.cohesive.nodes[j_n, i_b].x[col % 6]
            pivot = solver.cohesive.factor[start + row * dofs + row, i_b]
            if pivot <= 0.0 or qd.math.isnan(pivot) or qd.math.isinf(pivot) or qd.math.isnan(value) or qd.math.isinf(value):
                valid = False
            if valid:
                solver.cohesive.nodes[i_n, i_b].x[row % 6] = value / pivot
            else:
                solver.cohesive.nodes[i_n, i_b].x[row % 6] = 0.0
        for reverse in range(dofs):
            row = dofs - reverse - 1
            i_n = graph.node_start + row // 6
            value = solver.cohesive.nodes[i_n, i_b].x[row % 6]
            for col in range(row + 1, dofs):
                j_n = graph.node_start + col // 6
                value -= solver.cohesive.factor[start + col * dofs + row, i_b] * solver.cohesive.nodes[j_n, i_b].x[col % 6]
            if valid:
                value /= solver.cohesive.factor[start + row * dofs + row, i_b]
            else:
                value = 0.0
            solver.cohesive.nodes[i_n, i_b].x[row % 6] = value
            if qd.math.isnan(value) or qd.math.isinf(value):
                valid = False
        if not valid:
            qd.atomic_or(solver._errno[i_b], ErrorCode.INVALID_PBD_COHESIVE_REACTION)


@qd.func
def func_validate_cohesive_closure(solver: V_ANNOTATION, i_b):
    # Equilibrium includes each component's gauge node, even with zero free DOFs.
    closure = qd.f64(0.0)
    for root in range(solver.cohesive.n_nodes):
        if solver.cohesive.roots[root, i_b] == root:
            total = qd.Vector.zero(qd.f64, 6)
            scale, roundoff = qd.f64(0.0), qd.f64(0.0)
            for i_n in range(solver.cohesive.n_nodes):
                if solver.cohesive.roots[i_n, i_b] == root:
                    if qd.math.isnan(solver.cohesive.nodes[i_n, i_b].rhs).any() or qd.math.isinf(solver.cohesive.nodes[i_n, i_b].rhs).any() or qd.math.isnan(solver.cohesive.nodes[i_n, i_b].roundoff) or qd.math.isinf(solver.cohesive.nodes[i_n, i_b].roundoff):
                        qd.atomic_or(solver._errno[i_b], ErrorCode.INVALID_PBD_COHESIVE_REACTION)
                    total += solver.cohesive.nodes[i_n, i_b].rhs
                    scale += solver.cohesive.nodes[i_n, i_b].rhs.norm_sqr()
                    roundoff += solver.cohesive.nodes[i_n, i_b].roundoff
            bound = 2.220446049250313e-16 * 64.0 * solver.cohesive.n_members * roundoff
            if qd.math.isnan(total).any() or qd.math.isinf(total).any() or qd.math.isnan(scale) or qd.math.isinf(scale) or qd.math.isnan(bound) or qd.math.isinf(bound):
                qd.atomic_or(solver._errno[i_b], ErrorCode.INVALID_PBD_COHESIVE_REACTION)
            if total.norm_sqr() > solver.cohesive.relative_tol**2 * scale + bound**2:
                qd.atomic_or(solver._errno[i_b], ErrorCode.INVALID_PBD_COHESIVE_REACTION)
            closure = qd.max(closure, total.norm())
    return closure


@qd.func
def func_compute_cohesive_reaction(solver: V_ANNOTATION):
    for i_b in range(solver._B):
        solver.cohesive.rebuild[i_b] = solver.cohesive.rebuild[i_b] or solver.cluster_roots_dirty[i_b]
    # This only measures the predicted configuration; no position/velocity changes.
    func_fit_pbd_clusters(solver, welded=True, render_only=False)
    for i_n, i_b in qd.ndrange(solver.cohesive.n_nodes, solver._B):
        i_c = solver.cohesive.clusters[i_n]
        solver.cohesive.roots[i_n, i_b] = solver.cohesive.node_map[solver.cluster_roots[i_c, i_b]]
        solver.cohesive.nodes[i_n, i_b].rhs.fill(0.0)
        solver.cohesive.nodes[i_n, i_b].roundoff = 0.0
    for i_m, i_b in qd.ndrange(solver.cohesive.n_members, solver._B):
        i_c = solver.cohesive.members[i_m].cluster
        i_n = solver.cohesive.members[i_m].node
        if i_n >= 0:
            i_p = solver.cohesive.members[i_m].particle
            root = solver.cluster_roots[i_c, i_b]
            fit = solver.clusters_fit[root, i_b]
            R = fit.rot
            pos = solver.particles[i_p, i_b].pos.cast(qd.f64)
            rest = solver.particles_solid_rest[i_p, i_b].cast(qd.f64)
            goal = fit.cm + R @ (rest - fit.rest_cm)
            delta = R.transpose() @ (goal - pos)
            r = R.transpose() @ (pos - fit.cm) + fit.rest_cm - solver.cohesive.origin
            weight = qd.f64(solver.particles_info[i_p].mass) * solver.clusters_info[i_c].stiffness / solver.particle_cluster_ptr[i_p].member_count
            force = weight * delta
            torque = r.cross(force)
            for k in qd.static(range(3)):
                solver.cohesive.nodes[i_n, i_b].rhs[k] += force[k] * solver.cohesive.length
                solver.cohesive.nodes[i_n, i_b].rhs[k + 3] += torque[k]
            # Bound only evaluation/accumulation roundoff, never material strength.
            solver.cohesive.nodes[i_n, i_b].roundoff += weight * (r.norm() + solver.cohesive.length) ** 2
    for i_e, i_b in qd.ndrange(solver.cohesive.n_edges, solver._B):
        if solver.cohesive.rebuild[i_b]:
            solver.cohesive.K[i_e, i_b].fill(0.0)
    for i_l, i_b in qd.ndrange(solver._n_bonds, solver._B):
        if solver.cohesive.rebuild[i_b] and solver.bonds_state[i_l, i_b].alive:
            i_p, j_p = solver.bonds_info[i_l].i_p, solver.bonds_info[i_l].j_p
            reduced_mass = 1.0 / (1.0 / qd.f64(solver.particles_info[i_p].mass) + 1.0 / qd.f64(solver.particles_info[j_p].mass))
            r = 0.5 * (solver.particles_solid_rest[i_p, i_b].cast(qd.f64) + solver.particles_solid_rest[j_p, i_b].cast(qd.f64)) - solver.cohesive.origin
            i_e = solver.cohesive.edge_map[solver.bonds_info[i_l].seam]
            solver.cohesive.K[i_e, i_b] += func_cohesive_patch(r, reduced_mass, qd.f64(solver.cohesive.length))
    func_factor_cohesive_graphs(solver)
    func_solve_cohesive_graphs(solver)
    for i_b in range(solver._B):
        solver.cohesive.rebuild[i_b] = False
        closure = func_validate_cohesive_closure(solver, i_b)
        norm_rhs = qd.f64(0.0)
        for i_n in range(solver.cohesive.n_nodes):
            solver.cohesive.nodes[i_n, i_b].r.fill(0.0)
            if i_n != solver.cohesive.roots[i_n, i_b]:
                solver.cohesive.nodes[i_n, i_b].r = -solver.cohesive.nodes[i_n, i_b].rhs
                norm_rhs += solver.cohesive.nodes[i_n, i_b].rhs.norm_sqr()
        for i_e in range(solver.cohesive.n_edges):
            edge = solver.cohesive.edges_info[i_e]
            value = solver.cohesive.K[i_e, i_b] @ (solver.cohesive.nodes[edge.i, i_b].x - solver.cohesive.nodes[edge.j, i_b].x)
            solver.cohesive.nodes[edge.i, i_b].r += value
            solver.cohesive.nodes[edge.j, i_b].r -= value
        residual = qd.f64(0.0)
        for i_n in range(solver.cohesive.n_nodes):
            if i_n != solver.cohesive.roots[i_n, i_b]:
                residual += solver.cohesive.nodes[i_n, i_b].r.norm_sqr()
        if qd.math.isnan(residual) or qd.math.isinf(residual) or qd.math.isnan(norm_rhs) or qd.math.isinf(norm_rhs) or residual > solver.cohesive.relative_tol**2 * norm_rhs:
            qd.atomic_or(solver._errno[i_b], ErrorCode.INVALID_PBD_COHESIVE_REACTION)
        solver.cohesive.stats[i_b][0] = 1.0
        solver.cohesive.stats[i_b][1] = 0.0
        if norm_rhs > 0.0:
            solver.cohesive.stats[i_b][1] = qd.sqrt(residual / norm_rhs)
        solver.cohesive.stats[i_b][2] = closure
        solver.cohesive.stats[i_b][3] = qd.sqrt(norm_rhs)
    for i_l, i_b in qd.ndrange(solver._n_bonds, solver._B):
        solver.cohesive.overload[i_l, i_b] = 0.0
        if solver.bonds_state[i_l, i_b].alive:
            bond = solver.bonds_info[i_l]
            edge = solver.cohesive.edges_info[solver.cohesive.edge_map[bond.seam]]
            potential = solver.cohesive.nodes[edge.i, i_b].x - solver.cohesive.nodes[edge.j, i_b].x
            translation = qd.Vector([potential[0], potential[1], potential[2]]) * solver.cohesive.length
            angle = qd.Vector([potential[3], potential[4], potential[5]])
            rest_i = solver.particles_solid_rest[bond.i_p, i_b].cast(qd.f64)
            rest_j = solver.particles_solid_rest[bond.j_p, i_b].cast(qd.f64)
            r = (rest_i + rest_j) * 0.5 - solver.cohesive.origin
            shift = translation - r.cross(angle)
            axis = (rest_i - rest_j) / bond.rest
            normal = qd.abs(axis.dot(shift)) / bond.rest / bond.threshold
            overload = normal
            seam = solver.seams_info[bond.seam]
            if seam.shear_threshold > 0.0:
                overload = qd.max(overload, (shift - axis * axis.dot(shift)).norm() / bond.rest / seam.shear_threshold)
            if seam.rotation_threshold > 0.0:
                overload = qd.max(overload, angle.norm() / seam.rotation_threshold)
            solver.cohesive.overload[i_l, i_b] = overload



@qd.data_oriented
class PBDSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    class MATERIAL(gs.IntEnum):
        CLOTH = 0
        ELASTIC = 1
        LIQUID = 2
        PARTICLE = 3  # non-physics particles
        SOLID = 4  # shape-matching cluster rigid / fracturable solid

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self.cohesive = None
        self._n_entity_particles = None
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)
        self._particle_size = options.particle_size
        self._max_stretch_solver_iterations = options.max_stretch_solver_iterations
        self._max_bending_solver_iterations = options.max_bending_solver_iterations
        self._max_volume_solver_iterations = options.max_volume_solver_iterations
        self._max_density_solver_iterations = options.max_density_solver_iterations
        self._max_viscosity_solver_iterations = options.max_viscosity_solver_iterations
        self._max_shape_matching_solver_iterations = options.max_shape_matching_solver_iterations
        self._diffusion_coeff = options.diffusion_coeff

        self._is_st_enabled = options.is_surface_tension_enabled
        self._st_compliance = options.st_compliance
        self._is_st_distance_enabled = options.is_st_distance_enabled
        self._st_distance_compliance = options.st_distance_compliance
        self._st_ring_radius = options.st_ring_radius_factor * options.particle_size
        self._st_topo_interval = options.st_topo_interval
        self._st_lit_threshold = options.st_lit_threshold
        self._st_normal_compat = options.st_normal_compat
        self._st_surface_density_factor = options.st_surface_density_factor
        self._st_max_surface_neighbors = options.st_max_surface_neighbors
        self._st_max_localmesh_neighbors = options.st_max_localmesh_neighbors
        self._st_topo_counter = 0
        self._st_default_mass = 1000.0  # resolved from the first entity's material in build()
        # spherical-illumination screen resolution (Shibata 2015; PBSTF reference values)
        self._N_THETA = 18
        self._N_PHI = 36
        # topology overflow codes (raised like the PBSTF reference)
        self._ST_SURFACE_NEIGHBOR_OVERFLOW = 1
        self._ST_LOCAL_MESH_QUEUE_OVERFLOW = 2
        self._ST_LOCAL_MESH_NEIGHBOR_OVERFLOW = 3

        self._velocity_damping = options.velocity_damping

        self._is_wall_adhesion_enabled = options.is_wall_adhesion_enabled
        self._wall_adhesion_compliance = options.wall_adhesion_compliance
        self._wall_friction = options.wall_friction
        if (
            self._is_wall_adhesion_enabled
            and options.boundary_cylinder is None
            and options.boundary_plane is None
            and options.boundary_pitcher is None
            and options.boundary_pitcher_shell is None
            and options.boundary_cup_shell is None
            and not options.boundary_ball_sets
        ):
            gs.raise_exception(
                "is_wall_adhesion_enabled requires a boundary to adhere to "
                "(boundary_cylinder, boundary_plane, boundary_pitcher, boundary_pitcher_shell, "
                "boundary_cup_shell or boundary_ball_sets)."
            )

        self._boundary_cylinder = options.boundary_cylinder
        self._has_boundary_cylinder = options.boundary_cylinder is not None
        self._boundary_pitcher = options.boundary_pitcher
        self._boundary_pitcher_shell = options.boundary_pitcher_shell
        self._is_pitcher_shell = options.boundary_pitcher_shell is not None
        self._has_boundary_pitcher = options.boundary_pitcher is not None or self._is_pitcher_shell
        self._boundary_cup_shell = options.boundary_cup_shell
        self._has_boundary_cup_shell = options.boundary_cup_shell is not None
        # any shell in the scene selects the shell adhesion semantics (global nearest-surface
        # arbitration in _adh_query_nearest)
        self._has_boundary_shell = self._is_pitcher_shell or self._has_boundary_cup_shell
        self._boundary_plane = options.boundary_plane
        self._has_boundary_plane = options.boundary_plane is not None
        self._plane_friction = options.plane_friction
        self._boundary_ball_sets = options.boundary_ball_sets
        self._boundary_ball_initial_poses = options.boundary_ball_initial_poses
        self._has_boundary_balls = len(self._boundary_ball_sets) > 0
        self._n_ball_sets = len(self._boundary_ball_sets)
        self._boundary_ball_radius = (
            0.5 * options.particle_size if options.boundary_ball_radius is None else options.boundary_ball_radius
        )
        self._is_boundary_ball_mass_weight_enabled = options.is_boundary_ball_mass_weight_enabled
        self._boundary_ball_smooth_barrel = options.boundary_ball_smooth_barrel
        self._has_boundary_ball_smooth_barrel = self._boundary_ball_smooth_barrel is not None
        if self._has_boundary_ball_smooth_barrel:
            (
                self._smooth_barrel_set,
                self._smooth_barrel_cx,
                self._smooth_barrel_cy,
                self._smooth_barrel_r_inner_visible,
                self._smooth_barrel_r_outer_visible,
                self._smooth_barrel_z_min,
                self._smooth_barrel_z_max,
                self._smooth_barrel_excluded_azimuth,
                self._smooth_barrel_excluded_half_angle,
                self._smooth_barrel_excluded_z_min,
                self._smooth_barrel_excluded_z_max,
                self._smooth_barrel_witness_margin,
            ) = self._boundary_ball_smooth_barrel
        self.n_boundary_balls = 0
        self.boundary_ball_ranges = []
        # A pose upload happens outside the substep loop.  Keep the previous ball positions alive
        # and interpolate to the uploaded target over the following frame's substeps.  Updating
        # the full pose in one jump made a rotating floor teleport several millimetres through
        # fluid before collision was evaluated.
        self._is_boundary_pose_pending = False
        self._boundary_pose_substep = 0

        self._n_vvert_supports = self.scene.vis_options.n_support_neighbors

        # -Neighbours_Setting-
        self.dist_scale = self.particle_radius / 0.4  # @Zhenjia: double check this
        self.h = 1.0
        self.h_2 = self.h**2
        self.h_6 = self.h**6
        self.h_9 = self.h**9

        # representative particle volume (ST color-field normals; same empirical 0.8 factor as PBSTF/IPBF)
        self._particle_volume = 0.8 * self._particle_size**3

        # -POLY6_KERNEL-
        self.poly6_Coe = 315.0 / (64 * math.pi)

        # -SPIKY_KERNEL-
        self.spiky_Coe = -45.0 / math.pi

        self.lambda_epsilon = options.density_lambda_epsilon
        self._is_density_clamped_negative = options.is_density_clamped_negative

        # -S_CORR-
        self.S_Corr_delta_q = 0.3
        self.S_Corr_k = 0.0001

        # -Gradient Approx. delta difference-
        self.g_del = 0.01

        self.vorqd_epsilon = 0.01

        # spatial hasher
        self.sh = SpatialHasher(
            cell_size=options.hash_grid_cell_size,
            grid_res=options._hash_grid_res,
        )

        # boundary
        self.setup_boundary()

    def setup_boundary(self):
        if self._boundary_cylinder is None:
            self.boundary = CubeBoundary(
                lower=self._lower_bound,
                upper=self._upper_bound,
            )
        else:
            cx, cy, radius, z_bottom = self._boundary_cylinder[:4]
            z_top = self._boundary_cylinder[4] if len(self._boundary_cylinder) > 4 else None
            escape_band = self._boundary_cylinder[5] if len(self._boundary_cylinder) > 5 else None
            self.boundary = CylinderBoundary(
                center_xy=(cx, cy),
                radius=radius,
                z_bottom=z_bottom,
                z_top=z_top,
                band=1.5 * self._particle_size,
                escape_band=escape_band,
            )
        self.boundary2 = None
        if self._is_pitcher_shell:
            # P2 shell model: finite-thickness wall with an outer surface (exact SDF)
            ox, oy, oz, ax, ay, az, p_radius, p_length, p_twall, p_tbottom = self._boundary_pitcher_shell[:10]
            p_lip = self._boundary_pitcher_shell[10] if len(self._boundary_pitcher_shell) > 10 else 0.0
            self.boundary2 = TiltedCylinderShellBoundary(
                origin=(ox, oy, oz),
                axis=(ax, ay, az),
                radius=p_radius,
                length=p_length,
                t_wall=p_twall,
                t_bottom=p_tbottom,
                lip_round=p_lip,
                band=1.5 * self._particle_size,
            )
        elif self._has_boundary_pitcher:
            ox, oy, oz, ax, ay, az, p_radius, p_length = self._boundary_pitcher
            self.boundary2 = TiltedCylinderBoundary(
                origin=(ox, oy, oz),
                axis=(ax, ay, az),
                radius=p_radius,
                length=p_length,
                band=1.5 * self._particle_size,
            )
        self.boundary_cup = None
        if self._has_boundary_cup_shell:
            c_cx, c_cy, c_zbot, c_rin, c_length, c_twall, c_tbottom = self._boundary_cup_shell[:7]
            c_lip = self._boundary_cup_shell[7] if len(self._boundary_cup_shell) > 7 else 0.0
            self.boundary_cup = TiltedCylinderShellBoundary(
                origin=(c_cx, c_cy, c_zbot),
                axis=(0.0, 0.0, 1.0),
                radius=c_rin,
                length=c_length,
                t_wall=c_twall,
                t_bottom=c_tbottom,
                lip_round=c_lip,
                band=1.5 * self._particle_size,
            )
        self.boundary3 = None
        if self._has_boundary_plane:
            if len(self._boundary_plane) == 1:
                self.boundary3 = PlaneBoundary(z0=self._boundary_plane[0])
            else:
                pz0, pcx, pcy, p_radius = self._boundary_plane
                self.boundary3 = PlaneBoundary(z0=pz0, center_xy=(pcx, pcy), radius=p_radius)

    def set_pitcher_pose(self, origin, axis):
        """Update the origin and axis of the configured movable container."""
        if self._has_boundary_pitcher:
            self.boundary2.set_pose(origin, axis)

    def init_vvert_fields(self):
        struct_vvert_info = qd.types.struct(
            support_idxs=qd.types.vector(self._n_vvert_supports, gs.qd_int),
            support_weights=qd.types.vector(self._n_vvert_supports, gs.qd_float),
            # solid vverts follow one cluster's rigid transform instead of the support weights
            is_solid=gs.qd_bool,
        )
        self.vverts_info = struct_vvert_info.field(shape=(max(self._n_vverts, 1),), layout=qd.Layout.SOA)
        self.vverts_info.is_solid.fill(False)

        # per-vvert solid binding: the owning cluster and the rest offset from its rest centroid; faces on a
        # crack boundary bind to different fragments and split apart without stretching
        self.vverts_solid_cluster = qd.field(dtype=gs.qd_int, shape=(max(self._n_vverts, 1),))
        self.vverts_solid_offset = qd.field(dtype=gs.qd_vec3, shape=(max(self._n_vverts, 1),))
        # Interior crack walls carry their pair seam id; connected endpoint clusters keep them hidden.
        self.vverts_solid_seam = qd.field(dtype=gs.qd_int, shape=(max(self._n_vverts, 1),))
        self.vverts_solid_seam.fill(-1)

        struct_vvert_state_render = qd.types.struct(
            pos=gs.qd_vec3,
            active=gs.qd_bool,
        )
        self.vverts_render = struct_vvert_state_render.field(
            shape=(max(self._n_vverts, 1), self._B), layout=qd.Layout.SOA
        )

        # UV coordinates for visual vertices (static, same across all batch envs)
        self.vverts_uvs = qd.field(dtype=gs.qd_vec2, shape=(max(self._n_vverts, 1),))

        # Triangle face indices for visual mesh (static)
        self.vfaces_indices = qd.field(dtype=gs.qd_ivec3, shape=(max(self._n_vfaces, 1),))

    def init_particle_fields(self):
        # particles information (static)
        struct_particle_info = qd.types.struct(
            mass=gs.qd_float,
            pos_rest=gs.qd_vec3,
            rho_rest=gs.qd_float,
            material_type=gs.qd_int,
            mu_s=gs.qd_float,
            mu_k=gs.qd_float,
            air_resistance=gs.qd_float,
            density_relaxation=gs.qd_float,
            viscosity_relaxation=gs.qd_float,
            # rigid shard owning a SOLID particle; globally unique across entities, fixed at build time
            fragment=gs.qd_int,
        )
        # particles state (dynamic)
        struct_particle_state = qd.types.struct(
            free=gs.qd_bool,  # if not free, the particle is not affected by internal forces and solely controlled by external user until released
            pos=gs.qd_vec3,  # position
            ipos=gs.qd_vec3,  # initial position
            dpos=gs.qd_vec3,  # delta position
            vel=gs.qd_vec3,  # velocity
            lam=gs.qd_float,
            rho=gs.qd_float,
            c=gs.qd_float,
            dc=gs.qd_float,  # Jacobi buffer for the concentration diffusion pass
        )

        # dynamic particle state without gradient
        struct_particle_state_ng = qd.types.struct(
            reordered_idx=gs.qd_int,
            original_idx=gs.qd_int,
            active=gs.qd_bool,
            boundary_group=gs.qd_int,
            is_boundary=gs.qd_bool,
            boundary_set=gs.qd_int,  # owning boundary-ball set; meaningful only when is_boundary
            # The PBD-fragment bridge owns this particle's fragment: every prediction, force, fit, contact and
            # coupling path skips it, and the bridge rewrites its pos/vel from the native link at the substep tail.
            is_native_owned=gs.qd_bool,
        )

        # single frame particle state for rendering
        struct_particle_state_render = qd.types.struct(
            pos=gs.qd_vec3,
            vel=gs.qd_vec3,
            active=gs.qd_bool,
            c=gs.qd_float,
        )

        self.particles_info = struct_particle_info.field(shape=(self._n_particles,), layout=qd.Layout.SOA)
        self.particles_info_reordered = struct_particle_info.field(
            shape=(self._n_particles, self._B), layout=qd.Layout.SOA
        )
        self.particles = struct_particle_state.field(shape=(self._n_particles, self._B), layout=qd.Layout.SOA)
        self.particles_reordered = struct_particle_state.field(shape=(self._n_particles, self._B), layout=qd.Layout.SOA)
        self.particles_ng = struct_particle_state_ng.field(shape=(self._n_particles, self._B), layout=qd.Layout.SOA)
        self.particles_ng_reordered = struct_particle_state_ng.field(
            shape=(self._n_particles, self._B), layout=qd.Layout.SOA
        )
        self.particles_ng.is_native_owned.fill(False)
        self.particles_ng_reordered.is_native_owned.fill(False)
        self.particles_render = struct_particle_state_render.field(
            shape=(self._n_particles, self._B), layout=qd.Layout.SOA
        )

        if self._is_st_enabled:
            n, b = self._n_particles, self._B
            self.on_surface = qd.field(gs.qd_bool, shape=(n, b))
            self.topology_valid = qd.field(gs.qd_bool, shape=(n, b))
            self.normal = qd.field(gs.qd_vec3, shape=(n, b))
            self.n_ring = qd.field(gs.qd_int, shape=(n, b))
            # int difference array of the illumination screen (N_THETA+1 x N_PHI+1); the classify
            # pass restores per-cell counts with a 2D prefix sum -- a bool 18x36 screen cannot do this
            self._screen_blocked = qd.field(gs.qd_int, shape=(n, b, self._N_THETA + 1, self._N_PHI + 1))
            cand = (n, b, self._st_max_surface_neighbors)
            self.neighbor_ids = qd.field(gs.qd_int, shape=cand)  # one-ring candidates (mesh build workset)
            self.projected_positions = qd.field(gs.qd_vec2, shape=cand)  # tangent-plane projections
            self._chain_pre = qd.field(gs.qd_int, shape=cand)  # polar insertion-sort linked list
            self._chain_nxt = qd.field(gs.qd_int, shape=cand)
            self._node_queue = qd.field(gs.qd_int, shape=(n, b, 3 * self._st_max_surface_neighbors))
            self.local_mesh_neighbors = qd.field(
                gs.qd_int, shape=(n, b, self._st_max_localmesh_neighbors)
            )  # final one-ring (polar order)
            # area-constraint assembly buffers (PBSTF _kernel_apply_surface_constraints)
            self._surface_gradient = qd.field(gs.qd_vec3, shape=(n, b, self._st_max_localmesh_neighbors))
            self._surface_lambda = qd.field(gs.qd_float, shape=(n, b))
            self._surface_grad_i = qd.field(gs.qd_vec3, shape=(n, b))
            self._overflow = qd.field(gs.qd_int, shape=())

        if self._has_boundary_balls:
            self._ball_pos_local = qd.field(gs.qd_vec3, shape=(self.n_boundary_balls,))
            self._ball_motion_start = qd.field(gs.qd_vec3, shape=(self.n_boundary_balls, self._B))
            self._ball_motion_target = qd.field(gs.qd_vec3, shape=(self.n_boundary_balls, self._B))
            self._ball_set_pos = qd.field(gs.qd_vec3, shape=(self._n_ball_sets,))
            self._ball_set_quat = qd.field(gs.qd_vec4, shape=(self._n_ball_sets,))
            self._ball_set_start = qd.field(gs.qd_int, shape=(self._n_ball_sets,))
            self._ball_set_size = qd.field(gs.qd_int, shape=(self._n_ball_sets,))
            self._ball_set_kinematic = qd.field(gs.qd_bool, shape=(self._n_ball_sets,))
            self.bnd_dist = qd.field(gs.qd_float, shape=(self._n_particles, self._B))
            self.bnd_dir = qd.field(gs.qd_vec3, shape=(self._n_particles, self._B))

    def init_edge_fields(self):
        # edges information for stretch. edge: (v1, v2)
        struct_edge_info = qd.types.struct(
            len_rest=gs.qd_float,
            stretch_compliance=gs.qd_float,
            stretch_relaxation=gs.qd_float,
            v1=gs.qd_int,
            v2=gs.qd_int,
        )
        self.edges_info = struct_edge_info.field(shape=(max(1, self._n_edges),), layout=qd.Layout.SOA)

        # inner edges information for bending. edge: (v1, v2), adjacent faces: (v1, v2, v3) and (v1, v2, v4)
        struct_inner_edge_info = qd.types.struct(
            len_rest=gs.qd_float,
            bending_compliance=gs.qd_float,
            bending_relaxation=gs.qd_float,
            v1=gs.qd_int,
            v2=gs.qd_int,
            v3=gs.qd_int,
            v4=gs.qd_int,
        )
        self.inner_edges_info = struct_inner_edge_info.field(shape=(max(self._n_inner_edges, 1),), layout=qd.Layout.SOA)

    def init_elem_fields(self):
        struct_elem_info = qd.types.struct(
            vol_rest=gs.qd_float,
            volume_compliance=gs.qd_float,
            volume_relaxation=gs.qd_float,
            v1=gs.qd_int,
            v2=gs.qd_int,
            v3=gs.qd_int,
            v4=gs.qd_int,
        )
        self.elems_info = struct_elem_info.field(shape=(max(self._n_elems, 1),), layout=qd.Layout.SOA)

    def init_cluster_fields(self):
        # shape-matching clusters for SOLID entities (static parameters, shared across envs)
        struct_cluster_info = qd.types.struct(
            member_start=gs.qd_int,
            member_count=gs.qd_int,
            stiffness=gs.qd_float,
            yield_threshold=gs.qd_float,
            plastic_creep=gs.qd_float,
        )
        self.clusters_info = struct_cluster_info.field(shape=(max(self._n_clusters, 1),), layout=qd.Layout.SOA)

        # rest centroid per env: plasticity rewrites it at runtime, so it cannot be shared across envs
        struct_cluster_rest = qd.types.struct(
            cm=gs.qd_vec3,
        )
        self.clusters_rest = struct_cluster_rest.field(shape=(max(self._n_clusters, 1), self._B), layout=qd.Layout.SOA)
        self.cluster_member_particles = qd.field(dtype=gs.qd_int, shape=(max(self._n_cluster_members, 1),))
        self.cluster_member_clusters = (
            qd.field(dtype=gs.qd_int, shape=(self._n_cluster_members,)) if self._n_cluster_members else None
        )
        struct_cluster_fit = qd.types.struct(
            weight=qd.f64,
            p_sum=qd.types.vector(3, qd.f64),
            q_sum=qd.types.vector(3, qd.f64),
            covariance=qd.types.matrix(3, 3, qd.f64),
            rot=qd.types.matrix(3, 3, qd.f64),
            cm=qd.types.vector(3, qd.f64),
            rest_cm=qd.types.vector(3, qd.f64),
        )
        self.clusters_fit = (
            struct_cluster_fit.field(shape=(self._n_clusters, self._B), layout=qd.Layout.SOA)
            if self._n_clusters else None
        )
        self.is_velocity_projectable = qd.field(dtype=gs.qd_int, shape=(self._n_clusters, self._B)) if self._n_clusters else None
        self.cluster_roots = (
            qd.field(dtype=gs.qd_int, shape=(self._n_clusters, self._B)) if self._n_bonds else None
        )
        self.is_render_dirty = qd.field(dtype=gs.qd_bool, shape=(self._B,)) if self._n_clusters else None
        if self._n_clusters:
            self.is_render_dirty.fill(True)
        self.cluster_roots_dirty = qd.field(dtype=gs.qd_int, shape=(self._B,)) if self._n_bonds else None
        if self._n_bonds:
            self.cluster_roots_dirty.fill(1)

        # per-particle rest positions for SOLID entities. Plasticity creeps these toward the current shape;
        # one shared rest per particle (instead of per membership) keeps overlapping clusters consistent
        self.particles_solid_rest = qd.field(dtype=gs.qd_vec3, shape=(self._n_particles, self._B))
        # scratch for the plastic flow accumulation pass
        self.plastic_q_sum = qd.field(dtype=gs.qd_vec3, shape=(self._n_particles, self._B))
        self.plastic_w_sum = qd.field(dtype=gs.qd_float, shape=(self._n_particles, self._B))

        # dynamic cluster state
        struct_cluster_state = qd.types.struct(
            cm=gs.qd_vec3,
            rot=gs.qd_mat3,
            is_rest_dirty=gs.qd_bool,  # rest centroid must be rebuilt after plasticity moved a member's rest
        )
        self.clusters_state = struct_cluster_state.field(
            shape=(max(self._n_clusters, 1), self._B), layout=qd.Layout.SOA
        )

        # particle -> clusters dual map
        struct_particle_cluster_ptr = qd.types.struct(
            member_start=gs.qd_int,
            member_count=gs.qd_int,
        )
        self.particle_cluster_ptr = struct_particle_cluster_ptr.field(shape=(self._n_particles,), layout=qd.Layout.SOA)
        self.particle_cluster_ids = qd.field(dtype=gs.qd_int, shape=(max(self._n_cluster_members, 1),))

    def init_bond_fields(self):
        # breakable links joining the fragments of fracturable SOLID entities (static, shared across envs)
        struct_bond_info = qd.types.struct(
            i_p=gs.qd_int,
            j_p=gs.qd_int,
            rest=gs.qd_float,
            stiffness=gs.qd_float,
            # strain |d - d0| / d0 beyond which the bond breaks
            threshold=gs.qd_float,
            seam=gs.qd_int,
        )
        self.bonds_info = struct_bond_info.field(shape=(max(self._n_bonds, 1),), layout=qd.Layout.SOA)
        struct_bond_state = qd.types.struct(
            alive=gs.qd_bool,
        )
        self.bonds_state = struct_bond_state.field(shape=(max(self._n_bonds, 1), self._B), layout=qd.Layout.SOA)


        # One seam per bonded fragment pair or geometric cut interface, including zero-bond interfaces.
        # Death changes cohesion; crack walls become visible only after the endpoint clusters separate.
        struct_seam_info = qd.types.struct(
            i_c=gs.qd_int,
            j_c=gs.qd_int,
            # tangential slip strain (slip / bond rest length) beyond which a bond overloads the seam
            shear_threshold=gs.qd_float,
            # relative fragment rotation (rad) beyond which a bond overloads the seam
            rotation_threshold=gs.qd_float,
            # accumulated overload fraction at which the seam dies; <= 0 restricts death to its last bond
            failure_threshold=gs.qd_float,
        )
        self.seams_info = struct_seam_info.field(shape=(max(self._n_seams, 1),), layout=qd.Layout.SOA)
        struct_seam_state = qd.types.struct(
            alive_count=gs.qd_int,
            # monotonic sum of per-substep overloaded-bond fractions; the scratch overload_count holds
            # the current substep's numerator before the per-seam pass folds it in
            damage=gs.qd_float,
            overload_count=gs.qd_int,
            is_dead=gs.qd_bool,
        )
        self.seams_state = struct_seam_state.field(shape=(max(self._n_seams, 1), self._B), layout=qd.Layout.SOA)

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        pass

    def build(self):
        super().build()

        self._B = self._sim._B
        # Keep the entity-owned prefix separate from the solver-owned boundary balls.  Using
        # `_n_fluid_particles` as the append point would overwrite cloth/elastic/free-particle
        # rows in a mixed PBD scene, because those rows are part of `n_particles` but not of
        # `n_fluid_particles`.
        self._n_entity_particles = self.n_particles
        self._n_particles = self._n_entity_particles
        self._n_fluid_particles = self.n_fluid_particles
        # The ball sets append after every entity's particles. Runtime/render fields cover the
        # full solver range; Scene state snapshots intentionally remain entity-only because the
        # balls are solver-owned geometry reconstructed from local points and set poses.
        ball_pos_local, ball_mass_weight = self._load_boundary_balls()
        self.n_boundary_balls = len(ball_pos_local)
        self._n_particles = self._n_entity_particles + self.n_boundary_balls
        self._n_edges = self.n_edges
        self._n_inner_edges = self.n_inner_edges
        self._n_elems = self.n_elems
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces
        self._n_clusters = self.n_clusters
        self._n_cluster_members = self.n_cluster_members
        self._n_bonds = self.n_bonds
        self._n_seams = self.n_seams
        self._n_fragments = self.n_fragments
        self._has_plasticity = any(
            entity.material.plastic_creep > 0.0 for entity in self._entities if isinstance(entity, PBD3DSolidEntity)
        )

        if self._is_st_enabled and self._entities:
            # PBSTF denominators normalize the compliance by a default mass (the material density
            # in this solver's mass=rho convention); all demo entities share one fluid material
            self._st_default_mass = self._entities[0].material.rho

        if self.is_active:
            self.sh.build(self._B)
            self._errno = qd.field(dtype=gs.qd_int, shape=(self._B,))
            self._errno.fill(0)

            self.init_particle_fields()
            self.init_edge_fields()
            self.init_elem_fields()
            self.init_vvert_fields()
            self.init_cluster_fields()
            self.init_bond_fields()

            self.init_ckpt()

            for entity in self._entities:
                entity._add_to_solver()
                if isinstance(entity, PBD3DSolidEntity) and entity.n_seams > 0:
                    self._kernel_add_seam_fracture_params(
                        entity.seam_start,
                        entity.n_seams,
                        (
                            DEFAULT_SHEAR_THRESHOLD
                            if entity.material.shear_threshold is None
                            else entity.material.shear_threshold
                        ),
                        (
                            DEFAULT_ROTATION_THRESHOLD
                            if entity.material.rotation_threshold is None
                            else entity.material.rotation_threshold
                        ),
                        (
                            DEFAULT_SEAM_FAILURE_THRESHOLD
                            if entity.material.seam_failure_threshold is None
                            else entity.material.seam_failure_threshold
                        ),
                    )

            if self._n_bonds > 0:
                self.cohesive = CohesiveProjection(self)

            if self.n_boundary_balls > 0:
                # the ball mass mirrors the fluid material: with ball layers one spacing below the
                # fluid lattice, a wall-layer particle reproduces the interior density exactly
                liquids = [entity for entity in self._entities if isinstance(entity, PBDParticleEntity)]
                if not liquids:
                    gs.raise_exception("boundary_ball_sets require a PBD.Liquid entity to mirror the mass from.")
                mat_rho = liquids[0].material.rho
                starts = [ball_range[0] for ball_range in self.boundary_ball_ranges]
                sizes = [ball_range[1] - ball_range[0] for ball_range in self.boundary_ball_ranges]
                kinematic = [ball_range[2] for ball_range in self.boundary_ball_ranges]
                self._ball_set_start.from_numpy(np.array(starts, dtype=gs.np_int))
                self._ball_set_size.from_numpy(np.array(sizes, dtype=gs.np_int))
                self._ball_set_kinematic.from_numpy(np.array(kinematic, dtype=bool))
                initial_pos, initial_quat = self._normalized_boundary_ball_initial_poses()
                self._ball_set_pos.from_numpy(initial_pos)
                self._ball_set_quat.from_numpy(initial_quat)
                kernel_add_boundary_balls(
                    self._n_entity_particles, self.n_boundary_balls, mat_rho, ball_pos_local, ball_mass_weight, self
                )
                kernel_initialize_boundary_ball_pos(self)
            gs.logger.info(
                f"PBDSolver: {self._n_entity_particles} entity particles "
                f"({self._n_fluid_particles} fluid) + {self.n_boundary_balls} boundary ball "
                f"particles in {self._n_ball_sets} set(s)."
            )

        # FIXME: _gravity must be a raw qd.field() -- see comment in mpm_solver.py
        # Only when active -- see the SNode-tree note in mpm_solver.py.
        if self.is_active and self._gravity is not None:
            gravity = qd_to_numpy(self._gravity, transpose=True)
            self._gravity = qd.field(dtype=gs.qd_vec3, shape=(self._B,))
            self._gravity.from_numpy(gravity)

    def _load_boundary_balls(self):
        """Local positions and density-mass weights, concatenated in boundary-set order."""
        if not self._has_boundary_balls:
            return np.zeros((0, 3)), np.zeros((0,), dtype=gs.np_float)
        chunks = []
        weight_chunks = []
        start = self._n_entity_particles
        initial_pos, initial_quat = self._normalized_boundary_ball_initial_poses()
        world_chunks = []
        for i_set, (npz_path, kinematic) in enumerate(self._boundary_ball_sets):
            if not os.path.isfile(npz_path):
                gs.raise_exception(f"boundary ball set npz not found: {npz_path}.")
            data = np.load(npz_path)
            if "pos_local" not in data.files:
                gs.raise_exception(f"boundary ball set npz {npz_path} is missing the `pos_local` key.")
            pos_local = data["pos_local"]
            if len(pos_local) == 0:
                gs.raise_exception(f"boundary ball set npz {npz_path} contains no particles.")
            if pos_local.ndim != 2 or pos_local.shape[1] != 3:
                gs.raise_exception(f"`pos_local` in {npz_path} must have shape (N, 3), got {pos_local.shape}.")
            if self._is_boundary_ball_mass_weight_enabled:
                if "mass_weight" not in data.files:
                    gs.raise_exception(
                        f"boundary ball set npz {npz_path} is missing `mass_weight` while "
                        "is_boundary_ball_mass_weight_enabled=True."
                    )
                mass_weight = data["mass_weight"]
                if mass_weight.ndim != 1 or mass_weight.shape != (len(pos_local),):
                    gs.raise_exception(
                        f"`mass_weight` in {npz_path} must have shape ({len(pos_local)},), got {mass_weight.shape}."
                    )
                if not np.issubdtype(mass_weight.dtype, np.number):
                    gs.raise_exception(f"`mass_weight` in {npz_path} must be numeric.")
                if not np.isfinite(mass_weight).all() or np.any(mass_weight <= 0.0):
                    gs.raise_exception(f"`mass_weight` in {npz_path} must be finite and strictly positive.")
            else:
                mass_weight = np.ones(len(pos_local), dtype=gs.np_float)
            self.boundary_ball_ranges.append((start, start + len(pos_local), bool(kinematic)))
            chunks.append(pos_local)
            weight_chunks.append(mass_weight)
            world_chunks.append(gu.transform_by_trans_quat(pos_local, initial_pos[i_set], initial_quat[i_set]))
            start += len(pos_local)
        pos = np.concatenate(chunks, axis=0)
        pos_world = np.concatenate(world_chunks, axis=0)
        # the hash grid wraps modulo the cell count, so a ball outside [lower_bound, upper_bound)
        # would land on the opposite side of the domain and corrupt every neighbor list
        if np.any(pos_world < self._lower_bound) or np.any(pos_world >= self._upper_bound):
            gs.raise_exception(
                "boundary ball set positions transformed by `boundary_ball_initial_poses` must "
                "lie inside [lower_bound, upper_bound)."
            )
        return pos, np.concatenate(weight_chunks, axis=0)

    def _normalized_boundary_ball_initial_poses(self):
        """Return build-time world poses, expanding the public empty tuple to identities."""
        if not self._boundary_ball_initial_poses:
            pos = np.zeros((self._n_ball_sets, 3), dtype=gs.np_float)
            quat = np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=gs.np_float),
                (self._n_ball_sets, 1),
            )
            return pos, quat
        pos = np.array([pose[0] for pose in self._boundary_ball_initial_poses])
        quat = np.array([pose[1] for pose in self._boundary_ball_initial_poses])
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        return pos, quat

    def set_boundary_ball_pose(self, i_set, pos, quat):
        """
        Set the pose of one kinematic boundary ball set.

        Parameters
        ----------
        i_set : int
            Index into `boundary_ball_sets`.
        pos : array-like, shape (3,)
            World-frame position of the set's local origin.
        quat : array-like, shape (4,)
            World-frame orientation as (w, x, y, z).

        Call once per frame before `scene.step()`. Static sets reject the call.
        """
        if not 0 <= i_set < self._n_ball_sets:
            gs.raise_exception(f"boundary ball set index {i_set} out of range.")
        if not self.boundary_ball_ranges[i_set][2]:
            gs.raise_exception("set_boundary_ball_pose requires a kinematic boundary ball set.")
        pos = tensor_to_array(pos)
        quat = tensor_to_array(quat)
        if pos.shape != (3,) or quat.shape != (4,):
            gs.raise_exception("set_boundary_ball_pose requires `pos` (3,) and `quat` (4, wxyz).")
        if not np.isfinite(pos).all() or not np.isfinite(quat).all():
            gs.raise_exception("set_boundary_ball_pose requires finite `pos` and `quat` values.")
        quat_norm = np.linalg.norm(quat)
        if not np.isfinite(quat_norm) or quat_norm <= np.finfo(gs.np_float).eps:
            gs.raise_exception("set_boundary_ball_pose requires a non-zero finite quaternion.")
        kernel_set_boundary_ball_pose(i_set, pos, quat / quat_norm, self)
        kernel_stage_boundary_ball_pose(i_set, self)
        self._is_boundary_pose_pending = True
        self._boundary_pose_substep = 0

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_entity(
        self, idx, material, morph, surface, name: str | None = None
    ) -> "PBD2DEntity | PBD3DEntity | PBDParticleEntity | PBDFreeParticleEntity | PBD3DSolidEntity":
        if isinstance(material, gs.materials.PBD.Cloth):
            entity = PBD2DEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
                edge_start=self.n_edges,
                inner_edge_start=self.n_inner_edges,
                vvert_start=self.n_vverts,
                vface_start=self.n_vfaces,
                name=name,
            )

        elif isinstance(material, gs.materials.PBD.Elastic):
            entity = PBD3DEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
                edge_start=self.n_edges,
                elem_start=self.n_elems,
                vvert_start=self.n_vverts,
                vface_start=self.n_vfaces,
                name=name,
            )

        elif isinstance(material, gs.materials.PBD.Liquid):
            entity = PBDParticleEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
                name=name,
            )

        elif isinstance(material, gs.materials.PBD.Particle):
            entity = PBDFreeParticleEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
                name=name,
            )

        elif isinstance(material, gs.materials.PBD.Solid):
            entity = PBD3DSolidEntity(
                scene=self.scene,
                solver=self,
                material=material,
                morph=morph,
                surface=surface,
                particle_size=self._particle_size,
                idx=idx,
                particle_start=self.n_particles,
                cluster_start=self.n_clusters,
                cluster_member_start=self.n_cluster_members,
                bond_start=self.n_bonds,
                seam_start=self.n_seams,
                fragment_start=self.n_fragments,
                vvert_start=self.n_vverts,
                vface_start=self.n_vfaces,
                name=name,
            )

        else:
            raise NotImplementedError()

        self._entities.append(entity)

        return entity

    # ------------------------------------------------------------------------------------
    # ------------------------------------- utils ----------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def poly6(self, dist):
        # dist is a VECTOR
        result = gs.qd_float(0.0)
        d = dist.norm() / self.dist_scale
        if 0 < d < self.h:
            rhs = (self.h_2 - d * d) * (self.h_2 - d * d) * (self.h_2 - d * d)
            result = self.poly6_Coe * rhs / self.h_9
        return result

    @qd.func
    def poly6_scalar(self, dist):
        # dist is a SCALAR
        result = gs.qd_float(0.0)
        d = dist
        if 0 < d < self.h:
            rhs = (self.h_2 - d * d) * (self.h_2 - d * d) * (self.h_2 - d * d)
            result = self.poly6_Coe * rhs / self.h_9
        return result

    @qd.func
    def spiky(self, dist):
        # dist is a VECTOR
        result = qd.Vector.zero(gs.qd_float, 3)
        d = dist.norm() / self.dist_scale
        if 0 < d < self.h:
            m = (self.h - d) * (self.h - d)
            result = (self.spiky_Coe * m / (self.h_6 * d)) * dist / self.dist_scale
        return result

    @qd.func
    def S_Corr(self, dist):
        upper = self.poly6(dist)
        lower = self.poly6_scalar(self.S_Corr_delta_q)
        m = upper / lower
        return -1.0 * self.S_Corr_k * m * m * m * m

    @qd.func
    def _st_cubic_dW(self, r):
        """First radial derivative of the cubic spline with support R_st (PBSTF reference form)."""
        res = gs.qd_float(0.0)
        h = self._st_ring_radius
        coefficient = 48.0 / (math.pi * h**4)
        q = r / h
        if q < 0.5:
            res = coefficient * q * (3.0 * q - 2.0)
        elif q < 1.0:
            res = coefficient * (1.0 - q) * (q - 1.0)
        return res

    @qd.func
    def _st_angle_of_vec2(self, a, b):
        result = gs.qd_float(0.0)
        a_len = a.norm()
        b_len = b.norm()
        if a_len > gs.EPS and b_len > gs.EPS:
            result = qd.acos(qd.max(-1.0, qd.min(1.0, a.dot(b) / (a_len * b_len))))
        return result

    @qd.func
    def _st_need_flip(self, i, i_b, u):
        result = False
        u_pre = self._chain_pre[i, i_b, u]
        u_nxt = self._chain_nxt[i, i_b, u]
        if u_pre >= 0 and u_nxt >= 0:
            x_pre = self.projected_positions[i, i_b, u_pre]
            x_u = self.projected_positions[i, i_b, u]
            x_nxt = self.projected_positions[i, i_b, u_nxt]
            result = self._st_angle_of_vec2(-x_pre, x_u - x_pre) + self._st_angle_of_vec2(-x_nxt, x_u - x_nxt) > math.pi
        return result

    def _st_rebuild_topology(self, f):
        # topology (surface detect -> normals -> local meshes) is rebuilt OUTSIDE the constraint
        # loop; the area constraint values/gradients are still recomputed at current positions
        # inside every density iteration
        self._screen_blocked.fill(0)
        kernel_st_mark_surface_screen(self)
        kernel_st_classify_surface(self)
        kernel_st_compute_normals(self)
        self._overflow.fill(0)
        kernel_st_build_local_meshes(self)
        # overflow strategy: raise like the PBSTF reference
        overflow = qd_to_numpy(self._overflow, transpose=True)[()]
        if overflow == self._ST_LOCAL_MESH_NEIGHBOR_OVERFLOW:
            gs.raise_exception(
                "PBD-ST local mesh exceeded its one-ring capacity; increase "
                f"`st_max_localmesh_neighbors` (currently {self._st_max_localmesh_neighbors})."
            )
        if overflow:
            detail = "candidate capacity" if overflow == self._ST_SURFACE_NEIGHBOR_OVERFLOW else "queue capacity"
            gs.raise_exception(
                f"PBD-ST local mesh exceeded its {detail}; increase `st_max_surface_neighbors` "
                f"(currently {self._st_max_surface_neighbors})."
            )

    @qd.func
    def _st_triangle_area(self, a, b, c, i_b):
        pa = self.particles_reordered[a, i_b].pos
        pb = self.particles_reordered[b, i_b].pos
        pc = self.particles_reordered[c, i_b].pos
        return 0.5 * (pb - pa).cross(pc - pa).norm()

    @qd.func
    def _st_triangle_area_gradient(self, a, b, c, i_b):
        pa = self.particles_reordered[a, i_b].pos
        pb = self.particles_reordered[b, i_b].pos
        pc = self.particles_reordered[c, i_b].pos
        cross = (pb - pa).cross(pc - pa)
        result = qd.Vector.zero(gs.qd_float, 3)
        if cross.norm() > gs.EPS:
            result = 0.5 * cross.normalized().cross(pc - pb)
        return result

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------
    @qd.kernel
    def _kernel_store_initial_pos(self, f: qd.i32, preserve_boundary_ipos: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if not (preserve_boundary_ipos and self.particles_ng[i_p, i_b].is_boundary):
                if not self.particles_ng[i_p, i_b].is_native_owned:
                    self.particles[i_p, i_b].ipos = self.particles[i_p, i_b].pos

    @qd.kernel
    def _kernel_reorder_particles(self, f: qd.i32):
        self.sh.compute_reordered_idx(
            self._n_particles, self.particles.pos, self.particles_ng.active, self.particles_ng.reordered_idx
        )

        # copy to reordered
        self.particles_ng_reordered.active.fill(False)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                reordered_idx = self.particles_ng[i_p, i_b].reordered_idx

                self.particles_reordered[reordered_idx, i_b] = self.particles[i_p, i_b]
                self.particles_info_reordered[reordered_idx, i_b] = self.particles_info[i_p]
                self.particles_ng_reordered[reordered_idx, i_b].active = self.particles_ng[i_p, i_b].active
                self.particles_ng_reordered[reordered_idx, i_b].original_idx = i_p
                self.particles_ng_reordered[reordered_idx, i_b].boundary_group = self.particles_ng[
                    i_p, i_b
                ].boundary_group
                # a missed field here silently reads as False in every reordered-space gate
                self.particles_ng_reordered[reordered_idx, i_b].is_boundary = self.particles_ng[i_p, i_b].is_boundary
                self.particles_ng_reordered[reordered_idx, i_b].boundary_set = self.particles_ng[i_p, i_b].boundary_set
                self.particles_ng_reordered[reordered_idx, i_b].is_native_owned = self.particles_ng[
                    i_p, i_b
                ].is_native_owned

    @qd.kernel
    def _kernel_apply_external_force(self, f: qd.i32, t: qd.f32):
        if qd.static(self._n_clusters > 0):
            for i_b in range(self._B):
                self.is_render_dirty[i_b] = True
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            # boundary balls are frozen: no force, no integration (a kinematic set is moved by
            # set_boundary_ball_pose writing the position array directly); bridge-owned particles are
            # likewise advanced by the native solver, with pos/vel synced from it at the substep tail
            if not self.particles_ng[i_p, i_b].is_boundary and not self.particles_ng[i_p, i_b].is_native_owned:
                if self.particles[i_p, i_b].free:
                    # gravity
                    self.particles[i_p, i_b].vel = self.particles[i_p, i_b].vel + self._gravity[i_b] * self._substep_dt

                    # external force fields
                    acc = qd.Vector.zero(gs.qd_float, 3)
                    for i_ff in qd.static(range(len(self._ffs))):
                        acc += self._ffs[i_ff].get_acc(
                            self.particles[i_p, i_b].pos, self.particles[i_p, i_b].vel, t, i_p
                        )
                    self.particles[i_p, i_b].vel = self.particles[i_p, i_b].vel + acc * self._substep_dt

                    if self.particles_info[i_p].material_type == self.MATERIAL.CLOTH:
                        f_air_resistance = (
                            self.particles_info[i_p].air_resistance
                            * self.particles[i_p, i_b].vel.norm()
                            * self.particles[i_p, i_b].vel
                        )
                        self.particles[i_p, i_b].vel = (
                            self.particles[i_p, i_b].vel
                            - f_air_resistance / self.particles_info[i_p].mass * self._substep_dt
                        )

                # attached particles are not free but still need to update position to follow the link
                self.particles[i_p, i_b].pos = (
                    self.particles[i_p, i_b].pos + self.particles[i_p, i_b].vel * self._substep_dt
                )

    @qd.kernel
    def _kernel_solve_stretch(self, f: qd.i32):
        for _ in qd.static(range(self._max_stretch_solver_iterations)):
            for i_e, i_b in qd.ndrange(self._n_edges, self._B):
                v1 = self.edges_info[i_e].v1
                v2 = self.edges_info[i_e].v2

                w1 = self.particles[v1, i_b].free / self.particles_info[v1].mass
                w2 = self.particles[v2, i_b].free / self.particles_info[v2].mass
                n = self.particles[v1, i_b].pos - self.particles[v2, i_b].pos
                C = n.norm() - self.edges_info[i_e].len_rest
                alpha = self.edges_info[i_e].stretch_compliance / (self._substep_dt**2)
                dp = -C / (w1 + w2 + alpha) * n / n.norm(gs.EPS) * self.edges_info[i_e].stretch_relaxation
                self.particles[v1, i_b].dpos += dp * w1
                self.particles[v2, i_b].dpos -= dp * w2

            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if self.particles[i_p, i_b].free and self.particles_info[i_p].material_type != self.MATERIAL.PARTICLE:
                    self.particles[i_p, i_b].pos = self.particles[i_p, i_b].pos + self.particles[i_p, i_b].dpos
                    self.particles[i_p, i_b].dpos.fill(0)

    @qd.kernel
    def _kernel_solve_bending(self, f: qd.i32):
        for _ in qd.static(range(self._max_bending_solver_iterations)):
            for i_ie, i_b in qd.ndrange(self._n_inner_edges, self._B):  # 140 - 142
                v1 = self.inner_edges_info[i_ie].v1
                v2 = self.inner_edges_info[i_ie].v2
                v3 = self.inner_edges_info[i_ie].v3
                v4 = self.inner_edges_info[i_ie].v4

                w1 = self.particles[v1, i_b].free / self.particles_info[v1].mass
                w2 = self.particles[v2, i_b].free / self.particles_info[v2].mass
                w3 = self.particles[v3, i_b].free / self.particles_info[v3].mass
                w4 = self.particles[v4, i_b].free / self.particles_info[v4].mass

                if w1 + w2 + w3 + w4 > 0.0:
                    # https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
                    # Appendix A: Bending Constraint Projection
                    p2 = self.particles[v2, i_b].pos - self.particles[v1, i_b].pos
                    p3 = self.particles[v3, i_b].pos - self.particles[v1, i_b].pos
                    p4 = self.particles[v4, i_b].pos - self.particles[v1, i_b].pos
                    l23 = p2.cross(p3).norm()
                    l24 = p2.cross(p4).norm()
                    n1 = p2.cross(p3) / l23
                    n2 = p2.cross(p4) / l24
                    d = qd.math.clamp(n1.dot(n2), -1.0, 1.0)

                    q3 = (p2.cross(n2) + n1.cross(p2) * d) / l23  # eq. (25)
                    q4 = (p2.cross(n1) + n2.cross(p2) * d) / l24  # eq. (26)
                    q2 = -(p3.cross(n2) + n1.cross(p3) * d) / l23 - (p4.cross(n1) + n2.cross(p4) * d) / l24  # eq. (27)
                    q1 = -q2 - q3 - q4
                    # eq. (29)
                    sum_wq = w1 * q1.norm_sqr() + w2 * q2.norm_sqr() + w3 * q3.norm_sqr() + w4 * q4.norm_sqr()
                    constraint = qd.acos(d) - qd.acos(-1.0)

                    # XPBD
                    alpha = self.inner_edges_info[i_ie].bending_compliance / (self._substep_dt**2)
                    constraint = (
                        -qd.sqrt(1 - d**2)
                        * constraint
                        / (sum_wq + alpha)
                        * self.inner_edges_info[i_ie].bending_relaxation
                    )

                    self.particles[v1, i_b].dpos += w1 * constraint * q1
                    self.particles[v2, i_b].dpos += w2 * constraint * q2
                    self.particles[v3, i_b].dpos += w3 * constraint * q3
                    self.particles[v4, i_b].dpos += w4 * constraint * q4

            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if self.particles[i_p, i_b].free and self.particles_info[i_p].material_type != self.MATERIAL.PARTICLE:
                    self.particles[i_p, i_b].pos = self.particles[i_p, i_b].pos + self.particles[i_p, i_b].dpos
                    self.particles[i_p, i_b].dpos.fill(0)

    @qd.kernel
    def _kernel_solve_volume(self, f: qd.i32):
        for _ in qd.static(range(self._max_volume_solver_iterations)):
            for i_el, i_b in qd.ndrange(self._n_elems, self._B):
                v1 = self.elems_info[i_el].v1
                v2 = self.elems_info[i_el].v2
                v3 = self.elems_info[i_el].v3
                v4 = self.elems_info[i_el].v4

                p1 = self.particles[v1, i_b].pos
                p2 = self.particles[v2, i_b].pos
                p3 = self.particles[v3, i_b].pos
                p4 = self.particles[v4, i_b].pos

                grad1 = (p4 - p2).cross(p3 - p2) / 6.0
                grad2 = (p3 - p1).cross(p4 - p1) / 6.0
                grad3 = (p4 - p1).cross(p2 - p1) / 6.0
                grad4 = (p2 - p1).cross(p3 - p1) / 6.0

                w1 = self.particles[v1, i_b].free / self.particles_info[v1].mass * grad1.norm_sqr()
                w2 = self.particles[v2, i_b].free / self.particles_info[v2].mass * grad2.norm_sqr()
                w3 = self.particles[v3, i_b].free / self.particles_info[v3].mass * grad3.norm_sqr()
                w4 = self.particles[v4, i_b].free / self.particles_info[v4].mass * grad4.norm_sqr()

                if w1 + w2 + w3 + w4 > 0.0:
                    vol = gu.qd_tet_vol(p1, p2, p3, p4)
                    C = vol - self.elems_info[i_el].vol_rest
                    alpha = self.elems_info[i_el].volume_compliance / (self._substep_dt**2)
                    s = -C / (w1 + w2 + w3 + w4 + alpha) * self.elems_info[i_el].volume_relaxation

                    self.particles[v1, i_b].dpos += s * w1 * grad1
                    self.particles[v2, i_b].dpos += s * w2 * grad2
                    self.particles[v3, i_b].dpos += s * w3 * grad3
                    self.particles[v4, i_b].dpos += s * w4 * grad4

            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if self.particles[i_p, i_b].free and self.particles_info[i_p].material_type != self.MATERIAL.PARTICLE:
                    self.particles[i_p, i_b].pos = self.particles[i_p, i_b].pos + self.particles[i_p, i_b].dpos
                    self.particles[i_p, i_b].dpos.fill(0)

    @qd.func
    def _func_cluster_rigid_fit(self, i_c, i_b):
        """Best rigid transform in f64 fitting the cluster's current particle positions.

        The centroid sums run in f64: at f32 the mass-weighted accumulation over a large cluster carries a
        systematic bias at the 1e-7 m scale, and the projection then steadily cancels that bias against
        whatever moves the body (e.g. gravity).
        """
        member_start = self.clusters_info[i_c].member_start
        member_count = self.clusters_info[i_c].member_count
        cm64 = qd.Vector.zero(qd.f64, 3)
        rest_cm64 = qd.Vector.zero(qd.f64, 3)
        mass_sum64 = qd.f64(0.0)
        for i_m in range(member_start, member_start + member_count):
            i_p = self.cluster_member_particles[i_m]
            mass64 = qd.f64(self.particles_info[i_p].mass) / qd.max(self.particle_cluster_ptr[i_p].member_count, 1)
            cm64 += mass64 * self.particles[i_p, i_b].pos.cast(qd.f64)
            rest_cm64 += mass64 * self.particles_solid_rest[i_p, i_b].cast(qd.f64)
            mass_sum64 += mass64
        cm64 /= mass_sum64
        rest_cm64 /= mass_sum64

        A_pq = qd.Matrix.zero(qd.f64, 3, 3)
        for i_m in range(member_start, member_start + member_count):
            i_p = self.cluster_member_particles[i_m]
            mass = qd.f64(self.particles_info[i_p].mass) / qd.max(self.particle_cluster_ptr[i_p].member_count, 1)
            p_off = self.particles[i_p, i_b].pos.cast(qd.f64) - cm64
            q_off = self.particles_solid_rest[i_p, i_b].cast(qd.f64) - rest_cm64
            A_pq += mass * p_off.outer_product(q_off)
        return cm64, rest_cm64, func_pbd_polar_rotation(A_pq)

    @qd.kernel
    def _kernel_solve_shape_matching(self, f: qd.i32):
        if qd.static(self._n_clusters > 0):
            for i_b in range(self._B):
                self.is_render_dirty[i_b] = True
        for _ in qd.static(range(self._max_shape_matching_solver_iterations)):
            func_fit_pbd_clusters(self, welded=True, render_only=False)
            for i_m, i_b in qd.ndrange(self._n_cluster_members, self._B):
                i_c = self.cluster_member_clusters[i_m]
                i_p = self.cluster_member_particles[i_m]
                i_fit = i_c
                if qd.static(self._n_bonds > 0):
                    i_fit = self.cluster_roots[i_c, i_b]
                if self.particles[i_p, i_b].free and not self.particles_ng[i_p, i_b].is_native_owned:
                    goal = (
                        self.clusters_fit[i_fit, i_b].rot
                        @ (self.particles_solid_rest[i_p, i_b].cast(qd.f64) - self.clusters_fit[i_fit, i_b].rest_cm)
                        + self.clusters_fit[i_fit, i_b].cm
                        - self.particles[i_p, i_b].pos.cast(qd.f64)
                    ).cast(gs.qd_float)
                    self.particles[i_p, i_b].dpos += (
                        self.clusters_info[i_c].stiffness * goal
                        / qd.max(self.particle_cluster_ptr[i_p].member_count, 1)
                    )
            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if (
                    self.particles[i_p, i_b].free
                    and not self.particles_ng[i_p, i_b].is_native_owned
                    and self.particles_info[i_p].material_type == self.MATERIAL.SOLID
                ):
                    self.particles[i_p, i_b].vel += self.particles[i_p, i_b].dpos / self._substep_dt
                    self.particles[i_p, i_b].pos = self.particles[i_p, i_b].pos + self.particles[i_p, i_b].dpos
                    self.particles[i_p, i_b].dpos.fill(0)

        func_project_pbd_rigid_velocities(self)

    @qd.kernel
    def _kernel_add_seam_fracture_params(
        self,
        seam_start: qd.i32,
        n_seams: qd.i32,
        shear_threshold: qd.f32,
        rotation_threshold: qd.f32,
        failure_threshold: qd.f32,
    ):
        for i_s_ in range(n_seams):
            self.seams_info[i_s_ + seam_start].shear_threshold = shear_threshold
            self.seams_info[i_s_ + seam_start].rotation_threshold = rotation_threshold
            self.seams_info[i_s_ + seam_start].failure_threshold = failure_threshold

    @qd.func
    def _func_relative_cluster_rotation(self, i_p, j_p, i_b):
        """Rotation angle between independently fitted fragments before welded projection.

        Each fracturable fragment owns one cluster. Independent fits retain the angular
        load residual even when the fragments subsequently share a connected-group transform.
        """
        i_c_i = self.particle_cluster_ids[self.particle_cluster_ptr[i_p].member_start]
        i_c_j = self.particle_cluster_ids[self.particle_cluster_ptr[j_p].member_start]
        R_rel = self.clusters_state[i_c_i, i_b].rot.transpose() @ self.clusters_state[i_c_j, i_b].rot
        # vee(R - R^T) has norm 2 sin(angle) and trace(R) - 1 = 2 cos(angle) for a pure rotation
        skew_norm = qd.sqrt(
            (R_rel[2, 1] - R_rel[1, 2]) ** 2 + (R_rel[0, 2] - R_rel[2, 0]) ** 2 + (R_rel[1, 0] - R_rel[0, 1]) ** 2
        )
        return qd.atan2(skew_norm, R_rel.trace() - 1.0)

    @qd.kernel
    def _kernel_fracture_bonds(self, f: qd.i32):
        # runs before the constraint projection, so the strain measures the deformation left by the
        # previous substep's contacts instead of an already-healed configuration
        # Fit each fragment independently before healing any contact/prediction
        # residual. Shared weld rotations must not disable angular failure.
        func_compute_cohesive_reaction(self)
        func_fit_pbd_clusters(self, welded=False, render_only=False)
        for i_l, i_b in qd.ndrange(self._n_bonds, self._B):
            if self.bonds_state[i_l, i_b].alive:
                i_p = self.bonds_info[i_l].i_p
                j_p = self.bonds_info[i_l].j_p
                seam = self.bonds_info[i_l].seam
                rest = self.bonds_info[i_l].rest
                dist = (self.particles[i_p, i_b].pos - self.particles[j_p, i_b].pos).norm()
                stretch = qd.abs(dist - rest) / rest / self.bonds_info[i_l].threshold
                if stretch > 1.0 and self.seams_info[seam].failure_threshold <= 0.0:
                    self.bonds_state[i_l, i_b].alive = False
                    self.is_render_dirty[i_b] = True
                    qd.atomic_or(self.cluster_roots_dirty[i_b], 1)
                    self.cohesive.rebuild[i_b] = True
                    qd.atomic_sub(self.seams_state[seam, i_b].alive_count, 1)
                else:
                    # Keep all overload samples until cohesive seam failure; removing
                    # the stretched sample first would erase its share of accumulated load.
                    seam_info = self.seams_info[seam]
                    overload = qd.max(stretch, self.cohesive.overload[i_l, i_b])
                    if seam_info.shear_threshold > 0.0:
                        # the bond offset pulled back through fragment i's fit and differenced against
                        # the rest offset: whole-body rigid motion cancels because the fit undoes the
                        # common rotation, the axial component is stretch (distance threshold), and the
                        # perpendicular remainder is tangential slip
                        i_c_i = self.particle_cluster_ids[self.particle_cluster_ptr[i_p].member_start]
                        rest_offset = self.particles_solid_rest[i_p, i_b] - self.particles_solid_rest[j_p, i_b]
                        axis = rest_offset / rest
                        local_offset = (
                            self.clusters_state[i_c_i, i_b].rot.transpose()
                            @ (self.particles[i_p, i_b].pos - self.particles[j_p, i_b].pos)
                            - rest_offset
                        )
                        slip = (local_offset - axis * axis.dot(local_offset)).norm()
                        overload = qd.max(overload, slip / rest / seam_info.shear_threshold)
                    if seam_info.rotation_threshold > 0.0:
                        overload = qd.max(
                            overload,
                            self._func_relative_cluster_rotation(i_p, j_p, i_b) / seam_info.rotation_threshold,
                        )
                    if overload >= 1.0:
                        qd.atomic_add(self.seams_state[seam, i_b].overload_count, 1)

        # fold the overloaded share of the still-alive bonds into the seam damage, then decide death:
        # a seam dies atomically when its accumulated damage reaches the failure threshold or its last
        # bond broke; the bond pass below then kills every remaining bond of the seam at once
        for i_s, i_b in qd.ndrange(self._n_seams, self._B):
            if self.seams_state[i_s, i_b].overload_count > 0:
                if self.seams_state[i_s, i_b].alive_count > 0:
                    self.seams_state[i_s, i_b].damage += qd.cast(
                        self.seams_state[i_s, i_b].overload_count, gs.qd_float
                    ) / qd.cast(self.seams_state[i_s, i_b].alive_count, gs.qd_float)
                self.seams_state[i_s, i_b].overload_count = 0
            failure_threshold = self.seams_info[i_s].failure_threshold
            is_overloaded = failure_threshold > 0.0 and self.seams_state[i_s, i_b].damage >= failure_threshold
            if self.seams_state[i_s, i_b].alive_count == 0 or is_overloaded:
                self.seams_state[i_s, i_b].is_dead = True
                self.is_render_dirty[i_b] = True

        for i_l, i_b in qd.ndrange(self._n_bonds, self._B):
            if self.bonds_state[i_l, i_b].alive and self.seams_state[self.bonds_info[i_l].seam, i_b].is_dead:
                self.bonds_state[i_l, i_b].alive = False
                self.is_render_dirty[i_b] = True
                qd.atomic_or(self.cluster_roots_dirty[i_b], 1)
                self.cohesive.rebuild[i_b] = True
                qd.atomic_sub(self.seams_state[self.bonds_info[i_l].seam, i_b].alive_count, 1)

    @qd.kernel
    def _kernel_update_plasticity(self, f: qd.i32):
        if qd.static(self._n_clusters > 0):
            for i_b in range(self._B):
                self.is_render_dirty[i_b] = True
        # runs before the shape-matching projection, so the rigid fit measures the deformation left by the
        # previous substep's contacts instead of an already-healed configuration
        for i_c, i_b in qd.ndrange(self._n_clusters, self._B):
            if self.clusters_info[i_c].plastic_creep > 0.0:
                cm, rest_cm64, R = self._func_cluster_rigid_fit(i_c, i_b)
                cm = cm.cast(gs.qd_float)
                member_start = self.clusters_info[i_c].member_start
                member_count = self.clusters_info[i_c].member_count
                rest_cm = rest_cm64.cast(gs.qd_float)

                max_dev = gs.qd_float(0.0)
                for i_m in range(member_start, member_start + member_count):
                    i_p = self.cluster_member_particles[i_m]
                    goal = R @ (self.particles_solid_rest[i_p, i_b] - rest_cm) + cm
                    dev = (self.particles[i_p, i_b].pos - goal).norm()
                    max_dev = qd.max(max_dev, dev)

                if max_dev > self.clusters_info[i_c].yield_threshold:
                    # accumulate the cluster's back-transformed desired rest positions per member; the
                    # particle pass below blends them into one shared rest position, so overlapping
                    # clusters never pull toward conflicting rest shapes
                    creep = self.clusters_info[i_c].plastic_creep
                    for i_m in range(member_start, member_start + member_count):
                        i_p = self.cluster_member_particles[i_m]
                        q_hat = R.transpose() @ (self.particles[i_p, i_b].pos - cm) + rest_cm
                        self.plastic_q_sum[i_p, i_b] += creep * q_hat
                        self.plastic_w_sum[i_p, i_b] += creep
                        # every cluster owning this member must rebuild its rest centroid afterwards
                        for i_k in range(
                            self.particle_cluster_ptr[i_p].member_start,
                            self.particle_cluster_ptr[i_p].member_start + self.particle_cluster_ptr[i_p].member_count,
                        ):
                            self.clusters_state[self.particle_cluster_ids[i_k], i_b].is_rest_dirty = True

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_info[i_p].material_type == self.MATERIAL.SOLID:
                w_sum = self.plastic_w_sum[i_p, i_b]
                if w_sum > 0.0:
                    self.particles_solid_rest[i_p, i_b] += qd.min(w_sum, 1.0) * (
                        self.plastic_q_sum[i_p, i_b] / w_sum - self.particles_solid_rest[i_p, i_b]
                    )
                    self.plastic_q_sum[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    self.plastic_w_sum[i_p, i_b] = 0.0

        # rebuild the rest centroid of clusters whose members' rest positions crept (f64 sums; see
        # _func_cluster_rigid_fit for why the centroid cannot be accumulated in f32)
        for i_c, i_b in qd.ndrange(self._n_clusters, self._B):
            if self.clusters_state[i_c, i_b].is_rest_dirty:
                self.clusters_state[i_c, i_b].is_rest_dirty = False
                member_start = self.clusters_info[i_c].member_start
                member_count = self.clusters_info[i_c].member_count

                rest_cm64 = qd.Vector.zero(qd.f64, 3)
                mass_sum64 = qd.f64(0.0)
                for i_m in range(member_start, member_start + member_count):
                    i_p = self.cluster_member_particles[i_m]
                    mass64 = qd.f64(self.particles_info[i_p].mass)
                    rest_cm64 += mass64 * self.particles_solid_rest[i_p, i_b].cast(qd.f64)
                    mass_sum64 += mass64
                self.clusters_rest[i_c, i_b].cm = (rest_cm64 / mass_sum64).cast(gs.qd_float)

    @qd.func
    def _func_solve_collision(self, i, j, i_b):
        """j -> i"""

        cur_dist = (self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos).norm(gs.EPS)
        rest_dist = (
            self.particles_info_reordered[i, i_b].pos_rest - self.particles_info_reordered[j, i_b].pos_rest
        ).norm(gs.EPS)
        target_dist = self._particle_size  # target particle distance is 2 * particle radius, i.e. particle_size

        # Solid particles in one welded connected cluster share a rigid fit.
        # Only separated components may collide, including after seam failure.
        is_separated = rest_dist > target_dist
        if (
            self.particles_info_reordered[i, i_b].material_type == self.MATERIAL.SOLID
            and self.particles_info_reordered[j, i_b].material_type == self.MATERIAL.SOLID
        ):
            is_separated = (
                self.particles_info_reordered[i, i_b].fragment != self.particles_info_reordered[j, i_b].fragment
            )
            if is_separated:
                orig_i = self.particles_ng_reordered[i, i_b].original_idx
                orig_j = self.particles_ng_reordered[j, i_b].original_idx
                if qd.static(self._n_bonds > 0):
                    i_c = self.particle_cluster_ids[self.particle_cluster_ptr[orig_i].member_start]
                    j_c = self.particle_cluster_ids[self.particle_cluster_ptr[orig_j].member_start]
                    is_separated = self.cluster_roots[i_c, i_b] != self.cluster_roots[j_c, i_b]

        if cur_dist < target_dist and is_separated:
            wi = self.particles_reordered[i, i_b].free / self.particles_info_reordered[i, i_b].mass
            wj = self.particles_reordered[j, i_b].free / self.particles_info_reordered[j, i_b].mass
            n = (self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos) / cur_dist

            ### resolve collision ###
            self.particles_reordered[i, i_b].dpos += wi / (wi + wj) * (target_dist - cur_dist) * n

            ### apply friction ###
            # https://mmacklin.com/uppfrta_preprint.pdf
            # equation (23)
            dv = (self.particles_reordered[i, i_b].pos - self.particles_reordered[i, i_b].ipos) - (
                self.particles_reordered[j, i_b].pos - self.particles_reordered[j, i_b].ipos
            )
            dpos = -(dv - n * n.dot(dv))
            # equation (24)
            d = target_dist - cur_dist
            mu_s = qd.max(self.particles_info_reordered[i, i_b].mu_s, self.particles_info_reordered[j, i_b].mu_s)
            mu_k = qd.max(self.particles_info_reordered[i, i_b].mu_k, self.particles_info_reordered[j, i_b].mu_k)
            if dpos.norm() < mu_s * d:
                self.particles_reordered[i, i_b].dpos += wi / (wi + wj) * dpos
            else:
                self.particles_reordered[i, i_b].dpos += (
                    wi / (wi + wj) * dpos * qd.min(1.0, mu_k * d / dpos.norm(gs.EPS))
                )

    @qd.func
    def _func_smooth_barrel_projection(self, i, i_b):
        """Project against the particle-radius offset of the visible upright mug shell.

        A selected-set ball remains the broad-phase/CCD witness.  The analytic narrow phase is
        the exact 2-D Minkowski offset of the visible annular cross-section in (radius, z), so
        the side wall, top rim, bottom and their rounded corners share one source surface.
        Returning ``active`` suppresses the corrugated sphere response even when no correction
        is needed.  The previous position selects the connected free-space side and prevents a
        through-wall endpoint from flipping the projection normal.
        """

        active = False
        correction = qd.Vector.zero(gs.qd_float, 3)
        normal = qd.Vector.zero(gs.qd_float, 3)
        penetration = gs.qd_float(0.0)
        pos_old = self.particles_reordered[i, i_b].ipos
        pos_now = self.particles_reordered[i, i_b].pos
        old_x = pos_old[0] - self._smooth_barrel_cx
        old_y = pos_old[1] - self._smooth_barrel_cy
        now_x = pos_now[0] - self._smooth_barrel_cx
        now_y = pos_now[1] - self._smooth_barrel_cy
        r_old = qd.sqrt(old_x * old_x + old_y * old_y)
        r_now = qd.sqrt(now_x * now_x + now_y * now_y)
        theta_old = qd.atan2(old_y, old_x)
        theta_now = qd.atan2(now_y, now_x)
        dtheta_old = qd.atan2(
            qd.sin(theta_old - self._smooth_barrel_excluded_azimuth),
            qd.cos(theta_old - self._smooth_barrel_excluded_azimuth),
        )
        dtheta_now = qd.atan2(
            qd.sin(theta_now - self._smooth_barrel_excluded_azimuth),
            qd.cos(theta_now - self._smooth_barrel_excluded_azimuth),
        )
        segment_hits_excluded_z = (
            qd.min(pos_old[2], pos_now[2]) <= self._smooth_barrel_excluded_z_max
            and qd.max(pos_old[2], pos_now[2]) >= self._smooth_barrel_excluded_z_min
        )
        in_handle_seam = segment_hits_excluded_z and (
            qd.abs(dtheta_old) <= self._smooth_barrel_excluded_half_angle
            or qd.abs(dtheta_now) <= self._smooth_barrel_excluded_half_angle
        )
        if not in_handle_seam:
            active = True
            radial = qd.Vector([1.0, 0.0, 0.0])
            if r_now > gs.EPS:
                radial = qd.Vector([now_x / r_now, now_y / r_now, 0.0])
            elif r_old > gs.EPS:
                radial = qd.Vector([old_x / r_old, old_y / r_old, 0.0])

            # The strict legal-overflow history requires a particle centre to clear both the
            # outer contact cylinder and the top contact plane.  Stage a near-rim outward
            # crossing on the top plane for one solve instead of letting the rounded corner
            # cross r_outer below z_top+radius.  The guard is finite (only the real rim
            # transition band and only up to r_outer+radius), so it cannot create an infinite
            # invisible collar outside the mug.
            rim_staged = False
            outer_contact = self._smooth_barrel_r_outer_visible + self.particle_radius
            top_contact = self._smooth_barrel_z_max + self.particle_radius
            if r_old < outer_contact and r_now >= outer_contact and r_now > r_old:
                t_outer = (outer_contact - r_old) / (r_now - r_old)
                z_outer = pos_old[2] + t_outer * (pos_now[2] - pos_old[2])
                if z_outer >= self._smooth_barrel_z_max - 3.0 * self.particle_radius and z_outer < top_contact:
                    target_r = outer_contact - 1.0e-4 * self._particle_size
                    target_z = qd.max(pos_now[2], top_contact)
                    target = qd.Vector(
                        [
                            self._smooth_barrel_cx + target_r * radial[0],
                            self._smooth_barrel_cy + target_r * radial[1],
                            target_z,
                        ]
                    )
                    correction = target - pos_now
                    normal = correction.normalized(gs.EPS)
                    penetration = correction.norm()
                    rim_staged = True

            # Closest point on the visible solid rectangle in the (radius,z) half-plane.
            q_r_now = qd.min(qd.max(r_now, self._smooth_barrel_r_inner_visible), self._smooth_barrel_r_outer_visible)
            q_z_now = qd.min(qd.max(pos_now[2], self._smooth_barrel_z_min), self._smooth_barrel_z_max)
            v_r_now = r_now - q_r_now
            v_z_now = pos_now[2] - q_z_now
            dist_now = qd.sqrt(v_r_now * v_r_now + v_z_now * v_z_now)
            if not rim_staged and dist_now < self.particle_radius:
                q_r_old = qd.min(
                    qd.max(r_old, self._smooth_barrel_r_inner_visible), self._smooth_barrel_r_outer_visible
                )
                q_z_old = qd.min(qd.max(pos_old[2], self._smooth_barrel_z_min), self._smooth_barrel_z_max)
                v_r_old = r_old - q_r_old
                v_z_old = pos_old[2] - q_z_old
                dist_old = qd.sqrt(v_r_old * v_r_old + v_z_old * v_z_old)
                n_r = gs.qd_float(0.0)
                n_z = gs.qd_float(0.0)
                if dist_old > gs.EPS:
                    n_r = v_r_old / dist_old
                    n_z = v_z_old / dist_old
                else:
                    # ipos should be admissible.  This fallback is deterministic for a state
                    # restored inside the solid and keeps it on its previous radial side.
                    mid_radius = 0.5 * (self._smooth_barrel_r_inner_visible + self._smooth_barrel_r_outer_visible)
                    if r_old <= mid_radius:
                        n_r = -1.0
                    else:
                        n_r = 1.0
                # Keep the anchor on the feature occupied at ipos.  Using q_now directly is
                # wrong for a one-substep through-wall endpoint: a cavity-side normal combined
                # with the outer-face closest point places the target inside the solid.  The
                # old feature fixes the connected side, while q_now supplies only the tangent
                # coordinate along that same face/corner.
                anchor_r = q_r_now
                anchor_z = q_z_now
                if qd.abs(n_r) > gs.EPS:
                    if n_r < 0.0:
                        anchor_r = self._smooth_barrel_r_inner_visible
                    else:
                        anchor_r = self._smooth_barrel_r_outer_visible
                if qd.abs(n_z) > gs.EPS:
                    if n_z < 0.0:
                        anchor_z = self._smooth_barrel_z_min
                    else:
                        anchor_z = self._smooth_barrel_z_max
                target_r = anchor_r + self.particle_radius * n_r
                target_z = anchor_z + self.particle_radius * n_z
                target = qd.Vector(
                    [
                        self._smooth_barrel_cx + target_r * radial[0],
                        self._smooth_barrel_cy + target_r * radial[1],
                        target_z,
                    ]
                )
                correction = target - pos_now
                normal = qd.Vector([n_r * radial[0], n_r * radial[1], n_z])
                penetration = correction.norm()
        return active, correction, normal, penetration

    @qd.func
    def _func_ball_collision_candidate(self, i, j, i_b):
        """Return one fluid-vs-boundary-ball contact candidate without mutating `dpos`.

        Boundary-ball influence spheres overlap.  Summing every pair projection can therefore
        move a particle through the wall; the caller instead selects the earliest swept hit (or
        the deepest endpoint overlap when no sweep exists).
        """

        rel_now = self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos
        rel_old = self.particles_reordered[i, i_b].ipos - self.particles_reordered[j, i_b].ipos
        cur_dist = rel_now.norm(gs.EPS)
        target_dist = self._boundary_ball_radius + self.particle_radius
        old_dist = rel_old.norm(gs.EPS)
        hit_kind = gs.qd_int(0)  # 0 none, 1 overlap, 2 swept
        order = gs.qd_float(1.0e30)
        correction = qd.Vector.zero(gs.qd_float, 3)
        normal = qd.Vector.zero(gs.qd_float, 3)
        penetration = gs.qd_float(0.0)
        # A resting layer starts exactly at `target_dist`.  Roundoff can put that start a few
        # ulps inside the sphere, making `c >= 0` below false; if a density iteration then crosses
        # the whole sphere, the end point is outside again and the overlap branch also misses it.
        # Treat an on-surface start moving into the ball as a t=0 swept hit and keep the correction
        # on the side occupied at the beginning of the substep.
        contact_tol = 1.0e-4 * self._particle_size
        if old_dist > gs.EPS and old_dist <= target_dist + contact_tol and (rel_now - rel_old).dot(rel_old) < 0.0:
            normal = rel_old / old_dist
            correction = normal * target_dist - rel_now
            penetration = correction.norm()
            hit_kind = 2
            # Many overlapping wall balls can all report a t=0 hit.  Prefer the one that has
            # advanced furthest into the particle; hash iteration order is not a geometric rule.
            order = -penetration / target_dist
        # Prefer the first swept contact even when the end point still overlaps the sphere. Using
        # the end-point normal after crossing the center plane would push the particle through the
        # opposite side of the wall.
        seg = rel_now - rel_old
        a = seg.dot(seg)
        c = rel_old.dot(rel_old) - target_dist * target_dist
        if hit_kind == 0 and a > gs.EPS and c >= 0.0:
            b = 2.0 * rel_old.dot(seg)
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                t_hit = (-b - qd.sqrt(disc)) / (2.0 * a)
                if t_hit >= 0.0 and t_hit <= 1.0:
                    rel_hit = rel_old + t_hit * seg
                    normal = rel_hit.normalized(gs.EPS)
                    correction = rel_hit - rel_now
                    penetration = correction.norm()
                    hit_kind = 2
                    order = t_hit

        if hit_kind == 0 and cur_dist < target_dist:
            # Keep the contact on the side occupied at the beginning of this substep. At an exact
            # resting contact, roundoff can classify rel_old just inside the sphere; using rel_now
            # after it crossed the center plane would then flip the normal and eject through the
            # wall. ipos is refreshed after every kinematic pose update, so this remains valid for
            # moving ball sets.
            n = rel_now / cur_dist
            if old_dist > gs.EPS:
                n = rel_old / old_dist
            normal = n
            correction = n * target_dist - rel_now
            penetration = target_dist - cur_dist
            hit_kind = 1
            # Endpoint overlaps sort after every swept hit; among overlaps, deepest wins.
            order = 2.0 - penetration / target_dist
        return hit_kind, order, correction, normal, penetration

    @qd.func
    def _func_apply_ball_friction(self, i, j, i_b, normal, penetration):
        # Same tangential friction as _func_solve_collision, applied only for the selected ball.
        wi = self.particles_reordered[i, i_b].free / self.particles_info_reordered[i, i_b].mass
        wj = self.particles_reordered[j, i_b].free / self.particles_info_reordered[j, i_b].mass
        dv = (self.particles_reordered[i, i_b].pos - self.particles_reordered[i, i_b].ipos) - (
            self.particles_reordered[j, i_b].pos - self.particles_reordered[j, i_b].ipos
        )
        dpos = -(dv - normal * normal.dot(dv))
        mu_s = qd.max(self.particles_info_reordered[i, i_b].mu_s, self.particles_info_reordered[j, i_b].mu_s)
        mu_k = qd.max(self.particles_info_reordered[i, i_b].mu_k, self.particles_info_reordered[j, i_b].mu_k)
        if dpos.norm() < mu_s * penetration:
            self.particles_reordered[i, i_b].dpos += wi / (wi + wj) * dpos
        else:
            self.particles_reordered[i, i_b].dpos += (
                wi / (wi + wj) * dpos * qd.min(1.0, mu_k * penetration / dpos.norm(gs.EPS))
            )

    @qd.kernel
    def _kernel_solve_collision(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_info_reordered[i_p, i_b].material_type != self.MATERIAL.PARTICLE
                and not self.particles_ng_reordered[i_p, i_b].is_native_owned
            ):
                if qd.static(self._has_boundary_balls):
                    # The hash is built before density/viscosity projection.  Querying boundary
                    # balls from the projected end position can therefore miss a wall completely
                    # after a large correction crosses more than one cell.  Ball CCD must use the
                    # substep-start cell, whose neighborhood contains the wall before crossing.
                    # Non-ball contacts retain the stock current-position query below.
                    base_old = self.sh.pos_to_grid(self.particles_reordered[i_p, i_b].ipos)
                    best_kind = gs.qd_int(0)
                    best_order = gs.qd_float(1.0e30)
                    best_j = gs.qd_int(-1)
                    best_correction = qd.Vector.zero(gs.qd_float, 3)
                    best_normal = qd.Vector.zero(gs.qd_float, 3)
                    best_penetration = gs.qd_float(0.0)
                    smooth_witness_j = gs.qd_int(-1)
                    smooth_witness_dist = gs.qd_float(1.0e30)
                    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        slot_idx = self.sh.grid_to_slot(base_old + offset)
                        for j in range(
                            self.sh.slot_start[slot_idx, i_b],
                            self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                        ):
                            if (
                                i_p != j
                                and self.particles_ng_reordered[j, i_b].is_boundary
                                and (self.particles_reordered[i_p, i_b].free or self.particles_reordered[j, i_b].free)
                            ):
                                if qd.static(self._has_boundary_ball_smooth_barrel):
                                    ball_dx = self.particles_reordered[j, i_b].pos[0] - self._smooth_barrel_cx
                                    ball_dy = self.particles_reordered[j, i_b].pos[1] - self._smooth_barrel_cy
                                    ball_r = qd.sqrt(ball_dx * ball_dx + ball_dy * ball_dy)
                                    is_wall_ball = (
                                        ball_r >= self._smooth_barrel_r_inner_visible - self._boundary_ball_radius
                                        and ball_r <= self._smooth_barrel_r_outer_visible + self._boundary_ball_radius
                                    )
                                    if (
                                        self.particles_ng_reordered[j, i_b].boundary_set == self._smooth_barrel_set
                                        and is_wall_ball
                                    ):
                                        witness_dist = qd.min(
                                            (
                                                self.particles_reordered[i_p, i_b].ipos
                                                - self.particles_reordered[j, i_b].ipos
                                            ).norm(),
                                            (
                                                self.particles_reordered[i_p, i_b].pos
                                                - self.particles_reordered[j, i_b].pos
                                            ).norm(),
                                        )
                                        witness_limit = (
                                            self._boundary_ball_radius
                                            + self.particle_radius
                                            + self._smooth_barrel_witness_margin
                                        )
                                        if witness_dist <= witness_limit and witness_dist < smooth_witness_dist:
                                            smooth_witness_dist = witness_dist
                                            smooth_witness_j = j
                                hit_kind, order, correction, normal, penetration = self._func_ball_collision_candidate(
                                    i_p, j, i_b
                                )
                                if hit_kind != 0 and order < best_order:
                                    best_kind = hit_kind
                                    best_order = order
                                    best_j = j
                                    best_correction = correction
                                    best_normal = normal
                                    best_penetration = penetration
                    if best_j >= 0 or smooth_witness_j >= 0:
                        smooth_barrel = False
                        if qd.static(self._has_boundary_ball_smooth_barrel):
                            best_is_smooth_set = False
                            best_is_wall_ball = False
                            if best_j >= 0:
                                best_is_smooth_set = (
                                    self.particles_ng_reordered[best_j, i_b].boundary_set == self._smooth_barrel_set
                                )
                                if best_is_smooth_set:
                                    # Same wall-slab test as the witness loop: a floor-ball
                                    # contact of the smooth set is never replaced by the
                                    # analytic wall projection.
                                    best_dx = self.particles_reordered[best_j, i_b].pos[0] - self._smooth_barrel_cx
                                    best_dy = self.particles_reordered[best_j, i_b].pos[1] - self._smooth_barrel_cy
                                    best_r = qd.sqrt(best_dx * best_dx + best_dy * best_dy)
                                    best_is_wall_ball = (
                                        best_r >= self._smooth_barrel_r_inner_visible - self._boundary_ball_radius
                                        and best_r <= self._smooth_barrel_r_outer_visible + self._boundary_ball_radius
                                    )
                            if smooth_witness_j >= 0 and (best_j < 0 or best_is_smooth_set):
                                smooth_barrel, smooth_correction, smooth_normal, smooth_penetration = (
                                    self._func_smooth_barrel_projection(i_p, i_b)
                                )
                                if smooth_barrel and (best_j < 0 or best_is_wall_ball):
                                    best_correction = smooth_correction
                                    best_normal = smooth_normal
                                    best_penetration = smooth_penetration
                                else:
                                    # Analytic response rejected: keep the discrete ball
                                    # response (and the union cleanup below) untouched.
                                    smooth_barrel = False

                        trial_pos = self.particles_reordered[i_p, i_b].pos + best_correction
                        if not smooth_barrel:
                            for _ in qd.static(range(3)):
                                cleanup_penetration = gs.qd_float(0.0)
                                cleanup_correction = qd.Vector.zero(gs.qd_float, 3)
                                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                                    slot_idx = self.sh.grid_to_slot(base_old + offset)
                                    for j in range(
                                        self.sh.slot_start[slot_idx, i_b],
                                        self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                                    ):
                                        if i_p != j and self.particles_ng_reordered[j, i_b].is_boundary:
                                            rel = trial_pos - self.particles_reordered[j, i_b].pos
                                            dist = rel.norm()
                                            penetration = self._boundary_ball_radius + self.particle_radius - dist
                                            if penetration > cleanup_penetration and dist > gs.EPS:
                                                cleanup_penetration = penetration
                                                cleanup_correction = rel / dist * penetration
                                trial_pos += cleanup_correction
                        self.particles_reordered[i_p, i_b].dpos += trial_pos - self.particles_reordered[i_p, i_b].pos
                        if best_penetration > 0.0:
                            friction_j = best_j
                            if smooth_barrel:
                                friction_j = smooth_witness_j
                            self._func_apply_ball_friction(i_p, friction_j, i_b, best_normal, best_penetration)

                    base = self.sh.pos_to_grid(self.particles_reordered[i_p, i_b].pos)
                    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        slot_idx = self.sh.grid_to_slot(base + offset)
                        for j in range(
                            self.sh.slot_start[slot_idx, i_b],
                            self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                        ):
                            if (
                                i_p != j
                                and not self.particles_ng_reordered[j, i_b].is_boundary
                                and (self.particles_reordered[i_p, i_b].free or self.particles_reordered[j, i_b].free)
                                and not (
                                    self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID
                                    and self.particles_info_reordered[j, i_b].material_type == self.MATERIAL.LIQUID
                                )
                            ):
                                self._func_solve_collision(i_p, j, i_b)
                else:
                    # Preserve the stock path exactly when boundary_ball_sets is empty.
                    base = self.sh.pos_to_grid(self.particles_reordered[i_p, i_b].pos)
                    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        slot_idx = self.sh.grid_to_slot(base + offset)
                        for j in range(
                            self.sh.slot_start[slot_idx, i_b],
                            self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                        ):
                            if (
                                i_p != j
                                and not self.particles_ng_reordered[j, i_b].is_native_owned
                                and (self.particles_reordered[i_p, i_b].free or self.particles_reordered[j, i_b].free)
                                and not (
                                    self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID
                                    and self.particles_info_reordered[j, i_b].material_type == self.MATERIAL.LIQUID
                                )
                            ):
                                self._func_solve_collision(i_p, j, i_b)

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_reordered[i_p, i_b].free
                and not self.particles_ng_reordered[i_p, i_b].is_native_owned
                and self.particles_info_reordered[i_p, i_b].material_type != self.MATERIAL.PARTICLE
            ):
                if self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.SOLID:
                    self.particles_reordered[i_p, i_b].vel += (
                        self.particles_reordered[i_p, i_b].dpos / self._substep_dt
                    )
                self.particles_reordered[i_p, i_b].pos = (
                    self.particles_reordered[i_p, i_b].pos + self.particles_reordered[i_p, i_b].dpos
                )
                self.particles_reordered[i_p, i_b].dpos.fill(0)

    @qd.kernel
    def _kernel_solve_boundary_collision(self, f: qd.i32):
        if qd.static(self._n_clusters > 0):
            for i_b in range(self._B):
                self.is_render_dirty[i_b] = True
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            # boundary is enforced regardless of whether free
            # pre-declare before the branches: qd's AST resolver cannot see names that are
            # only assigned inside if/else branches
            corrected_pos = self.particles[i_p, i_b].pos
            corrected_vel = self.particles[i_p, i_b].vel
            if qd.static(self._has_boundary_pitcher):
                gid = self.particles_ng[i_p, i_b].boundary_group
                if qd.static(self._is_pitcher_shell):
                    # P2 shell keep region: R_out radial bound + past-rim adhesion exemption
                    if gid == 1 and not self.boundary2.in_keep_region(
                        corrected_pos, self._particle_size, self.particle_radius
                    ):
                        gid = 0
                        self.particles_ng[i_p, i_b].boundary_group = 0
                else:
                    if gid == 1 and not self.boundary2.in_keep_region(corrected_pos, self._particle_size):
                        gid = 0
                        self.particles_ng[i_p, i_b].boundary_group = 0
                if gid == 1:
                    corrected_pos, corrected_vel = self.boundary2.impose_pos_vel(corrected_pos, corrected_vel)
                else:
                    corrected_pos, corrected_vel = self.boundary.impose_pos_vel(corrected_pos, corrected_vel)
                    if qd.static(self._is_pitcher_shell):
                        # the shell is a world solid: it clamps EVERY particle inside its wall,
                        # regardless of group ownership (poured-out milk, blob impacts, coffee
                        # splashes). The exterior stays free -- impose only acts for d < 0.
                        corrected_pos, corrected_vel = self.boundary2.impose_pos_vel(corrected_pos, corrected_vel)
            else:
                corrected_pos, corrected_vel = self.boundary.impose_pos_vel(corrected_pos, corrected_vel)
            if qd.static(self._has_boundary_cup_shell):
                corrected_pos, corrected_vel = self.boundary_cup.impose_pos_vel(corrected_pos, corrected_vel)
            if qd.static(self._has_boundary_plane):
                corrected_pos, corrected_vel = self.boundary3.impose_pos_vel(corrected_pos, corrected_vel)
            # boundary balls ARE the boundary: no analytic clamp, no velocity writeback; bridge-owned
            # particles keep their derived pose untouched (their fragment collides natively)
            if not self.particles_ng[i_p, i_b].is_boundary and not self.particles_ng[i_p, i_b].is_native_owned:
                if self.particles[i_p, i_b].free and self.particles_info[i_p].material_type == self.MATERIAL.SOLID:
                    friction_pos = self._func_solve_boundary_friction(
                        self.particles[i_p, i_b].pos,
                        corrected_pos,
                        self.particles[i_p, i_b].ipos,
                        self.particles_info[i_p].mu_s,
                        self.particles_info[i_p].mu_k,
                    )
                    friction_shift = friction_pos - corrected_pos
                    # the velocity folds in the tangential correction, so the next substep integrates
                    # the damped motion instead of re-advancing the pre-friction velocity
                    corrected_vel = corrected_vel + friction_shift / self._substep_dt
                    corrected_pos = friction_pos
                self.particles[i_p, i_b].pos = corrected_pos
                self.particles[i_p, i_b].vel = corrected_vel

    @qd.func
    def _func_solve_boundary_friction(self, pos_pre, pos_clamped, ipos, mu_s, mu_k):
        """Tangential Coulomb friction for a solid particle against an analytic boundary projection.

        Applies the unified particle physics contact friction (Macklin et al. 2014, eq. 23/24), the
        same model as `_func_apply_ball_friction`: the tangential displacement accumulated since the
        substep start is canceled when it fits inside the static cone (`|dpos_tan| < mu_s * d`) and
        otherwise scaled by `mu_k * d / |dpos_tan|`, with `d` the normal correction the projection
        just applied. The boundary is static, so the particle carries the full correction. Returns
        the corrected position.
        """
        pos_frictioned = pos_clamped
        correction = pos_clamped - pos_pre
        d = correction.norm()
        if d > gs.EPS:
            n = correction / d
            dpos_tan = pos_clamped - ipos
            dpos_tan = -(dpos_tan - n * n.dot(dpos_tan))
            dpos_tan_norm = dpos_tan.norm()
            # the stick test and the slip scale both divide by the tangential displacement, which is
            # far below sqrt(f32 epsilon); a floored norm there crushes the correction, so the guard
            # is an explicit tiny constant instead of the squared-epsilon `.norm(eps)` convention
            if dpos_tan_norm < mu_s * d:
                pos_frictioned = pos_clamped + dpos_tan
            else:
                pos_frictioned = pos_clamped + dpos_tan * qd.min(1.0, mu_k * d / qd.max(dpos_tan_norm, 1e-30))
        return pos_frictioned

    @qd.func
    def _adh_take(self, ok, C, n, ok_b, C_b, n_b):
        tol = 0.1 * self.particle_radius
        take = False
        if ok_b:
            if not ok:
                take = True
            elif C_b < C - tol:
                take = True
            elif C_b < C + tol and qd.abs(n_b[2]) > qd.abs(n[2]):
                take = True
        if take:
            ok = True
            C = C_b
            n = n_b
        return ok, C, n

    @qd.func
    def _adh_query_nearest(self, i_p, i_b, pos, gid):
        ok = False
        C = gs.qd_float(1e30)
        n = qd.Vector.zero(gs.qd_float, 3)
        ok_b = False
        C_b = gs.qd_float(0.0)
        n_b = qd.Vector.zero(gs.qd_float, 3)
        if qd.static(self._has_boundary_shell):
            if qd.static(self._is_pitcher_shell):
                ok_b, C_b, n_b = self.boundary2.adhesion_query(pos, self.particle_radius)
                ok, C, n = self._adh_take(ok, C, n, ok_b, C_b, n_b)
            if qd.static(self._has_boundary_cup_shell):
                ok_b, C_b, n_b = self.boundary_cup.adhesion_query(pos, self.particle_radius)
                ok, C, n = self._adh_take(ok, C, n, ok_b, C_b, n_b)
            if qd.static(self._has_boundary_cylinder):
                ok_b, C_b, n_b = self.boundary.adhesion_query(pos)
                ok, C, n = self._adh_take(ok, C, n, ok_b, C_b, n_b)
        else:
            if qd.static(self._has_boundary_pitcher):
                if gid == 1:
                    ok_b, C_b, n_b = self.boundary2.adhesion_query(pos)
                else:
                    if qd.static(self._has_boundary_cylinder):
                        ok_b, C_b, n_b = self.boundary.adhesion_query(pos)
            else:
                if qd.static(self._has_boundary_cylinder):
                    ok_b, C_b, n_b = self.boundary.adhesion_query(pos)
            if ok_b:
                ok = True
                C = C_b
                n = n_b
        if qd.static(self._has_boundary_plane):
            ok_b, C_b, n_b = self.boundary3.adhesion_query(pos)
            if qd.static(self._has_boundary_shell):
                ok, C, n = self._adh_take(ok, C, n, ok_b, C_b, n_b)
            else:
                if ok_b and C_b < C:
                    ok = True
                    C = C_b
                    n = n_b
        if qd.static(self._has_boundary_balls):
            # frozen-fluid ball wall: the nearest ball center is tracked in the density pass (see
            # _kernel_solve_density), C is the distance to its surface and the stored
            # center-pointing direction flips into the surface normal pointing into the fluid
            C_b = self.bnd_dist[i_p, i_b] - self._boundary_ball_radius
            if C_b <= self.particle_radius:
                ok_b = True
                n_b = -self.bnd_dir[i_p, i_b]
                ok, C, n = self._adh_take(ok, C, n, ok_b, C_b, n_b)
        return ok, C, n

    @qd.kernel
    def _kernel_solve_density(self, f: qd.i32):
        for it in qd.static(range(self._max_density_solver_iterations)):
            # ---Calculate lambdas---
            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID:
                    rho_rest_i = self.particles_info_reordered[i_p, i_b].rho_rest
                    if qd.static(self._is_st_enabled):
                        if self.on_surface[i_p, i_b]:
                            rho_rest_i *= self._st_surface_density_factor
                    pos_i = self.particles_reordered[i_p, i_b].pos
                    base = self.sh.pos_to_grid(pos_i)
                    lower_sum = gs.qd_float(0.0)
                    rho = gs.qd_float(0.0)
                    spiky_i = qd.Vector.zero(gs.qd_float, 3)
                    if qd.static(self._has_boundary_balls):
                        # nearest frozen-fluid ball for the adhesion query (reset every iteration:
                        # the neighbor slot scan below is the only traversal it costs)
                        self.bnd_dist[i_p, i_b] = 1.0e30
                        self.bnd_dir[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
                    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        slot_idx = self.sh.grid_to_slot(base + offset)
                        for j in range(
                            self.sh.slot_start[slot_idx, i_b],
                            self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                        ):
                            pos_j = self.particles_reordered[j, i_b].pos
                            if qd.static(self._has_boundary_balls):
                                if self.particles_ng_reordered[j, i_b].is_boundary:
                                    delta_b = pos_j - pos_i
                                    dist_b = delta_b.norm()
                                    if dist_b > gs.EPS and dist_b < self.bnd_dist[i_p, i_b]:
                                        self.bnd_dist[i_p, i_b] = dist_b
                                        self.bnd_dir[i_p, i_b] = delta_b / dist_b
                            # ---Poly6---
                            rho += self.poly6(pos_i - pos_j) * self.particles_info_reordered[j, i_b].mass
                            # ---Spiky---
                            s = self.spiky(pos_i - pos_j) / rho_rest_i
                            spiky_i += s
                            lower_sum += s.dot(s)
                    constraint = (rho / rho_rest_i) - 1.0
                    if qd.static(self._is_density_clamped_negative):
                        # tension-free liquid: a negative constraint (free-surface dilatation)
                        # makes lambda positive and pulls the particle toward its neighbors and
                        # the boundary balls -- cohesion the explicit ST constraints already model
                        constraint = qd.max(constraint, gs.qd_float(0.0))
                    lower_sum += spiky_i.dot(spiky_i)
                    self.particles_reordered[i_p, i_b].lam = -1.0 * (constraint / (lower_sum + self.lambda_epsilon))

            # ---Calculate delta pos---
            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID:
                    rho_rest_i = self.particles_info_reordered[i_p, i_b].rho_rest
                    if qd.static(self._is_st_enabled):
                        if self.on_surface[i_p, i_b]:
                            rho_rest_i *= self._st_surface_density_factor
                    pos_i = self.particles_reordered[i_p, i_b].pos
                    base = self.sh.pos_to_grid(pos_i)
                    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        slot_idx = self.sh.grid_to_slot(base + offset)
                        for j in range(
                            self.sh.slot_start[slot_idx, i_b],
                            self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                        ):
                            if i_p != j:
                                pos_j = self.particles_reordered[j, i_b].pos
                                # ---S_Corr---
                                scorr = self.S_Corr(pos_i - pos_j)
                                left = (
                                    self.particles_reordered[i_p, i_b].lam
                                    + self.particles_reordered[j, i_b].lam
                                    + scorr
                                )
                                right = self.spiky(pos_i - pos_j)
                                self.particles_reordered[i_p, i_b].dpos = (
                                    self.particles_reordered[i_p, i_b].dpos
                                    + left
                                    * right
                                    / rho_rest_i
                                    * self.dist_scale
                                    * self.particles_info_reordered[i_p, i_b].density_relaxation
                                )

            if qd.static(self._is_st_enabled):
                # ---Surface tension: area constraint (PBSTF _kernel_apply_surface_constraints)---
                # lambda pass: constraint = sum of the fan triangle areas (minimized directly,
                # no rest area); gradients w.r.t. the center and each ring member
                self._surface_gradient.fill(0.0)
                for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                    if self.particles_ng_reordered[i_p, i_b].active and self.topology_valid[i_p, i_b]:
                        n = self.n_ring[i_p, i_b]
                        constraint = gs.qd_float(0.0)
                        grad_i = qd.Vector.zero(gs.qd_float, 3)
                        for k in range(n):
                            k_next = 0 if k == n - 1 else k + 1
                            j = self.local_mesh_neighbors[i_p, i_b, k]
                            j_next = self.local_mesh_neighbors[i_p, i_b, k_next]
                            constraint += self._st_triangle_area(i_p, j, j_next, i_b)
                            grad_i += self._st_triangle_area_gradient(i_p, j, j_next, i_b)
                            self._surface_gradient[i_p, i_b, k] += self._st_triangle_area_gradient(j, j_next, i_p, i_b)
                            self._surface_gradient[i_p, i_b, k_next] += self._st_triangle_area_gradient(
                                j_next, i_p, j, i_b
                            )

                        mass_i = self.particles_info_reordered[i_p, i_b].mass
                        denominator = self._st_compliance / self._st_default_mass
                        denominator += grad_i.norm_sqr() / mass_i
                        for k in range(n):
                            j = self.local_mesh_neighbors[i_p, i_b, k]
                            denominator += (
                                self._surface_gradient[i_p, i_b, k].norm_sqr()
                                / self.particles_info_reordered[j, i_b].mass
                            )
                        lmd = gs.qd_float(0.0)
                        if denominator > gs.EPS:
                            lmd = -constraint / denominator
                        self._surface_lambda[i_p, i_b] = lmd
                        self._surface_grad_i[i_p, i_b] = grad_i

                # apply pass: accumulate into the shared dpos buffer (the common apply below moves
                # the particles once per iteration)
                for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                    if self.particles_ng_reordered[i_p, i_b].active and self.topology_valid[i_p, i_b]:
                        lmd = self._surface_lambda[i_p, i_b]
                        mass_i = self.particles_info_reordered[i_p, i_b].mass
                        correction_i = lmd / mass_i * self._surface_grad_i[i_p, i_b]
                        for axis in qd.static(range(3)):
                            qd.atomic_add(self.particles_reordered[i_p, i_b].dpos[axis], correction_i[axis])
                        for k in range(self.n_ring[i_p, i_b]):
                            j = self.local_mesh_neighbors[i_p, i_b, k]
                            correction_j = (
                                lmd / self.particles_info_reordered[j, i_b].mass * self._surface_gradient[i_p, i_b, k]
                            )
                            for axis in qd.static(range(3)):
                                qd.atomic_add(self.particles_reordered[j, i_b].dpos[axis], correction_j[axis])

                # ---Surface tension: one-sided distance constraint between surface pairs---
                # (PBSTF _kernel_apply_distance_constraints, surface branch only), every 2nd iteration
                if qd.static(self._is_st_distance_enabled):
                    if it % 2 == 1:
                        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                            if (
                                self.particles_ng_reordered[i_p, i_b].active
                                and self.on_surface[i_p, i_b]
                                and self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID
                            ):
                                pos_i = self.particles_reordered[i_p, i_b].pos
                                base = self.sh.pos_to_grid(pos_i)
                                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                                    slot_idx = self.sh.grid_to_slot(base + offset)
                                    for j in range(
                                        self.sh.slot_start[slot_idx, i_b],
                                        self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                                    ):
                                        if (
                                            i_p < j
                                            and self.on_surface[j, i_b]
                                            and self.particles_info_reordered[j, i_b].material_type
                                            == self.MATERIAL.LIQUID
                                        ):
                                            pos_j = self.particles_reordered[j, i_b].pos
                                            delta = pos_i - pos_j
                                            distance = delta.norm()
                                            mass_i = self.particles_info_reordered[i_p, i_b].mass
                                            mass_j = self.particles_info_reordered[j, i_b].mass
                                            target = (
                                                self._particle_size
                                                * 0.5
                                                * (
                                                    qd.pow(mass_i / self._st_default_mass, 1.0 / 1.5)
                                                    + qd.pow(mass_j / self._st_default_mass, 1.0 / 1.5)
                                                )
                                            )
                                            if distance > gs.EPS and distance < target:
                                                constraint = distance - target
                                                denominator = (
                                                    self._st_distance_compliance / self._st_default_mass
                                                    + 1.0 / mass_i
                                                    + 1.0 / mass_j
                                                )
                                                lmd = -constraint / denominator
                                                grad_i = delta / distance
                                                correction_i = lmd / mass_i * grad_i
                                                correction_j = -lmd / mass_j * grad_i
                                                for axis in qd.static(range(3)):
                                                    qd.atomic_add(
                                                        self.particles_reordered[i_p, i_b].dpos[axis],
                                                        correction_i[axis],
                                                    )
                                                    qd.atomic_add(
                                                        self.particles_reordered[j, i_b].dpos[axis],
                                                        correction_j[axis],
                                                    )

            if qd.static(self._is_wall_adhesion_enabled):
                # ---Wall adhesion (PBSTF _kernel_apply_static_collider_adhesion)---
                # Signed two-sided distance constraint to the analytic container wall (big cup
                # or pitcher, dispatched per-particle by boundary_group), accumulated into the
                # shared dpos buffer EVERY Jacobi iteration and applied by the common pass below.
                # Gated to surface particles within one particle radius of the wall: interior
                # particles never stick (the key guard against near-wall clumping).
                for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                    if (
                        self.particles_ng_reordered[i_p, i_b].active
                        and self.on_surface[i_p, i_b]
                        and self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID
                    ):
                        pos_i = self.particles_reordered[i_p, i_b].pos
                        ok, constraint, normal = self._adh_query_nearest(
                            i_p, i_b, pos_i, self.particles_ng_reordered[i_p, i_b].boundary_group
                        )
                        if ok and constraint <= self.particle_radius:
                            mass_i = self.particles_info_reordered[i_p, i_b].mass
                            denominator = self._wall_adhesion_compliance / self._st_default_mass + 1.0 / mass_i
                            if denominator > gs.EPS:
                                self.particles_reordered[i_p, i_b].dpos += (
                                    -constraint
                                    / denominator
                                    / mass_i
                                    * normal
                                    * self.particles_info_reordered[i_p, i_b].density_relaxation
                                )

            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if (
                    self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID
                    and self.particles_reordered[i_p, i_b].free
                ):
                    pos_i = self.particles_reordered[i_p, i_b].pos + self.particles_reordered[i_p, i_b].dpos
                    # Container walls constrain intermediate density corrections to prevent thin-wall crossings.
                    if qd.static(
                        self._has_boundary_cylinder
                        or self._has_boundary_pitcher
                        or self._has_boundary_cup_shell
                        or self._has_boundary_plane
                    ):
                        pos_c = pos_i
                        if qd.static(self._has_boundary_pitcher):
                            gid = self.particles_ng_reordered[i_p, i_b].boundary_group
                            if gid == 1:
                                pos_c = self.boundary2.impose_pos(pos_c)
                            elif gid == 0:
                                pos_c = self.boundary.impose_pos(pos_c)
                            if qd.static(self._is_pitcher_shell):
                                if gid == 0:
                                    pos_c = self.boundary2.impose_pos(pos_c)
                        else:
                            pos_c = self.boundary.impose_pos(pos_c)
                        if qd.static(self._has_boundary_cup_shell):
                            pos_c = self.boundary_cup.impose_pos(pos_c)
                        if qd.static(self._has_boundary_plane):
                            pos_c = self.boundary3.impose_pos(pos_c)
                        if (pos_c - pos_i).norm() <= self.particle_radius:
                            pos_i = pos_c
                    self.particles_reordered[i_p, i_b].pos = pos_i
                    self.particles_reordered[i_p, i_b].dpos.fill(0)

    @qd.kernel
    def _kernel_solve_viscosity(self, f: qd.i32):
        for _ in qd.static(range(self._max_viscosity_solver_iterations)):
            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID:
                    pos_i = self.particles_reordered[i_p, i_b].pos
                    base = self.sh.pos_to_grid(pos_i)
                    xsph_sum = qd.Vector.zero(gs.qd_float, 3)
                    omega_sum = qd.Vector.zero(gs.qd_float, 3)
                    # -For Gradient Approx.-
                    dx_sum = qd.Vector.zero(gs.qd_float, 3)
                    dy_sum = qd.Vector.zero(gs.qd_float, 3)
                    dz_sum = qd.Vector.zero(gs.qd_float, 3)
                    n_dx_sum = qd.Vector.zero(gs.qd_float, 3)
                    n_dy_sum = qd.Vector.zero(gs.qd_float, 3)
                    n_dz_sum = qd.Vector.zero(gs.qd_float, 3)
                    dx = qd.Vector([self.g_del, 0.0, 0.0], dt=gs.qd_float)
                    dy = qd.Vector([0.0, self.g_del, 0.0], dt=gs.qd_float)
                    dz = qd.Vector([0.0, 0.0, self.g_del], dt=gs.qd_float)

                    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        slot_idx = self.sh.grid_to_slot(base + offset)
                        for j in range(
                            self.sh.slot_start[slot_idx, i_b],
                            self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                        ):
                            # boundary balls are excluded on both sides: their zero displacement
                            # would act as wall friction, and XSPH is a fluid-fluid smoother
                            if not self.particles_ng_reordered[j, i_b].is_boundary:
                                pos_j = self.particles_reordered[j, i_b].pos
                                v_ij = (
                                    self.particles_reordered[j, i_b].pos - self.particles_reordered[j, i_b].ipos
                                ) - (self.particles_reordered[i_p, i_b].pos - self.particles_reordered[i_p, i_b].ipos)

                                dist = pos_i - pos_j
                                # ---Vorticity---
                                omega_sum += v_ij.cross(self.spiky(dist))
                                # -Gradient Approx.-
                                dx_sum += v_ij.cross(self.spiky(dist + dx))
                                dy_sum += v_ij.cross(self.spiky(dist + dy))
                                dz_sum += v_ij.cross(self.spiky(dist + dz))
                                n_dx_sum += v_ij.cross(self.spiky(dist - dx))
                                n_dy_sum += v_ij.cross(self.spiky(dist - dy))
                                n_dz_sum += v_ij.cross(self.spiky(dist - dz))
                                # ---Viscosity---
                                poly = self.poly6(dist)
                                xsph_sum += poly * v_ij

                    # # ---Vorticity---
                    # n_x = (dx_sum.norm() - n_dx_sum.norm()) / (2 * self.g_del)
                    # n_y = (dy_sum.norm() - n_dy_sum.norm()) / (2 * self.g_del)
                    # n_z = (dz_sum.norm() - n_dz_sum.norm()) / (2 * self.g_del)
                    # n = qd.Vector([n_x, n_y, n_z])
                    # big_n = n.normalized()
                    # if not omega_sum.norm() == 0.0:
                    #     vorticity[p] = vorqd_epsilon * big_n.cross(omega_sum)

                    # ---Viscosity---
                    self.particles_reordered[i_p, i_b].dpos = (
                        self.particles_reordered[i_p, i_b].dpos
                        + xsph_sum * self.particles_info_reordered[i_p, i_b].viscosity_relaxation
                    )

            for i_p, i_b in qd.ndrange(self._n_particles, self._B):
                if (
                    self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.LIQUID
                    and self.particles_reordered[i_p, i_b].free
                ):
                    self.particles_reordered[i_p, i_b].pos = (
                        self.particles_reordered[i_p, i_b].pos + self.particles_reordered[i_p, i_b].dpos
                    )
                    self.particles_reordered[i_p, i_b].dpos.fill(0)

    @qd.kernel
    def _kernel_compute_velocity(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if not (
                self.particles_info_reordered[i_p, i_b].material_type == self.MATERIAL.SOLID
                and self.particles_reordered[i_p, i_b].free
                and not self.particles_ng_reordered[i_p, i_b].is_boundary
            ):
                self.particles_reordered[i_p, i_b].vel = (
                    self.particles_reordered[i_p, i_b].pos - self.particles_reordered[i_p, i_b].ipos
                ) / self._substep_dt

    @qd.kernel
    def _kernel_copy_from_reordered(self, f: qd.i32):
        if qd.static(self._n_clusters > 0):
            for i_b in range(self._B):
                self.is_render_dirty[i_b] = True
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                reordered_idx = self.particles_ng[i_p, i_b].reordered_idx
                self.particles[i_p, i_b] = self.particles_reordered[reordered_idx, i_b]
                func_validate_pbd_particle(
                    i_b, self.particles[i_p, i_b].pos, self.particles[i_p, i_b].vel, self._errno
                )

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        pass

    def check_errno(self):
        errno = np.bitwise_or.reduce(qd_to_numpy(self._errno, transpose=True))
        if errno & ErrorCode.INVALID_PBD_VELOCITY_PROJECTION:
            gs.raise_exception("PBD rigid velocity projection has invalid inertia or momentum. Simulation halted.")
        if errno & ErrorCode.INVALID_PBD_COHESIVE_REACTION:
            gs.raise_exception("PBD cohesive reaction graph has invalid factors, residual or balance. Simulation halted.")
        if errno & ErrorCode.INVALID_PBD_STATE:
            gs.raise_exception("PBD particle position or velocity is non-finite. Simulation halted.")

    def substep_pre_coupling(self, f):
        if self.is_active:
            preserve_boundary_ipos = 0
            if self._is_boundary_pose_pending:
                kernel_advance_boundary_ball_pose(self._boundary_pose_substep, self._sim._substeps, self)
                preserve_boundary_ipos = 1
                self._boundary_pose_substep += 1
                if self._boundary_pose_substep >= self._sim._substeps:
                    self._is_boundary_pose_pending = False
            self._kernel_store_initial_pos(f, preserve_boundary_ipos)
            self._kernel_apply_external_force(f, self._sim.cur_t)

            # topology constraints (doesn't require spatial hashing)
            if self._n_edges > 0:
                self._kernel_solve_stretch(f)

            if self._n_inner_edges > 0:
                self._kernel_solve_bending(f)

            if self._n_elems > 0:
                self._kernel_solve_volume(f)

            if self._n_bonds > 0:
                self._kernel_fracture_bonds(f)

            if self._has_plasticity:
                self._kernel_update_plasticity(f)

            if self._n_clusters > 0:
                self._kernel_solve_shape_matching(f)

            # perform spatial hashing
            self._kernel_reorder_particles(f)

            if self._is_st_enabled:
                if self._st_topo_counter % self._st_topo_interval == 0:
                    self._st_rebuild_topology(f)
                self._st_topo_counter += 1

            # spatial constraints
            if self._n_fluid_particles > 0:
                self._kernel_solve_density(f)
                self._kernel_solve_viscosity(f)

            self._kernel_solve_collision(f)

            # compute effective velocity
            self._kernel_compute_velocity(f)

            if self._wall_friction > 0.0:
                kernel_apply_wall_friction(f, self)

            if qd.static(self._has_boundary_plane) and self._plane_friction > 0.0:
                kernel_apply_plane_friction(f, self)

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if self.is_active:
            self._kernel_copy_from_reordered(f)

            # boundary collision
            self._kernel_solve_boundary_collision(f)

            if self._velocity_damping < 1.0:
                kernel_apply_velocity_damping(f, self)
            if self._diffusion_coeff > 0.0:
                self._kernel_reorder_particles(f)
                kernel_solve_diffusion(f, self)
                self._kernel_copy_from_reordered(f)

    def substep_post_coupling_grad(self, f):
        pass

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        pass

    def add_grad_from_state(self, state):
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(
                f,
                self.scene._sanitize_envs_idx(envs_idx),
                state.pos,
                state.vel,
                state.free,
                state.c,
                state.boundary_group,
                state.active,
                state.solid_rest,
                state.cluster_rest_cm,
                state.bonds_alive,
                state.seams_alive_count,
                state.seams_damage,
                state.seams_is_dead,
            )
            if self._n_clusters > 0:
                # the restored trajectory's last fit is meaningless for the new positions; the bond
                # fracture pass reads the cluster rotations before shape matching refits them, so a
                # stale fit would measure a phantom relative rotation and kill restored seams
                self._kernel_refit_clusters()

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        envs_idx: qd.types.ndarray(),
        pos: qd.types.ndarray(),  # shape [B, _n_entity_particles, 3]
        vel: qd.types.ndarray(),  # shape [B, _n_entity_particles, 3]
        free: qd.types.ndarray(),  # shape [B, _n_entity_particles]
        c: qd.types.ndarray(),
        boundary_group: qd.types.ndarray(),
        active: qd.types.ndarray(),
        solid_rest: qd.types.ndarray(),  # shape [B, _n_entity_particles, 3]
        cluster_rest_cm: qd.types.ndarray(),  # shape [B, _n_clusters, 3]
        bonds_alive: qd.types.ndarray(),  # shape [B, _n_bonds]
        seams_alive_count: qd.types.ndarray(),  # shape [B, _n_seams]
        seams_damage: qd.types.ndarray(),  # shape [B, _n_seams]
        seams_is_dead: qd.types.ndarray(),  # shape [B, _n_seams]
    ):
        # Boundary balls are solver-owned geometry, never restored through a Scene state.
        for i_b_local in range(envs_idx.shape[0]):
            self._errno[envs_idx[i_b_local]] = 0
            if qd.static(self._n_clusters > 0):
                self.is_render_dirty[envs_idx[i_b_local]] = True
            if qd.static(self._n_bonds > 0):
                self.cluster_roots_dirty[envs_idx[i_b_local]] = True
                self.cohesive.rebuild[envs_idx[i_b_local]] = True
        for i_p, i_b_local in qd.ndrange(self._n_entity_particles, envs_idx.shape[0]):
            i_b = envs_idx[i_b_local]
            for j in qd.static(range(3)):
                self.particles[i_p, i_b].pos[j] = pos[i_b, i_p, j]
                self.particles[i_p, i_b].vel[j] = vel[i_b, i_p, j]
                self.particles_solid_rest[i_p, i_b][j] = solid_rest[i_b, i_p, j]
            self.particles[i_p, i_b].free = free[i_b, i_p]
            self.particles[i_p, i_b].c = c[i_b, i_p]
            self.particles_ng[i_p, i_b].boundary_group = boundary_group[i_b, i_p]
            self.particles_ng[i_p, i_b].active = active[i_b, i_p]
            if self.particles_ng[i_p, i_b].active:
                func_validate_pbd_particle(
                    i_b, self.particles[i_p, i_b].pos, self.particles[i_p, i_b].vel, self._errno
                )

        for i_c, i_b_local in qd.ndrange(self._n_clusters, envs_idx.shape[0]):
            i_b = envs_idx[i_b_local]
            for j in qd.static(range(3)):
                self.clusters_rest[i_c, i_b].cm[j] = cluster_rest_cm[i_b, i_c, j]

        for i_l, i_b_local in qd.ndrange(self._n_bonds, envs_idx.shape[0]):
            i_b = envs_idx[i_b_local]
            self.bonds_state[i_l, i_b].alive = bonds_alive[i_b, i_l]

        for i_s, i_b_local in qd.ndrange(self._n_seams, envs_idx.shape[0]):
            i_b = envs_idx[i_b_local]
            self.seams_state[i_s, i_b].alive_count = seams_alive_count[i_b, i_s]
            self.seams_state[i_s, i_b].damage = seams_damage[i_b, i_s]
            self.seams_state[i_s, i_b].is_dead = seams_is_dead[i_b, i_s]

    def get_state(self, f):
        if self.is_active:
            state = PBDFluidSolverState(self.scene)
            self._kernel_get_state(
                f,
                state.pos,
                state.vel,
                state.free,
                state.c,
                state.boundary_group,
                state.active,
                state.solid_rest,
                state.cluster_rest_cm,
                state.bonds_alive,
                state.seams_alive_count,
                state.seams_damage,
                state.seams_is_dead,
            )
        else:
            state = None
        return state

    def get_state_render(self):
        """
        Get visual vertex positions, UVs, and face indices for rendering.

        Returns
        -------
        tuple
            (vverts_pos, vverts_uvs, vfaces_indices) - vertex positions, UV coords, and triangle indices
        """
        if not self.is_active or self._n_vverts == 0:
            return None, None, None

        # Make sure render fields are up to date
        self.update_render_fields()

        # Return the Quadrants fields directly for GPU access
        return self.vverts_render.pos, self.vverts_uvs, self.vfaces_indices

    @qd.kernel
    def _kernel_get_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),  # shape [B, _n_entity_particles, 3]
        vel: qd.types.ndarray(),  # shape [B, _n_entity_particles, 3]
        free: qd.types.ndarray(),  # shape [B, _n_entity_particles]
        c: qd.types.ndarray(),
        boundary_group: qd.types.ndarray(),
        active: qd.types.ndarray(),
        solid_rest: qd.types.ndarray(),  # shape [B, _n_entity_particles, 3]
        cluster_rest_cm: qd.types.ndarray(),  # shape [B, _n_clusters, 3]
        bonds_alive: qd.types.ndarray(),  # shape [B, _n_bonds]
        seams_alive_count: qd.types.ndarray(),  # shape [B, _n_seams]
        seams_damage: qd.types.ndarray(),  # shape [B, _n_seams]
        seams_is_dead: qd.types.ndarray(),  # shape [B, _n_seams]
    ):
        for i_p, i_b in qd.ndrange(self._n_entity_particles, self._B):
            for j in qd.static(range(3)):
                pos[i_b, i_p, j] = self.particles[i_p, i_b].pos[j]
                vel[i_b, i_p, j] = self.particles[i_p, i_b].vel[j]
                solid_rest[i_b, i_p, j] = self.particles_solid_rest[i_p, i_b][j]
            free[i_b, i_p] = qd.cast(self.particles[i_p, i_b].free, gs.qd_bool)
            c[i_b, i_p] = self.particles[i_p, i_b].c
            boundary_group[i_b, i_p] = self.particles_ng[i_p, i_b].boundary_group
            active[i_b, i_p] = self.particles_ng[i_p, i_b].active

        for i_c, i_b in qd.ndrange(self._n_clusters, self._B):
            for j in qd.static(range(3)):
                cluster_rest_cm[i_b, i_c, j] = self.clusters_rest[i_c, i_b].cm[j]

        for i_l, i_b in qd.ndrange(self._n_bonds, self._B):
            bonds_alive[i_b, i_l] = self.bonds_state[i_l, i_b].alive

        for i_s, i_b in qd.ndrange(self._n_seams, self._B):
            seams_alive_count[i_b, i_s] = self.seams_state[i_s, i_b].alive_count
            seams_damage[i_b, i_s] = self.seams_state[i_s, i_b].damage
            seams_is_dead[i_b, i_s] = self.seams_state[i_s, i_b].is_dead

    def update_render_fields(self):
        # the cached cluster transforms lag the solver state: set_particles_pos, set_state, and the
        # corrections after the last shape-matching iteration all move particles without refitting
        if self._n_clusters > 0:
            self._kernel_refit_clusters()
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @qd.kernel
    def _kernel_refit_clusters(self):
        func_fit_pbd_clusters(self, welded=True, render_only=True)

    @qd.kernel
    def _kernel_update_render_fields(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            # boundary balls stay invisible: they are a numerical wall, not rendered geometry
            if self.particles_ng[i_p, i_b].active and not self.particles_ng[i_p, i_b].is_boundary:
                self.particles_render[i_p, i_b].pos = self.particles[i_p, i_b].pos
                self.particles_render[i_p, i_b].vel = self.particles[i_p, i_b].vel
                self.particles_render[i_p, i_b].c = self.particles[i_p, i_b].c
            else:
                self.particles_render[i_p, i_b].pos = gu.qd_nowhere()
            self.particles_render[i_p, i_b].active = (
                self.particles_ng[i_p, i_b].active and not self.particles_ng[i_p, i_b].is_boundary
            )

        for i_v, i_b in qd.ndrange(self._n_vverts, self._B):
            vvert_pos = qd.Vector.zero(gs.qd_float, 3)
            if self.vverts_info[i_v].is_solid:
                vvert_pos = self.vverts_render[i_v, i_b].pos
                if qd.static(self._n_clusters > 0):
                    if self.is_render_dirty[i_b]:
                        i_c = self.vverts_solid_cluster[i_v]
                        seam = self.vverts_solid_seam[i_v]
                        is_exposed = seam < 0
                        if seam >= 0:
                            if self.seams_state[seam, i_b].is_dead:
                                i_c_i = self.seams_info[seam].i_c
                                i_c_j = self.seams_info[seam].j_c
                                is_exposed = i_c_i != i_c_j
                                if qd.static(self._n_bonds > 0):
                                    is_exposed = self.cluster_roots[i_c_i, i_b] != self.cluster_roots[i_c_j, i_b]
                        if is_exposed:
                            # Surface faces follow their cluster; separated dead interfaces reveal both walls.
                            vvert_pos = (
                                self.clusters_state[i_c, i_b].rot @ self.vverts_solid_offset[i_v]
                                + self.clusters_state[i_c, i_b].cm
                            )
                        else:
                            # Unexposed interface: collapse onto the centroid, hiding the zero-area face.
                            vvert_pos = self.clusters_state[i_c, i_b].cm
            else:
                # support-weight skinning
                for j in range(self._n_vvert_supports):
                    vvert_pos += (
                        self.particles[self.vverts_info.support_idxs[i_v][j], i_b].pos
                        * self.vverts_info.support_weights[i_v][j]
                    )
            self.vverts_render[i_v, i_b].pos = vvert_pos
            self.vverts_render[i_v, i_b].active = self.particles_render[
                self.vverts_info.support_idxs[i_v][0], i_b
            ].active

        if qd.static(self._n_clusters > 0):
            for i_b in range(self._B):
                self.is_render_dirty[i_b] = False

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    @qd.kernel
    def _kernel_set_particles_pos(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        poss: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            if qd.static(self._n_clusters > 0):
                self.is_render_dirty[i_b] = True
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].pos[i] = poss[i_b_, i_p_, i]
            self.particles[i_p, i_b].vel.fill(0.0)
            func_validate_pbd_particle(i_b, self.particles[i_p, i_b].pos, self.particles[i_p, i_b].vel, self._errno)

    @qd.kernel
    def _kernel_get_particles_pos(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        poss: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                poss[i_b_, i_p_, i] = self.particles[i_p, i_b].pos[i]

    @mutates(StateChange.DYNAMICS)
    @qd.kernel
    def _kernel_set_particles_vel(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].vel[i] = vels[i_b_, i_p_, i]
            func_validate_pbd_particle(i_b, self.particles[i_p, i_b].pos, self.particles[i_p, i_b].vel, self._errno)

    @gs.assert_built
    def set_animate_particles_by_link(
        self,
        particles_idx,
        link_idx: int,
        links_state: LinksState,
        envs_idx=None,
    ) -> None:
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        self._sim._coupler.kernel_attach_pbd_to_rigid_link(particles_idx, envs_idx, link_idx, links_state)

    @qd.kernel
    def _kernel_get_particles_vel(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                vels[i_b_, i_p_, i] = self.particles[i_p, i_b].vel[i]

    @mutates(StateChange.GEOMETRY)
    @qd.kernel
    def _kernel_set_particles_active(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles_ng[i_p, i_b].active = qd.cast(actives[i_b_, i_p_], gs.qd_bool)

    @qd.kernel
    def _kernel_get_particles_active(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles_ng[i_p, i_b].active

    @qd.kernel
    def _kernel_fix_particles(self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray()):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles[i_p, i_b].free = False

    @qd.kernel
    def _kernel_release_particle(self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray()):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles[i_p, i_b].free = True

    @qd.kernel
    def _kernel_get_mass(
        self, particle_start: qd.i32, n_particles: qd.i32, mass: qd.types.ndarray(), envs_idx: qd.types.ndarray()
    ):
        total_mass = gs.qd_float(0.0)
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            total_mass += self.particles_info[i_p].mass
        for i_b_ in range(envs_idx.shape[0]):
            mass[i_b_] = total_mass

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        return sum([entity.n_particles for entity in self._entities])

    @property
    def n_entity_particles(self):
        """Number of entity-owned particles represented by Scene state snapshots."""
        if self._n_entity_particles is not None:
            return self._n_entity_particles
        return sum(entity.n_particles for entity in self._entities)

    @property
    def n_fluid_particles(self):
        if self.is_built:
            return self._n_fluid_particles
        return sum(entity.n_fluid_particles for entity in self._entities if isinstance(entity, PBDParticleEntity))

    @property
    def n_clusters(self):
        if self.is_built:
            return self._n_clusters
        return sum(entity.n_clusters for entity in self._entities if isinstance(entity, PBD3DSolidEntity))

    @property
    def n_bonds(self):
        if self.is_built:
            return self._n_bonds
        return sum(entity.n_bonds for entity in self._entities if isinstance(entity, PBD3DSolidEntity))

    @property
    def n_seams(self):
        if self.is_built:
            return self._n_seams
        return sum(entity.n_seams for entity in self._entities if isinstance(entity, PBD3DSolidEntity))

    @property
    def n_fragments(self):
        if self.is_built:
            return self._n_fragments
        return sum(entity.n_fragments for entity in self._entities if isinstance(entity, PBD3DSolidEntity))

    @property
    def n_cluster_members(self):
        if self.is_built:
            return self._n_cluster_members
        return sum(entity.n_cluster_members for entity in self._entities if isinstance(entity, PBD3DSolidEntity))

    @property
    def n_edges(self):
        if self.is_built:
            return self._n_edges
        return sum(entity.n_edges for entity in self._entities if isinstance(entity, PBDTetEntity))

    @property
    def n_inner_edges(self):
        if self.is_built:
            return self._n_inner_edges
        return sum(entity.n_inner_edges for entity in self._entities if isinstance(entity, PBD2DEntity))

    @property
    def n_elems(self):
        if self.is_built:
            return self._n_elems
        return sum(entity.n_elems for entity in self._entities if isinstance(entity, PBD3DEntity))

    @property
    def n_vverts(self):
        if self.is_built:
            return self._n_vverts
        return sum(entity.n_vverts for entity in self._entities)

    @property
    def n_vfaces(self):
        if self.is_built:
            return self._n_vfaces
        return sum(entity.n_vfaces for entity in self._entities)

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def hash_grid_res(self):
        return self.sh.grid_res

    @property
    def hash_grid_cell_size(self):
        return self.sh.cell_size

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound


@qd.kernel
def kernel_add_boundary_balls(
    particle_start: int,
    n_balls: int,
    mat_rho: float,
    pos: qd.types.ndarray(),
    mass_weight: qd.types.ndarray(),
    solver: V_ANNOTATION,
):
    for i_p_, i_b in qd.ndrange(n_balls, solver._B):
        i_p = i_p_ + particle_start
        for i in qd.static(range(3)):
            solver.particles_info[i_p].pos_rest[i] = pos[i_p_, i]
            solver.particles[i_p, i_b].pos[i] = pos[i_p_, i]
            solver._ball_pos_local[i_p_][i] = pos[i_p_, i]
        solver.particles_info[i_p].material_type = solver.MATERIAL.PARTICLE
        solver.particles_info[i_p].mass = mat_rho * mass_weight[i_p_]
        solver.particles_info[i_p].rho_rest = mat_rho
        solver.particles[i_p, i_b].free = False
        solver.particles_ng[i_p, i_b].active = True
        solver.particles_ng[i_p, i_b].boundary_group = 0
        solver.particles_ng[i_p, i_b].is_boundary = True
        set_id = gs.qd_int(-1)
        for i_set in range(solver._n_ball_sets):
            if (
                i_p >= solver._ball_set_start[i_set]
                and i_p < solver._ball_set_start[i_set] + solver._ball_set_size[i_set]
            ):
                set_id = i_set
        solver.particles_ng[i_p, i_b].boundary_set = set_id


@qd.kernel
def kernel_set_boundary_ball_pose(i_set: int, pos: qd.types.ndarray(), quat: qd.types.ndarray(), solver: V_ANNOTATION):
    for i in qd.static(range(3)):
        solver._ball_set_pos[i_set][i] = pos[i]
    for i in qd.static(range(4)):
        solver._ball_set_quat[i_set][i] = quat[i]


@qd.kernel
def kernel_initialize_boundary_ball_pos(solver: V_ANNOTATION):
    for i_set in range(solver._n_ball_sets):
        start = solver._ball_set_start[i_set]
        local_start = start - solver._n_entity_particles
        trans = solver._ball_set_pos[i_set]
        quat = solver._ball_set_quat[i_set]
        for i_p_, i_b in qd.ndrange(solver._ball_set_size[i_set], solver._B):
            i_p = start + i_p_
            i_local = local_start + i_p_
            new_pos = gu.qd_transform_by_trans_quat_fast(solver._ball_pos_local[i_local], trans, quat)
            solver.particles[i_p, i_b].pos = new_pos
            solver.particles[i_p, i_b].ipos = new_pos
            solver._ball_motion_start[i_local, i_b] = new_pos
            solver._ball_motion_target[i_local, i_b] = new_pos


@qd.kernel
def kernel_stage_boundary_ball_pose(i_set: int, solver: V_ANNOTATION):
    start = solver._ball_set_start[i_set]
    local_start = start - solver._n_entity_particles
    trans = solver._ball_set_pos[i_set]
    quat = solver._ball_set_quat[i_set]
    for i_p_, i_b in qd.ndrange(solver._ball_set_size[i_set], solver._B):
        i_p = start + i_p_
        i_local = local_start + i_p_
        solver._ball_motion_start[i_local, i_b] = solver.particles[i_p, i_b].pos
        new_pos = gu.qd_transform_by_trans_quat_fast(solver._ball_pos_local[i_local], trans, quat)
        solver._ball_motion_target[i_local, i_b] = new_pos
        solver.particles[i_p, i_b].pos = new_pos


@qd.kernel
def kernel_advance_boundary_ball_pose(i_substep: int, n_substeps: int, solver: V_ANNOTATION):
    a0 = qd.cast(i_substep, gs.qd_float) / qd.cast(n_substeps, gs.qd_float)
    a1 = qd.cast(i_substep + 1, gs.qd_float) / qd.cast(n_substeps, gs.qd_float)
    for i_local, i_b in qd.ndrange(solver.n_boundary_balls, solver._B):
        i_p = solver._n_entity_particles + i_local
        p0 = solver._ball_motion_start[i_local, i_b]
        dp = solver._ball_motion_target[i_local, i_b] - p0
        solver.particles[i_p, i_b].ipos = p0 + a0 * dp
        solver.particles[i_p, i_b].pos = p0 + a1 * dp
        if i_substep + 1 >= n_substeps:
            solver._ball_motion_start[i_local, i_b] = solver._ball_motion_target[i_local, i_b]


@qd.kernel
def kernel_st_mark_surface_screen(solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if (
            solver.particles_ng_reordered[i_p, i_b].active
            and solver.particles_info_reordered[i_p, i_b].material_type == solver.MATERIAL.LIQUID
        ):
            pos_i = solver.particles_reordered[i_p, i_b].pos
            base = solver.sh.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-3, 4), (-3, 4), (-3, 4))):
                slot_idx = solver.sh.grid_to_slot(base + offset)
                for j in range(
                    solver.sh.slot_start[slot_idx, i_b],
                    solver.sh.slot_size[slot_idx, i_b] + solver.sh.slot_start[slot_idx, i_b],
                ):
                    if j != i_p and solver.particles_info_reordered[j, i_b].material_type == solver.MATERIAL.LIQUID:
                        delta = solver.particles_reordered[j, i_b].pos - pos_i
                        distance = delta.norm()
                        if distance > gs.EPS and distance < solver._st_ring_radius:
                            unit_theta = math.pi / solver._N_THETA
                            unit_phi = 2.0 * math.pi / solver._N_PHI
                            block_radius = qd.min(0.5 * solver._particle_size, 0.5 * distance)
                            delta_angle = qd.asin(qd.min(block_radius / distance, 1.0))
                            theta = qd.acos(qd.max(-1.0, qd.min(1.0, delta[1] / distance)))
                            phi = qd.atan2(delta[2], delta[0])
                            start_theta = qd.max(theta - delta_angle, 0.0)
                            end_theta = qd.min(theta + delta_angle, math.pi)
                            start_phi = phi - delta_angle
                            end_phi = phi + delta_angle
                            if start_phi < -math.pi:
                                start_phi += 2.0 * math.pi
                            if end_phi > math.pi:
                                end_phi -= 2.0 * math.pi
                            st_t = qd.min(qd.cast(qd.floor(start_theta / unit_theta), gs.qd_int), solver._N_THETA - 1)
                            en_t = qd.min(qd.cast(qd.ceil(end_theta / unit_theta), gs.qd_int), solver._N_THETA)
                            st_p = qd.min(
                                qd.cast(qd.floor((start_phi + math.pi) / unit_phi), gs.qd_int), solver._N_PHI - 1
                            )
                            en_p = qd.min(qd.cast(qd.ceil((end_phi + math.pi) / unit_phi), gs.qd_int), solver._N_PHI)
                            if st_p < en_p:
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, st_t, st_p], 1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, st_t, en_p], -1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, en_t, st_p], -1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, en_t, en_p], 1)
                            else:
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, st_t, st_p], 1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, st_t, solver._N_PHI], -1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, en_t, st_p], -1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, en_t, solver._N_PHI], 1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, st_t, 0], 1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, st_t, en_p], -1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, en_t, 0], -1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, en_t, en_p], 1)


@qd.kernel
def kernel_st_classify_surface(solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if (
            solver.particles_ng_reordered[i_p, i_b].active
            and solver.particles_info_reordered[i_p, i_b].material_type == solver.MATERIAL.LIQUID
        ):
            illuminated = gs.qd_float(0.0)
            total = gs.qd_float(0.0)
            for t in range(solver._N_THETA):
                weight = qd.sin(math.pi / solver._N_THETA * (t + 0.5))
                for p in range(solver._N_PHI):
                    if t > 0 and p > 0:
                        solver._screen_blocked[i_p, i_b, t, p] += (
                            solver._screen_blocked[i_p, i_b, t - 1, p]
                            + solver._screen_blocked[i_p, i_b, t, p - 1]
                            - solver._screen_blocked[i_p, i_b, t - 1, p - 1]
                        )
                    elif t > 0:
                        solver._screen_blocked[i_p, i_b, t, p] += solver._screen_blocked[i_p, i_b, t - 1, p]
                    elif p > 0:
                        solver._screen_blocked[i_p, i_b, t, p] += solver._screen_blocked[i_p, i_b, t, p - 1]
                    if solver._screen_blocked[i_p, i_b, t, p] == 0:
                        illuminated += weight
                    total += weight
            solver.on_surface[i_p, i_b] = illuminated >= solver._st_lit_threshold * total
        else:
            solver.on_surface[i_p, i_b] = False


@qd.kernel
def kernel_st_compute_normals(solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if (
            solver.particles_ng_reordered[i_p, i_b].active
            and solver.particles_info_reordered[i_p, i_b].material_type == solver.MATERIAL.LIQUID
            and solver.on_surface[i_p, i_b]
        ):
            pos_i = solver.particles_reordered[i_p, i_b].pos
            n_raw = qd.Vector.zero(gs.qd_float, 3)
            base = solver.sh.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-3, 4), (-3, 4), (-3, 4))):
                slot_idx = solver.sh.grid_to_slot(base + offset)
                for j in range(
                    solver.sh.slot_start[slot_idx, i_b],
                    solver.sh.slot_size[slot_idx, i_b] + solver.sh.slot_start[slot_idx, i_b],
                ):
                    if j != i_p and solver.particles_info_reordered[j, i_b].material_type == solver.MATERIAL.LIQUID:
                        delta = solver.particles_reordered[j, i_b].pos - pos_i
                        distance = delta.norm()
                        if distance > gs.EPS and distance < solver._st_ring_radius:
                            n_raw += solver._st_cubic_dW(distance) * (delta / distance) * solver._particle_volume
            solver.normal[i_p, i_b] = n_raw
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if (
            solver.particles_ng_reordered[i_p, i_b].active
            and solver.particles_info_reordered[i_p, i_b].material_type == solver.MATERIAL.LIQUID
            and solver.on_surface[i_p, i_b]
        ):
            raw_normal = solver.normal[i_p, i_b]
            raw_length = raw_normal.norm()
            if raw_length <= 1.0:
                solver.on_surface[i_p, i_b] = False
                solver.normal[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
            else:
                solver.normal[i_p, i_b] = raw_normal / raw_length


@qd.kernel
def kernel_st_build_local_meshes(solver: V_ANNOTATION):
    solver.topology_valid.fill(False)
    solver.n_ring.fill(0)
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if (
            solver.particles_ng_reordered[i_p, i_b].active
            and solver.particles_info_reordered[i_p, i_b].material_type == solver.MATERIAL.LIQUID
            and solver.on_surface[i_p, i_b]
        ):
            pos_i = solver.particles_reordered[i_p, i_b].pos
            normal = solver.normal[i_p, i_b]
            axis_x = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
            if axis_x.cross(normal).norm() < gs.EPS:
                axis_x = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
            axis_x = axis_x.cross(normal).normalized()
            axis_y = normal.cross(axis_x).normalized()
            count = gs.qd_int(0)
            cand_overflow = False
            base = solver.sh.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-3, 4), (-3, 4), (-3, 4))):
                slot_idx = solver.sh.grid_to_slot(base + offset)
                for j in range(
                    solver.sh.slot_start[slot_idx, i_b],
                    solver.sh.slot_size[slot_idx, i_b] + solver.sh.slot_start[slot_idx, i_b],
                ):
                    if j != i_p and solver.on_surface[j, i_b]:
                        delta = solver.particles_reordered[j, i_b].pos - pos_i
                        distance = delta.norm()
                        if distance < solver._st_ring_radius:
                            normal_j = solver.normal[j, i_b]
                            if normal.dot(normal_j) > solver._st_normal_compat or (
                                (normal - normal_j).dot(delta) > 0.0 and distance < 2.0 * solver._particle_size
                            ):
                                if count < solver._st_max_surface_neighbors:
                                    solver.neighbor_ids[i_p, i_b, count] = j
                                    projected = delta - delta.dot(normal) * normal
                                    solver.projected_positions[i_p, i_b, count] = qd.Vector(
                                        [projected.dot(axis_x), projected.dot(axis_y)], dt=gs.qd_float
                                    )
                                    count += 1
                                else:
                                    cand_overflow = True
            if cand_overflow:
                qd.atomic_max(solver._overflow[None], solver._ST_SURFACE_NEIGHBOR_OVERFLOW)
            n = count
            solver.n_ring[i_p, i_b] = n
            for k in range(n):
                solver._node_queue[i_p, i_b, k] = k
                solver._chain_pre[i_p, i_b, k] = -1
                solver._chain_nxt[i_p, i_b, k] = -1
            for k in range(1, n):
                cursor = k
                while cursor > 0:
                    u = solver._node_queue[i_p, i_b, cursor - 1]
                    v = solver._node_queue[i_p, i_b, cursor]
                    x_u = solver.projected_positions[i_p, i_b, u]
                    x_v = solver.projected_positions[i_p, i_b, v]
                    angle_u = qd.atan2(x_u[1], x_u[0])
                    angle_v = qd.atan2(x_v[1], x_v[0])
                    if angle_u > angle_v or (angle_u == angle_v and x_u.norm() > x_v.norm()):
                        solver._node_queue[i_p, i_b, cursor - 1] = v
                        solver._node_queue[i_p, i_b, cursor] = u
                        cursor -= 1
                    else:
                        cursor = 0
            ring_size = gs.qd_int(0)
            if n >= 3:
                for k in range(n - 1):
                    u = solver._node_queue[i_p, i_b, k]
                    v = solver._node_queue[i_p, i_b, k + 1]
                    solver._chain_nxt[i_p, i_b, u] = v
                    solver._chain_pre[i_p, i_b, v] = u
                u_last = solver._node_queue[i_p, i_b, n - 1]
                u_first = solver._node_queue[i_p, i_b, 0]
                solver._chain_nxt[i_p, i_b, u_last] = u_first
                solver._chain_pre[i_p, i_b, u_first] = u_last
                queue_start = gs.qd_int(0)
                queue_end = gs.qd_int(0)
                for k in range(n):
                    u = solver._node_queue[i_p, i_b, k]
                    if solver._st_need_flip(i_p, i_b, u):
                        if queue_end < 3 * solver._st_max_surface_neighbors:
                            solver._node_queue[i_p, i_b, queue_end] = u
                            queue_end += 1
                        else:
                            qd.atomic_max(solver._overflow[None], solver._ST_LOCAL_MESH_QUEUE_OVERFLOW)
                while queue_start < queue_end:
                    u = solver._node_queue[i_p, i_b, queue_start]
                    queue_start += 1
                    if solver._chain_nxt[i_p, i_b, u] >= 0 and solver._st_need_flip(i_p, i_b, u):
                        u_pre = solver._chain_pre[i_p, i_b, u]
                        u_nxt = solver._chain_nxt[i_p, i_b, u]
                        solver._chain_nxt[i_p, i_b, u_pre] = u_nxt
                        solver._chain_pre[i_p, i_b, u_nxt] = u_pre
                        if solver._st_need_flip(i_p, i_b, u_nxt):
                            if queue_end < 3 * solver._st_max_surface_neighbors:
                                solver._node_queue[i_p, i_b, queue_end] = u_nxt
                                queue_end += 1
                            else:
                                qd.atomic_max(solver._overflow[None], solver._ST_LOCAL_MESH_QUEUE_OVERFLOW)
                        if solver._st_need_flip(i_p, i_b, u_pre):
                            if queue_end < 3 * solver._st_max_surface_neighbors:
                                solver._node_queue[i_p, i_b, queue_end] = u_pre
                                queue_end += 1
                            else:
                                qd.atomic_max(solver._overflow[None], solver._ST_LOCAL_MESH_QUEUE_OVERFLOW)
                        solver._chain_nxt[i_p, i_b, u] = -1
                        solver._chain_pre[i_p, i_b, u] = -1
                start = gs.qd_int(-1)
                for k in range(n):
                    if start < 0 and solver._chain_nxt[i_p, i_b, k] >= 0:
                        start = k
                projected_area_twice = gs.qd_float(0.0)
                if start >= 0:
                    u = start
                    keep_walking = True
                    while keep_walking and ring_size < solver._st_max_localmesh_neighbors:
                        u_next = solver._chain_nxt[i_p, i_b, u]
                        x_u = solver.projected_positions[i_p, i_b, u]
                        x_next = solver.projected_positions[i_p, i_b, u_next]
                        projected_area_twice += x_u[0] * x_next[1] - x_u[1] * x_next[0]
                        solver.local_mesh_neighbors[i_p, i_b, ring_size] = solver.neighbor_ids[i_p, i_b, u]
                        ring_size += 1
                        u = u_next
                        if u == start:
                            keep_walking = False
                    if keep_walking:
                        qd.atomic_max(solver._overflow[None], solver._ST_LOCAL_MESH_NEIGHBOR_OVERFLOW)
                    else:
                        solver.topology_valid[i_p, i_b] = ring_size >= 3 and qd.abs(projected_area_twice) > gs.EPS
                solver.n_ring[i_p, i_b] = ring_size


@qd.kernel
def kernel_apply_velocity_damping(f: int, solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        solver.particles[i_p, i_b].vel = solver.particles[i_p, i_b].vel * solver._velocity_damping


@qd.kernel
def kernel_apply_wall_friction(f: int, solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if (
            solver.particles_ng_reordered[i_p, i_b].active
            and solver.particles_info_reordered[i_p, i_b].material_type == solver.MATERIAL.LIQUID
        ):
            pos_i = solver.particles_reordered[i_p, i_b].pos
            ok, constraint, normal = solver._adh_query_nearest(
                i_p, i_b, pos_i, solver.particles_ng_reordered[i_p, i_b].boundary_group
            )
            if ok and constraint <= solver.particle_radius:
                v = solver.particles_reordered[i_p, i_b].vel
                solver.particles_reordered[i_p, i_b].vel = v - solver._wall_friction * (v - v.dot(normal) * normal)


@qd.kernel
def kernel_apply_plane_friction(f: int, solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if (
            solver.particles_ng_reordered[i_p, i_b].active
            and solver.particles_info_reordered[i_p, i_b].material_type == solver.MATERIAL.LIQUID
        ):
            pos_i = solver.particles_reordered[i_p, i_b].pos
            ok, constraint, normal = solver.boundary3.adhesion_query(pos_i)
            if ok and constraint <= solver.particle_radius:
                v = solver.particles_reordered[i_p, i_b].vel
                solver.particles_reordered[i_p, i_b].vel = v - solver._plane_friction * (v - v.dot(normal) * normal)


@qd.kernel
def kernel_solve_diffusion(f: int, solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if (
            solver.particles_ng_reordered[i_p, i_b].active
            and not solver.particles_ng_reordered[i_p, i_b].is_boundary
            and solver.particles_info_reordered[i_p, i_b].material_type == solver.MATERIAL.LIQUID
        ):
            pos_i = solver.particles_reordered[i_p, i_b].pos
            base = solver.sh.pos_to_grid(pos_i)
            dc_sum = gs.qd_float(0.0)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = solver.sh.grid_to_slot(base + offset)
                for j in range(
                    solver.sh.slot_start[slot_idx, i_b],
                    solver.sh.slot_size[slot_idx, i_b] + solver.sh.slot_start[slot_idx, i_b],
                ):
                    if solver.particles_info_reordered[j, i_b].material_type == solver.MATERIAL.LIQUID and (
                        not solver.particles_ng_reordered[j, i_b].is_boundary
                    ):
                        dist = pos_i - solver.particles_reordered[j, i_b].pos
                        dc_sum += solver.poly6(dist) * (
                            solver.particles_reordered[j, i_b].c - solver.particles_reordered[i_p, i_b].c
                        )
            solver.particles_reordered[i_p, i_b].dc = dc_sum * solver._diffusion_coeff
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if (
            solver.particles_info_reordered[i_p, i_b].material_type == solver.MATERIAL.LIQUID
            and solver.particles_ng_reordered[i_p, i_b].active
            and not solver.particles_ng_reordered[i_p, i_b].is_boundary
        ):
            solver.particles_reordered[i_p, i_b].c = (
                solver.particles_reordered[i_p, i_b].c + solver.particles_reordered[i_p, i_b].dc
            )







