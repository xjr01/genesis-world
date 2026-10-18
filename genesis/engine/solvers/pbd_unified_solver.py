import math
from dataclasses import dataclass

import numpy as np
import torch

import igl

import quadrants as qd

import genesis as gs
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.boundaries.rigid_surface import RigidSurface, build_rigid_surface
from genesis.engine.bvh import AABB, RefitBVH, func_refit_bvh, point_aabb_distance_sqr
from genesis.engine.entities.pbd_entity import PBD2DEntity, PBD3DEntity
from genesis.engine.states.solvers import PBDSolverState
from genesis.utils import array_class, geom, sdf
from genesis.utils.array_class import ErrorCode, V_ANNOTATION
from genesis.utils.misc import indices_to_mask, qd_to_numpy, qd_to_torch, tensor_to_array
from genesis.utils.triangle_qd import (
    closest_point_on_triangle,
    edge_edge_contact,
    point_triangle_constraint,
    point_triangle_entry_time,
    ray_aabb_intersection,
    ray_projection,
    ray_triangle_intersection,
    segment_triangle_crossing,
    triangle_crossing_edges,
)

from .base_solver import Solver, StateChange, mutates


@dataclass(frozen=True)
class PBDConstraintResiduals:
    """Signed relative edge lengths, dihedral angles in radians, and relative tetrahedron volumes."""

    stretch: torch.Tensor
    bending: torch.Tensor
    volume: torch.Tensor


@dataclass(frozen=True)
class PBDCollisionDiagnostics:
    """Current transverse triangle intersections and unique edge contacts, with shape [B]; penetration is in meters."""

    triangle_intersections: torch.Tensor
    edge_contacts: torch.Tensor
    unpaired_intersections: torch.Tensor
    edge_penetration: torch.Tensor


@dataclass(frozen=True)
class PBDEdgeContactHistory:
    """Detection positions [substep, iteration, B, particle, 3], edge flags, endpoints, and rigid link poses."""

    positions: torch.Tensor
    is_detected: torch.Tensor
    edges: torch.Tensor
    links_pos: torch.Tensor
    links_quat: torch.Tensor


@dataclass(frozen=True)
class TriangleEdgeAdjacency:
    """Face-to-edge indices and compressed, sorted incident face lists for unique undirected edges."""

    face_edges: qd.Tensor
    edge_offsets: qd.Tensor
    edge_faces: qd.Tensor
    endpoints: torch.Tensor


def build_triangle_edge_adjacency(faces):
    """Build exact-size device adjacency arrays for triangle edges identified by their vertex indices."""
    endpoints = np.sort(faces[:, ((0, 1), (1, 2), (2, 0))].reshape((-1, 2)), axis=1)
    endpoints, edge_indices, counts = np.unique(endpoints, axis=0, return_inverse=True, return_counts=True)
    offsets = np.concatenate((np.zeros(1, dtype=gs.np_int), counts.cumsum()))
    incident_faces = np.argsort(edge_indices, kind="stable") // 3
    adjacency = TriangleEdgeAdjacency(
        qd.field(gs.qd_ivec3, shape=len(faces)),
        qd.field(gs.qd_int, shape=len(offsets)),
        qd.field(gs.qd_int, shape=len(incident_faces)),
        torch.from_numpy(endpoints).to(device=gs.device),
    )
    adjacency.face_edges.from_numpy(edge_indices.reshape((-1, 3)))
    adjacency.edge_offsets.from_numpy(offsets)
    adjacency.edge_faces.from_numpy(incident_faces)
    return adjacency


class PBDUnifiedSolverState(PBDSolverState):
    def __init__(self, scene):
        super().__init__(scene)
        self.active = None


@qd.func
def func_dihedral(p1, p2, p3, p4):
    edge = p2 - p1
    normal1 = edge.cross(p3 - p1)
    normal2 = edge.cross(p4 - p1)
    length = edge.norm()
    normal1_sqr = normal1.norm_sqr()
    normal2_sqr = normal2.norm_sqr()
    angle = gs.qd_float(0.0)
    gradients = qd.Matrix.zero(gs.qd_float, 3, 4)
    is_valid = (
        length > 0.0
        and normal1_sqr > gs.EPS**2 * edge.norm_sqr() * (p3 - p1).norm_sqr()
        and normal2_sqr > gs.EPS**2 * edge.norm_sqr() * (p4 - p1).norm_sqr()
    )
    if is_valid:
        angle = qd.atan2(edge.dot(normal1.cross(normal2)) / length, normal1.dot(normal2))
        gradients[:, 2] = -length / normal1_sqr * normal1
        gradients[:, 3] = length / normal2_sqr * normal2
        gradients[:, 1] = (
            -edge.dot(p3 - p1) / length**2 * gradients[:, 2] - edge.dot(p4 - p1) / length**2 * gradients[:, 3]
        )
        gradients[:, 0] = -gradients[:, 1] - gradients[:, 2] - gradients[:, 3]
    return angle, gradients, is_valid


@qd.kernel
def kernel_init_bending(particles_info: V_ANNOTATION, inner_edges_info: V_ANNOTATION, errno: qd.Tensor):
    for i_c in range(inner_edges_info.shape[0]):
        info = inner_edges_info[i_c]
        angle, _, is_valid = func_dihedral(
            particles_info[info.v1].pos_rest,
            particles_info[info.v2].pos_rest,
            particles_info[info.v3].pos_rest,
            particles_info[info.v4].pos_rest,
        )
        inner_edges_info[i_c].angle_rest = angle
        if not is_valid:
            qd.atomic_or(errno[0], ErrorCode.INVALID_PBD_STATE)


@qd.kernel
def kernel_predict(
    time: float,
    dt: float,
    particles: V_ANNOTATION,
    particles_ng: V_ANNOTATION,
    particles_info: V_ANNOTATION,
    gravity: qd.Tensor,
    solver: V_ANNOTATION,
    errno: qd.Tensor,
):
    for i_p, i_b in qd.ndrange(particles.shape[0], particles.shape[1]):
        particles[i_p, i_b].ipos = particles[i_p, i_b].pos
        if particles_ng[i_p, i_b].active:
            velocity = particles[i_p, i_b].vel
            if particles[i_p, i_b].free:
                acceleration = gravity[i_b]
                for i_ff in qd.static(range(len(solver._ffs))):
                    acceleration += solver._ffs[i_ff].get_acc(particles[i_p, i_b].pos, velocity, time, i_p)
                acceleration -= (
                    particles_info[i_p].air_resistance / particles_info[i_p].mass * velocity.norm() * velocity
                )
                velocity += dt * acceleration
            particles[i_p, i_b].pos += dt * velocity
            particles[i_p, i_b].vel = velocity
            if qd.math.isnan(velocity).any() or qd.math.isinf(velocity).any():
                qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_STATE)
        particles[i_p, i_b].pos_iter = particles[i_p, i_b].pos


@qd.func
def func_accumulate_stretch(
    dt: float,
    particles: V_ANNOTATION,
    particles_ng: V_ANNOTATION,
    particles_info: V_ANNOTATION,
    edges_info: V_ANNOTATION,
):
    for i_c, i_b in qd.ndrange(edges_info.shape[0], particles.shape[1]):
        info = edges_info[i_c]
        i1, i2 = info.v1, info.v2
        if particles_ng[i1, i_b].active and particles_ng[i2, i_b].active:
            w1 = particles[i1, i_b].free / particles_info[i1].mass
            w2 = particles[i2, i_b].free / particles_info[i2].mass
            edge = particles[i1, i_b].pos - particles[i2, i_b].pos
            length = edge.norm()
            if w1 + w2 > 0.0:
                direction = geom.qd_normalize(edge, gs.EPS)
                if length <= gs.EPS:
                    direction = geom.qd_normalize(particles_info[i1].pos_rest - particles_info[i2].pos_rest, gs.EPS)
                correction = (
                    -info.stretch_relaxation
                    * (length - info.len_rest)
                    / (w1 + w2 + info.stretch_compliance / dt**2)
                    * direction
                )
                particles[i1, i_b].dpos += w1 * correction
                particles[i2, i_b].dpos -= w2 * correction


@qd.func
def func_accumulate_volume(
    dt: float,
    particles: V_ANNOTATION,
    particles_ng: V_ANNOTATION,
    particles_info: V_ANNOTATION,
    elems_info: V_ANNOTATION,
):
    for i_c, i_b in qd.ndrange(elems_info.shape[0], particles.shape[1]):
        info = elems_info[i_c]
        vertices = qd.Vector([info.v1, info.v2, info.v3, info.v4])
        points = qd.Matrix.zero(gs.qd_float, 3, 4)
        weights = qd.Vector.zero(gs.qd_float, 4)
        is_active = True
        for i_v in qd.static(range(4)):
            i_p = vertices[i_v]
            points[:, i_v] = particles[i_p, i_b].pos
            weights[i_v] = particles[i_p, i_b].free / particles_info[i_p].mass
            is_active = is_active and particles_ng[i_p, i_b].active
        if is_active:
            p1, p2, p3, p4 = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
            gradients = qd.Matrix.cols(
                [
                    (p4 - p2).cross(p3 - p2) / 6.0,
                    (p3 - p1).cross(p4 - p1) / 6.0,
                    (p4 - p1).cross(p2 - p1) / 6.0,
                    (p2 - p1).cross(p3 - p1) / 6.0,
                ]
            )
            denominator = gs.qd_float(0.0)
            for i_v in qd.static(range(4)):
                denominator += weights[i_v] * gradients[:, i_v].norm_sqr()
            if denominator > 0.0:
                residual = geom.qd_tet_vol(p1, p2, p3, p4) - info.vol_rest
                multiplier = -info.volume_relaxation * residual / (denominator + info.volume_compliance / dt**2)
                for i_v in qd.static(range(4)):
                    particles[vertices[i_v], i_b].dpos += multiplier * weights[i_v] * gradients[:, i_v]


@qd.func
def func_accumulate_bending(
    dt: float,
    particles: V_ANNOTATION,
    particles_ng: V_ANNOTATION,
    particles_info: V_ANNOTATION,
    inner_edges_info: V_ANNOTATION,
    errno: qd.Tensor,
):
    for i_c, i_b in qd.ndrange(inner_edges_info.shape[0], particles.shape[1]):
        info = inner_edges_info[i_c]
        vertices = qd.Vector([info.v1, info.v2, info.v3, info.v4])
        points = qd.Matrix.zero(gs.qd_float, 3, 4)
        weights = qd.Vector.zero(gs.qd_float, 4)
        is_active = True
        for i_v in qd.static(range(4)):
            i_p = vertices[i_v]
            points[:, i_v] = particles[i_p, i_b].pos
            weights[i_v] = particles[i_p, i_b].free / particles_info[i_p].mass
            is_active = is_active and particles_ng[i_p, i_b].active
        if is_active and weights.sum() > 0.0:
            angle, gradients, is_valid = func_dihedral(points[:, 0], points[:, 1], points[:, 2], points[:, 3])
            if is_valid:
                residual = qd.atan2(qd.sin(angle - info.angle_rest), qd.cos(angle - info.angle_rest))
                denominator = gs.qd_float(0.0)
                for i_v in qd.static(range(4)):
                    denominator += weights[i_v] * gradients[:, i_v].norm_sqr()
                if denominator > 0.0:
                    multiplier = -info.bending_relaxation * residual / (denominator + info.bending_compliance / dt**2)
                    for i_v in qd.static(range(4)):
                        particles[vertices[i_v], i_b].dpos += multiplier * weights[i_v] * gradients[:, i_v]
            else:
                qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_STATE)


@qd.func
def func_apply_delta(acceleration: float, particles: V_ANNOTATION, particles_ng: V_ANNOTATION, errno: qd.Tensor):
    for i_p, i_b in qd.ndrange(particles.shape[0], particles.shape[1]):
        if particles_ng[i_p, i_b].active and particles[i_p, i_b].free:
            previous = particles[i_p, i_b].pos_iter
            particles[i_p, i_b].pos_iter = particles[i_p, i_b].pos
            pos = (
                particles[i_p, i_b].pos + particles[i_p, i_b].dpos + acceleration * (particles[i_p, i_b].pos - previous)
            )
            if qd.math.isnan(pos).any() or qd.math.isinf(pos).any():
                qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_STATE)
            else:
                particles[i_p, i_b].pos = pos
        particles[i_p, i_b].dpos.fill(0.0)


@qd.func
def func_apply_particle_contact_delta(i_p, i_b, particles: V_ANNOTATION, errno: qd.Tensor):
    pos = particles[i_p, i_b].pos + particles[i_p, i_b].dpos
    if qd.math.isnan(pos).any() or qd.math.isinf(pos).any():
        qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_STATE)
    else:
        particles[i_p, i_b].pos = pos
        # Translate the elastic history with the contact correction so extrapolation remains elastic-only.
        particles[i_p, i_b].pos_iter += particles[i_p, i_b].dpos
    particles[i_p, i_b].dpos = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_apply_contact_delta(particles: V_ANNOTATION, errno: qd.Tensor):
    for i_p, i_b in qd.ndrange(particles.shape[0], particles.shape[1]):
        func_apply_particle_contact_delta(i_p, i_b, particles, errno)


@qd.func
def func_accumulate_contact(i_b, indices, residual, gradients, weights, particles: V_ANNOTATION, errno: qd.Tensor):
    if residual < 0.0:
        denominator = gs.qd_float(0.0)
        for i_v in qd.static(range(indices.n)):
            denominator += weights[i_v] * gradients[:, i_v].norm_sqr()
        if denominator > 0.0:
            multiplier = -residual / denominator
            for i_v in qd.static(range(indices.n)):
                i_p = indices[i_v]
                delta = multiplier * weights[i_v] * gradients[:, i_v]
                if qd.math.isnan(delta).any() or qd.math.isinf(delta).any():
                    qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_STATE)
                else:
                    qd.atomic_add(particles[i_p, i_b].dpos, delta)
        else:
            qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_STATE)


@qd.func
def func_accumulate_vertex_contacts(
    particles: V_ANNOTATION,
    particles_ng: V_ANNOTATION,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    surface_info: array_class.RigidSurfaceInfo,
    solver: V_ANNOTATION,
    errno: qd.Tensor,
):
    for i_p, i_b in qd.ndrange(particles.shape[0], particles.shape[1]):
        if particles_ng[i_p, i_b].active and particles[i_p, i_b].free:
            for i_axis in qd.static(range(3)):
                normal = qd.Vector.zero(gs.qd_float, 3)
                normal[i_axis] = 1.0
                func_accumulate_contact(
                    i_b,
                    qd.Vector([i_p]),
                    particles[i_p, i_b].pos[i_axis]
                    - solver.boundary.lower_qd[i_axis]
                    - solver._options.collision_thickness,
                    qd.Matrix.cols([normal]),
                    qd.Vector([1.0]),
                    particles,
                    errno,
                )
                func_accumulate_contact(
                    i_b,
                    qd.Vector([i_p]),
                    solver.boundary.upper_qd[i_axis]
                    - particles[i_p, i_b].pos[i_axis]
                    - solver._options.collision_thickness,
                    qd.Matrix.cols([-normal]),
                    qd.Vector([1.0]),
                    particles,
                    errno,
                )
            if qd.static(isinstance(solver._rigid_surface, RigidSurface)):
                for i_g_ in range(surface_info.projection_geoms_idx.shape[0]):
                    i_g = surface_info.projection_geoms_idx[i_g_]
                    pos = particles[i_p, i_b].pos
                    residual = gs.qd_float(0.0)
                    normal = qd.Vector.zero(gs.qd_float, 3)
                    if (
                        dyn_info.geoms.type[i_g] != gs.GEOM_TYPE.SPHERE
                        and dyn_info.geoms.type[i_g] != gs.GEOM_TYPE.PLANE
                    ):
                        pos_local = geom.qd_inv_transform_by_trans_quat(
                            pos, dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b]
                        )
                        if (
                            pos_local >= rigid_info.geoms_init_AABB[i_g, 0] - solver._options.contact_detection_distance
                        ).all() and (
                            pos_local <= rigid_info.geoms_init_AABB[i_g, 7] + solver._options.contact_detection_distance
                        ).all():
                            _, direction, is_inside, distance, face_idx = sdf.sdf_func_exact_mesh_surface_bvh_local(
                                i_g,
                                pos_local,
                                solver._rigid_surface.bvh.nodes,
                                solver._rigid_surface.bvh.morton_codes,
                                dyn_info,
                                rigid_info,
                                surface_info,
                            )
                            if is_inside:
                                face = dyn_info.faces.verts_idx[face_idx]
                                residual, gradients, is_valid = point_triangle_constraint(
                                    pos_local,
                                    dyn_info.verts.init_pos[face[0]],
                                    dyn_info.verts.init_pos[face[1]],
                                    dyn_info.verts.init_pos[face[2]],
                                )
                                normal = gradients[:, 0]
                                if normal.dot(direction) < 0.0:
                                    normal = -normal
                                    residual = -residual
                                normal = geom.qd_transform_by_quat(normal, dyn_state.geoms.quat[i_g, i_b])
                                if not is_valid:
                                    qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_STATE)
                                residual -= solver._options.collision_thickness
                            elif distance <= solver._options.contact_detection_distance:
                                residual = distance - solver._options.collision_thickness
                                normal = geom.qd_transform_by_quat(direction, dyn_state.geoms.quat[i_g, i_b])
                    else:
                        residual = sdf.sdf_func_world(i_g, i_b, pos, dyn_state.geoms, dyn_info.geoms, collider_info.sdf)
                        residual -= solver._options.collision_thickness
                        if residual < 0.0:
                            normal = geom.qd_normalize(
                                sdf.sdf_func_grad_world_local_consistent(
                                    i_g,
                                    pos,
                                    dyn_state.geoms.pos[i_g, i_b],
                                    dyn_state.geoms.quat[i_g, i_b],
                                    dyn_info,
                                    rigid_info,
                                    collider_info,
                                ),
                                rigid_info.EPS[None],
                            )
                    if qd.math.isnan(residual) or qd.math.isinf(residual) or qd.math.isnan(normal).any():
                        qd.atomic_or(errno[i_b], ErrorCode.INVALID_CONTACT_NAN)
                    else:
                        func_accumulate_contact(
                            i_b,
                            qd.Vector([i_p]),
                            residual,
                            qd.Matrix.cols([normal]),
                            qd.Vector([1.0]),
                            particles,
                            errno,
                        )


@qd.func
def func_bvh_successor(i_b, node_idx, root_idx, nodes: V_ANNOTATION):
    """Return the next depth-first node after skipping the current subtree."""
    previous_node = node_idx
    node_idx = nodes[i_b, node_idx].parent
    while node_idx >= 0:
        if previous_node == root_idx:
            node_idx = -1
            break
        if nodes[i_b, node_idx].right != previous_node:
            break
        previous_node = node_idx
        node_idx = nodes[i_b, node_idx].parent
    if node_idx >= 0:
        node_idx = nodes[i_b, node_idx].right
    return node_idx


@qd.func
def func_refit_contact_surface(solver: V_ANNOTATION):
    for i_b, i_f in qd.ndrange(solver._B, solver._surface_faces.shape[0]):
        face = solver._surface_faces[i_f]
        lower = qd.Vector([1.0e30, 1.0e30, 1.0e30])
        upper = -lower
        is_active = True
        for i_v in qd.static(range(3)):
            i_p = face[i_v]
            is_active = is_active and solver.particles_ng[i_p, i_b].active
            lower = qd.min(lower, solver.particles[i_p, i_b].pos, solver.particles[i_p, i_b].ipos)
            upper = qd.max(upper, solver.particles[i_p, i_b].pos, solver.particles[i_p, i_b].ipos)
        solver._surface_bvh.aabbs[i_b, i_f].min = qd.select(is_active, lower, 1.0e30)
        solver._surface_bvh.aabbs[i_b, i_f].max = qd.select(is_active, upper, -1.0e30)
    func_refit_bvh(solver._surface_bvh)
    for i_e, i_b in qd.ndrange(solver.n_entities, solver._B):
        i_root = solver._surface_roots[i_e]
        lower = qd.Vector([1.0e30, 1.0e30, 1.0e30])
        upper = -lower
        if i_root >= 0:
            lower = solver._surface_bvh.nodes[i_b, i_root].bound.min
            upper = solver._surface_bvh.nodes[i_b, i_root].bound.max
        solver._surface_bounds[i_e, i_b].lower = lower
        solver._surface_bounds[i_e, i_b].upper = upper


@qd.func
def func_accumulate_face_contacts(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    surface_info: array_class.RigidSurfaceInfo,
    solver: V_ANNOTATION,
    is_projecting: qd.template(),
):
    for i_v, i_b in qd.ndrange(solver._contact_vertices.shape[0], solver._B):
        counts = qd.Vector.zero(gs.qd_int, 6)
        vertex_idx, geom_slot = solver._contact_vertices[i_v]
        geom_idx = surface_info.surface_geoms_idx[geom_slot]
        point_local = dyn_info.verts.init_pos[vertex_idx]
        point = geom.qd_transform_by_trans_quat(
            point_local, dyn_state.geoms.pos[geom_idx, i_b], dyn_state.geoms.quat[geom_idx, i_b]
        )
        previous_point = geom.qd_transform_by_trans_quat(
            point_local, solver._previous_geoms_pos[geom_slot, i_b], solver._previous_geoms_quat[geom_slot, i_b]
        )
        ray_dir = qd.Vector([0.8192319205, 0.4630140578, 0.3395271683])
        axes, shear, _ = ray_projection(ray_dir, gs.EPS)
        for i_e in range(solver.n_entities):
            bounds = solver._surface_bounds[i_e, i_b]
            if (point >= bounds.lower - solver._options.contact_detection_distance).all() and (
                point <= bounds.upper + solver._options.contact_detection_distance
            ).all():
                winding_crossings = 0
                counts[0] += 1
                node_idx = solver._surface_roots[i_e]
                while node_idx >= 0:
                    counts[1] += 1
                    node = solver._surface_bvh.nodes[i_b, node_idx]
                    has_hit = ray_aabb_intersection(point, ray_dir, node.bound.min, node.bound.max, gs.EPS) >= 0.0
                    if has_hit and node.left >= 0:
                        node_idx = node.left
                    else:
                        if has_hit:
                            leaf_idx = node_idx - (solver._surface_faces.shape[0] - 1)
                            i_f = solver._surface_bvh.leaves_idx[i_b, leaf_idx]
                            if solver._surface_face_entities[i_f] == i_e:
                                counts[2] += 1
                                face = solver._surface_faces[i_f]
                                v0 = solver.particles[face[0], i_b].pos
                                v1 = solver.particles[face[1], i_b].pos
                                v2 = solver.particles[face[2], i_b].pos
                                hit = ray_triangle_intersection(axes, point, shear, v0, v1, v2, gs.EPS)
                                if hit >= 0.0:
                                    alignment = (v1 - v0).cross(v2 - v0).dot(ray_dir)
                                    if alignment > 0.0:
                                        winding_crossings += 1
                                    elif alignment < 0.0:
                                        winding_crossings -= 1
                        node_idx = func_bvh_successor(
                            i_b, node_idx, solver._surface_roots[i_e], solver._surface_bvh.nodes
                        )
                if winding_crossings != 0 or qd.static(solver._options.contact_detection_distance > 0.0):
                    closest_face = -1
                    closest_distance_sqr = gs.qd_float(solver._options.contact_detection_distance**2)
                    if winding_crossings != 0:
                        closest_distance_sqr = 1.0e30
                    entry_face = -1
                    last_entry_time = gs.qd_float(-1.0)
                    segment_lower = qd.min(point, previous_point)
                    segment_upper = qd.max(point, previous_point)
                    counts[3] += 1
                    node_idx = solver._surface_roots[i_e]
                    while node_idx >= 0:
                        counts[4] += 1
                        node = solver._surface_bvh.nodes[i_b, node_idx]
                        has_overlap = (
                            winding_crossings != 0
                            and (segment_lower <= node.bound.max).all()
                            and (segment_upper >= node.bound.min).all()
                        )
                        is_near = point_aabb_distance_sqr(point, node.bound.min, node.bound.max) <= closest_distance_sqr
                        if (has_overlap or is_near) and node.left >= 0:
                            node_idx = node.left
                        else:
                            if has_overlap or is_near:
                                leaf_idx = node_idx - (solver._surface_faces.shape[0] - 1)
                                i_f = solver._surface_bvh.leaves_idx[i_b, leaf_idx]
                                if solver._surface_face_entities[i_f] == i_e:
                                    counts[5] += 1
                                    face = solver._surface_faces[i_f]
                                    triangle = qd.Matrix.zero(gs.qd_float, 3, 3)
                                    previous_triangle = qd.Matrix.zero(gs.qd_float, 3, 3)
                                    for i_fv in qd.static(range(3)):
                                        triangle[:, i_fv] = solver.particles[face[i_fv], i_b].pos
                                        previous_triangle[:, i_fv] = solver.particles[face[i_fv], i_b].ipos
                                    closest = closest_point_on_triangle(
                                        point, triangle[:, 0], triangle[:, 1], triangle[:, 2]
                                    )
                                    distance_sqr = (point - closest).norm_sqr()
                                    if distance_sqr < closest_distance_sqr:
                                        closest_distance_sqr = distance_sqr
                                        closest_face = i_f
                                    if has_overlap:
                                        entry_time = point_triangle_entry_time(
                                            previous_point, point, previous_triangle, triangle
                                        )
                                        if entry_time > last_entry_time:
                                            last_entry_time = entry_time
                                            entry_face = i_f
                            node_idx = func_bvh_successor(
                                i_b, node_idx, solver._surface_roots[i_e], solver._surface_bvh.nodes
                            )
                    if qd.static(is_projecting) and closest_face >= 0:
                        face = solver._surface_faces[qd.select(entry_face >= 0, entry_face, closest_face)]
                        residual, gradients, is_valid = point_triangle_constraint(
                            point,
                            solver.particles[face[0], i_b].pos,
                            solver.particles[face[1], i_b].pos,
                            solver.particles[face[2], i_b].pos,
                        )
                        # Separated point-face contacts require a projection inside the finite triangle.
                        # Boundary features are covered by finite-edge proximity contacts.
                        is_contact = winding_crossings != 0
                        if winding_crossings == 0:
                            is_contact = residual >= 0.0
                            for i_fv in qd.static(range(3)):
                                is_contact = (
                                    is_contact and -gradients[:, i_fv + 1].dot(gradients[:, 0]) >= -32.0 * gs.EPS
                                )
                        residual -= solver._options.collision_thickness
                        weights = qd.Vector.zero(gs.qd_float, 3)
                        for i_fv in qd.static(range(3)):
                            i_p = face[i_fv]
                            weights[i_fv] = solver.particles[i_p, i_b].free / solver.particles_info[i_p].mass
                        if not is_valid:
                            qd.atomic_or(solver._errno[i_b], ErrorCode.INVALID_PBD_STATE)
                        elif is_contact and weights.sum() > 0.0:
                            func_accumulate_contact(
                                i_b, face, residual, gradients[:, 1:4], weights, solver.particles, solver._errno
                            )
        solver._surface_query_counts[i_v, i_b] = counts


@qd.kernel
def kernel_measure_surface_queries(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    surface_info: array_class.RigidSurfaceInfo,
    solver: V_ANNOTATION,
):
    func_refit_contact_surface(solver)
    func_accumulate_face_contacts(dyn_state, dyn_info, surface_info, solver, is_projecting=False)


@qd.func
def func_edge_contacts(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    surface_info: array_class.RigidSurfaceInfo,
    solver: V_ANNOTATION,
    is_diagnostic: qd.template(),
):
    """Accumulate unique edge-pair corrections at fixed positions and average them per participating vertex."""
    if qd.static(not is_diagnostic):
        for i_p, i_b in qd.ndrange(solver.n_particles, solver._B):
            solver._edge_contact_counts[i_p, i_b] = 0
    for i_e, i_b in qd.ndrange(solver._surface_edge_adjacency.edge_offsets.shape[0] - 1, solver._B):
        if qd.static(not is_diagnostic and solver._options.is_recording_edge_contacts):
            solver._detected_edge_contacts[i_e, i_b] = False
        for geom_slot in range(surface_info.surface_geoms_idx.shape[0]):
            for face_slot in range(
                solver._surface_edge_adjacency.edge_offsets[i_e],
                solver._surface_edge_adjacency.edge_offsets[i_e + 1],
            ):
                i_f = solver._surface_edge_adjacency.edge_faces[face_slot]
                edge_slot = 0
                for slot in qd.static(range(3)):
                    if solver._surface_edge_adjacency.face_edges[i_f][slot] == i_e:
                        edge_slot = slot
                face = solver._surface_faces[i_f]
                triangle_world = qd.Matrix.zero(gs.qd_float, 3, 3)
                is_active = True
                for slot in qd.static(range(3)):
                    triangle_world[:, slot] = solver.particles[face[slot], i_b].pos
                    is_active = is_active and solver.particles_ng[face[slot], i_b].active
                if not is_active:
                    continue
                geom_idx = surface_info.surface_geoms_idx[geom_slot]
                geom_pos = dyn_state.geoms.pos[geom_idx, i_b]
                geom_quat = dyn_state.geoms.quat[geom_idx, i_b]
                triangle = qd.Matrix.zero(gs.qd_float, 3, 3)
                for slot in qd.static(range(3)):
                    triangle[:, slot] = geom.qd_inv_transform_by_trans_quat(
                        triangle_world[:, slot], geom_pos, geom_quat
                    )
                atlas_offset = surface_info.atlas_offsets[geom_slot]
                lower = qd.min(triangle[:, 0], triangle[:, 1], triangle[:, 2])
                upper = qd.max(triangle[:, 0], triangle[:, 1], triangle[:, 2])
                if qd.static(not is_diagnostic):
                    lower -= solver._options.contact_detection_distance
                    upper += solver._options.contact_detection_distance
                bounds = solver._contact_bounds[geom_slot]
                if (lower > bounds.upper).any() or (upper < bounds.lower).any():
                    continue
                lower += atlas_offset
                upper += atlas_offset
                node_idx = 0
                while node_idx >= 0:
                    node = solver._rigid_surface.bvh.nodes[0, node_idx]
                    has_overlap = (lower <= node.bound.max).all() and (upper >= node.bound.min).all()
                    if has_overlap and node.left >= 0:
                        node_idx = node.left
                        continue
                    if has_overlap:
                        leaf_idx = node_idx - (solver._rigid_surface.bvh.morton_codes.shape[1] - 1)
                        rigid_face_idx = qd.cast(solver._rigid_surface.bvh.morton_codes[0, leaf_idx][1], gs.qd_int)
                        if dyn_info.faces.geom_idx[rigid_face_idx] == geom_idx:
                            i_rf = rigid_face_idx + solver._contact_face_offsets[geom_slot]
                            rigid_face = solver._contact_faces[i_rf]
                            rigid_triangle = qd.Matrix.zero(gs.qd_float, 3, 3)
                            for slot in qd.static(range(3)):
                                rigid_triangle[:, slot] = dyn_info.verts.init_pos[rigid_face[slot]]
                            crossing_edges, rigid_crossing_edges, is_valid = triangle_crossing_edges(
                                triangle, rigid_triangle
                            )
                            if not is_valid:
                                qd.atomic_or(solver._errno[i_b], ErrorCode.INVALID_PBD_STATE)
                            # Each triangle is visited through its three edges; its first edge owns the face count.
                            if (
                                qd.static(is_diagnostic)
                                and edge_slot == 0
                                and (crossing_edges.any() or rigid_crossing_edges.any())
                            ):
                                qd.atomic_add(solver._collision_counts[i_b][0], 1)
                                if not (crossing_edges.any() and rigid_crossing_edges.any()):
                                    qd.atomic_add(solver._collision_counts[i_b][2], 1)
                            for rigid_edge_slot in range(3):
                                is_crossing = crossing_edges[edge_slot] and rigid_crossing_edges[rigid_edge_slot]
                                if (
                                    qd.static(is_diagnostic or solver._options.contact_detection_distance == 0.0)
                                    and not is_crossing
                                ):
                                    continue
                                edge = qd.Matrix.cols([triangle[:, edge_slot], triangle[:, (edge_slot + 1) % 3]])
                                rigid_edge = qd.Matrix.cols(
                                    [rigid_triangle[:, rigid_edge_slot], rigid_triangle[:, (rigid_edge_slot + 1) % 3]]
                                )
                                is_near = False
                                if qd.static(not is_diagnostic and solver._options.contact_detection_distance > 0.0):
                                    normal = (rigid_triangle[:, 1] - rigid_triangle[:, 0]).cross(
                                        rigid_triangle[:, 2] - rigid_triangle[:, 0]
                                    )
                                    gap, _, is_valid = edge_edge_contact(edge, rigid_edge, normal)
                                    is_near = is_valid and qd.abs(gap) <= solver._options.contact_detection_distance
                                if not (is_crossing or is_near):
                                    continue
                                i_re = solver._rigid_edge_adjacency.face_edges[i_rf][rigid_edge_slot]
                                is_owner = True
                                # Proximity depends only on the two edges, so their first active incident faces own it.
                                # Deep crossings require the first incident faces that actually intersect.
                                for adjacent_slot in range(
                                    solver._surface_edge_adjacency.edge_offsets[i_e],
                                    solver._surface_edge_adjacency.edge_offsets[i_e + 1],
                                ):
                                    adjacent_idx = solver._surface_edge_adjacency.edge_faces[adjacent_slot]
                                    if adjacent_idx >= i_f:
                                        break
                                    adjacent = solver._surface_faces[adjacent_idx]
                                    adjacent_triangle = qd.Matrix.zero(gs.qd_float, 3, 3)
                                    is_adjacent_active = True
                                    for slot in qd.static(range(3)):
                                        adjacent_triangle[:, slot] = geom.qd_inv_transform_by_trans_quat(
                                            solver.particles[adjacent[slot], i_b].pos, geom_pos, geom_quat
                                        )
                                        is_adjacent_active = (
                                            is_adjacent_active and solver.particles_ng[adjacent[slot], i_b].active
                                        )
                                    is_hit = is_near
                                    if not is_near:
                                        is_hit, _ = segment_triangle_crossing(
                                            rigid_edge[:, 0], rigid_edge[:, 1], adjacent_triangle
                                        )
                                    if is_adjacent_active and is_hit:
                                        is_owner = False
                                        break
                                if not is_owner:
                                    continue
                                rigid_normal = qd.Vector.zero(gs.qd_float, 3)
                                for adjacent_slot in range(
                                    solver._rigid_edge_adjacency.edge_offsets[i_re],
                                    solver._rigid_edge_adjacency.edge_offsets[i_re + 1],
                                ):
                                    adjacent_idx = solver._rigid_edge_adjacency.edge_faces[adjacent_slot]
                                    adjacent = solver._contact_faces[adjacent_idx]
                                    adjacent_triangle = qd.Matrix.zero(gs.qd_float, 3, 3)
                                    for slot in qd.static(range(3)):
                                        adjacent_triangle[:, slot] = dyn_info.verts.init_pos[adjacent[slot]]
                                    normal = (adjacent_triangle[:, 1] - adjacent_triangle[:, 0]).cross(
                                        adjacent_triangle[:, 2] - adjacent_triangle[:, 0]
                                    )
                                    if normal.norm_sqr() > 0.0:
                                        rigid_normal += normal.normalized()
                                    if adjacent_idx < i_rf:
                                        is_hit = is_near
                                        if not is_near:
                                            is_hit, _ = segment_triangle_crossing(
                                                edge[:, 0], edge[:, 1], adjacent_triangle
                                            )
                                        if is_hit:
                                            is_owner = False
                                if not is_owner:
                                    continue
                                residual, gradients, is_valid = edge_edge_contact(edge, rigid_edge, rigid_normal)
                                if not is_valid or qd.math.isnan(residual) or qd.math.isinf(residual):
                                    qd.atomic_or(solver._errno[i_b], ErrorCode.INVALID_PBD_STATE)
                                else:
                                    if qd.static(is_diagnostic):
                                        if residual < 0.0:
                                            qd.atomic_add(solver._collision_counts[i_b][1], 1)
                                            qd.atomic_max(solver._collision_depth[i_b], -residual)
                                    else:
                                        residual -= solver._options.collision_thickness
                                        if not (is_near or residual < 0.0):
                                            continue
                                        if qd.static(solver._options.is_recording_edge_contacts):
                                            solver._detected_edge_contacts[i_e, i_b] = True
                                        indices = qd.Vector([face[edge_slot], face[(edge_slot + 1) % 3]])
                                        weights = qd.Vector.zero(gs.qd_float, 2)
                                        for slot in qd.static(range(2)):
                                            i_p = indices[slot]
                                            weights[slot] = (
                                                solver.particles[i_p, i_b].free / solver.particles_info[i_p].mass
                                            )
                                            gradients[:, slot] = geom.qd_transform_by_quat(
                                                gradients[:, slot], geom_quat
                                            )
                                        if weights.sum() > 0.0:
                                            func_accumulate_contact(
                                                i_b,
                                                indices,
                                                residual,
                                                gradients,
                                                weights,
                                                solver.particles,
                                                solver._errno,
                                            )
                                            # Count separated candidates in the detection band too, keeping averaging
                                            # stable as individual constraints enter or leave the target gap.
                                            for slot in qd.static(range(2)):
                                                if weights[slot] > 0.0:
                                                    qd.atomic_add(solver._edge_contact_counts[indices[slot], i_b], 1)
                    node_idx = func_bvh_successor(0, node_idx, -1, solver._rigid_surface.bvh.nodes)

    if qd.static(not is_diagnostic):
        for i_p, i_b in qd.ndrange(solver.n_particles, solver._B):
            count = solver._edge_contact_counts[i_p, i_b]
            if count > 0:
                solver.particles[i_p, i_b].dpos /= count
                func_apply_particle_contact_delta(i_p, i_b, solver.particles, solver._errno)


@qd.kernel
def kernel_collision_diagnostics(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    surface_info: array_class.RigidSurfaceInfo,
    solver: V_ANNOTATION,
):
    for i_b in range(solver._B):
        solver._collision_counts[i_b].fill(0)
        solver._collision_depth[i_b] = 0.0
    func_edge_contacts(dyn_state, dyn_info, surface_info, solver, is_diagnostic=True)


@qd.func
def func_store_contact_positions(particles: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(particles.shape[0], particles.shape[1]):
        particles[i_p, i_b].contact_pos = particles[i_p, i_b].pos
        particles[i_p, i_b].dpos = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_store_contact_poses(
    dyn_state: array_class.DynState, surface_info: array_class.RigidSurfaceInfo, solver: V_ANNOTATION
):
    for i_g_, i_b in qd.ndrange(surface_info.surface_geoms_idx.shape[0], solver._B):
        i_g = surface_info.surface_geoms_idx[i_g_]
        solver._previous_geoms_pos[i_g_, i_b] = dyn_state.geoms.pos[i_g, i_b]
        solver._previous_geoms_quat[i_g_, i_b] = dyn_state.geoms.quat[i_g, i_b]


@qd.func
def func_check_volume(particles: V_ANNOTATION, particles_ng: V_ANNOTATION, elems_info: V_ANNOTATION, errno: qd.Tensor):
    for i_c, i_b in qd.ndrange(elems_info.shape[0], particles.shape[1]):
        info = elems_info[i_c]
        if (
            particles_ng[info.v1, i_b].active
            and particles_ng[info.v2, i_b].active
            and particles_ng[info.v3, i_b].active
            and particles_ng[info.v4, i_b].active
        ):
            volume = geom.qd_tet_vol(
                particles[info.v1, i_b].pos,
                particles[info.v2, i_b].pos,
                particles[info.v3, i_b].pos,
                particles[info.v4, i_b].pos,
            )
            if volume * info.vol_rest <= 0.0 and not (
                particles[info.v1, i_b].free
                or particles[info.v2, i_b].free
                or particles[info.v3, i_b].free
                or particles[info.v4, i_b].free
            ):
                qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_VOLUME)


@qd.func
def func_compute_momentum(solver: V_ANNOTATION, is_initial: qd.template(), is_contact: qd.template()):
    for i_e, i_b in qd.ndrange(solver.n_entities, solver._B):
        solver._momentum[i_e, i_b].center = qd.Vector.zero(gs.qd_float, 3)
        solver._momentum[i_e, i_b].linear = qd.Vector.zero(gs.qd_float, 3)
        solver._momentum[i_e, i_b].angular = qd.Vector.zero(gs.qd_float, 3)
        if qd.static(is_initial):
            solver._momentum[i_e, i_b].mass = 0.0
            solver._momentum[i_e, i_b].is_free = True
        if qd.static(not is_initial and not is_contact):
            solver._momentum[i_e, i_b].covariance = qd.Matrix.zero(gs.qd_float, 3, 3)

    for i_p, i_b in qd.ndrange(solver.n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active:
            i_e = solver.particles_info[i_p].entity_idx
            mass = solver.particles_info[i_p].mass
            pos = solver.particles[i_p, i_b].pos
            vel = (pos - solver.particles[i_p, i_b].ipos) / solver.substep_dt
            if qd.static(is_initial):
                pos = solver.particles[i_p, i_b].ipos
                vel = solver.particles[i_p, i_b].vel
                qd.atomic_add(solver._momentum[i_e, i_b].mass, mass)
                if not solver.particles[i_p, i_b].free:
                    solver._momentum[i_e, i_b].is_free = False
            if qd.static(is_contact):
                pos = solver.particles[i_p, i_b].contact_pos
                vel = (solver.particles[i_p, i_b].pos - pos) / solver.substep_dt
            solver.particles[i_p, i_b].momentum_vel = vel
            origin = solver.particles[solver._entities_particle_start[i_e], i_b].ipos
            qd.atomic_add(solver._momentum[i_e, i_b].center, mass * (pos - origin))
            qd.atomic_add(solver._momentum[i_e, i_b].linear, mass * vel)

    for i_e, i_b in qd.ndrange(solver.n_entities, solver._B):
        mass = solver._momentum[i_e, i_b].mass
        denominator = mass if mass > 0.0 else 1.0
        origin = solver.particles[solver._entities_particle_start[i_e], i_b].ipos
        solver._momentum[i_e, i_b].center = origin + solver._momentum[i_e, i_b].center / denominator

    for i_p, i_b in qd.ndrange(solver.n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active:
            i_e = solver.particles_info[i_p].entity_idx
            mass = solver.particles_info[i_p].mass
            total_mass = solver._momentum[i_e, i_b].mass
            denominator = total_mass if total_mass > 0.0 else 1.0
            pos = solver.particles[i_p, i_b].pos
            if qd.static(is_initial):
                pos = solver.particles[i_p, i_b].ipos
            if qd.static(is_contact):
                pos = solver.particles[i_p, i_b].contact_pos
            relative = pos - solver._momentum[i_e, i_b].center
            # Centering velocity suppresses angular roundoff from uniform translation.
            vel = solver.particles[i_p, i_b].momentum_vel - solver._momentum[i_e, i_b].linear / denominator
            qd.atomic_add(solver._momentum[i_e, i_b].angular, mass * relative.cross(vel))
            if qd.static(not is_initial and not is_contact):
                qd.atomic_add(solver._momentum[i_e, i_b].covariance, mass * relative.outer_product(relative))

    for i_e, i_b in qd.ndrange(solver.n_entities, solver._B):
        if qd.static(is_initial):
            solver._momentum[i_e, i_b].linear_target = solver._momentum[i_e, i_b].linear
            solver._momentum[i_e, i_b].angular_target = solver._momentum[i_e, i_b].angular
        elif qd.static(is_contact):
            solver._momentum[i_e, i_b].linear_target += solver._momentum[i_e, i_b].linear
            solver._momentum[i_e, i_b].angular_target += solver._momentum[i_e, i_b].angular
        else:
            mass = solver._momentum[i_e, i_b].mass
            denominator = mass if mass > 0.0 else 1.0
            # Muller et al., Position Based Dynamics, section 3.5: damp deformation while preserving rigid modes.
            # Scaling the measured momenta folds damping into the final momentum correction.
            decay = solver._entities_velocity_decay[i_e]
            solver._momentum[i_e, i_b].linear = (
                solver._momentum[i_e, i_b].linear_target - decay * solver._momentum[i_e, i_b].linear
            ) / denominator
            covariance = solver._momentum[i_e, i_b].covariance
            trace = covariance.trace()
            scale = trace if trace > 0.0 else 1.0
            inertia = (trace * qd.Matrix.identity(gs.qd_float, 3) - covariance) / scale
            angular_error = (
                solver._momentum[i_e, i_b].angular_target - decay * solver._momentum[i_e, i_b].angular
            ) / scale
            cofactors = qd.Matrix.rows(
                [
                    inertia[1, :].cross(inertia[2, :]),
                    inertia[2, :].cross(inertia[0, :]),
                    inertia[0, :].cross(inertia[1, :]),
                ]
            )
            determinant = inertia[0, :].dot(cofactors[0, :])
            angular = inertia @ angular_error
            # A line's normalized inertia equals its pseudoinverse.
            if determinant > gs.EPS:
                angular = cofactors.transpose() @ angular_error / determinant
            solver._momentum[i_e, i_b].angular = angular


@qd.func
def func_accumulate_elastic_constraints(solver: V_ANNOTATION):
    if qd.static(solver.n_edges > 0):
        func_accumulate_stretch(
            solver.substep_dt, solver.particles, solver.particles_ng, solver.particles_info, solver.edges_info
        )
    if qd.static(solver.n_inner_edges > 0):
        func_accumulate_bending(
            solver.substep_dt,
            solver.particles,
            solver.particles_ng,
            solver.particles_info,
            solver.inner_edges_info,
            solver._errno,
        )
    if qd.static(solver.n_elems > 0):
        func_accumulate_volume(
            solver.substep_dt, solver.particles, solver.particles_ng, solver.particles_info, solver.elems_info
        )


@qd.kernel
def kernel_project_elastic_constraints(solver: V_ANNOTATION):
    func_accumulate_elastic_constraints(solver)
    func_apply_delta(solver._options.constraint_acceleration, solver.particles, solver.particles_ng, solver._errno)


@qd.func
def func_project_collision(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    surface_info: array_class.RigidSurfaceInfo,
    solver: V_ANNOTATION,
    is_recording: qd.template(),
):
    func_store_contact_positions(solver.particles)
    if qd.static(isinstance(solver._surface_bvh, RefitBVH)):
        func_refit_contact_surface(solver)
        func_accumulate_face_contacts(dyn_state, dyn_info, surface_info, solver, is_projecting=True)
    func_accumulate_vertex_contacts(
        solver.particles,
        solver.particles_ng,
        dyn_state,
        dyn_info,
        rigid_info,
        collider_info,
        surface_info,
        solver,
        solver._errno,
    )
    func_apply_contact_delta(solver.particles, solver._errno)
    if qd.static(is_recording and solver._options.is_recording_edge_contacts):
        for i_p, i_b in qd.ndrange(solver.n_particles, solver._B):
            i_substep = solver._edge_history_substep[()]
            i_iteration = solver._options.max_solver_iterations - solver._graph_counter[()]
            solver._edge_history_positions[i_substep, i_iteration, i_p, i_b] = solver.particles[i_p, i_b].pos
    if qd.static(isinstance(solver._surface_bvh, RefitBVH)):
        func_edge_contacts(dyn_state, dyn_info, surface_info, solver, is_diagnostic=False)
        if qd.static(is_recording and solver._options.is_recording_edge_contacts):
            for i_e, i_b in qd.ndrange(solver._detected_edge_contacts.shape[0], solver._B):
                i_substep = solver._edge_history_substep[()]
                i_iteration = solver._options.max_solver_iterations - solver._graph_counter[()]
                solver._edge_history_masks[i_substep, i_iteration, i_e, i_b] = solver._detected_edge_contacts[i_e, i_b]


@qd.kernel(graph=True)
def kernel_project_collision(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    surface_info: array_class.RigidSurfaceInfo,
    solver: V_ANNOTATION,
):
    func_project_collision(dyn_state, dyn_info, rigid_info, collider_info, surface_info, solver, is_recording=False)


@qd.kernel(graph=True)
def kernel_solve_constraints(
    history: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    surface_info: array_class.RigidSurfaceInfo,
    solver: V_ANNOTATION,
):
    if qd.static(solver._options.is_recording_edge_contacts and solver.scene.rigid_solver.n_links > 0):
        for i_l, i_b in qd.ndrange(solver.scene.rigid_solver.n_links, solver._B):
            i_substep = solver._edge_history_substep[()]
            solver._edge_history_links_pos[i_substep, i_l, i_b] = dyn_state.links.pos[i_l, i_b]
            solver._edge_history_links_quat[i_substep, i_l, i_b] = dyn_state.links.quat[i_l, i_b]
    func_compute_momentum(solver, is_initial=True, is_contact=False)
    solver._graph_counter[()] = solver._options.max_solver_iterations
    while qd.graph.do_while(solver._graph_counter):
        if qd.static(solver._options.is_recording_constraint_history):
            for i_p, i_b in qd.ndrange(solver.n_particles, solver._B):
                i_iteration = solver._options.max_solver_iterations - solver._graph_counter[()]
                for i_axis in qd.static(range(3)):
                    history[i_iteration, 0, i_b, i_p, i_axis] = solver.particles[i_p, i_b].pos[i_axis]
        func_accumulate_elastic_constraints(solver)
        func_apply_delta(solver._options.constraint_acceleration, solver.particles, solver.particles_ng, solver._errno)
        if qd.static(solver._options.is_recording_constraint_history):
            for i_p, i_b in qd.ndrange(solver.n_particles, solver._B):
                i_iteration = solver._options.max_solver_iterations - solver._graph_counter[()]
                for i_axis in qd.static(range(3)):
                    history[i_iteration, 1, i_b, i_p, i_axis] = solver.particles[i_p, i_b].pos[i_axis]
        func_project_collision(
            dyn_state,
            dyn_info,
            rigid_info,
            collider_info,
            surface_info,
            solver,
            is_recording=True,
        )
        func_compute_momentum(solver, is_initial=False, is_contact=True)
        if qd.static(solver._options.is_recording_constraint_history):
            for i_p, i_b in qd.ndrange(solver.n_particles, solver._B):
                i_iteration = solver._options.max_solver_iterations - solver._graph_counter[()]
                for i_axis in qd.static(range(3)):
                    history[i_iteration, 2, i_b, i_p, i_axis] = solver.particles[i_p, i_b].pos[i_axis]
        solver._graph_counter[()] -= 1
    if qd.static(solver.n_elems > 0):
        func_check_volume(solver.particles, solver.particles_ng, solver.elems_info, solver._errno)
    func_compute_momentum(solver, is_initial=False, is_contact=False)
    for i_p, i_b in qd.ndrange(solver.n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active:
            i_e = solver.particles_info[i_p].entity_idx
            vel = solver.particles[i_p, i_b].momentum_vel
            if solver.particles[i_p, i_b].free:
                # See func_compute_momentum for the matching rigid-mode correction.
                vel *= solver._entities_velocity_decay[i_e]
            if solver._momentum[i_e, i_b].is_free:
                relative = solver.particles[i_p, i_b].pos - solver._momentum[i_e, i_b].center
                vel += solver._momentum[i_e, i_b].linear + solver._momentum[i_e, i_b].angular.cross(relative)
            if qd.math.isnan(vel).any() or qd.math.isinf(vel).any():
                qd.atomic_or(solver._errno[i_b], ErrorCode.INVALID_PBD_STATE)
            else:
                solver.particles[i_p, i_b].vel = vel


@qd.kernel
def kernel_update_render(
    particles: V_ANNOTATION,
    particles_ng: V_ANNOTATION,
    particles_render: V_ANNOTATION,
    vverts_render: V_ANNOTATION,
    vverts_info: V_ANNOTATION,
):
    for i_p, i_b in qd.ndrange(particles.shape[0], particles.shape[1]):
        particles_render[i_p, i_b].pos = geom.qd_nowhere()
        if particles_ng[i_p, i_b].active:
            particles_render[i_p, i_b].pos = particles[i_p, i_b].pos
        particles_render[i_p, i_b].vel = particles[i_p, i_b].vel
        particles_render[i_p, i_b].active = particles_ng[i_p, i_b].active
    for i_v, i_b in qd.ndrange(vverts_info.shape[0], particles.shape[1]):
        pos = qd.Vector.zero(gs.qd_float, 3)
        for i_s in qd.static(range(vverts_info.support_idxs.n)):
            pos += particles[vverts_info[i_v].support_idxs[i_s], i_b].pos * vverts_info[i_v].support_weights[i_s]
        vverts_render[i_v, i_b].pos = pos
        vverts_render[i_v, i_b].active = particles_ng[vverts_info[i_v].support_idxs[0], i_b].active


@qd.kernel
def kernel_set_field(
    particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), values: qd.types.ndarray(), data: V_ANNOTATION
):
    for i_b_, i_p_ in qd.ndrange(envs_idx.shape[0], particles_idx.shape[1]):
        i_p, i_b = particles_idx[i_b_, i_p_], envs_idx[i_b_]
        if qd.static(len(values.shape) == 3):
            for i_axis in range(values.shape[-1]):
                data[i_p, i_b][i_axis] = values[i_b_, i_p_, i_axis]
        else:
            data[i_p, i_b] = values[i_b_, i_p_]


@qd.kernel
def kernel_reset_errno(envs_idx: qd.types.ndarray(), errno: qd.Tensor):
    for i_b in range(envs_idx.shape[0]):
        errno[envs_idx[i_b]] = 0


class PBDUnifiedSolver(Solver):
    """Position-based dynamics (PBD) with simultaneous elastic corrections followed by unilateral contact projections.

    Supports cloth and tetrahedral elastic entities. Rigid geoms with ``needs_coup=True`` supply prescribed
    boundaries. Each iteration applies simultaneous elastic corrections, then projects vertex-face and domain contacts,
    followed by edge-edge contacts detected at the updated positions. Vertex-face and domain corrections accumulate
    before a simultaneous update. All unique elastic-rigid edge pairs contribute inverse-mass corrections from the
    same positions, averaged by each vertex's incident edge-contact count before updating positions.
    Surface contacts use elastic boundary faces and rigid collision-mesh triangles.
    Material compliance regularizes each elastic projection denominator; effective stiffness also depends on the
    iteration count and time step. Extrapolation accelerates recovery, with history initialized after motion prediction.
    Fully free entities receive a final velocity correction preserving linear and angular momentum after external
    forces and contact impulses, independently in each environment.
    """

    class MATERIAL(gs.IntEnum):
        CLOTH = 0
        ELASTIC = 1

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        self._options = options
        self._particle_size = options.particle_size
        self.boundary = CubeBoundary(lower=options.lower_bound, upper=options.upper_bound)
        self._n_vvert_supports = scene.vis_options.n_support_neighbors
        self.particles = None
        self.particles_ng = None
        self.particles_info = None
        self.edges_info = None
        self.inner_edges_info = None
        self.elems_info = None
        self.particles_render = None
        self.vverts_info = None
        self.vverts_render = None
        self.vverts_uvs = None
        self.vfaces_indices = None
        self._rigid_surface = None
        self._surface_info = None
        self._contact_vertices = None
        self._surface_faces = None
        self._surface_face_entities = None
        self._surface_bounds = None
        self._surface_bvh = None
        self._surface_roots = None
        self._surface_query_counts = None
        self._surface_query_baseline = None
        self._surface_query_updates = 0
        self._surface_edge_adjacency = None
        self._edge_contact_counts = None
        self._detected_edge_contacts = None
        self._edge_history_substep = None
        self._edge_history_positions = None
        self._edge_history_masks = None
        self._edge_history_links_pos = None
        self._edge_history_links_quat = None
        self._rigid_edge_adjacency = None
        self._contact_faces = None
        self._contact_face_offsets = None
        self._contact_bounds = None
        self._collision_counts = None
        self._collision_depth = None
        self._contact_geoms_idx = None
        self._previous_geoms_pos = None
        self._previous_geoms_quat = None
        self._errno = None
        self._iteration_positions = None
        self._momentum = None
        self._entities_particle_start = None
        self._entities_velocity_decay = None
        self._graph_counter = None
        self._edges = None
        self._inner_edges = None
        self._elems = None
        self._lengths_rest = None
        self._angles_rest = None
        self._volumes_rest = None

    def add_entity(self, idx, material, morph, surface, name=None):
        if isinstance(material, gs.materials.PBD.Cloth):
            entity = PBD2DEntity(
                self.scene,
                self,
                material,
                morph,
                surface,
                self.particle_size,
                idx,
                self.n_particles,
                self.n_edges,
                self.n_inner_edges,
                self.n_vverts,
                self.n_vfaces,
                name=name,
            )
        elif isinstance(material, gs.materials.PBD.Elastic):
            entity = PBD3DEntity(
                self.scene,
                self,
                material,
                morph,
                surface,
                self.particle_size,
                idx,
                self.n_particles,
                self.n_edges,
                self.n_elems,
                self.n_vverts,
                self.n_vfaces,
                name=name,
            )
        else:
            gs.raise_exception("PBDUnifiedSolver supports PBD.Cloth and PBD.Elastic materials.")
        self._entities.append(entity)
        return entity

    def build(self):
        super().build()
        if not self.is_active:
            return
        if self.scene.requires_grad:
            gs.raise_exception("PBDUnifiedSolver requires requires_grad=False.")
        self._n_vvert_supports = min(self._n_vvert_supports, min(entity.n_particles for entity in self.entities))
        self.particles = qd.types.struct(
            free=gs.qd_bool,
            pos=gs.qd_vec3,
            ipos=gs.qd_vec3,
            pos_iter=gs.qd_vec3,
            contact_pos=gs.qd_vec3,
            momentum_vel=gs.qd_vec3,
            dpos=gs.qd_vec3,
            vel=gs.qd_vec3,
        ).field(shape=(self.n_particles, self._B), layout=qd.Layout.SOA)
        self.particles_ng = qd.types.struct(active=gs.qd_bool).field(
            shape=(self.n_particles, self._B), layout=qd.Layout.SOA
        )
        self.particles_info = qd.types.struct(
            mass=gs.qd_float,
            pos_rest=gs.qd_vec3,
            material_type=gs.qd_int,
            entity_idx=gs.qd_int,
            mu_s=gs.qd_float,
            mu_k=gs.qd_float,
            air_resistance=gs.qd_float,
        ).field(shape=(self.n_particles,), layout=qd.Layout.SOA)
        if self.n_edges:
            self.edges_info = qd.types.struct(
                len_rest=gs.qd_float,
                stretch_compliance=gs.qd_float,
                stretch_relaxation=gs.qd_float,
                v1=gs.qd_int,
                v2=gs.qd_int,
            ).field(shape=(self.n_edges,), layout=qd.Layout.SOA)
        if self.n_inner_edges:
            self.inner_edges_info = qd.types.struct(
                len_rest=gs.qd_float,
                angle_rest=gs.qd_float,
                bending_compliance=gs.qd_float,
                bending_relaxation=gs.qd_float,
                v1=gs.qd_int,
                v2=gs.qd_int,
                v3=gs.qd_int,
                v4=gs.qd_int,
            ).field(shape=(self.n_inner_edges,), layout=qd.Layout.SOA)
        if self.n_elems:
            self.elems_info = qd.types.struct(
                vol_rest=gs.qd_float,
                volume_compliance=gs.qd_float,
                volume_relaxation=gs.qd_float,
                v1=gs.qd_int,
                v2=gs.qd_int,
                v3=gs.qd_int,
                v4=gs.qd_int,
            ).field(shape=(self.n_elems,), layout=qd.Layout.SOA)
        self.particles_render = qd.types.struct(pos=gs.qd_vec3, vel=gs.qd_vec3, active=gs.qd_bool).field(
            shape=(self.n_particles, self._B), layout=qd.Layout.SOA
        )
        self.vverts_info = qd.types.struct(
            support_idxs=qd.types.vector(self._n_vvert_supports, gs.qd_int),
            support_weights=qd.types.vector(self._n_vvert_supports, gs.qd_float),
        ).field(shape=(self.n_vverts,), layout=qd.Layout.SOA)
        self.vverts_render = qd.types.struct(pos=gs.qd_vec3, active=gs.qd_bool).field(
            shape=(self.n_vverts, self._B), layout=qd.Layout.SOA
        )
        self.vverts_uvs = qd.field(gs.qd_vec2, shape=(self.n_vverts,))
        self.vfaces_indices = qd.field(gs.qd_ivec3, shape=(self.n_vfaces,))
        self._errno = array_class.V(dtype=gs.qd_int, shape=(self._B,))
        self._momentum = qd.types.struct(
            mass=gs.qd_float,
            center=gs.qd_vec3,
            linear=gs.qd_vec3,
            angular=gs.qd_vec3,
            covariance=gs.qd_mat3,
            linear_target=gs.qd_vec3,
            angular_target=gs.qd_vec3,
            is_free=gs.qd_bool,
        ).field(shape=(self.n_entities, self._B), layout=qd.Layout.SOA)
        self._entities_particle_start = qd.field(gs.qd_int, shape=(self.n_entities,))
        self._entities_particle_start.from_numpy(
            np.array([entity.particle_start for entity in self.entities], dtype=gs.np_int)
        )
        self._entities_velocity_decay = qd.field(gs.qd_float, shape=(self.n_entities,))
        self._entities_velocity_decay.from_numpy(
            np.array(
                [
                    math.exp(-entity.material.damping * self.substep_dt)
                    if isinstance(entity.material, gs.materials.PBD.Elastic)
                    else 1.0
                    for entity in self.entities
                ],
                dtype=gs.np_float,
            )
        )
        self._graph_counter = qd.ndarray(gs.qd_int, shape=())
        particles_entity_idx = np.empty(self.n_particles, dtype=gs.np_int)
        for i_e, entity in enumerate(self.entities):
            entity._add_to_solver()
            particles_entity_idx[entity.particle_start : entity.particle_end] = i_e
        self.particles_info.entity_idx.from_numpy(particles_entity_idx)
        if self.n_inner_edges:
            kernel_init_bending(self.particles_info, self.inner_edges_info, self._errno)
        self._edges = torch.as_tensor(
            np.concatenate(tuple(entity.edges + entity.particle_start for entity in self.entities)), device=gs.device
        )
        self._inner_edges = torch.as_tensor(
            np.concatenate(
                (
                    np.empty((0, 4), dtype=gs.np_int),
                    *(
                        entity.inner_edges + entity.particle_start
                        for entity in self.entities
                        if isinstance(entity, PBD2DEntity)
                    ),
                )
            ),
            device=gs.device,
        )
        self._elems = torch.as_tensor(
            np.concatenate(
                (
                    np.empty((0, 4), dtype=gs.np_int),
                    *(
                        entity.elems + entity.particle_start
                        for entity in self.entities
                        if isinstance(entity, PBD3DEntity)
                    ),
                )
            ),
            device=gs.device,
        )
        self._lengths_rest = qd_to_torch(self.edges_info.len_rest, transpose=True, copy=True)
        self._angles_rest = (
            qd_to_torch(self.inner_edges_info.angle_rest, transpose=True, copy=True)
            if self.n_inner_edges
            else torch.empty(0, dtype=gs.tc_float, device=gs.device)
        )
        self._volumes_rest = (
            qd_to_torch(self.elems_info.vol_rest, transpose=True, copy=True)
            if self.n_elems
            else torch.empty(0, dtype=gs.tc_float, device=gs.device)
        )
        if self.n_elems and ((self._volumes_rest == 0.0).any() or not torch.isfinite(self._volumes_rest).all()):
            gs.raise_exception("PBD elastic rest tetrahedra must have nonzero volume.")
        rigid = self.scene.rigid_solver
        projection_geoms = [
            geom for entity in rigid.entities for link in entity.links for geom in link.geoms if geom.needs_coup
        ]
        if projection_geoms:
            if not any(geom.n_faces for geom in projection_geoms):
                gs.raise_exception("PBD rigid collision requires surface triangles.")
            self._rigid_surface = build_rigid_surface(rigid, projection_geoms)
            surface_geoms = [geom for geom in projection_geoms if geom.n_faces]
            self._contact_geoms_idx = tuple(geom.idx for geom in surface_geoms)
            contact_vertices = []
            for i_g, geom in enumerate(surface_geoms):
                vertices_idx = np.unique(geom.init_faces) + geom.vert_start
                contact_vertices.append(np.column_stack((vertices_idx, np.full(len(vertices_idx), i_g))))
            contact_vertices = np.concatenate(contact_vertices)
            self._contact_vertices = qd.field(gs.qd_ivec2, shape=(len(contact_vertices),))
            self._contact_vertices.from_numpy(contact_vertices)
            self._previous_geoms_pos = qd.field(gs.qd_vec3, shape=(len(surface_geoms), self._B))
            self._previous_geoms_quat = qd.field(gs.qd_vec4, shape=(len(surface_geoms), self._B))
        if self._rigid_surface is not None and self.n_elems:
            surface_faces = []
            surface_face_entities = []
            surface_triangles = []
            surface_entities = []
            for i_e, entity in enumerate(self.entities):
                if isinstance(entity, PBD3DEntity):
                    surface_entities.append(i_e)
                    faces, adjacent_tets, _ = igl.boundary_facets(entity.elems)
                    vertices = entity.init_particles
                    triangles = vertices[faces]
                    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                    centers = vertices[entity.elems[adjacent_tets]].mean(axis=1)
                    is_inward = np.einsum("ij,ij->i", normals, centers - triangles[:, 0]) > 0.0
                    faces[is_inward] = faces[is_inward][:, (0, 2, 1)]
                    surface_faces.append(faces + entity.particle_start)
                    surface_face_entities.append(np.full(len(faces), i_e))
                    surface_triangles.append(triangles)
            faces = np.concatenate(surface_faces)
            triangles = np.concatenate(surface_triangles)
            self._surface_faces = qd.field(gs.qd_ivec3, shape=(len(faces),))
            self._surface_faces.from_numpy(faces)
            self._surface_edge_adjacency = build_triangle_edge_adjacency(faces)
            if self._options.is_recording_edge_contacts:
                self._detected_edge_contacts = qd.field(
                    gs.qd_bool, shape=(len(self._surface_edge_adjacency.endpoints), self._B)
                )
            self._edge_contact_counts = qd.field(gs.qd_int, shape=(self.n_particles, self._B))
            rigid_faces = np.concatenate(tuple(geom.init_faces + geom.vert_start for geom in surface_geoms))
            self._contact_faces = qd.field(gs.qd_ivec3, shape=len(rigid_faces))
            self._contact_faces.from_numpy(rigid_faces)
            self._rigid_edge_adjacency = build_triangle_edge_adjacency(rigid_faces)
            face_starts = np.concatenate(
                (np.zeros(1, dtype=gs.np_int), np.cumsum([geom.n_faces for geom in surface_geoms]))
            )
            face_offsets = face_starts[:-1] - np.array([geom.face_start for geom in surface_geoms])
            self._contact_face_offsets = qd.field(gs.qd_int, shape=len(surface_geoms))
            self._contact_face_offsets.from_numpy(face_offsets)
            self._contact_bounds = qd.types.struct(lower=gs.qd_vec3, upper=gs.qd_vec3).field(shape=len(surface_geoms))
            self._contact_bounds.lower.from_numpy(
                np.stack(tuple(geom.init_verts.min(axis=0) for geom in surface_geoms))
            )
            self._contact_bounds.upper.from_numpy(
                np.stack(tuple(geom.init_verts.max(axis=0) for geom in surface_geoms))
            )
            self._collision_counts = qd.field(gs.qd_ivec3, shape=self._B)
            self._collision_depth = qd.field(gs.qd_float, shape=self._B)
            self._surface_face_entities = qd.field(gs.qd_int, shape=(len(faces),))
            self._surface_face_entities.from_numpy(np.concatenate(surface_face_entities))
            self._surface_bounds = qd.types.struct(lower=gs.qd_vec3, upper=gs.qd_vec3).field(
                shape=(self.n_entities, self._B)
            )
            surface_aabbs = AABB(n_batches=self._B, n_aabbs=len(faces))
            surface_aabbs.aabbs.min.from_numpy(np.tile(triangles.min(axis=1)[None], (self._B, 1, 1)))
            surface_aabbs.aabbs.max.from_numpy(np.tile(triangles.max(axis=1)[None], (self._B, 1, 1)))
            self._surface_bvh = RefitBVH(surface_aabbs, tuple(len(faces) for faces in surface_faces))
            roots = np.full(self.n_entities, -1, dtype=gs.np_int)
            roots[surface_entities] = qd_to_numpy(self._surface_bvh.roots_idx, transpose=True)
            self._surface_roots = qd.field(gs.qd_int, shape=self.n_entities)
            self._surface_roots.from_numpy(roots)
            # Each query type stores its count, visited nodes, and tested triangles independently.
            self._surface_query_counts = qd.field(qd.types.vector(6, gs.qd_int), shape=(len(contact_vertices), self._B))
        self._surface_info = (
            self._rigid_surface.info
            if self._rigid_surface is not None
            else array_class.RigidSurfaceInfo(None, None, None, None)
        )
        if self._options.is_recording_constraint_history:
            self._iteration_positions = torch.empty(
                (self._options.max_solver_iterations, 3, self._B, self.n_particles, 3),
                dtype=gs.tc_float,
                device=gs.device,
            )
        if self._options.is_recording_edge_contacts:
            self._edge_history_substep = qd.ndarray(gs.qd_int, shape=())
            self._edge_history_positions = qd.field(
                gs.qd_vec3, shape=(self.sim.substeps, self._options.max_solver_iterations, self.n_particles, self._B)
            )
            if self._detected_edge_contacts is not None:
                self._edge_history_masks = qd.field(
                    gs.qd_bool,
                    shape=(
                        self.sim.substeps,
                        self._options.max_solver_iterations,
                        self._detected_edge_contacts.shape[0],
                        self._B,
                    ),
                )
            if rigid.n_links:
                self._edge_history_links_pos = qd.field(gs.qd_vec3, shape=(self.sim.substeps, rigid.n_links, self._B))
                self._edge_history_links_quat = qd.field(gs.qd_vec4, shape=(self.sim.substeps, rigid.n_links, self._B))
        self.check_errno()
        self._store_contact_poses()

    def _store_contact_poses(self):
        if self._rigid_surface is not None:
            rigid = self.scene.rigid_solver
            if gs.use_zerocopy:
                previous_pos = qd_to_torch(self._previous_geoms_pos, transpose=True, copy=False)
                previous_quat = qd_to_torch(self._previous_geoms_quat, transpose=True, copy=False)
                previous_pos.copy_(rigid.get_geoms_pos(self._contact_geoms_idx))
                previous_quat.copy_(rigid.get_geoms_quat(self._contact_geoms_idx))
            else:
                kernel_store_contact_poses(rigid.dyn_state, self._surface_info, self)

    def process_input(self, in_backward=False):
        for entity in self.entities:
            entity.process_input(in_backward=in_backward)
        self._store_contact_poses()

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def substep_pre_coupling(self, f):
        if not self.is_active:
            return
        if self._edge_history_substep is not None:
            # Captured graphs read changing indices from device storage on each launch.
            if gs.use_zerocopy:
                substep = qd_to_torch(self._edge_history_substep, transpose=True, copy=False)
                substep.fill_(self.sim.cur_substep_global % self.sim.substeps)
            else:
                self._edge_history_substep.fill(self.sim.cur_substep_global % self.sim.substeps)
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()
        kernel_predict(
            self.sim.cur_t,
            self.substep_dt,
            self.particles,
            self.particles_ng,
            self.particles_info,
            self._gravity,
            self,
            self._errno,
        )
        self.update_contact_bvh()
        rigid = self.scene.rigid_solver
        kernel_solve_constraints(
            self._iteration_positions if self._iteration_positions is not None else self._graph_counter,
            rigid.dyn_state,
            rigid.dyn_info,
            rigid.rigid_info,
            rigid.collider._collider_info,
            self._surface_info,
            self,
        )
        self.check_errno()

    def project_elastic_constraints(self):
        """Apply one simultaneous stretch, bending, and volume correction at the current positions."""
        kernel_project_elastic_constraints(self)

    def project_collision(self):
        """Apply simultaneous vertex-face and domain corrections, then average edge-edge corrections per vertex."""
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()
        self.update_contact_bvh()
        rigid = self.scene.rigid_solver
        kernel_project_collision(
            rigid.dyn_state,
            rigid.dyn_info,
            rigid.rigid_info,
            rigid.collider._collider_info,
            self._surface_info,
            self,
        )

    def update_contact_bvh(self):
        """Check query work periodically and retain a spatial rebuild when it reduces the current traversal work."""
        if self._surface_bvh is None:
            return
        self._surface_query_updates += 1
        if self._surface_query_updates % self._options.bvh_rebuild_check_interval:
            return
        counts = tensor_to_array(self.get_contact_query_counts())
        work = counts[:, 1:].sum(axis=1) / np.maximum(counts[:, 0], 1)
        if self._surface_query_baseline is None:
            self._surface_query_baseline = work
            return
        is_populated = self._surface_query_baseline > 0
        is_degraded = is_populated & (work > self._options.bvh_rebuild_threshold * self._surface_query_baseline)
        self._surface_query_baseline[~is_populated] = work[~is_populated]
        if not is_degraded.any():
            return
        rigid = self.scene.rigid_solver
        # Evaluate both orders on the same positions; changing contact populations alone can increase query work.
        kernel_measure_surface_queries(rigid.dyn_state, rigid.dyn_info, self._surface_info, self)
        before = tensor_to_array(self.get_contact_query_counts())
        leaves_idx = qd_to_torch(self._surface_bvh.leaves_idx, transpose=True, copy=True).T.contiguous()
        self._surface_bvh.rebuild()
        kernel_measure_surface_queries(rigid.dyn_state, rigid.dyn_info, self._surface_info, self)
        after = tensor_to_array(self.get_contact_query_counts())
        # Require a margin while keeping both work categories bounded to avoid a traversal-order tradeoff.
        is_improved = (after[:, 1:].sum(axis=0) <= before[:, 1:].sum(axis=0)).all()
        is_improved = is_improved and after[:, 1:].sum() < 0.9 * before[:, 1:].sum()
        if not is_improved:
            self._surface_bvh.set_leaves(leaves_idx)
        # Establish the baseline at the regular sampling phase, matching the work used to trigger a trial.
        self._surface_query_baseline = None if is_improved else work
        gs.logger.debug(
            f"PBD surface BVH rebuild: accepted={is_improved}, "
            f"node tests={before[:, 1].sum()} -> {after[:, 1].sum()}, "
            f"triangle tests={before[:, 2].sum()} -> {after[:, 2].sum()}."
        )

    def get_contact_query_counts(self):
        """Return [ray/proximity, query/node/triangle] counts summed over environments for the last surface query pass."""
        if self._surface_query_counts is None:
            return torch.zeros((2, 3), dtype=gs.tc_int, device=gs.device)
        return qd_to_torch(self._surface_query_counts, transpose=True).sum(dim=(0, 1)).reshape(2, 3)

    def substep_post_coupling(self, f):
        self._store_contact_poses()

    def check_errno(self):
        if self._errno is None:
            return
        errno = np.bitwise_or.reduce(qd_to_numpy(self._errno, transpose=True))
        if errno & (ErrorCode.INVALID_PBD_STATE | ErrorCode.INVALID_CONTACT_NAN):
            gs.raise_exception(
                "PBDUnifiedSolver encountered a non-finite state or a degenerate or immovable constraint."
            )
        if errno & ErrorCode.INVALID_PBD_VOLUME:
            gs.raise_exception("PBDUnifiedSolver has a collapsed or inverted tetrahedron with every vertex fixed.")

    def get_state(self, f):
        if not self.is_active:
            return None
        state = PBDUnifiedSolverState(self.scene)
        state.pos.copy_(qd_to_torch(self.particles.pos, transpose=True))
        state.vel.copy_(qd_to_torch(self.particles.vel, transpose=True))
        state.free.copy_(qd_to_torch(self.particles.free, transpose=True))
        state.active = qd_to_torch(self.particles_ng.active, transpose=True, copy=True)
        return state

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def set_state(self, f, state, envs_idx=None):
        if not self.is_active:
            return
        self._surface_query_baseline = None
        self._surface_query_updates = 0
        envs_idx = self.scene._sanitize_envs_idx(envs_idx)
        particles_idx = torch.arange(self.n_particles, device=gs.device).repeat(len(envs_idx), 1)
        for field, value in (
            (self.particles.pos, state.pos),
            (self.particles.ipos, state.pos),
            (self.particles.pos_iter, state.pos),
            (self.particles.vel, state.vel),
            (self.particles.free, state.free),
            (self.particles_ng.active, state.active),
        ):
            self.set_particle_field(field, value[envs_idx].contiguous(), particles_idx, envs_idx)
        self.set_particle_field(self.particles.dpos, torch.zeros_like(state.pos[envs_idx]), particles_idx, envs_idx)
        if gs.use_zerocopy:
            errno = qd_to_torch(self._errno, transpose=True, copy=False)
            errno[envs_idx] = 0
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            kernel_reset_errno(envs_idx, self._errno)

    def set_particle_field(self, field, values, particles_idx, envs_idx):
        """Write a scalar or three-component particle field using environment-local indices."""
        if gs.use_zerocopy:
            view = qd_to_torch(field, transpose=True, copy=False)
            view[envs_idx[:, None], particles_idx] = values
        else:
            kernel_set_field(particles_idx, envs_idx, values, field)

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def _kernel_set_particles_pos(self, particles_idx, envs_idx, poss):
        self.set_particle_field(self.particles.pos, poss, particles_idx, envs_idx)
        self.set_particle_field(self.particles.ipos, poss, particles_idx, envs_idx)
        self.set_particle_field(self.particles.pos_iter, poss, particles_idx, envs_idx)
        self.set_particle_field(self.particles.vel, torch.zeros_like(poss), particles_idx, envs_idx)
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def _kernel_get_particles_pos(self, particle_start, n_particles, envs_idx, poss):
        poss.copy_(
            qd_to_torch(
                self.particles.pos, envs_idx, slice(particle_start, particle_start + n_particles), transpose=True
            )
        )

    @mutates(StateChange.DYNAMICS)
    def _kernel_set_particles_vel(self, particles_idx, envs_idx, vels):
        self.set_particle_field(self.particles.vel, vels, particles_idx, envs_idx)
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def _kernel_get_particles_vel(self, particle_start, n_particles, envs_idx, vels):
        vels.copy_(
            qd_to_torch(
                self.particles.vel, envs_idx, slice(particle_start, particle_start + n_particles), transpose=True
            )
        )

    @mutates(StateChange.GEOMETRY)
    def _kernel_set_particles_active(self, particles_idx, envs_idx, actives):
        self.set_particle_field(self.particles_ng.active, actives, particles_idx, envs_idx)
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def _kernel_get_particles_active(self, particle_start, n_particles, envs_idx, actives):
        actives.copy_(
            qd_to_torch(
                self.particles_ng.active, envs_idx, slice(particle_start, particle_start + n_particles), transpose=True
            )
        )

    def _kernel_fix_particles(self, particles_idx, envs_idx):
        self.set_particle_field(
            self.particles.free, torch.zeros_like(particles_idx, dtype=gs.tc_bool), particles_idx, envs_idx
        )
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def _kernel_release_particle(self, particles_idx, envs_idx):
        self.set_particle_field(
            self.particles.free, torch.ones_like(particles_idx, dtype=gs.tc_bool), particles_idx, envs_idx
        )
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def _kernel_get_mass(self, particle_start, n_particles, mass, envs_idx):
        mass[:] = qd_to_torch(
            self.particles_info.mass, slice(particle_start, particle_start + n_particles), transpose=True
        ).sum()

    def get_embedded_positions(self, elements_idx, barycentric, envs_idx=None):
        """Return tetrahedral material points with shape [B, n_points, 3] for the selected environments."""
        envs_idx = self.scene._sanitize_envs_idx(envs_idx)
        vertices = self._elems[indices_to_mask(elements_idx)]
        positions = qd_to_torch(self.particles.pos, envs_idx, transpose=True)
        return (positions[:, vertices] * barycentric[None, :, :, None]).sum(dim=-2)

    def get_constraint_residuals(self, positions=None, envs_idx=None):
        """Compute signed stretch/volume ratios and bending angles from positions with trailing shape [N, 3]."""
        if positions is None:
            envs_idx = self.scene._sanitize_envs_idx(envs_idx)
            positions = qd_to_torch(self.particles.pos, envs_idx, transpose=True)
        edges = positions[..., self._edges, :]
        stretch = torch.linalg.vector_norm(edges[..., 0, :] - edges[..., 1, :], dim=-1) / self._lengths_rest - 1.0
        tetrahedra = positions[..., self._elems, :]
        volume = torch.linalg.det(tetrahedra[..., 1:, :] - tetrahedra[..., :1, :]) / (6.0 * self._volumes_rest) - 1.0
        points = positions[..., self._inner_edges, :]
        edge = points[..., 1, :] - points[..., 0, :]
        normal1 = torch.linalg.cross(edge, points[..., 2, :] - points[..., 0, :])
        normal2 = torch.linalg.cross(edge, points[..., 3, :] - points[..., 0, :])
        angle = torch.atan2(
            (edge * torch.linalg.cross(normal1, normal2)).sum(dim=-1) / torch.linalg.vector_norm(edge, dim=-1),
            (normal1 * normal2).sum(dim=-1),
        )
        bending = torch.atan2(torch.sin(angle - self._angles_rest), torch.cos(angle - self._angles_rest))
        return PBDConstraintResiduals(stretch, bending, volume)

    def get_constraint_history(self):
        """Return residuals with shape [iteration, phase, B, constraint]; phases are input, elastic, and contact."""
        if self._iteration_positions is None:
            gs.raise_exception("Constraint history requires is_recording_constraint_history=True.")
        return self.get_constraint_residuals(self._iteration_positions)

    def get_collision_diagnostics(self):
        """Measure current surface crossings and edge contact gaps for every environment without moving particles.

        Counts exclude coplanar and endpoint touching. Unpaired intersections have crossing edges on only one side.
        All violating edge pairs contribute to diagnostics; each elastic edge responds to its first pair per iteration.
        Edge penetration measures the closest-point gap of crossed edge pairs, rather than solid penetration depth.
        Geometry queries add the cost of a collision detection pass and synchronize the device to check errors.
        """
        if self._surface_edge_adjacency is None:
            counts = torch.zeros((self._B, 3), dtype=gs.tc_int, device=gs.device)
            depth = torch.zeros(self._B, dtype=gs.tc_float, device=gs.device)
        else:
            if gs.use_zerocopy and gs.backend == gs.metal:
                torch.mps.synchronize()
            rigid = self.scene.rigid_solver
            kernel_collision_diagnostics(rigid.dyn_state, rigid.dyn_info, self._surface_info, self)
            self.check_errno()
            counts = qd_to_torch(self._collision_counts, transpose=True, copy=True)
            depth = qd_to_torch(self._collision_depth, transpose=True, copy=True)
        return PBDCollisionDiagnostics(counts[:, 0], counts[:, 1], counts[:, 2], depth)

    def get_edge_contact_history(self):
        """Copy each iteration's state after point/face projection and before edge projection in the latest step.

        Edge flags have shape [substep, iteration, B, edge]; edges contains particle index pairs [edge, 2].
        Rigid link poses have shape [substep, B, link, 3/4]. Enable is_recording_edge_contacts before scene.build().
        """
        if not self._options.is_recording_edge_contacts:
            gs.raise_exception("Edge contact history requires is_recording_edge_contacts=True.")
        positions = qd_to_torch(self._edge_history_positions, transpose=True, copy=True).movedim(0, 2)
        if self._edge_history_masks is None:
            is_detected = torch.zeros(positions.shape[:3] + (0,), dtype=gs.tc_bool, device=gs.device)
            edges = torch.empty((0, 2), dtype=gs.tc_int, device=gs.device)
        else:
            is_detected = qd_to_torch(self._edge_history_masks, transpose=True, copy=True).movedim(0, 2)
            edges = self._surface_edge_adjacency.endpoints.clone()
        if self._edge_history_links_pos is None:
            links_pos = torch.empty((self.sim.substeps, self._B, 0, 3), dtype=gs.tc_float, device=gs.device)
            links_quat = torch.empty((self.sim.substeps, self._B, 0, 4), dtype=gs.tc_float, device=gs.device)
        else:
            links_pos = qd_to_torch(self._edge_history_links_pos, transpose=True, copy=True).movedim(0, 1)
            links_quat = qd_to_torch(self._edge_history_links_quat, transpose=True, copy=True).movedim(0, 1)
        return PBDEdgeContactHistory(positions, is_detected, edges, links_pos, links_quat)

    def get_visual_vertex_positions(self, positions):
        """Interpolate visual mesh vertices from particle positions with trailing shape [particle, 3]."""
        supports = qd_to_torch(self.vverts_info.support_idxs, transpose=True)
        weights = qd_to_torch(self.vverts_info.support_weights, transpose=True)
        return (positions[..., supports, :] * weights[..., None]).sum(dim=-2)

    def update_render_fields(self):
        if self.is_active:
            kernel_update_render(
                self.particles, self.particles_ng, self.particles_render, self.vverts_render, self.vverts_info
            )

    def get_state_render(self):
        if not self.is_active:
            return None, None, None
        self.update_render_fields()
        return self.vverts_render.pos, self.vverts_uvs, self.vfaces_indices

    def get_tetrahedral_state_render(self, f):
        return self.particles.pos

    def reset_grad(self):
        pass

    def process_input_grad(self):
        pass

    def collect_output_grads(self):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    @property
    def is_active(self):
        return bool(self.entities)

    @property
    def n_particles(self):
        return sum(entity.n_particles for entity in self.entities)

    @property
    def n_edges(self):
        return sum(entity.n_edges for entity in self.entities)

    @property
    def n_inner_edges(self):
        return sum(entity.n_inner_edges for entity in self.entities if isinstance(entity, PBD2DEntity))

    @property
    def n_elems(self):
        return sum(entity.n_elems for entity in self.entities if isinstance(entity, PBD3DEntity))

    @property
    def n_vverts(self):
        return sum(entity.n_vverts for entity in self.entities)

    @property
    def n_vfaces(self):
        return sum(entity.n_vfaces for entity in self.entities)

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self.particle_size / 2.0
