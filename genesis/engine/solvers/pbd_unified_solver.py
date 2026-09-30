from dataclasses import dataclass

import numpy as np
import torch

import quadrants as qd

import genesis as gs
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.boundaries.rigid_surface import RigidSurface, build_rigid_surface
from genesis.engine.entities.pbd_entity import PBD2DEntity, PBD3DEntity
from genesis.engine.states.solvers import PBDSolverState
from genesis.utils import array_class, geom, sdf
from genesis.utils.array_class import ErrorCode, V_ANNOTATION
from genesis.utils.misc import indices_to_mask, qd_to_numpy, qd_to_torch

from .base_solver import Solver, StateChange, mutates


@dataclass(frozen=True)
class PBDConstraintResiduals:
    """Signed relative edge lengths, dihedral angles in radians, and relative tetrahedron volumes."""

    stretch: torch.Tensor
    bending: torch.Tensor
    volume: torch.Tensor


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
        particles[i_p, i_b].contact_pos = particles[i_p, i_b].pos
        particles[i_p, i_b].dpos.fill(0.0)


@qd.func
def func_project_vertices(
    particles: V_ANNOTATION,
    particles_ng: V_ANNOTATION,
    bvh_nodes: V_ANNOTATION,
    bvh_morton_codes: V_ANNOTATION,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    surface_info: array_class.RigidSurfaceInfo,
    boundary: V_ANNOTATION,
    errno: qd.Tensor,
):
    for i_p, i_b in qd.ndrange(particles.shape[0], particles.shape[1]):
        if particles_ng[i_p, i_b].active and particles[i_p, i_b].free:
            projected = boundary.impose_pos(particles[i_p, i_b].pos)
            for i_g_ in range(surface_info.projection_geoms_idx.shape[0]):
                i_g = surface_info.projection_geoms_idx[i_g_]
                projected, _, _ = sdf.sdf_func_project_vertex_outside_geom(
                    i_g,
                    i_b,
                    projected,
                    bvh_nodes,
                    bvh_morton_codes,
                    dyn_state,
                    dyn_info,
                    rigid_info,
                    collider_info,
                    surface_info,
                )
            if qd.math.isnan(projected).any() or qd.math.isinf(projected).any():
                qd.atomic_or(errno[i_b], ErrorCode.INVALID_CONTACT_NAN)
            else:
                particles[i_p, i_b].pos = projected


@qd.func
def func_project_boundary(boundary: V_ANNOTATION, particles: V_ANNOTATION, particles_ng: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(particles.shape[0], particles.shape[1]):
        if particles_ng[i_p, i_b].active and particles[i_p, i_b].free:
            particles[i_p, i_b].pos = boundary.impose_pos(particles[i_p, i_b].pos)


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
                # Extrapolation repeats contact corrections with geometrically decreasing weights.
                acceleration = solver._options.constraint_acceleration
                weight = (1.0 - acceleration ** solver._graph_counter[()]) / (1.0 - acceleration)
                vel = weight * (solver.particles[i_p, i_b].pos - pos) / solver.substep_dt
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
            solver._momentum[i_e, i_b].linear = (
                solver._momentum[i_e, i_b].linear_target - solver._momentum[i_e, i_b].linear
            ) / denominator
            covariance = solver._momentum[i_e, i_b].covariance
            trace = covariance.trace()
            scale = trace if trace > 0.0 else 1.0
            inertia = (trace * qd.Matrix.identity(gs.qd_float, 3) - covariance) / scale
            angular_error = (solver._momentum[i_e, i_b].angular_target - solver._momentum[i_e, i_b].angular) / scale
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
def func_project_elastic_constraints(solver: V_ANNOTATION):
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
    func_apply_delta(solver._options.constraint_acceleration, solver.particles, solver.particles_ng, solver._errno)


@qd.kernel
def kernel_project_elastic_constraints(solver: V_ANNOTATION):
    func_project_elastic_constraints(solver)


@qd.func
def func_project_collision(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    surface_info: array_class.RigidSurfaceInfo,
    solver: V_ANNOTATION,
):
    if qd.static(not isinstance(solver._rigid_surface, RigidSurface)):
        func_project_boundary(solver.boundary, solver.particles, solver.particles_ng)
    else:
        func_project_vertices(
            solver.particles,
            solver.particles_ng,
            solver._rigid_surface.bvh.nodes,
            solver._rigid_surface.bvh.morton_codes,
            dyn_state,
            dyn_info,
            rigid_info,
            collider_info,
            surface_info,
            solver.boundary,
            solver._errno,
        )


@qd.kernel
def kernel_project_collision(
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    surface_info: array_class.RigidSurfaceInfo,
    solver: V_ANNOTATION,
):
    func_project_collision(dyn_state, dyn_info, rigid_info, collider_info, surface_info, solver)


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
    func_compute_momentum(solver, is_initial=True, is_contact=False)
    solver._graph_counter[()] = solver._options.max_solver_iterations
    while qd.graph.do_while(solver._graph_counter):
        if qd.static(solver._options.is_recording_constraint_history):
            for i_p, i_b in qd.ndrange(solver.n_particles, solver._B):
                i_iteration = solver._options.max_solver_iterations - solver._graph_counter[()]
                for i_axis in qd.static(range(3)):
                    history[i_iteration, 0, i_b, i_p, i_axis] = solver.particles[i_p, i_b].pos[i_axis]
        func_project_elastic_constraints(solver)
        if qd.static(solver._options.is_recording_constraint_history):
            for i_p, i_b in qd.ndrange(solver.n_particles, solver._B):
                i_iteration = solver._options.max_solver_iterations - solver._graph_counter[()]
                for i_axis in qd.static(range(3)):
                    history[i_iteration, 1, i_b, i_p, i_axis] = solver.particles[i_p, i_b].pos[i_axis]
        func_project_collision(dyn_state, dyn_info, rigid_info, collider_info, surface_info, solver)
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
    """Position-based dynamics (PBD) with simultaneous elastic corrections and per-iteration rigid vertex contact.

    Supports cloth and tetrahedral elastic entities. Rigid geoms with ``needs_coup=True`` supply prescribed
    boundaries. Each iteration reads a common position state for every elastic constraint, then projects vertices
    through the domain boundary and each rigid geom once. Separated contact regions keep successive projections
    compatible. Material compliance regularizes each projection denominator; the effective stiffness also depends on
    the iteration count and time step. Extrapolating consecutive iterates accelerates shape recovery, with the iteration
    history initialized after each substep's motion prediction.
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
        self._errno = None
        self._iteration_positions = None
        self._momentum = None
        self._entities_particle_start = None
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
            rigid.collider._sdf.activate()
            self._rigid_surface = build_rigid_surface(rigid, projection_geoms)
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
        self.check_errno()

    def process_input(self, in_backward=False):
        for entity in self.entities:
            entity.process_input(in_backward=in_backward)

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def substep_pre_coupling(self, f):
        if not self.is_active:
            return
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
        """Project free vertices through the domain boundary and each rigid geom once."""
        rigid = self.scene.rigid_solver
        kernel_project_collision(
            rigid.dyn_state,
            rigid.dyn_info,
            rigid.rigid_info,
            rigid.collider._collider_info,
            self._surface_info,
            self,
        )

    def substep_post_coupling(self, f):
        pass

    def check_errno(self):
        if self._errno is None:
            return
        errno = np.bitwise_or.reduce(qd_to_numpy(self._errno, transpose=True))
        if errno & (ErrorCode.INVALID_PBD_STATE | ErrorCode.INVALID_CONTACT_NAN):
            gs.raise_exception("PBDUnifiedSolver encountered a non-finite state or a degenerate bending constraint.")
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
