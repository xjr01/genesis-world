import math
from dataclasses import dataclass

import numpy as np
import torch

import quadrants as qd

import genesis as gs
from genesis.engine.boundaries import (
    CubeBoundary,
    StaticCollider,
    create_static_collider,
    project_out_static_collider,
    static_collider_separates,
)
from genesis.engine.entities.ipbstf_entity import IPBSTFEntity
from genesis.engine.solvers.base_solver import Solver
from genesis.engine.states.solvers import IPBSTFSolverState
from genesis.utils import particle
from genesis.utils.array_class import ErrorCode
import genesis.utils.geom as gu
from genesis.utils.misc import (
    assign_indexed_tensor,
    broadcast_tensor,
    indices_to_mask,
    qd_to_numpy,
    qd_to_torch,
    sanitize_index,
)


@dataclass(frozen=True)
class _StaticColliderConfig:
    values: qd.template()


@qd.func
def _cubic_kernel(distance, support_radius):
    value = gs.qd_float(0.0)
    q = distance / support_radius
    coefficient = 8.0 / (math.pi * support_radius**3)
    if q < 0.5:
        value = coefficient * (6.0 * q * q * (q - 1.0) + 1.0)
    elif q < 1.0:
        value = 2.0 * coefficient * (1.0 - q) ** 3
    return value


@qd.func
def _cubic_kernel_first_derivative(distance, support_radius):
    value = gs.qd_float(0.0)
    q = distance / support_radius
    coefficient = 48.0 / (math.pi * support_radius**4)
    if q < 0.5:
        value = coefficient * q * (3.0 * q - 2.0)
    elif q < 1.0:
        value = -coefficient * (1.0 - q) ** 2
    return value


@qd.func
def _cubic_kernel_second_derivative(distance, support_radius):
    value = gs.qd_float(0.0)
    q = distance / support_radius
    coefficient = 96.0 / (math.pi * support_radius**5)
    if q < 0.5:
        value = coefficient * (3.0 * q - 1.0)
    elif q < 1.0:
        value = coefficient * (1.0 - q)
    return value


@qd.func
def _cubic_gradient_hessian(delta, support_radius):
    gradient = qd.Vector.zero(gs.qd_float, 3)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    distance = delta.norm()
    if distance < support_radius:
        second_derivative = _cubic_kernel_second_derivative(distance, support_radius)
        if distance > gs.EPS:
            direction = delta / distance
            first_derivative = _cubic_kernel_first_derivative(distance, support_radius)
            first_over_distance = first_derivative / distance
            gradient = first_derivative * direction
            hessian = first_over_distance * qd.Matrix.identity(gs.qd_float, 3)
            hessian += (second_derivative - first_over_distance) * direction.outer_product(direction)
        else:
            hessian = second_derivative * qd.Matrix.identity(gs.qd_float, 3)
    return gradient, hessian


@qd.func
def _hessian_column_norms(hessian):
    norms = qd.Vector.zero(gs.qd_float, 3)
    for column in qd.static(range(3)):
        norms[column] = qd.sqrt(
            hessian[0, column] * hessian[0, column]
            + hessian[1, column] * hessian[1, column]
            + hessian[2, column] * hessian[2, column]
        )
    return norms


@qd.func
def _is_separated_by_static_colliders(
    env_idx,
    pos_i,
    pos_j,
    particle_radius,
    static_colliders_pos,
    static_colliders_quat,
    static_colliders: _StaticColliderConfig,
):
    is_separated = False
    for collider_idx in qd.static(range(len(static_colliders.values))):
        if static_collider_separates(
            collider_idx,
            env_idx,
            pos_i,
            pos_j,
            particle_radius,
            static_colliders_pos,
            static_colliders_quat,
            static_colliders.values[collider_idx],
        ):
            is_separated = True
    return is_separated


@qd.func
def _project_out_static_colliders(
    env_idx,
    pos,
    static_colliders_pos,
    static_colliders_quat,
    static_colliders: _StaticColliderConfig,
):
    for collider_idx in qd.static(range(len(static_colliders.values))):
        pos = project_out_static_collider(
            collider_idx,
            env_idx,
            pos,
            static_colliders_pos,
            static_colliders_quat,
            static_colliders.values[collider_idx],
        )
    return pos


@qd.func
def _accumulate_density_constraint(
    particle_idx,
    env_idx,
    particle_radius,
    support_radius,
    particles,
    particles_status,
    particles_info,
    static_colliders_pos,
    static_colliders_quat,
    spatial_hasher: qd.template(),
    static_colliders: _StaticColliderConfig,
    is_fixed_influence_enabled: qd.template(),
):
    pos_i = particles[particle_idx, env_idx].pos
    mass_i = particles_info[particle_idx, env_idx].mass
    rho_rest = particles_info[particle_idx, env_idx].rho_rest
    density = mass_i * _cubic_kernel(0.0, support_radius)
    gradient = qd.Vector.zero(gs.qd_float, 3)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    grid = spatial_hasher.pos_to_grid(pos_i)
    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
        slot_idx = spatial_hasher.grid_to_slot(grid + offset)
        slot_start = spatial_hasher.slot_start[slot_idx, env_idx]
        slot_end = slot_start + spatial_hasher.slot_size[slot_idx, env_idx]
        for neighbor_idx in range(slot_start, slot_end):
            if (
                neighbor_idx != particle_idx
                and particles_status[neighbor_idx, env_idx].active
                and (is_fixed_influence_enabled or not particles_info[neighbor_idx, env_idx].is_fixed)
            ):
                pos_j = particles[neighbor_idx, env_idx].pos
                delta = pos_i - pos_j
                distance = delta.norm()
                if distance < support_radius and not _is_separated_by_static_colliders(
                    env_idx,
                    pos_i,
                    pos_j,
                    particle_radius,
                    static_colliders_pos,
                    static_colliders_quat,
                    static_colliders,
                ):
                    mass_j = particles_info[neighbor_idx, env_idx].mass
                    density += mass_j * _cubic_kernel(distance, support_radius)
                    kernel_gradient, kernel_hessian = _cubic_gradient_hessian(delta, support_radius)
                    gradient += mass_j / rho_rest * kernel_gradient
                    hessian += mass_j / rho_rest * kernel_hessian

    constraint = qd.max(density / rho_rest - 1.0, 0.0)
    particles[particle_idx, env_idx].density_constraint = constraint
    particles[particle_idx, env_idx].density_gradient = gradient
    particles[particle_idx, env_idx].density_hessian_diag = _hessian_column_norms(hessian)
    particles[particle_idx, env_idx].density = density


@qd.func
def _accumulate_neighbor_density_energy(
    particle_idx,
    env_idx,
    particle_radius,
    support_radius,
    force,
    hessian,
    particles,
    particles_status,
    particles_info,
    static_colliders_pos,
    static_colliders_quat,
    spatial_hasher: qd.template(),
    static_colliders: _StaticColliderConfig,
):
    pos_i = particles[particle_idx, env_idx].pos
    mass_i = particles_info[particle_idx, env_idx].mass
    grid = spatial_hasher.pos_to_grid(pos_i)
    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
        slot_idx = spatial_hasher.grid_to_slot(grid + offset)
        slot_start = spatial_hasher.slot_start[slot_idx, env_idx]
        slot_end = slot_start + spatial_hasher.slot_size[slot_idx, env_idx]
        for neighbor_idx in range(slot_start, slot_end):
            if neighbor_idx != particle_idx and particles_status[neighbor_idx, env_idx].active:
                constraint = particles[neighbor_idx, env_idx].density_constraint
                pos_j = particles[neighbor_idx, env_idx].pos
                delta = pos_i - pos_j
                distance = delta.norm()
                if (
                    distance < support_radius
                    and not _is_separated_by_static_colliders(
                        env_idx,
                        pos_i,
                        pos_j,
                        particle_radius,
                        static_colliders_pos,
                        static_colliders_quat,
                        static_colliders,
                    )
                ):
                    rho_rest_j = particles_info[neighbor_idx, env_idx].rho_rest
                    kernel_gradient, kernel_hessian = _cubic_gradient_hessian(delta, support_radius)
                    gradient = mass_i / rho_rest_j * kernel_gradient
                    constraint_hessian = mass_i / rho_rest_j * kernel_hessian
                    hessian_diag = _hessian_column_norms(constraint_hessian)
                    force -= constraint * gradient
                    hessian += gradient.outer_product(gradient)
                    for axis in qd.static(range(3)):
                        hessian[axis, axis] += constraint * hessian_diag[axis]
    return force, hessian


@qd.kernel
def _kernel_reorder_particles(
    n_particles: qd.i32,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    particles_reordered: qd.template(),
    particles_status_reordered: qd.template(),
    particles_info_reordered: qd.template(),
    spatial_hasher: qd.template(),
):
    spatial_hasher.compute_reordered_idx(
        n_particles, particles.pos, particles_status.active, particles_status.reordered_idx
    )
    particles_status_reordered.active.fill(False)
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active:
            reordered_idx = particles_status[particle_idx, env_idx].reordered_idx
            particles_reordered[reordered_idx, env_idx] = particles[particle_idx, env_idx]
            particles_info_reordered[reordered_idx, env_idx] = particles_info[particle_idx]
            particles_status_reordered[reordered_idx, env_idx].active = True


@qd.kernel
def _kernel_copy_from_reordered(
    n_particles: qd.i32,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_reordered: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active:
            reordered_idx = particles_status[particle_idx, env_idx].reordered_idx
            particles[particle_idx, env_idx] = particles_reordered[reordered_idx, env_idx]


@qd.kernel
def _kernel_compute_density_constraints(
    n_particles: qd.i32,
    particle_radius: float,
    support_radius: float,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    static_colliders_pos: qd.Tensor,
    static_colliders_quat: qd.Tensor,
    spatial_hasher: qd.template(),
    static_colliders: _StaticColliderConfig,
    is_fixed_influence_enabled: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        # Boundary particles are fixed but carry density constraints as sources for neighboring fluid (see solver_math.md).
        if particles_status[particle_idx, env_idx].active:
            _accumulate_density_constraint(
                particle_idx,
                env_idx,
                particle_radius,
                support_radius,
                particles,
                particles_status,
                particles_info,
                static_colliders_pos,
                static_colliders_quat,
                spatial_hasher,
                static_colliders,
                is_fixed_influence_enabled,
            )


@qd.kernel
def _kernel_reduce_max_density(
    n_particles: qd.i32,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    max_density: qd.Tensor,
):
    for particle_idx in range(n_particles):
        if particles_status[particle_idx, 0].active and not particles_info[particle_idx, 0].is_fixed:
            qd.atomic_max(max_density[None], particles[particle_idx, 0].density)


@qd.kernel
def _kernel_set_particle_mass(n_particles: qd.i32, mass: float, particles_info: qd.template()):
    for particle_idx in range(n_particles):
        particles_info[particle_idx].mass = mass


@qd.kernel
def _kernel_predict_positions(
    n_particles: qd.i32,
    substep_dt: float,
    gravity: qd.Tensor,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            pos = particles[particle_idx, env_idx].pos
            pos_predicted = pos + substep_dt * particles[particle_idx, env_idx].vel
            pos_predicted += substep_dt * substep_dt * gravity[env_idx]
            particles[particle_idx, env_idx].pos_prev = pos
            particles[particle_idx, env_idx].pos_predicted = pos_predicted
            particles[particle_idx, env_idx].pos = pos_predicted


@qd.kernel
def _kernel_assemble_density_local_systems(
    n_particles: qd.i32,
    substep_dt: float,
    alpha: float,
    particle_radius: float,
    support_radius: float,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    static_colliders_pos: qd.Tensor,
    static_colliders_quat: qd.Tensor,
    spatial_hasher: qd.template(),
    static_colliders: _StaticColliderConfig,
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            particle = particles[particle_idx, env_idx]
            inertia = alpha * particles_info[particle_idx, env_idx].mass / (substep_dt * substep_dt)
            force = -inertia * (particle.pos - particle.pos_predicted)
            hessian = inertia * qd.Matrix.identity(gs.qd_float, 3)

            constraint = particle.density_constraint
            gradient = particle.density_gradient
            force -= constraint * gradient
            hessian += gradient.outer_product(gradient)
            for axis in qd.static(range(3)):
                hessian[axis, axis] += constraint * particle.density_hessian_diag[axis]

            force, hessian = _accumulate_neighbor_density_energy(
                particle_idx,
                env_idx,
                particle_radius,
                support_radius,
                force,
                hessian,
                particles,
                particles_status,
                particles_info,
                static_colliders_pos,
                static_colliders_quat,
                spatial_hasher,
                static_colliders,
            )
            particles[particle_idx, env_idx].local_force = force
            particles[particle_idx, env_idx].local_hessian = hessian


@qd.kernel
def _kernel_solve_local_systems(
    n_particles: qd.i32,
    hessian_determinant_epsilon: float,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    errno: qd.Tensor,
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            hessian = particles[particle_idx, env_idx].local_hessian
            determinant = hessian.determinant()
            delta_pos = qd.Vector.zero(gs.qd_float, 3)
            is_valid = not qd.math.isnan(determinant) and not qd.math.isinf(determinant)
            if is_valid and determinant >= hessian_determinant_epsilon:
                delta_pos = hessian.inverse() @ particles[particle_idx, env_idx].local_force
            for axis in qd.static(range(3)):
                is_valid = is_valid and not qd.math.isnan(delta_pos[axis]) and not qd.math.isinf(delta_pos[axis])
            if is_valid:
                particles[particle_idx, env_idx].delta_pos = delta_pos
            else:
                particles[particle_idx, env_idx].delta_pos = qd.Vector.zero(gs.qd_float, 3)
                errno[env_idx] = ErrorCode.INVALID_IPBSTF_STATE_NAN


@qd.kernel
def _kernel_apply_position_updates(
    n_particles: qd.i32,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    static_colliders_pos: qd.Tensor,
    static_colliders_quat: qd.Tensor,
    boundary: qd.template(),
    static_colliders: _StaticColliderConfig,
    errno: qd.Tensor,
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            pos = boundary.impose_pos(
                particles[particle_idx, env_idx].pos + 0.5 * particles[particle_idx, env_idx].delta_pos
            )
            pos = _project_out_static_colliders(
                env_idx, pos, static_colliders_pos, static_colliders_quat, static_colliders
            )
            is_valid = True
            for axis in qd.static(range(3)):
                is_valid = is_valid and not qd.math.isnan(pos[axis]) and not qd.math.isinf(pos[axis])
            if is_valid:
                particles[particle_idx, env_idx].pos = pos
            else:
                errno[env_idx] = ErrorCode.INVALID_IPBSTF_STATE_NAN


@qd.kernel
def _kernel_update_velocities(
    n_particles: qd.i32,
    substep_dt: float,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    errno: qd.Tensor,
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            velocity = (particles[particle_idx, env_idx].pos - particles[particle_idx, env_idx].pos_prev) / substep_dt
            is_valid = True
            for axis in qd.static(range(3)):
                is_valid = is_valid and not qd.math.isnan(velocity[axis]) and not qd.math.isinf(velocity[axis])
            if is_valid:
                particles[particle_idx, env_idx].vel = velocity
            else:
                errno[env_idx] = ErrorCode.INVALID_IPBSTF_STATE_NAN


@qd.kernel
def _kernel_set_static_colliders_pose(
    colliders_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    pos: qd.types.ndarray(),
    quat: qd.types.ndarray(),
    static_colliders_pos: qd.Tensor,
    static_colliders_quat: qd.Tensor,
):
    for env_idx_local, collider_idx_local in qd.ndrange(envs_idx.shape[0], colliders_idx.shape[0]):
        env_idx = envs_idx[env_idx_local]
        collider_idx = colliders_idx[collider_idx_local]
        for axis in qd.static(range(3)):
            static_colliders_pos[collider_idx, env_idx][axis] = pos[env_idx_local, collider_idx_local, axis]
        for axis in qd.static(range(4)):
            static_colliders_quat[collider_idx, env_idx][axis] = quat[env_idx_local, collider_idx_local, axis]


@qd.kernel
def _kernel_add_particles(
    particle_start: qd.i32,
    n_particles: qd.i32,
    active: qd.i32,
    is_fixed: qd.i32,
    rho_rest: float,
    mass: float,
    pos: qd.types.ndarray(),
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
):
    for particle_idx_local, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        particle_idx = particle_idx_local + particle_start
        particles_status[particle_idx, env_idx].active = qd.cast(active, gs.qd_bool)
        for axis in qd.static(range(3)):
            particles[particle_idx, env_idx].pos[axis] = pos[particle_idx_local, axis]
        particles[particle_idx, env_idx].pos_prev = particles[particle_idx, env_idx].pos
        particles[particle_idx, env_idx].pos_predicted = particles[particle_idx, env_idx].pos
        particles[particle_idx, env_idx].vel = qd.Vector.zero(gs.qd_float, 3)

    for particle_idx_local in range(n_particles):
        particle_idx = particle_idx_local + particle_start
        particles_info[particle_idx].mass = mass
        particles_info[particle_idx].rho_rest = rho_rest
        particles_info[particle_idx].is_fixed = qd.cast(is_fixed, gs.qd_bool)


@qd.kernel
def _kernel_set_state(
    n_particles: qd.i32,
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    active: qd.types.ndarray(),
    particles: qd.template(),
    particles_status: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        for axis in qd.static(range(3)):
            particles[particle_idx, env_idx].pos[axis] = pos[env_idx, particle_idx, axis]
            particles[particle_idx, env_idx].vel[axis] = vel[env_idx, particle_idx, axis]
        particles_status[particle_idx, env_idx].active = active[env_idx, particle_idx]


@qd.kernel
def _kernel_get_state(
    n_particles: qd.i32,
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    active: qd.types.ndarray(),
    particles: qd.template(),
    particles_status: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        for axis in qd.static(range(3)):
            pos[env_idx, particle_idx, axis] = particles[particle_idx, env_idx].pos[axis]
            vel[env_idx, particle_idx, axis] = particles[particle_idx, env_idx].vel[axis]
        active[env_idx, particle_idx] = particles_status[particle_idx, env_idx].active


@qd.kernel
def _kernel_update_render_fields(
    n_particles: qd.i32,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_render: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active:
            particles_render[particle_idx, env_idx].pos = particles[particle_idx, env_idx].pos
            particles_render[particle_idx, env_idx].vel = particles[particle_idx, env_idx].vel
        else:
            particles_render[particle_idx, env_idx].pos = gu.qd_nowhere()
        particles_render[particle_idx, env_idx].active = particles_status[particle_idx, env_idx].active


@qd.kernel
def _kernel_set_particles_pos(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
    particles: qd.template(),
):
    for particle_idx_local, env_idx_local in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        particle_idx = particles_idx[env_idx_local, particle_idx_local]
        env_idx = envs_idx[env_idx_local]
        for axis in qd.static(range(3)):
            particles[particle_idx, env_idx].pos[axis] = poss[env_idx_local, particle_idx_local, axis]
        particles[particle_idx, env_idx].vel = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def _kernel_get_particles_pos(
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
    particles: qd.template(),
):
    for particle_idx_local, env_idx_local in qd.ndrange(n_particles, envs_idx.shape[0]):
        particle_idx = particle_idx_local + particle_start
        env_idx = envs_idx[env_idx_local]
        for axis in qd.static(range(3)):
            poss[env_idx_local, particle_idx_local, axis] = particles[particle_idx, env_idx].pos[axis]


@qd.kernel
def _kernel_set_particles_vel(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    vels: qd.types.ndarray(),
    particles: qd.template(),
):
    for particle_idx_local, env_idx_local in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        particle_idx = particles_idx[env_idx_local, particle_idx_local]
        env_idx = envs_idx[env_idx_local]
        for axis in qd.static(range(3)):
            particles[particle_idx, env_idx].vel[axis] = vels[env_idx_local, particle_idx_local, axis]


@qd.kernel
def _kernel_get_particles_vel(
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    vels: qd.types.ndarray(),
    particles: qd.template(),
):
    for particle_idx_local, env_idx_local in qd.ndrange(n_particles, envs_idx.shape[0]):
        particle_idx = particle_idx_local + particle_start
        env_idx = envs_idx[env_idx_local]
        for axis in qd.static(range(3)):
            vels[env_idx_local, particle_idx_local, axis] = particles[particle_idx, env_idx].vel[axis]


@qd.kernel
def _kernel_set_particles_active(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    actives: qd.types.ndarray(),
    particles_status: qd.template(),
):
    for particle_idx_local, env_idx_local in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        particle_idx = particles_idx[env_idx_local, particle_idx_local]
        env_idx = envs_idx[env_idx_local]
        particles_status[particle_idx, env_idx].active = actives[env_idx_local, particle_idx_local]


@qd.kernel
def _kernel_get_particles_active(
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    actives: qd.types.ndarray(),
    particles_status: qd.template(),
):
    for particle_idx_local, env_idx_local in qd.ndrange(n_particles, envs_idx.shape[0]):
        particle_idx = particle_idx_local + particle_start
        env_idx = envs_idx[env_idx_local]
        actives[env_idx_local, particle_idx_local] = particles_status[particle_idx, env_idx].active


@qd.kernel
def _kernel_get_mass(
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    mass: qd.types.ndarray(),
    particles_status: qd.template(),
    particles_info: qd.template(),
):
    for env_idx_local in range(envs_idx.shape[0]):
        env_idx = envs_idx[env_idx_local]
        total = gs.qd_float(0.0)
        for particle_idx_local in range(n_particles):
            particle_idx = particle_idx_local + particle_start
            if particles_status[particle_idx, env_idx].active:
                total += particles_info[particle_idx].mass
        mass[env_idx_local] = total


class IPBSTFSolver(Solver):
    """Implicit position-based surface-tension fluid (IPBSTF) solver using parallel local Newton updates."""

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        self._alpha = options.alpha
        self._hessian_determinant_epsilon = options.hessian_determinant_epsilon
        self._particle_size = options.particle_size
        self._particle_radius = 0.5 * options.particle_size
        self._support_radius = options._support_radius
        self._max_solver_iterations = options.max_solver_iterations
        self._static_colliders = _StaticColliderConfig(
            tuple(create_static_collider(collider_options) for collider_options in options.static_colliders)
        )
        self._n_static_colliders = len(self._static_colliders.values)
        self._static_colliders_pos = None
        self._static_colliders_quat = None
        self._kernel_static_colliders_pos = None
        self._kernel_static_colliders_quat = None
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)

        self.sh = gu.SpatialHasher(cell_size=options.hash_grid_cell_size, grid_res=options._hash_grid_res)
        self.boundary = CubeBoundary(lower=self._lower_bound, upper=self._upper_bound)

        self._default_mass = 1.0
        self._material = None
        self._errno = None

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_entity(self, idx, material, morph, surface, name: str | None = None) -> IPBSTFEntity:
        entity = IPBSTFEntity(
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
        self.entities.append(entity)
        return entity

    def build(self):
        super().build()
        self._B = self._sim._B
        self._n_particles = self.n_particles

        if self._n_static_colliders > 0:
            self._static_colliders_pos = qd.field(gs.qd_vec3, shape=(self._n_static_colliders, self._B))
            self._static_colliders_quat = qd.field(gs.qd_vec4, shape=(self._n_static_colliders, self._B))
            colliders_pos = np.repeat(
                np.stack([collider.pos for collider in self._static_colliders.values])[:, None, :],
                repeats=self._B,
                axis=1,
            )
            colliders_quat = np.repeat(
                np.stack([collider.quat for collider in self._static_colliders.values])[:, None, :],
                repeats=self._B,
                axis=1,
            )
            self._static_colliders_pos.from_numpy(colliders_pos)
            self._static_colliders_quat.from_numpy(colliders_quat)
            self._kernel_static_colliders_pos = self._static_colliders_pos
            self._kernel_static_colliders_quat = self._static_colliders_quat

        if not self.is_active:
            return
        if gs.backend != gs.cuda:
            gs.raise_exception("IPBSTFSolver requires the CUDA backend.")
        if self.sim.requires_grad:
            gs.raise_exception("IPBSTFSolver does not support differentiable simulation.")

        liquid_entities = [
            entity for entity in self.entities if isinstance(entity.material, gs.materials.IPBSTF.Liquid)
        ]
        if not liquid_entities:
            gs.raise_exception("IPBSTFSolver requires at least one liquid entity.")
        self._material = liquid_entities[0].material
        for entity in liquid_entities[1:]:
            if entity.material.rho != self._material.rho:
                gs.raise_exception(
                    "All entities in one IPBSTFSolver must use the same rest density because the current solver is "
                    "single-phase."
                )

        self.sh.build(self._B)
        particle_state = qd.types.struct(
            pos=gs.qd_vec3,
            pos_prev=gs.qd_vec3,
            pos_predicted=gs.qd_vec3,
            delta_pos=gs.qd_vec3,
            vel=gs.qd_vec3,
            density=gs.qd_float,
            density_constraint=gs.qd_float,
            density_gradient=gs.qd_vec3,
            density_hessian_diag=gs.qd_vec3,
            local_force=gs.qd_vec3,
            local_hessian=gs.qd_mat3,
        )
        particle_status = qd.types.struct(reordered_idx=gs.qd_int, active=gs.qd_bool)
        particle_info = qd.types.struct(mass=gs.qd_float, rho_rest=gs.qd_float, is_fixed=gs.qd_bool)
        particle_render = qd.types.struct(pos=gs.qd_vec3, vel=gs.qd_vec3, active=gs.qd_bool)

        shape = (self._n_particles, self._B)
        self.particles = particle_state.field(shape=shape, layout=qd.Layout.SOA)
        self.particles_status = particle_status.field(shape=shape, layout=qd.Layout.SOA)
        self.particles_info = particle_info.field(shape=(self._n_particles,), layout=qd.Layout.SOA)
        self.particles_reordered = particle_state.field(shape=shape, layout=qd.Layout.SOA)
        self.particles_status_reordered = particle_status.field(shape=shape, layout=qd.Layout.SOA)
        self.particles_info_reordered = particle_info.field(shape=shape, layout=qd.Layout.SOA)
        self.particles_render = particle_render.field(shape=shape, layout=qd.Layout.SOA)
        self._max_density = qd.field(gs.qd_float, shape=())
        self._errno = qd.field(gs.qd_int, shape=(self._B,))

        if self._n_static_colliders == 0:
            self._kernel_static_colliders_pos = self.particles.pos
            self._kernel_static_colliders_quat = self.particles.pos
        for entity in self.entities:
            entity._add_to_solver()

        if any(entity.active for entity in liquid_entities):
            self._reorder_particles()
            self._compute_density_constraints(is_fixed_influence_enabled=False)
            self._max_density.fill(0.0)
            _kernel_reduce_max_density(
                self._n_particles,
                self.particles_reordered,
                self.particles_status_reordered,
                self.particles_info_reordered,
                self._max_density,
            )
            max_density = qd_to_numpy(self._max_density, transpose=True)[()]
        else:
            reference_half_extent = self._support_radius + 2.0 * self._particle_size
            reference_particles = particle.box_to_particles(
                p_size=self._particle_size,
                size=(2.0 * reference_half_extent,) * 3,
                sampler="staggered",
            )
            center_idx = np.linalg.norm(reference_particles, axis=1).argmin()
            distances = np.linalg.norm(reference_particles - reference_particles[center_idx], axis=1)
            q = distances / self._support_radius
            weights = np.zeros_like(distances)
            coefficient = 8.0 / (math.pi * self._support_radius**3)
            is_inner = q < 0.5
            is_outer = (q >= 0.5) & (q < 1.0)
            weights[is_inner] = coefficient * (6.0 * q[is_inner] ** 2 * (q[is_inner] - 1.0) + 1.0)
            weights[is_outer] = 2.0 * coefficient * (1.0 - q[is_outer]) ** 3
            max_density = weights.sum()

        if max_density <= gs.EPS:
            gs.raise_exception("IPBSTF particle mass calibration requires a positive reference density.")
        self._default_mass = float(self._material.rho / max_density)
        _kernel_set_particle_mass(self._n_particles, self._default_mass, self.particles_info)
        self._reorder_particles()
        self._compute_density_constraints()

    def _reorder_particles(self):
        _kernel_reorder_particles(
            self._n_particles,
            self.particles,
            self.particles_status,
            self.particles_info,
            self.particles_reordered,
            self.particles_status_reordered,
            self.particles_info_reordered,
            self.sh,
        )

    def _compute_density_constraints(self, is_fixed_influence_enabled=True):
        _kernel_compute_density_constraints(
            self._n_particles,
            self._particle_radius,
            self._support_radius,
            self.particles_reordered,
            self.particles_status_reordered,
            self.particles_info_reordered,
            self._kernel_static_colliders_pos,
            self._kernel_static_colliders_quat,
            self.sh,
            self._static_colliders,
            is_fixed_influence_enabled,
        )

    @gs.assert_built
    def set_static_colliders_pose(self, pos, quat, colliders_idx=None, envs_idx=None):
        """Set positions and w-x-y-z orientations of selected one-way static colliders."""
        if self._n_static_colliders == 0:
            gs.raise_exception("Cannot set IPBSTF static collider poses because the scene has no static colliders.")

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        colliders_idx = sanitize_index(colliders_idx, -1, self._n_static_colliders, 1, "colliders_idx")
        pos = broadcast_tensor(
            pos, gs.tc_float, (len(envs_idx), len(colliders_idx), 3), ("envs_idx", "colliders_idx", "")
        ).contiguous()
        quat = broadcast_tensor(
            quat, gs.tc_float, (len(envs_idx), len(colliders_idx), 4), ("envs_idx", "colliders_idx", "")
        ).contiguous()
        quat_norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        if (quat_norm <= gs.EPS).any():
            gs.raise_exception("IPBSTF static collider quaternions must be non-zero.")
        quat = quat / quat_norm

        if gs.use_zerocopy:
            colliders_pos = qd_to_torch(self._static_colliders_pos, transpose=True, copy=False)
            colliders_quat = qd_to_torch(self._static_colliders_quat, transpose=True, copy=False)
            mask = indices_to_mask(envs_idx, colliders_idx)
            assign_indexed_tensor(colliders_pos, mask, pos, ("envs_idx", "colliders_idx", ""))
            assign_indexed_tensor(colliders_quat, mask, quat, ("envs_idx", "colliders_idx", ""))
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            _kernel_set_static_colliders_pose(
                colliders_idx,
                envs_idx,
                pos,
                quat,
                self._kernel_static_colliders_pos,
                self._kernel_static_colliders_quat,
            )

    def process_input(self, in_backward=False):
        for entity in self.entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        pass

    def substep_pre_coupling(self, f):
        if not self.is_active:
            return

        self._reorder_particles()
        _kernel_predict_positions(
            self._n_particles,
            self._substep_dt,
            self._gravity,
            self.particles_reordered,
            self.particles_status_reordered,
            self.particles_info_reordered,
        )
        _kernel_copy_from_reordered(self._n_particles, self.particles, self.particles_status, self.particles_reordered)

        for _ in range(self._max_solver_iterations):
            self._reorder_particles()
            self._compute_density_constraints()
            _kernel_assemble_density_local_systems(
                self._n_particles,
                self._substep_dt,
                self._alpha,
                self._particle_radius,
                self._support_radius,
                self.particles_reordered,
                self.particles_status_reordered,
                self.particles_info_reordered,
                self._kernel_static_colliders_pos,
                self._kernel_static_colliders_quat,
                self.sh,
                self._static_colliders,
            )
            _kernel_solve_local_systems(
                self._n_particles,
                self._hessian_determinant_epsilon,
                self.particles_reordered,
                self.particles_status_reordered,
                self.particles_info_reordered,
                self._errno,
            )
            _kernel_apply_position_updates(
                self._n_particles,
                self.particles_reordered,
                self.particles_status_reordered,
                self.particles_info_reordered,
                self._kernel_static_colliders_pos,
                self._kernel_static_colliders_quat,
                self.boundary,
                self._static_colliders,
                self._errno,
            )
            _kernel_copy_from_reordered(
                self._n_particles, self.particles, self.particles_status, self.particles_reordered
            )

        self._reorder_particles()
        _kernel_update_velocities(
            self._n_particles,
            self._substep_dt,
            self.particles_reordered,
            self.particles_status_reordered,
            self.particles_info_reordered,
            self._errno,
        )

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if self.is_active:
            _kernel_copy_from_reordered(
                self._n_particles, self.particles, self.particles_status, self.particles_reordered
            )

    def substep_post_coupling_grad(self, f):
        pass

    def collect_output_grads(self):
        pass

    def add_grad_from_state(self, state):
        pass

    def reset_grad(self):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    def check_errno(self):
        errno = np.bitwise_or.reduce(qd_to_numpy(self._errno, transpose=True))
        if errno & ErrorCode.INVALID_IPBSTF_STATE_NAN:
            gs.raise_exception(
                "IPBSTF produced a non-finite position or velocity. Increase alpha, reduce the time step, or increase "
                "the particle resolution."
            )

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            _kernel_set_state(
                self._n_particles, state.pos, state.vel, state.active, self.particles, self.particles_status
            )

    def get_state(self, f):
        if not self.is_active:
            return None
        state = IPBSTFSolverState(self.scene)
        _kernel_get_state(self._n_particles, state.pos, state.vel, state.active, self.particles, self.particles_status)
        return state

    def update_render_fields(self):
        _kernel_update_render_fields(self._n_particles, self.particles, self.particles_status, self.particles_render)

    def _kernel_add_particles(self, f, active, particle_start, n_particles, is_fixed, pos):
        _kernel_add_particles(
            particle_start,
            n_particles,
            active,
            is_fixed,
            self._material.rho,
            self._default_mass,
            pos,
            self.particles,
            self.particles_status,
            self.particles_info,
        )

    def _kernel_set_particles_pos(self, particles_idx, envs_idx, poss):
        _kernel_set_particles_pos(particles_idx, envs_idx, poss, self.particles)

    def _kernel_get_particles_pos(self, particle_start, n_particles, envs_idx, poss):
        _kernel_get_particles_pos(particle_start, n_particles, envs_idx, poss, self.particles)

    def _kernel_set_particles_vel(self, particles_idx, envs_idx, vels):
        _kernel_set_particles_vel(particles_idx, envs_idx, vels, self.particles)

    def _kernel_get_particles_vel(self, particle_start, n_particles, envs_idx, vels):
        _kernel_get_particles_vel(particle_start, n_particles, envs_idx, vels, self.particles)

    def _kernel_set_particles_active(self, particles_idx, envs_idx, actives):
        _kernel_set_particles_active(particles_idx, envs_idx, actives, self.particles_status)

    def _kernel_get_particles_active(self, particle_start, n_particles, envs_idx, actives):
        _kernel_get_particles_active(particle_start, n_particles, envs_idx, actives, self.particles_status)

    def _kernel_get_mass(self, particle_start, n_particles, mass, envs_idx):
        _kernel_get_mass(particle_start, n_particles, envs_idx, mass, self.particles_status, self.particles_info)

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        return sum(entity.n_particles for entity in self.entities)

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_radius

    @property
    def support_radius(self):
        return self._support_radius

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
