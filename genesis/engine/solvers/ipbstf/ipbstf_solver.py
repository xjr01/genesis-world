import math
from dataclasses import dataclass

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
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
    density_constraint_tolerance,
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
    is_line_search: qd.template(),
):
    pos_i = particles[particle_idx, env_idx].pos
    mass_i = particles_info[particle_idx, env_idx].mass
    rho_rest = particles_info[particle_idx, env_idx].rho_rest
    density = mass_i * _cubic_kernel(0.0, support_radius)
    gradient = qd.Vector.zero(gs.qd_float, 3)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    grid_pos = pos_i
    search_radius = 1
    if qd.static(is_line_search):
        grid_pos = particles[particle_idx, env_idx].pos_iteration
        search_radius = 2
    grid = spatial_hasher.pos_to_grid(grid_pos)
    for offset in qd.grouped(
        qd.ndrange(
            (-search_radius, search_radius + 1),
            (-search_radius, search_radius + 1),
            (-search_radius, search_radius + 1),
        )
    ):
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

    constraint = density / rho_rest - 1.0
    # Roundoff-scale residuals remain inactive so rigid translations preserve the zero-energy state.
    if constraint > density_constraint_tolerance:
        particles[particle_idx, env_idx].density_constraint = constraint
        particles[particle_idx, env_idx].density_gradient = gradient
        particles[particle_idx, env_idx].density_hessian_diag = _hessian_column_norms(hessian)
    else:
        particles[particle_idx, env_idx].density_constraint = 0.0
        particles[particle_idx, env_idx].density_gradient = qd.Vector.zero(gs.qd_float, 3)
        particles[particle_idx, env_idx].density_hessian_diag = qd.Vector.zero(gs.qd_float, 3)
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
                    constraint > 0.0
                    and distance < support_radius
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


@qd.func
def _assemble_density_local_system(
    particle_idx,
    env_idx,
    substep_dt,
    alpha,
    particle_radius,
    support_radius,
    particles,
    particles_status,
    particles_info,
    static_colliders_pos,
    static_colliders_quat,
    spatial_hasher: qd.template(),
    static_colliders: _StaticColliderConfig,
):
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

    return _accumulate_neighbor_density_energy(
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


@qd.func
def _solve_local_system(hessian_determinant_epsilon, force, hessian):
    delta_pos = qd.Vector.zero(gs.qd_float, 3)
    hessian_scale = gs.qd_float(0.0)
    for axis in qd.static(range(3)):
        hessian_scale = qd.max(hessian_scale, qd.abs(hessian[axis, axis]))
    is_valid = not qd.math.isnan(hessian_scale) and not qd.math.isinf(hessian_scale)
    if is_valid and hessian_scale > 0.0:
        delta_pos = force / hessian_scale
        hessian_normalized = hessian / hessian_scale
        determinant_normalized = hessian_normalized.determinant()
        is_valid = not qd.math.isnan(determinant_normalized) and not qd.math.isinf(determinant_normalized)
        if is_valid and determinant_normalized >= hessian_determinant_epsilon:
            delta_pos = hessian_normalized.inverse() @ (force / hessian_scale)
    for axis in qd.static(range(3)):
        is_valid = is_valid and not qd.math.isnan(delta_pos[axis]) and not qd.math.isinf(delta_pos[axis])
    return delta_pos, is_valid


@qd.func
def _limit_density_position_update(
    density_update_fraction,
    density_update_limit,
    surface_update_limit,
    density_constraint,
    delta_pos,
    density_gradient,
):
    distance = delta_pos.norm()
    max_distance = surface_update_limit
    constraint_change = density_gradient.dot(delta_pos)
    if density_constraint > 0.0 and constraint_change < 0.0:
        # Parallel neighbor corrections share the remaining compression budget.
        max_distance = qd.max(
            max_distance,
            density_update_fraction * density_constraint / -constraint_change * distance,
        )
    max_distance = qd.min(max_distance, density_update_limit)
    if distance > max_distance:
        delta_pos *= max_distance / distance
    return delta_pos


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
    density_constraint_tolerance: float,
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
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            _accumulate_density_constraint(
                particle_idx,
                env_idx,
                density_constraint_tolerance,
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
                False,
            )


@qd.kernel
def _kernel_compute_line_search_density_constraints(
    n_particles: qd.i32,
    density_constraint_tolerance: float,
    particle_radius: float,
    support_radius: float,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    static_colliders_pos: qd.Tensor,
    static_colliders_quat: qd.Tensor,
    line_search_state: qd.template(),
    spatial_hasher: qd.template(),
    static_colliders: _StaticColliderConfig,
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if (
            line_search_state[env_idx].is_active
            and particles_status[particle_idx, env_idx].active
            and not particles_info[particle_idx, env_idx].is_fixed
        ):
            _accumulate_density_constraint(
                particle_idx,
                env_idx,
                density_constraint_tolerance,
                particle_radius,
                support_radius,
                particles,
                particles_status,
                particles_info,
                static_colliders_pos,
                static_colliders_quat,
                spatial_hasher,
                static_colliders,
                True,
                True,
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
def _kernel_project_positions(
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
            pos = boundary.impose_pos(particles[particle_idx, env_idx].pos)
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
def _kernel_initialize_line_search(
    n_particles: qd.i32,
    iteration_idx: qd.i32,
    substep_dt: float,
    alpha: float,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    line_search_state: qd.template(),
    solver_iteration_energy: qd.template(),
):
    for env_idx in range(particles.shape[1]):
        line_search_state[env_idx].initial_energy = 0.0
        line_search_state[env_idx].candidate_energy = 0.0
        line_search_state[env_idx].accepted_energy = 0.0
        line_search_state[env_idx].directional_decrease = 0.0
        line_search_state[env_idx].step_size = 1.0
        line_search_state[env_idx].is_active = False

    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            particle = particles[particle_idx, env_idx]
            particles[particle_idx, env_idx].pos_iteration = particle.pos
            displacement = particle.pos - particle.pos_predicted
            inertia = alpha * particles_info[particle_idx, env_idx].mass / (substep_dt * substep_dt)
            line_search_state[env_idx].initial_energy += 0.5 * (
                inertia * displacement.norm_sqr() + particle.density_constraint * particle.density_constraint
            )

    for env_idx in range(particles.shape[1]):
        line_search_state[env_idx].accepted_energy = line_search_state[env_idx].initial_energy
        if iteration_idx == 0:
            solver_iteration_energy[0, env_idx] = line_search_state[env_idx].initial_energy
        else:
            line_search_state[env_idx].accepted_energy = solver_iteration_energy[iteration_idx, env_idx]
        line_search_state[env_idx].is_active = line_search_state[env_idx].initial_energy > 0.0


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
            force, hessian = _assemble_density_local_system(
                particle_idx,
                env_idx,
                substep_dt,
                alpha,
                particle_radius,
                support_radius,
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
    density_update_fraction: float,
    density_update_limit: float,
    surface_update_limit: float,
    hessian_determinant_epsilon: float,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    line_search_state: qd.template(),
    errno: qd.Tensor,
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            delta_pos, is_valid = _solve_local_system(
                hessian_determinant_epsilon,
                particles[particle_idx, env_idx].local_force,
                particles[particle_idx, env_idx].local_hessian,
            )
            delta_pos = _limit_density_position_update(
                density_update_fraction,
                density_update_limit,
                surface_update_limit,
                particles[particle_idx, env_idx].density_constraint,
                0.5 * delta_pos,
                particles[particle_idx, env_idx].density_gradient,
            )
            if is_valid:
                particles[particle_idx, env_idx].delta_pos = delta_pos
                line_search_state[env_idx].directional_decrease += qd.max(
                    0.0,
                    particles[particle_idx, env_idx].local_force.dot(delta_pos),
                )
            else:
                particles[particle_idx, env_idx].delta_pos = qd.Vector.zero(gs.qd_float, 3)
                errno[env_idx] = ErrorCode.INVALID_IPBSTF_STATE_NAN


@qd.kernel
def _kernel_compute_damping_positions(
    n_particles: qd.i32,
    substep_dt: float,
    damping_alpha: float,
    density_update_fraction: float,
    density_update_limit: float,
    surface_update_limit: float,
    hessian_determinant_epsilon: float,
    particle_radius: float,
    support_radius: float,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    static_colliders_pos: qd.Tensor,
    static_colliders_quat: qd.Tensor,
    spatial_hasher: qd.template(),
    boundary: qd.template(),
    static_colliders: _StaticColliderConfig,
    errno: qd.Tensor,
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            force, hessian = _assemble_density_local_system(
                particle_idx,
                env_idx,
                substep_dt,
                damping_alpha,
                particle_radius,
                support_radius,
                particles,
                particles_status,
                particles_info,
                static_colliders_pos,
                static_colliders_quat,
                spatial_hasher,
                static_colliders,
            )
            delta_pos, is_valid = _solve_local_system(hessian_determinant_epsilon, force, hessian)
            delta_pos = _limit_density_position_update(
                density_update_fraction,
                density_update_limit,
                surface_update_limit,
                particles[particle_idx, env_idx].density_constraint,
                0.5 * delta_pos,
                particles[particle_idx, env_idx].density_gradient,
            )
            pos_damping = boundary.impose_pos(particles[particle_idx, env_idx].pos + delta_pos)
            pos_damping = _project_out_static_colliders(
                env_idx, pos_damping, static_colliders_pos, static_colliders_quat, static_colliders
            )
            for axis in qd.static(range(3)):
                is_valid = is_valid and not qd.math.isnan(pos_damping[axis]) and not qd.math.isinf(pos_damping[axis])
            if is_valid:
                particles[particle_idx, env_idx].pos_damping = pos_damping
            else:
                particles[particle_idx, env_idx].pos_damping = particles[particle_idx, env_idx].pos
                errno[env_idx] = ErrorCode.INVALID_IPBSTF_STATE_NAN


@qd.kernel
def _kernel_apply_position_updates(
    n_particles: qd.i32,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    static_colliders_pos: qd.Tensor,
    static_colliders_quat: qd.Tensor,
    line_search_state: qd.template(),
    boundary: qd.template(),
    static_colliders: _StaticColliderConfig,
    errno: qd.Tensor,
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if (
            line_search_state[env_idx].is_active
            and particles_status[particle_idx, env_idx].active
            and not particles_info[particle_idx, env_idx].is_fixed
        ):
            pos = boundary.impose_pos(
                particles[particle_idx, env_idx].pos_iteration
                + line_search_state[env_idx].step_size * particles[particle_idx, env_idx].delta_pos
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
def _kernel_update_line_search(
    n_particles: qd.i32,
    substep_dt: float,
    alpha: float,
    reduction: float,
    energy_decrease_fraction: float,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    line_search_state: qd.template(),
):
    for env_idx in range(particles.shape[1]):
        if line_search_state[env_idx].is_active:
            line_search_state[env_idx].candidate_energy = 0.0

    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if (
            line_search_state[env_idx].is_active
            and particles_status[particle_idx, env_idx].active
            and not particles_info[particle_idx, env_idx].is_fixed
        ):
            particle = particles[particle_idx, env_idx]
            displacement = particle.pos - particle.pos_predicted
            inertia = alpha * particles_info[particle_idx, env_idx].mass / (substep_dt * substep_dt)
            line_search_state[env_idx].candidate_energy += 0.5 * (
                inertia * displacement.norm_sqr() + particle.density_constraint * particle.density_constraint
            )

    for env_idx in range(particles.shape[1]):
        if line_search_state[env_idx].is_active:
            if line_search_state[env_idx].candidate_energy < line_search_state[env_idx].accepted_energy - (
                energy_decrease_fraction
                * line_search_state[env_idx].step_size
                * line_search_state[env_idx].directional_decrease
            ):
                line_search_state[env_idx].accepted_energy = line_search_state[env_idx].candidate_energy
                line_search_state[env_idx].is_active = False
            else:
                line_search_state[env_idx].step_size *= reduction


@qd.kernel
def _kernel_restore_failed_line_search(
    n_particles: qd.i32,
    iteration_idx: qd.i32,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    line_search_state: qd.template(),
    solver_iteration_energy: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if (
            line_search_state[env_idx].is_active
            and particles_status[particle_idx, env_idx].active
            and not particles_info[particle_idx, env_idx].is_fixed
        ):
            particles[particle_idx, env_idx].pos = particles[particle_idx, env_idx].pos_iteration

    for env_idx in range(particles.shape[1]):
        solver_iteration_energy[iteration_idx + 1, env_idx] = line_search_state[env_idx].accepted_energy


@qd.kernel
def _kernel_update_velocities(
    n_particles: qd.i32,
    substep_dt: float,
    damping_beta: float,
    support_radius: float,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    damping_state: qd.template(),
    is_damping_enabled: qd.template(),
    errno: qd.Tensor,
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            velocity = (particles[particle_idx, env_idx].pos - particles[particle_idx, env_idx].pos_prev) / substep_dt
            if qd.static(is_damping_enabled) and damping_state[env_idx].is_active:
                velocity_damping = (
                    particles[particle_idx, env_idx].pos_damping - particles[particle_idx, env_idx].pos_prev
                ) / substep_dt
                speed_sq = velocity.norm_sqr()
                speed_damping_sq = velocity_damping.norm_sqr()
                position_delta = (
                    particles[particle_idx, env_idx].pos - particles[particle_idx, env_idx].pos_damping
                ).norm()
                if (
                    position_delta < damping_beta * support_radius
                    and speed_damping_sq < speed_sq
                    and speed_sq > gs.EPS
                ):
                    damping_weight = 1.0 - position_delta / (damping_beta * support_radius)
                    velocity *= qd.sqrt(
                        qd.max(0.0, 1.0 - damping_weight * (speed_sq - speed_damping_sq) / speed_sq)
                    )
            is_valid = True
            for axis in qd.static(range(3)):
                is_valid = is_valid and not qd.math.isnan(velocity[axis]) and not qd.math.isinf(velocity[axis])
            if is_valid:
                particles[particle_idx, env_idx].vel = velocity
            else:
                errno[env_idx] = ErrorCode.INVALID_IPBSTF_STATE_NAN


@qd.kernel
def _kernel_compute_damping_state(
    n_particles: qd.i32,
    damping_velocity_scale: float,
    support_radius: float,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    damping_state: qd.template(),
):
    for env_idx in range(particles.shape[1]):
        damping_state[env_idx].mass = 0.0
        damping_state[env_idx].mass_displacement_sqr = 0.0
        damping_state[env_idx].is_active = False

    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            mass = particles_info[particle_idx, env_idx].mass
            displacement = particles[particle_idx, env_idx].pos - particles[particle_idx, env_idx].pos_prev
            damping_state[env_idx].mass += mass
            damping_state[env_idx].mass_displacement_sqr += mass * displacement.norm_sqr()

    for env_idx in range(particles.shape[1]):
        displacement_limit = damping_velocity_scale * support_radius
        damping_state[env_idx].is_active = (
            damping_state[env_idx].mass > gs.EPS
            and damping_state[env_idx].mass_displacement_sqr
            < damping_state[env_idx].mass * displacement_limit * displacement_limit
        )


@qd.kernel
def _kernel_compute_viscosity_velocity_updates(
    n_particles: qd.i32,
    viscosity: float,
    support_radius: float,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    spatial_hasher: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            particle = particles[particle_idx, env_idx]
            delta_vel = qd.Vector.zero(gs.qd_float, 3)
            grid = spatial_hasher.pos_to_grid(particle.pos)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = spatial_hasher.grid_to_slot(grid + offset)
                slot_start = spatial_hasher.slot_start[slot_idx, env_idx]
                slot_end = slot_start + spatial_hasher.slot_size[slot_idx, env_idx]
                for neighbor_idx in range(slot_start, slot_end):
                    if (
                        neighbor_idx != particle_idx
                        and particles_status[neighbor_idx, env_idx].active
                        and not particles_info[neighbor_idx, env_idx].is_fixed
                    ):
                        neighbor = particles[neighbor_idx, env_idx]
                        distance = (particle.pos - neighbor.pos).norm()
                        if distance < support_radius and neighbor.density > gs.EPS:
                            delta_vel += (
                                particles_info[neighbor_idx, env_idx].mass
                                / neighbor.density
                                * (neighbor.vel - particle.vel)
                                * _cubic_kernel(distance, support_radius)
                            )
            particles[particle_idx, env_idx].delta_vel = viscosity * delta_vel


@qd.kernel
def _kernel_apply_viscosity_velocity_updates(
    n_particles: qd.i32,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    errno: qd.Tensor,
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            velocity = particles[particle_idx, env_idx].vel + particles[particle_idx, env_idx].delta_vel
            is_valid = True
            for axis in qd.static(range(3)):
                is_valid = is_valid and not qd.math.isnan(velocity[axis]) and not qd.math.isinf(velocity[axis])
            if is_valid:
                particles[particle_idx, env_idx].vel = velocity
            else:
                errno[env_idx] = ErrorCode.INVALID_IPBSTF_STATE_NAN


@qd.kernel
def _kernel_initialize_kinetic_smoothing(
    n_particles: qd.i32,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    kinetic_smoothing_state: qd.template(),
):
    for env_idx in range(particles.shape[1]):
        kinetic_smoothing_state[env_idx].mass = 0.0
        kinetic_smoothing_state[env_idx].momentum = qd.Vector.zero(gs.qd_float, 3)
        kinetic_smoothing_state[env_idx].second_moment = qd.Matrix.zero(gs.qd_float, 3, 3)
        kinetic_smoothing_state[env_idx].filtered_momentum = qd.Vector.zero(gs.qd_float, 3)
        kinetic_smoothing_state[env_idx].filtered_second_moment = qd.Matrix.zero(gs.qd_float, 3, 3)
        kinetic_smoothing_state[env_idx].velocity_transform = qd.Matrix.identity(gs.qd_float, 3)
        kinetic_smoothing_state[env_idx].is_active = False

    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            mass = particles_info[particle_idx, env_idx].mass
            velocity = particles[particle_idx, env_idx].vel
            kinetic_smoothing_state[env_idx].mass += mass
            for axis in qd.static(range(3)):
                kinetic_smoothing_state[env_idx].momentum[axis] += mass * velocity[axis]
                for axis_j in qd.static(range(3)):
                    kinetic_smoothing_state[env_idx].second_moment[axis, axis_j] += (
                        mass * velocity[axis] * velocity[axis_j]
                    )


@qd.kernel
def _kernel_reduce_kinetic_smoothing(
    n_particles: qd.i32,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    kinetic_smoothing_state: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            mass = particles_info[particle_idx, env_idx].mass
            velocity = particles[particle_idx, env_idx].vel + particles[particle_idx, env_idx].delta_vel
            for axis in qd.static(range(3)):
                kinetic_smoothing_state[env_idx].filtered_momentum[axis] += mass * velocity[axis]
                for axis_j in qd.static(range(3)):
                    kinetic_smoothing_state[env_idx].filtered_second_moment[axis, axis_j] += (
                        mass * velocity[axis] * velocity[axis_j]
                    )


@qd.kernel
def _kernel_prepare_kinetic_smoothing(kinetic_smoothing_state: qd.template()):
    for env_idx in range(kinetic_smoothing_state.shape[0]):
        state = kinetic_smoothing_state[env_idx]
        if state.mass > gs.EPS:
            velocity_center = state.momentum / state.mass
            velocity_center_filtered = state.filtered_momentum / state.mass
            covariance = state.second_moment / state.mass - velocity_center.outer_product(velocity_center)
            covariance_filtered = (
                state.filtered_second_moment / state.mass
                - velocity_center_filtered.outer_product(velocity_center_filtered)
            )
            eigenvalues, eigenvectors = qd.sym_eig(covariance)
            eigenvalues_filtered, eigenvectors_filtered = qd.sym_eig(covariance_filtered)
            covariance_trace = qd.max(0.0, eigenvalues[0] + eigenvalues[1] + eigenvalues[2])
            covariance_trace_filtered = qd.max(
                0.0,
                eigenvalues_filtered[0] + eigenvalues_filtered[1] + eigenvalues_filtered[2],
            )
            eigenvalue_tolerance = gs.EPS * qd.max(1.0, covariance_trace_filtered)
            if covariance_trace > eigenvalue_tolerance and covariance_trace_filtered > eigenvalue_tolerance:
                kinetic_smoothing_state[env_idx].is_active = True
                if eigenvalues_filtered[0] > eigenvalue_tolerance:
                    covariance_sqrt = qd.Matrix.zero(gs.qd_float, 3, 3)
                    covariance_filtered_inv_sqrt = qd.Matrix.zero(gs.qd_float, 3, 3)
                    for mode in qd.static(range(3)):
                        sqrt_eigenvalue = qd.sqrt(qd.max(0.0, eigenvalues[mode]))
                        inv_sqrt_eigenvalue_filtered = 1.0 / qd.sqrt(eigenvalues_filtered[mode])
                        for row in qd.static(range(3)):
                            for column in qd.static(range(3)):
                                covariance_sqrt[row, column] += (
                                    sqrt_eigenvalue * eigenvectors[row, mode] * eigenvectors[column, mode]
                                )
                                covariance_filtered_inv_sqrt[row, column] += (
                                    inv_sqrt_eigenvalue_filtered
                                    * eigenvectors_filtered[row, mode]
                                    * eigenvectors_filtered[column, mode]
                                )
                    kinetic_smoothing_state[env_idx].velocity_transform = (
                        covariance_sqrt @ covariance_filtered_inv_sqrt
                    )
                else:
                    scale = qd.sqrt(covariance_trace / covariance_trace_filtered)
                    kinetic_smoothing_state[env_idx].velocity_transform = (
                        scale * qd.Matrix.identity(gs.qd_float, 3)
                    )


@qd.kernel
def _kernel_apply_kinetic_smoothing(
    n_particles: qd.i32,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    kinetic_smoothing_state: qd.template(),
    errno: qd.Tensor,
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active and not particles_info[particle_idx, env_idx].is_fixed:
            state = kinetic_smoothing_state[env_idx]
            velocity = particles[particle_idx, env_idx].vel
            if state.mass > gs.EPS and state.is_active:
                velocity_center = state.momentum / state.mass
                velocity_center_filtered = state.filtered_momentum / state.mass
                velocity = velocity_center + state.velocity_transform @ (
                    particles[particle_idx, env_idx].vel
                    + particles[particle_idx, env_idx].delta_vel
                    - velocity_center_filtered
                )
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
    """Implicit position-based surface-tension fluid (IPBSTF) density solver using parallel local Newton updates."""

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        self._alpha = options.alpha
        self._is_damping_enabled = options.is_damping_enabled
        self._damping_alpha = options.damping_alpha
        self._damping_beta = options.damping_beta
        self._damping_velocity_scale = options.damping_velocity_scale
        self._density_constraint_tolerance = max(1e-8, 512.0 * gs.EPS)
        self._density_update_fraction = options.density_update_fraction
        # Candidate motion stays within half a hash cell between each pair, so a two-cell stencil remains complete.
        self._density_update_limit = 0.25 * options.support_radius
        self._hessian_determinant_epsilon = options.hessian_determinant_epsilon
        self._particle_size = options.particle_size
        self._particle_radius = 0.5 * options.particle_size
        self._support_radius = options.support_radius
        self._surface_update_limit = options.surface_update_scale * self._support_radius
        self._max_solver_iterations = options.max_solver_iterations
        self._max_line_search_iterations = 24
        self._line_search_reduction = 0.5
        self._energy_decrease_fraction = 1e-4
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
        self._collision_boundary = CubeBoundary(
            lower=options.collision_lower_bound,
            upper=options.collision_upper_bound,
        )

        self._default_mass = 1.0
        self._material = None
        self._viscosity = None
        self._kinetic_smoothing = None
        self._kinetic_smoothing_state = None
        self._damping_state = None
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
            if (
                entity.material.rho != self._material.rho
                or entity.material.viscosity != self._material.viscosity
                or entity.material.kinetic_smoothing != self._material.kinetic_smoothing
            ):
                gs.raise_exception(
                    "All liquid entities in one IPBSTFSolver must use the same rest density, viscosity, and kinetic "
                    "smoothing because the current solver is single-phase."
                )
        self._viscosity = self._material.viscosity
        self._kinetic_smoothing = self._material.kinetic_smoothing

        self.sh.build(self._B)
        particle_state = qd.types.struct(
            pos=gs.qd_vec3,
            pos_prev=gs.qd_vec3,
            pos_predicted=gs.qd_vec3,
            pos_iteration=gs.qd_vec3,
            pos_damping=gs.qd_vec3,
            delta_pos=gs.qd_vec3,
            delta_vel=gs.qd_vec3,
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
        line_search_state = qd.types.struct(
            initial_energy=gs.qd_float,
            candidate_energy=gs.qd_float,
            accepted_energy=gs.qd_float,
            directional_decrease=gs.qd_float,
            step_size=gs.qd_float,
            is_active=gs.qd_bool,
        )
        kinetic_smoothing_state = qd.types.struct(
            mass=gs.qd_float,
            momentum=gs.qd_vec3,
            second_moment=gs.qd_mat3,
            filtered_momentum=gs.qd_vec3,
            filtered_second_moment=gs.qd_mat3,
            velocity_transform=gs.qd_mat3,
            is_active=gs.qd_bool,
        )
        damping_state = qd.types.struct(
            mass=gs.qd_float,
            mass_displacement_sqr=gs.qd_float,
            is_active=gs.qd_bool,
        )

        shape = (self._n_particles, self._B)
        self.particles = particle_state.field(shape=shape, layout=qd.Layout.SOA)
        self.particles_status = particle_status.field(shape=shape, layout=qd.Layout.SOA)
        self.particles_info = particle_info.field(shape=(self._n_particles,), layout=qd.Layout.SOA)
        self.particles_reordered = particle_state.field(shape=shape, layout=qd.Layout.SOA)
        self.particles_status_reordered = particle_status.field(shape=shape, layout=qd.Layout.SOA)
        self.particles_info_reordered = particle_info.field(shape=shape, layout=qd.Layout.SOA)
        self.particles_render = particle_render.field(shape=shape, layout=qd.Layout.SOA)
        self._line_search_state = line_search_state.field(shape=(self._B,), layout=qd.Layout.SOA)
        self._kinetic_smoothing_state = kinetic_smoothing_state.field(shape=(self._B,), layout=qd.Layout.SOA)
        self._damping_state = damping_state.field(shape=(self._B,), layout=qd.Layout.SOA)
        self._solver_iteration_energy = qd.field(
            gs.qd_float, shape=(self._max_solver_iterations + 1, self._B)
        )
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
            self._density_constraint_tolerance,
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
        _kernel_project_positions(
            self._n_particles,
            self.particles_reordered,
            self.particles_status_reordered,
            self.particles_info_reordered,
            self._kernel_static_colliders_pos,
            self._kernel_static_colliders_quat,
            self._collision_boundary,
            self._static_colliders,
            self._errno,
        )
        _kernel_copy_from_reordered(self._n_particles, self.particles, self.particles_status, self.particles_reordered)

        for iteration_idx in range(self._max_solver_iterations):
            self._reorder_particles()
            self._compute_density_constraints()
            _kernel_initialize_line_search(
                self._n_particles,
                iteration_idx,
                self._substep_dt,
                self._alpha,
                self.particles_reordered,
                self.particles_status_reordered,
                self.particles_info_reordered,
                self._line_search_state,
                self._solver_iteration_energy,
            )
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
                self._density_update_fraction,
                self._density_update_limit,
                self._surface_update_limit,
                self._hessian_determinant_epsilon,
                self.particles_reordered,
                self.particles_status_reordered,
                self.particles_info_reordered,
                self._line_search_state,
                self._errno,
            )
            if self._is_damping_enabled and iteration_idx == self._max_solver_iterations - 1:
                _kernel_compute_damping_positions(
                    self._n_particles,
                    self._substep_dt,
                    self._damping_alpha,
                    self._density_update_fraction,
                    self._density_update_limit,
                    self._surface_update_limit,
                    self._hessian_determinant_epsilon,
                    self._particle_radius,
                    self._support_radius,
                    self.particles_reordered,
                    self.particles_status_reordered,
                    self.particles_info_reordered,
                    self._kernel_static_colliders_pos,
                    self._kernel_static_colliders_quat,
                    self.sh,
                    self._collision_boundary,
                    self._static_colliders,
                    self._errno,
                )
            for _ in range(self._max_line_search_iterations):
                _kernel_apply_position_updates(
                    self._n_particles,
                    self.particles_reordered,
                    self.particles_status_reordered,
                    self.particles_info_reordered,
                    self._kernel_static_colliders_pos,
                    self._kernel_static_colliders_quat,
                    self._line_search_state,
                    self._collision_boundary,
                    self._static_colliders,
                    self._errno,
                )
                _kernel_compute_line_search_density_constraints(
                    self._n_particles,
                    self._density_constraint_tolerance,
                    self._particle_radius,
                    self._support_radius,
                    self.particles_reordered,
                    self.particles_status_reordered,
                    self.particles_info_reordered,
                    self._kernel_static_colliders_pos,
                    self._kernel_static_colliders_quat,
                    self._line_search_state,
                    self.sh,
                    self._static_colliders,
                )
                _kernel_update_line_search(
                    self._n_particles,
                    self._substep_dt,
                    self._alpha,
                    self._line_search_reduction,
                    self._energy_decrease_fraction,
                    self.particles_reordered,
                    self.particles_status_reordered,
                    self.particles_info_reordered,
                    self._line_search_state,
                )

            _kernel_restore_failed_line_search(
                self._n_particles,
                iteration_idx,
                self.particles_reordered,
                self.particles_status_reordered,
                self.particles_info_reordered,
                self._line_search_state,
                self._solver_iteration_energy,
            )
            _kernel_copy_from_reordered(
                self._n_particles, self.particles, self.particles_status, self.particles_reordered
            )

        self._reorder_particles()
        _kernel_project_positions(
            self._n_particles,
            self.particles_reordered,
            self.particles_status_reordered,
            self.particles_info_reordered,
            self._kernel_static_colliders_pos,
            self._kernel_static_colliders_quat,
            self._collision_boundary,
            self._static_colliders,
            self._errno,
        )
        if self._is_damping_enabled:
            _kernel_compute_damping_state(
                self._n_particles,
                self._damping_velocity_scale,
                self._support_radius,
                self.particles_reordered,
                self.particles_status_reordered,
                self.particles_info_reordered,
                self._damping_state,
            )
        _kernel_update_velocities(
            self._n_particles,
            self._substep_dt,
            self._damping_beta,
            self._support_radius,
            self.particles_reordered,
            self.particles_status_reordered,
            self.particles_info_reordered,
            self._damping_state,
            self._is_damping_enabled,
            self._errno,
        )
        if self._viscosity > 0.0:
            self._compute_density_constraints()
            _kernel_compute_viscosity_velocity_updates(
                self._n_particles,
                self._viscosity,
                self._support_radius,
                self.particles_reordered,
                self.particles_status_reordered,
                self.particles_info_reordered,
                self.sh,
            )
            _kernel_apply_viscosity_velocity_updates(
                self._n_particles,
                self.particles_reordered,
                self.particles_status_reordered,
                self.particles_info_reordered,
                self._errno,
            )
        if self._kinetic_smoothing > 0.0:
            self._compute_density_constraints()
            _kernel_initialize_kinetic_smoothing(
                self._n_particles,
                self.particles_reordered,
                self.particles_status_reordered,
                self.particles_info_reordered,
                self._kinetic_smoothing_state,
            )
            _kernel_compute_viscosity_velocity_updates(
                self._n_particles,
                self._kinetic_smoothing,
                self._support_radius,
                self.particles_reordered,
                self.particles_status_reordered,
                self.particles_info_reordered,
                self.sh,
            )
            _kernel_reduce_kinetic_smoothing(
                self._n_particles,
                self.particles_reordered,
                self.particles_status_reordered,
                self.particles_info_reordered,
                self._kinetic_smoothing_state,
            )
            _kernel_prepare_kinetic_smoothing(self._kinetic_smoothing_state)
            _kernel_apply_kinetic_smoothing(
                self._n_particles,
                self.particles_reordered,
                self.particles_status_reordered,
                self.particles_info_reordered,
                self._kinetic_smoothing_state,
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
                "IPBSTF produced a non-finite local solve, position, or velocity. Reduce the time step, increase the "
                "particle resolution, or use larger alpha and damping_alpha values."
            )

    @gs.assert_built
    def get_kinetic_energy(self, envs_idx=None):
        """Get the total translational kinetic energy of active liquid particles in Joules [J]."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        velocity = qd_to_torch(self.particles.vel, envs_idx, transpose=True)
        is_active = qd_to_torch(self.particles_status.active, envs_idx, transpose=True)
        is_fixed = qd_to_torch(self.particles_info.is_fixed)
        mass = qd_to_torch(self.particles_info.mass)
        speed_sqr = torch.sum(velocity * velocity, dim=-1)
        kinetic_energy = 0.5 * torch.sum((is_active & ~is_fixed) * mass * speed_sqr, dim=-1)
        return kinetic_energy[0] if self._sim.n_envs == 0 else kinetic_energy

    @gs.assert_built
    def get_last_step_variational_energy(self, envs_idx=None):
        """Get energy before and after each density iteration of the most recent solver substep.

        Returns a tensor with shape ``(max_solver_iterations + 1,)`` or
        ``(n_envs, max_solver_iterations + 1)``. Entry zero is the energy of the collision-projected prediction;
        each remaining entry is the accepted energy after one relaxed local Newton iteration.
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        energy = qd_to_torch(self._solver_iteration_energy, envs_idx, transpose=True, copy=True)
        return energy[0] if self._sim.n_envs == 0 else energy

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
