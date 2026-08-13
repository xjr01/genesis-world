import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

import quadrants as qd

import genesis as gs
from genesis.typing import NDArrayType


@dataclass(frozen=True)
class PorousRestTopology:
    """Exact-size fixed-neighborhood data for meshless porous elasticity."""

    neighbor_offsets: NDArrayType
    neighbor_indices: NDArrayType
    rest_offsets: NDArrayType
    corrected_gradients: NDArrayType
    density_reference: NDArrayType


def _cubic_kernel_numpy(distances, support_radius):
    q = distances / support_radius
    coefficient = 8.0 / (math.pi * support_radius**3)
    weights = np.zeros_like(distances)
    is_inner = q < 0.5
    is_outer = (q >= 0.5) & (q < 1.0)
    weights[is_inner] = coefficient * (6.0 * q[is_inner] ** 2 * (q[is_inner] - 1.0) + 1.0)
    weights[is_outer] = 2.0 * coefficient * (1.0 - q[is_outer]) ** 3
    return weights


def _cubic_gradient_numpy(deltas, support_radius):
    distances = np.linalg.norm(deltas, axis=1)
    q = distances / support_radius
    coefficient = 48.0 / (math.pi * support_radius**4)
    derivatives = np.zeros_like(distances)
    is_inner = (distances > gs.EPS) & (q < 0.5)
    is_outer = (q >= 0.5) & (q < 1.0)
    derivatives[is_inner] = coefficient * q[is_inner] * (3.0 * q[is_inner] - 2.0)
    derivatives[is_outer] = -coefficient * (1.0 - q[is_outer]) ** 2
    gradients = np.zeros_like(deltas)
    is_nonzero = distances > gs.EPS
    gradients[is_nonzero] = -derivatives[is_nonzero, None] * deltas[is_nonzero] / distances[is_nonzero, None]
    return gradients


def build_porous_rest_topology(entities, support_radius):
    """Build entity-isolated corrected smoothed-particle hydrodynamics (SPH) reference neighborhoods."""

    n_particles = sum(entity.n_particles for entity in entities)
    counts = np.zeros(n_particles, dtype=gs.np_int)
    density_reference = np.zeros(n_particles, dtype=gs.np_float)
    sources_parts = []
    indices_parts = []
    raw_gradients_parts = []
    particle_start = 0

    for entity in entities:
        positions = entity.init_particles
        n_entity_particles = entity.n_particles
        pairs = cKDTree(positions).query_pairs(support_radius, output_type="ndarray")
        if pairs.size > 0:
            sources_local = np.concatenate((pairs[:, 0], pairs[:, 1]))
            indices_local = np.concatenate((pairs[:, 1], pairs[:, 0]))
            order = np.lexsort((indices_local, sources_local))
            sources_local = sources_local[order]
            indices_local = indices_local[order]
            deltas = positions[sources_local] - positions[indices_local]
            sources_parts.append(sources_local + particle_start)
            indices_parts.append(indices_local + particle_start)
            raw_gradients_parts.append(_cubic_gradient_numpy(deltas, support_radius))
            counts[particle_start : particle_start + n_entity_particles] = np.bincount(
                sources_local, minlength=n_entity_particles
            )

        mass = (1.0 - entity.material.porosity) * entity.material.rho * entity.rest_volume
        density_reference[particle_start : particle_start + n_entity_particles] = mass * _cubic_kernel_numpy(
            np.zeros(n_entity_particles, dtype=gs.np_float), support_radius
        )
        if pairs.size > 0:
            weights = mass * _cubic_kernel_numpy(np.linalg.norm(deltas, axis=1), support_radius)
            np.add.at(density_reference, sources_local + particle_start, weights)
        particle_start += n_entity_particles

    neighbor_offsets = np.empty(n_particles + 1, dtype=gs.np_int)
    neighbor_offsets[0] = 0
    neighbor_offsets[1:] = np.cumsum(counts)
    if not sources_parts:
        gs.raise_exception("PBSTF porous sampling has no reference neighbors inside the support radius.")

    sources = np.concatenate(sources_parts, dtype=gs.np_int)
    neighbor_indices = np.concatenate(indices_parts, dtype=gs.np_int)
    raw_gradients = np.concatenate(raw_gradients_parts, dtype=gs.np_float)
    rest_positions = np.concatenate(tuple(entity.init_particles for entity in entities), dtype=gs.np_float)
    rest_volumes = np.concatenate(
        tuple(np.full(entity.n_particles, entity.rest_volume, dtype=gs.np_float) for entity in entities)
    )
    rest_offsets = np.subtract(rest_positions[neighbor_indices], rest_positions[sources], dtype=gs.np_float)
    moment_matrices = np.zeros((n_particles, 3, 3), dtype=gs.np_float)
    contributions = rest_volumes[neighbor_indices, None, None] * np.einsum(
        "ni,nj->nij", rest_offsets, raw_gradients
    )
    np.add.at(moment_matrices, sources, contributions)
    singular_values = np.linalg.svd(moment_matrices, compute_uv=False)
    if (singular_values[:, -1] <= 1e-4 * singular_values[:, 0]).any():
        gs.raise_exception(
            "PBSTF porous sampling has an ill-conditioned reference neighborhood; use a volumetric staggered sample "
            "with at least three well-spaced particle layers."
        )
    correction_matrices = np.linalg.pinv(moment_matrices).transpose(0, 2, 1)
    corrected_gradients = np.einsum(
        "nij,nj->ni", correction_matrices[sources], raw_gradients, dtype=gs.np_float
    )
    return PorousRestTopology(neighbor_offsets, neighbor_indices, rest_offsets, corrected_gradients, density_reference)


@qd.func
def cubic_kernel(distance, support_radius):
    value = gs.qd_float(0.0)
    q = distance / support_radius
    coefficient = 8.0 / (math.pi * support_radius**3)
    if q < 0.5:
        value = coefficient * (6.0 * q * q * (q - 1.0) + 1.0)
    elif q < 1.0:
        value = 2.0 * coefficient * (1.0 - q) ** 3
    return value


@qd.func
def cubic_kernel_first_derivative(distance, support_radius):
    value = gs.qd_float(0.0)
    q = distance / support_radius
    coefficient = 48.0 / (math.pi * support_radius**4)
    if q < 0.5:
        value = coefficient * q * (3.0 * q - 2.0)
    elif q < 1.0:
        value = -coefficient * (1.0 - q) ** 2
    return value


@qd.func
def cubic_gradient(delta, support_radius):
    gradient = qd.Vector.zero(gs.qd_float, 3)
    distance = delta.norm()
    if distance > gs.EPS and distance < support_radius:
        gradient = -cubic_kernel_first_derivative(distance, support_radius) * delta / distance
    return gradient


@qd.func
def _inverse_mass(particle_idx, env_idx, particles_status, particles_info):
    inverse_mass = gs.qd_float(0.0)
    if particles_status[particle_idx, env_idx].active and not particles_status[particle_idx, env_idx].is_fixed:
        mass = particles_info[particle_idx, env_idx].mass
        if mass > gs.EPS:
            inverse_mass = 1.0 / mass
    return inverse_mass


@qd.kernel
def kernel_add_porous_particles(
    particle_start: qd.i32,
    n_particles: qd.i32,
    active: qd.i32,
    material_idx: qd.i32,
    rho: float,
    porosity: float,
    rest_volume: float,
    deviatoric_compliance: float,
    volumetric_compliance: float,
    pore_compliance: float,
    capillary_compliance: float,
    capillary_saturation_falloff: float,
    drag: float,
    wet_deviatoric_compliance_scale: float,
    wet_volumetric_compliance_scale: float,
    bloating_volume_strain: float,
    pos: qd.types.ndarray(),
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    is_capillary_enabled: qd.template(),
):
    for particle_idx_local, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        particle_idx = particle_start + particle_idx_local
        particle_pos = qd.Vector.zero(gs.qd_float, 3)
        for axis in qd.static(range(3)):
            particle_pos[axis] = pos[particle_idx_local, axis]
        particles[particle_idx, env_idx].pos = particle_pos
        particles[particle_idx, env_idx].ipos = particle_pos
        particles[particle_idx, env_idx].dpos = qd.Vector.zero(gs.qd_float, 3)
        particles[particle_idx, env_idx].vel = qd.Vector.zero(gs.qd_float, 3)
        particles[particle_idx, env_idx].density = 0.0
        particles[particle_idx, env_idx].porosity = porosity
        particles[particle_idx, env_idx].saturation = 0.0
        particles[particle_idx, env_idx].rotation = qd.Matrix.identity(gs.qd_float, 3)
        particles[particle_idx, env_idx].strain = qd.Matrix.zero(gs.qd_float, 3, 3)
        particles_status[particle_idx, env_idx].active = qd.cast(active, gs.qd_bool)
        particles_status[particle_idx, env_idx].is_fixed = False

    for particle_idx_local in range(n_particles):
        particle_idx = particle_start + particle_idx_local
        particles_info[particle_idx].mass = (1.0 - porosity) * rho * rest_volume
        particles_info[particle_idx].rest_volume = rest_volume
        particles_info[particle_idx].porosity = porosity
        particles_info[particle_idx].material_idx = material_idx
        particles_info[particle_idx].deviatoric_compliance = deviatoric_compliance
        particles_info[particle_idx].volumetric_compliance = volumetric_compliance
        particles_info[particle_idx].pore_compliance = pore_compliance
        particles_info[particle_idx].capillary_compliance = capillary_compliance
        particles_info[particle_idx].capillary_saturation_falloff = capillary_saturation_falloff
        particles_info[particle_idx].drag = drag
        particles_info[particle_idx].wet_deviatoric_compliance_scale = wet_deviatoric_compliance_scale
        particles_info[particle_idx].wet_volumetric_compliance_scale = wet_volumetric_compliance_scale
        particles_info[particle_idx].bloating_volume_strain = bloating_volume_strain
        particles_info[particle_idx].is_capillary_enabled = is_capillary_enabled


@qd.kernel
def kernel_reorder_porous_particles(
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
            particles_status_reordered[reordered_idx, env_idx] = particles_status[particle_idx, env_idx]
            particles_info_reordered[reordered_idx, env_idx] = particles_info[particle_idx]


@qd.kernel
def kernel_copy_porous_from_reordered(
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
def kernel_clear_porous_position_delta(n_particles: qd.i32, particles: qd.template()):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        particles[particle_idx, env_idx].dpos = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_predict_porous_positions(
    n_particles: qd.i32,
    substep_dt: float,
    gravity: qd.template(),
    particles: qd.template(),
    particles_status: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active:
            pos = particles[particle_idx, env_idx].pos
            if particles_status[particle_idx, env_idx].is_fixed:
                particles[particle_idx, env_idx].ipos = pos - substep_dt * particles[particle_idx, env_idx].vel
            else:
                particles[particle_idx, env_idx].ipos = pos
                vel = particles[particle_idx, env_idx].vel + substep_dt * gravity[env_idx]
                particles[particle_idx, env_idx].vel = vel
                particles[particle_idx, env_idx].pos = pos + substep_dt * vel


@qd.kernel
def kernel_compute_porous_density(
    n_particles: qd.i32,
    support_radius: float,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    spatial_hasher: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active:
            pos_i = particles[particle_idx, env_idx].pos
            info_i = particles_info[particle_idx, env_idx]
            density = info_i.mass * cubic_kernel(0.0, support_radius)
            base = spatial_hasher.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = spatial_hasher.grid_to_slot(base + offset)
                slot_start = spatial_hasher.slot_start[slot_idx, env_idx]
                slot_end = slot_start + spatial_hasher.slot_size[slot_idx, env_idx]
                for neighbor_idx in range(slot_start, slot_end):
                    if (
                        neighbor_idx != particle_idx
                        and particles_status[neighbor_idx, env_idx].active
                        and particles_info[neighbor_idx, env_idx].material_idx == info_i.material_idx
                    ):
                        distance = (pos_i - particles[neighbor_idx, env_idx].pos).norm()
                        if distance < support_radius:
                            density += particles_info[neighbor_idx, env_idx].mass * cubic_kernel(
                                distance, support_radius
                            )
            particles[particle_idx, env_idx].density = density
            density_reference = info_i.density_reference
            porosity = info_i.porosity
            if density_reference > gs.EPS:
                porosity = 1.0 - (1.0 - info_i.porosity) * density / density_reference
            particles[particle_idx, env_idx].porosity = qd.max(0.0, qd.min(1.0, porosity))


@qd.kernel
def kernel_compute_porous_saturation(
    n_particles: qd.i32,
    support_radius: float,
    fluid_particles: qd.template(),
    fluid_particles_status: qd.template(),
    fluid_particles_info: qd.template(),
    porous_particles: qd.template(),
    porous_particles_status: qd.template(),
    porous_particles_info: qd.template(),
    fluid_spatial_hasher: qd.template(),
    porous_spatial_hasher: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, porous_particles.shape[1]):
        if porous_particles_status[particle_idx, env_idx].active:
            pos_i = porous_particles[particle_idx, env_idx].pos
            info_i = porous_particles_info[particle_idx, env_idx]
            fluid_volume = gs.qd_float(0.0)
            base = fluid_spatial_hasher.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = fluid_spatial_hasher.grid_to_slot(base + offset)
                slot_start = fluid_spatial_hasher.slot_start[slot_idx, env_idx]
                slot_end = slot_start + fluid_spatial_hasher.slot_size[slot_idx, env_idx]
                for fluid_idx in range(slot_start, slot_end):
                    if fluid_particles_status[fluid_idx, env_idx].active:
                        distance = (pos_i - fluid_particles[fluid_idx, env_idx].pos).norm()
                        if distance < support_radius:
                            fluid_volume += (
                                fluid_particles_info[fluid_idx, env_idx].mass
                                / fluid_particles_info[fluid_idx, env_idx].rho_rest
                                * cubic_kernel(distance, support_radius)
                            )

            shepard_volume = info_i.mass / qd.max(porous_particles[particle_idx, env_idx].density, gs.EPS)
            shepard_volume *= cubic_kernel(0.0, support_radius)
            base = porous_spatial_hasher.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = porous_spatial_hasher.grid_to_slot(base + offset)
                slot_start = porous_spatial_hasher.slot_start[slot_idx, env_idx]
                slot_end = slot_start + porous_spatial_hasher.slot_size[slot_idx, env_idx]
                for neighbor_idx in range(slot_start, slot_end):
                    if (
                        neighbor_idx != particle_idx
                        and porous_particles_status[neighbor_idx, env_idx].active
                        and porous_particles_info[neighbor_idx, env_idx].material_idx == info_i.material_idx
                    ):
                        distance = (pos_i - porous_particles[neighbor_idx, env_idx].pos).norm()
                        if distance < support_radius:
                            shepard_volume += (
                                porous_particles_info[neighbor_idx, env_idx].mass
                                / qd.max(porous_particles[neighbor_idx, env_idx].density, gs.EPS)
                                * cubic_kernel(distance, support_radius)
                            )
            saturation = gs.qd_float(0.0)
            denominator = porous_particles[particle_idx, env_idx].porosity * shepard_volume
            if denominator > gs.EPS:
                saturation = qd.max(0.0, qd.min(1.0, fluid_volume / denominator))
            porous_particles[particle_idx, env_idx].saturation = saturation


@qd.kernel
def kernel_classify_fluid_in_porous(
    n_fluid_particles: qd.i32,
    support_radius: float,
    fluid_particles: qd.template(),
    fluid_particles_status: qd.template(),
    porous_particles: qd.template(),
    porous_particles_status: qd.template(),
    porous_spatial_hasher: qd.template(),
    is_fluid_in_porous: qd.template(),
    on_surface: qd.template(),
    topology_valid: qd.template(),
    density_constraint_enabled: qd.template(),
):
    for fluid_idx, env_idx in qd.ndrange(n_fluid_particles, fluid_particles.shape[1]):
        is_inside = False
        if fluid_particles_status[fluid_idx, env_idx].active:
            pos_i = fluid_particles[fluid_idx, env_idx].pos
            base = porous_spatial_hasher.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = porous_spatial_hasher.grid_to_slot(base + offset)
                slot_start = porous_spatial_hasher.slot_start[slot_idx, env_idx]
                slot_end = slot_start + porous_spatial_hasher.slot_size[slot_idx, env_idx]
                for porous_idx in range(slot_start, slot_end):
                    if porous_particles_status[porous_idx, env_idx].active:
                        distance = (pos_i - porous_particles[porous_idx, env_idx].pos).norm()
                        if distance < support_radius and porous_particles[porous_idx, env_idx].porosity > gs.EPS:
                            is_inside = True
        is_fluid_in_porous[fluid_idx, env_idx] = is_inside
        if is_inside:
            fluid_particles[fluid_idx, env_idx].surface = False
            on_surface[fluid_idx, env_idx] = False
            topology_valid[fluid_idx, env_idx] = False
            density_constraint_enabled[fluid_idx, env_idx] = True


@qd.func
def accumulate_porous_capacity(
    fluid_idx,
    env_idx,
    support_radius,
    rho_rest,
    result: qd.template(),
    fluid_particles: qd.template(),
    porous_particles: qd.template(),
    porous_particles_status: qd.template(),
    porous_particles_info: qd.template(),
    porous_spatial_hasher: qd.template(),
):
    pos_i = fluid_particles[fluid_idx, env_idx].pos
    base = porous_spatial_hasher.pos_to_grid(pos_i)
    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
        slot_idx = porous_spatial_hasher.grid_to_slot(base + offset)
        slot_start = porous_spatial_hasher.slot_start[slot_idx, env_idx]
        slot_end = slot_start + porous_spatial_hasher.slot_size[slot_idx, env_idx]
        for porous_idx in range(slot_start, slot_end):
            if porous_particles_status[porous_idx, env_idx].active:
                delta = pos_i - porous_particles[porous_idx, env_idx].pos
                if delta.norm() < support_radius:
                    info_s = porous_particles_info[porous_idx, env_idx]
                    solid_volume = (1.0 - info_s.porosity) * info_s.rest_volume
                    result.density += rho_rest * solid_volume * cubic_kernel(delta.norm(), support_radius)
                    gradient = solid_volume * cubic_gradient(delta, support_radius)
                    result.grad_i -= gradient
                    inverse_mass = _inverse_mass(
                        porous_idx, env_idx, porous_particles_status, porous_particles_info
                    )
                    result.denominator += inverse_mass * gradient.norm_sqr()
    return result


@qd.func
def apply_porous_capacity(
    fluid_idx,
    env_idx,
    support_radius,
    lmd,
    fluid_particles: qd.template(),
    porous_particles: qd.template(),
    porous_particles_status: qd.template(),
    porous_particles_info: qd.template(),
    porous_spatial_hasher: qd.template(),
):
    pos_i = fluid_particles[fluid_idx, env_idx].pos
    base = porous_spatial_hasher.pos_to_grid(pos_i)
    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
        slot_idx = porous_spatial_hasher.grid_to_slot(base + offset)
        slot_start = porous_spatial_hasher.slot_start[slot_idx, env_idx]
        slot_end = slot_start + porous_spatial_hasher.slot_size[slot_idx, env_idx]
        for porous_idx in range(slot_start, slot_end):
            if porous_particles_status[porous_idx, env_idx].active:
                delta = pos_i - porous_particles[porous_idx, env_idx].pos
                if delta.norm() < support_radius:
                    info_s = porous_particles_info[porous_idx, env_idx]
                    solid_volume = (1.0 - info_s.porosity) * info_s.rest_volume
                    gradient = solid_volume * cubic_gradient(delta, support_radius)
                    inverse_mass = _inverse_mass(
                        porous_idx, env_idx, porous_particles_status, porous_particles_info
                    )
                    correction = inverse_mass * lmd * gradient
                    for axis in qd.static(range(3)):
                        qd.atomic_add(porous_particles[porous_idx, env_idx].dpos[axis], correction[axis])


@qd.func
def _strain_basis(constraint_idx):
    basis = qd.Matrix.zero(gs.qd_float, 3, 3)
    if constraint_idx == 0:
        basis[0, 0] = 0.7071067811865476
        basis[1, 1] = -0.7071067811865476
    elif constraint_idx == 1:
        basis[0, 0] = 0.4082482904638631
        basis[1, 1] = 0.4082482904638631
        basis[2, 2] = -0.8164965809277261
    elif constraint_idx == 2:
        basis[0, 1] = 0.7071067811865476
        basis[1, 0] = 0.7071067811865476
    elif constraint_idx == 3:
        basis[0, 2] = 0.7071067811865476
        basis[2, 0] = 0.7071067811865476
    elif constraint_idx == 4:
        basis[1, 2] = 0.7071067811865476
        basis[2, 1] = 0.7071067811865476
    else:
        basis = qd.Matrix.identity(gs.qd_float, 3)
    return basis


@qd.kernel
def kernel_compute_porous_kinematics(
    n_particles: qd.i32,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    particles_reordered: qd.template(),
    neighbor_offsets: qd.template(),
    neighbor_indices: qd.template(),
    rest_offsets: qd.template(),
    corrected_gradients: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active:
            reordered_idx = particles_status[particle_idx, env_idx].reordered_idx
            pos_i = particles_reordered[reordered_idx, env_idx].pos
            deformation_gradient = qd.Matrix.zero(gs.qd_float, 3, 3)
            for neighbor_slot in range(neighbor_offsets[particle_idx], neighbor_offsets[particle_idx + 1]):
                neighbor_idx = neighbor_indices[neighbor_slot]
                if particles_status[neighbor_idx, env_idx].active:
                    neighbor_reordered_idx = particles_status[neighbor_idx, env_idx].reordered_idx
                    delta = particles_reordered[neighbor_reordered_idx, env_idx].pos - pos_i
                    deformation_gradient += particles_info[neighbor_idx].rest_volume * delta.outer_product(
                        corrected_gradients[neighbor_slot]
                    )
            U, unused_singular_values, V = qd.svd(deformation_gradient)
            rotation = U @ V.transpose()
            if rotation.determinant() < 0.0:
                for row in qd.static(range(3)):
                    U[row, 2] = -U[row, 2]
                rotation = U @ V.transpose()
            particles_reordered[reordered_idx, env_idx].rotation = rotation
            corotated_gradient = qd.Matrix.identity(gs.qd_float, 3)
            for neighbor_slot in range(neighbor_offsets[particle_idx], neighbor_offsets[particle_idx + 1]):
                neighbor_idx = neighbor_indices[neighbor_slot]
                if particles_status[neighbor_idx, env_idx].active:
                    neighbor_reordered_idx = particles_status[neighbor_idx, env_idx].reordered_idx
                    delta = particles_reordered[neighbor_reordered_idx, env_idx].pos - pos_i
                    rest_delta = rest_offsets[neighbor_slot]
                    gradient = rotation @ corrected_gradients[neighbor_slot]
                    corotated_gradient += particles_info[neighbor_idx].rest_volume * (
                        delta - rotation @ rest_delta
                    ).outer_product(gradient)
            particles_reordered[reordered_idx, env_idx].strain = 0.5 * (
                corotated_gradient + corotated_gradient.transpose()
            ) - qd.Matrix.identity(gs.qd_float, 3)


@qd.kernel
def kernel_apply_porous_elastic_constraints(
    n_particles: qd.i32,
    substep_dt: float,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    particles_reordered: qd.template(),
    particles_status_reordered: qd.template(),
    particles_info_reordered: qd.template(),
    neighbor_offsets: qd.template(),
    neighbor_indices: qd.template(),
    corrected_gradients: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active:
            reordered_idx = particles_status[particle_idx, env_idx].reordered_idx
            rotation = particles_reordered[reordered_idx, env_idx].rotation
            strain = particles_reordered[reordered_idx, env_idx].strain
            saturation = particles_reordered[reordered_idx, env_idx].saturation
            for constraint_idx in qd.static(range(6)):
                basis = _strain_basis(constraint_idx)
                constraint = gs.qd_float(0.0)
                for row, column in qd.static(qd.ndrange(3, 3)):
                    constraint += basis[row, column] * strain[row, column]
                compliance = particles_info[particle_idx].deviatoric_compliance
                wet_scale = particles_info[particle_idx].wet_deviatoric_compliance_scale
                if constraint_idx == 5:
                    constraint -= saturation * particles_info[particle_idx].bloating_volume_strain
                    compliance = particles_info[particle_idx].volumetric_compliance
                    wet_scale = particles_info[particle_idx].wet_volumetric_compliance_scale
                compliance *= (1.0 - saturation) + saturation * wet_scale
                center_gradient = qd.Vector.zero(gs.qd_float, 3)
                denominator = compliance / (substep_dt * substep_dt)
                denominator /= qd.max(particles_info[particle_idx].rest_volume, gs.EPS)
                for neighbor_slot in range(neighbor_offsets[particle_idx], neighbor_offsets[particle_idx + 1]):
                    neighbor_idx = neighbor_indices[neighbor_slot]
                    if particles_status[neighbor_idx, env_idx].active:
                        neighbor_reordered_idx = particles_status[neighbor_idx, env_idx].reordered_idx
                        gradient = particles_info[neighbor_idx].rest_volume * basis @ (
                            rotation @ corrected_gradients[neighbor_slot]
                        )
                        center_gradient -= gradient
                        inverse_mass = _inverse_mass(
                            neighbor_reordered_idx, env_idx, particles_status_reordered, particles_info_reordered
                        )
                        denominator += inverse_mass * gradient.norm_sqr()
                inverse_mass_i = _inverse_mass(
                    reordered_idx, env_idx, particles_status_reordered, particles_info_reordered
                )
                denominator += inverse_mass_i * center_gradient.norm_sqr()
                lmd = gs.qd_float(0.0)
                if denominator > gs.EPS:
                    lmd = -constraint / denominator
                constraint_relaxation = 1.0 / (
                    6.0 * (neighbor_offsets[particle_idx + 1] - neighbor_offsets[particle_idx] + 1)
                )
                correction_i = constraint_relaxation * inverse_mass_i * lmd * center_gradient
                for axis in qd.static(range(3)):
                    qd.atomic_add(particles_reordered[reordered_idx, env_idx].dpos[axis], correction_i[axis])
                for neighbor_slot in range(neighbor_offsets[particle_idx], neighbor_offsets[particle_idx + 1]):
                    neighbor_idx = neighbor_indices[neighbor_slot]
                    if particles_status[neighbor_idx, env_idx].active:
                        neighbor_reordered_idx = particles_status[neighbor_idx, env_idx].reordered_idx
                        gradient = particles_info[neighbor_idx].rest_volume * basis @ (
                            rotation @ corrected_gradients[neighbor_slot]
                        )
                        inverse_mass = _inverse_mass(
                            neighbor_reordered_idx, env_idx, particles_status_reordered, particles_info_reordered
                        )
                        correction = constraint_relaxation * inverse_mass * lmd * gradient
                        for axis in qd.static(range(3)):
                            qd.atomic_add(
                                particles_reordered[neighbor_reordered_idx, env_idx].dpos[axis], correction[axis]
                            )


@qd.kernel
def kernel_apply_porous_pore_constraints(
    n_particles: qd.i32,
    substep_dt: float,
    support_radius: float,
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    spatial_hasher: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active:
            info_i = particles_info[particle_idx, env_idx]
            density_reference = info_i.density_reference
            if density_reference > gs.EPS:
                constraint = (
                    (1.0 - info_i.porosity) * particles[particle_idx, env_idx].density / density_reference - 1.0
                )
                if constraint > 0.0:
                    pos_i = particles[particle_idx, env_idx].pos
                    gradient_i = qd.Vector.zero(gs.qd_float, 3)
                    denominator = info_i.pore_compliance / (substep_dt * substep_dt)
                    denominator /= qd.max(info_i.rest_volume, gs.EPS)
                    base = spatial_hasher.pos_to_grid(pos_i)
                    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        slot_idx = spatial_hasher.grid_to_slot(base + offset)
                        slot_start = spatial_hasher.slot_start[slot_idx, env_idx]
                        slot_end = slot_start + spatial_hasher.slot_size[slot_idx, env_idx]
                        for neighbor_idx in range(slot_start, slot_end):
                            if (
                                neighbor_idx != particle_idx
                                and particles_status[neighbor_idx, env_idx].active
                                and particles_info[neighbor_idx, env_idx].material_idx == info_i.material_idx
                            ):
                                delta = pos_i - particles[neighbor_idx, env_idx].pos
                                if delta.norm() < support_radius:
                                    gradient = (
                                        (1.0 - info_i.porosity)
                                        * particles_info[neighbor_idx, env_idx].mass
                                        / density_reference
                                        * cubic_gradient(delta, support_radius)
                                    )
                                    gradient_i -= gradient
                                    inverse_mass = _inverse_mass(
                                        neighbor_idx, env_idx, particles_status, particles_info
                                    )
                                    denominator += inverse_mass * gradient.norm_sqr()
                    inverse_mass_i = _inverse_mass(particle_idx, env_idx, particles_status, particles_info)
                    denominator += inverse_mass_i * gradient_i.norm_sqr()
                    lmd = gs.qd_float(0.0)
                    if denominator > gs.EPS:
                        lmd = -constraint / denominator
                    correction_i = inverse_mass_i * lmd * gradient_i
                    for axis in qd.static(range(3)):
                        qd.atomic_add(particles[particle_idx, env_idx].dpos[axis], correction_i[axis])
                    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                        slot_idx = spatial_hasher.grid_to_slot(base + offset)
                        slot_start = spatial_hasher.slot_start[slot_idx, env_idx]
                        slot_end = slot_start + spatial_hasher.slot_size[slot_idx, env_idx]
                        for neighbor_idx in range(slot_start, slot_end):
                            if (
                                neighbor_idx != particle_idx
                                and particles_status[neighbor_idx, env_idx].active
                                and particles_info[neighbor_idx, env_idx].material_idx == info_i.material_idx
                            ):
                                delta = pos_i - particles[neighbor_idx, env_idx].pos
                                if delta.norm() < support_radius:
                                    gradient = (
                                        (1.0 - info_i.porosity)
                                        * particles_info[neighbor_idx, env_idx].mass
                                        / density_reference
                                        * cubic_gradient(delta, support_radius)
                                    )
                                    inverse_mass = _inverse_mass(
                                        neighbor_idx, env_idx, particles_status, particles_info
                                    )
                                    correction = inverse_mass * lmd * gradient
                                    for axis in qd.static(range(3)):
                                        qd.atomic_add(
                                            particles[neighbor_idx, env_idx].dpos[axis], correction[axis]
                                        )


@qd.kernel
def kernel_compute_porous_coupling_weights(
    n_fluid_particles: qd.i32,
    n_porous_particles: qd.i32,
    support_radius: float,
    fluid_weight_sums: qd.template(),
    porous_weight_sums: qd.template(),
    fluid_particles: qd.template(),
    fluid_particles_status: qd.template(),
    porous_particles: qd.template(),
    porous_particles_status: qd.template(),
    porous_spatial_hasher: qd.template(),
):
    for fluid_idx, env_idx in qd.ndrange(n_fluid_particles, fluid_particles.shape[1]):
        fluid_weight_sums[fluid_idx, env_idx] = 0.0
    for porous_idx, env_idx in qd.ndrange(n_porous_particles, porous_particles.shape[1]):
        porous_weight_sums[porous_idx, env_idx] = 0.0

    kernel_zero = cubic_kernel(0.0, support_radius)
    for fluid_idx, env_idx in qd.ndrange(n_fluid_particles, fluid_particles.shape[1]):
        if fluid_particles_status[fluid_idx, env_idx].active:
            pos_i = fluid_particles[fluid_idx, env_idx].pos
            weight_sum = gs.qd_float(0.0)
            base = porous_spatial_hasher.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = porous_spatial_hasher.grid_to_slot(base + offset)
                slot_start = porous_spatial_hasher.slot_start[slot_idx, env_idx]
                slot_end = slot_start + porous_spatial_hasher.slot_size[slot_idx, env_idx]
                for porous_idx in range(slot_start, slot_end):
                    if porous_particles_status[porous_idx, env_idx].active:
                        distance = (pos_i - porous_particles[porous_idx, env_idx].pos).norm()
                        if distance < support_radius:
                            weight = cubic_kernel(distance, support_radius) / kernel_zero
                            weight_sum += weight
                            qd.atomic_add(porous_weight_sums[porous_idx, env_idx], weight)
            fluid_weight_sums[fluid_idx, env_idx] = weight_sum


@qd.kernel
def kernel_apply_porous_capillary_drag(
    n_fluid_particles: qd.i32,
    substep_dt: float,
    support_radius: float,
    fluid_default_mass: float,
    fluid_weight_sums: qd.template(),
    porous_weight_sums: qd.template(),
    fluid_particles: qd.template(),
    fluid_particles_status: qd.template(),
    fluid_particles_info: qd.template(),
    porous_particles: qd.template(),
    porous_particles_status: qd.template(),
    porous_particles_info: qd.template(),
    porous_spatial_hasher: qd.template(),
):
    kernel_zero = cubic_kernel(0.0, support_radius)
    for fluid_idx, env_idx in qd.ndrange(n_fluid_particles, fluid_particles.shape[1]):
        if fluid_particles_status[fluid_idx, env_idx].active:
            pos_i = fluid_particles[fluid_idx, env_idx].pos
            inverse_mass_i = 1.0 / fluid_particles_info[fluid_idx, env_idx].mass
            base = porous_spatial_hasher.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = porous_spatial_hasher.grid_to_slot(base + offset)
                slot_start = porous_spatial_hasher.slot_start[slot_idx, env_idx]
                slot_end = slot_start + porous_spatial_hasher.slot_size[slot_idx, env_idx]
                for porous_idx in range(slot_start, slot_end):
                    if porous_particles_status[porous_idx, env_idx].active:
                        delta = pos_i - porous_particles[porous_idx, env_idx].pos
                        distance = delta.norm()
                        if distance > gs.EPS and distance < support_radius:
                            info_s = porous_particles_info[porous_idx, env_idx]
                            inverse_mass_s = _inverse_mass(
                                porous_idx, env_idx, porous_particles_status, porous_particles_info
                            )
                            weight = cubic_kernel(distance, support_radius) / kernel_zero
                            weight_denominator = qd.max(
                                fluid_weight_sums[fluid_idx, env_idx], porous_weight_sums[porous_idx, env_idx]
                            )
                            pair_scale = weight / qd.max(weight_denominator, gs.EPS)
                            if info_s.is_capillary_enabled:
                                strength = qd.max(
                                    0.0,
                                    1.0
                                    - info_s.capillary_saturation_falloff
                                    * porous_particles[porous_idx, env_idx].saturation,
                                )
                                denominator = inverse_mass_i + inverse_mass_s
                                denominator += info_s.capillary_compliance / qd.max(
                                    fluid_default_mass * weight * strength, gs.EPS
                                )
                                if denominator > gs.EPS and strength > gs.EPS:
                                    lmd = -distance / denominator
                                    gradient = delta / distance
                                    correction_i = pair_scale * inverse_mass_i * lmd * gradient
                                    correction_s = -pair_scale * inverse_mass_s * lmd * gradient
                                    for axis in qd.static(range(3)):
                                        qd.atomic_add(
                                            fluid_particles[fluid_idx, env_idx].dpos[axis], correction_i[axis]
                                        )
                                        qd.atomic_add(
                                            porous_particles[porous_idx, env_idx].dpos[axis], correction_s[axis]
                                        )
                            if info_s.drag > gs.EPS and weight > gs.EPS:
                                relative_displacement = (
                                    pos_i
                                    - fluid_particles[fluid_idx, env_idx].ipos
                                    - porous_particles[porous_idx, env_idx].pos
                                    + porous_particles[porous_idx, env_idx].ipos
                                )
                                denominator = inverse_mass_i + inverse_mass_s
                                denominator += 1.0 / (substep_dt * info_s.drag * weight)
                                correction = -relative_displacement / denominator
                                correction_i = pair_scale * inverse_mass_i * correction
                                correction_s = -pair_scale * inverse_mass_s * correction
                                for axis in qd.static(range(3)):
                                    qd.atomic_add(
                                        fluid_particles[fluid_idx, env_idx].dpos[axis], correction_i[axis]
                                    )
                                    qd.atomic_add(
                                        porous_particles[porous_idx, env_idx].dpos[axis], correction_s[axis]
                                    )


@qd.kernel
def kernel_update_porous_velocities(
    n_particles: qd.i32,
    substep_dt: float,
    particles: qd.template(),
    particles_status: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active:
            particles[particle_idx, env_idx].vel = (
                particles[particle_idx, env_idx].pos - particles[particle_idx, env_idx].ipos
            ) / substep_dt


@qd.kernel
def kernel_update_porous_render_fields(
    n_particles: qd.i32,
    particles: qd.template(),
    particles_status: qd.template(),
    render_indices: qd.template(),
    particles_render: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        render_idx = render_indices[particle_idx]
        if particles_status[particle_idx, env_idx].active:
            particles_render[render_idx, env_idx].pos = particles[particle_idx, env_idx].pos
            particles_render[render_idx, env_idx].vel = particles[particle_idx, env_idx].vel
        else:
            particles_render[render_idx, env_idx].pos = qd.Vector([1.0e6, 1.0e6, 1.0e6])
        particles_render[render_idx, env_idx].active = particles_status[particle_idx, env_idx].active


@qd.kernel
def kernel_update_porous_visual_vertices(
    n_vverts: qd.i32,
    n_vvert_supports: qd.i32,
    particles_render: qd.template(),
    vverts_info: qd.template(),
    vverts_render: qd.template(),
):
    for vvert_idx, env_idx in qd.ndrange(n_vverts, particles_render.shape[1]):
        pos = qd.Vector.zero(gs.qd_float, 3)
        for support_idx in range(n_vvert_supports):
            particle_idx = vverts_info[vvert_idx].support_idxs[support_idx]
            pos += particles_render[particle_idx, env_idx].pos * vverts_info[vvert_idx].support_weights[support_idx]
        vverts_render[vvert_idx, env_idx].pos = pos
        vverts_render[vvert_idx, env_idx].active = particles_render[
            vverts_info[vvert_idx].support_idxs[0], env_idx
        ].active


@qd.kernel
def kernel_set_porous_particles_pos(
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
        particles[particle_idx, env_idx].ipos = particles[particle_idx, env_idx].pos
        particles[particle_idx, env_idx].vel = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel
def kernel_get_porous_particles_pos(
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    particles: qd.template(),
    poss: qd.types.ndarray(),
):
    for particle_idx_local, env_idx_local in qd.ndrange(n_particles, envs_idx.shape[0]):
        particle_idx = particle_start + particle_idx_local
        env_idx = envs_idx[env_idx_local]
        for axis in qd.static(range(3)):
            poss[env_idx_local, particle_idx_local, axis] = particles[particle_idx, env_idx].pos[axis]


@qd.kernel
def kernel_set_porous_particles_vel(
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
def kernel_get_porous_particles_vel(
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    particles: qd.template(),
    vels: qd.types.ndarray(),
):
    for particle_idx_local, env_idx_local in qd.ndrange(n_particles, envs_idx.shape[0]):
        particle_idx = particle_start + particle_idx_local
        env_idx = envs_idx[env_idx_local]
        for axis in qd.static(range(3)):
            vels[env_idx_local, particle_idx_local, axis] = particles[particle_idx, env_idx].vel[axis]


@qd.kernel
def kernel_set_porous_particles_active(
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
def kernel_get_porous_particles_active(
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    particles_status: qd.template(),
    actives: qd.types.ndarray(),
):
    for particle_idx_local, env_idx_local in qd.ndrange(n_particles, envs_idx.shape[0]):
        particle_idx = particle_start + particle_idx_local
        env_idx = envs_idx[env_idx_local]
        actives[env_idx_local, particle_idx_local] = particles_status[particle_idx, env_idx].active


@qd.kernel
def kernel_set_porous_particles_fixed(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    is_fixed: qd.i32,
    particles_status: qd.template(),
):
    for particle_idx_local, env_idx_local in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        particle_idx = particles_idx[env_idx_local, particle_idx_local]
        env_idx = envs_idx[env_idx_local]
        particles_status[particle_idx, env_idx].is_fixed = qd.cast(is_fixed, gs.qd_bool)


@qd.kernel
def kernel_get_porous_particles_fixed(
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    particles_status: qd.template(),
    is_fixed: qd.types.ndarray(),
):
    for particle_idx_local, env_idx_local in qd.ndrange(n_particles, envs_idx.shape[0]):
        particle_idx = particle_start + particle_idx_local
        env_idx = envs_idx[env_idx_local]
        is_fixed[env_idx_local, particle_idx_local] = particles_status[particle_idx, env_idx].is_fixed


@qd.kernel
def kernel_get_porous_particles_frame(
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    particles: qd.template(),
    particles_status: qd.template(),
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    active: qd.types.ndarray(),
    is_fixed: qd.types.ndarray(),
):
    for particle_idx_local, env_idx_local in qd.ndrange(n_particles, envs_idx.shape[0]):
        particle_idx = particle_start + particle_idx_local
        env_idx = envs_idx[env_idx_local]
        for axis in qd.static(range(3)):
            pos[env_idx_local, particle_idx_local, axis] = particles[particle_idx, env_idx].pos[axis]
            vel[env_idx_local, particle_idx_local, axis] = particles[particle_idx, env_idx].vel[axis]
        active[env_idx_local, particle_idx_local] = particles_status[particle_idx, env_idx].active
        is_fixed[env_idx_local, particle_idx_local] = particles_status[particle_idx, env_idx].is_fixed


@qd.kernel
def kernel_set_porous_particles_frame(
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    active: qd.types.ndarray(),
    is_fixed: qd.types.ndarray(),
    particles: qd.template(),
    particles_status: qd.template(),
):
    for particle_idx_local, env_idx_local in qd.ndrange(n_particles, envs_idx.shape[0]):
        particle_idx = particle_start + particle_idx_local
        env_idx = envs_idx[env_idx_local]
        for axis in qd.static(range(3)):
            particles[particle_idx, env_idx].pos[axis] = pos[env_idx, particle_idx_local, axis]
            particles[particle_idx, env_idx].vel[axis] = vel[env_idx, particle_idx_local, axis]
        particles[particle_idx, env_idx].ipos = particles[particle_idx, env_idx].pos
        particles_status[particle_idx, env_idx].active = active[env_idx, particle_idx_local]
        particles_status[particle_idx, env_idx].is_fixed = is_fixed[env_idx, particle_idx_local]


@qd.kernel
def kernel_get_porous_particles_saturation(
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    particles: qd.template(),
    saturation: qd.types.ndarray(),
):
    for particle_idx_local, env_idx_local in qd.ndrange(n_particles, envs_idx.shape[0]):
        particle_idx = particle_start + particle_idx_local
        env_idx = envs_idx[env_idx_local]
        saturation[env_idx_local, particle_idx_local] = particles[particle_idx, env_idx].saturation


@qd.kernel
def kernel_get_porous_particles_porosity(
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    particles: qd.template(),
    porosity: qd.types.ndarray(),
):
    for particle_idx_local, env_idx_local in qd.ndrange(n_particles, envs_idx.shape[0]):
        particle_idx = particle_start + particle_idx_local
        env_idx = envs_idx[env_idx_local]
        porosity[env_idx_local, particle_idx_local] = particles[particle_idx, env_idx].porosity


@qd.kernel
def kernel_get_absorbed_fluid_volume(
    particle_start: qd.i32,
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    particles: qd.template(),
    particles_status: qd.template(),
    particles_info: qd.template(),
    volume: qd.types.ndarray(),
):
    for env_idx_local in range(envs_idx.shape[0]):
        env_idx = envs_idx[env_idx_local]
        absorbed_volume = gs.qd_float(0.0)
        for particle_idx_local in range(n_particles):
            particle_idx = particle_start + particle_idx_local
            if particles_status[particle_idx, env_idx].active:
                absorbed_volume += (
                    particles[particle_idx, env_idx].saturation
                    * particles[particle_idx, env_idx].porosity
                    * particles_info[particle_idx].rest_volume
                    * particles_info[particle_idx].density_reference
                    / qd.max(particles[particle_idx, env_idx].density, gs.EPS)
                )
        volume[env_idx_local] = absorbed_volume


@qd.kernel
def kernel_check_fluid_state(
    n_fluid_particles: qd.i32,
    fluid_particles: qd.template(),
    fluid_particles_status: qd.template(),
    error_code: qd.i32,
    errno: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_fluid_particles, fluid_particles.shape[1]):
        if fluid_particles_status[particle_idx, env_idx].active:
            for axis in qd.static(range(3)):
                pos = fluid_particles[particle_idx, env_idx].pos[axis]
                vel = fluid_particles[particle_idx, env_idx].vel[axis]
                if qd.math.isnan(pos) or qd.math.isinf(pos) or qd.math.isnan(vel) or qd.math.isinf(vel):
                    errno[env_idx] = error_code


@qd.kernel
def kernel_check_porous_state(
    n_porous_particles: qd.i32,
    porous_particles: qd.template(),
    porous_particles_status: qd.template(),
    error_code: qd.i32,
    errno: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_porous_particles, porous_particles.shape[1]):
        if porous_particles_status[particle_idx, env_idx].active:
            for axis in qd.static(range(3)):
                pos = porous_particles[particle_idx, env_idx].pos[axis]
                vel = porous_particles[particle_idx, env_idx].vel[axis]
                if qd.math.isnan(pos) or qd.math.isinf(pos) or qd.math.isnan(vel) or qd.math.isinf(vel):
                    errno[env_idx] = error_code
            porosity = porous_particles[particle_idx, env_idx].porosity
            saturation = porous_particles[particle_idx, env_idx].saturation
            if (
                qd.math.isnan(porosity)
                or qd.math.isinf(porosity)
                or qd.math.isnan(saturation)
                or qd.math.isinf(saturation)
            ):
                errno[env_idx] = error_code


@qd.kernel
def kernel_clear_errno(envs_idx: qd.types.ndarray(), errno: qd.template()):
    for env_idx_local in range(envs_idx.shape[0]):
        errno[envs_idx[env_idx_local]] = 0
