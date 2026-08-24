import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import query_static_collider


@qd.func
def is_particle_absorbed(particle_idx, env_idx, absorption_particles: qd.template()):
    return absorption_particles[particle_idx, env_idx].collider_idx >= 0


@qd.func
def unbind_particle(
    particle_idx,
    env_idx,
    n_voxels,
    absorption_particles: qd.template(),
    voxel_capacity: qd.template(),
    voxel_occupancy: qd.template(),
    voxel_wetness: qd.template(),
):
    voxel_idx = absorption_particles[particle_idx, env_idx].voxel_idx
    if 0 <= voxel_idx and voxel_idx < n_voxels:
        qd.atomic_add(voxel_occupancy[voxel_idx, env_idx], -1)
        capacity = voxel_capacity[voxel_idx]
        if capacity > 0:
            qd.atomic_add(
                voxel_wetness[voxel_idx, env_idx],
                -absorption_particles[particle_idx, env_idx].progress / capacity,
            )
    absorption_particles[particle_idx, env_idx].collider_idx = -1
    absorption_particles[particle_idx, env_idx].voxel_idx = -1
    absorption_particles[particle_idx, env_idx].voxel_distance = -1
    absorption_particles[particle_idx, env_idx].local_pos = qd.Vector.zero(gs.qd_float, 3)
    absorption_particles[particle_idx, env_idx].target_local_pos = qd.Vector.zero(gs.qd_float, 3)
    absorption_particles[particle_idx, env_idx].progress = 0.0


@qd.kernel
def kernel_initialize_absorption_particles(n_particles: qd.i32, absorption_particles: qd.template()):
    for particle_idx, env_idx in qd.ndrange(n_particles, absorption_particles.shape[1]):
        absorption_particles[particle_idx, env_idx].collider_idx = -1
        absorption_particles[particle_idx, env_idx].voxel_idx = -1
        absorption_particles[particle_idx, env_idx].voxel_distance = -1
        absorption_particles[particle_idx, env_idx].local_pos = qd.Vector.zero(gs.qd_float, 3)
        absorption_particles[particle_idx, env_idx].target_local_pos = qd.Vector.zero(gs.qd_float, 3)
        absorption_particles[particle_idx, env_idx].progress = 0.0


@qd.kernel
def kernel_replenish_capture_budget(
    absorption_idx: qd.i32,
    budget_increment: float,
    budget_limit: float,
    absorption_capture_budget: qd.template(),
):
    for env_idx in range(absorption_capture_budget.shape[1]):
        budget = absorption_capture_budget[absorption_idx, env_idx] + budget_increment
        absorption_capture_budget[absorption_idx, env_idx] = qd.min(budget, budget_limit)


@qd.kernel
def kernel_set_deformable_collider_geometry(
    envs_idx: qd.types.ndarray(),
    surface_positions: qd.types.ndarray(),
    voxel_positions: qd.types.ndarray(),
    voxel_search_order: qd.types.ndarray(),
    collider: qd.template(),
):
    for surface_vertex_idx, env_idx_local in qd.ndrange(collider.n_surface_vertices, envs_idx.shape[0]):
        env_idx = envs_idx[env_idx_local]
        for axis in qd.static(range(3)):
            collider.surface_vertices[surface_vertex_idx, env_idx][axis] = surface_positions[
                env_idx_local, surface_vertex_idx, axis
            ]
    for voxel_idx, env_idx_local in qd.ndrange(collider.n_voxels, envs_idx.shape[0]):
        env_idx = envs_idx[env_idx_local]
        for axis in qd.static(range(3)):
            collider.voxel_positions[voxel_idx, env_idx][axis] = voxel_positions[env_idx_local, voxel_idx, axis]
    for origin_voxel_idx, search_idx, env_idx_local in qd.ndrange(
        collider.n_voxels, collider.n_voxels, envs_idx.shape[0]
    ):
        env_idx = envs_idx[env_idx_local]
        collider.voxel_search_order[origin_voxel_idx, search_idx, env_idx] = voxel_search_order[
            env_idx_local, origin_voxel_idx, search_idx
        ]


@qd.kernel
def kernel_disable_deformable_collider_sdf(envs_idx: qd.types.ndarray(), collider: qd.template()):
    for env_idx_local in range(envs_idx.shape[0]):
        collider.is_sdf_active[envs_idx[env_idx_local]] = False


@qd.kernel
def kernel_set_deformable_collider_sdf(
    envs_idx: qd.types.ndarray(),
    sdf: qd.types.ndarray(),
    sdf_lower: qd.types.ndarray(),
    sdf_inv_cell_size: qd.types.ndarray(),
    collider: qd.template(),
):
    for x, y, z, env_idx_local in qd.ndrange(
        collider.sdf_res, collider.sdf_res, collider.sdf_res, envs_idx.shape[0]
    ):
        collider.sdf[x, y, z, envs_idx[env_idx_local]] = sdf[env_idx_local, x, y, z]
    for env_idx_local in range(envs_idx.shape[0]):
        env_idx = envs_idx[env_idx_local]
        collider.sdf_lower[env_idx] = sdf_lower[env_idx_local]
        collider.sdf_inv_cell_size[env_idx] = sdf_inv_cell_size[env_idx_local]
        collider.is_sdf_active[env_idx] = True


@qd.kernel
def kernel_get_deformable_collider_sdf_active(
    sdf_state_idx: qd.i32,
    is_sdf_active: qd.types.ndarray(),
    collider: qd.template(),
):
    for env_idx in range(collider.is_sdf_active.shape[0]):
        is_sdf_active[env_idx, sdf_state_idx] = collider.is_sdf_active[env_idx]


@qd.kernel
def kernel_get_deformable_collider_geometry(
    surface_vertex_state_start: qd.i32,
    voxel_state_start: qd.i32,
    voxel_search_order_state_start: qd.i32,
    surface_vertices: qd.types.ndarray(),
    voxel_positions: qd.types.ndarray(),
    voxel_search_order: qd.types.ndarray(),
    collider: qd.template(),
):
    for surface_vertex_idx, env_idx in qd.ndrange(collider.n_surface_vertices, collider.surface_vertices.shape[1]):
        for axis in qd.static(range(3)):
            surface_vertices[env_idx, surface_vertex_state_start + surface_vertex_idx, axis] = (
                collider.surface_vertices[surface_vertex_idx, env_idx][axis]
            )
    for voxel_idx, env_idx in qd.ndrange(collider.n_voxels, collider.voxel_positions.shape[1]):
        for axis in qd.static(range(3)):
            voxel_positions[env_idx, voxel_state_start + voxel_idx, axis] = collider.voxel_positions[
                voxel_idx, env_idx
            ][axis]
    for origin_voxel_idx, search_idx, env_idx in qd.ndrange(
        collider.n_voxels, collider.n_voxels, collider.voxel_search_order.shape[2]
    ):
        state_idx = voxel_search_order_state_start + origin_voxel_idx * collider.n_voxels + search_idx
        voxel_search_order[env_idx, state_idx] = collider.voxel_search_order[
            origin_voxel_idx, search_idx, env_idx
        ]


@qd.kernel
def kernel_set_deformable_collider_state(
    surface_vertex_state_start: qd.i32,
    voxel_state_start: qd.i32,
    voxel_search_order_state_start: qd.i32,
    envs_idx: qd.types.ndarray(),
    surface_vertices: qd.types.ndarray(),
    voxel_positions: qd.types.ndarray(),
    voxel_search_order: qd.types.ndarray(),
    collider: qd.template(),
):
    for surface_vertex_idx, env_idx_local in qd.ndrange(collider.n_surface_vertices, envs_idx.shape[0]):
        env_idx = envs_idx[env_idx_local]
        for axis in qd.static(range(3)):
            collider.surface_vertices[surface_vertex_idx, env_idx][axis] = surface_vertices[
                env_idx, surface_vertex_state_start + surface_vertex_idx, axis
            ]
    for voxel_idx, env_idx_local in qd.ndrange(collider.n_voxels, envs_idx.shape[0]):
        env_idx = envs_idx[env_idx_local]
        for axis in qd.static(range(3)):
            collider.voxel_positions[voxel_idx, env_idx][axis] = voxel_positions[
                env_idx, voxel_state_start + voxel_idx, axis
            ]
    for origin_voxel_idx, search_idx, env_idx_local in qd.ndrange(
        collider.n_voxels, collider.n_voxels, envs_idx.shape[0]
    ):
        env_idx = envs_idx[env_idx_local]
        state_idx = voxel_search_order_state_start + origin_voxel_idx * collider.n_voxels + search_idx
        collider.voxel_search_order[origin_voxel_idx, search_idx, env_idx] = voxel_search_order[env_idx, state_idx]


@qd.kernel
def kernel_check_deformable_collider_geometry(
    collider: qd.template(),
    error_code: qd.i32,
    errno: qd.template(),
):
    for surface_vertex_idx, env_idx in qd.ndrange(collider.n_surface_vertices, collider.surface_vertices.shape[1]):
        pos = collider.surface_vertices[surface_vertex_idx, env_idx]
        is_valid = True
        for axis in qd.static(range(3)):
            is_valid = is_valid and not qd.math.isnan(pos[axis]) and not qd.math.isinf(pos[axis])
        if not is_valid:
            errno[env_idx] = error_code
    for triangle_idx, env_idx in qd.ndrange(collider.n_surface_triangles, collider.surface_vertices.shape[1]):
        face = collider.surface_faces[triangle_idx]
        v0 = collider.surface_vertices[face[0], env_idx]
        v1 = collider.surface_vertices[face[1], env_idx]
        v2 = collider.surface_vertices[face[2], env_idx]
        if (v1 - v0).cross(v2 - v0).norm_sqr() <= gs.EPS**2:
            errno[env_idx] = error_code
    for voxel_idx, env_idx in qd.ndrange(collider.n_voxels, collider.voxel_positions.shape[1]):
        pos = collider.voxel_positions[voxel_idx, env_idx]
        is_valid = True
        for axis in qd.static(range(3)):
            is_valid = is_valid and not qd.math.isnan(pos[axis]) and not qd.math.isinf(pos[axis])
        if not is_valid:
            errno[env_idx] = error_code
    for origin_voxel_idx, search_idx, env_idx in qd.ndrange(
        collider.n_voxels, collider.n_voxels, collider.voxel_search_order.shape[2]
    ):
        voxel_idx = collider.voxel_search_order[origin_voxel_idx, search_idx, env_idx]
        if voxel_idx < 0 or voxel_idx >= collider.n_voxels:
            errno[env_idx] = error_code


@qd.kernel
def kernel_update_absorbed_particles(
    n_particles: qd.i32,
    collider_idx: qd.i32,
    substep_dt: float,
    absorption_rate: float,
    particles: qd.template(),
    particles_status: qd.template(),
    absorption_particles: qd.template(),
    colliders_pos: qd.template(),
    colliders_quat: qd.template(),
    collider: qd.template(),
    error_code: qd.i32,
    errno: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if (
            particles_status[particle_idx, env_idx].active
            and absorption_particles[particle_idx, env_idx].collider_idx == collider_idx
        ):
            local_pos = absorption_particles[particle_idx, env_idx].local_pos
            target_local_pos = absorption_particles[particle_idx, env_idx].target_local_pos
            if qd.static(collider.is_deformable):
                voxel_idx_local = absorption_particles[particle_idx, env_idx].voxel_idx - collider.voxel_start
                target_local_pos = collider.voxel_positions[voxel_idx_local, env_idx]
                absorption_particles[particle_idx, env_idx].target_local_pos = target_local_pos
            voxel_distance = absorption_particles[particle_idx, env_idx].voxel_distance
            progress = absorption_particles[particle_idx, env_idx].progress
            distance_scale = qd.max(voxel_distance + 1, 1)
            beta = 1.0 - qd.exp(-absorption_rate * substep_dt / distance_scale)
            local_pos += beta * (target_local_pos - local_pos)
            progress += beta * (1.0 - progress)
            pos_prev = particles[particle_idx, env_idx].pos
            pos = gu.qd_transform_by_trans_quat(
                local_pos,
                colliders_pos[collider_idx, env_idx],
                colliders_quat[collider_idx, env_idx],
            )
            vel = (pos - pos_prev) / substep_dt

            is_valid = voxel_distance >= 0 and not qd.math.isnan(progress) and not qd.math.isinf(progress)
            for axis in qd.static(range(3)):
                is_valid = (
                    is_valid
                    and not qd.math.isnan(local_pos[axis])
                    and not qd.math.isinf(local_pos[axis])
                    and not qd.math.isnan(pos[axis])
                    and not qd.math.isinf(pos[axis])
                    and not qd.math.isnan(vel[axis])
                    and not qd.math.isinf(vel[axis])
                )
            if is_valid:
                particles[particle_idx, env_idx].ipos = pos_prev
                particles[particle_idx, env_idx].pos = pos
                particles[particle_idx, env_idx].vel = vel
                particles[particle_idx, env_idx].density = 0.0
                particles[particle_idx, env_idx].dpos = qd.Vector.zero(gs.qd_float, 3)
                particles[particle_idx, env_idx].surface = False
                absorption_particles[particle_idx, env_idx].local_pos = local_pos
                absorption_particles[particle_idx, env_idx].progress = progress
            else:
                errno[env_idx] = error_code


@qd.kernel
def kernel_capture_particles(
    n_particles: qd.i32,
    collider_idx: qd.i32,
    absorption_idx: qd.i32,
    particle_radius: float,
    substep_dt: float,
    absorption_rate: float,
    particles: qd.template(),
    particles_status: qd.template(),
    absorption_particles: qd.template(),
    absorption_capture_budget: qd.template(),
    voxel_capacity: qd.template(),
    voxel_occupancy: qd.template(),
    voxel_search_offsets: qd.template(),
    colliders_pos: qd.template(),
    colliders_quat: qd.template(),
    collider: qd.template(),
    error_code: qd.i32,
    errno: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if (
            particles_status[particle_idx, env_idx].active
            and absorption_particles[particle_idx, env_idx].collider_idx < 0
        ):
            pos = particles[particle_idx, env_idx].pos
            _, _, is_inside, surface_distance = query_static_collider(
                collider_idx,
                env_idx,
                pos,
                colliders_pos,
                colliders_quat,
                collider,
            )
            if is_inside or surface_distance <= particle_radius + gs.EPS:
                collider_quat = colliders_quat[collider_idx, env_idx]
                local_pos = gu.qd_inv_transform_by_trans_quat(
                    pos,
                    colliders_pos[collider_idx, env_idx],
                    collider_quat,
                )
                origin_voxel_idx = qd.Vector.zero(gs.qd_int, 3)
                origin_voxel_idx_local = 0
                if qd.static(collider.is_deformable):
                    origin_distance_sqr = gs.qd_float(1.0e20)
                    for candidate_voxel_idx in range(collider.n_voxels):
                        candidate_distance_sqr = (
                            local_pos - collider.voxel_positions[candidate_voxel_idx, env_idx]
                        ).norm_sqr()
                        if candidate_distance_sqr < origin_distance_sqr:
                            origin_distance_sqr = candidate_distance_sqr
                            origin_voxel_idx_local = candidate_voxel_idx
                else:
                    for axis in qd.static(range(3)):
                        coordinate = qd.cast(
                            qd.floor((local_pos[axis] - collider.lower_qd[axis]) / collider.voxel_size_qd[axis]),
                            gs.qd_int,
                        )
                        origin_voxel_idx[axis] = qd.max(0, qd.min(coordinate, collider.grid_res_qd[axis] - 1))
                    origin_voxel_idx_local = origin_voxel_idx[0] * collider.grid_res_qd[1] + origin_voxel_idx[1]
                    origin_voxel_idx_local = origin_voxel_idx_local * collider.grid_res_qd[2] + origin_voxel_idx[2]

                search_offset_idx = 0
                is_captured = False
                is_search_active = True
                n_search_candidates = collider.n_voxels
                if qd.static(not collider.is_deformable):
                    n_search_candidates = collider.n_voxel_search_offsets
                while search_offset_idx < n_search_candidates and not is_captured and is_search_active:
                    is_voxel_inside = True
                    voxel_offset = qd.Vector.zero(gs.qd_int, 3)
                    voxel_idx_local = -1
                    if qd.static(collider.is_deformable):
                        voxel_idx_local = collider.voxel_search_order[
                            origin_voxel_idx_local, search_offset_idx, env_idx
                        ]
                    else:
                        voxel_offset = voxel_search_offsets[collider.voxel_search_offset_start + search_offset_idx]
                        voxel_idx = origin_voxel_idx + voxel_offset
                        for axis in qd.static(range(3)):
                            is_voxel_inside = (
                                is_voxel_inside
                                and voxel_idx[axis] >= 0
                                and voxel_idx[axis] < collider.grid_res_qd[axis]
                            )
                        voxel_idx_local = voxel_idx[0] * collider.grid_res_qd[1] + voxel_idx[1]
                        voxel_idx_local = voxel_idx_local * collider.grid_res_qd[2] + voxel_idx[2]
                    if is_voxel_inside:
                        voxel_idx_global = collider.voxel_start + voxel_idx_local
                        capacity = voxel_capacity[voxel_idx_global]
                        if capacity > 0:
                            occupied = qd.atomic_add(voxel_occupancy[voxel_idx_global, env_idx], 1)
                            if occupied < capacity:
                                available_budget = absorption_capture_budget[absorption_idx, env_idx]
                                has_budget_debit = available_budget >= 1.0
                                if has_budget_debit:
                                    available_budget = qd.atomic_add(
                                        absorption_capture_budget[absorption_idx, env_idx], -1.0
                                    )
                                if available_budget >= 1.0:
                                    voxel_distance = (
                                        qd.abs(voxel_offset[0]) + qd.abs(voxel_offset[1]) + qd.abs(voxel_offset[2])
                                    )
                                    target_local_pos = collider.lower_qd + (
                                        origin_voxel_idx + voxel_offset + 0.5
                                    ) * collider.voxel_size_qd
                                    if qd.static(collider.is_deformable):
                                        yz_resolution = collider.grid_res_qd[1] * collider.grid_res_qd[2]
                                        origin_x = origin_voxel_idx_local // yz_resolution
                                        origin_remainder = origin_voxel_idx_local - origin_x * yz_resolution
                                        origin_y = origin_remainder // collider.grid_res_qd[2]
                                        origin_z = origin_remainder - origin_y * collider.grid_res_qd[2]
                                        voxel_x = voxel_idx_local // yz_resolution
                                        voxel_remainder = voxel_idx_local - voxel_x * yz_resolution
                                        voxel_y = voxel_remainder // collider.grid_res_qd[2]
                                        voxel_z = voxel_remainder - voxel_y * collider.grid_res_qd[2]
                                        voxel_distance = (
                                            qd.abs(voxel_x - origin_x)
                                            + qd.abs(voxel_y - origin_y)
                                            + qd.abs(voxel_z - origin_z)
                                        )
                                        target_local_pos = collider.voxel_positions[voxel_idx_local, env_idx]
                                    beta = 1.0 - qd.exp(-absorption_rate * substep_dt / (voxel_distance + 1))
                                    local_pos += beta * (target_local_pos - local_pos)
                                    progress = beta
                                    absorbed_pos = gu.qd_transform_by_trans_quat(
                                        local_pos,
                                        colliders_pos[collider_idx, env_idx],
                                        collider_quat,
                                    )
                                    absorbed_vel = (absorbed_pos - particles[particle_idx, env_idx].ipos) / substep_dt
                                    is_valid = not qd.math.isnan(progress) and not qd.math.isinf(progress)
                                    for axis in qd.static(range(3)):
                                        is_valid = (
                                            is_valid
                                            and not qd.math.isnan(local_pos[axis])
                                            and not qd.math.isinf(local_pos[axis])
                                            and not qd.math.isnan(absorbed_pos[axis])
                                            and not qd.math.isinf(absorbed_pos[axis])
                                            and not qd.math.isnan(absorbed_vel[axis])
                                            and not qd.math.isinf(absorbed_vel[axis])
                                        )
                                    if is_valid:
                                        absorption_particles[particle_idx, env_idx].collider_idx = collider_idx
                                        absorption_particles[particle_idx, env_idx].voxel_idx = voxel_idx_global
                                        absorption_particles[particle_idx, env_idx].voxel_distance = voxel_distance
                                        absorption_particles[particle_idx, env_idx].local_pos = local_pos
                                        absorption_particles[particle_idx, env_idx].target_local_pos = target_local_pos
                                        absorption_particles[particle_idx, env_idx].progress = progress
                                        particles[particle_idx, env_idx].pos = absorbed_pos
                                        particles[particle_idx, env_idx].vel = absorbed_vel
                                        particles[particle_idx, env_idx].density = 0.0
                                        particles[particle_idx, env_idx].dpos = qd.Vector.zero(gs.qd_float, 3)
                                        particles[particle_idx, env_idx].surface = False
                                        is_captured = True
                                    else:
                                        qd.atomic_add(absorption_capture_budget[absorption_idx, env_idx], 1.0)
                                        qd.atomic_add(voxel_occupancy[voxel_idx_global, env_idx], -1)
                                        errno[env_idx] = error_code
                                else:
                                    if has_budget_debit:
                                        qd.atomic_add(absorption_capture_budget[absorption_idx, env_idx], 1.0)
                                    qd.atomic_add(voxel_occupancy[voxel_idx_global, env_idx], -1)
                                    is_search_active = False
                            else:
                                qd.atomic_add(voxel_occupancy[voxel_idx_global, env_idx], -1)
                    search_offset_idx += 1


@qd.kernel
def kernel_rebuild_voxels(
    n_particles: qd.i32,
    n_voxels: qd.i32,
    n_absorbent_colliders: qd.i32,
    absorption_particles: qd.template(),
    absorption_capture_budget: qd.template(),
    voxel_capacity: qd.template(),
    voxel_occupancy: qd.template(),
    voxel_wetness: qd.template(),
    error_code: qd.i32,
    errno: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, absorption_particles.shape[1]):
        collider_idx = absorption_particles[particle_idx, env_idx].collider_idx
        if collider_idx >= 0:
            voxel_idx = absorption_particles[particle_idx, env_idx].voxel_idx
            voxel_distance = absorption_particles[particle_idx, env_idx].voxel_distance
            progress = absorption_particles[particle_idx, env_idx].progress
            is_valid = 0 <= voxel_idx and voxel_idx < n_voxels and voxel_distance >= 0
            is_valid = is_valid and not qd.math.isnan(progress) and not qd.math.isinf(progress)
            local_pos = absorption_particles[particle_idx, env_idx].local_pos
            target_local_pos = absorption_particles[particle_idx, env_idx].target_local_pos
            for axis in qd.static(range(3)):
                is_valid = (
                    is_valid
                    and not qd.math.isnan(local_pos[axis])
                    and not qd.math.isinf(local_pos[axis])
                    and not qd.math.isnan(target_local_pos[axis])
                    and not qd.math.isinf(target_local_pos[axis])
                )
            if is_valid:
                qd.atomic_add(voxel_occupancy[voxel_idx, env_idx], 1)
                capacity = voxel_capacity[voxel_idx]
                if capacity > 0:
                    qd.atomic_add(voxel_wetness[voxel_idx, env_idx], progress / capacity)
            else:
                errno[env_idx] = error_code

    for voxel_idx, env_idx in qd.ndrange(n_voxels, voxel_wetness.shape[1]):
        wetness = voxel_wetness[voxel_idx, env_idx]
        if qd.math.isnan(wetness) or qd.math.isinf(wetness):
            errno[env_idx] = error_code
        else:
            voxel_wetness[voxel_idx, env_idx] = qd.max(0.0, qd.min(1.0, wetness))

    for absorption_idx, env_idx in qd.ndrange(n_absorbent_colliders, absorption_capture_budget.shape[1]):
        budget = absorption_capture_budget[absorption_idx, env_idx]
        if budget < 0.0 or qd.math.isnan(budget) or qd.math.isinf(budget):
            errno[env_idx] = error_code


@qd.kernel
def kernel_check_fluid_state(
    n_particles: qd.i32,
    particles: qd.template(),
    particles_status: qd.template(),
    error_code: qd.i32,
    errno: qd.template(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, particles.shape[1]):
        if particles_status[particle_idx, env_idx].active:
            is_valid = True
            for axis in qd.static(range(3)):
                pos = particles[particle_idx, env_idx].pos[axis]
                vel = particles[particle_idx, env_idx].vel[axis]
                is_valid = (
                    is_valid
                    and not qd.math.isnan(pos)
                    and not qd.math.isinf(pos)
                    and not qd.math.isnan(vel)
                    and not qd.math.isinf(vel)
                )
            density = particles[particle_idx, env_idx].density
            is_valid = is_valid and not qd.math.isnan(density) and not qd.math.isinf(density)
            if not is_valid:
                errno[env_idx] = error_code


@qd.kernel
def kernel_get_absorption_state(
    n_particles: qd.i32,
    absorption_particles: qd.template(),
    collider_idx: qd.types.ndarray(),
    voxel_idx: qd.types.ndarray(),
    voxel_distance: qd.types.ndarray(),
    local_pos: qd.types.ndarray(),
    target_local_pos: qd.types.ndarray(),
    progress: qd.types.ndarray(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, absorption_particles.shape[1]):
        collider_idx[env_idx, particle_idx] = absorption_particles[particle_idx, env_idx].collider_idx
        voxel_idx[env_idx, particle_idx] = absorption_particles[particle_idx, env_idx].voxel_idx
        voxel_distance[env_idx, particle_idx] = absorption_particles[particle_idx, env_idx].voxel_distance
        for axis in qd.static(range(3)):
            local_pos[env_idx, particle_idx, axis] = absorption_particles[particle_idx, env_idx].local_pos[axis]
            target_local_pos[env_idx, particle_idx, axis] = absorption_particles[
                particle_idx, env_idx
            ].target_local_pos[axis]
        progress[env_idx, particle_idx] = absorption_particles[particle_idx, env_idx].progress


@qd.kernel
def kernel_set_absorption_state(
    n_particles: qd.i32,
    envs_idx: qd.types.ndarray(),
    collider_idx: qd.types.ndarray(),
    voxel_idx: qd.types.ndarray(),
    voxel_distance: qd.types.ndarray(),
    local_pos: qd.types.ndarray(),
    target_local_pos: qd.types.ndarray(),
    progress: qd.types.ndarray(),
    absorption_particles: qd.template(),
):
    for particle_idx, env_idx_local in qd.ndrange(n_particles, envs_idx.shape[0]):
        env_idx = envs_idx[env_idx_local]
        absorption_particles[particle_idx, env_idx].collider_idx = collider_idx[env_idx, particle_idx]
        absorption_particles[particle_idx, env_idx].voxel_idx = voxel_idx[env_idx, particle_idx]
        absorption_particles[particle_idx, env_idx].voxel_distance = voxel_distance[env_idx, particle_idx]
        for axis in qd.static(range(3)):
            absorption_particles[particle_idx, env_idx].local_pos[axis] = local_pos[env_idx, particle_idx, axis]
            absorption_particles[particle_idx, env_idx].target_local_pos[axis] = target_local_pos[
                env_idx, particle_idx, axis
            ]
        absorption_particles[particle_idx, env_idx].progress = progress[env_idx, particle_idx]


@qd.kernel
def kernel_get_absorption_capture_budget(
    n_absorbent_colliders: qd.i32,
    absorption_capture_budget: qd.template(),
    capture_budget: qd.types.ndarray(),
):
    for absorption_idx, env_idx in qd.ndrange(n_absorbent_colliders, absorption_capture_budget.shape[1]):
        capture_budget[env_idx, absorption_idx] = absorption_capture_budget[absorption_idx, env_idx]


@qd.kernel
def kernel_set_absorption_capture_budget(
    n_absorbent_colliders: qd.i32,
    envs_idx: qd.types.ndarray(),
    capture_budget: qd.types.ndarray(),
    absorption_capture_budget: qd.template(),
):
    for absorption_idx, env_idx_local in qd.ndrange(n_absorbent_colliders, envs_idx.shape[0]):
        env_idx = envs_idx[env_idx_local]
        absorption_capture_budget[absorption_idx, env_idx] = capture_budget[env_idx, absorption_idx]


@qd.kernel
def kernel_get_static_colliders_pose(
    n_colliders: qd.i32,
    colliders_pos: qd.template(),
    colliders_quat: qd.template(),
    pos: qd.types.ndarray(),
    quat: qd.types.ndarray(),
):
    for collider_idx, env_idx in qd.ndrange(n_colliders, colliders_pos.shape[1]):
        for axis in qd.static(range(3)):
            pos[env_idx, collider_idx, axis] = colliders_pos[collider_idx, env_idx][axis]
        for axis in qd.static(range(4)):
            quat[env_idx, collider_idx, axis] = colliders_quat[collider_idx, env_idx][axis]


@qd.kernel
def kernel_set_static_colliders_pose(
    n_colliders: qd.i32,
    envs_idx: qd.types.ndarray(),
    pos: qd.types.ndarray(),
    quat: qd.types.ndarray(),
    colliders_pos: qd.template(),
    colliders_quat: qd.template(),
):
    for collider_idx, env_idx_local in qd.ndrange(n_colliders, envs_idx.shape[0]):
        env_idx = envs_idx[env_idx_local]
        for axis in qd.static(range(3)):
            colliders_pos[collider_idx, env_idx][axis] = pos[env_idx, collider_idx, axis]
        for axis in qd.static(range(4)):
            colliders_quat[collider_idx, env_idx][axis] = quat[env_idx, collider_idx, axis]


@qd.kernel
def kernel_clear_errno(envs_idx: qd.types.ndarray(), errno: qd.template()):
    for env_idx_local in range(envs_idx.shape[0]):
        errno[envs_idx[env_idx_local]] = 0
