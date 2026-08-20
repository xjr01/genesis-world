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
    absorption_particles[particle_idx, env_idx].local_pos = qd.Vector.zero(gs.qd_float, 3)
    absorption_particles[particle_idx, env_idx].target_local_pos = qd.Vector.zero(gs.qd_float, 3)
    absorption_particles[particle_idx, env_idx].progress = 0.0


@qd.kernel
def kernel_initialize_absorption_particles(n_particles: qd.i32, absorption_particles: qd.template()):
    for particle_idx, env_idx in qd.ndrange(n_particles, absorption_particles.shape[1]):
        absorption_particles[particle_idx, env_idx].collider_idx = -1
        absorption_particles[particle_idx, env_idx].voxel_idx = -1
        absorption_particles[particle_idx, env_idx].local_pos = qd.Vector.zero(gs.qd_float, 3)
        absorption_particles[particle_idx, env_idx].target_local_pos = qd.Vector.zero(gs.qd_float, 3)
        absorption_particles[particle_idx, env_idx].progress = 0.0


@qd.kernel
def kernel_update_absorbed_particles(
    n_particles: qd.i32,
    collider_idx: qd.i32,
    substep_dt: float,
    beta: float,
    particles: qd.template(),
    particles_status: qd.template(),
    absorption_particles: qd.template(),
    colliders_pos: qd.template(),
    colliders_quat: qd.template(),
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
            progress = absorption_particles[particle_idx, env_idx].progress
            local_pos += beta * (target_local_pos - local_pos)
            progress += beta * (1.0 - progress)
            pos_prev = particles[particle_idx, env_idx].pos
            pos = gu.qd_transform_by_trans_quat(
                local_pos,
                colliders_pos[collider_idx, env_idx],
                colliders_quat[collider_idx, env_idx],
            )
            vel = (pos - pos_prev) / substep_dt

            is_valid = not qd.math.isnan(progress) and not qd.math.isinf(progress)
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
    particle_radius: float,
    substep_dt: float,
    beta: float,
    particles: qd.template(),
    particles_status: qd.template(),
    absorption_particles: qd.template(),
    voxel_capacity: qd.template(),
    voxel_occupancy: qd.template(),
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
            _, normal, is_inside, surface_distance = query_static_collider(
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
                normal_local = gu.qd_inv_transform_by_quat(normal, collider_quat)
                face_axis = 0
                face_normal = normal_local[0]
                face_res = collider.grid_res_qd[0]
                face_normal_abs = qd.abs(face_normal)
                if qd.abs(normal_local[1]) > face_normal_abs:
                    face_axis = 1
                    face_normal = normal_local[1]
                    face_res = collider.grid_res_qd[1]
                    face_normal_abs = qd.abs(face_normal)
                if qd.abs(normal_local[2]) > face_normal_abs:
                    face_axis = 2
                    face_normal = normal_local[2]
                    face_res = collider.grid_res_qd[2]

                voxel_idx = qd.Vector.zero(gs.qd_int, 3)
                for axis in qd.static(range(3)):
                    coordinate = qd.cast(
                        qd.floor((local_pos[axis] - collider.lower_qd[axis]) / collider.voxel_size_qd[axis]),
                        gs.qd_int,
                    )
                    voxel_idx[axis] = qd.max(0, qd.min(coordinate, collider.grid_res_qd[axis] - 1))

                depth = 0
                is_captured = False
                while depth < face_res and not is_captured:
                    face_coordinate = depth
                    if face_normal >= 0.0:
                        face_coordinate = face_res - 1 - depth
                    if face_axis == 0:
                        voxel_idx[0] = face_coordinate
                    elif face_axis == 1:
                        voxel_idx[1] = face_coordinate
                    else:
                        voxel_idx[2] = face_coordinate
                    voxel_idx_local = voxel_idx[0] * collider.grid_res_qd[1] + voxel_idx[1]
                    voxel_idx_local = voxel_idx_local * collider.grid_res_qd[2] + voxel_idx[2]
                    voxel_idx_global = collider.voxel_start + voxel_idx_local
                    capacity = voxel_capacity[voxel_idx_global]
                    if capacity > 0:
                        occupied = qd.atomic_add(voxel_occupancy[voxel_idx_global, env_idx], 1)
                        if occupied < capacity:
                            target_local_pos = collider.lower_qd + (voxel_idx + 0.5) * collider.voxel_size_qd
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
                                qd.atomic_add(voxel_occupancy[voxel_idx_global, env_idx], -1)
                                errno[env_idx] = error_code
                        else:
                            qd.atomic_add(voxel_occupancy[voxel_idx_global, env_idx], -1)
                    depth += 1


@qd.kernel
def kernel_rebuild_voxels(
    n_particles: qd.i32,
    n_voxels: qd.i32,
    absorption_particles: qd.template(),
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
            progress = absorption_particles[particle_idx, env_idx].progress
            is_valid = 0 <= voxel_idx and voxel_idx < n_voxels
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
    local_pos: qd.types.ndarray(),
    target_local_pos: qd.types.ndarray(),
    progress: qd.types.ndarray(),
):
    for particle_idx, env_idx in qd.ndrange(n_particles, absorption_particles.shape[1]):
        collider_idx[env_idx, particle_idx] = absorption_particles[particle_idx, env_idx].collider_idx
        voxel_idx[env_idx, particle_idx] = absorption_particles[particle_idx, env_idx].voxel_idx
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
    local_pos: qd.types.ndarray(),
    target_local_pos: qd.types.ndarray(),
    progress: qd.types.ndarray(),
    absorption_particles: qd.template(),
):
    for particle_idx, env_idx_local in qd.ndrange(n_particles, envs_idx.shape[0]):
        env_idx = envs_idx[env_idx_local]
        absorption_particles[particle_idx, env_idx].collider_idx = collider_idx[env_idx, particle_idx]
        absorption_particles[particle_idx, env_idx].voxel_idx = voxel_idx[env_idx, particle_idx]
        for axis in qd.static(range(3)):
            absorption_particles[particle_idx, env_idx].local_pos[axis] = local_pos[env_idx, particle_idx, axis]
            absorption_particles[particle_idx, env_idx].target_local_pos[axis] = target_local_pos[
                env_idx, particle_idx, axis
            ]
        absorption_particles[particle_idx, env_idx].progress = progress[env_idx, particle_idx]


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
