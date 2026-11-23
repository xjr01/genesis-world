import math
from typing import NamedTuple

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.bvh import AABB, LBVH, kernel_remap_leaf_faces
from genesis.utils import array_class, sdf
from genesis.utils.array_class import V_ANNOTATION


class RigidSurface(NamedTuple):
    info: array_class.RigidSurfaceInfo
    bvh: LBVH


def build_rigid_surface(rigid_solver, projection_geoms):
    """Build a local-space triangle atlas shared by rigid collision projections."""
    surface_geoms = [geom for geom in projection_geoms if geom.n_faces > 0]
    projection_geoms_idx = np.array([geom.idx for geom in projection_geoms], dtype=gs.np_int)
    surface_geoms_idx = np.array([geom.idx for geom in surface_geoms], dtype=gs.np_int)
    surface_geom_slots = np.full(rigid_solver.n_geoms, -1, dtype=gs.np_int)
    surface_geom_slots[surface_geoms_idx] = np.arange(len(surface_geoms), dtype=gs.np_int)

    geoms_lower = np.stack(tuple(geom.init_verts.min(axis=0) for geom in surface_geoms))
    geoms_upper = np.stack(tuple(geom.init_verts.max(axis=0) for geom in surface_geoms))
    max_geom_diagonal = np.linalg.norm(geoms_upper - geoms_lower, axis=1).max()
    # Bounded atlas coordinates retain local-feature precision. Large geoms may overlap neighboring cells;
    # leaf geom filtering preserves correctness and their coarse faces add little traversal work.
    atlas_spacing = 4.0 * max(min(max_geom_diagonal, 1.0), 1.0e-3)
    atlas_width = math.ceil(len(surface_geoms) ** (1.0 / 3.0))
    atlas_slots = np.arange(len(surface_geoms), dtype=gs.np_int)
    atlas_cells = np.empty((len(surface_geoms), 3), dtype=gs.np_float)
    atlas_cells[:, 0] = atlas_slots % atlas_width
    atlas_cells[:, 1] = atlas_slots // atlas_width % atlas_width
    atlas_cells[:, 2] = atlas_slots // (atlas_width * atlas_width)
    atlas_offsets = np.empty((len(surface_geoms), 3), dtype=gs.np_float)
    atlas_offsets[:] = atlas_spacing * atlas_cells - 0.5 * (geoms_lower + geoms_upper)

    projection_geoms_idx_qd = qd.field(dtype=gs.qd_int, shape=(len(projection_geoms_idx),))
    projection_geoms_idx_qd.from_numpy(projection_geoms_idx)
    surface_geom_slots_qd = qd.field(dtype=gs.qd_int, shape=(len(surface_geom_slots),))
    surface_geom_slots_qd.from_numpy(surface_geom_slots)
    surface_geoms_idx_qd = qd.field(dtype=gs.qd_int, shape=(len(surface_geoms_idx),))
    surface_geoms_idx_qd.from_numpy(surface_geoms_idx)
    atlas_offsets_qd = qd.Vector.field(3, dtype=gs.qd_float, shape=(len(atlas_offsets),))
    atlas_offsets_qd.from_numpy(atlas_offsets)
    surface_info = array_class.RigidSurfaceInfo(
        projection_geoms_idx=projection_geoms_idx_qd,
        surface_geom_slots=surface_geom_slots_qd,
        surface_geoms_idx=surface_geoms_idx_qd,
        atlas_offsets=atlas_offsets_qd,
    )
    faces_idx = np.concatenate(
        tuple(np.arange(geom.face_start, geom.face_end, dtype=gs.np_int) for geom in surface_geoms)
    )
    surface_aabb = AABB(n_batches=1, n_aabbs=len(faces_idx))
    surface_bvh = LBVH(surface_aabb, max_n_query_result_per_aabb=0)
    kernel_init_rigid_surface_aabbs(
        faces_idx,
        surface_bvh.aabbs,
        rigid_solver.dyn_info,
        surface_info,
    )
    surface_bvh.build()
    kernel_remap_leaf_faces(faces_idx, surface_bvh.morton_codes)
    return RigidSurface(surface_info, surface_bvh)


@qd.kernel
def kernel_init_rigid_surface_aabbs(
    faces_idx: qd.types.ndarray(ndim=1),
    surface_aabbs: V_ANNOTATION,
    dyn_info: array_class.DynInfo,
    surface_info: array_class.RigidSurfaceInfo,
):
    for face_slot in range(faces_idx.shape[0]):
        face_idx = faces_idx[face_slot]
        geom_idx = dyn_info.faces.geom_idx[face_idx]
        atlas_offset = surface_info.atlas_offsets[surface_info.surface_geom_slots[geom_idx]]
        face = dyn_info.faces.verts_idx[face_idx]
        v0 = dyn_info.verts.init_pos[face[0]] + atlas_offset
        v1 = dyn_info.verts.init_pos[face[1]] + atlas_offset
        v2 = dyn_info.verts.init_pos[face[2]] + atlas_offset
        surface_aabbs[0, face_slot].min = qd.min(v0, v1, v2)
        surface_aabbs[0, face_slot].max = qd.max(v0, v1, v2)


@qd.kernel
def kernel_reset_rigid_surface_contact(
    envs_idx: qd.types.ndarray(ndim=1),
    dyn_state: array_class.DynState,
    surface_state: array_class.RigidSurfaceContactState,
    surface_info: array_class.RigidSurfaceInfo,
):
    for env_slot, vertex_idx in qd.ndrange(envs_idx.shape[0], surface_state.corrections.shape[1]):
        env_idx = envs_idx[env_slot]
        surface_state.corrections[env_idx, vertex_idx] = qd.Vector.zero(gs.qd_float, 3)
        surface_state.n_corrections[env_idx, vertex_idx] = 0
    for env_slot in range(envs_idx.shape[0]):
        env_idx = envs_idx[env_slot]
        surface_state.is_active[env_idx] = False
        surface_state.has_intersection[env_idx] = 0
    for env_slot, surface_geom_slot in qd.ndrange(envs_idx.shape[0], surface_info.surface_geoms_idx.shape[0]):
        env_idx = envs_idx[env_slot]
        geom_idx = surface_info.surface_geoms_idx[surface_geom_slot]
        surface_state.previous_geoms_pos[env_idx, surface_geom_slot] = dyn_state.geoms.pos[geom_idx, env_idx]
        surface_state.previous_geoms_quat[env_idx, surface_geom_slot] = dyn_state.geoms.quat[geom_idx, env_idx]


@qd.kernel
def kernel_store_rigid_surface_poses(
    dyn_state: array_class.DynState,
    surface_state: array_class.RigidSurfaceContactState,
    surface_info: array_class.RigidSurfaceInfo,
):
    for env_idx, surface_geom_slot in qd.ndrange(
        surface_state.previous_geoms_pos.shape[0], surface_info.surface_geoms_idx.shape[0]
    ):
        geom_idx = surface_info.surface_geoms_idx[surface_geom_slot]
        surface_state.previous_geoms_pos[env_idx, surface_geom_slot] = dyn_state.geoms.pos[geom_idx, env_idx]
        surface_state.previous_geoms_quat[env_idx, surface_geom_slot] = dyn_state.geoms.quat[geom_idx, env_idx]


@qd.kernel
def kernel_detect_pbd_rigid_surface_intersections(
    i_iteration: int,
    surface_faces: qd.Tensor,
    particles: V_ANNOTATION,
    particles_ng: V_ANNOTATION,
    bvh_nodes: V_ANNOTATION,
    bvh_morton_codes: V_ANNOTATION,
    dyn_state: array_class.DynState,
    surface_state: array_class.RigidSurfaceContactState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    surface_info: array_class.RigidSurfaceInfo,
    is_audit: V_ANNOTATION,
    errno: qd.Tensor,
):
    for i_b, i_f in qd.ndrange(particles.shape[1], surface_faces.shape[0]):
        if not qd.static(is_audit) and i_iteration > 0 and not surface_state.is_active[i_b]:
            continue
        vertices_idx = surface_faces[i_f]
        vertices_world = qd.Matrix.zero(gs.qd_float, 3, 3)
        previous_vertices_world = qd.Matrix.zero(gs.qd_float, 3, 3)
        has_free_vertex = False
        is_active = True
        for i_v_ in qd.static(range(3)):
            i_v = vertices_idx[i_v_]
            is_active = is_active and particles_ng[i_v, i_b].active
            i_p = particles_ng[i_v, i_b].reordered_idx
            vertices_world[:, i_v_] = particles[i_p, i_b].pos
            previous_vertices_world[:, i_v_] = particles[i_p, i_b].ipos
            has_free_vertex = has_free_vertex or particles[i_p, i_b].free
        if not is_active or not has_free_vertex:
            continue
        for i_g_ in range(surface_info.surface_geoms_idx.shape[0]):
            i_g = surface_info.surface_geoms_idx[i_g_]
            clearance = sdf.sdf_func_collision_clearance(i_g, rigid_info)
            corrections, n_corrections, has_intersection = sdf.sdf_func_triangle_surface_corrections(
                i_b,
                i_g_,
                clearance,
                vertices_world,
                previous_vertices_world,
                bvh_nodes,
                bvh_morton_codes,
                dyn_state,
                surface_state,
                dyn_info,
                rigid_info,
                surface_info,
                is_audit=is_audit,
            )
            if has_intersection:
                qd.atomic_max(surface_state.has_intersection[i_b], 1)
                if qd.static(is_audit):
                    qd.atomic_or(errno[i_b], array_class.ErrorCode.INVALID_PBD_RIGID_SURFACE_INTERSECTION)
                else:
                    for i_v_ in qd.static(range(3)):
                        i_v = vertices_idx[i_v_]
                        i_p = particles_ng[i_v, i_b].reordered_idx
                        if particles[i_p, i_b].free:
                            for i_axis in qd.static(range(3)):
                                qd.atomic_add(surface_state.corrections[i_b, i_v][i_axis], corrections[i_axis, i_v_])
                            qd.atomic_add(surface_state.n_corrections[i_b, i_v], n_corrections[i_v_])


@qd.kernel
def kernel_apply_pbd_rigid_surface_corrections(
    substep_dt: float,
    particles: V_ANNOTATION,
    particles_ng: V_ANNOTATION,
    surface_state: array_class.RigidSurfaceContactState,
    errno: qd.Tensor,
):
    for i_b in range(particles.shape[1]):
        surface_state.is_active[i_b] = surface_state.has_intersection[i_b] != 0
        surface_state.has_intersection[i_b] = 0
    for i_b, i_v in qd.ndrange(particles.shape[1], particles.shape[0]):
        n_corrections = surface_state.n_corrections[i_b, i_v]
        if n_corrections > 0:
            i_p = particles_ng[i_v, i_b].reordered_idx
            corrected_pos = particles[i_p, i_b].pos + surface_state.corrections[i_b, i_v] / n_corrections
            corrected_vel = (corrected_pos - particles[i_p, i_b].ipos) / substep_dt
            if (
                qd.math.isnan(corrected_pos).any()
                or qd.math.isinf(corrected_pos).any()
                or qd.math.isnan(corrected_vel).any()
                or qd.math.isinf(corrected_vel).any()
            ):
                qd.atomic_or(errno[i_b], array_class.ErrorCode.INVALID_CONTACT_NAN)
            else:
                particles[i_p, i_b].pos = corrected_pos
                particles[i_p, i_b].vel = corrected_vel
        surface_state.corrections[i_b, i_v] = qd.Vector.zero(gs.qd_float, 3)
        surface_state.n_corrections[i_b, i_v] = 0
