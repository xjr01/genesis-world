import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.bvh import STACK_SIZE, point_aabb_distance_sqr
from genesis.utils.triangle_qd import (
    closest_point_on_triangle,
    ray_aabb_intersection,
    ray_projection,
    ray_triangle_intersection,
    triangle_face_normal,
    triangle_separating_corrections,
    triangle_triangle_intersection,
    triangle_triangle_previous_separating_correction,
)


class SDF:
    def __init__(self, rigid_solver):
        self.solver = rigid_solver
        self._geoms_sdf_coarse_res = np.array(
            [(geom.sdf_res - 2) // 4 + 1 for geom in rigid_solver.geoms], dtype=gs.np_int
        ).reshape((-1, 3))
        n_coarse_cells = int(self._geoms_sdf_coarse_res.prod(axis=-1).sum())
        self._sdf_info = array_class.get_sdf_info(self.solver.n_geoms, self.solver.n_cells, n_coarse_cells)
        self._is_active = False

    def activate(self):
        if self._is_active:
            return

        if self.solver.n_geoms > 0:
            geoms = self.solver.geoms
            # Coarse min-grid companion of each SDF: block minima over the grid nodes, with blocks overlapping by
            # one node so that every interpolation cell's 8 nodes lie inside the block of its coarse cell. Derived
            # from the loaded grids, so the preprocessing cache is untouched.
            geoms_sdf_coarse_val = []
            for geom, coarse_res in zip(geoms, self._geoms_sdf_coarse_res):
                coarse_val = geom.sdf_val
                for axis in range(3):
                    windows = np.minimum(
                        4 * np.arange(coarse_res[axis])[:, None] + np.arange(5), coarse_val.shape[axis] - 1
                    )
                    coarse_val = np.take(coarse_val, windows, axis=axis).min(axis=axis + 1)
                geoms_sdf_coarse_val.append(coarse_val.reshape((-1,)))
            sdf_kernel_init_geom_fields(
                np.array([geom.cell_start for geom in geoms], dtype=gs.np_int),
                np.concatenate(([0], self._geoms_sdf_coarse_res.prod(axis=-1).cumsum()[:-1]), dtype=gs.np_int),
                np.array([geom.T_mesh_to_sdf for geom in geoms], dtype=gs.np_float),
                np.array([geom.sdf_res for geom in geoms], dtype=gs.np_int),
                np.concatenate([geom.sdf_val_flattened for geom in geoms], dtype=gs.np_float),
                np.concatenate([geom.sdf_grad_flattened for geom in geoms], dtype=gs.np_float),
                np.array([geom.sdf_max for geom in geoms], dtype=gs.np_float),
                np.array(
                    [np.broadcast_to(np.asarray(geom.sdf_cell_size, dtype=gs.np_float), (3,)) for geom in geoms],
                    dtype=gs.np_float,
                ),
                np.concatenate([geom.sdf_closest_vert_flattened for geom in geoms], dtype=gs.np_int),
                self._geoms_sdf_coarse_res,
                np.concatenate(geoms_sdf_coarse_val, dtype=gs.np_float),
                self._sdf_info,
                self.solver.rigid_config,
            )

        self._is_active = True

    @property
    def is_active(self):
        return self._is_active


@qd.kernel
def sdf_kernel_init_geom_fields(
    geoms_sdf_cell_start: qd.types.ndarray(),
    geoms_sdf_coarse_cell_start: qd.types.ndarray(),
    geoms_T_mesh_to_sdf: qd.types.ndarray(),
    geoms_sdf_res: qd.types.ndarray(),
    geoms_sdf_val: qd.types.ndarray(),
    geoms_sdf_grad: qd.types.ndarray(),
    geoms_sdf_max: qd.types.ndarray(),
    geoms_sdf_cell_size: qd.types.ndarray(),
    geoms_sdf_closest_vert: qd.types.ndarray(),
    geoms_sdf_coarse_res: qd.types.ndarray(),
    geoms_sdf_coarse_val: qd.types.ndarray(),
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
):
    n_geoms = sdf_info.geoms_sdf_start.shape[0]
    n_cells = sdf_info.geoms_sdf_val.shape[0]
    n_coarse_cells = sdf_info.geoms_sdf_coarse_val.shape[0]

    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i in range(n_geoms):
        for j, k in qd.static(qd.ndrange(4, 4)):
            sdf_info.geoms_info.T_mesh_to_sdf[i][j, k] = geoms_T_mesh_to_sdf[i, j, k]

        for j in qd.static(range(3)):
            sdf_info.geoms_info.sdf_res[i][j] = geoms_sdf_res[i, j]

        sdf_info.geoms_info.sdf_cell_start[i] = geoms_sdf_cell_start[i]
        sdf_info.geoms_info.sdf_max[i] = geoms_sdf_max[i]
        for j in qd.static(range(3)):
            sdf_info.geoms_info.sdf_cell_size[i][j] = geoms_sdf_cell_size[i, j]
        sdf_info.geoms_info.sdf_coarse_cell_start[i] = geoms_sdf_coarse_cell_start[i]
        for j in qd.static(range(3)):
            sdf_info.geoms_info.sdf_coarse_res[i][j] = geoms_sdf_coarse_res[i, j]

    for i in range(n_cells):
        sdf_info.geoms_sdf_val[i] = geoms_sdf_val[i]
        sdf_info.geoms_sdf_closest_vert[i] = geoms_sdf_closest_vert[i]
        for j in qd.static(range(3)):
            sdf_info.geoms_sdf_grad[i][j] = geoms_sdf_grad[i, j]

    for i in range(n_coarse_cells):
        sdf_info.geoms_sdf_coarse_val[i] = geoms_sdf_coarse_val[i]


@qd.func
def sdf_func_world(
    geom_idx,
    batch_idx,
    pos_world,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
):
    """
    sdf value from world coordinate
    """

    g_pos = geoms_state.pos[geom_idx, batch_idx]
    g_quat = geoms_state.quat[geom_idx, batch_idx]

    return sdf_func_world_local(geom_idx, pos_world, g_pos, g_quat, geoms_info, sdf_info)


@qd.func
def sdf_func_world_local(
    geom_idx,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
):
    """
    Computes SDF value from world coordinate, using provided geometry pose
    instead of reading from geoms_state.
    """
    sd = gs.qd_float(0.0)

    if geoms_info.type[geom_idx] == gs.GEOM_TYPE.SPHERE:
        sd = (pos_world - geom_pos).norm() - geoms_info.data[geom_idx][0]

    elif geoms_info.type[geom_idx] == gs.GEOM_TYPE.PLANE:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        geom_data = geoms_info.data[geom_idx]
        plane_normal = gs.qd_vec3([geom_data[0], geom_data[1], geom_data[2]])
        sd = pos_mesh.dot(plane_normal)

    else:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[geom_idx])
        sd = sdf_func_sdf(geom_idx, pos_sdf, sdf_info)

    return sd


@qd.func
def sdf_func_coarse_sd_lower_bound(geom_idx, pos_sdf, collider_info: array_class.ColliderInfo):
    """
    Certified lower bound on the trilinear sd at an in-grid point: the minimum node value over the 4^3-cell node
    block containing its interpolation cell. Exact by convexity - the interpolant only combines nodes of that
    block - at the cost of a single load instead of the 8-node gather.
    """
    res = collider_info.sdf.geoms_info.sdf_res[geom_idx]
    base = qd.min(qd.floor(pos_sdf, gs.qd_int), res - 2)
    coarse_cell = base // 4
    coarse_res = collider_info.sdf.geoms_info.sdf_coarse_res[geom_idx]
    return collider_info.sdf.geoms_sdf_coarse_val[
        collider_info.sdf.geoms_info.sdf_coarse_cell_start[geom_idx]
        + (coarse_cell[0] * coarse_res[1] + coarse_cell[1]) * coarse_res[2]
        + coarse_cell[2]
    ]


@qd.func
def sdf_func_world_local_banded(
    geom_idx,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    band,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
):
    """
    Band-gated variant of sdf_func_world_local, returning (is_in_band, sd).

    is_in_band == (sd < band) for exactly the sd sdf_func_world_local would return, but sd itself is only computed
    (and meaningful) when in band: an in-grid query whose coarse block minimum already clears the band skips the
    8-node gather for a single coarse load.
    """
    is_in_band = False
    sd = gs.qd_float(0.0)

    if dyn_info.geoms.type[geom_idx] == gs.GEOM_TYPE.SPHERE:
        sd = (pos_world - geom_pos).norm() - dyn_info.geoms.data[geom_idx][0]
        is_in_band = sd < band

    elif dyn_info.geoms.type[geom_idx] == gs.GEOM_TYPE.PLANE:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        geom_data = dyn_info.geoms.data[geom_idx]
        plane_normal = gs.qd_vec3([geom_data[0], geom_data[1], geom_data[2]])
        sd = pos_mesh.dot(plane_normal)
        is_in_band = sd < band

    else:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        pos_sdf = gu.qd_transform_by_T(pos_mesh, collider_info.sdf.geoms_info.T_mesh_to_sdf[geom_idx])
        if sdf_func_is_outside_sdf_grid(geom_idx, pos_sdf, collider_info.sdf):
            sd = sdf_func_proxy_sdf(geom_idx, pos_sdf, collider_info.sdf)
            is_in_band = sd < band
        else:
            coarse_lower_bound = sdf_func_coarse_sd_lower_bound(geom_idx, pos_sdf, collider_info)
            # The bound holds in exact arithmetic, but the floating-point evaluation of the trilinear sum can
            # round below the block minimum; the relative guard keeps a vertex whose exact interpolant clears the
            # band from being misclassified when its rounded value dips just inside.
            if not (coarse_lower_bound - 1e-6 * (1.0 + qd.abs(coarse_lower_bound)) >= band):
                sd = sdf_func_true_sdf(geom_idx, pos_sdf, collider_info.sdf)
                is_in_band = sd < band

    return is_in_band, sd


@qd.func
def sdf_func_ray_exit_distance(
    geom_idx,
    origin: qd.types.vector(3),
    direction: qd.types.vector(3),
    max_dist,
    tolerance,
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
):
    """
    Distance from a point inside the geom to its surface along a unit direction, bisected down to tolerance.
    """
    dist = max_dist
    sd_end = sdf_func_world_local(
        geom_idx, origin + max_dist * direction, geom_pos, geom_quat, dyn_info.geoms, collider_info.sdf
    )
    if sd_end > 0.0:
        t_lo = gs.qd_float(0.0)
        t_hi = max_dist
        while t_hi - t_lo > tolerance:
            t_mid = 0.5 * (t_lo + t_hi)
            sd_mid = sdf_func_world_local(
                geom_idx, origin + t_mid * direction, geom_pos, geom_quat, dyn_info.geoms, collider_info.sdf
            )
            if sd_mid < 0.0:
                t_lo = t_mid
            else:
                t_hi = t_mid
        dist = 0.5 * (t_lo + t_hi)
    return dist


@qd.func
def sdf_func_sdf(geom_idx, pos_sdf, sdf_info: array_class.SDFInfo):
    """
    sdf value at sdf frame coordinate.
    Note that the stored sdf magnitude is already w.r.t world/mesh frame.
    """
    signed_dist = gs.qd_float(0.0)
    if sdf_func_is_outside_sdf_grid(geom_idx, pos_sdf, sdf_info):
        signed_dist = sdf_func_proxy_sdf(geom_idx, pos_sdf, sdf_info)
    else:
        signed_dist = sdf_func_true_sdf(geom_idx, pos_sdf, sdf_info)
    return signed_dist


@qd.func
def sdf_func_is_outside_sdf_grid(geom_idx, pos_sdf, sdf_info: array_class.SDFInfo):
    res = sdf_info.geoms_info.sdf_res[geom_idx]
    return (pos_sdf >= res - 1).any() or (pos_sdf <= 0).any()


@qd.func
def sdf_func_proxy_sdf(geom_idx, pos_sdf, sdf_info: array_class.SDFInfo):
    """
    Use distance to center as a proxy sdf, strictly greater than any point inside the cube to ensure value comparison
    is valid.

    Only considers region outside of cube. For anisotropic SDF grids the per-axis cell sizes are applied before taking
    the norm so the result remains a world distance.
    """
    center = (sdf_info.geoms_info.sdf_res[geom_idx] - 1) / 2.0
    delta = pos_sdf - center
    cs = sdf_info.geoms_info.sdf_cell_size[geom_idx]
    scaled = qd.Vector([delta[0] * cs[0], delta[1] * cs[1], delta[2] * cs[2]], dt=gs.qd_float)
    return scaled.norm() + sdf_info.geoms_info.sdf_max[geom_idx]


@qd.func
def sdf_func_true_sdf(geom_idx, pos_sdf, sdf_info: array_class.SDFInfo):
    """
    True sdf interpolated using stored sdf grid.
    """
    geom_sdf_res = sdf_info.geoms_info.sdf_res[geom_idx]
    base = qd.min(qd.floor(pos_sdf, gs.qd_int), geom_sdf_res - 2)
    signed_dist = gs.qd_float(0.0)
    for offset in qd.grouped(qd.ndrange(2, 2, 2)):
        pos_cell = base + offset
        w_xyz = 1 - qd.abs(pos_sdf - pos_cell)
        w = w_xyz[0] * w_xyz[1] * w_xyz[2]
        signed_dist = (
            signed_dist
            + w * sdf_info.geoms_sdf_val[sdf_func_ravel_cell_idx(pos_cell, geom_idx, geom_sdf_res, sdf_info)]
        )

    return signed_dist


@qd.func
def sdf_func_ravel_cell_idx(cell_idx, geom_idx, sdf_res, sdf_info: array_class.SDFInfo):
    return (
        sdf_info.geoms_info.sdf_cell_start[geom_idx]
        + cell_idx[0] * sdf_res[1] * sdf_res[2]
        + cell_idx[1] * sdf_res[2]
        + cell_idx[2]
    )


@qd.func
def sdf_func_grad_world(
    geom_idx,
    batch_idx,
    pos_world,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
):
    g_pos = dyn_state.geoms.pos[geom_idx, batch_idx]
    g_quat = dyn_state.geoms.quat[geom_idx, batch_idx]

    return sdf_func_grad_world_local(
        geom_idx, pos_world, g_pos, g_quat, dyn_info.geoms, rigid_info, collider_info.sdf, collider_static_config
    )


@qd.func
def sdf_func_grad(
    geom_idx,
    pos_sdf,
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    """
    sdf grad at sdf frame coordinate.

    Note that the stored sdf magnitude is already w.r.t world/mesh frame.
    """
    grad_sdf = qd.Vector.zero(gs.qd_float, 3)
    if sdf_func_is_outside_sdf_grid(geom_idx, pos_sdf, sdf_info):
        grad_sdf = sdf_func_proxy_grad(geom_idx, pos_sdf, rigid_info, sdf_info)
    else:
        grad_sdf = sdf_func_true_grad(geom_idx, pos_sdf, geoms_info, sdf_info, collider_static_config)
    return grad_sdf


@qd.func
def sdf_func_proxy_grad(geom_idx, pos_sdf, rigid_info: array_class.RigidInfo, sdf_info: array_class.SDFInfo):
    """
    Use direction from sdf center, scaled per-axis by the anisotropic cell size, to approximate the gradient
    direction outside the cube.

    The matching :func:`sdf_func_proxy_sdf` distance is `||(pos_sdf - center) * cs||` in world units, whose gradient
    direction (after the chain rule for the diagonal SDF<->mesh transform) is the per-axis-scaled delta. Using the raw
    `pos_sdf - center` would skew outside-grid normals toward fine-resolution axes on anisotropic grids and yield
    directionally wrong contact normals for points falling back on this proxy.
    """
    center = (sdf_info.geoms_info.sdf_res[geom_idx] - 1) / 2.0
    delta = pos_sdf - center
    cs = sdf_info.geoms_info.sdf_cell_size[geom_idx]
    scaled = qd.Vector([delta[0] * cs[0], delta[1] * cs[1], delta[2] * cs[2]], dt=gs.qd_float)
    proxy_sdf_grad = gu.qd_normalize(scaled, rigid_info.EPS[None])
    return proxy_sdf_grad


@qd.func
def sdf_func_true_grad(
    geom_idx,
    pos_sdf,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    """
    True sdf grad interpolated using stored sdf grad grid.
    """
    sdf_grad_sdf = qd.Vector.zero(gs.qd_float, 3)
    if geoms_info.type[geom_idx] == gs.GEOM_TYPE.TERRAIN:  # Terrain uses finite difference
        if qd.static(collider_static_config.has_terrain):  # for speed up compilation
            # since we are in sdf frame, delta can be a relatively big value
            delta = gs.qd_float(1e-2)

            for i in qd.static(range(3)):
                inc = pos_sdf
                dec = pos_sdf
                inc[i] += delta
                dec[i] -= delta
                sdf_grad_sdf[i] = (
                    sdf_func_true_sdf(geom_idx, inc, sdf_info) - sdf_func_true_sdf(geom_idx, dec, sdf_info)
                ) / (2 * delta)

    else:
        geom_sdf_res = sdf_info.geoms_info.sdf_res[geom_idx]
        base = qd.min(qd.floor(pos_sdf, gs.qd_int), geom_sdf_res - 2)
        for offset in qd.grouped(qd.ndrange(2, 2, 2)):
            pos_cell = base + offset
            w_xyz = 1 - qd.abs(pos_sdf - pos_cell)
            w = w_xyz[0] * w_xyz[1] * w_xyz[2]
            sdf_grad_sdf = (
                sdf_grad_sdf
                + w * sdf_info.geoms_sdf_grad[sdf_func_ravel_cell_idx(pos_cell, geom_idx, geom_sdf_res, sdf_info)]
            )

    return sdf_grad_sdf


@qd.func
def sdf_func_grad_world_local_consistent(
    geom_idx,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
):
    """
    SDF gradient in world coordinates as the analytic gradient of the trilinear value interpolant, NOT the
    interpolation of precomputed lattice gradients that sdf_func_grad_world_local returns.

    A contact whose normal is the exact gradient of the field supplying its penetration derives from a potential
    and can do no net work around a closed micro-cycle; the lattice-grad interpolation is smoother but tilted off
    the penetration level sets, which makes a settled stack of nested shells ratchet sideways. That smoothness is
    load-bearing for sliding thin features (bolt threads), so only the nested-shell contact path uses this variant.
    """
    EPS = rigid_info.EPS[None]
    grad_world = qd.Vector.zero(gs.qd_float, 3)
    if dyn_info.geoms.type[geom_idx] == gs.GEOM_TYPE.SPHERE:
        grad_world = gu.qd_normalize(pos_world - geom_pos, EPS)
    elif dyn_info.geoms.type[geom_idx] == gs.GEOM_TYPE.PLANE:
        geom_data = dyn_info.geoms.data[geom_idx]
        plane_normal = gs.qd_vec3([geom_data[0], geom_data[1], geom_data[2]])
        grad_world = gu.qd_transform_by_quat(plane_normal, geom_quat)
    else:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        pos_sdf = gu.qd_transform_by_T(pos_mesh, collider_info.sdf.geoms_info.T_mesh_to_sdf[geom_idx])
        grad_mesh = qd.Vector.zero(gs.qd_float, 3)
        if sdf_func_is_outside_sdf_grid(geom_idx, pos_sdf, collider_info.sdf):
            grad_mesh = sdf_func_proxy_grad(geom_idx, pos_sdf, rigid_info, collider_info.sdf)
        else:
            geom_sdf_res = collider_info.sdf.geoms_info.sdf_res[geom_idx]
            cs = collider_info.sdf.geoms_info.sdf_cell_size[geom_idx]
            base = qd.min(qd.floor(pos_sdf, gs.qd_int), geom_sdf_res - 2)
            for offset in qd.grouped(qd.ndrange(2, 2, 2)):
                pos_cell = base + offset
                w_xyz = 1 - qd.abs(pos_sdf - pos_cell)
                val = collider_info.sdf.geoms_sdf_val[
                    sdf_func_ravel_cell_idx(pos_cell, geom_idx, geom_sdf_res, collider_info.sdf)
                ]
                grad_mesh[0] += (2 * offset[0] - 1) * w_xyz[1] * w_xyz[2] * val / cs[0]
                grad_mesh[1] += w_xyz[0] * (2 * offset[1] - 1) * w_xyz[2] * val / cs[1]
                grad_mesh[2] += w_xyz[0] * w_xyz[1] * (2 * offset[2] - 1) * val / cs[2]
        grad_world = gu.qd_transform_by_quat(grad_mesh, geom_quat)
    return grad_world


@qd.func
def sdf_func_exact_mesh_surface_bvh_local(
    geom_idx,
    pos_mesh,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    surface_info: array_class.RigidSurfaceInfo,
):
    """Query one rigid geom in a shared local-coordinate bounding volume hierarchy (BVH)."""
    surface_geom_slot = surface_info.surface_geom_slots[geom_idx]
    atlas_offset = surface_info.atlas_offsets[surface_geom_slot]
    pos_atlas = pos_mesh + atlas_offset
    geom_extent = rigid_info.geoms_init_AABB[geom_idx, 7] - rigid_info.geoms_init_AABB[geom_idx, 0]
    max_distance = 2.0 * qd.max(1.0e-3, geom_extent.norm())
    surface_distance_sqr = gs.qd_float(1.0e30)
    closest_position = pos_atlas
    closest_normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    closest_normal_sum = qd.Vector.zero(gs.qd_float, 3)
    closest_face_idx = dyn_info.faces.verts_idx.shape[0]
    has_closest = False

    n_triangles = bvh_morton_codes.shape[1]
    node_stack = qd.Vector.zero(gs.qd_int, qd.static(STACK_SIZE))
    node_stack[0] = 0
    stack_idx = 1
    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]
        node = bvh_nodes[0, node_idx]
        distance_tolerance = 8.0 * rigid_info.EPS[None] * qd.max(1.0, surface_distance_sqr)
        node_distance_sqr = point_aabb_distance_sqr(pos_atlas, node.bound.min, node.bound.max)
        if node_distance_sqr <= surface_distance_sqr + distance_tolerance:
            if node.left == -1:
                sorted_leaf_idx = node_idx - (n_triangles - 1)
                face_idx = qd.cast(bvh_morton_codes[0, sorted_leaf_idx][1], gs.qd_int)
                if dyn_info.faces.geom_idx[face_idx] == geom_idx:
                    face = dyn_info.faces.verts_idx[face_idx]
                    v0 = dyn_info.verts.init_pos[face[0]] + atlas_offset
                    v1 = dyn_info.verts.init_pos[face[1]] + atlas_offset
                    v2 = dyn_info.verts.init_pos[face[2]] + atlas_offset
                    candidate_normal = triangle_face_normal(v0, v1, v2)
                    candidate = closest_point_on_triangle(pos_atlas, v0, v1, v2)
                    candidate_distance_sqr = (pos_atlas - candidate).norm_sqr()
                    candidate_tolerance = (
                        8.0 * rigid_info.EPS[None] * qd.max(1.0, candidate_distance_sqr, surface_distance_sqr)
                    )
                    if candidate_distance_sqr <= surface_distance_sqr + candidate_tolerance:
                        if not has_closest or candidate_distance_sqr < surface_distance_sqr - candidate_tolerance:
                            closest_normal_sum = candidate_normal
                        elif qd.abs(candidate_distance_sqr - surface_distance_sqr) <= candidate_tolerance:
                            closest_normal_sum += candidate_normal
                        if (
                            not has_closest
                            or candidate_distance_sqr < surface_distance_sqr
                            or (candidate_distance_sqr == surface_distance_sqr and face_idx < closest_face_idx)
                        ):
                            closest_position = candidate
                            closest_normal = candidate_normal
                            closest_face_idx = face_idx
                            surface_distance_sqr = candidate_distance_sqr
                        has_closest = True
            elif stack_idx < qd.static(STACK_SIZE - 2):
                left = node.left
                right = node.right
                left_distance_sqr = point_aabb_distance_sqr(
                    pos_atlas, bvh_nodes[0, left].bound.min, bvh_nodes[0, left].bound.max
                )
                right_distance_sqr = point_aabb_distance_sqr(
                    pos_atlas, bvh_nodes[0, right].bound.min, bvh_nodes[0, right].bound.max
                )
                if left_distance_sqr < right_distance_sqr:
                    node_stack[stack_idx] = right
                    node_stack[stack_idx + 1] = left
                else:
                    node_stack[stack_idx] = left
                    node_stack[stack_idx + 1] = right
                stack_idx += 2

    surface_distance = max_distance
    inside_normal = closest_normal
    if has_closest:
        surface_distance = qd.sqrt(surface_distance_sqr)
        if closest_normal_sum.norm_sqr() > rigid_info.EPS[None] ** 2:
            inside_normal = closest_normal_sum.normalized()

    delta = pos_atlas - closest_position
    is_inside_candidate = delta.dot(inside_normal) <= 0.0
    is_inside = surface_distance <= rigid_info.EPS[None]
    if is_inside_candidate and not is_inside:
        ray_dir = qd.Vector([0.8192319205, 0.4630140578, 0.3395271683], dt=gs.qd_float)
        axes, shear, is_valid_dir = ray_projection(ray_dir, rigid_info.EPS[None])
        winding_crossings = gs.qd_int(0)
        node_stack[0] = 0
        stack_idx = 1
        if not is_valid_dir:
            stack_idx = 0
        while stack_idx > 0:
            stack_idx -= 1
            node_idx = node_stack[stack_idx]
            node = bvh_nodes[0, node_idx]
            aabb_distance = ray_aabb_intersection(
                pos_atlas, ray_dir, node.bound.min, node.bound.max, rigid_info.EPS[None]
            )
            if aabb_distance >= 0.0 and aabb_distance <= max_distance:
                if node.left == -1:
                    sorted_leaf_idx = node_idx - (n_triangles - 1)
                    face_idx = qd.cast(bvh_morton_codes[0, sorted_leaf_idx][1], gs.qd_int)
                    if dyn_info.faces.geom_idx[face_idx] == geom_idx:
                        face = dyn_info.faces.verts_idx[face_idx]
                        v0 = dyn_info.verts.init_pos[face[0]] + atlas_offset
                        v1 = dyn_info.verts.init_pos[face[1]] + atlas_offset
                        v2 = dyn_info.verts.init_pos[face[2]] + atlas_offset
                        hit_distance = ray_triangle_intersection(
                            axes, pos_atlas, shear, v0, v1, v2, rigid_info.EPS[None]
                        )
                        if hit_distance >= 0.0 and hit_distance <= max_distance:
                            alignment = triangle_face_normal(v0, v1, v2).dot(ray_dir)
                            if alignment > rigid_info.EPS[None]:
                                winding_crossings += 1
                            elif alignment < -rigid_info.EPS[None]:
                                winding_crossings -= 1
                elif stack_idx < qd.static(STACK_SIZE - 2):
                    node_stack[stack_idx] = node.left
                    node_stack[stack_idx + 1] = node.right
                    stack_idx += 2
        is_inside = winding_crossings != 0

    if surface_distance > rigid_info.EPS[None]:
        if is_inside:
            closest_normal = -delta / surface_distance
        else:
            closest_normal = delta / surface_distance

    return closest_position - atlas_offset, closest_normal, is_inside, surface_distance


@qd.func
def sdf_func_exact_mesh_surface_bvh(
    geom_idx,
    env_idx,
    pos_world,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    surface_info: array_class.RigidSurfaceInfo,
):
    geom_pos = dyn_state.geoms.pos[geom_idx, env_idx]
    geom_quat = dyn_state.geoms.quat[geom_idx, env_idx]
    pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
    closest_position, closest_normal, is_inside, surface_distance = sdf_func_exact_mesh_surface_bvh_local(
        geom_idx,
        pos_mesh,
        bvh_nodes,
        bvh_morton_codes,
        dyn_info,
        rigid_info,
        surface_info,
    )
    return (
        gu.qd_transform_by_trans_quat(closest_position, geom_pos, geom_quat),
        gu.qd_transform_by_quat(closest_normal, geom_quat),
        is_inside,
        surface_distance,
    )


@qd.func
def sdf_func_surface_bvh_ray_cast_local(
    geom_idx,
    ray_start_mesh,
    ray_dir,
    max_range,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    surface_info: array_class.RigidSurfaceInfo,
):
    """Return the first hit along a geom-local ray in the shared rigid-surface BVH."""
    atlas_offset = surface_info.atlas_offsets[surface_info.surface_geom_slots[geom_idx]]
    ray_start = ray_start_mesh + atlas_offset
    axes, shear, is_valid_dir = ray_projection(ray_dir, rigid_info.EPS[None])
    closest_distance = max_range
    has_hit = False
    n_triangles = bvh_morton_codes.shape[1]
    node_stack = qd.Vector.zero(gs.qd_int, qd.static(STACK_SIZE))
    node_stack[0] = 0
    stack_idx = 1
    if not is_valid_dir:
        stack_idx = 0
    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]
        node = bvh_nodes[0, node_idx]
        aabb_distance = ray_aabb_intersection(ray_start, ray_dir, node.bound.min, node.bound.max, rigid_info.EPS[None])
        if aabb_distance >= 0.0 and aabb_distance < closest_distance:
            if node.left == -1:
                sorted_leaf_idx = node_idx - (n_triangles - 1)
                face_idx = qd.cast(bvh_morton_codes[0, sorted_leaf_idx][1], gs.qd_int)
                if dyn_info.faces.geom_idx[face_idx] == geom_idx:
                    face = dyn_info.faces.verts_idx[face_idx]
                    hit_distance = ray_triangle_intersection(
                        axes,
                        ray_start,
                        shear,
                        dyn_info.verts.init_pos[face[0]] + atlas_offset,
                        dyn_info.verts.init_pos[face[1]] + atlas_offset,
                        dyn_info.verts.init_pos[face[2]] + atlas_offset,
                        rigid_info.EPS[None],
                    )
                    if hit_distance >= 0.0 and hit_distance < closest_distance:
                        closest_distance = hit_distance
                        has_hit = True
            elif stack_idx < qd.static(STACK_SIZE - 2):
                node_stack[stack_idx] = node.left
                node_stack[stack_idx + 1] = node.right
                stack_idx += 2
    return closest_distance, has_hit


@qd.func
def sdf_func_collision_clearance(geom_idx, rigid_info: array_class.RigidInfo):
    """Return a scale-aware clearance that exceeds accumulated transform and projection roundoff."""
    geom_extent = rigid_info.geoms_init_AABB[geom_idx, 7] - rigid_info.geoms_init_AABB[geom_idx, 0]
    return 512.0 * rigid_info.EPS[None] * qd.max(1.0, geom_extent.norm())


@qd.func
def sdf_func_project_vertex_outside_geom(
    geom_idx,
    env_idx,
    pos_world: qd.types.vector(3),
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    surface_info: array_class.RigidSurfaceInfo,
):
    """Project a point onto the feasible side of one rigid geom and return its active contact normal."""
    # The local bounds remain valid for coupling geoms whose rigid broadphase and runtime AABBs are disabled.
    geom_lower = rigid_info.geoms_init_AABB[geom_idx, 0]
    geom_upper = rigid_info.geoms_init_AABB[geom_idx, 7]
    clearance = sdf_func_collision_clearance(geom_idx, rigid_info)
    contact_tolerance = 2.0 * clearance
    corrected_position = pos_world
    normal = qd.Vector.zero(gs.qd_float, 3)
    is_active = False

    if dyn_info.geoms.type[geom_idx] == gs.GEOM_TYPE.MESH:
        geom_pos = dyn_state.geoms.pos[geom_idx, env_idx]
        geom_quat = dyn_state.geoms.quat[geom_idx, env_idx]
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        is_in_query_aabb = (pos_mesh >= geom_lower - clearance).all() and (pos_mesh <= geom_upper + clearance).all()
        if is_in_query_aabb:
            closest_position, normal, is_inside, surface_distance = sdf_func_exact_mesh_surface_bvh(
                geom_idx,
                env_idx,
                pos_world,
                bvh_nodes,
                bvh_morton_codes,
                dyn_state,
                dyn_info,
                rigid_info,
                surface_info,
            )
            is_active = is_inside or surface_distance <= clearance + contact_tolerance
            if is_inside or surface_distance < clearance:
                corrected_position = closest_position + clearance * normal
    else:
        signed_distance = sdf_func_world(
            geom_idx, env_idx, pos_world, dyn_state.geoms, dyn_info.geoms, collider_info.sdf
        )
        if signed_distance <= clearance + contact_tolerance:
            normal = gu.qd_normalize(
                sdf_func_grad_world_local_consistent(
                    geom_idx,
                    pos_world,
                    dyn_state.geoms.pos[geom_idx, env_idx],
                    dyn_state.geoms.quat[geom_idx, env_idx],
                    dyn_info,
                    rigid_info,
                    collider_info,
                ),
                rigid_info.EPS[None],
            )
            is_active = True
            if signed_distance < clearance:
                corrected_position += (clearance - signed_distance) * normal

    return corrected_position, normal, is_active


@qd.func
def sdf_func_normal_world(
    geom_idx,
    batch_idx,
    pos_world,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    g_pos = geoms_state.pos[geom_idx, batch_idx]
    g_quat = geoms_state.quat[geom_idx, batch_idx]

    return sdf_func_normal_world_local(
        geom_idx, pos_world, g_pos, g_quat, geoms_info, rigid_info, sdf_info, collider_static_config
    )


@qd.func
def sdf_func_normal_world_local(
    geom_idx,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    """
    Computes normalized SDF gradient (surface normal) in world coordinates,
    using provided geometry pose instead of reading from geoms_state.
    """
    return gu.qd_normalize(
        sdf_func_grad_world_local(
            geom_idx, pos_world, geom_pos, geom_quat, geoms_info, rigid_info, sdf_info, collider_static_config
        ),
        rigid_info.EPS[None],
    )


@qd.func
def sdf_func_grad_world_local(
    geom_idx,
    pos_world: qd.types.vector(3),
    geom_pos: qd.types.vector(3),
    geom_quat: qd.types.vector(4),
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    """
    Computes SDF gradient in world coordinates, using provided geometry pose
    instead of reading from geoms_state.
    """
    EPS = rigid_info.EPS[None]

    grad_world = qd.Vector.zero(gs.qd_float, 3)

    if geoms_info.type[geom_idx] == gs.GEOM_TYPE.SPHERE:
        grad_world = gu.qd_normalize(pos_world - geom_pos, EPS)

    elif geoms_info.type[geom_idx] == gs.GEOM_TYPE.PLANE:
        geom_data = geoms_info.data[geom_idx]
        plane_normal = gs.qd_vec3([geom_data[0], geom_data[1], geom_data[2]])
        grad_world = gu.qd_transform_by_quat(plane_normal, geom_quat)

    else:
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, geom_pos, geom_quat)
        pos_sdf = gu.qd_transform_by_T(pos_mesh, sdf_info.geoms_info.T_mesh_to_sdf[geom_idx])
        grad_sdf = sdf_func_grad(geom_idx, pos_sdf, geoms_info, rigid_info, sdf_info, collider_static_config)

        grad_mesh = grad_sdf  # no rotation between mesh and sdf frame
        grad_world = gu.qd_transform_by_quat(grad_mesh, geom_quat)

    return grad_world


@qd.func
def sdf_func_find_closest_vert(
    geom_idx,
    i_b,
    pos_world,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    collider_info: array_class.ColliderInfo,
):
    """
    Returns vert of geom that's closest to pos_world
    """
    g_pos = dyn_state.geoms.pos[geom_idx, i_b]
    g_quat = dyn_state.geoms.quat[geom_idx, i_b]
    geom_sdf_res = collider_info.sdf.geoms_info.sdf_res[geom_idx]
    pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, g_pos, g_quat)
    pos_sdf = gu.qd_transform_by_T(pos_mesh, collider_info.sdf.geoms_info.T_mesh_to_sdf[geom_idx])
    nearest_cell = qd.cast(qd.min(qd.max(pos_sdf, 0), geom_sdf_res - 1), gs.qd_int)
    return (
        collider_info.sdf.geoms_sdf_closest_vert[
            sdf_func_ravel_cell_idx(nearest_cell, geom_idx, geom_sdf_res, collider_info.sdf)
        ]
        + dyn_info.geoms.vert_start[geom_idx]
    )


@qd.func
def sdf_func_triangle_surface_corrections(
    i_b,
    i_g_,
    clearance,
    vertices_world,
    previous_vertices_world,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    dyn_state: array_class.DynState,
    surface_state: array_class.RigidSurfaceContactState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    surface_info: array_class.RigidSurfaceInfo,
    is_audit: qd.template(),
):
    """Accumulate triangle corrections for rigid face and edge intersections, preserving the previous exterior side."""
    n_surface_faces = bvh_morton_codes.shape[1]
    previous_centroid_world = (
        previous_vertices_world[:, 0] + previous_vertices_world[:, 1] + previous_vertices_world[:, 2]
    ) / 3.0
    corrections = qd.Matrix.zero(gs.qd_float, 3, 3)
    n_corrections = qd.Vector.zero(gs.qd_int, 3)
    has_intersection = False
    i_g = surface_info.surface_geoms_idx[i_g_]
    geom_pos = dyn_state.geoms.pos[i_g, i_b]
    geom_quat = dyn_state.geoms.quat[i_g, i_b]
    atlas_offset = surface_info.atlas_offsets[i_g_]
    vertices_atlas = qd.Matrix.zero(gs.qd_float, 3, 3)
    previous_vertices_atlas = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i_v_ in qd.static(range(3)):
        vertices_atlas[:, i_v_] = (
            gu.qd_inv_transform_by_trans_quat(vertices_world[:, i_v_], geom_pos, geom_quat) + atlas_offset
        )
        previous_vertices_atlas[:, i_v_] = (
            gu.qd_inv_transform_by_trans_quat(
                previous_vertices_world[:, i_v_],
                surface_state.previous_geoms_pos[i_b, i_g_],
                surface_state.previous_geoms_quat[i_b, i_g_],
            )
            + atlas_offset
        )

    geom_lower = rigid_info.geoms_init_AABB[i_g, 0]
    geom_upper = rigid_info.geoms_init_AABB[i_g, 7]
    geom_extent = geom_upper - geom_lower
    query_lower = qd.min(vertices_atlas[:, 0], vertices_atlas[:, 1], vertices_atlas[:, 2]) - clearance
    query_upper = qd.max(vertices_atlas[:, 0], vertices_atlas[:, 1], vertices_atlas[:, 2]) + clearance
    previous_direction_mesh = qd.Vector.zero(gs.qd_float, 3)
    has_previous_direction = False

    node_stack = qd.Vector.zero(gs.qd_int, qd.static(STACK_SIZE))
    node_stack[0] = 0
    i_stack = 1
    while i_stack > 0:
        i_stack -= 1
        i_node = node_stack[i_stack]
        node = bvh_nodes[0, i_node]
        is_node_overlapping = (query_lower <= node.bound.max).all() and (query_upper >= node.bound.min).all()
        if is_node_overlapping:
            if node.left == -1:
                i_leaf = i_node - (n_surface_faces - 1)
                i_f = qd.cast(bvh_morton_codes[0, i_leaf][1], gs.qd_int)
                if dyn_info.faces.geom_idx[i_f] != i_g:
                    continue
                face = dyn_info.faces.verts_idx[i_f]
                rigid_vertices_atlas = qd.Matrix.cols(
                    [
                        dyn_info.verts.init_pos[face[0]] + atlas_offset,
                        dyn_info.verts.init_pos[face[1]] + atlas_offset,
                        dyn_info.verts.init_pos[face[2]] + atlas_offset,
                    ]
                )
                is_intersecting, hit_position_atlas = triangle_triangle_intersection(
                    vertices_atlas, rigid_vertices_atlas, rigid_info.EPS[None]
                )
                if not is_intersecting:
                    continue

                has_intersection = True
                if qd.static(is_audit):
                    continue

                # A prior separating axis preserves the contact topology with a minimum normal
                # displacement in the current geom frame.
                has_history_correction, history_correction_mesh = triangle_triangle_previous_separating_correction(
                    vertices_atlas,
                    previous_vertices_atlas,
                    rigid_vertices_atlas,
                    clearance,
                    rigid_info.EPS[None],
                )
                if has_history_correction:
                    history_corrections = triangle_separating_corrections(vertices_atlas, history_correction_mesh)
                    for i_v_ in qd.static(range(3)):
                        if history_corrections[:, i_v_].norm_sqr() > 0.0:
                            corrections[:, i_v_] += gu.qd_transform_by_quat(history_corrections[:, i_v_], geom_quat)
                            n_corrections[i_v_] += 1
                    continue

                if not has_previous_direction:
                    # The previous valid configuration selects the exterior side when the current
                    # triangle spans both sides of a collider.
                    previous_centroid_mesh = gu.qd_inv_transform_by_trans_quat(
                        previous_centroid_world,
                        surface_state.previous_geoms_pos[i_b, i_g_],
                        surface_state.previous_geoms_quat[i_b, i_g_],
                    )
                    _, previous_direction_mesh, _, _ = sdf_func_exact_mesh_surface_bvh_local(
                        i_g,
                        previous_centroid_mesh,
                        bvh_nodes,
                        bvh_morton_codes,
                        dyn_info,
                        rigid_info,
                        surface_info,
                    )
                    if previous_direction_mesh.norm_sqr() > rigid_info.EPS[None] ** 2:
                        previous_direction_mesh = previous_direction_mesh.normalized()
                        has_previous_direction = True

                rigid_normal_mesh = (rigid_vertices_atlas[:, 1] - rigid_vertices_atlas[:, 0]).cross(
                    rigid_vertices_atlas[:, 2] - rigid_vertices_atlas[:, 0]
                )
                if not has_previous_direction and rigid_normal_mesh.norm_sqr() > rigid_info.EPS[None] ** 2:
                    previous_direction_mesh = rigid_normal_mesh.normalized()
                    previous_centroid_mesh = gu.qd_inv_transform_by_trans_quat(
                        previous_centroid_world,
                        surface_state.previous_geoms_pos[i_b, i_g_],
                        surface_state.previous_geoms_quat[i_b, i_g_],
                    )
                    if (previous_centroid_mesh - (rigid_vertices_atlas[:, 0] - atlas_offset)).dot(
                        previous_direction_mesh
                    ) < 0.0:
                        previous_direction_mesh = -previous_direction_mesh
                    has_previous_direction = True

                if not has_previous_direction:
                    continue

                hit_position_mesh = hit_position_atlas - atlas_offset
                exit_position_mesh = hit_position_mesh
                rigid_normal_mesh = rigid_normal_mesh.normalized()
                if rigid_normal_mesh.dot(previous_direction_mesh) <= rigid_info.EPS[None]:
                    ray_start_mesh = hit_position_mesh + clearance * previous_direction_mesh
                    exit_distance, has_exit = sdf_func_surface_bvh_ray_cast_local(
                        i_g,
                        ray_start_mesh,
                        previous_direction_mesh,
                        2.0 * qd.max(1.0e-3, geom_extent.norm()),
                        bvh_nodes,
                        bvh_morton_codes,
                        dyn_info,
                        rigid_info,
                        surface_info,
                    )
                    if has_exit:
                        exit_position_mesh = ray_start_mesh + exit_distance * previous_direction_mesh
                    else:
                        for i_axis in qd.static(range(3)):
                            exit_position_mesh[i_axis] = qd.select(
                                previous_direction_mesh[i_axis] >= 0.0, geom_upper[i_axis], geom_lower[i_axis]
                            )

                # A shared exit plane moves the whole intersecting feature coherently and eliminates
                # residual edge crossings.
                target_projection = exit_position_mesh.dot(previous_direction_mesh) + clearance
                for i_v_ in qd.static(range(3)):
                    vertex_projection = (vertices_atlas[:, i_v_] - atlas_offset).dot(previous_direction_mesh)
                    penetration = target_projection - vertex_projection
                    if penetration > 0.0:
                        correction_mesh = penetration * previous_direction_mesh
                        correction_world = gu.qd_transform_by_quat(correction_mesh, geom_quat)
                        corrections[:, i_v_] += correction_world
                        n_corrections[i_v_] += 1
            elif i_stack < qd.static(STACK_SIZE - 2):
                node_stack[i_stack] = node.left
                node_stack[i_stack + 1] = node.right
                i_stack += 2
    return corrections, n_corrections, has_intersection
