from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import pickle as pkl

import numpy as np

import igl
import trimesh

import quadrants as qd

import genesis as gs
from genesis.engine.bvh import AABB, LBVH, STACK_SIZE
import genesis.utils.geom as gu
import genesis.utils.mesh as mesh_utils
from genesis.utils.misc import get_assets_dir, get_gsd_cache_dir
from genesis.utils.triangle_qd import (
    closest_point_on_triangle,
    ray_aabb_intersection,
    ray_projection,
    ray_triangle_intersection,
    triangle_face_normal,
)

_COLLIDER_CONE = 0
_COLLIDER_MESH = 1
_COLLIDER_BOX = 2
_SDF_CACHE_SCHEMA = "pbstf-static-collider-v1"


@dataclass(frozen=True)
class MeshSDFData:
    values: np.ndarray
    lower: np.ndarray
    cell_size: np.ndarray


class StaticCollider(ABC):
    """One-way PBSTF collider geometry expressed in a mutable local frame."""

    kind: int

    def __init__(self, pos, quat):
        self.pos = np.array(pos)
        self.quat = np.array(quat)
        self.is_deformable = False

    @classmethod
    @abstractmethod
    def from_options(cls, options):
        """Construct a collider from its solver-facing options object."""


class BoxStaticCollider(StaticCollider):
    """Finite analytic box whose geometry is fixed in the collider's local frame."""

    kind = _COLLIDER_BOX
    type = "box"

    def __init__(self, lower, upper, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
        super().__init__(pos, quat)
        self.lower = np.array(lower)
        self.upper = np.array(upper)

        if self.lower.shape != (3,):
            gs.raise_exception("Box static collider `lower` must have shape (3,).")
        if self.upper.shape != (3,):
            gs.raise_exception("Box static collider `upper` must have shape (3,).")
        if not np.all(self.upper > self.lower):
            gs.raise_exception("Box static collider `upper` must be greater than `lower` along every axis.")

        self.lower_qd = qd.Vector(self.lower, dt=gs.qd_float)
        self.upper_qd = qd.Vector(self.upper, dt=gs.qd_float)

    @classmethod
    def from_options(cls, options):
        return cls(
            lower=options.lower,
            upper=options.upper,
            pos=options.pos,
            quat=options.quat,
        )


class AbsorbentStaticCollider:
    """Absorption parameters and compact voxel metadata for a position-based surface tension flow (PBSTF) collider."""

    def __init__(self, absorption_rate, absorption_capacity_fraction):
        self.absorption_rate = absorption_rate
        self.absorption_capacity_fraction = absorption_capacity_fraction
        self.grid_res = None
        self.grid_res_qd = None
        self.voxel_size = None
        self.voxel_size_qd = None
        self.voxel_capacity = None
        self.voxel_start = 0
        self.n_voxels = 0
        self.voxel_search_offset_start = 0
        self.n_voxel_search_offsets = 0
        self.total_capacity = 0

        if self.absorption_rate <= 0.0:
            gs.raise_exception("Absorbent static collider `absorption_rate` must be positive.")
        if not 0.0 < self.absorption_capacity_fraction <= 1.0:
            gs.raise_exception("Absorbent static collider `absorption_capacity_fraction` must be in (0, 1].")


class AbsorbentBoxStaticCollider(BoxStaticCollider, AbsorbentStaticCollider):
    """Finite box with rate-limited nearby-voxel capture for position-based surface tension flow (PBSTF)."""

    type = "absorbent_box"

    def __init__(
        self,
        lower,
        upper,
        absorption_rate,
        absorption_capacity_fraction,
        fem_entity_name=None,
        sdf_res=None,
        pos=(0.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
    ):
        BoxStaticCollider.__init__(self, lower=lower, upper=upper, pos=pos, quat=quat)
        AbsorbentStaticCollider.__init__(
            self,
            absorption_rate=absorption_rate,
            absorption_capacity_fraction=absorption_capacity_fraction,
        )
        self.fem_entity_name = fem_entity_name
        self.sdf_res = sdf_res
        self.is_deformable = fem_entity_name is not None
        self.has_sdf = sdf_res is not None
        self.fem_entity = None
        self.embedding_elements_idx = None
        self.embedding_barycentric = None
        self.n_surface_vertices = 0
        self.n_surface_triangles = 0
        self.surface_faces_array = None
        self.surface_faces_tensor = None
        self.surface_faces = None
        self.surface_vertices = None
        self.surface_bvh = None
        self.sdf = None
        self.sdf_lower = None
        self.sdf_inv_cell_size = None
        self.is_sdf_active = None
        self.sdf_state_idx = -1
        self.voxel_graph_distance = None
        self.voxel_positions = None
        self.voxel_search_order = None
        self.surface_vertex_state_start = 0
        self.voxel_state_start = 0
        self.voxel_search_order_state_start = 0

    @classmethod
    def from_options(cls, options):
        return cls(
            lower=options.lower,
            upper=options.upper,
            absorption_rate=options.absorption_rate,
            absorption_capacity_fraction=options.absorption_capacity_fraction,
            fem_entity_name=options.fem_entity_name,
            sdf_res=options.sdf_res,
            pos=options.pos,
            quat=options.quat,
        )


class ConeStaticCollider(StaticCollider):
    """Finite analytic cone whose geometry is fixed in the collider's local frame."""

    kind = _COLLIDER_CONE
    type = "cone"

    def __init__(self, center, height, radius, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
        super().__init__(pos, quat)
        self.center = np.array(center)
        self.height = np.array(height)
        self.radius = radius

        if self.center.shape != (3,):
            gs.raise_exception("Cone static collider `center` must have shape (3,).")
        if self.height.shape != (3,):
            gs.raise_exception("Cone static collider `height` must have shape (3,).")
        if np.linalg.norm(self.height) <= gs.EPS:
            gs.raise_exception("Cone static collider `height` must be non-zero.")
        if self.radius <= 0.0:
            gs.raise_exception("Cone static collider `radius` must be positive.")

        self.center_qd = qd.Vector(self.center, dt=gs.qd_float)
        self.height_qd = qd.Vector(self.height, dt=gs.qd_float)

    @classmethod
    def from_options(cls, options):
        return cls(
            center=options.center,
            height=options.height,
            radius=options.radius,
            pos=options.pos,
            quat=options.quat,
        )


class MeshStaticCollider(StaticCollider):
    """Watertight triangle mesh represented by a cached signed distance field (SDF)."""

    kind = _COLLIDER_MESH
    type = "mesh"

    def __init__(self, file, scale, sdf_res, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
        super().__init__(pos, quat)
        self.file = file
        self.scale = scale
        self.sdf_res = sdf_res

        file_path = os.path.abspath(file)
        if not os.path.exists(file_path):
            file_path = os.path.join(get_assets_dir(), file)
        if not os.path.exists(file_path):
            gs.raise_exception(f"PBSTF static collider mesh file not found: '{file}'.")

        source_mesh = mesh_utils.load_mesh(file_path)
        mesh = trimesh.Trimesh(
            vertices=source_mesh.vertices * scale,
            faces=source_mesh.faces,
            process=False,
        )
        # Signed-distance-field topology welds coincident shading seams into a closed surface.
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        if not mesh.is_watertight:
            gs.raise_exception("PBSTF mesh static colliders require a watertight triangle mesh.")

        sdf_data = load_or_build_mesh_sdf(mesh.vertices, mesh.faces, sdf_res)
        self.sdf = qd.field(gs.qd_float, shape=sdf_data.values.shape)
        self.sdf.from_numpy(sdf_data.values)
        self.sdf_lower_qd = qd.Vector(sdf_data.lower, dt=gs.qd_float)
        self.sdf_inv_cell_size_qd = qd.Vector(1.0 / sdf_data.cell_size, dt=gs.qd_float)

    @classmethod
    def from_options(cls, options):
        return cls(
            file=options.file,
            scale=options.scale,
            sdf_res=options.sdf_res,
            pos=options.pos,
            quat=options.quat,
        )


def load_or_build_mesh_sdf(vertices, faces, sdf_res):
    """Load or build a cached signed distance field for one watertight triangle surface."""
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    grid_size = upper - lower + (upper - lower).max() * 0.2
    cell_size = grid_size / (sdf_res - 1)
    voxel_lower = 0.5 * (lower + upper - grid_size)
    cache_key = mesh_utils.get_hashkey(vertices, faces, sdf_res, cell_size, _SDF_CACHE_SCHEMA)
    cache_path = os.path.join(get_gsd_cache_dir(), f"{cache_key}.pbstf.gsd")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as file:
                sdf_data = pkl.load(file)
            if isinstance(sdf_data, MeshSDFData):
                return sdf_data
            gs.logger.info("Ignoring PBSTF static collider cache with an incompatible schema.")
        except (EOFError, OSError, AttributeError, pkl.UnpicklingError, TypeError):
            gs.logger.info("Ignoring corrupted PBSTF static collider cache.")

    values = np.empty((sdf_res, sdf_res, sdf_res), dtype=gs.np_float)
    x = np.linspace(voxel_lower[0], voxel_lower[0] + grid_size[0], sdf_res)
    y = np.linspace(voxel_lower[1], voxel_lower[1] + grid_size[1], sdf_res)
    z = np.linspace(voxel_lower[2], voxel_lower[2] + grid_size[2], sdf_res)
    slab_size = 4
    with gs.logger.timer(f"Preprocessing PBSTF static collider mesh at {sdf_res}^3 resolution."):
        for z_start in range(0, sdf_res, slab_size):
            z_end = min(z_start + slab_size, sdf_res)
            grid_x, grid_y, grid_z = np.meshgrid(x, y, z[z_start:z_end], indexing="ij")
            query_points = np.stack((grid_x, grid_y, grid_z), axis=-1).reshape((-1, 3))
            signed_distance, *_ = igl.signed_distance(query_points, vertices, faces)
            values[:, :, z_start:z_end] = signed_distance.reshape((sdf_res, sdf_res, z_end - z_start))

    sdf_data = MeshSDFData(values=values, lower=voxel_lower, cell_size=cell_size)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as file:
        pkl.dump(sdf_data, file)
    return sdf_data


@qd.kernel
def kernel_update_deformable_surface_aabbs(
    surface_vertices: qd.template(),
    surface_faces: qd.template(),
    surface_aabbs: qd.template(),
):
    for env_idx, triangle_idx in qd.ndrange(surface_vertices.shape[1], surface_faces.shape[0]):
        face = surface_faces[triangle_idx]
        v0 = surface_vertices[face[0], env_idx]
        v1 = surface_vertices[face[1], env_idx]
        v2 = surface_vertices[face[2], env_idx]
        surface_aabbs[env_idx, triangle_idx].min = qd.min(v0, v1, v2)
        surface_aabbs[env_idx, triangle_idx].max = qd.max(v0, v1, v2)


def build_deformable_surface_bvh(collider, n_batches):
    """Build one linear bounding volume hierarchy (BVH) per batch over a deformable collider surface."""
    surface_aabbs = AABB(n_batches=n_batches, n_aabbs=collider.n_surface_triangles)
    collider.surface_bvh = LBVH(surface_aabbs, max_n_query_result_per_aabb=0)
    kernel_update_deformable_surface_aabbs(
        collider.surface_vertices,
        collider.surface_faces,
        collider.surface_bvh.aabbs,
    )
    collider.surface_bvh.build()


def refit_deformable_surface_bvh(collider):
    """Refit a deformable surface BVH after its vertices change while preserving its leaf topology."""
    kernel_update_deformable_surface_aabbs(
        collider.surface_vertices,
        collider.surface_faces,
        collider.surface_bvh.aabbs,
    )
    collider.surface_bvh.compute_bounds()


@qd.func
def _cone_radial_direction(pos, collider: qd.template()):
    axis = collider.height_qd.normalized()
    axial_distance = (pos - collider.center_qd).dot(axis)
    radial = pos - collider.center_qd - axial_distance * axis
    if radial.norm_sqr() <= gs.EPS**2:
        fallback = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
        if axis[0] >= 0.1:
            fallback = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
        radial = axis.cross(fallback).normalized()
    else:
        radial = radial.normalized()
    return radial


@qd.func
def _query_sdf_local(
    env_idx,
    pos,
    sdf_lower,
    sdf_inv_cell_size,
    sdf: qd.template(),
    sdf_res,
    is_sdf_batched: qd.template(),
):
    grid_pos = (pos - sdf_lower) * sdf_inv_cell_size
    is_in_grid = True
    for axis in qd.static(range(3)):
        if grid_pos[axis] < 0.0 or grid_pos[axis] > sdf_res - 1:
            is_in_grid = False

    closest_position = pos
    closest_normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    is_inside = False
    surface_distance = gs.qd_float(1.0e20)
    if is_in_grid:
        cell = qd.Vector.zero(gs.qd_int, 3)
        fraction = qd.Vector.zero(gs.qd_float, 3)
        for axis in qd.static(range(3)):
            cell[axis] = qd.min(qd.cast(qd.floor(grid_pos[axis]), gs.qd_int), sdf_res - 2)
            fraction[axis] = grid_pos[axis] - cell[axis]

        x, y, z = cell[0], cell[1], cell[2]
        fx, fy, fz = fraction[0], fraction[1], fraction[2]
        v000 = gs.qd_float(0.0)
        v001 = gs.qd_float(0.0)
        v010 = gs.qd_float(0.0)
        v011 = gs.qd_float(0.0)
        v100 = gs.qd_float(0.0)
        v101 = gs.qd_float(0.0)
        v110 = gs.qd_float(0.0)
        v111 = gs.qd_float(0.0)
        if qd.static(is_sdf_batched):
            v000 = sdf[x, y, z, env_idx]
            v001 = sdf[x, y, z + 1, env_idx]
            v010 = sdf[x, y + 1, z, env_idx]
            v011 = sdf[x, y + 1, z + 1, env_idx]
            v100 = sdf[x + 1, y, z, env_idx]
            v101 = sdf[x + 1, y, z + 1, env_idx]
            v110 = sdf[x + 1, y + 1, z, env_idx]
            v111 = sdf[x + 1, y + 1, z + 1, env_idx]
        else:
            v000 = sdf[x, y, z]
            v001 = sdf[x, y, z + 1]
            v010 = sdf[x, y + 1, z]
            v011 = sdf[x, y + 1, z + 1]
            v100 = sdf[x + 1, y, z]
            v101 = sdf[x + 1, y, z + 1]
            v110 = sdf[x + 1, y + 1, z]
            v111 = sdf[x + 1, y + 1, z + 1]

        v00 = v000 * (1.0 - fx) + v100 * fx
        v01 = v001 * (1.0 - fx) + v101 * fx
        v10 = v010 * (1.0 - fx) + v110 * fx
        v11 = v011 * (1.0 - fx) + v111 * fx
        v0 = v00 * (1.0 - fy) + v10 * fy
        v1 = v01 * (1.0 - fy) + v11 * fy
        signed_distance = v0 * (1.0 - fz) + v1 * fz

        gradient = qd.Vector(
            [
                (
                    (1.0 - fy) * (1.0 - fz) * (v100 - v000)
                    + (1.0 - fy) * fz * (v101 - v001)
                    + fy * (1.0 - fz) * (v110 - v010)
                    + fy * fz * (v111 - v011)
                )
                * sdf_inv_cell_size[0],
                (
                    (1.0 - fx) * (1.0 - fz) * (v010 - v000)
                    + (1.0 - fx) * fz * (v011 - v001)
                    + fx * (1.0 - fz) * (v110 - v100)
                    + fx * fz * (v111 - v101)
                )
                * sdf_inv_cell_size[1],
                (
                    (1.0 - fx) * (1.0 - fy) * (v001 - v000)
                    + (1.0 - fx) * fy * (v011 - v010)
                    + fx * (1.0 - fy) * (v101 - v100)
                    + fx * fy * (v111 - v110)
                )
                * sdf_inv_cell_size[2],
            ],
            dt=gs.qd_float,
        )
        if gradient.norm_sqr() > gs.EPS**2:
            closest_normal = gradient.normalized()
        closest_position = pos - signed_distance * closest_normal
        is_inside = signed_distance < 0.0
        surface_distance = qd.abs(signed_distance)

    return closest_position, closest_normal, is_inside, surface_distance


@qd.func
def _point_aabb_distance_sqr(pos, lower, upper):
    delta = qd.Vector.zero(gs.qd_float, 3)
    for axis in qd.static(range(3)):
        if pos[axis] < lower[axis]:
            delta[axis] = lower[axis] - pos[axis]
        elif pos[axis] > upper[axis]:
            delta[axis] = pos[axis] - upper[axis]
    return delta.norm_sqr()


@qd.func
def _query_deformable_surface_bvh_local(env_idx, max_distance, pos, collider: qd.template()):
    closest_position = pos
    closest_normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    closest_normal_sum = qd.Vector.zero(gs.qd_float, 3)
    closest_triangle_idx = collider.n_surface_triangles
    surface_distance_sqr = max_distance * max_distance
    has_closest = False

    node_stack = qd.Vector.zero(gs.qd_int, qd.static(STACK_SIZE))
    node_stack[0] = 0
    stack_idx = 1
    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]
        node = collider.surface_bvh.nodes[env_idx, node_idx]
        distance_tolerance = 8.0 * gs.EPS * qd.max(1.0, surface_distance_sqr)
        node_distance_sqr = _point_aabb_distance_sqr(pos, node.bound.min, node.bound.max)
        if node_distance_sqr <= surface_distance_sqr + distance_tolerance:
            if node.left == -1:
                sorted_leaf_idx = node_idx - (collider.n_surface_triangles - 1)
                triangle_idx = qd.cast(collider.surface_bvh.morton_codes[env_idx, sorted_leaf_idx][1], gs.qd_int)
                face = collider.surface_faces[triangle_idx]
                v0 = collider.surface_vertices[face[0], env_idx]
                v1 = collider.surface_vertices[face[1], env_idx]
                v2 = collider.surface_vertices[face[2], env_idx]
                candidate_normal = triangle_face_normal(v0, v1, v2)
                candidate = closest_point_on_triangle(pos, v0, v1, v2)
                candidate_distance_sqr = (pos - candidate).norm_sqr()
                candidate_tolerance = 8.0 * gs.EPS * qd.max(1.0, candidate_distance_sqr, surface_distance_sqr)
                if candidate_distance_sqr <= surface_distance_sqr + candidate_tolerance:
                    if not has_closest or candidate_distance_sqr < surface_distance_sqr - candidate_tolerance:
                        closest_normal_sum = candidate_normal
                    elif qd.abs(candidate_distance_sqr - surface_distance_sqr) <= candidate_tolerance:
                        # Equidistant faces share normals so edge and vertex queries have a stable direction.
                        closest_normal_sum += candidate_normal
                    if (
                        not has_closest
                        or candidate_distance_sqr < surface_distance_sqr
                        or (candidate_distance_sqr == surface_distance_sqr and triangle_idx < closest_triangle_idx)
                    ):
                        closest_position = candidate
                        closest_normal = candidate_normal
                        closest_triangle_idx = triangle_idx
                        surface_distance_sqr = candidate_distance_sqr
                    has_closest = True
            elif stack_idx < qd.static(STACK_SIZE - 2):
                left = node.left
                right = node.right
                left_node = collider.surface_bvh.nodes[env_idx, left]
                right_node = collider.surface_bvh.nodes[env_idx, right]
                left_distance_sqr = _point_aabb_distance_sqr(pos, left_node.bound.min, left_node.bound.max)
                right_distance_sqr = _point_aabb_distance_sqr(pos, right_node.bound.min, right_node.bound.max)
                if left_distance_sqr < right_distance_sqr:
                    node_stack[stack_idx] = right
                    node_stack[stack_idx + 1] = left
                else:
                    node_stack[stack_idx] = left
                    node_stack[stack_idx + 1] = right
                stack_idx += 2

    surface_distance = 2.0 * max_distance
    inside_normal = closest_normal
    if has_closest:
        surface_distance = qd.sqrt(surface_distance_sqr)
        if closest_normal_sum.norm_sqr() > gs.EPS**2:
            inside_normal = closest_normal_sum.normalized()
    return closest_position, closest_normal, inside_normal, surface_distance


@qd.func
def _is_inside_deformable_surface_bvh_local(env_idx, pos, collider: qd.template()):
    # An oblique unit ray avoids systematic alignment with axis-aligned surface edges.
    ray_dir = qd.Vector([0.8192319205, 0.4630140578, 0.3395271683], dt=gs.qd_float)
    axes, shear, is_valid_dir = ray_projection(ray_dir, gs.EPS)
    # Oriented crossings cancel outside a closed surface and leave one net exit for a point inside it.
    winding_crossings = gs.qd_int(0)

    node_stack = qd.Vector.zero(gs.qd_int, qd.static(STACK_SIZE))
    node_stack[0] = 0
    stack_idx = 1
    if not is_valid_dir:
        stack_idx = 0
    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]
        node = collider.surface_bvh.nodes[env_idx, node_idx]
        if ray_aabb_intersection(pos, ray_dir, node.bound.min, node.bound.max, gs.EPS) >= 0.0:
            if node.left == -1:
                sorted_leaf_idx = node_idx - (collider.n_surface_triangles - 1)
                triangle_idx = qd.cast(collider.surface_bvh.morton_codes[env_idx, sorted_leaf_idx][1], gs.qd_int)
                face = collider.surface_faces[triangle_idx]
                v0 = collider.surface_vertices[face[0], env_idx]
                v1 = collider.surface_vertices[face[1], env_idx]
                v2 = collider.surface_vertices[face[2], env_idx]
                hit_distance = ray_triangle_intersection(axes, pos, shear, v0, v1, v2, gs.EPS)
                if hit_distance >= 0.0:
                    alignment = triangle_face_normal(v0, v1, v2).dot(ray_dir)
                    if alignment > gs.EPS:
                        winding_crossings += 1
                    elif alignment < -gs.EPS:
                        winding_crossings -= 1
            elif stack_idx < qd.static(STACK_SIZE - 2):
                node_stack[stack_idx] = node.left
                node_stack[stack_idx + 1] = node.right
                stack_idx += 2

    return winding_crossings != 0


@qd.func
def _query_deformable_surface_local(env_idx, pos, collider: qd.template()):
    closest_position, closest_normal, inside_normal, surface_distance = _query_deformable_surface_bvh_local(
        env_idx, 1.0e10, pos, collider
    )
    delta = pos - closest_position
    is_inside_candidate = delta.dot(inside_normal) <= 0.0
    is_inside = surface_distance <= gs.EPS
    if is_inside_candidate and not is_inside:
        is_inside = _is_inside_deformable_surface_bvh_local(env_idx, pos, collider)
    if not is_inside and surface_distance > gs.EPS:
        closest_normal = delta / surface_distance
    return closest_position, closest_normal, is_inside, surface_distance


@qd.func
def _query_box_local(env_idx, pos, collider: qd.template()):
    if qd.static(collider.is_deformable):
        closest_position = pos
        closest_normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
        is_inside = False
        surface_distance = gs.qd_float(1.0e20)
        if qd.static(collider.has_sdf):
            if collider.is_sdf_active[env_idx]:
                closest_position, closest_normal, is_inside, surface_distance = _query_sdf_local(
                    env_idx,
                    pos,
                    collider.sdf_lower[env_idx],
                    collider.sdf_inv_cell_size[env_idx],
                    collider.sdf,
                    collider.sdf_res,
                    is_sdf_batched=True,
                )
            else:
                closest_position, closest_normal, is_inside, surface_distance = _query_deformable_surface_local(
                    env_idx, pos, collider
                )
        else:
            closest_position, closest_normal, is_inside, surface_distance = _query_deformable_surface_local(
                env_idx, pos, collider
            )
        return closest_position, closest_normal, is_inside, surface_distance

    closest_position = pos
    closest_normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    is_inside = True
    for axis in qd.static(range(3)):
        if pos[axis] < collider.lower_qd[axis]:
            closest_position[axis] = collider.lower_qd[axis]
            is_inside = False
        elif pos[axis] > collider.upper_qd[axis]:
            closest_position[axis] = collider.upper_qd[axis]
            is_inside = False

    delta = pos - closest_position
    surface_distance = delta.norm()
    if is_inside:
        surface_distance = gs.qd_float(1.0e20)
        for axis in qd.static(range(3)):
            lower_distance = pos[axis] - collider.lower_qd[axis]
            if lower_distance < surface_distance:
                closest_position = pos
                closest_position[axis] = collider.lower_qd[axis]
                closest_normal = qd.Vector.zero(gs.qd_float, 3)
                closest_normal[axis] = -1.0
                surface_distance = lower_distance

            upper_distance = collider.upper_qd[axis] - pos[axis]
            if upper_distance < surface_distance:
                closest_position = pos
                closest_position[axis] = collider.upper_qd[axis]
                closest_normal = qd.Vector.zero(gs.qd_float, 3)
                closest_normal[axis] = 1.0
                surface_distance = upper_distance
    elif surface_distance > gs.EPS:
        closest_normal = delta / surface_distance

    return closest_position, closest_normal, is_inside, surface_distance


@qd.func
def _query_cone_local(pos, collider: qd.template()):
    axis = collider.height_qd.normalized()
    axial_distance = (pos - collider.center_qd).dot(axis)
    radial = _cone_radial_direction(pos, collider)

    base_position = pos - axial_distance * axis
    is_inside_base = axial_distance >= 0.0
    if (base_position - collider.center_qd).norm() > collider.radius:
        base_position = collider.center_qd + collider.radius * radial
        is_inside_base = False

    side_direction = (collider.radius * radial - collider.height_qd).normalized()
    side_parameter = (pos - collider.center_qd - collider.height_qd).dot(side_direction)
    side_position = collider.center_qd + collider.height_qd
    side_normal = axis
    is_inside_side = False
    slant_length_sqr = collider.height_qd.norm_sqr() + collider.radius * collider.radius
    if side_parameter >= 0.0:
        if side_parameter * side_parameter > slant_length_sqr:
            side_position = collider.center_qd + collider.radius * radial
            side_normal = -axis
        else:
            side_position = collider.center_qd + collider.height_qd + side_parameter * side_direction
            side_normal = (radial * collider.height_qd.norm() + axis * collider.radius).normalized()
            is_inside_side = (pos - side_position).dot(side_normal) <= 0.0

    closest_position = side_position
    closest_normal = side_normal
    is_inside = is_inside_side
    if (base_position - pos).norm_sqr() < (side_position - pos).norm_sqr():
        closest_position = base_position
        closest_normal = -axis
        is_inside = is_inside_base

    return closest_position, closest_normal, is_inside, (closest_position - pos).norm()


@qd.func
def _query_mesh_local(env_idx, pos, collider: qd.template()):
    return _query_sdf_local(
        env_idx,
        pos,
        collider.sdf_lower_qd,
        collider.sdf_inv_cell_size_qd,
        collider.sdf,
        collider.sdf_res,
        is_sdf_batched=False,
    )


@qd.func
def query_static_collider(collider_idx, env_idx, pos, colliders_pos, colliders_quat, collider: qd.template()):
    collider_pos = colliders_pos[collider_idx, env_idx]
    collider_quat = colliders_quat[collider_idx, env_idx]
    pos_local = gu.qd_inv_transform_by_trans_quat(pos, collider_pos, collider_quat)
    closest_local = pos_local
    normal_local = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    is_inside = False
    surface_distance = gs.qd_float(1.0e20)
    if qd.static(collider.kind == _COLLIDER_BOX):
        closest_local, normal_local, is_inside, surface_distance = _query_box_local(env_idx, pos_local, collider)
    elif qd.static(collider.kind == _COLLIDER_CONE):
        closest_local, normal_local, is_inside, surface_distance = _query_cone_local(pos_local, collider)
    elif qd.static(collider.kind == _COLLIDER_MESH):
        closest_local, normal_local, is_inside, surface_distance = _query_mesh_local(env_idx, pos_local, collider)
    closest = gu.qd_transform_by_trans_quat(closest_local, collider_pos, collider_quat)
    normal = gu.qd_transform_by_quat(normal_local, collider_quat)
    return closest, normal, is_inside, surface_distance


@qd.func
def query_deformable_static_collider_surface(
    collider_idx,
    env_idx,
    max_distance,
    pos,
    colliders_pos,
    colliders_quat,
    collider: qd.template(),
):
    collider_pos = colliders_pos[collider_idx, env_idx]
    collider_quat = colliders_quat[collider_idx, env_idx]
    pos_local = gu.qd_inv_transform_by_trans_quat(pos, collider_pos, collider_quat)
    closest_local = pos_local
    normal_local = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    inside_normal_local = normal_local
    is_inside = False
    is_bvh_query = True
    surface_distance = 2.0 * max_distance
    if qd.static(collider.has_sdf):
        if collider.is_sdf_active[env_idx]:
            closest_local, normal_local, is_inside, surface_distance = _query_sdf_local(
                env_idx,
                pos_local,
                collider.sdf_lower[env_idx],
                collider.sdf_inv_cell_size[env_idx],
                collider.sdf,
                collider.sdf_res,
                is_sdf_batched=True,
            )
            is_bvh_query = False

    if is_bvh_query:
        closest_local, normal_local, inside_normal_local, surface_distance = _query_deformable_surface_bvh_local(
            env_idx, max_distance, pos_local, collider
        )
        delta = pos_local - closest_local
        is_inside_candidate = surface_distance <= max_distance and delta.dot(inside_normal_local) <= 0.0
        is_inside = surface_distance <= gs.EPS
        if is_inside_candidate and not is_inside:
            is_inside = _is_inside_deformable_surface_bvh_local(env_idx, pos_local, collider)
        if not is_inside and surface_distance > gs.EPS:
            normal_local = delta / surface_distance
    closest = gu.qd_transform_by_trans_quat(closest_local, collider_pos, collider_quat)
    normal = gu.qd_transform_by_quat(normal_local, collider_quat)
    return closest, normal, is_inside, surface_distance


@qd.func
def query_static_collider_contact(
    collider_idx,
    env_idx,
    pos,
    particle_radius,
    colliders_pos,
    colliders_quat,
    collider: qd.template(),
):
    closest, normal, is_inside, surface_distance = query_static_collider(
        collider_idx, env_idx, pos, colliders_pos, colliders_quat, collider
    )
    anchor = closest + particle_radius * normal
    is_penetrating = is_inside or surface_distance < particle_radius
    return anchor, normal, is_penetrating, surface_distance


@qd.func
def project_out_static_collider(
    collider_idx,
    env_idx,
    pos,
    particle_radius,
    colliders_pos,
    colliders_quat,
    collider: qd.template(),
):
    anchor, _, is_penetrating, _ = query_static_collider_contact(
        collider_idx, env_idx, pos, particle_radius, colliders_pos, colliders_quat, collider
    )
    if is_penetrating:
        pos = anchor
    return pos


@qd.func
def static_collider_separates(
    collider_idx, env_idx, pos_i, pos_j, particle_radius, colliders_pos, colliders_quat, collider: qd.template()
):
    normal_i = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    normal_j = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    distance_i = gs.qd_float(1.0e20)
    distance_j = gs.qd_float(1.0e20)
    if qd.static(collider.kind == _COLLIDER_BOX and collider.is_deformable):
        _, normal_i, _, distance_i = query_deformable_static_collider_surface(
            collider_idx, env_idx, particle_radius, pos_i, colliders_pos, colliders_quat, collider
        )
        _, normal_j, _, distance_j = query_deformable_static_collider_surface(
            collider_idx, env_idx, particle_radius, pos_j, colliders_pos, colliders_quat, collider
        )
    else:
        _, normal_i, _, distance_i = query_static_collider(
            collider_idx, env_idx, pos_i, colliders_pos, colliders_quat, collider
        )
        _, normal_j, _, distance_j = query_static_collider(
            collider_idx, env_idx, pos_j, colliders_pos, colliders_quat, collider
        )
    return distance_i <= particle_radius and distance_j <= particle_radius and normal_i.dot(normal_j) < 0.0


_STATIC_COLLIDER_TYPES = {
    BoxStaticCollider.type: BoxStaticCollider,
    AbsorbentBoxStaticCollider.type: AbsorbentBoxStaticCollider,
    ConeStaticCollider.type: ConeStaticCollider,
    MeshStaticCollider.type: MeshStaticCollider,
}


def create_static_collider(options) -> StaticCollider:
    """Create the collider selected by ``options.type``."""
    collider_cls = _STATIC_COLLIDER_TYPES.get(options.type)
    if collider_cls is None:
        gs.raise_exception(f"Unsupported static collider type: {options.type!r}.")
    return collider_cls.from_options(options)


__all__ = [
    "StaticCollider",
    "AbsorbentStaticCollider",
    "AbsorbentBoxStaticCollider",
    "BoxStaticCollider",
    "ConeStaticCollider",
    "MeshStaticCollider",
    "build_deformable_surface_bvh",
    "create_static_collider",
    "load_or_build_mesh_sdf",
    "project_out_static_collider",
    "query_deformable_static_collider_surface",
    "query_static_collider",
    "query_static_collider_contact",
    "refit_deformable_surface_bvh",
    "static_collider_separates",
]
