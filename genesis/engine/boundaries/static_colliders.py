import os
import pickle as pkl
from abc import ABC, abstractmethod
from dataclasses import dataclass

import igl
import numpy as np
import quadrants as qd
import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mesh_utils
from genesis.utils.misc import get_assets_dir, get_gsd_cache_dir

_COLLIDER_CONE = 0
_COLLIDER_MESH = 1
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

    @classmethod
    @abstractmethod
    def from_options(cls, options):
        """Construct a collider from its solver-facing options object."""


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

        sdf_data = _load_or_build_mesh_sdf(mesh, sdf_res)
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


def _load_or_build_mesh_sdf(mesh, sdf_res):
    vertices = mesh.vertices
    faces = mesh.faces
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
def _query_mesh_local(pos, collider: qd.template()):
    grid_pos = (pos - collider.sdf_lower_qd) * collider.sdf_inv_cell_size_qd
    is_in_grid = True
    for axis in qd.static(range(3)):
        if grid_pos[axis] < 0.0 or grid_pos[axis] > collider.sdf_res - 1:
            is_in_grid = False

    closest_position = pos
    closest_normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    is_inside = False
    surface_distance = gs.qd_float(1.0e20)
    if is_in_grid:
        cell = qd.Vector.zero(gs.qd_int, 3)
        fraction = qd.Vector.zero(gs.qd_float, 3)
        for axis in qd.static(range(3)):
            cell[axis] = qd.min(qd.cast(qd.floor(grid_pos[axis]), gs.qd_int), collider.sdf_res - 2)
            fraction[axis] = grid_pos[axis] - cell[axis]

        x, y, z = cell[0], cell[1], cell[2]
        fx, fy, fz = fraction[0], fraction[1], fraction[2]
        v000 = collider.sdf[x, y, z]
        v001 = collider.sdf[x, y, z + 1]
        v010 = collider.sdf[x, y + 1, z]
        v011 = collider.sdf[x, y + 1, z + 1]
        v100 = collider.sdf[x + 1, y, z]
        v101 = collider.sdf[x + 1, y, z + 1]
        v110 = collider.sdf[x + 1, y + 1, z]
        v111 = collider.sdf[x + 1, y + 1, z + 1]

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
                * collider.sdf_inv_cell_size_qd[0],
                (
                    (1.0 - fx) * (1.0 - fz) * (v010 - v000)
                    + (1.0 - fx) * fz * (v011 - v001)
                    + fx * (1.0 - fz) * (v110 - v100)
                    + fx * fz * (v111 - v101)
                )
                * collider.sdf_inv_cell_size_qd[1],
                (
                    (1.0 - fx) * (1.0 - fy) * (v001 - v000)
                    + (1.0 - fx) * fy * (v011 - v010)
                    + fx * (1.0 - fy) * (v101 - v100)
                    + fx * fy * (v111 - v110)
                )
                * collider.sdf_inv_cell_size_qd[2],
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
def query_static_collider(collider_idx, env_idx, pos, colliders_pos, colliders_quat, collider: qd.template()):
    collider_pos = colliders_pos[collider_idx, env_idx]
    collider_quat = colliders_quat[collider_idx, env_idx]
    pos_local = gu.qd_inv_transform_by_trans_quat(pos, collider_pos, collider_quat)
    closest_local = pos_local
    normal_local = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    is_inside = False
    surface_distance = gs.qd_float(1.0e20)
    if qd.static(collider.kind == _COLLIDER_CONE):
        closest_local, normal_local, is_inside, surface_distance = _query_cone_local(pos_local, collider)
    elif qd.static(collider.kind == _COLLIDER_MESH):
        closest_local, normal_local, is_inside, surface_distance = _query_mesh_local(pos_local, collider)
    closest = gu.qd_transform_by_trans_quat(closest_local, collider_pos, collider_quat)
    normal = gu.qd_transform_by_quat(normal_local, collider_quat)
    return closest, normal, is_inside, surface_distance


@qd.func
def project_out_static_collider(collider_idx, env_idx, pos, colliders_pos, colliders_quat, collider: qd.template()):
    closest, _, is_inside, _ = query_static_collider(
        collider_idx, env_idx, pos, colliders_pos, colliders_quat, collider
    )
    if is_inside:
        pos = closest
    return pos


@qd.func
def static_collider_separates(
    collider_idx, env_idx, pos_i, pos_j, particle_radius, colliders_pos, colliders_quat, collider: qd.template()
):
    _, normal_i, _, distance_i = query_static_collider(
        collider_idx, env_idx, pos_i, colliders_pos, colliders_quat, collider
    )
    _, normal_j, _, distance_j = query_static_collider(
        collider_idx, env_idx, pos_j, colliders_pos, colliders_quat, collider
    )
    return distance_i <= particle_radius and distance_j <= particle_radius and normal_i.dot(normal_j) < 0.0


_STATIC_COLLIDER_TYPES = {
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
    "ConeStaticCollider",
    "MeshStaticCollider",
    "create_static_collider",
    "project_out_static_collider",
    "query_static_collider",
    "static_collider_separates",
]
