import math

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import (
    AbsorbentStaticCollider,
    CubeBoundary,
    create_static_collider,
    load_or_build_mesh_sdf,
    project_out_static_collider,
    query_static_collider,
    query_static_collider_contact,
    static_collider_separates,
)
from genesis.engine.entities import FEMEntity, PBSTFEntity
from genesis.engine.states.solvers import PBSTFSolverState
from genesis.utils import particle
from genesis.utils.array_class import ErrorCode
from genesis.utils.misc import (
    assign_indexed_tensor,
    broadcast_tensor,
    indices_to_mask,
    qd_to_numpy,
    qd_to_torch,
    sanitize_index,
    tensor_to_array,
)

from . import pbstf_absorption
from .base_solver import Solver


@qd.data_oriented
class PBSTFSolver(Solver):
    """
    GPU implementation of *Position-Based Surface Tension Flow*.

    This is deliberately independent of :class:`PBDSolver`: it has only fluid
    particles, uses the reference cubic-spline kernel for density, normals and
    every density gradient, and uses collision-distance constraints instead of
    PBF artificial pressure.
    """

    _N_THETA = 18
    _N_PHI = 36
    _ILLUMINATED_THRESHOLD = 1.0 / 9.0
    _COS_PI_OVER_4 = 0.7071067811865476
    _SURFACE_NEIGHBOR_OVERFLOW = 1
    _LOCAL_MESH_QUEUE_OVERFLOW = 2
    _LOCAL_MESH_NEIGHBOR_OVERFLOW = 3

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        self._particle_size = options.particle_size
        self._particle_radius = 0.5 * options.particle_size
        self._support_radius = options._support_radius
        self._max_solver_iterations = options.max_solver_iterations
        self._topology_rebuild_interval = options.topology_rebuild_interval
        self._max_surface_neighbors = options.max_surface_neighbors
        self._max_localmesh_neighbors = options.max_localmesh_neighbors
        self._enable_pca_normals = options.enable_pca_normals
        self._static_colliders = tuple(
            create_static_collider(collider_options) for collider_options in options.static_colliders
        )
        self._n_static_colliders = len(self._static_colliders)
        self._absorbent_static_colliders_idx = tuple(
            collider_idx
            for collider_idx, collider in enumerate(self._static_colliders)
            if isinstance(collider, AbsorbentStaticCollider)
        )
        self._n_absorbent_static_colliders = len(self._absorbent_static_colliders_idx)
        self._static_colliders_pos = None
        self._static_colliders_quat = None
        self._upper_bound = np.asarray(options.upper_bound, dtype=gs.np_float)
        self._lower_bound = np.asarray(options.lower_bound, dtype=gs.np_float)

        self.sh = gu.SpatialHasher(cell_size=options.hash_grid_cell_size, grid_res=options._hash_grid_res)
        self.boundary = CubeBoundary(lower=self._lower_bound, upper=self._upper_bound)

        self._default_mass = 1.0
        self._material = None
        self._n_absorption_voxels = 0
        self._n_deformable_static_colliders = 0
        self._n_deformable_sdf_colliders = 0
        self._n_deformable_surface_vertices = 0
        self._n_deformable_voxels = 0
        self._n_deformable_voxel_search_order = 0
        self._absorption_particles = None
        self._absorption_particles_reordered = None
        self._absorption_voxel_capacity = None
        self._absorption_voxel_occupancy = None
        self._absorption_voxel_wetness = None
        self._absorption_voxel_search_offsets = None
        self._absorption_capture_budget = None
        self._errno = None

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_entity(self, idx, material, morph, surface, name: str | None = None) -> "PBSTFEntity":
        entity = PBSTFEntity(
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

    def _validate_materials(self):
        if not self.entities:
            return
        self._material = self.entities[0].material
        for entity in self.entities[1:]:
            material = entity.material
            if (
                material.rho != self._material.rho
                or material.density_compliance != self._material.density_compliance
                or material.surface_tension_compliance != self._material.surface_tension_compliance
                or material.surface_distance_compliance != self._material.surface_distance_compliance
                or material.interior_distance_compliance != self._material.interior_distance_compliance
                or material.surface_viscosity != self._material.surface_viscosity
                or material.interior_viscosity != self._material.interior_viscosity
                or material.is_collider_adhesion_friction_enabled
                != self._material.is_collider_adhesion_friction_enabled
                or material.collider_adhesion_compliance != self._material.collider_adhesion_compliance
                or material.collider_friction != self._material.collider_friction
            ):
                gs.raise_exception(
                    "All entities in one PBSTFSolver must use identical PBSTF liquid properties. "
                    "The reference algorithm is a single-phase fluid solver."
                )

    def build(self):
        super().build()
        self._B = self._sim._B
        self._n_particles = self.n_particles
        if self._n_static_colliders > 0:
            self._static_colliders_pos = qd.field(gs.qd_vec3, shape=(self._n_static_colliders, self._B))
            self._static_colliders_quat = qd.field(gs.qd_vec4, shape=(self._n_static_colliders, self._B))
            colliders_pos = np.repeat(
                np.stack([collider.pos for collider in self._static_colliders])[:, None, :],
                repeats=self._B,
                axis=1,
            )
            colliders_quat = np.repeat(
                np.stack([collider.quat for collider in self._static_colliders])[:, None, :],
                repeats=self._B,
                axis=1,
            )
            self._static_colliders_pos.from_numpy(colliders_pos)
            self._static_colliders_quat.from_numpy(colliders_quat)

        # Convert before compiling any PBSTF kernel so every compiled instance
        # sees one stable gravity-field type.
        if self._gravity is not None:
            gravity = qd_to_numpy(self._gravity, transpose=True)
            self._gravity = qd.field(dtype=gs.qd_vec3, shape=(self._B,))
            self._gravity.from_numpy(gravity)

        if self.is_active:
            if gs.backend != gs.cuda:
                gs.raise_exception("PBSTFSolver requires the CUDA backend.")
            if self.sim.requires_grad:
                gs.raise_exception("PBSTFSolver does not support differentiable simulation.")

            self._validate_materials()
            self.sh.build(self._B)
            self._init_particle_fields()
            self._errno = qd.field(gs.qd_int, shape=(self._B,))
            self._errno.fill(0)
            if self._n_absorbent_static_colliders > 0:
                self._init_absorption_fields()
            self._init_surface_fields()

            for entity in self.entities:
                entity._add_to_solver()

            has_active_particles = any(entity.active for entity in self.entities)
            if has_active_particles:
                self._kernel_reorder_particles(0)
                self._kernel_compute_density(0)
                self._max_density[None] = 0.0
                self._kernel_reduce_max_density()
                max_density = qd_to_numpy(self._max_density, transpose=True)[()]
            else:
                # Empty emitters use the interior staggered lattice density that defines the PBSTF particle mass.
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
                gs.raise_exception("PBSTF particle mass calibration requires a positive reference density.")
            self._default_mass = float(self._material.rho / max_density)
            self._kernel_set_particle_mass(self._default_mass)
            self._kernel_reorder_particles(0)
            self._kernel_compute_density(0)
            if self._n_absorbent_static_colliders > 0:
                particle_volume = self._default_mass / self._material.rho
                voxel_capacities = []
                for collider_idx in self._absorbent_static_colliders_idx:
                    collider = self._static_colliders[collider_idx]
                    collider.total_capacity = int(
                        np.floor(
                            collider.absorption_capacity_fraction
                            * np.prod(collider.upper - collider.lower)
                            / particle_volume
                        )
                    )
                    capacity = collider.total_capacity // collider.n_voxels
                    remainder = collider.total_capacity % collider.n_voxels
                    collider.voxel_capacity = np.full(collider.n_voxels, capacity, dtype=gs.np_int)
                    collider.voxel_capacity[:remainder] += 1
                    voxel_capacities.append(collider.voxel_capacity)
                self._absorption_voxel_capacity.from_numpy(np.concatenate(voxel_capacities))
                self._rebuild_absorption_fields(self._absorption_particles)

    @gs.assert_built
    def set_static_colliders_pose(self, pos, quat, colliders_idx=None, envs_idx=None):
        """Set positions and orientations of selected one-way static colliders.

        ``pos`` and ``quat`` broadcast to ``(n_envs, n_colliders, 3)`` and ``(n_envs, n_colliders, 4)`` respectively.
        Quaternion values use the w-x-y-z convention.
        """
        if self._n_static_colliders == 0:
            gs.raise_exception("Cannot set PBSTF static collider poses because the scene has no static colliders.")

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        colliders_idx = sanitize_index(colliders_idx, -1, self._n_static_colliders, 1, "colliders_idx")
        pos = broadcast_tensor(
            pos,
            gs.tc_float,
            (len(envs_idx), len(colliders_idx), 3),
            ("envs_idx", "colliders_idx", ""),
        ).contiguous()
        quat = broadcast_tensor(
            quat,
            gs.tc_float,
            (len(envs_idx), len(colliders_idx), 4),
            ("envs_idx", "colliders_idx", ""),
        ).contiguous()
        quat_norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        if not torch.isfinite(pos).all() or not torch.isfinite(quat).all():
            gs.raise_exception("PBSTF static collider poses must be finite.")
        if (quat_norm <= gs.EPS).any():
            gs.raise_exception("PBSTF static collider quaternions must be non-zero.")
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
            self._kernel_set_static_colliders_pose(colliders_idx, envs_idx, pos, quat)

    def _set_deformable_collider_sdf(self, collider, envs_idx, surface_positions, is_sdf_active):
        is_sdf_active = broadcast_tensor(
            is_sdf_active,
            gs.tc_bool,
            (len(envs_idx),),
            ("envs_idx",),
        ).contiguous()
        if not collider.has_sdf:
            if is_sdf_active.any():
                gs.raise_exception("PBSTF FEM-bound collider SDF activation requires a configured `sdf_res`.")
            return

        if gs.use_zerocopy:
            is_sdf_active_dst = qd_to_torch(collider.is_sdf_active, transpose=True, copy=False)
            is_sdf_active_dst[envs_idx] = False
        else:
            pbstf_absorption.kernel_disable_deformable_collider_sdf(envs_idx, collider)

        if not is_sdf_active.any():
            if gs.use_zerocopy and gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        active_envs_idx = envs_idx[is_sdf_active]
        active_surface_positions = tensor_to_array(surface_positions[is_sdf_active])
        n_active_envs = len(active_envs_idx)
        sdf = np.empty(
            (n_active_envs, collider.sdf_res, collider.sdf_res, collider.sdf_res),
            dtype=gs.np_float,
        )
        sdf_lower = np.empty((n_active_envs, 3), dtype=gs.np_float)
        sdf_inv_cell_size = np.empty((n_active_envs, 3), dtype=gs.np_float)
        for env_idx_local in range(n_active_envs):
            sdf_data = load_or_build_mesh_sdf(
                active_surface_positions[env_idx_local], collider.surface_faces_array, collider.sdf_res
            )
            sdf[env_idx_local] = sdf_data.values
            sdf_lower[env_idx_local] = sdf_data.lower
            sdf_inv_cell_size[env_idx_local] = 1.0 / sdf_data.cell_size

        sdf = torch.as_tensor(sdf, device=gs.device)
        sdf_lower = torch.as_tensor(sdf_lower, device=gs.device)
        sdf_inv_cell_size = torch.as_tensor(sdf_inv_cell_size, device=gs.device)
        if gs.use_zerocopy:
            sdf_dst = qd_to_torch(collider.sdf, transpose=True, copy=False)
            sdf_lower_dst = qd_to_torch(collider.sdf_lower, transpose=True, copy=False)
            sdf_inv_cell_size_dst = qd_to_torch(collider.sdf_inv_cell_size, transpose=True, copy=False)
            sdf_dst[active_envs_idx] = sdf
            sdf_lower_dst[active_envs_idx] = sdf_lower
            sdf_inv_cell_size_dst[active_envs_idx] = sdf_inv_cell_size
            is_sdf_active_dst[active_envs_idx] = True
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            pbstf_absorption.kernel_set_deformable_collider_sdf(
                active_envs_idx, sdf, sdf_lower, sdf_inv_cell_size, collider
            )

    @gs.assert_built
    def update_static_collider_deformation(self, collider_idx, envs_idx=None, is_sdf_enabled=False):
        """Synchronize a FEM-bound absorbent collider with its current deformed material points.

        The finite element method (FEM) entity supplies geometry only: position-based surface tension flow (PBSTF)
        forces remain one-way. Call this after each FEM step whose deformation should affect later fluid steps.
        Enabling the signed distance field (SDF) builds a cached field from the synchronized surface. It makes later
        queries independent of triangle count at cubic preprocessing and memory cost, and suits a shape whose local
        deformation has stopped. A later synchronization with ``is_sdf_enabled=False`` resumes exact triangle queries.
        """
        if not isinstance(collider_idx, (int, np.integer)):
            gs.raise_exception("PBSTF collider deformation requires one integer `collider_idx`.")
        if collider_idx < 0 or collider_idx >= self._n_static_colliders:
            gs.raise_exception(f"PBSTF static collider index {collider_idx} is out of range.")
        collider = self._static_colliders[collider_idx]
        if not isinstance(collider, AbsorbentStaticCollider) or not collider.is_deformable:
            gs.raise_exception(f"PBSTF static collider {collider_idx} has no FEM deformation binding.")
        if is_sdf_enabled and not collider.has_sdf:
            gs.raise_exception(f"PBSTF static collider {collider_idx} has no configured SDF resolution.")

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        embedded_positions = collider.fem_entity.get_embedded_positions(
            collider.embedding_elements_idx,
            collider.embedding_barycentric,
            envs_idx if self._scene.n_envs > 0 else None,
        )
        collider_pos = qd_to_torch(
            self._static_colliders_pos, envs_idx, (collider_idx,), transpose=True,
        )[..., 0, :]
        collider_quat = qd_to_torch(
            self._static_colliders_quat, envs_idx, (collider_idx,), transpose=True,
        )[..., 0, :]
        local_positions = gu.inv_transform_by_trans_quat(
            embedded_positions, collider_pos[:, None, :], collider_quat[:, None, :],
        )
        surface_positions = local_positions[:, : collider.n_surface_vertices]
        voxel_positions = local_positions[:, collider.n_surface_vertices :]
        if not torch.isfinite(local_positions).all():
            gs.raise_exception("PBSTF FEM-bound collider positions must be finite.")

        face_v0 = surface_positions[:, collider.surface_faces_tensor[:, 0]]
        face_v1 = surface_positions[:, collider.surface_faces_tensor[:, 1]]
        face_v2 = surface_positions[:, collider.surface_faces_tensor[:, 2]]
        face_area_twice = torch.linalg.vector_norm(torch.linalg.cross(face_v1 - face_v0, face_v2 - face_v0), dim=-1)
        if (face_area_twice <= gs.EPS).any():
            gs.raise_exception("PBSTF FEM-bound collider surface triangles must remain non-degenerate.")

        voxel_delta = voxel_positions[:, :, None, :] - voxel_positions[:, None, :, :]
        physical_distance_sqr = torch.sum(voxel_delta * voxel_delta, dim=-1)
        physical_order = torch.argsort(physical_distance_sqr, dim=-1, stable=True)
        graph_distance = collider.voxel_graph_distance[None].expand(len(envs_idx), -1, -1)
        graph_distance_ordered = torch.gather(graph_distance, dim=-1, index=physical_order)
        graph_order = torch.argsort(graph_distance_ordered, dim=-1, stable=True)
        voxel_search_order = torch.gather(physical_order, dim=-1, index=graph_order)
        voxel_search_order = broadcast_tensor(
            voxel_search_order,
            gs.tc_int,
            (len(envs_idx), collider.n_voxels, collider.n_voxels),
            ("envs_idx", "origin_voxel_idx", "search_idx"),
        ).contiguous()

        if gs.use_zerocopy:
            surface_vertices = qd_to_torch(collider.surface_vertices, transpose=True, copy=False)
            voxel_positions_dst = qd_to_torch(collider.voxel_positions, transpose=True, copy=False)
            voxel_search_order_dst = qd_to_torch(collider.voxel_search_order, transpose=True, copy=False)
            surface_vertices[envs_idx] = surface_positions
            voxel_positions_dst[envs_idx] = voxel_positions
            voxel_search_order_dst[envs_idx] = voxel_search_order
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            pbstf_absorption.kernel_set_deformable_collider_geometry(
                envs_idx, surface_positions, voxel_positions, voxel_search_order, collider,
            )
        self._set_deformable_collider_sdf(collider, envs_idx, surface_positions, is_sdf_enabled)

    @gs.assert_built
    def get_static_collider_wetness(self, collider_idx, envs_idx=None):
        """Return local-grid wetness for one absorbent position-based surface tension flow (PBSTF) static collider.

        Values are ordered from the collider's local ``lower`` corner to ``upper`` corner along each grid axis and lie
        in ``[0, 1]``. A single-environment scene returns ``[nx, ny, nz]``; a batched scene returns
        ``[B, nx, ny, nz]``.
        """
        if not isinstance(collider_idx, (int, np.integer)):
            gs.raise_exception("PBSTF static collider wetness requires one integer `collider_idx`.")
        if collider_idx < 0 or collider_idx >= self._n_static_colliders:
            gs.raise_exception(f"PBSTF static collider index {collider_idx} is out of range.")
        collider = self._static_colliders[collider_idx]
        if not isinstance(collider, AbsorbentStaticCollider):
            gs.raise_exception(f"PBSTF static collider {collider_idx} is not absorbent.")

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        wetness = qd_to_torch(
            self._absorption_voxel_wetness,
            envs_idx,
            slice(collider.voxel_start, collider.voxel_start + collider.n_voxels),
            transpose=True,
        ).clamp(0.0, 1.0)
        wetness = wetness.reshape((len(envs_idx), *collider.grid_res))
        return wetness[0] if self._sim.n_envs == 0 else wetness

    @qd.kernel
    def _kernel_set_static_colliders_pose(
        self,
        colliders_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        pos: qd.types.ndarray(),
        quat: qd.types.ndarray(),
    ):
        for env_idx_local, collider_idx_local in qd.ndrange(envs_idx.shape[0], colliders_idx.shape[0]):
            env_idx = envs_idx[env_idx_local]
            collider_idx = colliders_idx[collider_idx_local]
            for axis in qd.static(range(3)):
                self._static_colliders_pos[collider_idx, env_idx][axis] = pos[env_idx_local, collider_idx_local, axis]
            for axis in qd.static(range(4)):
                self._static_colliders_quat[collider_idx, env_idx][axis] = quat[env_idx_local, collider_idx_local, axis]

    def _init_particle_fields(self):
        particle_state = qd.types.struct(
            pos=gs.qd_vec3,
            ipos=gs.qd_vec3,
            dpos=gs.qd_vec3,
            vel=gs.qd_vec3,
            density=gs.qd_float,
            lmd=gs.qd_float,
            grad_i=gs.qd_vec3,
            surface=gs.qd_bool,
        )
        particle_state_ng = qd.types.struct(reordered_idx=gs.qd_int, active=gs.qd_bool)
        particle_info = qd.types.struct(mass=gs.qd_float, rho_rest=gs.qd_float)
        particle_render = qd.types.struct(pos=gs.qd_vec3, vel=gs.qd_vec3, active=gs.qd_bool)

        shape = (self._n_particles, self._B)
        self.particles = particle_state.field(shape=shape, layout=qd.Layout.SOA)
        self.particles_ng = particle_state_ng.field(shape=shape, layout=qd.Layout.SOA)
        self.particles_info = particle_info.field(shape=(self._n_particles,), layout=qd.Layout.SOA)
        self.particles_reordered = particle_state.field(shape=shape, layout=qd.Layout.SOA)
        self.particles_ng_reordered = particle_state_ng.field(shape=shape, layout=qd.Layout.SOA)
        self.particles_info_reordered = particle_info.field(shape=shape, layout=qd.Layout.SOA)
        self.particles_render = particle_render.field(shape=shape, layout=qd.Layout.SOA)
        self._max_density = qd.field(gs.qd_float, shape=())

    def _init_absorption_fields(self):
        absorption_particle_state = qd.types.struct(
            collider_idx=gs.qd_int,
            voxel_idx=gs.qd_int,
            voxel_distance=gs.qd_int,
            local_pos=gs.qd_vec3,
            target_local_pos=gs.qd_vec3,
            progress=gs.qd_float,
        )

        voxel_start = 0
        voxel_search_offset_start = 0
        surface_vertex_state_start = 0
        voxel_state_start = 0
        voxel_search_order_state_start = 0
        voxel_search_offsets = []
        for collider_idx in self._absorbent_static_colliders_idx:
            collider = self._static_colliders[collider_idx]
            collider.grid_res = np.ceil((collider.upper - collider.lower) / self._support_radius).astype(gs.np_int)
            collider.grid_res_qd = qd.Vector(collider.grid_res, dt=gs.qd_int)
            collider.voxel_size = (collider.upper - collider.lower) / collider.grid_res
            collider.voxel_size_qd = qd.Vector(collider.voxel_size, dt=gs.qd_float)
            collider.voxel_start = voxel_start
            collider.n_voxels = int(np.prod(collider.grid_res))
            voxel_start += collider.n_voxels

            if collider.is_deformable:
                fem_entity = self.scene.get_entity(name=collider.fem_entity_name)
                if not isinstance(fem_entity, FEMEntity) or fem_entity.elems.shape[1] != 4:
                    gs.raise_exception(
                        f"PBSTF absorbent collider FEM binding {collider.fem_entity_name!r} requires a volumetric "
                        "FEM entity."
                    )

                surface_vertices_idx = np.unique(fem_entity.surface_triangles)
                surface_vertex_mapping = np.full(fem_entity.n_vertices, -1, dtype=gs.np_int)
                surface_vertex_mapping[surface_vertices_idx] = np.arange(len(surface_vertices_idx))
                surface_faces = surface_vertex_mapping[fem_entity.surface_triangles]
                fem_init_positions = tensor_to_array(fem_entity.init_positions)

                grid_coordinates = np.stack(
                    np.meshgrid(*(np.arange(resolution) + 0.5 for resolution in collider.grid_res), indexing="ij"),
                    axis=-1,
                ).reshape((-1, 3))
                voxel_positions = collider.lower + grid_coordinates * collider.voxel_size
                voxel_positions_world = gu.transform_by_trans_quat(voxel_positions, collider.pos, collider.quat)
                query_positions = np.concatenate(
                    (
                        fem_init_positions[surface_vertices_idx],
                        voxel_positions_world,
                    )
                )
                element_vertices = fem_init_positions[fem_entity.elems]
                element_edges = np.swapaxes(element_vertices[:, 1:] - element_vertices[:, :1], 1, 2)
                element_edges_inv = np.linalg.inv(element_edges)
                query_offsets = query_positions[:, None, :] - element_vertices[None, :, 0, :]
                barycentric_tail = np.einsum("eij,pej->pei", element_edges_inv, query_offsets)
                barycentric = np.concatenate(
                    (
                        1.0 - barycentric_tail.sum(axis=-1, keepdims=True),
                        barycentric_tail,
                    ),
                    axis=-1,
                )
                containing_score = barycentric.min(axis=-1)
                embedding_elements_idx = containing_score.argmax(axis=-1)
                embedding_barycentric = barycentric[np.arange(len(query_positions)), embedding_elements_idx]
                if (embedding_barycentric < -1.0e-5).any():
                    gs.raise_exception(
                        f"PBSTF absorbent collider bounds for {collider.fem_entity_name!r} must lie inside its FEM "
                        "tetrahedral mesh."
                    )

                surface_positions = gu.inv_transform_by_trans_quat(
                    fem_init_positions[surface_vertices_idx], collider.pos, collider.quat
                )
                material_coordinates = np.stack(
                    np.meshgrid(*(np.arange(resolution) for resolution in collider.grid_res), indexing="ij"), axis=-1
                ).reshape((-1, 3))
                graph_distance = np.abs(
                    material_coordinates[:, None, :] - material_coordinates[None, :, :]
                ).sum(axis=-1)
                physical_distance_sqr = np.square(
                    voxel_positions[:, None, :] - voxel_positions[None, :, :]
                ).sum(axis=-1)
                physical_order = np.argsort(physical_distance_sqr, axis=-1, kind="stable")
                graph_distance_ordered = np.take_along_axis(graph_distance, physical_order, axis=-1)
                graph_order = np.argsort(graph_distance_ordered, axis=-1, kind="stable")
                voxel_search_order = np.take_along_axis(physical_order, graph_order, axis=-1)

                collider.fem_entity = fem_entity
                collider.embedding_elements_idx = torch.as_tensor(
                    embedding_elements_idx, dtype=gs.tc_int, device=gs.device
                )
                collider.embedding_barycentric = torch.as_tensor(
                    embedding_barycentric, dtype=gs.tc_float, device=gs.device
                )
                collider.n_surface_vertices = len(surface_vertices_idx)
                collider.n_surface_triangles = len(surface_faces)
                collider.surface_faces_array = surface_faces
                collider.surface_faces_tensor = torch.as_tensor(surface_faces, device=gs.device)
                collider.voxel_graph_distance = torch.as_tensor(graph_distance, device=gs.device)
                collider.surface_vertex_state_start = surface_vertex_state_start
                collider.voxel_state_start = voxel_state_start
                collider.voxel_search_order_state_start = voxel_search_order_state_start
                collider.surface_faces = qd.field(gs.qd_ivec3, shape=(collider.n_surface_triangles,))
                collider.surface_vertices = qd.field(gs.qd_vec3, shape=(collider.n_surface_vertices, self._B))
                if collider.has_sdf:
                    collider.sdf = qd.field(
                        gs.qd_float, shape=(collider.sdf_res, collider.sdf_res, collider.sdf_res, self._B)
                    )
                    collider.sdf_lower = qd.field(gs.qd_vec3, shape=(self._B,))
                    collider.sdf_inv_cell_size = qd.field(gs.qd_vec3, shape=(self._B,))
                    collider.is_sdf_active = qd.field(gs.qd_bool, shape=(self._B,))
                    collider.is_sdf_active.fill(False)
                    collider.sdf_state_idx = self._n_deformable_sdf_colliders
                    self._n_deformable_sdf_colliders += 1
                collider.voxel_positions = qd.field(gs.qd_vec3, shape=(collider.n_voxels, self._B))
                collider.voxel_search_order = qd.field(
                    gs.qd_int, shape=(collider.n_voxels, collider.n_voxels, self._B)
                )
                collider.surface_faces.from_numpy(surface_faces)
                collider.surface_vertices.from_numpy(
                    np.repeat(surface_positions[:, None, :], repeats=self._B, axis=1)
                )
                collider.voxel_positions.from_numpy(
                    np.repeat(voxel_positions[:, None, :], repeats=self._B, axis=1)
                )
                collider.voxel_search_order.from_numpy(
                    np.repeat(voxel_search_order[:, :, None], repeats=self._B, axis=2)
                )
                surface_vertex_state_start += collider.n_surface_vertices
                voxel_state_start += collider.n_voxels
                voxel_search_order_state_start += collider.n_voxels * collider.n_voxels
                self._n_deformable_static_colliders += 1

            collider.voxel_search_offset_start = voxel_search_offset_start
            collider.n_voxel_search_offsets = 0
            if not collider.is_deformable:
                axis_offsets = tuple(
                    np.arange(1 - collider.grid_res[axis], collider.grid_res[axis], dtype=gs.np_int)
                    for axis in range(3)
                )
                collider_search_offsets = np.stack(
                    np.meshgrid(*axis_offsets, indexing="ij"), axis=-1
                ).reshape((-1, 3))
                voxel_distances = np.abs(collider_search_offsets).sum(axis=-1)
                physical_distance_sq = np.square(collider_search_offsets * collider.voxel_size).sum(axis=-1)
                search_order = np.lexsort(
                    (
                        collider_search_offsets[:, 2],
                        collider_search_offsets[:, 1],
                        collider_search_offsets[:, 0],
                        physical_distance_sq,
                        voxel_distances,
                    )
                )
                collider_search_offsets = collider_search_offsets[search_order]
                collider.n_voxel_search_offsets = len(collider_search_offsets)
                voxel_search_offset_start += collider.n_voxel_search_offsets
                voxel_search_offsets.append(collider_search_offsets)

        self._n_absorption_voxels = voxel_start
        self._n_deformable_surface_vertices = surface_vertex_state_start
        self._n_deformable_voxels = voxel_state_start
        self._n_deformable_voxel_search_order = voxel_search_order_state_start
        shape = (self._n_particles, self._B)
        self._absorption_particles = absorption_particle_state.field(shape=shape, layout=qd.Layout.SOA)
        self._absorption_particles_reordered = absorption_particle_state.field(shape=shape, layout=qd.Layout.SOA)
        self._absorption_voxel_capacity = qd.field(gs.qd_int, shape=(self._n_absorption_voxels,))
        self._absorption_voxel_occupancy = qd.field(gs.qd_int, shape=(self._n_absorption_voxels, self._B))
        self._absorption_voxel_wetness = qd.field(gs.qd_float, shape=(self._n_absorption_voxels, self._B))
        if voxel_search_offset_start > 0:
            self._absorption_voxel_search_offsets = qd.field(gs.qd_ivec3, shape=(voxel_search_offset_start,))
        self._absorption_capture_budget = qd.field(gs.qd_float, shape=(self._n_absorbent_static_colliders, self._B))
        self._absorption_voxel_capacity.fill(0)
        self._absorption_voxel_occupancy.fill(0)
        self._absorption_voxel_wetness.fill(0.0)
        if self._absorption_voxel_search_offsets is not None:
            self._absorption_voxel_search_offsets.from_numpy(np.concatenate(voxel_search_offsets))
        self._absorption_capture_budget.fill(0.0)
        pbstf_absorption.kernel_initialize_absorption_particles(self._n_particles, self._absorption_particles)
        pbstf_absorption.kernel_initialize_absorption_particles(
            self._n_particles, self._absorption_particles_reordered
        )

    def _init_surface_fields(self):
        n = self._n_particles
        b = self._B

        self.on_surface = qd.field(gs.qd_bool, shape=(n, b))
        self.topology_valid = qd.field(gs.qd_bool, shape=(n, b))
        self.density_constraint_enabled = qd.field(gs.qd_bool, shape=(n, b))
        self.normals = qd.field(gs.qd_vec3, shape=(n, b))
        self._has_interior_neighbor = qd.field(gs.qd_bool, shape=(n, b))
        self._pca_covariance = qd.field(gs.qd_mat3, shape=(n, b))

        # Difference grid used by the reference spherical-illumination surface
        # classifier (18 latitude x 36 longitude samples).
        self._screen_blocked = qd.field(gs.qd_int, shape=(n, b, self._N_THETA + 1, self._N_PHI + 1))

        candidate_shape = (n, b, self._max_surface_neighbors)
        self.projected_positions = qd.field(gs.qd_vec2, shape=candidate_shape)
        self.neighbor_ids = qd.field(gs.qd_int, shape=candidate_shape)
        self._chain_pre = qd.field(gs.qd_int, shape=candidate_shape)
        self._chain_nxt = qd.field(gs.qd_int, shape=candidate_shape)
        local_mesh_shape = (n, b, self._max_localmesh_neighbors)
        self.local_mesh_neighbors = qd.field(gs.qd_int, shape=local_mesh_shape)
        self._surface_gradient = qd.field(gs.qd_vec3, shape=local_mesh_shape)
        self.n_neighbors = qd.field(gs.qd_int, shape=(n, b))
        self._mesh_axis_x = qd.field(gs.qd_vec3, shape=(n, b))
        self._mesh_axis_y = qd.field(gs.qd_vec3, shape=(n, b))
        # Polar-order scratch reuses the queue because initial queue writes stay behind the scan cursor.
        self._node_queue = qd.field(gs.qd_int, shape=(n, b, 3 * self._max_surface_neighbors))
        self._surface_lambda = qd.field(gs.qd_float, shape=(n, b))
        self._surface_grad_i = qd.field(gs.qd_vec3, shape=(n, b))
        self._overflow = qd.field(gs.qd_int, shape=())

    @qd.func
    def _is_particle_absorbed_reordered(self, particle_idx, env_idx):
        is_absorbed = False
        if qd.static(self._n_absorbent_static_colliders > 0):
            is_absorbed = pbstf_absorption.is_particle_absorbed(
                particle_idx, env_idx, self._absorption_particles_reordered
            )
        return is_absorbed

    def _capture_absorbent_contacts(self):
        error_code = int(ErrorCode.INVALID_PBSTF_STATE_NAN)
        for absorption_idx, collider_idx in enumerate(self._absorbent_static_colliders_idx):
            collider = self._static_colliders[collider_idx]
            voxel_search_offsets = self._absorption_voxel_search_offsets
            if collider.is_deformable:
                voxel_search_offsets = collider.voxel_search_order
            pbstf_absorption.kernel_capture_particles(
                self._n_particles,
                collider_idx,
                absorption_idx,
                self._particle_radius,
                self._substep_dt,
                collider.absorption_rate,
                self.particles_reordered,
                self.particles_ng_reordered,
                self._absorption_particles_reordered,
                self._absorption_capture_budget,
                self._absorption_voxel_capacity,
                self._absorption_voxel_occupancy,
                voxel_search_offsets,
                self._static_colliders_pos,
                self._static_colliders_quat,
                collider,
                error_code,
                self._errno,
            )
        self._kernel_invalidate_absorbed_topology()

    def _rebuild_absorption_fields(self, absorption_particles):
        self._absorption_voxel_occupancy.fill(0)
        self._absorption_voxel_wetness.fill(0.0)
        pbstf_absorption.kernel_rebuild_voxels(
            self._n_particles,
            self._n_absorption_voxels,
            self._n_absorbent_static_colliders,
            absorption_particles,
            self._absorption_capture_budget,
            self._absorption_voxel_capacity,
            self._absorption_voxel_occupancy,
            self._absorption_voxel_wetness,
            int(ErrorCode.INVALID_PBSTF_STATE_NAN),
            self._errno,
        )

    @qd.kernel
    def _kernel_invalidate_absorbed_topology(self):
        for particle_idx, env_idx in qd.ndrange(self._n_particles, self._B):
            is_valid = self.topology_valid[particle_idx, env_idx]
            if self._is_particle_absorbed_reordered(particle_idx, env_idx):
                is_valid = False
            for neighbor_local_idx in range(self.n_neighbors[particle_idx, env_idx]):
                neighbor_idx = self.local_mesh_neighbors[particle_idx, env_idx, neighbor_local_idx]
                if self._is_particle_absorbed_reordered(neighbor_idx, env_idx):
                    is_valid = False
            self.topology_valid[particle_idx, env_idx] = is_valid

    @qd.func
    def _project_out_static_colliders(self, env_idx, pos):
        for collider_idx in qd.static(range(self._n_static_colliders)):
            pos = project_out_static_collider(
                collider_idx,
                env_idx,
                pos,
                self._particle_radius,
                self._static_colliders_pos,
                self._static_colliders_quat,
                self._static_colliders[collider_idx],
            )
        return pos

    @qd.func
    def _separated_by_static_colliders(self, env_idx, pos_i, pos_j):
        separated = False
        for collider_idx in qd.static(range(self._n_static_colliders)):
            if static_collider_separates(
                collider_idx,
                env_idx,
                pos_i,
                pos_j,
                self._particle_radius,
                self._static_colliders_pos,
                self._static_colliders_quat,
                self._static_colliders[collider_idx],
            ):
                separated = True
        return separated

    # ------------------------------------------------------------------
    # Cubic spline used everywhere in PBSTF
    # ------------------------------------------------------------------

    @qd.func
    def cubic_kernel(self, distance):
        result = gs.qd_float(0.0)
        q = distance / self._support_radius
        coefficient = 8.0 / (math.pi * self._support_radius**3)
        if q < 0.5:
            result = coefficient * (6.0 * q * q * (q - 1.0) + 1.0)
        elif q < 1.0:
            result = 2.0 * coefficient * (1.0 - q) ** 3
        return result

    @qd.func
    def cubic_kernel_first_derivative(self, distance):
        result = gs.qd_float(0.0)
        q = distance / self._support_radius
        coefficient = 48.0 / (math.pi * self._support_radius**4)
        if q < 0.5:
            result = coefficient * q * (3.0 * q - 2.0)
        elif q < 1.0:
            result = coefficient * (1.0 - q) * (q - 1.0)
        return result

    @qd.func
    def cubic_gradient_kernel(self, delta):
        """Reference ``gradientKernel``: gradient with respect to the second point."""
        result = qd.Vector.zero(gs.qd_float, 3)
        distance = delta.norm()
        if distance > gs.EPS and distance < self._support_radius:
            result = -self.cubic_kernel_first_derivative(distance) * delta / distance
        return result

    # ------------------------------------------------------------------
    # Reordering and density
    # ------------------------------------------------------------------

    @qd.kernel
    def _kernel_reorder_particles(self, f: qd.i32):
        self.sh.compute_reordered_idx(
            self._n_particles, self.particles.pos, self.particles_ng.active, self.particles_ng.reordered_idx
        )
        self.particles_ng_reordered.active.fill(False)
        if qd.static(self._n_absorbent_static_colliders > 0):
            self._absorption_particles_reordered.collider_idx.fill(-1)
            self._absorption_particles_reordered.voxel_idx.fill(-1)
            self._absorption_particles_reordered.progress.fill(0.0)
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i, i_b].active:
                j = self.particles_ng[i, i_b].reordered_idx
                self.particles_reordered[j, i_b] = self.particles[i, i_b]
                self.particles_info_reordered[j, i_b] = self.particles_info[i]
                self.particles_ng_reordered[j, i_b].active = True
                if qd.static(self._n_absorbent_static_colliders > 0):
                    self._absorption_particles_reordered[j, i_b] = self._absorption_particles[i, i_b]

    @qd.kernel
    def _kernel_copy_from_reordered(self, f: qd.i32):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i, i_b].active:
                j = self.particles_ng[i, i_b].reordered_idx
                self.particles[i, i_b] = self.particles_reordered[j, i_b]
                if qd.static(self._n_absorbent_static_colliders > 0):
                    self._absorption_particles[i, i_b] = self._absorption_particles_reordered[j, i_b]

    @qd.func
    def _task_density(self, i, j, result: qd.template(), i_b):
        if self.particles_ng_reordered[j, i_b].active and not self._is_particle_absorbed_reordered(j, i_b):
            pos_i = self.particles_reordered[i, i_b].pos
            pos_j = self.particles_reordered[j, i_b].pos
            if not self._separated_by_static_colliders(i_b, pos_i, pos_j):
                distance = (pos_i - pos_j).norm()
                result += self.particles_info_reordered[j, i_b].mass * self.cubic_kernel(distance)

    @qd.kernel
    def _kernel_compute_density(self, f: qd.i32):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i, i_b].active and not self._is_particle_absorbed_reordered(i, i_b):
                density = self.particles_info_reordered[i, i_b].mass * self.cubic_kernel(0.0)
                self.sh.for_all_neighbors(
                    i, self.particles_reordered.pos, self._support_radius, density, self._task_density, i_b
                )
                self.particles_reordered[i, i_b].density = density
            elif self.particles_ng_reordered[i, i_b].active:
                self.particles_reordered[i, i_b].density = 0.0

    @qd.kernel
    def _kernel_reduce_max_density(self):
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i, 0].active and not self._is_particle_absorbed_reordered(i, 0):
                qd.atomic_max(self._max_density[None], self.particles_reordered[i, 0].density)

    @qd.kernel
    def _kernel_set_particle_mass(self, mass: qd.f32):
        for i in range(self._n_particles):
            self.particles_info[i].mass = mass

    # ------------------------------------------------------------------
    # Surface classification and normals
    # ------------------------------------------------------------------

    @qd.func
    def _task_mark_screen(self, i, j, unused: qd.template(), i_b):
        delta = self.particles_reordered[j, i_b].pos - self.particles_reordered[i, i_b].pos
        distance = delta.norm()
        if (
            not self._is_particle_absorbed_reordered(j, i_b)
            and distance > gs.EPS
            and not self._separated_by_static_colliders(
                i_b, self.particles_reordered[i, i_b].pos, self.particles_reordered[j, i_b].pos
            )
        ):
            unit_theta = math.pi / self._N_THETA
            unit_phi = 2.0 * math.pi / self._N_PHI
            block_radius = qd.min(self._particle_radius, 0.5 * distance)
            delta_angle = qd.asin(qd.min(block_radius / distance, 1.0))
            theta = qd.acos(qd.max(-1.0, qd.min(1.0, delta[1] / distance)))
            phi = qd.atan2(delta[2], delta[0])

            start_theta = qd.max(theta - delta_angle, 0.0)
            end_theta = qd.min(theta + delta_angle, math.pi)
            start_phi = phi - delta_angle
            end_phi = phi + delta_angle
            if start_phi < -math.pi:
                start_phi += 2.0 * math.pi
            if end_phi > math.pi:
                end_phi -= 2.0 * math.pi

            st_t = qd.min(qd.cast(qd.floor(start_theta / unit_theta), gs.qd_int), self._N_THETA - 1)
            en_t = qd.min(qd.cast(qd.ceil(end_theta / unit_theta), gs.qd_int), self._N_THETA)
            st_p = qd.min(qd.cast(qd.floor((start_phi + math.pi) / unit_phi), gs.qd_int), self._N_PHI - 1)
            en_p = qd.min(qd.cast(qd.ceil((end_phi + math.pi) / unit_phi), gs.qd_int), self._N_PHI)

            if st_p < en_p:
                qd.atomic_add(self._screen_blocked[i, i_b, st_t, st_p], 1)
                qd.atomic_add(self._screen_blocked[i, i_b, st_t, en_p], -1)
                qd.atomic_add(self._screen_blocked[i, i_b, en_t, st_p], -1)
                qd.atomic_add(self._screen_blocked[i, i_b, en_t, en_p], 1)
            else:
                qd.atomic_add(self._screen_blocked[i, i_b, st_t, st_p], 1)
                qd.atomic_add(self._screen_blocked[i, i_b, st_t, self._N_PHI], -1)
                qd.atomic_add(self._screen_blocked[i, i_b, en_t, st_p], -1)
                qd.atomic_add(self._screen_blocked[i, i_b, en_t, self._N_PHI], 1)
                qd.atomic_add(self._screen_blocked[i, i_b, st_t, 0], 1)
                qd.atomic_add(self._screen_blocked[i, i_b, st_t, en_p], -1)
                qd.atomic_add(self._screen_blocked[i, i_b, en_t, 0], -1)
                qd.atomic_add(self._screen_blocked[i, i_b, en_t, en_p], 1)

    @qd.kernel
    def _kernel_mark_surface_screen(self):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i, i_b].active and not self._is_particle_absorbed_reordered(i, i_b):
                unused = gs.qd_int(0)
                self.sh.for_all_neighbors(
                    i, self.particles_reordered.pos, self._support_radius, unused, self._task_mark_screen, i_b
                )

    @qd.kernel
    def _kernel_classify_surface(self):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i, i_b].active and not self._is_particle_absorbed_reordered(i, i_b):
                illuminated = gs.qd_float(0.0)
                total = gs.qd_float(0.0)
                for t in range(self._N_THETA):
                    weight = qd.sin(math.pi / self._N_THETA * (t + 0.5))
                    for p in range(self._N_PHI):
                        if t > 0 and p > 0:
                            self._screen_blocked[i, i_b, t, p] += (
                                self._screen_blocked[i, i_b, t - 1, p]
                                + self._screen_blocked[i, i_b, t, p - 1]
                                - self._screen_blocked[i, i_b, t - 1, p - 1]
                            )
                        elif t > 0:
                            self._screen_blocked[i, i_b, t, p] += self._screen_blocked[i, i_b, t - 1, p]
                        elif p > 0:
                            self._screen_blocked[i, i_b, t, p] += self._screen_blocked[i, i_b, t, p - 1]
                        if self._screen_blocked[i, i_b, t, p] == 0:
                            illuminated += weight
                        total += weight
                self.on_surface[i, i_b] = illuminated >= self._ILLUMINATED_THRESHOLD * total
            else:
                self.on_surface[i, i_b] = False

    @qd.func
    def _task_normal_covariance(self, i, j, unused: qd.template(), i_b):
        if not self._is_particle_absorbed_reordered(j, i_b):
            delta = self.particles_reordered[j, i_b].pos - self.particles_reordered[i, i_b].pos
            separated = self._separated_by_static_colliders(
                i_b, self.particles_reordered[i, i_b].pos, self.particles_reordered[j, i_b].pos
            )
            if not separated:
                density_j = self.particles_reordered[j, i_b].density
                if density_j > gs.EPS:
                    # Keep the C++ formula m_j / rho_j verbatim. PBSTF currently
                    # calibrates one shared mass, but the neighbor-density weight
                    # is still spatially varying and is part of the reference.
                    self.normals[i, i_b] += (
                        -self.cubic_gradient_kernel(delta) * self.particles_info_reordered[j, i_b].mass / density_j
                    )
                if not self.on_surface[j, i_b]:
                    self._has_interior_neighbor[i, i_b] = True
            if self.on_surface[j, i_b]:
                self._pca_covariance[i, i_b] += delta.outer_product(delta)

    @qd.kernel
    def _kernel_compute_normals(self):
        self.normals.fill(0.0)
        self._pca_covariance.fill(0.0)
        self._has_interior_neighbor.fill(False)
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_ng_reordered[i, i_b].active
                and not self._is_particle_absorbed_reordered(i, i_b)
                and self.on_surface[i, i_b]
            ):
                unused = gs.qd_int(0)
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    self._support_radius,
                    unused,
                    self._task_normal_covariance,
                    i_b,
                )

        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_ng_reordered[i, i_b].active
                and not self._is_particle_absorbed_reordered(i, i_b)
                and self.on_surface[i, i_b]
            ):
                raw_normal = self.normals[i, i_b]
                raw_length = raw_normal.norm()
                if raw_length <= 1.0:
                    self.on_surface[i, i_b] = False
                    self.normals[i, i_b] = qd.Vector.zero(gs.qd_float, 3)
                else:
                    normal = raw_normal / raw_length
                    if qd.static(self._enable_pca_normals) and not self._has_interior_neighbor[i, i_b]:
                        eigenvalues, eigenvectors = qd.sym_eig(self._pca_covariance[i, i_b])
                        normal_pca = qd.Vector(
                            [eigenvectors[0, 0], eigenvectors[1, 0], eigenvectors[2, 0]], dt=gs.qd_float
                        )
                        if normal.dot(normal_pca) < 0.0:
                            normal_pca = -normal_pca
                        if normal_pca.norm() > gs.EPS:
                            normal = normal_pca.normalized()
                    self.normals[i, i_b] = normal

        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i, i_b].active:
                self.particles_reordered[i, i_b].surface = (
                    self.on_surface[i, i_b] and not self._is_particle_absorbed_reordered(i, i_b)
                )

    @qd.func
    def _task_surface_covariance(self, i, j, unused: qd.template(), i_b):
        if self.on_surface[j, i_b] and not self._is_particle_absorbed_reordered(j, i_b):
            delta = self.particles_reordered[j, i_b].pos - self.particles_reordered[i, i_b].pos
            self._pca_covariance[i, i_b] += delta.outer_product(delta)

    @qd.kernel
    def _kernel_mark_density_constraints(self):
        self._pca_covariance.fill(0.0)
        self.density_constraint_enabled.fill(False)
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i, i_b].active and not self._is_particle_absorbed_reordered(i, i_b):
                unused = gs.qd_int(0)
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    self._support_radius,
                    unused,
                    self._task_surface_covariance,
                    i_b,
                )

        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i, i_b].active and not self._is_particle_absorbed_reordered(i, i_b):
                eigenvalues, unused_eigenvectors = qd.sym_eig(self._pca_covariance[i, i_b])
                eigen_sum = eigenvalues[0] + eigenvalues[1] + eigenvalues[2]
                eigen_max = qd.max(eigenvalues[0], qd.max(eigenvalues[1], eigenvalues[2]))
                self.density_constraint_enabled[i, i_b] = eigen_sum <= gs.EPS or eigen_max / eigen_sum <= 0.8

    # ------------------------------------------------------------------
    # GPU local mesh (ported from pbstf-accelerator/delaunator_2d.py)
    # ------------------------------------------------------------------

    @qd.func
    def _angle_of_vec2(self, a, b):
        result = gs.qd_float(0.0)
        a_len = a.norm()
        b_len = b.norm()
        if a_len > gs.EPS and b_len > gs.EPS:
            result = qd.acos(qd.max(-1.0, qd.min(1.0, a.dot(b) / (a_len * b_len))))
        return result

    @qd.func
    def _need_flip(self, i, i_b, u):
        result = False
        u_pre = self._chain_pre[i, i_b, u]
        u_nxt = self._chain_nxt[i, i_b, u]
        if u_pre >= 0 and u_nxt >= 0:
            x_pre = self.projected_positions[i, i_b, u_pre]
            x_u = self.projected_positions[i, i_b, u]
            x_nxt = self.projected_positions[i, i_b, u_nxt]
            result = self._angle_of_vec2(-x_pre, x_u - x_pre) + self._angle_of_vec2(-x_nxt, x_u - x_nxt) > math.pi
        return result

    @qd.func
    def _task_collect_mesh_neighbor(self, i, j, result: qd.template(), i_b):
        if (
            self.on_surface[j, i_b]
            and not self._is_particle_absorbed_reordered(j, i_b)
            and not self._separated_by_static_colliders(
                i_b, self.particles_reordered[i, i_b].pos, self.particles_reordered[j, i_b].pos
            )
        ):
            delta = self.particles_reordered[j, i_b].pos - self.particles_reordered[i, i_b].pos
            distance = delta.norm()
            normal_i = self.normals[i, i_b]
            normal_j = self.normals[j, i_b]
            if normal_i.dot(normal_j) > self._COS_PI_OVER_4 or (
                (normal_i - normal_j).dot(delta) > 0.0 and distance < 4.0 * self._particle_radius
            ):
                if result.count < self._max_surface_neighbors:
                    slot = result.count
                    self.neighbor_ids[i, i_b, slot] = j
                    projected = delta - delta.dot(normal_i) * normal_i
                    self.projected_positions[i, i_b, slot] = qd.Vector(
                        [projected.dot(self._mesh_axis_x[i, i_b]), projected.dot(self._mesh_axis_y[i, i_b])],
                        dt=gs.qd_float,
                    )
                    result.count += 1
                else:
                    result.overflow = 1

    @qd.kernel
    def _kernel_build_local_meshes(self):
        self.topology_valid.fill(False)
        self.n_neighbors.fill(0)
        self._overflow[None] = 0

        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_ng_reordered[i, i_b].active
                and not self._is_particle_absorbed_reordered(i, i_b)
                and self.on_surface[i, i_b]
            ):
                normal = self.normals[i, i_b]
                axis_x = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
                if axis_x.cross(normal).norm() < gs.EPS:
                    axis_x = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
                axis_x = axis_x.cross(normal).normalized()
                axis_y = normal.cross(axis_x).normalized()
                self._mesh_axis_x[i, i_b] = axis_x
                self._mesh_axis_y[i, i_b] = axis_y

                result = qd.Struct(count=0, overflow=0)
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    self._support_radius,
                    result,
                    self._task_collect_mesh_neighbor,
                    i_b,
                )
                if result.overflow:
                    qd.atomic_max(self._overflow[None], self._SURFACE_NEIGHBOR_OVERFLOW)
                n = result.count
                self.n_neighbors[i, i_b] = n

                for k in range(n):
                    self._node_queue[i, i_b, k] = k
                    self._chain_pre[i, i_b, k] = -1
                    self._chain_nxt[i, i_b, k] = -1

                # Insertion sort by polar angle, then by radius. This is the
                # same order as the accelerator's bubble sort.
                for k in range(1, n):
                    cursor = k
                    while cursor > 0:
                        u = self._node_queue[i, i_b, cursor - 1]
                        v = self._node_queue[i, i_b, cursor]
                        x_u = self.projected_positions[i, i_b, u]
                        x_v = self.projected_positions[i, i_b, v]
                        angle_u = qd.atan2(x_u[1], x_u[0])
                        angle_v = qd.atan2(x_v[1], x_v[0])
                        if angle_u > angle_v or (angle_u == angle_v and x_u.norm() > x_v.norm()):
                            self._node_queue[i, i_b, cursor - 1] = v
                            self._node_queue[i, i_b, cursor] = u
                            cursor -= 1
                        else:
                            cursor = 0

                if n >= 3:
                    for k in range(n - 1):
                        u = self._node_queue[i, i_b, k]
                        v = self._node_queue[i, i_b, k + 1]
                        self._chain_nxt[i, i_b, u] = v
                        self._chain_pre[i, i_b, v] = u
                    u_last = self._node_queue[i, i_b, n - 1]
                    u_first = self._node_queue[i, i_b, 0]
                    self._chain_nxt[i, i_b, u_last] = u_first
                    self._chain_pre[i, i_b, u_first] = u_last

                    queue_start = 0
                    queue_end = 0
                    for k in range(n):
                        u = self._node_queue[i, i_b, k]
                        if self._need_flip(i, i_b, u):
                            if queue_end < 3 * self._max_surface_neighbors:
                                self._node_queue[i, i_b, queue_end] = u
                                queue_end += 1
                            else:
                                qd.atomic_max(self._overflow[None], self._LOCAL_MESH_QUEUE_OVERFLOW)

                    while queue_start < queue_end:
                        u = self._node_queue[i, i_b, queue_start]
                        queue_start += 1
                        if self._chain_nxt[i, i_b, u] >= 0 and self._need_flip(i, i_b, u):
                            u_pre = self._chain_pre[i, i_b, u]
                            u_nxt = self._chain_nxt[i, i_b, u]
                            self._chain_nxt[i, i_b, u_pre] = u_nxt
                            self._chain_pre[i, i_b, u_nxt] = u_pre
                            if self._need_flip(i, i_b, u_nxt):
                                if queue_end < 3 * self._max_surface_neighbors:
                                    self._node_queue[i, i_b, queue_end] = u_nxt
                                    queue_end += 1
                                else:
                                    qd.atomic_max(self._overflow[None], self._LOCAL_MESH_QUEUE_OVERFLOW)
                            if self._need_flip(i, i_b, u_pre):
                                if queue_end < 3 * self._max_surface_neighbors:
                                    self._node_queue[i, i_b, queue_end] = u_pre
                                    queue_end += 1
                                else:
                                    qd.atomic_max(self._overflow[None], self._LOCAL_MESH_QUEUE_OVERFLOW)
                            self._chain_nxt[i, i_b, u] = -1
                            self._chain_pre[i, i_b, u] = -1

                    start = -1
                    for k in range(n):
                        if start < 0 and self._chain_nxt[i, i_b, k] >= 0:
                            start = k

                    ring_size = 0
                    projected_area_twice = gs.qd_float(0.0)
                    if start >= 0:
                        u = start
                        keep_walking = True
                        while keep_walking and ring_size < self._max_localmesh_neighbors:
                            u_next = self._chain_nxt[i, i_b, u]
                            x_u = self.projected_positions[i, i_b, u]
                            x_next = self.projected_positions[i, i_b, u_next]
                            projected_area_twice += x_u[0] * x_next[1] - x_u[1] * x_next[0]
                            self.local_mesh_neighbors[i, i_b, ring_size] = self.neighbor_ids[i, i_b, u]
                            ring_size += 1
                            u = u_next
                            if u == start:
                                keep_walking = False

                        if keep_walking:
                            qd.atomic_max(self._overflow[None], self._LOCAL_MESH_NEIGHBOR_OVERFLOW)
                        else:
                            self.topology_valid[i, i_b] = ring_size >= 3 and qd.abs(projected_area_twice) > gs.EPS

                    self.n_neighbors[i, i_b] = ring_size

    def _rebuild_topology(self, f):
        self._kernel_compute_density(f)
        self._screen_blocked.fill(0)
        self._kernel_mark_surface_screen()
        self._kernel_classify_surface()
        self._kernel_compute_normals()
        self._kernel_mark_density_constraints()
        self._kernel_build_local_meshes()
        overflow = qd_to_numpy(self._overflow, transpose=True)[()]
        if overflow == self._LOCAL_MESH_NEIGHBOR_OVERFLOW:
            gs.raise_exception(
                "PBSTF local mesh exceeded its one-ring neighbor capacity; increase "
                f"`max_localmesh_neighbors` (currently {self._max_localmesh_neighbors})."
            )
        if overflow:
            detail = "projected neighbor capacity" if overflow == self._SURFACE_NEIGHBOR_OVERFLOW else "queue capacity"
            gs.raise_exception(
                f"PBSTF local mesh exceeded its {detail}; increase `max_surface_neighbors` "
                f"(currently {self._max_surface_neighbors})."
            )

    # ------------------------------------------------------------------
    # Position constraints
    # ------------------------------------------------------------------

    @qd.func
    def _density_target(self, i, i_b):
        target = self.particles_info_reordered[i, i_b].rho_rest
        if self.on_surface[i, i_b]:
            target *= 0.7
        return target

    @qd.func
    def _task_density_constraint(self, i, j, result: qd.template(), i_b):
        pos_i = self.particles_reordered[i, i_b].pos
        pos_j = self.particles_reordered[j, i_b].pos
        if not self._is_particle_absorbed_reordered(j, i_b) and not self._separated_by_static_colliders(
            i_b, pos_i, pos_j
        ):
            mass_j = self.particles_info_reordered[j, i_b].mass
            rho_rest = self._density_target(i, i_b)
            delta = pos_i - pos_j
            result.density += mass_j * self.cubic_kernel(delta.norm())
            grad_j = mass_j / rho_rest * self.cubic_gradient_kernel(delta)
            result.grad_i -= grad_j
            result.denominator += grad_j.norm_sqr() / mass_j

    @qd.kernel
    def _kernel_prepare_density_constraints(self):
        self.particles_reordered.dpos.fill(0.0)
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_ng_reordered[i, i_b].active
                and not self._is_particle_absorbed_reordered(i, i_b)
                and self.density_constraint_enabled[i, i_b]
            ):
                mass_i = self.particles_info_reordered[i, i_b].mass
                rho_rest = self._density_target(i, i_b)
                result = qd.Struct(
                    density=mass_i * self.cubic_kernel(0.0),
                    grad_i=qd.Vector.zero(gs.qd_float, 3),
                    denominator=self._material.density_compliance / self._default_mass,
                )
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    self._support_radius,
                    result,
                    self._task_density_constraint,
                    i_b,
                )
                result.denominator += result.grad_i.norm_sqr() / mass_i
                constraint = result.density / rho_rest - 1.0
                lmd = gs.qd_float(0.0)
                if result.denominator > gs.EPS:
                    lmd = -constraint / result.denominator
                self.particles_reordered[i, i_b].density = result.density
                self.particles_reordered[i, i_b].grad_i = result.grad_i
                self.particles_reordered[i, i_b].lmd = lmd

    @qd.func
    def _task_apply_density_constraint(self, i, j, unused: qd.template(), i_b):
        pos_i = self.particles_reordered[i, i_b].pos
        pos_j = self.particles_reordered[j, i_b].pos
        if not self._is_particle_absorbed_reordered(j, i_b) and not self._separated_by_static_colliders(
            i_b, pos_i, pos_j
        ):
            rho_rest = self._density_target(i, i_b)
            mass_j = self.particles_info_reordered[j, i_b].mass
            delta = pos_i - pos_j
            grad_j = mass_j / rho_rest * self.cubic_gradient_kernel(delta)
            correction = self.particles_reordered[i, i_b].lmd / mass_j * grad_j
            for axis in qd.static(range(3)):
                qd.atomic_add(self.particles_reordered[j, i_b].dpos[axis], correction[axis])

    @qd.kernel
    def _kernel_apply_density_constraints(self):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_ng_reordered[i, i_b].active
                and not self._is_particle_absorbed_reordered(i, i_b)
                and self.density_constraint_enabled[i, i_b]
            ):
                mass_i = self.particles_info_reordered[i, i_b].mass
                correction_i = self.particles_reordered[i, i_b].lmd / mass_i * self.particles_reordered[i, i_b].grad_i
                for axis in qd.static(range(3)):
                    qd.atomic_add(self.particles_reordered[i, i_b].dpos[axis], correction_i[axis])
                unused = gs.qd_int(0)
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    self._support_radius,
                    unused,
                    self._task_apply_density_constraint,
                    i_b,
                )

    @qd.func
    def _triangle_area(self, a, b, c, i_b):
        pa = self.particles_reordered[a, i_b].pos
        pb = self.particles_reordered[b, i_b].pos
        pc = self.particles_reordered[c, i_b].pos
        return 0.5 * (pb - pa).cross(pc - pa).norm()

    @qd.func
    def _triangle_area_gradient(self, a, b, c, i_b):
        pa = self.particles_reordered[a, i_b].pos
        pb = self.particles_reordered[b, i_b].pos
        pc = self.particles_reordered[c, i_b].pos
        cross = (pb - pa).cross(pc - pa)
        result = qd.Vector.zero(gs.qd_float, 3)
        if cross.norm() > gs.EPS:
            result = 0.5 * cross.normalized().cross(pc - pb)
        return result

    @qd.kernel
    def _kernel_apply_surface_constraints(self):
        self._surface_gradient.fill(0.0)
        self._surface_lambda.fill(0.0)
        self._surface_grad_i.fill(0.0)

        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_ng_reordered[i, i_b].active
                and not self._is_particle_absorbed_reordered(i, i_b)
                and self.topology_valid[i, i_b]
            ):
                n = self.n_neighbors[i, i_b]
                constraint = gs.qd_float(0.0)
                grad_i = qd.Vector.zero(gs.qd_float, 3)
                for k in range(n):
                    k_next = 0 if k == n - 1 else k + 1
                    j = self.local_mesh_neighbors[i, i_b, k]
                    j_next = self.local_mesh_neighbors[i, i_b, k_next]
                    constraint += self._triangle_area(i, j, j_next, i_b)
                    grad_i += self._triangle_area_gradient(i, j, j_next, i_b)
                    self._surface_gradient[i, i_b, k] += self._triangle_area_gradient(j, j_next, i, i_b)
                    self._surface_gradient[i, i_b, k_next] += self._triangle_area_gradient(j_next, i, j, i_b)

                mass_i = self.particles_info_reordered[i, i_b].mass
                denominator = self._material.surface_tension_compliance / self._default_mass
                denominator += grad_i.norm_sqr() / mass_i
                for k in range(n):
                    j = self.local_mesh_neighbors[i, i_b, k]
                    denominator += (
                        self._surface_gradient[i, i_b, k].norm_sqr() / self.particles_info_reordered[j, i_b].mass
                    )
                lmd = gs.qd_float(0.0)
                if denominator > gs.EPS:
                    lmd = -constraint / denominator
                self._surface_lambda[i, i_b] = lmd
                self._surface_grad_i[i, i_b] = grad_i

        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_ng_reordered[i, i_b].active
                and not self._is_particle_absorbed_reordered(i, i_b)
                and self.topology_valid[i, i_b]
            ):
                lmd = self._surface_lambda[i, i_b]
                mass_i = self.particles_info_reordered[i, i_b].mass
                correction_i = lmd / mass_i * self._surface_grad_i[i, i_b]
                for axis in qd.static(range(3)):
                    qd.atomic_add(self.particles_reordered[i, i_b].dpos[axis], correction_i[axis])
                for k in range(self.n_neighbors[i, i_b]):
                    j = self.local_mesh_neighbors[i, i_b, k]
                    correction_j = lmd / self.particles_info_reordered[j, i_b].mass * self._surface_gradient[i, i_b, k]
                    for axis in qd.static(range(3)):
                        qd.atomic_add(self.particles_reordered[j, i_b].dpos[axis], correction_j[axis])

    @qd.func
    def _task_apply_distance_constraint(self, i, j, unused: qd.template(), i_b):
        if (
            i < j
            and not self._is_particle_absorbed_reordered(i, i_b)
            and not self._is_particle_absorbed_reordered(j, i_b)
            and self.on_surface[i, i_b] == self.on_surface[j, i_b]
            and not self._separated_by_static_colliders(
                i_b, self.particles_reordered[i, i_b].pos, self.particles_reordered[j, i_b].pos
            )
        ):
            pi = self.particles_reordered[i, i_b].pos
            pj = self.particles_reordered[j, i_b].pos
            delta = pi - pj
            distance = delta.norm()
            mass_i = self.particles_info_reordered[i, i_b].mass
            mass_j = self.particles_info_reordered[j, i_b].mass
            target = (
                self._particle_size
                * 0.5
                * (qd.pow(mass_i / self._default_mass, 1.0 / 1.5) + qd.pow(mass_j / self._default_mass, 1.0 / 1.5))
            )
            if distance > gs.EPS and distance < target:
                constraint = distance - target
                compliance = self._material.interior_distance_compliance
                if self.on_surface[i, i_b]:
                    compliance = self._material.surface_distance_compliance
                denominator = compliance / self._default_mass + 1.0 / mass_i + 1.0 / mass_j
                lmd = -constraint / denominator
                grad_i = delta / distance
                correction_i = lmd / mass_i * grad_i
                correction_j = -lmd / mass_j * grad_i
                for axis in qd.static(range(3)):
                    qd.atomic_add(self.particles_reordered[i, i_b].dpos[axis], correction_i[axis])
                    qd.atomic_add(self.particles_reordered[j, i_b].dpos[axis], correction_j[axis])

    @qd.kernel
    def _kernel_apply_distance_constraints(self):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i, i_b].active and not self._is_particle_absorbed_reordered(i, i_b):
                unused = gs.qd_int(0)
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    self._support_radius,
                    unused,
                    self._task_apply_distance_constraint,
                    i_b,
                )

    @qd.kernel
    def _kernel_apply_static_collider_adhesion(self):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_ng_reordered[i, i_b].active
                and not self._is_particle_absorbed_reordered(i, i_b)
                and self.on_surface[i, i_b]
            ):
                pos = self.particles_reordered[i, i_b].pos
                mass = self.particles_info_reordered[i, i_b].mass
                for collider_idx in qd.static(range(self._n_static_colliders)):
                    anchor, normal, _, surface_distance = query_static_collider_contact(
                        collider_idx,
                        i_b,
                        pos,
                        self._particle_radius,
                        self._static_colliders_pos,
                        self._static_colliders_quat,
                        self._static_colliders[collider_idx],
                    )
                    if surface_distance <= 2.0 * self._particle_radius:
                        anchor_delta = pos - anchor
                        if anchor_delta.dot(anchor_delta) <= self._particle_radius * self._particle_radius:
                            constraint = anchor_delta.dot(normal)
                            denominator = self._material.collider_adhesion_compliance / self._default_mass + 1.0 / mass
                            if denominator > gs.EPS:
                                self.particles_reordered[i, i_b].dpos += -constraint / denominator / mass * normal

    @qd.kernel
    def _kernel_apply_position_delta(self):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i, i_b].active and not self._is_particle_absorbed_reordered(i, i_b):
                pos = self.boundary.impose_pos(
                    self.particles_reordered[i, i_b].pos + self.particles_reordered[i, i_b].dpos
                )
                self.particles_reordered[i, i_b].pos = self._project_out_static_colliders(i_b, pos)

    # ------------------------------------------------------------------
    # Time integration and XSPH velocity filtering
    # ------------------------------------------------------------------

    @qd.kernel
    def _kernel_predict_positions(self, f: qd.i32, t: qd.f32):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i, i_b].active and not self._is_particle_absorbed_reordered(i, i_b):
                pos = self.particles_reordered[i, i_b].pos
                vel = self.particles_reordered[i, i_b].vel + self._substep_dt * self._gravity[i_b]
                for i_ff in qd.static(range(len(self._ffs))):
                    vel += self._substep_dt * self._ffs[i_ff].get_acc(pos, vel, t, i)
                self.particles_reordered[i, i_b].ipos = pos
                self.particles_reordered[i, i_b].vel = vel
                self.particles_reordered[i, i_b].pos = pos + self._substep_dt * vel

    @qd.kernel
    def _kernel_update_velocities_from_positions(self):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i, i_b].active:
                self.particles_reordered[i, i_b].vel = (
                    self.particles_reordered[i, i_b].pos - self.particles_reordered[i, i_b].ipos
                ) / self._substep_dt

    @qd.func
    def _task_viscosity(self, i, j, result: qd.template(), i_b):
        if (
            not self._is_particle_absorbed_reordered(j, i_b)
            and self.particles_reordered[i, i_b].surface == self.particles_reordered[j, i_b].surface
        ):
            density_j = self.particles_reordered[j, i_b].density
            if density_j > gs.EPS:
                distance = (self.particles_reordered[i, i_b].pos - self.particles_reordered[j, i_b].pos).norm()
                result += (
                    self.particles_info_reordered[j, i_b].mass
                    / density_j
                    * (self.particles_reordered[j, i_b].vel - self.particles_reordered[i, i_b].vel)
                    * self.cubic_kernel(distance)
                )

    @qd.kernel
    def _kernel_compute_viscosity(self):
        self.particles_reordered.dpos.fill(0.0)
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i, i_b].active and not self._is_particle_absorbed_reordered(i, i_b):
                delta_vel = qd.Vector.zero(gs.qd_float, 3)
                self.sh.for_all_neighbors(
                    i,
                    self.particles_reordered.pos,
                    self._support_radius,
                    delta_vel,
                    self._task_viscosity,
                    i_b,
                )
                coefficient = self._material.interior_viscosity
                if self.particles_reordered[i, i_b].surface:
                    coefficient = self._material.surface_viscosity
                self.particles_reordered[i, i_b].dpos = coefficient * delta_vel

    @qd.kernel
    def _kernel_apply_viscosity(self):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i, i_b].active and not self._is_particle_absorbed_reordered(i, i_b):
                self.particles_reordered[i, i_b].vel += self.particles_reordered[i, i_b].dpos
                if qd.static(self._material.is_collider_adhesion_friction_enabled):
                    if self.particles_reordered[i, i_b].surface:
                        pos = self.particles_reordered[i, i_b].pos
                        vel = self.particles_reordered[i, i_b].vel
                        for collider_idx in qd.static(range(self._n_static_colliders)):
                            _, normal, _, surface_distance = query_static_collider(
                                collider_idx,
                                i_b,
                                pos,
                                self._static_colliders_pos,
                                self._static_colliders_quat,
                                self._static_colliders[collider_idx],
                            )
                            if surface_distance <= self._particle_radius:
                                vel_normal = vel.dot(normal) * normal
                                vel_tangent = vel - vel_normal
                                vel = vel_normal + (1.0 - self._material.collider_friction) * vel_tangent
                        self.particles_reordered[i, i_b].vel = vel
                pos = self.boundary.impose_pos(
                    self.particles_reordered[i, i_b].ipos + self._substep_dt * self.particles_reordered[i, i_b].vel
                )
                self.particles_reordered[i, i_b].pos = self._project_out_static_colliders(i_b, pos)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self.entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        pass

    def substep_pre_coupling(self, f):
        if not self.is_active:
            return

        self._kernel_reorder_particles(f)
        if self._n_absorbent_static_colliders > 0:
            absorption_capture_budget = None
            if gs.use_zerocopy:
                absorption_capture_budget = qd_to_torch(self._absorption_capture_budget, copy=False)
            for absorption_idx, collider_idx in enumerate(self._absorbent_static_colliders_idx):
                budget_increment = self._static_colliders[collider_idx].absorption_rate * self._substep_dt
                # One carried token preserves fractional throughput and bounds the burst after an idle interval.
                budget_limit = budget_increment + 1.0
                if gs.use_zerocopy:
                    collider_capture_budget = absorption_capture_budget[absorption_idx]
                    collider_capture_budget.add_(budget_increment)
                    collider_capture_budget.clamp_(min=0.0, max=budget_limit)
                else:
                    pbstf_absorption.kernel_replenish_capture_budget(
                        absorption_idx,
                        budget_increment,
                        budget_limit,
                        self._absorption_capture_budget,
                    )
            if gs.use_zerocopy and gs.backend == gs.metal:
                torch.mps.synchronize()
            error_code = int(ErrorCode.INVALID_PBSTF_STATE_NAN)
            for collider_idx in self._absorbent_static_colliders_idx:
                collider = self._static_colliders[collider_idx]
                pbstf_absorption.kernel_update_absorbed_particles(
                    self._n_particles,
                    collider_idx,
                    self._substep_dt,
                    collider.absorption_rate,
                    self.particles_reordered,
                    self.particles_ng_reordered,
                    self._absorption_particles_reordered,
                    self._static_colliders_pos,
                    self._static_colliders_quat,
                    collider,
                    error_code,
                    self._errno,
                )
        self._kernel_predict_positions(f, self._sim.cur_t)
        if self._n_absorbent_static_colliders > 0:
            self._capture_absorbent_contacts()

        # The reference rebuilds its neighbor search after prediction.
        self._kernel_copy_from_reordered(f)
        self._kernel_reorder_particles(f)

        for iteration in range(self._max_solver_iterations):
            if iteration % self._topology_rebuild_interval == 0:
                if iteration > 0:
                    self._kernel_copy_from_reordered(f)
                    self._kernel_reorder_particles(f)
                self._rebuild_topology(f)

            # One Jacobi accumulation combines density and area constraints,
            # plus collision-distance constraints on even iterations. There is
            # intentionally no PBF artificial-pressure term.
            self._kernel_prepare_density_constraints()
            self._kernel_apply_density_constraints()
            self._kernel_apply_surface_constraints()
            if iteration % 2 == 0:
                self._kernel_apply_distance_constraints()
            if self._material.is_collider_adhesion_friction_enabled:
                self._kernel_apply_static_collider_adhesion()
            self._kernel_apply_position_delta()
            if self._n_absorbent_static_colliders > 0:
                self._capture_absorbent_contacts()

        self._kernel_update_velocities_from_positions()

        # XSPH uses a fresh final neighbor search, as in the CPU reference.
        self._kernel_copy_from_reordered(f)
        self._kernel_reorder_particles(f)
        self._kernel_compute_density(f)
        self._kernel_compute_viscosity()
        self._kernel_apply_viscosity()
        if self._n_absorbent_static_colliders > 0:
            self._capture_absorbent_contacts()
            self._rebuild_absorption_fields(self._absorption_particles_reordered)
        pbstf_absorption.kernel_check_fluid_state(
            self._n_particles,
            self.particles_reordered,
            self.particles_ng_reordered,
            int(ErrorCode.INVALID_PBSTF_STATE_NAN),
            self._errno,
        )

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if self.is_active:
            self._kernel_copy_from_reordered(f)

    def check_errno(self):
        errno = np.bitwise_or.reduce(qd_to_numpy(self._errno, transpose=True))
        if errno & ErrorCode.INVALID_PBSTF_DEFORMABLE_COLLIDER:
            gs.raise_exception(
                "PBSTF deformable collider state contains non-finite points, degenerate surface triangles, or invalid "
                "voxel search indices. Synchronize the collider from a valid FEM entity or restore a valid state."
            )
        if errno & ErrorCode.INVALID_PBSTF_STATE_NAN:
            gs.raise_exception(
                "PBSTF produced a non-finite fluid or absorption state. Increase compliance, reduce the time step, "
                "or increase the particle resolution."
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

    # ------------------------------------------------------------------
    # State, rendering and particle control
    # ------------------------------------------------------------------

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            pbstf_absorption.kernel_clear_errno(envs_idx, self._errno)
            self._kernel_set_state(f, envs_idx, state.pos, state.vel, state.active)
            if self._n_static_colliders > 0:
                pbstf_absorption.kernel_set_static_colliders_pose(
                    self._n_static_colliders,
                    envs_idx,
                    state.static_colliders_pos,
                    state.static_colliders_quat,
                    self._static_colliders_pos,
                    self._static_colliders_quat,
                )
            if self._n_deformable_static_colliders > 0:
                for collider in self._static_colliders:
                    if collider.is_deformable:
                        pbstf_absorption.kernel_set_deformable_collider_state(
                            collider.surface_vertex_state_start,
                            collider.voxel_state_start,
                            collider.voxel_search_order_state_start,
                            envs_idx,
                            state.deformable_static_colliders_surface_vertices,
                            state.deformable_static_colliders_voxel_positions,
                            state.deformable_static_colliders_voxel_search_order,
                            collider,
                        )
                        pbstf_absorption.kernel_check_deformable_collider_geometry(
                            collider, error_code=int(ErrorCode.INVALID_PBSTF_DEFORMABLE_COLLIDER), errno=self._errno,
                        )
                        if collider.has_sdf:
                            surface_state_end = collider.surface_vertex_state_start + collider.n_surface_vertices
                            surface_positions = state.deformable_static_colliders_surface_vertices[
                                envs_idx, collider.surface_vertex_state_start : surface_state_end
                            ]
                            self._set_deformable_collider_sdf(
                                collider,
                                envs_idx,
                                surface_positions,
                                state.is_deformable_static_colliders_sdf_active[
                                    envs_idx, collider.sdf_state_idx
                                ],
                            )
            if self._n_absorbent_static_colliders > 0:
                pbstf_absorption.kernel_set_absorption_capture_budget(
                    self._n_absorbent_static_colliders,
                    envs_idx,
                    state.absorption_capture_budget,
                    self._absorption_capture_budget,
                )
                pbstf_absorption.kernel_set_absorption_state(
                    self._n_particles,
                    envs_idx,
                    state.absorbed_collider_idx,
                    state.absorbed_voxel_idx,
                    state.absorption_voxel_distance,
                    state.absorption_local_pos,
                    state.absorption_target_local_pos,
                    state.absorption_progress,
                    self._absorption_particles,
                )
                self._rebuild_absorption_fields(self._absorption_particles)

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        envs_idx: qd.types.ndarray(),
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i, i_b_local in qd.ndrange(self._n_particles, envs_idx.shape[0]):
            i_b = envs_idx[i_b_local]
            for axis in qd.static(range(3)):
                self.particles[i, i_b].pos[axis] = pos[i_b, i, axis]
                self.particles[i, i_b].vel[axis] = vel[i_b, i, axis]
            self.particles_ng[i, i_b].active = active[i_b, i]

    def get_state(self, f):
        if not self.is_active:
            return None
        state = PBSTFSolverState(self.scene)
        self._kernel_get_state(f, state.pos, state.vel, state.active)
        if self._n_static_colliders > 0:
            pbstf_absorption.kernel_get_static_colliders_pose(
                self._n_static_colliders,
                self._static_colliders_pos,
                self._static_colliders_quat,
                state.static_colliders_pos,
                state.static_colliders_quat,
            )
        if self._n_deformable_static_colliders > 0:
            for collider in self._static_colliders:
                if collider.is_deformable:
                    pbstf_absorption.kernel_get_deformable_collider_geometry(
                        collider.surface_vertex_state_start,
                        collider.voxel_state_start,
                        collider.voxel_search_order_state_start,
                        state.deformable_static_colliders_surface_vertices,
                        state.deformable_static_colliders_voxel_positions,
                        state.deformable_static_colliders_voxel_search_order,
                        collider,
                    )
                    if collider.has_sdf:
                        if gs.use_zerocopy:
                            state.is_deformable_static_colliders_sdf_active[:, collider.sdf_state_idx] = qd_to_torch(
                                collider.is_sdf_active, transpose=True
                            )
                        else:
                            pbstf_absorption.kernel_get_deformable_collider_sdf_active(
                                collider.sdf_state_idx,
                                state.is_deformable_static_colliders_sdf_active,
                                collider,
                            )
        if self._n_absorbent_static_colliders > 0:
            pbstf_absorption.kernel_get_absorption_capture_budget(
                self._n_absorbent_static_colliders,
                self._absorption_capture_budget,
                state.absorption_capture_budget,
            )
            pbstf_absorption.kernel_get_absorption_state(
                self._n_particles,
                self._absorption_particles,
                state.absorbed_collider_idx,
                state.absorbed_voxel_idx,
                state.absorption_voxel_distance,
                state.absorption_local_pos,
                state.absorption_target_local_pos,
                state.absorption_progress,
            )
        return state

    @qd.kernel
    def _kernel_get_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            for axis in qd.static(range(3)):
                pos[i_b, i, axis] = self.particles[i, i_b].pos[axis]
                vel[i_b, i, axis] = self.particles[i, i_b].vel[axis]
            active[i_b, i] = self.particles_ng[i, i_b].active

    def update_render_fields(self):
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @qd.kernel
    def _kernel_update_render_fields(self, f: qd.i32):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i, i_b].active:
                self.particles_render[i, i_b].pos = self.particles[i, i_b].pos
                self.particles_render[i, i_b].vel = self.particles[i, i_b].vel
            else:
                self.particles_render[i, i_b].pos = gu.qd_nowhere()
            self.particles_render[i, i_b].active = self.particles_ng[i, i_b].active

    @qd.kernel
    def _kernel_add_particles(
        self,
        f: qd.i32,
        active: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        rho_rest: qd.f32,
        pos: qd.types.ndarray(),
    ):
        for i_, i_b in qd.ndrange(n_particles, self._B):
            i = i_ + particle_start
            self.particles_ng[i, i_b].active = qd.cast(active, gs.qd_bool)
            for axis in qd.static(range(3)):
                self.particles[i, i_b].pos[axis] = pos[i_, axis]
            self.particles[i, i_b].ipos = self.particles[i, i_b].pos
            self.particles[i, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
            self.particles[i, i_b].surface = False

        for i_ in range(n_particles):
            i = i_ + particle_start
            self.particles_info[i].mass = 1.0
            self.particles_info[i].rho_rest = rho_rest

    @qd.kernel
    def _kernel_set_particles_pos(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        poss: qd.types.ndarray(),
    ):
        for i_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i = particles_idx[i_b_, i_]
            i_b = envs_idx[i_b_]
            if qd.static(self._n_absorbent_static_colliders > 0):
                pbstf_absorption.unbind_particle(
                    i,
                    i_b,
                    self._n_absorption_voxels,
                    self._absorption_particles,
                    self._absorption_voxel_capacity,
                    self._absorption_voxel_occupancy,
                    self._absorption_voxel_wetness,
                )
            for axis in qd.static(range(3)):
                self.particles[i, i_b].pos[axis] = poss[i_b_, i_, axis]
            self.particles[i, i_b].vel = qd.Vector.zero(gs.qd_float, 3)

    @qd.kernel
    def _kernel_get_particles_pos(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        poss: qd.types.ndarray(),
    ):
        for i_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i = i_ + particle_start
            i_b = envs_idx[i_b_]
            for axis in qd.static(range(3)):
                poss[i_b_, i_, axis] = self.particles[i, i_b].pos[axis]

    @qd.kernel
    def _kernel_set_particles_vel(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),
    ):
        for i_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i = particles_idx[i_b_, i_]
            i_b = envs_idx[i_b_]
            if qd.static(self._n_absorbent_static_colliders > 0):
                pbstf_absorption.unbind_particle(
                    i,
                    i_b,
                    self._n_absorption_voxels,
                    self._absorption_particles,
                    self._absorption_voxel_capacity,
                    self._absorption_voxel_occupancy,
                    self._absorption_voxel_wetness,
                )
            for axis in qd.static(range(3)):
                self.particles[i, i_b].vel[axis] = vels[i_b_, i_, axis]

    @qd.kernel
    def _kernel_get_particles_vel(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),
    ):
        for i_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i = i_ + particle_start
            i_b = envs_idx[i_b_]
            for axis in qd.static(range(3)):
                vels[i_b_, i_, axis] = self.particles[i, i_b].vel[axis]

    @qd.kernel
    def _kernel_set_particles_active(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),
    ):
        for i_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i = particles_idx[i_b_, i_]
            i_b = envs_idx[i_b_]
            if qd.static(self._n_absorbent_static_colliders > 0):
                pbstf_absorption.unbind_particle(
                    i,
                    i_b,
                    self._n_absorption_voxels,
                    self._absorption_particles,
                    self._absorption_voxel_capacity,
                    self._absorption_voxel_occupancy,
                    self._absorption_voxel_wetness,
                )
            self.particles_ng[i, i_b].active = actives[i_b_, i_]

    @qd.kernel
    def _kernel_get_particles_active(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),
    ):
        for i_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i = i_ + particle_start
            i_b = envs_idx[i_b_]
            actives[i_b_, i_] = self.particles_ng[i, i_b].active

    @qd.kernel
    def _kernel_get_mass(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        mass: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
    ):
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            total = gs.qd_float(0.0)
            for i_ in range(n_particles):
                i = i_ + particle_start
                if self.particles_ng[i, i_b].active:
                    total += self.particles_info[i].mass
            mass[i_b_] = total

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
