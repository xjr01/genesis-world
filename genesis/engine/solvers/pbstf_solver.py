import math
from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import (
    CubeBoundary,
    create_static_collider,
    project_out_static_collider,
    query_static_collider,
    query_static_collider_contact,
    static_collider_separates,
)
from genesis.engine.entities import PBSTFEntity
from genesis.engine.states.solvers import PBSTFSolverState
from genesis.utils import particle
from genesis.utils.misc import (
    assign_indexed_tensor,
    broadcast_tensor,
    indices_to_mask,
    qd_to_numpy,
    qd_to_torch,
    sanitize_index,
)

from .base_solver import Solver

if TYPE_CHECKING:
    from genesis.engine.entities import PBSTFEntity


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
        self._static_colliders_pos = None
        self._static_colliders_quat = None
        self._upper_bound = np.asarray(options.upper_bound, dtype=gs.np_float)
        self._lower_bound = np.asarray(options.lower_bound, dtype=gs.np_float)

        self.sh = gu.SpatialHasher(cell_size=options.hash_grid_cell_size, grid_res=options._hash_grid_res)
        self.boundary = CubeBoundary(lower=self._lower_bound, upper=self._upper_bound)

        self._default_mass = 1.0
        self._material = None

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
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i, i_b].active:
                j = self.particles_ng[i, i_b].reordered_idx
                self.particles_reordered[j, i_b] = self.particles[i, i_b]
                self.particles_info_reordered[j, i_b] = self.particles_info[i]
                self.particles_ng_reordered[j, i_b].active = True

    @qd.kernel
    def _kernel_copy_from_reordered(self, f: qd.i32):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i, i_b].active:
                self.particles[i, i_b] = self.particles_reordered[self.particles_ng[i, i_b].reordered_idx, i_b]

    @qd.func
    def _task_density(self, i, j, result: qd.template(), i_b):
        if self.particles_ng_reordered[j, i_b].active:
            pos_i = self.particles_reordered[i, i_b].pos
            pos_j = self.particles_reordered[j, i_b].pos
            if not self._separated_by_static_colliders(i_b, pos_i, pos_j):
                distance = (pos_i - pos_j).norm()
                result += self.particles_info_reordered[j, i_b].mass * self.cubic_kernel(distance)

    @qd.kernel
    def _kernel_compute_density(self, f: qd.i32):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i, i_b].active:
                density = self.particles_info_reordered[i, i_b].mass * self.cubic_kernel(0.0)
                self.sh.for_all_neighbors(
                    i, self.particles_reordered.pos, self._support_radius, density, self._task_density, i_b
                )
                self.particles_reordered[i, i_b].density = density

    @qd.kernel
    def _kernel_reduce_max_density(self):
        for i in range(self._n_particles):
            if self.particles_ng_reordered[i, 0].active:
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
        if distance > gs.EPS and not self._separated_by_static_colliders(
            i_b, self.particles_reordered[i, i_b].pos, self.particles_reordered[j, i_b].pos
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
            if self.particles_ng_reordered[i, i_b].active:
                unused = gs.qd_int(0)
                self.sh.for_all_neighbors(
                    i, self.particles_reordered.pos, self._support_radius, unused, self._task_mark_screen, i_b
                )

    @qd.kernel
    def _kernel_classify_surface(self):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i, i_b].active:
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
            if self.particles_ng_reordered[i, i_b].active and self.on_surface[i, i_b]:
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
            if self.particles_ng_reordered[i, i_b].active and self.on_surface[i, i_b]:
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
                self.particles_reordered[i, i_b].surface = self.on_surface[i, i_b]

    @qd.func
    def _task_surface_covariance(self, i, j, unused: qd.template(), i_b):
        if self.on_surface[j, i_b]:
            delta = self.particles_reordered[j, i_b].pos - self.particles_reordered[i, i_b].pos
            self._pca_covariance[i, i_b] += delta.outer_product(delta)

    @qd.kernel
    def _kernel_mark_density_constraints(self):
        self._pca_covariance.fill(0.0)
        self.density_constraint_enabled.fill(False)
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i, i_b].active:
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
            if self.particles_ng_reordered[i, i_b].active:
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
        if self.on_surface[j, i_b] and not self._separated_by_static_colliders(
            i_b, self.particles_reordered[i, i_b].pos, self.particles_reordered[j, i_b].pos
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
            if self.particles_ng_reordered[i, i_b].active and self.on_surface[i, i_b]:
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
        if not self._separated_by_static_colliders(i_b, pos_i, pos_j):
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
            if self.particles_ng_reordered[i, i_b].active and self.density_constraint_enabled[i, i_b]:
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
        if not self._separated_by_static_colliders(i_b, pos_i, pos_j):
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
            if self.particles_ng_reordered[i, i_b].active and self.density_constraint_enabled[i, i_b]:
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
            if self.particles_ng_reordered[i, i_b].active and self.topology_valid[i, i_b]:
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
            if self.particles_ng_reordered[i, i_b].active and self.topology_valid[i, i_b]:
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
            if self.particles_ng_reordered[i, i_b].active:
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
            if self.particles_ng_reordered[i, i_b].active and self.on_surface[i, i_b]:
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
            if self.particles_ng_reordered[i, i_b].active:
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
            if self.particles_ng_reordered[i, i_b].active:
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
        if self.particles_reordered[i, i_b].surface == self.particles_reordered[j, i_b].surface:
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
            if self.particles_ng_reordered[i, i_b].active:
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
            if self.particles_ng_reordered[i, i_b].active:
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
        self._kernel_predict_positions(f, self._sim.cur_t)

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

        self._kernel_update_velocities_from_positions()

        # XSPH uses a fresh final neighbor search, as in the CPU reference.
        self._kernel_copy_from_reordered(f)
        self._kernel_reorder_particles(f)
        self._kernel_compute_density(f)
        self._kernel_compute_viscosity()
        self._kernel_apply_viscosity()

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if self.is_active:
            self._kernel_copy_from_reordered(f)

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
            self._kernel_set_state(f, state.pos, state.vel, state.active)

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i, i_b in qd.ndrange(self._n_particles, self._B):
            for axis in qd.static(range(3)):
                self.particles[i, i_b].pos[axis] = pos[i_b, i, axis]
                self.particles[i, i_b].vel[axis] = vel[i_b, i, axis]
            self.particles_ng[i, i_b].active = active[i_b, i]

    def get_state(self, f):
        if not self.is_active:
            return None
        state = PBSTFSolverState(self.scene)
        self._kernel_get_state(f, state.pos, state.vel, state.active)
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
