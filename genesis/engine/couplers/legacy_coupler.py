import math
from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.bvh import AABB, LBVH, kernel_remap_leaf_faces
from genesis.options.solvers import LegacyCouplerOptions
from genesis.repr_base import RBC
from genesis.utils import array_class
from genesis.utils.array_class import LinksState
from genesis.utils.geom import qd_inv_transform_by_trans_quat, qd_transform_by_trans_quat
import genesis.utils.sdf as sdf

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

CLAMPED_INV_DT = 50.0


@qd.kernel
def kernel_init_fem_rigid_surface_aabbs(
    faces_idx: qd.types.ndarray(ndim=1),
    surface_aabbs: qd.template(),
    dyn_info: array_class.DynInfo,
    surface_info: array_class.FEMRigidSurfaceInfo,
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
def kernel_reset_fem_rigid_surface_state(
    envs_idx: qd.types.ndarray(ndim=1),
    dyn_state: array_class.DynState,
    surface_state: array_class.FEMRigidSurfaceState,
    surface_info: array_class.FEMRigidSurfaceInfo,
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
def kernel_store_fem_rigid_surface_poses(
    dyn_state: array_class.DynState,
    surface_state: array_class.FEMRigidSurfaceState,
    surface_info: array_class.FEMRigidSurfaceInfo,
):
    for env_idx, surface_geom_slot in qd.ndrange(
        surface_state.previous_geoms_pos.shape[0], surface_info.surface_geoms_idx.shape[0]
    ):
        geom_idx = surface_info.surface_geoms_idx[surface_geom_slot]
        surface_state.previous_geoms_pos[env_idx, surface_geom_slot] = dyn_state.geoms.pos[geom_idx, env_idx]
        surface_state.previous_geoms_quat[env_idx, surface_geom_slot] = dyn_state.geoms.quat[geom_idx, env_idx]


@qd.data_oriented
class LegacyCoupler(RBC):
    """
    This class handles all the coupling between different solvers. LegacyCoupler will be deprecated in the future.
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, simulator: "Simulator", options: "LegacyCouplerOptions") -> None:
        self.sim = simulator
        self.options = options

        self.tool_solver = self.sim.tool_solver
        self.rigid_solver = self.sim.rigid_solver
        self.mpm_solver = self.sim.mpm_solver
        self.sph_solver = self.sim.sph_solver
        self.pbd_solver = self.sim.pbd_solver
        self.fem_solver = self.sim.fem_solver
        self.sf_solver = self.sim.sf_solver
        self.fem_projection_state = None
        self.fem_rigid_surface_info = None
        self.fem_rigid_surface_state = None
        self.fem_rigid_surface_bvh = None

    def build(self) -> None:
        self._rigid_mpm = self.rigid_solver.is_active and self.mpm_solver.is_active and self.options.rigid_mpm
        self._rigid_sph = self.rigid_solver.is_active and self.sph_solver.is_active and self.options.rigid_sph
        self._rigid_pbd = self.rigid_solver.is_active and self.pbd_solver.is_active and self.options.rigid_pbd
        self._rigid_fem = self.rigid_solver.is_active and self.fem_solver.is_active and self.options.rigid_fem
        self._mpm_sph = self.mpm_solver.is_active and self.sph_solver.is_active and self.options.mpm_sph
        self._mpm_pbd = self.mpm_solver.is_active and self.pbd_solver.is_active and self.options.mpm_pbd
        self._fem_mpm = self.fem_solver.is_active and self.mpm_solver.is_active and self.options.fem_mpm
        self._fem_sph = self.fem_solver.is_active and self.sph_solver.is_active and self.options.fem_sph

        self._is_implicit_fem_projection_enabled = (
            self._rigid_fem
            and self.fem_solver._use_implicit_solver
            and any(geom.needs_coup and not geom.is_coup_reaction_enabled for geom in self.rigid_solver.geoms)
        )
        self.fem_solver._is_implicit_rigid_projection_enabled = self._is_implicit_fem_projection_enabled
        if self._is_implicit_fem_projection_enabled:
            self.fem_projection_state = array_class.FEMProjectionState(
                normals=qd.Vector.field(
                    3,
                    dtype=gs.qd_float,
                    shape=(self.fem_solver._B, self.fem_solver.n_vertices),
                ),
                is_active=qd.field(
                    dtype=gs.qd_bool,
                    shape=(self.fem_solver._B, self.fem_solver.n_vertices),
                ),
                is_processed=qd.field(dtype=gs.qd_bool, shape=(self.fem_solver._B,)),
                has_changed=qd.field(dtype=gs.qd_int, shape=(self.fem_solver._B,)),
                has_contact=qd.field(dtype=gs.qd_int, shape=(self.fem_solver._B,)),
                is_pcg_active_saved=qd.field(dtype=gs.qd_bool, shape=(self.fem_solver._B,)),
            )

            projection_geoms = [
                geom for geom in self.rigid_solver.geoms if geom.needs_coup and not geom.is_coup_reaction_enabled
            ]
            surface_geoms = [geom for geom in projection_geoms if geom.n_faces > 0]
            if not surface_geoms:
                gs.raise_exception("One-way implicit FEM coupling requires collider surface triangles.")

            projection_geoms_idx = np.array([geom.idx for geom in projection_geoms], dtype=gs.np_int)
            surface_geoms_idx = np.array([geom.idx for geom in surface_geoms], dtype=gs.np_int)
            surface_geom_slots = np.full(self.rigid_solver.n_geoms, -1, dtype=gs.np_int)
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
            self.fem_rigid_surface_info = array_class.FEMRigidSurfaceInfo(
                projection_geoms_idx=projection_geoms_idx_qd,
                surface_geom_slots=surface_geom_slots_qd,
                surface_geoms_idx=surface_geoms_idx_qd,
                atlas_offsets=atlas_offsets_qd,
            )
            self.fem_rigid_surface_state = array_class.FEMRigidSurfaceState(
                corrections=qd.Vector.field(
                    3,
                    dtype=gs.qd_float,
                    shape=(self.fem_solver._B, self.fem_solver.n_vertices),
                ),
                n_corrections=qd.field(
                    dtype=gs.qd_int,
                    shape=(self.fem_solver._B, self.fem_solver.n_vertices),
                ),
                is_active=qd.field(dtype=gs.qd_bool, shape=(self.fem_solver._B,)),
                has_intersection=qd.field(dtype=gs.qd_int, shape=(self.fem_solver._B,)),
                previous_geoms_pos=qd.Vector.field(
                    3,
                    dtype=gs.qd_float,
                    shape=(self.fem_solver._B, len(surface_geoms)),
                ),
                previous_geoms_quat=qd.Vector.field(
                    4,
                    dtype=gs.qd_float,
                    shape=(self.fem_solver._B, len(surface_geoms)),
                ),
            )

            faces_idx = np.concatenate(
                tuple(np.arange(geom.face_start, geom.face_end, dtype=gs.np_int) for geom in surface_geoms)
            )
            surface_aabb = AABB(n_batches=1, n_aabbs=len(faces_idx))
            self.fem_rigid_surface_bvh = LBVH(surface_aabb, max_n_query_result_per_aabb=0)
            kernel_init_fem_rigid_surface_aabbs(
                faces_idx,
                self.fem_rigid_surface_bvh.aabbs,
                self.rigid_solver.dyn_info,
                self.fem_rigid_surface_info,
            )
            self.fem_rigid_surface_bvh.build()
            kernel_remap_leaf_faces(faces_idx, self.fem_rigid_surface_bvh.morton_codes)

        if (self._rigid_mpm or self._rigid_sph or self._rigid_pbd or self._rigid_fem) and any(
            geom.needs_coup for geom in self.rigid_solver.geoms
        ):
            self.rigid_solver.collider._sdf.activate()

        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            # this field stores the geom index of the thin shell rigid object (if any) that separates particle and its surrounding grid cell
            self.cpic_flag = qd.field(gs.qd_int, shape=(self.mpm_solver.n_particles, 3, 3, 3, self.mpm_solver._B))
            self.mpm_rigid_normal = qd.Vector.field(
                3,
                dtype=gs.qd_float,
                shape=(self.mpm_solver.n_particles, self.rigid_solver.n_geoms_, self.mpm_solver._B),
            )

        if self._rigid_sph:
            self.sph_rigid_normal = qd.Vector.field(
                3,
                dtype=gs.qd_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )
            self.sph_rigid_normal_reordered = qd.Vector.field(
                3,
                dtype=gs.qd_float,
                shape=(self.sph_solver.n_particles, self.rigid_solver.n_geoms_, self.sph_solver._B),
            )

        if self._rigid_pbd:
            self.pbd_rigid_normal_reordered = qd.Vector.field(
                3, dtype=gs.qd_float, shape=(self.pbd_solver.n_particles, self.pbd_solver._B, self.rigid_solver.n_geoms)
            )

            struct_particle_attach_info = qd.types.struct(link_idx=gs.qd_int, local_pos=gs.qd_vec3)

            self.particle_attach_info = struct_particle_attach_info.field(
                shape=(self.pbd_solver._n_particles, self.pbd_solver._B), layout=qd.Layout.SOA
            )
            self.particle_attach_info.link_idx.fill(-1)
            self.particle_attach_info.local_pos.fill(0.0)

        if self._mpm_sph:
            self.mpm_sph_stencil_size = int(np.floor(self.mpm_solver.dx / self.sph_solver.hash_grid_cell_size) + 2)

        if self._mpm_pbd:
            self.mpm_pbd_stencil_size = int(np.floor(self.mpm_solver.dx / self.pbd_solver.hash_grid_cell_size) + 2)

        ## DEBUG
        self._dx = 1 / 1024
        self._stencil_size = int(np.floor(self._dx / self.sph_solver.hash_grid_cell_size) + 2)

        self.reset(envs_idx=self.sim.scene._envs_idx)

    def reset(self, envs_idx=None) -> None:
        if self.fem_projection_state is not None:
            self.fem_projection_state.normals.fill(0.0)
            self.fem_projection_state.is_active.fill(False)
            self.fem_projection_state.is_processed.fill(False)
            self.fem_projection_state.has_changed.fill(0)
            self.fem_projection_state.has_contact.fill(0)
            self.fem_projection_state.is_pcg_active_saved.fill(False)

        if self.fem_rigid_surface_state is not None:
            if envs_idx is None:
                envs_idx = self.sim.scene._envs_idx
            kernel_reset_fem_rigid_surface_state(
                envs_idx,
                self.rigid_solver.dyn_state,
                self.fem_rigid_surface_state,
                self.fem_rigid_surface_info,
            )

        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            if envs_idx is None:
                self.mpm_rigid_normal.fill(0)
            else:
                self._kernel_reset_mpm(envs_idx)

        if self._rigid_sph:
            if envs_idx is None:
                self.sph_rigid_normal.fill(0)
            else:
                self._kernel_reset_sph(envs_idx)

    @qd.kernel
    def _kernel_reset_mpm(self, envs_idx: qd.types.ndarray()):
        for i_p, i_g, i_b_ in qd.ndrange(self.mpm_solver.n_particles, self.rigid_solver.n_geoms, envs_idx.shape[0]):
            self.mpm_rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0

    @qd.kernel
    def _kernel_reset_sph(self, envs_idx: qd.types.ndarray()):
        for i_p, i_g, i_b_ in qd.ndrange(self.sph_solver.n_particles, self.rigid_solver.n_geoms, envs_idx.shape[0]):
            self.sph_rigid_normal[i_p, i_g, envs_idx[i_b_]] = 0.0

    @qd.func
    def _func_collide_with_rigid(
        self,
        f,
        pos_world,
        vel,
        mass,
        i_b,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_info: array_class.RigidInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
        is_position_projection_enabled: qd.template(),
        is_imposed_boundary_projection_enabled: qd.template(),
    ):
        for i_g in range(self.rigid_solver.n_geoms):
            if geoms_info.needs_coup[i_g] and (
                not qd.static(is_imposed_boundary_projection_enabled) or geoms_info.is_coup_reaction_enabled[i_g]
            ):
                vel = self._func_collide_with_rigid_geom(
                    pos_world,
                    vel,
                    mass,
                    i_g,
                    i_b,
                    geoms_state=geoms_state,
                    geoms_info=geoms_info,
                    links_state=links_state,
                    rigid_info=rigid_info,
                    sdf_info=sdf_info,
                    collider_static_config=collider_static_config,
                    is_position_projection_enabled=is_position_projection_enabled,
                )
        return vel

    @qd.func
    def _func_collide_with_rigid_geom(
        self,
        pos_world,
        vel,
        mass,
        geom_idx,
        batch_idx,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_info: array_class.RigidInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
        is_position_projection_enabled: qd.template(),
    ):
        signed_dist = sdf.sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)

        if influence > 0.1:
            normal_rigid = sdf.sdf_func_normal_world(
                geom_idx, batch_idx, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_static_config
            )
            vel = self._func_collide_in_rigid_geom(
                pos_world, vel, mass, normal_rigid, influence, geom_idx, batch_idx, geoms_info, links_state, rigid_info
            )

        # Predicted-position projection recovers resting deformable vertices that enter the signed distance field.
        if qd.static(is_position_projection_enabled):
            vel_rigid = self.rigid_solver._func_vel_at_point(
                pos_world=pos_world,
                link_idx=geoms_info.link_idx[geom_idx],
                i_b=batch_idx,
                links_state=links_state,
            )
            predicted_pos = pos_world + rigid_info.substep_dt[None] * (vel - vel_rigid)
            predicted_signed_dist = sdf.sdf_func_world(
                geom_idx, batch_idx, predicted_pos, geoms_state, geoms_info, sdf_info
            )
            if predicted_signed_dist < 0:
                predicted_normal = sdf.sdf_func_normal_world(
                    geom_idx,
                    batch_idx,
                    predicted_pos,
                    geoms_state,
                    geoms_info,
                    rigid_info,
                    sdf_info,
                    collider_static_config,
                )
                corrected_pos = predicted_pos - predicted_signed_dist * predicted_normal
                vel_old = vel
                vel = vel_rigid + (corrected_pos - pos_world) / rigid_info.substep_dt[None]
                if geoms_info.is_coup_reaction_enabled[geom_idx]:
                    delta_mv = mass * (vel - vel_old)
                    force = -delta_mv / rigid_info.substep_dt[None]
                    self.rigid_solver._func_apply_coupling_force(
                        geoms_info.link_idx[geom_idx], batch_idx, predicted_pos, force, links_state
                    )

        return vel

    @qd.func
    def _func_collide_with_rigid_geom_robust(
        self,
        pos_world,
        vel,
        mass,
        pressure,
        normal_prev,
        geom_idx,
        batch_idx,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        links_info: array_class.LinksInfo,
        rigid_info: array_class.RigidInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        """Resolves Smoothed Particle Hydrodynamics (SPH) particle collisions with velocity response, predicted
        position projection, and entry-side normal preservation across thin walls.
        """
        signed_dist = sdf.sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)
        normal_rigid = sdf.sdf_func_normal_world(
            geom_idx, batch_idx, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_static_config
        )

        if normal_prev.dot(normal_prev) > gs.EPS and normal_rigid.dot(normal_prev) < 0:
            normal_rigid = normal_prev

        # bigger coup_softness implies that the coupling influence extends further away from the object.
        influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)

        if influence > 0.1:
            vel = self._func_collide_in_rigid_geom(
                pos_world, vel, mass, normal_rigid, influence, geom_idx, batch_idx, geoms_info, links_state, rigid_info
            )

        # Static fluid pressure pushes on the geom even at rest, where the velocity-gated collision response above
        # transfers nothing; this is what makes submerged geoms buoyant. Mirroring the particle pressure across the
        # surface and integrating the symmetric pressure force over the truncated kernel support yields the factor
        # 2 * sigma(signed_dist), with sigma the kernel plane integral (see cubic_kernel_plane_integral in
        # sph_solver.py). Sigma integrates to 1/2 across the support band, so a covering particle layer transmits
        # exactly p per unit area: Archimedes buoyancy with no tuning constant. Pressure feedback is limited to movable
        # links whose material enables coupling reactions; applying it to an imposed boundary creates a conservative
        # stiff kick that keeps resting fluid ringing through its acoustic pressure fluctuations.
        link_idx = geoms_info.link_idx[geom_idx]
        I_l = [link_idx, batch_idx] if qd.static(self.rigid_solver._options.batch_links_info) else link_idx
        if (
            signed_dist < self.sph_solver._support_radius
            and pressure > 0
            and not links_info.is_fixed[I_l]
            and geoms_info.is_coup_reaction_enabled[geom_idx]
        ):
            pressure_force = (
                -2.0
                * pressure
                * self.sph_solver._particle_volume
                * self.sph_solver.cubic_kernel_plane_integral(signed_dist)
                * normal_rigid
            )
            self.rigid_solver._func_apply_coupling_force(link_idx, batch_idx, pos_world, pressure_force, links_state)
            vel = vel - pressure_force * (rigid_info.substep_dt[None] / mass)

        predicted_pos = pos_world + rigid_info.substep_dt[None] * vel
        predicted_signed_dist = sdf.sdf_func_world(
            geom_idx, batch_idx, predicted_pos, geoms_state, geoms_info, sdf_info
        )
        predicted_normal = sdf.sdf_func_normal_world(
            geom_idx,
            batch_idx,
            predicted_pos,
            geoms_state,
            geoms_info,
            rigid_info,
            sdf_info,
            collider_static_config,
        )
        particle_radius = 0.5 * self.sph_solver.particle_size

        # A normal flip across one substep identifies a complete crossing between opposite exterior sides of a thin
        # wall. Resolve the velocity on the entry side before projecting the predicted particle center.
        if predicted_normal.dot(normal_rigid) < 0:
            vel_rigid = self.rigid_solver._func_vel_at_point(
                pos_world=pos_world,
                link_idx=geoms_info.link_idx[geom_idx],
                i_b=batch_idx,
                links_state=links_state,
            )
            rvel_normal_magnitude = (vel - vel_rigid).dot(normal_rigid)
            if rvel_normal_magnitude < 0:
                corrected_vel = vel - normal_rigid * rvel_normal_magnitude * (
                    1.0 + geoms_info.coup_restitution[geom_idx]
                )
                delta_mv = mass * (corrected_vel - vel)
                if geoms_info.is_coup_reaction_enabled[geom_idx]:
                    self.rigid_solver._func_apply_coupling_force(
                        link_idx=geoms_info.link_idx[geom_idx],
                        env_idx=batch_idx,
                        pos=pos_world,
                        force=-delta_mv / rigid_info.substep_dt[None],
                        links_state=links_state,
                    )
                vel = corrected_vel
        elif predicted_signed_dist < particle_radius:
            corrected_pos = predicted_pos + predicted_normal * (particle_radius - predicted_signed_dist)
            corrected_vel = (corrected_pos - pos_world) / rigid_info.substep_dt[None]
            delta_mv = mass * (corrected_vel - vel)
            if geoms_info.is_coup_reaction_enabled[geom_idx]:
                self.rigid_solver._func_apply_coupling_force(
                    link_idx=geoms_info.link_idx[geom_idx],
                    env_idx=batch_idx,
                    pos=predicted_pos,
                    force=-delta_mv / rigid_info.substep_dt[None],
                    links_state=links_state,
                )
            vel = corrected_vel
            normal_rigid = predicted_normal

        return vel, normal_rigid

    @qd.func
    def _func_collide_in_rigid_geom(
        self,
        pos_world,
        vel,
        mass,
        normal_rigid,
        influence,
        geom_idx,
        i_b,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_info: array_class.RigidInfo,
    ):
        """
        Resolves collision when a particle is already in collision with a rigid object.
        This function assumes known normal_rigid and influence.
        """
        vel_rigid = self.rigid_solver._func_vel_at_point(
            pos_world=pos_world, link_idx=geoms_info.link_idx[geom_idx], i_b=i_b, links_state=links_state
        )

        # v w.r.t rigid
        rvel = vel - vel_rigid
        rvel_normal_magnitude = rvel.dot(normal_rigid)  # negative if inward

        if rvel_normal_magnitude < 0:  # colliding
            #################### rigid -> particle ####################
            # tangential component
            rvel_tan = rvel - rvel_normal_magnitude * normal_rigid
            rvel_tan_norm = rvel_tan.norm(gs.EPS)

            # tangential component after friction
            rvel_tan = (
                rvel_tan
                / rvel_tan_norm
                * qd.max(0, rvel_tan_norm + rvel_normal_magnitude * geoms_info.coup_friction[geom_idx])
            )

            # normal component after collision
            rvel_normal = -normal_rigid * rvel_normal_magnitude * geoms_info.coup_restitution[geom_idx]

            # normal + tangential component
            rvel_new = rvel_tan + rvel_normal

            # apply influence
            vel_old = vel
            vel = vel_rigid + rvel_new * influence + rvel * (1 - influence)

            #################### particle -> rigid ####################
            # Compute delta momentum and apply to rigid body.
            if geoms_info.is_coup_reaction_enabled[geom_idx]:
                delta_mv = mass * (vel - vel_old)
                force = -delta_mv / rigid_info.substep_dt[None]
                self.rigid_solver._func_apply_coupling_force(
                    geoms_info.link_idx[geom_idx], i_b, pos_world, force, links_state
                )

        return vel

    @qd.func
    def _func_mpm_tool(self, f, pos_world, vel, i_b):
        for entity in qd.static(self.tool_solver.entities):
            if qd.static(entity.material.collision):
                vel = entity.collide(f, pos_world, vel, i_b)
        return vel

    @qd.kernel
    def mpm_grid_op(
        self,
        f: qd.i32,
        t: qd.f32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_info: array_class.RigidInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        for ii, jj, kk, i_b in qd.ndrange(*self.mpm_solver.grid_res, self.mpm_solver._B):
            I = (ii, jj, kk)
            if self.mpm_solver.grid[f, I, i_b].mass > gs.EPS:
                #################### MPM grid op ####################
                # Momentum to velocity
                vel_mpm = (1 / self.mpm_solver.grid[f, I, i_b].mass) * self.mpm_solver.grid[f, I, i_b].vel_in

                # gravity
                vel_mpm += self.mpm_solver.substep_dt * self.mpm_solver._gravity[i_b]

                pos = (I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                mass_mpm = self.mpm_solver.grid[f, I, i_b].mass / self.mpm_solver._particle_volume_scale

                # external force fields
                for i_ff in qd.static(range(len(self.mpm_solver._ffs))):
                    vel_mpm += self.mpm_solver._ffs[i_ff].get_acc(pos, vel_mpm, t, -1) * self.mpm_solver.substep_dt

                #################### MPM <-> Tool ####################
                if qd.static(self.tool_solver.is_active):
                    vel_mpm = self._func_mpm_tool(f, pos, vel_mpm, i_b)

                #################### MPM <-> Rigid ####################
                if qd.static(self._rigid_mpm):
                    vel_mpm = self._func_collide_with_rigid(
                        f,
                        pos,
                        vel_mpm,
                        mass_mpm,
                        i_b,
                        geoms_state=geoms_state,
                        geoms_info=geoms_info,
                        links_state=links_state,
                        rigid_info=rigid_info,
                        sdf_info=sdf_info,
                        collider_static_config=collider_static_config,
                        is_position_projection_enabled=False,
                        is_imposed_boundary_projection_enabled=False,
                    )

                #################### MPM <-> SPH ####################
                if qd.static(self._mpm_sph):
                    # using the lower corner of MPM cell to find the corresponding SPH base cell
                    base = self.sph_solver.sh.pos_to_grid(pos - 0.5 * self.mpm_solver.dx)

                    # ---------- SPH -> MPM ----------
                    sph_vel = qd.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in qd.grouped(
                        qd.ndrange(self.mpm_sph_stencil_size, self.mpm_sph_stencil_size, self.mpm_sph_stencil_size)
                    ):
                        slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                        for i in range(
                            self.sph_solver.sh.slot_start[slot_idx, i_b],
                            self.sph_solver.sh.slot_start[slot_idx, i_b] + self.sph_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if (
                                qd.abs(pos - self.sph_solver.particles_reordered.pos[i, i_b]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                sph_vel += self.sph_solver.particles_reordered.vel[i, i_b]
                                colliding_particles += 1
                    if colliding_particles > 0:
                        vel_old = vel_mpm
                        vel_mpm = sph_vel / colliding_particles

                        # ---------- MPM -> SPH ----------
                        delta_mv = mass_mpm * (vel_mpm - vel_old)

                        for offset in qd.grouped(
                            qd.ndrange(self.mpm_sph_stencil_size, self.mpm_sph_stencil_size, self.mpm_sph_stencil_size)
                        ):
                            slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                            for i in range(
                                self.sph_solver.sh.slot_start[slot_idx, i_b],
                                self.sph_solver.sh.slot_start[slot_idx, i_b]
                                + self.sph_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if (
                                    qd.abs(pos - self.sph_solver.particles_reordered.pos[i, i_b]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    self.sph_solver.particles_reordered[i, i_b].vel = (
                                        self.sph_solver.particles_reordered[i, i_b].vel
                                        - delta_mv / self.sph_solver.particles_info_reordered[i, i_b].mass
                                    )

                #################### MPM <-> PBD ####################
                if qd.static(self._mpm_pbd):
                    # using the lower corner of MPM cell to find the corresponding PBD base cell
                    base = self.pbd_solver.sh.pos_to_grid(pos - 0.5 * self.mpm_solver.dx)

                    # ---------- PBD -> MPM ----------
                    pbd_vel = qd.Vector([0.0, 0.0, 0.0])
                    colliding_particles = 0
                    for offset in qd.grouped(
                        qd.ndrange(self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size)
                    ):
                        slot_idx = self.pbd_solver.sh.grid_to_slot(base + offset)
                        for i in range(
                            self.pbd_solver.sh.slot_start[slot_idx, i_b],
                            self.pbd_solver.sh.slot_start[slot_idx, i_b] + self.pbd_solver.sh.slot_size[slot_idx, i_b],
                        ):
                            if (
                                qd.abs(pos - self.pbd_solver.particles_reordered.pos[i, i_b]).max()
                                < self.mpm_solver.dx * 0.5
                            ):
                                pbd_vel += self.pbd_solver.particles_reordered.vel[i, i_b]
                                colliding_particles += 1
                    if colliding_particles > 0:
                        vel_old = vel_mpm
                        vel_mpm = pbd_vel / colliding_particles

                        # ---------- MPM -> PBD ----------
                        delta_mv = mass_mpm * (vel_mpm - vel_old)

                        for offset in qd.grouped(
                            qd.ndrange(self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size, self.mpm_pbd_stencil_size)
                        ):
                            slot_idx = self.pbd_solver.sh.grid_to_slot(base + offset)
                            for i in range(
                                self.pbd_solver.sh.slot_start[slot_idx, i_b],
                                self.pbd_solver.sh.slot_start[slot_idx, i_b]
                                + self.pbd_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if (
                                    qd.abs(pos - self.pbd_solver.particles_reordered.pos[i, i_b]).max()
                                    < self.mpm_solver.dx * 0.5
                                ):
                                    if self.pbd_solver.particles_reordered[i, i_b].free:
                                        self.pbd_solver.particles_reordered[i, i_b].vel = (
                                            self.pbd_solver.particles_reordered[i, i_b].vel
                                            - delta_mv / self.pbd_solver.particles_info_reordered[i, i_b].mass
                                        )

                #################### MPM boundary ####################
                _, self.mpm_solver.grid[f, I, i_b].vel_out = self.mpm_solver.boundary.impose_pos_vel(pos, vel_mpm)

    @qd.kernel
    def mpm_surface_to_particle(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        sdf_info: array_class.SDFInfo,
        rigid_info: array_class.RigidInfo,
        collider_static_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self.mpm_solver.n_particles, self.mpm_solver._B):
            if self.mpm_solver.particles_ng[f, i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        sdf_normal = sdf.sdf_func_normal_world(
                            i_g,
                            i_b,
                            self.mpm_solver.particles[f, i_p, i_b].pos,
                            geoms_state,
                            geoms_info,
                            rigid_info,
                            sdf_info,
                            collider_static_config,
                        )
                        # we only update the normal if the particle does not the object
                        if sdf_normal.dot(self.mpm_rigid_normal[i_p, i_g, i_b]) >= 0:
                            self.mpm_rigid_normal[i_p, i_g, i_b] = sdf_normal

    def fem_rigid_link_constraints(self):
        if self.fem_solver._constraints_initialized and self.rigid_solver.is_active:
            self.fem_solver._kernel_update_linked_vertex_constraints(self.rigid_solver.dyn_state.links)

    def project_fem_implicit_pcg(self, f, is_initial):
        if is_initial:
            self.fem_solver.project_initial_pcg_positions(
                f,
                self.fem_rigid_surface_bvh.nodes,
                self.fem_rigid_surface_bvh.morton_codes,
                self.rigid_solver.dyn_state,
                self.fem_projection_state,
                self.rigid_solver.dyn_info,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider._collider_info,
                self.fem_rigid_surface_info,
            )
        else:
            self.fem_solver.one_projected_pcg_iter(
                f,
                self.fem_rigid_surface_bvh.nodes,
                self.fem_rigid_surface_bvh.morton_codes,
                self.rigid_solver.dyn_state,
                self.fem_projection_state,
                self.rigid_solver.dyn_info,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider._collider_info,
                self.fem_rigid_surface_info,
            )

    def project_fem_implicit_positions(self, f, is_committed):
        self.fem_solver.project_implicit_positions(
            f,
            self.fem_rigid_surface_bvh.nodes,
            self.fem_rigid_surface_bvh.morton_codes,
            self.rigid_solver.dyn_state,
            self.rigid_solver.dyn_info,
            self.rigid_solver.rigid_info,
            self.rigid_solver.collider._collider_info,
            self.fem_rigid_surface_info,
            is_committed=is_committed,
        )

    def project_fem_implicit_surface(self, f):
        self.fem_solver.project_implicit_surface(
            f,
            self.fem_rigid_surface_bvh.nodes,
            self.fem_rigid_surface_bvh.morton_codes,
            self.rigid_solver.dyn_state,
            self.fem_rigid_surface_state,
            self.rigid_solver.dyn_info,
            self.rigid_solver.rigid_info,
            self.rigid_solver.collider._collider_info,
            self.fem_rigid_surface_info,
            self.rigid_solver._errno,
        )
        kernel_store_fem_rigid_surface_poses(
            self.rigid_solver.dyn_state,
            self.fem_rigid_surface_state,
            self.fem_rigid_surface_info,
        )

    @qd.kernel
    def fem_surface_force(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        rigid_info: array_class.RigidInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        # TODO: all collisions are on vertices instead of surface and edge
        for i_s, i_b in qd.ndrange(self.fem_solver.n_surfaces, self.fem_solver._B):
            if self.fem_solver.surface[i_s].active:
                dt = self.fem_solver.substep_dt
                iel = self.fem_solver.surface[i_s].tri2el
                mass = self.fem_solver.elements_i[iel].mass_scaled / self.fem_solver.vol_scale

                p1 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[0], i_b].pos
                p2 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[1], i_b].pos
                p3 = self.fem_solver.elements_v[f, self.fem_solver.surface[i_s].tri2v[2], i_b].pos
                u = p2 - p1
                v = p3 - p1
                surface_normal = qd.math.cross(u, v)
                surface_normal = surface_normal / surface_normal.norm(gs.EPS)

                # FEM <-> Rigid
                if qd.static(self._rigid_fem):
                    # NOTE: collision only on surface vertices
                    for j in qd.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        vel_fem_sv = self._func_collide_with_rigid(
                            f,
                            self.fem_solver.elements_v[f, iv, i_b].pos,
                            self.fem_solver.elements_v[f + 1, iv, i_b].vel,
                            mass / 3.0,  # assume element mass uniformly distributed to vertices
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            rigid_info,
                            sdf_info,
                            collider_static_config,
                            is_position_projection_enabled=True,
                            is_imposed_boundary_projection_enabled=self._is_implicit_fem_projection_enabled,
                        )
                        self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

                # FEM <-> MPM (interact with MPM grid instead of particles)
                # NOTE: not doing this in mpm_grid_op otherwise we need to search for fem surface for each particles
                #       however, this function is called after mpm boundary conditions.
                if qd.static(self._fem_mpm):
                    for j in qd.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        pos = self.fem_solver.elements_v[f, iv, i_b].pos
                        vel_fem_sv = self.fem_solver.elements_v[f + 1, iv, i_b].vel
                        mass_fem_sv = mass / 4.0  # assume element mass uniformly distributed

                        # follow MPM p2g scheme
                        vel_mpm = qd.Vector([0.0, 0.0, 0.0])
                        mass_mpm = 0.0
                        mpm_base = qd.floor(pos * self.mpm_solver.inv_dx - 0.5).cast(gs.qd_int)
                        mpm_fx = pos * self.mpm_solver.inv_dx - mpm_base.cast(gs.qd_float)
                        mpm_w = [0.5 * (1.5 - mpm_fx) ** 2, 0.75 - (mpm_fx - 1.0) ** 2, 0.5 * (mpm_fx - 0.5) ** 2]
                        new_vel_fem_sv = vel_fem_sv
                        for mpm_offset in qd.static(qd.grouped(self.mpm_solver.stencil_range())):
                            mpm_grid_I = mpm_base - self.mpm_solver.grid_offset + mpm_offset
                            mpm_grid_mass = (
                                self.mpm_solver.grid[f, mpm_grid_I, i_b].mass / self.mpm_solver.particle_volume_scale
                            )

                            mpm_weight = gs.qd_float(1.0)
                            for d in qd.static(range(3)):
                                mpm_weight *= mpm_w[mpm_offset[d]][d]

                            # FEM -> MPM
                            mpm_grid_pos = (mpm_grid_I + self.mpm_solver.grid_offset) * self.mpm_solver.dx
                            signed_dist = (mpm_grid_pos - pos).dot(surface_normal)
                            if signed_dist <= self.mpm_solver.dx:  # NOTE: use dx as minimal unit for collision
                                vel_mpm_at_cell = mpm_weight * self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out
                                mass_mpm_at_cell = mpm_weight * mpm_grid_mass

                                vel_mpm += vel_mpm_at_cell
                                mass_mpm += mass_mpm_at_cell

                                if mass_mpm_at_cell > gs.EPS:
                                    delta_mpm_vel_at_cell_unmul = (
                                        vel_fem_sv * mpm_weight - self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out
                                    )
                                    mass_mul_at_cell = (
                                        mpm_grid_mass / mass_fem_sv
                                    )  # NOTE: use un-reweighted mass instead of mass_mpm_at_cell
                                    delta_mpm_vel_at_cell = delta_mpm_vel_at_cell_unmul * mass_mul_at_cell
                                    self.mpm_solver.grid[f, mpm_grid_I, i_b].vel_out += delta_mpm_vel_at_cell

                                    new_vel_fem_sv -= delta_mpm_vel_at_cell * mass_mpm_at_cell / mass_fem_sv

                        # MPM -> FEM
                        if mass_mpm > gs.EPS:
                            # delta_mv = (vel_mpm - vel_fem_sv) * mass_mpm
                            # delta_vel_fem_sv = delta_mv / mass_fem_sv
                            # self.fem_solver.elements_v[f + 1, iv].vel += delta_vel_fem_sv
                            self.fem_solver.elements_v[f + 1, iv, i_b].vel = new_vel_fem_sv

                # FEM <-> SPH TODO: this doesn't work well
                if qd.static(self._fem_sph):
                    for j in qd.static(range(3)):
                        iv = self.fem_solver.surface[i_s].tri2v[j]
                        pos = self.fem_solver.elements_v[f, iv, i_b].pos
                        vel_fem_sv = self.fem_solver.elements_v[f + 1, iv, i_b].vel
                        mass_fem_sv = mass / 4.0

                        dx = self.sph_solver.hash_grid_cell_size  # self._dx
                        stencil_size = 2  # self._stencil_size

                        base = self.sph_solver.sh.pos_to_grid(pos - 0.5 * dx)

                        # ---------- SPH -> FEM ----------
                        sph_vel = qd.Vector([0.0, 0.0, 0.0])
                        colliding_particles = 0
                        for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                            slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                            for k in range(
                                self.sph_solver.sh.slot_start[slot_idx, i_b],
                                self.sph_solver.sh.slot_start[slot_idx, i_b]
                                + self.sph_solver.sh.slot_size[slot_idx, i_b],
                            ):
                                if qd.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                    sph_vel += self.sph_solver.particles_reordered.vel[k, i_b]
                                    colliding_particles += 1

                        if colliding_particles > 0:
                            vel_old = vel_fem_sv
                            vel_fem_sv_unprojected = sph_vel / colliding_particles
                            vel_fem_sv = (
                                vel_fem_sv_unprojected.dot(surface_normal) * surface_normal
                            )  # exclude tangential velocity

                            # ---------- FEM -> SPH ----------
                            delta_mv = mass_fem_sv * (vel_fem_sv - vel_old)

                            for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                                slot_idx = self.sph_solver.sh.grid_to_slot(base + offset)
                                for k in range(
                                    self.sph_solver.sh.slot_start[slot_idx, i_b],
                                    self.sph_solver.sh.slot_start[slot_idx, i_b]
                                    + self.sph_solver.sh.slot_size[slot_idx, i_b],
                                ):
                                    if qd.abs(pos - self.sph_solver.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                        self.sph_solver.particles_reordered[k, i_b].vel = (
                                            self.sph_solver.particles_reordered[k, i_b].vel
                                            - delta_mv / self.sph_solver.particles_info_reordered[k, i_b].mass
                                        )

                            self.fem_solver.elements_v[f + 1, iv, i_b].vel = vel_fem_sv

                # boundary condition
                for j in qd.static(range(3)):
                    iv = self.fem_solver.surface[i_s].tri2v[j]
                    _, self.fem_solver.elements_v[f + 1, iv, i_b].vel = self.fem_solver.boundary.impose_pos_vel(
                        self.fem_solver.elements_v[f, iv, i_b].pos, self.fem_solver.elements_v[f + 1, iv, i_b].vel
                    )

    def fem_hydroelastic(self, f: qd.i32):
        # Floor contact

        # collision detection
        self.fem_solver.floor_hydroelastic_detection(f)

    @qd.kernel
    def sph_rigid(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        links_info: array_class.LinksInfo,
        rigid_info: array_class.RigidInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self.sph_solver._n_particles, self.sph_solver._B):
            if self.sph_solver.particles_ng_reordered[i_p, i_b].active:
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        (
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                        ) = self._func_collide_with_rigid_geom_robust(
                            self.sph_solver.particles_reordered[i_p, i_b].pos,
                            self.sph_solver.particles_reordered[i_p, i_b].vel,
                            self.sph_solver.particles_info_reordered[i_p, i_b].mass,
                            self.sph_solver.particles_reordered[i_p, i_b].p,
                            self.sph_rigid_normal_reordered[i_p, i_g, i_b],
                            i_g,
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            links_info,
                            rigid_info,
                            sdf_info,
                            collider_static_config,
                        )

    @qd.kernel
    def kernel_pbd_rigid_collide(
        self,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        sdf_info: array_class.SDFInfo,
        rigid_info: array_class.RigidInfo,
        collider_static_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self.pbd_solver._n_particles, self.sph_solver._B):
            if self.pbd_solver.particles_ng_reordered[i_p, i_b].active:
                # NOTE: Couldn't figure out a good way to handle collision with non-free particle. Such collision is not phsically plausible anyway.
                for i_g in range(self.rigid_solver.n_geoms):
                    if geoms_info.needs_coup[i_g]:
                        (
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                        ) = self._func_pbd_collide_with_rigid_geom(
                            i_p,
                            self.pbd_solver.particles_reordered[i_p, i_b].pos,
                            self.pbd_solver.particles_reordered[i_p, i_b].vel,
                            self.pbd_solver.particles_info_reordered[i_p, i_b].mass,
                            self.pbd_rigid_normal_reordered[i_p, i_b, i_g],
                            i_g,
                            i_b,
                            geoms_state,
                            geoms_info,
                            links_state,
                            sdf_info,
                            rigid_info,
                            collider_static_config,
                        )

    @qd.kernel
    def kernel_attach_pbd_to_rigid_link(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), link_idx: qd.i32, links_state: LinksState
    ) -> None:
        """
        Sets listed particles in listed environments to be animated by the link.

        Current position of the particle, relatively to the link, is stored and preserved.
        """
        pdb = self.pbd_solver

        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            link_pos = links_state.pos[link_idx, i_b]
            link_quat = links_state.quat[link_idx, i_b]

            # compute local offset from link to the particle
            world_pos = pdb.particles[i_p, i_b].pos
            local_pos = qd_inv_transform_by_trans_quat(world_pos, link_pos, link_quat)

            # set particle to be animated (not free) and store animation info
            pdb.particles[i_p, i_b].free = False
            self.particle_attach_info[i_p, i_b].link_idx = link_idx
            self.particle_attach_info[i_p, i_b].local_pos = local_pos

    @qd.kernel
    def kernel_pbd_rigid_clear_animate_particles_by_link(
        self, particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray()
    ) -> None:
        """Detach listed particles from links, and simulate them freely."""
        pdb = self.pbd_solver
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            pdb.particles[i_p, i_b].free = True
            self.particle_attach_info[i_p, i_b].link_idx = -1
            self.particle_attach_info[i_p, i_b].local_pos = qd.math.vec3([0.0, 0.0, 0.0])

    @qd.kernel
    def kernel_pbd_rigid_solve_animate_particles_by_link(self, clamped_inv_dt: qd.f32, links_state: LinksState):
        """
        Itearates all particles and environments, and sets corrective velocity for all animated particle.

        Computes target position and velocity from the attachment/reference link and local offset position.

        Note, that this step shoudl be done after rigid solver update, and before PDB solver update.
        Currently, this is done after both rigid and PBD solver updates, hence the corrective velocity
        is off by a frame.

        Note, it's adviced to clamp inv_dt to avoid large jerks and instability. 1/0.02 might be a good max value.
        """
        pdb = self.pbd_solver
        for i_p, i_env in qd.ndrange(pdb._n_particles, pdb._B):
            if self.particle_attach_info[i_p, i_env].link_idx >= 0:
                # read link state
                link_idx = self.particle_attach_info[i_p, i_env].link_idx
                link_pos = links_state.pos[link_idx, i_env]
                link_quat = links_state.quat[link_idx, i_env]

                link_lin_vel = links_state.cd_vel[link_idx, i_env]
                link_ang_vel = links_state.cd_ang[link_idx, i_env]
                link_com_in_world = links_state.root_COM[link_idx, i_env] + links_state.i_pos[link_idx, i_env]

                # calculate target pos and vel of the particle
                local_pos = self.particle_attach_info[i_p, i_env].local_pos
                target_world_pos = qd_transform_by_trans_quat(local_pos, link_pos, link_quat)

                world_arm = target_world_pos - link_com_in_world
                target_world_vel = link_lin_vel + link_ang_vel.cross(world_arm)

                # compute and apply corrective velocity
                i_rp = pdb.particles_ng[i_p, i_env].reordered_idx
                particle_pos = pdb.particles_reordered[i_rp, i_env].pos
                pos_correction = target_world_pos - particle_pos
                corrective_vel = pos_correction * clamped_inv_dt
                pdb.particles_reordered[i_rp, i_env].vel = corrective_vel + target_world_vel

    @qd.func
    def _func_pbd_collide_with_rigid_geom(
        self,
        i,
        pos_world,
        vel,
        mass,
        normal_prev,
        geom_idx,
        batch_idx,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        sdf_info: array_class.SDFInfo,
        rigid_info: array_class.RigidInfo,
        collider_static_config: qd.template(),
    ):
        """
        Resolves collision when a particle is already in collision with a rigid object.
        This function assumes known normal_rigid and influence.
        """
        signed_dist = sdf.sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)
        contact_normal = sdf.sdf_func_normal_world(
            geom_idx, batch_idx, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_static_config
        )
        new_pos = pos_world
        new_vel = vel
        if signed_dist < self.pbd_solver.particle_size / 2:  # skip non-penetration particles
            stiffness = 1.0  # value in [0, 1]

            # we don't consider friction for now
            # friction = 0.15
            # vel_rigid = self.rigid_solver._func_vel_at_point(
            #     pos_world=pos_world,
            #     link_idx=geoms_info.link_idx[geom_idx],
            #     i_b=batch_idx,
            #     links_state=links_state,
            # )
            # rvel = vel - vel_rigid
            # rvel_normal_magnitude = rvel.dot(contact_normal)  # negative if inward
            # rvel_tan = rvel - rvel_normal_magnitude * contact_normal
            # rvel_tan_norm = rvel_tan.norm(gs.EPS)

            #################### rigid -> particle ####################

            energy_loss = 0.0  # value in [0, 1]
            new_pos = pos_world + stiffness * contact_normal * (self.pbd_solver.particle_size / 2 - signed_dist)
            prev_pos = self.pbd_solver.particles_reordered[i, batch_idx].ipos
            new_vel = (new_pos - prev_pos) / self.pbd_solver._substep_dt

            #################### particle -> rigid ####################
            if geoms_info.is_coup_reaction_enabled[geom_idx]:
                delta_mv = mass * (new_vel - vel)
                force = (-delta_mv / self.rigid_solver._substep_dt) * (1 - energy_loss)
                self.rigid_solver._func_apply_coupling_force(
                    geoms_info.link_idx[geom_idx], batch_idx, pos_world, force, links_state
                )

        return new_pos, new_vel, contact_normal

    def preprocess(self, f):
        # Implicit finite element method (FEM) constraints participate in inertia assembly before the coupling phase.
        if self.fem_solver.is_active and self.fem_solver._use_implicit_solver:
            self.fem_rigid_link_constraints()

        # preprocess for MPM CPIC
        if self._rigid_mpm and self.mpm_solver.enable_CPIC:
            self.mpm_surface_to_particle(
                f,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.collider._sdf._sdf_info,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider._collider_static_config,
            )

    def couple(self, f):
        # MPM <-> all others
        if self.mpm_solver.is_active:
            self.mpm_grid_op(
                f,
                self.sim.cur_t,
                geoms_state=self.rigid_solver.dyn_state.geoms,
                geoms_info=self.rigid_solver.dyn_info.geoms,
                links_state=self.rigid_solver.dyn_state.links,
                rigid_info=self.rigid_solver.rigid_info,
                sdf_info=self.rigid_solver.collider._sdf._sdf_info,
                collider_static_config=self.rigid_solver.collider._collider_static_config,
            )

        # SPH <-> Rigid
        if self._rigid_sph:
            self.sph_rigid(
                f,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.dyn_state.links,
                self.rigid_solver.dyn_info.links,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider._sdf._sdf_info,
                self.rigid_solver.collider._collider_static_config,
            )

        # PBD <-> Rigid
        if self._rigid_pbd:
            self.kernel_pbd_rigid_collide(
                geoms_state=self.rigid_solver.dyn_state.geoms,
                geoms_info=self.rigid_solver.dyn_info.geoms,
                links_state=self.rigid_solver.dyn_state.links,
                sdf_info=self.rigid_solver.collider._sdf._sdf_info,
                rigid_info=self.rigid_solver.rigid_info,
                collider_static_config=self.rigid_solver.collider._collider_static_config,
            )

            # 1-way: animate particles by links
            full_step_inv_dt = 1.0 / self.pbd_solver._dt
            clamped_inv_dt = min(full_step_inv_dt, CLAMPED_INV_DT)
            self.kernel_pbd_rigid_solve_animate_particles_by_link(clamped_inv_dt, self.rigid_solver.dyn_state.links)

        if self.fem_solver.is_active:
            self.fem_surface_force(
                f,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.dyn_state.links,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider._sdf._sdf_info,
                self.rigid_solver.collider._collider_static_config,
            )
            self.fem_rigid_link_constraints()

    def couple_grad(self, f):
        if self.fem_solver.is_active:
            self.fem_surface_force.grad(
                f,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.dyn_state.links,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider._sdf._sdf_info,
                self.rigid_solver.collider._collider_static_config,
            )
        if self.mpm_solver.is_active:
            self.mpm_grid_op.grad(
                f,
                self.sim.cur_t,
                geoms_state=self.rigid_solver.dyn_state.geoms,
                geoms_info=self.rigid_solver.dyn_info.geoms,
                links_state=self.rigid_solver.dyn_state.links,
                rigid_info=self.rigid_solver.rigid_info,
                sdf_info=self.rigid_solver.collider._sdf._sdf_info,
                collider_static_config=self.rigid_solver.collider._collider_static_config,
            )

    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers
