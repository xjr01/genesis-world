from dataclasses import dataclass

import numpy as np
import torch

import quadrants as qd

import genesis as gs
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.boundaries.rigid_surface import (
    build_rigid_surface,
    kernel_apply_pbd_rigid_surface_corrections,
    kernel_detect_pbd_rigid_surface_intersections,
    kernel_reset_rigid_surface_contact,
    kernel_store_rigid_surface_poses,
)
from genesis.engine.entities.pbd_entity import PBD2DEntity, PBD3DEntity
from genesis.engine.states.solvers import PBDSolverState
from genesis.utils import array_class, geom, sdf
from genesis.utils.array_class import V_ANNOTATION, ErrorCode
from genesis.utils.misc import indices_to_mask, qd_to_numpy, qd_to_torch

from .base_solver import Solver, StateChange, mutates


@dataclass(frozen=True)
class PBDConstraintResiduals:
    """Signed relative edge lengths, dihedral angles in radians, and relative tetrahedron volumes."""

    stretch: torch.Tensor
    bending: torch.Tensor
    volume: torch.Tensor


class PBDUnifiedSolverState(PBDSolverState):
    def __init__(self, scene):
        super().__init__(scene)
        self.active = None
        self.previous_geoms_pos = None
        self.previous_geoms_quat = None


@qd.func
def func_dihedral(p1, p2, p3, p4):
    edge = p2 - p1
    normal1 = edge.cross(p3 - p1)
    normal2 = edge.cross(p4 - p1)
    length = edge.norm()
    normal1_sqr = normal1.norm_sqr()
    normal2_sqr = normal2.norm_sqr()
    angle = gs.qd_float(0.0)
    gradients = qd.Matrix.zero(gs.qd_float, 3, 4)
    is_valid = (
        length > 0.0
        and normal1_sqr > gs.EPS**2 * edge.norm_sqr() * (p3 - p1).norm_sqr()
        and normal2_sqr > gs.EPS**2 * edge.norm_sqr() * (p4 - p1).norm_sqr()
    )
    if is_valid:
        angle = qd.atan2(edge.dot(normal1.cross(normal2)) / length, normal1.dot(normal2))
        gradients[:, 2] = -length / normal1_sqr * normal1
        gradients[:, 3] = length / normal2_sqr * normal2
        gradients[:, 1] = (
            -edge.dot(p3 - p1) / length**2 * gradients[:, 2] - edge.dot(p4 - p1) / length**2 * gradients[:, 3]
        )
        gradients[:, 0] = -gradients[:, 1] - gradients[:, 2] - gradients[:, 3]
    return angle, gradients, is_valid


@qd.kernel
def kernel_init_bending(particles_info: V_ANNOTATION, inner_edges_info: V_ANNOTATION, errno: qd.Tensor):
    for i_c in range(inner_edges_info.shape[0]):
        info = inner_edges_info[i_c]
        angle, _, is_valid = func_dihedral(
            particles_info[info.v1].pos_rest,
            particles_info[info.v2].pos_rest,
            particles_info[info.v3].pos_rest,
            particles_info[info.v4].pos_rest,
        )
        inner_edges_info[i_c].angle_rest = angle
        if not is_valid:
            qd.atomic_or(errno[0], ErrorCode.INVALID_PBD_STATE)


@qd.kernel
def kernel_predict(
    time: float,
    dt: float,
    particles: V_ANNOTATION,
    particles_ng: V_ANNOTATION,
    particles_info: V_ANNOTATION,
    gravity: qd.Tensor,
    solver: V_ANNOTATION,
    errno: qd.Tensor,
):
    for i_p, i_b in qd.ndrange(particles.shape[0], particles.shape[1]):
        particles[i_p, i_b].ipos = particles[i_p, i_b].pos
        if particles_ng[i_p, i_b].active:
            velocity = particles[i_p, i_b].vel
            if particles[i_p, i_b].free:
                acceleration = gravity[i_b]
                for i_ff in qd.static(range(len(solver._ffs))):
                    acceleration += solver._ffs[i_ff].get_acc(particles[i_p, i_b].pos, velocity, time, i_p)
                acceleration -= (
                    particles_info[i_p].air_resistance / particles_info[i_p].mass * velocity.norm() * velocity
                )
                velocity += dt * acceleration
            particles[i_p, i_b].pos += dt * velocity
            particles[i_p, i_b].vel = velocity
            if qd.math.isnan(velocity).any() or qd.math.isinf(velocity).any():
                qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_STATE)
        particles[i_p, i_b].pos_iter = particles[i_p, i_b].pos


@qd.kernel
def kernel_accumulate_stretch(
    dt: float,
    particles: V_ANNOTATION,
    particles_ng: V_ANNOTATION,
    particles_info: V_ANNOTATION,
    edges_info: V_ANNOTATION,
):
    for i_c, i_b in qd.ndrange(edges_info.shape[0], particles.shape[1]):
        info = edges_info[i_c]
        i1, i2 = info.v1, info.v2
        if particles_ng[i1, i_b].active and particles_ng[i2, i_b].active:
            w1 = particles[i1, i_b].free / particles_info[i1].mass
            w2 = particles[i2, i_b].free / particles_info[i2].mass
            edge = particles[i1, i_b].pos - particles[i2, i_b].pos
            length = edge.norm()
            if w1 + w2 > 0.0:
                direction = geom.qd_normalize(edge, gs.EPS)
                if length <= gs.EPS:
                    direction = geom.qd_normalize(particles_info[i1].pos_rest - particles_info[i2].pos_rest, gs.EPS)
                correction = (
                    -info.stretch_relaxation
                    * (length - info.len_rest)
                    / (w1 + w2 + info.stretch_compliance / dt**2)
                    * direction
                )
                particles[i1, i_b].dpos += w1 * correction
                particles[i2, i_b].dpos -= w2 * correction


@qd.kernel
def kernel_accumulate_volume(
    dt: float,
    particles: V_ANNOTATION,
    particles_ng: V_ANNOTATION,
    particles_info: V_ANNOTATION,
    elems_info: V_ANNOTATION,
):
    for i_c, i_b in qd.ndrange(elems_info.shape[0], particles.shape[1]):
        info = elems_info[i_c]
        vertices = qd.Vector([info.v1, info.v2, info.v3, info.v4])
        points = qd.Matrix.zero(gs.qd_float, 3, 4)
        weights = qd.Vector.zero(gs.qd_float, 4)
        is_active = True
        for i_v in qd.static(range(4)):
            i_p = vertices[i_v]
            points[:, i_v] = particles[i_p, i_b].pos
            weights[i_v] = particles[i_p, i_b].free / particles_info[i_p].mass
            is_active = is_active and particles_ng[i_p, i_b].active
        if is_active:
            p1, p2, p3, p4 = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
            gradients = qd.Matrix.cols(
                [
                    (p4 - p2).cross(p3 - p2) / 6.0,
                    (p3 - p1).cross(p4 - p1) / 6.0,
                    (p4 - p1).cross(p2 - p1) / 6.0,
                    (p2 - p1).cross(p3 - p1) / 6.0,
                ]
            )
            denominator = gs.qd_float(0.0)
            for i_v in qd.static(range(4)):
                denominator += weights[i_v] * gradients[:, i_v].norm_sqr()
            if denominator > 0.0:
                residual = geom.qd_tet_vol(p1, p2, p3, p4) - info.vol_rest
                multiplier = -info.volume_relaxation * residual / (denominator + info.volume_compliance / dt**2)
                for i_v in qd.static(range(4)):
                    particles[vertices[i_v], i_b].dpos += multiplier * weights[i_v] * gradients[:, i_v]


@qd.kernel
def kernel_accumulate_bending(
    dt: float,
    particles: V_ANNOTATION,
    particles_ng: V_ANNOTATION,
    particles_info: V_ANNOTATION,
    inner_edges_info: V_ANNOTATION,
    errno: qd.Tensor,
):
    for i_c, i_b in qd.ndrange(inner_edges_info.shape[0], particles.shape[1]):
        info = inner_edges_info[i_c]
        vertices = qd.Vector([info.v1, info.v2, info.v3, info.v4])
        points = qd.Matrix.zero(gs.qd_float, 3, 4)
        weights = qd.Vector.zero(gs.qd_float, 4)
        is_active = True
        for i_v in qd.static(range(4)):
            i_p = vertices[i_v]
            points[:, i_v] = particles[i_p, i_b].pos
            weights[i_v] = particles[i_p, i_b].free / particles_info[i_p].mass
            is_active = is_active and particles_ng[i_p, i_b].active
        if is_active and weights.sum() > 0.0:
            angle, gradients, is_valid = func_dihedral(points[:, 0], points[:, 1], points[:, 2], points[:, 3])
            if is_valid:
                residual = qd.atan2(qd.sin(angle - info.angle_rest), qd.cos(angle - info.angle_rest))
                denominator = gs.qd_float(0.0)
                for i_v in qd.static(range(4)):
                    denominator += weights[i_v] * gradients[:, i_v].norm_sqr()
                if denominator > 0.0:
                    multiplier = -info.bending_relaxation * residual / (denominator + info.bending_compliance / dt**2)
                    for i_v in qd.static(range(4)):
                        particles[vertices[i_v], i_b].dpos += multiplier * weights[i_v] * gradients[:, i_v]
            else:
                qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_STATE)


@qd.kernel
def kernel_apply_delta(acceleration: float, particles: V_ANNOTATION, particles_ng: V_ANNOTATION, errno: qd.Tensor):
    for i_p, i_b in qd.ndrange(particles.shape[0], particles.shape[1]):
        if particles_ng[i_p, i_b].active and particles[i_p, i_b].free:
            previous = particles[i_p, i_b].pos_iter
            particles[i_p, i_b].pos_iter = particles[i_p, i_b].pos
            pos = (
                particles[i_p, i_b].pos + particles[i_p, i_b].dpos + acceleration * (particles[i_p, i_b].pos - previous)
            )
            if qd.math.isnan(pos).any() or qd.math.isinf(pos).any():
                qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_STATE)
            else:
                particles[i_p, i_b].pos = pos
        particles[i_p, i_b].dpos.fill(0.0)


@qd.kernel
def kernel_project_vertices(
    i_iteration: int,
    boundary: V_ANNOTATION,
    particles: V_ANNOTATION,
    particles_ng: V_ANNOTATION,
    bvh_nodes: V_ANNOTATION,
    bvh_morton_codes: V_ANNOTATION,
    collision_iterations: V_ANNOTATION,
    dyn_state: array_class.DynState,
    surface_state: array_class.RigidSurfaceContactState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    surface_info: array_class.RigidSurfaceInfo,
    is_audit: V_ANNOTATION,
    errno: qd.Tensor,
):
    for i_p, i_b in qd.ndrange(particles.shape[0], particles.shape[1]):
        if not qd.static(is_audit) and not surface_state.is_active[i_b]:
            continue
        if particles_ng[i_p, i_b].active:
            pos = particles[i_p, i_b].pos
            projected = boundary.impose_pos(pos)
            for i_g in range(surface_info.projection_geoms_idx.shape[0]):
                geom_idx = surface_info.projection_geoms_idx[i_g]
                projected, _, _ = sdf.sdf_func_project_vertex_outside_geom(
                    geom_idx,
                    i_b,
                    projected,
                    bvh_nodes,
                    bvh_morton_codes,
                    dyn_state,
                    dyn_info,
                    rigid_info,
                    collider_info,
                    surface_info,
                )
            if (projected - pos).norm_sqr() > gs.EPS**2:
                if qd.static(is_audit):
                    qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_RIGID_SURFACE_INTERSECTION)
                elif particles[i_p, i_b].free:
                    particles[i_p, i_b].pos = projected
                    qd.atomic_max(surface_state.has_intersection[i_b], 1)
                else:
                    qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_RIGID_SURFACE_INTERSECTION)
    if not qd.static(is_audit):
        for i_b in range(particles.shape[1]):
            if surface_state.is_active[i_b]:
                collision_iterations[i_b, i_iteration] += 1


@qd.kernel
def kernel_project_boundary(boundary: V_ANNOTATION, particles: V_ANNOTATION, particles_ng: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(particles.shape[0], particles.shape[1]):
        if particles_ng[i_p, i_b].active and particles[i_p, i_b].free:
            particles[i_p, i_b].pos = boundary.impose_pos(particles[i_p, i_b].pos)


@qd.kernel
def kernel_update_velocity(dt: float, particles: V_ANNOTATION, particles_ng: V_ANNOTATION, errno: qd.Tensor):
    for i_p, i_b in qd.ndrange(particles.shape[0], particles.shape[1]):
        if particles_ng[i_p, i_b].active:
            velocity = (particles[i_p, i_b].pos - particles[i_p, i_b].ipos) / dt
            if qd.math.isnan(velocity).any() or qd.math.isinf(velocity).any():
                qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_STATE)
            else:
                particles[i_p, i_b].vel = velocity


@qd.kernel
def kernel_check_volume(
    particles: V_ANNOTATION, particles_ng: V_ANNOTATION, elems_info: V_ANNOTATION, errno: qd.Tensor
):
    for i_c, i_b in qd.ndrange(elems_info.shape[0], particles.shape[1]):
        info = elems_info[i_c]
        if (
            particles_ng[info.v1, i_b].active
            and particles_ng[info.v2, i_b].active
            and particles_ng[info.v3, i_b].active
            and particles_ng[info.v4, i_b].active
        ):
            volume = geom.qd_tet_vol(
                particles[info.v1, i_b].pos,
                particles[info.v2, i_b].pos,
                particles[info.v3, i_b].pos,
                particles[info.v4, i_b].pos,
            )
            if volume * info.vol_rest <= 0.0 and not (
                particles[info.v1, i_b].free
                or particles[info.v2, i_b].free
                or particles[info.v3, i_b].free
                or particles[info.v4, i_b].free
            ):
                qd.atomic_or(errno[i_b], ErrorCode.INVALID_PBD_VOLUME)


@qd.kernel
def kernel_update_render(
    particles: V_ANNOTATION,
    particles_ng: V_ANNOTATION,
    particles_render: V_ANNOTATION,
    vverts_render: V_ANNOTATION,
    vverts_info: V_ANNOTATION,
):
    for i_p, i_b in qd.ndrange(particles.shape[0], particles.shape[1]):
        particles_render[i_p, i_b].pos = geom.qd_nowhere()
        if particles_ng[i_p, i_b].active:
            particles_render[i_p, i_b].pos = particles[i_p, i_b].pos
        particles_render[i_p, i_b].vel = particles[i_p, i_b].vel
        particles_render[i_p, i_b].active = particles_ng[i_p, i_b].active
    for i_v, i_b in qd.ndrange(vverts_info.shape[0], particles.shape[1]):
        pos = qd.Vector.zero(gs.qd_float, 3)
        for i_s in qd.static(range(vverts_info.support_idxs.n)):
            pos += particles[vverts_info[i_v].support_idxs[i_s], i_b].pos * vverts_info[i_v].support_weights[i_s]
        vverts_render[i_v, i_b].pos = pos
        vverts_render[i_v, i_b].active = particles_ng[vverts_info[i_v].support_idxs[0], i_b].active


@qd.kernel
def kernel_set_field(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    values: qd.types.ndarray(),
    data: V_ANNOTATION,
    is_batch_first: V_ANNOTATION,
):
    for i_b, i_p in qd.ndrange(envs_idx.shape[0], particles_idx.shape[1]):
        i_row, i_col = particles_idx[i_b, i_p], envs_idx[i_b]
        if qd.static(is_batch_first):
            i_row, i_col = i_col, i_row
        if qd.static(len(values.shape) == 3):
            for i_axis in qd.static(range(values.shape[-1])):
                data[i_row, i_col][i_axis] = values[i_b, i_p, i_axis]
        else:
            data[i_row, i_col] = values[i_b, i_p]


@qd.kernel
def kernel_reset_diagnostics(envs_idx: qd.types.ndarray(), collision_iterations: V_ANNOTATION, errno: qd.Tensor):
    for i_b, i_iteration in qd.ndrange(envs_idx.shape[0], collision_iterations.shape[1]):
        collision_iterations[envs_idx[i_b], i_iteration] = 0
    for i_b in range(envs_idx.shape[0]):
        errno[envs_idx[i_b]] = 0


class PBDUnifiedSolver(Solver):
    """Position-based dynamics (PBD) with simultaneous elastic corrections and per-iteration hard rigid contact.

    Supports cloth and tetrahedral elastic entities. Rigid geoms with ``needs_coup=True`` supply prescribed
    boundaries. Each iteration reads a common position state for every elastic constraint, then projects the
    combined update onto the contact constraints. Material compliance regularizes each projection denominator;
    the effective stiffness also depends on the iteration count and time step. Extrapolating consecutive iterates
    accelerates shape recovery, with the iteration history initialized after each substep's motion prediction.
    """

    class MATERIAL(gs.IntEnum):
        CLOTH = 0
        ELASTIC = 1

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        self._options = options
        self._particle_size = options.particle_size
        self.boundary = CubeBoundary(lower=options.lower_bound, upper=options.upper_bound)
        self._n_vvert_supports = scene.vis_options.n_support_neighbors
        self.particles = None
        self.particles_ng = None
        self.particles_info = None
        self.edges_info = None
        self.inner_edges_info = None
        self.elems_info = None
        self.particles_render = None
        self.vverts_info = None
        self.vverts_render = None
        self.vverts_uvs = None
        self.vfaces_indices = None
        self._rigid_surface = None
        self._surface_state = None
        self._surface_faces = None
        self._collision_iterations = None
        self._errno = None
        self._iteration_positions = None
        self._edges = None
        self._inner_edges = None
        self._elems = None
        self._lengths_rest = None
        self._angles_rest = None
        self._volumes_rest = None

    def add_entity(self, idx, material, morph, surface, name=None):
        if isinstance(material, gs.materials.PBD.Cloth):
            entity = PBD2DEntity(
                self.scene,
                self,
                material,
                morph,
                surface,
                self.particle_size,
                idx,
                self.n_particles,
                self.n_edges,
                self.n_inner_edges,
                self.n_vverts,
                self.n_vfaces,
                name=name,
            )
        elif isinstance(material, gs.materials.PBD.Elastic):
            entity = PBD3DEntity(
                self.scene,
                self,
                material,
                morph,
                surface,
                self.particle_size,
                idx,
                self.n_particles,
                self.n_edges,
                self.n_elems,
                self.n_vverts,
                self.n_vfaces,
                name=name,
            )
        else:
            gs.raise_exception("PBDUnifiedSolver supports PBD.Cloth and PBD.Elastic materials.")
        self._entities.append(entity)
        return entity

    def build(self):
        super().build()
        if not self.is_active:
            return
        if self.scene.requires_grad:
            gs.raise_exception("PBDUnifiedSolver requires requires_grad=False.")
        self._n_vvert_supports = min(self._n_vvert_supports, min(entity.n_particles for entity in self.entities))
        self.particles = qd.types.struct(
            free=gs.qd_bool,
            pos=gs.qd_vec3,
            ipos=gs.qd_vec3,
            pos_iter=gs.qd_vec3,
            dpos=gs.qd_vec3,
            vel=gs.qd_vec3,
        ).field(shape=(self.n_particles, self._B), layout=qd.Layout.SOA)
        self.particles_ng = qd.types.struct(active=gs.qd_bool, reordered_idx=gs.qd_int).field(
            shape=(self.n_particles, self._B), layout=qd.Layout.SOA
        )
        self.particles_info = qd.types.struct(
            mass=gs.qd_float,
            pos_rest=gs.qd_vec3,
            material_type=gs.qd_int,
            mu_s=gs.qd_float,
            mu_k=gs.qd_float,
            air_resistance=gs.qd_float,
        ).field(shape=(self.n_particles,), layout=qd.Layout.SOA)
        if self.n_edges:
            self.edges_info = qd.types.struct(
                len_rest=gs.qd_float,
                stretch_compliance=gs.qd_float,
                stretch_relaxation=gs.qd_float,
                v1=gs.qd_int,
                v2=gs.qd_int,
            ).field(shape=(self.n_edges,), layout=qd.Layout.SOA)
        if self.n_inner_edges:
            self.inner_edges_info = qd.types.struct(
                len_rest=gs.qd_float,
                angle_rest=gs.qd_float,
                bending_compliance=gs.qd_float,
                bending_relaxation=gs.qd_float,
                v1=gs.qd_int,
                v2=gs.qd_int,
                v3=gs.qd_int,
                v4=gs.qd_int,
            ).field(shape=(self.n_inner_edges,), layout=qd.Layout.SOA)
        if self.n_elems:
            self.elems_info = qd.types.struct(
                vol_rest=gs.qd_float,
                volume_compliance=gs.qd_float,
                volume_relaxation=gs.qd_float,
                v1=gs.qd_int,
                v2=gs.qd_int,
                v3=gs.qd_int,
                v4=gs.qd_int,
            ).field(shape=(self.n_elems,), layout=qd.Layout.SOA)
        self.particles_render = qd.types.struct(pos=gs.qd_vec3, vel=gs.qd_vec3, active=gs.qd_bool).field(
            shape=(self.n_particles, self._B), layout=qd.Layout.SOA
        )
        self.vverts_info = qd.types.struct(
            support_idxs=qd.types.vector(self._n_vvert_supports, gs.qd_int),
            support_weights=qd.types.vector(self._n_vvert_supports, gs.qd_float),
        ).field(shape=(self.n_vverts,), layout=qd.Layout.SOA)
        self.vverts_render = qd.types.struct(pos=gs.qd_vec3, active=gs.qd_bool).field(
            shape=(self.n_vverts, self._B), layout=qd.Layout.SOA
        )
        self.vverts_uvs = qd.field(gs.qd_vec2, shape=(self.n_vverts,))
        self.vfaces_indices = qd.field(gs.qd_ivec3, shape=(self.n_vfaces,))
        self._errno = array_class.V(dtype=gs.qd_int, shape=(self._B,))
        self._collision_iterations = qd.field(gs.qd_int, shape=(self._B, self._options.max_solver_iterations))
        for entity in self.entities:
            entity._add_to_solver()
        self.particles_ng.reordered_idx.from_numpy(
            np.broadcast_to(np.arange(self.n_particles, dtype=gs.np_int)[:, None], (self.n_particles, self._B)).copy()
        )
        if self.n_inner_edges:
            kernel_init_bending(self.particles_info, self.inner_edges_info, self._errno)
        self._edges = torch.as_tensor(
            np.concatenate(tuple(entity.edges + entity.particle_start for entity in self.entities)), device=gs.device
        )
        self._inner_edges = torch.as_tensor(
            np.concatenate(
                (
                    np.empty((0, 4), dtype=gs.np_int),
                    *(
                        entity.inner_edges + entity.particle_start
                        for entity in self.entities
                        if isinstance(entity, PBD2DEntity)
                    ),
                )
            ),
            device=gs.device,
        )
        self._elems = torch.as_tensor(
            np.concatenate(
                (
                    np.empty((0, 4), dtype=gs.np_int),
                    *(
                        entity.elems + entity.particle_start
                        for entity in self.entities
                        if isinstance(entity, PBD3DEntity)
                    ),
                )
            ),
            device=gs.device,
        )
        self._lengths_rest = qd_to_torch(self.edges_info.len_rest, transpose=True, copy=True)
        self._angles_rest = (
            qd_to_torch(self.inner_edges_info.angle_rest, transpose=True, copy=True)
            if self.n_inner_edges
            else torch.empty(0, dtype=gs.tc_float, device=gs.device)
        )
        self._volumes_rest = (
            qd_to_torch(self.elems_info.vol_rest, transpose=True, copy=True)
            if self.n_elems
            else torch.empty(0, dtype=gs.tc_float, device=gs.device)
        )
        if self.n_elems and ((self._volumes_rest == 0.0).any() or not torch.isfinite(self._volumes_rest).all()):
            gs.raise_exception("PBD elastic rest tetrahedra must have nonzero volume.")
        rigid = self.scene.rigid_solver
        projection_geoms = [
            geom for entity in rigid.entities for link in entity.links for geom in link.geoms if geom.needs_coup
        ]
        if projection_geoms:
            if not any(geom.n_faces for geom in projection_geoms):
                gs.raise_exception("PBD rigid collision requires surface triangles.")
            rigid.collider._sdf.activate()
            self._rigid_surface = build_rigid_surface(rigid, projection_geoms)
            self._surface_state = array_class.get_rigid_surface_contact_state(
                self._B, self.n_particles, sum(geom.n_faces > 0 for geom in projection_geoms)
            )
            faces = np.concatenate(tuple(entity.surface_triangles + entity.particle_start for entity in self.entities))
            self._surface_faces = qd.field(gs.qd_ivec3, shape=(len(faces),))
            self._surface_faces.from_numpy(faces)
            kernel_reset_rigid_surface_contact(
                self.scene._envs_idx, rigid.dyn_state, self._surface_state, self._rigid_surface.info
            )
        if self._options.is_recording_constraint_history:
            self._iteration_positions = torch.empty(
                (self._options.max_solver_iterations, 3, self._B, self.n_particles, 3),
                dtype=gs.tc_float,
                device=gs.device,
            )
        self.check_errno()

    def process_input(self, in_backward=False):
        for entity in self.entities:
            entity.process_input(in_backward=in_backward)

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def substep_pre_coupling(self, f):
        if not self.is_active:
            return
        self._collision_iterations.fill(0)
        kernel_predict(
            self.sim.cur_t,
            self.substep_dt,
            self.particles,
            self.particles_ng,
            self.particles_info,
            self._gravity,
            self,
            self._errno,
        )
        for i_iteration in range(self._options.max_solver_iterations):
            if self._iteration_positions is not None:
                self._iteration_positions[i_iteration, 0].copy_(qd_to_torch(self.particles.pos, transpose=True))
            self.project_elastic_constraints()
            if self._iteration_positions is not None:
                self._iteration_positions[i_iteration, 1].copy_(qd_to_torch(self.particles.pos, transpose=True))
            self.project_collision(i_iteration)
            if self._iteration_positions is not None:
                self._iteration_positions[i_iteration, 2].copy_(qd_to_torch(self.particles.pos, transpose=True))
        kernel_update_velocity(self.substep_dt, self.particles, self.particles_ng, self._errno)
        if self.n_elems:
            kernel_check_volume(self.particles, self.particles_ng, self.elems_info, self._errno)
        self.check_errno()

    def project_elastic_constraints(self):
        """Apply one simultaneous stretch, bending, and volume correction at the current positions."""
        self.particles.dpos.fill(0.0)
        if self.n_edges:
            kernel_accumulate_stretch(
                self.substep_dt, self.particles, self.particles_ng, self.particles_info, self.edges_info
            )
        if self.n_inner_edges:
            kernel_accumulate_bending(
                self.substep_dt,
                self.particles,
                self.particles_ng,
                self.particles_info,
                self.inner_edges_info,
                self._errno,
            )
        if self.n_elems:
            kernel_accumulate_volume(
                self.substep_dt, self.particles, self.particles_ng, self.particles_info, self.elems_info
            )
        kernel_apply_delta(self._options.constraint_acceleration, self.particles, self.particles_ng, self._errno)

    def project_collision(self, i_iteration):
        """Enforce hard rigid and domain contacts for one elastic iteration."""
        if self._rigid_surface is None:
            kernel_project_boundary(self.boundary, self.particles, self.particles_ng)
            return
        rigid = self.scene.rigid_solver
        surface = self._rigid_surface
        self._surface_state.is_active.fill(True)
        self._surface_state.has_intersection.fill(0)
        for i_contact in range(self._options.max_collision_iterations + 1):
            is_audit = i_contact == self._options.max_collision_iterations
            kernel_project_vertices(
                i_iteration,
                self.boundary,
                self.particles,
                self.particles_ng,
                surface.bvh.nodes,
                surface.bvh.morton_codes,
                self._collision_iterations,
                rigid.dyn_state,
                self._surface_state,
                rigid.dyn_info,
                rigid.rigid_info,
                rigid.collider._collider_info,
                surface.info,
                is_audit=is_audit,
                errno=self._errno,
            )
            kernel_detect_pbd_rigid_surface_intersections(
                i_contact,
                self._surface_faces,
                self.particles,
                self.particles_ng,
                surface.bvh.nodes,
                surface.bvh.morton_codes,
                rigid.dyn_state,
                self._surface_state,
                rigid.dyn_info,
                rigid.rigid_info,
                surface.info,
                is_audit=is_audit,
                errno=self._errno,
            )
            if not is_audit:
                kernel_apply_pbd_rigid_surface_corrections(
                    self.substep_dt, self.particles, self.particles_ng, self._surface_state, self._errno
                )
                if not qd_to_torch(self._surface_state.is_active, transpose=True).any():
                    break

    def substep_post_coupling(self, f):
        if self._rigid_surface is not None:
            kernel_store_rigid_surface_poses(
                self.scene.rigid_solver.dyn_state, self._surface_state, self._rigid_surface.info
            )

    def check_errno(self):
        if self._errno is None:
            return
        errno = np.bitwise_or.reduce(qd_to_numpy(self._errno, transpose=True))
        if errno & (ErrorCode.INVALID_PBD_STATE | ErrorCode.INVALID_CONTACT_NAN):
            gs.raise_exception("PBDUnifiedSolver encountered a non-finite state or a degenerate bending constraint.")
        if errno & ErrorCode.INVALID_PBD_VOLUME:
            gs.raise_exception("PBDUnifiedSolver has a collapsed or inverted tetrahedron with every vertex fixed.")
        if errno & ErrorCode.INVALID_PBD_RIGID_SURFACE_INTERSECTION:
            gs.raise_exception(
                "PBDUnifiedSolver hard collision projection left a rigid intersection or a fixed particle in contact."
            )

    def get_state(self, f):
        if not self.is_active:
            return None
        state = PBDUnifiedSolverState(self.scene)
        state.pos.copy_(qd_to_torch(self.particles.pos, transpose=True))
        state.vel.copy_(qd_to_torch(self.particles.vel, transpose=True))
        state.free.copy_(qd_to_torch(self.particles.free, transpose=True))
        state.active = qd_to_torch(self.particles_ng.active, transpose=True, copy=True)
        if self._surface_state is not None:
            state.previous_geoms_pos = qd_to_torch(
                self._surface_state.previous_geoms_pos, transpose=True, copy=True
            ).transpose(0, 1)
            state.previous_geoms_quat = qd_to_torch(
                self._surface_state.previous_geoms_quat, transpose=True, copy=True
            ).transpose(0, 1)
        return state

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def set_state(self, f, state, envs_idx=None):
        if not self.is_active:
            return
        envs_idx = self.scene._sanitize_envs_idx(envs_idx)
        particles_idx = torch.arange(self.n_particles, device=gs.device).expand(len(envs_idx), -1)
        for field, value in (
            (self.particles.pos, state.pos),
            (self.particles.ipos, state.pos),
            (self.particles.pos_iter, state.pos),
            (self.particles.vel, state.vel),
            (self.particles.free, state.free),
            (self.particles_ng.active, state.active),
        ):
            self.set_particle_field(field, value[envs_idx].contiguous(), particles_idx, envs_idx)
        self.set_particle_field(self.particles.dpos, torch.zeros_like(state.pos[envs_idx]), particles_idx, envs_idx)
        if gs.use_zerocopy:
            errno = qd_to_torch(self._errno, transpose=True, copy=False)
            collision_iterations = qd_to_torch(self._collision_iterations, transpose=True, copy=False)
            errno[envs_idx] = 0
            collision_iterations[:, envs_idx] = 0
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            kernel_reset_diagnostics(envs_idx, self._collision_iterations, self._errno)
        if self._surface_state is not None:
            kernel_reset_rigid_surface_contact(
                envs_idx, self.scene.rigid_solver.dyn_state, self._surface_state, self._rigid_surface.info
            )
            for field, value in (
                (self._surface_state.previous_geoms_pos, state.previous_geoms_pos),
                (self._surface_state.previous_geoms_quat, state.previous_geoms_quat),
            ):
                if gs.use_zerocopy:
                    view = qd_to_torch(field, transpose=True, copy=False).transpose(0, 1)
                    view[envs_idx] = value[envs_idx]
                else:
                    geoms_idx = torch.arange(value.shape[1], device=gs.device).expand(len(envs_idx), -1)
                    kernel_set_field(geoms_idx, envs_idx, value[envs_idx].contiguous(), field, is_batch_first=True)
            if gs.backend == gs.metal:
                torch.mps.synchronize()

    def set_particle_field(self, field, values, particles_idx, envs_idx):
        """Write a scalar or three-component particle field using environment-local indices."""
        if gs.use_zerocopy:
            view = qd_to_torch(field, transpose=True, copy=False)
            view[envs_idx[:, None], particles_idx] = values
        else:
            kernel_set_field(particles_idx, envs_idx, values, field, is_batch_first=False)

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def _kernel_set_particles_pos(self, particles_idx, envs_idx, poss):
        self.set_particle_field(self.particles.pos, poss, particles_idx, envs_idx)
        self.set_particle_field(self.particles.ipos, poss, particles_idx, envs_idx)
        self.set_particle_field(self.particles.pos_iter, poss, particles_idx, envs_idx)
        self.set_particle_field(self.particles.vel, torch.zeros_like(poss), particles_idx, envs_idx)
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def _kernel_get_particles_pos(self, particle_start, n_particles, envs_idx, poss):
        poss.copy_(
            qd_to_torch(
                self.particles.pos, envs_idx, slice(particle_start, particle_start + n_particles), transpose=True
            )
        )

    @mutates(StateChange.DYNAMICS)
    def _kernel_set_particles_vel(self, particles_idx, envs_idx, vels):
        self.set_particle_field(self.particles.vel, vels, particles_idx, envs_idx)
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def _kernel_get_particles_vel(self, particle_start, n_particles, envs_idx, vels):
        vels.copy_(
            qd_to_torch(
                self.particles.vel, envs_idx, slice(particle_start, particle_start + n_particles), transpose=True
            )
        )

    @mutates(StateChange.GEOMETRY)
    def _kernel_set_particles_active(self, particles_idx, envs_idx, actives):
        self.set_particle_field(self.particles_ng.active, actives, particles_idx, envs_idx)
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def _kernel_get_particles_active(self, particle_start, n_particles, envs_idx, actives):
        actives.copy_(
            qd_to_torch(
                self.particles_ng.active, envs_idx, slice(particle_start, particle_start + n_particles), transpose=True
            )
        )

    def _kernel_fix_particles(self, particles_idx, envs_idx):
        self.set_particle_field(
            self.particles.free, torch.zeros_like(particles_idx, dtype=gs.tc_bool), particles_idx, envs_idx
        )
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def _kernel_release_particle(self, particles_idx, envs_idx):
        self.set_particle_field(
            self.particles.free, torch.ones_like(particles_idx, dtype=gs.tc_bool), particles_idx, envs_idx
        )
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    def _kernel_get_mass(self, particle_start, n_particles, mass, envs_idx):
        mass[:] = qd_to_torch(
            self.particles_info.mass, slice(particle_start, particle_start + n_particles), transpose=True
        ).sum()

    def get_embedded_positions(self, elements_idx, barycentric, envs_idx=None):
        """Return tetrahedral material points with shape [B, n_points, 3] for the selected environments."""
        envs_idx = self.scene._sanitize_envs_idx(envs_idx)
        vertices = self._elems[indices_to_mask(elements_idx)]
        positions = qd_to_torch(self.particles.pos, envs_idx, transpose=True)
        return (positions[:, vertices] * barycentric[None, :, :, None]).sum(dim=-2)

    def get_constraint_residuals(self, positions=None, envs_idx=None):
        """Compute signed stretch/volume ratios and bending angles from positions with trailing shape [N, 3]."""
        if positions is None:
            envs_idx = self.scene._sanitize_envs_idx(envs_idx)
            positions = qd_to_torch(self.particles.pos, envs_idx, transpose=True)
        edges = positions[..., self._edges, :]
        stretch = torch.linalg.vector_norm(edges[..., 0, :] - edges[..., 1, :], dim=-1) / self._lengths_rest - 1.0
        tetrahedra = positions[..., self._elems, :]
        volume = torch.linalg.det(tetrahedra[..., 1:, :] - tetrahedra[..., :1, :]) / (6.0 * self._volumes_rest) - 1.0
        points = positions[..., self._inner_edges, :]
        edge = points[..., 1, :] - points[..., 0, :]
        normal1 = torch.linalg.cross(edge, points[..., 2, :] - points[..., 0, :])
        normal2 = torch.linalg.cross(edge, points[..., 3, :] - points[..., 0, :])
        angle = torch.atan2(
            (edge * torch.linalg.cross(normal1, normal2)).sum(dim=-1) / torch.linalg.vector_norm(edge, dim=-1),
            (normal1 * normal2).sum(dim=-1),
        )
        bending = torch.atan2(torch.sin(angle - self._angles_rest), torch.cos(angle - self._angles_rest))
        return PBDConstraintResiduals(stretch, bending, volume)

    def get_constraint_history(self):
        """Return residuals with shape [iteration, phase, B, constraint]; phases are input, elastic, and contact."""
        if self._iteration_positions is None:
            gs.raise_exception("Constraint history requires is_recording_constraint_history=True.")
        return self.get_constraint_residuals(self._iteration_positions)

    def get_collision_iterations(self):
        """Return the collision sweep count with shape [B, elastic_iteration] for the last substep."""
        return qd_to_torch(self._collision_iterations, transpose=True, copy=True).T

    def update_render_fields(self):
        if self.is_active:
            kernel_update_render(
                self.particles, self.particles_ng, self.particles_render, self.vverts_render, self.vverts_info
            )

    def get_state_render(self):
        if not self.is_active:
            return None, None, None
        self.update_render_fields()
        return self.vverts_render.pos, self.vverts_uvs, self.vfaces_indices

    def get_tetrahedral_state_render(self, f):
        return self.particles.pos

    def reset_grad(self):
        pass

    def process_input_grad(self):
        pass

    def collect_output_grads(self):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    @property
    def is_active(self):
        return bool(self.entities)

    @property
    def n_particles(self):
        return sum(entity.n_particles for entity in self.entities)

    @property
    def n_edges(self):
        return sum(entity.n_edges for entity in self.entities)

    @property
    def n_inner_edges(self):
        return sum(entity.n_inner_edges for entity in self.entities if isinstance(entity, PBD2DEntity))

    @property
    def n_elems(self):
        return sum(entity.n_elems for entity in self.entities if isinstance(entity, PBD3DEntity))

    @property
    def n_vverts(self):
        return sum(entity.n_vverts for entity in self.entities)

    @property
    def n_vfaces(self):
        return sum(entity.n_vfaces for entity in self.entities)

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self.particle_size / 2.0
