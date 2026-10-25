from itertools import combinations

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.bvh import AABB, RefitBVH, kernel_refit_bvh
from genesis.engine.entities.vbd_entity import VBDEntity
from genesis.engine.solvers.base_solver import Solver, StateChange, mutates
from genesis.engine.states.vbd import VBDState
from genesis.utils.array_class import ErrorCode, V, V_ANNOTATION
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array

from .collision import find_shell_contacts, find_volume_contacts, initialize_shell_bounds, update_bounds
from .contact import solve_colors
from .data import (
    CollisionData,
    ColorData,
    ElementData,
    HingeData,
    IterationData,
    PlaneData,
    VBDConfig,
    VertexData,
    array_to_field,
    color_vertices,
    incidence,
)
from .physics import (
    apply_prediction,
    finish,
    limit_initialization,
    limit_prediction,
    predict,
    prepare_prediction,
    validate_state,
)


@qd.kernel
def write_field(envs_idx: qd.types.ndarray(), values: qd.types.ndarray(), field: qd.Tensor):
    for index, batch in qd.ndrange(field.shape[0], envs_idx.shape[0]):
        if qd.static(len(values.shape) == 2):
            field[index, envs_idx[batch]] = values[batch, index]
        elif qd.static(len(values.shape) == 3):
            for axis in qd.static(range(values.shape[2])):
                field[index, envs_idx[batch]][axis] = values[batch, index, axis]
        else:
            for row, col in qd.static(qd.ndrange(3, 3)):
                field[index, envs_idx[batch]][row, col] = values[batch, index, row, col]


@qd.kernel
def render_positions(
    indices: qd.Tensor,
    offsets: qd.Tensor,
    positions: qd.Tensor,
    vertices: V_ANNOTATION,
):
    for index, batch in positions:
        positions[index, batch] = vertices.pos[indices[index], batch] + offsets[batch]


def read_field(field, batch_size, n_entries=0):
    """Copy a batched field to a public tensor, including empty optional components."""
    if field is None:
        return torch.empty((batch_size, n_entries), dtype=gs.tc_float, device=gs.device)
    return qd_to_torch(field, transpose=True, copy=True)


def replace_field(field, values, envs_idx):
    """Write a batched tensor through a shared view or the portable transfer kernel."""
    if field is not None:
        if gs.use_zerocopy:
            field_view = qd_to_torch(field, transpose=True, copy=False)
            field_view[envs_idx] = values
        else:
            write_field(envs_idx, values.contiguous(), field)


class VBDSolver(Solver):
    """Implicit Euler integration with colored vertex block descent (VBD) and penalty contact."""

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)
        self.options = options
        self.config = VBDConfig(
            options.contact_d_hat,
            options.contact_search_radius,
            options.friction_epsilon,
            options.contact_stiffness,
            options.shell_contact_stiffness,
            options.convergence_tolerance,
            options.is_contact_diagnostics_enabled,
            options.is_line_search_enabled,
            options.line_search_iterations,
            n_vertex_threads=8 if gs.backend == gs.cuda else 1,
        )
        self.vertices = None
        self.elements = None
        self.hinges = None
        self.planes = None
        self.colors = None
        self.iteration = None
        self.collision = None
        self.errno = None
        self.has_warned_penetration = False

    def add_entity(self, idx, material, morph, surface, name=None):
        entity = VBDEntity(idx, self.scene, morph, self, material, surface, name=name)
        self._entities.append(entity)
        return entity

    def build(self):
        super().build()
        self.n_envs = self.sim.n_envs
        if not self.is_active:
            return
        if self.sim.requires_grad:
            gs.raise_exception("VBD supports forward simulation only.")
        if not isinstance(self.sim.coupler_options, gs.options.LegacyCouplerOptions):
            gs.raise_exception("VBD supplies its own contact solver; use the default coupler options.")
        for solver in self.sim.solvers:
            if solver.is_active and solver not in (self, self.sim.rigid_solver):
                gs.raise_exception("VBD scenes support VBD deformables and fixed rigid planes.")
        if not isinstance(self.scene.renderer_options, gs.renderers.Rasterizer):
            gs.raise_exception("VBD visualization requires the Rasterizer renderer.")
        if self._ffs:
            gs.raise_exception("Use VBD vertex forces to apply spatially varying loads.")
        n_vertices, n_elements = self.n_vertices, self.n_elements
        positions = np.concatenate([entity.init_positions for entity in self.entities], dtype=gs.np_float)
        indices = np.empty((n_elements, 4), dtype=gs.np_int)
        gradients = np.zeros((n_elements, 4, 3), dtype=gs.np_float)
        measure = np.empty(n_elements, dtype=gs.np_float)
        mu, lam = np.empty_like(measure), np.empty_like(measure)
        is_volume = np.empty(n_elements, dtype=gs.np_bool)
        is_log_neohookean = np.zeros(n_elements, dtype=gs.np_bool)
        is_vertex_volume = np.empty(n_vertices, dtype=gs.np_bool)
        is_plastic = np.zeros(n_elements, dtype=gs.np_bool)
        yield_strain, hardening = np.zeros_like(measure), np.zeros_like(measure)
        mass = np.zeros(n_vertices, dtype=gs.np_float)
        radius, friction = np.zeros_like(mass), np.zeros_like(mass)
        entity_indices = np.empty(n_vertices, dtype=gs.np_int)
        hinge_indices, hinge_stiffness, hinge_yield, hinge_hardening, hinge_plastic = [], [], [], [], []
        for i_entity, entity in enumerate(self.entities):
            material = entity.material
            vs = slice(entity.v_start, entity.v_start + entity.n_vertices)
            entity_indices[vs] = i_entity
            es = slice(entity._el_start, entity._el_start + entity.n_elements)
            connectivity = entity.elems + entity.v_start
            is_shell = isinstance(material, gs.materials.VBD.Cloth)
            n_corners = 3 if is_shell else 4
            indices[es, :n_corners] = connectivity
            is_volume[es] = not is_shell
            is_vertex_volume[vs] = not is_shell
            friction[vs] = material.friction_mu
            corners = positions[connectivity]
            edges = corners[:, 1:, :] - corners[:, :1, :]
            if is_shell:
                indices[es, 3] = connectivity[:, 0]
                length = np.linalg.norm(edges[:, 0], axis=-1)
                normal = np.cross(edges[:, 0], edges[:, 1])
                twice_area = np.linalg.norm(normal, axis=-1)
                if (twice_area <= 0).any():
                    gs.raise_exception("VBD shells require nondegenerate triangles.")
                rest = np.zeros((entity.n_elements, 2, 2), dtype=gs.np_float)
                rest[:, 0, 0] = length
                rest[:, 0, 1] = np.sum(edges[:, 0] * edges[:, 1], axis=-1) / length
                rest[:, 1, 1] = twice_area / length
                gradients[es, 1:3, :2] = np.linalg.inv(rest)
                measure[es] = twice_area / 2
                mu[es] = material.E * material.thickness / (2 * (1 + material.nu)) + material.shear_stiffness
                lam[es] = (
                    material.E * material.thickness * material.nu / (1 - material.nu**2) - material.shear_stiffness
                )
                density = material.rho * material.thickness
                radius[vs] = material.thickness / 2
                if material.bending_model is not None and material.bending_rigidity > 0:
                    oriented = np.concatenate(
                        (connectivity[:, [0, 1, 2]], connectivity[:, [1, 2, 0]], connectivity[:, [2, 0, 1]])
                    )
                    _, inverse, counts = np.unique(
                        np.sort(oriented[:, :2], axis=1), axis=0, return_inverse=True, return_counts=True
                    )
                    if (counts > 2).any():
                        gs.raise_exception("VBD shells require manifold edges.")
                    order = np.argsort(inverse, kind="stable")
                    starts = np.concatenate(([0], np.cumsum(counts)))[:-1][counts == 2]
                    first, second = order[starts], order[starts + 1]
                    hinges = np.column_stack((oriented[first], oriented[second, 2]))
                    entity._hinge_start = sum(len(value) for value in hinge_indices)
                    entity._n_hinges = len(hinges)
                    hinge_indices.append(hinges)
                    area_sum = measure[es][first % entity.n_elements] + measure[es][second % entity.n_elements]
                    edge_sq = np.sum((positions[hinges[:, 1]] - positions[hinges[:, 0]]) ** 2, axis=-1)
                    hinge_stiffness.append(3 * material.bending_rigidity * edge_sq / area_sum)
                    hinge_yield.append(np.full(len(hinges), np.deg2rad(material.bending_yield_angle_deg)))
                    hinge_hardening.append(np.full(len(hinges), material.bending_plastic_hardening))
                    hinge_plastic.append(np.full(len(hinges), material.is_bending_plasticity_enabled))
            else:
                rest = edges.transpose(0, 2, 1)
                determinant = np.linalg.det(rest)
                if (np.abs(determinant) <= 0).any():
                    gs.raise_exception("VBD solids require nondegenerate tetrahedra.")
                gradients[es, 1:, :] = np.linalg.inv(rest)
                measure[es] = np.abs(determinant) / 6
                mu[es] = material.E / (2 * (1 + material.nu))
                lam[es] = material.E * material.nu / ((1 + material.nu) * (1 - 2 * material.nu))
                density = material.rho
                is_log_neohookean[es] = material.model == "log_neohookean"
                is_plastic[es] = material.is_volume_plasticity_enabled
                yield_strain[es] = np.log1p(material.volume_yield_strain_percent / 100)
                hardening[es] = material.volume_plastic_hardening
            gradients[es, 0] = -gradients[es, 1:].sum(axis=1)
            np.add.at(mass, connectivity, (density * measure[es] / n_corners)[:, None])
        if (mass <= 0).any():
            gs.raise_exception("Every VBD vertex must belong to an element with positive mass.")
        hinges = np.concatenate(hinge_indices) if hinge_indices else np.empty((0, 4), dtype=gs.np_int)
        hinge_positions = positions[hinges]
        edge = hinge_positions[:, 1] - hinge_positions[:, 0]
        normal0 = np.cross(edge, hinge_positions[:, 2] - hinge_positions[:, 0])
        normal1 = np.cross(hinge_positions[:, 3] - hinge_positions[:, 0], edge)
        angle = np.arctan2(
            np.sum(edge * np.cross(normal0, normal1), axis=-1) / np.linalg.norm(edge, axis=-1),
            np.sum(normal0 * normal1, axis=-1),
        )
        pairs = np.concatenate(
            [indices[:, pair] for pair in combinations(range(4), 2)]
            + [hinges[:, pair] for pair in combinations(range(4), 2)]
        )
        pairs = np.unique(np.concatenate((pairs, pairs[:, ::-1])), axis=0)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        neighbor_offsets, order = incidence(pairs[:, 0], n_vertices)
        neighbors = pairs[order, 1]
        colors = color_vertices(neighbor_offsets, neighbors)
        self.colors = ColorData(
            array_to_field(np.argsort(colors, kind="stable"), gs.qd_int),
            tuple(int(value) for value in np.concatenate(([0], np.cumsum(np.bincount(colors))))),
        )
        elem_offsets, elem_entries = incidence(indices, n_vertices)
        # The fourth triangle slot is padding in the element layout; it has no incidence or energy.
        elem_entries = elem_entries[is_volume[elem_entries // 4] | (elem_entries % 4 != 3)]
        elem_counts = np.bincount(indices.flat[elem_entries], minlength=n_vertices)
        elem_offsets = np.concatenate(([0], np.cumsum(elem_counts)))
        hinge_offsets, hinge_entries = incidence(hinges, n_vertices)
        batch_positions = np.repeat(positions[:, None], self._B, axis=1)
        is_fixed = np.repeat(
            np.concatenate([entity._init_is_fixed for entity in self.entities])[:, None], self._B, axis=1
        )
        fixed_pos = np.repeat(
            np.concatenate([entity._init_fixed_pos for entity in self.entities])[:, None], self._B, axis=1
        )
        forces = np.repeat(np.concatenate([entity._init_forces for entity in self.entities])[:, None], self._B, axis=1)
        self.vertices = VertexData(
            array_to_field(batch_positions, gs.qd_float, element_shape=(3,)),
            array_to_field(np.zeros_like(batch_positions), gs.qd_float, element_shape=(3,)),
            array_to_field(batch_positions, gs.qd_float, element_shape=(3,)),
            array_to_field(batch_positions, gs.qd_float, element_shape=(3,)),
            array_to_field(batch_positions, gs.qd_float, element_shape=(3,)),
            array_to_field(batch_positions, gs.qd_float, element_shape=(3,)),
            array_to_field(forces, gs.qd_float, element_shape=(3,)),
            array_to_field(is_fixed, gs.qd_bool),
            array_to_field(fixed_pos, gs.qd_float, element_shape=(3,)),
            array_to_field(np.zeros((n_vertices, self._B)), gs.qd_float),
            array_to_field(mass, gs.qd_float),
            array_to_field(radius, gs.qd_float),
            array_to_field(is_vertex_volume, gs.qd_bool),
            array_to_field(friction, gs.qd_float),
            array_to_field(entity_indices, gs.qd_int),
            array_to_field(elem_offsets, gs.qd_int),
            array_to_field(elem_entries, gs.qd_int),
            array_to_field(hinge_offsets, gs.qd_int),
            array_to_field(hinge_entries, gs.qd_int),
        )
        self.elements = ElementData(
            array_to_field(indices, gs.qd_int, element_shape=(4,)),
            array_to_field(gradients, gs.qd_float, element_shape=(4, 3)),
            array_to_field(measure, gs.qd_float),
            array_to_field(mu, gs.qd_float),
            array_to_field(lam, gs.qd_float),
            array_to_field(is_volume, gs.qd_bool),
            array_to_field(is_log_neohookean, gs.qd_bool) if is_log_neohookean.any() else None,
            array_to_field(is_plastic, gs.qd_bool),
            array_to_field(yield_strain, gs.qd_float),
            array_to_field(hardening, gs.qd_float),
            array_to_field(np.tile(np.eye(3), (n_elements, self._B, 1, 1)), gs.qd_float, element_shape=(3, 3)),
            array_to_field(np.zeros((n_elements, self._B)), gs.qd_float),
            bool((~is_volume).any()),
            bool(is_plastic.any()),
            bool(is_log_neohookean.any()),
        )
        self.hinges = HingeData(
            array_to_field(hinges, gs.qd_int, element_shape=(4,)),
            array_to_field(np.concatenate(hinge_stiffness) if hinge_stiffness else np.empty(0), gs.qd_float),
            array_to_field(np.repeat(angle[:, None], self._B, axis=1), gs.qd_float),
            array_to_field(np.zeros((len(hinges), self._B)), gs.qd_float),
            array_to_field(np.concatenate(hinge_plastic) if hinge_plastic else np.empty(0), gs.qd_bool),
            array_to_field(np.concatenate(hinge_yield) if hinge_yield else np.empty(0), gs.qd_float),
            array_to_field(np.concatenate(hinge_hardening) if hinge_hardening else np.empty(0), gs.qd_float),
            len(hinges),
        )
        normals, plane_offsets = [], []
        rigid_solver = self.sim.rigid_solver
        links_pos, links_quat = None, None
        if rigid_solver.is_active:
            links_pos = tensor_to_array(rigid_solver.get_links_pos()).reshape(self._B, rigid_solver.n_links, 3)
            links_quat = tensor_to_array(rigid_solver.get_links_quat()).reshape(self._B, rigid_solver.n_links, 4)
        for entity in self.sim.rigid_solver.entities:
            for link in entity.links:
                for geom in link.geoms:
                    if geom.type != gs.GEOM_TYPE.PLANE or not geom.is_fixed:
                        gs.raise_exception("VBD rigid contact currently requires fixed Plane geoms.")
                    normal = gu.transform_by_quat(geom.data[:3], geom.init_quat)
                    normal = gu.transform_by_quat(normal, links_quat[0, link.idx])
                    normal /= np.linalg.norm(normal)
                    position = links_pos[0, link.idx] + gu.transform_by_quat(geom.init_pos, links_quat[0, link.idx])
                    normals.append(normal)
                    plane_offsets.append(np.dot(position, normal))
        self.planes = PlaneData(
            array_to_field(np.array(normals), gs.qd_float, element_shape=(3,)),
            array_to_field(np.array(plane_offsets), gs.qd_float),
            len(normals),
        )
        faces = np.concatenate([entity.surface_triangles + entity.v_start for entity in self.entities])
        tets = indices[is_volume]
        tet_faces = tets[:, ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))].reshape(4 * len(tets), 3)
        unique_faces, inverse, counts = np.unique(
            np.sort(tet_faces, axis=1), axis=0, return_inverse=True, return_counts=True
        )
        if (counts > 2).any():
            gs.raise_exception("VBD tetrahedra require manifold face adjacency.")
        order = np.argsort(inverse, kind="stable")
        starts = np.concatenate(([0], np.cumsum(counts)))[:-1]
        tet_neighbors = np.full(len(tet_faces), -1, dtype=gs.np_int)
        interior = starts[counts == 2]
        tet_neighbors[order[interior]] = order[interior + 1] // 4
        tet_neighbors[order[interior + 1]] = order[interior] // 4
        _, combined = np.unique(np.concatenate((unique_faces, np.sort(faces, axis=1))), axis=0, return_inverse=True)
        owners = np.full(combined.max() + 1, -1, dtype=gs.np_int)
        owners[combined[: len(unique_faces)]] = order[starts] // 4
        face_tets = owners[combined[len(unique_faces) :]]
        volume_faces = np.flatnonzero(face_tets >= 0)
        triangles = positions[faces[volume_faces]]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        centers = positions[tets[face_tets[volume_faces]]].mean(axis=1)
        reverse = volume_faces[(normals * (centers - triangles[:, 0])).sum(axis=1) > 0]
        faces[reverse] = faces[reverse][:, [0, 2, 1]]
        edges, edge_inverse, edge_counts = np.unique(
            np.sort(faces[:, ((0, 1), (1, 2), (2, 0))].reshape(3 * len(faces), 2), axis=1),
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        if (edge_counts > 2).any():
            gs.raise_exception("VBD contact surfaces require manifold edges.")
        edge_order = np.argsort(edge_inverse, kind="stable")
        edge_starts = np.concatenate(([0], np.cumsum(edge_counts)))[:-1]
        edge_faces = np.full((len(edges), 2), -1, dtype=gs.np_int)
        edge_faces[:, 0] = edge_order[edge_starts] // 3
        edge_faces[edge_counts == 2, 1] = edge_order[edge_starts[edge_counts == 2] + 1] // 3
        vertex_faces = np.full(n_vertices, len(faces), dtype=gs.np_int)
        np.minimum.at(vertex_faces, faces.reshape(3 * len(faces)), np.repeat(np.arange(len(faces)), 3))
        surface_indices = np.unique(faces)
        surface_pairs = np.concatenate((edges, edges[:, ::-1]))
        neighbor_offsets, order = incidence(surface_pairs[:, 0], n_vertices)
        self.config.has_shell = bool((~is_volume).any())
        self.config.max_contact_thickness = float(2.0 * radius.max())
        if (
            self.config.has_shell
            and self.options.contact_search_radius <= self.config.max_contact_thickness + self.options.contact_d_hat
        ):
            gs.raise_exception("VBD contact_search_radius must exceed the maximum shell thickness plus contact_d_hat.")
        self.collision = CollisionData(
            array_to_field(surface_indices, gs.qd_int),
            array_to_field(faces, gs.qd_int, element_shape=(3,)),
            array_to_field(edges, gs.qd_int, element_shape=(2,)),
            array_to_field(edge_inverse.reshape(len(faces), 3), gs.qd_int, element_shape=(3,)),
            array_to_field(edge_faces, gs.qd_int, element_shape=(2,)),
            array_to_field(vertex_faces, gs.qd_int),
            array_to_field(neighbor_offsets, gs.qd_int),
            array_to_field(surface_pairs[order, 1], gs.qd_int),
            array_to_field(face_tets, gs.qd_int),
            array_to_field(tets, gs.qd_int, element_shape=(4,)),
            array_to_field(6.0 * self.options.contact_min_volume_ratio * measure[is_volume], gs.qd_float),
            array_to_field(tet_neighbors.reshape(len(tets), 4), gs.qd_int, element_shape=(4,)),
            V(gs.qd_int, (len(surface_indices), self._B)),
            V(gs.qd_int, (len(surface_indices), self._B)),
            V(gs.qd_vec4, (len(self.entities), self._B)) if self.config.has_shell else None,
            V(gs.qd_float, (len(self.entities), self._B)) if self.config.has_shell else None,
            V(gs.qd_int, (len(self.entities), self._B)) if self.config.has_shell else None,
            array_to_field(np.bincount(entity_indices, weights=mass), gs.qd_float) if self.config.has_shell else None,
            len(faces),
            len(edges) if self.config.has_shell else 0,
            len(tets),
            self.config.has_shell,
        )
        self.collision.intersected_tets.fill(-1)
        self.collision.contact_faces.fill(-1)
        self.contact_counts = torch.zeros(
            (len(surface_indices) + self.collision.n_edges) * self._B if self.config.has_shell else 0,
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.initial_scale = V(gs.qd_float, (len(self.entities), self._B))
        self.prediction_scale = V(gs.qd_float, (n_vertices, self._B))
        self.prediction_correction = V(gs.qd_int, ())
        self.iteration = IterationData(
            V(gs.qd_int, (3,)), V(gs.qd_float, ()) if self.options.convergence_tolerance else None
        )
        leaf_counts = [self.collision.n_faces]
        if self.collision.n_edges:
            leaf_counts.append(self.collision.n_edges)
        if self.collision.n_tets:
            leaf_counts.append(self.collision.n_tets)
        self.bvh = RefitBVH(AABB(self._B, sum(leaf_counts)), leaf_counts)
        update_bounds(False, False, self.vertices, self.collision, self.bvh)
        self.bvh.rebuild()
        self.errno = V(gs.qd_int, ())
        self.errno.fill(0)
        visual_indices = np.concatenate(
            [geom.sim_verts_idx + entity.v_start for entity in self.entities for geom in entity.vgeoms]
        )
        visual_faces = np.concatenate(
            [geom.vmesh.faces + geom.vvert_start for entity in self.entities for geom in entity.vgeoms]
        )
        self.render_indices = array_to_field(visual_indices, gs.qd_int)
        self.render_pos = V(gs.qd_vec3, (self.n_vverts, self._B))
        self.render_tet_pos = V(gs.qd_vec3, (self.n_vertices, self._B))
        self.vertex_indices = array_to_field(np.arange(self.n_vertices), gs.qd_int)
        self.render_faces = array_to_field(visual_faces, gs.qd_int, element_shape=(3,))
        self.envs_offset = array_to_field(self.scene.envs_offset, gs.qd_float, element_shape=(3,))
        gs.logger.info(
            f"VBD: {n_vertices} vertices, {n_elements} elements, "
            f"{len(hinges)} hinges, {len(self.colors.offsets) - 1} colors, "
            f"{self.config.n_vertex_threads} threads per vertex."
        )

    def process_input(self, in_backward=False):
        self.check_errno()

    def substep_pre_coupling(self, f):
        if not self.is_active:
            return
        predict(self.substep_dt, self._gravity, self.vertices, self.errno)
        prepare_prediction(self.vertices)
        for _ in range(32):
            self.prediction_correction.fill(0)
            limit_prediction(self.prediction_scale, self.prediction_correction, self.vertices, self.elements)
            if not qd_to_numpy(self.prediction_correction, transpose=True).item():
                break
        self.initial_scale.fill(1.0)
        limit_initialization(self.initial_scale, self.vertices, self.elements)
        update_bounds(is_previous=True, is_swept=False, vertices=self.vertices, collision=self.collision, bvh=self.bvh)
        kernel_refit_bvh(self.bvh)
        if self.sim.cur_substep_global % (16 * self.sim.substeps) == 0:
            self.bvh.rebuild()
        find_volume_contacts(
            is_ccd=False,
            vertices=self.vertices,
            collision=self.collision,
            bvh=self.bvh,
            options=self.config,
            errno=self.errno,
        )
        apply_prediction(
            is_shell=False, scale=self.initial_scale, vertices=self.vertices, collision=self.collision, errno=self.errno
        )
        update_bounds(is_previous=False, is_swept=True, vertices=self.vertices, collision=self.collision, bvh=self.bvh)
        kernel_refit_bvh(self.bvh)
        find_volume_contacts(
            is_ccd=True,
            vertices=self.vertices,
            collision=self.collision,
            bvh=self.bvh,
            options=self.config,
            errno=self.errno,
        )
        contact_faces = qd_to_torch(self.collision.contact_faces, transpose=True)
        active = (contact_faces >= 0).nonzero()
        volume_batches = active[:, 0]
        volume_pairs = torch.cat(
            (
                qd_to_torch(self.collision.surface_indices, transpose=True)[active[:, 1], None],
                qd_to_torch(self.collision.faces, transpose=True)[contact_faces[active[:, 0], active[:, 1]]],
            ),
            dim=1,
        )
        is_rebuild_required = True
        has_iteration_control = self.config.has_shell or self.options.convergence_tolerance > 0.0
        iteration = 0
        check_interval = 1 if has_iteration_control else self.options.iterations
        while iteration < self.options.iterations:
            if is_rebuild_required:
                pairs, batches = volume_pairs, volume_batches
                kinds = torch.full((len(pairs),), 2, dtype=gs.tc_int, device=gs.device)
                if self.config.has_shell:
                    update_bounds(
                        is_previous=False,
                        is_swept=False,
                        vertices=self.vertices,
                        collision=self.collision,
                        bvh=self.bvh,
                    )
                    kernel_refit_bvh(self.bvh)
                    initialize_shell_bounds(self.vertices, self.collision, self.config)
                    empty_pairs = torch.empty((0, 4), dtype=gs.tc_int, device=gs.device)
                    empty_indices = torch.empty(0, dtype=gs.tc_int, device=gs.device)
                    find_shell_contacts(
                        empty_indices,
                        empty_pairs,
                        empty_indices,
                        empty_indices,
                        self.contact_counts,
                        is_fill=False,
                        vertices=self.vertices,
                        collision=self.collision,
                        bvh=self.bvh,
                        options=self.config,
                        errno=self.errno,
                    )
                    offsets = torch.cat(
                        (self.contact_counts.new_zeros(1), self.contact_counts.cumsum(0, dtype=gs.tc_int))
                    )
                    n_shell_contacts = int(offsets[-1].item())
                    shell_pairs = torch.empty((n_shell_contacts, 4), dtype=gs.tc_int, device=gs.device)
                    shell_batches = torch.empty(n_shell_contacts, dtype=gs.tc_int, device=gs.device)
                    shell_kinds = torch.empty(n_shell_contacts, dtype=gs.tc_int, device=gs.device)
                    find_shell_contacts(
                        offsets,
                        shell_pairs,
                        shell_batches,
                        shell_kinds,
                        self.contact_counts,
                        is_fill=True,
                        vertices=self.vertices,
                        collision=self.collision,
                        bvh=self.bvh,
                        options=self.config,
                        errno=self.errno,
                    )
                    pairs = torch.cat((pairs, shell_pairs))
                    batches = torch.cat((batches, shell_batches))
                    kinds = torch.cat((kinds, shell_kinds))
                    if iteration == 0:
                        apply_prediction(
                            is_shell=True,
                            scale=self.initial_scale,
                            vertices=self.vertices,
                            collision=self.collision,
                            errno=self.errno,
                        )
                n_contacts = len(pairs)
                slots = (pairs * self._B + batches[:, None]).reshape(n_contacts * 4)
                entries = torch.argsort(slots, stable=True)
                counts = torch.bincount(slots, minlength=self.n_vertices * self._B)
                vertex_offsets = torch.cat((self.contact_counts.new_zeros(1), counts.cumsum(0, dtype=gs.tc_int)))
            for i_s in range(min(check_interval, self.options.iterations - iteration)):
                solve_colors(
                    vertex_offsets,
                    entries,
                    pairs,
                    kinds,
                    self.substep_dt,
                    i_s == 0,
                    self.iteration,
                    self.vertices,
                    self.elements,
                    self.hinges,
                    self.planes,
                    self.colors,
                    self.config,
                    self.errno,
                )
            if not has_iteration_control:
                iteration += check_interval
                break
            status = qd_to_numpy(self.iteration.status, transpose=True)
            iteration += int(status[1])
            is_rebuild_required = bool(status[0])
            if not is_rebuild_required and not status[2]:
                break
            check_interval = 1 if is_rebuild_required else min(2 * check_interval, 8)
        finish(
            pairs,
            kinds,
            batches,
            self.substep_dt,
            self._gravity,
            self.vertices,
            self.elements,
            self.hinges,
            self.planes,
            self.config,
            self.errno,
        )
        validate_state(self.vertices, self.elements, self.hinges, self.errno)

    def substep_post_coupling(self, f):
        pass

    def reset_grad(self):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def check_errno(self):
        if self.is_active and self.errno is not None:
            code = qd_to_numpy(self.errno, transpose=True).item()
            if code & ErrorCode.INVALID_VBD_STATE:
                gs.raise_exception("VBD encountered an invalid deformation. Reset the scene before continuing.")
            if (
                self.options.is_contact_diagnostics_enabled
                and code & ErrorCode.VBD_CONTACT_PENETRATION
                and not self.has_warned_penetration
            ):
                gs.logger.warning(
                    "VBD detected interpenetration. Simulation continues; consider increasing "
                    "VBDOptions.contact_d_hat to activate contact earlier."
                )
                self.has_warned_penetration = True

    def get_state(self, f):
        if not self.is_active:
            return None
        self.check_errno()
        return VBDState(
            read_field(self.vertices.pos, self._B),
            read_field(self.vertices.vel, self._B),
            read_field(self.vertices.forces, self._B),
            read_field(self.vertices.is_fixed, self._B),
            read_field(self.vertices.fixed_pos, self._B),
            read_field(self.elements.plastic_inverse, self._B),
            read_field(self.elements.plastic_strain, self._B),
            read_field(self.hinges.rest, self._B),
            read_field(self.hinges.strain, self._B),
        )

    def get_entity_state(self, entity):
        state = self.get_state(0)
        vs = slice(entity.v_start, entity.v_start + entity.n_vertices)
        es = slice(entity._el_start, entity._el_start + entity.n_elements)
        hs = slice(entity._hinge_start, entity._hinge_start + entity._n_hinges)
        return VBDState(
            state.pos[:, vs],
            state.vel[:, vs],
            state.forces[:, vs],
            state.is_fixed[:, vs],
            state.fixed_pos[:, vs],
            state.plastic_inverse[:, es],
            state.plastic_strain[:, es],
            state.bending_rest[:, hs],
            state.bending_strain[:, hs],
        )

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def set_state(self, f, state, envs_idx=None):
        envs_idx = self.scene._sanitize_envs_idx(envs_idx)
        if len(envs_idx) != self._B and qd_to_numpy(self.errno, transpose=True).item() & ErrorCode.INVALID_VBD_STATE:
            gs.raise_exception("Reset every environment after a VBD simulation failure.")
        fields = (
            self.vertices.pos,
            self.vertices.vel,
            self.vertices.forces,
            self.vertices.is_fixed,
            self.vertices.fixed_pos,
            self.elements.plastic_inverse,
            self.elements.plastic_strain,
            self.hinges.rest,
            self.hinges.strain,
        )
        values = (
            state.pos,
            state.vel,
            state.forces,
            state.is_fixed,
            state.fixed_pos,
            state.plastic_inverse,
            state.plastic_strain,
            state.bending_rest,
            state.bending_strain,
        )
        shapes = (
            (self._B, self.n_vertices, 3),
            (self._B, self.n_vertices, 3),
            (self._B, self.n_vertices, 3),
            (self._B, self.n_vertices),
            (self._B, self.n_vertices, 3),
            (self._B, self.n_elements, 3, 3),
            (self._B, self.n_elements),
            (self._B, self.hinges.n_hinges),
            (self._B, self.hinges.n_hinges),
        )
        for value, shape in zip(values, shapes):
            if not isinstance(value, torch.Tensor) or value.shape != shape:
                gs.raise_exception(f"VBD state tensor must have shape {shape}.")
            if not torch.isfinite(value).all():
                gs.raise_exception("VBD state must be finite.")
        for field, value in zip(fields, values):
            replace_field(field, value[envs_idx], envs_idx)
        if gs.backend == gs.metal:
            torch.mps.synchronize()
        self.errno.fill(0)
        self.has_warned_penetration = False
        validate_state(self.vertices, self.elements, self.hinges, self.errno)
        self.check_errno()

    def set_vertex_constraints(self, entity, indices, targets, envs_idx):
        self.check_errno()
        envs_idx = self.scene._sanitize_envs_idx(envs_idx)
        ids = indices + entity.v_start
        is_fixed = read_field(self.vertices.is_fixed, self._B)
        positions = read_field(self.vertices.fixed_pos, self._B)
        if targets is None:
            targets = read_field(self.vertices.pos, self._B)[envs_idx[:, None], ids]
        targets = torch.as_tensor(targets, dtype=gs.tc_float, device=gs.device)
        if not torch.isfinite(targets).all():
            gs.raise_exception("Vertex targets must be finite.")
        is_fixed[envs_idx[:, None], ids] = True
        positions[envs_idx[:, None], ids] = targets
        replace_field(self.vertices.is_fixed, is_fixed[envs_idx], envs_idx)
        replace_field(self.vertices.fixed_pos, positions[envs_idx], envs_idx)
        if gs.backend == gs.metal:
            torch.mps.synchronize()

    def remove_vertex_constraints(self, entity, indices, envs_idx):
        self.check_errno()
        envs_idx = self.scene._sanitize_envs_idx(envs_idx)
        values = read_field(self.vertices.is_fixed, self._B)
        values[envs_idx[:, None], indices + entity.v_start] = False
        replace_field(self.vertices.is_fixed, values[envs_idx], envs_idx)
        if gs.backend == gs.metal:
            torch.mps.synchronize()

    def set_vertex_forces(self, entity, indices, forces, envs_idx):
        self.check_errno()
        envs_idx = self.scene._sanitize_envs_idx(envs_idx)
        forces = torch.as_tensor(forces, dtype=gs.tc_float, device=gs.device).expand(len(envs_idx), len(indices), 3)
        if not torch.isfinite(forces).all():
            gs.raise_exception("Vertex forces must be finite.")
        values = read_field(self.vertices.forces, self._B)[envs_idx]
        values[:, entity.v_start : entity.v_start + entity.n_vertices] = 0
        values.index_add_(1, indices + entity.v_start, forces)
        replace_field(self.vertices.forces, values, envs_idx)
        if gs.backend == gs.metal:
            torch.mps.synchronize()

    def clear_vertex_forces(self, entity, indices, envs_idx):
        self.check_errno()
        envs_idx = self.scene._sanitize_envs_idx(envs_idx)
        values = read_field(self.vertices.forces, self._B)
        values[envs_idx[:, None], indices + entity.v_start] = 0
        replace_field(self.vertices.forces, values[envs_idx], envs_idx)
        if gs.backend == gs.metal:
            torch.mps.synchronize()

    def get_state_render(self, f):
        self.check_errno()
        render_positions(self.render_indices, self.envs_offset, self.render_pos, self.vertices)
        return self.render_pos, None, self.render_faces

    def get_tetrahedral_state_render(self, f):
        self.check_errno()
        render_positions(self.vertex_indices, self.envs_offset, self.render_tet_pos, self.vertices)
        return self.render_tet_pos

    @property
    def is_active(self):
        return bool(self.entities)

    @property
    def n_vertices(self):
        return sum(entity.n_vertices for entity in self.entities)

    @property
    def n_elements(self):
        return sum(entity.n_elements for entity in self.entities)

    @property
    def n_vverts(self):
        return sum(geom.n_vverts for entity in self.entities for geom in entity.vgeoms)

    @property
    def n_vfaces(self):
        return sum(geom.n_vfaces for entity in self.entities for geom in entity.vgeoms)
