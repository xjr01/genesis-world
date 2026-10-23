import numpy as np

import igl
import trimesh

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.entities.base_entity import Entity
from genesis.engine.entities.deformable_geom import DeformableVisGeom
from genesis.utils.misc import sanitize_index, tensor_to_array


def vertex_indices(indices, n_vertices):
    """Normalize an entity's local vertex selection and reject out-of-range entries."""
    indices = sanitize_index(indices, -1, n_vertices, 0, "verts_idx_local")
    if ((indices < 0) | (indices >= n_vertices)).any():
        gs.raise_exception("Vertex indices are out of range.")
    return indices


class VBDEntity(Entity):
    """A triangle shell or tetrahedral solid advanced by vertex block descent (VBD).

    Simulation vertex indices are local to the entity. surface_triangles and each visual geom's sim_verts_idx retain
    the mapping between the simulation and the original surface. Vertex controls accept batched values through
    envs_idx; get_state always retains the leading environment dimension.
    """

    def __init__(self, idx, scene, morph, solver, material, surface, name=None):
        super().__init__(idx, scene, morph, solver, material, surface, name=name)
        meshes = gs.Mesh.from_morph_surface(morph, surface)
        vertices, faces, maps = mu.merge_submeshes([mesh.verts for mesh in meshes], [mesh.faces for mesh in meshes])
        if isinstance(material, gs.materials.VBD.Cloth):
            elements = faces
        elif isinstance(morph, gs.morphs.TetrahedralMesh):
            vertices = np.array(morph.vertices)
            elements = np.array(morph.elements)
        else:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            if not mesh.is_watertight:
                gs.raise_exception("VBD volume meshes must have a closed surface.")
            surface_vertices = vertices
            vertices, elements = eu.mesh_to_elements(mesh, tet_cfg=mu.generate_tetgen_config_from_morph(morph))
            if not np.allclose(vertices[: len(surface_vertices)], surface_vertices):
                gs.raise_exception("Tetrahedralization must preserve input surface vertices and their ordering.")
        if not len(vertices) or not len(elements) or not np.isfinite(vertices).all():
            gs.raise_exception("VBD meshes require finite vertices and nonempty elements.")
        quat = gu.transform_quat_by_quat(np.array(morph.offset_quat), np.array(morph.quat))
        center = vertices.mean(axis=0)
        vertices = (vertices - center) @ gu.quat_to_R(quat).T + center + morph.pos
        vertices += gu.transform_by_quat(np.array(morph.offset_pos), np.array(morph.quat))
        self.init_positions = vertices
        self.elems = elements
        self._surface_triangles = (
            faces if isinstance(material, gs.materials.VBD.Cloth) else igl.boundary_facets(elements)[0]
        )
        self._v_start = solver.n_vertices
        self._el_start = solver.n_elements
        self._hinge_start = 0
        self._n_hinges = 0
        self._vgeoms = gs.List()
        vvert_start, vface_start = solver.n_vverts, solver.n_vfaces
        for mesh, indices in zip(meshes, maps):
            self._vgeoms.append(DeformableVisGeom(self, vvert_start, vface_start, mesh, indices))
            vvert_start += len(mesh.verts)
            vface_start += len(mesh.faces)
        self._init_is_fixed = np.zeros(self.n_vertices, dtype=gs.np_bool)
        if isinstance(morph, (gs.morphs.Primitive, gs.morphs.Mesh)):
            self._init_is_fixed.fill(morph.fixed)
        self._init_fixed_pos = vertices.copy()
        self._init_forces = np.zeros_like(vertices)

    def _get_morph_identifier(self):
        return self.morph.__class__.__name__

    @gs.assert_built
    def get_state(self):
        """Return a detached snapshot of deformation, controls and plastic history."""
        return self.solver.get_entity_state(self)

    def set_vertex_constraints(self, verts_idx_local, target_poss=None, envs_idx=None):
        """Fix selected vertices at target_poss, or their current positions when targets are omitted.

        Before build, controls form part of the initial state restored by scene.reset(). After build, targets apply
        at the next substep and retain the existing deformation and plastic history of free vertices.
        """
        indices = vertex_indices(verts_idx_local, self.n_vertices)
        if self.is_built:
            self.solver.set_vertex_constraints(self, indices, target_poss, envs_idx)
        else:
            if envs_idx is not None:
                gs.raise_exception("Environment selection is available after scene.build().")
            indices = tensor_to_array(indices)
            if target_poss is not None:
                targets = tensor_to_array(target_poss)
                if not np.isfinite(targets).all():
                    gs.raise_exception("Vertex targets must be finite.")
                self._init_fixed_pos[indices] = targets
            self._init_is_fixed[indices] = True

    def remove_vertex_constraints(self, verts_idx_local=None, envs_idx=None):
        """Release selected vertices, preserving their current positions and material history."""
        indices = vertex_indices(verts_idx_local, self.n_vertices)
        if self.is_built:
            self.solver.remove_vertex_constraints(self, indices, envs_idx)
        else:
            if envs_idx is not None:
                gs.raise_exception("Environment selection is available after scene.build().")
            self._init_is_fixed[tensor_to_array(indices)] = False

    def set_vertex_forces(self, forces, verts_idx_local, envs_idx=None):
        """Replace the complete persistent force set in the selected environments, in world-space newtons.

        Repeated vertex indices sum their forces. Unlisted vertices receive zero force. Updating the force set
        preserves positions, velocities and plastic history; fixed vertices remain fixed under load.
        """
        indices = vertex_indices(verts_idx_local, self.n_vertices)
        if self.is_built:
            self.solver.set_vertex_forces(self, indices, forces, envs_idx)
        else:
            if envs_idx is not None:
                gs.raise_exception("Environment selection is available after scene.build().")
            values = tensor_to_array(forces)
            if not np.isfinite(values).all():
                gs.raise_exception("Vertex forces must be finite.")
            self._init_forces.fill(0)
            np.add.at(self._init_forces, tensor_to_array(indices), np.broadcast_to(values, (len(indices), 3)))

    def clear_vertex_forces(self, verts_idx_local=None, envs_idx=None):
        """Clear persistent forces on selected vertices, or the entire entity when indices are omitted."""
        indices = vertex_indices(verts_idx_local, self.n_vertices)
        if self.is_built:
            self.solver.clear_vertex_forces(self, indices, envs_idx)
        else:
            if envs_idx is not None:
                gs.raise_exception("Environment selection is available after scene.build().")
            self._init_forces[tensor_to_array(indices)] = 0

    @property
    def n_vertices(self):
        return len(self.init_positions)

    @property
    def n_elements(self):
        return len(self.elems)

    @property
    def v_start(self):
        return self._v_start

    @property
    def surface_triangles(self):
        return self._surface_triangles

    @property
    def vgeoms(self):
        return self._vgeoms
