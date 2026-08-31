import numpy as np
import torch

import igl

import quadrants as qd

import genesis as gs
from genesis.engine.boundaries import FloorBoundary
from genesis.engine.bvh import STACK_SIZE
from genesis.engine.entities.fem_entity import FEMEntity
from genesis.engine.states.solvers import FEMSolverState
import genesis.utils.array_class as array_class
from genesis.utils.geom import qd_inv_transform_by_trans_quat, qd_transform_by_quat, qd_transform_quat_by_quat
from genesis.utils.misc import qd_to_torch
import genesis.utils.sdf as sdf
from genesis.utils.triangle_qd import (
    triangle_triangle_intersection,
    triangle_triangle_previous_separating_correction,
)

from .base_solver import Solver

FEM_RIGID_SURFACE_PROJECTION_ITERATIONS = 16


@qd.data_oriented
class FEMSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._floor_height = options.floor_height
        self._damping = options.damping
        self._use_implicit_solver = options.use_implicit_solver
        self._n_newton_iterations = options.n_newton_iterations
        self._newton_dx_threshold = options.newton_dx_threshold
        self._n_pcg_iterations = options.n_pcg_iterations
        self._pcg_threshold = options.pcg_threshold
        self._n_linesearch_iterations = options.n_linesearch_iterations
        self._linesearch_c = options.linesearch_c
        self._linesearch_tau = options.linesearch_tau
        self._damping_alpha = options.damping_alpha
        self._damping_beta = options.damping_beta
        self._enable_vertex_constraints = options.enable_vertex_constraints

        # use scaled volume for better numerical stability, similar to p_vol_scale in mpm
        self._vol_scale = float(1e4)

        # materials
        self._mats = list()
        self._mats_idx = list()
        self._mats_update_stress = list()
        self._mats_compute_energy_gradient_hessian = list()
        self._mats_compute_energy = list()

        # boundary
        self.setup_boundary()

        # lazy initialization
        self._constraints_initialized = False
        self._has_tetrahedral_visual = False
        self._is_implicit_rigid_projection_enabled = False
        self.vertices_render = None

    def setup_boundary(self):
        self.boundary = FloorBoundary(height=self._floor_height)

    def init_batch_fields(self):
        self.batch_active = qd.field(dtype=gs.qd_bool, shape=(self._B,), needs_grad=False)
        self.batch_pcg_active = qd.field(dtype=gs.qd_bool, shape=(self._B,), needs_grad=False)
        self.batch_linesearch_active = qd.field(dtype=gs.qd_bool, shape=(self._B,), needs_grad=False)

        pcg_state = qd.types.struct(
            rTr=gs.qd_float,
            rTz=gs.qd_float,
            rTr_new=gs.qd_float,
            rTz_new=gs.qd_float,
            pTAp=gs.qd_float,
            alpha=gs.qd_float,
            beta=gs.qd_float,
        )
        self.pcg_state = pcg_state.field(shape=(self._B,), needs_grad=False, layout=qd.Layout.SOA)

        linesearch_state = qd.types.struct(
            prev_energy=gs.qd_float,
            energy=gs.qd_float,
            step_size=gs.qd_float,
            m=gs.qd_float,
        )
        self.linesearch_state = linesearch_state.field(shape=(self._B,), needs_grad=False, layout=qd.Layout.SOA)

    def init_element_fields(self):
        # element state in vertices
        element_state_v = qd.types.struct(
            pos=gs.qd_vec3,  # position
            vel=gs.qd_vec3,  # velocity
        )

        # element state in elements
        element_state_el = qd.types.struct(
            actu=gs.qd_float,  # actuation
        )

        # element state without gradient
        element_state_el_ng = qd.types.struct(
            active=gs.qd_bool,
        )

        # element info (properties that remain static through time)
        element_info = qd.types.struct(
            el2v=gs.qd_ivec4,  # vertex index of an element
            mu=gs.qd_float,  # lame parameters (1)
            lam=gs.qd_float,  # lame parameters (2)
            mass_scaled=gs.qd_float,  # scaled element mass. The real mass is mass_scaled / self._vol_scale
            mat_idx=gs.qd_int,  # material model index
            B=gs.qd_mat3,  # inverse of the deformation gradient at rest state
            V=gs.qd_float,  # rest volume of the element
            V_scaled=gs.qd_float,  # scaled rest volume of the element
            friction_mu=gs.qd_float,  # friction coefficient for contact
            # for muscle
            muscle_group=gs.qd_int,
            muscle_direction=gs.qd_vec3,
        )

        # element state for energy
        element_state_el_energy = qd.types.struct(
            energy=gs.qd_float,  # energy density for the element
            gradient=gs.qd_mat3,  # gradient density for the element, del energy / del F
        )

        element_state_v_energy = qd.types.struct(
            inertia=gs.qd_vec3,  # inertia for the vertex
            force=gs.qd_vec3,
        )

        element_v_info = qd.types.struct(
            mass=gs.qd_float,  # mass of the vertex
            mass_inv=gs.qd_float,  # inverse mass of the vertex
            mass_over_dt2=gs.qd_float,  # scaled mass of the vertex over dt^2
            friction_mu=gs.qd_float,  # friction coefficient for contact
        )

        pcg_state_v = qd.types.struct(
            diag3x3=gs.qd_mat3,  # diagonal 3-by-3 block of the hessian
            prec=gs.qd_mat3,  # preconditioner
            x=gs.qd_vec3,  # solution vector
            r=gs.qd_vec3,  # residual vector
            z=gs.qd_vec3,  # preconditioned residual vector
            p=gs.qd_vec3,  # search direction vector
            Ap=gs.qd_vec3,  # matrix-vector product
        )

        linesearch_state_v = qd.types.struct(
            x_prev=gs.qd_vec3,  # solution vector
        )

        # construct field
        self.elements_v = element_state_v.field(
            shape=(self.sim.substeps_local + 1, self.n_vertices, self._B),
            needs_grad=True,
            layout=qd.Layout.SOA,
        )
        self.elements_el = element_state_el.field(
            shape=(self.sim.substeps_local + 1, self.n_elements, self._B),
            needs_grad=True,
            layout=qd.Layout.SOA,
        )
        self.elements_el_ng = element_state_el_ng.field(
            shape=(self.sim.substeps_local + 1, self.n_elements, self._B),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )
        self.elements_i = element_info.field(
            shape=(self.n_elements),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        self.elements_el_energy = element_state_el_energy.field(
            shape=(self._B, self.n_elements),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        self.elements_el_hessian = qd.field(shape=(self._B, 3, 3, self.n_elements), dtype=gs.qd_mat3)

        self.elements_v_energy = element_state_v_energy.field(
            shape=(self._B, self.n_vertices),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        self.elements_v_info = element_v_info.field(
            shape=(self.n_vertices),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        self.pcg_state_v = pcg_state_v.field(
            shape=(self._B, self.n_vertices),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

        self.linesearch_state_v = linesearch_state_v.field(
            shape=(self._B, self.n_vertices),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

    def init_surface_fields(self):
        n_surfaces_max = self.n_surfaces

        # surface info (for coupling)
        surface_state = qd.types.struct(
            tri2v=gs.qd_ivec3,  # vertex index of a triangle
            tri2el=gs.qd_int,  # element index of a triangle
            active=gs.qd_bool,
        )

        self.surface = surface_state.field(
            shape=(n_surfaces_max),
            needs_grad=False,
            layout=qd.Layout.SOA,
        )

    def init_vvert_fields(self):
        """Allocate the render geometry of every visual geom of every entity, laid out back-to-back.

        Several vverts may stand for a single simulated vertex, so each one carries its own UVs and gathers its
        position through 'vert_idx' (see 'FEMVisGeom'). A contiguous layout lets a renderer consume positions, UVs and
        topology as three flat arrays.
        """
        struct_vvert_info = qd.types.struct(
            vert_idx=gs.qd_int,  # simulated vertex standing for this vvert
        )
        self.vverts_info = struct_vvert_info.field(shape=(max(self._n_vverts, 1),), layout=qd.Layout.SOA)

        # environment-offset vvert positions
        struct_vvert_state_render = qd.types.struct(
            pos=gs.qd_vec3,
        )
        self.vverts_render = struct_vvert_state_render.field(
            shape=(max(self._n_vverts, 1), self._B), layout=qd.Layout.SOA
        )

        self._has_tetrahedral_visual = any(entity.surface.vis_mode == "tetrahedral" for entity in self.entities)
        if self._has_tetrahedral_visual:
            self.vertices_render = struct_vvert_state_render.field(
                shape=(self.n_vertices, self._B), layout=qd.Layout.SOA
            )

        # static, shared across all batch envs
        self.vverts_uvs = qd.field(dtype=gs.qd_vec2, shape=(max(self._n_vverts, 1),))

        # static, in the solver's global vvert space
        self.vfaces_indices = qd.field(dtype=gs.qd_ivec3, shape=(max(self._n_vfaces, 1),))

    def _init_surface_info(self):
        self.vertices_on_surface = qd.field(dtype=gs.qd_bool, shape=(self.n_vertices,))
        self.elements_on_surface = qd.field(dtype=gs.qd_bool, shape=(self.n_elements,))
        self.compute_surface_vertices()
        self.compute_surface_elements()
        vertices_on_surface_np = self.vertices_on_surface.to_numpy()
        elements_on_surface_np = self.elements_on_surface.to_numpy()
        (surface_vertices_np,) = vertices_on_surface_np.nonzero()
        self.surface_vertices = qd.field(
            dtype=qd.i32,
            shape=(len(surface_vertices_np),),
            needs_grad=False,
        )
        self.surface_vertices.from_numpy(surface_vertices_np.astype(np.int32, copy=False))
        (surface_elements_np,) = elements_on_surface_np.nonzero()
        self.surface_elements = qd.field(
            dtype=qd.i32,
            shape=(len(surface_elements_np),),
            needs_grad=False,
        )
        self.surface_elements.from_numpy(surface_elements_np.astype(np.int32, copy=False))

        surface_triangles_np = self.surface.tri2v.to_numpy()
        pos_np = self.elements_v.pos.to_numpy()[0, :, 0, :][surface_vertices_np]
        surface_vertices_mapping = np.full(self.n_vertices, -1, dtype=np.int32)
        surface_vertices_mapping[surface_vertices_np] = np.arange(len(surface_vertices_np))
        mass = igl.massmatrix(pos_np, surface_vertices_mapping[surface_triangles_np])
        surface_vert_mass_np = mass.diagonal().astype(gs.np_float, copy=False)
        self.surface_vert_mass = qd.field(
            dtype=gs.qd_float,
            shape=(len(surface_vertices_np),),
            needs_grad=False,
        )
        self.surface_vert_mass.from_numpy(surface_vert_mass_np)

    @qd.kernel
    def compute_surface_vertices(self):
        for i_v in range(self.n_vertices):
            self.vertices_on_surface[i_v] = False

        for i_s in range(self.n_surfaces):
            tri2v = self.surface[i_s].tri2v
            for i in qd.static(range(3)):
                self.vertices_on_surface[tri2v[i]] = True

    @qd.kernel
    def compute_surface_elements(self):
        for i_e in range(self.n_elements):
            i_v = self.elements_i[i_e].el2v
            self.elements_on_surface[i_e] = (
                self.vertices_on_surface[i_v[0]]
                or self.vertices_on_surface[i_v[1]]
                or self.vertices_on_surface[i_v[2]]
                or self.vertices_on_surface[i_v[3]]
            )

    def init_ckpt(self):
        self._ckpt = dict()

    def init_constraints(self):
        self._constraints_initialized = True

        vertex_constraint_info = qd.types.struct(
            is_constrained=gs.qd_bool,  # boolean flag indicating if vertex is constrained
            target_pos=gs.qd_vec3,  # target position for the constraint
            is_soft_constraint=gs.qd_bool,  # use spring for soft constraints
            stiffness=gs.qd_float,  # spring stiffness
            link_idx=gs.qd_int,  # index of the rigid link (-1 if not linked)
            link_offset_pos=gs.qd_vec3,  # offset position of link
            link_init_quat_inv=gs.qd_vec4,  # inverse link rotation when the constraint is created
        )

        # FIXME: AOS, which does not match other Genesis structs. Old, untested code. We prefer not to touch for now.
        self.vertex_constraints = vertex_constraint_info.field(
            shape=(self.n_vertices, self._B), needs_grad=False, layout=qd.Layout.AOS
        )

        self.vertex_constraints.is_constrained.fill(False)
        self.vertex_constraints.link_idx.fill(-1)

    def reset_grad(self):
        self.elements_v.grad.fill(0)
        self.elements_el.grad.fill(0)

        for entity in self._entities:
            entity.reset_grad()

    def build(self):
        super().build()

        self.n_envs = self.sim.n_envs
        self._B = self.sim._B
        self.tet_wrong_order = qd.field(dtype=gs.qd_bool, shape=(), needs_grad=False)

        # batch fields
        self.init_batch_fields()

        # rendering
        self.envs_offset = qd.Vector.field(3, dtype=qd.f32, shape=self._B)
        self.envs_offset.from_numpy(self._scene.envs_offset.astype(np.float32))

        # elements and bodies
        self._n_elements_max = self.n_elements
        self._n_vertices_max = self.n_vertices
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces
        if self.n_elements_max > 0:
            self.init_element_fields()
            self.init_surface_fields()
            self.init_vvert_fields()
            self.init_ckpt()

            for entity in self._entities:
                entity._add_to_solver()

        for mat in self._mats:
            mat.build(self)

        if self.n_elements_max > 0:
            self._init_surface_info()
            if self.tet_wrong_order[None]:
                raise RuntimeError(
                    "The order of vertices in the tetrahedral elements is not correct. "
                    "Please check the input mesh or the FEM solver implementation."
                )

        if self.n_vertices_max > 0 and self._enable_vertex_constraints and not self._constraints_initialized:
            self.init_constraints()

        # FIXME: _gravity must be a raw qd.field() -- see comment in mpm_solver.py
        if self._gravity is not None:
            gravity = self._gravity.to_numpy()
            self._gravity = qd.field(dtype=gs.qd_vec3, shape=(self._B,))
            self._gravity.from_numpy(gravity)

    @property
    def is_active(self):
        return self.n_elements_max > 0

    def add_entity(self, idx, material, morph, surface, name: str | None = None) -> "FEMEntity":
        # add material's update methods if not matching any existing material
        exist = False
        for mat in self._mats:
            if material == mat:
                material.idx = mat.idx
                exist = True
                break
        self._mats.append(material)
        if not exist:
            material.idx = len(self._mats_idx)
            self._mats_idx.append(material.idx)
            self._mats_update_stress.append(material.update_stress)
            self._mats_compute_energy_gradient_hessian.append(material.compute_energy_gradient_hessian)
            self._mats_compute_energy.append(material.compute_energy)

        # create entity
        entity = FEMEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            v_start=self.n_vertices,
            el_start=self.n_elements,
            s_start=self.n_surfaces,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            name=name,
        )

        self._entities.append(entity)
        return entity

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def init_pos_and_vel(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            self.elements_v[f + 1, i_v, i_b].pos = self.elements_v[f, i_v, i_b].pos
            self.elements_v[f + 1, i_v, i_b].vel = self.elements_v[f, i_v, i_b].vel

    @qd.kernel
    def compute_vel(self, f: qd.i32):
        for i_e, i_b in qd.ndrange(self.n_elements, self._B):
            i_v0, i_v1, i_v2, i_v3 = self.elements_i[i_e].el2v
            pos_v0 = self.elements_v[f, i_v0, i_b].pos
            pos_v1 = self.elements_v[f, i_v1, i_b].pos
            pos_v2 = self.elements_v[f, i_v2, i_b].pos
            pos_v3 = self.elements_v[f, i_v3, i_b].pos
            D = qd.Matrix.cols([pos_v0 - pos_v3, pos_v1 - pos_v3, pos_v2 - pos_v3])

            V_scaled = self.elements_i[i_e].V_scaled
            B = self.elements_i[i_e].B
            F = D @ B
            J = F.determinant()

            stress = qd.Matrix.zero(gs.qd_float, 3, 3)
            for mat_idx in qd.static(self._mats_idx):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    stress = self._mats_update_stress[mat_idx](
                        mu=self.elements_i[i_e].mu,
                        lam=self.elements_i[i_e].lam,
                        J=J,
                        F=F,
                        actu=self.elements_el[f, i_e, i_b].actu,
                        m_dir=self.elements_i[i_e].muscle_direction,
                    )

            verts = self.elements_i[i_e].el2v
            mass_scaled = self.elements_i[i_e].mass_scaled
            H_scaled = -V_scaled * stress @ B.transpose()
            for k in qd.static(range(3)):
                force_scaled = qd.Vector([H_scaled[j, k] for j in range(3)])

                # store so forces can be read out
                self.elements_v_energy[i_b, verts[k]].force = force_scaled

                dv = self.substep_dt * force_scaled / mass_scaled
                self.elements_v[f + 1, verts[k], i_b].vel += dv
                self.elements_v[f + 1, verts[3], i_b].vel -= dv

    @qd.kernel
    def apply_uniform_force(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            # NOTE: damping should only be applied to velocity from internal force and thus come first here
            #       given the immediate previous function call is compute_internal_vel --> however, shouldn't
            #       be done at dv only and need to wait for all elements updated (cannot be in the compute_internal_vel kernel)
            #       however, this inevitably damp the gravity.
            self.elements_v[f + 1, i_v, i_b].vel *= qd.exp(-self.substep_dt * self.damping)
            # Add gravity (avoiding damping on gravity)
            self.elements_v[f + 1, i_v, i_b].vel += self.substep_dt * self._gravity[i_b]

    @qd.kernel
    def compute_pos(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            self.elements_v[f + 1, i_v, i_b].pos = (
                self.substep_dt * self.elements_v[f + 1, i_v, i_b].vel + self.elements_v[f, i_v, i_b].pos
            )

    @qd.kernel
    def precompute_material_data(self, f: qd.i32):
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            J, F = self._compute_ele_J_F(f, i_e, i_b)  # use last time step's pos to compute
            for mat_idx in qd.static(self._mats_idx):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    self._mats[mat_idx].pre_compute(J=J, F=F, i_e=i_e, i_b=i_b)

    @qd.kernel
    def init_pos_and_inertia(self, f: qd.i32):
        dt2 = self.substep_dt**2
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            if qd.static(self._enable_vertex_constraints):
                if self.vertex_constraints.is_constrained[i_v, i_b]:
                    self.elements_v[f + 1, i_v, i_b].pos = self.vertex_constraints.target_pos[i_v, i_b]
                    self.elements_v_energy[i_b, i_v].inertia = self.vertex_constraints.target_pos[i_v, i_b]
                else:
                    self.elements_v_energy[i_b, i_v].inertia = (
                        self.elements_v[f, i_v, i_b].pos
                        + self.elements_v[f, i_v, i_b].vel * self.substep_dt
                        + self._gravity[i_b] * dt2
                    )
                    self.elements_v[f + 1, i_v, i_b].pos = self.elements_v[f, i_v, i_b].pos
            else:
                self.elements_v_energy[i_b, i_v].inertia = (
                    self.elements_v[f, i_v, i_b].pos
                    + self.elements_v[f, i_v, i_b].vel * self.substep_dt
                    + self._gravity[i_b] * dt2
                )
                self.elements_v[f + 1, i_v, i_b].pos = self.elements_v[f, i_v, i_b].pos

    @qd.func
    def _compute_ele_J_F(self, f: qd.i32, i_e: qd.i32, i_b: qd.i32):
        """
        Compute the determinant (J) and deformation gradient (F) for an element.
        """
        i_v0, i_v1, i_v2, i_v3 = self.elements_i[i_e].el2v
        pos_v0 = self.elements_v[f, i_v0, i_b].pos
        pos_v1 = self.elements_v[f, i_v1, i_b].pos
        pos_v2 = self.elements_v[f, i_v2, i_b].pos
        pos_v3 = self.elements_v[f, i_v3, i_b].pos
        D = qd.Matrix.cols([pos_v0 - pos_v3, pos_v1 - pos_v3, pos_v2 - pos_v3])

        B = self.elements_i[i_e].B
        F = D @ B
        J = F.determinant()

        return J, F

    @qd.kernel
    def compute_ele_hessian_gradient(self, f: qd.i32):
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_active[i_b]:
                continue

            J, F = self._compute_ele_J_F(f + 1, i_e, i_b)

            for mat_idx in qd.static(self._mats_idx):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    if self._mats[mat_idx]._hessian_ready:
                        (
                            self.elements_el_energy[i_b, i_e].energy,
                            self.elements_el_energy[i_b, i_e].gradient,
                        ) = self._mats[mat_idx].compute_energy_gradient(
                            mu=self.elements_i[i_e].mu,
                            lam=self.elements_i[i_e].lam,
                            J=J,
                            F=F,
                            actu=self.elements_el[f, i_e, i_b].actu,
                            m_dir=self.elements_i[i_e].muscle_direction,
                            i_e=i_e,
                            i_b=i_b,
                        )
                    else:
                        (
                            self.elements_el_energy[i_b, i_e].energy,
                            self.elements_el_energy[i_b, i_e].gradient,
                        ) = self._mats[mat_idx].compute_energy_gradient_hessian(
                            mu=self.elements_i[i_e].mu,
                            lam=self.elements_i[i_e].lam,
                            J=J,
                            F=F,
                            actu=self.elements_el[f, i_e, i_b].actu,
                            m_dir=self.elements_i[i_e].muscle_direction,
                            i_e=i_e,
                            i_b=i_b,
                            hessian_field=self.elements_el_hessian,
                        )

    @qd.func
    def _func_compute_element_mapping_matrix(self, i_vs, B, i_b):
        """
        Compute the element mapping matrix S for an element.
        """
        S = qd.Matrix.zero(gs.qd_float, 4, 3)
        S[:3, :] = B
        S[3, :] = -B[0, :] - B[1, :] - B[2, :]

        if qd.static(self._enable_vertex_constraints):
            for i in qd.static(range(4)):
                if self.vertex_constraints.is_constrained[i_vs[i], i_b]:
                    S[i, :] = qd.Vector.zero(gs.qd_float, 3)
        return S

    @qd.func
    def _func_compute_ele_energy(self, f: qd.i32):
        """
        Compute the energy for each element in the batch. Should only be used in linesearch.
        """
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_linesearch_active[i_b]:
                continue

            J, F = self._compute_ele_J_F(f + 1, i_e, i_b)

            for mat_idx in qd.static(self._mats_idx):
                if self.elements_i[i_e].mat_idx == mat_idx:
                    self.elements_el_energy[i_b, i_e].energy = self._mats[mat_idx].compute_energy(
                        mu=self.elements_i[i_e].mu,
                        lam=self.elements_i[i_e].lam,
                        J=J,
                        F=F,
                        actu=self.elements_el[f, i_e, i_b].actu,
                        m_dir=self.elements_i[i_e].muscle_direction,
                        i_e=i_e,
                        i_b=i_b,
                    )

            # add linearized damping energy
            if self._damping_beta > gs.EPS:
                damping_beta_over_dt = self._damping_beta / self._substep_dt
                i_vs = self.elements_i[i_e].el2v
                B = self.elements_i[i_e].B
                S = self._func_compute_element_mapping_matrix(i_vs, B, i_b)

                x_diff = qd.Vector.zero(gs.qd_float, 12)
                for i in qd.static(range(4)):
                    x_diff[i * 3 : i * 3 + 3] = (
                        self.elements_v[f + 1, i_vs[i], i_b].pos - self.elements_v[f, i_vs[i], i_b].pos
                    )
                St_x_diff = qd.Vector.zero(gs.qd_float, 9)
                for i, j in qd.static(qd.ndrange(3, 4)):
                    St_x_diff[i * 3 : i * 3 + 3] += S[j, i] * x_diff[j * 3 : j * 3 + 3]

                H_St_x_diff = qd.Vector.zero(gs.qd_float, 9)
                for i, j in qd.static(qd.ndrange(3, 3)):
                    H_St_x_diff[i * 3 : i * 3 + 3] += (
                        self.elements_el_hessian[i_b, i, j, i_e] @ St_x_diff[j * 3 : j * 3 + 3]
                    )

                self.elements_el_energy[i_b, i_e].energy += 0.5 * damping_beta_over_dt * St_x_diff.dot(H_St_x_diff)

    @qd.kernel
    def accumulate_vertex_force_preconditioner(self, f: qd.i32):
        damping_alpha_dt = self._damping_alpha * self._substep_dt
        damping_alpha_factor = damping_alpha_dt + 1.0
        damping_beta_over_dt = self._damping_beta / self._substep_dt
        damping_beta_factor = damping_beta_over_dt + 1.0
        # inertia
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_active[i_b]:
                continue
            self.elements_v_energy[i_b, i_v].force = -self.elements_v_info[i_v].mass_over_dt2 * (
                (self.elements_v[f + 1, i_v, i_b].pos - self.elements_v_energy[i_b, i_v].inertia)
                + (self.elements_v[f + 1, i_v, i_b].pos - self.elements_v[f, i_v, i_b].pos) * damping_alpha_dt
            )
            self.pcg_state_v[i_b, i_v].diag3x3 = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i in qd.static(range(3)):
                self.pcg_state_v[i_b, i_v].diag3x3[i, i] = (
                    self.elements_v_info[i_v].mass_over_dt2 * damping_alpha_factor
                )

        # elastic
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_active[i_b]:
                continue
            V = self.elements_i[i_e].V
            B = self.elements_i[i_e].B
            gradient = self.elements_el_energy[i_b, i_e].gradient
            i_vs = self.elements_i[i_e].el2v
            S = self._func_compute_element_mapping_matrix(i_vs, B, i_b)
            force = -V * gradient @ S.transpose()

            # atomic
            for i in qd.static(range(4)):
                self.elements_v_energy[i_b, i_vs[i]].force += force[:, i]

            if self._damping_beta > gs.EPS:
                x_diff = qd.Vector.zero(gs.qd_float, 12)
                for i in qd.static(range(4)):
                    x_diff[i * 3 : i * 3 + 3] = (
                        self.elements_v[f + 1, i_vs[i], i_b].pos - self.elements_v[f, i_vs[i], i_b].pos
                    )
                St_x_diff = qd.Vector.zero(gs.qd_float, 9)
                for i, j in qd.static(qd.ndrange(3, 4)):
                    St_x_diff[i * 3 : i * 3 + 3] += S[j, i] * x_diff[j * 3 : j * 3 + 3]

                H_St_x_diff = qd.Vector.zero(gs.qd_float, 9)
                for i, j in qd.static(qd.ndrange(3, 3)):
                    H_St_x_diff[i * 3 : i * 3 + 3] += (
                        self.elements_el_hessian[i_b, i, j, i_e] @ St_x_diff[j * 3 : j * 3 + 3]
                    )
                S_H_St_x_diff = qd.Vector.zero(gs.qd_float, 12)
                for i, j in qd.static(qd.ndrange(4, 3)):
                    S_H_St_x_diff[i * 3 : i * 3 + 3] += S[i, j] * H_St_x_diff[j * 3 : j * 3 + 3]
                for i in qd.static(range(4)):
                    self.elements_v_energy[i_b, i_vs[i]].force += (
                        -damping_beta_over_dt * V * S_H_St_x_diff[i * 3 : i * 3 + 3]
                    )

            # diagonal 3-by-3 block of hessian
            for k, i, j in qd.ndrange(4, 3, 3):
                self.pcg_state_v[i_b, i_vs[k]].diag3x3 += (
                    V * damping_beta_factor * S[k, i] * S[k, j] * self.elements_el_hessian[i_b, i, j, i_e]
                )

        # inverse
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_active[i_b]:
                continue
            # Use 3-by-3 diagonal block inverse for preconditioner
            self.pcg_state_v[i_b, i_v].prec = self.pcg_state_v[i_b, i_v].diag3x3.inverse()

            # Other options for preconditioner:
            # Uncomment one of the following lines to test different preconditioners
            # Use identity for preconditioner
            # self.pcg_state_v[i_b, i_v].prec = qd.Matrix.identity(gs.qd_float, 3)

            # Use diagonal for preconditioner
            # self.pcg_state_v[i_b, i_v].prec = qd.Matrix([[1 / self.pcg_state_v[i_b, i_v].diag3x3[0, 0], 0, 0],
            #                                            [0, 1 / self.pcg_state_v[i_b, i_v].diag3x3[1, 1], 0],
            #                                            [0, 0, 1 / self.pcg_state_v[i_b, i_v].diag3x3[2, 2]]])

    @qd.func
    def compute_Ap(self):
        damping_alpha_dt = self._damping_alpha * self._substep_dt
        damping_alpha_factor = damping_alpha_dt + 1.0
        damping_beta_over_dt = self._damping_beta / self._substep_dt
        damping_beta_factor = damping_beta_over_dt + 1.0
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].Ap = (
                self.elements_v_info[i_v].mass_over_dt2 * damping_alpha_factor * self.pcg_state_v[i_b, i_v].p
            )

        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_pcg_active[i_b]:
                continue
            V = self.elements_i[i_e].V
            B = self.elements_i[i_e].B
            i_vs = self.elements_i[i_e].el2v
            S = self._func_compute_element_mapping_matrix(i_vs, B, i_b)

            p9 = qd.Vector([0.0] * 9, dt=gs.qd_float)

            for i, j in qd.static(qd.ndrange(3, 4)):
                p9[i * 3 : i * 3 + 3] = p9[i * 3 : i * 3 + 3] + S[j, i] * self.pcg_state_v[i_b, i_vs[j]].p

            new_p9 = qd.Vector([0.0] * 9, dt=gs.qd_float)

            for i, j in qd.static(qd.ndrange(3, 3)):
                new_p9[i * 3 : i * 3 + 3] = (
                    new_p9[i * 3 : i * 3 + 3] + self.elements_el_hessian[i_b, i, j, i_e] @ p9[j * 3 : j * 3 + 3]
                )

            # atomic
            for i in qd.static(range(4)):
                self.pcg_state_v[i_b, i_vs[i]].Ap += (
                    (S[i, 0] * new_p9[0:3] + S[i, 1] * new_p9[3:6] + S[i, 2] * new_p9[6:9]) * V * damping_beta_factor
                )

    @qd.kernel
    def init_pcg_solve(self):
        for i_b in range(self._B):
            self.batch_pcg_active[i_b] = self.batch_active[i_b]
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].rTr = 0.0
            self.pcg_state[i_b].rTz = 0.0
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].x = 0
            self.pcg_state_v[i_b, i_v].r = self.elements_v_energy[i_b, i_v].force
            self.pcg_state_v[i_b, i_v].z = self.pcg_state_v[i_b, i_v].prec @ self.pcg_state_v[i_b, i_v].r
            self.pcg_state_v[i_b, i_v].p = self.pcg_state_v[i_b, i_v].z
            qd.atomic_add(self.pcg_state[i_b].rTr, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].r))
            qd.atomic_add(self.pcg_state[i_b].rTz, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].z))
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr > self._pcg_threshold

    @qd.func
    def _func_one_pcg_iter(self):
        self.compute_Ap()

        # compute pTAp
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].pTAp = 0.0
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            qd.atomic_add(self.pcg_state[i_b].pTAp, self.pcg_state_v[i_b, i_v].p.dot(self.pcg_state_v[i_b, i_v].Ap))

        # compute alpha and update x, r, z, rTr, rTz
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].alpha = self.pcg_state[i_b].rTz / self.pcg_state[i_b].pTAp
            self.pcg_state[i_b].rTr_new = 0.0
            self.pcg_state[i_b].rTz_new = 0.0
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].x += self.pcg_state[i_b].alpha * self.pcg_state_v[i_b, i_v].p
            self.pcg_state_v[i_b, i_v].r -= self.pcg_state[i_b].alpha * self.pcg_state_v[i_b, i_v].Ap
            self.pcg_state_v[i_b, i_v].z = self.pcg_state_v[i_b, i_v].prec @ self.pcg_state_v[i_b, i_v].r
            qd.atomic_add(self.pcg_state[i_b].rTr_new, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].r))
            qd.atomic_add(self.pcg_state[i_b].rTz_new, self.pcg_state_v[i_b, i_v].r.dot(self.pcg_state_v[i_b, i_v].z))

        # check convergence
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.batch_pcg_active[i_b] = self.pcg_state[i_b].rTr_new > self._pcg_threshold

        # update beta, rTr, rTz
        for i_b in range(self._B):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state[i_b].beta = self.pcg_state[i_b].rTz_new / self.pcg_state[i_b].rTz
            self.pcg_state[i_b].rTr = self.pcg_state[i_b].rTr_new
            self.pcg_state[i_b].rTz = self.pcg_state[i_b].rTz_new

        # update p
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_pcg_active[i_b]:
                continue
            self.pcg_state_v[i_b, i_v].p = (
                self.pcg_state_v[i_b, i_v].z + self.pcg_state[i_b].beta * self.pcg_state_v[i_b, i_v].p
            )

    @qd.func
    def _func_is_vertex_projection_enabled(self, vertex_idx, env_idx):
        is_enabled = True
        if qd.static(self._enable_vertex_constraints):
            is_enabled = not self.vertex_constraints.is_constrained[vertex_idx, env_idx]
        return is_enabled

    @qd.func
    def _func_project_vertex_against_rigid(
        self,
        env_idx,
        pos,
        bvh_nodes: qd.template(),
        bvh_morton_codes: qd.template(),
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
        collider_info: array_class.ColliderInfo,
        surface_info: array_class.FEMRigidSurfaceInfo,
    ):
        corrected_pos = pos
        collision_normal = qd.Vector.zero(gs.qd_float, 3)
        is_collision_active = False
        for projection_pass in qd.static(range(2)):
            if qd.static(projection_pass == 1):
                collision_normal = qd.Vector.zero(gs.qd_float, 3)
                is_collision_active = False
            for projection_geom_slot in range(surface_info.projection_geoms_idx.shape[0]):
                geom_idx = surface_info.projection_geoms_idx[projection_geom_slot]
                corrected_pos, normal, is_active = sdf.sdf_func_project_vertex_outside_geom(
                    geom_idx,
                    env_idx,
                    corrected_pos,
                    bvh_nodes,
                    bvh_morton_codes,
                    dyn_state,
                    dyn_info,
                    rigid_info,
                    collider_info,
                    surface_info,
                )
                if qd.static(projection_pass == 1) and is_active:
                    collision_normal += normal
                    is_collision_active = True
        if collision_normal.norm_sqr() > rigid_info.EPS[None] ** 2:
            collision_normal = collision_normal.normalized()
        return corrected_pos, collision_normal, is_collision_active

    @qd.func
    def _func_project_pcg_positions(
        self,
        f,
        bvh_nodes: qd.template(),
        bvh_morton_codes: qd.template(),
        dyn_state: array_class.DynState,
        projection_state: array_class.FEMProjectionState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
        collider_info: array_class.ColliderInfo,
        surface_info: array_class.FEMRigidSurfaceInfo,
    ):
        for env_idx in range(self._B):
            if projection_state.is_processed[env_idx]:
                projection_state.has_changed[env_idx] = 0
                projection_state.has_contact[env_idx] = 0

        for env_idx, vertex_idx in qd.ndrange(self._B, self.n_vertices):
            if not projection_state.is_processed[env_idx]:
                continue
            self.pcg_state_v[env_idx, vertex_idx].Ap = qd.Vector.zero(gs.qd_float, 3)
            projection_state.normals[env_idx, vertex_idx] = qd.Vector.zero(gs.qd_float, 3)
            projection_state.is_active[env_idx, vertex_idx] = False
            if not self._func_is_vertex_projection_enabled(vertex_idx, env_idx):
                continue
            pos = self.elements_v[f + 1, vertex_idx, env_idx].pos + self.pcg_state_v[env_idx, vertex_idx].x
            corrected_pos, collision_normal, is_collision_active = self._func_project_vertex_against_rigid(
                env_idx,
                pos,
                bvh_nodes,
                bvh_morton_codes,
                dyn_state,
                dyn_info,
                rigid_info,
                collider_info,
                surface_info,
            )
            correction = corrected_pos - pos
            self.pcg_state_v[env_idx, vertex_idx].Ap = correction
            projection_state.normals[env_idx, vertex_idx] = collision_normal
            projection_state.is_active[env_idx, vertex_idx] = is_collision_active
            if correction.norm_sqr() > rigid_info.EPS[None] ** 2:
                self.pcg_state_v[env_idx, vertex_idx].x += correction
                qd.atomic_max(projection_state.has_changed[env_idx], 1)
            if is_collision_active:
                qd.atomic_max(projection_state.has_contact[env_idx], 1)

        for env_idx in range(self._B):
            if projection_state.is_processed[env_idx]:
                self.batch_pcg_active[env_idx] = projection_state.has_changed[env_idx] != 0
        for env_idx, vertex_idx in qd.ndrange(self._B, self.n_vertices):
            if projection_state.is_processed[env_idx] and projection_state.has_changed[env_idx] != 0:
                self.pcg_state_v[env_idx, vertex_idx].p = self.pcg_state_v[env_idx, vertex_idx].Ap

        self.compute_Ap()

        for env_idx in range(self._B):
            if projection_state.is_processed[env_idx] and (
                projection_state.has_changed[env_idx] != 0 or projection_state.has_contact[env_idx] != 0
            ):
                self.pcg_state[env_idx].rTr_new = 0.0
                self.pcg_state[env_idx].rTz_new = 0.0
        for env_idx, vertex_idx in qd.ndrange(self._B, self.n_vertices):
            if not projection_state.is_processed[env_idx] or (
                projection_state.has_changed[env_idx] == 0 and projection_state.has_contact[env_idx] == 0
            ):
                continue
            if projection_state.has_changed[env_idx] != 0:
                self.pcg_state_v[env_idx, vertex_idx].r -= self.pcg_state_v[env_idx, vertex_idx].Ap
            if projection_state.is_active[env_idx, vertex_idx]:
                normal = projection_state.normals[env_idx, vertex_idx]
                normal_component = self.pcg_state_v[env_idx, vertex_idx].r.dot(normal)
                if normal_component < 0.0:
                    self.pcg_state_v[env_idx, vertex_idx].r -= normal_component * normal
            self.pcg_state_v[env_idx, vertex_idx].z = (
                self.pcg_state_v[env_idx, vertex_idx].prec @ self.pcg_state_v[env_idx, vertex_idx].r
            )
            if projection_state.is_active[env_idx, vertex_idx]:
                normal = projection_state.normals[env_idx, vertex_idx]
                normal_component = self.pcg_state_v[env_idx, vertex_idx].z.dot(normal)
                if normal_component < 0.0:
                    self.pcg_state_v[env_idx, vertex_idx].z -= normal_component * normal
            self.pcg_state_v[env_idx, vertex_idx].p = self.pcg_state_v[env_idx, vertex_idx].z
            qd.atomic_add(
                self.pcg_state[env_idx].rTr_new,
                self.pcg_state_v[env_idx, vertex_idx].r.dot(self.pcg_state_v[env_idx, vertex_idx].r),
            )
            qd.atomic_add(
                self.pcg_state[env_idx].rTz_new,
                self.pcg_state_v[env_idx, vertex_idx].r.dot(self.pcg_state_v[env_idx, vertex_idx].z),
            )
        for env_idx in range(self._B):
            if not projection_state.is_processed[env_idx]:
                continue
            if projection_state.has_changed[env_idx] != 0 or projection_state.has_contact[env_idx] != 0:
                self.pcg_state[env_idx].rTr = self.pcg_state[env_idx].rTr_new
                self.pcg_state[env_idx].rTz = self.pcg_state[env_idx].rTz_new
                self.batch_pcg_active[env_idx] = self.pcg_state[env_idx].rTr > self._pcg_threshold
            else:
                self.batch_pcg_active[env_idx] = projection_state.is_pcg_active_saved[env_idx]

    @qd.kernel
    def one_pcg_iter(self):
        self._func_one_pcg_iter()

    @qd.kernel
    def project_initial_pcg_positions(
        self,
        f: qd.i32,
        bvh_nodes: qd.template(),
        bvh_morton_codes: qd.template(),
        dyn_state: array_class.DynState,
        projection_state: array_class.FEMProjectionState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
        collider_info: array_class.ColliderInfo,
        surface_info: array_class.FEMRigidSurfaceInfo,
    ):
        for env_idx in range(self._B):
            projection_state.is_processed[env_idx] = self.batch_active[env_idx]
            projection_state.is_pcg_active_saved[env_idx] = self.batch_pcg_active[env_idx]
        self._func_project_pcg_positions(
            f,
            bvh_nodes,
            bvh_morton_codes,
            dyn_state,
            projection_state,
            dyn_info,
            rigid_info,
            collider_info,
            surface_info,
        )

    @qd.kernel
    def one_projected_pcg_iter(
        self,
        f: qd.i32,
        bvh_nodes: qd.template(),
        bvh_morton_codes: qd.template(),
        dyn_state: array_class.DynState,
        projection_state: array_class.FEMProjectionState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
        collider_info: array_class.ColliderInfo,
        surface_info: array_class.FEMRigidSurfaceInfo,
    ):
        for env_idx in range(self._B):
            projection_state.is_processed[env_idx] = self.batch_pcg_active[env_idx]
        self._func_one_pcg_iter()
        for env_idx in range(self._B):
            projection_state.is_pcg_active_saved[env_idx] = self.batch_pcg_active[env_idx]
        self._func_project_pcg_positions(
            f,
            bvh_nodes,
            bvh_morton_codes,
            dyn_state,
            projection_state,
            dyn_info,
            rigid_info,
            collider_info,
            surface_info,
        )

    @qd.kernel
    def project_implicit_positions(
        self,
        f: qd.i32,
        bvh_nodes: qd.template(),
        bvh_morton_codes: qd.template(),
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
        collider_info: array_class.ColliderInfo,
        surface_info: array_class.FEMRigidSurfaceInfo,
        is_committed: qd.template(),
    ):
        for env_idx, vertex_idx in qd.ndrange(self._B, self.n_vertices):
            is_env_active = self.batch_active[env_idx]
            if qd.static(is_committed):
                is_env_active = True
            if is_env_active and self._func_is_vertex_projection_enabled(vertex_idx, env_idx):
                corrected_pos, _, _ = self._func_project_vertex_against_rigid(
                    env_idx,
                    self.elements_v[f + 1, vertex_idx, env_idx].pos,
                    bvh_nodes,
                    bvh_morton_codes,
                    dyn_state,
                    dyn_info,
                    rigid_info,
                    collider_info,
                    surface_info,
                )
                correction = corrected_pos - self.elements_v[f + 1, vertex_idx, env_idx].pos
                self.elements_v[f + 1, vertex_idx, env_idx].pos = corrected_pos
                if correction.norm_sqr() > rigid_info.EPS[None] ** 2:
                    self.elements_v[f + 1, vertex_idx, env_idx].vel += correction / self.substep_dt

    @qd.kernel
    def init_implicit_surface_projection(self, surface_state: array_class.FEMRigidSurfaceState):
        for env_idx in range(self._B):
            surface_state.is_active[env_idx] = True
            surface_state.has_intersection[env_idx] = 0

    @qd.kernel
    def detect_implicit_surface_intersections(
        self,
        f: qd.i32,
        bvh_nodes: qd.template(),
        bvh_morton_codes: qd.template(),
        dyn_state: array_class.DynState,
        surface_state: array_class.FEMRigidSurfaceState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
        surface_info: array_class.FEMRigidSurfaceInfo,
        is_audit: qd.template(),
        errno: qd.Tensor,
    ):
        """Detect deformable-rigid surface intersections and accumulate frictionless one-way corrections."""
        n_surface_faces = bvh_morton_codes.shape[1]
        if qd.static(is_audit):
            for env_idx in range(self._B):
                surface_state.has_intersection[env_idx] = 0
        for env_idx, surface_idx in qd.ndrange(self._B, self.n_surfaces):
            is_env_active = surface_state.is_active[env_idx]
            if qd.static(is_audit):
                is_env_active = True
            if not is_env_active or not self.surface[surface_idx].active:
                continue

            vertices_idx = self.surface[surface_idx].tri2v
            vertices_world = qd.Matrix.zero(gs.qd_float, 3, 3)
            previous_vertices_world = qd.Matrix.zero(gs.qd_float, 3, 3)
            previous_centroid_world = qd.Vector.zero(gs.qd_float, 3)
            for vertex_slot in qd.static(range(3)):
                vertex_idx = vertices_idx[vertex_slot]
                vertices_world[:, vertex_slot] = self.elements_v[f + 1, vertex_idx, env_idx].pos
                previous_vertices_world[:, vertex_slot] = self.elements_v[f, vertex_idx, env_idx].pos
                previous_centroid_world += previous_vertices_world[:, vertex_slot] / 3.0

            for surface_geom_slot in range(surface_info.surface_geoms_idx.shape[0]):
                geom_idx = surface_info.surface_geoms_idx[surface_geom_slot]
                geom_pos = dyn_state.geoms.pos[geom_idx, env_idx]
                geom_quat = dyn_state.geoms.quat[geom_idx, env_idx]
                atlas_offset = surface_info.atlas_offsets[surface_geom_slot]
                vertices_atlas = qd.Matrix.zero(gs.qd_float, 3, 3)
                previous_vertices_atlas = qd.Matrix.zero(gs.qd_float, 3, 3)
                for vertex_slot in qd.static(range(3)):
                    vertices_atlas[:, vertex_slot] = (
                        qd_inv_transform_by_trans_quat(vertices_world[:, vertex_slot], geom_pos, geom_quat)
                        + atlas_offset
                    )
                    previous_vertices_atlas[:, vertex_slot] = (
                        qd_inv_transform_by_trans_quat(
                            previous_vertices_world[:, vertex_slot],
                            surface_state.previous_geoms_pos[env_idx, surface_geom_slot],
                            surface_state.previous_geoms_quat[env_idx, surface_geom_slot],
                        )
                        + atlas_offset
                    )

                geom_lower = rigid_info.geoms_init_AABB[geom_idx, 0]
                geom_upper = rigid_info.geoms_init_AABB[geom_idx, 7]
                geom_extent = geom_upper - geom_lower
                clearance = sdf.sdf_func_collision_clearance(geom_idx, rigid_info)
                query_lower = qd.min(vertices_atlas[:, 0], vertices_atlas[:, 1], vertices_atlas[:, 2]) - clearance
                query_upper = qd.max(vertices_atlas[:, 0], vertices_atlas[:, 1], vertices_atlas[:, 2]) + clearance
                previous_direction_mesh = qd.Vector.zero(gs.qd_float, 3)
                has_previous_direction = False

                node_stack = qd.Vector.zero(gs.qd_int, qd.static(STACK_SIZE))
                node_stack[0] = 0
                stack_idx = 1
                while stack_idx > 0:
                    stack_idx -= 1
                    node_idx = node_stack[stack_idx]
                    node = bvh_nodes[0, node_idx]
                    is_node_overlapping = (query_lower <= node.bound.max).all() and (
                        query_upper >= node.bound.min
                    ).all()
                    if is_node_overlapping:
                        if node.left == -1:
                            sorted_leaf_idx = node_idx - (n_surface_faces - 1)
                            face_idx = qd.cast(bvh_morton_codes[0, sorted_leaf_idx][1], gs.qd_int)
                            if dyn_info.faces.geom_idx[face_idx] != geom_idx:
                                continue
                            face = dyn_info.faces.verts_idx[face_idx]
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

                            qd.atomic_max(surface_state.has_intersection[env_idx], 1)
                            if qd.static(is_audit):
                                qd.atomic_or(
                                    errno[env_idx], array_class.ErrorCode.INVALID_FEM_RIGID_SURFACE_INTERSECTION
                                )
                                continue

                            # A prior separating axis preserves the contact topology with a minimum normal
                            # displacement in the current geom frame.
                            has_history_correction, history_correction_mesh = (
                                triangle_triangle_previous_separating_correction(
                                    vertices_atlas,
                                    previous_vertices_atlas,
                                    rigid_vertices_atlas,
                                    clearance,
                                    rigid_info.EPS[None],
                                )
                            )
                            if has_history_correction:
                                history_correction_world = qd_transform_by_quat(history_correction_mesh, geom_quat)
                                for vertex_slot in qd.static(range(3)):
                                    vertex_idx = vertices_idx[vertex_slot]
                                    if self._func_is_vertex_projection_enabled(vertex_idx, env_idx):
                                        for axis in qd.static(range(3)):
                                            qd.atomic_add(
                                                surface_state.corrections[env_idx, vertex_idx][axis],
                                                history_correction_world[axis],
                                            )
                                        qd.atomic_add(surface_state.n_corrections[env_idx, vertex_idx], 1)
                                continue

                            if not has_previous_direction:
                                # The previous valid configuration selects the exterior side when the current
                                # triangle spans both sides of a collider.
                                previous_centroid_mesh = qd_inv_transform_by_trans_quat(
                                    previous_centroid_world,
                                    surface_state.previous_geoms_pos[env_idx, surface_geom_slot],
                                    surface_state.previous_geoms_quat[env_idx, surface_geom_slot],
                                )
                                _, previous_direction_mesh, _, _ = sdf.sdf_func_exact_mesh_surface_bvh_local(
                                    geom_idx,
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
                                previous_centroid_mesh = qd_inv_transform_by_trans_quat(
                                    previous_centroid_world,
                                    surface_state.previous_geoms_pos[env_idx, surface_geom_slot],
                                    surface_state.previous_geoms_quat[env_idx, surface_geom_slot],
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
                                exit_distance, has_exit = sdf.sdf_func_surface_bvh_ray_cast_local(
                                    geom_idx,
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
                                    for axis in qd.static(range(3)):
                                        exit_position_mesh[axis] = qd.select(
                                            previous_direction_mesh[axis] >= 0.0, geom_upper[axis], geom_lower[axis]
                                        )

                            # A shared exit plane moves the whole intersecting feature coherently and eliminates
                            # residual edge crossings.
                            target_projection = exit_position_mesh.dot(previous_direction_mesh) + clearance
                            for vertex_slot in qd.static(range(3)):
                                vertex_idx = vertices_idx[vertex_slot]
                                vertex_projection = (vertices_atlas[:, vertex_slot] - atlas_offset).dot(
                                    previous_direction_mesh
                                )
                                penetration = target_projection - vertex_projection
                                if penetration > 0.0 and self._func_is_vertex_projection_enabled(vertex_idx, env_idx):
                                    correction_mesh = penetration * previous_direction_mesh
                                    correction_world = qd_transform_by_quat(correction_mesh, geom_quat)
                                    for axis in qd.static(range(3)):
                                        qd.atomic_add(
                                            surface_state.corrections[env_idx, vertex_idx][axis],
                                            correction_world[axis],
                                        )
                                    qd.atomic_add(surface_state.n_corrections[env_idx, vertex_idx], 1)
                        elif stack_idx < qd.static(STACK_SIZE - 2):
                            node_stack[stack_idx] = node.left
                            node_stack[stack_idx + 1] = node.right
                            stack_idx += 2

    @qd.kernel
    def apply_implicit_surface_projection(
        self,
        f: qd.i32,
        surface_state: array_class.FEMRigidSurfaceState,
        rigid_info: array_class.RigidInfo,
    ):
        for env_idx in range(self._B):
            if surface_state.is_active[env_idx]:
                surface_state.is_active[env_idx] = surface_state.has_intersection[env_idx] != 0
            surface_state.has_intersection[env_idx] = 0
        for env_idx, vertex_idx in qd.ndrange(self._B, self.n_vertices):
            n_corrections = surface_state.n_corrections[env_idx, vertex_idx]
            if n_corrections > 0:
                pos = self.elements_v[f + 1, vertex_idx, env_idx].pos
                corrected_pos = pos + surface_state.corrections[env_idx, vertex_idx] / n_corrections
                correction = corrected_pos - pos
                self.elements_v[f + 1, vertex_idx, env_idx].pos = corrected_pos
                if correction.norm_sqr() > rigid_info.EPS[None] ** 2:
                    self.elements_v[f + 1, vertex_idx, env_idx].vel += correction / self.substep_dt
            surface_state.corrections[env_idx, vertex_idx] = qd.Vector.zero(gs.qd_float, 3)
            surface_state.n_corrections[env_idx, vertex_idx] = 0

    def project_implicit_surface(
        self,
        f,
        bvh_nodes,
        bvh_morton_codes,
        dyn_state,
        surface_state,
        dyn_info,
        rigid_info,
        collider_info,
        surface_info,
        errno,
    ):
        """Resolve committed surface intersections with a bounded active set and audit the final geometry."""
        self.init_implicit_surface_projection(surface_state)
        for _ in range(FEM_RIGID_SURFACE_PROJECTION_ITERATIONS):
            self.detect_implicit_surface_intersections(
                f,
                bvh_nodes,
                bvh_morton_codes,
                dyn_state,
                surface_state,
                dyn_info,
                rigid_info,
                surface_info,
                is_audit=False,
                errno=errno,
            )
            self.apply_implicit_surface_projection(
                f,
                surface_state,
                rigid_info,
            )
        self.detect_implicit_surface_intersections(
            f,
            bvh_nodes,
            bvh_morton_codes,
            dyn_state,
            surface_state,
            dyn_info,
            rigid_info,
            surface_info,
            is_audit=True,
            errno=errno,
        )

    def pcg_solve(self, f):
        self.init_pcg_solve()
        if self._is_implicit_rigid_projection_enabled:
            self.sim._coupler.project_fem_implicit_pcg(f, is_initial=True)
        for i in range(self._n_pcg_iterations):
            if self._is_implicit_rigid_projection_enabled:
                self.sim._coupler.project_fem_implicit_pcg(f, is_initial=False)
            else:
                self.one_pcg_iter()

    @qd.kernel
    def init_linesearch(self, f: qd.i32):
        for i_b in range(self._B):
            self.batch_linesearch_active[i_b] = self.batch_active[i_b]
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].prev_energy = 0.0
            self.linesearch_state[i_b].step_size = 1.0
            self.linesearch_state[i_b].m = 0.0

        # Inertia, x_prev, m
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            diff = self.elements_v[f + 1, i_v, i_b].pos - self.elements_v_energy[i_b, i_v].inertia
            self.linesearch_state[i_b].prev_energy += 0.5 * self.elements_v_info[i_v].mass_over_dt2 * diff.dot(diff)
            self.linesearch_state_v[i_b, i_v].x_prev = self.elements_v[f + 1, i_v, i_b].pos
            self.linesearch_state[i_b].m -= self.pcg_state_v[i_b, i_v].x.dot(self.elements_v_energy[i_b, i_v].force)
        # Elastic
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].prev_energy += self.elements_el_energy[i_b, i_e].energy * self.elements_i[i_e].V

    @qd.kernel
    def one_linesearch_iter(self, f: qd.i32):
        for i_b in range(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].energy = 0.0

        # update pos and compute Inertia energy
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.elements_v[f + 1, i_v, i_b].pos = (
                self.linesearch_state_v[i_b, i_v].x_prev
                + self.linesearch_state[i_b].step_size * self.pcg_state_v[i_b, i_v].x
            )
            diff = self.elements_v[f + 1, i_v, i_b].pos - self.elements_v_energy[i_b, i_v].inertia
            self.linesearch_state[i_b].energy += 0.5 * self.elements_v_info[i_v].mass_over_dt2 * diff.dot(diff)
            # damping
            if self._damping_alpha > 0.0:
                damping_alpha_dt = self._damping_alpha * self._substep_dt
                diff = self.elements_v[f + 1, i_v, i_b].pos - self.elements_v[f, i_v, i_b].pos
                self.linesearch_state[i_b].energy += (
                    0.5 * self.elements_v_info[i_v].mass_over_dt2 * diff.dot(diff) * damping_alpha_dt
                )

        # compute elastic energy
        self._func_compute_ele_energy(f)
        for i_b, i_e in qd.ndrange(self._B, self.n_elements):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].energy += self.elements_el_energy[i_b, i_e].energy * self.elements_i[i_e].V

        # check condition
        for i_b in range(self._B):
            if not self.batch_linesearch_active[i_b]:
                continue
            self.batch_linesearch_active[i_b] = (
                self.linesearch_state[i_b].energy
                > self.linesearch_state[i_b].prev_energy
                + self._linesearch_c * self.linesearch_state[i_b].step_size * self.linesearch_state[i_b].m
            )
            if not self.batch_linesearch_active[i_b]:
                continue
            self.linesearch_state[i_b].step_size *= self._linesearch_tau

    @qd.kernel
    def skip_linesearch(self, f: qd.i32):
        # Inertia, x_prev, m
        for i_b, i_v in qd.ndrange(self._B, self.n_vertices):
            if not self.batch_active[i_b]:
                continue
            self.elements_v[f + 1, i_v, i_b].pos = self.elements_v[f + 1, i_v, i_b].pos + self.pcg_state_v[i_b, i_v].x

    def linesearch(self, f: qd.i32):
        """
        Note
        ------
        https://en.wikipedia.org/wiki/Backtracking_line_search#Algorithm
        """
        if self._n_linesearch_iterations <= 0:
            self.skip_linesearch(f)
            return
        self.init_linesearch(f)
        for i in range(self._n_linesearch_iterations):
            self.one_linesearch_iter(f)

    def batch_solve(self, f: qd.i32):
        self.batch_active.fill(True)

        for i in range(self._n_newton_iterations):
            # compute element energy and gradient
            self.compute_ele_hessian_gradient(f)

            # If the hessian is invariant, we only need to compute it once
            for mat_idx in self._mats_idx:
                if self._mats[mat_idx].hessian_invariant:
                    self._mats[mat_idx]._hessian_ready = True

            # accumulate vertex force and preconditioner
            self.accumulate_vertex_force_preconditioner(f)

            # solve for the vertex positions
            self.pcg_solve(f)

            # line search
            self.linesearch(f)
            if self._is_implicit_rigid_projection_enabled:
                self.sim._coupler.project_fem_implicit_positions(f, is_committed=False)

    @qd.kernel
    def setup_pos_vel(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            # set pos and vel
            self.elements_v[f + 1, i_v, i_b].vel = (
                self.elements_v[f + 1, i_v, i_b].pos - self.elements_v[f, i_v, i_b].pos
            ) / self.substep_dt

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self._entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        if self.is_active:
            # Skip FEM solver step if using IPCCoupler (IPC handles FEM simulation)
            from genesis.engine.couplers import IPCCoupler

            if isinstance(self.sim._coupler, IPCCoupler):
                pass  # IPC coupler handles FEM simulation
            elif self._use_implicit_solver:
                self.precompute_material_data(f)
                self.init_pos_and_inertia(f)
                self.batch_solve(f)
                self.setup_pos_vel(f)
            else:
                self.init_pos_and_vel(f)
                self.compute_vel(f)
                self.apply_uniform_force(f)
                if self._constraints_initialized:
                    self.apply_soft_constraints(f)

    def substep_pre_coupling_grad(self, f):
        if self.is_active:
            if self._use_implicit_solver:
                gs.raise_exception("Gradient computation is not supported for implicit solver.")
            self.apply_uniform_force.grad(f)
            self.compute_vel.grad(f)
            self.init_pos_and_vel.grad(f)

    def substep_post_coupling(self, f):
        if self.is_active:
            self.compute_pos(f)
            if self._constraints_initialized:
                self.apply_hard_constraints(f)
            if self._is_implicit_rigid_projection_enabled:
                # Coupling and hard constraints write positions after Newton, so the committed frame is projected too.
                self.sim._coupler.project_fem_implicit_positions(f, is_committed=True)
                self.sim._coupler.project_fem_implicit_surface(f)

    def substep_post_coupling_grad(self, f):
        if self.is_active:
            self.compute_pos.grad(f)

    @qd.kernel
    def copy_frame(self, source: qd.i32, target: qd.i32):
        # Copy pos/vel for all vertices and all batch indices
        for i_v, i_b in qd.ndrange(self.n_vertices_max, self._B):
            self.elements_v[target, i_v, i_b].pos = self.elements_v[source, i_v, i_b].pos
            self.elements_v[target, i_v, i_b].vel = self.elements_v[source, i_v, i_b].vel

        # Copy 'active' for all elements and all batch indices
        for i_e, i_b in qd.ndrange(self.n_elements_max, self._B):
            self.elements_el_ng[target, i_e, i_b].active = self.elements_el_ng[source, i_e, i_b].active

    @qd.kernel
    def copy_grad(self, source: qd.i32, target: qd.i32):
        # Copy gradients for vertices
        for i_v, i_b in qd.ndrange(self.n_vertices_max, self._B):
            self.elements_v.grad[target, i_v, i_b].pos = self.elements_v.grad[source, i_v, i_b].pos
            self.elements_v.grad[target, i_v, i_b].vel = self.elements_v.grad[source, i_v, i_b].vel

        # Copy 'active' for elements
        for i_e, i_b in qd.ndrange(self.n_elements_max, self._B):
            self.elements_el_ng[target, i_e, i_b].active = self.elements_el_ng[source, i_e, i_b].active

    @qd.kernel
    def reset_grad_till_frame(self, f: qd.i32):
        # Zero out v.grad in frame 0..(f-1) for all vertices, all batch indices
        for frame_i, vert_i, i_b in qd.ndrange(f, self.n_vertices_max, self._B):
            self.elements_v.grad[frame_i, vert_i, i_b].pos = 0
            self.elements_v.grad[frame_i, vert_i, i_b].vel = 0

        # Zero out elements_el.grad in frame 0..(f-1) for all elements, all batch indices
        for frame_i, elem_i, i_b in qd.ndrange(f, self.n_elements_max, self._B):
            self.elements_el.grad[frame_i, elem_i, i_b].actu = 0

    # ------------------------------------------------------------------------------------
    # ----------------------------------- gradient ---------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        for entity in self._entities:
            entity.collect_output_grads()

    def add_grad_from_state(self, state):
        if self.is_active:
            if state.pos.grad is not None:
                state.pos.assert_contiguous()
                self._kernel_add_grad_from_pos(self._sim.cur_substep_local, state.pos.grad)

            if state.vel.grad is not None:
                state.vel.assert_contiguous()
                self._kernel_add_grad_from_vel(self._sim.cur_substep_local, state.vel.grad)

    def save_ckpt(self, ckpt_name):
        if self.is_active:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = dict()
                self._ckpt[ckpt_name]["pos"] = torch.zeros((self._B, self.n_vertices, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["vel"] = torch.zeros((self._B, self.n_vertices, 3), dtype=gs.tc_float)
                self._ckpt[ckpt_name]["active"] = torch.zeros((self._B, self.n_elements), dtype=gs.tc_int)

            self._kernel_get_state(
                0, self._ckpt[ckpt_name]["pos"], self._ckpt[ckpt_name]["vel"], self._ckpt[ckpt_name]["active"]
            )

            self.copy_frame(self.sim.substeps_local, 0)

    def load_ckpt(self, ckpt_name):
        self.copy_frame(0, self._sim.substeps_local)
        self.copy_grad(0, self._sim.substeps_local)

        if self._sim.requires_grad:
            self.reset_grad_till_frame(self._sim.substeps_local)

            self._kernel_set_state(
                0,
                self._ckpt[ckpt_name]["pos"],
                self._ckpt[ckpt_name]["vel"],
                self._ckpt[ckpt_name]["active"],
            )

            for entity in self._entities:
                entity.load_ckpt(ckpt_name=ckpt_name)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel, state.active)

    def get_state(self, f):
        if self.is_active:
            state = FEMSolverState(self._scene)
            self._kernel_get_state(f, state.pos, state.vel, state.active)
        else:
            state = None
        return state

    def get_embedded_positions(self, f, elements_idx, barycentric, envs_idx):
        """Return positions of tetrahedron material points for selected environments."""
        positions = gs.zeros(
            (len(envs_idx), len(elements_idx), 3),
            dtype=gs.tc_float,
            requires_grad=False,
            scene=self.scene,
        )
        self._kernel_get_embedded_positions(f, elements_idx, envs_idx, barycentric, positions)
        return positions

    @qd.kernel
    def _kernel_get_embedded_positions(
        self,
        f: qd.i32,
        elements_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        barycentric: qd.types.ndarray(),
        positions: qd.types.ndarray(),
    ):
        for point_idx, env_idx_local in qd.ndrange(elements_idx.shape[0], envs_idx.shape[0]):
            element_idx = elements_idx[point_idx]
            env_idx = envs_idx[env_idx_local]
            pos = qd.Vector.zero(gs.qd_float, 3)
            for element_vertex_idx in qd.static(range(4)):
                vertex_idx = self.elements_i[element_idx].el2v[element_vertex_idx]
                pos += barycentric[point_idx, element_vertex_idx] * self.elements_v[f, vertex_idx, env_idx].pos
            for axis in qd.static(range(3)):
                positions[env_idx_local, point_idx, axis] = pos[axis]

    def get_state_render(self, f):
        """
        Refresh and return the render geometry of every visual geom, laid out contiguously.

        Returns
        -------
        tuple
            (vverts_pos, vverts_uvs, vfaces_indices) - environment-offset render vertex positions with shape
            (n_vverts, B), their UV coordinates, and the render triangles in global render vertex space.
        """
        if not self.is_active or self._n_vverts == 0:
            return None, None, None

        self._kernel_get_state_render(f)
        return self.vverts_render.pos, self.vverts_uvs, self.vfaces_indices

    def get_tetrahedral_state_render(self, f):
        """Refresh and return environment-offset positions for tetrahedral skeleton rendering."""
        if not self.is_active or not self._has_tetrahedral_visual:
            return None

        self._kernel_get_tetrahedral_state_render(f)
        return self.vertices_render.pos

    def get_forces(self):
        """
        Get forces on all vertices.

        Returns:
            torch.Tensor : shape (B, n_vertices, 3) where B is batch size
        """
        if not self.is_active:
            return None

        return qd_to_torch(self.elements_v_energy.force, copy=True)

    @qd.kernel
    def _kernel_add_elements(
        self,
        f: qd.i32,
        mat_idx: qd.i32,
        mat_mu: qd.f32,
        mat_lam: qd.f32,
        mat_rho: qd.f32,
        mat_friction_mu: qd.f32,
        v_start: qd.i32,
        el_start: qd.i32,
        s_start: qd.i32,
        verts: qd.types.ndarray(),
        elems: qd.types.ndarray(),
        tri2v: qd.types.ndarray(),
        tri2el: qd.types.ndarray(),
    ):
        n_verts_local = verts.shape[0]
        for i_v, i_b in qd.ndrange(n_verts_local, self._B):
            i_global = i_v + v_start
            for j in qd.static(range(3)):
                self.elements_v[f, i_global, i_b].pos[j] = verts[i_v, j]
            self.elements_v[f, i_global, i_b].vel = qd.Vector.zero(gs.qd_float, 3)

        for i_v in range(n_verts_local):
            i_global = i_v + v_start
            self.elements_v_info[i_global].mass = 0.0
            self.elements_v_info[i_global].mass_over_dt2 = 0.0
            self.elements_v_info[i_global].friction_mu = mat_friction_mu

        dt2_inv = 1.0 / (self.substep_dt**2)
        n_elems_local = elems.shape[0]
        for i_e in range(n_elems_local):
            i_global = i_e + el_start

            a = self.elements_v[f, elems[i_e, 0] + v_start, 0].pos
            b = self.elements_v[f, elems[i_e, 1] + v_start, 0].pos
            c = self.elements_v[f, elems[i_e, 2] + v_start, 0].pos
            d = self.elements_v[f, elems[i_e, 3] + v_start, 0].pos
            B_inv = qd.Matrix.cols([a - d, b - d, c - d])
            self.elements_i[i_global].B = B_inv.inverse()
            det = B_inv.determinant()
            # Determinant should be consistently smaller than 0
            if det >= 0.0:
                self.tet_wrong_order[None] = True
            V = qd.abs(det) / 6.0
            self.elements_i[i_global].V = V
            V_scaled = V * self._vol_scale
            self.elements_i[i_global].V_scaled = V_scaled

            for j in qd.static(range(4)):
                self.elements_i[i_global].el2v[j] = elems[i_e, j] + v_start
            self.elements_i[i_global].mat_idx = mat_idx
            self.elements_i[i_global].mu = mat_mu
            self.elements_i[i_global].lam = mat_lam
            self.elements_i[i_global].friction_mu = mat_friction_mu
            self.elements_i[i_global].mass_scaled = mat_rho * V_scaled
            for j in qd.static(range(4)):
                mass = 0.25 * mat_rho * V
                self.elements_v_info[self.elements_i[i_global].el2v[j]].mass += mass
                self.elements_v_info[self.elements_i[i_global].el2v[j]].mass_over_dt2 += mass * dt2_inv
            self.elements_i[i_global].muscle_group = 0
            self.elements_i[i_global].muscle_direction = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)

        for i_v in range(n_verts_local):
            i_global = i_v + v_start
            self.elements_v_info[i_global].mass_inv = 1.0 / self.elements_v_info[i_global].mass

        for i_e, i_b in qd.ndrange(n_elems_local, self._B):
            i_global = i_e + el_start
            self.elements_el[f, i_global, i_b].actu = 0.0
            self.elements_el_ng[f, i_global, i_b].active = True

        for i_s in range(tri2v.shape[0]):
            i_global = i_s + s_start
            for j in qd.static(range(3)):
                self.surface[i_global].tri2v[j] = tri2v[i_s, j] + v_start
            self.surface[i_global].tri2el = tri2el[i_s] + el_start
            self.surface[i_global].active = True

    @qd.kernel
    def _kernel_add_cloth(
        self,
        f: qd.i32,
        v_start: qd.i32,
        s_start: qd.i32,
        verts: qd.types.ndarray(),
        tri2v: qd.types.ndarray(),
    ):
        """
        Add cloth vertices and surface triangles to the solver, for position tracking and coupling only.

        Cloth elements and mass are owned by the IPC coupler, so the vertex info holds placeholder values and each
        surface triangle references itself as element.
        """
        n_verts_local = verts.shape[0]
        for i_v, i_b in qd.ndrange(n_verts_local, self._B):
            i_global = i_v + v_start
            for j in qd.static(range(3)):
                self.elements_v[f, i_global, i_b].pos[j] = verts[i_v, j]
            self.elements_v[f, i_global, i_b].vel = qd.Vector.zero(gs.qd_float, 3)

        for i_v in range(n_verts_local):
            i_global = i_v + v_start
            self.elements_v_info[i_global].mass = 1.0
            self.elements_v_info[i_global].mass_over_dt2 = 0.0
            self.elements_v_info[i_global].friction_mu = 0.0

        for i_s in range(tri2v.shape[0]):
            i_global = i_s + s_start
            for j in qd.static(range(3)):
                self.surface[i_global].tri2v[j] = tri2v[i_s, j] + v_start
            self.surface[i_global].tri2el = i_global
            self.surface[i_global].active = True

    @qd.kernel
    def _kernel_set_elements_pos(
        self,
        f: qd.i32,
        element_v_start: qd.i32,
        n_vertices: qd.i32,
        pos: qd.types.ndarray(),
    ):
        for i_v, i_b in qd.ndrange(n_vertices, self._B):
            i_global = i_v + element_v_start
            for k in qd.static(range(3)):
                self.elements_v[f, i_global, i_b].pos[k] = pos[i_b, i_v, k]

    @qd.kernel
    def _kernel_set_elements_pos_grad(
        self,
        f: qd.i32,
        element_v_start: qd.i32,
        n_vertices: qd.i32,
        pos_grad: qd.types.ndarray(),
    ):
        for i_v, i_b in qd.ndrange(n_vertices, self._B):
            i_global = i_v + element_v_start
            for k in qd.static(range(3)):
                self.elements_v.grad[f, i_global, i_b].pos[k] = pos_grad[i_b, i_v, k]

    @qd.kernel
    def _kernel_set_elements_vel(
        self,
        f: qd.i32,
        element_v_start: qd.i32,
        n_vertices: qd.i32,
        vel: qd.types.ndarray(),  # shape [B, n_vertices, 3]
    ):
        for i_v, i_b in qd.ndrange(n_vertices, self._B):
            i_global = i_v + element_v_start
            for k in qd.static(range(3)):
                self.elements_v[f, i_global, i_b].vel[k] = vel[i_b, i_v, k]

    @qd.kernel
    def _kernel_set_elements_vel_grad(
        self,
        f: qd.i32,
        element_v_start: qd.i32,
        n_vertices: qd.i32,
        vel_grad: qd.types.ndarray(),  # shape [B, n_vertices, 3]
    ):
        for i_v, i_b in qd.ndrange(n_vertices, self._B):
            i_global = i_v + element_v_start
            for k in qd.static(range(3)):
                self.elements_v.grad[f, i_global, i_b].vel[k] = vel_grad[i_b, i_v, k]

    @qd.kernel
    def _kernel_set_elements_actu(
        self,
        f: qd.i32,
        element_el_start: qd.i32,
        n_elements: qd.i32,
        n_groups: qd.i32,
        actu: qd.types.ndarray(),  # shape [B, n_elements, n_groups]
    ):
        for i_e, j_g, i_b in qd.ndrange(n_elements, n_groups, self._B):
            i_global = i_e + element_el_start
            if self.elements_i[i_global].muscle_group == j_g:
                self.elements_el[f, i_global, i_b].actu = actu[i_b, j_g]

    @qd.kernel
    def _kernel_set_elements_actu_grad(
        self,
        f: qd.i32,
        element_el_start: qd.i32,
        n_elements: qd.i32,
        actu_grad: qd.types.ndarray(),  # shape [B, n_elements]
    ):
        for i_e, i_b in qd.ndrange(n_elements, self._B):
            i_global = i_e + element_el_start
            self.elements_el.grad[f, i_global, i_b].actu = actu_grad[i_b, i_e]

    @qd.kernel
    def _kernel_set_active(
        self,
        f: qd.i32,
        element_el_start: qd.i32,
        n_elements: qd.i32,
        active: qd.types.ndarray(),  # shape [B, n_elements]
    ):
        for i_e, i_b in qd.ndrange(n_elements, self._B):
            i_global = i_e + element_el_start
            self.elements_el_ng[f, i_global, i_b].active = active[i_b, i_e]

    @qd.kernel
    def _kernel_set_muscle_group(
        self,
        element_el_start: qd.i32,
        n_elements: qd.i32,
        muscle_group: qd.types.ndarray(),
    ):
        for i_e in range(n_elements):
            i_global = i_e + element_el_start
            self.elements_i[i_global].muscle_group = muscle_group[i_e]

    @qd.kernel
    def _kernel_set_muscle_direction(
        self,
        element_el_start: qd.i32,
        n_elements: qd.i32,
        muscle_direction: qd.types.ndarray(),
    ):
        for i_e in range(n_elements):
            i_global = i_e + element_el_start
            for j in qd.static(range(3)):
                self.elements_i[i_global].muscle_direction[j] = muscle_direction[i_e, j]

    @qd.kernel
    def _kernel_get_el2v(
        self,
        element_el_start: qd.i32,
        n_elements: qd.i32,
        el2v: qd.types.ndarray(),
    ):
        for i_e in range(n_elements):
            i_global = i_e + element_el_start
            for j in qd.static(range(4)):
                el2v[i_global, j] = self.elements_i[i_global].el2v[j]

    @qd.kernel
    def _kernel_get_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),  # shape [B, n_vertices, 3]
        vel: qd.types.ndarray(),  # shape [B, n_vertices, 3]
        active: qd.types.ndarray(),  # shape [B, n_elements]
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            for j in qd.static(range(3)):
                pos[i_b, i_v, j] = self.elements_v[f, i_v, i_b].pos[j]
                vel[i_b, i_v, j] = self.elements_v[f, i_v, i_b].vel[j]

        for i_e, i_b in qd.ndrange(self.n_elements, self._B):
            active[i_b, i_e] = self.elements_el_ng[f, i_e, i_b].active

    @qd.kernel
    def _kernel_get_state_render(self, f: qd.i32):
        for i_vv, i_b in qd.ndrange(self._n_vverts, self._B):
            i_v = self.vverts_info[i_vv].vert_idx
            for j in qd.static(range(3)):
                pos_j = qd.cast(self.elements_v[f, i_v, i_b].pos[j], qd.f32)
                self.vverts_render[i_vv, i_b].pos[j] = pos_j + self.envs_offset[i_b][j]

    @qd.kernel
    def _kernel_get_tetrahedral_state_render(self, f: qd.i32):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            for j in qd.static(range(3)):
                pos_j = qd.cast(self.elements_v[f, i_v, i_b].pos[j], qd.f32)
                self.vertices_render[i_v, i_b].pos[j] = pos_j + self.envs_offset[i_b][j]

    @qd.kernel
    def _kernel_add_vverts(
        self,
        vvert_start: qd.i32,
        vface_start: qd.i32,
        v_start: qd.i32,
        verts_idx: qd.types.ndarray(),
        uvs: qd.types.ndarray(element_dim=1),
        vfaces: qd.types.ndarray(element_dim=1),
    ):
        n_vverts_local = verts_idx.shape[0]
        for i_vv_ in range(n_vverts_local):
            self.vverts_info[i_vv_ + vvert_start].vert_idx = verts_idx[i_vv_] + v_start

        n_uvs = uvs.shape[0]
        for i_vv_ in range(n_uvs):
            self.vverts_uvs[i_vv_ + vvert_start] = uvs[i_vv_]

        n_vfaces_local = vfaces.shape[0]
        for i_vf_ in range(n_vfaces_local):
            self.vfaces_indices[i_vf_ + vface_start] = vfaces[i_vf_] + vvert_start

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),  # shape [B, n_vertices, 3]
        vel: qd.types.ndarray(),  # shape [B, n_vertices, 3]
        active: qd.types.ndarray(),  # shape [B, n_elements]
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            for j in qd.static(range(3)):
                self.elements_v[f, i_v, i_b].pos[j] = pos[i_b, i_v, j]
                self.elements_v[f, i_v, i_b].vel[j] = vel[i_b, i_v, j]

        for i_e, i_b in qd.ndrange(self.n_elements, self._B):
            self.elements_el_ng[f, i_e, i_b].active = active[i_b, i_e]

    @qd.kernel
    def _kernel_add_grad_from_pos(self, f: qd.i32, pos_grad: qd.types.ndarray()):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            for j in qd.static(range(3)):
                self.elements_v.grad[f, i_v, i_b].pos[j] += pos_grad[i_b, i_v, j]

    @qd.kernel
    def _kernel_add_grad_from_vel(self, f: qd.i32, vel_grad: qd.types.ndarray()):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            for j in qd.static(range(3)):
                self.elements_v.grad[f, i_v, i_b].vel[j] += vel_grad[i_b, i_v, j]

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def floor_height(self):
        return self._floor_height

    @property
    def damping(self):
        return self._damping

    @property
    def n_vertices(self):
        return sum([entity.n_vertices for entity in self._entities])

    @property
    def n_elements(self):
        return sum([entity.n_elements for entity in self._entities])

    @property
    def n_surfaces(self):
        return sum([entity.n_surfaces for entity in self.entities])

    @property
    def n_vverts(self):
        return sum([entity.n_vverts for entity in self._entities])

    @property
    def n_vfaces(self):
        return sum([entity.n_vfaces for entity in self._entities])

    @property
    def n_vertices_max(self):
        return self._n_vertices_max

    @property
    def n_elements_max(self):
        return self._n_elements_max

    @property
    def vol_scale(self):
        return self._vol_scale

    @property
    def n_surface_vertices(self):
        return self.surface_vertices.shape[0]

    @property
    def n_surface_elements(self):
        return self.surface_elements.shape[0]

    # ------------------------------------------------------------------------------------
    # -------------------------------- vertex constraints --------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def _kernel_update_linked_vertex_constraints(
        self,
        links_state: array_class.LinksState,
    ):
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            vc = self.vertex_constraints[i_v, i_b]
            if vc.is_constrained and vc.link_idx >= 0:
                i_l = vc.link_idx
                pos = links_state.pos[i_l, i_b]
                quat = links_state.quat[i_l, i_b]

                offset_pos = vc.link_offset_pos
                offset_quat = qd_transform_quat_by_quat(vc.link_init_quat_inv, quat)
                self.vertex_constraints[i_v, i_b].target_pos = pos + qd_transform_by_quat(offset_pos, offset_quat)

    @qd.kernel
    def apply_hard_constraints(self, f: qd.i32):
        """Apply hard constraints by directly overriding positions and velocities."""
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            vc = self.vertex_constraints[i_v, i_b]
            if vc.is_constrained and not vc.is_soft_constraint:
                self.elements_v[f + 1, i_v, i_b].pos = vc.target_pos
                self.elements_v[f + 1, i_v, i_b].vel.fill(0.0)

    @qd.kernel
    def apply_soft_constraints(self, f: qd.i32):
        """Apply soft constraints as spring forces for explicit solver."""
        for i_v, i_b in qd.ndrange(self.n_vertices, self._B):
            vc = self.vertex_constraints[i_v, i_b]
            if vc.is_constrained and vc.is_soft_constraint:
                pos_error = self.elements_v[f, i_v, i_b].pos - vc.target_pos
                vel_error = self.elements_v[f + 1, i_v, i_b].vel - self.elements_v[f, i_v, i_b].vel
                spring_force = -vc.stiffness * pos_error
                damping_force = -2.0 * qd.math.sqrt(vc.stiffness) * vel_error

                dv = self.substep_dt * (spring_force + damping_force)
                self.elements_v[f + 1, i_v, i_b].vel += dv

    @qd.kernel
    def _kernel_set_vertex_constraints(
        self,
        f: qd.i32,
        verts_idx: qd.types.ndarray(),  # shape [B, V]
        target_poss: qd.types.ndarray(),  # shape [B, V, 3]
        is_soft_constraint: qd.i32,
        stiffness: qd.f32,
        link_idx: qd.i32,
        link_init_pos: qd.types.ndarray(),  # shape [B, 3]
        link_init_quat_inv: qd.types.ndarray(),  # shape [B, 4]
        envs_idx: qd.types.ndarray(),  # shape [B]
    ):
        for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            i_v = verts_idx[i_b, i_v_]
            self.vertex_constraints[i_v, i_b].is_constrained = True
            self.vertex_constraints[i_v, i_b].is_soft_constraint = qd.cast(is_soft_constraint, gs.qd_bool)
            self.vertex_constraints[i_v, i_b].stiffness = stiffness
            self.vertex_constraints[i_v, i_b].link_idx = link_idx

            cur_pos = self.elements_v[f, i_v, i_b].pos
            for j in qd.static(range(3)):
                self.vertex_constraints[i_v, i_b].target_pos[j] = target_poss[i_b_, i_v_, j]
                self.vertex_constraints[i_v, i_b].link_offset_pos[j] = cur_pos[j] - link_init_pos[i_b_, j]
            for j in qd.static(range(4)):
                self.vertex_constraints[i_v, i_b].link_init_quat_inv[j] = link_init_quat_inv[i_b_, j]

    @qd.kernel
    def _kernel_update_constraint_targets(
        self, verts_idx: qd.types.ndarray(), new_target_poss: qd.types.ndarray(), envs_idx: qd.types.ndarray()
    ):
        for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            i_v = verts_idx[i_b, i_v_]
            for j in qd.static(range(3)):
                self.vertex_constraints[i_v, i_b].target_pos[j] = new_target_poss[i_b_, i_v_, j]

    @qd.kernel
    def _kernel_remove_specific_constraints(self, verts_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray()):
        for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            i_v = verts_idx[i_b, i_v_]
            self.vertex_constraints[i_v, i_b].is_constrained = False
