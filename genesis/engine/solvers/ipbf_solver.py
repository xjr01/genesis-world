import math

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.boundaries import CubeBoundary, CylinderBoundary, PlaneBoundary, TiltedCylinderBoundary
from genesis.engine.entities import IPBFEntity
from genesis.engine.states.solvers import IPBFSolverState
from genesis.utils.array_class import ErrorCode, V_ANNOTATION
import genesis.utils.geom as gu
from genesis.utils.misc import qd_to_numpy

from .base_solver import Solver


@qd.data_oriented
class IPBFSolver(Solver):
    # spherical-illumination surface classifier resolution (Shibata 2015, PBSTF reference)
    _N_THETA = 18
    _N_PHI = 36
    _ST_SURFACE_NEIGHBOR_OVERFLOW = 1
    _ST_LOCAL_MESH_QUEUE_OVERFLOW = 2
    _ST_LOCAL_MESH_NEIGHBOR_OVERFLOW = 3
    _ST_REVERSE_RING_OVERFLOW = 4
    # reverse one-ring entries pack (owner << 7) | ring_slot, so ring capacity is limited to 128
    _ST_REV_SHIFT = 7

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._n_particles = None
        self._errno = None
        self._particle_size = options.particle_size
        self._support_radius = options._support_radius

        # IPBF parameters
        self._ipbf_iterations = options.ipbf_iterations
        self._alpha = options.alpha
        self._is_hash_rebuilt_per_iteration = options.is_neighbor_search_rebuilt_per_iteration

        self._is_damping_enabled = options.is_damping_enabled
        self._damping_alpha_star = options.damping_alpha_star
        self._damping_beta = options.damping_beta

        self._has_boundary_particles = options.has_boundary_particles
        self._boundary_layers = options.boundary_layers
        self._boundary_cylinder = options.boundary_cylinder
        self._boundary_pitcher = options.boundary_pitcher
        self._has_boundary_pitcher = options.boundary_pitcher is not None

        self._boundary_plane = options.boundary_plane
        self._has_boundary_plane = options.boundary_plane is not None

        self._viscosity_xsph = options.viscosity_xsph
        # interface-restricted variant: surface flags come from the ST topology pass, so a
        # positive coefficient requires the ST machinery to run
        self._surface_viscosity_xsph = options.surface_viscosity_xsph
        if self._surface_viscosity_xsph > 0.0 and not options.is_surface_tension_enabled:
            gs.raise_exception("surface_viscosity_xsph requires is_surface_tension_enabled=True.")

        self._diffusion_coeff = options.diffusion_coeff

        self._is_st_enabled = options.is_surface_tension_enabled
        self._is_st_model_linear = options.st_model == "linear"
        self._st_stiffness = options.st_stiffness
        self._is_st_distance_enabled = options.is_st_distance_enabled
        self._st_distance_stiffness = options.st_distance_stiffness
        self._st_ring_radius = options.st_ring_radius_factor * options.particle_size
        self._st_topo_interval = options.st_topo_interval
        self._st_lit_threshold = options.st_lit_threshold
        self._st_normal_compat = options.st_normal_compat
        self._st_max_surface_neighbors = options.st_max_surface_neighbors
        self._st_max_localmesh_neighbors = options.st_max_localmesh_neighbors
        self._st_topo_counter = 0
        self._alpha_eff = options.st_alpha_ref if (self._is_st_enabled and self._is_st_model_linear) else options.alpha
        # area-energy weight: quadratic -> k~_st directly; linear -> w_st = sigma * alpha_eff
        # (sigma is a material property, resolved in build(); placeholder here)
        self._st_area_weight = options.st_stiffness

        self._eps_r = 1e-6 * self._support_radius

        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)

        self._particle_volume = 0.8 * self._particle_size**3  # 0.8 is an empirical value

        # spatial hasher
        self.sh = gu.SpatialHasher(
            cell_size=options.hash_grid_cell_size,
            grid_res=options._hash_grid_res,
        )
        # boundary
        self.setup_boundary()

    def setup_boundary(self):
        if self._boundary_cylinder is None:
            self.boundary = CubeBoundary(
                lower=self._lower_bound,
                upper=self._upper_bound,
            )
        else:
            cx, cy, radius, z_bottom = self._boundary_cylinder[:4]
            z_top = self._boundary_cylinder[4] if len(self._boundary_cylinder) > 4 else None
            self.boundary = CylinderBoundary(
                center_xy=(cx, cy),
                radius=radius,
                z_bottom=z_bottom,
                z_top=z_top,
                band=1.5 * self._particle_size,
            )
        self.boundary2 = None
        if self._has_boundary_pitcher:
            ox, oy, oz, ax, ay, az, p_radius, p_length = self._boundary_pitcher
            self.boundary2 = TiltedCylinderBoundary(
                origin=(ox, oy, oz),
                axis=(ax, ay, az),
                radius=p_radius,
                length=p_length,
                band=1.5 * self._particle_size,
            )

        self.boundary3 = None
        if self._has_boundary_plane:
            if len(self._boundary_plane) == 1:
                self.boundary3 = PlaneBoundary(z0=self._boundary_plane[0])
            else:
                pz0, pcx, pcy, p_radius = self._boundary_plane
                self.boundary3 = PlaneBoundary(z0=pz0, center_xy=(pcx, pcy), radius=p_radius)

    def init_particle_fields(self):
        self._errno = qd.field(gs.qd_int, shape=(self._B,))
        self._errno.fill(0)
        # dynamic particle state
        struct_particle_state = qd.types.struct(
            pos=gs.qd_vec3,  # position
            vel=gs.qd_vec3,  # velocity
            rho=gs.qd_float,  # volume-normalized density sum_j V W (diagnostic)
            ipos=gs.qd_vec3,  # position at the start of the substep (x^t)
            y=gs.qd_vec3,  # inertial position (eq. 3)
            dpos=gs.qd_vec3,  # Newton step (Delta x) scratch
            C=gs.qd_float,  # clamped density constraint max(rho - 1, 0)
            xstar=gs.qd_vec3,  # alternative position x* with compliance alpha* (artificial damping)
            dv=gs.qd_vec3,
            c=gs.qd_float,
            dc=gs.qd_float,  # Jacobi buffer for the concentration diffusion pass
        )

        # dynamic particle state without gradient
        struct_particle_state_ng = qd.types.struct(
            reordered_idx=gs.qd_int,
            active=gs.qd_bool,
            is_boundary=gs.qd_bool,  # static Akinci-style boundary particle (fixed, no own constraint)
            boundary_group=gs.qd_int,
        )

        # static particle info
        struct_particle_info = qd.types.struct(
            rho=gs.qd_float,  # rest density
            mass=gs.qd_float,  # mass
        )

        # single frame particle state for rendering
        struct_particle_state_render = qd.types.struct(
            pos=gs.qd_vec3,
            vel=gs.qd_vec3,
            active=gs.qd_bool,
            c=gs.qd_float,
        )

        # construct fields
        self.particles = struct_particle_state.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_ng = struct_particle_state_ng.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_info = struct_particle_info.field(
            shape=(self._n_particles,), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_reordered = struct_particle_state.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_ng_reordered = struct_particle_state_ng.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )
        self.particles_info_reordered = struct_particle_info.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )

        self.particles_render = struct_particle_state_render.field(
            shape=(self._n_particles, self._B), needs_grad=False, layout=qd.Layout.SOA
        )

        # reordered-slot position -> original particle index (hash grid lookup without particle reorder)
        self.particle_idx = qd.field(gs.qd_int, shape=(self._n_particles, self._B))

        if self._is_st_enabled:
            n, b = self._n_particles, self._B
            self.on_surface = qd.field(gs.qd_bool, shape=(n, b))
            self.topology_valid = qd.field(gs.qd_bool, shape=(n, b))
            self.normal = qd.field(gs.qd_vec3, shape=(n, b))
            self.CA = qd.field(gs.qd_float, shape=(n, b))  # area constraint C_i^A (recomputed every iteration)
            self.n_ring = qd.field(gs.qd_int, shape=(n, b))
            # int difference array of the illumination screen (N_THETA+1 x N_PHI+1); the classify
            # pass restores per-cell counts with a 2D prefix sum -- a bool 18x36 screen cannot do this
            self._screen_blocked = qd.field(gs.qd_int, shape=(n, b, self._N_THETA + 1, self._N_PHI + 1))
            cand = (n, b, self._st_max_surface_neighbors)
            self.neighbor_ids = qd.field(gs.qd_int, shape=cand)  # one-ring candidates (mesh build workset)
            self.projected_positions = qd.field(gs.qd_vec2, shape=cand)  # tangent-plane projections
            self._chain_pre = qd.field(gs.qd_int, shape=cand)  # polar insertion-sort linked list
            self._chain_nxt = qd.field(gs.qd_int, shape=cand)
            self._node_queue = qd.field(gs.qd_int, shape=(n, b, 3 * self._st_max_surface_neighbors))
            self.local_mesh_neighbors = qd.field(
                gs.qd_int, shape=(n, b, self._st_max_localmesh_neighbors)
            )  # final one-ring (polar order)
            self._mesh_axis_x = qd.field(gs.qd_vec3, shape=(n, b))
            self._mesh_axis_y = qd.field(gs.qd_vec3, shape=(n, b))
            self._overflow = qd.field(gs.qd_int, shape=())
            self._rev_ids = qd.field(gs.qd_int, shape=(n, b, self._st_max_surface_neighbors))
            self._rev_count = qd.field(gs.qd_int, shape=(n, b))

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        pass

    def build(self):
        super().build()

        self._B = self._sim._B

        # particles and entities
        self._n_particles = self.n_particles

        if self.is_active:
            # fluid particles come first; static boundary particles are appended after them
            self._n_fluid_particles = self._n_particles
            boundary_pos = self._sample_boundary_particles() if self._has_boundary_particles else np.zeros((0, 3))
            self._n_boundary_particles = len(boundary_pos)
            self._n_particles = self._n_fluid_particles + self._n_boundary_particles

            self.sh.build(self._B)
            self.init_particle_fields()
            self.init_ckpt()

            for entity in self.entities:
                entity._add_to_solver()

            if self._n_boundary_particles > 0:
                kernel_add_boundary_particles(
                    self._n_fluid_particles,
                    self._n_boundary_particles,
                    self.entities[0].material.rho,
                    boundary_pos,
                    self,
                )
            gs.logger.info(
                f"IPBFSolver: {self._n_fluid_particles} fluid + {self._n_boundary_particles} boundary particles."
            )

        if self._is_st_enabled and self._is_st_model_linear:
            if not self.entities:
                gs.raise_exception("IPBF linear surface-tension model requires at least one entity.")
            self._st_area_weight = self.entities[0].material.surface_tension * self._alpha_eff / 3.0

        # FIXME: _gravity must be a raw qd.field() -- see comment in mpm_solver.py
        # Only when active -- see the SNode-tree note in mpm_solver.py.
        if self.is_active and self._gravity is not None:
            gravity = qd_to_numpy(self._gravity, transpose=True)
            self._gravity = qd.field(dtype=gs.qd_vec3, shape=(self._B,))
            self._gravity.from_numpy(gravity)

    def _sample_boundary_particles(self):
        """
        Sample the box floor and side walls on a regular grid:
        bottom face (z = lower.z) + 4 side faces spanning the full box height, on a regular grid with
        spacing `particle_size`; layer 0 sits on the wall plane, further layers are shifted one
        `particle_size` outward each. Corner/edge duplicates are removed.
        """
        ps = self._particle_size
        lx, ly, lz = self._lower_bound
        ux, uy, uz = self._upper_bound
        xs = np.arange(lx, ux + 0.5 * ps, ps)
        ys = np.arange(ly, uy + 0.5 * ps, ps)
        zs = np.arange(lz, uz + 0.5 * ps, ps)

        pts = []
        for l in range(self._boundary_layers):
            off = -(l + 1) * ps
            # bottom face
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            pts.append(np.stack([X, Y, np.full_like(X, lz + off)], axis=-1).reshape(-1, 3))
            # side faces at x = lx / ux (full y, z)
            for xv in (lx + off, ux - off):
                Y, Z = np.meshgrid(ys, zs, indexing="ij")
                pts.append(np.stack([np.full_like(Y, xv), Y, Z], axis=-1).reshape(-1, 3))
            # side faces at y = ly / uy (full x, z)
            for yv in (ly + off, uy - off):
                X, Z = np.meshgrid(xs, zs, indexing="ij")
                pts.append(np.stack([X, np.full_like(X, yv), Z], axis=-1).reshape(-1, 3))

        pos = np.concatenate(pts, axis=0)
        # dedupe corner/edge duplicates on the integer grid
        grid_idx = np.round(pos / ps)
        _, uniq_idx = np.unique(grid_idx, axis=0, return_index=True)
        return pos[np.sort(uniq_idx)]

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_entity(self, idx, material, morph, surface, name: str | None = None) -> "IPBFEntity":
        entity = IPBFEntity(
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

    # ------------------------------------------------------------------------------------
    # ------------------------------------- utils ----------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def cubic_kernel_W(self, r):
        """
        Cubic spline smoothing kernel W(r) (Koschier et al. 2019), same coefficients as
        sph_solver.cubic_kernel: sigma = 8 / (pi R^3), q = r / R, support q <= 1.
        """
        res = gs.qd_float(0.0)
        h = self._support_radius
        k = 8.0 / np.pi / h**3
        q = r / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q**2
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1.0)
            else:
                res = 2 * k * (1.0 - q) ** 3
        return res

    @qd.func
    def cubic_kernel_dW(self, r):
        """
        First radial derivative W'(r) of the cubic spline kernel; grad W(r_vec) = W'(r) r_hat.
        Consistent with sph_solver.cubic_kernel_derivative.
        """
        res = gs.qd_float(0.0)
        h = self._support_radius
        k = 8.0 / np.pi / h**3
        q = r / h
        if q <= 1.0:
            if q <= 0.5:
                res = k / h * (18.0 * q**2 - 12.0 * q)
            else:
                res = -6.0 * k / h * (1.0 - q) ** 2
        return res

    @qd.func
    def cubic_kernel_ddW(self, r):
        """
        Second radial derivative W''(r) of the cubic spline kernel. W''(0) = -12 sigma / R^2.
        """
        res = gs.qd_float(0.0)
        h = self._support_radius
        k = 8.0 / np.pi / h**3
        q = r / h
        if q <= 1.0:
            if q <= 0.5:
                res = k / h**2 * (36.0 * q - 12.0)
            else:
                res = 12.0 * k / h**2 * (1.0 - q)
        return res

    @qd.func
    def cubic_kernel_hessian(self, r_vec):
        """
        Kernel Hessian H_W(r_vec) = W''(r) r_hat r_hat^T + (W'(r) / r) (I - r_hat r_hat^T).
        Only valid for r >= eps_r; the r -> 0 limit W''(0) I is handled by the caller (self term).
        """
        r = r_vec.norm()
        r_hat = r_vec / r
        rrT = r_hat.outer_product(r_hat)
        return self.cubic_kernel_ddW(r) * rrT + (self.cubic_kernel_dW(r) / r) * (
            qd.Matrix.identity(dt=gs.qd_float, n=3) - rrT
        )

    @qd.func
    def _st_cubic_dW(self, r):
        """First radial derivative of the cubic spline with support R_st (PBSTF reference form)."""
        res = gs.qd_float(0.0)
        h = self._st_ring_radius
        coefficient = 48.0 / (math.pi * h**4)
        q = r / h
        if q < 0.5:
            res = coefficient * q * (3.0 * q - 2.0)
        elif q < 1.0:
            res = coefficient * (1.0 - q) * (q - 1.0)
        return res

    @qd.func
    def _st_angle_of_vec2(self, a, b):
        result = gs.qd_float(0.0)
        a_len = a.norm()
        b_len = b.norm()
        if a_len > gs.EPS and b_len > gs.EPS:
            result = qd.acos(qd.max(-1.0, qd.min(1.0, a.dot(b) / (a_len * b_len))))
        return result

    @qd.func
    def _st_need_flip(self, i, i_b, u):
        result = False
        u_pre = self._chain_pre[i, i_b, u]
        u_nxt = self._chain_nxt[i, i_b, u]
        if u_pre >= 0 and u_nxt >= 0:
            x_pre = self.projected_positions[i, i_b, u_pre]
            x_u = self.projected_positions[i, i_b, u]
            x_nxt = self.projected_positions[i, i_b, u_nxt]
            result = self._st_angle_of_vec2(-x_pre, x_u - x_pre) + self._st_angle_of_vec2(-x_nxt, x_u - x_nxt) > math.pi
        return result

    def _st_rebuild_topology(self, f):
        kernel_compute_density_C(f, self)
        self._screen_blocked.fill(0)
        kernel_st_mark_surface_screen(self)
        kernel_st_classify_surface(self)
        kernel_st_compute_normals(self)
        self._overflow.fill(0)
        kernel_st_build_local_meshes(self)
        self._rev_count.fill(0)
        kernel_st_build_reverse_rings(self)
        overflow = qd_to_numpy(self._overflow, transpose=True)[()]
        if overflow == self._ST_LOCAL_MESH_NEIGHBOR_OVERFLOW:
            gs.raise_exception(
                "IPBF-ST local mesh exceeded its one-ring capacity; increase "
                f"`st_max_localmesh_neighbors` (currently {self._st_max_localmesh_neighbors})."
            )
        if overflow == self._ST_REVERSE_RING_OVERFLOW:
            gs.raise_exception(
                "IPBF-ST reverse one-ring index exceeded its capacity; increase "
                f"`st_max_surface_neighbors` (currently {self._st_max_surface_neighbors})."
            )
        if overflow:
            detail = "candidate capacity" if overflow == self._ST_SURFACE_NEIGHBOR_OVERFLOW else "queue capacity"
            gs.raise_exception(
                f"IPBF-ST local mesh exceeded its {detail}; increase `st_max_surface_neighbors` "
                f"(currently {self._st_max_surface_neighbors})."
            )

    @qd.func
    def _st_tri_grad_hess_uu(self, p1, p2, p3, u: qd.i32):
        """Return the triangle-area gradient and diagonal Hessian block for vertex u.

        With opposite edge e and area A, H = (|e|^2 I - e e^T) / (4A) - grad grad^T / A is positive
        semi-definite (PSD). Degenerate triangles contribute zero to keep the inverse-area factors finite.
        """
        grad = qd.Vector.zero(gs.qd_float, 3)
        H_uu = qd.Matrix.zero(gs.qd_float, 3, 3)
        X = (p2 - p1).cross(p3 - p1)
        Xn = X.norm()
        if Xn > gs.EPS:
            A = 0.5 * Xn
            n_hat = X / Xn
            e = p3 - p2
            if u == 2:
                e = p1 - p3
            elif u == 3:
                e = p2 - p1
            grad = 0.5 * n_hat.cross(e)
            H_uu = (e.dot(e) * qd.Matrix.identity(dt=gs.qd_float, n=3) - e.outer_product(e)) / (4.0 * A) - (
                grad.outer_product(grad) / A
            )
        return grad, H_uu

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self.entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self.entities[::-1]:
            entity.process_input_grad()

    @qd.func
    def _solve_newton_direction(
        self,
        alpha_over_h2,
        x_minus_y,
        C_i,
        g_ii,
        A_ii,
        sum_Cj_gij,
        sum_gij_gijT,
        sum_Cj_DAij,
        f_st,
        H_st,
        d_st_diag,
    ):
        """Solve the local symmetric 3x3 system for a particle position update.

        Surface forces and Hessians augment the density and inertia terms. Column-norm diagonal stabilization
        bounds indefinite geometric stiffness; numerically singular systems produce a zero update.
        """
        # f_i: negative gradient (eq. 10)
        f_i = -alpha_over_h2 * x_minus_y - C_i * g_ii - sum_Cj_gij

        # H_i: eq. 11 with the geometric stiffness replaced by its column-norm diagonal
        # approximation (eq. 15, Andrews et al. 2017) -- mandatory for stability
        H_i = alpha_over_h2 * qd.Matrix.identity(dt=gs.qd_float, n=3)
        H_i += g_ii.outer_product(g_ii) + sum_gij_gijT
        for c in qd.static(range(3)):
            H_i[c, c] += C_i * qd.sqrt(A_ii[0, c] ** 2 + A_ii[1, c] ** 2 + A_ii[2, c] ** 2) + sum_Cj_DAij[c]

        if qd.static(self._is_st_enabled):
            f_i += f_st
            H_i += H_st
            for c in qd.static(range(3)):
                H_i[c, c] += d_st_diag[c]

        # 3x3 analytic inverse via adjugate / determinant (H_i is symmetric)
        m00 = H_i[0, 0]
        m01 = H_i[0, 1]
        m02 = H_i[0, 2]
        m11 = H_i[1, 1]
        m12 = H_i[1, 2]
        m22 = H_i[2, 2]
        K00 = m11 * m22 - m12 * m12
        K01 = m02 * m12 - m01 * m22
        K02 = m01 * m12 - m02 * m11
        det = m00 * K00 + m01 * K01 + m02 * K02
        dpos = qd.Vector.zero(gs.qd_float, 3)
        if det > 1e-12 * m00 * m11 * m22:
            K11 = m00 * m22 - m02 * m02
            K12 = m01 * m02 - m00 * m12
            K22 = m00 * m11 - m01 * m01
            inv_det = 1.0 / det
            dpos = inv_det * qd.Vector(
                [
                    K00 * f_i[0] + K01 * f_i[1] + K02 * f_i[2],
                    K01 * f_i[0] + K11 * f_i[1] + K12 * f_i[2],
                    K02 * f_i[0] + K12 * f_i[1] + K22 * f_i[2],
                ],
                dt=gs.qd_float,
            )
        return dpos

    def set_pitcher_pose(self, origin, axis):
        """Update the origin and axis of the configured movable container."""
        if self._has_boundary_pitcher:
            self.boundary2.set_pose(origin, axis)

    def substep_pre_coupling(self, f):
        if self.is_active:
            # Algorithm 1: predict (x <- y) -> hash rebuild -> relaxed-Jacobi Newton iterations -> finalize
            kernel_predict(f, self)
            kernel_build_hash(f, self)
            if self._is_st_enabled:
                if self._st_topo_counter % self._st_topo_interval == 0:
                    self._st_rebuild_topology(f)
                self._st_topo_counter += 1
            for it in range(self._ipbf_iterations):
                if self._is_hash_rebuilt_per_iteration:
                    kernel_build_hash(f, self)
                kernel_compute_density_C(f, self)
                if self._is_st_enabled:
                    kernel_st_compute_area_C(f, self)
                # during the last iteration, also compute the alternative solution x* (artificial damping)
                with_star = 1 if it == self._ipbf_iterations - 1 else 0
                kernel_compute_newton_step(f, with_star, self)
                kernel_apply_relaxed_update(f, self)
            kernel_finalize(f, self)
            if self._viscosity_xsph > 0.0:
                kernel_apply_viscosity(f, self)
            if self._surface_viscosity_xsph > 0.0:
                kernel_apply_surface_viscosity(f, self)
            if self._diffusion_coeff > 0.0:
                kernel_build_hash(f, self)
                kernel_solve_diffusion(f, self)

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if self.is_active:
            kernel_impose_boundary(f, error_code=int(ErrorCode.INVALID_IPBF_STATE_NAN), solver=self, errno=self._errno)

    def check_errno(self):
        errno = np.bitwise_or.reduce(qd_to_numpy(self._errno, transpose=True))
        if errno & ErrorCode.INVALID_IPBF_STATE_NAN:
            gs.raise_exception("Implicit position-based fluid state is non-finite. Reduce the time step or stiffness.")

    def substep_post_coupling_grad(self, f):
        pass

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        pass

    def add_grad_from_state(self, state):
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            assert state.pos.shape[1] == self._n_particles, (
                f"state size mismatch: {state.pos.shape[1]} != {self._n_particles}"
            )
            kernel_set_state(
                f,
                self.scene._sanitize_envs_idx(envs_idx),
                state.pos,
                state.vel,
                state.active,
                state.c,
                state.boundary_group,
                self,
                self._errno,
            )

    def get_state(self, f):
        if self.is_active:
            state = IPBFSolverState(self.scene)
            assert state.pos.shape[1] == self._n_particles, (
                f"state size mismatch: {state.pos.shape[1]} != {self._n_particles}"
            )
            kernel_get_state(f, state.pos, state.vel, state.active, state.c, state.boundary_group, self)
        else:
            state = None
        return state

    def update_render_fields(self):
        kernel_update_render_fields(self.sim.cur_substep_local, self)

    # ----------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    def _kernel_add_particles(
        self, f, active, particle_start, n_particles, mat_rho, mat_c_init, mat_boundary_group, pos
    ):
        kernel_add_particles(f, active, particle_start, n_particles, mat_rho, mat_c_init, mat_boundary_group, pos, self)

    def _kernel_set_particles_pos(self, particles_idx, envs_idx, poss):
        kernel_set_particles_pos(particles_idx, envs_idx, poss, self)

    def _kernel_get_particles_pos(self, particle_start, n_particles, envs_idx, poss):
        kernel_get_particles_pos(particle_start, n_particles, envs_idx, poss, self)

    def _kernel_set_particles_vel(self, particles_idx, envs_idx, vels):
        kernel_set_particles_vel(particles_idx, envs_idx, vels, self)

    def _kernel_get_particles_vel(self, particle_start, n_particles, envs_idx, vels):
        kernel_get_particles_vel(particle_start, n_particles, envs_idx, vels, self)

    def _kernel_get_particles_rho(self, particle_start, n_particles, envs_idx, rhos):
        kernel_get_particles_rho(particle_start, n_particles, envs_idx, rhos, self)

    def _kernel_set_particles_active(self, particles_idx, envs_idx, actives):
        kernel_set_particles_active(particles_idx, envs_idx, actives, self)

    def _kernel_get_particles_active(self, particle_start, n_particles, envs_idx, actives):
        kernel_get_particles_active(particle_start, n_particles, envs_idx, actives, self)

    @property
    def n_particles(self):
        # NOTE: after build, `_n_particles` includes the static boundary particles, which do not belong
        # to any entity. Prefer it whenever it exists so state arrays are sized correctly even when
        # captured while `scene._is_built` is still False (e.g. Scene._init_state during build).
        # `add_entity` queries this property before build (attribute absent) and gets the entity sum.
        n = self._n_particles
        if n is not None:
            return n
        return sum([entity.n_particles for entity in self._entities])

    @property
    def particle_volume(self):
        return self._particle_volume

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

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


@qd.kernel
def kernel_add_boundary_particles(
    particle_start: int, n_particles: int, mat_rho: float, pos: qd.types.ndarray(), solver: V_ANNOTATION
):
    for i_p_, i_b in qd.ndrange(n_particles, solver._B):
        i_p = i_p_ + particle_start
        solver.particles_ng[i_p, i_b].active = True
        solver.particles_ng[i_p, i_b].is_boundary = True
        for i in qd.static(range(3)):
            solver.particles[i_p, i_b].pos[i] = pos[i_p_, i]
        solver.particles[i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
    for i_p_ in range(n_particles):
        i_p = i_p_ + particle_start
        solver.particles_info[i_p].rho = mat_rho
        solver.particles_info[i_p].mass = solver._particle_volume * mat_rho


@qd.kernel
def kernel_st_mark_surface_screen(solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active and (not solver.particles_ng[i_p, i_b].is_boundary):
            pos_i = solver.particles[i_p, i_b].pos
            base = solver.sh.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = solver.sh.grid_to_slot(base + offset)
                for j_r in range(
                    solver.sh.slot_start[slot_idx, i_b],
                    solver.sh.slot_size[slot_idx, i_b] + solver.sh.slot_start[slot_idx, i_b],
                ):
                    j = solver.particle_idx[j_r, i_b]
                    if j != i_p and (not solver.particles_ng[j, i_b].is_boundary):
                        delta = solver.particles[j, i_b].pos - pos_i
                        distance = delta.norm()
                        if distance > gs.EPS and distance < solver._st_ring_radius:
                            unit_theta = math.pi / solver._N_THETA
                            unit_phi = 2.0 * math.pi / solver._N_PHI
                            block_radius = qd.min(0.5 * solver._particle_size, 0.5 * distance)
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
                            st_t = qd.min(qd.cast(qd.floor(start_theta / unit_theta), gs.qd_int), solver._N_THETA - 1)
                            en_t = qd.min(qd.cast(qd.ceil(end_theta / unit_theta), gs.qd_int), solver._N_THETA)
                            st_p = qd.min(
                                qd.cast(qd.floor((start_phi + math.pi) / unit_phi), gs.qd_int), solver._N_PHI - 1
                            )
                            en_p = qd.min(qd.cast(qd.ceil((end_phi + math.pi) / unit_phi), gs.qd_int), solver._N_PHI)
                            if st_p < en_p:
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, st_t, st_p], 1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, st_t, en_p], -1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, en_t, st_p], -1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, en_t, en_p], 1)
                            else:
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, st_t, st_p], 1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, st_t, solver._N_PHI], -1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, en_t, st_p], -1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, en_t, solver._N_PHI], 1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, st_t, 0], 1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, st_t, en_p], -1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, en_t, 0], -1)
                                qd.atomic_add(solver._screen_blocked[i_p, i_b, en_t, en_p], 1)


@qd.kernel
def kernel_st_classify_surface(solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active and (not solver.particles_ng[i_p, i_b].is_boundary):
            illuminated = gs.qd_float(0.0)
            total = gs.qd_float(0.0)
            for t in range(solver._N_THETA):
                weight = qd.sin(math.pi / solver._N_THETA * (t + 0.5))
                for p in range(solver._N_PHI):
                    if t > 0 and p > 0:
                        solver._screen_blocked[i_p, i_b, t, p] += (
                            solver._screen_blocked[i_p, i_b, t - 1, p]
                            + solver._screen_blocked[i_p, i_b, t, p - 1]
                            - solver._screen_blocked[i_p, i_b, t - 1, p - 1]
                        )
                    elif t > 0:
                        solver._screen_blocked[i_p, i_b, t, p] += solver._screen_blocked[i_p, i_b, t - 1, p]
                    elif p > 0:
                        solver._screen_blocked[i_p, i_b, t, p] += solver._screen_blocked[i_p, i_b, t, p - 1]
                    if solver._screen_blocked[i_p, i_b, t, p] == 0:
                        illuminated += weight
                    total += weight
            solver.on_surface[i_p, i_b] = illuminated >= solver._st_lit_threshold * total
        else:
            solver.on_surface[i_p, i_b] = False


@qd.kernel
def kernel_st_compute_normals(solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if (
            solver.particles_ng[i_p, i_b].active
            and (not solver.particles_ng[i_p, i_b].is_boundary)
            and solver.on_surface[i_p, i_b]
        ):
            pos_i = solver.particles[i_p, i_b].pos
            n_raw = qd.Vector.zero(gs.qd_float, 3)
            base = solver.sh.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = solver.sh.grid_to_slot(base + offset)
                for j_r in range(
                    solver.sh.slot_start[slot_idx, i_b],
                    solver.sh.slot_size[slot_idx, i_b] + solver.sh.slot_start[slot_idx, i_b],
                ):
                    j = solver.particle_idx[j_r, i_b]
                    if j != i_p:
                        delta = solver.particles[j, i_b].pos - pos_i
                        distance = delta.norm()
                        if distance > gs.EPS and distance < solver._st_ring_radius:
                            if solver.particles_ng[j, i_b].is_boundary:
                                n_raw += solver._st_cubic_dW(distance) * (delta / distance) * solver._particle_volume
                            else:
                                rho_j = solver.particles[j, i_b].rho
                                if rho_j > gs.EPS:
                                    n_raw += (
                                        solver._st_cubic_dW(distance)
                                        * (delta / distance)
                                        * (solver._particle_volume / rho_j)
                                    )
            solver.normal[i_p, i_b] = n_raw
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if (
            solver.particles_ng[i_p, i_b].active
            and (not solver.particles_ng[i_p, i_b].is_boundary)
            and solver.on_surface[i_p, i_b]
        ):
            raw_normal = solver.normal[i_p, i_b]
            raw_length = raw_normal.norm()
            if raw_length <= 1.0:
                solver.on_surface[i_p, i_b] = False
                solver.normal[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
            else:
                solver.normal[i_p, i_b] = raw_normal / raw_length


@qd.kernel
def kernel_st_build_local_meshes(solver: V_ANNOTATION):
    solver.topology_valid.fill(False)
    solver.n_ring.fill(0)
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if (
            solver.particles_ng[i_p, i_b].active
            and (not solver.particles_ng[i_p, i_b].is_boundary)
            and solver.on_surface[i_p, i_b]
        ):
            pos_i = solver.particles[i_p, i_b].pos
            normal = solver.normal[i_p, i_b]
            axis_x = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
            if axis_x.cross(normal).norm() < gs.EPS:
                axis_x = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
            axis_x = axis_x.cross(normal).normalized()
            axis_y = normal.cross(axis_x).normalized()
            solver._mesh_axis_x[i_p, i_b] = axis_x
            solver._mesh_axis_y[i_p, i_b] = axis_y
            count = gs.qd_int(0)
            cand_overflow = False
            base = solver.sh.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = solver.sh.grid_to_slot(base + offset)
                for j_r in range(
                    solver.sh.slot_start[slot_idx, i_b],
                    solver.sh.slot_size[slot_idx, i_b] + solver.sh.slot_start[slot_idx, i_b],
                ):
                    j = solver.particle_idx[j_r, i_b]
                    if j != i_p and solver.on_surface[j, i_b]:
                        delta = solver.particles[j, i_b].pos - pos_i
                        distance = delta.norm()
                        if distance < solver._st_ring_radius:
                            normal_j = solver.normal[j, i_b]
                            if normal.dot(normal_j) > solver._st_normal_compat or (
                                (normal - normal_j).dot(delta) > 0.0 and distance < 2.0 * solver._particle_size
                            ):
                                if count < solver._st_max_surface_neighbors:
                                    solver.neighbor_ids[i_p, i_b, count] = j
                                    projected = delta - delta.dot(normal) * normal
                                    solver.projected_positions[i_p, i_b, count] = qd.Vector(
                                        [projected.dot(axis_x), projected.dot(axis_y)], dt=gs.qd_float
                                    )
                                    count += 1
                                else:
                                    cand_overflow = True
            if cand_overflow:
                qd.atomic_max(solver._overflow[None], solver._ST_SURFACE_NEIGHBOR_OVERFLOW)
            n = count
            solver.n_ring[i_p, i_b] = n
            for k in range(n):
                solver._node_queue[i_p, i_b, k] = k
                solver._chain_pre[i_p, i_b, k] = -1
                solver._chain_nxt[i_p, i_b, k] = -1
            for k in range(1, n):
                cursor = k
                while cursor > 0:
                    u = solver._node_queue[i_p, i_b, cursor - 1]
                    v = solver._node_queue[i_p, i_b, cursor]
                    x_u = solver.projected_positions[i_p, i_b, u]
                    x_v = solver.projected_positions[i_p, i_b, v]
                    angle_u = qd.atan2(x_u[1], x_u[0])
                    angle_v = qd.atan2(x_v[1], x_v[0])
                    if angle_u > angle_v or (angle_u == angle_v and x_u.norm() > x_v.norm()):
                        solver._node_queue[i_p, i_b, cursor - 1] = v
                        solver._node_queue[i_p, i_b, cursor] = u
                        cursor -= 1
                    else:
                        cursor = 0
            ring_size = gs.qd_int(0)
            if n >= 3:
                for k in range(n - 1):
                    u = solver._node_queue[i_p, i_b, k]
                    v = solver._node_queue[i_p, i_b, k + 1]
                    solver._chain_nxt[i_p, i_b, u] = v
                    solver._chain_pre[i_p, i_b, v] = u
                u_last = solver._node_queue[i_p, i_b, n - 1]
                u_first = solver._node_queue[i_p, i_b, 0]
                solver._chain_nxt[i_p, i_b, u_last] = u_first
                solver._chain_pre[i_p, i_b, u_first] = u_last
                queue_start = gs.qd_int(0)
                queue_end = gs.qd_int(0)
                for k in range(n):
                    u = solver._node_queue[i_p, i_b, k]
                    if solver._st_need_flip(i_p, i_b, u):
                        if queue_end < 3 * solver._st_max_surface_neighbors:
                            solver._node_queue[i_p, i_b, queue_end] = u
                            queue_end += 1
                        else:
                            qd.atomic_max(solver._overflow[None], solver._ST_LOCAL_MESH_QUEUE_OVERFLOW)
                while queue_start < queue_end:
                    u = solver._node_queue[i_p, i_b, queue_start]
                    queue_start += 1
                    if solver._chain_nxt[i_p, i_b, u] >= 0 and solver._st_need_flip(i_p, i_b, u):
                        u_pre = solver._chain_pre[i_p, i_b, u]
                        u_nxt = solver._chain_nxt[i_p, i_b, u]
                        solver._chain_nxt[i_p, i_b, u_pre] = u_nxt
                        solver._chain_pre[i_p, i_b, u_nxt] = u_pre
                        if solver._st_need_flip(i_p, i_b, u_nxt):
                            if queue_end < 3 * solver._st_max_surface_neighbors:
                                solver._node_queue[i_p, i_b, queue_end] = u_nxt
                                queue_end += 1
                            else:
                                qd.atomic_max(solver._overflow[None], solver._ST_LOCAL_MESH_QUEUE_OVERFLOW)
                        if solver._st_need_flip(i_p, i_b, u_pre):
                            if queue_end < 3 * solver._st_max_surface_neighbors:
                                solver._node_queue[i_p, i_b, queue_end] = u_pre
                                queue_end += 1
                            else:
                                qd.atomic_max(solver._overflow[None], solver._ST_LOCAL_MESH_QUEUE_OVERFLOW)
                        solver._chain_nxt[i_p, i_b, u] = -1
                        solver._chain_pre[i_p, i_b, u] = -1
                start = gs.qd_int(-1)
                for k in range(n):
                    if start < 0 and solver._chain_nxt[i_p, i_b, k] >= 0:
                        start = k
                projected_area_twice = gs.qd_float(0.0)
                if start >= 0:
                    u = start
                    keep_walking = True
                    while keep_walking and ring_size < solver._st_max_localmesh_neighbors:
                        u_next = solver._chain_nxt[i_p, i_b, u]
                        x_u = solver.projected_positions[i_p, i_b, u]
                        x_next = solver.projected_positions[i_p, i_b, u_next]
                        projected_area_twice += x_u[0] * x_next[1] - x_u[1] * x_next[0]
                        solver.local_mesh_neighbors[i_p, i_b, ring_size] = solver.neighbor_ids[i_p, i_b, u]
                        ring_size += 1
                        u = u_next
                        if u == start:
                            keep_walking = False
                    if keep_walking:
                        qd.atomic_max(solver._overflow[None], solver._ST_LOCAL_MESH_NEIGHBOR_OVERFLOW)
                    else:
                        solver.topology_valid[i_p, i_b] = ring_size >= 3 and qd.abs(projected_area_twice) > gs.EPS
                solver.n_ring[i_p, i_b] = ring_size


@qd.kernel
def kernel_st_build_reverse_rings(solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if (
            solver.particles_ng[i_p, i_b].active
            and (not solver.particles_ng[i_p, i_b].is_boundary)
            and solver.on_surface[i_p, i_b]
            and solver.topology_valid[i_p, i_b]
        ):
            n = solver.n_ring[i_p, i_b]
            for k in range(n):
                u = solver.local_mesh_neighbors[i_p, i_b, k]
                slot = qd.atomic_add(solver._rev_count[u, i_b], 1)
                if slot < solver._st_max_surface_neighbors:
                    solver._rev_ids[u, i_b, slot] = i_p * (1 << solver._ST_REV_SHIFT) + k
                else:
                    qd.atomic_max(solver._overflow[None], solver._ST_REVERSE_RING_OVERFLOW)


@qd.kernel
def kernel_st_compute_area_C(f: int, solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        CA = gs.qd_float(0.0)
        if (
            solver.particles_ng[i_p, i_b].active
            and (not solver.particles_ng[i_p, i_b].is_boundary)
            and solver.on_surface[i_p, i_b]
            and solver.topology_valid[i_p, i_b]
        ):
            pos_i = solver.particles[i_p, i_b].pos
            n = solver.n_ring[i_p, i_b]
            for k in range(n):
                k_next = 0 if k == n - 1 else k + 1
                j1 = solver.local_mesh_neighbors[i_p, i_b, k]
                j2 = solver.local_mesh_neighbors[i_p, i_b, k_next]
                CA += 0.5 * (solver.particles[j1, i_b].pos - pos_i).cross(solver.particles[j2, i_b].pos - pos_i).norm()
        solver.CA[i_p, i_b] = CA


@qd.kernel
def kernel_predict(f: int, solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active and (not solver.particles_ng[i_p, i_b].is_boundary):
            solver.particles[i_p, i_b].ipos = solver.particles[i_p, i_b].pos
            y = solver.particles[i_p, i_b].pos + solver._substep_dt * (
                solver.particles[i_p, i_b].vel + solver._substep_dt * solver._gravity[i_b]
            )
            solver.particles[i_p, i_b].y = y
            solver.particles[i_p, i_b].pos = y


@qd.kernel
def kernel_build_hash(f: int, solver: V_ANNOTATION):
    solver.sh.compute_reordered_idx(
        solver._n_particles, solver.particles.pos, solver.particles_ng.active, solver.particles_ng.reordered_idx
    )
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active:
            solver.particle_idx[solver.particles_ng[i_p, i_b].reordered_idx, i_b] = i_p


@qd.kernel
def kernel_compute_density_C(f: int, solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active and (not solver.particles_ng[i_p, i_b].is_boundary):
            pos_i = solver.particles[i_p, i_b].pos
            rho = solver._particle_volume * solver.cubic_kernel_W(0.0)
            base = solver.sh.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = solver.sh.grid_to_slot(base + offset)
                for j_r in range(
                    solver.sh.slot_start[slot_idx, i_b],
                    solver.sh.slot_size[slot_idx, i_b] + solver.sh.slot_start[slot_idx, i_b],
                ):
                    j = solver.particle_idx[j_r, i_b]
                    if j != i_p:
                        r = (solver.particles[j, i_b].pos - pos_i).norm()
                        if r < solver._support_radius:
                            rho += solver._particle_volume * solver.cubic_kernel_W(r)
            solver.particles[i_p, i_b].rho = rho
            solver.particles[i_p, i_b].C = qd.max(rho - 1.0, 0.0)


@qd.kernel
def kernel_compute_newton_step(f: int, with_star: int, solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active and (not solver.particles_ng[i_p, i_b].is_boundary):
            pos_i = solver.particles[i_p, i_b].pos
            C_i = solver.particles[i_p, i_b].C
            m_i = solver.particles_info[i_p].mass
            V = solver._particle_volume
            g_ii = qd.Vector.zero(gs.qd_float, 3)
            A_ii = qd.Matrix.zero(gs.qd_float, 3, 3)
            sum_Cj_gij = qd.Vector.zero(gs.qd_float, 3)
            sum_gij_gijT = qd.Matrix.zero(gs.qd_float, 3, 3)
            sum_Cj_DAij = qd.Vector.zero(gs.qd_float, 3)
            f_st = qd.Vector.zero(gs.qd_float, 3)
            H_st = qd.Matrix.zero(gs.qd_float, 3, 3)
            d_st_diag = qd.Vector.zero(gs.qd_float, 3)
            base = solver.sh.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = solver.sh.grid_to_slot(base + offset)
                for j_r in range(
                    solver.sh.slot_start[slot_idx, i_b],
                    solver.sh.slot_size[slot_idx, i_b] + solver.sh.slot_start[slot_idx, i_b],
                ):
                    j = solver.particle_idx[j_r, i_b]
                    if j != i_p:
                        r_vec = solver.particles[j, i_b].pos - pos_i
                        r = r_vec.norm()
                        if r >= solver._eps_r and r < solver._support_radius:
                            r_hat = r_vec / r
                            g_ij = -V * solver.cubic_kernel_dW(r) * r_hat
                            g_ii += g_ij
                            A_ij = V * solver.cubic_kernel_hessian(r_vec)
                            A_ii += A_ij
                            C_j = solver.particles[j, i_b].C
                            sum_Cj_gij += C_j * g_ij
                            sum_gij_gijT += g_ij.outer_product(g_ij)
                            for c in qd.static(range(3)):
                                sum_Cj_DAij[c] += C_j * qd.sqrt(A_ij[0, c] ** 2 + A_ij[1, c] ** 2 + A_ij[2, c] ** 2)
                            if qd.static(solver._is_st_enabled and solver._is_st_distance_enabled):
                                if (
                                    r < solver._particle_size
                                    and (not solver.particles_ng[j, i_b].is_boundary)
                                    and (solver.on_surface[i_p, i_b] == solver.on_surface[j, i_b])
                                ):
                                    Cd = r - solver._particle_size
                                    kd = solver._st_distance_stiffness
                                    f_st += kd * Cd * r_hat
                                    H_st += kd * r_hat.outer_product(r_hat)
                                    for c in qd.static(range(3)):
                                        d_st_diag[c] += kd * Cd * qd.sqrt(1.0 - r_hat[c] ** 2) / r
            ddW0 = V * solver.cubic_kernel_ddW(0.0)
            for c in qd.static(range(3)):
                A_ii[c, c] += ddW0
            if qd.static(solver._is_st_enabled):
                w_st = solver._st_area_weight
                if solver.on_surface[i_p, i_b] and solver.topology_valid[i_p, i_b]:
                    grad_i = qd.Vector.zero(gs.qd_float, 3)
                    H_geo = qd.Matrix.zero(gs.qd_float, 3, 3)
                    n = solver.n_ring[i_p, i_b]
                    for k in range(n):
                        k_next = 0 if k == n - 1 else k + 1
                        j1 = solver.local_mesh_neighbors[i_p, i_b, k]
                        j2 = solver.local_mesh_neighbors[i_p, i_b, k_next]
                        g1, H11 = solver._st_tri_grad_hess_uu(
                            pos_i, solver.particles[j1, i_b].pos, solver.particles[j2, i_b].pos, 1
                        )
                        grad_i += g1
                        H_geo += H11
                    if qd.static(solver._is_st_model_linear):
                        f_st -= w_st * grad_i
                        H_st += w_st * H_geo
                    else:
                        CA_i = solver.CA[i_p, i_b]
                        f_st -= w_st * CA_i * grad_i
                        H_st += w_st * (grad_i.outer_product(grad_i) + CA_i * H_geo)
                for s in range(solver._rev_count[i_p, i_b]):
                    entry = solver._rev_ids[i_p, i_b, s]
                    j = entry // (1 << solver._ST_REV_SHIFT)
                    kk = entry % (1 << solver._ST_REV_SHIFT)
                    nj = solver.n_ring[j, i_b]
                    kk_next = 0 if kk == nj - 1 else kk + 1
                    kk_prev = nj - 1 if kk == 0 else kk - 1
                    pos_j = solver.particles[j, i_b].pos
                    pos_k = solver.particles[solver.local_mesh_neighbors[j, i_b, kk], i_b].pos
                    g2, H22 = solver._st_tri_grad_hess_uu(
                        pos_j, pos_k, solver.particles[solver.local_mesh_neighbors[j, i_b, kk_next], i_b].pos, 2
                    )
                    g3, H33 = solver._st_tri_grad_hess_uu(
                        pos_j, solver.particles[solver.local_mesh_neighbors[j, i_b, kk_prev], i_b].pos, pos_k, 3
                    )
                    grad_ij = g2 + g3
                    H_geo_ij = H22 + H33
                    if qd.static(solver._is_st_model_linear):
                        f_st -= w_st * grad_ij
                        H_st += w_st * H_geo_ij
                    else:
                        CA_j = solver.CA[j, i_b]
                        f_st -= w_st * CA_j * grad_ij
                        H_st += w_st * (grad_ij.outer_product(grad_ij) + CA_j * H_geo_ij)
            x_minus_y = pos_i - solver.particles[i_p, i_b].y
            solver.particles[i_p, i_b].dpos = solver._solve_newton_direction(
                solver._alpha_eff * m_i / solver._substep_dt**2,
                x_minus_y,
                C_i,
                g_ii,
                A_ii,
                sum_Cj_gij,
                sum_gij_gijT,
                sum_Cj_DAij,
                f_st,
                H_st,
                d_st_diag,
            )
            if qd.static(solver._is_damping_enabled):
                if with_star == 1:
                    dpos_star = solver._solve_newton_direction(
                        solver._damping_alpha_star * m_i / solver._substep_dt**2,
                        x_minus_y,
                        C_i,
                        g_ii,
                        A_ii,
                        sum_Cj_gij,
                        sum_gij_gijT,
                        sum_Cj_DAij,
                        f_st,
                        H_st,
                        d_st_diag,
                    )
                    solver.particles[i_p, i_b].xstar = pos_i + 0.5 * dpos_star


@qd.kernel
def kernel_apply_relaxed_update(f: int, solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active and (not solver.particles_ng[i_p, i_b].is_boundary):
            solver.particles[i_p, i_b].pos += 0.5 * solver.particles[i_p, i_b].dpos


@qd.kernel
def kernel_finalize(f: int, solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active and (not solver.particles_ng[i_p, i_b].is_boundary):
            v = (solver.particles[i_p, i_b].pos - solver.particles[i_p, i_b].ipos) / solver._substep_dt
            if qd.static(solver._is_damping_enabled):
                dist = (solver.particles[i_p, i_b].xstar - solver.particles[i_p, i_b].pos).norm()
                beta_r = solver._damping_beta * solver._support_radius
                if dist < beta_r:
                    v_star = (solver.particles[i_p, i_b].xstar - solver.particles[i_p, i_b].ipos) / solver._substep_dt
                    v2 = v.norm_sqr()
                    v_star2 = v_star.norm_sqr()
                    if v_star2 < v2 and v2 > 0.0:
                        d = 1.0 - dist / beta_r
                        v = v * qd.sqrt(1.0 - d * (v2 - v_star2) / v2)
            solver.particles[i_p, i_b].vel = v


@qd.kernel
def kernel_apply_viscosity(f: int, solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active and (not solver.particles_ng[i_p, i_b].is_boundary):
            pos_i = solver.particles[i_p, i_b].pos
            vel_i = solver.particles[i_p, i_b].vel
            dv = qd.Vector.zero(gs.qd_float, 3)
            base = solver.sh.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = solver.sh.grid_to_slot(base + offset)
                for j_r in range(
                    solver.sh.slot_start[slot_idx, i_b],
                    solver.sh.slot_size[slot_idx, i_b] + solver.sh.slot_start[slot_idx, i_b],
                ):
                    j = solver.particle_idx[j_r, i_b]
                    if j != i_p and (not solver.particles_ng[j, i_b].is_boundary):
                        r = (solver.particles[j, i_b].pos - pos_i).norm()
                        if r >= solver._eps_r and r < solver._support_radius:
                            dv += (solver.particles[j, i_b].vel - vel_i) * solver.cubic_kernel_W(r)
            solver.particles[i_p, i_b].dv = solver._viscosity_xsph * solver._particle_volume * dv
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active and (not solver.particles_ng[i_p, i_b].is_boundary):
            solver.particles[i_p, i_b].vel += solver.particles[i_p, i_b].dv


@qd.kernel
def kernel_apply_surface_viscosity(f: int, solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if (
            solver.particles_ng[i_p, i_b].active
            and (not solver.particles_ng[i_p, i_b].is_boundary)
            and solver.on_surface[i_p, i_b]
        ):
            pos_i = solver.particles[i_p, i_b].pos
            vel_i = solver.particles[i_p, i_b].vel
            dv = qd.Vector.zero(gs.qd_float, 3)
            base = solver.sh.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = solver.sh.grid_to_slot(base + offset)
                for j_r in range(
                    solver.sh.slot_start[slot_idx, i_b],
                    solver.sh.slot_size[slot_idx, i_b] + solver.sh.slot_start[slot_idx, i_b],
                ):
                    j = solver.particle_idx[j_r, i_b]
                    if j != i_p and (not solver.particles_ng[j, i_b].is_boundary) and solver.on_surface[j, i_b]:
                        r = (solver.particles[j, i_b].pos - pos_i).norm()
                        if r >= solver._eps_r and r < solver._support_radius:
                            dv += (solver.particles[j, i_b].vel - vel_i) * solver.cubic_kernel_W(r)
            solver.particles[i_p, i_b].dv = solver._surface_viscosity_xsph * solver._particle_volume * dv
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if (
            solver.particles_ng[i_p, i_b].active
            and (not solver.particles_ng[i_p, i_b].is_boundary)
            and solver.on_surface[i_p, i_b]
        ):
            solver.particles[i_p, i_b].vel += solver.particles[i_p, i_b].dv


@qd.kernel
def kernel_solve_diffusion(f: int, solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active and (not solver.particles_ng[i_p, i_b].is_boundary):
            pos_i = solver.particles[i_p, i_b].pos
            c_i = solver.particles[i_p, i_b].c
            dc = gs.qd_float(0.0)
            base = solver.sh.pos_to_grid(pos_i)
            for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = solver.sh.grid_to_slot(base + offset)
                for j_r in range(
                    solver.sh.slot_start[slot_idx, i_b],
                    solver.sh.slot_size[slot_idx, i_b] + solver.sh.slot_start[slot_idx, i_b],
                ):
                    j = solver.particle_idx[j_r, i_b]
                    if j != i_p and (not solver.particles_ng[j, i_b].is_boundary):
                        r = (solver.particles[j, i_b].pos - pos_i).norm()
                        if r >= solver._eps_r and r < solver._support_radius:
                            dc += (solver.particles[j, i_b].c - c_i) * solver.cubic_kernel_W(r)
            solver.particles[i_p, i_b].dc = solver._diffusion_coeff * solver._particle_volume * dc
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active and (not solver.particles_ng[i_p, i_b].is_boundary):
            solver.particles[i_p, i_b].c += solver.particles[i_p, i_b].dc


@qd.kernel
def kernel_impose_boundary(f: int, error_code: int, solver: V_ANNOTATION, errno: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active and (not solver.particles_ng[i_p, i_b].is_boundary):
            corrected_pos = solver.particles[i_p, i_b].pos
            corrected_vel = solver.particles[i_p, i_b].vel
            gid = solver.particles_ng[i_p, i_b].boundary_group
            if qd.static(solver._has_boundary_pitcher):
                if gid == 1 and (
                    not solver.boundary2.in_keep_region(solver.particles[i_p, i_b].pos, solver._particle_size)
                ):
                    gid = 0
                    solver.particles_ng[i_p, i_b].boundary_group = 0
                if gid == 1:
                    corrected_pos, corrected_vel = solver.boundary2.impose_pos_vel(corrected_pos, corrected_vel)
                else:
                    corrected_pos, corrected_vel = solver.boundary.impose_pos_vel(corrected_pos, corrected_vel)
            else:
                corrected_pos, corrected_vel = solver.boundary.impose_pos_vel(corrected_pos, corrected_vel)
            if qd.static(solver._has_boundary_plane):
                corrected_pos, corrected_vel = solver.boundary3.impose_pos_vel(corrected_pos, corrected_vel)
            solver.particles[i_p, i_b].pos = corrected_pos
            solver.particles[i_p, i_b].vel = corrected_vel
            c = solver.particles[i_p, i_b].c
            is_valid = not qd.math.isnan(c) and not qd.math.isinf(c)
            for axis in qd.static(range(3)):
                is_valid = (
                    is_valid
                    and not qd.math.isnan(corrected_pos[axis])
                    and not qd.math.isinf(corrected_pos[axis])
                    and not qd.math.isnan(corrected_vel[axis])
                    and not qd.math.isinf(corrected_vel[axis])
                )
            if not is_valid:
                qd.atomic_or(errno[i_b], error_code)


@qd.kernel
def kernel_set_state(
    f: int,
    envs_idx: qd.types.ndarray(),
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    active: qd.types.ndarray(),
    c: qd.types.ndarray(),
    boundary_group: qd.types.ndarray(),
    solver: V_ANNOTATION,
    errno: V_ANNOTATION,
):
    for i_b_local in range(envs_idx.shape[0]):
        errno[envs_idx[i_b_local]] = 0
    for i_p, i_b_local in qd.ndrange(solver._n_particles, envs_idx.shape[0]):
        i_b = envs_idx[i_b_local]
        for j in qd.static(range(3)):
            solver.particles[i_p, i_b].pos[j] = pos[i_b, i_p, j]
            solver.particles[i_p, i_b].vel[j] = vel[i_b, i_p, j]
        solver.particles_ng[i_p, i_b].active = active[i_b, i_p]
        solver.particles[i_p, i_b].c = c[i_b, i_p]
        solver.particles_ng[i_p, i_b].boundary_group = boundary_group[i_b, i_p]


@qd.kernel
def kernel_get_state(
    f: int,
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    active: qd.types.ndarray(),
    c: qd.types.ndarray(),
    boundary_group: qd.types.ndarray(),
    solver: V_ANNOTATION,
):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        for j in qd.static(range(3)):
            pos[i_b, i_p, j] = solver.particles[i_p, i_b].pos[j]
            vel[i_b, i_p, j] = solver.particles[i_p, i_b].vel[j]
        active[i_b, i_p] = solver.particles_ng[i_p, i_b].active
        c[i_b, i_p] = solver.particles[i_p, i_b].c
        boundary_group[i_b, i_p] = solver.particles_ng[i_p, i_b].boundary_group


@qd.kernel
def kernel_update_render_fields(f: int, solver: V_ANNOTATION):
    for i_p, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles_ng[i_p, i_b].active and (not solver.particles_ng[i_p, i_b].is_boundary):
            solver.particles_render[i_p, i_b].pos = solver.particles[i_p, i_b].pos
            solver.particles_render[i_p, i_b].vel = solver.particles[i_p, i_b].vel
            solver.particles_render[i_p, i_b].c = solver.particles[i_p, i_b].c
            solver.particles_render[i_p, i_b].active = True
        else:
            solver.particles_render[i_p, i_b].pos = gu.qd_nowhere()
            solver.particles_render[i_p, i_b].active = False


@qd.kernel
def kernel_add_particles(
    f: int,
    active: int,
    particle_start: int,
    n_particles: int,
    mat_rho: float,
    mat_c_init: float,
    mat_boundary_group: int,
    pos: qd.types.ndarray(),
    solver: V_ANNOTATION,
):
    for i_p_, i_b in qd.ndrange(n_particles, solver._B):
        i_p = i_p_ + particle_start
        solver.particles_ng[i_p, i_b].active = qd.cast(active, gs.qd_bool)
        solver.particles_ng[i_p, i_b].boundary_group = mat_boundary_group
        for i in qd.static(range(3)):
            solver.particles[i_p, i_b].pos[i] = pos[i_p_, i]
        solver.particles[i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
        solver.particles[i_p, i_b].c = mat_c_init
        solver.particles[i_p, i_b].dc = 0.0
    for i_p_ in range(n_particles):
        i_p = i_p_ + particle_start
        solver.particles_info[i_p].rho = mat_rho
        solver.particles_info[i_p].mass = solver._particle_volume * mat_rho


@qd.kernel
def kernel_set_particles_pos(
    particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), poss: qd.types.ndarray(), solver: V_ANNOTATION
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        for i in qd.static(range(3)):
            solver.particles[i_p, i_b].pos[i] = poss[i_b_, i_p_, i]
        solver.particles[i_p, i_b].vel.fill(0.0)


@qd.kernel
def kernel_get_particles_pos(
    particle_start: int, n_particles: int, envs_idx: qd.types.ndarray(), poss: qd.types.ndarray(), solver: V_ANNOTATION
):
    for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
        i_p = i_p_ + particle_start
        i_b = envs_idx[i_b_]
        for i in qd.static(range(3)):
            poss[i_b_, i_p_, i] = solver.particles[i_p, i_b].pos[i]


@qd.kernel
def kernel_set_particles_vel(
    particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), vels: qd.types.ndarray(), solver: V_ANNOTATION
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        for i in qd.static(range(3)):
            solver.particles[i_p, i_b].vel[i] = vels[i_b_, i_p_, i]


@qd.kernel
def kernel_get_particles_vel(
    particle_start: int, n_particles: int, envs_idx: qd.types.ndarray(), vels: qd.types.ndarray(), solver: V_ANNOTATION
):
    for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
        i_p = i_p_ + particle_start
        i_b = envs_idx[i_b_]
        for i in qd.static(range(3)):
            vels[i_b_, i_p_, i] = solver.particles[i_p, i_b].vel[i]


@qd.kernel
def kernel_get_particles_rho(
    particle_start: int, n_particles: int, envs_idx: qd.types.ndarray(), rhos: qd.types.ndarray(), solver: V_ANNOTATION
):
    for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
        i_p = i_p_ + particle_start
        i_b = envs_idx[i_b_]
        rhos[i_b_, i_p_] = solver.particles[i_p, i_b].rho


@qd.kernel
def kernel_set_particles_active(
    particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), actives: qd.types.ndarray(), solver: V_ANNOTATION
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        solver.particles_ng[i_p, i_b].active = actives[i_b_, i_p_]


@qd.kernel
def kernel_get_particles_active(
    particle_start: int,
    n_particles: int,
    envs_idx: qd.types.ndarray(),
    actives: qd.types.ndarray(),
    solver: V_ANNOTATION,
):
    for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
        i_p = i_p_ + particle_start
        i_b = envs_idx[i_b_]
        actives[i_b_, i_p_] = solver.particles_ng[i_p, i_b].active
