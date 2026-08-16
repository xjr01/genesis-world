from typing import TYPE_CHECKING

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import IPBFEntity
from genesis.engine.states.solvers import IPBFSolverState

from .base_solver import Solver

if TYPE_CHECKING:
    from genesis.engine.entities import IPBFEntity


@qd.data_oriented
class IPBFSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._particle_size = options.particle_size
        self._support_radius = options._support_radius

        # IPBF parameters (damping_beta is reserved for the artificial damping of Phase 3)
        self._ipbf_iterations = options.ipbf_iterations
        self._alpha = options.alpha
        self._damping_beta = options.damping_beta

        # numerical guard (PLAN P2.4): skip neighbor pairs closer than eps * support radius
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
        self.boundary = CubeBoundary(
            lower=self._lower_bound,
            upper=self._upper_bound,
        )

    def init_particle_fields(self):
        # dynamic particle state
        struct_particle_state = qd.types.struct(
            pos=gs.qd_vec3,  # position
            vel=gs.qd_vec3,  # velocity
            rho=gs.qd_float,  # volume-normalized density sum_j V W (diagnostic)
            ipos=gs.qd_vec3,  # position at the start of the substep (x^t)
            y=gs.qd_vec3,  # inertial position (eq. 3)
            dpos=gs.qd_vec3,  # Newton step (Delta x) scratch
            C=gs.qd_float,  # clamped density constraint max(rho - 1, 0)
        )

        # dynamic particle state without gradient
        struct_particle_state_ng = qd.types.struct(
            reordered_idx=gs.qd_int,
            active=gs.qd_bool,
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
            self.sh.build(self._B)
            self.init_particle_fields()
            self.init_ckpt()

            for entity in self.entities:
                entity._add_to_solver()

        # FIXME: _gravity must be a raw qd.field() — see comment in mpm_solver.py
        # Only when active — see the SNode-tree note in mpm_solver.py.
        if self.is_active and self._gravity is not None:
            gravity = self._gravity.to_numpy()
            self._gravity = qd.field(dtype=gs.qd_vec3, shape=(self._B,))
            self._gravity.from_numpy(gravity)

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

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self.entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self.entities[::-1]:
            entity.process_input_grad()

    @qd.kernel
    def _kernel_predict(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                self.particles[i_p, i_b].ipos = self.particles[i_p, i_b].pos
                # inertial position y = x^t + h (v^t + h a*), a* = gravity (eq. 3; viscosity joins a* in Phase 3)
                y = self.particles[i_p, i_b].pos + self._substep_dt * (
                    self.particles[i_p, i_b].vel + self._substep_dt * self._gravity[i_b]
                )
                self.particles[i_p, i_b].y = y
                # initial guess x <- y (the only momentum entry when alpha = 0)
                self.particles[i_p, i_b].pos = y

    @qd.kernel
    def _kernel_build_hash(self, f: qd.i32):
        # rebuilt once per substep; the neighborhood structure stays fixed across Newton iterations
        self.sh.compute_reordered_idx(
            self._n_particles, self.particles.pos, self.particles_ng.active, self.particles_ng.reordered_idx
        )
        # hash slots hold reordered positions; map them back to original particle indices
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                self.particle_idx[self.particles_ng[i_p, i_b].reordered_idx, i_b] = i_p

    @qd.kernel
    def _kernel_compute_density_C(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                pos_i = self.particles[i_p, i_b].pos
                # volume-normalized density rho~_i = sum_j V W_ij (includes the self term V W(0))
                rho = self._particle_volume * self.cubic_kernel_W(0.0)
                base = self.sh.pos_to_grid(pos_i)
                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    slot_idx = self.sh.grid_to_slot(base + offset)
                    for j_r in range(
                        self.sh.slot_start[slot_idx, i_b],
                        self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                    ):
                        j = self.particle_idx[j_r, i_b]
                        if j != i_p:
                            r = (self.particles[j, i_b].pos - pos_i).norm()
                            if r < self._support_radius:
                                rho += self._particle_volume * self.cubic_kernel_W(r)
                self.particles[i_p, i_b].rho = rho
                # negative-pressure clamp: C_i = max(rho~_i - 1, 0) (paper: enabled in all tests)
                self.particles[i_p, i_b].C = qd.max(rho - 1.0, 0.0)

    @qd.kernel
    def _kernel_compute_newton_step(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                pos_i = self.particles[i_p, i_b].pos
                C_i = self.particles[i_p, i_b].C
                m_i = self.particles_info[i_p].mass
                V = self._particle_volume

                g_ii = qd.Vector.zero(gs.qd_float, 3)  # dC_i/dx_i = V sum_k grad W(x_i - x_k)
                A_ii = qd.Matrix.zero(gs.qd_float, 3, 3)  # d2C_i/dx_i^2 = V sum_k H_W(x_i - x_k)
                sum_Cj_gij = qd.Vector.zero(gs.qd_float, 3)  # sum_j C_j g_ij
                sum_gij_gijT = qd.Matrix.zero(gs.qd_float, 3, 3)  # Gauss-Newton neighbor terms
                # diagonal of sum_j C_j D(A_ij); D(A) = diag of column norms (Andrews et al. 2017)
                sum_Cj_DAij = qd.Vector.zero(gs.qd_float, 3)

                base = self.sh.pos_to_grid(pos_i)
                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    slot_idx = self.sh.grid_to_slot(base + offset)
                    for j_r in range(
                        self.sh.slot_start[slot_idx, i_b],
                        self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                    ):
                        j = self.particle_idx[j_r, i_b]
                        if j != i_p:
                            r_vec = self.particles[j, i_b].pos - pos_i  # x_j - x_i
                            r = r_vec.norm()
                            # skip degenerate neighbor pairs (PLAN P2.4); W'/r is regular as r -> 0
                            if r >= self._eps_r and r < self._support_radius:
                                r_hat = r_vec / r
                                # g_ij = -V grad W(x_j - x_i) = -V W'(r) r_hat;
                                # identical to the k=j term of g_ii since grad W(x_i - x_j) = -W'(r) r_hat
                                g_ij = -V * self.cubic_kernel_dW(r) * r_hat
                                g_ii += g_ij
                                # A_ij = V H_W(x_j - x_i) (H_W is even in its argument)
                                A_ij = V * self.cubic_kernel_hessian(r_vec)
                                A_ii += A_ij
                                C_j = self.particles[j, i_b].C
                                sum_Cj_gij += C_j * g_ij
                                sum_gij_gijT += g_ij.outer_product(g_ij)
                                for c in qd.static(range(3)):
                                    sum_Cj_DAij[c] += C_j * qd.sqrt(
                                        A_ij[0, c] ** 2 + A_ij[1, c] ** 2 + A_ij[2, c] ** 2
                                    )

                # self term of A_ii: r -> 0 limit H_W(0) = W''(0) I
                ddW0 = V * self.cubic_kernel_ddW(0.0)
                for c in qd.static(range(3)):
                    A_ii[c, c] += ddW0

                alpha_over_h2 = self._alpha * m_i / self._substep_dt**2
                # f_i: negative gradient (eq. 10)
                f_i = -alpha_over_h2 * (pos_i - self.particles[i_p, i_b].y) - C_i * g_ii - sum_Cj_gij

                # H_i: eq. 11 with the geometric stiffness replaced by its column-norm diagonal
                # approximation (eq. 15, Andrews et al. 2017) — mandatory for stability
                H_i = alpha_over_h2 * qd.Matrix.identity(dt=gs.qd_float, n=3)
                H_i += g_ii.outer_product(g_ii) + sum_gij_gijT
                for c in qd.static(range(3)):
                    H_i[c, c] += C_i * qd.sqrt(A_ii[0, c] ** 2 + A_ii[1, c] ** 2 + A_ii[2, c] ** 2) + sum_Cj_DAij[c]

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
                # degenerate guard (PLAN P2.4): isolated particle / fully clamped neighborhood => H_i = 0
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
                self.particles[i_p, i_b].dpos = dpos

    @qd.kernel
    def _kernel_apply_relaxed_update(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                # relaxed Jacobi: simultaneous half-step update (relaxation factor fixed at 1/2)
                self.particles[i_p, i_b].pos += 0.5 * self.particles[i_p, i_b].dpos

    @qd.kernel
    def _kernel_finalize(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                # PBD-style velocity update: v^{t+1} = (x^{t+1} - x^t) / h
                self.particles[i_p, i_b].vel = (
                    self.particles[i_p, i_b].pos - self.particles[i_p, i_b].ipos
                ) / self._substep_dt

    @qd.kernel
    def _kernel_impose_boundary(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                corrected_pos, corrected_vel = self.boundary.impose_pos_vel(
                    self.particles[i_p, i_b].pos, self.particles[i_p, i_b].vel
                )
                self.particles[i_p, i_b].pos = corrected_pos
                self.particles[i_p, i_b].vel = corrected_vel

    def substep_pre_coupling(self, f):
        if self.is_active:
            # Algorithm 1: predict (x <- y) -> hash rebuild -> relaxed-Jacobi Newton iterations -> finalize
            self._kernel_predict(f)
            self._kernel_build_hash(f)
            for _ in range(self._ipbf_iterations):
                self._kernel_compute_density_C(f)
                self._kernel_compute_newton_step(f)
                self._kernel_apply_relaxed_update(f)
            self._kernel_finalize(f)

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if self.is_active:
            self._kernel_impose_boundary(f)

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
            self._kernel_set_state(f, state.pos, state.vel, state.active)

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                self.particles[i_p, i_b].pos[j] = pos[i_b, i_p, j]
                self.particles[i_p, i_b].vel[j] = vel[i_b, i_p, j]
            self.particles_ng[i_p, i_b].active = active[i_b, i_p]

    def get_state(self, f):
        if self.is_active:
            state = IPBFSolverState(self.scene)
            self._kernel_get_state(f, state.pos, state.vel, state.active)
        else:
            state = None
        return state

    @qd.kernel
    def _kernel_get_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for j in qd.static(range(3)):
                pos[i_b, i_p, j] = self.particles[i_p, i_b].pos[j]
                vel[i_b, i_p, j] = self.particles[i_p, i_b].vel[j]
            active[i_b, i_p] = self.particles_ng[i_p, i_b].active

    def update_render_fields(self):
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    @qd.kernel
    def _kernel_update_render_fields(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                self.particles_render[i_p, i_b].pos = self.particles[i_p, i_b].pos
                self.particles_render[i_p, i_b].vel = self.particles[i_p, i_b].vel
            else:
                self.particles_render[i_p, i_b].pos = gu.qd_nowhere()
            self.particles_render[i_p, i_b].active = self.particles_ng[i_p, i_b].active

    @qd.kernel
    def _kernel_add_particles(
        self,
        f: qd.i32,
        active: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        mat_rho: qd.f32,
        pos: qd.types.ndarray(),
    ):
        for i_p_, i_b in qd.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            self.particles_ng[i_p, i_b].active = qd.cast(active, gs.qd_bool)
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].pos[i] = pos[i_p_, i]
            self.particles[i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)

        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            self.particles_info[i_p].rho = mat_rho
            self.particles_info[i_p].mass = self._particle_volume * mat_rho

    # ----------------------------------------------------------------------

    @qd.kernel
    def _kernel_set_particles_pos(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        poss: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].pos[i] = poss[i_b_, i_p_, i]
            self.particles[i_p, i_b].vel.fill(0.0)

    @qd.kernel
    def _kernel_get_particles_pos(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        poss: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                poss[i_b_, i_p_, i] = self.particles[i_p, i_b].pos[i]

    @qd.kernel
    def _kernel_set_particles_vel(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].vel[i] = vels[i_b_, i_p_, i]

    @qd.kernel
    def _kernel_get_particles_vel(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        vels: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            for i in qd.static(range(3)):
                vels[i_b_, i_p_, i] = self.particles[i_p, i_b].vel[i]

    @qd.kernel
    def _kernel_get_particles_rho(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        rhos: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            rhos[i_b_, i_p_] = self.particles[i_p, i_b].rho

    @qd.kernel
    def _kernel_set_particles_active(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles_ng[i_p, i_b].active = actives[i_b_, i_p_]

    @qd.kernel
    def _kernel_get_particles_active(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles_ng[i_p, i_b].active

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        else:
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
