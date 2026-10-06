import math

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import FLIPEntity
from genesis.utils.misc import qd_to_torch

from .base_solver import Solver


@qd.data_oriented
class FLIPSolver(Solver):
    """
    FLIP/PIC free-surface liquid solver on a MAC grid, ported from the reference implementation
    `sand-water-coupling-PIC-DEM-3d` (water-only path of `Simulation::Advance`):

    1. Advect particles through the current grid velocity (RK2 midpoint), enforce the box collider
       (projection + reflection), and rebuild the cell level set from particle positions
       (`min` of `|x_p - x_cell| - rad` over a 4^3 window, `rad = 1.01 * sqrt(3) / 2 * dx`).
    2. Density correction (IDP): deposit particles to cells (trilinear, normalized by 27 per cell),
       solve a Poisson-like system, and apply the resulting displacement to particle positions.
    3. P2G: scatter particle velocities to MAC faces (trilinear, atomic), normalize, extrapolate.
    4. Gravity on faces.
    5. Pressure projection: Poisson solve on fluid cells (levelSet <= 0) with collider face weights
       and the free-surface theta condition; extrapolate; enforce solid faces; record the velocity diff.
    6. G2P: FLIP (particle velocity + interpolated diff) blended with PIC by `blend_factor` (0.95).

    The fluid time step is CFL-adaptive, following the reference's `GetCourantTimeStep`: each simulator
    substep is split into water steps of `min(2 * dx / max|v_face|, remaining_dt)`.

    Differences from the reference (all recorded in the project log): the AMGCL solve is replaced by a
    GPU Jacobi-PCG with matrix-free matvecs; the sand-coupling solid-fraction terms reduce to 1 until
    the coupling phase; volume smoothing / redistancing of the level set are skipped (they only affect
    rendering); extrapolation uses Jacobi fill iterations.

    Batched simulation (n_envs > 0) is not supported.
    """

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        self._lower_bound = np.array(options.lower_bound, dtype=gs.np_float)
        self._upper_bound = np.array(options.upper_bound, dtype=gs.np_float)
        lengths = self._upper_bound - self._lower_bound
        self._dx = float(lengths.max() / options.grid_res)
        self._res = np.ceil(lengths / self._dx).astype(gs.np_int)  # (Nx, Ny, Nz) cell counts
        self._inv_dx = 1.0 / self._dx
        # qd vector copies for use inside kernels (numpy arrays cannot mix with qd vectors elementwise)
        self._lower_v = qd.Vector(options.lower_bound, dt=gs.qd_float)
        self._upper_v = qd.Vector(options.upper_bound, dt=gs.qd_float)

        self._seed_sub_factor = options.seed_sub_factor
        self._blend_factor = options.blend_factor
        self._density_correction = options.density_correction
        self._pcg_max_iter = options.pcg_max_iter
        self._pcg_tol = options.pcg_tol

        # reference: m_ParticleRadFactor = 1.01 * sqrt(3) / 2
        self._particle_rad = 1.01 * math.sqrt(3.0) / 2.0 * self._dx
        # C++ m_ViscosityCoeff = 1 (quadratic drag on sand grains)
        self._viscosity_coeff = options.viscosity_coeff

        self._cylinder_radius = options.cylinder_radius
        self._has_cylinder = options.cylinder_radius is not None
        self._rotate_omega = options.rotate_omega
        self._rotate_duration = options.rotate_duration
        self._center_x = 0.5 * float(self._lower_bound[0] + self._upper_bound[0])
        self._center_y = 0.5 * float(self._lower_bound[1] + self._upper_bound[1])

        # used for the entity seeding bounds check only
        self.boundary = CubeBoundary(lower=self._lower_bound, upper=self._upper_bound)

    def build(self):
        super().build()

        self._B = self._sim._B
        if self.is_active and self._B != 1:
            gs.raise_exception("FLIPSolver does not support batched simulation (n_envs must be 0 or 1).")
        self._n_particles = self.n_particles

        if self.is_active:
            self.init_fields()
            for entity in self._entities:
                entity._add_to_solver()
            self._build_open_masks()

        # coupled mode: drive the DEM advance before each water step (C++ Advance order)
        self._dem_coupling = self.is_active and self._sim.flip_options.dem_coupling and self._sim.dem_solver.is_active
        if self.is_active:
            self.target_fraction.fill(1.0)
        # C++ m_LastDeltaTime = 1000000 (first-step coupling forces vanish)
        self._last_dt = 1e6
        if self._dem_coupling:
            # C++ m_single_ratio = 1 / floor(V_dem / dx^3 * NumPartPerCell)
            dem = self._sim.dem_solver
            n_absorbable = int(dem._particle_volume / self._dx**3 * self._seed_sub_factor**3)
            if n_absorbable < 1:
                gs.raise_exception(
                    "DEM grains are too small relative to the FLIP cell for water absorption: "
                    f"floor(V_dem/dx^3 * {self._seed_sub_factor}^3) = 0 "
                    f"(V_dem={dem._particle_volume:.3e}, dx={self._dx:.4f}). Increase grid_res or particle_size."
                )
            self._single_ratio = 1.0 / n_absorbable
        # C++ DEMParticle::max_ratio (0.1 by default); fraction lower bound 1 - 0.74 * (1 + max_ratio)
        self._max_ratio = float(self._sim.dem_solver._entities[0].material.max_ratio) if self._dem_coupling else 0.1
        # The C++ lower bound goes non-positive for max_ratio >~ 0.35, and the projection divides
        # by the cell fraction (A = diag(1/f) * M), so a non-positive fraction blows up the solve;
        # clamp to a small positive value to keep max_ratio up to 0.5 usable.
        self._fraction_floor = max(1.0 - 0.74 * (1.0 + self._max_ratio), 1e-3)

        # FIXME: _gravity must be a raw qd.field() -- see comment in mpm_solver.py
        if self._gravity is not None:
            gravity = self._gravity.to_numpy()
            self._gravity = qd.field(dtype=gs.qd_vec3, shape=(self._B,))
            self._gravity.from_numpy(gravity)

    def init_fields(self):
        struct_particle_state = qd.types.struct(
            pos=gs.qd_vec3,
            vel=gs.qd_vec3,
            active=gs.qd_bool,
        )
        struct_particle_state_render = qd.types.struct(
            pos=gs.qd_vec3,
            active=gs.qd_bool,
        )
        self.particles = struct_particle_state.field(shape=(self._n_particles, self._B), layout=qd.Layout.SOA)
        self.particles_render = struct_particle_state_render.field(
            shape=(self._n_particles, self._B), layout=qd.Layout.SOA
        )

        Nx, Ny, Nz = (int(v) for v in self._res)
        shape_u = (Nx + 1, Ny, Nz)
        shape_v = (Nx, Ny + 1, Nz)
        shape_w = (Nx, Ny, Nz + 1)
        self.vel_u = qd.field(gs.qd_float, shape=shape_u)
        self.vel_v = qd.field(gs.qd_float, shape=shape_v)
        self.vel_w = qd.field(gs.qd_float, shape=shape_w)
        self.wsum_u = qd.field(gs.qd_float, shape=shape_u)
        self.wsum_v = qd.field(gs.qd_float, shape=shape_v)
        self.wsum_w = qd.field(gs.qd_float, shape=shape_w)
        self.vdiff_u = qd.field(gs.qd_float, shape=shape_u)
        self.vdiff_v = qd.field(gs.qd_float, shape=shape_v)
        self.vdiff_w = qd.field(gs.qd_float, shape=shape_w)
        self.dpos_u = qd.field(gs.qd_float, shape=shape_u)
        self.dpos_v = qd.field(gs.qd_float, shape=shape_v)
        self.dpos_w = qd.field(gs.qd_float, shape=shape_w)
        # scratch buffers (used by the extrapolation sweeps; vdiff keeps the saved velocity for FLIP)
        self.scratch_u = qd.field(gs.qd_float, shape=shape_u)
        self.scratch_v = qd.field(gs.qd_float, shape=shape_v)
        self.scratch_w = qd.field(gs.qd_float, shape=shape_w)
        self.mark_u = qd.field(gs.qd_int, shape=shape_u)
        self.mark_v = qd.field(gs.qd_int, shape=shape_v)
        self.mark_w = qd.field(gs.qd_int, shape=shape_w)
        # per-face open masks: 0 on solid faces (box boundary and, if enabled, outside the cylinder)
        self.open_u = qd.field(gs.qd_float, shape=shape_u)
        self.open_v = qd.field(gs.qd_float, shape=shape_v)
        self.open_w = qd.field(gs.qd_float, shape=shape_w)
        # sand-coupling face fields (C++ m_CouplingForce / m_Pressure.GetGradPressure1())
        self.coupling_force_u = qd.field(gs.qd_float, shape=shape_u)
        self.coupling_force_v = qd.field(gs.qd_float, shape=shape_v)
        self.coupling_force_w = qd.field(gs.qd_float, shape=shape_w)
        self.grad_p_u = qd.field(gs.qd_float, shape=shape_u)
        self.grad_p_v = qd.field(gs.qd_float, shape=shape_v)
        self.grad_p_w = qd.field(gs.qd_float, shape=shape_w)

        shape_c = (Nx, Ny, Nz)
        self.levelset = qd.field(gs.qd_float, shape=shape_c)
        self.density = qd.field(gs.qd_float, shape=shape_c)
        self.grid2mat = qd.field(gs.qd_int, shape=shape_c)
        # sand-coupling fields (C++ m_TargetFraction / m_NeededRatio / m_AbsorbRatio / m_AbsorbVelocity)
        self.target_fraction = qd.field(gs.qd_float, shape=shape_c)
        self.needed_ratio = qd.field(gs.qd_float, shape=shape_c)
        self.absorb_ratio = qd.field(gs.qd_float, shape=shape_c)
        self.absorb_vel = qd.Vector.field(3, gs.qd_float, shape=shape_c)
        self.absorb_count = qd.field(gs.qd_int, shape=shape_c)
        # PCG vectors
        self.p_x = qd.field(gs.qd_float, shape=shape_c)
        self.p_r = qd.field(gs.qd_float, shape=shape_c)
        self.p_d = qd.field(gs.qd_float, shape=shape_c)
        self.p_z = qd.field(gs.qd_float, shape=shape_c)
        self.p_ap = qd.field(gs.qd_float, shape=shape_c)
        self.p_diag = qd.field(gs.qd_float, shape=shape_c)
        self.p_rhs = qd.field(gs.qd_float, shape=shape_c)

    @property
    def is_active(self):
        return self.n_particles > 0

    @property
    def dx(self):
        return self._dx

    @property
    def seed_sub_factor(self):
        return self._seed_sub_factor

    @property
    def particle_radius(self):
        # rendering radius of a fluid particle
        return 0.5 * self._dx

    def add_entity(self, idx, material, morph, surface, name=None):
        entity = FLIPEntity(
            scene=self.scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._dx,
            idx=idx,
            particle_start=self.n_particles,
            name=name,
        )
        self._entities.append(entity)
        return entity

    @staticmethod
    def _corner_fraction(phi0, phi1, phi2):
        # C++ Collider::CalcFaceFraction's corner subdivision: solid fraction of the triangle
        # (phi <= 0 is solid). theta(p, q) = p / (p - q)
        p = np.sort(np.stack([phi0, phi1, phi2], axis=0), axis=0)
        a, b, c = p[0], p[1], p[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_ca = np.nan_to_num(c / (c - a), nan=1.0, posinf=1.0)
            t_cb = np.nan_to_num(c / (c - b), nan=1.0, posinf=1.0)
            t_ab = np.nan_to_num(a / (a - b), nan=0.0)
            t_ac = np.nan_to_num(a / (a - c), nan=0.0)
        frac = np.where(
            c <= 0,
            1.0,
            np.where(b <= 0, 1.0 - t_ca * t_cb, np.where(a <= 0, t_ab * t_ac, 0.0)),
        )
        return frac

    def _build_open_masks(self):
        # per-face open weight in [0, 1]: 0 on fully solid faces (box boundary / outside the cylinder),
        # fractional on faces cut by the cylinder surface (C++ Collider::CalcFaceFraction)
        R = self._cylinder_radius
        lo = self._lower_bound
        hi = self._upper_bound
        for axis, field in enumerate([self.open_u, self.open_v, self.open_w]):
            shape = field.shape
            offsets = np.array([0.5, 0.5, 0.5])
            offsets[axis] = 0.0
            grids = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
            pos = lo[None, None, None, :] + (np.stack(grids, axis=-1) + offsets) * self._dx
            open_mask = np.ones(shape, dtype=np.float64)
            # box boundary faces are fully solid
            on_box = np.isclose(pos[..., axis], lo[axis]) | np.isclose(pos[..., axis], hi[axis])
            open_mask[on_box] = 0.0
            if R is not None:
                # node SDF of the cylinder (positive = free): nodes sit at lower + idx * dx
                node_shape = tuple(n + 1 for n in shape)
                ngrids = np.meshgrid(*(np.arange(n) for n in node_shape), indexing="ij")
                npos = lo[None, None, None, :] + np.stack(ngrids, axis=-1) * self._dx
                phi = R - np.hypot(npos[..., 0] - self._center_x, npos[..., 1] - self._center_y)
                # the 4 corner nodes of a face: c00, c01, c11, c10 along the two non-axis directions
                axes2 = [a for a in range(3) if a != axis]

                def corner(o1, o2, *, face_axes=axes2, face_shape=shape, face_axis=axis, node_phi=phi):
                    sl = [slice(None)] * 3
                    sl[face_axes[0]] = slice(o1, o1 + face_shape[face_axes[0]])
                    sl[face_axes[1]] = slice(o2, o2 + face_shape[face_axes[1]])
                    sl[face_axis] = slice(0, face_shape[face_axis])
                    return node_phi[tuple(sl)]

                c00 = corner(0, 0)
                c01 = corner(0, 1)
                c11 = corner(1, 1)
                c10 = corner(1, 0)
                center = (c00 + c01 + c11 + c10) * 0.25
                frac = (
                    self._corner_fraction(center, c00, c01)
                    + self._corner_fraction(center, c01, c11)
                    + self._corner_fraction(center, c11, c10)
                    + self._corner_fraction(center, c10, c00)
                ) * 0.25
                frac = np.where(frac > 0.9, 1.0, frac)
                open_mask = np.minimum(open_mask, 1.0 - frac)
            field.from_numpy(np.ascontiguousarray(open_mask.astype(gs.np_float)))

    # ------------------------------------------------------------------------------------
    # --------------------------------- grid helpers -------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def _func_clamp_idx(self, idx, max_idx):
        return qd.Vector(
            [
                qd.min(qd.max(idx[0], 0), max_idx[0]),
                qd.min(qd.max(idx[1], 0), max_idx[1]),
                qd.min(qd.max(idx[2], 0), max_idx[2]),
            ],
            dt=gs.qd_int,
        )

    @qd.func
    def _func_in_bounds(self, idx, max_idx):
        return (
            idx[0] >= 0
            and idx[0] <= max_idx[0]
            and idx[1] >= 0
            and idx[1] <= max_idx[1]
            and idx[2] >= 0
            and idx[2] <= max_idx[2]
        )

    @qd.func
    def _func_trilerp_weights(self, pos, offsets):
        # trilinear base index and fractional weights on a staggered grid whose samples sit at
        # lower_bound + (index + offsets) * dx
        rel = (pos - self._lower_v) * self._inv_dx - offsets
        base = qd.floor(rel, gs.qd_int)
        frac = rel - base
        return base, frac

    @qd.func
    def _func_sample_vel(self, pos, diff: qd.template()):
        # trilinear interpolation of the MAC velocity (or velocity-diff) field at pos
        vel = qd.Vector.zero(gs.qd_float, 3)
        for axis in qd.static(range(3)):
            offsets = qd.Vector([0.5, 0.5, 0.5])
            offsets[axis] = 0.0
            base, frac = self._func_trilerp_weights(pos, offsets)
            val = gs.qd_float(0.0)
            for i, j, k in qd.static(qd.ndrange(2, 2, 2)):
                w = (
                    (frac[0] if i else 1.0 - frac[0])
                    * (frac[1] if j else 1.0 - frac[1])
                    * (frac[2] if k else 1.0 - frac[2])
                )
                idx = base + qd.Vector([i, j, k], dt=gs.qd_int)
                if axis == 0:
                    idx = self._func_clamp_idx(idx, qd.Vector(self.vel_u.shape, dt=gs.qd_int) - 1)
                    val += (self.vdiff_u[idx] if diff else self.vel_u[idx]) * w
                elif axis == 1:
                    idx = self._func_clamp_idx(idx, qd.Vector(self.vel_v.shape, dt=gs.qd_int) - 1)
                    val += (self.vdiff_v[idx] if diff else self.vel_v[idx]) * w
                else:
                    idx = self._func_clamp_idx(idx, qd.Vector(self.vel_w.shape, dt=gs.qd_int) - 1)
                    val += (self.vdiff_w[idx] if diff else self.vel_w[idx]) * w
            vel[axis] = val
        return vel

    @qd.func
    def _func_boundary_phi_normal(self, pos):
        # combined domain boundary: box intersected with the (optional) z-axis cylinder
        phi, n = self._func_box_phi_normal(pos)
        if qd.static(self._has_cylinder):
            dx_ = pos[0] - self._center_x
            dy_ = pos[1] - self._center_y
            rho = qd.sqrt(dx_ * dx_ + dy_ * dy_)
            phi_c = self._cylinder_radius - rho
            if phi_c < phi:
                phi = phi_c
                n = qd.Vector.zero(gs.qd_float, 3)
                if rho > gs.EPS:
                    n[0] = -dx_ / rho
                    n[1] = -dy_ / rho
        return phi, n

    @qd.func
    def _func_box_phi_normal(self, pos):
        # signed distance (positive inside) and inward normal of the domain box
        phi = gs.qd_float(1e30)
        n = qd.Vector.zero(gs.qd_float, 3)
        for i_a in qd.static(range(3)):
            if pos[i_a] - self._lower_bound[i_a] < phi:
                phi = pos[i_a] - self._lower_bound[i_a]
                n = qd.Vector.zero(gs.qd_float, 3)
                n[i_a] = 1.0
            if self._upper_bound[i_a] - pos[i_a] < phi:
                phi = self._upper_bound[i_a] - pos[i_a]
                n = qd.Vector.zero(gs.qd_float, 3)
                n[i_a] = -1.0
        return phi, n

    # ------------------------------------------------------------------------------------
    # ------------------------------------- kernels --------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def _kernel_add_particles(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        positions: qd.types.ndarray(element_dim=1),
    ):
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            for i_b in range(self._B):
                self.particles[i_p, i_b].pos = positions[i_p_]
                self.particles[i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
                self.particles[i_p, i_b].active = True

    @qd.kernel
    def _kernel_advect_particles(self, f: qd.i32, dt: qd.f32):
        # C++ AdvectFields: RK2 midpoint trace through the grid velocity, then collider enforcement
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                pos = self.particles[i_p, i_b].pos
                vel0 = self._func_sample_vel(pos, diff=False)
                vel1 = self._func_sample_vel(pos + vel0 * dt * 0.5, diff=False)
                pos = pos + vel1 * dt

                # C++ Collider::Enforce(particles): project out and reflect (restitution 1)
                phi, n = self._func_boundary_phi_normal(pos)
                if phi < 0.0:
                    pos = pos - n * phi
                    vel = self.particles[i_p, i_b].vel
                    vel_n = vel.dot(n)
                    if vel_n < 0.0:
                        self.particles[i_p, i_b].vel = vel - 2.0 * vel_n * n
                self.particles[i_p, i_b].pos = pos

    @qd.kernel
    def _kernel_reconstruct_levelset(self, f: qd.i32):
        # C++ ReconstructLevelSet: init to 2*dx - rad, then min with |x_p - x_cell| - rad over a 4^3 window
        self.levelset.fill(2.0 * self._dx - self._particle_rad)
        max_cell = qd.Vector(self.levelset.shape, dt=gs.qd_int) - 1
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                pos = self.particles[i_p, i_b].pos
                # C++ CalcLower<3> on the cell grid: floor(rel - 0.5) - 1
                base = qd.floor((pos - self._lower_v) * self._inv_dx - 0.5, gs.qd_int) - 1
                for i, j, k in qd.static(qd.ndrange(4, 4, 4)):
                    cell = base + qd.Vector([i, j, k], dt=gs.qd_int)
                    if self._func_in_bounds(cell, max_cell):
                        cell_pos = self._lower_v + (cell + 0.5) * self._dx
                        qd.atomic_min(self.levelset[cell], (pos - cell_pos).norm() - self._particle_rad)

    @qd.kernel
    def _kernel_set_unknowns(self, f: qd.i32):
        # C++ SetUnKnowns: fluid cells are those with levelset <= 0
        self.grid2mat.fill(-1)
        for cell in qd.grouped(self.grid2mat):
            if self.levelset[cell] <= 0.0:
                self.grid2mat[cell] = 1

    @qd.kernel
    def _kernel_enforce_particles_boundary(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                pos = self.particles[i_p, i_b].pos
                phi, n = self._func_boundary_phi_normal(pos)
                if phi < 0.0:
                    pos = pos - n * phi
                    vel = self.particles[i_p, i_b].vel
                    vel_n = vel.dot(n)
                    if vel_n < 0.0:
                        self.particles[i_p, i_b].vel = vel - 2.0 * vel_n * n
                    self.particles[i_p, i_b].pos = pos

    @qd.kernel
    def _kernel_cal_density(self, f: qd.i32):
        # C++ CalDensity: trilinear deposit normalized by the number of particles per cell (27)
        self.density.fill(0.0)
        n_per_cell = float(self._seed_sub_factor**3)
        max_cell = qd.Vector(self.density.shape, dt=gs.qd_int) - 1
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                base, frac = self._func_trilerp_weights(self.particles[i_p, i_b].pos, qd.Vector([0.5, 0.5, 0.5]))
                for i, j, k in qd.static(qd.ndrange(2, 2, 2)):
                    w = (
                        (frac[0] if i else 1.0 - frac[0])
                        * (frac[1] if j else 1.0 - frac[1])
                        * (frac[2] if k else 1.0 - frac[2])
                    )
                    cell = self._func_clamp_idx(base + qd.Vector([i, j, k], dt=gs.qd_int), max_cell)
                    qd.atomic_add(self.density[cell], w / n_per_cell)

    @qd.kernel
    def _kernel_cal_fraction(self, f: qd.i32):
        # C++ CalFraction: deposit DEM grain volumes onto the cell grid, then
        # target_fraction = clamp(1 - sum, 1 - 0.74 * (1 + max_ratio), 1)
        self.target_fraction.fill(0.0)
        dem = self._sim.dem_solver
        n_per_cell = float(dem._particle_volume * self._inv_dx**3)
        max_cell = qd.Vector(self.target_fraction.shape, dt=gs.qd_int) - 1
        for i_p, i_b in qd.ndrange(dem._n_particles, self._B):
            if dem.particles[i_p, i_b].active:
                pfrac = n_per_cell * (1.0 + dem.particles[i_p, i_b].ratio)
                base, frac = self._func_trilerp_weights(dem.particles[i_p, i_b].pos, qd.Vector([0.5, 0.5, 0.5]))
                for i, j, k in qd.static(qd.ndrange(2, 2, 2)):
                    w = (
                        (frac[0] if i else 1.0 - frac[0])
                        * (frac[1] if j else 1.0 - frac[1])
                        * (frac[2] if k else 1.0 - frac[2])
                    )
                    cell = self._func_clamp_idx(base + qd.Vector([i, j, k], dt=gs.qd_int), max_cell)
                    qd.atomic_add(self.target_fraction[cell], w * pfrac)
        for cell in qd.grouped(self.target_fraction):
            # C++: clamp(1 - acc, 1 - 0.74 * (1 + max_ratio), 1)
            self.target_fraction[cell] = qd.min(qd.max(1.0 - self.target_fraction[cell], self._fraction_floor), 1.0)

    @qd.kernel
    def _kernel_apply_rotation(self, f: qd.i32, dt: qd.f32):
        # C++ Advance rotation drive (rotate scene): fluid faces in sand-free cells
        # (target fraction >= 0.8 on both sides) receive the rigid-body increment omega x r * dt
        omega = gs.qd_float(self._rotate_omega)
        max_cell = qd.Vector(self.levelset.shape, dt=gs.qd_int) - 1
        for idx in qd.grouped(self.vel_u):
            if self.open_u[idx] > 0.0:
                i, j, k = idx
                c0 = self._func_clamp_idx(qd.Vector([i - 1, j, k], dt=gs.qd_int), max_cell)
                c1 = self._func_clamp_idx(qd.Vector([i, j, k], dt=gs.qd_int), max_cell)
                if self.target_fraction[c0] >= 0.8 and self.target_fraction[c1] >= 0.8:
                    pos_y = self._lower_bound[1] + (j + 0.5) * self._dx
                    self.vel_u[idx] = self.vel_u[idx] - omega * (pos_y - self._center_y) * dt
        for idx in qd.grouped(self.vel_v):
            if self.open_v[idx] > 0.0:
                i, j, k = idx
                c0 = self._func_clamp_idx(qd.Vector([i, j - 1, k], dt=gs.qd_int), max_cell)
                c1 = self._func_clamp_idx(qd.Vector([i, j, k], dt=gs.qd_int), max_cell)
                if self.target_fraction[c0] >= 0.8 and self.target_fraction[c1] >= 0.8:
                    pos_x = self._lower_bound[0] + (i + 0.5) * self._dx
                    self.vel_v[idx] = self.vel_v[idx] + omega * (pos_x - self._center_x) * dt

    # ---------------------------------- P2G ----------------------------------

    @qd.kernel
    def _kernel_p2g_scatter(self, f: qd.i32):
        self.vel_u.fill(0.0)
        self.vel_v.fill(0.0)
        self.vel_w.fill(0.0)
        self.wsum_u.fill(0.0)
        self.wsum_v.fill(0.0)
        self.wsum_w.fill(0.0)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                pos = self.particles[i_p, i_b].pos
                vel = self.particles[i_p, i_b].vel
                for axis in qd.static(range(3)):
                    offsets = qd.Vector([0.5, 0.5, 0.5])
                    offsets[axis] = 0.0
                    base, frac = self._func_trilerp_weights(pos, offsets)
                    for i, j, k in qd.static(qd.ndrange(2, 2, 2)):
                        w = (
                            (frac[0] if i else 1.0 - frac[0])
                            * (frac[1] if j else 1.0 - frac[1])
                            * (frac[2] if k else 1.0 - frac[2])
                        )
                        idx = base + qd.Vector([i, j, k], dt=gs.qd_int)
                        if axis == 0:
                            idx = self._func_clamp_idx(idx, qd.Vector(self.vel_u.shape, dt=gs.qd_int) - 1)
                            qd.atomic_add(self.vel_u[idx], vel[0] * w)
                            qd.atomic_add(self.wsum_u[idx], w)
                        elif axis == 1:
                            idx = self._func_clamp_idx(idx, qd.Vector(self.vel_v.shape, dt=gs.qd_int) - 1)
                            qd.atomic_add(self.vel_v[idx], vel[1] * w)
                            qd.atomic_add(self.wsum_v[idx], w)
                        else:
                            idx = self._func_clamp_idx(idx, qd.Vector(self.vel_w.shape, dt=gs.qd_int) - 1)
                            qd.atomic_add(self.vel_w[idx], vel[2] * w)
                            qd.atomic_add(self.wsum_w[idx], w)

    @qd.kernel
    def _kernel_p2g_normalize(self, f: qd.i32):
        for idx in qd.grouped(self.vel_u):
            if self.wsum_u[idx] > 0.0:
                self.vel_u[idx] = self.vel_u[idx] / self.wsum_u[idx]
        for idx in qd.grouped(self.vel_v):
            if self.wsum_v[idx] > 0.0:
                self.vel_v[idx] = self.vel_v[idx] / self.wsum_v[idx]
        for idx in qd.grouped(self.vel_w):
            if self.wsum_w[idx] > 0.0:
                self.vel_w[idx] = self.vel_w[idx] / self.wsum_w[idx]

    @qd.kernel
    def _kernel_extrapolate_mark_weights(self, f: qd.i32):
        # marker for the post-P2G extrapolation: faces that received any particle weight
        for idx in qd.grouped(self.mark_u):
            self.mark_u[idx] = 1 if self.wsum_u[idx] > 0.0 else 0
        for idx in qd.grouped(self.mark_v):
            self.mark_v[idx] = 1 if self.wsum_v[idx] > 0.0 else 0
        for idx in qd.grouped(self.mark_w):
            self.mark_w[idx] = 1 if self.wsum_w[idx] > 0.0 else 0

    @qd.kernel
    def _kernel_extrapolate_mark_fluid(self, f: qd.i32):
        # marker for the post-projection extrapolation (C++ ProjectVelocity): open faces (not on the domain
        # boundary) adjacent to at least one fluid cell
        max_cell = qd.Vector(self.levelset.shape, dt=gs.qd_int) - 1
        for idx in qd.grouped(self.mark_u):
            i, j, k = idx
            if self.open_u[idx] == 0.0:
                self.mark_u[idx] = 0
            else:
                c0 = self._func_clamp_idx(qd.Vector([i - 1, j, k], dt=gs.qd_int), max_cell)
                c1 = self._func_clamp_idx(qd.Vector([i, j, k], dt=gs.qd_int), max_cell)
                self.mark_u[idx] = 1 if (self.levelset[c0] <= 0.0 or self.levelset[c1] <= 0.0) else 0
        for idx in qd.grouped(self.mark_v):
            i, j, k = idx
            if self.open_v[idx] == 0.0:
                self.mark_v[idx] = 0
            else:
                c0 = self._func_clamp_idx(qd.Vector([i, j - 1, k], dt=gs.qd_int), max_cell)
                c1 = self._func_clamp_idx(qd.Vector([i, j, k], dt=gs.qd_int), max_cell)
                self.mark_v[idx] = 1 if (self.levelset[c0] <= 0.0 or self.levelset[c1] <= 0.0) else 0
        for idx in qd.grouped(self.mark_w):
            i, j, k = idx
            if self.open_w[idx] == 0.0:
                self.mark_w[idx] = 0
            else:
                c0 = self._func_clamp_idx(qd.Vector([i, j, k - 1], dt=gs.qd_int), max_cell)
                c1 = self._func_clamp_idx(qd.Vector([i, j, k], dt=gs.qd_int), max_cell)
                self.mark_w[idx] = 1 if (self.levelset[c0] <= 0.0 or self.levelset[c1] <= 0.0) else 0

    @qd.kernel
    def _kernel_extrapolate_sweep(self, f: qd.i32):
        # one Jacobi fill sweep per axis: unmarked OPEN faces take the average of their marked neighbors
        # (closed faces are excluded as targets; they stay zero, keeping vdiff clean near walls)
        for idx in qd.grouped(self.vel_u):
            self.scratch_u[idx] = 0.0
            if self.mark_u[idx] == 0 and self.open_u[idx] > 0.0:
                total = gs.qd_float(0.0)
                count = 0
                for d in qd.static(range(3)):
                    for s in qd.static((-1, 1)):
                        nb = idx + qd.Vector.unit(3, d, gs.qd_int) * s
                        if self._func_in_bounds(nb, qd.Vector(self.vel_u.shape, dt=gs.qd_int) - 1):
                            if self.mark_u[nb] == 1:
                                total += self.vel_u[nb]
                                count += 1
                if count > 0:
                    self.scratch_u[idx] = total / count  # scratch buffer
        for idx in qd.grouped(self.vel_u):
            if self.mark_u[idx] == 0 and self.scratch_u[idx] != 0.0:
                self.vel_u[idx] = self.scratch_u[idx]
                self.mark_u[idx] = 1
        for idx in qd.grouped(self.vel_v):
            self.scratch_v[idx] = 0.0
            if self.mark_v[idx] == 0 and self.open_v[idx] > 0.0:
                total = gs.qd_float(0.0)
                count = 0
                for d in qd.static(range(3)):
                    for s in qd.static((-1, 1)):
                        nb = idx + qd.Vector.unit(3, d, gs.qd_int) * s
                        if self._func_in_bounds(nb, qd.Vector(self.vel_v.shape, dt=gs.qd_int) - 1):
                            if self.mark_v[nb] == 1:
                                total += self.vel_v[nb]
                                count += 1
                if count > 0:
                    self.scratch_v[idx] = total / count
        for idx in qd.grouped(self.vel_v):
            if self.mark_v[idx] == 0 and self.scratch_v[idx] != 0.0:
                self.vel_v[idx] = self.scratch_v[idx]
                self.mark_v[idx] = 1
        for idx in qd.grouped(self.vel_w):
            self.scratch_w[idx] = 0.0
            if self.mark_w[idx] == 0 and self.open_w[idx] > 0.0:
                total = gs.qd_float(0.0)
                count = 0
                for d in qd.static(range(3)):
                    for s in qd.static((-1, 1)):
                        nb = idx + qd.Vector.unit(3, d, gs.qd_int) * s
                        if self._func_in_bounds(nb, qd.Vector(self.vel_w.shape, dt=gs.qd_int) - 1):
                            if self.mark_w[nb] == 1:
                                total += self.vel_w[nb]
                                count += 1
                if count > 0:
                    self.scratch_w[idx] = total / count
        for idx in qd.grouped(self.vel_w):
            if self.mark_w[idx] == 0 and self.scratch_w[idx] != 0.0:
                self.vel_w[idx] = self.scratch_w[idx]
                self.mark_w[idx] = 1

    @qd.kernel
    def _kernel_apply_gravity(self, f: qd.i32, dt: qd.f32):
        # C++ ApplyBodyForces: gravity plus the coupling reaction force (v -= F * dt / dx^3);
        # the coupling fields are zero in uncoupled mode, keeping this bit-identical
        g = self._gravity[0]
        inv_cell_vol = self._inv_dx**3
        for idx in qd.grouped(self.vel_u):
            self.vel_u[idx] = self.vel_u[idx] + g[0] * dt - self.coupling_force_u[idx] * dt * inv_cell_vol
        for idx in qd.grouped(self.vel_v):
            self.vel_v[idx] = self.vel_v[idx] + g[1] * dt - self.coupling_force_v[idx] * dt * inv_cell_vol
        for idx in qd.grouped(self.vel_w):
            self.vel_w[idx] = self.vel_w[idx] + g[2] * dt - self.coupling_force_w[idx] * dt * inv_cell_vol
        self.coupling_force_u.fill(0.0)
        self.coupling_force_v.fill(0.0)
        self.coupling_force_w.fill(0.0)

    @qd.kernel
    def _kernel_enforce_velocity_boundary(self, f: qd.i32):
        # C++ Collider::Enforce(velocity): fully solid faces (box boundary or outside the cylinder) take
        # the wall velocity 0
        for idx in qd.grouped(self.vel_u):
            if self.open_u[idx] == 0.0:
                self.vel_u[idx] = 0.0
        for idx in qd.grouped(self.vel_v):
            if self.open_v[idx] == 0.0:
                self.vel_v[idx] = 0.0
        for idx in qd.grouped(self.vel_w):
            if self.open_w[idx] == 0.0:
                self.vel_w[idx] = 0.0

    # ------------------------------ pressure solve ------------------------------

    @qd.kernel
    def _kernel_projection_rhs_diag(self, f: qd.i32):
        # C++ BuildProjectionMatrix (fraction == 1): rhs = -div; Jacobi diagonal with theta on air faces.
        # Faces touching the domain boundary have collider weight 0 and are simply skipped: for a cell at
        # the grid edge the corresponding neighbor lies outside the cell grid.
        max_cell = qd.Vector(self.levelset.shape, dt=gs.qd_int) - 1
        for cell in qd.grouped(self.levelset):
            self.p_rhs[cell] = 0.0
            self.p_diag[cell] = 0.0
            if self.grid2mat[cell] >= 0:
                f_r = self.target_fraction[cell]
                # divergence with symmetric fraction coefficients; this sum already equals f_r * rhs_cpp
                # (A = diag(1/f) * M with M symmetric; we solve M x = f .* rhs_cpp, same solution)
                rhs = gs.qd_float(0.0)
                diag = gs.qd_float(0.0)
                for d in qd.static(range(3)):
                    for s in qd.static((-1, 1)):
                        nb = cell + qd.Vector.unit(3, d, gs.qd_int) * s
                        if self._func_in_bounds(nb, max_cell):
                            face = nb if s > 0 else cell
                            w_open = gs.qd_float(0.0)
                            if d == 0:
                                w_open = self.open_u[face]
                            elif d == 1:
                                w_open = self.open_v[face]
                            else:
                                w_open = self.open_w[face]
                            if w_open > 0.0:
                                coef = w_open * 0.5 * (f_r + self.target_fraction[nb])
                                face_vel = gs.qd_float(0.0)
                                if d == 0:
                                    face_vel = self.vel_u[face]
                                elif d == 1:
                                    face_vel = self.vel_v[face]
                                else:
                                    face_vel = self.vel_w[face]
                                rhs -= s * face_vel * coef
                                if self.grid2mat[nb] >= 0:
                                    diag += coef
                                else:
                                    theta = self.levelset[cell] / (self.levelset[cell] - self.levelset[nb])
                                    diag += coef / max(theta, 0.001)
                self.p_rhs[cell] = rhs
                # C++ falls back to an identity row when the diagonal vanishes (isolated fluid cell);
                # a near-zero diagonal would blow up the Jacobi preconditioner
                self.p_diag[cell] = diag if diag > 1e-8 else 1.0

    @qd.kernel
    def _kernel_projection_matvec(self, f: qd.i32):
        # matrix-free (M x): symmetric fraction-weighted Laplacian (see _kernel_projection_rhs_diag)
        max_cell = qd.Vector(self.levelset.shape, dt=gs.qd_int) - 1
        for cell in qd.grouped(self.levelset):
            self.p_ap[cell] = 0.0
            if self.grid2mat[cell] >= 0:
                acc = gs.qd_float(0.0)
                for d in qd.static(range(3)):
                    for s in qd.static((-1, 1)):
                        nb = cell + qd.Vector.unit(3, d, gs.qd_int) * s
                        if self._func_in_bounds(nb, max_cell):
                            face = nb if s > 0 else cell
                            w_open = gs.qd_float(0.0)
                            if d == 0:
                                w_open = self.open_u[face]
                            elif d == 1:
                                w_open = self.open_v[face]
                            else:
                                w_open = self.open_w[face]
                            if w_open > 0.0:
                                if self.grid2mat[nb] >= 0:
                                    acc -= (
                                        self.p_d[nb]
                                        * w_open
                                        * 0.5
                                        * (self.target_fraction[cell] + self.target_fraction[nb])
                                    )
                self.p_ap[cell] = self.p_diag[cell] * self.p_d[cell] + acc

    @qd.kernel
    def _kernel_apply_projection(self, f: qd.i32):
        # C++ ApplyProjection: subtract the (scaled) pressure gradient on open faces; free-surface faces
        # use the theta condition with p = 0 in air
        for idx in qd.grouped(self.vel_u):
            i, j, k = idx
            if self.open_u[idx] == 0.0:
                continue
            c0 = qd.Vector([i - 1, j, k], dt=gs.qd_int)
            c1 = qd.Vector([i, j, k], dt=gs.qd_int)
            id0 = self.grid2mat[c0]
            id1 = self.grid2mat[c1]
            if id0 >= 0 and id1 >= 0:
                # C++ ApplyProjection also stores the gradient (m_GradPressure1); both scale with the open weight
                w_f = self.open_u[idx]
                self.vel_u[idx] = self.vel_u[idx] - (self.p_x[c1] - self.p_x[c0]) * w_f
                self.grad_p_u[idx] = (self.p_x[c1] - self.p_x[c0]) * w_f
            elif id0 >= 0 or id1 >= 0:
                phi0 = self.levelset[c0]
                phi1 = self.levelset[c1]
                theta = phi0 / (phi0 - phi1)
                p_inner = self.p_x[c0] if id0 >= 0 else self.p_x[c1]
                intf = 1.0 / max(theta if id0 >= 0 else 1.0 - theta, 0.001)
                self.vel_u[idx] = self.vel_u[idx] + (1.0 if id0 >= 0 else -1.0) * p_inner * intf * self.open_u[idx]
        for idx in qd.grouped(self.vel_v):
            i, j, k = idx
            if self.open_v[idx] == 0.0:
                continue
            c0 = qd.Vector([i, j - 1, k], dt=gs.qd_int)
            c1 = qd.Vector([i, j, k], dt=gs.qd_int)
            id0 = self.grid2mat[c0]
            id1 = self.grid2mat[c1]
            if id0 >= 0 and id1 >= 0:
                # C++ ApplyProjection also stores the gradient (m_GradPressure1); both scale with the open weight
                w_f = self.open_v[idx]
                self.vel_v[idx] = self.vel_v[idx] - (self.p_x[c1] - self.p_x[c0]) * w_f
                self.grad_p_v[idx] = (self.p_x[c1] - self.p_x[c0]) * w_f
            elif id0 >= 0 or id1 >= 0:
                phi0 = self.levelset[c0]
                phi1 = self.levelset[c1]
                theta = phi0 / (phi0 - phi1)
                p_inner = self.p_x[c0] if id0 >= 0 else self.p_x[c1]
                intf = 1.0 / max(theta if id0 >= 0 else 1.0 - theta, 0.001)
                self.vel_v[idx] = self.vel_v[idx] + (1.0 if id0 >= 0 else -1.0) * p_inner * intf * self.open_v[idx]
        for idx in qd.grouped(self.vel_w):
            i, j, k = idx
            if self.open_w[idx] == 0.0:
                continue
            c0 = qd.Vector([i, j, k - 1], dt=gs.qd_int)
            c1 = qd.Vector([i, j, k], dt=gs.qd_int)
            id0 = self.grid2mat[c0]
            id1 = self.grid2mat[c1]
            if id0 >= 0 and id1 >= 0:
                # C++ ApplyProjection also stores the gradient (m_GradPressure1); both scale with the open weight
                w_f = self.open_w[idx]
                self.vel_w[idx] = self.vel_w[idx] - (self.p_x[c1] - self.p_x[c0]) * w_f
                self.grad_p_w[idx] = (self.p_x[c1] - self.p_x[c0]) * w_f
            elif id0 >= 0 or id1 >= 0:
                phi0 = self.levelset[c0]
                phi1 = self.levelset[c1]
                theta = phi0 / (phi0 - phi1)
                p_inner = self.p_x[c0] if id0 >= 0 else self.p_x[c1]
                intf = 1.0 / max(theta if id0 >= 0 else 1.0 - theta, 0.001)
                self.vel_w[idx] = self.vel_w[idx] + (1.0 if id0 >= 0 else -1.0) * p_inner * intf * self.open_w[idx]

    @qd.kernel
    def _kernel_save_velocity(self, f: qd.i32):
        for idx in qd.grouped(self.vel_u):
            self.vdiff_u[idx] = self.vel_u[idx]
        for idx in qd.grouped(self.vel_v):
            self.vdiff_v[idx] = self.vel_v[idx]
        for idx in qd.grouped(self.vel_w):
            self.vdiff_w[idx] = self.vel_w[idx]

    @qd.kernel
    def _kernel_update_veldiff(self, f: qd.i32):
        for idx in qd.grouped(self.vel_u):
            self.vdiff_u[idx] = (self.vel_u[idx] - self.vdiff_u[idx]) if self.open_u[idx] > 0.0 else 0.0
        for idx in qd.grouped(self.vel_v):
            self.vdiff_v[idx] = (self.vel_v[idx] - self.vdiff_v[idx]) if self.open_v[idx] > 0.0 else 0.0
        for idx in qd.grouped(self.vel_w):
            self.vdiff_w[idx] = (self.vel_w[idx] - self.vdiff_w[idx]) if self.open_w[idx] > 0.0 else 0.0

    @qd.kernel
    def _kernel_max_face_vel(self, max_vel: qd.types.ndarray()):
        # C++ GetMaxAbsComponent: max absolute component over all MAC faces
        for idx in qd.grouped(self.vel_u):
            qd.atomic_max(max_vel[0], abs(self.vel_u[idx]))
        for idx in qd.grouped(self.vel_v):
            qd.atomic_max(max_vel[0], abs(self.vel_v[idx]))
        for idx in qd.grouped(self.vel_w):
            qd.atomic_max(max_vel[0], abs(self.vel_w[idx]))

    @qd.kernel
    def _kernel_g2p(self, f: qd.i32):
        # C++ TransferFromGridToParticles (FLIP): v += interp(veldiff), blended with PIC by blend_factor
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                pos = self.particles[i_p, i_b].pos
                v_flip = self.particles[i_p, i_b].vel + self._func_sample_vel(pos, diff=True)
                v_pic = self._func_sample_vel(pos, diff=False)
                self.particles[i_p, i_b].vel = self._blend_factor * v_flip + (1.0 - self._blend_factor) * v_pic

    # ------------------------------ density correction ------------------------------

    @qd.kernel
    def _kernel_correction_rhs_diag(self, f: qd.i32):
        # C++ BuildCorrectionMatrix (fraction == 1): rhs = 1 - clamp(density, .5, 1.5), 0 at free-surface cells
        max_cell = qd.Vector(self.levelset.shape, dt=gs.qd_int) - 1
        for cell in qd.grouped(self.levelset):
            self.p_rhs[cell] = 0.0
            self.p_diag[cell] = 0.0
            if self.grid2mat[cell] >= 0:
                f_r = self.target_fraction[cell]
                diag = gs.qd_float(0.0)
                is_boundary = False
                for d in qd.static(range(3)):
                    for s in qd.static((-1, 1)):
                        nb = cell + qd.Vector.unit(3, d, gs.qd_int) * s
                        if self._func_in_bounds(nb, max_cell):
                            face = nb if s > 0 else cell
                            w_open = gs.qd_float(0.0)
                            if d == 0:
                                w_open = self.open_u[face]
                            elif d == 1:
                                w_open = self.open_v[face]
                            else:
                                w_open = self.open_w[face]
                            if w_open > 0.0:
                                diag += w_open * 0.5 * (f_r + self.target_fraction[nb])
                                if self.grid2mat[nb] < 0:
                                    is_boundary = True
                self.p_diag[cell] = diag if diag > 1e-8 else 1.0
                if not is_boundary:
                    # C++ rhs 1 - clamp(density / f_r, .5, 1.5), times f_r (symmetrized system)
                    self.p_rhs[cell] = f_r - qd.min(qd.max(self.density[cell], 0.5 * f_r), 1.5 * f_r)

    @qd.kernel
    def _kernel_correction_matvec(self, f: qd.i32):
        max_cell = qd.Vector(self.levelset.shape, dt=gs.qd_int) - 1
        for cell in qd.grouped(self.levelset):
            self.p_ap[cell] = 0.0
            if self.grid2mat[cell] >= 0:
                acc = gs.qd_float(0.0)
                for d in qd.static(range(3)):
                    for s in qd.static((-1, 1)):
                        nb = cell + qd.Vector.unit(3, d, gs.qd_int) * s
                        if self._func_in_bounds(nb, max_cell):
                            face = nb if s > 0 else cell
                            w_open = gs.qd_float(0.0)
                            if d == 0:
                                w_open = self.open_u[face]
                            elif d == 1:
                                w_open = self.open_v[face]
                            else:
                                w_open = self.open_w[face]
                            if w_open > 0.0:
                                if self.grid2mat[nb] >= 0:
                                    acc -= (
                                        self.p_d[nb]
                                        * w_open
                                        * 0.5
                                        * (self.target_fraction[cell] + self.target_fraction[nb])
                                    )
                self.p_ap[cell] = self.p_diag[cell] * self.p_d[cell] + acc

    @qd.kernel
    def _kernel_apply_correction_faces(self, f: qd.i32):
        # C++ ApplyCorrection: per-face displacement (p1 - p0) * dx, zero on closed and air-air faces
        for idx in qd.grouped(self.dpos_u):
            i, j, k = idx
            self.dpos_u[idx] = 0.0
            if 0 < i < self.vel_u.shape[0] - 1:
                c0 = qd.Vector([i - 1, j, k], dt=gs.qd_int)
                c1 = qd.Vector([i, j, k], dt=gs.qd_int)
                id0 = self.grid2mat[c0]
                id1 = self.grid2mat[c1]
                if id0 >= 0 or id1 >= 0:
                    p0 = self.p_x[c0] if id0 >= 0 else 0.0
                    p1 = self.p_x[c1] if id1 >= 0 else 0.0
                    self.dpos_u[idx] = (p1 - p0) * self._dx
        for idx in qd.grouped(self.dpos_v):
            i, j, k = idx
            self.dpos_v[idx] = 0.0
            if 0 < j < self.vel_v.shape[1] - 1:
                c0 = qd.Vector([i, j - 1, k], dt=gs.qd_int)
                c1 = qd.Vector([i, j, k], dt=gs.qd_int)
                id0 = self.grid2mat[c0]
                id1 = self.grid2mat[c1]
                if id0 >= 0 or id1 >= 0:
                    p0 = self.p_x[c0] if id0 >= 0 else 0.0
                    p1 = self.p_x[c1] if id1 >= 0 else 0.0
                    self.dpos_v[idx] = (p1 - p0) * self._dx
        for idx in qd.grouped(self.dpos_w):
            i, j, k = idx
            self.dpos_w[idx] = 0.0
            if 0 < k < self.vel_w.shape[2] - 1:
                c0 = qd.Vector([i, j, k - 1], dt=gs.qd_int)
                c1 = qd.Vector([i, j, k], dt=gs.qd_int)
                id0 = self.grid2mat[c0]
                id1 = self.grid2mat[c1]
                if id0 >= 0 or id1 >= 0:
                    p0 = self.p_x[c0] if id0 >= 0 else 0.0
                    p1 = self.p_x[c1] if id1 >= 0 else 0.0
                    self.dpos_w[idx] = (p1 - p0) * self._dx

    @qd.kernel
    def _kernel_apply_correction_particles(self, f: qd.i32):
        # C++ ApplyCorrection: particle positions += trilerp(face displacement field)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                pos = self.particles[i_p, i_b].pos
                dpos = qd.Vector.zero(gs.qd_float, 3)
                for axis in qd.static(range(3)):
                    offsets = qd.Vector([0.5, 0.5, 0.5])
                    offsets[axis] = 0.0
                    base, frac = self._func_trilerp_weights(pos, offsets)
                    val = gs.qd_float(0.0)
                    for i, j, k in qd.static(qd.ndrange(2, 2, 2)):
                        w = (
                            (frac[0] if i else 1.0 - frac[0])
                            * (frac[1] if j else 1.0 - frac[1])
                            * (frac[2] if k else 1.0 - frac[2])
                        )
                        idx = base + qd.Vector([i, j, k], dt=gs.qd_int)
                        if axis == 0:
                            idx = self._func_clamp_idx(idx, qd.Vector(self.dpos_u.shape, dt=gs.qd_int) - 1)
                            val += self.dpos_u[idx] * w
                        elif axis == 1:
                            idx = self._func_clamp_idx(idx, qd.Vector(self.dpos_v.shape, dt=gs.qd_int) - 1)
                            val += self.dpos_v[idx] * w
                        else:
                            idx = self._func_clamp_idx(idx, qd.Vector(self.dpos_w.shape, dt=gs.qd_int) - 1)
                            val += self.dpos_w[idx] * w
                    dpos[axis] = val
                self.particles[i_p, i_b].pos = pos + dpos

    # ------------------------------ water absorption ------------------------------

    @qd.kernel
    def _kernel_wet_reset(self, f: qd.i32):
        # C++ WetDEMParticle first half: per-cell water demand of the sand, max_ratio = 0.1
        self.needed_ratio.fill(0.0)
        self.absorb_ratio.fill(0.0)
        self.absorb_count.fill(0)
        self.absorb_vel.fill(qd.Vector.zero(gs.qd_float, 3))
        dem = self._sim.dem_solver
        max_cell = qd.Vector(self.needed_ratio.shape, dt=gs.qd_int) - 1
        for i_p, i_b in qd.ndrange(dem._n_particles, self._B):
            if dem.particles[i_p, i_b].active:
                need = self._max_ratio - dem.particles[i_p, i_b].ratio
                base, frac = self._func_trilerp_weights(dem.particles[i_p, i_b].pos, qd.Vector([0.5, 0.5, 0.5]))
                for i, j, k in qd.static(qd.ndrange(2, 2, 2)):
                    w = (
                        (frac[0] if i else 1.0 - frac[0])
                        * (frac[1] if j else 1.0 - frac[1])
                        * (frac[2] if k else 1.0 - frac[2])
                    )
                    cell = self._func_clamp_idx(base + qd.Vector([i, j, k], dt=gs.qd_int), max_cell)
                    qd.atomic_add(self.needed_ratio[cell], w * need)

    @qd.kernel
    def _kernel_absorb_mark(self, f: qd.i32):
        # C++ CacheFluidIndex + WetDEMParticle middle: per cell, the first
        # floor(needed_ratio / single_ratio) visiting fluid particles are absorbed (the reference picks
        # them via std::shuffle; the GPU atomic visit order is equally unbiased)
        max_cell = qd.Vector(self.needed_ratio.shape, dt=gs.qd_int) - 1
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                cell = qd.floor((self.particles[i_p, i_b].pos - self._lower_v) * self._inv_dx, gs.qd_int)
                cell = self._func_clamp_idx(cell, max_cell)
                ticket = qd.atomic_add(self.absorb_count[cell], 1)
                if ticket < qd.floor(self.needed_ratio[cell] / self._single_ratio, gs.qd_int):
                    self.particles[i_p, i_b].active = False
                    qd.atomic_add(self.absorb_ratio[cell], self._single_ratio)
                    qd.atomic_add(self.absorb_vel[cell], self.particles[i_p, i_b].vel)

    @qd.kernel
    def _kernel_absorb_finalize(self, f: qd.i32):
        # average the absorbed particles' velocity per cell (C++ accumulates v / num)
        for cell in qd.grouped(self.absorb_ratio):
            n = qd.round(self.absorb_ratio[cell] / self._single_ratio, gs.qd_int)
            if n > 0:
                self.absorb_vel[cell] = self.absorb_vel[cell] / n

    @qd.kernel
    def _kernel_wet_dem_particles(self, f: qd.i32):
        # C++ WetDEMParticle last half: each grain gathers its share of the absorbed water
        dem = self._sim.dem_solver
        max_cell = qd.Vector(self.needed_ratio.shape, dt=gs.qd_int) - 1
        for i_p, i_b in qd.ndrange(dem._n_particles, self._B):
            if dem.particles[i_p, i_b].active:
                ratio = dem.particles[i_p, i_b].ratio
                base, frac = self._func_trilerp_weights(dem.particles[i_p, i_b].pos, qd.Vector([0.5, 0.5, 0.5]))
                add_ratio = gs.qd_float(0.0)
                add_vel = qd.Vector.zero(gs.qd_float, 3)
                for i, j, k in qd.static(qd.ndrange(2, 2, 2)):
                    w = (
                        (frac[0] if i else 1.0 - frac[0])
                        * (frac[1] if j else 1.0 - frac[1])
                        * (frac[2] if k else 1.0 - frac[2])
                    )
                    cell = self._func_clamp_idx(base + qd.Vector([i, j, k], dt=gs.qd_int), max_cell)
                    ar = (self._max_ratio - ratio) * self.absorb_ratio[cell] / qd.max(self.needed_ratio[cell], 1e-12)
                    add_ratio += w * ar
                    add_vel += self.absorb_vel[cell] * (w * ar)
                if add_ratio > 0.0:
                    add_vel = add_vel / add_ratio - dem.particles[i_p, i_b].vel
                    dem.particles[i_p, i_b].add_ratio = add_ratio
                    dem.particles[i_p, i_b].add_velocity = add_vel
                    dem.particles[i_p, i_b].ratio = ratio + add_ratio
                else:
                    dem.particles[i_p, i_b].add_ratio = 0.0
                    dem.particles[i_p, i_b].add_velocity = qd.Vector.zero(gs.qd_float, 3)

    # ------------------------------ PCG driver ------------------------------

    @qd.kernel
    def _kernel_pcg_init(self, f: qd.i32):
        for cell in qd.grouped(self.p_x):
            self.p_x[cell] = 0.0
            if self.grid2mat[cell] >= 0:
                self.p_z[cell] = self.p_r[cell] / self.p_diag[cell]
            else:
                self.p_r[cell] = 0.0
                self.p_z[cell] = 0.0
            self.p_d[cell] = self.p_z[cell]

    @qd.kernel
    def _kernel_pcg_update(self, f: qd.i32, alpha: qd.f32):
        for cell in qd.grouped(self.p_x):
            self.p_x[cell] = self.p_x[cell] + alpha * self.p_d[cell]
            self.p_r[cell] = self.p_r[cell] - alpha * self.p_ap[cell]

    @qd.kernel
    def _kernel_pcg_precond(self, f: qd.i32):
        for cell in qd.grouped(self.p_x):
            if self.grid2mat[cell] >= 0:
                self.p_z[cell] = self.p_r[cell] / self.p_diag[cell]
            else:
                self.p_z[cell] = 0.0

    @qd.kernel
    def _kernel_pcg_update_dir(self, f: qd.i32, beta: qd.f32):
        for cell in qd.grouped(self.p_x):
            self.p_d[cell] = self.p_z[cell] + beta * self.p_d[cell]

    def _dot(self, fa, fb):
        return float((qd_to_torch(fa) * qd_to_torch(fb)).sum().item())

    def _pcg_solve(self, f, matvec_kernel):
        # Jacobi-PCG, matrix-free; solves A x = rhs with the system prepared in p_rhs / p_diag
        self.p_x.fill(0.0)
        self.p_r.copy_from(self.p_rhs)
        self._kernel_pcg_init(f)
        rho = self._dot(self.p_r, self.p_z)
        rhs_norm = math.sqrt(max(self._dot(self.p_rhs, self.p_rhs), 1e-30))
        for _ in range(self._pcg_max_iter):
            matvec_kernel(f)
            alpha = rho / max(self._dot(self.p_d, self.p_ap), 1e-30)
            self._kernel_pcg_update(f, alpha)
            res = math.sqrt(max(self._dot(self.p_r, self.p_r), 0.0))
            if res < self._pcg_tol * rhs_norm:
                break
            self._kernel_pcg_precond(f)
            rho_new = self._dot(self.p_r, self.p_z)
            self._kernel_pcg_update_dir(f, rho_new / max(rho, 1e-30))
            rho = rho_new

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        pass

    def substep_pre_coupling(self, f):
        if not self.is_active:
            return

        # C++ GetCourantTimeStep: fluid steps of dt_f = min(2 * dx / max|v_face|, remaining dt) within
        # each simulator substep (the reference's upper clamp m_ddt * 1000 is always looser than the
        # simulator substep here, and its lower clamp is 0)
        dt_remaining = float(self._substep_dt)
        while dt_remaining > 0.0:
            max_vel = np.zeros(1, dtype=gs.np_float)
            self._kernel_max_face_vel(max_vel)
            cfl_dt = 2.0 * self._dx / max(float(max_vel[0]), 1e-12)
            dt_f = min(cfl_dt, dt_remaining)
            if self._dem_coupling:
                # C++ Advance: MoveDEMParticles(dt_f) runs before the water path of each fluid step
                self._sim.dem_solver.advance_coupled(f, dt_f)
            self._water_step(f, dt_f)
            dt_remaining -= dt_f

    def _water_step(self, f, dt):
        # one full water step of the reference's Advance (water-only path)
        # 1. advect particles with the current grid velocity, rebuild the level set
        self._kernel_advect_particles(f, dt)
        self._kernel_reconstruct_levelset(f)
        self._kernel_set_unknowns(f)

        # sand volume fraction for this water step (C++ calls CalFraction before ProjectDensity and
        # inside ProjectVelocity; the grains do not move during a water step, so once suffices)
        if self._dem_coupling:
            self._kernel_cal_fraction(f)

        # 2. density correction (IDP)
        if self._density_correction:
            self._kernel_cal_density(f)
            self._kernel_correction_rhs_diag(f)
            self._pcg_solve(f, self._kernel_correction_matvec)
            self._kernel_apply_correction_faces(f)
            self._kernel_apply_correction_particles(f)
            self._kernel_enforce_particles_boundary(f)
            self._kernel_reconstruct_levelset(f)
            self._kernel_set_unknowns(f)

        # 3. P2G
        self._kernel_p2g_scatter(f)
        self._kernel_p2g_normalize(f)
        self._kernel_extrapolate_mark_weights(f)
        for _ in range(3):
            self._kernel_extrapolate_sweep(f)
        self._kernel_save_velocity(f)

        # rotation drive (C++ Advance: after P2G, before body forces, while t < rotate_duration)
        if self._rotate_omega != 0.0 and self._sim.cur_t < self._rotate_duration:
            self._kernel_apply_rotation(f, dt)

        # 4. body forces, then zero the boundary faces: the reference divergence uses the wall
        # velocity (0) on solid faces, so gravity must not leak into the rhs through them
        self._kernel_apply_gravity(f, dt)
        self._kernel_enforce_velocity_boundary(f)

        # 5. pressure projection
        self.grad_p_u.fill(0.0)
        self.grad_p_v.fill(0.0)
        self.grad_p_w.fill(0.0)
        self._kernel_projection_rhs_diag(f)
        self._pcg_solve(f, self._kernel_projection_matvec)
        self._kernel_apply_projection(f)
        self._kernel_extrapolate_mark_fluid(f)
        for _ in range(6):
            self._kernel_extrapolate_sweep(f)
        self._kernel_enforce_velocity_boundary(f)
        self._kernel_update_veldiff(f)

        # 6. G2P (FLIP/PIC blend)
        self._kernel_g2p(f)

        # 7. water absorption by sand (C++ CacheFluidIndex / WetDEMParticle / RemoveDeadParticle;
        # absorbed fluid particles are deactivated in place instead of being erased)
        if self._dem_coupling:
            self._kernel_wet_reset(f)
            self._kernel_absorb_mark(f)
            self._kernel_absorb_finalize(f)
            self._kernel_wet_dem_particles(f)

        # C++ m_LastDeltaTime = deltaTime at the end of Advance
        self._last_dt = dt

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if self.is_active:
            self._kernel_update_render_fields(f)

    def substep_post_coupling_grad(self, f):
        pass

    @qd.kernel
    def _kernel_update_render_fields(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            self.particles_render[i_p, i_b].pos = self.particles[i_p, i_b].pos
            self.particles_render[i_p, i_b].active = self.particles[i_p, i_b].active

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def reset_grad(self):
        pass

    def collect_output_grads(self):
        pass

    def add_grad_from_state(self, state):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    # ------------------------------------------------------------------------------------
    # -------------------------------------- state ---------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state_render(self):
        if not self.is_active:
            return None, None, None
        self.update_render_fields()
        return self.particles_render.pos, None, None

    def update_render_fields(self):
        self._kernel_update_render_fields(self.sim.cur_substep_local)

    def get_state(self, f):
        if not self.is_active:
            return None
        from genesis.engine.states.solvers import FLIPSolverState

        state = FLIPSolverState(self.scene)
        self._kernel_get_state(f, state.pos, state.vel, state.active)
        self._kernel_get_grid_state(state.grid_vel_u, state.grid_vel_v, state.grid_vel_w)
        return state

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            self._kernel_set_state(f, state.pos, state.vel, state.active)
            self._kernel_set_grid_state(state.grid_vel_u, state.grid_vel_v, state.grid_vel_w)
            self._last_dt = float(state.last_dt)

    @qd.kernel
    def _kernel_get_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for axis in qd.static(range(3)):
                pos[i_b, i_p, axis] = self.particles[i_p, i_b].pos[axis]
                vel[i_b, i_p, axis] = self.particles[i_p, i_b].vel[axis]
            active[i_b, i_p] = self.particles[i_p, i_b].active

    @qd.kernel
    def _kernel_set_state(
        self,
        f: qd.i32,
        pos: qd.types.ndarray(),
        vel: qd.types.ndarray(),
        active: qd.types.ndarray(),
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            for axis in qd.static(range(3)):
                self.particles[i_p, i_b].pos[axis] = pos[i_b, i_p, axis]
                self.particles[i_p, i_b].vel[axis] = vel[i_b, i_p, axis]
            self.particles[i_p, i_b].active = qd.cast(active[i_b, i_p], gs.qd_bool)

    @qd.kernel
    def _kernel_get_grid_state(
        self,
        vel_u: qd.types.ndarray(),
        vel_v: qd.types.ndarray(),
        vel_w: qd.types.ndarray(),
    ):
        for i, j, k in qd.ndrange(*self.vel_u.shape):
            vel_u[i, j, k] = self.vel_u[i, j, k]
        for i, j, k in qd.ndrange(*self.vel_v.shape):
            vel_v[i, j, k] = self.vel_v[i, j, k]
        for i, j, k in qd.ndrange(*self.vel_w.shape):
            vel_w[i, j, k] = self.vel_w[i, j, k]

    @qd.kernel
    def _kernel_set_grid_state(
        self,
        vel_u: qd.types.ndarray(),
        vel_v: qd.types.ndarray(),
        vel_w: qd.types.ndarray(),
    ):
        for i, j, k in qd.ndrange(*self.vel_u.shape):
            self.vel_u[i, j, k] = vel_u[i, j, k]
        for i, j, k in qd.ndrange(*self.vel_v.shape):
            self.vel_v[i, j, k] = vel_v[i, j, k]
        for i, j, k in qd.ndrange(*self.vel_w.shape):
            self.vel_w[i, j, k] = vel_w[i, j, k]

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
    def _kernel_set_particles_active(
        self,
        particles_idx: qd.types.ndarray(),
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
            i_p = particles_idx[i_b_, i_p_]
            i_b = envs_idx[i_b_]
            self.particles[i_p, i_b].active = qd.cast(actives[i_b_, i_p_], gs.qd_bool)

    @qd.kernel
    def _kernel_get_particles_active(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        envs_idx: qd.types.ndarray(),
        actives: qd.types.ndarray(),
    ):
        for i_p_, i_b_ in qd.ndrange(n_particles, envs_idx.shape[0]):
            i_p = i_p_ + particle_start
            i_b = envs_idx[i_b_]
            actives[i_b_, i_p_] = self.particles[i_p, i_b].active

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        return sum([entity.n_particles for entity in self._entities])

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound
