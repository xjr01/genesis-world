from typing import TYPE_CHECKING

import math

import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary, CylinderBoundary, PlaneBoundary, TiltedCylinderBoundary
from genesis.engine.entities import IPBFEntity
from genesis.engine.states.solvers import IPBFSolverState
from genesis.utils.misc import qd_to_numpy

from .base_solver import Solver

if TYPE_CHECKING:
    from genesis.engine.entities import IPBFEntity


@qd.data_oriented
class IPBFSolver(Solver):
    # spherical-illumination surface classifier resolution (Shibata 2015, PBSTF reference)
    _N_THETA = 18
    _N_PHI = 36
    # capacity overflow codes (spec D10: report via gs.raise_exception, like the PBSTF reference)
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
        self._particle_size = options.particle_size
        self._support_radius = options._support_radius

        # IPBF parameters
        self._ipbf_iterations = options.ipbf_iterations
        self._alpha = options.alpha
        # non-paper experiment: rebuild the spatial hash inside the Newton loop (paper fixes it)
        self._rebuild_hash_per_iter = options.rebuild_neighbors_per_iter

        # artificial damping (paper section 3.6, eqs. 16-18)
        self._damping_enabled = options.damping_enabled
        self._damping_alpha_star = options.damping_alpha_star
        self._damping_beta = options.damping_beta

        # static boundary particles (PLAN P4.1)
        self._boundary_particles_enabled = options.boundary_particles
        self._boundary_layers = options.boundary_layers
        # cylindrical clamp boundary (multiflow MF-7); None keeps the CubeBoundary box
        self._boundary_cylinder = options.boundary_cylinder
        # tilted open-mouth pitcher clamp (multiflow MF-10); None disables it (bit-identical)
        self._boundary_pitcher = options.boundary_pitcher
        self._has_boundary_pitcher = options.boundary_pitcher is not None

        # horizontal table plane (multiflow MF-13 pattern, PBD parity); None disables it
        self._boundary_plane = options.boundary_plane
        self._has_boundary_plane = options.boundary_plane is not None

        # explicit XSPH viscosity (PLAN P7; user-requested, NOT part of the IPBF paper)
        self._viscosity_xsph = options.viscosity_xsph
        # interface-restricted variant: surface flags come from the ST topology pass, so a
        # positive coefficient requires the ST machinery to run
        self._surface_viscosity_xsph = options.surface_viscosity_xsph
        if self._surface_viscosity_xsph > 0.0 and not options.surface_tension_enabled:
            gs.raise_exception("surface_viscosity_xsph requires surface_tension_enabled=True.")

        # demo-level XSPH-style concentration diffusion (multiflow demo; 0 = off)
        self._diffusion_coeff = options.diffusion_coeff

        # surface tension (ST_IPBF_DERIVATION.md; all ST logic is compile-time conditioned so that
        # surface_tension_enabled=False keeps the solver bit-identical to the pre-ST version)
        self._st_enabled = options.surface_tension_enabled
        self._st_model_linear = options.st_model == "linear"
        self._st_stiffness = options.st_stiffness
        self._st_distance_enabled = options.st_distance_enabled
        self._st_distance_stiffness = options.st_distance_stiffness
        self._st_ring_radius = options.st_ring_radius_factor * options.particle_size
        self._st_topo_interval = options.st_topo_interval
        self._st_lit_threshold = options.st_lit_threshold
        self._st_normal_compat = options.st_normal_compat
        self._st_max_surface_neighbors = options.st_max_surface_neighbors
        self._st_max_localmesh_neighbors = options.st_max_localmesh_neighbors
        self._st_topo_counter = 0
        # linear mode (plan (a)) needs a finite density compliance (D2); quadratic keeps alpha
        self._alpha_eff = options.st_alpha_ref if (self._st_enabled and self._st_model_linear) else options.alpha
        # area-energy weight: quadratic -> k~_st directly; linear -> w_st = sigma * alpha_eff
        # (sigma is a material property, resolved in build(); placeholder here)
        self._st_area_weight = options.st_stiffness

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
        if self._boundary_cylinder is None:
            self.boundary = CubeBoundary(
                lower=self._lower_bound,
                upper=self._upper_bound,
            )
        else:
            # 4-tuple: (cx, cy, radius, z_bottom), wall infinitely tall (MF-7..9);
            # 5-tuple: + z_top = rim height — radial wall clamp applies only below the rim
            # (open-top cup; MF-10, so a pitcher hovering above the rim is not sucked back)
            cx, cy, radius, z_bottom = self._boundary_cylinder[:4]
            z_top = self._boundary_cylinder[4] if len(self._boundary_cylinder) > 4 else None
            self.boundary = CylinderBoundary(
                center_xy=(cx, cy),
                radius=radius,
                z_bottom=z_bottom,
                z_top=z_top,
                band=1.5 * self._particle_size,
            )
        # optional second clamp (multiflow MF-10 pitcher); applied after `boundary`
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

        # optional table plane (multiflow MF-13 pattern, PBD parity); applied to every particle
        # after the container clamps, so liquid escaping a container lands on it
        self.boundary3 = None
        if self._has_boundary_plane:
            if len(self._boundary_plane) == 1:
                self.boundary3 = PlaneBoundary(z0=self._boundary_plane[0])
            else:
                pz0, pcx, pcy, p_radius = self._boundary_plane
                self.boundary3 = PlaneBoundary(z0=pz0, center_xy=(pcx, pcy), radius=p_radius)

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
            xstar=gs.qd_vec3,  # alternative position x* with compliance alpha* (artificial damping)
            dv=gs.qd_vec3,  # Jacobi delta-v scratch (explicit XSPH viscosity, PLAN P7)
            c=gs.qd_float,  # concentration (multiflow demo: 0=water, 1=coffee)
            dc=gs.qd_float,  # Jacobi buffer for the concentration diffusion pass
        )

        # dynamic particle state without gradient
        struct_particle_state_ng = qd.types.struct(
            reordered_idx=gs.qd_int,
            active=gs.qd_bool,
            is_boundary=gs.qd_bool,  # static Akinci-style boundary particle (fixed, no own constraint)
            boundary_group=gs.qd_int,  # clamp ownership (multiflow MF-11): 0 = primary boundary, 1 = pitcher
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
            c=gs.qd_float,  # concentration (multiflow demo)
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

        # surface-tension topology fields (spec section 6.1). Allocated only when ST is enabled
        # (~8 KB/particle); everything lives in the ORIGINAL index space (neighbors are resolved
        # through `particle_idx`), unlike the PBSTF reference which works in reordered space (E5).
        if self._st_enabled:
            n, b = self._n_particles, self._B
            self.on_surface = qd.field(gs.qd_bool, shape=(n, b))
            self.topology_valid = qd.field(gs.qd_bool, shape=(n, b))
            self.normal = qd.field(gs.qd_vec3, shape=(n, b))
            self.CA = qd.field(gs.qd_float, shape=(n, b))  # area constraint C_i^A (recomputed every iteration)
            self.n_ring = qd.field(gs.qd_int, shape=(n, b))
            # int difference array of the illumination screen (N_THETA+1 x N_PHI+1); the classify
            # pass restores per-cell counts with a 2D prefix sum — a bool 18x36 screen cannot do this
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
            # reverse one-ring index (spec section 6.2): for each particle i, the list of
            # (owner j, ring slot k) with i == local_mesh_neighbors[j, k]; entry = (j << 7) | k
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
            if self._boundary_cylinder is not None and self._boundary_particles_enabled:
                gs.logger.warning(
                    "IPBFOptions: boundary_cylinder is set while boundary_particles=True — boundary "
                    "particles are still sampled on the box walls (lower/upper_bound) and will sit in "
                    "the corners outside the cylinder. Set boundary_particles=False for cylinder scenes."
                )
            # fluid particles come first; static boundary particles are appended after them
            self._n_fluid_particles = self._n_particles
            boundary_pos = (
                self._sample_boundary_particles() if self._boundary_particles_enabled else np.zeros((0, 3))
            )
            self._n_boundary_particles = len(boundary_pos)
            self._n_particles = self._n_fluid_particles + self._n_boundary_particles

            self.sh.build(self._B)
            self.init_particle_fields()
            self.init_ckpt()

            for entity in self.entities:
                entity._add_to_solver()

            if self._n_boundary_particles > 0:
                self._kernel_add_boundary_particles(
                    self._n_fluid_particles,
                    self._n_boundary_particles,
                    self.entities[0].material.rho,
                    boundary_pos,
                )
            gs.logger.info(f"IPBFSolver: {self._n_fluid_particles} fluid + {self._n_boundary_particles} boundary particles.")

        # linear ST model: w_st = sigma * alpha_eff / 3 (spec eq. 1.4 with the measured fan-overlap
        # correction). The discrete area sum Σ_j C_j^A = 3 × A_true on a lattice-ordered surface
        # (ST-0: fan = 3 × Voronoi; dbg_st2_grad.py: net linear gradient = 3.00 × the physical
        # mean-curvature residual (2/R)·A_p), so the spec's w_st = sigma*alpha would apply 3× the
        # physical surface tension. sigma is a material property. Quadratic mode is unaffected.
        if self._st_enabled and self._st_model_linear:
            if not self.entities:
                gs.raise_exception("IPBF linear surface-tension model requires at least one entity.")
            self._st_area_weight = self.entities[0].material.surface_tension * self._alpha_eff / 3.0

        # FIXME: _gravity must be a raw qd.field() — see comment in mpm_solver.py
        # Only when active — see the SNode-tree note in mpm_solver.py.
        if self.is_active and self._gravity is not None:
            gravity = self._gravity.to_numpy()
            self._gravity = qd.field(dtype=gs.qd_vec3, shape=(self._B,))
            self._gravity.from_numpy(gravity)

    def _sample_boundary_particles(self):
        """
        Sample static boundary particles on the boundary box walls (PLAN P4.1, Akinci et al. 2012 style):
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
            # layer 0 sits ONE particle spacing outside the wall plane, further layers one more
            # spacing each. (Deviation from PLAN P4.1's literal "layer 0 on the wall": fluid
            # particles clamped by CubeBoundary land exactly on the wall plane, so a wall-plane
            # boundary lattice would produce r=0 duplicate pairs and contact-density spikes.
            # At 1x spacing the contact geometry is the standard Akinci interleaved lattice.)
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
        grid_idx = np.round(pos / ps).astype(np.int64)
        _, uniq_idx = np.unique(grid_idx, axis=0, return_index=True)
        return pos[np.sort(uniq_idx)].astype(gs.np_float)

    @qd.kernel
    def _kernel_add_boundary_particles(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        mat_rho: qd.f32,
        pos: qd.types.ndarray(),
    ):
        for i_p_, i_b in qd.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            self.particles_ng[i_p, i_b].active = True
            self.particles_ng[i_p, i_b].is_boundary = True
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].pos[i] = pos[i_p_, i]
            self.particles[i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)

        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            self.particles_info[i_p].rho = mat_rho
            self.particles_info[i_p].mass = self._particle_volume * mat_rho

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
    # ----------------------- surface tension: topology kernels --------------------------
    # ------------------------------------------------------------------------------------
    # Ported from the PBSTF reference (surface tension/genesis-world-main/.../pbstf_solver.py)
    # into the ORIGINAL index space: all fields are indexed [i_p, i_b] and hash-slot neighbors
    # are mapped back through `particle_idx` (spec section 6.1, E5). ST topology passes use the
    # one-ring radius R_st = st_ring_radius_factor * particle_size (= 3 ps, spec section 3.2),
    # while the density kernel keeps R = 2 ps.

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

    @qd.kernel
    def _kernel_st_mark_surface_screen(self):
        # spherical-illumination screen marking (Shibata 2015; pbstf_solver.py:433-482).
        # Boundary particles neither own a screen nor block light (spec section 3.3, D7).
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active and not self.particles_ng[i_p, i_b].is_boundary:
                pos_i = self.particles[i_p, i_b].pos
                base = self.sh.pos_to_grid(pos_i)
                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    slot_idx = self.sh.grid_to_slot(base + offset)
                    for j_r in range(
                        self.sh.slot_start[slot_idx, i_b],
                        self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                    ):
                        j = self.particle_idx[j_r, i_b]
                        if j != i_p and not self.particles_ng[j, i_b].is_boundary:
                            delta = self.particles[j, i_b].pos - pos_i
                            distance = delta.norm()
                            if distance > gs.EPS and distance < self._st_ring_radius:
                                unit_theta = math.pi / self._N_THETA
                                unit_phi = 2.0 * math.pi / self._N_PHI
                                block_radius = qd.min(0.5 * self._particle_size, 0.5 * distance)
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
                                st_p = qd.min(
                                    qd.cast(qd.floor((start_phi + math.pi) / unit_phi), gs.qd_int), self._N_PHI - 1
                                )
                                en_p = qd.min(qd.cast(qd.ceil((end_phi + math.pi) / unit_phi), gs.qd_int), self._N_PHI)

                                # int difference array: add +1/-1 at the rectangle corners
                                if st_p < en_p:
                                    qd.atomic_add(self._screen_blocked[i_p, i_b, st_t, st_p], 1)
                                    qd.atomic_add(self._screen_blocked[i_p, i_b, st_t, en_p], -1)
                                    qd.atomic_add(self._screen_blocked[i_p, i_b, en_t, st_p], -1)
                                    qd.atomic_add(self._screen_blocked[i_p, i_b, en_t, en_p], 1)
                                else:
                                    # phi interval wraps around the +-pi seam: two rectangles
                                    qd.atomic_add(self._screen_blocked[i_p, i_b, st_t, st_p], 1)
                                    qd.atomic_add(self._screen_blocked[i_p, i_b, st_t, self._N_PHI], -1)
                                    qd.atomic_add(self._screen_blocked[i_p, i_b, en_t, st_p], -1)
                                    qd.atomic_add(self._screen_blocked[i_p, i_b, en_t, self._N_PHI], 1)
                                    qd.atomic_add(self._screen_blocked[i_p, i_b, st_t, 0], 1)
                                    qd.atomic_add(self._screen_blocked[i_p, i_b, st_t, en_p], -1)
                                    qd.atomic_add(self._screen_blocked[i_p, i_b, en_t, 0], -1)
                                    qd.atomic_add(self._screen_blocked[i_p, i_b, en_t, en_p], 1)

    @qd.kernel
    def _kernel_st_classify_surface(self):
        # 2D prefix sum restores per-cell blocked counts; a cell with count 0 is illuminated.
        # Solid-angle weights sin(theta); surface iff illuminated fraction >= 1/9 (pbstf :485-508).
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active and not self.particles_ng[i_p, i_b].is_boundary:
                illuminated = gs.qd_float(0.0)
                total = gs.qd_float(0.0)
                for t in range(self._N_THETA):
                    weight = qd.sin(math.pi / self._N_THETA * (t + 0.5))
                    for p in range(self._N_PHI):
                        if t > 0 and p > 0:
                            self._screen_blocked[i_p, i_b, t, p] += (
                                self._screen_blocked[i_p, i_b, t - 1, p]
                                + self._screen_blocked[i_p, i_b, t, p - 1]
                                - self._screen_blocked[i_p, i_b, t - 1, p - 1]
                            )
                        elif t > 0:
                            self._screen_blocked[i_p, i_b, t, p] += self._screen_blocked[i_p, i_b, t - 1, p]
                        elif p > 0:
                            self._screen_blocked[i_p, i_b, t, p] += self._screen_blocked[i_p, i_b, t, p - 1]
                        if self._screen_blocked[i_p, i_b, t, p] == 0:
                            illuminated += weight
                        total += weight
                self.on_surface[i_p, i_b] = illuminated >= self._st_lit_threshold * total
            else:
                self.on_surface[i_p, i_b] = False

    @qd.kernel
    def _kernel_st_compute_normals(self):
        # color-field gradient normals (Muller 2003; pbstf :511-565). The m_j/rho_j weight becomes
        # V / rho~_j in the volume-normalized IPBF density (requires the topology density pre-pass,
        # spec section 3.4 E4); boundary neighbors participate with rho~_j = 1 (spec section 3.5-2).
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_ng[i_p, i_b].active
                and not self.particles_ng[i_p, i_b].is_boundary
                and self.on_surface[i_p, i_b]
            ):
                pos_i = self.particles[i_p, i_b].pos
                n_raw = qd.Vector.zero(gs.qd_float, 3)
                base = self.sh.pos_to_grid(pos_i)
                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    slot_idx = self.sh.grid_to_slot(base + offset)
                    for j_r in range(
                        self.sh.slot_start[slot_idx, i_b],
                        self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                    ):
                        j = self.particle_idx[j_r, i_b]
                        if j != i_p:
                            delta = self.particles[j, i_b].pos - pos_i
                            distance = delta.norm()
                            if distance > gs.EPS and distance < self._st_ring_radius:
                                if self.particles_ng[j, i_b].is_boundary:
                                    # compromise of spec section 3.5-2: rho~_j = 1 for boundary neighbors
                                    n_raw += self._st_cubic_dW(distance) * (delta / distance) * self._particle_volume
                                else:
                                    rho_j = self.particles[j, i_b].rho
                                    if rho_j > gs.EPS:
                                        n_raw += (
                                            self._st_cubic_dW(distance)
                                            * (delta / distance)
                                            * (self._particle_volume / rho_j)
                                        )
                self.normal[i_p, i_b] = n_raw

        # normalize; |n| <= 1 downgrades the particle to non-surface (pbstf :547-565)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_ng[i_p, i_b].active
                and not self.particles_ng[i_p, i_b].is_boundary
                and self.on_surface[i_p, i_b]
            ):
                raw_normal = self.normal[i_p, i_b]
                raw_length = raw_normal.norm()
                if raw_length <= 1.0:
                    self.on_surface[i_p, i_b] = False
                    self.normal[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
                else:
                    self.normal[i_p, i_b] = raw_normal / raw_length

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

    @qd.kernel
    def _kernel_st_build_local_meshes(self):
        # tangent-plane projection -> polar insertion sort -> fan triangulation + flip queue
        # (pbstf :614-774); normal-compatibility filter separates thin-film sides.
        self.topology_valid.fill(False)
        self.n_ring.fill(0)

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_ng[i_p, i_b].active
                and not self.particles_ng[i_p, i_b].is_boundary
                and self.on_surface[i_p, i_b]
            ):
                pos_i = self.particles[i_p, i_b].pos
                normal = self.normal[i_p, i_b]
                axis_x = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
                if axis_x.cross(normal).norm() < gs.EPS:
                    axis_x = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
                axis_x = axis_x.cross(normal).normalized()
                axis_y = normal.cross(axis_x).normalized()
                self._mesh_axis_x[i_p, i_b] = axis_x
                self._mesh_axis_y[i_p, i_b] = axis_y

                # collect one-ring candidates (surface neighbors within R_st, normal-compatible)
                count = gs.qd_int(0)
                cand_overflow = False
                base = self.sh.pos_to_grid(pos_i)
                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    slot_idx = self.sh.grid_to_slot(base + offset)
                    for j_r in range(
                        self.sh.slot_start[slot_idx, i_b],
                        self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                    ):
                        j = self.particle_idx[j_r, i_b]
                        if j != i_p and self.on_surface[j, i_b]:
                            delta = self.particles[j, i_b].pos - pos_i
                            distance = delta.norm()
                            if distance < self._st_ring_radius:
                                normal_j = self.normal[j, i_b]
                                if normal.dot(normal_j) > self._st_normal_compat or (
                                    (normal - normal_j).dot(delta) > 0.0 and distance < 2.0 * self._particle_size
                                ):
                                    if count < self._st_max_surface_neighbors:
                                        self.neighbor_ids[i_p, i_b, count] = j
                                        projected = delta - delta.dot(normal) * normal
                                        self.projected_positions[i_p, i_b, count] = qd.Vector(
                                            [projected.dot(axis_x), projected.dot(axis_y)], dt=gs.qd_float
                                        )
                                        count += 1
                                    else:
                                        cand_overflow = True
                if cand_overflow:
                    qd.atomic_max(self._overflow[None], self._ST_SURFACE_NEIGHBOR_OVERFLOW)
                n = count
                self.n_ring[i_p, i_b] = n

                for k in range(n):
                    self._node_queue[i_p, i_b, k] = k
                    self._chain_pre[i_p, i_b, k] = -1
                    self._chain_nxt[i_p, i_b, k] = -1

                # insertion sort by polar angle, then by radius
                for k in range(1, n):
                    cursor = k
                    while cursor > 0:
                        u = self._node_queue[i_p, i_b, cursor - 1]
                        v = self._node_queue[i_p, i_b, cursor]
                        x_u = self.projected_positions[i_p, i_b, u]
                        x_v = self.projected_positions[i_p, i_b, v]
                        angle_u = qd.atan2(x_u[1], x_u[0])
                        angle_v = qd.atan2(x_v[1], x_v[0])
                        if angle_u > angle_v or (angle_u == angle_v and x_u.norm() > x_v.norm()):
                            self._node_queue[i_p, i_b, cursor - 1] = v
                            self._node_queue[i_p, i_b, cursor] = u
                            cursor -= 1
                        else:
                            cursor = 0

                ring_size = gs.qd_int(0)
                if n >= 3:
                    for k in range(n - 1):
                        u = self._node_queue[i_p, i_b, k]
                        v = self._node_queue[i_p, i_b, k + 1]
                        self._chain_nxt[i_p, i_b, u] = v
                        self._chain_pre[i_p, i_b, v] = u
                    u_last = self._node_queue[i_p, i_b, n - 1]
                    u_first = self._node_queue[i_p, i_b, 0]
                    self._chain_nxt[i_p, i_b, u_last] = u_first
                    self._chain_pre[i_p, i_b, u_first] = u_last

                    queue_start = gs.qd_int(0)
                    queue_end = gs.qd_int(0)
                    for k in range(n):
                        u = self._node_queue[i_p, i_b, k]
                        if self._st_need_flip(i_p, i_b, u):
                            if queue_end < 3 * self._st_max_surface_neighbors:
                                self._node_queue[i_p, i_b, queue_end] = u
                                queue_end += 1
                            else:
                                qd.atomic_max(self._overflow[None], self._ST_LOCAL_MESH_QUEUE_OVERFLOW)

                    while queue_start < queue_end:
                        u = self._node_queue[i_p, i_b, queue_start]
                        queue_start += 1
                        if self._chain_nxt[i_p, i_b, u] >= 0 and self._st_need_flip(i_p, i_b, u):
                            u_pre = self._chain_pre[i_p, i_b, u]
                            u_nxt = self._chain_nxt[i_p, i_b, u]
                            self._chain_nxt[i_p, i_b, u_pre] = u_nxt
                            self._chain_pre[i_p, i_b, u_nxt] = u_pre
                            if self._st_need_flip(i_p, i_b, u_nxt):
                                if queue_end < 3 * self._st_max_surface_neighbors:
                                    self._node_queue[i_p, i_b, queue_end] = u_nxt
                                    queue_end += 1
                                else:
                                    qd.atomic_max(self._overflow[None], self._ST_LOCAL_MESH_QUEUE_OVERFLOW)
                            if self._st_need_flip(i_p, i_b, u_pre):
                                if queue_end < 3 * self._st_max_surface_neighbors:
                                    self._node_queue[i_p, i_b, queue_end] = u_pre
                                    queue_end += 1
                                else:
                                    qd.atomic_max(self._overflow[None], self._ST_LOCAL_MESH_QUEUE_OVERFLOW)
                            self._chain_nxt[i_p, i_b, u] = -1
                            self._chain_pre[i_p, i_b, u] = -1

                    start = gs.qd_int(-1)
                    for k in range(n):
                        if start < 0 and self._chain_nxt[i_p, i_b, k] >= 0:
                            start = k

                    projected_area_twice = gs.qd_float(0.0)
                    if start >= 0:
                        u = start
                        keep_walking = True
                        while keep_walking and ring_size < self._st_max_localmesh_neighbors:
                            u_next = self._chain_nxt[i_p, i_b, u]
                            x_u = self.projected_positions[i_p, i_b, u]
                            x_next = self.projected_positions[i_p, i_b, u_next]
                            projected_area_twice += x_u[0] * x_next[1] - x_u[1] * x_next[0]
                            self.local_mesh_neighbors[i_p, i_b, ring_size] = self.neighbor_ids[i_p, i_b, u]
                            ring_size += 1
                            u = u_next
                            if u == start:
                                keep_walking = False

                        if keep_walking:
                            qd.atomic_max(self._overflow[None], self._ST_LOCAL_MESH_NEIGHBOR_OVERFLOW)
                        else:
                            self.topology_valid[i_p, i_b] = ring_size >= 3 and qd.abs(projected_area_twice) > gs.EPS

                    self.n_ring[i_p, i_b] = ring_size

    @qd.kernel
    def _kernel_st_build_reverse_rings(self):
        # reverse one-ring index (spec section 6.2): for each mesh owner j and ring slot k, append
        # (j, k) to the reverse list of u = ring[j][k]; entry packs (j << 7) | k. The Newton
        # assembly of particle i must collect dC_j^A/dx_i from every owner j whose ring contains i.
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_ng[i_p, i_b].active
                and not self.particles_ng[i_p, i_b].is_boundary
                and self.on_surface[i_p, i_b]
                and self.topology_valid[i_p, i_b]
            ):
                n = self.n_ring[i_p, i_b]
                for k in range(n):
                    u = self.local_mesh_neighbors[i_p, i_b, k]
                    slot = qd.atomic_add(self._rev_count[u, i_b], 1)
                    if slot < self._st_max_surface_neighbors:
                        self._rev_ids[u, i_b, slot] = i_p * (1 << self._ST_REV_SHIFT) + k
                    else:
                        qd.atomic_max(self._overflow[None], self._ST_REVERSE_RING_OVERFLOW)

    @qd.kernel
    def _kernel_st_compute_area_C(self, f: qd.i32):
        # C_i^A = sum of the fan triangle areas over the polar-ordered one-ring (PBSTF eq. 7);
        # recomputed at the CURRENT positions every Newton iteration (topology stays fixed).
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            CA = gs.qd_float(0.0)
            if (
                self.particles_ng[i_p, i_b].active
                and not self.particles_ng[i_p, i_b].is_boundary
                and self.on_surface[i_p, i_b]
                and self.topology_valid[i_p, i_b]
            ):
                pos_i = self.particles[i_p, i_b].pos
                n = self.n_ring[i_p, i_b]
                for k in range(n):
                    k_next = 0 if k == n - 1 else k + 1
                    j1 = self.local_mesh_neighbors[i_p, i_b, k]
                    j2 = self.local_mesh_neighbors[i_p, i_b, k_next]
                    CA += 0.5 * (self.particles[j1, i_b].pos - pos_i).cross(self.particles[j2, i_b].pos - pos_i).norm()
            self.CA[i_p, i_b] = CA

    def _st_rebuild_topology(self, f):
        # spec section 3.4: the density pre-pass is mandatory before normals (m_j/rho_j weights, E4)
        self._kernel_compute_density_C(f)
        self._screen_blocked.fill(0)
        self._kernel_st_mark_surface_screen()
        self._kernel_st_classify_surface()
        self._kernel_st_compute_normals()
        self._overflow.fill(0)
        self._kernel_st_build_local_meshes()
        self._rev_count.fill(0)
        self._kernel_st_build_reverse_rings()
        # overflow strategy (D10): raise like the PBSTF reference
        overflow = int(qd_to_numpy(self._overflow, transpose=True)[()])
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
        """
        Gradient (spec eq. 2.3) and diagonal Hessian block (eq. 2.7 with s_uu = 0) of the triangle
        area A(p1, p2, p3) with respect to vertex u in {1, 2, 3}. The diagonal block
        H_uu = (|e_u|^2 I - e_u e_u^T) / (4A) - grad grad^T / A is always PSD (spec section 2.4),
        so NO column-norm diagonal approximation is applied to the area term. Degenerate triangles
        (2A <= EPS) contribute nothing (A -> 0 guard, spec section 2.3).
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

    @qd.kernel
    def _kernel_predict(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            # boundary particles are static: no prediction
            if self.particles_ng[i_p, i_b].active and not self.particles_ng[i_p, i_b].is_boundary:
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
            # boundary particles only contribute density; they carry no constraint of their own
            if self.particles_ng[i_p, i_b].active and not self.particles_ng[i_p, i_b].is_boundary:
                pos_i = self.particles[i_p, i_b].pos
                # volume-normalized density rho~_i = sum_j V W_ij over fluid AND boundary neighbors
                # (includes the self term V W(0)); boundary neighbors contribute with V_b = V
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
        """
        Assemble f_i (eq. 10) and H_i (eq. 11 + column-norm diagonal approximation, eq. 15) from the
        per-particle accumulators and solve Delta x_i = H_i^-1 f_i via the 3x3 adjugate / determinant.
        Shared by the plain Newton step (alpha) and the alternative solution x* (alpha*).
        Surface-tension accumulators (spec eqs. 2.11/2.12): f_st / H_st hold the area term
        (quadratic eq. 2.9 or linear eq. 2.10) plus the distance-constraint Gauss-Newton term;
        d_st_diag holds the distance-constraint geometric term after the column-norm diagonal
        approximation (C^d < 0, spec eq. 2.15 / section 2.8). All three are exactly zero and the
        additions compile away when surface_tension_enabled=False (zero regression).
        """
        # f_i: negative gradient (eq. 10)
        f_i = -alpha_over_h2 * x_minus_y - C_i * g_ii - sum_Cj_gij

        # H_i: eq. 11 with the geometric stiffness replaced by its column-norm diagonal
        # approximation (eq. 15, Andrews et al. 2017) — mandatory for stability
        H_i = alpha_over_h2 * qd.Matrix.identity(dt=gs.qd_float, n=3)
        H_i += g_ii.outer_product(g_ii) + sum_gij_gijT
        for c in qd.static(range(3)):
            H_i[c, c] += C_i * qd.sqrt(A_ii[0, c] ** 2 + A_ii[1, c] ** 2 + A_ii[2, c] ** 2) + sum_Cj_DAij[c]

        if qd.static(self._st_enabled):
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
        return dpos

    @qd.kernel
    def _kernel_compute_newton_step(self, f: qd.i32, with_star: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            # boundary particles are never updated; note their C is always 0 (never computed), so they
            # contribute to g_ii / A_ii sums but produce no neighbor-constraint terms (PLAN P4.1)
            if self.particles_ng[i_p, i_b].active and not self.particles_ng[i_p, i_b].is_boundary:
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

                # surface-tension accumulators (spec eqs. 2.11/2.12); stay exactly zero when ST is off
                f_st = qd.Vector.zero(gs.qd_float, 3)
                H_st = qd.Matrix.zero(gs.qd_float, 3, 3)
                d_st_diag = qd.Vector.zero(gs.qd_float, 3)  # distance-constraint geometric diagonal

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
                                # one-sided distance constraint (spec eqs. 2.13-2.15): fluid-fluid
                                # pairs of the SAME type (surface-surface / interior-interior) with
                                # r < d0 = ps. Folded into this loop since d0 = ps < R = 2 ps; the
                                # r >= eps_r loop guard replaces the reference's distance > gs.EPS
                                # guard (equivalent up to the degenerate-pair cutoff).
                                if qd.static(self._st_enabled and self._st_distance_enabled):
                                    if (
                                        r < self._particle_size
                                        and not self.particles_ng[j, i_b].is_boundary
                                        and self.on_surface[i_p, i_b] == self.on_surface[j, i_b]
                                    ):
                                        Cd = r - self._particle_size  # < 0 (active branch only)
                                        kd = self._st_distance_stiffness
                                        # f contribution: -kd * Cd * grad_i C^d, grad_i C^d = -r_hat
                                        f_st += kd * Cd * r_hat
                                        # Gauss-Newton term kd * grad grad^T (PSD)
                                        H_st += kd * r_hat.outer_product(r_hat)
                                        # geometric term kd * Cd * (I - r_hat r_hat^T)/r: Cd < 0 makes
                                        # it indefinite -> column-norm diagonal approximation (spec 2.8);
                                        # column c norm of (I - r r^T)/r is sqrt(1 - r_c^2)/r
                                        for c in qd.static(range(3)):
                                            d_st_diag[c] += kd * Cd * qd.sqrt(1.0 - r_hat[c] ** 2) / r

                # self term of A_ii: r -> 0 limit H_W(0) = W''(0) I
                ddW0 = V * self.cubic_kernel_ddW(0.0)
                for c in qd.static(range(3)):
                    A_ii[c, c] += ddW0

                if qd.static(self._st_enabled):
                    w_st = self._st_area_weight
                    # --- own local mesh: the j == i term of S(i) (spec section 2.5/2.6) ---
                    if self.on_surface[i_p, i_b] and self.topology_valid[i_p, i_b]:
                        grad_i = qd.Vector.zero(gs.qd_float, 3)
                        H_geo = qd.Matrix.zero(gs.qd_float, 3, 3)
                        n = self.n_ring[i_p, i_b]
                        for k in range(n):
                            k_next = 0 if k == n - 1 else k + 1
                            j1 = self.local_mesh_neighbors[i_p, i_b, k]
                            j2 = self.local_mesh_neighbors[i_p, i_b, k_next]
                            # fan triangle (p1 = i, p2 = ring[k], p3 = ring[k+1]); i is vertex 1
                            g1, H11 = self._st_tri_grad_hess_uu(
                                pos_i, self.particles[j1, i_b].pos, self.particles[j2, i_b].pos, 1
                            )
                            grad_i += g1
                            H_geo += H11
                        if qd.static(self._st_model_linear):
                            # plan (a), eq. 2.10: f = -w_st sum grad C^A, H = w_st sum H_uu
                            f_st -= w_st * grad_i
                            H_st += w_st * H_geo
                        else:
                            # plan (b), eq. 2.9: f = -k~ C^A grad C^A, H = k~ (grad grad^T + C^A H_uu)
                            CA_i = self.CA[i_p, i_b]
                            f_st -= w_st * CA_i * grad_i
                            H_st += w_st * (grad_i.outer_product(grad_i) + CA_i * H_geo)
                    # --- reverse one-ring: owners j != i whose ring contains i (spec 2.1 S(i)) ---
                    for s in range(self._rev_count[i_p, i_b]):
                        entry = self._rev_ids[i_p, i_b, s]
                        j = entry // (1 << self._ST_REV_SHIFT)
                        kk = entry % (1 << self._ST_REV_SHIFT)
                        nj = self.n_ring[j, i_b]
                        kk_next = 0 if kk == nj - 1 else kk + 1
                        kk_prev = nj - 1 if kk == 0 else kk - 1
                        pos_j = self.particles[j, i_b].pos
                        pos_k = self.particles[self.local_mesh_neighbors[j, i_b, kk], i_b].pos
                        # triangle k: (j, ring[k], ring[k+1]) with i = ring[k] as vertex 2
                        g2, H22 = self._st_tri_grad_hess_uu(
                            pos_j, pos_k, self.particles[self.local_mesh_neighbors[j, i_b, kk_next], i_b].pos, 2
                        )
                        # triangle k-1: (j, ring[k-1], ring[k]) with i = ring[k] as vertex 3
                        g3, H33 = self._st_tri_grad_hess_uu(
                            pos_j, self.particles[self.local_mesh_neighbors[j, i_b, kk_prev], i_b].pos, pos_k, 3
                        )
                        grad_ij = g2 + g3  # dC_j^A/dx_i
                        H_geo_ij = H22 + H33
                        if qd.static(self._st_model_linear):
                            f_st -= w_st * grad_ij
                            H_st += w_st * H_geo_ij
                        else:
                            CA_j = self.CA[j, i_b]
                            f_st -= w_st * CA_j * grad_ij
                            H_st += w_st * (grad_ij.outer_product(grad_ij) + CA_j * H_geo_ij)

                x_minus_y = pos_i - self.particles[i_p, i_b].y
                self.particles[i_p, i_b].dpos = self._solve_newton_direction(
                    self._alpha_eff * m_i / self._substep_dt**2,
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

                # artificial damping (paper section 3.6): during the last solver iteration, compute the
                # alternative position x* = x_beg + Delta x* / 2 from the same accumulators with the larger
                # compliance alpha* (single extra Newton solve, same relaxed half-step). x* includes the
                # surface terms (D11: same energy, only the density stiffness is lowered).
                if qd.static(self._damping_enabled):
                    if with_star == 1:
                        dpos_star = self._solve_newton_direction(
                            self._damping_alpha_star * m_i / self._substep_dt**2,
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
                        self.particles[i_p, i_b].xstar = pos_i + 0.5 * dpos_star

    @qd.kernel
    def _kernel_apply_relaxed_update(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active and not self.particles_ng[i_p, i_b].is_boundary:
                # relaxed Jacobi: simultaneous half-step update (relaxation factor fixed at 1/2)
                self.particles[i_p, i_b].pos += 0.5 * self.particles[i_p, i_b].dpos

    @qd.kernel
    def _kernel_finalize(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active and not self.particles_ng[i_p, i_b].is_boundary:
                # PBD-style velocity update: v^{t+1} = (x^{t+1} - x^t) / h
                v = (self.particles[i_p, i_b].pos - self.particles[i_p, i_b].ipos) / self._substep_dt
                # artificial damping (paper section 3.6, eqs. 16-18): compare against the alternative
                # solution x* and extract kinetic energy only when the motion is coming close to a stop
                if qd.static(self._damping_enabled):
                    dist = (self.particles[i_p, i_b].xstar - self.particles[i_p, i_b].pos).norm()
                    beta_r = self._damping_beta * self._support_radius
                    if dist < beta_r:
                        v_star = (self.particles[i_p, i_b].xstar - self.particles[i_p, i_b].ipos) / self._substep_dt
                        v2 = v.norm_sqr()
                        v_star2 = v_star.norm_sqr()
                        if v_star2 < v2 and v2 > 0.0:
                            d = 1.0 - dist / beta_r
                            v = v * qd.sqrt(1.0 - d * (v2 - v_star2) / v2)
                self.particles[i_p, i_b].vel = v

    @qd.kernel
    def _kernel_apply_viscosity(self, f: qd.i32):
        # explicit XSPH viscosity (PLAN P7, NOT part of the IPBF paper): Jacobi velocity smoothing
        #   v_i <- v_i + eps * sum_j V (v_j - v_i) W(x_i - x_j),   V = 0.8 ps^3
        # fluid-fluid pairs only (boundary particles excluded — including them would act as wall
        # friction, not requested). Pairwise anti-symmetric => linear momentum conserved. Positions
        # are final for this substep; the hash of this substep is reused.
        # pass 1: compute per-particle delta-v into the dv scratch (Jacobi double buffering)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active and not self.particles_ng[i_p, i_b].is_boundary:
                pos_i = self.particles[i_p, i_b].pos
                vel_i = self.particles[i_p, i_b].vel
                dv = qd.Vector.zero(gs.qd_float, 3)
                base = self.sh.pos_to_grid(pos_i)
                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    slot_idx = self.sh.grid_to_slot(base + offset)
                    for j_r in range(
                        self.sh.slot_start[slot_idx, i_b],
                        self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                    ):
                        j = self.particle_idx[j_r, i_b]
                        if j != i_p and not self.particles_ng[j, i_b].is_boundary:
                            r = (self.particles[j, i_b].pos - pos_i).norm()
                            if r >= self._eps_r and r < self._support_radius:
                                dv += (self.particles[j, i_b].vel - vel_i) * self.cubic_kernel_W(r)
                self.particles[i_p, i_b].dv = self._viscosity_xsph * self._particle_volume * dv
        # pass 2: apply (top-level loops in one kernel run as separate passes, cf. normals kernel)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active and not self.particles_ng[i_p, i_b].is_boundary:
                self.particles[i_p, i_b].vel += self.particles[i_p, i_b].dv

    @qd.kernel
    def _kernel_apply_surface_viscosity(self, f: qd.i32):
        # interface-restricted XSPH viscosity: same Jacobi velocity smoothing as
        # _kernel_apply_viscosity, but only surface-surface pairs contribute, so tangential
        # velocity differences along the free surface dissipate while the bulk flow keeps the
        # bulk coefficient. The on_surface flags come from the ST topology pass and may lag the
        # current positions by up to st_topo_interval substeps (the same staleness the area
        # constraint neighborhood tolerates). Fluid-fluid pairs only; pairwise anti-symmetric
        # within the surface set => linear momentum conserved.
        # pass 1: compute per-particle delta-v into the dv scratch (Jacobi double buffering)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_ng[i_p, i_b].active
                and not self.particles_ng[i_p, i_b].is_boundary
                and self.on_surface[i_p, i_b]
            ):
                pos_i = self.particles[i_p, i_b].pos
                vel_i = self.particles[i_p, i_b].vel
                dv = qd.Vector.zero(gs.qd_float, 3)
                base = self.sh.pos_to_grid(pos_i)
                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    slot_idx = self.sh.grid_to_slot(base + offset)
                    for j_r in range(
                        self.sh.slot_start[slot_idx, i_b],
                        self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                    ):
                        j = self.particle_idx[j_r, i_b]
                        if (
                            j != i_p
                            and not self.particles_ng[j, i_b].is_boundary
                            and self.on_surface[j, i_b]
                        ):
                            r = (self.particles[j, i_b].pos - pos_i).norm()
                            if r >= self._eps_r and r < self._support_radius:
                                dv += (self.particles[j, i_b].vel - vel_i) * self.cubic_kernel_W(r)
                self.particles[i_p, i_b].dv = self._surface_viscosity_xsph * self._particle_volume * dv
        # pass 2: apply (top-level loops in one kernel run as separate passes)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if (
                self.particles_ng[i_p, i_b].active
                and not self.particles_ng[i_p, i_b].is_boundary
                and self.on_surface[i_p, i_b]
            ):
                self.particles[i_p, i_b].vel += self.particles[i_p, i_b].dv

    @qd.kernel
    def _kernel_solve_diffusion(self, f: qd.i32):
        # Demo-level XSPH-style concentration diffusion (multiflow demo, NOT physical Fick
        # diffusion): two-pass relaxed Jacobi, structured exactly like _kernel_apply_viscosity.
        #   pass 1: dc_i = eps_d * sum_j V (c_j - c_i) W(x_i - x_j),  V = 0.8 ps^3
        #   pass 2: c_i += dc_i
        # fluid-fluid pairs only (boundary particles excluded — they carry no concentration).
        # Pairwise anti-symmetric => sum(c) conserved. Positions are final for this substep;
        # the hash of this substep is reused.
        # pass 1
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active and not self.particles_ng[i_p, i_b].is_boundary:
                pos_i = self.particles[i_p, i_b].pos
                c_i = self.particles[i_p, i_b].c
                dc = gs.qd_float(0.0)
                base = self.sh.pos_to_grid(pos_i)
                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    slot_idx = self.sh.grid_to_slot(base + offset)
                    for j_r in range(
                        self.sh.slot_start[slot_idx, i_b],
                        self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                    ):
                        j = self.particle_idx[j_r, i_b]
                        if j != i_p and not self.particles_ng[j, i_b].is_boundary:
                            r = (self.particles[j, i_b].pos - pos_i).norm()
                            if r >= self._eps_r and r < self._support_radius:
                                dc += (self.particles[j, i_b].c - c_i) * self.cubic_kernel_W(r)
                self.particles[i_p, i_b].dc = self._diffusion_coeff * self._particle_volume * dc
        # pass 2: apply (top-level loops in one kernel run as separate passes)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active and not self.particles_ng[i_p, i_b].is_boundary:
                self.particles[i_p, i_b].c += self.particles[i_p, i_b].dc

    @qd.kernel
    def _kernel_impose_boundary(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            # boundary clamp (CubeBoundary or CylinderBoundary) stays as a fallback for escaped
            # fluid particles; boundary particles are fixed and never clamped
            if self.particles_ng[i_p, i_b].active and not self.particles_ng[i_p, i_b].is_boundary:
                # pre-declare before the branches: qd's AST resolver cannot see names that are
                # only assigned inside if/else branches
                corrected_pos = self.particles[i_p, i_b].pos
                corrected_vel = self.particles[i_p, i_b].vel
                gid = self.particles_ng[i_p, i_b].boundary_group
                if qd.static(self._has_boundary_pitcher):
                    # clamp ownership dispatch (multiflow MF-11): group-1 particles are clamped by
                    # the pitcher ONLY (the primary cup clamp never touches them — MF-10 bug: the
                    # global cup clamp froze the pitcher water into wall-hugging chunks). A group-1
                    # particle outside the pitcher's keep region has poured out: it permanently
                    # transfers to the primary boundary's group.
                    if gid == 1 and not self.boundary2.in_keep_region(
                        self.particles[i_p, i_b].pos, self._particle_size
                    ):
                        gid = 0
                        self.particles_ng[i_p, i_b].boundary_group = 0
                    if gid == 1:
                        corrected_pos, corrected_vel = self.boundary2.impose_pos_vel(
                            corrected_pos, corrected_vel
                        )
                    else:
                        corrected_pos, corrected_vel = self.boundary.impose_pos_vel(
                            corrected_pos, corrected_vel
                        )
                else:
                    corrected_pos, corrected_vel = self.boundary.impose_pos_vel(
                        corrected_pos, corrected_vel
                    )
                # table plane (multiflow MF-13 pattern): applies to every particle regardless of
                # group, after the container clamp that owns it
                if qd.static(self._has_boundary_plane):
                    corrected_pos, corrected_vel = self.boundary3.impose_pos_vel(
                        corrected_pos, corrected_vel
                    )
                self.particles[i_p, i_b].pos = corrected_pos
                self.particles[i_p, i_b].vel = corrected_vel

    def set_pitcher_pose(self, origin, axis):
        """Animate the pitcher clamp (multiflow MF-11): per-frame pose update of boundary2."""
        if self._has_boundary_pitcher:
            self.boundary2.set_pose(origin, axis)

    def substep_pre_coupling(self, f):
        if self.is_active:
            # Algorithm 1: predict (x <- y) -> hash rebuild -> relaxed-Jacobi Newton iterations -> finalize
            self._kernel_predict(f)
            self._kernel_build_hash(f)
            if self._st_enabled:
                # spec section 3.4: topology (surface detect -> normals -> local meshes) is rebuilt
                # once every st_topo_interval substeps, OUTSIDE the Newton loop; the area constraint
                # values/gradients are still recomputed at current positions inside the loop
                if self._st_topo_counter % self._st_topo_interval == 0:
                    self._st_rebuild_topology(f)
                self._st_topo_counter += 1
            for it in range(self._ipbf_iterations):
                if self._rebuild_hash_per_iter:
                    # non-paper experiment: re-search neighbors at the current guess before
                    # recomputing density (the reorder only rewrites index maps; particle
                    # fields are not moved, so the Jacobi buffers stay consistent)
                    self._kernel_build_hash(f)
                self._kernel_compute_density_C(f)
                if self._st_enabled:
                    self._kernel_st_compute_area_C(f)
                # during the last iteration, also compute the alternative solution x* (artificial damping)
                with_star = 1 if it == self._ipbf_iterations - 1 else 0
                self._kernel_compute_newton_step(f, with_star)
                self._kernel_apply_relaxed_update(f)
            self._kernel_finalize(f)
            # explicit XSPH viscosity (PLAN P7): velocity smoothing after the velocity update,
            # before the boundary clamp; skipped entirely when the coefficient is 0 (zero regression)
            if self._viscosity_xsph > 0.0:
                self._kernel_apply_viscosity(f)
            # interface-restricted variant runs after the bulk pass (it overwrites the dv scratch
            # for surface particles); skipped entirely when the coefficient is 0 (zero regression)
            if self._surface_viscosity_xsph > 0.0:
                self._kernel_apply_surface_viscosity(f)
            # demo-level concentration diffusion (multiflow): right after viscosity; skipped
            # entirely when the coefficient is 0 (zero regression)
            if self._diffusion_coeff > 0.0:
                self._kernel_solve_diffusion(f)

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
        # NOTE: concentration `c` is intentionally NOT part of the state snapshot (pos/vel/active
        # only), same as the PBD side; it survives scene.reset() unchanged.
        if self.is_active:
            assert state.pos.shape[1] == self._n_particles, (
                f"state size mismatch: {state.pos.shape[1]} != {self._n_particles}"
            )
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
            assert state.pos.shape[1] == self._n_particles, (
                f"state size mismatch: {state.pos.shape[1]} != {self._n_particles}"
            )
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
            # boundary particles are never rendered
            if self.particles_ng[i_p, i_b].active and not self.particles_ng[i_p, i_b].is_boundary:
                self.particles_render[i_p, i_b].pos = self.particles[i_p, i_b].pos
                self.particles_render[i_p, i_b].vel = self.particles[i_p, i_b].vel
                self.particles_render[i_p, i_b].c = self.particles[i_p, i_b].c
                self.particles_render[i_p, i_b].active = True
            else:
                self.particles_render[i_p, i_b].pos = gu.qd_nowhere()
                self.particles_render[i_p, i_b].active = False

    @qd.kernel
    def _kernel_add_particles(
        self,
        f: qd.i32,
        active: qd.i32,
        particle_start: qd.i32,
        n_particles: qd.i32,
        mat_rho: qd.f32,
        mat_c_init: qd.f32,
        mat_boundary_group: qd.i32,
        pos: qd.types.ndarray(),
    ):
        for i_p_, i_b in qd.ndrange(n_particles, self._B):
            i_p = i_p_ + particle_start
            self.particles_ng[i_p, i_b].active = qd.cast(active, gs.qd_bool)
            # clamp ownership (multiflow MF-11): 0 = primary boundary, 1 = pitcher
            self.particles_ng[i_p, i_b].boundary_group = mat_boundary_group
            for i in qd.static(range(3)):
                self.particles[i_p, i_b].pos[i] = pos[i_p_, i]
            self.particles[i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
            # concentration init (multiflow demo): per-entity constant from the material
            self.particles[i_p, i_b].c = mat_c_init
            self.particles[i_p, i_b].dc = 0.0

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
        # NOTE: after build, `_n_particles` includes the static boundary particles, which do not belong
        # to any entity. Prefer it whenever it exists so state arrays are sized correctly even when
        # captured while `scene._is_built` is still False (e.g. Scene._init_state during build).
        # `add_entity` queries this property before build (attribute absent) and gets the entity sum.
        n = getattr(self, "_n_particles", None)
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
