import math

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.boundaries import CubeBoundary
from genesis.engine.entities import DEMEntity
from genesis.utils.geom import SpatialHasher, qd_transform_by_quat_fast

from .base_solver import Solver

# capacity of the mesh-SDF obstacle's grid (allocated at build; ~33.5M cells = 134 MB f32)
SDFO_MAX_CELLS = 2**25


@qd.data_oriented
class DEMSolver(Solver):
    """
    DEM (Discrete Element Method) solver for granular materials (sand), ported from the reference
    implementation `sand-water-coupling-PIC-DEM-3d`. All formulas strictly follow the C++ code:

    - Base sub-substep: `ddt = radius * pi * sqrt(density / young_modulus) / 2`, optionally scaled by
      `DEMOptions.ddt_safety`. The reference's velocity-adaptive clamp `2 * radius / max_vel` is replaced
      by a fixed step count per simulator substep (user decision: avoids a global max-velocity reduction
      and GPU readback per sub-substep).
    - Contact force on grain i from grain j (penetration `pen = 2 * radius - |dij|`):
      normal `K_norm * pen * n` with `K_norm = young * radius`;
      shear `-K_tang * v_tangential` with `K_tang = K_norm * poisson`,
      clamped by the Coulomb limit `|f_shear| <= |f_normal| * tan(friction_angle)`.
    - Per sub-substep order: assemble acceleration (gravity + contact / mass), enforce the box collider,
      then `v += a * ddt`, `x += v * ddt`.
    - Box collider (static walls at the domain bounds, signed distance `phi` positive inside):
      trigger zone `phi < 0.5 * radius`; projection `x -= n * phi` and velocity reflection
      `v -= 1.5 * (v . n) * n` when `phi < 0`; wall friction `a += (a . n) * tan(friction_angle) * t_hat`
      on the acceleration when `a . n < 0`, where `t_hat` is the normalized tangential velocity.
    - Moving obstacles (optional: axis-aligned box, sieve, tilted box): same enforcement as the domain
      collider, but with the particle velocity replaced by the velocity relative to the obstacle
      (`delta_vel` in the C++ code), so a moving obstacle drags/scoops grains via the restitution and
      friction terms. The tilted box also carries an angular velocity about its center; its SDF is
      evaluated by inverse-rotating the query point into the box frame, and the obstacle velocity at a
      contact point is `v_center + omega x (pos - center)`.

    The capillary (liquid-bridge) cohesion of the reference is identically zero for dry sand
    (its coefficient vanishes at zero saturation) and is therefore omitted until the wet-sand coupling phase.
    """

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        self._particle_size = options.particle_size
        self._particle_radius = options.particle_size / 2.0
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)

        # spatial hasher for neighbor search (cell >= 1.25 * particle_size = 2.5 * radius > sqrt(6) * radius)
        self.sh = SpatialHasher(
            cell_size=options.hash_grid_cell_size,
            grid_res=options._hash_grid_res,
        )

        # used for the entity sampling bounds check only; contact handling follows the C++ box collider above
        self.boundary = CubeBoundary(
            lower=self._lower_bound,
            upper=self._upper_bound,
        )

        # base DEM sub-substep, computed at build time from the entity materials
        self._m_ddt = None
        self._ddt_safety = options.ddt_safety

        # optional hollow-cylinder collider (axis along z through the domain center), C++ rotate scene
        self._cylinder_radius = options.cylinder_radius
        self._has_cylinder = options.cylinder_radius is not None
        self._center_x = 0.5 * float(self._lower_bound[0] + self._upper_bound[0])
        self._center_y = 0.5 * float(self._lower_bound[1] + self._upper_bound[1])

    def build(self):
        super().build()

        self._B = self._sim._B
        self._n_particles = self.n_particles

        if self.is_active:
            self.sh.build(self._B)
            self.init_particle_fields()
            for entity in self._entities:
                entity._add_to_solver()

            self._m_ddt = min(
                self._particle_radius
                * math.pi
                * math.sqrt(entity.material.rho / entity.material.young_modulus)
                / 2.0
                for entity in self._entities
            )
            self._particle_volume = 4.0 / 3.0 * math.pi * self._particle_radius**3

        # coupled mode: the FLIP solver drives the DEM advance per water step (C++ Advance order)
        self._coupled = (
            self.is_active
            and self._sim.flip_options.dem_coupling
            and self._sim.flip_solver.is_active
        )

        # FIXME: _gravity must be a raw qd.field() -- see comment in mpm_solver.py
        if self._gravity is not None:
            gravity = self._gravity.to_numpy()
            self._gravity = qd.field(dtype=gs.qd_vec3, shape=(self._B,))
            self._gravity.from_numpy(gravity)

    def init_particle_fields(self):
        # static info
        struct_particle_info = qd.types.struct(
            mass=gs.qd_float,
            radius=gs.qd_float,
        )
        # dynamic state; `acc` is the C++ `AccVelocity`, assembling gravity and contact forces each sub-substep.
        # The wet-sand fields follow C++ DEMParticle (Particle.h); they stay zero in dry-sand mode.
        struct_particle_state = qd.types.struct(
            pos=gs.qd_vec3,
            vel=gs.qd_vec3,
            acc=gs.qd_vec3,
            active=gs.qd_bool,
            ratio=gs.qd_float,
            saturate_rate=gs.qd_float,
            add_ratio=gs.qd_float,
            add_velocity=gs.qd_vec3,
            coupling_force=gs.qd_vec3,
            inv_mass_eff=gs.qd_float,
        )
        # non-gradient state
        struct_particle_state_ng = qd.types.struct(
            reordered_idx=gs.qd_int,
        )
        # render state
        struct_particle_state_render = qd.types.struct(
            pos=gs.qd_vec3,
            active=gs.qd_bool,
        )

        self.particles_info = struct_particle_info.field(shape=(self._n_particles,), layout=qd.Layout.SOA)
        self.particles = struct_particle_state.field(shape=(self._n_particles, self._B), layout=qd.Layout.SOA)
        self.particles_reordered = struct_particle_state.field(shape=(self._n_particles, self._B), layout=qd.Layout.SOA)
        self.particles_ng = struct_particle_state_ng.field(shape=(self._n_particles, self._B), layout=qd.Layout.SOA)
        self.particles_render = struct_particle_state_render.field(
            shape=(self._n_particles, self._B), layout=qd.Layout.SOA
        )

        # optional moving box obstacle (disabled by default); see set_box_obstacle
        self.obstacle_pos = qd.field(gs.qd_vec3, shape=(self._B,))
        self.obstacle_vel = qd.field(gs.qd_vec3, shape=(self._B,))
        self.obstacle_half = qd.field(gs.qd_vec3, shape=(self._B,))
        self.obstacle_enabled = qd.field(gs.qd_int, shape=(self._B,))

        # optional moving sieve (grid of bars) obstacle; see set_sieve_obstacle
        self.sieve_pos = qd.field(gs.qd_vec3, shape=(self._B,))
        self.sieve_vel = qd.field(gs.qd_vec3, shape=(self._B,))
        self.sieve_params = qd.field(gs.qd_vec4, shape=(self._B,))  # half_extent, bar_width, bar_period, thickness
        self.sieve_enabled = qd.field(gs.qd_int, shape=(self._B,))

        # optional moving tilted box obstacle (rotated thin box, e.g. a shovel); see set_tilt_box_obstacle
        self.tilt_pos = qd.field(gs.qd_vec3, shape=(self._B,))
        self.tilt_vel = qd.field(gs.qd_vec3, shape=(self._B,))
        self.tilt_omega = qd.field(gs.qd_vec3, shape=(self._B,))  # angular velocity about the box center
        self.tilt_quat = qd.field(gs.qd_vec4, shape=(self._B,))  # (w, x, y, z)
        self.tilt_half = qd.field(gs.qd_vec3, shape=(self._B,))
        # optional handle attached at the box's local -x face: (length, half_thickness, angle), angle in the
        # local xz-plane measured from the box face (0 = straight back along -x); length 0 disables it
        self.tilt_handle = qd.field(gs.qd_vec3, shape=(self._B,))
        self.tilt_enabled = qd.field(gs.qd_int, shape=(self._B,))

        # optional moving mesh-SDF obstacle (a voxelized real asset, e.g. a litter scoop); see
        # set_sdf_obstacle. Same kinematics as the tilt box; the SDF grid is shared across envs and
        # expressed in the obstacle local frame (origin = grid min corner, numpy C-order (nx, ny, nz) flattening).
        # sdfo_val is allocated at build with a fixed capacity: fields created after build are NOT
        # visible to already-materialized kernels (a lazy replacement read garbage -> explosions).
        self.sdfo_pos = qd.field(gs.qd_vec3, shape=(self._B,))
        self.sdfo_vel = qd.field(gs.qd_vec3, shape=(self._B,))
        self.sdfo_omega = qd.field(gs.qd_vec3, shape=(self._B,))
        self.sdfo_quat = qd.field(gs.qd_vec4, shape=(self._B,))  # (w, x, y, z)
        self.sdfo_enabled = qd.field(gs.qd_int, shape=(self._B,))
        self.sdfo_val = qd.field(gs.qd_float, shape=(SDFO_MAX_CELLS,))
        self.sdfo_dims = qd.field(gs.qd_int, shape=(3,))  # (nx, ny, nz)
        self.sdfo_origin = qd.field(gs.qd_vec3, shape=(self._B,))  # grid min corner (local frame)
        self.sdfo_cell = qd.field(gs.qd_vec3, shape=(self._B,))  # cell size (local frame)
        self.sdfo_inv_cell = qd.field(gs.qd_vec3, shape=(self._B,))
        self.sdfo_extent = qd.field(gs.qd_vec3, shape=(self._B,))  # dims * cell (local AABB size)

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_entity(self, idx, material, morph, surface, name=None):
        entity = DEMEntity(
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
        self._entities.append(entity)
        return entity

    # ------------------------------------------------------------------------------------
    # --------------------------------- moving obstacle ----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def set_box_obstacle(self, half_extents, pos, vel=(0.0, 0.0, 0.0)):
        """
        Place (or replace) the moving box obstacle and enable it.

        The obstacle position is advanced internally by `vel * ddt` at every DEM sub-substep, so the
        obstacle motion is continuous at the contact timescale; per-frame teleportation would kick the
        resting grains. Use `set_box_obstacle_vel` to change the velocity during the simulation.

        Parameters
        ----------
        half_extents : tuple, shape (3,)
            Half extents of the box in meters.
        pos : tuple, shape (3,)
            Center of the box in meters.
        vel : tuple, shape (3,), optional
            Velocity of the box in m/s. Defaults to zero.
        """
        self.obstacle_half.from_numpy(np.tile(np.asarray(half_extents, dtype=gs.np_float), (self._B, 1)))
        self.obstacle_enabled.from_numpy(np.ones((self._B,), dtype=gs.np_int))
        self.obstacle_pos.from_numpy(np.tile(np.asarray(pos, dtype=gs.np_float), (self._B, 1)))
        self.obstacle_vel.from_numpy(np.tile(np.asarray(vel, dtype=gs.np_float), (self._B, 1)))

    @gs.assert_built
    def set_box_obstacle_vel(self, vel):
        """
        Set the velocity of the moving box obstacle. Its position is advanced internally every DEM sub-substep.
        """
        self.obstacle_vel.from_numpy(np.tile(np.asarray(vel, dtype=gs.np_float), (self._B, 1)))

    @gs.assert_built
    def get_box_obstacle_pos(self):
        """
        Current center position of the moving box obstacle (per env, shape (n_envs, 3)).
        """
        return self.obstacle_pos.to_numpy()

    @gs.assert_built
    def set_sieve_obstacle(self, half_extent, bar_width, bar_period, thickness, pos, vel=(0.0, 0.0, 0.0)):
        """
        Place a square mesh sieve (two perpendicular families of parallel bars) and enable it.

        The sieve plane is horizontal (xy) and moves like the box obstacle: its position advances
        internally by vel * ddt at every DEM sub-substep.

        Parameters
        ----------
        half_extent : float
            Half side length of the square sieve in meters.
        bar_width : float
            Rod diameter in meters.
        bar_period : float
            Bar center-to-center period in meters (opening = period - bar_width).
        thickness : float
            Unused for round rods (kept for API compatibility).
        pos : tuple, shape (3,)
            Center of the sieve plane in meters.
        vel : tuple, shape (3,), optional
            Velocity of the sieve in m/s. Defaults to zero.
        """
        self.sieve_params.from_numpy(
            np.tile(
                np.array([half_extent, bar_width, bar_period, thickness], dtype=gs.np_float), (self._B, 1)
            )
        )
        self.sieve_enabled.from_numpy(np.ones((self._B,), dtype=gs.np_int))
        self.sieve_pos.from_numpy(np.tile(np.asarray(pos, dtype=gs.np_float), (self._B, 1)))
        self.sieve_vel.from_numpy(np.tile(np.asarray(vel, dtype=gs.np_float), (self._B, 1)))
        self._kernel_init_delete_sieve()

    @qd.kernel
    def _kernel_init_delete_sieve(self):
        # C++ Collider::InitDelete: grains initialized inside the sieve bars are removed
        # (otherwise projecting them out at the first step creates deep overlaps and an explosion)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                if self._func_sieve_sdf(self.particles[i_p, i_b].pos, i_b) < 0.0:
                    self.particles[i_p, i_b].active = False

    @gs.assert_built
    def set_sieve_vel(self, vel):
        """
        Set the sieve velocity. Its position advances internally every DEM sub-substep.
        """
        self.sieve_vel.from_numpy(np.tile(np.asarray(vel, dtype=gs.np_float), (self._B, 1)))

    @gs.assert_built
    def get_sieve_pos(self):
        """
        Current center position of the sieve (per env, shape (n_envs, 3)).
        """
        return self.sieve_pos.to_numpy()

    @gs.assert_built
    def set_tilt_box_obstacle(self, half_extents, pos, quat=(1.0, 0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), handle=(0.0, 0.0, 0.0)):
        """
        Place (or replace) a rotated box obstacle (e.g. an inclined shovel blade) and enable it.

        Same motion model as the box/sieve obstacles: the position is advanced internally by `vel * ddt` at
        every DEM sub-substep; only the velocity setpoint is scripted (see set_box_obstacle). The orientation
        advances likewise from the angular velocity setpoint (see set_tilt_box_vel). Grains overlapping the
        box at placement are removed (C++ Collider::InitDelete), so the box may be placed inside the sand bed.

        Parameters
        ----------
        half_extents : tuple, shape (3,)
            Half extents of the box in its local frame in meters.
        pos : tuple, shape (3,)
            Center of the box in meters.
        quat : tuple, shape (4,), optional
            Orientation of the box as a quaternion (w, x, y, z); normalized internally. Defaults to identity.
        vel : tuple, shape (3,), optional
            Velocity of the box in m/s. Defaults to zero.
        handle : tuple, shape (3,), optional
            Optional handle bar (a shovel grip): (length, half_thickness, angle) in meters / radians. The bar
            is attached at the box's local -x face and points up-back in the local xz-plane; angle is measured
            from the box face (0 = straight back along -x). The default zero length disables the handle.
        """
        quat_np = np.asarray(quat, dtype=gs.np_float)
        quat_np = quat_np / np.linalg.norm(quat_np)
        self.tilt_half.from_numpy(np.tile(np.asarray(half_extents, dtype=gs.np_float), (self._B, 1)))
        self.tilt_quat.from_numpy(np.tile(quat_np, (self._B, 1)))
        self.tilt_handle.from_numpy(np.tile(np.asarray(handle, dtype=gs.np_float), (self._B, 1)))
        self.tilt_enabled.from_numpy(np.ones((self._B,), dtype=gs.np_int))
        self.tilt_pos.from_numpy(np.tile(np.asarray(pos, dtype=gs.np_float), (self._B, 1)))
        self.tilt_vel.from_numpy(np.tile(np.asarray(vel, dtype=gs.np_float), (self._B, 1)))
        self.tilt_omega.from_numpy(np.zeros((self._B, 3), dtype=gs.np_float))
        self._kernel_init_delete_tilt_box()

    @qd.kernel
    def _kernel_init_delete_tilt_box(self):
        # C++ Collider::InitDelete, same as _kernel_init_delete_sieve
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                if self._func_tilt_box_sdf(self.particles[i_p, i_b].pos, i_b) < 0.0:
                    self.particles[i_p, i_b].active = False

    @gs.assert_built
    def set_tilt_box_vel(self, vel, omega=(0.0, 0.0, 0.0)):
        """
        Set the velocity of the tilted box obstacle. Its position and orientation are advanced internally
        every DEM sub-substep.

        Parameters
        ----------
        vel : tuple, shape (3,)
            Velocity of the box center in m/s.
        omega : tuple, shape (3,), optional
            Angular velocity in rad/s (world frame); the box rotates about its own center. Defaults to zero.
        """
        self.tilt_vel.from_numpy(np.tile(np.asarray(vel, dtype=gs.np_float), (self._B, 1)))
        self.tilt_omega.from_numpy(np.tile(np.asarray(omega, dtype=gs.np_float), (self._B, 1)))

    @gs.assert_built
    def get_tilt_box_pos(self):
        """
        Current center position of the tilted box obstacle (per env, shape (n_envs, 3)).
        """
        return self.tilt_pos.to_numpy()

    @gs.assert_built
    def get_tilt_box_quat(self):
        """
        Current orientation of the tilted box obstacle as a quaternion (w, x, y, z) (per env, shape (n_envs, 4)).
        """
        return self.tilt_quat.to_numpy()

    @gs.assert_built
    def set_sdf_obstacle(self, sdf_val, dims, origin, cell_size, pos, quat=(1.0, 0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0)):
        """
        Place (or replace) a mesh-SDF obstacle (a voxelized real asset) and enable it.

        Same motion model as the tilt box obstacle: the position/orientation are advanced internally from
        the velocity setpoints at every DEM sub-substep (see set_sdf_obstacle_vel). Grains overlapping the
        obstacle at placement are removed (C++ Collider::InitDelete).

        Parameters
        ----------
        sdf_val : ndarray, shape (nx, ny, nz)
            Signed distance values of the mesh in its local frame (negative inside), numpy C-order (nx, ny, nz).
        dims : tuple, shape (3,)
            Grid dimensions (nx, ny, nz).
        origin : tuple, shape (3,)
            Grid min corner in the obstacle local frame in meters.
        cell_size : tuple, shape (3,)
            Grid cell size in meters.
        pos : tuple, shape (3,)
            Local-frame origin's world position in meters.
        quat : tuple, shape (4,), optional
            Orientation of the local frame as a quaternion (w, x, y, z); normalized internally.
        vel : tuple, shape (3,), optional
            Velocity of the obstacle in m/s. Defaults to zero.
        """
        val = np.asarray(sdf_val, dtype=gs.np_float).reshape(-1)
        dims_np = np.asarray(dims, dtype=gs.np_int)
        assert val.size == int(np.prod(dims_np)), f"sdf_val size {val.size} != dims {dims_np}"
        assert val.size <= SDFO_MAX_CELLS, f"sdf_val size {val.size} exceeds capacity {SDFO_MAX_CELLS}"
        # write into the prefix of the build-time-allocated field (post-build field replacement is
        # NOT visible to already-materialized kernels)
        buf = np.zeros((SDFO_MAX_CELLS,), dtype=gs.np_float)
        buf[: val.size] = val
        self.sdfo_val.from_numpy(buf)
        self.sdfo_dims.from_numpy(dims_np)
        cell_np = np.asarray(cell_size, dtype=gs.np_float)
        origin_np = np.asarray(origin, dtype=gs.np_float)
        self.sdfo_cell.from_numpy(np.tile(cell_np, (self._B, 1)))
        self.sdfo_inv_cell.from_numpy(np.tile(1.0 / cell_np, (self._B, 1)))
        self.sdfo_origin.from_numpy(np.tile(origin_np, (self._B, 1)))
        self.sdfo_extent.from_numpy(np.tile(cell_np * dims_np.astype(gs.np_float), (self._B, 1)))
        quat_np = np.asarray(quat, dtype=gs.np_float)
        quat_np = quat_np / np.linalg.norm(quat_np)
        self.sdfo_quat.from_numpy(np.tile(quat_np, (self._B, 1)))
        self.sdfo_pos.from_numpy(np.tile(np.asarray(pos, dtype=gs.np_float), (self._B, 1)))
        self.sdfo_vel.from_numpy(np.tile(np.asarray(vel, dtype=gs.np_float), (self._B, 1)))
        self.sdfo_omega.from_numpy(np.zeros((self._B, 3), dtype=gs.np_float))
        self.sdfo_enabled.from_numpy(np.ones((self._B,), dtype=gs.np_int))
        self._kernel_init_delete_sdfo()

    @qd.kernel
    def _kernel_init_delete_sdfo(self):
        # C++ Collider::InitDelete, same as _kernel_init_delete_tilt_box
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                if self._func_sdfo_sdf(self.particles[i_p, i_b].pos, i_b) < 0.0:
                    self.particles[i_p, i_b].active = False

    @gs.assert_built
    def set_sdf_obstacle_vel(self, vel, omega=(0.0, 0.0, 0.0)):
        """
        Set the velocity of the mesh-SDF obstacle. Its position and orientation are advanced internally
        every DEM sub-substep. `omega` is in the world frame (rotation about the obstacle center).
        """
        self.sdfo_vel.from_numpy(np.tile(np.asarray(vel, dtype=gs.np_float), (self._B, 1)))
        self.sdfo_omega.from_numpy(np.tile(np.asarray(omega, dtype=gs.np_float), (self._B, 1)))

    @gs.assert_built
    def get_sdf_obstacle_pos(self):
        """
        Current world position of the mesh-SDF obstacle's local-frame origin (per env, shape (n_envs, 3)).
        """
        return self.sdfo_pos.to_numpy()

    @gs.assert_built
    def get_sdf_obstacle_quat(self):
        """
        Current orientation of the mesh-SDF obstacle as a quaternion (w, x, y, z) (per env, shape (n_envs, 4)).
        """
        return self.sdfo_quat.to_numpy()

    @gs.assert_built
    def query_sdf_obstacle(self, points):
        """
        Debug/verification helper: evaluate the mesh-SDF obstacle's signed distance at the given world
        points. Returns an array of shape (n_points,) with NaN where the obstacle is disabled.
        """
        pts = np.asarray(points, dtype=gs.np_float).reshape(-1, 3)
        out = np.full(len(pts), np.nan, dtype=gs.np_float)
        self._kernel_query_sdfo(pts, out)
        return out

    @qd.kernel
    def _kernel_query_sdfo(self, pts: qd.types.ndarray(), out: qd.types.ndarray()):
        for i in range(pts.shape[0]):
            p = qd.Vector([pts[i, 0], pts[i, 1], pts[i, 2]])
            if self.sdfo_enabled[0] == 1:
                out[i] = self._func_sdfo_sdf(p, 0)

    @qd.kernel
    def _kernel_query_sdfo_debug(self, pts: qd.types.ndarray(), out: qd.types.ndarray()):
        # debug: per point, dump [phi, gx, gy, gz, x0, y0, z0, corner000]
        for i in range(pts.shape[0]):
            p = qd.Vector([pts[i, 0], pts[i, 1], pts[i, 2]])
            rel = p - self.sdfo_pos[0]
            q = self.sdfo_quat[0]
            q_conj = qd.Vector([q[0], -q[1], -q[2], -q[3]])
            pl = qd_transform_by_quat_fast(rel, q_conj)
            g = qd.Vector.zero(gs.qd_float, 3)
            for a in qd.static(range(3)):
                ga = (pl[a] - self.sdfo_origin[0][a]) * self.sdfo_inv_cell[0][a] - 0.5
                gmax = gs.qd_float(self.sdfo_dims[a] - 1) - 1e-4
                g[a] = min(max(ga, 0.0), gmax)
            x0 = int(qd.floor(g[0]))
            y0 = int(qd.floor(g[1]))
            z0 = int(qd.floor(g[2]))
            out[i, 0] = self._func_sdfo_sdf(p, 0)
            out[i, 1] = g[0]
            out[i, 2] = g[1]
            out[i, 3] = g[2]
            out[i, 4] = gs.qd_float(x0)
            out[i, 5] = gs.qd_float(y0)
            out[i, 6] = gs.qd_float(z0)
            out[i, 7] = self.sdfo_val[x0 * (self.sdfo_dims[1] * self.sdfo_dims[2]) + y0 * self.sdfo_dims[2] + z0]
            out[i, 8] = self.sdfo_val[0]
            out[i, 9] = gs.qd_float(self.sdfo_dims[0])
            out[i, 10] = self.sdfo_origin[0][0]
            out[i, 11] = self.sdfo_inv_cell[0][0]

    def query_sdf_obstacle_debug(self, points):
        pts = np.asarray(points, dtype=gs.np_float).reshape(-1, 3)
        out = np.zeros((len(pts), 12), dtype=gs.np_float)
        self._kernel_query_sdfo_debug(pts, out)
        return out

    # ------------------------------------------------------------------------------------
    # ------------------------------------- kernels --------------------------------------
    # ------------------------------------------------------------------------------------

    @qd.kernel
    def _kernel_add_particles(
        self,
        particle_start: qd.i32,
        n_particles: qd.i32,
        positions: qd.types.ndarray(element_dim=1),
        radius: qd.f32,
        density: qd.f32,
    ):
        volume = (4.0 / 3.0) * math.pi * radius * radius * radius
        mass = density * volume
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            self.particles_info[i_p].mass = mass
            self.particles_info[i_p].radius = radius
            for i_b in range(self._B):
                self.particles[i_p, i_b].pos = positions[i_p_]
                self.particles[i_p, i_b].vel = qd.Vector.zero(gs.qd_float, 3)
                self.particles[i_p, i_b].acc = qd.Vector.zero(gs.qd_float, 3)
                self.particles[i_p, i_b].active = True
                self.particles[i_p, i_b].ratio = 0.0
                self.particles[i_p, i_b].saturate_rate = 0.0
                self.particles[i_p, i_b].add_ratio = 0.0
                self.particles[i_p, i_b].add_velocity = qd.Vector.zero(gs.qd_float, 3)
                self.particles[i_p, i_b].coupling_force = qd.Vector.zero(gs.qd_float, 3)
                self.particles[i_p, i_b].inv_mass_eff = 1.0 / mass

    @qd.kernel
    def _kernel_reset_acc(self, f: qd.i32):
        # C++ ApplyDEMForces: AccVelocity = gravity (coupling and absorbed-water terms are zero for dry sand)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                self.particles[i_p, i_b].acc = self._gravity[i_b]

    @qd.kernel
    def _kernel_reorder_particles(self, f: qd.i32):
        self.sh.compute_reordered_idx(
            self._n_particles, self.particles.pos, self.particles.active, self.particles_ng.reordered_idx
        )

        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                reordered_idx = self.particles_ng[i_p, i_b].reordered_idx
                self.particles_reordered[reordered_idx, i_b] = self.particles[i_p, i_b]

    @qd.func
    def _func_bezier_G(self, sr: qd.f32):
        # C++ QuadraticBezierCoeff with (py0, py1, px1, py2) = (c0=0, cmc=1, cmcp=0.1, csat=-1.5)
        py0, py1, px1, py2 = 0.0, 1.0, 0.1, -1.5
        a = (px1 + 1.0) / 4.0
        b = -2.0 * a
        c = b * b
        d = -4.0 * (1.0 + b - px1)
        e = 2.0 * (1.0 + b - px1)
        res = gs.qd_float(0.0)
        if sr < 0.0:
            res = py0
        elif sr >= 1.0:
            res = py2
        elif sr <= px1:
            t = sr / px1
            omt = 1.0 - t
            # C++ keeps t*t*py1 here (not py2); preserved as-is, recorded in the project log
            res = omt * omt * py0 + 2.0 * t * omt * py1 + t * t * py1
        else:
            t = (b + qd.sqrt(c + d * (px1 - sr))) / e
            omt = 1.0 - t
            res = omt * omt * py1 + 2.0 * t * omt * py1 + t * t * py2
        return res

    @qd.func
    def _func_capillary_force(self, n, dist: qd.f32, sr: qd.f32, particle_radius: qd.f32):
        # C++ DEMForce::ComputeDemCapillaryForces (attractive, along -n), active for 0 < H < d_rupture.
        # contact_angle is uninitialized in the reference (UB); we use its commented-out reference
        # value of 30 degrees (recorded in the project log).
        contact_angle = 30.0 / 180.0 * 3.141592653589793
        r = particle_radius
        Vb = 4.0 / 3.0 * 3.141592653589793 * r * r * r * 1e-4
        d_rupture = (1.0 + 0.5 * contact_angle) * (Vb ** (1.0 / 3.0) + 0.1 * Vb ** (2.0 / 3.0))
        surface_tensor_cof = 0.007

        f = qd.Vector.zero(gs.qd_float, 3)
        H = dist - 2.0 * r
        if H < d_rupture and H > 0.0:
            coeff_c = qd.max(0.0, self._func_bezier_G(sr) * surface_tensor_cof)
            d = -H + qd.sqrt(H * H + Vb / (3.141592653589793 * r))
            phi = qd.sqrt(
                2.0 * H / r * (-1.0 + qd.sqrt(1.0 + Vb / (3.141592653589793 * r * H * H)))
            )
            neck = -2.0 * 3.141592653589793 * coeff_c * r * qd.cos(contact_angle) / (1.0 + H / (2.0 * d))
            st = -2.0 * 3.141592653589793 * coeff_c * r * phi * qd.sin(contact_angle)
            f = -n * (neck + st)
        return f

    @qd.func
    def _func_contact_force(
        self,
        i: qd.i32,
        j: qd.i32,
        i_b: qd.i32,
        particle_radius: qd.f32,
        k_norm: qd.f32,
        k_tang: qd.f32,
        tan_fric: qd.f32,
    ):
        # C++ DEMForce::getForce(pi, pj), force on particle i; strictly elastic, no damping
        pos_i = self.particles_reordered[i, i_b].pos
        pos_j = self.particles_reordered[j, i_b].pos
        dij = pos_j - pos_i
        dist2 = dij.dot(dij)
        if dist2 < 6.0 * particle_radius * particle_radius:
            dist = qd.sqrt(dist2)
            if dist >= 0.001 * particle_radius:
                n = dij / dist
                f = qd.Vector.zero(gs.qd_float, 3)
                penetration = 2.0 * particle_radius - dist
                if penetration > 0.0:
                    vij = self.particles_reordered[j, i_b].vel - self.particles_reordered[i, i_b].vel
                    vij_tangential = vij - vij.dot(n) * n

                    f_normal = k_norm * penetration * n
                    f_shear = -k_tang * vij_tangential

                    max_fs = k_norm * penetration * tan_fric
                    fs_norm = f_shear.norm()
                    if fs_norm > max_fs:
                        f_shear *= max_fs / fs_norm

                    f = -f_normal - f_shear
                # C++ getForce adds the capillary (liquid-bridge) force for separated pairs (0 < H < d_rupture)
                sr = (
                    self.particles_reordered[i, i_b].saturate_rate
                    + self.particles_reordered[j, i_b].saturate_rate
                ) * 0.5
                f += self._func_capillary_force(n, dist, sr, particle_radius)
                # C++ divides by the wet mass (m + ratio * V); inv_mass_eff == 1/m in dry-sand mode
                self.particles_reordered[i, i_b].acc = (
                    self.particles_reordered[i, i_b].acc + f * self.particles_reordered[i, i_b].inv_mass_eff
                )

    @qd.kernel
    def _kernel_contact_forces(
        self,
        f: qd.i32,
        particle_radius: qd.f32,
        k_norm: qd.f32,
        k_tang: qd.f32,
        tan_fric: qd.f32,
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_reordered[i_p, i_b].active:
                base = self.sh.pos_to_grid(self.particles_reordered[i_p, i_b].pos)
                for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    slot_idx = self.sh.grid_to_slot(base + offset)
                    for j in range(
                        self.sh.slot_start[slot_idx, i_b],
                        self.sh.slot_size[slot_idx, i_b] + self.sh.slot_start[slot_idx, i_b],
                    ):
                        if i_p != j and self.particles_reordered[j, i_b].active:
                            self._func_contact_force(i_p, j, i_b, particle_radius, k_norm, k_tang, tan_fric)

    @qd.kernel
    def _kernel_copy_acc_from_reordered(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                reordered_idx = self.particles_ng[i_p, i_b].reordered_idx
                self.particles[i_p, i_b].acc = self.particles_reordered[reordered_idx, i_b].acc

    @qd.func
    def _func_sieve_sdf(self, pos, i_b: qd.i32):
        # signed distance to the sieve solid (negative inside a rod): union of two perpendicular
        # families of horizontal cylinders (round rods, bw = rod diameter), clipped to the square extent
        rel = pos - self.sieve_pos[i_b]
        half = self.sieve_params[i_b][0]
        bw = self.sieve_params[i_b][1]
        period = self.sieve_params[i_b][2]
        thick = self.sieve_params[i_b][3]
        sdf = gs.qd_float(1e30)
        for fam in qd.static(range(2)):
            # distance to the nearest cylinder (rod along the family's axis, round cross-section)
            u = rel[fam]
            v = rel[1 - fam]
            d_center = abs(u - period * qd.floor(u / period + 0.5))
            d_fam = qd.sqrt(d_center * d_center + rel[2] * rel[2]) - 0.5 * bw
            d_fam = qd.max(d_fam, qd.max(abs(u) - half, abs(v) - half))
            sdf = qd.min(sdf, d_fam)
        return sdf

    @qd.func
    def _func_sieve_normal(self, pos, i_b: qd.i32):
        # inward normal of the sieve SDF by central differences (bar edges are non-smooth, as with
        # the reference's level-set gradient normals)
        eps = gs.qd_float(1e-5)
        n = qd.Vector.zero(gs.qd_float, 3)
        for a in qd.static(range(3)):
            dp = qd.Vector.zero(gs.qd_float, 3)
            dp[a] = eps
            n[a] = self._func_sieve_sdf(pos + dp, i_b) - self._func_sieve_sdf(pos - dp, i_b)
        norm = n.norm()
        if norm > gs.EPS:
            n = n / norm
        return n

    @qd.func
    def _func_tilt_box_sdf(self, pos, i_b: qd.i32):
        # signed distance to the rotated box (negative inside): transform the query point into the box
        # local frame with the conjugate quaternion, then evaluate the standard axis-aligned box SDF
        rel = pos - self.tilt_pos[i_b]
        q = self.tilt_quat[i_b]
        q_conj = qd.Vector([q[0], -q[1], -q[2], -q[3]])
        rel_local = qd_transform_by_quat_fast(rel, q_conj)
        half = self.tilt_half[i_b]
        qx = abs(rel_local[0]) - half[0]
        qy = abs(rel_local[1]) - half[1]
        qz = abs(rel_local[2]) - half[2]
        q_pos = qd.Vector([max(qx, 0.0), max(qy, 0.0), max(qz, 0.0)])
        phi = q_pos.norm() + min(max(qx, max(qy, qz)), 0.0)
        # optional handle: bar attached at the local -x face, pointing up-back in the local xz-plane
        # (d = bar axis, e2 = in-plane normal); unioned with the blade SDF
        hlen = self.tilt_handle[i_b][0]
        if hlen > 0.0:
            hrad = self.tilt_handle[i_b][1]
            hang = self.tilt_handle[i_b][2]
            d = qd.Vector([-qd.cos(hang), 0.0, qd.sin(hang)])
            e2 = qd.Vector([qd.sin(hang), 0.0, qd.cos(hang)])
            p = rel_local - qd.Vector([-half[0], 0.0, 0.0])
            hx = abs(p.dot(d) - 0.5 * hlen) - 0.5 * hlen
            hy = abs(p[1]) - hrad
            hz = abs(p.dot(e2)) - hrad
            h_pos = qd.Vector([max(hx, 0.0), max(hy, 0.0), max(hz, 0.0)])
            phi = min(phi, h_pos.norm() + min(max(hx, max(hy, hz)), 0.0))
        return phi

    @qd.func
    def _func_tilt_box_normal(self, pos, i_b: qd.i32):
        # inward normal of the tilted box SDF by central differences (box edges are non-smooth, as with
        # the reference's level-set gradient normals)
        eps = gs.qd_float(1e-5)
        n = qd.Vector.zero(gs.qd_float, 3)
        for a in qd.static(range(3)):
            dp = qd.Vector.zero(gs.qd_float, 3)
            dp[a] = eps
            n[a] = self._func_tilt_box_sdf(pos + dp, i_b) - self._func_tilt_box_sdf(pos - dp, i_b)
        norm = n.norm()
        if norm > gs.EPS:
            n = n / norm
        return n

    @qd.func
    def _func_sdfo_sdf(self, pos, i_b: qd.i32):
        # signed distance to the mesh-SDF obstacle (negative inside): transform the query point into the
        # obstacle local frame with the conjugate quaternion, then trilinearly interpolate the voxelized
        # SDF grid; outside the grid the clamped boundary value plus the out-of-grid distance is returned
        rel = pos - self.sdfo_pos[i_b]
        q = self.sdfo_quat[i_b]
        q_conj = qd.Vector([q[0], -q[1], -q[2], -q[3]])
        p = qd_transform_by_quat_fast(rel, q_conj)
        g = qd.Vector.zero(gs.qd_float, 3)
        for a in qd.static(range(3)):
            # grid nodes sit at origin + (i + 0.5) * cell (cell-center sampling), hence the -0.5
            ga = (p[a] - self.sdfo_origin[i_b][a]) * self.sdfo_inv_cell[i_b][a] - 0.5
            gmax = gs.qd_float(self.sdfo_dims[a] - 1) - 1e-4
            g[a] = min(max(ga, 0.0), gmax)
        x0 = int(qd.floor(g[0]))
        y0 = int(qd.floor(g[1]))
        z0 = int(qd.floor(g[2]))
        fx = g[0] - x0
        fy = g[1] - y0
        fz = g[2] - z0
        nx = self.sdfo_dims[0]
        ny = self.sdfo_dims[1]
        nz = self.sdfo_dims[2]
        phi = gs.qd_float(0.0)
        for i, j, k in qd.static(qd.ndrange(2, 2, 2)):
            w = (fx if i == 1 else 1.0 - fx) * (fy if j == 1 else 1.0 - fy) * (fz if k == 1 else 1.0 - fz)
            # numpy C-order (nx, ny, nz): x slowest, z fastest
            idx = (x0 + i) * (ny * nz) + (y0 + j) * nz + (z0 + k)
            phi = phi + w * self.sdfo_val[idx]
        # out-of-grid extension (physical units); zero inside the grid
        d_phys = qd.Vector.zero(gs.qd_float, 3)
        for a in qd.static(range(3)):
            ga = (p[a] - self.sdfo_origin[i_b][a]) * self.sdfo_inv_cell[i_b][a] - 0.5
            d_phys[a] = (ga - g[a]) * self.sdfo_cell[i_b][a]
        return phi + d_phys.norm()

    @qd.func
    def _func_sdfo_normal(self, pos, i_b: qd.i32):
        # inward normal of the mesh-SDF obstacle by central differences (same FD convention as the
        # tilt box / sieve normals)
        eps = gs.qd_float(1e-5)
        n = qd.Vector.zero(gs.qd_float, 3)
        for a in qd.static(range(3)):
            dp = qd.Vector.zero(gs.qd_float, 3)
            dp[a] = eps
            n[a] = self._func_sdfo_sdf(pos + dp, i_b) - self._func_sdfo_sdf(pos - dp, i_b)
        norm = n.norm()
        if norm > gs.EPS:
            n = n / norm
        return n

    @qd.kernel
    def _kernel_enforce_boundary(self, f: qd.i32, particle_radius: qd.f32, tan_fric: qd.f32):
        # C++ Collider::Enforce(DEMParticles, ddt, radius): domain box (static, phi positive inside) plus an
        # optional moving box obstacle (relative velocity delta_vel = v - v_obstacle, as in the C++ code)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                pos = self.particles[i_p, i_b].pos
                vel = self.particles[i_p, i_b].vel
                acc = self.particles[i_p, i_b].acc

                # domain box
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

                if phi < 0.5 * particle_radius:
                    if phi < 0.0:
                        pos = pos - n * phi
                        vel_n = vel.dot(n)
                        if vel_n < 0.0:
                            vel = vel - 1.5 * vel_n * n
                    acc_n = acc.dot(n)
                    if acc_n < 0.0:
                        vel_tangential = vel - vel.dot(n) * n
                        vt_norm = vel_tangential.norm()
                        # C++ normalizes the tangential velocity unconditionally; guard the zero-velocity case
                        if vt_norm > gs.EPS:
                            acc = acc + acc_n * tan_fric * vel_tangential / vt_norm

                # moving box obstacle
                if self.obstacle_enabled[i_b] == 1:
                    rel = pos - self.obstacle_pos[i_b]
                    half = self.obstacle_half[i_b]
                    qx = abs(rel[0]) - half[0]
                    qy = abs(rel[1]) - half[1]
                    qz = abs(rel[2]) - half[2]
                    q_pos = qd.Vector([max(qx, 0.0), max(qy, 0.0), max(qz, 0.0)])
                    phi_obs = q_pos.norm() + min(max(qx, max(qy, qz)), 0.0)

                    if phi_obs < 0.5 * particle_radius:
                        # outside: exact SDF gradient; inside: dominant-axis face normal
                        n_obs = qd.Vector.zero(gs.qd_float, 3)
                        if phi_obs > 0.0:
                            n_obs = q_pos / phi_obs
                        elif qx >= qy and qx >= qz:
                            n_obs[0] = 1.0 if rel[0] >= 0.0 else -1.0
                        elif qy >= qz:
                            n_obs[1] = 1.0 if rel[1] >= 0.0 else -1.0
                        else:
                            n_obs[2] = 1.0 if rel[2] >= 0.0 else -1.0

                        delta_vel = vel - self.obstacle_vel[i_b]
                        if phi_obs < 0.0:
                            pos = pos - n_obs * phi_obs
                            dv_n = delta_vel.dot(n_obs)
                            if dv_n < 0.0:
                                vel = vel - 1.5 * dv_n * n_obs
                        acc_n = acc.dot(n_obs)
                        if acc_n < 0.0:
                            dv_tangential = delta_vel - delta_vel.dot(n_obs) * n_obs
                            vt_norm = dv_tangential.norm()
                            if vt_norm > gs.EPS:
                                acc = acc + acc_n * tan_fric * dv_tangential / vt_norm

                # moving sieve obstacle (same enforcement semantics)
                if self.sieve_enabled[i_b] == 1:
                    phi_s = self._func_sieve_sdf(pos, i_b)
                    if phi_s < 0.5 * particle_radius:
                        n_s = self._func_sieve_normal(pos, i_b)
                        delta_vel = vel - self.sieve_vel[i_b]
                        if phi_s < 0.0:
                            pos = pos - n_s * phi_s
                            dv_n = delta_vel.dot(n_s)
                            if dv_n < 0.0:
                                vel = vel - 1.5 * dv_n * n_s
                        acc_n = acc.dot(n_s)
                        if acc_n < 0.0:
                            dv_tangential = delta_vel - delta_vel.dot(n_s) * n_s
                            vt_norm = dv_tangential.norm()
                            if vt_norm > gs.EPS:
                                acc = acc + acc_n * tan_fric * dv_tangential / vt_norm

                # moving tilted box obstacle (same enforcement semantics; the obstacle velocity is evaluated
                # at the contact point: v_col = v_center + omega x (pos - center))
                if self.tilt_enabled[i_b] == 1:
                    phi_t = self._func_tilt_box_sdf(pos, i_b)
                    if phi_t < 0.5 * particle_radius:
                        n_t = self._func_tilt_box_normal(pos, i_b)
                        v_col = self.tilt_vel[i_b] + self.tilt_omega[i_b].cross(pos - self.tilt_pos[i_b])
                        delta_vel = vel - v_col
                        if phi_t < 0.0:
                            pos = pos - n_t * phi_t
                            dv_n = delta_vel.dot(n_t)
                            if dv_n < 0.0:
                                vel = vel - 1.5 * dv_n * n_t
                        acc_n = acc.dot(n_t)
                        if acc_n < 0.0:
                            dv_tangential = delta_vel - delta_vel.dot(n_t) * n_t
                            vt_norm = dv_tangential.norm()
                            if vt_norm > gs.EPS:
                                acc = acc + acc_n * tan_fric * dv_tangential / vt_norm

                # moving mesh-SDF obstacle (same enforcement semantics as the tilt box; the obstacle
                # velocity is evaluated at the contact point: v_col = v_center + omega x (pos - center)).
                # A cheap local-frame AABB test gates the trilinear SDF query.
                if self.sdfo_enabled[i_b] == 1:
                    rel_m = pos - self.sdfo_pos[i_b]
                    q_m = self.sdfo_quat[i_b]
                    qc_m = qd.Vector([q_m[0], -q_m[1], -q_m[2], -q_m[3]])
                    p_m = qd_transform_by_quat_fast(rel_m, qc_m)
                    bmin = self.sdfo_origin[i_b]
                    bmax = self.sdfo_origin[i_b] + self.sdfo_extent[i_b]
                    qx = max(bmin[0] - p_m[0], p_m[0] - bmax[0])
                    qy = max(bmin[1] - p_m[1], p_m[1] - bmax[1])
                    qz = max(bmin[2] - p_m[2], p_m[2] - bmax[2])
                    phi_bb = qd.Vector([max(qx, 0.0), max(qy, 0.0), max(qz, 0.0)]).norm()
                    if phi_bb < 0.5 * particle_radius:
                        phi_m = self._func_sdfo_sdf(pos, i_b)
                        if phi_m < 0.5 * particle_radius:
                            n_m = self._func_sdfo_normal(pos, i_b)
                            v_col = self.sdfo_vel[i_b] + self.sdfo_omega[i_b].cross(pos - self.sdfo_pos[i_b])
                            delta_vel = vel - v_col
                            if phi_m < 0.0:
                                pos = pos - n_m * phi_m
                                dv_n = delta_vel.dot(n_m)
                                if dv_n < 0.0:
                                    vel = vel - 1.5 * dv_n * n_m
                            acc_n = acc.dot(n_m)
                            if acc_n < 0.0:
                                dv_tangential = delta_vel - delta_vel.dot(n_m) * n_m
                                vt_norm = dv_tangential.norm()
                                if vt_norm > gs.EPS:
                                    acc = acc + acc_n * tan_fric * dv_tangential / vt_norm

                self.particles[i_p, i_b].pos = pos
                self.particles[i_p, i_b].vel = vel
                self.particles[i_p, i_b].acc = acc

    @qd.kernel
    def _kernel_move_obstacle(self, ddt: qd.f32):
        # continuous obstacle motion at the DEM sub-substep timescale (see set_box_obstacle)
        for i_b in range(self._B):
            if self.obstacle_enabled[i_b] == 1:
                self.obstacle_pos[i_b] = self.obstacle_pos[i_b] + self.obstacle_vel[i_b] * ddt
            if self.sieve_enabled[i_b] == 1:
                self.sieve_pos[i_b] = self.sieve_pos[i_b] + self.sieve_vel[i_b] * ddt
            if self.tilt_enabled[i_b] == 1:
                self.tilt_pos[i_b] = self.tilt_pos[i_b] + self.tilt_vel[i_b] * ddt
                om = self.tilt_omega[i_b]
                if om.norm_sqr() > 0.0:
                    # q_dot = 0.5 * (0, omega) x q (quaternion product)
                    q = self.tilt_quat[i_b]
                    dq = qd.Vector(
                        [
                            -om[0] * q[1] - om[1] * q[2] - om[2] * q[3],
                            om[0] * q[0] + om[1] * q[3] - om[2] * q[2],
                            -om[0] * q[3] + om[1] * q[0] + om[2] * q[1],
                            om[0] * q[2] - om[1] * q[1] + om[2] * q[0],
                        ]
                    )
                    q_new = q + 0.5 * dq * ddt
                    self.tilt_quat[i_b] = q_new / q_new.norm()
            if self.sdfo_enabled[i_b] == 1:
                # same kinematic integration as the tilt box
                self.sdfo_pos[i_b] = self.sdfo_pos[i_b] + self.sdfo_vel[i_b] * ddt
                om = self.sdfo_omega[i_b]
                if om.norm_sqr() > 0.0:
                    q = self.sdfo_quat[i_b]
                    dq = qd.Vector(
                        [
                            -om[0] * q[1] - om[1] * q[2] - om[2] * q[3],
                            om[0] * q[0] + om[1] * q[3] - om[2] * q[2],
                            -om[0] * q[3] + om[1] * q[0] + om[2] * q[1],
                            om[0] * q[2] - om[1] * q[1] + om[2] * q[0],
                        ]
                    )
                    q_new = q + 0.5 * dq * ddt
                    self.sdfo_quat[i_b] = q_new / q_new.norm()

    @qd.kernel
    def _kernel_integrate(self, f: qd.i32, ddt: qd.f32):
        # C++ MoveDEMParticlesSplit: v += a * ddt, then x += v * ddt
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                self.particles[i_p, i_b].vel = self.particles[i_p, i_b].vel + self.particles[i_p, i_b].acc * ddt
                self.particles[i_p, i_b].pos = self.particles[i_p, i_b].pos + self.particles[i_p, i_b].vel * ddt

    @qd.kernel
    def _kernel_update_render_fields(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            self.particles_render[i_p, i_b].pos = self.particles[i_p, i_b].pos
            self.particles_render[i_p, i_b].active = self.particles[i_p, i_b].active

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------- coupled mode (C++ GIC coupling) ------------------------
    # ------------------------------------------------------------------------------------

    @qd.func
    def _func_sample_flip_faces(self, pos, fu: qd.template(), fv: qd.template(), fw: qd.template()):
        # trilinear sample of a FLIP MAC face field triple at pos (same staggering as FLIP p2g)
        flip = self._sim.flip_solver
        vel = qd.Vector.zero(gs.qd_float, 3)
        for axis in qd.static(range(3)):
            offsets = qd.Vector([0.5, 0.5, 0.5])
            offsets[axis] = 0.0
            rel = (pos - flip._lower_v) * flip._inv_dx - offsets
            base = qd.floor(rel, gs.qd_int)
            frac = rel - base
            val = gs.qd_float(0.0)
            for i, j, k in qd.static(qd.ndrange(2, 2, 2)):
                w = (frac[0] if i else 1.0 - frac[0]) * (frac[1] if j else 1.0 - frac[1]) * (
                    frac[2] if k else 1.0 - frac[2]
                )
                idx = base + qd.Vector([i, j, k], dt=gs.qd_int)
                if axis == 0:
                    idx = self._func_clamp_flip_idx(idx, qd.Vector(fu.shape, dt=gs.qd_int) - 1)
                    val += fu[idx] * w
                elif axis == 1:
                    idx = self._func_clamp_flip_idx(idx, qd.Vector(fv.shape, dt=gs.qd_int) - 1)
                    val += fv[idx] * w
                else:
                    idx = self._func_clamp_flip_idx(idx, qd.Vector(fw.shape, dt=gs.qd_int) - 1)
                    val += fw[idx] * w
            vel[axis] = val
        return vel

    @qd.func
    def _func_clamp_flip_idx(self, idx, max_idx):
        return qd.Vector(
            [
                qd.min(qd.max(idx[0], 0), max_idx[0]),
                qd.min(qd.max(idx[1], 0), max_idx[1]),
                qd.min(qd.max(idx[2], 0), max_idx[2]),
            ],
            dt=gs.qd_int,
        )

    @qd.func
    def _func_sample_flip_cell(self, pos, field: qd.template()):
        # trilinear sample of a FLIP cell-centered scalar field at pos
        flip = self._sim.flip_solver
        rel = (pos - flip._lower_v) * flip._inv_dx - 0.5
        base = qd.floor(rel, gs.qd_int)
        frac = rel - base
        val = gs.qd_float(0.0)
        max_cell = qd.Vector(field.shape, dt=gs.qd_int) - 1
        for i, j, k in qd.static(qd.ndrange(2, 2, 2)):
            w = (frac[0] if i else 1.0 - frac[0]) * (frac[1] if j else 1.0 - frac[1]) * (
                frac[2] if k else 1.0 - frac[2]
            )
            cell = self._func_clamp_flip_idx(base + qd.Vector([i, j, k], dt=gs.qd_int), max_cell)
            val += field[cell] * w
        return val

    @qd.kernel
    def _kernel_update_saturate_rate(self, f: qd.i32):
        # C++ MoveDEMParticlesSplit: SaturateRate = clamp(ratio + den / (1 - den), 0, 1)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                den = self._func_sample_flip_cell(
                    self.particles[i_p, i_b].pos, self._sim.flip_solver.density
                )
                self.particles[i_p, i_b].saturate_rate = qd.min(
                    qd.max(self.particles[i_p, i_b].ratio + den / (1.0 - den), 0.0), 1.0
                )

    @qd.kernel
    def _kernel_cal_coupling(self, f: qd.i32):
        # C++ CalCoupling (alg2): pressure gradient + relative acceleration (added mass) + quadratic drag,
        # applied only inside the fluid (levelset <= 0)
        flip = self._sim.flip_solver
        V = gs.qd_float(self._particle_volume)
        r = gs.qd_float(self._particle_radius)
        inv_dt = gs.qd_float(1.0 / flip._last_dt)
        visc = gs.qd_float(flip._viscosity_coeff)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                pos = self.particles[i_p, i_b].pos
                if self._func_sample_flip_cell(pos, flip.levelset) <= 0.0:
                    ratio = self.particles[i_p, i_b].ratio
                    grad_p = self._func_sample_flip_faces(pos, flip.grad_p_u, flip.grad_p_v, flip.grad_p_w)
                    vdiff = self._func_sample_flip_faces(pos, flip.vdiff_u, flip.vdiff_v, flip.vdiff_w)
                    v_grid = self._func_sample_flip_faces(pos, flip.vel_u, flip.vel_v, flip.vel_w)
                    cf = self.particles[i_p, i_b].coupling_force
                    cf -= grad_p * (inv_dt * V * (1.0 + ratio))
                    cf += (vdiff * inv_dt - self.particles[i_p, i_b].acc) * (V * 0.5 * (1.0 + ratio))
                    dv = v_grid - self.particles[i_p, i_b].vel
                    cf += dv * dv.norm() * (12.0 * 3.141592653589793 * r * r * visc)
                    self.particles[i_p, i_b].coupling_force = cf

    @qd.kernel
    def _kernel_reset_acc_coupled(self, f: qd.i32, dt_f: qd.f32):
        # C++ ApplyDEMForces: gravity (not mass-scaled) + (coupling + absorbed-water impulse) / (m + ratio*V)
        V = gs.qd_float(self._particle_volume)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                ratio = self.particles[i_p, i_b].ratio
                inv_mass_eff = 1.0 / (self.particles_info[i_p].mass + ratio * V)
                self.particles[i_p, i_b].inv_mass_eff = inv_mass_eff
                absorb = self.particles[i_p, i_b].add_velocity * (
                    self.particles[i_p, i_b].add_ratio * V / dt_f
                )
                self.particles[i_p, i_b].acc = self._gravity[i_b] + (
                    self.particles[i_p, i_b].coupling_force + absorb
                ) * inv_mass_eff

    @qd.kernel
    def _kernel_transfer_coupling_forces(self, f: qd.i32, ddt: qd.f32, dt_f: qd.f32):
        # C++ TransferCouplingForces: trilinear scatter to the MAC faces, scaled by ddt / dt_f
        # (no weight-sum normalization, as in the reference), then clear the particle force.
        # Semi-implicit addition (recorded deviation): the same kick (-cf * w * ddt / dx^3) is also
        # applied immediately to the current grid velocity, so the drag sampled by later
        # sub-substeps sees the already-corrected water velocity and the explicit reaction does
        # not overshoot. The accumulated force is still applied once after P2G, as in C++.
        flip = self._sim.flip_solver
        scale = ddt / dt_f
        kick = ddt * flip._inv_dx**3
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles[i_p, i_b].active:
                pos = self.particles[i_p, i_b].pos
                cf = self.particles[i_p, i_b].coupling_force
                for axis in qd.static(range(3)):
                    offsets = qd.Vector([0.5, 0.5, 0.5])
                    offsets[axis] = 0.0
                    rel = (pos - flip._lower_v) * flip._inv_dx - offsets
                    base = qd.floor(rel, gs.qd_int)
                    frac = rel - base
                    for i, j, k in qd.static(qd.ndrange(2, 2, 2)):
                        w = (frac[0] if i else 1.0 - frac[0]) * (frac[1] if j else 1.0 - frac[1]) * (
                            frac[2] if k else 1.0 - frac[2]
                        )
                        idx = base + qd.Vector([i, j, k], dt=gs.qd_int)
                        if axis == 0:
                            idx = self._func_clamp_flip_idx(idx, qd.Vector(flip.coupling_force_u.shape, dt=gs.qd_int) - 1)
                            qd.atomic_add(flip.coupling_force_u[idx], cf[0] * w * scale)
                            qd.atomic_add(flip.vel_u[idx], -cf[0] * w * kick)
                        elif axis == 1:
                            idx = self._func_clamp_flip_idx(idx, qd.Vector(flip.coupling_force_v.shape, dt=gs.qd_int) - 1)
                            qd.atomic_add(flip.coupling_force_v[idx], cf[1] * w * scale)
                            qd.atomic_add(flip.vel_v[idx], -cf[1] * w * kick)
                        else:
                            idx = self._func_clamp_flip_idx(idx, qd.Vector(flip.coupling_force_w.shape, dt=gs.qd_int) - 1)
                            qd.atomic_add(flip.coupling_force_w[idx], cf[2] * w * scale)
                            qd.atomic_add(flip.vel_w[idx], -cf[2] * w * kick)
                self.particles[i_p, i_b].coupling_force = qd.Vector.zero(gs.qd_float, 3)

    def _dem_substep_coupled(self, f, ddt, dt_f, k_norm, k_tang, tan_fric):
        # C++ MoveDEMParticlesSplit (coupled): saturate rate, coupling force (reads the previous
        # sub-substep's acc), force assembly, contact, reaction scatter, collider, integrate
        self._kernel_update_saturate_rate(f)
        self._kernel_cal_coupling(f)
        self._kernel_reset_acc_coupled(f, dt_f)
        self._kernel_reorder_particles(f)
        self._kernel_contact_forces(f, self._particle_radius, k_norm, k_tang, tan_fric)
        self._kernel_copy_acc_from_reordered(f)
        self._kernel_transfer_coupling_forces(f, ddt, dt_f)
        self._kernel_move_obstacle(ddt)
        self._kernel_enforce_boundary(f, self._particle_radius, tan_fric)
        self._kernel_integrate(f, ddt)

    def _dem_substep(self, f, ddt, inv_mass, k_norm, k_tang, tan_fric):
        self._kernel_reset_acc(f)
        self._kernel_reorder_particles(f)
        self._kernel_contact_forces(f, self._particle_radius, k_norm, k_tang, tan_fric)
        self._kernel_copy_acc_from_reordered(f)
        self._kernel_move_obstacle(ddt)
        self._kernel_enforce_boundary(f, self._particle_radius, tan_fric)
        self._kernel_integrate(f, ddt)

    def substep_pre_coupling(self, f):
        if not self.is_active or self._coupled:
            # in coupled mode the FLIP solver drives the DEM advance per water step
            return
        self._advance(float(self._substep_dt), f)

    def _material_constants(self):
        # contact stiffnesses and grain mass are global constants in C++ (m_ParticleMass, DEMForce);
        # they are taken from the (single) material here
        material = self._entities[0].material
        volume = 4.0 / 3.0 * math.pi * self._particle_radius**3
        inv_mass = float(1.0 / (material.rho * volume))
        k_norm = float(material.young_modulus * self._particle_radius)
        k_tang = float(k_norm * material.poisson_ratio)
        tan_fric = float(math.tan(material.friction_angle))
        return inv_mass, k_norm, k_tang, tan_fric

    def _advance(self, dt_total, f):
        # fixed sub-substep (the reference's velocity-adaptive clamp is removed by user decision):
        # as many full `ddt` steps as fit into dt_total, plus one remainder step
        inv_mass, k_norm, k_tang, tan_fric = self._material_constants()
        ddt = self._m_ddt * self._ddt_safety
        n_full = int(dt_total / ddt)
        for _ in range(n_full):
            self._dem_substep(f, ddt, inv_mass, k_norm, k_tang, tan_fric)
        remainder = dt_total - n_full * ddt
        if remainder > 0.0:
            self._dem_substep(f, remainder, inv_mass, k_norm, k_tang, tan_fric)

    def advance_coupled(self, f, dt_f):
        # C++ MoveDEMParticles(dt_f): fresh fluid density, then the ddt sub-substep loop whose
        # total duration is exactly dt_f (called by the FLIP solver before each water step)
        self._sim.flip_solver._kernel_cal_density(f)
        inv_mass, k_norm, k_tang, tan_fric = self._material_constants()
        ddt = self._m_ddt * self._ddt_safety
        n_full = int(dt_f / ddt)
        for _ in range(n_full):
            self._dem_substep_coupled(f, ddt, dt_f, k_norm, k_tang, tan_fric)
        remainder = dt_f - n_full * ddt
        if remainder > 0.0:
            self._dem_substep_coupled(f, remainder, dt_f, k_norm, k_tang, tan_fric)

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if self.is_active:
            self._kernel_update_render_fields(f)

    def substep_post_coupling_grad(self, f):
        pass

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
            self.particles[i_p, i_b].active = qd.cast(active[i_b, i_p], gs.qd_bool)

    def get_state(self, f):
        if self.is_active:
            from genesis.engine.states.solvers import DEMSolverState

            state = DEMSolverState(self.scene)
            self._kernel_get_state(f, state.pos, state.vel, state.active)
        else:
            state = None
        return state

    def get_state_render(self):
        if not self.is_active:
            return None, None, None
        self.update_render_fields()
        return self.particles_render.pos, None, None

    def update_render_fields(self):
        self._kernel_update_render_fields(self.sim.cur_substep_local)

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
            active[i_b, i_p] = qd.cast(self.particles[i_p, i_b].active, gs.qd_bool)

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

    @qd.kernel
    def _kernel_get_mass(
        self, particle_start: qd.i32, n_particles: qd.i32, mass: qd.types.ndarray(), envs_idx: qd.types.ndarray()
    ):
        total_mass = gs.qd_float(0.0)
        for i_p_ in range(n_particles):
            i_p = i_p_ + particle_start
            total_mass += self.particles_info[i_p].mass
        for i_b_ in range(envs_idx.shape[0]):
            mass[i_b_] = total_mass

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        return sum([entity.n_particles for entity in self._entities])

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_radius

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound
