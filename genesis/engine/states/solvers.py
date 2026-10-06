import genesis as gs
from genesis.repr_base import RBC


class SimState(RBC):
    """
    Dynamic state queried from a Scene's Simulator.
    """

    def __init__(
        self,
        scene,
        s_global,
        f_local,
        solvers,
        coupler_state=None,
    ):
        self._scene = scene
        self._s_global = s_global
        self._solvers_state = []
        for solver in solvers:
            self._solvers_state.append(solver.get_state(f_local))
        self._coupler_state = coupler_state

    def serializable(self):
        self._scene = None

        for solver_state in self._solvers_state:
            if solver_state is not None:
                solver_state.serializable()

    @property
    def scene(self):
        return self._scene

    @property
    def s_global(self):
        return self._s_global

    @property
    def solvers_state(self):
        return self._solvers_state

    @property
    def coupler_state(self):
        return self._coupler_state

    def __iter__(self):
        return iter(self._solvers_state)


class KinematicSolverState:
    """
    Dynamic state queried from a KinematicSolver.

    Only stores position-related fields (qpos, link poses). Physics fields
    (velocity, acceleration, mass, friction) are omitted since kinematic entities have no dynamics.
    """

    def __init__(self, scene, s_global):
        self.scene = scene
        self._s_global = s_global

        _B = scene.sim.kinematic_solver._B
        args = {
            "dtype": gs.tc_float,
            "requires_grad": scene.requires_grad,
            "scene": self.scene,
        }
        self.qpos = gs.zeros((_B, scene.sim.kinematic_solver.n_qs), **args)
        self.dofs_vel = gs.zeros((_B, scene.sim.kinematic_solver.n_dofs), **args)
        self.links_pos = gs.zeros((_B, scene.sim.kinematic_solver.n_links, 3), **args)
        self.links_quat = gs.zeros((_B, scene.sim.kinematic_solver.n_links, 4), **args)
        self.i_pos_shift = gs.zeros((_B, scene.sim.kinematic_solver.n_links, 3), **args)

    def serializable(self):
        self.scene = None
        self.qpos = self.qpos.detach()
        self.dofs_vel = self.dofs_vel.detach()
        self.links_pos = self.links_pos.detach()
        self.links_quat = self.links_quat.detach()
        self.i_pos_shift = self.i_pos_shift.detach()

    @property
    def s_global(self):
        return self._s_global


class RigidSolverState:
    """
    Dynamic state queried from a RigidSolver.
    """

    def __init__(self, scene, s_global):
        self.scene = scene

        self._s_global = s_global

        _B = scene.sim.rigid_solver._B
        args = {
            "dtype": gs.tc_float,
            "requires_grad": scene.requires_grad,
            "scene": self.scene,
        }
        self.qpos = gs.zeros((_B, scene.sim.rigid_solver.n_qs), **args)
        self.dofs_vel = gs.zeros((_B, scene.sim.rigid_solver.n_dofs), **args)
        self.dofs_acc = gs.zeros((_B, scene.sim.rigid_solver.n_dofs), **args)
        self.links_pos = gs.zeros((_B, scene.sim.rigid_solver.n_links, 3), **args)
        self.links_quat = gs.zeros((_B, scene.sim.rigid_solver.n_links, 4), **args)
        self.i_pos_shift = gs.zeros((_B, scene.sim.rigid_solver.n_links, 3), **args)
        self.mass_shift = gs.zeros((_B, scene.sim.rigid_solver.n_links), **args)
        self.friction_ratio = gs.ones((_B, scene.sim.rigid_solver.n_geoms), **args)

    def serializable(self):
        self.scene = None
        self.qpos = self.qpos.detach()
        self.dofs_vel = self.dofs_vel.detach()
        self.dofs_acc = self.dofs_acc.detach()
        self.links_pos = self.links_pos.detach()
        self.links_quat = self.links_quat.detach()
        self.i_pos_shift = self.i_pos_shift.detach()
        self.mass_shift = self.mass_shift.detach()
        self.friction_ratio = self.friction_ratio.detach()

    @property
    def s_global(self):
        return self._s_global


class ToolSolverState:
    """
    Dynamic state queried from a RigidSolver.
    """

    def __init__(self, scene):
        self.scene = scene
        self.entities = []

    def serializable(self):
        self.scene = None

        for entity_state in self.entities:
            entity_state.serializable()

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, index):
        return self.entities[index]

    # def __repr__(self):
    #     return f'{_repr(self)}\n' \
    #            f'entities : {_repr(self.entities)}'


class MPMSolverState(RBC):
    """
    Dynamic state queried from a MPMSolver.
    """

    def __init__(self, scene):
        self._scene = scene
        args = {
            "dtype": gs.tc_float,
            "requires_grad": scene.requires_grad,
            "scene": self._scene,
        }
        self._pos = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles, 3), **args)
        self._vel = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles, 3), **args)
        self._C = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles, 3, 3), **args)
        self._F = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles, 3, 3), **args)
        self._Jp = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles), **args)
        args["dtype"] = gs.tc_bool
        args["requires_grad"] = False
        self._active = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles), **args)

    def serializable(self):
        self._scene = None

        self._pos = self._pos.detach()
        self._vel = self._vel.detach()
        self._C = self._C.detach()
        self._F = self._F.detach()
        self._Jp = self._Jp.detach()
        self._active = self._active.detach()

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def C(self):
        return self._C

    @property
    def F(self):
        return self._F

    @property
    def Jp(self):
        return self._Jp

    @property
    def active(self):
        return self._active


class SPHSolverState:
    """
    Dynamic state queried from a SPHSolver.
    """

    def __init__(self, scene):
        self._scene = scene
        args = {
            "dtype": gs.tc_float,
            "requires_grad": scene.requires_grad,
            "scene": self._scene,
        }
        self._pos = gs.zeros((scene.sim._B, scene.sim.sph_solver.n_particles, 3), **args)
        self._vel = gs.zeros((self._scene.sim._B, scene.sim.sph_solver.n_particles, 3), **args)
        args["dtype"] = gs.tc_bool
        args["requires_grad"] = False
        self._active = gs.zeros((self._scene.sim._B, scene.sim.sph_solver.n_particles), **args)

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def active(self):
        return self._active


class _ParticleFluidSolverState:
    def __init__(self, scene, solver):
        self._scene = scene
        args = {
            "dtype": gs.tc_float,
            "requires_grad": False,
            "scene": scene,
        }
        self._pos = gs.zeros((scene.sim._B, solver.n_particles, 3), **args)
        self._vel = gs.zeros((scene.sim._B, solver.n_particles, 3), **args)
        args["dtype"] = gs.tc_bool
        self._active = gs.zeros((scene.sim._B, solver.n_particles), **args)

    def serializable(self):
        self._scene = None
        self._pos = self._pos.detach()
        self._vel = self._vel.detach()
        self._active = self._active.detach()

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def active(self):
        return self._active


class IPBSTFSolverState(_ParticleFluidSolverState):
    """Dynamic state queried from an implicit position-based surface-tension fluid (IPBSTF) solver."""

    def __init__(self, scene):
        super().__init__(scene, scene.sim.ipbstf_solver)
        solver = scene.sim.ipbstf_solver
        self._static_colliders_pos = None
        self._static_colliders_quat = None
        if solver._n_static_colliders > 0:
            args = {
                "dtype": gs.tc_float,
                "requires_grad": False,
                "scene": scene,
            }
            self._static_colliders_pos = gs.zeros((scene.sim._B, solver._n_static_colliders, 3), **args)
            self._static_colliders_quat = gs.zeros((scene.sim._B, solver._n_static_colliders, 4), **args)

    def serializable(self):
        super().serializable()
        if self._static_colliders_pos is not None:
            self._static_colliders_pos = self._static_colliders_pos.detach()
            self._static_colliders_quat = self._static_colliders_quat.detach()

    @property
    def static_colliders_pos(self):
        return self._static_colliders_pos

    @property
    def static_colliders_quat(self):
        return self._static_colliders_quat


class PBSTFSolverState(_ParticleFluidSolverState):
    """Dynamic state queried from a position-based surface tension flow (PBSTF) solver."""

    def __init__(self, scene):
        super().__init__(scene, scene.sim.pbstf_solver)
        solver = scene.sim.pbstf_solver
        args = {
            "dtype": gs.tc_float,
            "requires_grad": False,
            "scene": scene,
        }
        self._static_colliders_pos = gs.zeros((scene.sim._B, solver._n_static_colliders, 3), **args)
        self._static_colliders_quat = gs.zeros((scene.sim._B, solver._n_static_colliders, 4), **args)
        self._c = gs.zeros((scene.sim._B, solver.n_particles), **args)
        self._deformable_static_colliders_surface_vertices = None
        self._deformable_static_colliders_voxel_positions = None
        self._deformable_static_colliders_voxel_search_order = None
        self._is_deformable_static_colliders_sdf_active = None
        if solver._n_deformable_static_colliders > 0:
            self._deformable_static_colliders_surface_vertices = gs.zeros(
                (scene.sim._B, solver._n_deformable_surface_vertices, 3), **args
            )
            self._deformable_static_colliders_voxel_positions = gs.zeros(
                (scene.sim._B, solver._n_deformable_voxels, 3), **args
            )
            args["dtype"] = gs.tc_int
            self._deformable_static_colliders_voxel_search_order = gs.zeros(
                (scene.sim._B, solver._n_deformable_voxel_search_order), **args
            )
        if solver._n_deformable_sdf_colliders > 0:
            args["dtype"] = gs.tc_bool
            self._is_deformable_static_colliders_sdf_active = gs.zeros(
                (scene.sim._B, solver._n_deformable_sdf_colliders), **args
            )
        self._absorbed_collider_idx = None
        self._absorbed_voxel_idx = None
        self._absorption_voxel_distance = None
        self._absorption_local_pos = None
        self._absorption_target_local_pos = None
        self._absorption_progress = None
        self._absorption_capture_budget = None
        if solver._n_absorbent_static_colliders > 0:
            args["dtype"] = gs.tc_int
            self._absorbed_collider_idx = gs.zeros((scene.sim._B, solver.n_particles), **args)
            self._absorbed_voxel_idx = gs.zeros((scene.sim._B, solver.n_particles), **args)
            self._absorption_voxel_distance = gs.zeros((scene.sim._B, solver.n_particles), **args)
            args["dtype"] = gs.tc_float
            self._absorption_local_pos = gs.zeros((scene.sim._B, solver.n_particles, 3), **args)
            self._absorption_target_local_pos = gs.zeros((scene.sim._B, solver.n_particles, 3), **args)
            self._absorption_progress = gs.zeros((scene.sim._B, solver.n_particles), **args)
            self._absorption_capture_budget = gs.zeros((scene.sim._B, solver._n_absorbent_static_colliders), **args)

    def serializable(self):
        super().serializable()
        self._static_colliders_pos = self._static_colliders_pos.detach()
        self._static_colliders_quat = self._static_colliders_quat.detach()
        self._c = self._c.detach()
        if self._deformable_static_colliders_surface_vertices is not None:
            self._deformable_static_colliders_surface_vertices = (
                self._deformable_static_colliders_surface_vertices.detach()
            )
            self._deformable_static_colliders_voxel_positions = (
                self._deformable_static_colliders_voxel_positions.detach()
            )
            self._deformable_static_colliders_voxel_search_order = (
                self._deformable_static_colliders_voxel_search_order.detach()
            )
        if self._is_deformable_static_colliders_sdf_active is not None:
            self._is_deformable_static_colliders_sdf_active = self._is_deformable_static_colliders_sdf_active.detach()
        if self._absorbed_collider_idx is not None:
            self._absorbed_collider_idx = self._absorbed_collider_idx.detach()
            self._absorbed_voxel_idx = self._absorbed_voxel_idx.detach()
            self._absorption_voxel_distance = self._absorption_voxel_distance.detach()
            self._absorption_local_pos = self._absorption_local_pos.detach()
            self._absorption_target_local_pos = self._absorption_target_local_pos.detach()
            self._absorption_progress = self._absorption_progress.detach()
            self._absorption_capture_budget = self._absorption_capture_budget.detach()

    @property
    def static_colliders_pos(self):
        return self._static_colliders_pos

    @property
    def static_colliders_quat(self):
        return self._static_colliders_quat

    @property
    def c(self):
        return self._c

    @property
    def deformable_static_colliders_surface_vertices(self):
        return self._deformable_static_colliders_surface_vertices

    @property
    def deformable_static_colliders_voxel_positions(self):
        return self._deformable_static_colliders_voxel_positions

    @property
    def deformable_static_colliders_voxel_search_order(self):
        return self._deformable_static_colliders_voxel_search_order

    @property
    def is_deformable_static_colliders_sdf_active(self):
        return self._is_deformable_static_colliders_sdf_active

    @property
    def absorbed_collider_idx(self):
        return self._absorbed_collider_idx

    @property
    def absorbed_voxel_idx(self):
        return self._absorbed_voxel_idx

    @property
    def absorption_voxel_distance(self):
        return self._absorption_voxel_distance

    @property
    def absorption_local_pos(self):
        return self._absorption_local_pos

    @property
    def absorption_target_local_pos(self):
        return self._absorption_target_local_pos

    @property
    def absorption_progress(self):
        return self._absorption_progress

    @property
    def absorption_capture_budget(self):
        return self._absorption_capture_budget


class PBDSolverState:
    """
    Dynamic entity state queried from a PBDSolver.

    Solver-owned boundary balls are reconstructed from their local geometry and set poses, so
    they are deliberately excluded from Scene snapshots.
    """

    def __init__(self, scene):
        self._scene = scene
        args = {
            "dtype": gs.tc_float,
            "requires_grad": scene.requires_grad,
            "scene": self._scene,
        }
        n_entity_particles = sum(entity.n_particles for entity in scene.sim.pbd_solver.entities)
        self._pos = gs.zeros((scene.sim._B, n_entity_particles, 3), **args)
        self._vel = gs.zeros((self._scene.sim._B, n_entity_particles, 3), **args)
        args["dtype"] = gs.tc_bool
        args["requires_grad"] = False
        self._free = gs.zeros((self._scene.sim._B, n_entity_particles), **args)

    def serializable(self):
        self._scene = None
        self._pos = self._pos.detach()
        self._vel = self._vel.detach()
        self._free = self._free.detach()

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def free(self):
        return self._free


class DEMSolverState:
    """Dynamic granular state required by reset and coupled wet-sand checkpoints."""

    def __init__(self, scene):
        self._scene = scene
        solver = scene.dem_solver
        args = {"dtype": gs.tc_float, "requires_grad": False, "scene": scene}
        shape = (scene.sim._B, solver.n_particles)
        self._pos = gs.zeros((*shape, 3), **args)
        self._vel = gs.zeros((*shape, 3), **args)
        self._ratio = gs.zeros(shape, **args)
        self._tilt_pos = gs.zeros((scene.sim._B, 3), **args)
        self._tilt_quat = gs.zeros((scene.sim._B, 4), **args)
        self._tilt_vel = gs.zeros((scene.sim._B, 3), **args)
        self._tilt_omega = gs.zeros((scene.sim._B, 3), **args)
        self._sdf_pos = gs.zeros((scene.sim._B, 3), **args)
        self._sdf_quat = gs.zeros((scene.sim._B, 4), **args)
        self._sdf_vel = gs.zeros((scene.sim._B, 3), **args)
        self._sdf_omega = gs.zeros((scene.sim._B, 3), **args)
        args["dtype"] = gs.tc_bool
        self._active = gs.zeros(shape, **args)

    def serializable(self):
        self._scene = None
        self._pos = self._pos.detach()
        self._vel = self._vel.detach()
        self._ratio = self._ratio.detach()
        self._tilt_pos = self._tilt_pos.detach()
        self._tilt_quat = self._tilt_quat.detach()
        self._tilt_vel = self._tilt_vel.detach()
        self._tilt_omega = self._tilt_omega.detach()
        self._sdf_pos = self._sdf_pos.detach()
        self._sdf_quat = self._sdf_quat.detach()
        self._sdf_vel = self._sdf_vel.detach()
        self._sdf_omega = self._sdf_omega.detach()
        self._active = self._active.detach()

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def ratio(self):
        return self._ratio

    @property
    def active(self):
        return self._active

    @property
    def tilt_pos(self):
        return self._tilt_pos

    @property
    def tilt_quat(self):
        return self._tilt_quat

    @property
    def tilt_vel(self):
        return self._tilt_vel

    @property
    def tilt_omega(self):
        return self._tilt_omega

    @property
    def sdf_pos(self):
        return self._sdf_pos

    @property
    def sdf_quat(self):
        return self._sdf_quat

    @property
    def sdf_vel(self):
        return self._sdf_vel

    @property
    def sdf_omega(self):
        return self._sdf_omega


class FLIPSolverState:
    """Dynamic FLIP particle and face-velocity state used at a simulation-step boundary."""

    def __init__(self, scene):
        self._scene = scene
        solver = scene.flip_solver
        args = {"dtype": gs.tc_float, "requires_grad": False, "scene": scene}
        shape = (scene.sim._B, solver.n_particles)
        self._pos = gs.zeros((*shape, 3), **args)
        self._vel = gs.zeros((*shape, 3), **args)
        args["dtype"] = gs.tc_bool
        self._active = gs.zeros(shape, **args)
        args["dtype"] = gs.tc_float
        nx, ny, nz = (int(value) for value in solver._res)
        self._grid_vel_u = gs.zeros((nx + 1, ny, nz), **args)
        self._grid_vel_v = gs.zeros((nx, ny + 1, nz), **args)
        self._grid_vel_w = gs.zeros((nx, ny, nz + 1), **args)
        self._last_dt = float(solver._last_dt)

    def serializable(self):
        self._scene = None
        self._pos = self._pos.detach()
        self._vel = self._vel.detach()
        self._active = self._active.detach()
        self._grid_vel_u = self._grid_vel_u.detach()
        self._grid_vel_v = self._grid_vel_v.detach()
        self._grid_vel_w = self._grid_vel_w.detach()

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def active(self):
        return self._active

    @property
    def grid_vel_u(self):
        return self._grid_vel_u

    @property
    def grid_vel_v(self):
        return self._grid_vel_v

    @property
    def grid_vel_w(self):
        return self._grid_vel_w

    @property
    def last_dt(self):
        return self._last_dt


class FEMSolverState:
    def __init__(self, scene):
        self._scene = scene
        args = {
            "dtype": gs.tc_float,
            "requires_grad": scene.requires_grad,
            "scene": self._scene,
        }
        self._pos = gs.zeros((scene.sim._B, scene.sim.fem_solver.n_vertices, 3), **args)
        self._vel = gs.zeros((scene.sim._B, scene.sim.fem_solver.n_vertices, 3), **args)
        args["dtype"] = gs.tc_bool
        args["requires_grad"] = False
        self._active = gs.zeros((scene.sim._B, scene.sim.fem_solver.n_elements), **args)

    def serializable(self):
        self._scene = None

        self._pos = self._pos.detach()
        self._vel = self._vel.detach()
        self._active = self._active.detach()

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def active(self):
        return self._active


class IPBFSolverState(_ParticleFluidSolverState):
    """Dynamic state of an implicit position-based fluid solver."""

    def __init__(self, scene):
        super().__init__(scene, scene.sim.ipbf_solver)
        self._c = gs.zeros((scene.sim._B, scene.sim.ipbf_solver.n_particles), dtype=gs.tc_float, scene=scene)
        self._boundary_group = gs.zeros(self._active.shape, dtype=gs.tc_int, scene=scene)
        self._pitcher_origin = None
        self._pitcher_axis = None
        if scene.sim.ipbf_solver._has_boundary_pitcher:
            self._pitcher_origin = gs.zeros((3,), dtype=gs.tc_float, scene=scene)
            self._pitcher_axis = gs.zeros((3,), dtype=gs.tc_float, scene=scene)

    @property
    def c(self):
        return self._c

    @property
    def boundary_group(self):
        return self._boundary_group

    @property
    def pitcher_origin(self):
        return self._pitcher_origin

    @property
    def pitcher_axis(self):
        return self._pitcher_axis

    def serializable(self):
        super().serializable()
        self._c = self._c.detach()
        self._boundary_group = self._boundary_group.detach()
        if self._pitcher_origin is not None:
            self._pitcher_origin = self._pitcher_origin.detach()
            self._pitcher_axis = self._pitcher_axis.detach()


class PBDFluidSolverState(PBDSolverState):
    """Particle liquid state including concentrations, activation and container membership."""

    def __init__(self, scene):
        super().__init__(scene)
        shape = self.free.shape
        self._c = gs.zeros(shape, dtype=gs.tc_float, scene=scene)
        self._active = gs.zeros(shape, dtype=gs.tc_bool, scene=scene)
        self._boundary_group = gs.zeros(shape, dtype=gs.tc_int, scene=scene)

    def serializable(self):
        super().serializable()
        self._c = self._c.detach()
        self._active = self._active.detach()
        self._boundary_group = self._boundary_group.detach()

    @property
    def c(self):
        return self._c

    @property
    def active(self):
        return self._active

    @property
    def boundary_group(self):
        return self._boundary_group
