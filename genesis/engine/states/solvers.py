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
    ):
        self._scene = scene
        self._s_global = s_global
        self._solvers_state = list()
        for solver in solvers:
            self._solvers_state.append(solver.get_state(f_local))

    def serializable(self):
        self.scene = None

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
    Dynamic state queried from a PBDSolver.
    """

    def __init__(self, scene):
        self._scene = scene
        args = {
            "dtype": gs.tc_float,
            "requires_grad": scene.requires_grad,
            "scene": self._scene,
        }
        self._pos = gs.zeros((scene.sim._B, scene.sim.pbd_solver.n_particles, 3), **args)
        self._vel = gs.zeros((self._scene.sim._B, scene.sim.pbd_solver.n_particles, 3), **args)
        args["dtype"] = gs.tc_bool
        args["requires_grad"] = False
        self._free = gs.zeros((self._scene.sim._B, scene.sim.pbd_solver.n_particles), **args)

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
