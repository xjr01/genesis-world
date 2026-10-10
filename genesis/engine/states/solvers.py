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
        # Fragment lifecycle switches (0/1), restored bit-exactly by set_state (see is_externally_driven /
        # is_contact_enabled in array_class.py)
        self.links_externally_driven = gs.zeros((_B, scene.sim.rigid_solver.n_links), **args)
        self.links_contact_enabled = gs.ones((_B, scene.sim.rigid_solver.n_links), **args)

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
        self.links_externally_driven = self.links_externally_driven.detach()
        self.links_contact_enabled = self.links_contact_enabled.detach()

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
    """
    Dynamic state queried from a DEMSolver.
    """

    def __init__(self, scene):
        self._scene = scene
        args = {
            "dtype": gs.tc_float,
            "requires_grad": scene.requires_grad,
            "scene": self._scene,
        }
        self._pos = gs.zeros((scene.sim._B, scene.sim.dem_solver.n_particles, 3), **args)
        self._vel = gs.zeros((self._scene.sim._B, scene.sim.dem_solver.n_particles, 3), **args)
        args["dtype"] = gs.tc_bool
        args["requires_grad"] = False
        self._active = gs.zeros((self._scene.sim._B, scene.sim.dem_solver.n_particles), **args)

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

    @property
    def c(self):
        return self._c

    @property
    def boundary_group(self):
        return self._boundary_group

    def serializable(self):
        super().serializable()
        self._c = self._c.detach()
        self._boundary_group = self._boundary_group.detach()


class PBDFluidSolverState(PBDSolverState):
    """Particle liquid state including concentrations, activation and container membership.

    The solid extensions (rest positions, cluster rest centroids, bond and seam liveness) let a Scene
    snapshot restore fracturable / plastic bodies exactly.
    """

    def __init__(self, scene):
        super().__init__(scene)
        shape = self.free.shape
        self._c = gs.zeros(shape, dtype=gs.tc_float, scene=scene)
        self._active = gs.zeros(shape, dtype=gs.tc_bool, scene=scene)
        self._boundary_group = gs.zeros(shape, dtype=gs.tc_int, scene=scene)
        solver = scene.sim.pbd_solver
        self._solid_rest = gs.zeros(self._pos.shape, dtype=gs.tc_float, scene=scene)
        self._cluster_rest_cm = gs.zeros((shape[0], solver.n_clusters, 3), dtype=gs.tc_float, scene=scene)
        self._bonds_alive = gs.zeros((shape[0], solver.n_bonds), dtype=gs.tc_bool, scene=scene)
        self._seams_alive_count = gs.zeros((shape[0], solver.n_seams), dtype=gs.tc_int, scene=scene)
        self._seams_damage = gs.zeros((shape[0], solver.n_seams), dtype=gs.tc_float, scene=scene)
        self._seams_is_dead = gs.zeros((shape[0], solver.n_seams), dtype=gs.tc_bool, scene=scene)

    def serializable(self):
        super().serializable()
        self._c = self._c.detach()
        self._active = self._active.detach()
        self._boundary_group = self._boundary_group.detach()
        self._solid_rest = self._solid_rest.detach()
        self._cluster_rest_cm = self._cluster_rest_cm.detach()
        self._bonds_alive = self._bonds_alive.detach()
        self._seams_alive_count = self._seams_alive_count.detach()
        self._seams_damage = self._seams_damage.detach()
        self._seams_is_dead = self._seams_is_dead.detach()

    @property
    def c(self):
        return self._c

    @property
    def active(self):
        return self._active

    @property
    def boundary_group(self):
        return self._boundary_group

    @property
    def solid_rest(self):
        return self._solid_rest

    @property
    def cluster_rest_cm(self):
        return self._cluster_rest_cm

    @property
    def bonds_alive(self):
        return self._bonds_alive

    @property
    def seams_alive_count(self):
        return self._seams_alive_count

    @property
    def seams_damage(self):
        return self._seams_damage

    @property
    def seams_is_dead(self):
        return self._seams_is_dead
