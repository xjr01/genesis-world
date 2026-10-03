import enum
import math
from dataclasses import dataclass

import genesis as gs
from genesis.engine import materials
from genesis.engine.solvers import IPBSTFSolver, PBSTFSolver
from genesis.options.options import Options
from genesis.options.solvers import IPBFOptions, IPBSTFOptions, PBSTFOptions
from genesis.typing import PositiveFloat


class ParticleFluidSolver(enum.Enum):
    """Particle-fluid implementations covered by the stable integration helpers."""

    IPBF = "ipbf"
    IPBSTF = "ipbstf"
    PBSTF = "pbstf"


class ParticleFluidProperties(Options):
    """Solver-independent particle-fluid properties accepted from an adapter.

    Density maps to the liquid material and particle size maps to the solver. Solver-specific viscosity, surface
    tension, concentration and boundary settings stay in the corresponding Genesis material or solver options because
    their meanings and units differ between implementations.
    """

    density: PositiveFloat = 1000.0
    particle_size: PositiveFloat = 0.02


@dataclass(frozen=True)
class ParticleFluidSetup:
    """Validated solver options and material produced from common particle-fluid properties."""

    solver: ParticleFluidSolver
    solver_options: IPBFOptions | IPBSTFOptions | PBSTFOptions
    material: materials.IPBF.Liquid | materials.IPBSTF.Liquid | materials.PBSTF.Liquid


def create_particle_fluid_setup(
    properties: ParticleFluidProperties,
    solver: ParticleFluidSolver,
    solver_options: IPBFOptions | IPBSTFOptions | PBSTFOptions | None = None,
    material: materials.IPBF.Liquid | materials.IPBSTF.Liquid | materials.PBSTF.Liquid | None = None,
) -> ParticleFluidSetup:
    """Map common fluid properties onto one Genesis solver without hiding solver-specific settings.

    Passing explicit solver options or a material preserves every implementation-specific parameter. Their common
    particle size and density must agree with ``properties`` so an adapter cannot silently configure two conflicting
    values for the same physical quantity.
    """
    if not isinstance(properties, ParticleFluidProperties):
        gs.raise_exception("`properties` requires `ParticleFluidProperties`.")
    if not isinstance(solver, ParticleFluidSolver):
        gs.raise_exception("`solver` requires `ParticleFluidSolver`.")

    if solver is ParticleFluidSolver.IPBF:
        if solver_options is None:
            solver_options = IPBFOptions(particle_size=properties.particle_size)
        if material is None:
            material = materials.IPBF.Liquid(rho=properties.density, sampler="regular")
        expected_options_type = IPBFOptions
        expected_material_type = materials.IPBF.Liquid
    elif solver is ParticleFluidSolver.IPBSTF:
        if solver_options is None:
            solver_options = IPBSTFOptions(particle_size=properties.particle_size)
        if material is None:
            material = materials.IPBSTF.Liquid(rho=properties.density, sampler="regular")
        expected_options_type = IPBSTFOptions
        expected_material_type = materials.IPBSTF.Liquid
    else:
        if solver_options is None:
            solver_options = PBSTFOptions(particle_size=properties.particle_size)
        if material is None:
            material = materials.PBSTF.Liquid(rho=properties.density, sampler="regular")
        expected_options_type = PBSTFOptions
        expected_material_type = materials.PBSTF.Liquid

    if not isinstance(solver_options, expected_options_type):
        gs.raise_exception(f"{solver.value} requires `{expected_options_type.__name__}`.")
    if not isinstance(material, expected_material_type):
        gs.raise_exception(f"{solver.value} requires `{expected_material_type.__name__}`.")
    if not math.isclose(solver_options.particle_size, properties.particle_size):
        gs.raise_exception("Common `particle_size` conflicts with the solver-specific options.")
    if not math.isclose(material.rho, properties.density):
        gs.raise_exception("Common `density` conflicts with the solver-specific material.")
    return ParticleFluidSetup(solver=solver, solver_options=solver_options, material=material)


class StaticColliderLinkSynchronizer:
    """Copy selected rigid-link world poses to selected PBSTF or IPBSTF colliders before every scene step.

    Construct this after ``scene.build()`` and register the instance with ``scene.register_pre_step_callback``. Link
    and collider selections are positional: the first link drives the first collider, and so on.
    """

    def __init__(self, scene, solver, links_idx, colliders_idx):
        if not isinstance(solver, (IPBSTFSolver, PBSTFSolver)):
            gs.raise_exception("Static collider link synchronization requires an IPBSTF or PBSTF solver.")
        if solver.scene is not scene:
            gs.raise_exception("The synchronized solver must belong to the supplied scene.")
        if len(links_idx) != len(colliders_idx):
            gs.raise_exception("`links_idx` and `colliders_idx` must have the same length.")
        if len(links_idx) == 0:
            gs.raise_exception("Static collider link synchronization requires at least one link and collider.")
        self._scene = scene
        self._solver = solver
        self._links_idx = tuple(links_idx)
        self._colliders_idx = tuple(colliders_idx)

    def sync(self, envs_idx=None):
        """Synchronize collider poses immediately for all or selected parallel environments."""
        self._solver.set_static_colliders_pose(
            self._scene.rigid_solver.get_links_pos(self._links_idx, envs_idx, relative=False),
            self._scene.rigid_solver.get_links_quat(self._links_idx, envs_idx, relative=False),
            colliders_idx=self._colliders_idx,
            envs_idx=envs_idx,
        )

    def __call__(self):
        self.sync()
        return False
