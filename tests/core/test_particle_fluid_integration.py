import pytest

import genesis as gs
from genesis.integrations import ParticleFluidProperties, ParticleFluidSolver, create_particle_fluid_setup


@pytest.mark.parametrize(
    ("solver", "options_type", "material_type"),
    [
        (ParticleFluidSolver.IPBF, gs.options.IPBFOptions, gs.materials.IPBF.Liquid),
        (ParticleFluidSolver.IPBSTF, gs.options.IPBSTFOptions, gs.materials.IPBSTF.Liquid),
        (ParticleFluidSolver.PBSTF, gs.options.PBSTFOptions, gs.materials.PBSTF.Liquid),
    ],
)
def test_particle_fluid_common_properties(solver, options_type, material_type):
    setup = create_particle_fluid_setup(
        ParticleFluidProperties(
            density=997.0,
            particle_size=0.012,
        ),
        solver,
    )

    assert isinstance(setup.solver_options, options_type)
    assert isinstance(setup.material, material_type)
    assert setup.solver_options.particle_size == 0.012
    assert setup.material.rho == 997.0
    assert setup.material.sampler == "regular"


def test_particle_fluid_rejects_conflicting_specific_options():
    with pytest.raises(gs.GenesisException, match="particle_size.*conflicts"):
        create_particle_fluid_setup(
            ParticleFluidProperties(
                particle_size=0.01,
            ),
            ParticleFluidSolver.PBSTF,
            solver_options=gs.options.PBSTFOptions(
                particle_size=0.02,
            ),
        )
