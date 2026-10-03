from dataclasses import replace

import pytest

import genesis as gs
from examples.multiphysics.coffee_water import CoffeeWaterScenarioConfig
from examples.multiphysics.litter_scoop import LitterScoopScenarioConfig
from examples.multiphysics.table_wiping import TableWipingScenarioConfig
from genesis.integrations import GranularFluidProperties, create_granular_fluid_setup


def test_granular_fluid_common_properties():
    properties = GranularFluidProperties(
        particle_size=0.0125,
        sand_density=2.7,
        liquid_density=998.0,
        lower_bound=(-0.4, -0.3, 0.0),
        upper_bound=(0.4, 0.3, 0.8),
    )
    setup = create_granular_fluid_setup(properties)

    assert setup.dem_options.particle_size == properties.particle_size
    assert setup.dem_options.lower_bound == properties.lower_bound
    assert setup.flip_options.upper_bound == properties.upper_bound
    assert setup.sand_material.rho == properties.sand_density
    assert setup.liquid_material.rho == properties.liquid_density


def test_granular_fluid_rejects_conflicting_domain():
    properties = GranularFluidProperties()
    with pytest.raises(gs.GenesisException, match="bounds conflict"):
        create_granular_fluid_setup(
            properties,
            flip_options=gs.options.FLIPOptions(
                lower_bound=(-1.0, -1.0, 0.0),
                upper_bound=(1.0, 1.0, 1.0),
            ),
        )


def test_litter_scoop_asset_change_does_not_change_solver_config():
    config = LitterScoopScenarioConfig()
    moved = replace(config, assets=replace(config.assets, blade_initial_pos=(0.1, 0.0, 0.4)))

    assert moved.assets.blade_initial_pos == (0.1, 0.0, 0.4)
    assert moved.solver == config.solver
    assert moved.material == config.material


def test_scenario_asset_changes_are_independent_from_solver_parameters():
    coffee = CoffeeWaterScenarioConfig()
    moved_coffee = replace(
        coffee,
        assets=replace(coffee.assets, water_cup_pos=(0.2, -0.045, 0.0)),
    )
    wiping = TableWipingScenarioConfig()
    resized_table = replace(
        wiping,
        assets=replace(wiping.assets, table_size=(1.0, 1.0 / 30.0, 0.6)),
    )

    assert moved_coffee.solver == coffee.solver
    assert moved_coffee.assets.water_cup_pos == (0.2, -0.045, 0.0)
    assert resized_table.solver == wiping.solver
    assert resized_table.assets.table_size == (1.0, 1.0 / 30.0, 0.6)
