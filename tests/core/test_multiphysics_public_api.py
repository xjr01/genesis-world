import inspect
from dataclasses import is_dataclass

import pytest

from examples.multiphysics import butter_spreading, coffee_water, garment_folding, litter_scoop, table_wiping

SCENARIOS = (
    (coffee_water, "CoffeeWater"),
    (table_wiping, "TableWiping"),
    (litter_scoop, "LitterScoop"),
    (garment_folding, "GarmentFolding"),
    (butter_spreading, "ButterSpreading"),
)


@pytest.mark.parametrize(("scenario", "prefix"), SCENARIOS)
def test_normalized_scenario_exports_stable_adapter_surface(scenario, prefix):
    expected = {
        f"{prefix}Assets",
        f"{prefix}Controller",
        f"{prefix}ControllerState",
        f"{prefix}MaterialConfig",
        f"{prefix}Runtime",
        f"{prefix}ScenarioConfig",
        f"{prefix}SolverConfig",
        f"{prefix}TaskConfig",
        "build_scene",
    }

    assert expected <= set(scenario.__all__)
    assert expected <= vars(scenario).keys()
    assert next(iter(inspect.signature(scenario.build_scene).parameters)) == "config"

    config = vars(scenario)[f"{prefix}ScenarioConfig"]()
    for group in (config, config.solver, config.material, config.assets, config.task):
        assert is_dataclass(group)
        assert type(group).__dataclass_params__.frozen

    controller_type = vars(scenario)[f"{prefix}Controller"]
    for method in ("reset", "get_state", "set_state", "step"):
        assert callable(vars(controller_type)[method])
    assert tuple(inspect.signature(controller_type.step).parameters) == ("self", "runtime")
    assert tuple(inspect.signature(controller_type.set_state).parameters) == ("self", "runtime", "state")
    expected_snapshot_parameters = ("self", "runtime") if scenario is coffee_water else ("self",)
    assert tuple(inspect.signature(controller_type.get_state).parameters) == expected_snapshot_parameters
    assert isinstance(vars(vars(scenario)[f"{prefix}Runtime"])["control_dt"], property)
