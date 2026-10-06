import inspect

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
        f"{prefix}Controller",
        f"{prefix}ControllerState",
        f"{prefix}Runtime",
        f"{prefix}ScenarioConfig",
        "build_scene",
    }

    assert expected <= set(scenario.__all__)
    assert all(hasattr(scenario, name) for name in expected)
    assert next(iter(inspect.signature(scenario.build_scene).parameters)) == "config"

    controller_type = getattr(scenario, f"{prefix}Controller")
    for method in ("reset", "get_state", "set_state", "step"):
        assert callable(getattr(controller_type, method))
