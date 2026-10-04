from examples import pbstf_coffee_water
from examples.multiphysics.coffee_water import (
    CoffeeWaterController,
    CoffeeWaterRuntime,
    CoffeeWaterScenarioConfig,
    build_scene,
)


def test_coffee_water_public_api():
    config = CoffeeWaterScenarioConfig()
    controller = CoffeeWaterController(config.task)

    assert controller.config is config.task
    assert pbstf_coffee_water.build_scene is build_scene
    assert pbstf_coffee_water.CoffeeWaterRuntime is CoffeeWaterRuntime
