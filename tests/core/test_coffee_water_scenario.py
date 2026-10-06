import numpy as np

from examples import pbstf_coffee_water
from examples.multiphysics.coffee_water import (
    CoffeeWaterController,
    CoffeeWaterRuntime,
    CoffeeWaterScenarioConfig,
    build_scene,
)
from examples.multiphysics.coffee_water.implementation import (
    MOP_QUAT,
    MotionState,
    Phase,
    SceneObservation,
    ToolPose,
    motion_target,
)


def test_coffee_water_public_api():
    config = CoffeeWaterScenarioConfig()
    controller = CoffeeWaterController(config.task)

    assert controller.config is config.task
    assert pbstf_coffee_water.build_scene is build_scene
    assert pbstf_coffee_water.CoffeeWaterRuntime is CoffeeWaterRuntime


def test_grasp_sponge_phase_uses_configured_grip_height():
    config = CoffeeWaterScenarioConfig()
    hand = ToolPose(np.zeros(3), MOP_QUAT.copy())
    motion = MotionState(Phase.GRASP_SPONGE, 14.0, hand, hand, hand)
    observation = SceneObservation(
        cup=hand,
        rod=hand,
        hands=(hand, hand),
        cup_tilt=0.0,
        water_in_cup=0,
        is_cup_grasped=True,
        cup_grasp_span=0.0,
        sponge_pos=np.asarray(config.assets.sponge_start),
    )

    target = motion_target(15.0, motion, observation, config)

    expected_height = config.assets.sponge_size[1] - 0.005 - 0.5 * config.assets.sponge_size[1]
    assert motion.sponge_grasp_offset[1] == expected_height
    assert target.right_phase is Phase.GRASP_SPONGE
