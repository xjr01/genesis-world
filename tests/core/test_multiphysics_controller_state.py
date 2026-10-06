from types import SimpleNamespace

import numpy as np
import torch

from examples.multiphysics.butter_spreading import (
    ButterSpreadingController,
    ButterSpreadingControllerState,
    ButterSpreadingScenarioConfig,
)
from examples.multiphysics.coffee_water import (
    CoffeeWaterController,
    CoffeeWaterScenarioConfig,
)
from examples.multiphysics.coffee_water.implementation import MotionState, Phase, ToolPose
from examples.multiphysics.garment_folding import GarmentFoldingController, GarmentFoldingScenarioConfig
from examples.multiphysics.litter_scoop import LitterScoopController, LitterScoopScenarioConfig
from examples.multiphysics.table_wiping import TableWipingController, TableWipingScenarioConfig


def _motion_state():
    pose = ToolPose(np.zeros(3), np.array((1.0, 0.0, 0.0, 0.0)))
    return MotionState(Phase.GRASP_CUP, 0.0, pose, pose, pose)


def test_coffee_controller_checkpoint_is_an_independent_snapshot():
    config = CoffeeWaterScenarioConfig()
    runtime = SimpleNamespace(
        config=config,
        qpos=torch.arange(4, dtype=torch.float32),
        motion=_motion_state(),
    )
    controller = CoffeeWaterController(config.task, step_index=12, ik_error=0.25)

    state = controller.get_state(runtime)
    runtime.qpos.zero_()
    runtime.motion.spilled_particles = 7
    controller.step_index = 0
    controller.set_state(runtime, state)

    torch.testing.assert_close(runtime.qpos, torch.arange(4, dtype=torch.float32))
    assert runtime.motion.spilled_particles == 0
    assert controller.step_index == 12
    assert controller.ik_error == 0.25


def test_table_controller_checkpoint_clones_ik_seed():
    config = TableWipingScenarioConfig()
    controller = TableWipingController(config.task, step_index=17, qpos=torch.arange(3, dtype=torch.float32))
    state = controller.get_state()
    controller.qpos.zero_()
    controller.step_index = 0
    runtime = SimpleNamespace(config=config)

    controller.set_state(runtime, state)

    torch.testing.assert_close(controller.qpos, torch.arange(3, dtype=torch.float32))
    assert controller.step_index == 17


def test_litter_controller_checkpoint_preserves_phase_index():
    config = LitterScoopScenarioConfig()
    controller = LitterScoopController(config.task, step_index=123)

    state = controller.get_state()

    assert state.step_index == 123


def test_garment_controller_checkpoint_copies_regrasp_points():
    config = GarmentFoldingScenarioConfig()
    controller = GarmentFoldingController(config.task, step_index=456)
    controller.hem_points = (np.ones(3), np.full(3, 2.0))

    state = controller.get_state()
    controller.hem_points[0][0] = 99.0
    runtime = SimpleNamespace(config=config)
    controller.set_state(runtime, state)

    np.testing.assert_array_equal(controller.hem_points[0], np.ones(3))
    assert controller.step_index == 456


def test_butter_controller_checkpoint_preserves_task_time():
    config = ButterSpreadingScenarioConfig()
    controller = ButterSpreadingController(config.task, config.solver.dt, step_index=23)
    state = controller.get_state()
    controller.step_index = 0
    runtime = SimpleNamespace(config=config)

    controller.set_state(runtime, state)

    assert controller.get_state() == ButterSpreadingControllerState(step_index=23)
