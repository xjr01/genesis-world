from dataclasses import replace

import numpy as np
import pytest

from examples.multiphysics.garment_folding import (
    GarmentFoldingAssets,
    GarmentFoldingScenarioConfig,
    GarmentFoldingTaskConfig,
)
from examples.multiphysics.garment_folding.task import GarmentFoldingController
from examples.multiphysics.garment_folding.task import smooth_step


class _Scene:
    dt = 0.02


class _Jaw:
    def __init__(self):
        self.pos = None
        self.quat = None

    def set_pos(self, pos):
        self.pos = pos

    def set_quat(self, quat):
        self.quat = quat


class _Runtime:
    def __init__(self, assets, task):
        self.scene = _Scene()
        self.jaw_origins_local = assets.jaw_origins(task.task)
        self.jaws = tuple((_Jaw(), _Jaw()) for _ in self.jaw_origins_local)
        self.garment_pos = assets.garment_pos
        self.garment_scale = assets.garment_scale
        self.jaw_half_thickness = 0.5 * assets.jaw_size[2]


def test_garment_asset_changes_are_independent_from_solver_parameters():
    config = GarmentFoldingScenarioConfig()
    resized_table = replace(config, assets=replace(config.assets, table_size=(0.7, 0.5, 0.05)))

    assert resized_table.solver == config.solver
    assert resized_table.material == config.material
    assert resized_table.assets.table_size == (0.7, 0.5, 0.05)


def test_half_fold_is_the_validated_default():
    task = GarmentFoldingTaskConfig()
    assets = GarmentFoldingAssets()

    assert task.task == "half"
    assert task.is_validated
    assert task.resolved_duration == 16.0
    assert assets.jaw_origins(task.task) == (assets.sleeve_left, assets.hem_left)


def test_experimental_tasks_keep_their_distinct_layout_and_duration():
    task = GarmentFoldingTaskConfig(task="fold")
    assets = GarmentFoldingAssets()

    assert not task.is_validated
    assert task.resolved_duration == 30.0
    assert assets.jaw_origins(task.task) == (assets.sleeve_left, assets.sleeve_right)


def test_half_fold_initial_jaw_commands_match_asset_placement():
    task = GarmentFoldingTaskConfig()
    assets = GarmentFoldingAssets(garment_pos=(0.4, -0.2, 0.1), garment_scale=1.5)
    controller = GarmentFoldingController(task)
    runtime = _Runtime(assets, task)

    controller.before_step(runtime)

    for origin_local, jaws in zip(runtime.jaw_origins_local, runtime.jaws):
        origin = np.array(assets.garment_pos) + assets.garment_scale * np.array(origin_local)
        offset = 0.5 * task.open_gap + runtime.jaw_half_thickness
        np.testing.assert_allclose(jaws[0].pos, (origin[0], origin[1], origin[2] - offset))
        np.testing.assert_allclose(jaws[1].pos, (origin[0], origin[1], origin[2] + offset))
        np.testing.assert_allclose(jaws[0].quat, (1.0, 0.0, 0.0, 0.0))
        np.testing.assert_allclose(jaws[1].quat, (1.0, 0.0, 0.0, 0.0))


def test_quarter_regrasp_rotates_around_moved_garment_origin():
    task = GarmentFoldingTaskConfig(task="quarter")
    assets = GarmentFoldingAssets(garment_pos=(0.4, -0.2, 0.1), garment_scale=1.5)
    controller = GarmentFoldingController(task)
    runtime = _Runtime(assets, task)
    runtime.hem_points = None
    hem_y = assets.garment_pos[1] + assets.garment_scale * assets.hem_center[1]
    controller.hem_points = (
        np.array([0.4, hem_y, 0.2]),
        np.array([0.5, hem_y, 0.2]),
    )
    phase_time = 7.5
    controller.step_index = round((16.0 + phase_time) / runtime.scene.dt)

    controller.before_step(runtime)

    angle = np.pi * smooth_step((phase_time - 4.0) / 7.0)
    expected_y = assets.garment_pos[1] + (hem_y - assets.garment_pos[1]) * np.cos(angle)
    for jaws in runtime.jaws:
        jaw_center = 0.5 * (jaws[0].pos + jaws[1].pos)
        assert jaw_center[1] == pytest.approx(expected_y)
