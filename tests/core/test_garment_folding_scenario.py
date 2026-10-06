from dataclasses import replace

import numpy as np
import pytest

from examples.multiphysics.garment_folding import (
    GarmentFoldingAssets,
    GarmentFoldingScenarioConfig,
    GarmentFoldingTaskConfig,
    GarmentRobotTrajectory,
    create_scene527_config,
)
from examples.multiphysics.garment_folding.task import GarmentFoldingController, smooth_step


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


class _Robot:
    def __init__(self):
        self.qpos = None

    def set_qpos(self, qpos, *, zero_velocity):
        assert zero_velocity
        self.qpos = qpos


class _Runtime:
    def __init__(self, assets, task):
        self.scene = _Scene()
        self.jaw_origins_local = assets.jaw_origins(task.task)
        self.jaws = tuple((_Jaw(), _Jaw()) for _ in self.jaw_origins_local)
        self.garment_pos = assets.garment_pos
        self.garment_scale = assets.garment_scale
        self.jaw_half_thickness = 0.5 * assets.jaw_size[2]
        self.robot = None
        self.robot_trajectory = None


def test_garment_asset_changes_are_independent_from_solver_parameters():
    config = GarmentFoldingScenarioConfig()
    resized_table = replace(config, assets=replace(config.assets, table_size=(0.7, 0.5, 0.05)))
    tuned_contact = replace(
        config,
        material=replace(config.material, cloth_self_friction=0.8, cloth_table_friction=0.5),
    )

    assert resized_table.solver == config.solver
    assert resized_table.material == config.material
    assert resized_table.assets.table_size == (0.7, 0.5, 0.05)
    assert config.material.cloth_self_friction is None
    assert config.material.cloth_table_friction is None
    assert tuned_contact.solver == config.solver
    assert tuned_contact.assets == config.assets
    assert tuned_contact.task == config.task
    assert tuned_contact.material.cloth_self_friction == 0.8
    assert tuned_contact.material.cloth_table_friction == 0.5


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


def test_scene527_profile_uses_portable_bundle_paths_and_calibrated_physics(tmp_path):
    config = create_scene527_config(tmp_path)

    assert config.assets.asset_root == str(tmp_path)
    assert config.assets.garment_mesh == "assets/cloth/short-shirt-55068f.obj"
    assert config.assets.robot == "assets/robots/dual_x5_2025_ipc_v1/dual_x5_2025_ipc.urdf"
    assert config.assets.trajectory == "controls/trajectory.npz"
    assert config.assets.expected_garment_vertices == 27811
    assert config.assets.expected_garment_faces == 55068
    assert config.assets.expected_trajectory_frames == 4373
    assert config.assets.robot_watertighten is None
    assert config.task.task == "scene527"
    assert config.task.uses_robot_trajectory
    assert config.task.settle_steps == 60
    assert config.solver.dt == pytest.approx(1.0 / 120.0)
    assert config.solver.newton_min_iterations == 1
    assert config.solver.is_rigid_rigid_contact_enabled
    assert not config.solver.enable_genesis_rigid_collision
    assert config.solver.contact_d_hat == 0.0015
    assert config.material.cloth_self_friction == 2.0


def test_scene527_controller_interpolates_each_action_over_physics_steps():
    task = GarmentFoldingTaskConfig(task="scene527", physics_steps_per_action=2)
    controller = GarmentFoldingController(task)
    runtime = _Runtime(GarmentFoldingAssets(), task)
    runtime.robot = _Robot()
    runtime.robot_trajectory = GarmentRobotTrajectory(
        joint_q=np.array([[0.0, 1.0], [2.0, 3.0]]),
        action_fps=60.0,
        physics_steps_per_action=2,
    )

    expected = ([0.0, 1.0], [0.0, 1.0], [1.0, 2.0], [2.0, 3.0])
    for qpos in expected:
        controller.before_step(runtime)
        np.testing.assert_allclose(runtime.robot.qpos, qpos)
        controller.step_index += 1


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
