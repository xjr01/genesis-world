from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest

from examples.garment_folding import (
    GarmentFoldingScenarioConfig,
    GarmentRobotTrajectory,
    config as config_module,
    create_scene527_config,
)
from examples.garment_folding.task import GarmentFoldingController


class _Robot:
    def __init__(self):
        self.qpos = None

    def set_qpos(self, qpos, *, zero_velocity):
        assert zero_velocity
        self.qpos = qpos


@pytest.mark.parametrize("mesh,n_vertices,n_faces", [("8k", 4090, 8000), ("13k", 7021, 13767), ("55k", 27811, 55068)])
def test_scene527_configuration(tmp_path, mesh, n_vertices, n_faces):
    config = create_scene527_config(tmp_path, mesh=mesh)
    assert config.assets.asset_root == str(tmp_path)
    assert config.assets.garment_mesh == f"assets/cloth/short-shirt-{n_faces}f.obj"
    assert config.assets.robot == "assets/robots/dual_x5_2025_ipc_v1/dual_x5_2025_ipc.urdf"
    assert config.assets.trajectory == "controls/trajectory.npz"
    assert config.assets.expected_garment_vertices == n_vertices
    assert config.assets.expected_garment_faces == n_faces
    assert config.assets.expected_trajectory_frames == 4373
    assert config.assets.robot_watertighten is None
    assert config.task.settle_steps == 60
    assert config.solver.dt == pytest.approx(1.0 / 120.0)
    assert config.solver.is_rigid_rigid_contact_enabled
    assert not config.solver.enable_genesis_rigid_collision
    assert config.solver.contact_d_hat == 0.0015
    assert config.material.cloth_young_modulus == 20000.0
    assert config.material.cloth_thickness == 0.0001
    assert config.material.cloth_bending_stiffness == 40.0
    assert config.material.cloth_self_friction == 2.0
    resized_table = replace(config, assets=replace(config.assets, table_size=(1.0, 3.0, 0.8)))
    assert resized_table.solver == config.solver
    assert resized_table.material == config.material
    assert resized_table.task == config.task
    defaults = GarmentFoldingScenarioConfig()
    assert create_scene527_config() == defaults
    assert defaults.solver == config.solver
    assert defaults.material == config.material
    assert defaults.task == config.task
    package_root = Path(config_module.__file__).parent
    mesh_lines = (package_root / config.assets.garment_mesh).read_text().splitlines()
    assert sum(line.startswith("v ") for line in mesh_lines) == n_vertices
    assert sum(line.startswith("f ") for line in mesh_lines) == n_faces
    robot_path = package_root / defaults.assets.robot
    robot_tree = ElementTree.parse(robot_path)
    for robot_mesh in robot_tree.iter("mesh"):
        assert (robot_path.parent / robot_mesh.attrib["filename"]).is_file()
    with np.load(package_root / defaults.assets.trajectory, allow_pickle=False) as trajectory:
        assert trajectory["joint_q"].shape == (4373, len(trajectory["joint_names"]))
        assert np.isfinite(trajectory["joint_q"]).all()
        robot_joint_names = {
            joint.attrib["name"] for joint in robot_tree.iter("joint") if joint.attrib["type"] != "fixed"
        }
        assert set(trajectory["joint_names"]) == robot_joint_names


def test_scene527_interpolation_and_resume():
    config = GarmentFoldingScenarioConfig()
    runtime = SimpleNamespace(
        config=config,
        robot=_Robot(),
        robot_trajectory=GarmentRobotTrajectory(
            joint_q=np.array([[0.0, 1.0], [2.0, 3.0]]),
            action_fps=config.task.action_fps,
            physics_steps_per_action=config.task.physics_steps_per_action,
        ),
    )
    controller = GarmentFoldingController(config.task)
    expected = ([0.0, 1.0], [0.0, 1.0], [1.0, 2.0], [2.0, 3.0])
    for qpos in expected:
        controller.before_step(runtime)
        np.testing.assert_allclose(runtime.robot.qpos, qpos)
        controller.step_index += 1
    with pytest.raises(RuntimeError, match="complete"):
        controller.before_step(runtime)
    checkpoint = controller.get_state()
    controller.reset(runtime)
    assert controller.step_index == 0
    controller.set_state(runtime, checkpoint)
    assert controller.step_index == runtime.robot_trajectory.physics_steps
