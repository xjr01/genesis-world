from dataclasses import dataclass

from .config import GarmentFoldingTaskConfig
from .scene import GarmentFoldingRuntime


def step_settling(runtime: GarmentFoldingRuntime):
    """Hold the first recorded pose while the cloth falls onto the table."""
    runtime.robot.set_qpos(runtime.robot_trajectory.joint_q[0], zero_velocity=True)
    runtime.scene.step()


@dataclass
class GarmentFoldingController:
    """Interpolate recorded Scene527 joint targets over physics steps."""

    config: GarmentFoldingTaskConfig
    step_index: int = 0

    def reset(self, runtime: GarmentFoldingRuntime | None = None):
        if runtime is not None and self.config != runtime.config.task:
            raise ValueError("Controller task configuration must match the scene task configuration.")
        self.step_index = 0

    def get_state(self) -> int:
        return self.step_index

    def set_state(self, runtime: GarmentFoldingRuntime, state: int):
        if self.config != runtime.config.task:
            raise ValueError("Controller task configuration must match the scene task configuration.")
        self.step_index = state

    def before_step(self, runtime: GarmentFoldingRuntime):
        if self.step_index >= runtime.robot_trajectory.physics_steps:
            raise RuntimeError("Garment robot trajectory is complete.")
        action_index = self.step_index // runtime.robot_trajectory.physics_steps_per_action
        physics_index = self.step_index % runtime.robot_trajectory.physics_steps_per_action
        target_q = runtime.robot_trajectory.joint_q[action_index]
        previous_q = runtime.robot_trajectory.joint_q[max(0, action_index - 1)]
        alpha = (physics_index + 1) / runtime.robot_trajectory.physics_steps_per_action
        runtime.robot.set_qpos(previous_q + alpha * (target_q - previous_q), zero_velocity=True)

    def step(self, runtime: GarmentFoldingRuntime):
        self.before_step(runtime)
        runtime.scene.step()
        self.step_index += 1
