import math
from dataclasses import dataclass

from .config import LitterScoopTaskConfig
from .scene import LitterScoopRuntime


@dataclass(frozen=True)
class LitterScoopControllerState:
    step_index: int


@dataclass
class LitterScoopController:
    """Deterministic shovel command sequence kept separate from scene and solver configuration."""

    config: LitterScoopTaskConfig
    step_index: int = 0

    def reset(self, runtime: LitterScoopRuntime | None = None):
        self.step_index = 0
        if runtime is not None:
            if self.config != runtime.config.task:
                raise ValueError("Controller task configuration must match the scene task configuration.")
            assets = runtime.config.assets
            blade_quat = (
                math.cos(0.5 * assets.blade_angle),
                0.0,
                math.sin(0.5 * assets.blade_angle),
                0.0,
            )
            runtime.scene.dem_solver.set_sdf_obstacle_pose(assets.blade_initial_pos, blade_quat)
            runtime.scene.dem_solver.set_sdf_obstacle_vel((0.0, 0.0, 0.0))
            runtime.synchronize_shovel()

    def get_state(self) -> LitterScoopControllerState:
        return LitterScoopControllerState(step_index=self.step_index)

    def set_state(self, runtime: LitterScoopRuntime, state: LitterScoopControllerState):
        if self.config != runtime.config.task:
            raise ValueError("Controller task configuration must match the scene task configuration.")
        self.step_index = state.step_index
        runtime.synchronize_shovel()

    def before_step(self, runtime: LitterScoopRuntime):
        task = self.config
        descend = task.settle_steps
        insert = descend + task.descend_steps
        rotate = insert + task.insert_steps
        lift = rotate + task.rotate_steps
        hold = lift + task.lift_steps
        if self.step_index == descend:
            runtime.scene.dem_solver.set_sdf_obstacle_vel(task.descend_velocity)
        elif self.step_index == insert:
            runtime.scene.dem_solver.set_sdf_obstacle_vel(task.insert_velocity)
        elif self.step_index == rotate:
            runtime.scene.dem_solver.set_sdf_obstacle_vel(
                task.rotate_velocity,
                omega=task.rotate_angular_velocity,
            )
        elif self.step_index == lift:
            runtime.scene.dem_solver.set_sdf_obstacle_vel(task.lift_velocity)
        elif self.step_index == hold:
            runtime.scene.dem_solver.set_sdf_obstacle_vel((0.0, 0.0, 0.0))

    def step(self, runtime: LitterScoopRuntime):
        self.before_step(runtime)
        runtime.scene.step()
        runtime.synchronize_shovel()
        self.step_index += 1
