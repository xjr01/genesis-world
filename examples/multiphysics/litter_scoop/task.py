from dataclasses import dataclass

from .config import LitterScoopTaskConfig
from .scene import LitterScoopRuntime


@dataclass
class LitterScoopController:
    """Deterministic shovel command sequence kept separate from scene and solver configuration."""

    config: LitterScoopTaskConfig
    step_index: int = 0

    def reset(self):
        self.step_index = 0

    def before_step(self, runtime: LitterScoopRuntime):
        task = self.config
        descend = task.settle_steps
        insert = descend + task.descend_steps
        rotate = insert + task.insert_steps
        lift = rotate + task.rotate_steps
        hold = lift + task.lift_steps
        if self.step_index == descend:
            runtime.scene.dem_solver.set_tilt_box_vel(task.descend_velocity)
        elif self.step_index == insert:
            runtime.scene.dem_solver.set_tilt_box_vel(task.insert_velocity)
        elif self.step_index == rotate:
            runtime.scene.dem_solver.set_tilt_box_vel(
                task.rotate_velocity,
                omega=task.rotate_angular_velocity,
            )
        elif self.step_index == lift:
            runtime.scene.dem_solver.set_tilt_box_vel(task.lift_velocity)
        elif self.step_index == hold:
            runtime.scene.dem_solver.set_tilt_box_vel((0.0, 0.0, 0.0))

        runtime.shovel.set_pos(runtime.scene.dem_solver.get_tilt_box_pos()[0])
        runtime.shovel.set_quat(runtime.scene.dem_solver.get_tilt_box_quat()[0])

    def step(self, runtime: LitterScoopRuntime):
        self.before_step(runtime)
        runtime.scene.step()
        self.step_index += 1
