from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import CoffeeWaterTaskConfig
from .implementation import MotionTarget, SceneObservation, observe_scene, step_scene, update_motion

if TYPE_CHECKING:
    from .scene import CoffeeWaterRuntime


@dataclass
class CoffeeWaterController:
    config: CoffeeWaterTaskConfig
    step_index: int = 0
    observation: SceneObservation | None = None
    ik_error: float | None = None

    def reset(self, runtime: "CoffeeWaterRuntime"):
        if self.config != runtime.config.task:
            raise ValueError("Controller task configuration must match the scene task configuration.")
        self.step_index = 0
        self.observation = observe_scene(runtime)
        self.ik_error = None

    def step(self, runtime: "CoffeeWaterRuntime") -> MotionTarget:
        if self.observation is None:
            self.reset(runtime)
        time = (self.step_index + 1) * runtime.config.solver.control_dt
        target, self.ik_error = update_motion(runtime, time, self.observation)
        step_scene(runtime)
        self.observation = observe_scene(runtime)
        self.step_index += 1
        return target
