import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import genesis as gs

from .config import CoffeeWaterTaskConfig
from .implementation import (
    POUR_QUAT,
    MotionState,
    MotionTarget,
    SceneObservation,
    ToolPose,
    grasp_pose,
    observe_scene,
    step_scene,
    update_motion,
)

if TYPE_CHECKING:
    from .scene import CoffeeWaterRuntime


@dataclass(frozen=True)
class CoffeeWaterControllerState:
    step_index: int
    observation: SceneObservation | None
    ik_error: float | None
    qpos: gs.Tensor
    motion: MotionState


@dataclass
class CoffeeWaterController:
    config: CoffeeWaterTaskConfig
    step_index: int = 0
    observation: SceneObservation | None = None
    ik_error: float | None = None

    def reset(self, runtime: "CoffeeWaterRuntime"):
        if self.config != runtime.config.task:
            raise ValueError("Controller task configuration must match the scene task configuration.")
        runtime.qpos[...] = runtime.robot.get_qpos()
        cup_grasp = ToolPose(np.array((0.0, self.config.cup_grip_height, 0.0)), POUR_QUAT)
        right_start = grasp_pose(
            ToolPose(np.array(runtime.config.assets.water_cup_pos), np.array((1.0, 0.0, 0.0, 0.0))),
            cup_grasp,
        )
        runtime.motion.reset(right_start, cup_grasp)
        self.step_index = 0
        self.observation = observe_scene(runtime)
        self.ik_error = None
        update_motion(runtime, time=0.0, observation=self.observation)
        runtime.robot.set_qpos(runtime.qpos)
        self.observation = observe_scene(runtime)

    def get_state(self, runtime: "CoffeeWaterRuntime") -> CoffeeWaterControllerState:
        """Snapshot task phase state that must accompany a mid-trajectory Scene state."""
        return CoffeeWaterControllerState(
            step_index=self.step_index,
            observation=copy.deepcopy(self.observation),
            ik_error=self.ik_error,
            qpos=runtime.qpos.clone(),
            motion=copy.deepcopy(runtime.motion),
        )

    def set_state(self, runtime: "CoffeeWaterRuntime", state: CoffeeWaterControllerState):
        """Restore task phase state after the matching Scene state has been restored."""
        if self.config != runtime.config.task:
            raise ValueError("Controller task configuration must match the scene task configuration.")
        runtime.motion.restore(state.motion)
        runtime.qpos[...] = state.qpos
        self.step_index = state.step_index
        self.observation = copy.deepcopy(state.observation)
        self.ik_error = state.ik_error

    def step(self, runtime: "CoffeeWaterRuntime") -> MotionTarget:
        if self.observation is None:
            self.reset(runtime)
        time = (self.step_index + 1) * runtime.config.solver.control_dt
        target, self.ik_error = update_motion(runtime, time, self.observation)
        step_scene(runtime)
        self.observation = observe_scene(runtime)
        self.step_index += 1
        return target
