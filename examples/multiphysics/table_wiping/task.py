from dataclasses import dataclass

import numpy as np

from .config import TableWipingTaskConfig
from .scene import TableWipingRuntime


def wipe_pose(time: float, config: TableWipingTaskConfig) -> tuple[float, float, float]:
    progress = min(max((time - config.settle_time) / config.wipe_time, 0.0), 1.0)
    return tuple(start + progress * (end - start) for start, end in zip(config.start_pos, config.end_pos))


@dataclass(frozen=True)
class TableWipingControllerState:
    step_index: int
    qpos: object | None


@dataclass
class TableWipingController:
    config: TableWipingTaskConfig
    step_index: int = 0
    qpos: object | None = None

    def reset(self, runtime: TableWipingRuntime):
        if self.config != runtime.config.task:
            raise ValueError("Controller task configuration must match the scene task configuration.")
        self.step_index = 0
        self.qpos = runtime.initial_qpos.clone()
        runtime.robot.set_qpos(self.qpos, zero_velocity=True)
        runtime.scene.pbstf_solver.set_static_colliders_pose(
            pos=self.config.start_pos,
            quat=self.config.quat,
            colliders_idx=1,
        )

    def get_state(self) -> TableWipingControllerState:
        return TableWipingControllerState(
            step_index=self.step_index,
            qpos=None if self.qpos is None else self.qpos.clone(),
        )

    def set_state(self, runtime: TableWipingRuntime, state: TableWipingControllerState):
        if self.config != runtime.config.task:
            raise ValueError("Controller task configuration must match the scene task configuration.")
        self.step_index = state.step_index
        self.qpos = None if state.qpos is None else state.qpos.clone()

    def step(self, runtime: TableWipingRuntime):
        if self.qpos is None:
            self.reset(runtime)
        assets = runtime.config.assets
        task_time = self.step_index * runtime.config.solver.dt
        wipe_pos = wipe_pose(task_time, self.config)
        grip_phase = min(max((task_time + runtime.config.solver.dt) / self.config.settle_time, 0.0), 1.0)
        grip_progress = grip_phase**3 * (grip_phase * (grip_phase * 6.0 - 15.0) + 10.0)
        finger_qpos = self.config.finger_open_qpos + grip_progress * (
            self.config.finger_closed_qpos - self.config.finger_open_qpos
        )
        target_pos = (
            wipe_pos[0] + 0.5 * (assets.sponge_lower[0] + assets.sponge_upper[0]),
            wipe_pos[1] + assets.sponge_upper[1],
            wipe_pos[2] + 0.5 * (assets.sponge_lower[2] + assets.sponge_upper[2]),
        )
        target_pos = np.broadcast_to(np.array(target_pos), (*self.qpos.shape[:-1], 3)).copy()
        target_quat = np.broadcast_to(np.array(self.config.grasp_quat), (*self.qpos.shape[:-1], 4)).copy()
        self.qpos = runtime.robot.inverse_kinematics(
            link=runtime.robot.get_link("panda_link7"),
            pos=target_pos,
            quat=target_quat,
            local_point=self.config.tool_center_point,
            init_qpos=self.qpos,
            pos_tol=1.0e-4,
            rot_tol=1.0e-4,
            dofs_idx_local=range(7),
        )
        self.qpos[..., -2:] = finger_qpos
        runtime.robot.set_qpos(self.qpos, zero_velocity=True)
        runtime.scene.pbstf_solver.set_static_colliders_pose(
            pos=wipe_pos,
            quat=self.config.quat,
            colliders_idx=1,
        )
        runtime.scene.step()
        self.step_index += 1
