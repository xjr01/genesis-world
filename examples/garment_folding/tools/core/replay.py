"""Sample garment and robot states for CPU inspection of action segments."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from genesis.utils.misc import tensor_to_array

from .trajectory import file_hash


@dataclass
class ReplayRecorder:
    steps: np.ndarray
    cloth_pos: np.ndarray
    cloth_vel: np.ndarray
    robot_q: np.ndarray
    sample_index: int = 0

    @classmethod
    def create(cls, runtime, start_step: int, n_steps: int):
        steps = np.unique(np.append(np.arange(start_step, start_step + n_steps + 1, 2), start_step + n_steps))
        shape = (len(steps), runtime.garment.n_vertices, 3)
        return cls(steps, np.empty(shape), np.empty(shape), np.empty((len(steps), runtime.robot.n_qs)))

    def capture(self, runtime, step: int, garment_state=None):
        if self.sample_index >= len(self.steps) or step != self.steps[self.sample_index]:
            return
        state = runtime.garment.get_state() if garment_state is None else garment_state
        self.cloth_pos[self.sample_index] = tensor_to_array(state.pos).reshape(-1, 3)
        self.cloth_vel[self.sample_index] = tensor_to_array(state.vel).reshape(-1, 3)
        self.robot_q[self.sample_index] = tensor_to_array(runtime.robot.get_qpos())
        self.sample_index += 1

    def save(self, path: Path, runtime):
        if self.sample_index != len(self.steps):
            raise RuntimeError("Replay sample count differs from the planned action interval.")
        root = Path(__file__).resolve().parents[1]
        if runtime.config.assets.asset_root is not None:
            root = Path(runtime.config.assets.asset_root)
        np.savez_compressed(
            path,
            cloth_pos=self.cloth_pos,
            cloth_vel=self.cloth_vel,
            robot_q=self.robot_q,
            physics_steps=self.steps,
            source_frames=self.steps // 2 - 1,
            joint_names=tuple(joint.name for joint in runtime.robot.joints if joint.n_qs),
            cloth_sha256=file_hash(root / runtime.config.assets.garment_mesh),
            physics_dt=runtime.config.solver.dt,
        )
