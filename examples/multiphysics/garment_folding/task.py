from dataclasses import dataclass

import numpy as np

from genesis.utils.misc import tensor_to_array

from .config import GarmentFoldingTaskConfig
from .scene import GarmentFoldingRuntime


def smooth_step(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def smooth_lerp(start, end, value):
    return start + (end - start) * smooth_step(value)


@dataclass
class GarmentFoldingController:
    """Advance a contact-only garment task using prescribed rigid-jaw targets."""

    config: GarmentFoldingTaskConfig
    step_index: int = 0
    hem_points: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self):
        self.step_index = 0
        self.hem_points = None

    def _capture_hem_points(self, runtime: GarmentFoldingRuntime):
        positions = tensor_to_array(runtime.garment.get_state().pos).reshape(-1, 3)
        if self.config.task == "quarter":
            points = []
            for name in ("hem_center", "hem_right"):
                indices = runtime.landmarks.indices(name)
                point = np.take(positions, indices, axis=0).mean(axis=0)
                is_near = np.linalg.norm(positions[:, :2] - point[:2], axis=1) < 0.013
                point[2] = np.median(positions[is_near, 2])
                points.append(point)
            self.hem_points = (points[0], points[1])
        else:
            self.hem_points = tuple(
                np.take(positions, runtime.landmarks.indices(name), axis=0).mean(axis=0)
                for name in ("hem_left", "hem_right")
            )

    def before_step(self, runtime: GarmentFoldingRuntime):
        dt = runtime.scene.dt
        time = self.step_index * dt
        if self.config.task == "fold" and self.hem_points is None and time >= 14.0:
            self._capture_hem_points(runtime)
        if self.config.task == "quarter" and self.hem_points is None and time >= 16.0:
            self._capture_hem_points(runtime)

        garment_pos = np.array(runtime.garment_pos)
        garment_scale = runtime.garment_scale
        for side, (origin_values, jaws) in enumerate(zip(runtime.jaw_origins_local, runtime.jaws)):
            origin = np.array(origin_values)
            position = origin.copy()
            is_position_local = True
            roll = 0.0
            pitch = 0.0
            close = 0.0
            opening = 0.0
            if self.config.task == "grasp":
                close = smooth_step(time / 0.6)
                position[2] += 0.2 * smooth_step((time - 1.0) / 1.5)
                pitch = 0.5 * np.pi * smooth_step((time - 2.5) / 1.5)
                opening = 0.04 * smooth_step((time - 5.0) / 0.5)
            elif self.config.task == "half" or (self.config.task == "quarter" and time < 16.0):
                angle = np.pi * smooth_step((time - 1.0) / 8.0)
                radius = -origin[0]
                position = np.array(
                    [
                        -radius * np.cos(angle),
                        origin[1],
                        origin[2] + radius * np.sin(angle) + 0.02 * smooth_step((time - 7.0) / 2.0),
                    ]
                )
                pitch = angle
                close = smooth_step(time / 0.6)
                opening = 0.03 * smooth_step((time - 9.0) / 0.3)
                if time >= 9.3:
                    position[0] += 0.14 * smooth_step((time - 9.3) / 1.7)
                if time >= 11.0:
                    position[2] += 0.3 * smooth_step(time - 11.0)
            elif self.config.task == "quarter":
                if self.hem_points is None:
                    raise RuntimeError("The quarter-fold regrasp requires captured hem points.")
                is_position_local = False
                hem = self.hem_points[side]
                phase_time = time - 16.0
                outside = hem + np.array([0.0, -0.045 * garment_scale, 0.0])
                if phase_time < 1.0:
                    start = garment_pos + garment_scale * np.array(
                        [-origin[0] + 0.14, origin[1], origin[2] + 0.32]
                    )
                    position = smooth_lerp(
                        start,
                        outside + np.array([0.0, 0.0, 0.10 * garment_scale]),
                        phase_time,
                    )
                    pitch = np.pi * (1.0 - smooth_step(phase_time))
                    opening = 0.013 * (1.0 - smooth_step(phase_time))
                elif phase_time < 2.0:
                    position = smooth_lerp(
                        outside + np.array([0.0, 0.0, 0.10 * garment_scale]),
                        outside,
                        phase_time - 1.0,
                    )
                elif phase_time < 3.0:
                    position = smooth_lerp(outside, hem, phase_time - 2.0)
                elif phase_time < 4.0:
                    position = hem.copy()
                    close = smooth_step(phase_time - 3.0)
                elif phase_time < 11.0:
                    angle = np.pi * smooth_step((phase_time - 4.0) / 7.0)
                    close = 1.0
                    roll = -angle
                    position = np.array(
                        [
                            hem[0],
                            runtime.garment_pos[1]
                            + (hem[1] - runtime.garment_pos[1]) * np.cos(angle),
                            hem[2]
                            - 0.90 * hem[1] * np.sin(angle)
                            + 0.05 * garment_scale * smooth_step((phase_time - 9.0) / 2.0),
                        ]
                    )
                else:
                    local_hem_y = (hem[1] - runtime.garment_pos[1]) / garment_scale
                    position = np.array(
                        [
                            hem[0],
                            runtime.garment_pos[1] - garment_scale * local_hem_y,
                            hem[2] + 0.05 * garment_scale,
                        ]
                    )
                    roll = -np.pi
                    close = 1.0
                    opening = 0.03 * smooth_step((phase_time - 11.0) / 0.3)
                    if phase_time >= 11.3:
                        position[1] += 0.15 * garment_scale * smooth_step(phase_time - 11.3)
                    if phase_time >= 12.3:
                        position[2] += 0.3 * garment_scale * smooth_step(phase_time - 12.3)
                    if phase_time >= 13.3:
                        destination = garment_pos + garment_scale * np.array(
                            [(-1.0 if side == 0 else 1.0) * 0.35, 0.1, 0.4]
                        )
                        position = smooth_lerp(position, destination, phase_time - 13.3)
            else:
                sign = 1.0 if side else -1.0
                local_time = time - (0.0 if side == 0 else 7.0)
                if time < 14.0:
                    angle = np.pi * smooth_step((local_time - 1.0) / 3.0)
                    close = smooth_step(local_time / 0.6)
                    pitch = -sign * angle
                    position = np.array(
                        [
                            sign * (0.165 + 0.124 * np.cos(angle)),
                            0.15,
                            0.0625 + 0.14 * np.sin(angle) + 0.0275 * smooth_step((local_time - 3.0) / 1.0),
                        ]
                    )
                    if local_time >= 4.0:
                        opening = 0.03 * smooth_step((local_time - 4.0) / 0.2)
                    if local_time >= 4.2:
                        position[0] -= sign * 0.15 * smooth_step((local_time - 4.2) / 1.0)
                    if local_time >= 5.2:
                        position[2] = 0.09 + 0.21 * smooth_step((local_time - 5.2) / 0.5)
                    if local_time >= 5.7:
                        position = smooth_lerp(
                            position,
                            np.array([sign * 0.32, 0.15, 0.30]),
                            (local_time - 5.7) / 0.8,
                        )
                else:
                    if self.hem_points is None:
                        raise RuntimeError("The full-fold regrasp requires captured hem points.")
                    is_position_local = False
                    phase_time = time - 5.0
                    hem = self.hem_points[side]
                    outside = hem + np.array([0.0, -0.05 * garment_scale, 0.0])
                    if phase_time < 10.0:
                        position = smooth_lerp(
                            garment_pos + garment_scale * np.array([sign * 0.32, 0.15, 0.30]),
                            outside + np.array([0.0, 0.0, 0.08 * garment_scale]),
                            phase_time - 9.0,
                        )
                        pitch = -sign * np.pi * (1.0 - smooth_step(phase_time - 9.0))
                        opening = 0.03
                        close = 1.0
                    elif phase_time < 11.0:
                        position = smooth_lerp(
                            outside + np.array([0.0, 0.0, 0.08 * garment_scale]),
                            outside,
                            phase_time - 10.0,
                        )
                    elif phase_time < 12.0:
                        position = smooth_lerp(outside, hem, phase_time - 11.0)
                    elif phase_time < 13.5:
                        position = hem.copy()
                        close = smooth_step((phase_time - 12.0) / 0.8)
                    elif phase_time < 19.0:
                        angle = np.pi * smooth_step((phase_time - 13.5) / 5.5)
                        close = 1.0
                        roll = -angle
                        position = np.array(
                            [
                                hem[0],
                                runtime.garment_pos[1]
                                + (hem[1] - runtime.garment_pos[1]) * np.cos(angle),
                                hem[2]
                                + 0.19 * garment_scale * np.sin(angle)
                                + 0.05 * garment_scale * smooth_step((phase_time - 17.0) / 2.0),
                            ]
                        )
                    else:
                        local_hem_y = (hem[1] - runtime.garment_pos[1]) / garment_scale
                        position = np.array(
                            [
                                hem[0],
                                runtime.garment_pos[1] - garment_scale * local_hem_y,
                                hem[2] + 0.05 * garment_scale,
                            ]
                        )
                        roll = -np.pi
                        close = 1.0
                        opening = 0.03 * smooth_step((phase_time - 19.0) / 0.2)
                        if phase_time >= 19.2:
                            position[1] += 0.15 * garment_scale * smooth_step((phase_time - 19.2) / 1.0)
                        if phase_time >= 20.2:
                            target_height = runtime.garment_pos[2] + 0.32 * garment_scale
                            position[2] = hem[2] + 0.05 * garment_scale + (
                                target_height - hem[2] - 0.05 * garment_scale
                            ) * smooth_step((phase_time - 20.2) / 0.8)
                        if phase_time >= 21.0:
                            destination = garment_pos + garment_scale * np.array([sign * 0.32, 0.1, 0.32])
                            position = smooth_lerp(position, destination, phase_time - 21.0)

            closed_gap = (
                self.config.regrasp_gap
                if self.config.task == "quarter" and time >= 16.0
                else self.config.closed_gap
            )
            gap = self.config.open_gap * (1.0 - close) + closed_gap * close + opening
            if is_position_local:
                position = garment_pos + garment_scale * position
            half_roll = 0.5 * roll
            half_pitch = 0.5 * pitch
            sin_roll = np.sin(half_roll)
            cos_roll = np.cos(half_roll)
            sin_pitch = np.sin(half_pitch)
            cos_pitch = np.cos(half_pitch)
            quat = np.array(
                [
                    cos_pitch * cos_roll,
                    cos_pitch * sin_roll,
                    sin_pitch * cos_roll,
                    -sin_pitch * sin_roll,
                ]
            )
            sin_roll_full = np.sin(roll)
            cos_roll_full = np.cos(roll)
            sin_pitch_full = np.sin(pitch)
            cos_pitch_full = np.cos(pitch)
            rotation = np.array(
                [
                    [cos_pitch_full, sin_pitch_full * sin_roll_full, sin_pitch_full * cos_roll_full],
                    [0.0, cos_roll_full, -sin_roll_full],
                    [-sin_pitch_full, cos_pitch_full * sin_roll_full, cos_pitch_full * cos_roll_full],
                ]
            )
            for jaw, sign in zip(jaws, (-1.0, 1.0)):
                offset = rotation @ np.array(
                    [0.0, 0.0, sign * (0.5 * gap + runtime.jaw_half_thickness)]
                )
                jaw.set_pos(position + offset)
                jaw.set_quat(quat)

    def step(self, runtime: GarmentFoldingRuntime):
        self.before_step(runtime)
        runtime.scene.step()
        self.step_index += 1
