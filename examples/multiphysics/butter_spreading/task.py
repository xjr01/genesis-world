from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import ButterSpreadingTaskConfig

if TYPE_CHECKING:
    from .scene import ButterSpreadingRuntime


def _ease(value: float) -> float:
    return value * value * value * (10.0 + value * (-15.0 + 6.0 * value))


def _ease_integral(value: float) -> float:
    return value**4 * (2.5 + value * (-3.0 + value))


def knife_pose(config: ButterSpreadingTaskConfig, time_s: float) -> tuple[float, float, float]:
    """Evaluate the continuous level-blade trajectory in metres."""
    time_s = min(max(time_s, 0.0), config.lift_end_time)
    if time_s < config.press_end_time:
        phase = time_s / config.press_end_time
        x_pos = config.press_end_x
        z_pos = config.start_clearance + (config.press_clearance - config.start_clearance) * _ease(phase)
    elif time_s < config.acceleration_end_time:
        duration = config.acceleration_end_time - config.press_end_time
        phase = (time_s - config.press_end_time) / duration
        x_pos = config.press_end_x + config.spread_speed * duration * _ease_integral(phase)
        z_pos = config.press_clearance + (config.sweep_clearance - config.press_clearance) * _ease(phase)
    elif time_s < config.deceleration_start_time:
        x_pos = (
            config.press_end_x
            + 0.5 * config.spread_speed * (config.acceleration_end_time - config.press_end_time)
            + config.spread_speed * (time_s - config.acceleration_end_time)
        )
        phase = (time_s - config.acceleration_end_time) / (
            config.spread_end_time - config.acceleration_end_time
        )
        z_pos = config.sweep_clearance + (config.sweep_end_clearance - config.sweep_clearance) * _ease(phase)
    elif time_s < config.spread_end_time:
        duration = config.spread_end_time - config.deceleration_start_time
        phase = (time_s - config.deceleration_start_time) / duration
        x_pos = (
            config.press_end_x
            + 0.5 * config.spread_speed * (config.acceleration_end_time - config.press_end_time)
            + config.spread_speed * (config.deceleration_start_time - config.acceleration_end_time)
            + config.spread_speed * duration * (phase - _ease_integral(phase))
        )
        sweep_phase = (time_s - config.acceleration_end_time) / (
            config.spread_end_time - config.acceleration_end_time
        )
        z_pos = config.sweep_clearance + (config.sweep_end_clearance - config.sweep_clearance) * _ease(sweep_phase)
    else:
        phase = (time_s - config.spread_end_time) / (config.lift_end_time - config.spread_end_time)
        x_pos = config.spread_end_x
        z_pos = config.sweep_end_clearance + (config.lift_clearance - config.sweep_end_clearance) * _ease(phase)
    return x_pos, 0.0, z_pos


@dataclass
class ButterSpreadingController:
    config: ButterSpreadingTaskConfig
    dt: float

    def step(self, runtime: "ButterSpreadingRuntime"):
        current_local_pose = knife_pose(self.config, runtime.scene.cur_t)
        next_local_pose = knife_pose(self.config, runtime.scene.cur_t + self.dt)
        current_pose = tuple(origin + value for origin, value in zip(runtime.task_origin, current_local_pose))
        next_pose = tuple(origin + value for origin, value in zip(runtime.task_origin, next_local_pose))
        center = (
            current_pose[0] + runtime.blade_xy_offset[0],
            current_pose[1] + runtime.blade_xy_offset[1],
            current_pose[2],
        )
        velocity = tuple((next_value - value) / self.dt for value, next_value in zip(current_pose, next_pose))
        runtime.blade.set_pos(center, zero_velocity=False, relative=False, skip_forward=True)
        runtime.blade.set_dofs_velocity((*velocity, 0.0, 0.0, 0.0))
        runtime.scene.step()
