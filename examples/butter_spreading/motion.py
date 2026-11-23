"""Prescribed level-blade motion in world-frame metres."""

from examples.butter_spreading.config import MotionParameters

DEFAULT_MOTION = MotionParameters()


def _ease(value: float) -> float:
    return value * value * value * (10.0 + value * (-15.0 + 6.0 * value))


def _ease_integral(value: float) -> float:
    return value**4 * (2.5 + value * (-3.0 + value))


def knife_pose(time_s: float, motion: MotionParameters = DEFAULT_MOTION) -> tuple[float, float, float]:
    """Evaluate the continuous level-blade trajectory in world-frame metres."""
    acceleration_end_time = 1.65
    deceleration_start_time = 3.15
    lift_clearance = motion.lift_clearance
    lift_end_time = 4.0
    press_clearance = motion.press_clearance
    press_end_time = 0.9
    press_end_x = motion.press_end_x
    spread_end_time = 3.5
    spread_end_x = motion.spread_end_x
    spread_speed = (spread_end_x - press_end_x) / (
        deceleration_start_time
        - acceleration_end_time
        + 0.5 * (acceleration_end_time - press_end_time)
        + 0.5 * (spread_end_time - deceleration_start_time)
    )
    start_clearance = motion.start_clearance
    sweep_clearance = motion.sweep_clearance
    sweep_end_clearance = motion.sweep_end_clearance
    time_s = min(max(time_s, 0.0), lift_end_time)
    if time_s < press_end_time:
        phase = time_s / press_end_time
        x_pos = press_end_x
        z_pos = start_clearance + (press_clearance - start_clearance) * _ease(phase)
    elif time_s < acceleration_end_time:
        duration = acceleration_end_time - press_end_time
        phase = (time_s - press_end_time) / duration
        x_pos = press_end_x + spread_speed * duration * _ease_integral(phase)
        z_pos = press_clearance + (sweep_clearance - press_clearance) * _ease(phase)
    elif time_s < deceleration_start_time:
        x_pos = (
            press_end_x
            + 0.5 * spread_speed * (acceleration_end_time - press_end_time)
            + spread_speed * (time_s - acceleration_end_time)
        )
        phase = (time_s - acceleration_end_time) / (spread_end_time - acceleration_end_time)
        z_pos = sweep_clearance + (sweep_end_clearance - sweep_clearance) * _ease(phase)
    elif time_s < spread_end_time:
        duration = spread_end_time - deceleration_start_time
        phase = (time_s - deceleration_start_time) / duration
        x_pos = (
            press_end_x
            + 0.5 * spread_speed * (acceleration_end_time - press_end_time)
            + spread_speed * (deceleration_start_time - acceleration_end_time)
            + spread_speed * duration * (phase - _ease_integral(phase))
        )
        sweep_phase = (time_s - acceleration_end_time) / (spread_end_time - acceleration_end_time)
        z_pos = sweep_clearance + (sweep_end_clearance - sweep_clearance) * _ease(sweep_phase)
    else:
        phase = (time_s - spread_end_time) / (lift_end_time - spread_end_time)
        x_pos = spread_end_x
        z_pos = sweep_end_clearance + (lift_clearance - sweep_end_clearance) * _ease(phase)
    return x_pos, motion.y, z_pos
