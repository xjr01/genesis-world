from dataclasses import dataclass


@dataclass(frozen=True)
class CoffeeWaterTaskConfig:
    """Robot grasp geometry and task timing independent of assets and fluid numerics."""

    cup_grip_height: float = 0.09
    cup_grip_opening: float = 0.034
    recovery_grip_opening: float = 0.032
    rod_grip_height: float = 0.15
    tool_center: tuple[float, float, float] = (0.14747, 0.001786, 0.0)
    parallel_start: float = 6.4
    motion_end: float = 28.0
