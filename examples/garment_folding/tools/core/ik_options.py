"""Validated inverse kinematics settings and bounded tool-axis assistance."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial.transform import Rotation


class OrientationAssist(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    is_enabled: bool = False
    gain_deg_per_mm: float = -0.181379499
    max_abs_angle_deg: float = Field(default=15, ge=0)


class IKOptions(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    hand: Literal["both", "left", "right"] = "both"
    orientation_mode: Literal["full", "closing_y", "length_x", "tool_z"] = "full"
    orientation_assist: OrientationAssist = Field(default_factory=OrientationAssist)
    table_z_m: float = 0.8
    min_tool_table_clearance_m: float = -0.0035
    min_clearance_delta_m: float = -0.0035

    @property
    def rotation_mask(self):
        if self.orientation_assist.is_enabled or self.orientation_mode == "full":
            return (True, True, True)
        return tuple(self.orientation_mode == mode for mode in ("length_x", "closing_y", "tool_z"))


def assisted_quaternions(quat: np.ndarray, delta_m: np.ndarray, options: OrientationAssist):
    """Rotate around tool-local Y by the bounded correction-length gain."""
    angles = np.clip(
        np.linalg.norm(delta_m, axis=1) * 1000 * options.gain_deg_per_mm,
        -options.max_abs_angle_deg,
        options.max_abs_angle_deg,
    )
    rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    assist = Rotation.from_euler("y", angles[:, None], degrees=True)
    return (rotation * assist).as_quat()[:, [3, 0, 1, 2]], angles


def orientation_error(actual: Rotation, target: Rotation, options: IKOptions):
    """Measure full rotation error or the angle between constrained tool axes."""
    if all(options.rotation_mask):
        return np.degrees((actual.inv() * target).magnitude())
    axis = options.rotation_mask.index(True)
    return np.degrees(np.arccos(np.clip(np.dot(actual.as_matrix()[:, axis], target.as_matrix()[:, axis]), -1, 1)))
