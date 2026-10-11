"""Compile two-hand TCP actions into sampled paths for Genesis inverse kinematics."""

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from scipy.spatial.transform import Rotation, Slerp


class Endpoint(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] | None = None
    opening: float | None = Field(default=None, ge=0, le=0.044)


class Hands(BaseModel):
    left: Endpoint
    right: Endpoint


class HandVectors(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    left: tuple[float, float, float] = (0, 0, 0)
    right: tuple[float, float, float] = (0, 0, 0)


class HandHeights(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    left: float = Field(default=0, ge=0)
    right: float = Field(default=0, ge=0)


class RotationProfile(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    world_x_deg: float
    tool_x_deg: float


class HandRotations(BaseModel):
    left: RotationProfile | None = None
    right: RotationProfile | None = None


class Action(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    id: str = Field(min_length=1, pattern=r"\S")
    duration_frames: StrictInt = Field(ge=1)
    left: Endpoint
    right: Endpoint
    arc_height_m: float | HandHeights = 0
    position_end_velocity_mps: HandVectors = Field(default_factory=HandVectors)
    arc_with_endpoint_velocity: bool = False
    rotation_profile: HandRotations = Field(default_factory=HandRotations)


class PathLimits(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    max_speed_mps: float = Field(default=0.75, gt=0)
    max_accel_mps2: float = Field(default=20, gt=0)


class GripperEvent(BaseModel):
    frame: StrictInt = Field(ge=0)
    command: Literal["open", "close"]
    hand: Literal["both", "left", "right"] = "both"
    duration_frames: StrictInt = Field(default=18, ge=1)


class ActionPlan(BaseModel):
    """External JSON schema shared with the Scene527 handoff plans."""

    model_config = ConfigDict(allow_inf_nan=False, extra="allow")
    start_frame: StrictInt = Field(default=0, ge=0)
    reference_frame: StrictInt | None = Field(default=None, ge=0)
    fps: float = Field(default=60, gt=0, le=1000)
    start: Hands
    actions: list[Action] = Field(min_length=1)
    boundaries: list[str] = Field(default_factory=list)
    limits: PathLimits = Field(default_factory=PathLimits)
    gripper_events: list[GripperEvent] = Field(default_factory=list)

    @field_validator("gripper_events", mode="before")
    @classmethod
    def parse_event_text(cls, value):
        """Accept the upstream absolute-frame command notation as well as JSON events."""
        if not isinstance(value, str):
            return value
        events = []
        pattern = r"t=(\d+)\s+(open|close)\s+(both|left|right)(?:\s+duration=(\d+))?"
        for line in value.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            match = re.fullmatch(pattern, line)
            if match is None:
                raise ValueError("Expected t=120 close both duration=18.")
            frame, command, hand, duration = match.groups()
            events.append(
                GripperEvent(frame=int(frame), command=command, hand=hand, duration_frames=int(duration or 18))
            )
        return events


@dataclass(frozen=True)
class HandPath:
    pos: np.ndarray
    quat: np.ndarray
    opening: np.ndarray
    max_speed_mps: float
    max_accel_mps2: float


@dataclass(frozen=True)
class CompiledPlan:
    source_frames: np.ndarray
    fps: float
    left: HandPath
    right: HandPath
    checkpoint_frames: tuple[int, ...]
    is_accepted: bool


def minimum_jerk(progress):
    """Interpolate a stopped segment with zero endpoint velocity and acceleration."""
    return 10 * progress**3 - 15 * progress**4 + 6 * progress**5


def compile_action_plan(plan: ActionPlan) -> CompiledPlan:
    """Sample position, WXYZ orientation and per-finger aperture on a common timeline."""
    identifiers = [action.id for action in plan.actions]
    if len(set(identifiers)) != len(identifiers) or set(plan.boundaries) - set(identifiers):
        raise ValueError("Action IDs must be unique and boundaries must refer to existing actions.")
    if plan.reference_frame is not None and plan.reference_frame != plan.start_frame:
        raise ValueError("Reference frame must match the plan start frame.")
    duration = sum(action.duration_frames for action in plan.actions)
    if duration > 10000:
        raise ValueError("Plan duration exceeds 10000 action intervals.")
    paths = []
    for hand, initial in (("left", plan.start.left), ("right", plan.start.right)):
        if initial.quat is None or initial.opening is None:
            raise ValueError("Both starting hands require explicit quaternion and opening.")
        pos = np.array(initial.pos)
        quat = np.array(initial.quat)
        if np.linalg.norm(quat) < 1e-12:
            raise ValueError("Starting quaternion must have nonzero norm.")
        quat = quat / np.linalg.norm(quat)
        opening = initial.opening
        positions, quaternions, openings = [pos[None]], [quat[None]], [np.array([opening])]
        start_velocity = np.zeros(3)
        endpoint_speed_max = 0
        for action in plan.actions:
            endpoint = action.left if hand == "left" else action.right
            target = np.array(endpoint.pos)
            target_quat = quat.copy() if endpoint.quat is None else np.array(endpoint.quat)
            if np.linalg.norm(target_quat) < 1e-12:
                raise ValueError("Endpoint quaternion must have nonzero norm.")
            target_quat = target_quat / np.linalg.norm(target_quat)
            target_opening = opening if endpoint.opening is None else endpoint.opening
            progress = np.arange(1, action.duration_frames + 1) / action.duration_frames
            weight = minimum_jerk(progress)
            samples = pos + weight[:, None] * (target - pos)
            heights = action.arc_height_m
            height = heights.left if isinstance(heights, HandHeights) and hand == "left" else heights
            height = heights.right if isinstance(heights, HandHeights) and hand == "right" else height
            if height < 0:
                raise ValueError("Arc height must be non-negative.")
            velocities = action.position_end_velocity_mps
            end_velocity = np.array(velocities.left if hand == "left" else velocities.right)
            endpoint_speed_max = max(endpoint_speed_max, np.linalg.norm(end_velocity))
            if np.any(start_velocity) or np.any(end_velocity):
                if np.array_equal(pos, target):
                    raise ValueError("Position holds require zero endpoint velocity.")
                if height and not action.arc_with_endpoint_velocity:
                    raise ValueError("Set arc_with_endpoint_velocity to combine an arc with endpoint velocity.")
                h_start = progress - 6 * progress**3 + 8 * progress**4 - 3 * progress**5
                h_end = -4 * progress**3 + 7 * progress**4 - 3 * progress**5
                samples += (
                    action.duration_frames
                    / plan.fps
                    * (h_start[:, None] * start_velocity + h_end[:, None] * end_velocity)
                )
            samples[:, 2] += height * 16 * weight**2 * (1 - weight) ** 2
            samples[-1] = target
            rotations = Rotation.from_quat(np.stack((quat, target_quat))[:, [1, 2, 3, 0]])
            quat_samples = Slerp([0, 1], rotations)(weight).as_quat()[:, [3, 0, 1, 2]]
            aligned_quat = -target_quat if np.dot(quat, target_quat) < 0 else target_quat
            if np.dot(quat, aligned_quat) > 0.9995:
                quat_samples = (1 - weight[:, None]) * quat + weight[:, None] * aligned_quat
                quat_samples /= np.linalg.norm(quat_samples, axis=1)[:, None]
            profile = action.rotation_profile.left if hand == "left" else action.rotation_profile.right
            if profile is not None:
                start_rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
                world_rotations = Rotation.from_euler("x", weight[:, None] * profile.world_x_deg, degrees=True)
                tool_rotations = Rotation.from_euler("x", weight[:, None] * profile.tool_x_deg, degrees=True)
                quat_samples = (world_rotations * start_rotation * tool_rotations).as_quat()[:, [3, 0, 1, 2]]
                if abs(np.dot(quat_samples[-1], target_quat)) < 1 - 1e-6:
                    raise ValueError("Rotation profile endpoint differs from the target quaternion.")
                quat_samples[-1] = target_quat
            aperture = opening + weight * (target_opening - opening)
            positions.append(samples)
            quaternions.append(quat_samples)
            openings.append(aperture)
            pos, quat, opening, start_velocity = target, quat_samples[-1], target_opening, end_velocity
        if np.any(start_velocity):
            raise ValueError("The last action must finish with zero endpoint velocity.")
        positions = np.concatenate(positions)
        quaternions = np.concatenate(quaternions)
        openings = np.concatenate(openings)
        events = sorted((event for event in plan.gripper_events if event.hand in (hand, "both")), key=lambda e: e.frame)
        if plan.gripper_events:
            openings.fill(initial.opening)
        previous_end = 0
        for event in events:
            begin = event.frame - plan.start_frame
            end = begin + event.duration_frames
            if begin < previous_end or end > duration:
                raise ValueError("Gripper events must fit the plan and must not overlap for the same hand.")
            weight = minimum_jerk(np.arange(event.duration_frames + 1) / event.duration_frames)
            target = 0.044 if event.command == "open" else 0
            openings[begin : end + 1] = openings[begin] + weight * (target - openings[begin])
            openings[end:] = target
            previous_end = end
        speeds = np.diff(positions, axis=0) * plan.fps
        acceleration = np.diff(np.vstack((np.zeros(3), speeds, np.zeros(3))), axis=0) * plan.fps
        paths.append(
            HandPath(
                positions,
                quaternions,
                openings,
                max(np.max(np.linalg.norm(speeds, axis=1)), endpoint_speed_max),
                np.max(np.linalg.norm(acceleration, axis=1)),
            )
        )
    boundaries = []
    frame = plan.start_frame
    for action in plan.actions:
        frame += action.duration_frames
        if action.id in plan.boundaries:
            boundaries.append(frame)
    is_accepted = all(
        path.max_speed_mps <= plan.limits.max_speed_mps and path.max_accel_mps2 <= plan.limits.max_accel_mps2
        for path in paths
    )
    return CompiledPlan(np.arange(plan.start_frame, frame + 1), plan.fps, *paths, tuple(boundaries), is_accepted)
