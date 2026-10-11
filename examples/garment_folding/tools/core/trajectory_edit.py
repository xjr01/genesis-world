"""Apply smooth local corrections to an existing sampled TCP path."""

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .action_plan import CompiledPlan, HandPath, minimum_jerk


@dataclass(frozen=True)
class PositionEdit:
    frame: int
    delta_m: tuple[float, float, float]
    support_radius: int
    mode: Literal["local", "hold_after"] = "local"


class NodeEdit(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    hand: Literal["left", "right"]
    frame: StrictInt = Field(ge=0)
    delta_m: tuple[float, float, float]
    support_radius: StrictInt = Field(ge=2)
    mode: Literal["local", "hold_after"] = "local"


class SemanticAnchor(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    hand: Literal["left", "right"]
    frame: StrictInt = Field(ge=0)
    delta_m: tuple[float, float, float]


class PositionLimits(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    max_step_mm: float = Field(default=6.5, gt=0)
    max_speed_mps: float = Field(default=0.4, gt=0)
    max_accel_mps2: float = Field(default=15, gt=0)


class PathEdits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edits: list[NodeEdit] = Field(default_factory=list)
    semantic_keyframes: list[SemanticAnchor] = Field(default_factory=list)
    stage_range: tuple[StrictInt, StrictInt] | None = None
    limits: PositionLimits = Field(default_factory=PositionLimits)

    @model_validator(mode="after")
    def validate_edits(self):
        if not self.edits and not self.semantic_keyframes:
            raise ValueError("Supply nodes or semantic keyframes.")
        if self.stage_range is not None and not 0 <= self.stage_range[0] <= self.stage_range[1]:
            raise ValueError("Expected an ordered nonnegative stage range.")
        return self


def edit_positions(
    frames: np.ndarray,
    positions: np.ndarray,
    edits: tuple[PositionEdit, ...],
    stage_range: tuple[int, int] | None = None,
    semantic_keyframes: tuple[SemanticAnchor, ...] = (),
) -> np.ndarray:
    """Interpolate exact node corrections with compact Wendland C2 support."""
    if positions.shape != (len(frames), 3) or not np.isfinite(positions).all():
        raise ValueError("Expected finite XYZ positions for each frame.")
    if len(frames) == 0 or not np.all(np.diff(frames) == 1):
        raise ValueError("Expected consecutive source frames.")
    start, end = (frames[0], frames[-1]) if stage_range is None else stage_range
    if not frames[0] <= start <= end <= frames[-1]:
        raise ValueError("Stage range must lie within the path.")
    if any(edit.frame < start or edit.frame > end for edit in edits):
        raise ValueError("Node outside stage range.")
    local_edits = tuple(edit for edit in edits if edit.mode == "local")
    if len({edit.frame for edit in local_edits}) != len(local_edits):
        raise ValueError("Edited frame numbers must be unique.")
    if any(edit.support_radius < 2 or not np.isfinite(edit.delta_m).all() for edit in edits):
        raise ValueError("Edits require finite offsets, existing frames and support radii >= 2.")
    correction = np.zeros_like(positions)
    if local_edits:
        centers = np.array([edit.frame for edit in local_edits])
        radii = np.array([edit.support_radius for edit in local_edits])
        delta = np.array([edit.delta_m for edit in local_edits])
        distances = np.abs(frames[:, None] - centers) / radii
        weights = np.maximum(1 - distances, 0) ** 4 * (4 * distances + 1)
        center_distances = np.abs(centers[:, None] - centers) / radii
        matrix = np.maximum(1 - center_distances, 0) ** 4 * (4 * center_distances + 1)
        condition = np.linalg.cond(matrix)
        if not np.isfinite(condition) or condition > 1e8:
            raise ValueError("Node supports are ill-conditioned.")
        correction += weights @ np.linalg.solve(matrix, delta)
    anchors = sorted(semantic_keyframes, key=lambda anchor: anchor.frame)
    if len({anchor.frame for anchor in anchors}) != len(anchors):
        raise ValueError("Semantic anchor frames must be unique per hand.")
    if anchors:
        semantic = np.zeros_like(positions)
        semantic[frames <= anchors[0].frame] = anchors[0].delta_m
        semantic[frames >= anchors[-1].frame] = anchors[-1].delta_m
        for first, last in pairwise(anchors):
            mask = (frames >= first.frame) & (frames <= last.frame)
            weight = minimum_jerk((frames[mask] - first.frame) / (last.frame - first.frame))
            semantic[mask] = np.array(first.delta_m) + weight[:, None] * (np.array(last.delta_m) - first.delta_m)
        correction += semantic
    for edit in edits:
        if edit.mode == "hold_after":
            ramp_start = edit.frame - edit.support_radius
            if ramp_start < start:
                raise ValueError("hold_after ramp crosses stage start.")
            weight = minimum_jerk(np.clip((frames - ramp_start) / edit.support_radius, 0, 1))
            correction += weight[:, None] * edit.delta_m
    correction[(frames < start) | (frames > end)] = 0
    return positions + correction


def edit_path(source: CompiledPlan, edits: PathEdits) -> CompiledPlan:
    """Correct XYZ samples, preserve the anchor and recheck motion limits."""
    paths = []
    for hand, path in (("left", source.left), ("right", source.right)):
        modifications = tuple(
            PositionEdit(edit.frame, edit.delta_m, edit.support_radius, edit.mode)
            for edit in edits.edits
            if edit.hand == hand
        )
        anchors = tuple(anchor for anchor in edits.semantic_keyframes if anchor.hand == hand)
        pos = edit_positions(source.source_frames, path.pos, modifications, edits.stage_range, anchors)
        if not np.allclose(pos[0], path.pos[0], rtol=0, atol=1e-12):
            raise ValueError("Local edits must preserve the prepared starting TCP position.")
        velocity = np.diff(pos, axis=0) * source.fps
        acceleration = np.diff(pos, n=2, axis=0) * source.fps**2
        paths.append(
            HandPath(
                pos,
                path.quat,
                path.opening,
                np.max(np.linalg.norm(velocity, axis=1), initial=0),
                np.max(np.linalg.norm(acceleration, axis=1), initial=0),
            )
        )
    is_accepted = all(
        path.max_speed_mps <= edits.limits.max_speed_mps * (1 + 1e-9)
        and path.max_accel_mps2 <= edits.limits.max_accel_mps2 * (1 + 1e-9)
        and path.max_speed_mps / source.fps * 1000 <= edits.limits.max_step_mm * (1 + 1e-9)
        for path in paths
    )
    return CompiledPlan(source.source_frames, source.fps, *paths, source.checkpoint_frames, is_accepted)


def duration_scale(compiled: CompiledPlan, limits: PositionLimits) -> float:
    """Suggest the time multiplier required by step, speed and acceleration gates."""
    ratio = max(
        1,
        *(path.max_speed_mps / compiled.fps * 1000 / limits.max_step_mm for path in (compiled.left, compiled.right)),
        *(path.max_speed_mps / limits.max_speed_mps for path in (compiled.left, compiled.right)),
        *(np.sqrt(path.max_accel_mps2 / limits.max_accel_mps2) for path in (compiled.left, compiled.right)),
    )
    return ratio * (1 if compiled.is_accepted else 1.05)
