"""Translate bound action endpoints from selected cloth and placement positions."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from .action_plan import ActionPlan


class SelectedPoint(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="allow")
    material_world_m: tuple[float, float, float]


class SelectedHands(BaseModel):
    left: SelectedPoint | None = None
    right: SelectedPoint | None = None


class PlacementHands(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    left: tuple[float, float, float] | None = None
    right: tuple[float, float, float] | None = None


class Selections(BaseModel):
    model_config = ConfigDict(extra="allow", serialize_by_alias=True)
    schema_name: Literal["workbench-material-regrasp-v1"] = Field(alias="schema")
    source_frame: StrictInt = Field(ge=0)
    points: SelectedHands = Field(default_factory=SelectedHands)
    placement_targets_world_m: PlacementHands = Field(default_factory=PlacementHands)


class PointBinding(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    hand: Literal["left", "right"]
    kind: Literal["grasp", "placement"]
    anchor_action: str
    translate_actions: tuple[str, ...]
    tcp_offset_world_mm: tuple[float, float, float]


class Bindings(BaseModel):
    bindings: list[PointBinding] = Field(min_length=1)


class PointChange(BaseModel):
    hand: Literal["left", "right"]
    kind: Literal["grasp", "placement"]
    status: Literal["updated", "unspecified_preserved"]
    target_tcp_world_m: tuple[float, float, float] | None = None
    delta_mm: tuple[float, float, float] | None = None
    actions: tuple[str, ...] | None = None


class PointEditReport(BaseModel):
    source_frame: int
    changes: list[PointChange]
    ik_status: Literal["not_run"] = "not_run"
    physics_started: bool = False


def retarget_points(plan: ActionPlan, selections: Selections, bindings: Bindings) -> ActionPlan:
    """Apply explicit world-space TCP offsets while retaining timing, rotation and aperture."""
    if selections.source_frame != plan.start_frame:
        raise ValueError("Selections must be saved at the plan start frame.")
    if plan.reference_frame is not None and plan.reference_frame != plan.start_frame:
        raise ValueError("Plan start and reference frames differ.")
    result = plan.model_copy(deep=True)
    ids = [action.id for action in result.actions]
    if len(set(ids)) != len(ids):
        raise ValueError("Action IDs must be unique.")
    updated = set()
    records = []
    for binding in bindings.bindings:
        selected = selections.points.left if binding.hand == "left" else selections.points.right
        placement = selections.placement_targets_world_m
        target = selected.material_world_m if selected is not None else None
        if binding.kind == "placement":
            target = placement.left if binding.hand == "left" else placement.right
        if target is None:
            records.append(PointChange(hand=binding.hand, kind=binding.kind, status="unspecified_preserved"))
            continue
        if binding.anchor_action not in binding.translate_actions:
            raise ValueError("The anchor must be included in translated actions.")
        if set(binding.translate_actions) - set(ids) or binding.anchor_action not in ids:
            raise ValueError("Binding refers to an unknown action.")
        anchor = result.actions[ids.index(binding.anchor_action)]
        endpoint = anchor.left if binding.hand == "left" else anchor.right
        delta = np.array(target) + np.array(binding.tcp_offset_world_mm) * 0.001 - endpoint.pos
        records.append(
            PointChange(
                hand=binding.hand,
                kind=binding.kind,
                status="updated",
                target_tcp_world_m=tuple(np.array(target) + np.array(binding.tcp_offset_world_mm) * 0.001),
                delta_mm=tuple(delta * 1000),
                actions=binding.translate_actions,
            )
        )
        for identifier in binding.translate_actions:
            key = (identifier, binding.hand)
            if key in updated:
                raise ValueError("Bindings overlap for the same hand and action.")
            action = result.actions[ids.index(identifier)]
            endpoint = action.left if binding.hand == "left" else action.right
            endpoint.pos = tuple(np.array(endpoint.pos) + delta)
            updated.add(key)
    if not updated:
        raise ValueError("No selected point matched a binding.")
    report = PointEditReport(source_frame=plan.start_frame, changes=records)
    return result.model_copy(update={"point_edit": report.model_dump(mode="json", exclude_none=True)})
