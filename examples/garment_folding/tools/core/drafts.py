"""Draft short lift/pull actions from an explicitly supplied grasped pose."""

from .action_plan import Action, ActionPlan, Hands, compile_action_plan


def default_plan(start: Hands, start_frame: int = 0, pull_axis: str = "x", pull_sign: int = -1) -> ActionPlan:
    """Draft a 5 cm lift, 5 cm pull and half-second hold at the supplied aperture."""
    if pull_axis not in ("x", "y") or pull_sign not in (-1, 1):
        raise ValueError("Pull axis must be x/y and sign must be -1/+1.")
    current = start.model_copy(deep=True)
    actions = []
    for identifier, duration, axis, distance in (
        ("A_lift", 60, 2, 0.05),
        ("A_pull", 60, "xyz".index(pull_axis), 0.05 * pull_sign),
        ("A_hold", 30, 2, 0),
    ):
        for endpoint in (current.left, current.right):
            position = list(endpoint.pos)
            position[axis] += distance
            endpoint.pos = tuple(position)
        actions.append(
            Action(
                id=identifier,
                duration_frames=duration,
                left=current.left.model_copy(),
                right=current.right.model_copy(),
            )
        )
    plan = ActionPlan(start_frame=start_frame, start=start, actions=actions, boundaries=["A_hold"])
    compile_action_plan(plan)
    return plan
