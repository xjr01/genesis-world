"""Compile a standalone action JSON; optionally check a prepared robot with CPU IK."""

import argparse
from pathlib import Path

from .core.action_plan import ActionPlan, compile_action_plan
from .core.drafts import default_plan
from .core.plan_io import save_compiled
from .core.preflight import run_preflight
from .core.workspace import read_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--plan", type=Path, help="Action JSON to sample into TCP paths.")
    inputs.add_argument("--draft-from", type=Path, help="Plan supplying an already grasped start for lift/pull/hold.")
    parser.add_argument("--pull-axis", choices=("x", "y"), default="x")
    parser.add_argument("--pull-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--output-dir", type=Path, required=True, help="New TCP output directory.")
    parser.add_argument("--workspace", type=Path, help="Optional saved-state identity; required for --ik.")
    parser.add_argument("--ik", action="store_true", help="Also run CPU IK against --workspace.")
    args = parser.parse_args()
    if args.ik and args.workspace is None:
        parser.error("--ik requires --workspace; TCP-only compilation needs no workspace.")
    source = args.plan if args.plan is not None else args.draft_from
    plan = ActionPlan.model_validate_json(source.read_text(encoding="utf-8"))
    if args.draft_from is not None:
        plan = default_plan(plan.start, plan.start_frame, args.pull_axis, args.pull_sign)
    if args.workspace is not None:
        metadata = read_workspace(args.workspace)
        if plan.start_frame != metadata.start_frame or plan.fps != 60:
            raise ValueError("Plan must use the prepared start frame and 60 Hz action rate.")
    compiled = compile_action_plan(plan)
    save_compiled(args.output_dir, plan, compiled)
    if not compiled.is_accepted:
        raise SystemExit("TCP path rejected; inspect path-report.json.")
    if args.ik:
        run_preflight(args.workspace, args.output_dir)
    print(f"Compiled {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
