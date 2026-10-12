"""Write selected grasp or placement points into an existing action plan."""

import argparse
from pathlib import Path

from .core.action_plan import ActionPlan
from .core.plan_io import write_json
from .core.points import Bindings, Selections, retarget_points
from .core.trajectory import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New retargeted plan JSON.")
    args = parser.parse_args()
    plan = ActionPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
    selections = Selections.model_validate_json(args.points.read_text(encoding="utf-8"))
    bindings = Bindings.model_validate_json(args.bindings.read_text(encoding="utf-8"))
    result = retarget_points(plan, selections, bindings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(result.model_dump_json(indent=2))
    write_json(
        args.output.with_suffix(".provenance.json"),
        {
            "plan_sha256": file_hash(args.plan),
            "points_sha256": file_hash(args.points),
            "bindings_sha256": file_hash(args.bindings),
        },
    )


if __name__ == "__main__":
    main()
