"""Apply local XYZ node edits to a saved TCP path without requiring a simulator workspace."""

import argparse
from pathlib import Path

from .core.action_plan import ActionPlan, PathLimits
from .core.plan_io import load_compiled, save_compiled, write_json
from .core.preflight import run_preflight
from .core.trajectory import file_hash
from .core.trajectory_edit import PathEdits, duration_scale, edit_path
from .core.workspace import read_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, required=True, help="Existing TCP output directory.")
    parser.add_argument("--edits", type=Path, required=True, help="JSON hand/frame/delta_m/support_radius edits.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New corrected TCP directory.")
    parser.add_argument("--workspace", type=Path, help="Optional saved state for robot validation.")
    parser.add_argument("--ik", action="store_true", help="Also run CPU IK against --workspace.")
    args = parser.parse_args()
    if args.ik and args.workspace is None:
        parser.error("--ik requires --workspace; TCP-only edits need no workspace.")
    plan = ActionPlan.model_validate_json((args.compiled / "plan.json").read_text(encoding="utf-8"))
    source = load_compiled(args.compiled / "tcp-path.npz")
    edits = PathEdits.model_validate_json(args.edits.read_text(encoding="utf-8"))
    compiled = edit_path(source, edits)
    plan = plan.model_copy(
        update={
            "limits": PathLimits(max_speed_mps=edits.limits.max_speed_mps, max_accel_mps2=edits.limits.max_accel_mps2)
        }
    )
    if args.workspace is not None:
        metadata = read_workspace(args.workspace)
        if source.source_frames[0] != metadata.start_frame or source.fps != 60:
            raise ValueError("Source path must match the workspace start and action rate.")
    save_compiled(args.output_dir, plan, compiled)
    report = {
        "accepted": compiled.is_accepted,
        "limits": edits.limits.model_dump(),
        "left_max_step_mm": compiled.left.max_speed_mps / compiled.fps * 1000,
        "right_max_step_mm": compiled.right.max_speed_mps / compiled.fps * 1000,
        "left_max_speed_mps": compiled.left.max_speed_mps,
        "right_max_speed_mps": compiled.right.max_speed_mps,
        "left_max_accel_mps2": compiled.left.max_accel_mps2,
        "right_max_accel_mps2": compiled.right.max_accel_mps2,
        "required_duration_scale": duration_scale(compiled, edits.limits),
        "scope": "Node and semantic corrections; step, speed and interior acceleration gates",
    }
    write_json(args.output_dir / "path-report.json", report)
    write_json(
        args.output_dir / "edits.json",
        {
            "source_tcp_path_sha256": file_hash(args.compiled / "tcp-path.npz"),
            "source_compiled": str(args.compiled.resolve()),
            "edits": edits.model_dump(),
        },
    )
    if not compiled.is_accepted:
        raise SystemExit("TCP path rejected; inspect path-report.json.")
    if args.ik:
        run_preflight(args.workspace, args.output_dir)
    print(f"Edited {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
