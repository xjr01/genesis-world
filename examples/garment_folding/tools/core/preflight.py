"""Solve and validate an already compiled TCP path against a prepared robot state."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .ik_options import IKOptions, assisted_quaternions
from .kinematics import build_robot, solve_ik
from .plan_io import load_compiled, write_json
from .trajectory import file_hash
from .workspace import read_workspace


def run_preflight(workspace: Path, compiled_dir: Path, options: IKOptions | None = None):
    """Add an IK report and accepted or rejected joint file to a TCP output directory."""
    metadata = read_workspace(workspace)
    options = IKOptions() if options is None else options
    if (compiled_dir / "ik-report.json").exists():
        raise FileExistsError("IK output already exists; compile or edit to a new directory.")
    path_report = json.loads((compiled_dir / "path-report.json").read_text(encoding="utf-8"))
    compiled = load_compiled(compiled_dir / "tcp-path.npz")
    if not path_report["accepted"] or not compiled.is_accepted:
        raise ValueError("IK requires a TCP path that passed translation checks.")
    if compiled.source_frames[0] != metadata.start_frame or compiled.fps != 60:
        raise ValueError("TCP path must use the prepared start frame and 60 Hz action rate.")
    root = Path(metadata.asset_root)
    runtime = build_robot(
        root, None if metadata.robot_path is None else Path(metadata.robot_path), metadata.robot_base_pos_m
    )
    if runtime.joint_names != metadata.joint_names:
        raise ValueError("Prepared robot joint order has changed.")
    with np.load(workspace / "seed.npz", allow_pickle=False) as data:
        seed = data["robot_q"].copy()
    if options.orientation_assist.is_enabled:
        edits = json.loads((compiled_dir / "edits.json").read_text(encoding="utf-8"))
        baseline_file = Path(edits["source_compiled"]) / "tcp-path.npz"
        if file_hash(baseline_file) != edits["source_tcp_path_sha256"]:
            raise ValueError("Assistance baseline path has changed.")
        baseline = load_compiled(baseline_file)
        if not np.array_equal(baseline.source_frames, compiled.source_frames):
            raise ValueError("Assistance baseline frames differ from the edited path.")
        paths = []
        for hand, path, original in (("left", compiled.left, baseline.left), ("right", compiled.right, baseline.right)):
            if options.hand in ("both", hand):
                quat, angles = assisted_quaternions(original.quat, path.pos - original.pos, options.orientation_assist)
                path = replace(path, quat=quat)
                np.save(compiled_dir / f"{hand}-assist-angle-deg.npy", angles)
            paths.append(path)
        compiled = replace(compiled, left=paths[0], right=paths[1])
    baseline_joints = None
    if options.hand != "both" and (workspace / "baseline-joints.npz").exists():
        with np.load(workspace / "baseline-joints.npz", allow_pickle=False) as data:
            indices = compiled.source_frames - data["source_frames"][0]
            if np.any(indices < 0) or np.any(indices >= len(data["robot_q"])):
                if compiled.source_frames[0] < data["source_frames"][-1]:
                    raise ValueError(
                        "Single-hand replay edits must fit the baseline; start at the last observation to draft a new segment."
                    )
            else:
                baseline_joints = data["robot_q"][indices].copy()
    report, joints = solve_ik(runtime, compiled, seed, options=options, baseline_joints=baseline_joints)
    external_report = asdict(report)
    for key, value in tuple(external_report.items()):
        if isinstance(value, np.ndarray):
            external_report[key] = value.tolist()
    external_report["workspace_sha256"] = file_hash(workspace / "workspace.json")
    external_report["tcp_path_sha256"] = file_hash(compiled_dir / "tcp-path.npz")
    external_report["options"] = options.model_dump()
    write_json(compiled_dir / "ik-report.json", external_report)
    name = "robot_q.npz" if report.is_accepted else "rejected-robot_q.npz"
    np.savez_compressed(
        compiled_dir / name,
        robot_q=joints,
        joint_names=runtime.joint_names,
        source_frames=compiled.source_frames,
        accepted=report.is_accepted,
    )
    external_report["joint_file_sha256"] = file_hash(compiled_dir / name)
    write_json(compiled_dir / "ik-report.json", external_report)
    if not report.is_accepted:
        raise SystemExit("IK rejected; inspect ik-report.json.")
