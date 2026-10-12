"""Export accepted IK commands for the existing garment-folding entry point."""

import json
import sys
from pathlib import Path

import numpy as np

from .plan_io import write_json
from .trajectory import file_hash, prefix_hash, read_trajectory
from .workspace import read_workspace


def export_trajectory(workspace: Path, compiled_dir: Path, output_dir: Path):
    """Export checked joint commands and a launcher for the existing folding demo."""
    metadata = read_workspace(workspace)
    if not metadata.is_physics_ready:
        raise ValueError("This observation workspace requires a matching native .pt checkpoint before physics export.")
    report = json.loads((compiled_dir / "ik-report.json").read_text(encoding="utf-8"))
    if not report["is_accepted"] or report["workspace_sha256"] != file_hash(workspace / "workspace.json"):
        raise ValueError("Export requires accepted IK from this workspace.")
    joint_file = compiled_dir / "robot_q.npz"
    if report["joint_file_sha256"] != file_hash(joint_file):
        raise ValueError("IK joint output has changed after validation.")
    if report["tcp_path_sha256"] != file_hash(compiled_dir / "tcp-path.npz"):
        raise ValueError("Sampled TCP path has changed after IK validation.")
    with np.load(joint_file, allow_pickle=False) as data:
        joints = data["robot_q"].copy()
        names = tuple(data["joint_names"].tolist())
        frames = data["source_frames"].copy()
        if not data["accepted"].item() or names != metadata.joint_names:
            raise ValueError("IK acceptance or joint order differs from the workspace.")
    if not np.array_equal(frames, np.arange(metadata.start_frame, metadata.start_frame + len(joints))):
        raise ValueError("IK frames must begin at the prepared anchor.")
    with np.load(workspace / "seed.npz", allow_pickle=False) as data:
        if not np.array_equal(joints[0], data["robot_q"]):
            raise ValueError("IK first row must preserve the exact seed.")
    trajectory = read_trajectory(Path(metadata.trajectory))
    prefix_length = metadata.checkpoint_step // 2
    if file_hash(Path(metadata.trajectory)) != metadata.trajectory_sha256:
        raise ValueError("Source trajectory changed after preparation.")
    if prefix_hash(trajectory.joint_q[:prefix_length]) != metadata.prefix_sha256:
        raise ValueError("Historical command prefix changed after preparation.")
    prefix = trajectory.joint_q[:prefix_length, [trajectory.joint_names.index(name) for name in names]]
    if metadata.checkpoint is not None and file_hash(Path(metadata.checkpoint)) != metadata.checkpoint_sha256:
        raise ValueError("Checkpoint changed after preparation.")
    cumulative = np.concatenate((prefix, joints[1:])) if prefix_length else joints
    output_dir.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[4]
    physics_output = output_dir.resolve() / "physics"
    trajectory_path = output_dir.resolve() / "trajectory.npz"
    if metadata.checkpoint is None:
        np.savez_compressed(trajectory_path, joint_q=cumulative, joint_names=names, accepted=True)
    else:
        np.savez_compressed(
            trajectory_path,
            joint_q=cumulative,
            joint_names=names,
            accepted=True,
            checkpoint_sha256=metadata.checkpoint_sha256,
            checkpoint_step=metadata.checkpoint_step,
        )
    argv = [
        sys.executable,
        "-m",
        "examples.garment_folding.run",
        "--mesh",
        metadata.mesh,
        "--asset-root",
        metadata.asset_root,
        "--trajectory",
        str(trajectory_path),
        "--output-dir",
        str(physics_output),
        "--dump-replay",
        "--record",
    ]
    if metadata.checkpoint is not None:
        argv.extend(("--checkpoint", metadata.checkpoint))
    write_json(
        output_dir / "run-manifest.json",
        {
            "argv": argv,
            "cwd": str(repository),
            "workspace": metadata.model_dump(),
            "trajectory_sha256": file_hash(trajectory_path),
            "action_intervals": len(joints) - 1,
            "execution_physics_steps": 2 * (len(cumulative) - prefix_length),
        },
    )
    command = " ".join("'" + argument.replace("'", "''") + "'" for argument in argv)
    output_literal = str(physics_output).replace("'", "''")
    repository_literal = str(repository).replace("'", "''")
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        f"if (Test-Path -LiteralPath '{output_literal}') {{ throw 'Run output already exists; export a new bundle.' }}\n"
        f"Push-Location -LiteralPath '{repository_literal}'\n"
        "try {\n"
        f"    & {command}\n"
        "    if ($LASTEXITCODE -ne 0) { throw 'Garment-folding simulation failed.' }\n"
        "} finally {\n"
        "    Pop-Location\n"
        "}\n"
    )
    (output_dir / "run.ps1").write_text(script, encoding="utf-8")
    print(f"Exported {trajectory_path}; run the generated run.ps1 from the repository root.")
