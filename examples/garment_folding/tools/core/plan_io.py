"""Read and write action plans, sampled TCP paths and external reports."""

import json
from pathlib import Path

import numpy as np

from .action_plan import ActionPlan, CompiledPlan, HandPath


def write_json(path: Path, value):
    """Write external JSON with explicit finite-number and UTF-8 conventions."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_compiled(output: Path, plan: ActionPlan, compiled: CompiledPlan):
    """Persist sampled paths and a readable translation-limit report."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "tcp-path.npz",
        source_frames=compiled.source_frames,
        fps=compiled.fps,
        left_pos=compiled.left.pos,
        left_quat=compiled.left.quat,
        left_opening=compiled.left.opening,
        right_pos=compiled.right.pos,
        right_quat=compiled.right.quat,
        right_opening=compiled.right.opening,
        checkpoint_frames=compiled.checkpoint_frames,
        accepted=compiled.is_accepted,
    )
    write_json(
        output / "path-report.json",
        {
            "accepted": compiled.is_accepted,
            "frames": len(compiled.source_frames),
            "left_max_speed_mps": compiled.left.max_speed_mps,
            "right_max_speed_mps": compiled.right.max_speed_mps,
            "left_max_accel_mps2": compiled.left.max_accel_mps2,
            "right_max_accel_mps2": compiled.right.max_accel_mps2,
            "limits": plan.limits.model_dump(),
            "scope": "TCP translation; run IK and FEM/IPC for contact validation",
        },
    )


def load_compiled(path: Path) -> CompiledPlan:
    """Read the finite sampled arrays consumed by the IK solver."""
    with np.load(path, allow_pickle=False) as data:
        frames = data["source_frames"].copy()
        fps = data["fps"].item()
        paths = []
        for hand in ("left", "right"):
            pos, quat, opening = (data[f"{hand}_{name}"].copy() for name in ("pos", "quat", "opening"))
            if pos.shape != (len(frames), 3) or quat.shape != (len(frames), 4) or opening.shape != (len(frames),):
                raise ValueError("Sampled TCP arrays have incompatible shapes.")
            if not all(np.isfinite(value).all() for value in (pos, quat, opening)):
                raise ValueError("TCP arrays must be finite.")
            if np.any(np.abs(np.linalg.norm(quat, axis=1) - 1) > 1e-6) or np.any((opening < 0) | (opening > 0.044)):
                raise ValueError("TCP quaternions must be unit length and aperture must be in range.")
            paths.append(HandPath(pos, quat, opening, 0, 0))
        if not len(frames) or frames.ndim != 1 or not np.all(np.diff(frames) == 1):
            raise ValueError("Source frames must be consecutive.")
        return CompiledPlan(frames, fps, *paths, tuple(data["checkpoint_frames"].tolist()), data["accepted"].item())
