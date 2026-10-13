"""Validated sampled joint trajectories and action-prefix identity."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class JointTrajectory:
    joint_q: np.ndarray
    joint_names: tuple[str, ...]
    checkpoint_sha256: str | None = None
    checkpoint_step: int | None = None


def file_hash(path: Path) -> str:
    """Hash file contents incrementally, including large native IPC checkpoints."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prefix_hash(joint_q: np.ndarray) -> str:
    """Hash a command prefix in a platform-independent external binary format."""
    return hashlib.sha256(np.ascontiguousarray(joint_q, dtype="<f8").tobytes()).hexdigest()


def read_trajectory(path: Path) -> JointTrajectory:
    """Read finite joint commands with unique joint names and optional continuation identity."""
    with np.load(path, allow_pickle=False) as data:
        joints = data["joint_q"].copy()
        names = tuple(data["joint_names"].tolist())
        if "accepted" in data and not data["accepted"].item():
            raise ValueError("Rejected IK trajectories cannot be executed.")
        checkpoint_hash = data["checkpoint_sha256"].item() if "checkpoint_sha256" in data else None
        checkpoint_step = data["checkpoint_step"].item() if "checkpoint_step" in data else None
    if joints.ndim != 2 or joints.shape[1] != len(names) or not len(joints) or not np.isfinite(joints).all():
        raise ValueError("Trajectory must contain a nonempty finite joint_q matrix matching joint_names.")
    if len(names) != len(set(names)) or any(not isinstance(name, str) for name in names):
        raise ValueError("Trajectory joint_names must be unique strings.")
    if (checkpoint_hash is None) != (checkpoint_step is None):
        raise ValueError("Continuation identity requires both checkpoint hash and progress.")
    return JointTrajectory(joints, names, checkpoint_hash, checkpoint_step)
