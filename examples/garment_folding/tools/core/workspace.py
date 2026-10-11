"""Typed identities for optional saved-state tools and robot preflight."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ...config import create_scene527_config
from .trajectory import file_hash


class Preparation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mesh: str
    cloth_sha256: str
    robot_sha256: str
    asset_root: str
    trajectory: str
    trajectory_sha256: str
    prefix_sha256: str
    checkpoint: str | None
    checkpoint_sha256: str | None
    checkpoint_step: int
    start_frame: int
    joint_names: tuple[str, ...]
    is_physics_ready: bool = True
    cloth_path: str | None = None
    robot_path: str | None = None
    replay: str | None = None
    robot_base_pos_m: tuple[float, float, float] = (0, 0, 0.17)


def read_workspace(workspace: Path) -> Preparation:
    """Check that the robot and cloth still match the prepared state."""
    metadata = Preparation.model_validate_json((workspace / "workspace.json").read_text(encoding="utf-8"))
    root = Path(metadata.asset_root)
    config = create_scene527_config(root, mesh=metadata.mesh)
    cloth_path = root / config.assets.garment_mesh if metadata.cloth_path is None else Path(metadata.cloth_path)
    robot_path = root / config.assets.robot if metadata.robot_path is None else Path(metadata.robot_path)
    if file_hash(cloth_path) != metadata.cloth_sha256:
        raise ValueError("Workspace cloth asset has changed since preparation.")
    if file_hash(robot_path) != metadata.robot_sha256:
        raise ValueError("Workspace robot URDF has changed since preparation.")
    return metadata
