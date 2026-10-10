"""Translate saved observation configurations and native run manifests into workspaces."""

import json
from pathlib import Path

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict

from ...config import create_scene527_config
from .action_plan import Action, ActionPlan, CompiledPlan, HandPath
from .plan_io import save_compiled, write_json
from .preparation import prepare_workspace
from .preview import Camera, link_transforms, measured_hands
from .trajectory import file_hash, prefix_hash
from .workspace import Preparation


class ObservationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    replay: str
    cloth_obj: str
    robot_urdf: str
    robot_base_pos_m: tuple[float, float, float] = (0, 0, 0.17)
    default_frame: int | None = None
    display_frame_range: tuple[int, int] | None = None
    camera: Camera | None = None


def resolve_input(value: str, config_path: Path, source_root: Path | None):
    """Resolve portable source paths against an explicit root or the config folder."""
    path = Path(value)
    if not path.is_absolute():
        path = (config_path.parent if source_root is None else source_root) / path
    return path.resolve(strict=True)


def import_observation(
    config_path: Path, output: Path, source_root: Path | None = None, start_frame: int | None = None
):
    """Import measured cloth and joint observations with preserved frame identities."""
    config = ObservationConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    replay_path = resolve_input(config.replay, config_path, source_root)
    cloth_path = resolve_input(config.cloth_obj, config_path, source_root)
    robot_path = resolve_input(config.robot_urdf, config_path, source_root)
    mesh = trimesh.load(cloth_path, force="mesh", process=False)
    with np.load(replay_path, allow_pickle=False) as replay:
        frames = replay["source_frames"].copy()
        positions = replay["cloth_pos"].copy()
        joints = replay["robot_q"].copy()
        measured = {key: replay[key].copy() for key in ("ipc_link_names", "ipc_link_transforms") if key in replay}
        if "joint_names" in replay:
            names = tuple(replay["joint_names"].tolist())
        else:
            names = (
                "joint1",
                "joint2",
                "joint3",
                *(f"left_joint1{i}" for i in range(1, 9)),
                *(f"right_joint2{i}" for i in range(1, 9)),
            )
    if frames.ndim != 1 or len(frames) < 2 or not np.all(np.diff(frames) == 1):
        raise ValueError("Observation frames must be consecutive.")
    if positions.shape != (len(frames), len(mesh.vertices), 3) or joints.shape != (len(frames), len(names)):
        raise ValueError("Observation topology or joint order differs from its assets.")
    if not np.isfinite(positions).all() or not np.isfinite(joints).all():
        raise ValueError("Observation arrays must be finite.")
    anchor = frames[-1].item() if config.default_frame is None else config.default_frame
    if start_frame is not None:
        anchor = start_frame
    indices = np.flatnonzero(frames == anchor)
    if len(indices) != 1:
        raise ValueError("Default frame must occur in the replay.")
    hands = [
        measured_hands(link_transforms(robot_path, names, row, config.robot_base_pos_m), row, names) for row in joints
    ]
    initial = hands[indices[0]]
    plan = ActionPlan(
        start_frame=anchor,
        reference_frame=anchor,
        start=initial,
        actions=[Action(id="hold", duration_frames=60, left=initial.left, right=initial.right)],
    )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "seed.npz", robot_q=joints[indices[0]], joint_names=names)
    np.savez_compressed(
        output / "cloth-state.npz",
        cloth_pos=positions[indices[0]],
        faces=mesh.faces,
        rest_raw_xyz=mesh.vertices,
        source_frame=anchor,
    )
    np.savez_compressed(
        output / "replay.npz",
        source_frames=frames,
        cloth_pos=positions,
        robot_q=joints,
        joint_names=names,
        cloth_sha256=file_hash(cloth_path),
        **measured,
    )
    np.savez_compressed(output / "baseline-joints.npz", source_frames=frames, robot_q=joints, joint_names=names)
    root = Path(__file__).resolve().parents[2]
    trajectory = root / create_scene527_config(root).assets.trajectory
    metadata = Preparation(
        mesh="8k" if len(mesh.faces) == 8000 else "13k" if len(mesh.faces) == 13767 else "55k",
        cloth_sha256=file_hash(cloth_path),
        robot_sha256=file_hash(robot_path),
        asset_root=str(root),
        trajectory=str(trajectory),
        trajectory_sha256=file_hash(trajectory),
        prefix_sha256=prefix_hash(np.empty((0, len(names)))),
        checkpoint=None,
        checkpoint_sha256=None,
        checkpoint_step=0,
        start_frame=anchor,
        joint_names=names,
        is_physics_ready=False,
        cloth_path=str(cloth_path),
        robot_path=str(robot_path),
        replay=str((output / "replay.npz").resolve()),
        robot_base_pos_m=config.robot_base_pos_m,
    )
    (output / "workspace.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    (output / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    if config.camera is not None:
        write_json(output / "camera.json", config.camera.model_dump())
    # Reference samples preserve the original time sequence independently of the hold template.
    paths = []
    for hand in ("left", "right"):
        endpoints = [state.left if hand == "left" else state.right for state in hands]
        pos = np.array([state.pos for state in endpoints])
        paths.append(
            HandPath(
                pos,
                np.array([state.quat for state in endpoints]),
                np.array([state.opening for state in endpoints]),
                np.linalg.norm(np.diff(pos, axis=0), axis=1).max() * 60,
                np.linalg.norm(np.diff(pos, n=2, axis=0), axis=1).max(initial=0) * 60**2,
            )
        )
    baseline_plan = ActionPlan(
        start_frame=frames[0].item(),
        start=hands[0],
        actions=[Action(id="reference", duration_frames=len(frames) - 1, left=hands[-1].left, right=hands[-1].right)],
    )
    is_accepted = all(
        path.max_speed_mps <= baseline_plan.limits.max_speed_mps
        and path.max_accel_mps2 <= baseline_plan.limits.max_accel_mps2
        for path in paths
    )
    save_compiled(output / "reference-path", baseline_plan, CompiledPlan(frames, 60, *paths, (), is_accepted))
    write_json(
        output / "import-source.json",
        {
            "config": str(config_path.resolve()),
            "config_sha256": file_hash(config_path),
            "replay_sha256": file_hash(replay_path),
            "capability": "Measured observation; native checkpoint required for physics continuation",
        },
    )


def import_run_manifest(manifest: Path, checkpoint: Path | None, output: Path, source_root: Path | None = None):
    """Prepare the completed native checkpoint using its exported command trajectory."""
    source = json.loads(manifest.read_text(encoding="utf-8"))
    if "argv" not in source:
        if checkpoint is not None:
            raise ValueError(
                "Prepare native checkpoints with their native trajectory; upstream manifests import observations."
            )
        config = resolve_input(source["config"], manifest, source_root)
        import_observation(config, output, source_root)
        return
    if checkpoint is None:
        raise ValueError("A native run manifest requires its completed .pt checkpoint.")
    argv = source["argv"]
    trajectory = Path(argv[argv.index("--trajectory") + 1])
    if file_hash(trajectory) != source["trajectory_sha256"]:
        raise ValueError("Run trajectory changed after export.")
    metadata = Preparation.model_validate(source["workspace"])
    prepare_workspace(output, checkpoint, trajectory, Path(metadata.asset_root), metadata.mesh)
