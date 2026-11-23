"""Prepare a saved cloth view and exact robot seed from the initial pose or native checkpoint."""

from pathlib import Path

import numpy as np
import torch
import trimesh

from genesis.engine.states.solvers import FEMSolverState, RigidSolverState, SimState
from genesis.utils.misc import tensor_to_array

from ...config import create_scene527_config
from .action_plan import Action, ActionPlan
from .kinematics import build_robot, hand_states
from .trajectory import file_hash, prefix_hash, read_trajectory
from .workspace import Preparation


def prepare_workspace(
    output_dir: Path,
    checkpoint_path: Path | None = None,
    trajectory_file: Path | None = None,
    asset_root: Path | None = None,
    mesh_name: str = "55k",
):
    """Save the actual robot seed, cloth state and hold-plan template."""
    root = Path(__file__).resolve().parents[2] if asset_root is None else asset_root.resolve()
    config = create_scene527_config(root, mesh=mesh_name)
    trajectory_path = root / config.assets.trajectory if trajectory_file is None else trajectory_file.resolve()
    trajectory = read_trajectory(trajectory_path)
    checkpoint = None if checkpoint_path is None else checkpoint_path.resolve()
    step = 0
    cloth_pos = None
    seed = trajectory.joint_q[0].copy()
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, tuple):
            state, step = state
        if not isinstance(state, SimState) or step < 0 or step % 2:
            raise ValueError("Use a native checkpoint at a complete two-step action boundary.")
        rigid = tuple(value for value in state.solvers_state if isinstance(value, RigidSolverState))
        fem = tuple(value for value in state.solvers_state if isinstance(value, FEMSolverState))
        if len(rigid) != 1 or len(fem) != 1:
            raise ValueError("Expected one rigid and one FEM state in the garment scene.")
        seed = tensor_to_array(rigid[0].qpos)[0].copy()
        cloth_pos = tensor_to_array(fem[0].pos).reshape(-1, 3).copy()
        if cloth_pos.shape != (config.assets.expected_garment_vertices, 3) or not np.isfinite(cloth_pos).all():
            raise ValueError("Checkpoint cloth topology or positions do not match the selected mesh.")
    prefix_length = step // 2
    if prefix_length > len(trajectory.joint_q):
        raise ValueError("Checkpoint progress exceeds the supplied trajectory.")
    runtime = build_robot(root)
    if set(runtime.joint_names) != set(trajectory.joint_names):
        raise ValueError("Robot joint names differ from supplied trajectory.")
    # Native checkpoints follow the current robot order; command files are name-addressed.
    if checkpoint is None:
        seed = seed[[trajectory.joint_names.index(name) for name in runtime.joint_names]]
    expected = trajectory.joint_q[
        max(0, prefix_length - 1), [trajectory.joint_names.index(name) for name in runtime.joint_names]
    ]
    if seed.shape != expected.shape or np.max(np.abs(seed - expected)) > 0.005:
        raise ValueError("Checkpoint robot state differs from the supplied command endpoint.")
    initial = hand_states(runtime, seed)
    start_frame = max(0, prefix_length - 1)
    plan = ActionPlan(
        start_frame=start_frame,
        reference_frame=start_frame,
        start=initial,
        actions=[Action(id="hold", duration_frames=60, left=initial.left, right=initial.right)],
        boundaries=["hold"],
    )
    cloth_path = root / config.assets.garment_mesh
    robot_path = root / config.assets.robot
    mesh = trimesh.load(cloth_path, force="mesh", process=False)
    if cloth_pos is None:
        from_rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        cloth_pos = mesh.vertices @ from_rotation.T * config.assets.garment_scale + config.assets.garment_pos
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "seed.npz", robot_q=seed, joint_names=runtime.joint_names)
    np.savez_compressed(
        output_dir / "cloth-state.npz",
        cloth_pos=cloth_pos,
        faces=mesh.faces,
        source_frame=start_frame,
        rest_raw_xyz=mesh.vertices,
    )
    metadata = Preparation(
        mesh=mesh_name,
        cloth_sha256=file_hash(cloth_path),
        robot_sha256=file_hash(robot_path),
        asset_root=str(root),
        trajectory=str(trajectory_path),
        trajectory_sha256=file_hash(trajectory_path),
        prefix_sha256=prefix_hash(trajectory.joint_q[:prefix_length]),
        checkpoint=None if checkpoint is None else str(checkpoint),
        checkpoint_sha256=None if checkpoint is None else file_hash(checkpoint),
        checkpoint_step=step,
        start_frame=start_frame,
        joint_names=runtime.joint_names,
    )
    (output_dir / "workspace.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    print(f"Prepared {output_dir.resolve()}; edit plan.json or open the Workbench.")
