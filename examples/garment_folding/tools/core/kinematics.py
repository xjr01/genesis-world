"""CPU dual-X5 inverse kinematics and table-clearance diagnostics."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ...config import create_scene527_config
from .action_plan import CompiledPlan, Endpoint, Hands
from .ik_options import IKOptions, orientation_error

TCP_LOCAL = np.array([0.155, 0.001786, 0.014])


@dataclass(frozen=True)
class RobotKinematics:
    scene: object
    robot: object
    joint_names: tuple[str, ...]
    left_arm: tuple[int, ...]
    right_arm: tuple[int, ...]
    left_fingers: tuple[int, ...]
    right_fingers: tuple[int, ...]
    left_link: object
    right_link: object
    tool_links: tuple[object, ...]


@dataclass(frozen=True)
class IKReport:
    is_accepted: bool
    position_error_m: np.ndarray
    orientation_error_deg: np.ndarray
    joint_step_rad: np.ndarray
    joint_limit_margin: np.ndarray
    table_clearance_m: np.ndarray
    has_new_collision: np.ndarray
    inherited_collision_pairs: tuple[tuple[int, int], ...]
    failed_checks: tuple[str, ...]


def build_robot(asset_root: Path | None = None, urdf_path: Path | None = None, base_pos=None) -> RobotKinematics:
    """Build the same neutral-default URDF as the folding scene using the CPU backend."""
    config = create_scene527_config(asset_root)
    root = Path(__file__).resolve().parents[2] if asset_root is None else asset_root.resolve()
    gs.init(backend=gs.cpu, logging_level="warning", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / 60),
        rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True),
        show_viewer=False,
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=str(root / config.assets.robot if urdf_path is None else urdf_path),
            pos=config.assets.robot_pos if base_pos is None else base_pos,
            fixed=True,
            visualization=False,
            collision=True,
            convexify=False,
            decimate=False,
            watertighten=config.assets.robot_watertighten,
        ),
        material=gs.materials.Rigid(),
    )
    scene.build()
    joints = tuple(joint for joint in robot.joints if joint.n_qs)
    names = tuple(joint.name for joint in joints)
    left_arm = tuple(robot.get_joint(f"left_joint1{number}").dofs_idx_local[0] for number in range(1, 7))
    right_arm = tuple(robot.get_joint(f"right_joint2{number}").dofs_idx_local[0] for number in range(1, 7))
    left_fingers = tuple(names.index(f"left_joint1{number}") for number in (7, 8))
    right_fingers = tuple(names.index(f"right_joint2{number}") for number in (7, 8))
    return RobotKinematics(
        scene,
        robot,
        names,
        left_arm,
        right_arm,
        left_fingers,
        right_fingers,
        robot.get_link("left_link16"),
        robot.get_link("right_link26"),
        tuple(
            robot.get_link(f"{hand}_link{prefix}{number}")
            for hand, prefix in (("left", 1), ("right", 2))
            for number in (6, 7, 8)
        ),
    )


def hand_states(runtime: RobotKinematics, qpos: np.ndarray) -> Hands:
    """Measure world TCP pose and mean per-finger aperture through public getters."""
    runtime.robot.set_qpos(qpos, zero_velocity=True)
    states = []
    for link, fingers in ((runtime.left_link, runtime.left_fingers), (runtime.right_link, runtime.right_fingers)):
        quat = tensor_to_array(link.get_quat(relative=False)).reshape(4)
        rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
        pos = tensor_to_array(link.get_pos(relative=False)).reshape(3) + rotation.apply(TCP_LOCAL)
        states.append(Endpoint(pos=tuple(pos), quat=tuple(quat), opening=np.mean(qpos[list(fingers)])))
    return Hands(left=states[0], right=states[1])


def collision_pairs(robot) -> set[tuple[int, int]]:
    """Represent rigid collision pairs independently of reporting order."""
    pairs = tensor_to_array(robot.detect_collision()).reshape(-1, 2)
    return {tuple(sorted(pair)) for pair in pairs.tolist()}


def table_clearance(runtime: RobotKinematics, table_z: float):
    """Measure the minimum tool AABB height relative to the tabletop."""
    return min(tensor_to_array(link.get_AABB()).reshape(2, 3)[0, 2] - table_z for link in runtime.tool_links)


def solve_ik(
    runtime: RobotKinematics,
    compiled: CompiledPlan,
    seed: np.ndarray,
    *,
    options: IKOptions | None = None,
    baseline_joints: np.ndarray | None = None,
) -> tuple[IKReport, np.ndarray]:
    """Solve causal arm targets and retain the exact measured seed in the first row."""
    if not compiled.is_accepted:
        raise ValueError("TCP path limits must pass before solving IK.")
    if seed.shape != (len(runtime.joint_names),) or not np.isfinite(seed).all():
        raise ValueError("Seed must contain one finite robot joint state.")
    options = IKOptions() if options is None else options
    table_z = options.table_z_m
    if baseline_joints is not None and (
        baseline_joints.shape != (len(compiled.source_frames), len(seed)) or not np.isfinite(baseline_joints).all()
    ):
        raise ValueError("Baseline joints must match the sampled path.")
    states = hand_states(runtime, seed)
    for state, path, fingers in (
        (states.left, compiled.left, runtime.left_fingers),
        (states.right, compiled.right, runtime.right_fingers),
    ):
        if abs(state.opening - path.opening[0]) > 2e-5 or np.ptp(seed[list(fingers)]) > 2e-5:
            raise ValueError("Initial aperture must match the seed within 20 micrometres.")
    baseline_pairs = collision_pairs(runtime.robot)
    baseline_clearance = min(table_clearance(runtime, table_z), 0)
    limit_low, limit_high = (tensor_to_array(value) for value in runtime.robot.get_dofs_limit())
    count = len(compiled.source_frames)
    joints = np.empty((count, len(seed)))
    position_error, rotation_error = np.empty((count, 2)), np.empty((count, 2))
    step, margin, clearance = np.empty(count), np.empty(count), np.empty(count)
    collisions = np.empty(count, dtype=bool)
    previous = seed.copy()
    arms = list(runtime.left_arm + runtime.right_arm)
    for ordinal in range(count):
        current = previous.copy() if baseline_joints is None or ordinal == 0 else baseline_joints[ordinal].copy()
        for hand, path, link, arm, fingers in (
            ("left", compiled.left, runtime.left_link, runtime.left_arm, runtime.left_fingers),
            ("right", compiled.right, runtime.right_link, runtime.right_arm, runtime.right_fingers),
        ):
            if options.hand not in ("both", hand):
                continue
            if ordinal:
                current[list(arm)] = previous[list(arm)]
                runtime.robot.set_qpos(current, zero_velocity=True)
                solution = runtime.robot.inverse_kinematics(
                    link=link,
                    pos=path.pos[ordinal],
                    quat=path.quat[ordinal],
                    local_point=TCP_LOCAL,
                    init_qpos=current,
                    dofs_idx_local=arm,
                    max_samples=1,
                    max_solver_iters=80,
                    damping=0.05,
                    max_step_size=0.1,
                    pos_tol=1e-4,
                    rot_tol=1e-4,
                    rot_mask=options.rotation_mask,
                    respect_joint_limit=True,
                )
                current[list(arm)] = tensor_to_array(solution)[list(arm)]
                current[list(fingers)] = path.opening[ordinal]
        if not np.isfinite(current).all():
            raise ValueError(f"Non-finite IK state at sample {ordinal}.")
        measured = hand_states(runtime, current)
        for hand_index, state, path in ((0, measured.left, compiled.left), (1, measured.right, compiled.right)):
            position_error[ordinal, hand_index] = np.linalg.norm(np.array(state.pos) - path.pos[ordinal])
            actual = Rotation.from_quat(np.array(state.quat)[[1, 2, 3, 0]])
            target = Rotation.from_quat(path.quat[ordinal, [1, 2, 3, 0]])
            hand = "left" if hand_index == 0 else "right"
            rotation_error[ordinal, hand_index] = orientation_error(
                actual, target, options if options.hand in ("both", hand) else IKOptions()
            )
        step[ordinal] = np.max(np.abs(current[arms] - previous[arms]))
        margin[ordinal] = np.min(np.minimum(current - limit_low, limit_high - current))
        clearance[ordinal] = table_clearance(runtime, table_z)
        collisions[ordinal] = bool(collision_pairs(runtime.robot) - baseline_pairs)
        joints[ordinal] = current
        previous = current
    checks = (
        ("position", np.all(position_error <= 0.005)),
        ("orientation", np.all(rotation_error <= 2)),
        ("joint_step", np.all(step <= 0.05 + 1e-9)),
        ("joint_limits", np.all(margin >= -1e-6)),
        ("table_clearance", np.all(clearance >= options.min_tool_table_clearance_m)),
        ("clearance_delta", np.all(clearance - baseline_clearance >= options.min_clearance_delta_m)),
        ("new_self_collision", not np.any(collisions)),
    )
    failed_checks = tuple(name for name, is_valid in checks if not is_valid)
    is_accepted = not failed_checks
    return IKReport(
        is_accepted,
        position_error,
        rotation_error,
        step,
        margin,
        clearance,
        collisions,
        tuple(sorted(baseline_pairs)),
        failed_checks,
    ), joints
