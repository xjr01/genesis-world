"""Grasp and pour water, catch a tipped cup, and wipe the spill while the other arm stirs coffee.

Run headlessly with ``python -m examples.pbstf_coffee_water`` from the repository. Add ``--vis`` for the viewer,
``--record`` for a checkpoint at every control frame, or ``--surface`` for a reconstructed liquid surface.
Replay with ``python -m examples.rendering.replay_checkpoints out/pbstf_coffee_water/checkpoints``.
``--no-liquid`` skips liquid creation and simulation for faster trajectory tuning while keeping the soft sponge.
It supports ``--record``; ``--check-motion`` also replaces the sponge with a rigid preview.
From the robot's perspective, world X points right, Y points up, and negative Z points forward across the table.
The default particle scale resolves pouring and the wall film; coarse scales can retain liquid inside the tilted cup.
"""

import argparse
import csv
from dataclasses import dataclass
from enum import IntEnum
import math
from pathlib import Path

import numpy as np
import torch

from scipy.spatial.transform import Rotation
import trimesh

import genesis as gs
from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
from genesis.utils import element, geom, mesh, particle
from genesis.utils.checkpoint_replay import CheckpointWriter
from genesis.utils.misc import tensor_to_array

CUP_ASSET = "meshes/drinking_glass/12-oz-glass.obj"
CUP_CAVITY_ASSET = "meshes/drinking_glass/12-oz-glass-cavity.obj"
COFFEE_CUP_POS = (-0.09, -0.045, 0.0)
WATER_CUP_POS = (0.09, -0.045, 0.0)
CUP_MASS = 0.02
# Fine signed distance fields (SDFs) keep oblique finger contacts above the contact detector's grid-noise threshold.
CONTACT_SDF_CELL_SIZE = 0.0005
COFFEE_FILL_FRACTION = 0.5
WATER_FILL_FRACTION = 1.0
# The high grasp keeps the wrist above the table while the palm stays opposite the pouring lip.
CUP_GRIP_HEIGHT = 0.09
CUP_GRIP_OPENING = 0.034
RECOVERY_GRIP_OPENING = 0.032
ROBOT_ASSET = "urdf/sim1_acone/acone_collision.urdf"
ROD_ASSET = "meshes/glass_stirring_rod/glass_rod.obj"
ROBOT_POS = (-0.06, -0.55, 0.52)
ROBOT_QUAT = (0.5, -0.5, 0.5, 0.5)
TOOL_CENTER = np.array((0.14747, 0.001786, 0.0))
TABLE_Y = -0.045
SPONGE_SIZE = (0.04, 0.06, 0.08)
# The grip height keeps the wider finger mounts above the cup rims during the central wiping stroke.
SPONGE_GRIP_HEIGHT = SPONGE_SIZE[1] - 0.005
SPONGE_START = (0.0, TABLE_Y + 0.001, -0.17)
SPONGE_END = (0.0, TABLE_Y + 0.001, 0.17)
ROD_GRIP_HEIGHT = 0.15
ROD_PARK = (-0.19, -0.025, 0.0)
POUR_QUAT = (
    Rotation.from_euler("z", 10, degrees=True)
    * Rotation.from_quat((0.0, 0.0, math.sqrt(0.5), -math.sqrt(0.5)), scalar_first=True)
).as_quat(scalar_first=True)
# An upward palm pitch clears the finger mounts during the lower recovery grasp.
RECOVERY_GRIP_QUAT = (
    Rotation.from_euler("z", 4, degrees=True) * Rotation.from_quat(POUR_QUAT, scalar_first=True)
).as_quat(scalar_first=True)
MOP_QUAT = (
    Rotation.from_euler("x", 15, degrees=True) * Rotation.from_matrix(((0, -1, 0), (-1, 0, 0), (0, 0, -1)))
).as_quat(scalar_first=True)
ROD_GRIP_QUAT = (
    Rotation.from_euler("y", -20, degrees=True)
    * Rotation.from_euler("z", -30, degrees=True)
    * Rotation.from_matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
).as_quat(scalar_first=True)
PARALLEL_START = 6.4
MOTION_END = 28.0
CONTROL_DT = 0.002
CONTACT_SUBSTEPS = 2


class Phase(IntEnum):
    GRASP_CUP = 0
    POUR = 1
    RELEASE = 2
    REACH_SPONGE = 3
    CATCH = 4
    UPRIGHT = 5
    PLACE_CUP = 6
    RELEASE_CUP = 7
    CLEAR_CUP = 8
    TRANSFER = 9
    APPROACH_SPONGE = 10
    GRASP_SPONGE = 11
    WIPE = 12
    LIFT_ROD = 13
    APPROACH_CUP = 14
    INSERT_ROD = 15
    STIR = 16
    HOLD_ROD = 17
    WITHDRAW_ROD = 18
    PARK_ROD = 19
    REST = 20


@dataclass(frozen=True)
class ToolPose:
    pos: np.ndarray
    quat: np.ndarray


@dataclass(frozen=True)
class MotionTarget:
    right_phase: Phase
    left_phase: Phase
    right: ToolPose
    left: ToolPose
    right_opening: float
    left_opening: float
    rod_pos: np.ndarray
    sponge_pos: np.ndarray
    stir_angle: float


@dataclass(frozen=True)
class SceneObservation:
    cup: ToolPose
    rod: ToolPose
    hands: tuple[ToolPose, ToolPose]
    cup_tilt: float
    water_in_cup: int | None
    is_cup_grasped: bool
    cup_grasp_span: float
    sponge_pos: np.ndarray


@dataclass(frozen=True)
class MotionMeasurement:
    cup_position_error: float
    cup_rotation_error: float
    hands_position_error: np.ndarray
    hands_rotation_error: np.ndarray
    sponge_position_error: float


@dataclass(frozen=True)
class DistanceField:
    geom_idx: int | None
    values: gs.Tensor
    mesh_to_grid: gs.Tensor
    scale: gs.Tensor


@dataclass
class MotionState:
    right_phase: Phase
    phase_started: float
    right_start: ToolPose
    right_target: ToolPose
    cup_grasp: ToolPose
    right_velocity: np.ndarray | None = None
    catch_velocity: np.ndarray | None = None
    catch_acceleration: np.ndarray | None = None
    catch_goal: ToolPose | None = None
    cup_hold_time: float = 0.0
    rod_grasp: ToolPose | None = None
    recovery_cup: ToolPose | None = None
    water_before_reach: int | None = None
    spilled_particles: int = 0
    max_knock_tilt: float = 0.0
    has_caught_cup: bool = False
    left_approach_started: float | None = None
    left_withdraw_started: float | None = None
    sponge_grasp_offset: np.ndarray | None = None


@dataclass
class CoffeeWaterScene:
    scene: gs.Scene
    coffee: gs.engine.entities.PBSTFEntity | None
    water: gs.engine.entities.PBSTFEntity | None
    water_cup: gs.engine.entities.RigidEntity
    cup_cavity: trimesh.Trimesh
    cup_distance_field: DistanceField | None
    robot: gs.engine.entities.RigidEntity
    rod: gs.engine.entities.RigidEntity
    sponge: gs.engine.entities.PBD3DEntity | gs.engine.entities.RigidEntity
    hands_link: tuple[RigidLink, RigidLink]
    arms_dofs_idx: tuple[tuple[int, ...], tuple[int, ...]]
    hands_fingers_qs_idx: tuple[tuple[int, int], tuple[int, int]]
    hands_fingers_dofs_idx: tuple[tuple[int, int], tuple[int, int]]
    hands_finger_links: tuple[tuple[RigidLink, RigidLink], tuple[RigidLink, RigidLink]]
    observed_links_idx: tuple[int, int, int, int]
    gripper_distance_fields: tuple[DistanceField, ...]
    qpos: gs.Tensor
    motion: MotionState
    is_motion_only: bool


def smooth_progress(time, start, end):
    """Interpolate a phase with zero endpoint velocity and acceleration."""
    fraction = np.clip((time - start) / (end - start), 0.0, 1.0)
    return fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))


def interpolate_tool(start, end, progress):
    """Interpolate a tool center and its orientation in the world frame."""
    quat = geom.slerp(start.quat, end.quat, np.array(progress))
    return ToolPose(start.pos + progress * (end.pos - start.pos), quat)


def observe_scene(demo):
    """Read the manipulated link poses and the water still enclosed by the moving cup."""
    positions = demo.scene.rigid_solver.get_links_pos(demo.observed_links_idx)
    quaternions = demo.scene.rigid_solver.get_links_quat(demo.observed_links_idx)
    poses = tensor_to_array(torch.cat((positions, quaternions), dim=-1))
    cup = ToolPose(poses[0, :3], poses[0, 3:])
    rod = ToolPose(poses[1, :3], poses[1, 3:])
    hands = tuple(ToolPose(pose[:3] + geom.transform_by_quat(TOOL_CENTER, pose[3:]), pose[3:]) for pose in poses[2:])
    cup_axis = geom.transform_by_quat(np.array((0.0, 1.0, 0.0)), cup.quat)
    cup_tilt = math.degrees(math.acos(np.clip(cup_axis[1], -1.0, 1.0)))
    water_in_cup = None
    if demo.water is not None:
        distances = sample_distance_field(
            demo.water.get_particles_pos(), demo.cup_distance_field, positions[0], quaternions[0]
        )
        water_in_cup = (distances < 0.0).sum().item()
    contacts = demo.water_cup.get_contacts(with_entity=demo.robot)
    finger_indices = contacts["link_a"].new_tensor([link.idx for link in demo.hands_finger_links[0]])
    is_finger_contact = (contacts["link_a"][None] == finger_indices[:, None]) | (
        contacts["link_b"][None] == finger_indices[:, None]
    )
    force = torch.linalg.vector_norm(contacts["force_a"], dim=-1)
    weights = is_finger_contact * torch.where(force > 0.01, force, 0.0)[None]
    contact_pos = (contacts["position"] - positions[0]) @ geom.quat_to_R(quaternions[0])
    centers = weights @ contact_pos / weights.sum(dim=-1, keepdim=True).clamp_min(gs.EPS)
    cup_grasp_span = torch.linalg.vector_norm(centers[0, (0, 2)] - centers[1, (0, 2)]).item()
    is_cup_grasped = (
        cup_grasp_span > 0.06
        and (weights.sum(dim=-1) > 0.01).all().item()
        and ((centers[:, 1] > 0.025) & (centers[:, 1] < 0.100)).all().item()
    )
    if isinstance(demo.sponge, gs.engine.entities.RigidEntity):
        sponge_pos = demo.sponge.get_pos()
    else:
        sponge_pos = demo.sponge.get_particles_pos().mean(dim=-2)
    return SceneObservation(
        cup, rod, hands, cup_tilt, water_in_cup, is_cup_grasped, cup_grasp_span, tensor_to_array(sponge_pos)
    )


def grasp_pose(cup, grasp):
    """Transform a grasp expressed in an entity frame to the world frame."""
    pos, quat = geom.transform_pos_quat_by_trans_quat(grasp.pos, grasp.quat, cup.pos, cup.quat)
    return ToolPose(pos, quat)


def relative_grasp(hand, entity_pose):
    """Express a hand pose in the grasped entity's frame."""
    pos, quat = geom.inv_transform_pos_quat_by_trans_quat(hand.pos, hand.quat, entity_pose.pos, entity_pose.quat)
    return ToolPose(pos, quat)


def sample_distance_field(points, field, pos, quat):
    """Sample a posed mesh signed distance field at torch points in world coordinates."""
    local_points = geom.inv_transform_by_trans_quat(points, pos, quat)
    grid = local_points @ field.mesh_to_grid[:3, :3].T + field.mesh_to_grid[:3, 3]
    grid = grid * field.scale - 1.0
    return torch.nn.functional.grid_sample(
        field.values,
        grid[:, (2, 1, 0)][None, None, None],
        padding_mode="border",
        align_corners=True,
    )[0, 0, 0, 0]


def surface_clearance(points, fields, positions, quaternions):
    """Return the smallest signed distance from torch points to posed mesh distance fields."""
    return torch.stack(
        [
            sample_distance_field(points, field, pos, quat).amin()
            for field, pos, quat in zip(fields, positions, quaternions)
        ]
    ).amin()


def measure_motion(time, target, observation):
    """Compare measured cup and tool poses with the manipulation references, in meters and degrees."""
    cup_pos, cup_quat = water_cup_pose(time)
    cup_rotation = geom.transform_quat_by_quat(observation.cup.quat, geom.inv_quat(np.array(cup_quat)))
    hands_position_error = np.array(
        [np.linalg.norm(hand.pos - pose.pos) for hand, pose in zip(observation.hands, (target.right, target.left))]
    )
    hands_rotation_error = np.array(
        [
            np.linalg.norm(geom.quat_to_rotvec(geom.transform_quat_by_quat(hand.quat, geom.inv_quat(pose.quat))))
            for hand, pose in zip(observation.hands, (target.right, target.left))
        ]
    )
    return MotionMeasurement(
        np.linalg.norm(observation.cup.pos - cup_pos),
        np.rad2deg(np.linalg.norm(geom.quat_to_rotvec(cup_rotation))),
        hands_position_error,
        np.rad2deg(hands_rotation_error),
        np.linalg.norm(observation.sponge_pos - target.sponge_pos - (0.0, 0.5 * SPONGE_SIZE[1], 0.0)),
    )


def motion_target(time, motion, observation):
    """Advance independent arm phases using the measured cup pose for tipping and recovery."""
    cup_pos, cup_quat = water_cup_pose(time)
    right = grasp_pose(ToolPose(np.array(cup_pos), np.array(cup_quat)), motion.cup_grasp)
    right_opening = 0.044
    sponge_pos = np.array(SPONGE_START)
    rod_pos = np.array(ROD_PARK)
    stir_angle = 0.0
    left_phase = Phase.REST
    if time < 1.0:
        motion.right_phase = Phase.GRASP_CUP
        if time < 0.6:
            right_opening = 0.044 - 0.007 * smooth_progress(time, start=0.0, end=0.6)
        else:
            right_opening = 0.037 - (0.037 - CUP_GRIP_OPENING) * smooth_progress(time, start=0.6, end=0.85)
    elif time < 6.0:
        motion.right_phase = Phase.POUR
        right_opening = CUP_GRIP_OPENING
        if not observation.is_cup_grasped:
            raise RuntimeError(f"The water cup lost a finger contact during pouring at {time:.3f}s.")
    elif time < PARALLEL_START:
        motion.right_phase = Phase.RELEASE
        right_opening = CUP_GRIP_OPENING
        right_opening += (0.044 - right_opening) * smooth_progress(time, start=6.0, end=PARALLEL_START)
    else:
        if motion.right_phase == Phase.RELEASE:
            motion.right_phase = Phase.REACH_SPONGE
            motion.phase_started = PARALLEL_START
            motion.right_start = motion.right_target
            motion.water_before_reach = observation.water_in_cup
        elapsed = time - motion.phase_started
        next_phase = None
        if motion.right_phase in (Phase.REACH_SPONGE, Phase.CATCH, Phase.UPRIGHT):
            motion.max_knock_tilt = max(motion.max_knock_tilt, observation.cup_tilt)
            if observation.water_in_cup is not None:
                motion.spilled_particles = max(0, motion.water_before_reach - observation.water_in_cup)
            if observation.cup_tilt > 65.0:
                raise RuntimeError(f"The tipped cup exceeded the recovery angle at {time:.3f}s.")
        if motion.right_phase in (Phase.UPRIGHT, Phase.PLACE_CUP) and not observation.is_cup_grasped:
            raise RuntimeError(f"The cup lost its opposing wall contacts at {time:.3f}s.")
        if motion.right_phase == Phase.REACH_SPONGE:
            end = ToolPose(
                np.array((SPONGE_START[0], motion.right_start.pos[1], SPONGE_START[2])), motion.right_start.quat
            )
            right = interpolate_tool(motion.right_start, end, smooth_progress(elapsed, start=0.0, end=1.2))
            # Measured tilt keeps recovery timing independent of liquid sampling and simulation.
            if observation.cup_tilt >= 38.0:
                # Intercept the falling cup with open fingers before closing around its wall.
                axis = geom.transform_by_quat(np.array((0.0, 1.0, 0.0)), observation.cup.quat)
                direction = axis * (1.0, 0.0, 1.0)
                direction /= np.linalg.norm(direction)
                rotation_axis = np.cross(np.array((0.0, 1.0, 0.0)), direction)
                quat = geom.rotvec_to_quat(math.radians(54.0) * rotation_axis)
                pivot_offset = 0.037 * direction
                pivot = observation.cup.pos + geom.transform_by_quat(pivot_offset, observation.cup.quat)
                pivot[1] = TABLE_Y
                cup = ToolPose(pivot - geom.transform_by_quat(pivot_offset, quat), quat)
                motion.catch_goal = grasp_pose(cup, ToolPose(np.array((0.0, 0.06, 0.0)), RECOVERY_GRIP_QUAT))
                motion.catch_velocity = (right.pos - motion.right_target.pos) / CONTROL_DT
                motion.catch_acceleration = (motion.catch_velocity - motion.right_velocity) / CONTROL_DT
                next_phase = Phase.CATCH
            elif elapsed > 1.3:
                raise RuntimeError(f"The cup failed to tip at {time:.3f}s: tilt={observation.cup_tilt:.2f}deg.")
        elif motion.right_phase == Phase.CATCH:
            right = interpolate_tool(
                motion.right_start, motion.catch_goal, smooth_progress(elapsed, start=0.0, end=0.35)
            )
            # Quintic braking retains the incoming velocity and acceleration when the measured tilt triggers recovery.
            fraction = np.clip(elapsed / 0.12, 0.0, 1.0)
            carry = 0.12 * fraction * (1.0 - fraction) ** 3 * (1.0 + 3.0 * fraction)
            acceleration_carry = 0.5 * 0.12**2 * fraction**2 * (1.0 - fraction) ** 3
            right = ToolPose(
                right.pos + carry * motion.catch_velocity + acceleration_carry * motion.catch_acceleration,
                right.quat,
            )
            right_opening -= (0.044 - RECOVERY_GRIP_OPENING) * smooth_progress(elapsed, start=0.02, end=0.35)
            motion.cup_hold_time = motion.cup_hold_time + CONTROL_DT if observation.is_cup_grasped else 0.0
            if elapsed >= 0.43 and motion.cup_hold_time >= 0.12:
                motion.recovery_cup = observation.cup
                motion.cup_grasp = relative_grasp(right, observation.cup)
                motion.has_caught_cup = True
                next_phase = Phase.UPRIGHT
            elif elapsed > 0.8:
                raise RuntimeError(f"Both fingers failed to grip the cup wall at {time:.3f}s.")
        elif motion.right_phase == Phase.UPRIGHT:
            cup = interpolate_tool(
                motion.recovery_cup,
                ToolPose(np.array(WATER_CUP_POS) + (0.0, 0.015, 0.0), np.array((1.0, 0.0, 0.0, 0.0))),
                smooth_progress(elapsed, start=0.08, end=1.18),
            )
            right = grasp_pose(cup, motion.cup_grasp)
            right_opening = RECOVERY_GRIP_OPENING
            if elapsed >= 1.4 and observation.cup_tilt < 2.0:
                motion.recovery_cup = observation.cup
                motion.cup_grasp = relative_grasp(right, observation.cup)
                next_phase = Phase.PLACE_CUP
            elif elapsed > 1.9:
                raise RuntimeError(f"The cup failed to stand upright at {time:.3f}s: {observation.cup_tilt:.3f}deg.")
        elif motion.right_phase == Phase.PLACE_CUP:
            pos = motion.recovery_cup.pos + smooth_progress(elapsed, start=0.0, end=0.6) * (
                np.array(WATER_CUP_POS) - motion.recovery_cup.pos
            )
            # The measured orientation preserves the settled finger contacts during placement.
            right = grasp_pose(ToolPose(pos, motion.recovery_cup.quat), motion.cup_grasp)
            right_opening = RECOVERY_GRIP_OPENING
            if elapsed >= 0.8:
                next_phase = Phase.RELEASE_CUP
        elif motion.right_phase == Phase.RELEASE_CUP:
            right = motion.right_start
            right_opening = RECOVERY_GRIP_OPENING
            right_opening += (0.044 - right_opening) * smooth_progress(elapsed, start=0.0, end=0.4)
            if elapsed >= 0.4:
                next_phase = Phase.CLEAR_CUP
        elif motion.right_phase == Phase.CLEAR_CUP:
            withdrawn = ToolPose(motion.right_start.pos + (0.06, 0.09, 0.0), motion.right_start.quat)
            end = ToolPose(np.array((0.16, 0.14, 0.06)), MOP_QUAT)
            if elapsed < 0.45:
                right = interpolate_tool(motion.right_start, withdrawn, smooth_progress(elapsed, start=0.0, end=0.45))
            else:
                right = interpolate_tool(withdrawn, end, smooth_progress(elapsed, start=0.45, end=1.25))
            if elapsed >= 1.25:
                next_phase = Phase.TRANSFER
        elif motion.right_phase == Phase.TRANSFER:
            waypoint = ToolPose(np.array((0.13, 0.06, -0.13)), MOP_QUAT)
            end = ToolPose(np.array((0.0, 0.045, -0.17)), MOP_QUAT)
            if elapsed < 0.8:
                right = interpolate_tool(motion.right_start, waypoint, smooth_progress(elapsed, start=0.0, end=0.8))
            else:
                right = interpolate_tool(waypoint, end, smooth_progress(elapsed, start=0.8, end=1.6))
            right_opening = 0.044 - 0.016 * smooth_progress(elapsed, start=0.0, end=0.6)
            if elapsed >= 1.6:
                next_phase = Phase.APPROACH_SPONGE
        elif motion.right_phase == Phase.APPROACH_SPONGE:
            end = ToolPose(sponge_pos + (0.0, SPONGE_GRIP_HEIGHT, 0.0), MOP_QUAT)
            right = interpolate_tool(motion.right_start, end, smooth_progress(elapsed, start=0.0, end=1.0))
            right_opening = 0.028
            if elapsed >= 1.0:
                next_phase = Phase.GRASP_SPONGE
        elif motion.right_phase == Phase.GRASP_SPONGE:
            right = motion.right_start
            right_opening = 0.028 - 0.010 * smooth_progress(elapsed, start=0.0, end=0.6)
            # Horizontal compensation preserves the clearance set by SPONGE_GRIP_HEIGHT.
            if elapsed >= 0.9:
                if motion.sponge_grasp_offset is None:
                    motion.sponge_grasp_offset = observation.hands[0].pos - observation.sponge_pos
                    motion.sponge_grasp_offset[1] = SPONGE_GRIP_HEIGHT - 0.5 * SPONGE_SIZE[1]
                end = ToolPose(sponge_pos + (0.0, 0.5 * SPONGE_SIZE[1], 0.0) + motion.sponge_grasp_offset, MOP_QUAT)
                right = interpolate_tool(motion.right_start, end, smooth_progress(elapsed, start=0.9, end=1.5))
            if elapsed >= 1.5:
                next_phase = Phase.WIPE
        elif motion.right_phase == Phase.WIPE:
            sponge_pos += smooth_progress(elapsed, start=0.0, end=4.0) * (np.array(SPONGE_END) - sponge_pos)
            right = ToolPose(sponge_pos + (0.0, 0.5 * SPONGE_SIZE[1], 0.0) + motion.sponge_grasp_offset, MOP_QUAT)
            right_opening = 0.018
            if elapsed >= 4.0:
                next_phase = Phase.REST
        elif motion.right_phase == Phase.REST:
            right = motion.right_start
            right_opening = 0.018
            sponge_pos = np.array(SPONGE_END)
        if next_phase is not None:
            motion.right_phase = next_phase
            motion.phase_started = time
            motion.right_start = right
            if next_phase == Phase.PLACE_CUP:
                motion.left_approach_started = time

    elapsed = time - PARALLEL_START
    if elapsed >= 0.8:
        elapsed = 0.8 if motion.left_approach_started is None else time - motion.left_approach_started + 0.8
    # Keep the rod inside the cup until the right forearm clears its upward withdrawal path.
    if elapsed >= 10.4 and motion.right_phase == Phase.REST and motion.left_withdraw_started is None:
        motion.left_withdraw_started = time
    if motion.left_withdraw_started is not None:
        elapsed = time - motion.left_withdraw_started + 10.4
    raised_rod = np.array((-0.19, 0.075, 0.0))
    above_cup = np.array((-0.07, 0.075, 0.0))
    inserted_rod = np.array((-0.07, -0.024, 0.0))
    if 0.0 <= elapsed < 0.8:
        left_phase = Phase.LIFT_ROD
        rod_pos += smooth_progress(elapsed, start=0.0, end=0.8) * (raised_rod - rod_pos)
    elif 0.8 <= elapsed < 3.4:
        left_phase = Phase.APPROACH_CUP
        rod_pos = raised_rod + smooth_progress(elapsed, start=0.8, end=3.4) * (above_cup - raised_rod)
    elif 3.4 <= elapsed < 4.4:
        left_phase = Phase.INSERT_ROD
        rod_pos = above_cup + smooth_progress(elapsed, start=3.4, end=4.4) * (inserted_rod - above_cup)
    elif 4.4 <= elapsed < 10.4:
        left_phase = Phase.STIR
        stir_angle = 6.0 * math.pi * smooth_progress(elapsed, start=4.4, end=10.4)
        rod_pos = np.array((-0.09 + 0.02 * math.cos(stir_angle), -0.024, 0.02 * math.sin(stir_angle)))
    elif elapsed >= 10.4 and motion.left_withdraw_started is None:
        left_phase = Phase.HOLD_ROD
        rod_pos = inserted_rod
        stir_angle = 6.0 * math.pi
    elif 10.4 <= elapsed < 11.4:
        left_phase = Phase.WITHDRAW_ROD
        rod_pos = inserted_rod + smooth_progress(elapsed, start=10.4, end=11.4) * (above_cup - inserted_rod)
        stir_angle = 6.0 * math.pi
    elif 11.4 <= elapsed < 12.4:
        left_phase = Phase.PARK_ROD
        rod_pos = above_cup + smooth_progress(elapsed, start=11.4, end=12.4) * (raised_rod - above_cup)
        stir_angle = 6.0 * math.pi
    elif 12.4 <= elapsed < 13.4:
        left_phase = Phase.PARK_ROD
        rod_pos = raised_rod + smooth_progress(elapsed, start=12.4, end=13.4) * (np.array(ROD_PARK) - raised_rod)
        stir_angle = 6.0 * math.pi
    elif elapsed >= 13.4:
        stir_angle = 6.0 * math.pi
    rod_grasp = ToolPose(np.array((0.0, ROD_GRIP_HEIGHT, 0.0)), ROD_GRIP_QUAT)
    if time >= 1.0:
        if motion.rod_grasp is None:
            motion.rod_grasp = relative_grasp(observation.hands[1], observation.rod)
        elif time >= 1.4:
            # Compensate translational contact drift while retaining the planned wrist orientation.
            measured_offset = observation.hands[1].pos - observation.rod.pos
            motion.rod_grasp = ToolPose(
                motion.rod_grasp.pos + (CONTROL_DT / 0.2) * (measured_offset - motion.rod_grasp.pos),
                motion.rod_grasp.quat,
            )
        rod_grasp = interpolate_tool(rod_grasp, motion.rod_grasp, smooth_progress(time, start=1.0, end=1.4))
    left = grasp_pose(ToolPose(rod_pos, np.array((1.0, 0.0, 0.0, 0.0))), rod_grasp)
    left_opening = 0.0007 * (1.0 - smooth_progress(time, start=0.0, end=0.3))
    motion.right_velocity = (right.pos - motion.right_target.pos) / CONTROL_DT
    motion.right_target = right
    return MotionTarget(
        motion.right_phase, left_phase, right, left, right_opening, left_opening, rod_pos, sponge_pos, stir_angle
    )


def update_motion(demo, time, observation=None):
    """Drive arm joints and fingers while fluid boundaries follow measured rigid poses."""
    if observation is None:
        observation = observe_scene(demo)
    target = motion_target(time, demo.motion, observation)
    qpos = demo.qpos
    errors = []
    for link, dofs, pose in zip(demo.hands_link, demo.arms_dofs_idx, (target.right, target.left)):
        qpos, error = demo.robot.inverse_kinematics(
            link=link,
            pos=pose.pos,
            quat=pose.quat,
            local_point=TOOL_CENTER,
            init_qpos=qpos,
            max_samples=1,
            max_solver_iters=100,
            pos_tol=1e-6,
            rot_tol=1e-6,
            dofs_idx_local=dofs,
            return_error=True,
        )
        error_array = tensor_to_array(error)
        pos_error = np.linalg.norm(error_array[..., :3], axis=-1).max()
        rot_error = np.linalg.norm(error_array[..., 3:], axis=-1).max()
        if not np.isfinite(error_array).all() or pos_error > 2e-4 or rot_error > 2e-4:
            raise RuntimeError(f"{link.name} IK failed at {time:.3f}s: position={pos_error}, rotation={rot_error}.")
        velocity = (qpos[list(dofs)] - demo.qpos[list(dofs)]) / CONTROL_DT
        if time == 0.0:
            velocity.zero_()
        demo.robot.control_dofs_position_velocity(qpos[list(dofs)], velocity, dofs_idx_local=dofs)
        errors.append(pos_error)
    demo.qpos = qpos
    demo.robot.control_dofs_position(target.right_opening, dofs_idx_local=demo.hands_fingers_dofs_idx[0])
    demo.robot.control_dofs_position(target.left_opening, dofs_idx_local=demo.hands_fingers_dofs_idx[1])
    return target, max(errors)


def step_scene(demo):
    """Advance contact steps with one-way fluid boundaries synchronized to measured rigid poses."""
    for _ in range(CONTACT_SUBSTEPS):
        if demo.scene.pbstf_solver.is_active:
            demo.scene.pbstf_solver.set_static_colliders_pose(
                demo.scene.rigid_solver.get_links_pos(demo.observed_links_idx[:2]),
                demo.scene.rigid_solver.get_links_quat(demo.observed_links_idx[:2]),
                colliders_idx=(2, 3),
            )
        demo.scene.step()


def water_cup_pose(time):
    """Return the water-cup reference pose through grasping, lifting, pouring and returning, in seconds."""
    pour_x = COFFEE_CUP_POS[0] + 0.12
    lift_y = COFFEE_CUP_POS[1] + 0.05
    pour_y = COFFEE_CUP_POS[1] + 0.07
    times = np.array((0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0))
    poses = np.array(
        [
            (WATER_CUP_POS[0], WATER_CUP_POS[1], 0.0),
            (WATER_CUP_POS[0], WATER_CUP_POS[1], 0.0),
            (WATER_CUP_POS[0], lift_y, 0.0),
            (pour_x, lift_y, 0.0),
            (pour_x, pour_y, 40.0),
            (pour_x, pour_y, 30.0),
            (pour_x, lift_y, 0.0),
            (WATER_CUP_POS[0], WATER_CUP_POS[1], 0.0),
        ]
    )
    interval = np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 2)
    fraction = np.clip((time - times[interval]) / (times[interval + 1] - times[interval]), 0.0, 1.0)
    fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    pose = poses[interval] + fraction * (poses[interval + 1] - poses[interval])
    angle = math.radians(pose[2])
    return (pose[0], pose[1], WATER_CUP_POS[2]), (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def check_contacts(demo, time):
    """Reject unintended penetration while retaining the intended finger contacts for each manipulation phase."""
    contacts = demo.robot.get_contacts()
    is_forbidden = contacts["penetration"] > 2e-4
    if demo.motion.right_phase in (Phase.REACH_SPONGE, Phase.CATCH):
        is_palm_contact = (
            (contacts["link_a"] == demo.hands_link[0].idx) & (contacts["link_b"] == demo.water_cup.base_link.idx)
        ) | ((contacts["link_b"] == demo.hands_link[0].idx) & (contacts["link_a"] == demo.water_cup.base_link.idx))
        is_forbidden &= ~is_palm_contact
    for fingers, entity in ((demo.hands_finger_links[0], demo.water_cup), (demo.hands_finger_links[1], demo.rod)):
        is_finger_a = (contacts["link_a"] == fingers[0].idx) | (contacts["link_a"] == fingers[1].idx)
        is_finger_b = (contacts["link_b"] == fingers[0].idx) | (contacts["link_b"] == fingers[1].idx)
        is_grasp_contact = (is_finger_a & (contacts["link_b"] == entity.base_link.idx)) | (
            is_finger_b & (contacts["link_a"] == entity.base_link.idx)
        )
        if entity is demo.rod or demo.motion.right_phase <= Phase.RELEASE_CUP:
            is_forbidden &= ~is_grasp_contact
        if demo.is_motion_only and entity is demo.water_cup:
            is_sponge_contact = (is_finger_a & (contacts["link_b"] == demo.sponge.base_link.idx)) | (
                is_finger_b & (contacts["link_a"] == demo.sponge.base_link.idx)
            )
            if demo.motion.right_phase >= Phase.APPROACH_SPONGE:
                is_forbidden &= ~is_sponge_contact
    if is_forbidden.any():
        links_a = tensor_to_array(contacts["link_a"][is_forbidden])
        links_b = tensor_to_array(contacts["link_b"][is_forbidden])
        penetration = tensor_to_array(contacts["penetration"][is_forbidden])
        deepest = penetration.argmax()
        link_a = demo.scene.rigid_solver.links[links_a[deepest]]
        link_b = demo.scene.rigid_solver.links[links_b[deepest]]
        raise RuntimeError(
            f"Rigid penetration at {time:.3f}s: {link_a.name} / {link_b.name}, depth={penetration[deepest]:.6f}m."
        )


def build_scene(
    scale=1500,
    is_viewer_shown=False,
    is_surface=False,
    is_motion_only=False,
    is_liquid_enabled=True,
):
    """Build contact-driven manipulation in meters with a deformable absorbent sponge.

    Motion-only mode uses a rigid sponge preview. Disabling liquid skips its particles and solver steps while retaining
    rigid and sponge dynamics. Liquid boundaries receive their motion through one-way coupling.
    """
    if scale <= 0:
        raise ValueError("Particle scale must be positive.")
    dt = CONTROL_DT / CONTACT_SUBSTEPS
    particle_size = 2.0 / scale
    cup_path = Path(gs.utils.get_assets_dir()) / CUP_ASSET
    cup = mesh.load_mesh(cup_path)
    cup_cavity = mesh.load_mesh(Path(gs.utils.get_assets_dir()) / CUP_CAVITY_ASSET)
    bottom, rim = cup_cavity.bounds[:, 1]
    cups_pos = (COFFEE_CUP_POS, WATER_CUP_POS)
    camera_pos = (0.0, 0.60, 0.15)
    camera_lookat = (0.0, 0.03, -0.12)
    table_collider = gs.options.PBSTFBoxStaticColliderOptions(
        is_collider_adhesion_friction_enabled=True,
        collider_adhesion_compliance=50.0,
        collider_friction=0.5,
        lower=(-0.4, -0.065, -4.0 / 15.0),
        upper=(0.4, -0.045, 4.0 / 15.0),
    )
    # Earlier colliders preserve the supporting surface at incompatible contacts.
    colliders = [table_collider]
    colliders.extend(
        gs.options.PBSTFMeshStaticColliderOptions(
            pos=pos,
            is_collider_adhesion_friction_enabled=True,
            collider_adhesion_compliance=50.0,
            collider_friction=0.1,
            file=str(cup_path),
            sdf_res=128,
        )
        for pos in cups_pos
    )
    colliders.extend(
        (
            gs.options.PBSTFMeshStaticColliderOptions(
                pos=ROD_PARK,
                is_collider_adhesion_friction_enabled=True,
                collider_adhesion_compliance=50.0,
                collider_friction=0.1,
                file=str(Path(gs.utils.get_assets_dir()) / ROD_ASSET),
                sdf_res=128,
            ),
            gs.options.PBSTFAbsorbentBoxStaticColliderOptions(
                pos=SPONGE_START,
                is_collider_adhesion_friction_enabled=True,
                collider_adhesion_compliance=50.0,
                collider_friction=0.5,
                lower=(-0.5 * SPONGE_SIZE[0], 0.0, -0.5 * SPONGE_SIZE[2]),
                upper=(0.5 * SPONGE_SIZE[0], SPONGE_SIZE[1], 0.5 * SPONGE_SIZE[2]),
                absorption_rate=4000.0,
                absorption_capacity_fraction=1.0,
                pbd_entity_name="sponge",
            ),
        )
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, -9.8, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.implicitfast,
            enable_neutral_collision=True,
            noslip_iterations=10,
            enable_torsional_friction=True,
            enable_rolling_friction=True,
            constraint_timeconst=0.004,
        ),
        pbd_options=gs.options.PBDUnifiedOptions(
            particle_size=0.005,
            lower_bound=(-0.4, -1.0 / 15.0, -4.0 / 15.0),
            upper_bound=(0.4, 8.0 / 15.0, 4.0 / 15.0),
            max_solver_iterations=30,
            constraint_acceleration=0.85,
        ),
        pbstf_options=gs.options.PBSTFOptions(
            diffusion_coeff=0.001,
            particle_size=particle_size,
            max_solver_iterations=20,
            topology_rebuild_interval=10,
            max_surface_neighbors=128,
            max_localmesh_neighbors=64,
            enable_pca_normals=False,
            static_colliders=colliders if is_liquid_enabled and not is_motion_only else [],
            lower_bound=(-0.4, -1.0 / 15.0, -4.0 / 15.0),
            upper_bound=(0.4, 8.0 / 15.0, 4.0 / 15.0),
        ),
        vis_options=gs.options.VisOptions(
            ambient_light=(0.4, 0.4, 0.4),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_up=(0.0, 1.0, 0.0),
            camera_fov=80,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=is_viewer_shown,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(
            fixed=True,
            lower=table_collider.lower,
            upper=table_collider.upper,
        ),
        material=gs.materials.Rigid(friction=3.0, coup_friction=0.0, is_coup_reaction_enabled=False),
        surface=gs.surfaces.Default(color=(0.36, 0.24, 0.14)),
        name="table",
    )
    liquids = []
    cups = []
    for pos, fill_fraction, concentration, name in zip(
        cups_pos, (COFFEE_FILL_FRACTION, WATER_FILL_FRACTION), (1.0, 0.0), ("coffee", "water")
    ):
        # The water cup presents its convex envelope to the fingers; the coffee cup exposes its cavity to the rod.
        cups.append(
            scene.add_entity(
                morph=gs.morphs.Mesh(
                    pos=pos,
                    file=str(cup_path),
                    convexify=True,
                    decompose_object_error_threshold=0.05 if name == "coffee" else math.inf,
                    align=False,
                    fixed=name == "coffee",
                ),
                material=gs.materials.Rigid(is_coup_reaction_enabled=False, sdf_cell_size=CONTACT_SDF_CELL_SIZE),
                surface=gs.surfaces.Default(color=(0.65, 0.75, 0.85, 0.25)),
                name=f"{name}_cup",
            )
        )
        if is_motion_only or not is_liquid_enabled:
            continue
        liquid_top = bottom + fill_fraction * (rim - bottom)
        particles = particle.mesh_cavity_to_particles(
            cup,
            p_size=particle_size,
            seed=(0.0, liquid_top - 0.5 * particle_size, 0.0),
            max_height=liquid_top - 0.5 * particle_size,
        )
        liquids.append(
            scene.add_entity(
                morph=gs.morphs.Particles(
                    pos=pos,
                    positions=particles,
                ),
                material=gs.materials.PBSTF.Liquid(
                    sampler="regular",
                    rho=1000.0,
                    density_compliance=843750.0,
                    surface_tension_compliance=1.0 / 225.0,
                    surface_distance_compliance=40.0,
                    interior_distance_compliance=180.0,
                    surface_viscosity=0.5,
                    interior_viscosity=0.5,
                    c_init=concentration,
                ),
                surface=gs.surfaces.Default(vis_mode="recon" if is_surface else "particle"),
                name=name,
            )
        )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            pos=ROBOT_POS,
            quat=ROBOT_QUAT,
            file=ROBOT_ASSET,
            convexify=False,
            fixed=True,
        ),
        material=gs.materials.Rigid(
            coup_friction=0.5,
            is_coup_reaction_enabled=False,
            sdf_cell_size=CONTACT_SDF_CELL_SIZE,
            gravity_compensation=1.0,
        ),
        name="sim1",
    )
    rod = scene.add_entity(
        morph=gs.morphs.Mesh(
            pos=ROD_PARK,
            file=ROD_ASSET,
            convexify=True,
            align=False,
        ),
        material=gs.materials.Rigid(rho=2500.0, is_coup_reaction_enabled=False),
        surface=gs.surfaces.Default(color=(0.65, 0.8, 0.9, 0.45)),
        name="glass_rod",
    )
    sponge_pos = (SPONGE_START[0], SPONGE_START[1] + 0.5 * SPONGE_SIZE[1], SPONGE_START[2])
    if is_motion_only:
        sponge = scene.add_entity(
            morph=gs.morphs.Box(
                pos=sponge_pos,
                size=SPONGE_SIZE,
            ),
            material=gs.materials.Rigid(rho=30.0, needs_coup=False),
            surface=gs.surfaces.Default(color=(0.95, 0.68, 0.12)),
            name="sponge",
        )
    else:
        sponge_vertices, sponge_elements = element.create_tetrahedral_grid(
            lower=tuple(-0.5 * size for size in SPONGE_SIZE),
            upper=tuple(0.5 * size for size in SPONGE_SIZE),
            resolution=(10, 7, 20),
        )
        sponge = scene.add_entity(
            morph=gs.morphs.TetrahedralMesh(
                pos=sponge_pos,
                vertices=sponge_vertices,
                elements=sponge_elements,
            ),
            material=gs.materials.PBD.Elastic(rho=30.0, stretch_relaxation=0.25, volume_relaxation=0.15),
            surface=gs.surfaces.Default(color=(0.95, 0.68, 0.12)),
            name="sponge",
        )
    scene.build()
    # Normalize mass after convex decomposition so the empty-cup weight stays explicit.
    cups[1].set_mass(mass=CUP_MASS)
    hands_link = tuple(robot.get_link(name) for name in ("right_link26", "left_link16"))
    arms_dofs_idx = tuple(
        tuple(robot.get_joint(f"{side}_joint{idx}").dofs_idx_local[0] for idx in range(start, start + 6))
        for side, start in (("right", 21), ("left", 11))
    )
    hands_fingers_qs_idx = tuple(
        tuple(robot.get_joint(f"{side}_joint{idx}").qs_idx_local[0] for idx in indices)
        for side, indices in (("right", (27, 28)), ("left", (17, 18)))
    )
    hands_fingers_dofs_idx = tuple(
        tuple(robot.get_joint(f"{side}_joint{idx}").dofs_idx_local[0] for idx in indices)
        for side, indices in (("right", (27, 28)), ("left", (17, 18)))
    )
    hands_finger_links = tuple(
        tuple(robot.get_link(f"{side}_link{idx}") for idx in indices)
        for side, indices in (("right", (27, 28)), ("left", (17, 18)))
    )
    for dofs in arms_dofs_idx:
        robot.set_dofs_kp(kp=[5000, 5000, 5000, 1500, 1500, 1500], dofs_idx_local=dofs)
        robot.set_dofs_kv(kv=[100, 100, 100, 30, 30, 30], dofs_idx_local=dofs)
        robot.set_dofs_force_range(lower=[-27, -27, -27, -7, -7, -7], upper=[27, 27, 27, 7, 7, 7], dofs_idx_local=dofs)
    for dofs, fingers, friction, torsion in zip(hands_fingers_dofs_idx, hands_finger_links, (2.0, 5.0), (0.001, 0.02)):
        robot.set_dofs_kp(kp=3000, dofs_idx_local=dofs)
        robot.set_dofs_kv(kv=30, dofs_idx_local=dofs)
        robot.set_dofs_force_range(lower=-20, upper=20, dofs_idx_local=dofs)
        for link in fingers:
            link.set_friction(friction)
            link.set_friction_torsional(torsion)
    # Rolling resistance at the narrow rod contacts keeps its axis aligned with the fingers.
    scene.rigid_solver.set_geoms_friction_rolling(friction_rolling=0.0)
    for link in (*hands_finger_links[1], rod.base_link):
        link.set_friction_rolling(friction_rolling=0.003)
    # Firm table contacts support the cup's lower rim while friction resists sliding under a side impact.
    contact_links = (*hands_finger_links[0], *hands_finger_links[1], cups[1].base_link, rod.base_link, table.base_link)
    contact_geoms_idx = tuple(collision_geom.idx for link in contact_links for collision_geom in link.geoms)
    scene.rigid_solver.set_sol_params(
        sol_params=[0.004, 1.0, 0.99, 0.999, 0.001, 0.5, 2.0], geoms_idx=contact_geoms_idx
    )
    qpos = robot.get_qpos()
    for dofs in arms_dofs_idx:
        qpos[list(dofs)] = qpos.new_tensor((0.0, 3.0, 1.5, -1.4, -1.4, 0.0))
    qpos[list(arms_dofs_idx[1])] = qpos.new_tensor((0.5, 2.8, 1.3, -1.5, 1.0, 0.0))
    qpos[list(hands_fingers_qs_idx[0])] = 0.044
    qpos[list(hands_fingers_qs_idx[1])] = 0.0007
    gripper_distance_fields = ()
    cup_distance_field = None
    if liquids:
        cavity_scale = cup_cavity.extents.max()
        normalized_cavity = cup_cavity.copy()
        normalization = np.eye(4)
        normalization[:3, :3] /= cavity_scale
        normalization[:3, 3] = -cup_cavity.bounds.mean(axis=0) / cavity_scale
        normalized_cavity.apply_transform(normalization)
        cavity_sdf = mesh.compute_sdf_data(normalized_cavity, res=128)
        cup_distance_field = DistanceField(
            None,
            qpos.new_tensor(cavity_scale * cavity_sdf["voxels"])[None, None],
            qpos.new_tensor(cavity_sdf["T_mesh_to_sdf"] @ normalization),
            qpos.new_tensor(2.0 / (np.array(cavity_sdf["voxels"].shape) - 1)),
        )
        gripper_distance_fields = tuple(
            DistanceField(
                collision_geom.idx,
                qpos.new_tensor(collision_geom.sdf_val)[None, None],
                qpos.new_tensor(collision_geom.T_mesh_to_sdf),
                2.0 / qpos.new_tensor(collision_geom.sdf_res - 1),
            )
            for link in (hands_link[0], *hands_finger_links[0])
            for collision_geom in link.geoms
        )
    cup_grasp = ToolPose(np.array((0.0, CUP_GRIP_HEIGHT, 0.0)), POUR_QUAT)
    right_start = grasp_pose(ToolPose(np.array(WATER_CUP_POS), np.array((1.0, 0.0, 0.0, 0.0))), cup_grasp)
    demo = CoffeeWaterScene(
        scene,
        liquids[0] if liquids else None,
        liquids[1] if liquids else None,
        cups[1],
        cup_cavity,
        cup_distance_field,
        robot,
        rod,
        sponge,
        hands_link,
        arms_dofs_idx,
        hands_fingers_qs_idx,
        hands_fingers_dofs_idx,
        hands_finger_links,
        (cups[1].base_link.idx, rod.base_link.idx, hands_link[0].idx, hands_link[1].idx),
        gripper_distance_fields,
        qpos,
        MotionState(Phase.GRASP_CUP, 0.0, right_start, right_start, cup_grasp),
        is_motion_only,
    )
    update_motion(demo, time=0.0)
    robot.set_qpos(demo.qpos)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, default=1500)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--vis", dest="is_viewer_shown", action="store_true")
    parser.add_argument("--record", dest="is_recording", action="store_true")
    parser.add_argument("--surface", dest="is_surface", action="store_true")
    parser.add_argument("--check-motion", dest="is_motion_only", action="store_true")
    parser.add_argument(
        "--no-liquid",
        dest="is_liquid_enabled",
        action="store_false",
        help="Skip liquid creation and simulation for faster trajectory tuning; keep the soft sponge and support --record.",
    )
    parser.add_argument("--output", type=Path, default=Path("out/pbstf_coffee_water"))
    args = parser.parse_args()
    if args.scale <= 0 or (args.steps is not None and args.steps <= 0):
        parser.error("--scale and --steps must be positive")
    gs.init(backend=gs.cuda, precision="32", logging_level="info")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    demo = build_scene(
        args.scale,
        args.is_viewer_shown,
        args.is_surface,
        args.is_motion_only,
        args.is_liquid_enabled,
    )
    dt = demo.scene.dt * CONTACT_SUBSTEPS
    steps = args.steps if args.steps is not None else round(MOTION_END / dt)
    initial_mass = None
    if demo.coffee is not None:
        initial_mass = tensor_to_array(demo.coffee.get_mass() + demo.water.get_mass()).sum()
    checkpoint_writer = None
    if args.is_recording:
        checkpoint_writer = CheckpointWriter(demo.scene, output / "checkpoints", dt)
        checkpoint_writer.write_frame()
        gs.logger.info(f"Saving checkpoints to {checkpoint_writer.directory}")
    previous_phases = None
    observation = observe_scene(demo)
    milestone_steps = {round(time / dt) - 1 for time in (3.0, 4.0, 6.0, 6.4, 14.0, 20.0, MOTION_END)}
    absorbed_particles = 0
    absorbed_water_particles = 0
    gripper_clearance = math.inf
    previous_stir_angle = None
    measured_stir_angle = 0.0
    gripper_geoms_idx = tuple(field.geom_idx for field in demo.gripper_distance_fields)
    with (output / "metrics.csv").open("w", newline="", encoding="ascii") as metrics:
        writer = csv.writer(metrics)
        writer.writerow(
            (
                "step",
                "time",
                "right_phase",
                "left_phase",
                "ik_error_m",
                "right_hand_error_m",
                "left_hand_error_m",
                "cup_position_error_m",
                "cup_rotation_error_deg",
                "cup_tilt_deg",
                "knock_spilled_particles",
                "water_particles_in_cup",
                "water_particles_before_knock",
                "max_knock_tilt_deg",
                "cup_grasp_span_m",
                "right_finger_opening_m",
                "left_finger_opening_m",
                "pour_liquid_gripper_clearance_m",
                "stir_turns",
                "measured_stir_turns",
                "sponge_position_error_m",
                "active_particles",
                "mass_kg",
                "coffee_amount",
                "variance",
                "water_in_coffee_cup",
                "absorbed_particles",
                "absorbed_water_particles",
                "sponge_wetness",
            )
        )
        for step in range(steps):
            if args.steps is None and step > 0 and target.right_phase == target.left_phase == Phase.REST:
                break
            time = (step + 1) * dt
            target, ik_error = update_motion(demo, time, observation)
            step_scene(demo)
            observation = observe_scene(demo)
            measurement = measure_motion(time, target, observation)
            if measurement.hands_position_error[1] > 0.005 or (
                target.right_phase not in (Phase.REACH_SPONGE, Phase.CATCH)
                and measurement.hands_position_error[0] > 0.005
            ):
                raise RuntimeError(f"Arm tracking failed at {time:.3f}s: {measurement.hands_position_error}m.")
            if target.left_phase == Phase.STIR:
                rod_offset = observation.rod.pos - COFFEE_CUP_POS
                angle = math.atan2(rod_offset[2], rod_offset[0])
                if previous_stir_angle is not None:
                    difference = angle - previous_stir_angle
                    measured_stir_angle += math.atan2(math.sin(difference), math.cos(difference))
                previous_stir_angle = angle
            check_contacts(demo, time)
            demo.scene.rigid_solver.check_errno()
            if demo.water is not None and time <= 6.0:
                clearance = surface_clearance(
                    demo.water.get_particles_pos(),
                    demo.gripper_distance_fields,
                    demo.scene.rigid_solver.get_geoms_pos(gripper_geoms_idx),
                    demo.scene.rigid_solver.get_geoms_quat(gripper_geoms_idx),
                ).item()
                gripper_clearance = min(gripper_clearance, clearance)
                if clearance < 1.0 / args.scale:
                    raise RuntimeError(f"Pouring liquid reached the right gripper at {time:.3f}s: {clearance:.6f}m.")
            if time <= 6.0 and (measurement.cup_position_error > 0.002 or measurement.cup_rotation_error > 2.0):
                raise RuntimeError(
                    f"Pour tracking failed at {time:.3f}s: position={measurement.cup_position_error:.6f}m, "
                    f"rotation={measurement.cup_rotation_error:.3f}deg."
                )
            if target.right_phase >= Phase.CLEAR_CUP and (
                measurement.cup_position_error > 0.002 or observation.cup_tilt > 2.0
            ):
                raise RuntimeError(
                    f"Cup recovery failed at {time:.3f}s: position={measurement.cup_position_error:.6f}m, "
                    f"tilt={observation.cup_tilt:.3f}deg."
                )
            if target.right_phase in (Phase.WIPE, Phase.REST) and measurement.sponge_position_error > 0.01:
                raise RuntimeError(f"Sponge tracking failed at {time:.3f}s: {measurement.sponge_position_error:.6f}m.")
            phases = (target.right_phase, target.left_phase)
            is_milestone = phases != previous_phases or step + 1 == steps or step in milestone_steps
            if is_milestone:
                gs.logger.info(f"{time:.3f}s: right={target.right_phase.name}, left={target.left_phase.name}")
            previous_phases = phases
            if checkpoint_writer is not None:
                checkpoint_writer.write_frame()
            if (step + 1) % 100 == 0 or step + 1 == steps or is_milestone:
                motion_row = (
                    step + 1,
                    time,
                    target.right_phase.name,
                    target.left_phase.name,
                    ik_error,
                    *measurement.hands_position_error,
                    measurement.cup_position_error,
                    measurement.cup_rotation_error,
                    observation.cup_tilt,
                    demo.motion.spilled_particles,
                    observation.water_in_cup,
                    demo.motion.water_before_reach,
                    demo.motion.max_knock_tilt,
                    observation.cup_grasp_span,
                    target.right_opening,
                    target.left_opening,
                    gripper_clearance if demo.water is not None else None,
                    target.stir_angle / (2.0 * math.pi),
                    measured_stir_angle / (2.0 * math.pi),
                    measurement.sponge_position_error,
                )
                if demo.water is None:
                    writer.writerow(motion_row + (None,) * 8)
                    metrics.flush()
                    continue
                state = demo.scene.pbstf_solver.get_state(demo.scene.sim.cur_substep_local)
                concentrations = tensor_to_array(state.c)
                positions = tensor_to_array(state.pos)
                velocities = tensor_to_array(state.vel)
                if not all(np.isfinite(values).all() for values in (positions, velocities, concentrations)):
                    raise RuntimeError("Liquid state contains non-finite values.")
                mass = tensor_to_array(demo.coffee.get_mass() + demo.water.get_mass()).sum()
                if abs(mass - initial_mass) > initial_mass * 2e-6:
                    raise RuntimeError("Liquid mass changed during the simulation.")
                demo.scene.pbstf_solver.check_errno()
                is_absorbed = tensor_to_array(state.absorbed_collider_idx) >= 0
                absorbed_particles = is_absorbed.sum()
                absorbed_water_particles = is_absorbed[:, demo.water.particle_start : demo.water.particle_end].sum()
                wetness = tensor_to_array(demo.scene.pbstf_solver.get_static_collider_wetness(4))
                water_pos = positions[:, demo.water.particle_start : demo.water.particle_end] - COFFEE_CUP_POS
                is_in_cup = demo.cup_cavity.contains(water_pos.reshape((-1, 3)))
                writer.writerow(
                    motion_row
                    + (
                        tensor_to_array(state.active).sum(),
                        mass,
                        concentrations.sum(),
                        concentrations.var(),
                        is_in_cup.mean(),
                        absorbed_particles,
                        absorbed_water_particles,
                        wetness.mean(),
                    )
                )
                metrics.flush()
        if steps * dt >= MOTION_END:
            if target.right_phase != Phase.REST or target.left_phase != Phase.REST or not demo.motion.has_caught_cup:
                raise RuntimeError("The manipulation sequence did not complete.")
            if abs(measured_stir_angle - 6.0 * math.pi) > 0.05:
                raise RuntimeError(f"The rod completed {measured_stir_angle / (2.0 * math.pi):.6f} stirring turns.")
            if np.max(np.abs(observation.rod.pos - ROD_PARK)) > 0.002:
                raise RuntimeError(f"The rod missed its parking position: {observation.rod.pos}.")
            if demo.water is not None and absorbed_water_particles == 0:
                raise RuntimeError("The wiping stroke absorbed no water; inspect the spill and sponge path.")
            if demo.water is not None and not 0 < observation.water_in_cup < demo.motion.water_before_reach:
                raise RuntimeError("Cup recovery must spill some water and retain some in the cup.")


if __name__ == "__main__":
    main()
