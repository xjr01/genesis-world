"""Interactively edit the frame-0 Shadow Hand grasp used by the PBSTF teapot case.

The scene contains the teapot, KUKA, and Shadow Hand at the pose produced by the teapot case's frame-0 inverse
kinematics update. A Tk control window translates and rotates the hand through KUKA inverse kinematics, edits every
Shadow Hand joint, and saves the complete pose as JSON. The simulation remains at frame 0 for the lifetime of the
editor.
"""

from __future__ import annotations

import argparse
import json
import os
import tkinter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk
from typing import NamedTuple

import numpy as np

import genesis as gs
import genesis.utils.geom as geom_utils
from genesis.utils.misc import tensor_to_array
from pbstf_surface_tension import (
    CASE_TEAPOT,
    TeapotSettings,
    _case_settings,
    _teapot_pose,
    _update_teapot_manipulator,
)

SLIDER_LENGTH = 720
POSITION_SLIDER_HALF_RANGE = 5.0
ROTATION_SLIDER_HALF_RANGE_DEGREES = 180.0


@dataclass(frozen=True)
class PoseSnapshot:
    pos: tuple[float, ...]
    quat: tuple[float, ...]


@dataclass(frozen=True)
class JointSnapshot:
    name: str
    qpos: tuple[float, ...]


@dataclass(frozen=True)
class GraspReferenceSnapshot:
    grasp_pos: tuple[float, ...]
    grasp_quat: tuple[float, ...]
    tool_center_point: tuple[float, ...]
    hand_mount_pos: tuple[float, ...]
    hand_mount_quat: tuple[float, ...]
    case_hand_qpos: tuple[float, ...]


@dataclass(frozen=True)
class TeapotSnapshot:
    asset: str
    world_pose: PoseSnapshot


@dataclass(frozen=True)
class KukaSnapshot:
    asset: str
    qpos: tuple[float, ...]
    joints: tuple[JointSnapshot, ...]
    end_effector_world_pose: PoseSnapshot


@dataclass(frozen=True)
class ShadowHandSnapshot:
    asset: str
    qpos: tuple[float, ...]
    joints: tuple[JointSnapshot, ...]
    base_world_pose: PoseSnapshot
    palm_world_pose: PoseSnapshot


@dataclass(frozen=True)
class GraspSnapshot:
    format_version: int
    case: str
    frame: int
    created_utc: str
    reference: GraspReferenceSnapshot
    teapot: TeapotSnapshot
    kuka: KukaSnapshot
    shadow_hand: ShadowHandSnapshot


class GraspScene(NamedTuple):
    scene: gs.Scene
    teapot: gs.engine.entities.KinematicEntity
    kuka: gs.engine.entities.RigidEntity
    hand: gs.engine.entities.RigidEntity
    settings: TeapotSettings


class JointControl(NamedTuple):
    q_idx: int
    variable: tkinter.DoubleVar


class AxisControl(NamedTuple):
    coordinate_idx: int
    variable: tkinter.DoubleVar


def pose_snapshot(entity_or_link) -> PoseSnapshot:
    """Return the world pose of an entity or link using native Python values."""
    return PoseSnapshot(
        pos=tuple(tensor_to_array(entity_or_link.get_pos(relative=False)).tolist()),
        quat=tuple(tensor_to_array(entity_or_link.get_quat(relative=False)).tolist()),
    )


def joint_snapshots(entity) -> tuple[JointSnapshot, ...]:
    """Return every movable joint's local qpos components in entity order."""
    qpos = tensor_to_array(entity.get_qpos())
    return tuple(
        JointSnapshot(
            name=joint.name,
            qpos=tuple(qpos[joint.qs_idx_local].tolist()),
        )
        for joint in entity.joints
        if joint.n_qs > 0
    )


def create_grasp_snapshot(
    grasp_scene: GraspScene,
    grasp_pos: tuple[float, float, float] | None = None,
    grasp_quat: tuple[float, float, float, float] | None = None,
) -> GraspSnapshot:
    """Capture the teapot, KUKA, and Shadow Hand state needed to reproduce the edited grasp."""
    manipulator = grasp_scene.settings.manipulator
    reference_grasp_pos = manipulator.grasp_pos if grasp_pos is None else grasp_pos
    reference_grasp_quat = manipulator.grasp_quat if grasp_quat is None else grasp_quat
    kuka_end_effector = grasp_scene.kuka.get_link(manipulator.kuka_end_effector_link)
    palm = grasp_scene.hand.get_link("palm")
    return GraspSnapshot(
        format_version=1,
        case=CASE_TEAPOT,
        frame=0,
        created_utc=datetime.now(timezone.utc).isoformat(),
        reference=GraspReferenceSnapshot(
            grasp_pos=tuple(reference_grasp_pos),
            grasp_quat=tuple(reference_grasp_quat),
            tool_center_point=tuple(manipulator.tool_center_point),
            hand_mount_pos=tuple(manipulator.hand_mount_pos),
            hand_mount_quat=tuple(manipulator.hand_mount_quat),
            case_hand_qpos=tuple(manipulator.hand_qpos),
        ),
        teapot=TeapotSnapshot(
            asset=grasp_scene.settings.asset,
            world_pose=pose_snapshot(grasp_scene.teapot),
        ),
        kuka=KukaSnapshot(
            asset=manipulator.kuka_asset,
            qpos=tuple(tensor_to_array(grasp_scene.kuka.get_qpos()).tolist()),
            joints=joint_snapshots(grasp_scene.kuka),
            end_effector_world_pose=pose_snapshot(kuka_end_effector),
        ),
        shadow_hand=ShadowHandSnapshot(
            asset=manipulator.hand_asset,
            qpos=tuple(tensor_to_array(grasp_scene.hand.get_qpos()).tolist()),
            joints=joint_snapshots(grasp_scene.hand),
            base_world_pose=pose_snapshot(grasp_scene.hand),
            palm_world_pose=pose_snapshot(palm),
        ),
    )


def write_grasp_snapshot(snapshot: GraspSnapshot, output_path: Path) -> Path:
    """Write a captured grasp snapshot as formatted JSON and return its absolute path."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(asdict(snapshot), output_file, indent=2)
        output_file.write("\n")
    return output_path


def build_grasp_scene(is_viewer_shown: bool) -> GraspScene:
    """Build the teapot and attached manipulator in the case's post-IK frame-0 state."""
    settings = _case_settings(CASE_TEAPOT)
    if settings.teapot is None:
        gs.raise_exception("The teapot grasp editor requires teapot settings.")
    teapot_settings = settings.teapot
    manipulator = teapot_settings.manipulator

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=settings.dt,
            gravity=settings.gravity,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            disable_constraint=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            run_in_thread=True,
            camera_pos=manipulator.camera_pos,
            camera_lookat=manipulator.camera_lookat,
            camera_up=(0.0, 1.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=is_viewer_shown,
    )
    teapot = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=teapot_settings.asset,
            scale=teapot_settings.mesh_scale,
            pos=teapot_settings.offset,
            quat=teapot_settings.quat,
            collision=False,
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(0.66, 0.66, 0.66),
            opacity=0.3,
        ),
        name=teapot_settings.entity_name,
    )
    kuka = scene.add_entity(
        morph=gs.morphs.URDF(
            file=manipulator.kuka_asset,
            scale=manipulator.kuka_scale,
            pos=manipulator.kuka_base_pos,
            quat=manipulator.kuka_base_quat,
            collision=False,
            fixed=True,
        ),
        material=gs.materials.Rigid(
            needs_coup=False,
            gravity_compensation=1.0,
        ),
        name=manipulator.kuka_entity_name,
    )
    hand = scene.add_entity(
        morph=gs.morphs.URDF(
            file=manipulator.hand_asset,
            scale=manipulator.hand_scale,
            collision=False,
        ),
        material=gs.materials.Rigid(
            needs_coup=False,
            gravity_compensation=1.0,
        ),
        name=manipulator.hand_entity_name,
    )
    hand.attach(
        kuka,
        manipulator.kuka_end_effector_link,
        pos=manipulator.hand_mount_pos,
        quat=manipulator.hand_mount_quat,
    )
    scene.build()

    pose = _teapot_pose(0.0, teapot_settings)
    teapot.set_pos(
        pose.pos,
        relative=False,
        skip_forward=True,
    )
    teapot.set_quat(
        pose.quat,
        relative=False,
    )
    kuka.set_qpos(manipulator.kuka_qpos)
    hand.set_qpos(manipulator.hand_qpos)
    _update_teapot_manipulator(kuka, pose, teapot_settings, manipulator.kuka_qpos)
    scene.visualizer.update(force=True)
    return GraspScene(scene, teapot, kuka, hand, teapot_settings)


class GraspEditor:
    """Own the Tk controls that pose the Shadow Hand while the Genesis viewer renders frame 0."""

    def __init__(self, grasp_scene: GraspScene, output_path: Path):
        self.grasp_scene = grasp_scene
        self.output_path = output_path
        self.initial_qpos = tuple(tensor_to_array(grasp_scene.hand.get_qpos()).tolist())
        self.initial_hand_world_pos = tuple(tensor_to_array(grasp_scene.hand.get_pos(relative=False)).tolist())
        self.current_grasp_pos = grasp_scene.settings.manipulator.grasp_pos
        self.current_grasp_quat = grasp_scene.settings.manipulator.grasp_quat
        self.update_after_id: str | None = None
        self.has_pose_update_pending = False
        self.is_closed = False

        pose = _teapot_pose(0.0, grasp_scene.settings)
        manipulator = grasp_scene.settings.manipulator
        tool_world_pos, end_effector_world_quat = geom_utils.transform_pos_quat_by_trans_quat(
            np.array(manipulator.grasp_pos) * grasp_scene.settings.mesh_scale,
            np.array(manipulator.grasp_quat),
            np.array(pose.pos),
            np.array(pose.quat),
        )
        end_effector_world_pos = tool_world_pos - geom_utils.transform_by_quat(
            np.array(manipulator.tool_center_point),
            end_effector_world_quat,
        )
        initial_hand_target_pos, initial_hand_target_quat = geom_utils.transform_pos_quat_by_trans_quat(
            np.array(manipulator.hand_mount_pos),
            np.array(manipulator.hand_mount_quat),
            end_effector_world_pos,
            end_effector_world_quat,
        )
        self.initial_hand_target_pos = tuple(initial_hand_target_pos.tolist())
        self.initial_hand_target_quat = tuple(initial_hand_target_quat.tolist())

        hand_joints = tuple(joint for joint in grasp_scene.hand.joints if joint.n_qs > 0)
        if any(joint.n_qs != 1 or joint.n_dofs != 1 for joint in hand_joints):
            raise ValueError("The teapot grasp editor requires one scalar degree of freedom per Shadow Hand joint.")

        self.root = tkinter.Tk()
        self.root.title("PBSTF Teapot Shadow Hand Grasp Editor")
        self.root.geometry("1080x900")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        ttk.Label(
            self.root,
            text=(
                "Frame 0 is frozen after KUKA inverse kinematics. World-position and relative Roll-Pitch-Yaw "
                "sliders recompute KUKA inverse kinematics, while the 24 joint sliders edit the Shadow Hand grasp."
            ),
            wraplength=1030,
            justify="left",
        ).pack(fill="x", padx=12, pady=(12, 6))

        position_frame = ttk.LabelFrame(self.root, text="Shadow Hand base world position")
        position_frame.pack(fill="x", padx=12, pady=6)
        self.position_controls = self._create_axis_controls(
            frame=position_frame,
            axis_names=("X", "Y", "Z"),
            initial_values=self.initial_hand_world_pos,
            half_range=POSITION_SLIDER_HALF_RANGE,
            resolution=0.001,
            digits=8,
            command=self._schedule_pose_update,
        )

        rotation_frame = ttk.LabelFrame(self.root, text="Shadow Hand world rotation offset (degrees)")
        rotation_frame.pack(fill="x", padx=12, pady=6)
        self.rotation_controls = self._create_axis_controls(
            frame=rotation_frame,
            axis_names=("Roll X", "Pitch Y", "Yaw Z"),
            initial_values=(0.0, 0.0, 0.0),
            half_range=ROTATION_SLIDER_HALF_RANGE_DEGREES,
            resolution=0.1,
            digits=6,
            command=self._schedule_pose_update,
        )

        buttons = ttk.Frame(self.root)
        buttons.pack(fill="x", padx=12, pady=6)
        ttk.Button(buttons, text="Reset Pose", command=self.reset_pose).pack(side="left")
        ttk.Button(buttons, text="Save Grasp Pose", command=self.save).pack(side="left", padx=(8, 0))

        ttk.Label(self.root, text=f"Output: {self.output_path.resolve()}", wraplength=1030).pack(
            fill="x", padx=12, pady=(0, 4)
        )
        self.status = tkinter.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status, wraplength=1030).pack(fill="x", padx=12, pady=(0, 8))

        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.canvas = tkinter.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.joints_frame = ttk.Frame(self.canvas)
        self.joints_window = self.canvas.create_window((0, 0), window=self.joints_frame, anchor="nw")
        self.joints_frame.bind("<Configure>", self._resize_scroll_region)
        self.canvas.bind("<Configure>", self._resize_joints_frame)

        qpos = tensor_to_array(grasp_scene.hand.get_qpos())
        joint_controls = []
        for row, joint in enumerate(hand_joints):
            q_idx = joint.qs_idx_local[0]
            lower = float(joint.dofs_limit[0, 0])
            upper = float(joint.dofs_limit[0, 1])
            variable = tkinter.DoubleVar(value=float(qpos[q_idx]))
            ttk.Label(self.joints_frame, text=joint.name, width=26).grid(row=row, column=0, sticky="w", padx=(4, 8))
            tkinter.Scale(
                self.joints_frame,
                from_=lower,
                to=upper,
                variable=variable,
                command=self._schedule_joint_update,
                orient="horizontal",
                resolution=0.0001,
                digits=7,
                length=SLIDER_LENGTH,
            ).grid(row=row, column=1, sticky="ew")
            ttk.Label(self.joints_frame, text=f"[{lower:.3f}, {upper:.3f}]").grid(
                row=row, column=2, sticky="e", padx=(8, 4)
            )
            joint_controls.append(JointControl(q_idx, variable))
        self.joint_controls = tuple(joint_controls)
        self.joints_frame.columnconfigure(1, weight=1)
        self.root.after(100, self._poll_viewer)

    def _create_axis_controls(
        self,
        frame: ttk.LabelFrame,
        axis_names: tuple[str, str, str],
        initial_values: tuple[float, float, float],
        half_range: float,
        resolution: float,
        digits: int,
        command: Callable[[str], None],
    ) -> tuple[AxisControl, ...]:
        controls = []
        for coordinate_idx, axis_name in enumerate(axis_names):
            initial_value = initial_values[coordinate_idx]
            lower = initial_value - half_range
            upper = initial_value + half_range
            variable = tkinter.DoubleVar(value=initial_value)
            ttk.Label(frame, text=axis_name, width=26).grid(row=coordinate_idx, column=0, sticky="w", padx=(4, 8))
            tkinter.Scale(
                frame,
                from_=lower,
                to=upper,
                variable=variable,
                command=command,
                orient="horizontal",
                resolution=resolution,
                digits=digits,
                length=SLIDER_LENGTH,
            ).grid(row=coordinate_idx, column=1, sticky="ew")
            ttk.Label(frame, text=f"[{lower:.3f}, {upper:.3f}]").grid(
                row=coordinate_idx,
                column=2,
                sticky="e",
                padx=(8, 4),
            )
            controls.append(AxisControl(coordinate_idx, variable))
        frame.columnconfigure(1, weight=1)
        return tuple(controls)

    def _resize_scroll_region(self, _event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_joints_frame(self, event) -> None:
        self.canvas.itemconfigure(self.joints_window, width=event.width)

    def _schedule_joint_update(self, _value) -> None:
        self._schedule_update()

    def _schedule_pose_update(self, _value) -> None:
        self.has_pose_update_pending = True
        self._schedule_update()

    def _schedule_update(self) -> None:
        if self.update_after_id is None:
            self.update_after_id = self.root.after(16, self._apply_controls)

    def _apply_controls(self) -> None:
        self.update_after_id = None
        if self.is_closed or not self.grasp_scene.scene.viewer.is_alive():
            return
        qpos = tensor_to_array(self.grasp_scene.hand.get_qpos())
        for control in self.joint_controls:
            qpos[control.q_idx] = control.variable.get()
        with self.grasp_scene.scene.viewer.lock:
            self.grasp_scene.hand.set_qpos(qpos)
            if self.has_pose_update_pending:
                pose = _teapot_pose(0.0, self.grasp_scene.settings)
                manipulator = self.grasp_scene.settings.manipulator
                hand_world_pos = np.array([control.variable.get() for control in self.position_controls])
                hand_world_delta = hand_world_pos - np.array(self.initial_hand_world_pos)
                hand_target_world_pos = np.array(self.initial_hand_target_pos) + hand_world_delta
                rotation_degrees = np.array([control.variable.get() for control in self.rotation_controls])
                rotation_quat = geom_utils.xyz_to_quat(rotation_degrees, rpy=True, degrees=True)
                hand_target_world_quat = geom_utils.transform_quat_by_quat(
                    np.array(self.initial_hand_target_quat),
                    rotation_quat,
                )
                end_effector_world_quat = geom_utils.transform_quat_by_quat(
                    geom_utils.inv_quat(np.array(manipulator.hand_mount_quat)),
                    hand_target_world_quat,
                )
                end_effector_world_pos = hand_target_world_pos - geom_utils.transform_by_quat(
                    np.array(manipulator.hand_mount_pos),
                    end_effector_world_quat,
                )
                tool_world_pos = end_effector_world_pos + geom_utils.transform_by_quat(
                    np.array(manipulator.tool_center_point),
                    end_effector_world_quat,
                )
                grasp_pos, grasp_quat = geom_utils.inv_transform_pos_quat_by_trans_quat(
                    tool_world_pos,
                    end_effector_world_quat,
                    np.array(pose.pos),
                    np.array(pose.quat),
                )
                self.current_grasp_pos = tuple((grasp_pos / self.grasp_scene.settings.mesh_scale).tolist())
                self.current_grasp_quat = tuple(grasp_quat.tolist())
                current_manipulator = manipulator._replace(
                    grasp_pos=self.current_grasp_pos,
                    grasp_quat=self.current_grasp_quat,
                )
                settings = self.grasp_scene.settings._replace(manipulator=current_manipulator)
                _update_teapot_manipulator(self.grasp_scene.kuka, pose, settings, self.grasp_scene.kuka.get_qpos())
                self.has_pose_update_pending = False
            self.grasp_scene.scene.visualizer.update(force=True)

    def reset_pose(self) -> None:
        if self.update_after_id is not None:
            self.root.after_cancel(self.update_after_id)
            self.update_after_id = None
        for control in self.joint_controls:
            control.variable.set(self.initial_qpos[control.q_idx])
        for control in self.position_controls:
            control.variable.set(self.initial_hand_world_pos[control.coordinate_idx])
        for control in self.rotation_controls:
            control.variable.set(0.0)
        self.has_pose_update_pending = True
        self._apply_controls()
        self.status.set("Shadow Hand and KUKA reset to the frame-0 case pose")

    def save(self) -> None:
        if self.update_after_id is not None:
            self.root.after_cancel(self.update_after_id)
            self._apply_controls()
        with self.grasp_scene.scene.viewer.lock:
            snapshot = create_grasp_snapshot(self.grasp_scene, self.current_grasp_pos, self.current_grasp_quat)
        try:
            output_path = write_grasp_snapshot(snapshot, self.output_path)
        except OSError as error:
            self.status.set(f"Save failed: {error}")
            return
        self.status.set(f"Saved: {output_path}")
        print(f"Saved grasp pose to {output_path}")

    def _poll_viewer(self) -> None:
        if not self.grasp_scene.scene.viewer.is_alive():
            self.close()
            return
        self.root.after(100, self._poll_viewer)

    def close(self) -> None:
        if self.is_closed:
            return
        self.is_closed = True
        if self.update_after_id is not None:
            self.root.after_cancel(self.update_after_id)
            self.update_after_id = None
        self.grasp_scene.scene.viewer.stop()
        self.root.destroy()

    def run(self) -> None:
        """Run the editor until either the Tk controls or Genesis viewer closes."""
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Edit and save the PBSTF teapot frame-0 Shadow Hand grasp")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("teapot_grasp_pose.json"),
        help="JSON path written by the Save Grasp Pose button",
    )
    parser.add_argument(
        "--headless",
        dest="is_headless",
        action="store_true",
        help="Save the initial frame-0 state directly and exit",
    )
    args = parser.parse_args()
    is_test = "PYTEST_VERSION" in os.environ or "PYTEST_CURRENT_TEST" in os.environ
    is_headless = args.is_headless or is_test

    gs.init(backend=gs.cuda, precision="32", logging_level="warning")
    grasp_scene = build_grasp_scene(is_viewer_shown=not is_headless)
    try:
        if is_test:
            create_grasp_snapshot(grasp_scene)
            return
        if args.is_headless:
            output_path = write_grasp_snapshot(create_grasp_snapshot(grasp_scene), args.output)
            print(f"Saved grasp pose to {output_path}")
            return
        GraspEditor(grasp_scene, args.output).run()
    finally:
        grasp_scene.scene.destroy()


if __name__ == "__main__":
    main()
