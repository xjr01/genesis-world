"""CPU checks for action timing, material retargeting and continuation identity."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from .core.action_plan import (
    Action,
    ActionPlan,
    Endpoint,
    GripperEvent,
    HandHeights,
    HandPath,
    Hands,
    compile_action_plan,
)
from .core.points import Bindings, PointBinding, Selections, retarget_points
from .core.trajectory import prefix_hash, read_trajectory
from .core.trajectory_edit import PathEdits, PositionEdit, edit_path, edit_positions


class ActionToolsTests(unittest.TestCase):
    def setUp(self):
        initial = Endpoint(pos=(0.4, 0.2, 0.9), quat=(1, 0, 0, 0), opening=0.02)
        self.plan = ActionPlan(
            start_frame=12,
            start=Hands(left=initial, right=initial),
            actions=[
                Action(
                    id="lift",
                    duration_frames=60,
                    left=Endpoint(pos=(0.4, 0.2, 0.95)),
                    right=initial,
                    arc_height_m=HandHeights(left=0.02),
                )
            ],
            boundaries=["lift"],
        )

    def test_action_intervals_and_stationary_hand(self):
        compiled = compile_action_plan(self.plan)
        self.assertEqual(len(compiled.source_frames), 61)
        self.assertEqual(compiled.checkpoint_frames, (72,))
        np.testing.assert_allclose(compiled.left.pos[-1], (0.4, 0.2, 0.95))
        np.testing.assert_allclose(compiled.right.pos, np.tile((0.4, 0.2, 0.9), (61, 1)))
        self.assertTrue(compiled.is_accepted)

    def test_short_motion_rejected(self):
        self.plan.actions[0].duration_frames = 1
        self.assertFalse(compile_action_plan(self.plan).is_accepted)

    def test_invalid_quaternion_rejected(self):
        self.plan.start.left.quat = (0, 0, 0, 0)
        with self.assertRaises(ValueError):
            compile_action_plan(self.plan)

    def test_opposite_quaternion_sign(self):
        self.plan.actions[0].left.quat = (-1, 0, 0, 0)
        np.testing.assert_allclose(np.abs(compile_action_plan(self.plan).left.quat[:, 0]), 1)

    def test_gripper_event_timing(self):
        self.plan.gripper_events = [GripperEvent(frame=32, command="close", hand="right", duration_frames=10)]
        compiled = compile_action_plan(self.plan)
        np.testing.assert_allclose(compiled.right.opening[:21], 0.02)
        np.testing.assert_allclose(compiled.right.opening[30:], 0)
        np.testing.assert_allclose(compiled.left.opening, 0.02)

    def test_events_override_both_hands_endpoint_apertures(self):
        value = self.plan.model_dump()
        value["actions"][0]["left"]["opening"] = 0.044
        value["gripper_events"] = "t=32 close right duration=10 # absolute source frame"
        compiled = compile_action_plan(ActionPlan.model_validate(value))
        np.testing.assert_array_equal(compiled.left.opening, np.full(61, 0.02))
        self.assertEqual(compiled.right.opening[-1], 0)

    def test_empty_events_retain_endpoint_apertures(self):
        value = self.plan.model_dump()
        value["actions"][0]["left"]["opening"] = 0.044
        value["gripper_events"] = []
        self.assertEqual(compile_action_plan(ActionPlan.model_validate(value)).left.opening[-1], 0.044)

    def test_small_angle_matches_upstream_normalized_linear_branch(self):
        value = self.plan.model_dump()
        value["actions"][0]["right"]["quat"] = (0.9999619230641713, 0.008726535498373935, 0, 0)
        sample = compile_action_plan(ActionPlan.model_validate(value)).right.quat[20]
        np.testing.assert_allclose(sample, (0.9999983227902538, 0.0018315066691999104, 0, 0), rtol=0, atol=1e-14)

    def test_rotation_profile_reaches_exact_target(self):
        value = self.plan.model_dump()
        angle = np.deg2rad(45.02) / 2
        target = (np.cos(angle), np.sin(angle), 0, 0)
        value["actions"][0]["left"]["quat"] = target
        value["actions"][0]["rotation_profile"] = {"left": {"world_x_deg": 45, "tool_x_deg": 0}}
        np.testing.assert_array_equal(compile_action_plan(ActionPlan.model_validate(value)).left.quat[-1], target)

    def test_local_edit_step_gate_and_interior_acceleration(self):
        source = compile_action_plan(self.plan)
        positions = source.left.pos.copy()
        positions[:, 2] = positions[0, 2] + np.arange(61) * 0.005
        source = replace(source, left=HandPath(positions, source.left.quat, source.left.opening, 0, 0))
        request = PathEdits.model_validate(
            {"edits": [{"hand": "left", "frame": 32, "delta_m": [0, 0, 0], "support_radius": 12}]}
        )
        self.assertTrue(edit_path(source, request).is_accepted)
        request.limits.max_step_mm = 4
        self.assertFalse(edit_path(source, request).is_accepted)

    def test_ill_conditioned_node_supports_rejected(self):
        with self.assertRaises(ValueError):
            edit_positions(
                np.arange(61),
                np.zeros((61, 3)),
                (PositionEdit(30, (0, 0, 0.001), 1000000), PositionEdit(31, (0, 0, 0.002), 1000000)),
            )

    def test_nonfinite_tcp_binding_offset_rejected(self):
        with self.assertRaises(ValidationError):
            PointBinding(
                hand="left",
                kind="grasp",
                anchor_action="lift",
                translate_actions=("lift",),
                tcp_offset_world_mm=(0, 0, float("inf")),
            )

    def test_overlapping_events_rejected(self):
        self.plan.gripper_events = [GripperEvent(frame=20, command="close"), GripperEvent(frame=21, command="open")]
        with self.assertRaises(ValueError):
            compile_action_plan(self.plan)

    def test_unknown_boundary_rejected(self):
        self.plan.boundaries = ["missing"]
        with self.assertRaises(ValueError):
            compile_action_plan(self.plan)

    def test_reference_frame_rejected(self):
        self.plan.reference_frame = 13
        with self.assertRaises(ValueError):
            compile_action_plan(self.plan)

    def test_finite_vectors_and_integer_duration(self):
        value = self.plan.model_dump()
        value["actions"][0]["duration_frames"] = True
        with self.assertRaises(ValidationError):
            ActionPlan.model_validate(value)
        value["actions"][0]["duration_frames"] = 60
        value["start"]["left"]["pos"] = [float("nan"), 0, 0]
        with self.assertRaises(ValidationError):
            ActionPlan.model_validate(value)

    def test_material_retarget_preserves_source_and_units(self):
        selections = Selections.model_validate(
            {
                "schema": "workbench-material-regrasp-v1",
                "source_frame": 12,
                "points": {"left": {"material_world_m": [0.5, 0.2, 0.8]}},
            }
        )
        bindings = Bindings(
            bindings=[
                PointBinding(
                    hand="left",
                    kind="grasp",
                    anchor_action="lift",
                    translate_actions=("lift",),
                    tcp_offset_world_mm=(0, 0, 10),
                )
            ]
        )
        result = retarget_points(self.plan, selections, bindings)
        np.testing.assert_allclose(result.actions[0].left.pos, (0.5, 0.2, 0.81))
        self.assertEqual(self.plan.actions[0].left.pos, (0.4, 0.2, 0.95))
        self.assertEqual(result.actions[0].duration_frames, 60)
        self.assertEqual(result.actions[0].right, self.plan.actions[0].right)

    def test_cross_frame_retarget_rejected(self):
        selections = Selections.model_validate({"schema": "workbench-material-regrasp-v1", "source_frame": 13})
        bindings = Bindings(
            bindings=[
                PointBinding(
                    hand="left",
                    kind="grasp",
                    anchor_action="lift",
                    translate_actions=("lift",),
                    tcp_offset_world_mm=(0, 0, 0),
                )
            ]
        )
        with self.assertRaises(ValueError):
            retarget_points(self.plan, selections, bindings)

    def test_local_correction_exact_nodes_and_compact_support(self):
        frames = np.arange(31)
        positions = np.zeros((31, 3))
        edited = edit_positions(
            frames, positions, (PositionEdit(10, (0.01, 0, 0), 8), PositionEdit(15, (0, 0.02, 0), 8))
        )
        np.testing.assert_allclose(edited[10], (0.01, 0, 0), atol=1e-15)
        np.testing.assert_allclose(edited[15], (0, 0.02, 0), atol=1e-15)
        np.testing.assert_allclose(edited[[0, 30]], 0)

    def test_local_correction_rejects_duplicate_nodes(self):
        with self.assertRaises(ValueError):
            edit_positions(
                np.arange(10),
                np.zeros((10, 3)),
                (
                    PositionEdit(5, (0.01, 0, 0), 2),
                    PositionEdit(5, (0, 0.02, 0), 2),
                ),
            )

    def test_duplicate_point_bindings_rejected(self):
        selections = Selections.model_validate(
            {
                "schema": "workbench-material-regrasp-v1",
                "source_frame": 12,
                "points": {"left": {"material_world_m": [0.5, 0.2, 0.8]}},
            }
        )
        binding = PointBinding(
            hand="left",
            kind="grasp",
            anchor_action="lift",
            translate_actions=("lift",),
            tcp_offset_world_mm=(0, 0, 0),
        )
        with self.assertRaises(ValueError):
            retarget_points(self.plan, selections, Bindings(bindings=[binding, binding]))

    def test_duplicate_names_and_rejected_trajectory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trajectory.npz"
            np.savez(path, joint_q=np.zeros((2, 2)), joint_names=["a", "a"])
            with self.assertRaises(ValueError):
                read_trajectory(path)
            np.savez(path, joint_q=np.zeros((2, 2)), joint_names=["a", "b"], accepted=False)
            with self.assertRaises(ValueError):
                read_trajectory(path)

    def test_partial_continuation_identity_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trajectory.npz"
            np.savez(path, joint_q=np.zeros((2, 1)), joint_names=["a"], checkpoint_step=2)
            with self.assertRaises(ValueError):
                read_trajectory(path)

    def test_prefix_identity_changes_with_command(self):
        original = np.zeros((2, 3))
        modified = original.copy()
        modified[1, 1] = 0.001
        self.assertNotEqual(prefix_hash(original), prefix_hash(modified))

    def test_shipped_source_plans(self):
        for path in sorted(Path(__file__).with_name("plans").glob("*.json")):
            with self.subTest(plan=path.name):
                plan = ActionPlan.model_validate_json(path.read_text(encoding="utf-8"))
                compiled = compile_action_plan(plan)
                self.assertEqual(len(compiled.source_frames), sum(a.duration_frames for a in plan.actions) + 1)
                self.assertTrue(np.isfinite(compiled.left.pos).all())
                self.assertTrue(np.isfinite(compiled.right.quat).all())


if __name__ == "__main__":
    unittest.main()
