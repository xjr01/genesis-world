"""Regression checks for persistent corrections, material identity and pose constraints."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .core.action_plan import ActionPlan, compile_action_plan
from .core.browser import compiled_folder
from .core.drafts import default_plan
from .core.ik_options import IKOptions, OrientationAssist, assisted_quaternions, orientation_error
from .core.imports import import_run_manifest
from .core.material import material_atlas, save_atlas
from .core.preview import link_transforms, measured_hands, pose_preview, robot_preview
from .core.runtime import export_trajectory
from .core.trajectory_edit import PathEdits, PositionEdit, SemanticAnchor, duration_scale, edit_path, edit_positions

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parent / "assets"


class AdvancedToolsTests(unittest.TestCase):
    def test_hold_after_ramp_and_stage_clipping(self):
        frames = np.arange(61)
        result = edit_positions(
            frames, np.zeros((61, 3)), (PositionEdit(30, (0, 0, 0.01), 10, "hold_after"),), (10, 50)
        )
        np.testing.assert_allclose(result[:21], 0)
        self.assertAlmostEqual(result[25, 2], 0.005)
        np.testing.assert_allclose(result[30:51, 2], 0.01)
        np.testing.assert_allclose(result[51:], 0)

    def test_hold_after_rejects_crossing_stage_start(self):
        with self.assertRaises(ValueError):
            edit_positions(
                np.arange(61), np.zeros((61, 3)), (PositionEdit(15, (0, 0, 0.01), 10, "hold_after"),), (10, 50)
            )

    def test_semantic_interpolation_and_single_anchor(self):
        frames = np.arange(61)
        anchors = (
            SemanticAnchor(hand="right", frame=10, delta_m=(0, 0, 0)),
            SemanticAnchor(hand="right", frame=30, delta_m=(0, 0, 0.01)),
        )
        result = edit_positions(frames, np.zeros((61, 3)), (), semantic_keyframes=anchors)
        self.assertAlmostEqual(result[20, 2], 0.005)
        np.testing.assert_allclose(result[30:, 2], 0.01)
        single = edit_positions(frames, np.zeros((61, 3)), (), semantic_keyframes=(anchors[1],))
        np.testing.assert_allclose(single[:, 2], 0.01)
        clipped = edit_positions(frames, np.zeros((61, 3)), (), (15, 25), anchors)
        np.testing.assert_allclose(clipped[20], result[20])
        np.testing.assert_allclose(clipped[:15], 0)
        np.testing.assert_allclose(clipped[26:], 0)
        with self.assertRaises(ValueError):
            edit_positions(frames, np.zeros((61, 3)), (), semantic_keyframes=(anchors[0], anchors[0]))

    def test_start_anchor_protection_and_duration_hint(self):
        plan = ActionPlan.model_validate_json((ROOT / "samples/lift.json").read_text())
        source = compile_action_plan(plan)
        with self.assertRaises(ValueError):
            edit_path(
                source, PathEdits(semantic_keyframes=[SemanticAnchor(hand="right", frame=30, delta_m=(0, 0, 0.01))])
            )
        edits = PathEdits.model_validate(
            {"edits": [{"hand": "right", "frame": 30, "delta_m": [0, 0, 0.1], "support_radius": 2}]}
        )
        result = edit_path(source, edits)
        self.assertFalse(result.is_accepted)
        self.assertGreater(duration_scale(result, edits.limits), 1.05)

    def test_assistance_uses_correction_norm_and_local_y(self):
        quat = Rotation.from_euler("z", [[90]], degrees=True).as_quat()[:, [3, 0, 1, 2]]
        result, angles = assisted_quaternions(
            quat, np.array([[0.003, 0.004, 0]]), OrientationAssist(is_enabled=True, gain_deg_per_mm=2)
        )
        self.assertAlmostEqual(angles[0], 10)
        expected = Rotation.from_euler("z", 90, degrees=True) * Rotation.from_euler("y", 10, degrees=True)
        np.testing.assert_allclose(
            Rotation.from_quat(result[:, [1, 2, 3, 0]]).as_matrix()[0], expected.as_matrix(), atol=1e-15
        )
        _, bounded = assisted_quaternions(quat, np.array([[1, 0, 0]]), OrientationAssist(is_enabled=True))
        self.assertEqual(bounded[0], -15)

    def test_axis_relaxation_measures_the_selected_axis(self):
        actual = Rotation.from_euler("y", 45, degrees=True)
        target = Rotation.identity()
        self.assertAlmostEqual(orientation_error(actual, target, IKOptions(orientation_mode="closing_y")), 0)
        self.assertAlmostEqual(orientation_error(actual, target, IKOptions()), 45)
        for mode, expected in (
            ("length_x", (True, False, False)),
            ("closing_y", (False, True, False)),
            ("tool_z", (False, False, True)),
        ):
            self.assertEqual(IKOptions(orientation_mode=mode).rotation_mask, expected)
        options = IKOptions(orientation_mode="closing_y", orientation_assist=OrientationAssist(is_enabled=True))
        self.assertEqual(options.rotation_mask, (True, True, True))

    def test_draft_keeps_start_and_aperture(self):
        original = ActionPlan.model_validate_json((ROOT / "samples/lift.json").read_text())
        draft = default_plan(original.start, original.start_frame, "y", 1)
        self.assertEqual(draft.start, original.start)
        self.assertEqual([a.duration_frames for a in draft.actions], [60, 60, 30])
        np.testing.assert_allclose(np.array(draft.actions[-1].right.pos) - original.start.right.pos, [0, 0.05, 0.05])
        self.assertEqual(draft.actions[-1].right.opening, original.start.right.opening)

    def test_historical_atlas_matches_source_arrays(self):
        for mesh in ("8k", "13k"):
            with np.load(ROOT / "history" / mesh / "atlas.npz", allow_pickle=False) as reference:
                actual = material_atlas(reference["rest_raw_xyz"])
                for key in ("canonical_uv", "width_band", "length_zone", "surface_layer"):
                    value = (
                        actual.canonical_uv
                        if key == "canonical_uv"
                        else actual.width_band
                        if key == "width_band"
                        else actual.length_zone
                        if key == "length_zone"
                        else actual.surface_layer
                    )
                    np.testing.assert_allclose(value, reference[key], atol=1e-15, rtol=0)

    def test_pose_geometry_and_robot_primitives(self):
        with np.load(ROOT / "history/8k/replay.npz", allow_pickle=False) as replay:
            joints = replay["robot_q"][0]
        names = (
            "joint1",
            "joint2",
            "joint3",
            *(f"left_joint1{i}" for i in range(1, 9)),
            *(f"right_joint2{i}" for i in range(1, 9)),
        )
        urdf = ASSETS / "robots/dual_x5_2025_ipc_v1/dual_x5_2025_ipc.urdf"
        hands = measured_hands(link_transforms(urdf, names, joints), joints, names)
        pose = pose_preview(urdf, names, joints, "right")
        np.testing.assert_allclose(pose["position_m"], hands.right.pos, atol=1e-15)
        self.assertEqual(len(pose["fingers"]), 2)
        self.assertGreater(len(robot_preview(urdf, names, joints)), 10)

    def test_atlas_export_landmarks_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "atlas"
            save_atlas(ASSETS / "cloth/short-shirt-8000f.obj", output)
            self.assertTrue((output / "atlas.json").exists())
            with self.assertRaises(FileExistsError):
                save_atlas(ASSETS / "cloth/short-shirt-8000f.obj", output)

    def test_browser_path_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder) / "workspace"
            workspace.mkdir()
            other = Path(folder) / "other"
            other.mkdir()
            (other / "tcp-path.npz").write_bytes(b"test")
            with self.assertRaises(ValueError):
                compiled_folder(workspace, "../other")

    def test_observation_manifest_preserves_data_and_requires_native_execution_state(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "manifest.json"
            manifest.write_text(
                json.dumps({"config": str((ROOT / "history/8k/workbench.json").resolve())}), encoding="utf-8"
            )
            workspace = Path(folder) / "workspace"
            import_run_manifest(manifest, None, workspace)
            with (
                np.load(workspace / "replay.npz", allow_pickle=False) as actual,
                np.load(ROOT / "history/8k/replay.npz", allow_pickle=False) as original,
            ):
                for key in ("cloth_pos", "robot_q", "ipc_link_transforms", "source_frames"):
                    np.testing.assert_array_equal(actual[key], original[key])
            with self.assertRaisesRegex(ValueError, "native .pt checkpoint"):
                export_trajectory(workspace, workspace / "reference-path", Path(folder) / "run")


if __name__ == "__main__":
    unittest.main()
