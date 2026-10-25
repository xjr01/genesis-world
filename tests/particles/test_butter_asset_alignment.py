import json
from dataclasses import replace

import numpy as np
import pytest
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from examples.butter_cut_spread.aligned_motion import continue_motion
from examples.butter_cut_spread.asset_demo import asset_motion
from examples.butter_cut_spread.asset_runtime import file_digest, source_signature, verify_prepared
from examples.butter_cut_spread.config import Config
from examples.butter_cut_spread.geometry import rigid_fit


@pytest.mark.parametrize("backend", [None])
def test_horizontal_downstroke_and_single_draw_cut_taller_butter(backend):
    profile = np.array([[-0.017, 0.011], [0.018, 0.011], [0.016, -0.011], [-0.017, -0.004]])
    blade = trimesh.creation.extrude_triangulation(profile, np.array([[0, 1, 2], [0, 2, 3]]), 0.0008)
    blade.vertices = blade.vertices[:, [2, 0, 1]] - (0.0004, 0, 0)
    orientation = Rotation.from_euler("x", -15, degrees=True) * Rotation.from_euler("y", -90, degrees=True)
    blade.vertices = orientation.inv().apply(blade.vertices)
    config = replace(Config(), butter_size=(0.12, 0.026, 0.039), butter_center=(0, 0, 0.025), cut_end_time=2.9)
    assets = {"cutting_roll_degrees": -15, "cutting_direction": -1, "lift_outward_clearance_m": 0.003}
    times = np.array([0, 0.6, 1.5, 2.9, 3.7])
    centers, _, _ = asset_motion(times, config, assets, blade)
    lower = np.array(config.butter_center)[1:] - np.array(config.butter_size)[1:] / 2
    upper = lower + config.butter_size[1:]
    y, z = np.meshgrid(
        np.linspace(lower[0] + 0.0002, upper[0] - 0.0002, 91), np.linspace(lower[1] + 0.0002, upper[1] - 0.0002, 131)
    )
    points = np.column_stack((y.reshape(-1), z.reshape(-1)))
    covered = np.zeros(len(points), dtype=bool)
    for start, end in zip(centers[:3, 1:], centers[1:4, 1:]):
        hull = ConvexHull(np.concatenate((profile + start, profile + end)))
        covered |= np.all(points @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-12, axis=1)
    assert covered.all()
    assert np.all(np.diff(centers[:4, 1]) <= 0)
    np.testing.assert_allclose(centers[3:, 1], centers[3, 1])


@pytest.mark.parametrize("backend", [None])
def test_runtime_refuses_missing_cache_or_changed_prepared_assets(backend, tmp_path):
    geometry = tmp_path / "geometry.txt"
    geometry.write_text("prepared collision geometry")
    cache = tmp_path / "sdf.gsd"
    cache.write_bytes(b"prepared cache")
    report = {
        "trajectory_sources": source_signature(),
        "files": {geometry.name: file_digest(geometry)},
        "sdf": [{"cache_path": str(cache)}],
    }
    (tmp_path / "runtime-preparation.json").write_text(json.dumps(report))
    verify_prepared(tmp_path)
    cache.unlink()
    with pytest.raises(ValueError, match="cache is missing"):
        verify_prepared(tmp_path)
    cache.write_bytes(b"prepared cache")
    geometry.write_text("changed geometry")
    with pytest.raises(ValueError, match="Prepared asset changed"):
        verify_prepared(tmp_path)


@pytest.mark.parametrize("backend", [None])
@pytest.mark.parametrize("direction", [-1, 1])
def test_completed_cut_lifts_with_one_outward_peel_and_no_reverse_draw(backend, direction):
    config = replace(
        Config(),
        butter_size=(0.123, 0.026, 0.039),
        butter_center=(0, 0, 0.025),
        blade_size=(0.022, 0.035, 0.0008),
        cut_offset=0.123 / 2 - 0.010 - 0.0008 / 2,
        cut_end_time=2.9,
    )
    blade = trimesh.creation.box(extents=config.blade_size)
    orientation = Rotation.from_euler("x", -70, degrees=True) * Rotation.from_euler("y", -90, degrees=True)
    assets = {
        "cutting_roll_degrees": -70,
        "cutting_direction": direction,
        "lift_outward_clearance_m": 0.003,
        "blade_min_z_when_cutting_m": orientation.apply(blade.vertices)[:, 2].min(),
    }
    times = np.linspace(0, 4, 4001)
    centers, _, _ = asset_motion(times, config, assets, blade)
    lifted = centers[(times >= 2.9) & (times <= 3.7)]
    np.testing.assert_allclose(lifted[:, 1], lifted[0, 1])
    assert np.all(np.diff(lifted[:, 0]) >= 0)
    np.testing.assert_allclose(lifted[-1, 0] - lifted[0, 0], 0.003, atol=1e-12)
    assert np.all(np.diff(lifted[:, 2]) > 0)
    np.testing.assert_allclose(centers[times <= 2.9, 0], centers[0, 0])
    assert direction * (centers[np.searchsorted(times, 2.9), 1] - centers[np.searchsorted(times, 2.05), 1]) > 0
    np.testing.assert_allclose(config.butter_size[0] / 2 - centers[0, 0] - config.blade_size[2] / 2, 0.010)


@pytest.mark.parametrize("backend", [None])
def test_asset_rigid_fit_preserves_scale_and_rejects_changed_geometry(backend):
    source = np.array(((0, 0, 0), (0.02, 0, 0), (0, 0.1, 0), (0, 0, 0.003)))
    rotation = Rotation.from_euler("xyz", (0.2, 0.4, -0.3)).as_matrix()
    target = source @ rotation.T + (0.1, -0.2, 0.3)
    transform, error = rigid_fit(source, target)
    np.testing.assert_allclose(source @ transform[:3, :3].T + transform[:3, 3], target, atol=1e-14)
    assert error < 1e-14
    with pytest.raises(ValueError, match="identity"):
        rigid_fit(source, target * 1.02)


@pytest.mark.parametrize("backend", [None])
@pytest.mark.parametrize("pitch,yaw", [(0, 0), (10, 180)])
@pytest.mark.parametrize("cut_roll", [-15, -70])
def test_transfer_keeps_loaded_face_and_spreads_left_to_right(backend, pitch, yaw, cut_roll):
    config = replace(
        Config(),
        bread_center=(0.205, 0, 0.027),
        spread_start_offset=-0.031,
        spread_end_offset=0.023,
        spread_pitch_degrees=pitch,
        spread_yaw_degrees=yaw,
    )
    times = np.linspace(0, 8.4, 3361)
    positions = np.tile((0.058, 0, 0.080), (len(times), 1))
    angles = np.tile((-np.pi / 2, 0), (len(times), 1))
    orientation = Rotation.from_euler("x", cut_roll, degrees=True) * Rotation.from_euler("y", -90, degrees=True)
    quaternions = np.tile(orientation.as_quat()[[3, 0, 1, 2]], (len(times), 1))
    assets = {"cutting_roll_degrees": cut_roll, "bread_top_m": 0.032}
    centers, _, quats, _ = continue_motion(times, config, assets, positions, angles, quaternions)
    start, end = np.searchsorted(times, (6.0, 7.5))
    loaded_normal = Rotation.from_quat(quats[start, [1, 2, 3, 0]]).apply((0, 0, -1))
    np.testing.assert_allclose(loaded_normal[2], -np.cos(np.deg2rad(pitch)), atol=1e-12)
    corners = np.array(
        [
            (x, y, -config.blade_size[2] / 2)
            for x in (-config.blade_size[0] / 2, config.blade_size[0] / 2)
            for y in (-config.blade_size[1] / 2, config.blade_size[1] / 2)
        ]
    )
    world = Rotation.from_quat(quats[start, [1, 2, 3, 0]]).apply(corners) + centers[start]
    np.testing.assert_allclose(world[:, 2].min(), assets["bread_top_m"] + config.spread_start_gap, atol=1e-12)
    if yaw == 180:
        overhang = Rotation.from_quat(quats[start, [1, 2, 3, 0]]).apply((-0.02, 0, -0.003))
        assert overhang[0] > 0
    assert np.all(np.diff(centers[start : end + 1, 0]) > 0)
    assert np.linalg.norm(np.diff(centers, axis=0), axis=1).max() < 0.001
    rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]])
    angular_speed = np.linalg.norm((rotations[:-1].inv() * rotations[1:]).as_rotvec(), axis=1) / np.diff(times)
    assert angular_speed.max() < 5.0
    for boundary in (3.7, 5.2, 6.0, 7.5, 8.1):
        index = np.searchsorted(times, boundary)
        assert angular_speed[index - 1 : index + 1].max() < 0.001
