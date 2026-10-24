import math
from dataclasses import replace

import numpy as np
import pytest
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

import genesis as gs
from examples.butter_cut_spread.asset_contact import MeshProjection
from examples.butter_cut_spread.asset_demo import asset_motion
from examples.butter_cut_spread.compare import trajectory_difference
from examples.butter_cut_spread.config import PHASES, Config, angular_velocities, motion, rotation
from examples.butter_cut_spread.contact import ToolContact
from examples.butter_cut_spread.cutting_scene import CuttingScene
from examples.butter_cut_spread.geometry import convex_planes, tool_mesh
from examples.butter_spreading.config import ButterParameters, ContactParameters
from genesis.utils.misc import tensor_to_array
from tests.utils import assert_allclose, assert_equal


def test_asset_cut_sweeps_short_side_before_lifting():
    profile = np.array([[-0.030, 0.010], [0.025, 0.0], [0.020, -0.014], [-0.025, -0.004]])
    blade = trimesh.creation.extrude_triangulation(profile, np.array([[0, 1, 2], [0, 2, 3]]), 0.0008)
    blade.vertices = blade.vertices[:, [2, 0, 1]] - (0.0004, 0, 0)
    orientation = Rotation.from_euler("x", -15, degrees=True) * Rotation.from_euler("y", -90, degrees=True)
    blade.vertices = orientation.inv().apply(blade.vertices)
    config = replace(Config(), butter_size=(0.12, 0.026, 0.039), butter_center=(0, 0.008, 0.025))
    assets = {"cutting_roll_degrees": -15.0, "blade_min_z_when_cutting_m": -0.014}
    times = np.array([0.0, 0.6, 1.5, 2.6, 2.9, 3.7, 4.0])
    positions, _, quaternions = asset_motion(times, config, assets, blade)
    low = np.array(config.butter_center)[1:] - np.array(config.butter_size)[1:] / 2
    high = np.array(config.butter_center)[1:] + np.array(config.butter_size)[1:] / 2
    y, z = np.meshgrid(
        np.linspace(low[0] + 0.0002, high[0] - 0.0002, 91),
        np.linspace(low[1] + 0.0002, high[1] - 0.0002, 131),
    )
    points = np.column_stack((y.reshape(-1), z.reshape(-1)))
    covered = np.zeros(len(points), dtype=bool)
    # A convex section translated along a segment sweeps its endpoint hull.
    for start, end in zip(positions[:3, 1:], positions[1:4, 1:]):
        hull = ConvexHull(np.concatenate((profile + start, profile + end)))
        covered |= np.all(points @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-12, axis=1)
    assert covered.all(), "Entry and draw strokes must sweep every sampled location in the initial cutting section."
    assert positions[2, 1] + profile[2, 0] > high[0]
    assert positions[3, 1] + profile[2, 0] < low[0]
    assert_allclose(quaternions, np.tile(orientation.as_quat()[[3, 0, 1, 2]], (len(times), 1)), atol=1e-12)


@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_tool_selection_preserves_support_contact_and_separates_blade_sides(
    backend, n_envs, asset_tmp_path, show_viewer
):
    profile = np.array(
        [
            [-0.03, -0.005],
            [0.03, -0.005],
            [0.03, 0.03],
            [0.025, 0.03],
            [0.005, 0],
            [-0.005, 0],
            [-0.025, 0.03],
            [-0.03, 0.03],
        ]
    )
    triangles = np.array([[0, 1, 4], [0, 4, 5], [1, 2, 4], [2, 3, 4], [0, 5, 7], [5, 6, 7]])
    mesh = trimesh.creation.extrude_triangulation(profile, triangles, 0.04)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
    mesh.apply_translation((0, 0.02, 0))
    path = asset_tmp_path / "concave-support.obj"
    mesh.export(path)
    scene = CuttingScene(
        sim_options=gs.options.SimOptions(
            dt=0.00002,
            gravity=(0, 0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.09, -0.05, -0.02),
            upper_bound=(0.09, 0.05, 0.07),
            grid_density=1000,
            particle_size=0.0004,
            enable_CPIC=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.12, -0.18, 0.14),
            camera_lookat=(0, 0, 0.02),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(path),
            fixed=True,
            convexify=False,
            decimate=False,
            watertighten=None,
        ),
        material=gs.materials.Rigid(
            coup_friction=0.5,
            coup_softness=0,
            sdf_cell_size=0.0002,
            sdf_max_res=400,
        ),
    )
    blade = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.0002, 0.02, 0.02),
            pos=(0.06, 0, 0.03),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            coup_friction=0,
            coup_softness=0,
        ),
    )
    positions = np.array(
        [
            [-0.0002, 0, 0.02],
            [0.0002, 0, 0.02],
            [0, 0, 0.0003],
            [0.0596, 0, 0.03],
            [0.0604, 0, 0.03],
            [-0.0604, 0, 0.03],
            [-0.0596, 0, 0.03],
        ]
    )
    butter = scene.add_entity(
        morph=gs.morphs.Particles(
            positions=positions,
        ),
        material=gs.materials.MPM.HerschelBulkleyButter(
            **ButterParameters().model_dump(),
            dt=0.00002,
            particle_size=0.0004,
        ),
    )
    scene.build(cutting_tools=(blade,), n_envs=n_envs)
    velocities = np.zeros_like(positions)
    velocities[2, 2] = -0.5
    velocities[2, 0] = 0.1
    velocities[[3, 5], 1] = 0.1
    velocities[[4, 6], 1] = -0.1
    butter.set_particles_vel(velocities)
    scene.step(update_visualizer=False)
    measured = tensor_to_array(butter.get_particles_vel())
    separated = measured[..., 3, 1] - measured[..., 4, 1]
    shared = measured[..., 5, 1] - measured[..., 6, 1]
    assert np.all(separated > 1.25 * shared)
    assert_allclose(measured[..., :2, 2], -9.81 * 0.00002, atol=1e-7)
    # Contact needs about 4 ms to reduce tangential speed below 1 cm/s; allow 5 ms.
    for _ in range(249):
        scene.step(update_visualizer=False)
    points = tensor_to_array(butter.get_particles_pos())
    expected = 0.02 - 9.81 * 0.00002**2 * 250 * 251 / 2
    assert_allclose(points[..., :2, 2], expected, atol=2e-6)
    assert np.all(points[..., 2, 2] >= 0)
    assert np.all(points[..., 2, 2] <= 0.0003)
    assert np.all(abs(tensor_to_array(butter.get_particles_vel())[..., 2, 0]) < 0.01)
    scene.reset()
    butter.set_particles_vel(velocities)
    scene.step(update_visualizer=False)
    assert_allclose(tensor_to_array(butter.get_particles_vel())[..., :2, 2], -9.81 * 0.00002, atol=1e-7)
    counts = scene.get_cpic_transfer_counts()
    assert_equal(counts[0], 0)
    assert counts[1] > 0


@pytest.mark.parametrize("backend", [None])
def test_end_slice_trajectory_preserves_loaded_face_and_collision_clearance(backend):
    config = Config()
    times = np.linspace(0, config.duration, 7001)
    positions, angles, quaternions = motion(times, config)
    assert np.all(np.diff(positions[times >= 4, 0]) >= -1e-12)
    assert_allclose(rotation(angles[0]) @ np.array([0.0, 0.0, -1.0]), [1, 0, 0], atol=1e-12)
    normal = rotation(angles[np.argmin(abs(times - 4))]) @ np.array([0.0, 0.0, -1.0])
    assert normal[2] < -0.99
    assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-12)
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
    angular_speed = np.linalg.norm(np.diff(angles, axis=0), axis=1) / np.diff(times)
    radius = np.linalg.norm(config.blade_size) / 2
    assert np.max((speed + angular_speed * radius) * config.dt) < config.particle_size / 10
    for _, end in PHASES:
        near, theta, _ = motion(np.array([end - 1e-7, end, end + 1e-7]), config)
        assert np.max(abs(np.diff(near, axis=0))) < 1e-8
        assert np.max(abs(np.diff(theta, axis=0))) < 1e-10
        near, _, _ = motion(np.array([end - 1e-5, end, end + 1e-5]), config)
        velocity = np.diff(near, axis=0) / 1e-5
        assert_allclose(velocity[0], velocity[1], atol=1e-5)
    extent = (abs(np.sin(angles[:, 0])) * config.blade_size[0] + abs(np.cos(angles[:, 0])) * config.blade_size[2]) / 2
    assert np.min(positions[:, 2] - extent) > -1e-12
    is_carrying = (times >= 1.95) & (times <= 3.25)
    assert np.min(positions[is_carrying, 2] - extent[is_carrying]) > 0.045
    assert_allclose(angles[is_carrying, 0], -np.pi / 2, atol=1e-12)
    assert_equal(angles[:, 1], np.zeros(len(times)))
    slice_thickness = config.butter_size[0] / 2 - config.cut_offset - config.blade_size[2] / 2
    assert_allclose(slice_thickness, 0.0042, atol=1e-12)
    mesh = tool_mesh(config)
    assert mesh.is_volume
    assert_allclose(mesh.extents, config.blade_size, atol=1e-12)
    planes = convex_planes(mesh.vertices, mesh.faces)
    assert np.max(mesh.vertices @ planes[:, :3].T + planes[:, 3]) < 1e-12
    dt = 1e-4
    _, _, quaternions = motion(np.arange(3.25, 4.0, dt), config)
    body_rate, world_rate = angular_velocities(quaternions, dt)
    orientations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
    integrated = orientations[:-1] * Rotation.from_rotvec(body_rate * dt)
    assert_allclose(integrated.as_matrix(), orientations[1:].as_matrix(), atol=1e-12)
    integrated = Rotation.from_rotvec(world_rate * dt) * orientations[:-1]
    assert_allclose(integrated.as_matrix(), orientations[1:].as_matrix(), atol=1e-12)


@pytest.mark.parametrize("backend", [gs.gpu])
def test_two_sided_wet_contact_releases_anchor_memory_and_has_finite_range(backend, asset_tmp_path):
    config = Config()
    wet = ContactParameters()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=config.dt, gravity=(0, 0, 0)),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.08, -0.08, -0.08),
            upper_bound=(0.08, 0.08, 0.08),
            grid_density=64,
            particle_size=config.particle_size,
        ),
        show_viewer=False,
    )
    butter = scene.add_entity(
        morph=gs.morphs.Box(size=(2 * config.particle_size, config.particle_size, config.particle_size)),
        material=gs.materials.MPM.HerschelBulkleyButter(
            **ButterParameters().model_dump(), dt=config.dt, sampler="regular", particle_size=config.particle_size
        ),
    )
    blade_path = asset_tmp_path / "butter-projection-blade.obj"
    tool_mesh(config).export(blade_path)
    blade = scene.add_entity(
        morph=gs.morphs.Mesh(file=str(blade_path), fixed=True, convexify=False, decimate=False, watertighten=None),
        material=gs.materials.Rigid(sdf_cell_size=0.00015, sdf_max_res=512),
    )
    scene.build()
    assert butter.n_particles == 2
    contact = ToolContact(butter, config)
    signs = np.array([-1, 1])
    gap = 0.0004
    positions = np.zeros((2, 3))
    positions[:, 2] = signs * (config.blade_size[2] / 2 + config.particle_size / 2 + gap)
    velocity = np.tile([0.05, 0.025, 0.0], (2, 1))
    separation = gap / config.spread_adhesion_range
    expected = velocity.copy()
    expected[:, :2] *= 1 - (1 - math.exp(-config.dt / wet.blade_slip_time)) * (1 - separation) ** 2
    expected[:, 2] = (
        -signs
        * 4
        * separation
        * (1 - separation)
        * config.spread_normal_stress
        * config.dt
        / (butter.material.rho * 2 * config.spread_adhesion_range / 3)
    )
    for anchor in ((0.0, 0.0), (0.005, -0.005)):
        butter.set_particles_pos(gs.tensor(positions))
        butter.set_particles_vel(gs.tensor(velocity))
        contact.anchors.from_numpy(np.array([anchor] * 2, dtype=gs.np_float))
        contact.side.from_numpy(signs.astype(gs.np_int))
        contact.age.fill(1)
        contact.apply((0, 0, 0), (0, 0, 0), (0, 0), (0, 0, 0), 4.0)
        assert_allclose(tensor_to_array(butter.get_particles_vel()), expected, atol=2e-7, rtol=1e-5)
        assert_allclose(tensor_to_array(butter.get_particles_pos()), positions, atol=1e-10)
    tilted = Rotation.from_euler("x", 0.25).as_matrix()
    butter.set_particles_pos(gs.tensor(positions @ tilted.T))
    butter.set_particles_vel(gs.tensor(velocity @ tilted.T))
    contact.apply((0, 0, 0), (0, 0, 0), (0, 0), (0, 0, 0), 4.0, roll=0.25)
    assert_allclose(tensor_to_array(butter.get_particles_vel()), expected @ tilted.T, atol=2e-7, rtol=1e-5)
    for gap in (0.0, 0.0006, 0.0015, 1.1 * config.spread_adhesion_range):
        positions[:, 2] = signs * (config.blade_size[2] / 2 + config.particle_size / 2 + gap)
        butter.set_particles_pos(gs.tensor(positions))
        butter.set_particles_vel(gs.tensor(np.zeros((2, 3))))
        contact.side.fill(0)
        contact.apply((0, 0, 0), (0, 0, 0), (0, 0), (0, 0, 0), 4.0)
        measured = tensor_to_array(butter.get_particles_vel())
        if 0 < gap < config.spread_adhesion_range:
            assert np.all(signs * measured[:, 2] < 0)
        else:
            assert_allclose(measured, 0, atol=1e-7)
    positions[:, 2] = signs * config.blade_size[2] / 4
    butter.set_particles_pos(gs.tensor(positions))
    butter.set_particles_vel(gs.tensor(np.column_stack((np.zeros((2, 2)), -signs * 0.01))))
    contact.apply((0, 0, 0), (0, 0, 0), (0, 0), (0, 0, 0), 4.0, project=True)
    corrected = tensor_to_array(butter.get_particles_pos())
    assert_allclose(corrected[:, 2], signs * config.blade_size[2] / 2, atol=1e-9)
    assert_allclose(tensor_to_array(butter.get_particles_vel()), 0, atol=1e-9)

    # A translated support must apply the same finite cohesive law at its actual surface height.
    for height in (0.0, 0.006):
        raised = replace(config, board_centers=((0, 0, height - 0.005),), board_sizes=((0.02, 0.02, 0.01),))
        support_contact = ToolContact(butter, raised)
        positions[:] = (
            (0, 0, height + config.particle_size / 2 + 0.00045),
            (0, 0, height + config.particle_size / 2 + 0.0018),
        )
        butter.set_particles_pos(gs.tensor(positions))
        butter.set_particles_vel(gs.tensor(np.zeros((2, 3))))
        support_contact.apply((0, 0, 0.05), (0, 0, 0), (0, 0), (0, 0, 0), 0.0)
        expected = np.zeros((2, 3))
        expected[0, 2] = -config.board_normal_stress * config.dt / (butter.material.rho * 0.0006)
        assert_allclose(tensor_to_array(butter.get_particles_vel()), expected, atol=1e-7)
    projection = MeshProjection(butter, blade, np.eye(3), config.blade_size)
    outside = np.zeros((2, 3))
    outside[:, 2] = signs * (config.blade_size[2] / 2 + 0.0001)
    for target in (outside * 0.2, -outside):
        butter.set_particles_pos(gs.tensor(outside))
        projection.record((0, 0, 0))
        butter.set_particles_pos(gs.tensor(target))
        butter.set_particles_vel(gs.tensor(np.column_stack((np.zeros((2, 2)), -signs * 0.01))))
        projection.apply((0, 0, 0), (0, 0, 0))
        corrected = tensor_to_array(butter.get_particles_pos())
        assert np.all(corrected[:, 2] * signs >= config.blade_size[2] / 2)
        assert_allclose(tensor_to_array(butter.get_particles_vel()), 0, atol=1e-7)


@pytest.mark.parametrize("backend", [None])
def test_trajectory_comparison_detects_intermediate_particle_identity_changes(backend):
    original = np.zeros((3, 2, 3))
    original[:, 1, 0] = 0.01
    current = original.copy()
    assert trajectory_difference(current, original).is_identical
    current[1] = current[1, ::-1]
    result = trajectory_difference(current, original)
    assert not result.is_identical
    assert_allclose(result.max_frame_rms_m, 0.01, atol=1e-12)
    assert_allclose(result.max_m, 0.01, atol=1e-12)
    with pytest.raises(ValueError, match="identities"):
        trajectory_difference(current[:, :1], original)


@pytest.mark.parametrize("backend", [gs.gpu])
def test_authored_asymmetric_tool_pose_matches_native_collision_frame(backend, asset_tmp_path):
    mesh = tool_mesh(Config())
    mesh.vertices = Rotation.from_euler("z", 17, degrees=True).apply(mesh.vertices) + [0.003, 0.014, 0.002]
    path = asset_tmp_path / "asymmetric-authored-tool.obj"
    mesh.export(path)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2.5e-5, gravity=(0, 0, 0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, disable_constraint=True),
        show_viewer=False,
    )
    blade = scene.add_entity(
        morph=gs.morphs.Mesh(file=str(path), align=False, convexify=False, decimate=False, watertighten=None),
        material=gs.materials.Rigid(),
    )
    scene.build()
    position = np.array((0.04, 0.03, 0.08))
    quaternion = Rotation.from_euler("xyz", (-15, -90, 0), degrees=True).as_quat()[[3, 0, 1, 2]]
    blade.set_pos(gs.tensor(position), relative=False)
    blade.set_quat(gs.tensor(quaternion), relative=False)
    blade.set_dofs_velocity(gs.tensor(np.zeros(6)))
    scene.step(update_visualizer=False)
    geom_index = blade.geoms[0].idx
    assert_allclose(tensor_to_array(blade.solver.get_geoms_pos([geom_index]))[0], position, atol=1e-7)
    assert_allclose(tensor_to_array(blade.solver.get_geoms_quat([geom_index]))[0], quaternion, atol=1e-7)
