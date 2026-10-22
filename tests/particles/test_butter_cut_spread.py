import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import genesis as gs
from examples.butter_cut_spread.compare import trajectory_difference
from examples.butter_cut_spread.config import PHASES, Config, angular_velocities, motion, rotation
from examples.butter_cut_spread.contact import ToolContact
from examples.butter_cut_spread.geometry import convex_planes, tool_mesh
from examples.butter_spreading.config import ButterParameters, ContactParameters
from genesis.utils.misc import tensor_to_array
from tests.utils import assert_allclose, assert_equal


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
def test_two_sided_wet_contact_releases_anchor_memory_and_has_finite_range(backend):
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
