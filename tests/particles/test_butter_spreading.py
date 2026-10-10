import numpy as np
import pytest
import torch
from pydantic import ValidationError

import genesis as gs
from examples.butter_spreading.config import load_config
from examples.butter_spreading.contact import ButterContact
from examples.butter_spreading.demo import DT, prepare_knife_motion
from examples.butter_spreading.motion import knife_pose
from tests.utils import assert_allclose, assert_equal


def test_butter_configuration_rejects_invalid_samples(tmp_path):
    path = tmp_path / "parameters.json"
    path.write_text('{"seed": 18, "butter": {"yield_stress": 100.0}, "dt": 0.0000175}', encoding="utf-8")
    config = load_config(path, seed=19, butter_particle_size=0.0006)
    assert config.seed == 19
    assert config.butter.yield_stress == 100.0
    assert config.butter.consistency == 25.0
    assert config.butter_particle_size == 0.0006
    assert config.dt == 0.0000175
    for text in (
        '{"butter": {"yeild_stress": 100.0}}',
        '{"dt": 0}',
        '{"dt": NaN}',
        '{"seed": 1.5}',
        '{"contact": {"blade_slip_time": 0}}',
        '{"butter": {"flow_exponent": 0.1}}',
    ):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError):
            load_config(path)


def test_butter_knife_trajectory_is_continuous():
    for transition in (0.90, 1.65, 3.15, 3.50):
        assert knife_pose(transition - 3.5e-5) == pytest.approx(knife_pose(transition + 3.5e-5), abs=1e-4)
    assert knife_pose(0.0) == (-0.055, 0.0, 0.0565)
    assert knife_pose(4.0) == (0.049, 0.0, 0.0575)
    origin = (0.0, 0.0, 0.026)
    offset = (0.0021, 0.0035)
    motion = prepare_knife_motion(round(4.0 / DT), origin, offset)
    offset = np.array(offset, dtype=np.float32).tolist()
    for time_s in (0.0, 0.45, 0.9, 1.65, 2.0, 3.15, 3.5, 3.9):
        step = round(time_s / DT)
        current = knife_pose(step * DT)
        following = knife_pose(step * DT + DT)
        expected_position = gs.tensor((current[0] + offset[0], current[1] + offset[1], current[2]))
        expected_velocity = gs.tensor(
            (
                (following[0] - current[0]) / DT,
                (following[1] - current[1]) / DT,
                (following[2] - current[2]) / DT,
                0.0,
                0.0,
                0.0,
            )
        )
        assert_equal(motion.positions[step], expected_position)
        assert_equal(motion.velocities[step], expected_velocity)


@pytest.mark.parametrize("n_envs", [0, 2])
def test_butter_contact_conserves_batched_momentum(n_envs, tol):
    bread_positions = torch.tensor(
        [
            [[-0.001, -0.001, 0.0], [-0.001, 0.001, 0.0], [0.001, -0.001, 0.0], [0.001, 0.001, 0.0]],
            [[-0.001, -0.001, 0.0], [-0.001, 0.001, 0.0], [0.001, -0.001, 0.0], [0.001, 0.001, 0.0]],
        ]
    )
    butter_positions = torch.tensor([[[0.0, 0.0, 0.002]], [[0.0, 0.0, 0.002]]])
    butter_velocities = torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    if n_envs == 0:
        bread_positions = bread_positions[0]
        butter_positions = butter_positions[0]
        butter_velocities = butter_velocities[0]
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.0001, gravity=(0.0, 0.0, 0.0)),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.1, -0.1, -0.1),
            upper_bound=(0.1, 0.1, 0.1),
            particle_size=0.002,
            grid_density=64,
        ),
        show_viewer=False,
    )
    bread = scene.add_entity(
        morph=gs.morphs.Box(size=(0.004, 0.004, 0.002)),
        material=gs.materials.MPM.PorousBread(rho=1.0, sampler="regular"),
    )
    butter = scene.add_entity(
        morph=gs.morphs.Box(size=(0.001, 0.001, 0.001)),
        material=gs.materials.MPM.HerschelBulkleyButter(rho=1.0, sampler="regular", particle_size=0.001, dt=0.0001),
    )
    blade = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 1.0),
            size=(0.02, 0.02, 0.002),
        ),
        material=gs.materials.Rigid(
            needs_coup=False,
        ),
    )
    scene.build(n_envs=n_envs)
    bread.set_particles_pos(bread_positions)
    bread.set_particles_vel(torch.zeros_like(bread_positions))
    butter.set_particles_pos(butter_positions)
    butter.set_particles_vel(butter_velocities)
    bread_mass = bread.get_mass()[0] / bread.n_particles
    butter_mass = butter.get_mass()[0] / butter.n_particles
    initial_momentum = (butter.get_particles_vel() * butter_mass).sum(dim=-2)
    contact = ButterContact(
        bread,
        butter,
        blade,
        spacing=0.001,
        dt=0.0001,
        blade_size=(0.02, 0.02, 0.002),
        lower_bound=(-0.01, -0.01, -0.01),
        upper_bound=(0.01, 0.01, 0.02),
        bread_stress=1.0,
        blade_stress=1.0,
        bread_contact_range=0.003,
        bread_slip_time=0.001,
        bread_shear_stress=1.0,
        blade_contact_range=0.003,
        blade_slip_time=0.001,
        blade_shear_stress=1.0,
        blade_normal_relaxation=0.0,
        blade_max_separation_speed=0.35,
        blade_contact_margin=0.001,
        is_equilibrium_adhesion=True,
    )

    contact.apply()

    assert (butter.get_particles_vel()[..., 0] < 1.0).all()
    assert (butter.get_particles_vel()[..., 2] < 0.0).all()
    assert (torch.linalg.vector_norm(bread.get_particles_vel(), dim=-1).sum(dim=-1) > 0.0).all()

    momentum = (bread.get_particles_vel() * bread_mass).sum(dim=-2)
    momentum += (butter.get_particles_vel() * butter_mass).sum(dim=-2)
    torch.testing.assert_close(momentum, initial_momentum, atol=1e-15, rtol=tol)

    # Rotate the blade's thin face toward the particle, away from the bread.
    # Check the fused quaternion transform against an analytical cohesive impulse.
    butter_positions[..., 2] = 0.052
    butter.set_particles_pos(butter_positions)
    butter.set_particles_vel(butter_velocities)
    blade.set_quat((2**-0.5, 0.0, 2**-0.5, 0.0), relative=False)
    blade.set_pos((0.0035, 0.0, 0.052), relative=False)
    blade.set_dofs_velocity((0.1, 0.0, 0.0, 0.0, 0.0, 2.0))
    contact.apply()
    # gap=2 mm, range=3 mm: normal cohesive weight is 4*(2/3)*(1/3).
    assert_allclose(butter.get_particles_vel()[..., 0], 1.0 + 0.0001 / 0.002 * 8.0 / 9.0, tol=tol)

    butter.set_particles_vel(butter_velocities)
    contact.apply(
        center=(0.0, 0.0, 0.0555),
        velocity=(0.0, 0.0, 0.0),
    )
    assert_allclose(butter.get_particles_vel()[..., 2], 0.0001 / 0.002 * 8.0 / 9.0, tol=tol)


@pytest.mark.parametrize("n_envs", [0, 2])
def test_butter_materials_and_particle_spacing(n_envs, show_viewer, tol):
    dt = 3.5e-5
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.1, -0.1, -0.1),
            upper_bound=(0.1, 0.1, 0.1),
            particle_size=0.005,
            grid_density=64,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.12, -0.15, 0.1),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    bread = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.03, 0.0, 0.0),
            size=(0.02, 0.02, 0.02),
        ),
        material=gs.materials.MPM.PorousBread(
            rho=280.0,
            sampler="regular",
        ),
    )
    butter = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.03, 0.0, 0.0),
            size=(0.01, 0.01, 0.01),
        ),
        material=gs.materials.MPM.HerschelBulkleyButter(
            rho=900.0,
            dt=dt,
            sampler="regular",
            particle_size=0.0025,
        ),
    )
    scene.build(n_envs=n_envs)
    for entity, spacing in ((bread, 0.005), (butter, 0.0025)):
        assert entity.particle_size == spacing
        assert_allclose(entity.get_mass(), entity.n_particles * spacing**3 * entity.material.rho, rtol=tol, atol=0)
    initial_bread = bread.get_particles_pos().clone()
    initial_butter = butter.get_particles_pos().clone()
    checkpoint = scene.get_state()
    butter.set_particles_vel((0.01, 0.0, 0.0))
    scene.step()
    for entity in (bread, butter):
        assert torch.isfinite(entity.get_particles_pos()).all()
        assert torch.isfinite(entity.get_particles_vel()).all()
    scene.reset(checkpoint)
    torch.testing.assert_close(bread.get_particles_pos(), initial_bread)
    torch.testing.assert_close(butter.get_particles_pos(), initial_butter)
    scene.step()
    torch.testing.assert_close(bread.get_particles_pos(), initial_bread)
    torch.testing.assert_close(butter.get_particles_pos(), initial_butter)
