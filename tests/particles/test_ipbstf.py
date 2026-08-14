import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array
from tests.utils import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_unilateral_density_energy_and_kinetic_smoothing(n_envs, show_viewer):
    dt = 0.01
    translation_steps = 40
    translation_velocity = (0.0, -9.81 * dt, 0.0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, 0.0),
        ),
        ipbstf_options=gs.options.IPBSTFOptions(
            particle_size=0.1,
            lower_bound=(-2.0, -2.0, -2.0),
            upper_bound=(2.0, 2.0, 2.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.2, 2.2, 1.8),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Box(
            lower=(-0.25, 0.75, -0.25),
            upper=(0.25, 1.25, 0.25),
        ),
        material=gs.materials.IPBSTF.Liquid(
            sampler="staggered",
        ),
    )
    scene.build(n_envs=n_envs)

    pos_initial = tensor_to_array(liquid.get_state().pos)
    liquid.set_particles_vel(translation_velocity)
    kinetic_energy_initial = tensor_to_array(scene.ipbstf_solver.get_kinetic_energy())
    solver_iteration_energy = []
    for _ in range(translation_steps):
        scene.step()
        solver_iteration_energy.append(
            np.atleast_2d(tensor_to_array(scene.ipbstf_solver.get_last_step_variational_energy()))
        )

    pos_translated = pos_initial.copy()
    pos_translated[..., 1] += translation_steps * dt * translation_velocity[1]
    assert_allclose(tensor_to_array(liquid.get_state().pos), pos_translated, atol=1e-5)
    assert_allclose(
        tensor_to_array(scene.ipbstf_solver.get_kinetic_energy()), kinetic_energy_initial, tol=5e-4
    )
    solver_iteration_energy = np.stack(solver_iteration_energy)
    assert_equal(solver_iteration_energy, np.zeros_like(solver_iteration_energy))

    center = pos_translated.mean(axis=1, keepdims=True)
    pos_compressed = center + 0.65 * (pos_translated - center)
    liquid.set_particles_pos(pos_compressed)
    scene.step()
    pos_expanded = tensor_to_array(liquid.get_state().pos)
    solver_iteration_energy = np.atleast_2d(
        tensor_to_array(scene.ipbstf_solver.get_last_step_variational_energy())
    )

    assert_allclose(pos_expanded.mean(axis=1), center[..., 0, :], atol=1e-5)
    assert (np.ptp(pos_expanded, axis=1) > 1.1 * np.ptp(pos_compressed, axis=1)).all()
    assert (np.diff(solver_iteration_energy, axis=-1) <= 0.0).all()
    assert (solver_iteration_energy[..., -1] < solver_iteration_energy[..., 0]).all()

    pos_smoothing = center + 1.1 * (pos_initial - pos_initial.mean(axis=1, keepdims=True))
    pos_relative = pos_smoothing - pos_smoothing.mean(axis=1, keepdims=True)
    vel_smoothing = np.empty_like(pos_smoothing)
    vel_smoothing[..., 0] = 0.04 * pos_relative[..., 0] + 0.02 * pos_relative[..., 1]
    vel_smoothing[..., 1] = -0.01 * pos_relative[..., 0] + 0.03 * pos_relative[..., 1]
    vel_smoothing[..., 2] = 0.02 * pos_relative[..., 1] + 0.04 * pos_relative[..., 2]
    vel_smoothing += (0.03, -0.02, 0.01)
    liquid.set_particles_pos(pos_smoothing)
    liquid.set_particles_vel(vel_smoothing)
    kinetic_energy_initial = tensor_to_array(scene.ipbstf_solver.get_kinetic_energy())
    velocity_center_initial = vel_smoothing.mean(axis=1)
    velocity_relative_initial = vel_smoothing - velocity_center_initial[:, None, :]
    velocity_covariance_initial = np.einsum(
        "bni,bnj->bij", velocity_relative_initial, velocity_relative_initial
    ) / liquid.n_particles
    scene.step()
    vel_smoothed = tensor_to_array(liquid.get_state().vel)
    velocity_center_smoothed = vel_smoothed.mean(axis=1)
    velocity_relative_smoothed = vel_smoothed - velocity_center_smoothed[:, None, :]
    velocity_covariance_smoothed = np.einsum(
        "bni,bnj->bij", velocity_relative_smoothed, velocity_relative_smoothed
    ) / liquid.n_particles

    assert_allclose(velocity_center_smoothed, velocity_center_initial, atol=1e-5)
    assert_allclose(velocity_covariance_smoothed, velocity_covariance_initial, tol=5e-3)
    assert_allclose(scene.ipbstf_solver.get_kinetic_energy(), kinetic_energy_initial, tol=5e-3)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_dam_break_spreads_under_gravity(n_envs, show_viewer):
    dt = 0.005
    gravity = 9.81
    particle_size = 1.0 / 6.0
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, -gravity, 0.0),
        ),
        ipbstf_options=gs.options.IPBSTFOptions(
            alpha=0.0,
            is_damping_enabled=True,
            particle_size=particle_size,
            lower_bound=(-2.5, -0.5, -2.5),
            upper_bound=(2.5, 4.5, 2.5),
            collision_lower_bound=(-2.0, 0.0, -2.0),
            collision_upper_bound=(2.0, 4.0, 2.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.0, 3.0, 6.0),
            camera_lookat=(0.0, 1.0, 0.0),
            camera_up=(0.0, 1.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Box(
            lower=(-1.5, 0.1, -1.5),
            upper=(0.0, 2.0, 1.5),
        ),
        material=gs.materials.IPBSTF.Liquid(
            sampler="staggered",
        ),
    )
    floor_entity = scene.add_entity(
        morph=gs.morphs.Box(
            lower=(-2.0, -2.0 * particle_size, -2.0),
            upper=(2.0, 0.0, 2.0),
        ),
        material=gs.materials.IPBSTF.Solid(
            sampler="staggered",
        ),
    )
    scene.build(n_envs=n_envs)

    pos_initial = tensor_to_array(liquid.get_state().pos)
    floor_pos_initial = tensor_to_array(floor_entity.get_state().pos)
    particle_mass = tensor_to_array(liquid.get_mass()) / liquid.n_particles
    mechanical_energy_initial = particle_mass * gravity * pos_initial[..., 1].sum(axis=1)
    scene.step()
    solver_iteration_energy = [
        np.atleast_2d(tensor_to_array(scene.ipbstf_solver.get_last_step_variational_energy()))
    ]
    pos = tensor_to_array(liquid.get_state().pos)
    vel = tensor_to_array(liquid.get_state().vel)
    speed = np.linalg.norm(vel, axis=-1)
    assert (speed < 2.0 * gravity * dt).all()
    kinetic_energy = 0.5 * particle_mass * np.square(vel).sum(axis=(-2, -1))
    kinetic_energy_peak = kinetic_energy.copy()
    mechanical_energy = kinetic_energy + particle_mass * gravity * pos[..., 1].sum(axis=1)
    mechanical_energy_peak = mechanical_energy.copy()
    for _ in range(79):
        scene.step()
        solver_iteration_energy.append(
            np.atleast_2d(tensor_to_array(scene.ipbstf_solver.get_last_step_variational_energy()))
        )
        pos = tensor_to_array(liquid.get_state().pos)
        vel = tensor_to_array(liquid.get_state().vel)
        kinetic_energy = 0.5 * particle_mass * np.square(vel).sum(axis=(-2, -1))
        kinetic_energy_peak = np.maximum(kinetic_energy_peak, kinetic_energy)
        mechanical_energy = kinetic_energy + particle_mass * gravity * pos[..., 1].sum(axis=1)
        mechanical_energy_peak = np.maximum(mechanical_energy_peak, mechanical_energy)

    solver_iteration_energy = np.stack(solver_iteration_energy)
    is_positive_energy = solver_iteration_energy[..., 0] > 0.0
    assert np.isfinite(pos).all()
    assert_equal(tensor_to_array(floor_entity.get_state().pos), floor_pos_initial)
    assert_allclose(scene.ipbstf_solver.get_kinetic_energy(), kinetic_energy, tol=5e-4)
    assert (mechanical_energy_peak < 1.05 * mechanical_energy_initial).all()
    assert (mechanical_energy > 0.9 * mechanical_energy_initial).all()
    assert (kinetic_energy > 0.95 * kinetic_energy_peak).all()
    assert (pos[..., 1] >= 0.0).all()
    assert (pos[..., 1].mean(axis=1) < pos_initial[..., 1].mean(axis=1) - 0.3).all()
    assert (np.ptp(pos[..., 0], axis=1) > 1.5 * np.ptp(pos_initial[..., 0], axis=1)).all()
    assert (np.diff(solver_iteration_energy, axis=-1) <= 0.0).all()
    assert (
        solver_iteration_energy[..., -1][is_positive_energy]
        < solver_iteration_energy[..., 0][is_positive_energy]
    ).all()
    assert_equal(
        solver_iteration_energy[..., -1][~is_positive_energy],
        np.zeros_like(solver_iteration_energy[..., -1][~is_positive_energy]),
    )
