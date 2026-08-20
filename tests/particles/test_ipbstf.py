import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array
from tests.utils import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_unilateral_density_energy_and_viscosity(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0),
        ),
        ipbstf_options=gs.options.IPBSTFOptions(
            particle_size=0.1,
            static_colliders=[
                gs.options.PBSTFBoxStaticColliderOptions(
                    lower=(0.5, 0.0, -0.5),
                    upper=(1.5, 1.0, 0.5),
                ),
            ],
            lower_bound=(-2.0, -2.0, -2.0),
            upper_bound=(2.0, 2.0, 2.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.2, 2.2, 1.8),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    liquid_material = gs.materials.IPBSTF.Liquid(
        sampler="staggered",
        viscosity=1.0,
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Box(
            lower=(-0.25, -0.25, -0.25),
            upper=(0.25, 0.25, 0.25),
        ),
        material=liquid_material,
    )
    collider_liquid = scene.add_entity(
        morph=gs.morphs.Particles(
            positions=((1.0, 0.02, 0.0),),
        ),
        material=liquid_material,
    )
    moving_boundary_liquid = scene.add_entity(
        morph=gs.morphs.Particles(
            positions=((-1.5, 0.0, 0.0),),
        ),
        material=liquid_material,
    )
    viscous_pair = scene.add_entity(
        morph=gs.morphs.Particles(
            positions=((-0.05, -1.5, 0.0), (0.05, -1.5, 0.0)),
        ),
        material=liquid_material,
    )
    moving_boundary_positions = tuple(
        (-1.5 + x_offset, 1.0 + y_offset, z_offset)
        for x_offset in (0.05, 0.1)
        for y_offset in (-0.05, 0.0, 0.05)
        for z_offset in (-0.05, 0.0, 0.05)
    )
    moving_boundary = scene.add_entity(
        morph=gs.morphs.Particles(
            positions=moving_boundary_positions,
        ),
        material=gs.materials.IPBSTF.Solid(
            sampler="staggered",
        ),
    )
    scene.build(n_envs=n_envs)
    viscous_pair.set_particles_vel(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))

    pos_initial = tensor_to_array(liquid.get_state().pos)
    scene.step()
    viscous_pair_vel = tensor_to_array(viscous_pair.get_state().vel)
    assert_allclose(tensor_to_array(liquid.get_state().pos), pos_initial, atol=1e-6)
    assert_allclose(viscous_pair_vel.mean(axis=-2), 0.0, atol=1e-6)
    assert (np.ptp(viscous_pair_vel[..., 0], axis=-1) < 2.0).all()
    collider_pos = tensor_to_array(collider_liquid.get_state().pos)
    collider_pos_expected = np.zeros_like(collider_pos)
    collider_pos_expected[..., 0] = 1.0
    collider_pos_expected[..., 1] = -0.05
    assert_allclose(collider_pos, collider_pos_expected, atol=1e-6)
    moving_boundary_liquid_pos_initial = tensor_to_array(moving_boundary_liquid.get_state().pos)

    center = pos_initial.mean(axis=1, keepdims=True)
    pos_compressed = center + 0.65 * (pos_initial - center)
    liquid.set_particles_pos(pos_compressed)
    moving_boundary.set_particles_pos(tuple((x, y - 1.0, z) for x, y, z in moving_boundary_positions))
    scene.step()
    pos_expanded = tensor_to_array(liquid.get_state().pos)
    moving_boundary_liquid_pos = tensor_to_array(moving_boundary_liquid.get_state().pos)

    assert_allclose(pos_expanded.mean(axis=1), center[..., 0, :], atol=1e-5)
    assert (np.ptp(pos_expanded, axis=1) > 1.1 * np.ptp(pos_compressed, axis=1)).all()
    assert (moving_boundary_liquid_pos[..., 0] < moving_boundary_liquid_pos_initial[..., 0] - 0.01).all()


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_dam_break_spreads_under_gravity(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            gravity=(0.0, -9.81, 0.0),
        ),
        ipbstf_options=gs.options.IPBSTFOptions(
            particle_size=0.1,
            max_solver_iterations=5,
            static_colliders=[
                gs.options.PBSTFInverseBoxStaticColliderOptions(
                    is_density_blocking=False,
                    lower=(-1.0, 0.0, -0.5),
                    upper=(2.0, 2.0, 0.5),
                ),
            ],
            lower_bound=(-1.0, -0.5, -0.5),
            upper_bound=(2.0, 2.0, 0.5),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.2, 2.2, 4.0),
            camera_lookat=(0.4, 0.6, 0.0),
            camera_up=(0.0, 1.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Box(
            lower=(-0.8, 0.1, -0.35),
            upper=(-0.1, 1.2, 0.35),
        ),
        material=gs.materials.IPBSTF.Liquid(
            sampler="staggered",
        ),
    )
    floor_entity = scene.add_entity(
        morph=gs.morphs.Box(
            lower=(-1.0, -0.2, -0.5),
            upper=(2.0, 0.0, 0.5),
        ),
        material=gs.materials.IPBSTF.Solid(
            sampler="staggered",
        ),
    )
    scene.build(n_envs=n_envs)

    pos_initial = tensor_to_array(liquid.get_state().pos)
    floor_pos_initial = tensor_to_array(floor_entity.get_state().pos)
    for _ in range(60):
        scene.step()
    pos = tensor_to_array(liquid.get_state().pos)

    assert np.isfinite(pos).all()
    assert_equal(tensor_to_array(floor_entity.get_state().pos), floor_pos_initial)
    assert (pos[..., 1] >= 0.05).all()
    assert (pos[..., 2] >= -0.45).all()
    assert (pos[..., 2] <= 0.45).all()
    assert (pos[..., 1].mean(axis=1) < pos_initial[..., 1].mean(axis=1) - 0.3).all()
    assert (np.ptp(pos[..., 0], axis=1) > 1.5 * np.ptp(pos_initial[..., 0], axis=1)).all()
