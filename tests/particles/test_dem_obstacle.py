from pathlib import Path

import numpy as np
import pytest

import genesis as gs
from tests.utils import assert_allclose


@pytest.mark.parametrize("n_envs", [0, 2])
def test_mesh_slots_contact_and_checkpoint(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            gravity=(0.0, 0.0, -9.8),
        ),
        dem_options=gs.options.DEMOptions(
            particle_size=0.00625,
            lower_bound=(-0.6, -0.2, -0.2),
            upper_bound=(0.3, 0.2, 0.4),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.35, 0.25),
            camera_lookat=(0.05, 0.02, 0.01),
        ),
        show_viewer=show_viewer,
    )
    grains = []
    for pos in ((0.05, 0.0, 0.04), (0.05, 0.045, 0.04)):
        grains.append(
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=pos,
                    size=(0.001, 0.001, 0.001),
                ),
                material=gs.materials.DEM.Sand(
                    sampler="fcc",
                    rho=2500.0,
                    young_modulus=10000.0,
                ),
            )
        )
    scene.build(n_envs=n_envs)
    asset = Path(__file__).parents[2] / "examples/sand_water_coupling/assets/litter_scoop_sdf_slots.npz"
    with np.load(asset) as sdf:
        scene.dem_solver.set_sdf_obstacle(sdf["sdf_val"], sdf["dims"], sdf["origin"], sdf["cell"], (0.0, 0.0, 0.0))
    scene.dem_solver.set_sdf_obstacle_vel((0.01, 0.0, 0.0), omega=(0.0, 0.1, 0.0))
    scene.step()
    saved_pos = scene.dem_solver.get_sdf_obstacle_pos()
    saved_quat = scene.dem_solver.get_sdf_obstacle_quat()
    state = scene.get_state()
    scene.step()
    scene.restore(state)
    assert_allclose(scene.dem_solver.get_sdf_obstacle_pos(), saved_pos, atol=1e-7)
    assert_allclose(scene.dem_solver.get_sdf_obstacle_quat(), saved_quat, atol=1e-7)
    scene.step()
    assert_allclose(scene.dem_solver.get_sdf_obstacle_pos() - saved_pos, ((0.00005, 0.0, 0.0),), atol=1e-7)
    scene.dem_solver.set_sdf_obstacle_pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    scene.dem_solver.set_sdf_obstacle_vel((0.0, 0.0, 0.0))
    for _ in range(20):
        scene.step()
    assert (grains[0].get_particles_pos()[..., 2] > 0.011).all()
    assert (grains[1].get_particles_pos()[..., 2] < 0.003).all()
