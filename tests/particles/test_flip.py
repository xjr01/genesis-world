import math

import numpy as np

import igl
import pytest

import genesis as gs
from genesis.utils import mesh
from genesis.utils.misc import tensor_to_array

from ..utils import assert_equal


@pytest.mark.required
@pytest.mark.parametrize("substeps", [1, 2])
def test_prescribed_boundaries_displace_water(substeps, show_viewer):
    rotation = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    cup_asset = "meshes/drinking_glass/12-oz-glass.obj"
    radius = 0.9 * 0.070522047730 / 2.0
    start_height = 0.104291584329 + radius + 0.015
    stop_height = 0.012614391867 + radius
    dt = 1.0 / 240.0
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=substeps,
        ),
        flip_options=gs.options.FLIPOptions(
            static_colliders=[
                gs.options.FLIPMeshStaticColliderOptions(
                    quat=rotation,
                    file=cup_asset,
                    sdf_res=128,
                ),
                gs.options.FLIPSphereStaticColliderOptions(
                    pos=(0.0, 0.0, start_height),
                    is_kinematic=True,
                    radius=radius,
                ),
            ],
            grid_res=96,
            lower_bound=(-0.08, -0.08, 0.0),
            upper_bound=(0.08, 0.08, 0.192),
            seed_sub_factor=2,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.4, 0.25),
            camera_lookat=(0.0, 0.0, 0.08),
        ),
        show_viewer=show_viewer,
    )
    water = scene.add_entity(
        morph=gs.morphs.Mesh(
            quat=rotation,
            file="meshes/drinking_glass/12-oz-glass-cavity.obj",
        ),
        material=gs.materials.FLIP.Liquid(),
    )
    scene.build()
    for _ in range(8):
        scene.step()
    positions = water.get_particles_pos()[0]
    assert (positions[:, :2].norm(dim=-1) < 0.038).sum() > 0.99 * water.n_particles
    peak = 0.0
    for step in range(40):
        height = start_height + (stop_height - start_height) * (step + 1) / 40
        scene.flip_solver.set_static_colliders_pose([0.0, 0.0, height], [1.0, 0.0, 0.0, 0.0], colliders_idx=[1])
        scene.step()
        positions = water.get_particles_pos()[0]
        assert ((positions - positions.new_tensor([0.0, 0.0, height])).norm(dim=-1) >= radius - 1e-5).all()
        peak = max(peak, positions[:, 2].max().item())
    for _ in range(8):
        scene.step()
    positions = water.get_particles_pos()[0]
    assert_equal(water.get_particles_active().sum().item(), water.n_particles)
    assert np.isfinite(tensor_to_array(water.get_particles_vel())).all()
    is_ejected = (positions[:, :2].norm(dim=-1) > 0.0367) | (positions[:, 2] > 0.1043)
    ejected_volume = is_ejected.sum() * (scene.flip_solver.dx / scene.flip_solver.seed_sub_factor) ** 3
    assert ejected_volume > 0.5 * (4.0 / 3.0 * math.pi * radius**3)
    assert peak > 0.115
    assert (positions[:, :2].norm(dim=-1) > 0.038).sum() > 0.05 * water.n_particles
    assert ((positions - positions.new_tensor([0.0, 0.0, height])).norm(dim=-1) >= radius - 1e-5).all()
    glass = mesh.load_mesh(gs.utils.get_assets_dir() + "/" + cup_asset)
    local_positions = tensor_to_array(positions)[:, (0, 2, 1)].copy()
    local_positions[:, 2] *= -1
    distances, *_ = igl.signed_distance(local_positions, glass.vertices, glass.faces)
    assert (distances > -0.001).all()
