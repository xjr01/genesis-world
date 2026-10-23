from time import perf_counter

import pytest

import quadrants as qd

import genesis as gs
from genesis.utils.element import create_tetrahedral_grid

from ..utils import assert_allclose


@pytest.mark.benchmarks
@pytest.mark.parametrize("n_envs", [0, 2])
def test_vertex_descent(n_envs, show_viewer, record_property):
    n_steps = 20
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        vbd_options=gs.options.VBDOptions(
            convergence_tolerance=0.0,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.3, 0.7),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    vertices, elements = create_tetrahedral_grid(
        lower=(-0.05, -0.05, -0.05), upper=(0.05, 0.05, 0.05), resolution=(8, 8, 8)
    )
    entity = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=(0.0, 0.0, 0.5),
            vertices=vertices,
            elements=elements,
        ),
        material=gs.materials.VBD.Elastic(),
    )
    scene.build(n_envs=n_envs)
    pos_initial = entity.get_state().pos
    scene.step()
    scene.reset()
    qd.sync()

    time_start = perf_counter()
    for _ in range(n_steps):
        scene.step()
    qd.sync()
    runtime_fps = n_steps * max(n_envs, 1) / (perf_counter() - time_start)
    record_property("runtime_fps", runtime_fps)
    gs.logger.info(f"Vertex block descent (VBD): {runtime_fps:.2f} FPS across {max(n_envs, 1)} environments.")

    state = entity.get_state()
    assert_allclose(state.pos, pos_initial, atol=1e-6)
    assert_allclose(state.vel, desired=0.0, atol=1e-5)
