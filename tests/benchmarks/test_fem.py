from time import perf_counter

import pytest

import quadrants as qd

import genesis as gs
from genesis.utils.element import create_tetrahedral_grid

from ..utils import assert_allclose


@pytest.mark.benchmarks
@pytest.mark.parametrize("n_envs", [0, 2])
def test_implicit_convergence(n_envs, show_viewer, record_property):
    N_STEPS = 20
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=10,
            gravity=(0.0, 0.0, 0.0),
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
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
        material=gs.materials.FEM.Elastic(
            E=1e4,
            nu=0.3,
            rho=1000.0,
            model="linear_corotated",
        ),
    )
    scene.build(n_envs=n_envs)
    pos_initial = entity.get_state().pos
    scene.step()
    qd.sync()

    time_start = perf_counter()
    for _ in range(N_STEPS):
        scene.step()
    qd.sync()
    runtime_fps = N_STEPS * max(n_envs, 1) / (perf_counter() - time_start)
    record_property("runtime_fps", runtime_fps)
    gs.logger.info(f"Implicit finite element method (FEM): {runtime_fps:.2f} FPS across {max(n_envs, 1)} environments.")

    state = entity.get_state()
    assert_allclose(state.pos, pos_initial, atol=1e-6)
    assert_allclose(state.vel, desired=0.0, atol=1e-5)
