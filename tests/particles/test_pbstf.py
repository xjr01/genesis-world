import numpy as np
import pytest

import genesis as gs


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
def test_pbstf_cuda(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, gravity=(0.0, 0.0, 0.0)),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=0.08,
            lower_bound=(-0.5, -0.5, -0.5),
            upper_bound=(0.5, 0.5, 0.5),
            max_solver_iterations=2,
            topology_rebuild_interval=1,
            max_surface_neighbors=64,
        ),
        show_viewer=show_viewer,
    )
    liquid = scene.add_entity(
        material=gs.materials.PBSTF.Liquid(sampler="regular"),
        morph=gs.morphs.Box(lower=(-0.16, -0.16, -0.16), upper=(0.16, 0.16, 0.16)),
    )
    scene.build()

    solver = scene.pbstf_solver
    assert solver.n_particles == liquid.n_particles
    assert not scene.pbd_solver.is_active
    assert not hasattr(scene.pbd_solver, "_surface_cubic_spline")

    solver._kernel_reorder_particles(0)
    solver._kernel_compute_density(0)
    active = solver.particles_ng_reordered.active.to_numpy()[:, 0].astype(bool)
    densities = solver.particles_reordered.density.to_numpy()[:, 0][active]
    assert densities.max() == pytest.approx(liquid.material.rho, rel=2e-4)

    positions_initial = liquid.get_particles_pos().cpu().numpy()
    scene.step()
    positions = liquid.get_particles_pos().cpu().numpy()
    assert np.isfinite(positions).all()
    assert np.linalg.norm(positions - positions_initial) > 1e-7
    surface = solver.on_surface.to_numpy()[:, 0].astype(bool)
    assert surface.any()
    assert (solver.n_neighbors.to_numpy()[:, 0][surface] >= 3).any()


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
def test_pbstf_cpp_cube_converges_to_sphere(show_viewer):
    """The C++ reference cube must become a closed, nearly spherical droplet."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 30.0, gravity=(0.0, 0.0, 0.0)),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=0.2,
            lower_bound=(-2.0, -2.0, -2.0),
            upper_bound=(2.0, 2.0, 2.0),
            max_solver_iterations=100,
            topology_rebuild_interval=10,
            max_surface_neighbors=128,
            enable_pca_normals=False,
        ),
        show_viewer=show_viewer,
    )
    liquid = scene.add_entity(
        material=gs.materials.PBSTF.Liquid(
            sampler="staggered",
            rho=1000.0,
            density_compliance=500.0,
            surface_tension_compliance=0.8,
            surface_distance_compliance=40.0,
            interior_distance_compliance=180.0,
            surface_viscosity=0.3,
            interior_viscosity=0.3,
        ),
        morph=gs.morphs.Box(lower=(-1.0, -1.0, -1.0), upper=(1.0, 1.0, 1.0)),
    )
    scene.build()

    solver = scene.pbstf_solver
    assert liquid.n_particles == 1099
    assert solver._support_radius == pytest.approx(3.0 * solver.particle_size)

    def surface_radius_cv():
        solver._kernel_reorder_particles(0)
        solver._rebuild_topology(0)
        positions = solver.particles_reordered.pos.to_numpy()[:, 0]
        on_surface = solver.on_surface.to_numpy()[:, 0].astype(bool)
        topology_valid = solver.topology_valid.to_numpy()[:, 0].astype(bool)
        surface_positions = positions[on_surface]
        radii = np.linalg.norm(surface_positions - surface_positions.mean(axis=0), axis=1)
        return radii.std() / radii.mean(), on_surface, topology_valid

    initial_cv, initial_surface, initial_valid = surface_radius_cv()
    assert initial_surface.sum() == initial_valid.sum()

    for _ in range(5):
        scene.step()

    final_cv, final_surface, final_valid = surface_radius_cv()
    assert final_surface.sum() == final_valid.sum()
    assert final_cv < 0.01
    assert final_cv < 0.1 * initial_cv


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_pbstf_rejects_non_cuda(show_viewer):
    scene = gs.Scene(
        pbstf_options=gs.options.PBSTFOptions(particle_size=0.1),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        material=gs.materials.PBSTF.Liquid(sampler="regular"),
        morph=gs.morphs.Nowhere(n_particles=1),
    )
    with pytest.raises(gs.GenesisException, match="requires the CUDA backend"):
        scene.build()
