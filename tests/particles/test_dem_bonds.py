import numpy as np

import genesis as gs


def _bond_scene(positions, break_strain=None):
    positions = np.asarray(positions, dtype=np.float32)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0e-4, gravity=(0.0, 0.0, 0.0)),
        dem_options=gs.options.DEMOptions(
            particle_size=0.02,
            bond_enable=True,
            bond_break_strain=break_strain,
            restitution=1.0,
            lower_bound=(-0.2, -0.2, 0.0),
            upper_bound=(0.2, 0.2, 1.0),
        ),
        show_viewer=False,
    )
    grains = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.5), size=(0.08, 0.08, 0.08)),
        material=gs.materials.DEM.Sand(rho=2.5, young_modulus=1.0e5, sampler="poisson"),
    )
    scene.build()
    dem = scene.sim.dem_solver
    assert grains.n_particles >= len(positions)
    grains.set_particles_active(np.zeros((1, grains.n_particles), dtype=np.bool_))
    selected = np.arange(len(positions), dtype=np.int32)
    grains.set_particles_pos(positions[None], particles_idx_local=selected)
    grains.set_particles_vel(np.zeros_like(positions)[None], particles_idx_local=selected)
    grains.set_particles_active(np.ones((1, len(positions)), dtype=np.bool_), particles_idx_local=selected)
    ratio = np.zeros(dem.particles.ratio.shape, dtype=np.float32)
    ratio[grains.particle_start:grains.particle_start + len(positions), 0] = 1.0
    dem.particles.ratio.from_numpy(ratio)
    return scene, grains, dem


def test_wet_bonds_are_unique_reciprocal_and_restore_safe():
    spacing = 0.02001
    positions = [[-1.5 * spacing, 0.0, 0.5], [-0.5 * spacing, 0.0, 0.5],
                 [0.5 * spacing, 0.0, 0.5], [1.5 * spacing, 0.0, 0.5]]
    _, _, dem = _bond_scene(positions)

    # create_wet_bonds must rebuild the hash after a restore-like direct position setter.
    dem.create_wet_bonds(radius_factor=1.025)
    stats = dem.get_wet_bond_stats()
    assert stats["directed_edges"] == 6
    assert stats["unique_directed_edges"] == 6
    assert stats["undirected_edges"] == 3
    assert stats["duplicate_directed_edges"] == 0
    assert stats["asymmetric_edges"] == 0
    assert stats["overflow_particles"] == 0
    assert stats["max_rest_error_m"] < 1.0e-7
    assert stats["bond_ddt_limit_s"] > 0.0

    # Rebuilding must be idempotent rather than appending a second copy of every edge.
    dem.create_wet_bonds(radius_factor=1.025)
    assert dem.get_wet_bond_stats()["directed_edges"] == 6


def test_wet_bond_axial_force_is_equal_opposite_and_restoring():
    rest = 0.02001
    _, grains, dem = _bond_scene([[-0.5 * rest, 0.0, 0.5], [0.5 * rest, 0.0, 0.5]])
    dem.create_wet_bonds(radius_factor=1.025)

    stretched = np.asarray([[-0.5 * rest, 0.0, 0.5], [0.5 * rest + 0.001, 0.0, 0.5]], dtype=np.float32)
    grains.set_particles_pos(stretched[None])
    inv_mass, k_norm, *_ = dem._material_constants()
    dem._kernel_reset_acc(0)
    dem._kernel_reorder_particles(0)
    dem._kernel_bond_forces(0, dem._particle_radius, k_norm, inv_mass)
    dem._kernel_copy_acc_from_reordered(0)
    acceleration = dem.particles.acc.to_numpy()[:2, 0]

    assert acceleration[0, 0] > 0.0
    assert acceleration[1, 0] < 0.0
    np.testing.assert_allclose(acceleration[0], -acceleration[1], rtol=2.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(acceleration[:, 1:], 0.0, atol=1.0e-6)


def test_wet_bond_breaks_irreversibly_above_tensile_strain():
    rest = 0.02001
    _, grains, dem = _bond_scene(
        [[-0.5 * rest, 0.0, 0.5], [0.5 * rest, 0.0, 0.5]], break_strain=0.01
    )
    dem.create_wet_bonds(radius_factor=1.025)
    stretched = np.asarray([[-0.5 * rest, 0.0, 0.5], [0.5 * rest + 0.0003, 0.0, 0.5]], dtype=np.float32)
    grains.set_particles_pos(stretched[None])
    inv_mass, k_norm, *_ = dem._material_constants()
    dem._kernel_reset_acc(0)
    dem._kernel_reorder_particles(0)
    dem._kernel_bond_forces(0, dem._particle_radius, k_norm, inv_mass)
    stats = dem.get_wet_bond_stats()
    assert stats["created_directed_edges"] == 2
    assert stats["directed_edges"] == 0
    assert stats["broken_directed_edges"] == 2
