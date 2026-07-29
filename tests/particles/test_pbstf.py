import numpy as np
import pytest

import genesis as gs
from genesis.engine.boundaries import ConeStaticCollider, StaticCollider

from examples.pbstf_surface_tension import CASE_BOUNCE, CASE_CONE, CASE_MERGE, build_scene


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
def test_cone_static_collider_geometry():
    """The analytic cone exposes consistent GPU closest-point, normal, and collision queries."""
    import quadrants as qd

    collider = ConeStaticCollider(center=(0.0, 0.0, 0.0), height=(0.0, 2.0, 0.0), radius=2.0)
    points = np.array(
        [
            (0.5, 1.0, 0.0),
            (1.5, 1.0, 0.0),
            (0.25, -0.5, 0.0),
            (0.5, -0.05, 0.0),
            (1.95, 0.1, 0.0),
            (1.85, 0.2, 0.0),
        ],
        dtype=gs.np_float,
    )

    @qd.data_oriented
    class ColliderProbe:
        def __init__(self):
            self.n_points = len(points)
            self.points = qd.field(gs.qd_vec3, shape=(self.n_points,))
            self.closest = qd.field(gs.qd_vec3, shape=(self.n_points,))
            self.normals = qd.field(gs.qd_vec3, shape=(self.n_points,))
            self.projected = qd.field(gs.qd_vec3, shape=(self.n_points,))
            self.inside = qd.field(gs.qd_bool, shape=(self.n_points,))
            self.separated = qd.field(gs.qd_bool, shape=(2,))
            self.points.from_numpy(points)

        @qd.kernel
        def run(self):
            for i in range(self.n_points):
                self.closest[i] = collider.closest_position(self.points[i])
                self.normals[i] = collider.closest_normal(self.points[i])
                self.projected[i] = collider.project_out(self.points[i])
                self.inside[i] = collider.is_inside(self.points[i])
            self.separated[0] = collider.separates(self.points[3], self.points[4], 0.2)
            self.separated[1] = collider.separates(self.points[4], self.points[5], 0.2)

    probe = ColliderProbe()
    probe.run()

    closest = probe.closest.to_numpy()
    normals = probe.normals.to_numpy()
    projected = probe.projected.to_numpy()
    inside = probe.inside.to_numpy().astype(bool)
    separated = probe.separated.to_numpy().astype(bool)

    np.testing.assert_allclose(closest[0], (0.75, 1.25, 0.0), atol=1e-5)
    np.testing.assert_allclose(closest[1], (1.25, 0.75, 0.0), atol=1e-5)
    np.testing.assert_allclose(closest[2], (0.25, 0.0, 0.0), atol=1e-5)
    np.testing.assert_allclose(normals[0], np.sqrt(0.5) * np.array((1.0, 1.0, 0.0)), atol=1e-5)
    np.testing.assert_allclose(normals[2], (0.0, -1.0, 0.0), atol=1e-5)
    np.testing.assert_allclose(projected[0], closest[0], atol=1e-5)
    np.testing.assert_allclose(projected[1:], points[1:], atol=1e-5)
    np.testing.assert_array_equal(inside, (True, False, False, False, False, False))
    np.testing.assert_array_equal(separated, (True, False))

    with pytest.raises(TypeError):
        StaticCollider()


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
@pytest.mark.parametrize("backend", [gs.cuda])
def test_pbstf_cpp_two_cubes_merge(show_viewer):
    """C++ buildCase3 must turn two opposing box droplets into one connected drop."""
    scene, (left, right) = build_scene(case=CASE_MERGE, scale=10, show_viewer=show_viewer)
    solver = scene.pbstf_solver

    left_initial = left.get_particles_pos().cpu().numpy()
    right_initial = right.get_particles_pos().cpu().numpy()
    initial_gap = np.linalg.norm(left_initial[:, None, :] - right_initial[None, :, :], axis=2).min()
    assert initial_gap > solver._support_radius

    for _ in range(15):
        scene.step()

    left_pos = left.get_particles_pos().cpu().numpy()
    right_pos = right.get_particles_pos().cpu().numpy()
    velocities = np.concatenate(
        (left.get_particles_vel().cpu().numpy(), right.get_particles_vel().cpu().numpy()), axis=0
    )
    final_gap = np.linalg.norm(left_pos[:, None, :] - right_pos[None, :, :], axis=2).min()

    assert np.isfinite(left_pos).all()
    assert np.isfinite(right_pos).all()
    # Each staggered droplet is internally connected at one particle diameter;
    # this cross-edge therefore joins them into a single neighbor component.
    assert final_gap < solver.particle_size
    assert abs(velocities[:, 0].mean()) < 5e-3
    assert abs(velocities[:, 2].mean()) < 5e-3


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
def test_pbstf_cpp_drop_hits_floor_and_bounces(show_viewer):
    """C++ buildCase0 must hit y=-2, rebound, and leave the floor without penetration."""
    scene, (drop,) = build_scene(case=CASE_BOUNCE, scale=10, show_viewer=show_viewer)
    floor_height = -2.0
    touched_floor = False
    upward_after_contact = False
    left_floor = False

    for _ in range(25):
        scene.step()
        positions = drop.get_particles_pos().cpu().numpy()
        velocities = drop.get_particles_vel().cpu().numpy()
        min_y = positions[:, 1].min()

        assert np.isfinite(positions).all()
        assert min_y >= floor_height - 1e-6
        if min_y <= floor_height + 1e-5:
            touched_floor = True
        if touched_floor and velocities[:, 1].mean() > 0.05:
            upward_after_contact = True
        if touched_floor and min_y > floor_height + 0.01:
            left_floor = True

    assert touched_floor
    assert upward_after_contact
    assert left_floor


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
def test_pbstf_cpp_drop_hits_cone_tip(show_viewer):
    """C++ buildCase15 must collide with the analytic cone and rebound without penetration."""
    scene, (drop,) = build_scene(case=CASE_CONE, scale=10, show_viewer=show_viewer)
    solver = scene.pbstf_solver
    hit_cone = False
    rebounded = False

    assert solver._n_static_colliders == 1
    assert isinstance(solver._static_colliders[0], ConeStaticCollider)
    for _ in range(20):
        scene.step()
        positions = drop.get_particles_pos().cpu().numpy()
        velocities = drop.get_particles_vel().cpu().numpy()
        radial = np.linalg.norm(positions[:, (0, 2)], axis=1)
        cone_radius = np.sqrt(3.0) * (-2.0 - positions[:, 1])
        below_tip = positions[:, 1] <= -2.0
        above_base = positions[:, 1] >= -7.0

        assert np.isfinite(positions).all()
        assert not np.any(below_tip & above_base & (radial < cone_radius - 2e-5))
        if np.any(below_tip & (radial <= cone_radius + 2e-3)):
            hit_cone = True
        if hit_cone and velocities[:, 1].mean() > 0.05:
            rebounded = True

    on_surface = solver.on_surface.to_numpy()[:, 0].astype(bool)
    topology_valid = solver.topology_valid.to_numpy()[:, 0].astype(bool)
    assert hit_cone
    assert rebounded
    assert topology_valid[on_surface].all()


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
