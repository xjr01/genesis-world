import math
import os

import numpy as np
import pytest

import igl
import trimesh

import quadrants as qd

from examples.teapot.pbstf_surface_tension import (
    CASE_BOUNCE,
    CASE_CONE,
    CASE_MERGE,
    CASE_MOP,
    CASE_SWEEP,
    CASE_TAP,
    CASE_TEAPOT,
    CASES,
    build_scene,
    case_settings,
    get_wipe_settings,
    teapot_pose,
    update_mop_case,
    update_teapot_manipulator,
    update_wipe_case,
    wipe_pose,
)
import genesis as gs
from genesis.engine.boundaries import (
    AbsorbentBoxStaticCollider,
    BoxStaticCollider,
    ConeStaticCollider,
    StaticCollider,
    build_deformable_surface_bvh,
    load_or_build_mesh_sdf,
    project_out_static_collider,
    query_static_collider,
    refit_deformable_surface_bvh,
    static_collider_separates,
)
import genesis.utils.geom as geom_utils
import genesis.utils.mesh as mesh_utils
from genesis.utils.misc import qd_to_numpy, tensor_to_array
import genesis.utils.particle as particle_utils
from tests.utils import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
def test_analytic_static_collider_geometry():
    collider = ConeStaticCollider(center=(0.0, 0.0, 0.0), height=(0.0, 2.0, 0.0), radius=2.0)
    particle_radius = 0.2
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

    points_qd = qd.field(gs.qd_vec3, shape=(len(points),))
    closest_qd = qd.field(gs.qd_vec3, shape=(len(points),))
    normals_qd = qd.field(gs.qd_vec3, shape=(len(points),))
    projected_qd = qd.field(gs.qd_vec3, shape=(len(points),))
    inside_qd = qd.field(gs.qd_bool, shape=(len(points),))
    separated_qd = qd.field(gs.qd_bool, shape=(2,))
    colliders_pos_qd = qd.field(gs.qd_vec3, shape=(1, 1))
    colliders_quat_qd = qd.field(gs.qd_vec4, shape=(1, 1))
    points_qd.from_numpy(points)
    colliders_pos_qd.from_numpy(np.zeros((1, 1, 3), dtype=gs.np_float))
    colliders_quat_qd.from_numpy(np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=gs.np_float))

    @qd.kernel
    def run(
        particle_radius: float,
        points_field: qd.template(),
        closest_field: qd.template(),
        normals_field: qd.template(),
        projected_field: qd.template(),
        inside_field: qd.template(),
        separated_field: qd.template(),
        colliders_pos: qd.template(),
        colliders_quat: qd.template(),
        collider_geometry: qd.template(),
    ):
        for i in range(points_field.shape[0]):
            closest, normal, is_inside, _ = query_static_collider(
                0, 0, points_field[i], colliders_pos, colliders_quat, collider_geometry
            )
            closest_field[i] = closest
            normals_field[i] = normal
            projected_field[i] = project_out_static_collider(
                0, 0, points_field[i], particle_radius, colliders_pos, colliders_quat, collider_geometry
            )
            inside_field[i] = is_inside
        separated_field[0] = static_collider_separates(
            0, 0, points_field[3], points_field[4], particle_radius, colliders_pos, colliders_quat, collider_geometry
        )
        separated_field[1] = static_collider_separates(
            0, 0, points_field[4], points_field[5], particle_radius, colliders_pos, colliders_quat, collider_geometry
        )

    run(
        particle_radius,
        points_qd,
        closest_qd,
        normals_qd,
        projected_qd,
        inside_qd,
        separated_qd,
        colliders_pos_qd,
        colliders_quat_qd,
        collider,
    )

    closest = qd_to_numpy(closest_qd, transpose=True)
    normals = qd_to_numpy(normals_qd, transpose=True)
    projected = qd_to_numpy(projected_qd, transpose=True)
    inside = qd_to_numpy(inside_qd, transpose=True)
    separated = qd_to_numpy(separated_qd, transpose=True)

    assert_allclose(closest[0], (0.75, 1.25, 0.0), atol=1e-5)
    assert_allclose(closest[1], (1.25, 0.75, 0.0), atol=1e-5)
    assert_allclose(closest[2], (0.25, 0.0, 0.0), atol=1e-5)
    assert_allclose(normals[0], np.sqrt(0.5) * np.array((1.0, 1.0, 0.0)), atol=1e-5)
    assert_allclose(normals[2], (0.0, -1.0, 0.0), atol=1e-5)
    assert_allclose(projected[0] - closest[0], particle_radius * normals[0], atol=1e-5)
    assert_allclose(projected[1:3], points[1:3], atol=1e-5)
    assert_allclose(projected[3:] - closest[3:], particle_radius * normals[3:], atol=1e-5)
    assert_equal(inside, (True, False, False, False, False, False))
    assert_equal(separated, (True, False))

    box = BoxStaticCollider(lower=(-1.0, -2.0, -3.0), upper=(1.0, 2.0, 3.0))
    box_points = np.array(
        [
            (0.9, 0.0, 0.0),
            (1.1, 0.0, 0.0),
            (1.5, 0.0, 0.0),
            (-1.05, 0.0, 0.0),
            (1.05, 0.0, 0.0),
            (1.1, 2.1, 3.1),
        ],
        dtype=gs.np_float,
    )
    points_qd.from_numpy(box_points)
    run(
        particle_radius,
        points_qd,
        closest_qd,
        normals_qd,
        projected_qd,
        inside_qd,
        separated_qd,
        colliders_pos_qd,
        colliders_quat_qd,
        box,
    )

    closest = qd_to_numpy(closest_qd, transpose=True)
    normals = qd_to_numpy(normals_qd, transpose=True)
    projected = qd_to_numpy(projected_qd, transpose=True)
    inside = qd_to_numpy(inside_qd, transpose=True)
    separated = qd_to_numpy(separated_qd, transpose=True)

    assert_allclose(closest[:3], ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), atol=1e-6)
    assert_allclose(normals[:3], ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), atol=1e-6)
    assert_allclose(projected[:2], ((1.2, 0.0, 0.0), (1.2, 0.0, 0.0)), atol=1e-6)
    assert_allclose(projected[2], box_points[2], atol=1e-6)
    assert_allclose(closest[5], (1.0, 2.0, 3.0), atol=1e-6)
    assert_allclose(normals[5], np.sqrt(1.0 / 3.0) * np.ones(3), atol=1e-6)
    assert_allclose(projected[5] - closest[5], particle_radius * normals[5], atol=1e-6)
    assert_equal(inside, (True, False, False, False, False, False))
    assert_equal(separated, (True, False))

    deformable_box = AbsorbentBoxStaticCollider(
        lower=(-1.0, -2.0, -3.0),
        upper=(1.0, 2.0, 3.0),
        absorption_rate=8.0,
        absorption_capacity_fraction=0.25,
        fem_entity_name="sponge",
    )
    deformable_mesh = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
    deformable_vertices = np.array(deformable_mesh.vertices, dtype=gs.np_float)
    deformable_faces = np.array(deformable_mesh.faces, dtype=gs.np_int)
    deformable_box.n_surface_vertices = len(deformable_vertices)
    deformable_box.n_surface_triangles = len(deformable_faces)
    deformable_box.surface_faces = qd.field(gs.qd_ivec3, shape=(deformable_box.n_surface_triangles,))
    deformable_box.surface_vertices = qd.field(gs.qd_vec3, shape=(deformable_box.n_surface_vertices, 1))
    deformable_box.surface_faces.from_numpy(deformable_faces)
    deformable_box.surface_vertices.from_numpy(deformable_vertices[:, None, :])
    build_deformable_surface_bvh(deformable_box, n_batches=1)
    deformable_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (-1.0, -3.2, 0.0),
            (1.0, 2.4, 0.0),
            (-1.0, -2.0, 0.0),
            (1.1, 2.1, 3.1),
            (0.0, 2.1, 0.0),
        ],
        dtype=gs.np_float,
    )
    points_qd.from_numpy(deformable_points)
    run(
        particle_radius,
        points_qd,
        closest_qd,
        normals_qd,
        projected_qd,
        inside_qd,
        separated_qd,
        colliders_pos_qd,
        colliders_quat_qd,
        deformable_box,
    )

    closest = qd_to_numpy(closest_qd, transpose=True)
    normals = qd_to_numpy(normals_qd, transpose=True)
    projected = qd_to_numpy(projected_qd, transpose=True)
    inside = qd_to_numpy(inside_qd, transpose=True)

    assert_allclose(closest[1], (-1.0, -2.0, 0.0), atol=1e-6)
    assert_allclose(normals[1], (0.0, -1.0, 0.0), atol=1e-6)
    assert_allclose(projected[1], deformable_points[1], atol=1e-6)
    assert_allclose(projected[3], (-1.2, -2.0, 0.0), atol=1e-6)
    assert_equal(inside, (True, False, False, True, False, False))

    translated_deformable_vertices = deformable_vertices + (4.0, 0.0, 0.0)
    refit_points = deformable_points.copy()
    refit_points[0] = (2.9, 0.0, 0.0)
    deformable_box.surface_vertices.from_numpy(translated_deformable_vertices[:, None, :])
    points_qd.from_numpy(refit_points)
    refit_deformable_surface_bvh(deformable_box)
    run(
        particle_radius,
        points_qd,
        closest_qd,
        normals_qd,
        projected_qd,
        inside_qd,
        separated_qd,
        colliders_pos_qd,
        colliders_quat_qd,
        deformable_box,
    )
    assert_allclose(qd_to_numpy(closest_qd, transpose=True)[0], (3.0, 0.0, 0.0), atol=1e-6)
    assert_allclose(qd_to_numpy(normals_qd, transpose=True)[0], (-1.0, 0.0, 0.0), atol=1e-6)
    assert_equal(qd_to_numpy(inside_qd, transpose=True), (False,) * len(deformable_points))

    dented_box = AbsorbentBoxStaticCollider(
        lower=(-1.0, -2.0, -3.0),
        upper=(1.0, 2.0, 3.0),
        absorption_rate=8.0,
        absorption_capacity_fraction=0.25,
        fem_entity_name="sponge",
        sdf_res=32,
    )
    dented_mesh = trimesh.creation.box(extents=(2.0, 4.0, 6.0)).subdivide()
    dented_vertices = np.array(dented_mesh.vertices, dtype=gs.np_float)
    dented_vertices[
        np.isclose(dented_vertices[:, 0], 1.0)
        & np.isclose(dented_vertices[:, 1], 0.0)
        & np.isclose(dented_vertices[:, 2], 0.0),
        0,
    ] = 0.5
    dented_faces = np.array(dented_mesh.faces, dtype=gs.np_int)
    dented_box.n_surface_vertices = len(dented_vertices)
    dented_box.n_surface_triangles = len(dented_faces)
    dented_box.surface_faces = qd.field(gs.qd_ivec3, shape=(dented_box.n_surface_triangles,))
    dented_box.surface_vertices = qd.field(gs.qd_vec3, shape=(dented_box.n_surface_vertices, 1))
    dented_box.sdf = qd.field(gs.qd_float, shape=(dented_box.sdf_res, dented_box.sdf_res, dented_box.sdf_res, 1))
    dented_box.sdf_lower = qd.field(gs.qd_vec3, shape=(1,))
    dented_box.sdf_inv_cell_size = qd.field(gs.qd_vec3, shape=(1,))
    dented_box.is_sdf_active = qd.field(gs.qd_bool, shape=(1,))
    dented_box.surface_faces.from_numpy(dented_faces)
    dented_box.surface_vertices.from_numpy(dented_vertices[:, None, :])
    dented_box.is_sdf_active.fill(False)
    build_deformable_surface_bvh(dented_box, n_batches=1)
    dented_points = np.array(
        (
            (0.8, 0.0, 0.0),
            (0.4, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.2, 0.0, 0.0),
            (0.8, 0.0, 0.0),
            (0.4, 0.0, 0.0),
        ),
        dtype=gs.np_float,
    )
    points_qd.from_numpy(dented_points)
    run(
        particle_radius,
        points_qd,
        closest_qd,
        normals_qd,
        projected_qd,
        inside_qd,
        separated_qd,
        colliders_pos_qd,
        colliders_quat_qd,
        dented_box,
    )
    projected = qd_to_numpy(projected_qd, transpose=True)
    inside = qd_to_numpy(inside_qd, transpose=True)
    assert_allclose(projected[0], dented_points[0], atol=1e-6)
    assert_equal(inside[:4], (False, True, True, False))

    sdf_data = load_or_build_mesh_sdf(dented_vertices, dented_faces, dented_box.sdf_res)
    dented_box.sdf.from_numpy(sdf_data.values[..., None])
    dented_box.sdf_lower.from_numpy(sdf_data.lower[None])
    dented_box.sdf_inv_cell_size.from_numpy((1.0 / sdf_data.cell_size)[None])
    dented_box.is_sdf_active.fill(True)
    run(
        particle_radius,
        points_qd,
        closest_qd,
        normals_qd,
        projected_qd,
        inside_qd,
        separated_qd,
        colliders_pos_qd,
        colliders_quat_qd,
        dented_box,
    )
    assert_equal(qd_to_numpy(inside_qd, transpose=True)[:4], (False, True, True, False))

    absorbent_box = AbsorbentBoxStaticCollider(
        lower=(-1.0, -2.0, -3.0),
        upper=(1.0, 2.0, 3.0),
        absorption_rate=8.0,
        absorption_capacity_fraction=0.25,
    )
    assert isinstance(absorbent_box, BoxStaticCollider)
    assert absorbent_box.type == "absorbent_box"

    with pytest.raises(gs.GenesisException, match="greater than"):
        gs.options.PBSTFBoxStaticColliderOptions(
            lower=(-1.0, -1.0, -1.0),
            upper=(1.0, -1.0, 1.0),
        )

    with pytest.raises(gs.GenesisException, match="requires `fem_entity_name`"):
        gs.options.PBSTFAbsorbentBoxStaticColliderOptions(
            lower=(-1.0, -1.0, -1.0),
            upper=(1.0, 1.0, 1.0),
            absorption_rate=1.0,
            absorption_capacity_fraction=1.0,
            sdf_res=16,
        )

    with pytest.raises(TypeError):
        StaticCollider()


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_static_collider_pose_and_absorption(asset_tmp_path, n_envs, show_viewer):
    mesh_path = asset_tmp_path / f"pbstf_static_collider_box_{n_envs}.obj"
    trimesh.creation.box().export(mesh_path)
    target_pos = (1.0, 2.0, 3.0)
    mesh_target_quat = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    absorbent_pos = (2.0, 0.0, 0.0)
    collider_options = [
        gs.options.PBSTFMeshStaticColliderOptions(
            file=str(mesh_path),
            sdf_res=16,
        ),
        gs.options.PBSTFMeshStaticColliderOptions(
            file=str(mesh_path),
            sdf_res=16,
        ),
        gs.options.PBSTFBoxStaticColliderOptions(
            pos=(-2.0, 0.0, 0.0),
            lower=(-0.3, -0.3, -0.3),
            upper=(0.3, 0.3, 0.3),
        ),
        gs.options.PBSTFAbsorbentBoxStaticColliderOptions(
            pos=absorbent_pos,
            lower=(-0.3, -0.3, -0.3),
            upper=(0.3, 0.3, 0.3),
            absorption_rate=100.0,
            absorption_capacity_fraction=0.4,
        ),
    ]
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            gravity=(0.0, 0.0, 0.0),
        ),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=0.1,
            lower_bound=(-4.0, -4.0, -4.0),
            upper_bound=(4.0, 4.0, 4.0),
            max_solver_iterations=1,
            max_surface_neighbors=16,
            static_colliders=collider_options,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(6.0, 6.0, 6.0),
            camera_lookat=(0.0, 0.5, 1.0),
        ),
        show_viewer=show_viewer,
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Particles(
            positions=(
                target_pos,
                (-2.0, 0.0, 0.0),
                (2.0, -2.0, -2.0),
                (2.0, 2.0, -2.0),
                (2.0, -2.0, 2.0),
                (2.0, 2.0, 2.0),
                (3.0, -2.0, -2.0),
                (3.0, 2.0, -2.0),
                (3.0, -2.0, 2.0),
                (3.0, 2.0, 2.0),
                (0.0, 3.0, 3.0),
            ),
        ),
        material=gs.materials.PBSTF.Liquid(
            sampler="regular",
        ),
    )
    scene.build(n_envs=n_envs)

    scene.pbstf_solver.set_static_colliders_pose(
        pos=target_pos,
        quat=mesh_target_quat,
        colliders_idx=1,
        envs_idx=n_envs - 1 if n_envs else None,
    )
    liquid.set_particles_pos((1.65, -0.15, -0.15), particles_idx_local=(2, 3, 4, 5, 6, 7, 8, 9, 10))

    solver = scene.pbstf_solver
    solver_state = solver.get_state(0)
    colliders_pos = tensor_to_array(solver_state.static_colliders_pos)
    colliders_quat = tensor_to_array(solver_state.static_colliders_quat)
    expected_pos = np.zeros((max(n_envs, 1), 4, 3), dtype=gs.np_float)
    expected_quat = np.zeros((max(n_envs, 1), 4, 4), dtype=gs.np_float)
    expected_quat[..., 0] = 1.0
    expected_pos[-1, 1] = target_pos
    expected_quat[-1, 1] = mesh_target_quat
    expected_pos[:, 2] = (-2.0, 0.0, 0.0)
    expected_pos[:, 3] = absorbent_pos
    initial_wetness = tensor_to_array(solver.get_static_collider_wetness(3))
    if n_envs == 0:
        initial_wetness = initial_wetness[None]

    assert liquid.n_particles == 11
    assert_allclose(colliders_pos, expected_pos, atol=1e-6)
    assert_allclose(colliders_quat, expected_quat, atol=1e-6)
    assert_equal(initial_wetness, np.zeros((max(n_envs, 1), 2, 2, 2)))
    assert_equal(tensor_to_array(solver_state.absorption_capture_budget), np.zeros((max(n_envs, 1), 1)))
    with pytest.raises(gs.GenesisException, match="not absorbent"):
        solver.get_static_collider_wetness(2)

    scene.step()
    particles_pos = tensor_to_array(liquid.get_particles_pos())
    if n_envs == 0:
        particles_pos = particles_pos[None]
    else:
        assert_equal(particles_pos[:-1, 0], (target_pos,))
    assert np.linalg.norm(particles_pos[-1, 0] - target_pos) > 0.4
    assert np.linalg.norm(particles_pos[:, 1] - (-2.0, 0.0, 0.0), axis=-1).min() > 0.3

    pending_solver_state = solver.get_state(0)
    assert_equal((tensor_to_array(pending_solver_state.absorption_progress) > 0.0).sum(axis=-1), 0)
    assert_allclose(tensor_to_array(pending_solver_state.absorption_capture_budget), 0.1, atol=1e-6)

    for _ in range(9):
        scene.step()
    first_solver_state = solver.get_state(0)
    first_progress = tensor_to_array(first_solver_state.absorption_progress)
    is_first_captured = first_progress > 0.0
    assert_equal(is_first_captured.sum(axis=-1), 1)
    assert_equal(tensor_to_array(first_solver_state.absorption_voxel_distance)[is_first_captured], 0)
    assert_allclose(first_progress[is_first_captured], 1.0 - math.exp(-0.1), atol=1e-6)
    assert_allclose(tensor_to_array(first_solver_state.absorption_capture_budget), 0.0, atol=1e-6)

    for _ in range(70):
        scene.step()
    particles_pos = tensor_to_array(liquid.get_particles_pos())
    if n_envs == 0:
        particles_pos = particles_pos[None]

    solver_state = solver.get_state(0)
    assert_allclose(tensor_to_array(solver_state.absorption_capture_budget), 0.0, atol=1e-6)
    progress = tensor_to_array(solver_state.absorption_progress)
    voxel_idx = tensor_to_array(solver_state.absorbed_voxel_idx)
    voxel_distances = tensor_to_array(solver_state.absorption_voxel_distance)
    local_pos = tensor_to_array(solver_state.absorption_local_pos)
    target_local_pos = tensor_to_array(solver_state.absorption_target_local_pos)
    is_captured = progress > 0.0
    beta = np.zeros_like(progress)
    beta[is_captured] = 1.0 - np.exp(-collider_options[3].absorption_rate * 1e-3 / (voxel_distances[is_captured] + 1))
    wetness = tensor_to_array(solver.get_static_collider_wetness(3))
    if n_envs == 0:
        wetness = wetness[None]
    expected_local_start = np.array((-0.35, -0.15, -0.15))

    assert_equal(is_captured.sum(axis=-1), 8)
    for env_idx in range(max(n_envs, 1)):
        assert_equal(np.sort(voxel_idx[env_idx, is_captured[env_idx]]), tuple(range(8)))
        assert_equal(
            np.sort(voxel_distances[env_idx, is_captured[env_idx]]),
            (0, 1, 1, 1, 2, 2, 2, 3),
        )
        assert_allclose(
            np.sort(wetness[env_idx], axis=None),
            np.sort(progress[env_idx, is_captured[env_idx]]),
            atol=1e-6,
        )
    assert_allclose(
        local_pos[is_captured],
        expected_local_start + progress[is_captured][:, None] * (target_local_pos[is_captured] - expected_local_start),
        atol=1e-6,
    )
    assert ((0.0 <= wetness) & (wetness <= 1.0)).all()
    assert liquid.get_particles_active().all()
    assert_allclose(particles_pos[..., 2:, 0].min(axis=-1), 1.65, atol=1e-6)

    moved_pos = (2.5, 0.3, 0.2)
    moved_quat = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    solver.set_static_colliders_pose(
        pos=moved_pos,
        quat=moved_quat,
        colliders_idx=3,
    )
    scene.step()
    particles_moved = tensor_to_array(liquid.get_particles_pos())
    if n_envs == 0:
        particles_moved = particles_moved[None]
    progress_expected = progress + beta * (1.0 - progress)
    local_pos_expected = local_pos + beta[..., None] * (target_local_pos - local_pos)
    for env_idx in range(max(n_envs, 1)):
        expected_world = geom_utils.transform_by_quat(
            local_pos_expected[env_idx, is_captured[env_idx]], np.array(moved_quat)
        )
        expected_world += moved_pos
        assert_allclose(particles_moved[env_idx, is_captured[env_idx]], expected_world, atol=1e-5)
    solver_state = solver.get_state(0)
    capture_budget_saved = tensor_to_array(solver_state.absorption_capture_budget)
    assert_allclose(
        tensor_to_array(solver_state.absorption_progress)[is_captured], progress_expected[is_captured], atol=1e-6
    )
    assert_allclose(capture_budget_saved, 0.1, atol=1e-6)

    saved_state = scene.get_state()
    positions_saved = particles_moved.copy()
    wetness_saved = tensor_to_array(solver.get_static_collider_wetness(3))
    solver.set_static_colliders_pose(
        pos=(-2.5, -0.3, -0.2),
        quat=(1.0, 0.0, 0.0, 0.0),
        colliders_idx=3,
    )
    scene.step()
    scene.reset(saved_state)
    positions_restored = tensor_to_array(liquid.get_particles_pos())
    if n_envs == 0:
        positions_restored = positions_restored[None]
    restored_state = solver.get_state(0)
    assert_allclose(positions_restored, positions_saved, atol=1e-6)
    assert_allclose(solver.get_static_collider_wetness(3), wetness_saved, atol=1e-6)
    assert_allclose(tensor_to_array(restored_state.static_colliders_pos)[:, 3], moved_pos, atol=1e-6)
    assert_allclose(tensor_to_array(restored_state.static_colliders_quat)[:, 3], moved_quat, atol=1e-6)
    assert_equal(tensor_to_array(restored_state.absorption_voxel_distance), voxel_distances)
    assert_allclose(tensor_to_array(restored_state.absorption_capture_budget), capture_budget_saved, atol=1e-6)

    wetness_sum = tensor_to_array(solver.get_static_collider_wetness(3))
    if n_envs == 0:
        wetness_sum = wetness_sum[None]
    wetness_sum = wetness_sum.sum(axis=(1, 2, 3))
    if n_envs:
        particles_idx_local = np.argmax(is_captured, axis=-1)[:, None]
    else:
        particles_idx_local = np.argmax(is_captured[0])
    for setter, value in (
        (liquid.set_particles_pos, (0.0, 0.0, 0.0)),
        (liquid.set_particles_vel, (0.0, 0.0, 0.0)),
        (liquid.set_particles_active, False),
    ):
        scene.reset(saved_state)
        setter(value, particles_idx_local=particles_idx_local)
        wetness_sum_unbound = tensor_to_array(solver.get_static_collider_wetness(3))
        if n_envs == 0:
            wetness_sum_unbound = wetness_sum_unbound[None]
        wetness_sum_unbound = wetness_sum_unbound.sum(axis=(1, 2, 3))
        assert (wetness_sum_unbound < wetness_sum).all()

    saved_solver_state = saved_state.solvers_state[scene.solvers.index(solver)]
    saved_solver_state.absorption_capture_budget[:] = math.nan
    scene.reset(saved_state)
    with pytest.raises(gs.GenesisException, match="non-finite fluid or absorption"):
        solver.check_errno()
    saved_solver_state.absorption_capture_budget[:] = 0.0
    saved_solver_state.absorption_progress[:] = math.nan
    scene.reset(saved_state)
    with pytest.raises(gs.GenesisException, match="non-finite fluid or absorption"):
        solver.check_errno()


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_static_collider_adhesion_and_friction(asset_tmp_path, n_envs, show_viewer):
    mesh_path = asset_tmp_path / "pbstf_adhesion_remote_box.obj"
    trimesh.creation.box().export(mesh_path)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            gravity=(0.0, 0.0, 0.0),
        ),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=0.2,
            lower_bound=(-3.0, -3.0, -3.0),
            upper_bound=(3.0, 3.0, 3.0),
            max_solver_iterations=1,
            max_surface_neighbors=16,
            static_colliders=[
                gs.options.PBSTFConeStaticColliderOptions(
                    center=(0.0, 0.0, 0.0),
                    height=(0.0, 2.0, 0.0),
                    radius=2.0,
                ),
                gs.options.PBSTFMeshStaticColliderOptions(
                    pos=(3.0, 0.0, 0.0),
                    file=str(mesh_path),
                    sdf_res=16,
                ),
                gs.options.PBSTFBoxStaticColliderOptions(
                    pos=(1.5, 0.0, 0.0),
                    lower=(-0.4, -0.5, -0.4),
                    upper=(0.4, -0.05, 0.4),
                ),
            ],
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(4.0, -5.0, 4.0),
            camera_lookat=(-0.5, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Particles(
            positions=((0.0, -0.05, 0.0), (0.0, -0.15, 0.0), (1.5, -0.1, 0.0)),
        ),
        material=gs.materials.PBSTF.Liquid(
            sampler="regular",
            is_collider_adhesion_friction_enabled=True,
            collider_adhesion_compliance=10.0,
            collider_friction=0.25,
        ),
    )
    scene.build(n_envs=n_envs)
    solver = scene.pbstf_solver

    solver._kernel_reorder_particles(0)
    solver.particles_reordered.dpos.fill(0.0)
    solver.on_surface.fill(True)
    solver._kernel_apply_static_collider_adhesion()
    reordered_idx = qd_to_numpy(solver.particles_ng.reordered_idx, transpose=True)
    adhesion_delta = np.take_along_axis(
        qd_to_numpy(solver.particles_reordered.dpos, transpose=True), reordered_idx[..., None], axis=1
    )
    mass = np.take_along_axis(qd_to_numpy(solver.particles_info_reordered.mass, transpose=True), reordered_idx, axis=1)
    denominator = liquid.material.collider_adhesion_compliance / solver._default_mass + 1.0 / mass
    expected_adhesion_delta = np.zeros_like(adhesion_delta)
    expected_adhesion_delta[..., :2, 1] = np.array((-0.05, 0.05)) / denominator[..., :2] / mass[..., :2]
    assert_allclose(adhesion_delta, expected_adhesion_delta, atol=1e-6)

    liquid.set_particles_vel((1.0, 1.0, 0.0))
    solver._kernel_reorder_particles(0)
    solver.particles_reordered.dpos.fill(0.0)
    solver.particles_reordered.surface.fill(True)
    solver._kernel_apply_viscosity()
    velocity = np.take_along_axis(
        qd_to_numpy(solver.particles_reordered.vel, transpose=True), reordered_idx[..., None], axis=1
    )
    position = np.take_along_axis(
        qd_to_numpy(solver.particles_reordered.pos, transpose=True), reordered_idx[..., None], axis=1
    )
    assert_allclose(
        velocity,
        (((0.75, 1.0, 0.0), (1.0, 1.0, 0.0), (0.5625, 1.0, 0.0)),) * max(n_envs, 1),
        atol=1e-6,
    )
    assert_allclose(position[..., 2, 1], -0.1, atol=1e-6)

    liquid.set_particles_pos((2.5, -2.5, 0.0), particles_idx_local=2)
    liquid.set_particles_vel((0.0, 0.0, 0.0))
    solver.set_static_colliders_pose(
        pos=(0.001, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        colliders_idx=0,
    )
    solver._kernel_reorder_particles(0)
    solver.particles_reordered.dpos.fill(0.0)
    solver.particles_reordered.surface.fill(True)
    solver._kernel_apply_viscosity()
    reordered_idx = qd_to_numpy(solver.particles_ng.reordered_idx, transpose=True)
    velocity = np.take_along_axis(
        qd_to_numpy(solver.particles_reordered.vel, transpose=True), reordered_idx[..., None], axis=1
    )
    assert_allclose(
        velocity,
        (((0.25, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),) * max(n_envs, 1),
        atol=1e-6,
    )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_teapot_initial_particles_pose_and_case_time_steps():
    teapot_settings = case_settings(CASE_TEAPOT).teapot
    particle_size = 0.1
    teapot_mesh = mesh_utils.load_mesh(os.path.join(gs.utils.get_assets_dir(), teapot_settings.asset)).copy()
    teapot_mesh.merge_vertices(merge_tex=True, merge_norm=True)
    teapot_mesh.vertices = geom_utils.transform_by_quat(
        teapot_mesh.vertices * teapot_settings.mesh_scale,
        np.array(teapot_settings.quat),
    ) + np.array(teapot_settings.offset)
    particles = particle_utils.mesh_cavity_to_particles(
        teapot_mesh,
        p_size=particle_size,
        seed=teapot_settings.particles_seed,
        max_height=teapot_settings.particles_max_height,
        clearance=0.5 * particle_size,
    )
    signed_distance, *_ = igl.signed_distance(particles, teapot_mesh.vertices, teapot_mesh.faces)

    assert teapot_settings.asset == "meshes/utah_teapot_modified.obj"
    assert teapot_mesh.is_watertight
    assert_equal(len(particles), 188716)
    assert (signed_distance >= 0.5 * particle_size).all()
    assert particles[:, 1].min() < -3.2
    max_height_steps = math.floor(
        (teapot_settings.particles_max_height - teapot_settings.particles_seed[1]) / particle_size
    )
    expected_max_height = teapot_settings.particles_seed[1] + max_height_steps * particle_size
    assert_allclose(particles[:, 1].max(), expected_max_height, atol=1e-12)
    assert particles[:, 2].max() > 5.0
    for case in CASES:
        settings = case_settings(case)
        expected_scale = 20 if case in (CASE_MOP, CASE_SWEEP, CASE_TAP, CASE_TEAPOT) else 10
        expected_dt = 0.01 if case in (CASE_MOP, CASE_SWEEP, CASE_TEAPOT) else 1.0 / 30.0
        assert_equal(settings.scale, expected_scale)
        assert_equal(settings.dt, expected_dt)
    for time, angle_degrees in ((0.0, 0.0), (18.0, 27.0), (28.0, 27.0), (29.6, 19.0), (35.0, 19.0)):
        pose = teapot_pose(time, teapot_settings)
        angle = math.radians(angle_degrees)
        assert_allclose(
            pose.pos,
            (
                teapot_settings.offset[0],
                teapot_settings.turning_axis_pos[1]
                + (teapot_settings.offset[1] - teapot_settings.turning_axis_pos[1]) * math.cos(angle)
                - (teapot_settings.offset[2] - teapot_settings.turning_axis_pos[2]) * math.sin(angle),
                teapot_settings.turning_axis_pos[2]
                + (teapot_settings.offset[1] - teapot_settings.turning_axis_pos[1]) * math.sin(angle)
                + (teapot_settings.offset[2] - teapot_settings.turning_axis_pos[2]) * math.cos(angle),
            ),
            atol=1e-12,
        )
        expected_quat = geom_utils.transform_quat_by_quat(
            np.array(teapot_settings.quat),
            np.array((math.cos(0.5 * angle), math.sin(0.5 * angle), 0.0, 0.0)),
        )
        assert_allclose(pose.quat, expected_quat, atol=1e-12)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_case_settings():
    settings = case_settings(CASE_TAP)
    assert CASE_TAP in CASES
    assert settings.scale == 20
    assert_equal(settings.dt, 1.0 / 30.0)
    assert_equal(settings.gravity, (0.0, -9.8, 0.0))
    assert_equal(settings.lower_bound, (-500.0, -50.0, -500.0))
    assert_equal(settings.upper_bound, (500.0, 500.0, 500.0))
    assert settings.static_colliders == ()
    assert settings.max_surface_neighbors == 768
    assert settings.max_localmesh_neighbors == 64
    assert settings.enable_pca_normals
    assert settings.steps == 2000
    assert settings.emitter is not None
    assert settings.emitter.max_particles == 200000
    assert_equal(settings.emitter.pos, (0.0, 5.0, 0.0))
    assert_equal(settings.emitter.direction, (0.0, -1.0, 0.0))
    assert_equal(settings.emitter.droplet_size, 2.0)
    assert_equal(settings.emitter.generation_speed, 3.0)
    assert_equal(settings.emitter.initial_speed, 0.0)

    mop_settings = case_settings(CASE_MOP)
    sweep_settings = case_settings(CASE_SWEEP)
    mop = get_wipe_settings(mop_settings)
    sweep = get_wipe_settings(sweep_settings)
    assert CASE_MOP != CASE_SWEEP
    assert CASE_MOP in CASES
    assert CASE_SWEEP in CASES
    assert mop_settings.mop is mop
    assert mop_settings.sweep is None
    assert sweep_settings.mop is None
    assert sweep_settings.sweep is sweep
    assert mop is not sweep
    assert mop._replace(collider_entity_name=sweep.collider_entity_name, mop_manipulator=None) == sweep
    assert mop.mop_manipulator is not None
    assert sweep.mop_manipulator is None
    assert mop.collider_entity_name == "sponge"
    assert sweep.collider_entity_name == "sweep_collider"
    assert isinstance(mop_settings.static_colliders[1], gs.options.PBSTFAbsorbentBoxStaticColliderOptions)
    assert type(sweep_settings.static_colliders[1]) is gs.options.PBSTFBoxStaticColliderOptions
    assert_equal(mop_settings.static_colliders[1].lower, sweep_settings.static_colliders[1].lower)
    assert_equal(mop_settings.static_colliders[1].upper, sweep_settings.static_colliders[1].upper)
    assert_equal(mop_settings.static_colliders[1].pos, sweep_settings.static_colliders[1].pos)
    assert_equal(mop_settings.static_colliders[1].quat, sweep_settings.static_colliders[1].quat)
    assert_equal(mop_settings.static_colliders[1].absorption_rate, 2000.0)
    assert_equal(mop_settings.static_colliders[1].absorption_capacity_fraction, 1.0)
    assert mop_settings.static_colliders[1].fem_entity_name == mop.collider_entity_name
    assert mop_settings.static_colliders[1].sdf_res is None
    assert mop.mop_manipulator.asset == "urdf/panda_bullet/panda.urdf"
    assert mop.mop_manipulator.is_visible
    assert_equal(mop.mop_manipulator.sponge_grid_resolution, (15, 10, 30))
    assert_equal(mop.mop_manipulator.sponge_density, 30.0)
    assert_equal(mop.mop_manipulator.scale, 15.0)
    assert_equal(mop.mop_manipulator.finger_open_qpos, 0.6)
    assert mop.table_entity_name == "wipe_table"
    assert_equal(mop.mop_manipulator.tool_center_point, (0.0, 0.0, 3.02))
    assert_equal(mop.mop_manipulator.finger_closed_qpos, 0.4)
    for time in (0.0, mop.settle_time, mop.settle_time + 0.5 * mop.wipe_time, 20.0):
        assert_equal(wipe_pose(time, mop), wipe_pose(time, sweep))


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_local_mesh_neighbor_capacity():
    options = gs.options.PBSTFOptions(
        max_surface_neighbors=128,
    )
    assert options.max_localmesh_neighbors == 64

    options = gs.options.PBSTFOptions(
        max_surface_neighbors=32,
    )
    assert options.max_localmesh_neighbors == 32

    with pytest.raises(gs.GenesisException, match="must be at most"):
        gs.options.PBSTFOptions(
            max_surface_neighbors=32,
            max_localmesh_neighbors=33,
        )


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
def test_teapot_manipulator_tracks_grasp_pose(show_viewer):
    teapot_settings = case_settings(CASE_TEAPOT).teapot
    manipulator = teapot_settings.manipulator
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            disable_constraint=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=manipulator.camera_pos,
            camera_lookat=manipulator.camera_lookat,
            camera_up=(0.0, 1.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    kuka = scene.add_entity(
        morph=gs.morphs.URDF(
            file=manipulator.kuka_asset,
            scale=manipulator.kuka_scale,
            pos=manipulator.kuka_base_pos,
            quat=manipulator.kuka_base_quat,
            collision=False,
            fixed=True,
        ),
        material=gs.materials.Rigid(
            needs_coup=False,
            gravity_compensation=1.0,
        ),
        name=manipulator.kuka_entity_name,
    )
    hand = scene.add_entity(
        morph=gs.morphs.URDF(
            file=manipulator.hand_asset,
            scale=manipulator.hand_scale,
            collision=False,
        ),
        material=gs.materials.Rigid(
            needs_coup=False,
            gravity_compensation=1.0,
        ),
        name=manipulator.hand_entity_name,
    )
    hand.attach(
        kuka,
        manipulator.kuka_end_effector_link,
        pos=manipulator.hand_mount_pos,
        quat=manipulator.hand_mount_quat,
    )
    scene.build()
    hand.set_qpos(manipulator.hand_qpos)

    assert_equal(kuka.n_dofs, 7)
    assert_equal(hand.n_dofs, 24)
    kuka_limit_lower, kuka_limit_upper = map(tensor_to_array, kuka.get_dofs_limit())
    hand_limit_lower, hand_limit_upper = map(tensor_to_array, hand.get_dofs_limit())
    assert (np.array(manipulator.hand_qpos) >= hand_limit_lower).all()
    assert (np.array(manipulator.hand_qpos) <= hand_limit_upper).all()

    end_effector = kuka.get_link(manipulator.kuka_end_effector_link)
    qpos = kuka.get_qpos()
    times = tuple(angle_twice / 3.0 for angle_twice in range(55)) + tuple(
        28.0 + (54 - angle_twice) / 10.0 for angle_twice in range(53, 37, -1)
    )
    for time in times:
        pose = teapot_pose(time, teapot_settings)
        qpos = update_teapot_manipulator(kuka, pose, teapot_settings, qpos)
        scene.step()

        target_pos, target_quat = geom_utils.transform_pos_quat_by_trans_quat(
            np.array(manipulator.grasp_pos) * teapot_settings.mesh_scale,
            np.array(manipulator.grasp_quat),
            np.array(pose.pos),
            np.array(pose.quat),
        )
        end_effector_quat = tensor_to_array(end_effector.get_quat())
        tool_center_pos = tensor_to_array(end_effector.get_pos()) + geom_utils.transform_by_quat(
            np.array(manipulator.tool_center_point), end_effector_quat
        )
        qpos = kuka.get_qpos()
        qpos_array = tensor_to_array(qpos)
        hand_qpos = tensor_to_array(hand.get_qpos())
        orientation_error = 2.0 * np.arccos(np.minimum(np.abs(np.sum(end_effector_quat * target_quat)), 1.0))

        assert np.linalg.norm(tool_center_pos - target_pos) <= 5e-4
        assert orientation_error <= 5e-3
        assert (qpos_array >= kuka_limit_lower).all()
        assert (qpos_array <= kuka_limit_upper).all()
        assert (hand_qpos >= hand_limit_lower).all()
        assert (hand_qpos <= hand_limit_upper).all()
        assert_allclose(hand_qpos, manipulator.hand_qpos, atol=1e-6)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_mop_sponge_full_simulation_and_collision(n_envs, show_viewer):
    scene, (liquid_entity,) = build_scene(
        case=CASE_MOP,
        scale=5,
        show_viewer=show_viewer,
        dt=0.05,
        n_envs=n_envs,
    )
    settings = case_settings(CASE_MOP).mop
    manipulator = settings.mop_manipulator
    sponge_grid_resolution = manipulator.sponge_grid_resolution
    manipulator_entity = scene.get_entity(name=manipulator.entity_name)
    sponge_entity = scene.get_entity(name=settings.collider_entity_name)
    table_entity = scene.get_entity(name=settings.table_entity_name)
    assert sponge_entity.surface.vis_mode == "tetrahedral"
    assert sponge_entity.surface.opacity is None
    assert isinstance(sponge_entity.morph, gs.morphs.TetrahedralMesh)
    assert sponge_entity.n_vertices == math.prod(resolution + 1 for resolution in sponge_grid_resolution)
    assert sponge_entity.n_elements == 6 * math.prod(sponge_grid_resolution)
    sponge_init_positions = tensor_to_array(sponge_entity.init_positions)
    for axis, resolution in enumerate(sponge_grid_resolution):
        coordinates = np.unique(sponge_init_positions[:, axis])
        assert len(coordinates) == resolution + 1
        assert_allclose(
            np.diff(coordinates),
            (settings.collider_upper[axis] - settings.collider_lower[axis]) / resolution,
            atol=1e-6,
        )
    (sponge_vgeom,) = sponge_entity.vgeoms
    render_surface_triangles = np.sort(sponge_vgeom.sim_verts_idx[sponge_vgeom.vmesh.faces], axis=-1)
    simulation_surface_triangles = np.sort(sponge_entity.surface_triangles, axis=-1)
    sponge_surface_vertices_idx = np.unique(sponge_entity.surface_triangles)
    render_faces_order = np.lexsort(render_surface_triangles.T[::-1])
    simulation_faces_order = np.lexsort(simulation_surface_triangles.T[::-1])
    assert_equal(
        render_surface_triangles[render_faces_order],
        simulation_surface_triangles[simulation_faces_order],
    )
    left_finger_link = manipulator_entity.get_link(manipulator.left_finger_link_name)
    finger_links = (
        left_finger_link,
        manipulator_entity.get_link(manipulator.right_finger_link_name),
    )
    manipulator_collider_links = tuple(link for link in manipulator_entity.links if link.geoms)
    collider_links = manipulator_collider_links + tuple(link for link in table_entity.links if link.geoms)
    assert all(link.geoms for link in finger_links)
    assert any(
        link.name not in (manipulator.left_finger_link_name, manipulator.right_finger_link_name)
        for link in manipulator_collider_links
    )
    assert all(
        geom.needs_coup and not geom.is_coup_reaction_enabled and geom.coup_friction == 0.0
        for link in collider_links
        for geom in link.geoms
    )
    assert all(geom.get_trimesh().is_watertight for link in collider_links for geom in link.geoms)
    assert any(link.vgeoms for link in manipulator_entity.links) == manipulator.is_visible
    assert any(link.vgeoms for link in table_entity.links)
    assert_allclose(scene.fem_options.gravity, (0.0, -9.8, 0.0), atol=1e-12)
    assert_equal(sponge_entity.material.rho, manipulator.sponge_density)
    sponge_x = sponge_init_positions[:, 0]
    sponge_y = sponge_init_positions[:, 1]
    finger_contact_mask = np.isclose(sponge_y, sponge_y.max())
    left_contact_mask = finger_contact_mask & np.isclose(sponge_x, sponge_x.max())
    right_contact_mask = finger_contact_mask & np.isclose(sponge_x, sponge_x.min())
    lower_left_mask = np.isclose(sponge_x, sponge_x.max()) & np.isclose(sponge_y, sponge_y.min())
    lower_right_mask = np.isclose(sponge_x, sponge_x.min()) & np.isclose(sponge_y, sponge_y.min())
    left_finger_geoms = left_finger_link.geoms
    sponge_center_x = 0.5 * (sponge_x.min() + sponge_x.max())
    finger_contact_y = [[] for _ in range(max(n_envs, 1))]
    for geom in left_finger_geoms:
        geom_vertices = tensor_to_array(geom.get_verts())
        if geom_vertices.ndim == 2:
            geom_vertices = geom_vertices[None]
        for env_idx, vertices in enumerate(geom_vertices):
            distances_x = np.abs(vertices[:, 0] - sponge_center_x)
            contact_mask = distances_x <= distances_x.min() + 0.01 * np.ptp(vertices[:, 0])
            finger_contact_y[env_idx].append(vertices[contact_mask, 1])
    finger_contact_y = [np.concatenate(values) for values in finger_contact_y]
    finger_lower_y = min(tensor_to_array(link.get_AABB())[..., 0, 1].min() for link in finger_links)
    qpos = manipulator_entity.get_qpos()
    liquid_entity.set_particles_pos(
        tuple(
            settings.start_pos[axis] + 0.5 * (settings.collider_lower[axis] + settings.collider_upper[axis])
            for axis in range(3)
        ),
        particles_idx_local=0,
    )
    assert not tensor_to_array(scene.pbstf_solver.get_state(0).is_deformable_static_colliders_sdf_active).any()

    for _ in range(round(settings.settle_time / scene.dt)):
        update = update_mop_case(scene, scene.cur_t, settings, qpos)
        qpos = update.qpos
        scene.step()
    update = update_mop_case(scene, scene.cur_t, settings, qpos)
    qpos = update.qpos

    settled_positions = tensor_to_array(sponge_entity.get_state().pos)
    settled_extents = np.ptp(settled_positions, axis=1)
    finger_sdf_cell_sizes = []
    for link in finger_links:
        for geom in link.geoms:
            finger_sdf_cell_sizes.append(np.max(geom.sdf_cell_size))
    contact_inner_width = settled_positions[:, left_contact_mask, 0].min(axis=1) - settled_positions[
        :, right_contact_mask, 0
    ].max(axis=1)
    top_width = settled_positions[:, left_contact_mask, 0].mean(axis=1) - settled_positions[
        :, right_contact_mask, 0
    ].mean(axis=1)
    lower_width = settled_positions[:, lower_left_mask, 0].mean(axis=1) - settled_positions[
        :, lower_right_mask, 0
    ].mean(axis=1)
    assert left_contact_mask.sum() >= 3
    assert right_contact_mask.sum() >= 3
    assert all(values.min() >= sponge_init_positions[:, 1].min() - 1e-3 for values in finger_contact_y)
    assert all(values.max() <= sponge_init_positions[:, 1].max() + 1e-3 for values in finger_contact_y)
    assert finger_lower_y > settings.liquid_upper[1]
    assert (settled_positions[..., 1].min(axis=1) >= -1e-6).all()
    assert (settled_positions[..., 1].min(axis=1) <= 1e-3).all()
    assert_allclose(
        contact_inner_width,
        2.0 * manipulator.finger_closed_qpos,
        atol=2.0 * max(finger_sdf_cell_sizes),
    )
    assert (top_width - contact_inner_width > 0.05).all()
    assert (lower_width - contact_inner_width > 0.05).all()
    assert (settled_extents[:, 0] > 0.95 * (settings.collider_upper[0] - settings.collider_lower[0])).all()
    assert_allclose(qpos[..., -2:], manipulator.finger_closed_qpos, atol=1e-6)
    assert_allclose(manipulator_entity.get_qpos(), qpos, atol=1e-6)
    solver_state = scene.pbstf_solver.get_state(0)
    assert not tensor_to_array(solver_state.is_deformable_static_colliders_sdf_active).any()
    collider_surface_positions = tensor_to_array(solver_state.deformable_static_colliders_surface_vertices)
    sponge_surface_positions_local = geom_utils.inv_transform_by_trans_quat(
        settled_positions[:, sponge_surface_vertices_idx],
        np.array(settings.start_pos),
        np.array(settings.quat),
    )
    assert_allclose(collider_surface_positions, sponge_surface_positions_local, atol=1e-5)
    assert_equal(tensor_to_array(solver_state.absorbed_collider_idx)[:, 0], settings.collider_idx)
    assert tensor_to_array(scene.pbstf_solver.get_static_collider_wetness(settings.collider_idx)).sum() > 0.0
    hand_link = manipulator_entity.get_link(manipulator.hand_link_name)
    settled_state = scene.get_state()

    scene.step()
    update = update_mop_case(scene, scene.cur_t, settings, qpos)
    pre_move_positions = tensor_to_array(sponge_entity.get_state().pos)
    pre_move_hand_pos = np.atleast_2d(tensor_to_array(hand_link.get_pos()))
    pre_move_hand_quat = np.atleast_2d(tensor_to_array(hand_link.get_quat()))
    scene.step()
    moved_positions = tensor_to_array(sponge_entity.get_state().pos)
    hand_pos = np.atleast_2d(tensor_to_array(hand_link.get_pos()))
    hand_quat = np.atleast_2d(tensor_to_array(hand_link.get_quat()))
    sponge_hand_pos = geom_utils.transform_by_quat(
        pre_move_positions - pre_move_hand_pos[:, None, :], geom_utils.inv_quat(pre_move_hand_quat)[:, None, :]
    )
    expected_positions = hand_pos[:, None, :] + geom_utils.transform_by_quat(sponge_hand_pos, hand_quat[:, None, :])
    assert np.linalg.norm(moved_positions - expected_positions, axis=-1).max() > 1e-3
    collider_signed_distances = []
    for link in collider_links:
        for geom in link.geoms:
            geom_vertices = tensor_to_array(geom.get_verts())
            if geom_vertices.ndim == 2:
                geom_vertices = geom_vertices[None]
            for sponge_positions, vertices in zip(moved_positions, geom_vertices):
                signed_distances, *_ = igl.signed_distance(
                    sponge_positions,
                    vertices,
                    geom.get_trimesh().faces,
                )
                collider_signed_distances.append(signed_distances.min())
    assert min(collider_signed_distances) >= -1e-6
    scene.pbstf_solver.update_static_collider_deformation(settings.collider_idx, is_sdf_enabled=False)

    end_update = update_mop_case(
        scene,
        settings.settle_time + settings.wipe_time,
        settings,
        update.qpos,
    )
    end_hand_pos = np.atleast_2d(tensor_to_array(hand_link.get_pos()))
    end_hand_quat = np.atleast_2d(tensor_to_array(hand_link.get_quat()))
    tool_center_offset = np.broadcast_to(np.array(manipulator.tool_center_point), end_hand_pos.shape)
    tool_center_pos = end_hand_pos + geom_utils.transform_by_quat(tool_center_offset, end_hand_quat)
    expected_tool_center = np.array(
        (
            settings.end_pos[0] + 0.5 * (settings.collider_lower[0] + settings.collider_upper[0]),
            settings.end_pos[1] + settings.collider_upper[1],
            settings.end_pos[2] + 0.5 * (settings.collider_lower[2] + settings.collider_upper[2]),
        )
    )
    orientation_similarity = np.sum(end_hand_quat * manipulator.grasp_quat, axis=-1)
    orientation_error = 2.0 * np.arccos(np.minimum(np.abs(orientation_similarity), 1.0))
    assert_equal(end_update.wipe_pos, settings.end_pos)
    assert_allclose(tool_center_pos, expected_tool_center, atol=5e-4)
    assert (orientation_error <= 5e-3).all()

    scene.reset(settled_state)
    assert_allclose(sponge_entity.get_state().pos, settled_positions, atol=1e-6)
    assert not tensor_to_array(scene.pbstf_solver.get_state(0).is_deformable_static_colliders_sdf_active).any()


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_emitter_build_emit_and_wrap(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            gravity=(0.0, 0.0, 0.0),
        ),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=0.2,
            lower_bound=(-2.0, -2.0, -2.0),
            upper_bound=(2.0, 2.0, 2.0),
            max_solver_iterations=1,
            max_surface_neighbors=32,
            enable_pca_normals=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 3.0, 2.0),
            camera_lookat=(0.0, 0.5, 0.0),
        ),
        show_viewer=show_viewer,
    )
    emitter = scene.add_emitter(
        material=gs.materials.PBSTF.Liquid(
            sampler="regular",
        ),
        max_particles=20,
    )
    scene.build(n_envs=n_envs)

    assert not tensor_to_array(emitter.entity.get_particles_active()).any()
    emitter.emit(
        droplet_shape="circle",
        droplet_size=0.6,
        pos=(0.0, 0.5, 0.0),
        direction=(0.0, -1.0, 0.0),
        speed=0.5,
        generation_speed=200.0,
    )
    assert emitter.next_particle == 9
    emitter.emit(
        droplet_shape="circle",
        droplet_size=0.8,
        pos=(0.0, 0.5, 0.0),
        direction=(0.0, -1.0, 0.0),
        speed=0.5,
        generation_speed=200.0,
    )

    active = tensor_to_array(emitter.entity.get_particles_active())
    velocities = tensor_to_array(emitter.entity.get_particles_vel())
    assert emitter.next_particle == 12
    assert (active.sum(axis=-1) == 12).all()
    assert_allclose(velocities[active], (0.0, -0.5, 0.0), atol=1e-6)
    expected_mass = emitter.entity.material.rho * emitter.entity.particle_size**3 / math.sqrt(2.0)
    assert_allclose(tensor_to_array(emitter.entity.get_mass()) / active.sum(axis=-1), expected_mass, rtol=1e-3)

    scene.step()
    positions = tensor_to_array(emitter.entity.get_particles_pos())
    active = tensor_to_array(emitter.entity.get_particles_active())
    assert np.isfinite(positions[active]).all()

    scene.reset()
    assert emitter.next_particle == 0
    assert not tensor_to_array(emitter.entity.get_particles_active()).any()


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
    assert solver.projected_positions.shape[-1] == 128
    assert solver.local_mesh_neighbors.shape[-1] == 64
    assert solver._surface_gradient.shape[-1] == 64

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
    cone_entity = scene.get_entity(name="cone_collider")
    hit_cone = False
    rebounded = False

    assert solver._n_static_colliders == 1
    assert isinstance(solver._static_colliders[0], ConeStaticCollider)
    assert isinstance(cone_entity.morph, gs.morphs.MeshSet)
    assert cone_entity.surface.opacity is None
    assert any(link.vgeoms for link in cone_entity.links)
    assert_allclose(cone_entity.get_pos(), solver._static_colliders[0].center, atol=1e-6)
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
@pytest.mark.parametrize("backend", [gs.cuda])
def test_sweep_box_pushes_water(show_viewer):
    settings = case_settings(CASE_SWEEP)
    sweep = get_wipe_settings(settings)
    assert len(settings.static_colliders) == 2
    assert type(settings.static_colliders[1]) is gs.options.PBSTFBoxStaticColliderOptions

    scene, (liquid,) = build_scene(case=CASE_SWEEP, show_viewer=show_viewer)
    sweep_entity = scene.get_entity(name=sweep.collider_entity_name)
    table_entity = scene.get_entity(name=sweep.table_entity_name)
    positions_initial = tensor_to_array(liquid.get_particles_pos())
    assert isinstance(sweep_entity.morph, gs.morphs.MeshSet)
    assert sweep_entity.surface.opacity is None
    assert any(link.vgeoms for link in sweep_entity.links)
    assert any(link.vgeoms for link in table_entity.links)
    assert liquid.n_particles == 840
    assert liquid.material.collider_friction == 0.5
    with pytest.raises(gs.GenesisException, match="not absorbent"):
        scene.pbstf_solver.get_static_collider_wetness(1)

    for _ in range(5):
        scene.step()
    for step_idx in range(70):
        time = sweep.settle_time + sweep.wipe_time * step_idx / 69
        update_wipe_case(scene.pbstf_solver, sweep_entity, time, sweep)
        scene.step()

    positions = tensor_to_array(liquid.get_particles_pos())
    displacement_x = positions[:, 0] - positions_initial[:, 0]
    is_ahead_of_initial_water = positions[:, 0] > sweep.liquid_upper[0]

    assert np.isfinite(positions).all()
    assert positions[:, 1].min() >= -1e-5
    assert displacement_x.mean() > 3.0
    assert (displacement_x > 0.1).mean() > 0.99
    assert is_ahead_of_initial_water.mean() > 2.0 / 3.0
    assert_allclose(sweep_entity.get_pos(), sweep.end_pos, atol=1e-6)


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
