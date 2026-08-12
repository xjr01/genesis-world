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
    CASE_TAP,
    CASE_TEAPOT,
    CASES,
    build_scene,
    case_settings,
    draw_case_colliders,
    teapot_pose,
    update_mop_case,
    update_teapot_manipulator,
)
import genesis as gs
from genesis.engine.boundaries import (
    BoxStaticCollider,
    ConeStaticCollider,
    StaticCollider,
    project_out_static_collider,
    query_static_collider,
    static_collider_separates,
)
import genesis.utils.geom as geom_utils
import genesis.utils.mesh as mesh_utils
import genesis.utils.particle as particle_utils
from genesis.utils.misc import qd_to_numpy, tensor_to_array
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

    with pytest.raises(gs.GenesisException, match="greater than"):
        gs.options.PBSTFBoxStaticColliderOptions(
            lower=(-1.0, -1.0, -1.0),
            upper=(1.0, -1.0, 1.0),
        )

    with pytest.raises(TypeError):
        StaticCollider()


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_mesh_static_collider_pose(asset_tmp_path, n_envs, show_viewer):
    mesh_path = asset_tmp_path / f"pbstf_static_collider_box_{n_envs}.obj"
    trimesh.creation.box().export(mesh_path)
    target_pos = (1.0, 2.0, 3.0)
    target_quat = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    collider_options = [
        gs.options.PBSTFMeshStaticColliderOptions(
            file=str(mesh_path),
            sdf_res=16,
        ),
        gs.options.PBSTFMeshStaticColliderOptions(
            file=str(mesh_path),
            sdf_res=16,
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
            camera_lookat=(1.0, 2.0, 3.0),
        ),
        show_viewer=show_viewer,
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Particles(
            positions=(target_pos,),
        ),
        material=gs.materials.PBSTF.Liquid(
            sampler="regular",
        ),
    )
    scene.build(n_envs=n_envs)

    scene.pbstf_solver.set_static_colliders_pose(
        pos=target_pos,
        quat=target_quat,
        colliders_idx=1,
        envs_idx=n_envs - 1 if n_envs else None,
    )

    colliders_pos = qd_to_numpy(scene.pbstf_solver._static_colliders_pos, transpose=True)
    colliders_quat = qd_to_numpy(scene.pbstf_solver._static_colliders_quat, transpose=True)
    expected_pos = np.zeros((max(n_envs, 1), 2, 3), dtype=gs.np_float)
    expected_quat = np.zeros((max(n_envs, 1), 2, 4), dtype=gs.np_float)
    expected_quat[..., 0] = 1.0
    expected_pos[-1, 1] = target_pos
    expected_quat[-1, 1] = target_quat

    assert liquid.n_particles == 1
    assert_allclose(colliders_pos, expected_pos, atol=1e-6)
    assert_allclose(colliders_quat, expected_quat, atol=1e-6)

    scene.step()
    particles_pos = tensor_to_array(liquid.get_particles_pos())
    if n_envs:
        assert_equal(particles_pos[:-1, 0], (target_pos,))
        projected_pos = particles_pos[-1, 0]
    else:
        projected_pos = particles_pos[0]
    assert np.linalg.norm(projected_pos - target_pos) > 0.4


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
def test_static_collider_adhesion_and_friction(asset_tmp_path, show_viewer):
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
            ],
        ),
        show_viewer=show_viewer,
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Particles(
            positions=((0.0, -0.05, 0.0), (0.0, -0.15, 0.0)),
        ),
        material=gs.materials.PBSTF.Liquid(
            sampler="regular",
            is_collider_adhesion_friction_enabled=True,
            collider_adhesion_compliance=10.0,
            collider_friction=0.25,
        ),
    )
    scene.build()
    solver = scene.pbstf_solver

    solver._kernel_reorder_particles(0)
    solver.particles_reordered.dpos.fill(0.0)
    solver.on_surface.fill(True)
    solver._kernel_apply_static_collider_adhesion()
    adhesion_delta = qd_to_numpy(solver.particles_reordered.dpos, transpose=True)[0]
    mass = qd_to_numpy(solver.particles_info_reordered.mass, transpose=True)[0]
    denominator = liquid.material.collider_adhesion_compliance / solver._default_mass + 1.0 / mass
    expected_adhesion_delta = np.zeros_like(adhesion_delta)
    expected_adhesion_delta[:, 1] = np.array((-0.05, 0.05)) / denominator / mass
    assert_allclose(adhesion_delta, expected_adhesion_delta, atol=1e-6)

    liquid.set_particles_vel((1.0, 1.0, 0.0))
    solver._kernel_reorder_particles(0)
    solver.particles_reordered.dpos.fill(0.0)
    solver.particles_reordered.surface.fill(True)
    solver._kernel_apply_viscosity()
    velocity = qd_to_numpy(solver.particles_reordered.vel, transpose=True)[0]
    assert_allclose(velocity, ((0.75, 1.0, 0.0), (1.0, 1.0, 0.0)), atol=1e-6)


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
        expected_scale = 20 if case in (CASE_MOP, CASE_TAP, CASE_TEAPOT) else 10
        expected_dt = 0.01 if case in (CASE_MOP, CASE_TEAPOT) else 1.0 / 30.0
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
def test_tap_case_settings():
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
@pytest.mark.parametrize("backend", [gs.cuda])
def test_mop_box_pushes_water(show_viewer):
    settings = case_settings(CASE_MOP)
    assert settings.mop is not None
    mop = settings.mop
    mop_mesh = mesh_utils.load_mesh(os.path.join(gs.utils.get_assets_dir(), mop.asset))

    assert mop_mesh.is_watertight
    assert mop_mesh.is_winding_consistent
    assert mop_mesh.volume > 0.0
    assert_allclose(mop_mesh.bounds, ((-0.6, 0.02, -2.2), (0.6, 0.85, 2.2)), atol=1e-8)
    assert_allclose(mop_mesh.bounds, (mop.collider_lower, mop.collider_upper), atol=1e-8)
    assert len(settings.static_colliders) == 2
    assert all(
        isinstance(collider_options, gs.options.PBSTFBoxStaticColliderOptions)
        for collider_options in settings.static_colliders
    )

    scene, (liquid,) = build_scene(case=CASE_MOP, show_viewer=show_viewer)
    mop_debug_object = draw_case_colliders(scene, CASE_MOP)
    mop_debug_transform = geom_utils.trans_quat_to_T(np.array(mop.start_pos), np.array(mop.quat))
    scene.update_debug_objects((mop_debug_object,), (mop_debug_transform,))
    positions_initial = tensor_to_array(liquid.get_particles_pos())
    assert liquid.n_particles == 6000
    assert liquid.material.collider_friction == 0.1

    for _ in range(5):
        scene.step()
    for step_idx in range(70):
        time = mop.settle_time + mop.wipe_time * step_idx / 69
        update_mop_case(scene.pbstf_solver, time, mop)
        scene.step()

    positions = tensor_to_array(liquid.get_particles_pos())
    displacement_x = positions[:, 0] - positions_initial[:, 0]
    is_ahead_of_initial_water = positions[:, 0] > mop.liquid_upper[0]

    assert np.isfinite(positions).all()
    assert positions[:, 1].min() >= -1e-5
    assert displacement_x.mean() > 3.0
    assert (displacement_x > 0.1).mean() > 0.99
    assert is_ahead_of_initial_water.mean() > 2.0 / 3.0


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
