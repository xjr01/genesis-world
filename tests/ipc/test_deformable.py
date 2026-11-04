import math
from contextlib import nullcontext
from typing import TYPE_CHECKING, cast, Any

import numpy as np
import pytest

try:
    import uipc
except ImportError:
    pytest.skip("IPC Coupler is not supported because 'uipc' module is not available.", allow_module_level=True)


import genesis as gs
from genesis.utils.misc import tensor_to_array, qd_to_numpy

from ..conftest import TOL_SINGLE
from ..utils import assert_allclose, get_hf_dataset
from .utils import (
    find_ipc_geometries,
    get_ipc_positions,
    get_ipc_rigid_links_idx,
)

if TYPE_CHECKING:
    from genesis.engine.couplers import IPCCoupler


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.parametrize("coup_type", ["two_way_soft_constraint", "external_articulation"])
def test_robot_grasp_fem(coup_type, show_viewer):
    DT = 0.01
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)
    BOX_POS = (0.65, 0.0, 0.03)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            constraint_strength_translation=10.0,
            constraint_strength_rotation=10.0,
            newton_translation_tolerance=10.0,
            enable_rigid_rigid_contact=False,
            enable_rigid_ground_contact=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 1.0, 1.0),
            camera_lookat=(0.3, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_type="ipc_only",
            coup_friction=0.8,
        ),
    )

    material_kwargs: dict[str, Any] = dict(
        coup_friction=0.8,
        coup_type=coup_type,
    )
    if coup_type == "two_way_soft_constraint":
        material_kwargs["coup_links"] = ("left_finger", "right_finger")

    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda_non_overlap.xml",
        ),
        material=gs.materials.Rigid(**material_kwargs),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=BOX_POS,
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.FEM.Elastic(
            E=5.0e4,
            nu=0.45,
            rho=1000.0,
            friction_mu=0.5,
            model="stable_neohookean",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.8, 0.2, 0.5),
        ),
    )

    scene.build()
    assert scene.sim is not None
    coupler = cast("IPCCoupler", scene.sim.coupler)

    envs_idx = range(max(scene.n_envs, 1))
    motors_dof, fingers_dof = slice(0, 7), slice(7, 9)
    # end_effector = franka.get_link("hand")

    franka.set_dofs_kp([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 500.0, 500.0])

    box_entity_idx = scene.sim.fem_solver.entities.index(box)
    assert len(find_ipc_geometries(scene, solver_type="fem", idx=box_entity_idx, env_idx=0)) == 1

    franka_finger_links = {franka.get_link(name) for name in ("left_finger", "right_finger")}
    franka_finger_links_idx = {link.idx for link in franka_finger_links}
    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert franka_finger_links_idx.issubset(ipc_links_idx)
    for link_idx in franka_finger_links:
        assert link_idx in coupler._abd_slots_by_link

    franka_links_idx = {link.idx for link in franka.links}
    franka_ipc_links_idx = franka_links_idx.intersection(ipc_links_idx)
    if coup_type == "two_way_soft_constraint":
        assert coupler._coup_links.get(franka) == franka_finger_links
        assert franka_ipc_links_idx == franka_finger_links_idx
    else:
        assert franka_finger_links_idx.issubset(franka_ipc_links_idx)

    ipc_positions_0 = get_ipc_positions(scene, solver_type="fem", idx=box_entity_idx, envs_idx=envs_idx)
    gs_positions_0 = tensor_to_array(box.get_state().pos)
    assert_allclose(ipc_positions_0, gs_positions_0, atol=TOL_SINGLE)
    gs_centroid_0 = gs_positions_0.mean(axis=1)
    assert_allclose(gs_centroid_0, BOX_POS, atol=1e-4)

    def run_stage(target_qpos, finger_pos, duration):
        franka.control_dofs_position(target_qpos[motors_dof], motors_dof)
        franka.control_dofs_position(finger_pos, fingers_dof)
        for _ in range(int(duration / DT)):
            scene.step()

    # Setting initial configuration is not supported by coupling mode "external_articulation"
    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.4], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.9482, 0.6910, 1.2114, -1.6619, -0.6739, 1.8685, 1.1844, 0.0112, 0.0096]
    with pytest.raises(gs.GenesisException) if coup_type == "external_articulation" else nullcontext():
        franka.set_dofs_position(qpos)
        franka.control_dofs_position(qpos)
    if coup_type == "external_articulation":
        run_stage(qpos, finger_pos=0.04, duration=2.0)

    # Lower the grapper half way to grasping position
    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.25], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.8757, 0.8824, 1.0523, -1.7619, -0.8831, 2.0903, 1.2924, 0.0400, 0.0400]
    run_stage(qpos, finger_pos=0.04, duration=1.0)

    # Reach grasping position
    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.135], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.7711, 1.0502, 0.8850, -1.7182, -1.0210, 2.2350, 1.3489, 0.0400, 0.0400]
    run_stage(qpos, finger_pos=0.04, duration=0.5)

    # Grasp the cube
    run_stage(qpos, finger_pos=0.0, duration=0.1)

    # Lift the cube
    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.4], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.9488, 0.6916, 1.2123, -1.6627, -0.6750, 1.8683, 1.1855, 0.0301, 0.0319]
    run_stage(qpos, finger_pos=0.0, duration=0.5)

    ipc_positions_f = get_ipc_positions(scene, solver_type="fem", idx=box_entity_idx, envs_idx=envs_idx)
    gs_positions_f = tensor_to_array(box.get_state().pos)
    assert_allclose(ipc_positions_f, gs_positions_f, atol=TOL_SINGLE)
    assert (gs_positions_f[..., 2] - gs_positions_0[..., 2] >= 0.2).all()
    finger_aabb = tensor_to_array(franka.get_link("right_finger").get_AABB())
    assert (gs_positions_f[..., 2] - finger_aabb[..., 0, 2] > 0).any()


@pytest.mark.slow  # ~400s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_cloth_corner_drag(n_envs, show_viewer):
    DT = 0.01
    CLOTH_HALF = 0.5
    BOX_SIZE = 0.05
    GAP = 0.005
    SCALE = 0.5
    FREQ = 0.7

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -9.8),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_enable=True,
            enable_rigid_rigid_contact=True,
            contact_d_hat=GAP,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0 - CLOTH_HALF, -0.5, 1.0 - CLOTH_HALF),
            camera_lookat=(-CLOTH_HALF, 0.0, -CLOTH_HALF),
        ),
        show_viewer=show_viewer,
    )

    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=2 * CLOTH_HALF,
            pos=(-CLOTH_HALF, 0.0, -CLOTH_HALF),
        ),
        material=gs.materials.FEM.Cloth(
            E=1e4,
            nu=0.3,
            rho=200.0,
            thickness=0.001,
            bending_stiffness=None,
            friction_mu=0.8,
        ),
    )

    # Sandwich grip at one corner
    boxes = []
    for z_sign in (+1, -1):
        box = scene.add_entity(
            gs.morphs.Box(
                size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
                pos=(-BOX_SIZE, z_sign * (0.5 * BOX_SIZE + GAP), -BOX_SIZE),
            ),
            material=gs.materials.Rigid(
                coup_type="two_way_soft_constraint",
                coup_friction=0.8,
            ),
            surface=gs.surfaces.Plastic(
                color=(1.0, 0.0, 0.0, 1.0) if z_sign > 0 else (0.0, 1.0, 0.0, 1.0),
            ),
        )
        boxes.append(box)

    scene.build(n_envs=n_envs)
    assert scene.sim is not None

    # Close gap, hold position during settling
    for box in boxes:
        box.set_dofs_kp(2000.0)
        box.set_dofs_kv(400.0)
        init_dof = tensor_to_array(box.get_dofs_position())
        init_dof[..., 1] = 0.0
        box.control_dofs_position(init_dof)

    # Find corner vertices: closest to the gripped corner
    cloth_positions = tensor_to_array(cloth.get_state().pos)
    corner_idx = np.argmin(np.linalg.norm(cloth_positions, axis=-1), axis=-1)

    # Settle: let cloth conform to grip
    for _ in range(40):
        scene.step()

    # Make sure that the cloth did not fall
    cloth_pos = cloth.get_state().pos[range(scene.sim._B), corner_idx]
    assert_allclose(cloth_pos, 0.0, tol=5e-3)

    # Drag phase
    for i in range(int(1.0 / (DT * FREQ))):
        theta = (2.0 * np.pi * FREQ) * (i * scene.sim.dt)
        x = SCALE / math.sqrt(2.0) * (np.cos(theta) - 1.0)
        dx = -SCALE * math.sqrt(2.0) * np.pi * FREQ * np.sin(theta)
        y = SCALE / math.sqrt(2.0) * np.sin(theta)
        dy = SCALE * math.sqrt(2.0) * np.pi * FREQ * np.cos(theta)
        z = SCALE / math.sqrt(2.0) * (np.cos(theta) - 1.0)
        dz = -SCALE * math.sqrt(2.0) * np.pi * FREQ * np.sin(theta)
        for box in boxes:
            box.control_dofs_position_velocity(
                (x - BOX_SIZE, y, z - BOX_SIZE), (dx, dy, dz), dofs_idx_local=slice(0, 3)
            )
        scene.step()
        assert_allclose(cloth.get_state().pos[range(scene.sim._B), corner_idx], (x, y, z), tol=0.01)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("E, nu, strech_scale", [(1e4, 0.3, 1.0), (5e4, 0.49, 0.3)])
def test_cloth_uniform_biaxial_stretching(E, nu, strech_scale, n_envs, show_viewer):
    CLOTH_HALF = 0.5
    BOX_SIZE = 0.05
    GAP = 0.005
    THICKNESS = 0.001
    STRETCH_RATIO_1 = 1.0 + strech_scale * 0.15
    STRETCH_RATIO_2 = 1.4
    PULL_DISTANCE = 0.03  # Radial displacement per corner

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, 0.0),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_enable=True,
            enable_rigid_rigid_contact=True,
            contact_d_hat=GAP,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )

    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=2 * CLOTH_HALF,
            pos=(0.0, 0.0, 0.0),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=E,
            nu=nu,
            rho=200.0,
            thickness=THICKNESS,
            bending_stiffness=None,
            friction_mu=0.8,
        ),
    )

    # 8 boxes: 2 per corner (sandwich grip above/below cloth)
    boxes = []
    for x_sign, y_sign in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        for z_sign in (+1, -1):
            box = scene.add_entity(
                gs.morphs.Box(
                    size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
                    pos=(
                        x_sign * (CLOTH_HALF - BOX_SIZE),
                        y_sign * (CLOTH_HALF - BOX_SIZE),
                        z_sign * (0.5 * BOX_SIZE + GAP),
                    ),
                ),
                material=gs.materials.Rigid(
                    rho=200.0,
                    coup_type="two_way_soft_constraint",
                    coup_friction=0.8,
                ),
                surface=gs.surfaces.Plastic(
                    color=np.random.rand(3),
                ),
            )
            boxes.append(box)

    scene.build(n_envs=n_envs)

    # Configure PD: position-controlled outward pull on x,y; hold z + rotation
    for box in boxes:
        box.set_dofs_kp(2000.0)
        box.set_dofs_kv(500.0)
        init_dof = tensor_to_array(box.get_dofs_position())
        init_dof[..., 2] = 0.0
        box.control_dofs_position(init_dof)

    # Wait for steady state
    cloth_positions_0 = tensor_to_array(cloth.get_state().pos)
    for _ in range(20):
        scene.step()
    cloth_positions_f = tensor_to_array(cloth.get_state().pos)
    assert_allclose(cloth_positions_f, cloth_positions_0, atol=0.005)
    assert_allclose(cloth_positions_f[..., 2], cloth_positions_0[..., 2], tol=5e-3)

    # Stretch: phase one
    for box in boxes:
        init_dof = tensor_to_array(box.get_dofs_position())
        init_dof[..., :2] *= STRETCH_RATIO_1
        box.control_dofs_position(init_dof)
    for _ in range(80):
        scene.step()
    cloth_positions_f = tensor_to_array(cloth.get_state().pos)
    for box in boxes:
        init_dof = tensor_to_array(box.get_dofs_position())
        dist_vertices = np.linalg.norm(cloth_positions_f[..., :2] - init_dof[..., None, :2], axis=-1).min(axis=-1)
        assert_allclose(dist_vertices, 0.0, atol=0.02)
    assert_allclose(cloth_positions_f[..., 2], cloth_positions_0[..., 2], tol=5e-3)

    # Extract X/Y forces while making sure observed forces are consistent
    box_forces_xy = []
    applied_forces = qd_to_numpy(scene.rigid_solver.dyn_state.dofs.qf_applied, None, transpose=True)
    for box in boxes:
        dofs_idx = slice(box.dof_start, box.dof_end)
        box_forces = applied_forces[..., dofs_idx]
        assert_allclose(box_forces[..., 3:], 0.0, tol=0.02)
        assert_allclose(np.abs(box_forces[..., 0]), np.abs(box_forces[..., 1]), tol=0.02)
        box_forces_xy.append(box_forces[..., :2])

    # Check that deformation is roughly symmetric (sanity check)
    grid = cloth_positions_f.reshape((-1, 20, 20, 3))
    grid_flipped_x = np.flip(grid, axis=-3)
    assert_allclose(grid[..., 0], grid_flipped_x[..., 0], atol=0.01)
    assert_allclose(grid[..., 1], -grid_flipped_x[..., 1], atol=0.01)
    grid_flipped_y = np.flip(grid, axis=-2)
    assert_allclose(grid[..., 0], -grid_flipped_y[..., 0], atol=0.01)
    assert_allclose(grid[..., 1], grid_flipped_y[..., 1], atol=0.01)

    # Check that deformation is consistent with applied forces based on material properties.
    # Each corner bears the load from half the reference edge length (by symmetry,
    # 2 corners per edge). Use reference length since stress is in reference config.
    strain_GL = 0.5 * (STRETCH_RATIO_1**2 - 1.0)  # Green–Lagrange strain
    expected_stress = E * strain_GL / (1.0 - nu)  # Equal biaxial plane stress (2nd Piola–Kirchhoff)
    expected_force_per_box = expected_stress * THICKNESS * CLOTH_HALF
    # FIXME: The estimated force is not very accurate. Is it possible to do better?
    assert_allclose(np.abs(box_forces_xy), expected_force_per_box, tol=1e4 / E)

    # Stretch: phase two
    for box in boxes:
        init_dof = tensor_to_array(box.get_dofs_position())
        init_dof[..., :2] *= STRETCH_RATIO_2
        box.control_dofs_position(init_dof)
    for _ in range(50):
        scene.step()

    # Lost grip
    cloth_positions_f = tensor_to_array(cloth.get_state().pos)
    cloth_aabb_min, cloth_aabb_max = cloth_positions_f.min(axis=-2), cloth_positions_f.max(axis=-2)
    cloth_aabb_extent = cloth_aabb_max - cloth_aabb_min
    assert (cloth_aabb_extent[..., :2] < STRETCH_RATIO_1 * (2.0 * CLOTH_HALF)).all()
    assert ((0.001 < cloth_aabb_extent[..., 2]) & (cloth_aabb_extent[..., 2] < 0.2)).all()


@pytest.fixture(scope="session")
def ipc_state_cloth_paths(asset_tmp_path):
    quad = asset_tmp_path / "ipc_state_quad.obj"
    quad.write_text("v 0 0 0\nv 0.1 0 0\nv 0.1 0.1 0\nv 0 0.1 0\nf 1 2 3\nf 1 3 4\n")
    triangle = asset_tmp_path / "ipc_state_triangle.obj"
    triangle.write_text("v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nf 1 2 3\n")
    return str(quad), str(triangle)


@pytest.mark.required
@pytest.mark.precision(str(uipc.build_info()["float_bytes"] * 8))
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("substeps", [1, 2])
def test_cloth_position_velocity(n_envs, substeps, ipc_state_cloth_paths, show_viewer):
    dt = 0.01
    gravity = (0.0, 0.0, -2.0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=substeps,
            substeps_local=4,
            gravity=gravity,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            # Resolve the linear solve accurately enough to test ballistic state transfer.
            linear_system_tolerance=1e-12,
            contact_enable=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, -0.7, 1.5),
            camera_lookat=(0.2, 0.05, 1.0),
        ),
        show_viewer=show_viewer,
    )
    cloths = []
    for i_e, path in enumerate(ipc_state_cloth_paths):
        cloths.append(
            scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=path,
                    pos=(0.3 * i_e, 0.0, 1.0),
                    decimate=False,
                ),
                material=gs.materials.FEM.Cloth(
                    E=1000.0,
                    nu=0.25,
                ),
            )
        )
    scene.build(n_envs=n_envs)
    initial = [tensor_to_array(cloth.get_state().pos).copy() for cloth in cloths]
    velocities = []
    for i_e, cloth in enumerate(cloths):
        velocity = np.zeros_like(initial[i_e])
        velocity[..., 0] = np.arange(1, max(n_envs, 1) + 1)[:, None] * (0.2 + 0.1 * i_e)
        velocities.append(velocity)
        cloth.set_velocity(velocity)
    acceleration = np.array(gravity)
    substep_dt = dt / substeps
    # Unit-scale positions round at O(eps); velocity reconstruction divides by h.
    position_tolerance = 8 * np.finfo(gs.np_float).eps if gs.np_float == np.float32 else 1e-10
    velocity_tolerance = position_tolerance / substep_dt if gs.np_float == np.float32 else 1e-10
    for step in range(1, 3):
        scene.step()
        n = step * substeps
        for i_e, cloth in enumerate(cloths):
            state = cloth.get_state()
            assert_allclose(state.vel, velocities[i_e] + acceleration * dt * step, atol=velocity_tolerance, rtol=0.0)
            displacement = velocities[i_e] * dt * step + acceleration * substep_dt**2 * n * (n + 1) / 2
            assert_allclose(state.pos, initial[i_e] + displacement, atol=position_tolerance, rtol=0.0)

    target = initial[0] + (0.03, 0.02, 0.1)
    cloths[0].set_position(target)
    scene.step()
    state = cloths[0].get_state()
    previous_velocity = velocities[0] + acceleration * dt * 2
    assert_allclose(state.vel, previous_velocity + acceleration * dt, atol=velocity_tolerance, rtol=0.0)
    displacement = previous_velocity * dt + acceleration * substep_dt**2 * substeps * (substeps + 1) / 2
    assert_allclose(state.pos, target + displacement, atol=position_tolerance, rtol=0.0)

    cloths[0].set_position(initial[0])
    cloths[0].set_velocity(velocities[0])
    scene.step()
    state = cloths[0].get_state()
    assert_allclose(state.vel, velocities[0] + acceleration * dt, atol=velocity_tolerance, rtol=0.0)
    displacement = velocities[0] * dt + acceleration * substep_dt**2 * substeps * (substeps + 1) / 2
    assert_allclose(state.pos, initial[0] + displacement, atol=position_tolerance, rtol=0.0)
    state = cloths[1].get_state()
    n = 4 * substeps
    assert_allclose(state.vel, velocities[1] + acceleration * dt * 4, atol=velocity_tolerance, rtol=0.0)
    displacement = velocities[1] * dt * 4 + acceleration * substep_dt**2 * n * (n + 1) / 2
    assert_allclose(state.pos, initial[1] + displacement, atol=position_tolerance, rtol=0.0)

    scene.reset()
    for i_e, cloth in enumerate(cloths):
        state = cloth.get_state()
        assert_allclose(state.pos, initial[i_e], atol=position_tolerance, rtol=0.0)
        assert_allclose(state.vel, 0.0, atol=velocity_tolerance, rtol=0.0)
        cloth.set_velocity(velocities[i_e])
    scene.step()
    for i_e, cloth in enumerate(cloths):
        state = cloth.get_state()
        assert_allclose(state.vel, velocities[i_e] + acceleration * dt, atol=velocity_tolerance, rtol=0.0)
        displacement = velocities[i_e] * dt + acceleration * substep_dt**2 * substeps * (substeps + 1) / 2
        assert_allclose(state.pos, initial[i_e] + displacement, atol=position_tolerance, rtol=0.0)


@pytest.mark.required
@pytest.mark.precision(str(uipc.build_info()["float_bytes"] * 8))
@pytest.mark.parametrize("n_envs", [0, 2])
def test_volume_frame_state(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps_local=4,
            gravity=(0.0, 0.0, 0.0),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_enable=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, -0.5, 1.4),
            camera_lookat=(0.0, 0.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 1.0),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=1000.0,
            nu=0.25,
        ),
    )
    scene.build(n_envs=n_envs)
    initial = tensor_to_array(entity.get_state().pos).copy()
    tolerance = 8 * np.finfo(gs.np_float).eps if gs.np_float == np.float32 else 1e-10
    assert entity.n_elements > 0
    for _ in range(4):
        scene.step()
        state = entity.get_state()
        assert (state.active == gs.ACTIVE).all()
        assert_allclose(state.pos, initial, atol=tolerance, rtol=0.0)
        assert_allclose(state.vel, 0.0, atol=tolerance, rtol=0.0)
