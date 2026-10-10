from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import igl
import trimesh

from examples import pbstf_coffee_water

import genesis as gs
from genesis.utils import geom
from genesis.utils.misc import tensor_to_array

from ..utils import assert_allclose, assert_equal, get_hf_dataset


@pytest.mark.required
@pytest.mark.slow("gpu")
@pytest.mark.parametrize("backend", [gs.cuda])
def test_dual_arm_manipulation(show_viewer):
    demo = pbstf_coffee_water.build_scene(is_viewer_shown=show_viewer, is_motion_only=True)
    lower, upper = map(tensor_to_array, demo.robot.get_dofs_limit())
    rim = demo.cup_cavity.bounds[1, 1]
    phases = set()
    stir_positions = []
    max_error = 0.0
    previous_openings = np.array((0.044, 0.0007))
    previous_opening_velocity = np.zeros(2)
    observation = pbstf_coffee_water.observe_scene(demo)
    previous_positions = np.stack([hand.pos for hand in observation.hands])
    previous_quaternions = np.stack([hand.quat for hand in observation.hands])
    previous_velocity = np.zeros((2, 3))
    previous_angular_velocity = np.zeros((2, 3))
    has_parallel_start = False
    reach_start = np.array(pbstf_coffee_water.WATER_CUP_POS) + (0.0, pbstf_coffee_water.CUP_GRIP_HEIGHT, 0.0)
    reach_direction = np.array(pbstf_coffee_water.SPONGE_START) - reach_start
    reach_direction[1] = 0.0
    reach_direction /= np.linalg.norm(reach_direction)
    cup_vertices = tensor_to_array(demo.water_cup.get_verts()) - pbstf_coffee_water.WATER_CUP_POS
    # The contact-driven sequence needs the full pouring, recovery and wiping horizons in one scene.
    for step in range(round(pbstf_coffee_water.MOTION_END / pbstf_coffee_water.CONTROL_DT)):
        time = (step + 1) * pbstf_coffee_water.CONTROL_DT
        if demo.motion.right_phase == pbstf_coffee_water.Phase.REACH_SPONGE and observation.cup_tilt >= 38.0:
            for water_in_cup in (None, 10, 9):
                recovery_target = pbstf_coffee_water.motion_target(
                    time,
                    replace(demo.motion, water_before_reach=10),
                    replace(observation, water_in_cup=water_in_cup),
                )
                assert_equal(recovery_target.right_phase, pbstf_coffee_water.Phase.CATCH)
        target, error = pbstf_coffee_water.update_motion(demo, time, observation)
        positions = np.stack((target.right.pos, target.left.pos))
        quaternions = np.stack((target.right.quat, target.left.quat))
        velocity = (positions - previous_positions) / pbstf_coffee_water.CONTROL_DT
        acceleration = (velocity - previous_velocity) / pbstf_coffee_water.CONTROL_DT
        rotation = geom.transform_quat_by_quat(geom.inv_quat(previous_quaternions), quaternions)
        angular_velocity = geom.quat_to_rotvec(rotation) / pbstf_coffee_water.CONTROL_DT
        angular_acceleration = (angular_velocity - previous_angular_velocity) / pbstf_coffee_water.CONTROL_DT
        assert np.linalg.norm(velocity, axis=-1).max() < 0.8
        assert np.linalg.norm(acceleration, axis=-1).max() < 20.0
        assert np.linalg.norm(angular_velocity, axis=-1).max() < 6.0
        assert np.linalg.norm(angular_acceleration, axis=-1).max() < 50.0
        previous_positions = positions
        previous_quaternions = quaternions
        previous_velocity = velocity
        previous_angular_velocity = angular_velocity
        openings = np.array((target.right_opening, target.left_opening))
        opening_velocity = (openings - previous_openings) / pbstf_coffee_water.CONTROL_DT
        opening_acceleration = (opening_velocity - previous_opening_velocity) / pbstf_coffee_water.CONTROL_DT
        assert np.abs(opening_velocity).max() < 0.1
        assert np.abs(opening_acceleration).max() < 2.0
        previous_openings = openings
        previous_opening_velocity = opening_velocity
        pbstf_coffee_water.step_scene(demo)
        observation = pbstf_coffee_water.observe_scene(demo)
        cup_min_y = (cup_vertices @ geom.quat_to_R(observation.cup.quat)[1]).min() + observation.cup.pos[1]
        assert cup_min_y > pbstf_coffee_water.TABLE_Y - 0.002
        pbstf_coffee_water.check_contacts(demo, time)
        demo.scene.rigid_solver.check_errno()
        qpos = tensor_to_array(demo.qpos)
        assert ((lower <= qpos) & (qpos <= upper)).all()
        max_error = max(error, max_error)
        if target.right_phase not in phases or target.left_phase not in phases:
            gs.logger.info(f"{time:.3f}s: right={target.right_phase.name}, left={target.left_phase.name}")
        phases.update((target.right_phase, target.left_phase))
        measurement = pbstf_coffee_water.measure_motion(time, target, observation)
        assert measurement.hands_position_error[1] < 0.005
        if target.right_phase not in (pbstf_coffee_water.Phase.REACH_SPONGE, pbstf_coffee_water.Phase.CATCH):
            assert measurement.hands_position_error[0] < 0.005
        if target.right_phase == pbstf_coffee_water.Phase.REACH_SPONGE:
            displacement = target.right.pos - reach_start
            assert np.linalg.norm(np.cross(displacement, reach_direction)) < 1e-6
            assert displacement @ reach_direction >= 0.0
            assert np.linalg.norm(np.cross(observation.hands[0].pos - reach_start, reach_direction)) < 0.005
            assert_allclose(target.right.quat, pbstf_coffee_water.POUR_QUAT, atol=1e-6)
            assert_allclose(target.right_opening, 0.044, atol=1e-6)
        if time <= 6.0:
            assert measurement.cup_position_error < 0.002
            assert measurement.cup_rotation_error < 2.0
        if 1.0 <= time < 6.0:
            assert observation.is_cup_grasped
        if target.right_phase in (pbstf_coffee_water.Phase.UPRIGHT, pbstf_coffee_water.Phase.PLACE_CUP):
            assert observation.is_cup_grasped
        if time == pbstf_coffee_water.PARALLEL_START:
            assert_equal(
                (target.right_phase, target.left_phase),
                (pbstf_coffee_water.Phase.REACH_SPONGE, pbstf_coffee_water.Phase.LIFT_ROD),
            )
            has_parallel_start = True
        if step + 1 == round((pbstf_coffee_water.PARALLEL_START + 0.4) / pbstf_coffee_water.CONTROL_DT):
            assert observation.hands[0].pos[0] < pbstf_coffee_water.WATER_CUP_POS[0] - 0.01
            assert observation.hands[0].pos[2] < -0.02
            assert observation.rod.pos[1] > pbstf_coffee_water.ROD_PARK[1] + 0.04
        if target.left_phase in (
            pbstf_coffee_water.Phase.INSERT_ROD,
            pbstf_coffee_water.Phase.STIR,
            pbstf_coffee_water.Phase.HOLD_ROD,
            pbstf_coffee_water.Phase.WITHDRAW_ROD,
        ) and step % 10 == 0:
            vertices = tensor_to_array(demo.rod.get_verts()) - pbstf_coffee_water.COFFEE_CUP_POS
            is_below_rim = vertices[:, 1] < rim
            if is_below_rim.any():
                assert demo.cup_cavity.contains(vertices[is_below_rim]).all()
        if target.left_phase == pbstf_coffee_water.Phase.STIR:
            stir_positions.append(observation.rod.pos)
        if time > pbstf_coffee_water.PARALLEL_START and (
            target.right_phase == target.left_phase == pbstf_coffee_water.Phase.REST
        ):
            break
    assert max_error < 2e-4
    assert has_parallel_start
    assert demo.motion.has_caught_cup
    assert 50.0 <= demo.motion.max_knock_tilt <= 65.0
    assert_equal(sorted(phases), list(pbstf_coffee_water.Phase))
    stir_positions = np.stack(stir_positions) - pbstf_coffee_water.COFFEE_CUP_POS
    angles = np.unwrap(np.arctan2(stir_positions[:, 2], stir_positions[:, 0]))
    gs.logger.info(
        f"Measured stirring angle: {angles[-1] - angles[0]:.6f}rad; parked rod: {observation.rod.pos}; "
        f"returned cup: {observation.cup.pos}; sponge: {observation.sponge_pos}."
    )
    assert_allclose(angles[-1] - angles[0], 6.0 * np.pi, atol=0.05)
    assert_allclose(observation.rod.pos, pbstf_coffee_water.ROD_PARK, atol=0.002)
    assert_allclose(observation.cup.pos, pbstf_coffee_water.WATER_CUP_POS, atol=0.002)
    assert observation.cup_tilt < 2.0
    assert_allclose(
        demo.sponge.get_pos(),
        np.array(pbstf_coffee_water.SPONGE_END) + (0.0, 0.5 * pbstf_coffee_water.SPONGE_SIZE[1], 0.0),
        atol=0.005,
    )
    torso = trimesh.load(Path(gs.utils.get_assets_dir()) / "urdf/sim1_acone/meshes/body4.STL")
    torso_pos = np.array(pbstf_coffee_water.ROBOT_POS) + geom.transform_by_quat(
        np.array((0.094118, 0.0010067, 0.77)), np.array(pbstf_coffee_water.ROBOT_QUAT)
    )
    vertices = np.concatenate(
        [
            tensor_to_array(visual_geom.get_vverts())
            for idx in range(23, 29)
            for visual_geom in demo.robot.get_link(f"right_link{idx}").vgeoms
        ]
    )
    local_vertices = geom.inv_transform_by_trans_quat(vertices, torso_pos, np.array(pbstf_coffee_water.ROBOT_QUAT))
    distances, *_ = igl.signed_distance(local_vertices, torso.vertices, torso.faces)
    assert distances.min() > 0.005


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_pick_and_place(mode, show_viewer):
    # Add DoF armature to improve numerical stability if not using 'approximate_implicitfast' integrator.
    #
    # This is necessary because the first-order correction term involved in the implicit integration schemes
    # 'implicitfast' and 'Euler' are only able to stabilize each entity independently, from the forces that were
    # obtained from the instable accelerations. As a result, everything is fine as long as the entities are not
    # interacting with each other, but it induces unrealistic motion otherwise. In this case, the acceleration of the
    # cube being lifted is based on the acceleration that the gripper would have without implicit damping.
    #
    # The only way to correct this would be to take into account the derivative of the Jacobian of the constraints in
    # the first-order correction term. Doing this is challenging and would significantly increase the computation cost.
    #
    # In practice, it is more common to just go for a higher order integrator such as RK4.
    if mode == 0:
        integrator = gs.integrator.approximate_implicitfast
        substeps = 1
        armature = 0.0
    elif mode == 1:
        integrator = gs.integrator.implicitfast
        substeps = 4
        armature = 0.0
    elif mode == 2:
        integrator = gs.integrator.Euler
        substeps = 1
        armature = 2.0

    # Create and build the scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=substeps,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
            integrator=integrator,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.65, 0.0, 0.025),
        ),
        surface=gs.surfaces.Plastic(color=(1, 0, 0)),
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.4, 0.2, 0.025),
            fixed=True,
        ),
        surface=gs.surfaces.Plastic(color=(0, 1, 0)),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        vis_mode="collision",
        visualize_contact=True,
    )
    scene.build()

    franka.set_dofs_armature(franka.get_dofs_armature() + armature)

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")

    # set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.22]),
        quat=np.array([0, 1, 0, 0]),
    )
    # gripper open pos
    qpos[-2:] = 0.04
    path = franka.plan_path(qpos_goal=qpos, num_waypoints=300, resolution=0.05, max_retry=10)
    # execute the planned path
    franka.control_dofs_position(np.array([0.15, 0.15]), fingers_dof)
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        scene.step()

    # Get more time to the robot to reach the last waypoint
    for i in range(120):
        scene.step()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.13]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(60):
        scene.step()

    # grasp
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
    for i in range(50):
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.28]),
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(50):
        scene.step()

    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.4, 0.2, 0.2]),
        quat=np.array([0, 1, 0, 0]),
    )
    path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=100,
        resolution=0.05,
        max_retry=10,
        ee_link_name="hand",
        with_entity=cube,
    )
    for waypoint in path:
        franka.control_dofs_position(waypoint[:-2], motors_dof)
        scene.step()

    # Get more time to the robot to reach the last waypoint
    for i in range(50):
        scene.step()

    # release
    franka.control_dofs_position(np.array([0.15, 0.15]), fingers_dof)

    for i in range(180):
        scene.step()
        if i > 150:
            qvel = cube.get_dofs_velocity()
            assert_allclose(qvel, 0, atol=0.02)

    qpos = cube.get_dofs_position()
    assert_allclose(qpos[2], 0.075, atol=2e-3)


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_hanging_rigid_cable(show_viewer, tol):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.002,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/cable.xml",
        ),
    )
    scene.build()

    links_pos_0 = scene.rigid_solver.dyn_state.links.pos.to_numpy()[:, 0]
    links_quat_0 = scene.rigid_solver.dyn_state.links.quat.to_numpy()[:, 0]
    links_quat_0 /= np.linalg.norm(links_quat_0, axis=-1, keepdims=True)

    robot.set_dofs_position(robot.get_dofs_position())
    if show_viewer:
        scene.visualizer.update()
    for _ in range(100):
        scene.step()

    links_pos_f = scene.rigid_solver.dyn_state.links.pos.to_numpy()[:, 0]
    links_quat_f = scene.rigid_solver.dyn_state.links.quat.to_numpy()[:, 0]
    links_quat_f /= np.linalg.norm(links_quat_f, axis=-1, keepdims=True)
    links_quat_err = 2.0 * np.arccos(np.minimum(np.abs(np.sum(links_quat_f * links_quat_0, axis=-1)), 1.0))

    # FIXME: Why it is not possible to achieve better accuracy?
    assert_allclose(links_pos_0, links_pos_f, tol=1e-3)
    assert_allclose(links_quat_err, 0.0, tol=1e-3)


@pytest.mark.slow  # ~200s
@pytest.mark.parametrize("primitive_type", ["box", "sphere"])
@pytest.mark.parametrize("precision", ["64"])
def test_franka_panda_grasp_fem_entity(primitive_type, show_viewer):
    if gs.use_ndarray:
        pytest.skip("SAPCoupler does not support ndarray yet.")

    GRAPPER_POS_START = (0.65, 0.0, 0.13)
    GRAPPER_POS_END = (0.65, 0.0, 0.18)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60,
            substeps=2,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
            pcg_threshold=1e-10,
        ),
        coupler_options=gs.options.SAPCouplerOptions(
            pcg_threshold=1e-10,
            sap_convergence_atol=1e-10,
            sap_convergence_rtol=1e-10,
            linesearch_ftol=1e-10,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.3, 0.0, 0.15),
            camera_lookat=(0.65, 0.0, 0.15),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(
            coup_friction=1.0,
            friction=1.0,
        ),
    )
    # Only allow finger contact to accelerate
    for geom in franka.geoms:
        if "finger" not in geom.link.name:
            geom._contype = 0
            geom._conaffinity = 0
    if primitive_type == "sphere":
        obj = scene.add_entity(
            morph=gs.morphs.Sphere(
                pos=(0.65, 0.0, 0.02),
                radius=0.02,
            ),
            material=gs.materials.FEM.Elastic(
                model="linear_corotated",
                friction_mu=1.0,
                E=1e5,
                nu=0.4,
            ),
        )
    else:  # primitive_type == "box":
        asset_path = get_hf_dataset(pattern="meshes/cube8.obj")
        obj = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=f"{asset_path}/meshes/cube8.obj",
                pos=(0.65, 0.0, 0.02),
                scale=0.02,
            ),
            material=gs.materials.FEM.Elastic(
                model="linear_corotated",
                friction_mu=1.0,
            ),
        )
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")

    # init
    franka.set_qpos((-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04))
    box_pos_0 = obj.get_state().pos.mean(dim=-2)

    # hold
    qpos = franka.inverse_kinematics(link=end_effector, pos=GRAPPER_POS_START, quat=(0, 1, 0, 0))
    franka.control_dofs_position(qpos[motors_dof], motors_dof)
    for i in range(15):
        scene.step()

    # grasp
    for i in range(10):
        franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        scene.step()

    # lift and wait for while to give enough time for the robot to stop shaking
    qpos = franka.inverse_kinematics(link=end_effector, pos=GRAPPER_POS_END, quat=(0, 1, 0, 0))
    franka.control_dofs_position(qpos[motors_dof], motors_dof)
    for i in range(65):
        franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        scene.step()

    # Check that the box has moved by the expected delta, without slipping
    box_pos_f = obj.get_state().pos.mean(dim=-2)
    assert_allclose(box_pos_f - box_pos_0, np.array(GRAPPER_POS_END) - np.array(GRAPPER_POS_START), tol=5e-3)

    # wait for a while
    for i in range(25):
        franka.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
        scene.step()
    box_pos_post = obj.get_state().pos.mean(dim=-2)
    assert_allclose(box_pos_f, box_pos_post, atol=1e-3)
