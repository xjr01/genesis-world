import argparse
import math
import os
from typing import NamedTuple

import numpy as np

import genesis as gs
import genesis.utils.geom as geom_utils
import genesis.utils.mesh as mesh_utils
import genesis.utils.particle as particle_utils

CASE_CUBE = "cube"
CASE_MERGE = "merge"
CASE_BOUNCE = "bounce"
CASE_CONE = "cone"
CASE_TEAPOT = "teapot"
CASES = (CASE_CUBE, CASE_MERGE, CASE_BOUNCE, CASE_CONE, CASE_TEAPOT)

DEFAULT_CASE_DT = 1.0 / 30.0


class TeapotManipulatorSettings(NamedTuple):
    kuka_asset: str
    kuka_entity_name: str
    kuka_scale: float
    kuka_base_pos: tuple[float, float, float]
    kuka_base_quat: tuple[float, float, float, float]
    kuka_end_effector_link: str
    kuka_qpos: tuple[float, ...]
    hand_asset: str
    hand_entity_name: str
    hand_scale: float
    hand_mount_pos: tuple[float, float, float]
    hand_mount_quat: tuple[float, float, float, float]
    hand_qpos: tuple[float, ...]
    grasp_pos: tuple[float, float, float]
    grasp_quat: tuple[float, float, float, float]
    tool_center_point: tuple[float, float, float]
    camera_pos: tuple[float, float, float]
    camera_lookat: tuple[float, float, float]


class TeapotSettings(NamedTuple):
    asset: str
    entity_name: str
    mesh_scale: float
    offset: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    turning_axis_pos: tuple[float, float, float]
    particles_seed: tuple[float, float, float]
    particles_max_height: float
    particles_vel: tuple[float, float, float]
    manipulator: TeapotManipulatorSettings


class CaseSettings(NamedTuple):
    scale: int
    dt: float
    gravity: tuple[float, float, float]
    lower_bound: tuple[float, float, float]
    upper_bound: tuple[float, float, float]
    camera_pos: tuple[float, float, float]
    camera_lookat: tuple[float, float, float]
    static_colliders: tuple[gs.options.PBSTFStaticColliderOptions, ...]
    max_solver_iterations: int
    steps: int
    teapot: TeapotSettings | None


class TeapotPose(NamedTuple):
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    vel: tuple[float, float, float]
    ang: tuple[float, float, float]


def _liquid_material(
    sampler="staggered",
    rho=1000.0,
    density_compliance=500.0,
    surface_tension_compliance=0.8,
    surface_distance_compliance=40.0,
    interior_distance_compliance=180.0,
    surface_viscosity=0.3,
    interior_viscosity=0.3,
    is_collider_adhesion_friction_enabled=False,
    collider_adhesion_compliance=10.0,
    collider_friction=0.1,
):
    return gs.materials.PBSTF.Liquid(
        sampler=sampler,
        rho=rho,
        density_compliance=density_compliance,
        surface_tension_compliance=surface_tension_compliance,
        surface_distance_compliance=surface_distance_compliance,
        interior_distance_compliance=interior_distance_compliance,
        surface_viscosity=surface_viscosity,
        interior_viscosity=interior_viscosity,
        is_collider_adhesion_friction_enabled=is_collider_adhesion_friction_enabled,
        collider_adhesion_compliance=collider_adhesion_compliance,
        collider_friction=collider_friction,
    )


def _case_settings(case):
    if case == CASE_CUBE:
        return CaseSettings(
            scale=10,
            dt=DEFAULT_CASE_DT,
            gravity=(0.0, 0.0, 0.0),
            lower_bound=(-2.0, -2.0, -2.0),
            upper_bound=(2.0, 2.0, 2.0),
            camera_pos=(4.0, 4.0, 3.0),
            camera_lookat=(0.0, 0.0, 0.0),
            static_colliders=(),
            max_solver_iterations=100,
            steps=1500,
            teapot=None,
        )
    if case == CASE_MERGE:
        return CaseSettings(
            scale=10,
            dt=DEFAULT_CASE_DT,
            gravity=(0.0, 0.0, 0.0),
            lower_bound=(-10.0, -10.0, -10.0),
            upper_bound=(10.0, 10.0, 10.0),
            camera_pos=(4.0, 3.0, 4.0),
            camera_lookat=(0.0, -1.0, 0.0),
            static_colliders=(),
            max_solver_iterations=100,
            steps=300,
            teapot=None,
        )
    if case == CASE_BOUNCE:
        return CaseSettings(
            scale=10,
            dt=DEFAULT_CASE_DT,
            gravity=(0.0, 0.0, 0.0),
            lower_bound=(-10.0, -2.0, -10.0),
            upper_bound=(10.0, 10.0, 10.0),
            camera_pos=(4.0, 2.0, 4.0),
            camera_lookat=(0.0, -1.0, 0.0),
            static_colliders=(),
            max_solver_iterations=100,
            steps=300,
            teapot=None,
        )
    if case == CASE_CONE:
        return CaseSettings(
            scale=10,
            dt=DEFAULT_CASE_DT,
            gravity=(0.0, 0.0, 0.0),
            lower_bound=(-20.0, -20.0, -20.0),
            upper_bound=(20.0, 20.0, 20.0),
            camera_pos=(12.0, 2.0, 12.0),
            camera_lookat=(0.0, -3.0, 0.0),
            static_colliders=(
                gs.options.PBSTFConeStaticColliderOptions(
                    center=(0.0, -7.0, 0.0),
                    height=(0.0, 5.0, 0.0),
                    radius=5.0 * math.sqrt(3.0),
                ),
            ),
            max_solver_iterations=100,
            steps=300,
            teapot=None,
        )
    if case == CASE_TEAPOT:
        manipulator = TeapotManipulatorSettings(
            kuka_asset="urdf/kuka_iiwa/model.urdf",
            kuka_entity_name="teapot_kuka",
            kuka_scale=12.0,
            kuka_base_pos=(-6.165, -5.8945, -13.5525),
            kuka_base_quat=(math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0),
            kuka_end_effector_link="lbr_iiwa_link_7",
            kuka_qpos=(
                0.16527023911476135,
                1.8198974132537842,
                0.6981679201126099,
                1.1762551069259644,
                0.8250933289527893,
                -1.3872859477996826,
                0.9264655709266663,
            ),
            hand_asset="urdf/shadow_hand/shadow_hand.urdf",
            hand_entity_name="teapot_shadow_hand",
            hand_scale=14.0,
            hand_mount_pos=(0.0, 0.0, 0.54),
            hand_mount_quat=(1.0, 0.0, 0.0, 0.0),
            hand_qpos=(
                0.0000012000000424450263,
                -0.00003170000127283856,
                -0.396697998046875,
                1.2216999530792236,
                0.026160499081015587,
                0.31376829743385315,
                0.635699987411499,
                -0.06306590139865875,
                1.4007999897003174,
                0.44350001215934753,
                0.6726999878883362,
                -0.03246590122580528,
                1.5707999467849731,
                0.6097999811172485,
                0.2476000040769577,
                -0.349065899848938,
                1.3489999771118164,
                0.7651000022888184,
                0.41029998660087585,
                0.0,
                -0.349065899848938,
                0.9092000126838684,
                0.6690000295639038,
                0.8981000185012817,
            ),
            grasp_pos=(-2.6035435064512424, 1.692335, 0.2395262797782305),
            grasp_quat=(0.5957038027506156, 0.38096847558355235, 0.38096847558355235, 0.5957038027506156),
            tool_center_point=(0.0, -0.525, 5.5275),
            camera_pos=(63.0, 23.5, -7.25),
            camera_lookat=(-3.0, -8.0, -12.5),
        )
        teapot_mesh_scale = 2.25
        teapot_offset = (0.0, -3.79, 0.0)
        teapot_quat = (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)
        turning_axis_pos = tuple(
            geom_utils.transform_by_trans_quat(
                np.array(manipulator.grasp_pos) * teapot_mesh_scale,
                np.array(teapot_offset),
                np.array(teapot_quat),
            ).tolist()
        )
        teapot = TeapotSettings(
            asset="meshes/utah_teapot_modified.obj",
            entity_name="teapot_visual",
            mesh_scale=teapot_mesh_scale,
            offset=teapot_offset,
            quat=teapot_quat,
            turning_axis_pos=turning_axis_pos,
            particles_seed=(0.0, -3.15, 0.0),
            particles_max_height=0.7,
            particles_vel=(0.0, 0.0, 0.0),
            manipulator=manipulator,
        )
        return CaseSettings(
            scale=20,
            dt=0.01,
            gravity=(0.0, -9.8, 0.0),
            lower_bound=(-20.0, -6.04186, -20.0),
            upper_bound=(20.0, 15.0, 20.0),
            camera_pos=manipulator.camera_pos,
            camera_lookat=manipulator.camera_lookat,
            static_colliders=(
                gs.options.PBSTFMeshStaticColliderOptions(
                    file=teapot.asset,
                    scale=teapot.mesh_scale,
                    sdf_res=150,
                    pos=teapot.offset,
                    quat=teapot.quat,
                ),
            ),
            max_solver_iterations=5,
            steps=10000,
            teapot=teapot,
        )
    raise ValueError(f"Unknown PBSTF example case: {case}")


def _teapot_pose(time, settings):
    """Return the teapot pose for a rotation about the world-X line through its grasp point."""
    turning_rate_initial = math.radians(1.5)
    stop_angle_initial = math.radians(27.0)
    hold_end = stop_angle_initial / turning_rate_initial + 10.0
    turning_rate_final = math.radians(5.0)
    stop_angle_final = math.radians(19.0)
    turn_end = hold_end + abs(stop_angle_initial - stop_angle_final) / turning_rate_final

    if time <= stop_angle_initial / turning_rate_initial:
        angle = turning_rate_initial * time
        turning_rate = turning_rate_initial
    elif time <= hold_end:
        angle = stop_angle_initial
        turning_rate = 0.0
    elif time <= turn_end:
        angle = stop_angle_initial - turning_rate_final * (time - hold_end)
        turning_rate = -turning_rate_final
    else:
        angle = stop_angle_final
        turning_rate = 0.0

    half_angle = 0.5 * angle
    cos_half_angle = math.cos(half_angle)
    sin_half_angle = math.sin(half_angle)
    quat = settings.quat
    turning_axis_pos = settings.turning_axis_pos
    offset_y = settings.offset[1] - turning_axis_pos[1]
    offset_z = settings.offset[2] - turning_axis_pos[2]
    pos = (
        settings.offset[0],
        turning_axis_pos[1] + offset_y * math.cos(angle) - offset_z * math.sin(angle),
        turning_axis_pos[2] + offset_y * math.sin(angle) + offset_z * math.cos(angle),
    )
    return TeapotPose(
        pos=pos,
        quat=(
            cos_half_angle * quat[0] - sin_half_angle * quat[1],
            cos_half_angle * quat[1] + sin_half_angle * quat[0],
            cos_half_angle * quat[2] - sin_half_angle * quat[3],
            cos_half_angle * quat[3] + sin_half_angle * quat[2],
        ),
        vel=(
            0.0,
            -turning_rate * (pos[2] - turning_axis_pos[2]),
            turning_rate * (pos[1] - turning_axis_pos[1]),
        ),
        ang=(turning_rate, 0.0, 0.0),
    )


def sample_teapot_particles(settings, particle_size):
    teapot_mesh = mesh_utils.load_mesh(os.path.join(gs.utils.get_assets_dir(), settings.asset)).copy()
    teapot_mesh.merge_vertices(merge_tex=True, merge_norm=True)
    teapot_mesh.vertices = geom_utils.transform_by_quat(
        teapot_mesh.vertices * settings.mesh_scale,
        np.array(settings.quat),
    ) + np.array(settings.offset)
    return particle_utils.mesh_cavity_to_particles(
        teapot_mesh,
        p_size=particle_size,
        seed=settings.particles_seed,
        max_height=settings.particles_max_height,
        clearance=0.5 * particle_size,
    )


def update_rigid_teapot(teapot, time, settings):
    pose = _teapot_pose(time, settings)
    teapot.set_pos(
        pose.pos,
        zero_velocity=False,
        relative=False,
        skip_forward=True,
    )
    teapot.set_quat(
        pose.quat,
        zero_velocity=False,
        relative=False,
    )
    teapot.set_velocity(
        vel=pose.vel,
        ang=pose.ang,
    )


def _update_teapot_manipulator(kuka, pose, settings, init_qpos):
    """Set the KUKA joints so the attached hand keeps its teapot-local grasp transform."""
    manipulator = settings.manipulator
    target_pos, target_quat = geom_utils.transform_pos_quat_by_trans_quat(
        np.array(manipulator.grasp_pos) * settings.mesh_scale,
        np.array(manipulator.grasp_quat),
        np.array(pose.pos),
        np.array(pose.quat),
    )
    qpos = kuka.inverse_kinematics(
        link=kuka.get_link(manipulator.kuka_end_effector_link),
        pos=target_pos,
        quat=target_quat,
        local_point=manipulator.tool_center_point,
        init_qpos=init_qpos,
        pos_tol=1e-4,
        rot_tol=1e-4,
    )
    kuka.set_qpos(qpos)
    return qpos


def _add_case_entities(scene, case, particle_size, settings):
    if case == CASE_CUBE:
        liquid = scene.add_entity(
            morph=gs.morphs.Box(
                lower=(-1.0, -1.0, -1.0),
                upper=(1.0, 1.0, 1.0),
            ),
            material=_liquid_material(),
        )
        return ((liquid, (0.0, 0.0, 0.0)),)

    if case == CASE_MERGE:
        left = scene.add_entity(
            morph=gs.morphs.Box(
                lower=(-1.5, -0.625, -0.5),
                upper=(-0.5, 0.375, 0.5),
            ),
            material=_liquid_material(
                surface_tension_compliance=1.0,
                surface_viscosity=0.05,
                interior_viscosity=0.05,
            ),
        )
        right = scene.add_entity(
            morph=gs.morphs.Box(
                lower=(0.5, -0.375, -0.5),
                upper=(1.5, 0.625, 0.5),
            ),
            material=_liquid_material(
                surface_tension_compliance=1.0,
                surface_viscosity=0.05,
                interior_viscosity=0.05,
            ),
        )
        return (
            (left, (1.0, 0.0, 0.0)),
            (right, (-1.0, 0.0, 0.0)),
        )

    if case == CASE_BOUNCE:
        liquid = scene.add_entity(
            morph=gs.morphs.Sphere(
                pos=(0.0, 0.0, 0.0),
                radius=1.0,
            ),
            material=_liquid_material(),
        )
        return ((liquid, (0.0, -3.0, 0.0)),)

    if case == CASE_CONE:
        liquid = scene.add_entity(
            morph=gs.morphs.Sphere(
                pos=(0.0, 0.0, 0.0),
                radius=1.0,
            ),
            material=_liquid_material(
                surface_tension_compliance=1.0,
                interior_distance_compliance=90.0,
                surface_viscosity=0.05,
                interior_viscosity=0.05,
            ),
        )
        return ((liquid, (0.0, -4.0, 0.0)),)

    if case == CASE_TEAPOT:
        teapot = settings.teapot
        if teapot is None:
            gs.raise_exception("The teapot case requires teapot settings.")
        particles = sample_teapot_particles(teapot, particle_size)
        manipulator = teapot.manipulator

        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=teapot.asset,
                scale=teapot.mesh_scale,
                pos=teapot.offset,
                quat=teapot.quat,
                collision=False,
            ),
            material=gs.materials.Kinematic(),
            surface=gs.surfaces.Default(
                color=(0.66, 0.66, 0.66),
                opacity=0.3,
            ),
            name=teapot.entity_name,
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
        liquid = scene.add_entity(
            morph=gs.morphs.Particles(
                positions=particles,
            ),
            material=_liquid_material(
                sampler="regular",
                density_compliance=150.0,
                surface_tension_compliance=3.0,
                surface_distance_compliance=40.0,
                interior_distance_compliance=180.0,
                surface_viscosity=0.2,
                interior_viscosity=0.05,
                is_collider_adhesion_friction_enabled=True,
                collider_adhesion_compliance=20.0,
                collider_friction=0.01,
            ),
        )
        return ((liquid, teapot.particles_vel),)

    raise ValueError(f"Unknown PBSTF example case: {case}")


def _draw_case_colliders(scene, case):
    if case != CASE_CONE:
        return

    cone = mesh_utils.create_cone(
        radius=5.0 * math.sqrt(3.0),
        height=5.0,
        sections=96,
        color=(0.35, 0.38, 0.42, 1.0),
    )
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.array(
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        ),
        dtype=np.float32,
    )
    transform[:3, 3] = (0.0, -7.0, 0.0)
    scene.draw_debug_mesh(cone, T=transform)


def start_viewer_recording(scene, is_recording):
    if is_recording:
        scene.viewer.toggle_recording()


def stop_viewer(scene, is_viewer_shown):
    if not is_viewer_shown:
        return
    if scene.viewer.recording:
        scene.viewer.toggle_recording()
    scene.viewer.stop()


def build_scene(case=CASE_CUBE, scale=None, show_viewer=False, dt=None):
    """Build one of the C++ position-based surface-tension flow reference cases."""
    settings = _case_settings(case)
    if scale is None:
        scale = settings.scale
    if scale <= 0:
        raise ValueError("PBSTF particle scale must be positive")
    if case == CASE_TEAPOT and scale < settings.scale:
        raise ValueError(
            f"The teapot case requires scale >= {settings.scale} to keep more than 50,000 initial particles"
        )

    if dt is None:
        dt = settings.dt
    if dt <= 0.0:
        raise ValueError("PBSTF time step must be positive")
    particle_size = 2.0 / scale
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=settings.gravity,
        ),
        rigid_options=(
            gs.options.RigidOptions(
                enable_collision=False,
                disable_constraint=True,
            )
            if case == CASE_TEAPOT
            else None
        ),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=particle_size,
            lower_bound=settings.lower_bound,
            upper_bound=settings.upper_bound,
            max_solver_iterations=settings.max_solver_iterations,
            topology_rebuild_interval=10,
            max_surface_neighbors=128,
            enable_pca_normals=False,
            static_colliders=list(settings.static_colliders),
        ),
        viewer_options=gs.options.ViewerOptions(
            refresh_rate=round(1.0 / dt),
            camera_pos=settings.camera_pos,
            camera_lookat=settings.camera_lookat,
            camera_up=(0.0, 1.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    entities_and_velocities = _add_case_entities(scene, case, particle_size, settings)
    scene.build()
    for entity, velocity in entities_and_velocities:
        entity.set_particles_vel(velocity)
    if settings.teapot is not None:
        manipulator = settings.teapot.manipulator
        kuka = scene.get_entity(name=manipulator.kuka_entity_name)
        hand = scene.get_entity(name=manipulator.hand_entity_name)
        kuka.set_qpos(manipulator.kuka_qpos)
        hand.set_qpos(manipulator.hand_qpos)
        _update_teapot_manipulator(kuka, _teapot_pose(0.0, settings.teapot), settings.teapot, manipulator.kuka_qpos)
    if show_viewer:
        _draw_case_colliders(scene, case)

    return scene, tuple(entity for entity, _ in entities_and_velocities)


def main():
    parser = argparse.ArgumentParser(description="GPU Position-Based Surface Tension Flow examples")
    parser.add_argument("--case", choices=CASES, default=CASE_CUBE)
    parser.add_argument("--scale", type=int, default=None, help="C++ particle scale (radius = 1 / scale)")
    parser.add_argument("--dt", type=float, default=None, help="Override the case's default time step")
    parser.add_argument("--steps", type=int, default=None, help="Override the case's default simulation horizon")
    parser.add_argument("-v", "--vis", dest="show_viewer", action="store_true", default=False)
    parser.add_argument(
        "--record",
        dest="is_recording",
        action="store_true",
        default=False,
        help="Record the viewer and prompt for the output path on exit",
    )
    args = parser.parse_args()
    if args.scale is not None and args.scale <= 0:
        parser.error("--scale must be positive")
    if args.dt is not None and args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")

    gs.init(backend=gs.cuda, precision="32", logging_level="info")
    settings = _case_settings(args.case)
    is_viewer_shown = args.show_viewer or args.is_recording
    scene, _ = build_scene(case=args.case, scale=args.scale, show_viewer=is_viewer_shown, dt=args.dt)
    teapot_settings = settings.teapot
    teapot = scene.get_entity(name=teapot_settings.entity_name) if teapot_settings is not None else None
    if teapot_settings is None:
        kuka = None
        kuka_qpos = None
    else:
        kuka = scene.get_entity(name=teapot_settings.manipulator.kuka_entity_name)
        kuka_qpos = kuka.get_qpos()

    steps = settings.steps if args.steps is None else args.steps
    if "PYTEST_VERSION" in os.environ:
        steps = min(steps, 2)
    start_viewer_recording(scene, args.is_recording)
    try:
        for _ in range(steps):
            if teapot is not None:
                pose = _teapot_pose(scene.cur_t, teapot_settings)
                scene.pbstf_solver.set_static_colliders_pose(
                    pos=pose.pos,
                    quat=pose.quat,
                    colliders_idx=0,
                )
                teapot.set_pos(
                    pose.pos,
                    relative=False,
                    skip_forward=True,
                )
                teapot.set_quat(
                    pose.quat,
                    relative=False,
                )
                kuka_qpos = _update_teapot_manipulator(kuka, pose, teapot_settings, kuka_qpos)
            scene.step()
    finally:
        stop_viewer(scene, is_viewer_shown)


if __name__ == "__main__":
    main()
