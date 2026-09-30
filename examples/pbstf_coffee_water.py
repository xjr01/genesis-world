"""Pour water, absorb the spill with a sponge, and stir coffee with a Sim1 dual-arm robot.

Run headlessly with ``python -m examples.pbstf_coffee_water`` from the repository. Add ``--vis`` for the viewer,
``--record`` for a video and stage images, or ``--surface`` for a reconstructed liquid surface.
``--check-motion`` checks the articulated trajectory without simulating liquid or sponge dynamics.
World X points right, Y points up, and Z points toward the front of the table and the robot.
The default particle scale resolves pouring and the wall film; coarse scales can retain liquid inside the tilted cup.
"""

import argparse
import csv
from dataclasses import dataclass
from enum import IntEnum
import math
from pathlib import Path

import numpy as np

from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh

import genesis as gs
from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
import genesis.utils.element as element_utils
import genesis.utils.geom as geom_utils
import genesis.utils.mesh as mesh_utils
import genesis.utils.particle as particle_utils
from genesis.utils.misc import tensor_to_array

CUP_ASSET = "meshes/drinking_glass/12-oz-glass.obj"
CUP_CAVITY_ASSET = "meshes/drinking_glass/12-oz-glass-cavity.obj"
COFFEE_CUP_POS = (-0.09, -0.045, 0.0)
WATER_CUP_POS = (0.09, -0.045, 0.0)
COFFEE_FILL_FRACTION = 0.5
WATER_FILL_FRACTION = 1.0
# The high grasp keeps the wrist above the table while the palm stays opposite the pouring lip.
CUP_GRIP_HEIGHT = 0.09
ROBOT_ASSET = "urdf/sim1_acone/acone_collision.urdf"
ROD_ASSET = "meshes/glass_stirring_rod/glass_rod.obj"
ROBOT_POS = (-0.06, -0.55, 0.40)
ROBOT_QUAT = (0.5, -0.5, 0.5, 0.5)
TOOL_CENTER = np.array((0.14747, 0.001786, 0.0))
TABLE_Y = -0.045
# The sponge top keeps the wider finger mounts above the cup rims during the central wiping stroke.
SPONGE_SIZE = (0.04, 0.05, 0.08)
SPONGE_GRIP_HEIGHT = SPONGE_SIZE[1] - 0.005
SPONGE_START = (0.0, TABLE_Y + 0.001, -0.17)
SPONGE_END = (0.0, TABLE_Y + 0.001, 0.17)
ROD_GRIP_HEIGHT = 0.15
ROD_PARK = (-0.19, -0.025, 0.0)
POUR_QUAT = np.array((0.0, 0.0, math.sqrt(0.5), -math.sqrt(0.5)))
MOP_QUAT = Rotation.from_matrix(((0, -1, 0), (-1, 0, 0), (0, 0, -1))).as_quat(scalar_first=True)
ROD_GRIP_QUAT = (
    Rotation.from_euler("y", -20, degrees=True)
    * Rotation.from_euler("z", -30, degrees=True)
    * Rotation.from_matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
).as_quat(scalar_first=True)
MOTION_END = 28.0


class Phase(IntEnum):
    POUR = 0
    RELEASE = 1
    RETRACT = 2
    TRANSFER = 3
    APPROACH_SPONGE = 4
    GRASP_SPONGE = 5
    WIPE = 6
    LIFT_ROD = 7
    APPROACH_CUP = 8
    INSERT_ROD = 9
    STIR = 10
    WITHDRAW_ROD = 11
    PARK_ROD = 12
    REST = 13


@dataclass(frozen=True)
class ToolPose:
    pos: np.ndarray
    quat: np.ndarray


@dataclass(frozen=True)
class MotionTarget:
    phase: Phase
    right: ToolPose
    left: ToolPose
    right_opening: float
    rod_pos: np.ndarray
    sponge_pos: np.ndarray
    stir_angle: float


@dataclass
class CoffeeWaterScene:
    scene: gs.Scene
    coffee: gs.engine.entities.PBSTFEntity | None
    water: gs.engine.entities.PBSTFEntity | None
    water_cup: gs.engine.entities.RigidEntity
    camera: gs.vis.camera.Camera | None
    cup_cavity: trimesh.Trimesh
    robot: gs.engine.entities.RigidEntity
    rod: gs.engine.entities.RigidEntity
    sponge: gs.engine.entities.PBD3DEntity | gs.engine.entities.RigidEntity
    hands_link: tuple[RigidLink, RigidLink]
    arms_dofs_idx: tuple[tuple[int, ...], tuple[int, ...]]
    hands_fingers_qs_idx: tuple[tuple[int, int], tuple[int, int]]
    right_finger_link_pair: tuple[RigidLink, RigidLink]
    qpos: gs.Tensor
    is_motion_only: bool


def smooth_progress(time, start, end):
    """Interpolate a phase with zero endpoint velocity and acceleration."""
    fraction = np.clip((time - start) / (end - start), 0.0, 1.0)
    return fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))


def interpolate_tool(start, end, progress):
    """Interpolate a tool center and its orientation in the world frame."""
    quat = geom_utils.slerp(start.quat, end.quat, np.array(progress))
    return ToolPose(start.pos + progress * (end.pos - start.pos), quat)


def motion_target(time):
    """Return both grasp targets and the sequential pouring, wiping, and three-turn stirring phases."""
    cup_pos, cup_quat = water_cup_pose(time)
    cup_quat = np.array(cup_quat)
    right = ToolPose(
        np.array(cup_pos) + geom_utils.transform_by_quat(np.array((0.0, CUP_GRIP_HEIGHT, 0.0)), cup_quat),
        geom_utils.transform_quat_by_quat(POUR_QUAT, cup_quat),
    )
    right_opening = 0.038
    sponge_pos = np.array(SPONGE_START)
    rod_pos = np.array(ROD_PARK)
    stir_angle = 0.0
    phase = Phase.POUR
    release = ToolPose(np.array(WATER_CUP_POS) + (0.0, CUP_GRIP_HEIGHT, 0.0), POUR_QUAT)
    retract = ToolPose(release.pos + (0.07, 0.0, 0.0), POUR_QUAT)
    raised = ToolPose(np.array((0.12, 0.17, 0.10)), POUR_QUAT)
    above_sponge = ToolPose(np.array((0.0, 0.17, -0.17)), MOP_QUAT)
    grasp = ToolPose(sponge_pos + (0.0, SPONGE_GRIP_HEIGHT, 0.0), MOP_QUAT)
    if 6.0 <= time < 6.4:
        phase = Phase.RELEASE
        right_opening += (0.044 - right_opening) * smooth_progress(time, 6.0, 6.4)
    elif 6.4 <= time < 7.4:
        phase = Phase.RETRACT
        right = interpolate_tool(release, retract, smooth_progress(time, 6.4, 7.4))
        right_opening = 0.044
    elif 7.4 <= time < 8.2:
        phase = Phase.TRANSFER
        right = interpolate_tool(retract, raised, smooth_progress(time, 7.4, 8.2))
        right_opening = 0.044
    elif 8.2 <= time < 9.6:
        phase = Phase.TRANSFER
        progress = smooth_progress(time, 8.2, 9.6)
        right = interpolate_tool(raised, above_sponge, progress)
        right_opening = 0.044 - 0.016 * progress
    elif 9.6 <= time < 10.6:
        phase = Phase.APPROACH_SPONGE
        right = interpolate_tool(above_sponge, grasp, smooth_progress(time, 9.6, 10.6))
        right_opening = 0.028
    elif 10.6 <= time < 11.2:
        phase = Phase.GRASP_SPONGE
        right = grasp
        right_opening = 0.028 - 0.010 * smooth_progress(time, 10.6, 11.2)
    elif time >= 11.2:
        phase = Phase.WIPE
        sponge_pos += smooth_progress(time, 11.2, 15.2) * (np.array(SPONGE_END) - sponge_pos)
        right = ToolPose(sponge_pos + (0.0, SPONGE_GRIP_HEIGHT, 0.0), MOP_QUAT)
        right_opening = 0.018

    raised_rod = np.array((-0.19, 0.075, 0.0))
    above_cup = np.array((-0.07, 0.075, 0.0))
    inserted_rod = np.array((-0.07, -0.024, 0.0))
    if 15.2 <= time < 16.0:
        phase = Phase.LIFT_ROD
        rod_pos += smooth_progress(time, 15.2, 16.0) * (raised_rod - rod_pos)
    elif 16.0 <= time < 17.0:
        phase = Phase.APPROACH_CUP
        rod_pos = raised_rod + smooth_progress(time, 16.0, 17.0) * (above_cup - raised_rod)
    elif 17.0 <= time < 18.0:
        phase = Phase.INSERT_ROD
        rod_pos = above_cup + smooth_progress(time, 17.0, 18.0) * (inserted_rod - above_cup)
    elif 18.0 <= time < 24.0:
        phase = Phase.STIR
        stir_angle = 6.0 * math.pi * smooth_progress(time, 18.0, 24.0)
        rod_pos = np.array((-0.09 + 0.02 * math.cos(stir_angle), -0.024, 0.02 * math.sin(stir_angle)))
    elif 24.0 <= time < 25.0:
        phase = Phase.WITHDRAW_ROD
        rod_pos = inserted_rod + smooth_progress(time, 24.0, 25.0) * (above_cup - inserted_rod)
        stir_angle = 6.0 * math.pi
    elif 25.0 <= time < 26.0:
        phase = Phase.PARK_ROD
        rod_pos = above_cup + smooth_progress(time, 25.0, 26.0) * (raised_rod - above_cup)
        stir_angle = 6.0 * math.pi
    elif 26.0 <= time < 27.0:
        phase = Phase.PARK_ROD
        rod_pos = raised_rod + smooth_progress(time, 26.0, 27.0) * (np.array(ROD_PARK) - raised_rod)
        stir_angle = 6.0 * math.pi
    elif time >= 27.0:
        phase = Phase.REST
        stir_angle = 6.0 * math.pi
    left = ToolPose(rod_pos + (0.0, ROD_GRIP_HEIGHT, 0.0), ROD_GRIP_QUAT)
    return MotionTarget(phase, right, left, right_opening, rod_pos, sponge_pos, stir_angle)


def update_motion(demo, time):
    """Solve each arm with its own degrees of freedom and synchronize prescribed collider poses."""
    target = motion_target(time)
    errors = []
    for link, dofs, pose in zip(demo.hands_link, demo.arms_dofs_idx, (target.right, target.left)):
        qpos, error = demo.robot.inverse_kinematics(
            link=link,
            pos=pose.pos,
            quat=pose.quat,
            local_point=TOOL_CENTER,
            init_qpos=demo.qpos,
            max_samples=1,
            max_solver_iters=50,
            pos_tol=1e-4,
            rot_tol=1e-4,
            dofs_idx_local=dofs,
            return_error=True,
        )
        error_array = tensor_to_array(error)
        pos_error = np.linalg.norm(error_array[..., :3], axis=-1).max()
        rot_error = np.linalg.norm(error_array[..., 3:], axis=-1).max()
        if not np.isfinite(error_array).all() or pos_error > 2e-4 or rot_error > 2e-4:
            raise RuntimeError(f"{link.name} IK failed at {time:.3f}s: position={pos_error}, rotation={rot_error}.")
        demo.qpos = qpos
        errors.append(pos_error)
    demo.qpos[..., demo.hands_fingers_qs_idx[0]] = target.right_opening
    demo.qpos[..., demo.hands_fingers_qs_idx[1]] = 0.003
    demo.robot.set_qpos(demo.qpos, zero_velocity=True)
    cup_pos, cup_quat = water_cup_pose(time)
    demo.water_cup.set_pos(cup_pos)
    demo.water_cup.set_quat(cup_quat)
    demo.rod.set_pos(target.rod_pos)
    if demo.is_motion_only:
        demo.sponge.set_pos(target.sponge_pos + (0.0, 0.5 * SPONGE_SIZE[1], 0.0))
    elif demo.scene.pbstf_solver.is_active:
        demo.scene.pbstf_solver.set_static_colliders_pose(
            np.stack((cup_pos, target.rod_pos, target.sponge_pos)),
            np.array((cup_quat, (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))),
            colliders_idx=(2, 3, 4),
        )
    return target, max(errors)


def water_cup_pose(time):
    """Return the water cup pose through settling, lifting, pouring and returning, in seconds."""
    pour_x = COFFEE_CUP_POS[0] + 0.12
    lift_y = COFFEE_CUP_POS[1] + 0.05
    pour_y = COFFEE_CUP_POS[1] + 0.07
    times = np.array((0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0))
    poses = np.array(
        [
            (WATER_CUP_POS[0], WATER_CUP_POS[1], 0.0),
            (WATER_CUP_POS[0], WATER_CUP_POS[1], 0.0),
            (WATER_CUP_POS[0], lift_y, 0.0),
            (pour_x, lift_y, 0.0),
            (pour_x, pour_y, 40.0),
            (pour_x, pour_y, 30.0),
            (pour_x, lift_y, 0.0),
            (WATER_CUP_POS[0], WATER_CUP_POS[1], 0.0),
        ]
    )
    interval = np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 2)
    fraction = np.clip((time - times[interval]) / (times[interval + 1] - times[interval]), 0.0, 1.0)
    fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    pose = poses[interval] + fraction * (poses[interval + 1] - poses[interval])
    angle = math.radians(pose[2])
    return (pose[0], pose[1], WATER_CUP_POS[2]), (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def check_contacts(demo, time):
    """Reject rigid penetration while allowing the right fingers to compress the motion-preview sponge."""
    demo.robot.detect_collision()
    contacts = demo.robot.get_contacts()
    is_forbidden = contacts["penetration"] > 2e-4
    if demo.is_motion_only:
        is_finger_a = (contacts["link_a"] == demo.right_finger_link_pair[0].idx) | (
            contacts["link_a"] == demo.right_finger_link_pair[1].idx
        )
        is_finger_b = (contacts["link_b"] == demo.right_finger_link_pair[0].idx) | (
            contacts["link_b"] == demo.right_finger_link_pair[1].idx
        )
        is_sponge_contact = (is_finger_a & (contacts["link_b"] == demo.sponge.base_link.idx)) | (
            is_finger_b & (contacts["link_a"] == demo.sponge.base_link.idx)
        )
        is_forbidden &= ~is_sponge_contact
    if is_forbidden.any():
        links_a = tensor_to_array(contacts["link_a"][is_forbidden])
        links_b = tensor_to_array(contacts["link_b"][is_forbidden])
        penetration = tensor_to_array(contacts["penetration"][is_forbidden])
        deepest = penetration.argmax()
        link_a = demo.scene.rigid_solver.links[links_a[deepest]]
        link_b = demo.scene.rigid_solver.links[links_b[deepest]]
        raise RuntimeError(
            f"Rigid penetration at {time:.3f}s: {link_a.name} / {link_b.name}, depth={penetration[deepest]:.6f}m."
        )


def build_scene(
    scale=1500,
    is_viewer_shown=False,
    is_recording=False,
    is_surface=False,
    is_motion_only=False,
    is_liquid_enabled=True,
):
    """Build the dual-arm table scene in meters with prescribed cups and a deformable absorbent sponge.

    Motion-only mode uses a rigid sponge preview. Disabling liquid keeps sponge dynamics for isolated contact checks.
    """
    if scale <= 0:
        raise ValueError("Particle scale must be positive.")
    dt = 0.02 if is_motion_only else 0.002
    particle_size = 2.0 / scale
    cup_path = Path(gs.utils.get_assets_dir()) / CUP_ASSET
    cup = mesh_utils.load_mesh(cup_path)
    cup_cavity = mesh_utils.load_mesh(Path(gs.utils.get_assets_dir()) / CUP_CAVITY_ASSET)
    bottom, rim = cup_cavity.bounds[:, 1]
    cups_pos = (COFFEE_CUP_POS, WATER_CUP_POS)
    camera_pos = (0.0, 0.42, 0.04)
    camera_lookat = (0.0, 0.02, 0.0)
    table_collider = gs.options.PBSTFBoxStaticColliderOptions(
        is_collider_adhesion_friction_enabled=True,
        collider_adhesion_compliance=50.0,
        collider_friction=0.5,
        lower=(-0.4, -0.065, -4.0 / 15.0),
        upper=(0.4, -0.045, 4.0 / 15.0),
    )
    # Earlier colliders preserve the supporting surface at incompatible contacts.
    colliders = [table_collider]
    colliders.extend(
        gs.options.PBSTFMeshStaticColliderOptions(
            pos=pos,
            is_collider_adhesion_friction_enabled=True,
            collider_adhesion_compliance=50.0,
            collider_friction=0.1,
            file=str(cup_path),
            sdf_res=128,
        )
        for pos in cups_pos
    )
    colliders.extend(
        (
            gs.options.PBSTFMeshStaticColliderOptions(
                pos=ROD_PARK,
                is_collider_adhesion_friction_enabled=True,
                collider_adhesion_compliance=50.0,
                collider_friction=0.1,
                file=str(Path(gs.utils.get_assets_dir()) / ROD_ASSET),
                sdf_res=128,
            ),
            gs.options.PBSTFAbsorbentBoxStaticColliderOptions(
                pos=SPONGE_START,
                is_collider_adhesion_friction_enabled=True,
                collider_adhesion_compliance=50.0,
                collider_friction=0.5,
                lower=(-0.5 * SPONGE_SIZE[0], 0.0, -0.5 * SPONGE_SIZE[2]),
                upper=(0.5 * SPONGE_SIZE[0], SPONGE_SIZE[1], 0.5 * SPONGE_SIZE[2]),
                absorption_rate=4000.0,
                absorption_capacity_fraction=1.0,
                pbd_entity_name="sponge",
            ),
        )
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, -9.8, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            gravity=(0.0, 0.0, 0.0),
            enable_neutral_collision=True,
            disable_constraint=True,
        ),
        pbd_options=gs.options.PBDUnifiedOptions(
            particle_size=0.005,
            lower_bound=(-0.4, -1.0 / 15.0, -4.0 / 15.0),
            upper_bound=(0.4, 8.0 / 15.0, 4.0 / 15.0),
            max_solver_iterations=100,
            constraint_acceleration=0.85,
        ),
        pbstf_options=gs.options.PBSTFOptions(
            diffusion_coeff=0.005,
            particle_size=particle_size,
            max_solver_iterations=20,
            topology_rebuild_interval=10,
            max_surface_neighbors=128,
            max_localmesh_neighbors=64,
            enable_pca_normals=False,
            static_colliders=colliders if is_liquid_enabled and not is_motion_only else [],
            lower_bound=(-0.4, -1.0 / 15.0, -4.0 / 15.0),
            upper_bound=(0.4, 8.0 / 15.0, 4.0 / 15.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            refresh_rate=round(1.0 / dt),
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_up=(0.0, 1.0, 0.0),
            camera_fov=70,
        ),
        vis_options=gs.options.VisOptions(
            ambient_light=(0.4, 0.4, 0.4),
        ),
        show_viewer=is_viewer_shown,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            fixed=True,
            lower=table_collider.lower,
            upper=table_collider.upper,
        ),
        material=gs.materials.Rigid(
            coup_friction=0.0,
            is_coup_reaction_enabled=False,
        ),
        surface=gs.surfaces.Default(
            color=(0.36, 0.24, 0.14),
        ),
        name="table",
    )
    liquids = []
    cups = []
    for pos, fill_fraction, concentration, name in zip(
        cups_pos, (COFFEE_FILL_FRACTION, WATER_FILL_FRACTION), (1.0, 0.0), ("coffee", "water")
    ):
        cups.append(
            scene.add_entity(
                morph=gs.morphs.Mesh(
                    pos=pos,
                    file=str(cup_path),
                    convexify=False,
                    fixed=True,
                ),
                material=gs.materials.Rigid(
                    is_coup_reaction_enabled=False,
                ),
                surface=gs.surfaces.Default(
                    color=(0.65, 0.75, 0.85, 0.25),
                ),
                name=f"{name}_cup",
            )
        )
        if is_motion_only or not is_liquid_enabled:
            continue
        liquid_top = bottom + fill_fraction * (rim - bottom)
        particles = particle_utils.mesh_cavity_to_particles(
            cup,
            p_size=particle_size,
            seed=(0.0, liquid_top - 0.5 * particle_size, 0.0),
            max_height=liquid_top - 0.5 * particle_size,
        )
        liquids.append(
            scene.add_entity(
                morph=gs.morphs.Particles(
                    pos=pos,
                    positions=particles,
                ),
                material=gs.materials.PBSTF.Liquid(
                    sampler="regular",
                    rho=1000.0,
                    density_compliance=843750.0,
                    surface_tension_compliance=1.0 / 225.0,
                    surface_distance_compliance=40.0,
                    interior_distance_compliance=180.0,
                    surface_viscosity=0.5,
                    interior_viscosity=0.5,
                    c_init=concentration,
                ),
                surface=gs.surfaces.Default(
                    vis_mode="recon" if is_surface else "particle",
                ),
                name=name,
            )
        )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            pos=ROBOT_POS,
            quat=ROBOT_QUAT,
            file=ROBOT_ASSET,
            convexify=False,
            fixed=True,
        ),
        material=gs.materials.Rigid(
            coup_friction=0.5,
            is_coup_reaction_enabled=False,
        ),
        name="sim1",
    )
    rod = scene.add_entity(
        morph=gs.morphs.Mesh(
            pos=ROD_PARK,
            file=ROD_ASSET,
            convexify=False,
            fixed=True,
        ),
        material=gs.materials.Rigid(
            is_coup_reaction_enabled=False,
        ),
        surface=gs.surfaces.Default(
            color=(0.65, 0.80, 0.90, 0.45),
        ),
        name="glass_rod",
    )
    sponge_pos = (SPONGE_START[0], SPONGE_START[1] + 0.5 * SPONGE_SIZE[1], SPONGE_START[2])
    if is_motion_only:
        sponge = scene.add_entity(
            morph=gs.morphs.Box(
                pos=sponge_pos,
                fixed=True,
                size=SPONGE_SIZE,
            ),
            material=gs.materials.Rigid(
                needs_coup=False,
            ),
            surface=gs.surfaces.Default(
                color=(0.95, 0.68, 0.12),
            ),
            name="sponge",
        )
    else:
        sponge_vertices, sponge_elements = element_utils.create_tetrahedral_grid(
            lower=tuple(-0.5 * size for size in SPONGE_SIZE),
            upper=tuple(0.5 * size for size in SPONGE_SIZE),
            resolution=(10, 7, 20),
        )
        sponge = scene.add_entity(
            morph=gs.morphs.TetrahedralMesh(
                pos=sponge_pos,
                vertices=sponge_vertices,
                elements=sponge_elements,
            ),
            material=gs.materials.PBD.Elastic(
                rho=30.0,
                stretch_relaxation=0.25,
                volume_relaxation=0.15,
            ),
            surface=gs.surfaces.Default(
                color=(0.95, 0.68, 0.12),
            ),
            name="sponge",
        )
    camera = None
    if is_recording:
        camera = scene.add_camera(
            res=(960, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            up=(0.0, 1.0, 0.0),
            fov=70,
            GUI=False,
        )
    scene.build()
    hands_link = tuple(robot.get_link(name) for name in ("right_link26", "left_link16"))
    arms_dofs_idx = tuple(
        tuple(robot.get_joint(f"{side}_joint{idx}").dofs_idx_local[0] for idx in range(start, start + 6))
        for side, start in (("right", 21), ("left", 11))
    )
    hands_fingers_qs_idx = tuple(
        tuple(robot.get_joint(f"{side}_joint{idx}").qs_idx_local[0] for idx in indices)
        for side, indices in (("right", (27, 28)), ("left", (17, 18)))
    )
    qpos = robot.get_qpos()
    for dofs in arms_dofs_idx:
        qpos[list(dofs)] = qpos.new_tensor((0.0, 3.0, 1.5, -1.4, -1.4, 0.0))
    qpos[list(arms_dofs_idx[1])] = qpos.new_tensor((0.5, 2.8, 1.3, -1.5, 1.0, 0.0))
    demo = CoffeeWaterScene(
        scene,
        liquids[0] if liquids else None,
        liquids[1] if liquids else None,
        cups[1],
        camera,
        cup_cavity,
        robot,
        rod,
        sponge,
        hands_link,
        arms_dofs_idx,
        hands_fingers_qs_idx,
        tuple(robot.get_link(name) for name in ("right_link27", "right_link28")),
        qpos,
        is_motion_only,
    )
    update_motion(demo, time=0.0)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, default=1500)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--vis", dest="is_viewer_shown", action="store_true")
    parser.add_argument("--record", dest="is_recording", action="store_true")
    parser.add_argument("--surface", dest="is_surface", action="store_true")
    parser.add_argument("--check-motion", dest="is_motion_only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("out/pbstf_coffee_water"))
    args = parser.parse_args()
    if args.scale <= 0 or (args.steps is not None and args.steps <= 0):
        parser.error("--scale and --steps must be positive")
    gs.init(backend=gs.cuda, precision="32", logging_level="info")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    demo = build_scene(args.scale, args.is_viewer_shown, args.is_recording, args.is_surface, args.is_motion_only)
    dt = demo.scene.sim_options.dt
    steps = args.steps if args.steps is not None else round(MOTION_END / dt)
    initial_mass = None
    if demo.coffee is not None:
        initial_mass = tensor_to_array(demo.coffee.get_mass() + demo.water.get_mass()).sum()
    if demo.camera is not None:
        demo.camera.start_recording(save_to_filename=str(output / "coffee-water.mp4"), fps=50)
    previous_phase = None
    checkpoint_steps = {round(time / dt) - 1 for time in (3.0, 4.0, 13.0, 14.0, 20.0, 22.0)}
    try:
        with (output / "metrics.csv").open("w", newline="", encoding="ascii") as metrics:
            writer = csv.writer(metrics)
            writer.writerow(
                (
                    "step",
                    "time",
                    "phase",
                    "ik_error_m",
                    "stir_turns",
                    "active_particles",
                    "mass_kg",
                    "coffee_amount",
                    "variance",
                    "water_in_coffee_cup",
                    "absorbed_particles",
                    "sponge_wetness",
                )
            )
            for step in range(steps):
                time = (step + 1) * dt
                target, ik_error = update_motion(demo, time)
                check_contacts(demo, time)
                demo.scene.step()
                is_checkpoint = target.phase != previous_phase or step + 1 == steps or step in checkpoint_steps
                if is_checkpoint:
                    gs.logger.info(f"{time:.3f}s: {target.phase.name.lower()}")
                previous_phase = target.phase
                if demo.camera is not None and is_checkpoint:
                    rgb, *_ = demo.camera.render()
                    Image.fromarray(rgb).save(output / f"stage-{step:05d}-{target.phase.name.lower()}.png")
                if (step + 1) % 100 == 0 or step + 1 == steps or is_checkpoint:
                    motion_row = (step + 1, time, target.phase.name, ik_error, target.stir_angle / (2.0 * math.pi))
                    if args.is_motion_only:
                        writer.writerow(motion_row + (None,) * 7)
                        metrics.flush()
                        continue
                    state = demo.scene.pbstf_solver.get_state(demo.scene.sim.cur_substep_local)
                    concentrations = tensor_to_array(state.c)
                    positions = tensor_to_array(state.pos)
                    velocities = tensor_to_array(state.vel)
                    if not all(np.isfinite(values).all() for values in (positions, velocities, concentrations)):
                        raise RuntimeError("Liquid state contains non-finite values.")
                    mass = tensor_to_array(demo.coffee.get_mass() + demo.water.get_mass()).sum()
                    if abs(mass - initial_mass) > initial_mass * 2e-6:
                        raise RuntimeError("Liquid mass changed during the simulation.")
                    demo.scene.pbstf_solver.check_errno()
                    is_absorbed = tensor_to_array(state.absorbed_collider_idx) >= 0
                    wetness = tensor_to_array(demo.scene.pbstf_solver.get_static_collider_wetness(4))
                    if args.is_recording and is_checkpoint:
                        pos, quat = water_cup_pose(time)
                        np.savez_compressed(
                            output / f"state-{step:05d}.npz",
                            pos=positions,
                            vel=velocities,
                            c=concentrations,
                            cup_pos=pos,
                            cup_quat=quat,
                            robot_qpos=tensor_to_array(demo.qpos),
                            rod_pos=target.rod_pos,
                            sponge_pos=tensor_to_array(demo.sponge.get_particles_pos()),
                            absorbed=is_absorbed,
                            wetness=wetness,
                        )
                    water_pos = positions[:, demo.water.particle_start : demo.water.particle_end] - COFFEE_CUP_POS
                    is_in_cup = demo.cup_cavity.contains(water_pos.reshape((-1, 3)))
                    writer.writerow(
                        motion_row
                        + (
                            tensor_to_array(state.active).sum(),
                            mass,
                            concentrations.sum(),
                            concentrations.var(),
                            is_in_cup.mean(),
                            is_absorbed.sum(),
                            wetness.mean(),
                        )
                    )
                    metrics.flush()
            if not args.is_motion_only and steps * dt >= MOTION_END and not is_absorbed.any():
                raise RuntimeError("The completed wiping stroke absorbed no liquid; inspect the spill and sponge path.")
    finally:
        if demo.camera is not None:
            demo.camera.stop_recording()


if __name__ == "__main__":
    main()
