"""Clamp and wipe with the coffee demo's robot, tabletop, and deformable sponge.

Run with python -m examples.multiflow.pbd_box_squeeze -g -v. Unified position-based dynamics (PBD) is the default;
--solver original selects the separate stretch/volume solver. The right hand starts at the sponge, closes for
0.6 seconds, settles until 1.0 second, then wipes 0.34 meters in 4 seconds and holds. Measured sponge tracking error
above 1 cm stops the example. --check-penetration measures vertex intrusion with extra CPU geometry work.
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import igl
from scipy.spatial.transform import Rotation

# Select this checkout when the editable installation points elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import genesis as gs
from genesis.utils import geom
from genesis.utils.element import create_tetrahedral_grid
from genesis.utils.misc import tensor_to_array


@dataclass(frozen=True)
class ContactMesh:
    geom_idx: int
    vertices: np.ndarray
    faces: np.ndarray
    bounds: np.ndarray


def solve_tool_pose(robot, link, dofs_idx_local, pos, quat, local_point, qpos):
    """Solve a tool-center pose and reject unreachable or non-finite inverse kinematics targets."""
    qpos, error = robot.inverse_kinematics(
        link=link,
        pos=pos,
        quat=quat,
        local_point=local_point,
        init_qpos=qpos,
        max_samples=1,
        max_solver_iters=100,
        pos_tol=1e-6,
        rot_tol=1e-6,
        dofs_idx_local=dofs_idx_local,
        return_error=True,
    )
    error = tensor_to_array(error)
    if (
        not np.isfinite(error).all()
        or np.linalg.norm(error[..., :3]) > 2e-4
        or np.linalg.norm(error[..., 3:]) > 2e-4
    ):
        raise RuntimeError(f"Cannot reach {link.name} tool pose: {error}.")
    return qpos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", dest="is_viewer_shown", action="store_true")
    parser.add_argument("-g", "--gpu", dest="is_gpu_enabled", action="store_true")
    parser.add_argument("--solver", choices=("original", "unified"), default="unified")
    parser.add_argument("--dt", type=float, default=0.002, help="Simulation step duration in seconds")
    parser.add_argument("--substeps", type=int, default=1, help="More substeps resolve contacts at extra cost")
    parser.add_argument("-t", "--seconds", type=float, default=5.5, help="Duration; wiping finishes at 5 seconds")
    parser.add_argument(
        "--check-penetration",
        dest="is_penetration_checked",
        action="store_true",
        help="Measure vertices inside the right-hand meshes every 0.1 seconds; adds CPU geometry queries",
    )
    args = parser.parse_args()
    if not math.isfinite(args.dt) or args.dt <= 0 or args.substeps <= 0:
        parser.error("--dt and --substeps must be positive and finite")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    if "PYTEST_VERSION" in os.environ:
        args.seconds = min(args.seconds, 0.02)

    gs.init(backend=gs.gpu if args.is_gpu_enabled else gs.cpu, seed=0, precision="32")
    sponge_size = (0.04, 0.06, 0.08)
    sponge_start = (0.0, -0.044, -0.17)
    wipe_displacement = np.array((0.0, 0.0, 0.34))
    wipe_start = 1.0
    wipe_duration = 4.0
    tool_center = (0.14747, 0.001786, 0.0)
    grip_quat = (
        Rotation.from_euler("x", 15, degrees=True)
        * Rotation.from_matrix(((0, -1, 0), (-1, 0, 0), (0, 0, -1)))
    ).as_quat(scalar_first=True)
    left_quat = (
        Rotation.from_euler("y", -20, degrees=True)
        * Rotation.from_euler("z", -30, degrees=True)
        * Rotation.from_matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
    ).as_quat(scalar_first=True)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
            gravity=(0.0, -9.8, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.implicitfast,
            enable_neutral_collision=True,
            noslip_iterations=10,
            enable_torsional_friction=True,
            enable_rolling_friction=True,
            constraint_timeconst=0.004,
        ),
        pbd_options=(
            gs.options.PBDUnifiedOptions(
                particle_size=0.005,
                lower_bound=(-0.4, -1.0 / 15.0, -4.0 / 15.0),
                upper_bound=(0.4, 8.0 / 15.0, 4.0 / 15.0),
                max_solver_iterations=30,
                constraint_acceleration=0.85,
            )
            if args.solver == "unified"
            else gs.options.PBDOptions(
                particle_size=0.005,
                lower_bound=(-0.4, -1.0 / 15.0, -4.0 / 15.0),
                upper_bound=(0.4, 8.0 / 15.0, 4.0 / 15.0),
                max_stretch_solver_iterations=30,
                max_volume_solver_iterations=30,
            )
        ),
        vis_options=gs.options.VisOptions(
            ambient_light=(0.4, 0.4, 0.4),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.40, 0.38, -0.30),
            camera_lookat=(0.0, -0.005, 0.0),
            camera_up=(0.0, 1.0, 0.0),
            camera_fov=50,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.is_viewer_shown,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(
            fixed=True,
            lower=(-0.4, -0.065, -4.0 / 15.0),
            upper=(0.4, -0.045, 4.0 / 15.0),
        ),
        material=gs.materials.Rigid(friction=3.0, coup_friction=0.0, is_coup_reaction_enabled=False),
        surface=gs.surfaces.Default(color=(0.36, 0.24, 0.14)),
        name="table",
    )
    robot_path = Path(gs.utils.get_assets_dir()) / "urdf/sim1_acone/acone_collision.urdf"
    robot_xml = ET.parse(robot_path).getroot()
    # Finger contacts use the displayed surface and its authored local transform.
    for link in robot_xml.findall("link"):
        if link.attrib["name"] in ("right_link27", "right_link28"):
            visual = link.find("visual")
            collision = link.find("collision")
            for tag in ("origin", "geometry"):
                collision.remove(collision.find(tag))
                collision.append(deepcopy(visual.find(tag)))
    for mesh_node in robot_xml.iter("mesh"):
        mesh_node.set("filename", str(robot_path.parent / mesh_node.attrib["filename"]))
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            pos=(-0.06, -0.55, 0.52),
            quat=(0.5, -0.5, 0.5, 0.5),
            file=ET.tostring(robot_xml, encoding="unicode"),
            convexify=False,
            fixed=True,
        ),
        material=gs.materials.Rigid(
            coup_friction=0.5,
            is_coup_reaction_enabled=False,
            sdf_cell_size=0.0005,
            gravity_compensation=1.0,
        ),
        name="sim1",
    )
    vertices, elements = create_tetrahedral_grid(
        lower=tuple(-0.5 * size for size in sponge_size),
        upper=tuple(0.5 * size for size in sponge_size),
        resolution=(10, 7, 20),
    )
    sponge = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=(sponge_start[0], sponge_start[1] + 0.5 * sponge_size[1], sponge_start[2]),
            vertices=vertices,
            elements=elements,
        ),
        material=gs.materials.PBD.Elastic(rho=30.0, stretch_relaxation=0.25, volume_relaxation=0.15),
        surface=gs.surfaces.Default(color=(0.95, 0.68, 0.12), smooth=False),
        name="sponge",
    )
    scene.build()
    hands = tuple(robot.get_link(name) for name in ("right_link26", "left_link16"))
    arms_dofs_idx = tuple(
        tuple(robot.get_joint(f"{side}_joint{idx}").dofs_idx_local[0] for idx in range(start, start + 6))
        for side, start in (("right", 21), ("left", 11))
    )
    fingers_dofs_idx = tuple(
        tuple(robot.get_joint(f"{side}_joint{idx}").dofs_idx_local[0] for idx in indices)
        for side, indices in (("right", (27, 28)), ("left", (17, 18)))
    )
    fingers_qs_idx = tuple(
        tuple(robot.get_joint(f"{side}_joint{idx}").qs_idx_local[0] for idx in indices)
        for side, indices in (("right", (27, 28)), ("left", (17, 18)))
    )
    right_fingers = tuple(robot.get_link(f"right_link{idx}") for idx in (27, 28))
    for dofs in arms_dofs_idx:
        robot.set_dofs_kp(kp=[5000, 5000, 5000, 1500, 1500, 1500], dofs_idx_local=dofs)
        robot.set_dofs_kv(kv=[100, 100, 100, 30, 30, 30], dofs_idx_local=dofs)
        robot.set_dofs_force_range(lower=[-27, -27, -27, -7, -7, -7], upper=[27, 27, 27, 7, 7, 7], dofs_idx_local=dofs)
    for dofs in fingers_dofs_idx:
        robot.set_dofs_kp(kp=3000, dofs_idx_local=dofs)
        robot.set_dofs_kv(kv=30, dofs_idx_local=dofs)
        robot.set_dofs_force_range(lower=-20, upper=20, dofs_idx_local=dofs)
    for link in right_fingers:
        link.set_friction(2.0)
        link.set_friction_torsional(0.001)
    scene.rigid_solver.set_geoms_friction_rolling(friction_rolling=0.0)
    contact_geoms_idx = tuple(
        collision_geom.idx for link in (*right_fingers, table.base_link) for collision_geom in link.geoms
    )
    scene.rigid_solver.set_sol_params(
        sol_params=[0.004, 1.0, 0.99, 0.999, 0.001, 0.5, 2.0], geoms_idx=contact_geoms_idx
    )
    qpos = robot.get_qpos()
    qpos[list(arms_dofs_idx[0])] = qpos.new_tensor((0.0, 3.0, 1.5, -1.4, -1.4, 0.0))
    qpos[list(arms_dofs_idx[1])] = qpos.new_tensor((0.5, 2.8, 1.3, -1.5, 1.0, 0.0))
    qpos[list(fingers_qs_idx[0])] = 0.028
    qpos[list(fingers_qs_idx[1])] = 0.0007
    grip_pos = np.array(sponge_start) + (0.0, sponge_size[1] - 0.005, 0.0)
    for hand, dofs, pos, quat in zip(hands, arms_dofs_idx, (grip_pos, (-0.19, 0.125, 0.0)), (grip_quat, left_quat)):
        qpos = solve_tool_pose(robot, hand, dofs, pos, quat, tool_center, qpos)
    robot.set_qpos(qpos)
    for dofs in arms_dofs_idx:
        robot.control_dofs_position_velocity(qpos[list(dofs)], velocity=0.0, dofs_idx_local=dofs)
    robot.control_dofs_position(0.0007, dofs_idx_local=fingers_dofs_idx[1])

    contact_meshes = []
    if args.is_penetration_checked:
        for link in (hands[0], *right_fingers):
            for collision_geom in link.geoms:
                # libigl requires float64 vertices and int64 face indices.
                mesh_vertices = collision_geom.init_verts.astype(np.float64)
                contact_meshes.append(
                    ContactMesh(
                        collision_geom.idx,
                        mesh_vertices,
                        collision_geom.init_faces.astype(np.int64),
                        np.stack((mesh_vertices.min(axis=0), mesh_vertices.max(axis=0))),
                    )
                )
    geoms_idx = tuple(contact_mesh.geom_idx for contact_mesh in contact_meshes)
    surface_indices = np.unique(sponge.surface_triangles)
    measurement_stride = max(1, round(0.1 / scene.dt))
    steps = math.ceil(args.seconds / scene.dt)
    max_penetration = 0.0
    max_penetrating_vertices = 0
    sponge_wipe_start = None
    max_tracking_error = 0.0
    wipe_delta = np.zeros(3)
    for step in range(steps):
        time = (step + 1) * scene.dt
        fraction = min(time / 0.6, 1.0)
        progress = fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
        robot.control_dofs_position(0.028 - 0.01 * progress, dofs_idx_local=fingers_dofs_idx[0])
        if time >= wipe_start:
            if sponge_wipe_start is None:
                sponge_wipe_start = tensor_to_array(sponge.get_particles_pos().mean(dim=-2))
                gs.logger.info(f"{time:.3f}s: start wiping.")
            fraction = min((time - wipe_start) / wipe_duration, 1.0)
            progress = fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
            wipe_delta = progress * wipe_displacement
            target_qpos = solve_tool_pose(
                robot, hands[0], arms_dofs_idx[0], grip_pos + wipe_delta, grip_quat, tool_center, qpos
            )
            velocity = (target_qpos[list(arms_dofs_idx[0])] - qpos[list(arms_dofs_idx[0])]) / scene.dt
            robot.control_dofs_position_velocity(
                target_qpos[list(arms_dofs_idx[0])], velocity, dofs_idx_local=arms_dofs_idx[0]
            )
            qpos = target_qpos
        scene.step()
        scene.rigid_solver.check_errno()
        if (step + 1) % measurement_stride != 0 and step + 1 != steps:
            continue
        positions = tensor_to_array(
            sponge.get_particles_pos(), dtype=np.float64 if args.is_penetration_checked else None
        )
        if not np.isfinite(positions).all():
            raise RuntimeError("Sponge vertices contain non-finite values.")
        if sponge_wipe_start is not None:
            displacement = positions.mean(axis=0) - sponge_wipe_start
            tracking_error = np.linalg.norm(displacement - wipe_delta)
            max_tracking_error = max(max_tracking_error, tracking_error)
            gs.logger.info(
                f"{time:.3f}s: sponge travel={1000 * displacement[2]:.3f}mm, "
                f"target={1000 * wipe_delta[2]:.3f}mm, tracking error={1000 * tracking_error:.3f}mm."
            )
            if tracking_error > 0.01:
                raise RuntimeError(f"The sponge slipped from the gripper at {time:.3f}s: {tracking_error:.6f}m.")
        if args.is_penetration_checked:
            geoms_pos = tensor_to_array(scene.rigid_solver.get_geoms_pos(geoms_idx), dtype=np.float64)
            geoms_quat = tensor_to_array(scene.rigid_solver.get_geoms_quat(geoms_idx), dtype=np.float64)
            depths = np.zeros(sponge.n_particles)
            for contact_mesh, pos, quat in zip(contact_meshes, geoms_pos, geoms_quat):
                local = geom.inv_transform_by_trans_quat(positions, pos, quat)
                is_candidate = ((local >= contact_mesh.bounds[0]) & (local <= contact_mesh.bounds[1])).all(axis=-1)
                if is_candidate.any():
                    distances, _, _ = igl.point_mesh_squared_distance(
                        local[is_candidate], contact_mesh.vertices, contact_mesh.faces
                    )
                    winding = igl.winding_number(contact_mesh.vertices, contact_mesh.faces, local[is_candidate])
                    depths[is_candidate] = np.maximum(
                        depths[is_candidate], np.where(np.abs(winding) > 0.5, np.sqrt(distances), 0.0)
                    )
            count = np.count_nonzero(depths > 1e-6)
            surface_count = np.count_nonzero(depths[surface_indices] > 1e-6)
            max_penetration = max(max_penetration, depths.max())
            max_penetrating_vertices = max(max_penetrating_vertices, count)
            gs.logger.info(
                f"{time:.3f}s: penetrating vertices={count}/{sponge.n_particles}, "
                f"surface vertices={surface_count}, depth={1000 * depths.max():.6f}mm."
            )
    if sponge_wipe_start is not None:
        gs.logger.info(f"Maximum sponge tracking error: {1000 * max_tracking_error:.3f}mm.")
    if args.is_penetration_checked:
        gs.logger.info(
            f"Peak vertex penetration: {1000 * max_penetration:.6f}mm; "
            f"maximum penetrating vertices: {max_penetrating_vertices}."
        )


if __name__ == "__main__":
    main()
