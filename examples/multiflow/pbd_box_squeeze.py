"""Clamp and wipe with the coffee demo's robot, tabletop, and deformable sponge.

Run with python -m examples.multiflow.pbd_box_squeeze -g -v. Unified position-based dynamics (PBD) is the default;
--solver original selects the separate stretch/volume solver. --solver fem selects implicit corotated finite element
method (FEM) elasticity on a coarser mesh. The right hand starts at the sponge, closes for
0.6 seconds, settles until 1.0 second, then wipes 0.34 meters in 4 seconds and holds. Measured sponge tracking error
above 1 cm stops the example. --check-penetration measures vertex intrusion against the hand and fingers and unified
PBD surface crossings against every coupled rigid geom, including the table, with extra geometry queries.
With --vis, Space plays/pauses and Left/Right browse simulation time steps. For unified PBD, T switches to individual
solver iterations with colored collision edges, recorded after point/face projection and before edge projection.
T switches back to completed time steps with plain meshes. D cycles wireframe modes. The viewer starts paused
in time-step mode and keeps visited steps and iterations in memory.
"""

import argparse
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from colorsys import hsv_to_rgb
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue

import numpy as np

import igl
from scipy.spatial.transform import Rotation

# Select this checkout when the editable installation points elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import genesis as gs
from genesis.ext.pyrender.jit_render import RenderBuffer
from genesis.utils import geom
from genesis.utils.element import create_tetrahedral_grid
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, Keybind


@dataclass(frozen=True)
class ContactMesh:
    geom_idx: int
    vertices: np.ndarray
    faces: np.ndarray
    bounds: np.ndarray


@dataclass(frozen=True)
class SqueezeFrame:
    step: int
    substep: int
    iteration: int
    buffers: tuple[RenderBuffer, ...]
    vertices: np.ndarray | None
    edges_idx: np.ndarray
    edge_positions: np.ndarray


class SqueezePlayback:
    """Browse recorded render frames while retaining the latest physics and robot controller state."""

    def __init__(self, scene, sponge, is_unified):
        self.scene = scene
        self.is_unified = is_unified
        self.requests: SimpleQueue[Key] = SimpleQueue()
        self.frames: list[SqueezeFrame] = []
        self.step_frames: list[SqueezeFrame] = []
        self.markers = []
        self.frame = 0
        self.is_paused = True
        self.is_iteration_playback = False
        self.iterations_per_step = scene.sim.substeps * scene.pbd_options.max_solver_iterations if is_unified else 1
        self.sponge_node = scene.visualizer.context.static_nodes[(0, sponge.uid)] if is_unified else None
        self.rigid_geoms = tuple(
            geom
            for entity in scene.rigid_solver.entities
            for link in entity.links
            for geom in (link.vgeoms if entity.surface.vis_mode == "visual" else link.geoms)
            if geom.uid in scene.visualizer.context.rigid_nodes
        )
        self.rigid_nodes = tuple(scene.visualizer.context.rigid_nodes[geom.uid] for geom in self.rigid_geoms)
        self.rigid_links_idx = np.array([geom.link.idx for geom in self.rigid_geoms])
        self.rigid_geoms_pos = np.stack([geom.init_pos for geom in self.rigid_geoms])
        self.rigid_geoms_quat = np.stack([geom.init_quat for geom in self.rigid_geoms])
        scene.viewer.register_keybinds(
            Keybind(name="play_pause", key=Key.SPACE, callback=self.requests.put, args=(Key.SPACE,)),
            Keybind(name="previous_frame", key=Key.LEFT, callback=self.requests.put, args=(Key.LEFT,)),
            Keybind(name="next_frame", key=Key.RIGHT, callback=self.requests.put, args=(Key.RIGHT,)),
        )
        if is_unified:
            scene.viewer.register_keybinds(
                Keybind(name="toggle_iteration_playback", key=Key.T, callback=self.requests.put, args=(Key.T,))
            )
            gs.logger.info("T: switch between time steps and solver iterations with colored collision edges.")
        gs.logger.info(
            "Space: play/pause. Left/Right: previous/next frame in the current mode. D: wireframe. Close the viewer to exit."
        )

    def record_frame(self, step):
        """Retain a completed step and its iterations, then display the first new frame in the current mode."""
        first_frame = len(self.frames) if self.is_iteration_playback else len(self.step_frames)
        buffers = self.scene.viewer.capture_frame()
        edges_idx = np.empty(0, dtype=gs.np_int)
        edge_positions = np.empty((0, 2, 3), dtype=gs.np_float)
        step_frame = SqueezeFrame(step, 0, 0, buffers, None, edges_idx, edge_positions)
        self.step_frames.append(step_frame)
        if self.is_unified and step:
            history = self.scene.pbd_solver.get_edge_contact_history()
            vertices = tensor_to_array(self.scene.pbd_solver.get_visual_vertex_positions(history.positions))
            positions = tensor_to_array(history.positions)
            is_detected = tensor_to_array(history.is_detected)
            edges = tensor_to_array(history.edges)
            links_pos = tensor_to_array(history.links_pos)
            links_quat = tensor_to_array(history.links_quat)
            for substep in range(self.scene.sim.substeps):
                geoms_pos, geoms_quat = geom.transform_pos_quat_by_trans_quat(
                    self.rigid_geoms_pos,
                    self.rigid_geoms_quat,
                    links_pos[substep, 0, self.rigid_links_idx],
                    links_quat[substep, 0, self.rigid_links_idx],
                )
                geoms_T = geom.trans_quat_to_T(geoms_pos, geoms_quat)
                rigid_buffers = tuple(
                    RenderBuffer(node.mesh.primitives[0], "model", transform[None].transpose((0, 2, 1)))
                    for node, transform in zip(self.rigid_nodes, geoms_T)
                )
                substep_buffers = (*buffers, *rigid_buffers)
                for iteration in range(self.scene.pbd_options.max_solver_iterations):
                    edges_idx = np.flatnonzero(is_detected[substep, iteration, 0])
                    edge_positions = positions[substep, iteration, 0, edges[edges_idx]]
                    self.frames.append(
                        SqueezeFrame(
                            step,
                            substep,
                            iteration,
                            substep_buffers,
                            vertices[substep, iteration, 0],
                            edges_idx,
                            edge_positions,
                        )
                    )
        else:
            self.frames.append(step_frame)
        self.show_frame(first_frame)

    def show_frame(self, frame):
        """Display a stored time step or iteration, highlighting collision edges in iteration mode."""
        self.frame = frame
        saved = (self.frames if self.is_iteration_playback else self.step_frames)[frame]
        for marker in self.markers:
            self.scene.clear_debug_object(marker)
        self.markers.clear()
        if self.is_iteration_playback:
            for edge_idx, (start, end) in zip(saved.edges_idx, saved.edge_positions):
                color = (*hsv_to_rgb(h=(edge_idx * 0.618033988749895) % 1.0, s=0.85, v=1.0), 1.0)
                self.markers.append(self.scene.draw_debug_line(start, end, radius=0.00018, color=color))
        buffers = saved.buffers
        if saved.vertices is not None:
            context = self.scene.visualizer.context
            positions = context._scene.reorder_vertices(self.sponge_node, saved.vertices)
            normals = context.jit.update_normal(self.sponge_node, positions)
            buffers = (*buffers, RenderBuffer(self.sponge_node.mesh.primitives[0], "pos", positions))
            if normals is not None:
                buffers = (*buffers, RenderBuffer(self.sponge_node.mesh.primitives[0], "normal", normals))
        frame_time = saved.step * self.scene.dt
        if self.is_iteration_playback and saved.step:
            frame_time = (saved.step - 1 + (saved.substep + 1) / self.scene.sim.substeps) * self.scene.dt
        mode = "Iterations" if self.is_iteration_playback else "Time steps"
        label = f"{mode} | Step {saved.step} | t={frame_time:.4f}s"
        if self.is_iteration_playback and saved.step:
            label += f" | Substep {saved.substep + 1} | Iteration {saved.iteration + 1} | Edges {len(saved.edges_idx)}"
        if self.is_unified:
            label += " | T: switch mode"
        self.scene.viewer.show_frame(buffers, label=label)

    def advance(self, n_steps):
        """Pump viewer events until a new physics frame is requested or the window closes."""
        while self.scene.viewer.is_alive():
            frame_step = 0 if self.is_paused else 1
            if not self.requests.empty():
                key = self.requests.get()
                if key == Key.T:
                    self.is_iteration_playback = not self.is_iteration_playback
                    if self.is_iteration_playback:
                        next_frame = (self.frame - 1) * self.iterations_per_step + 1 if self.frame else 0
                    else:
                        next_frame = self.frames[self.frame].step
                    self.show_frame(next_frame)
                    continue
                if key == Key.SPACE:
                    self.is_paused = not self.is_paused
                    frame_step = 0
                else:
                    self.is_paused = True
                    frame_step = -1 if key == Key.LEFT else 1
            last_frame = n_steps * self.iterations_per_step if self.is_iteration_playback else n_steps
            next_frame = min(max(self.frame + frame_step, 0), last_frame)
            n_frames = len(self.frames) if self.is_iteration_playback else len(self.step_frames)
            if next_frame == n_frames:
                self.scene.viewer.show_frame(None)
                return True
            if next_frame != self.frame:
                self.show_frame(next_frame)
            else:
                self.scene.viewer.update()
            if self.frame == last_frame:
                self.is_paused = True
            time.sleep(0.01)
        return False


def get_sponge_positions(sponge):
    """Return simulation vertex positions with shape [n_vertices, 3] in this single-environment scene."""
    if isinstance(sponge, gs.engine.entities.FEMEntity):
        return sponge.get_state().pos[0]
    return sponge.get_particles_pos()


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
    if not np.isfinite(error).all() or np.linalg.norm(error[..., :3]) > 2e-4 or np.linalg.norm(error[..., 3:]) > 2e-4:
        raise RuntimeError(f"Cannot reach {link.name} tool pose: {error}.")
    return qpos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", dest="is_viewer_shown", action="store_true")
    parser.add_argument("-g", "--gpu", dest="is_gpu_enabled", action="store_true")
    parser.add_argument(
        "--solver",
        choices=("original", "unified", "fem"),
        default="unified",
        help="Sponge deformation: original or unified PBD on a finer mesh, or implicit FEM on a coarser mesh",
    )
    parser.add_argument("--dt", type=float, default=0.002, help="Simulation step duration in seconds")
    parser.add_argument("--substeps", type=int, default=1, help="More substeps resolve contacts at extra cost")
    parser.add_argument("-t", "--seconds", type=float, default=5.5, help="Duration; wiping finishes at 5 seconds")
    parser.add_argument(
        "--check-penetration",
        dest="is_penetration_checked",
        action="store_true",
        help="Measure hand/sponge vertex intrusion and unified surface crossings every 0.1 seconds; adds geometry queries",
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
        Rotation.from_euler("x", 15, degrees=True) * Rotation.from_matrix(((0, -1, 0), (-1, 0, 0), (0, 0, -1)))
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
        fem_options=(
            gs.options.FEMOptions(
                # The tabletop supports the sponge in this Y-up scene; the solver floor is Z-aligned.
                floor_height=-1.0,
                use_implicit_solver=True,
                n_pcg_iterations=120,
            )
            if args.solver == "fem"
            else None
        ),
        pbd_options=(
            gs.options.PBDUnifiedOptions(
                particle_size=0.005,
                lower_bound=(-0.4, -1.0 / 15.0, -4.0 / 15.0),
                upper_bound=(0.4, 8.0 / 15.0, 4.0 / 15.0),
                max_solver_iterations=30,
                constraint_acceleration=0.0,
                collision_thickness=5e-5,
                contact_detection_distance=1.5e-4,
                is_recording_edge_contacts=args.is_viewer_shown,
            )
            if args.solver == "unified"
            else gs.options.PBDOptions(
                particle_size=0.005,
                lower_bound=(-0.4, -1.0 / 15.0, -4.0 / 15.0),
                upper_bound=(0.4, 8.0 / 15.0, 4.0 / 15.0),
                max_stretch_solver_iterations=30,
                max_volume_solver_iterations=30,
            )
            if args.solver == "original"
            else None
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
        resolution=(4, 6, 8) if args.solver == "fem" else (10, 7, 20),
    )
    sponge = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=(sponge_start[0], sponge_start[1] + 0.5 * sponge_size[1], sponge_start[2]),
            vertices=vertices,
            elements=elements,
        ),
        material=(
            gs.materials.FEM.Elastic(E=1e4, rho=30.0, model="linear_corotated")
            if args.solver == "fem"
            else gs.materials.PBD.Elastic(rho=30.0, stretch_relaxation=0.1, volume_relaxation=0.1)
        ),
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
    surface_faces = sponge.surface_triangles.astype(np.int64) if args.is_penetration_checked else None
    measurement_stride = max(1, round(0.1 / scene.dt))
    steps = math.ceil(args.seconds / scene.dt)
    max_penetration = 0.0
    max_penetrating_vertices = 0
    max_rigid_penetration = 0.0
    max_penetrating_rigid_vertices = 0
    max_edge_penetration = 0.0
    max_edge_contacts = 0
    max_triangle_intersections = 0
    max_unpaired_intersections = 0
    sponge_wipe_start = None
    max_tracking_error = 0.0
    wipe_delta = np.zeros(3)
    playback = None
    if args.is_viewer_shown and "PYTEST_VERSION" not in os.environ:
        playback = SqueezePlayback(scene, sponge, is_unified=args.solver == "unified")
        playback.record_frame(step=0)
    step = 0
    while step < steps or playback is not None:
        if playback is not None and not playback.advance(steps):
            break
        time = (step + 1) * scene.dt
        fraction = min(time / 0.6, 1.0)
        progress = fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
        robot.control_dofs_position(0.028 - 0.01 * progress, dofs_idx_local=fingers_dofs_idx[0])
        if time >= wipe_start:
            if sponge_wipe_start is None:
                sponge_wipe_start = tensor_to_array(get_sponge_positions(sponge).mean(dim=-2))
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
        scene.step(update_visualizer=playback is None)
        scene.rigid_solver.check_errno()
        if playback is not None:
            playback.record_frame(step + 1)
        step += 1
        if step % measurement_stride != 0 and step != steps:
            continue
        positions = tensor_to_array(
            get_sponge_positions(sponge), dtype=np.float64 if args.is_penetration_checked else None
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
            depths = np.zeros(len(positions))
            rigid_depth = 0.0
            rigid_count = 0
            sponge_lower = positions.min(axis=0)
            sponge_upper = positions.max(axis=0)
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
                vertices_world = geom.transform_by_trans_quat(contact_mesh.vertices, pos, quat)
                is_candidate = ((vertices_world >= sponge_lower) & (vertices_world <= sponge_upper)).all(axis=-1)
                if is_candidate.any():
                    candidates = vertices_world[is_candidate]
                    distances, _, _ = igl.point_mesh_squared_distance(candidates, positions, surface_faces)
                    winding = igl.winding_number(positions, surface_faces, candidates)
                    rigid_depths = np.where(np.abs(winding) > 0.5, np.sqrt(distances), 0.0)
                    rigid_count += np.count_nonzero(rigid_depths > 1e-6)
                    rigid_depth = max(rigid_depth, rigid_depths.max())
            count = np.count_nonzero(depths > 1e-6)
            surface_count = np.count_nonzero(depths[surface_indices] > 1e-6)
            max_penetration = max(max_penetration, depths.max())
            max_penetrating_vertices = max(max_penetrating_vertices, count)
            max_rigid_penetration = max(max_rigid_penetration, rigid_depth)
            max_penetrating_rigid_vertices = max(max_penetrating_rigid_vertices, rigid_count)
            tetrahedra = positions[sponge.elems]
            volumes = np.linalg.det(tetrahedra[:, 1:, :] - tetrahedra[:, :1, :]) / 6.0
            gs.logger.info(
                f"{time:.3f}s: penetrating vertices={count}/{len(positions)}, "
                f"surface vertices={surface_count}, depth={1000 * depths.max():.6f}mm; "
                f"rigid penetrating vertices={rigid_count}, depth={1000 * rigid_depth:.6f}mm; "
                f"minimum tetrahedron volume={volumes.min():.6e}m^3."
            )
            if args.solver == "unified":
                diagnostics = scene.pbd_solver.get_collision_diagnostics()
                edge_penetration = diagnostics.edge_penetration[0].item()
                edge_contacts = diagnostics.edge_contacts[0].item()
                triangle_intersections = diagnostics.triangle_intersections[0].item()
                unpaired_intersections = diagnostics.unpaired_intersections[0].item()
                max_edge_penetration = max(max_edge_penetration, edge_penetration)
                max_edge_contacts = max(max_edge_contacts, edge_contacts)
                max_triangle_intersections = max(max_triangle_intersections, triangle_intersections)
                max_unpaired_intersections = max(max_unpaired_intersections, unpaired_intersections)
                gs.logger.info(
                    f"{time:.3f}s: surface triangle intersections={triangle_intersections}, "
                    f"edge contacts={edge_contacts}, edge gap={1000 * edge_penetration:.6f}mm, "
                    f"unpaired intersections={unpaired_intersections}."
                )
    if sponge_wipe_start is not None:
        gs.logger.info(f"Maximum sponge tracking error: {1000 * max_tracking_error:.3f}mm.")
    if args.is_penetration_checked:
        gs.logger.info(
            f"Peak vertex penetration: {1000 * max_penetration:.6f}mm; "
            f"maximum penetrating vertices: {max_penetrating_vertices}; "
            f"peak rigid vertex penetration: {1000 * max_rigid_penetration:.6f}mm; "
            f"maximum penetrating rigid vertices: {max_penetrating_rigid_vertices}."
        )
        if args.solver == "unified":
            gs.logger.info(
                f"Peak crossed-edge gap: {1000 * max_edge_penetration:.6f}mm; "
                f"maximum edge contacts: {max_edge_contacts}; "
                f"maximum surface triangle intersections: {max_triangle_intersections}; "
                f"maximum unpaired intersections: {max_unpaired_intersections}."
            )


if __name__ == "__main__":
    main()
