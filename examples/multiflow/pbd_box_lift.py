"""Replay the coffee robot's sponge grasp and pickup with unified position-based dynamics (PBD).

Run with python -m examples.multiflow.pbd_box_lift -g -v. The bundled coffee sponge contains 401 vertices and
1509 tetrahedra, spanning 100 x 73 x 31.8 mm. The Z-up scene uses the coffee lift robot and its recorded joint
position targets from 37.62 to 39.02 seconds. Fingers close over 0.6 seconds and settle until 1 second; the recorded
1.4-second pickup then plays at its original speed, followed by a 0.5-second hold. The sponge moves through contact.
The closure follows tools/coffee_sponge_lift_probe.py's quintic mode with 6 mm indentation per finger.
The rendered tetrahedral boundary exposes the imported discretization. Asset hashes and recording provenance
are in coffee_sponge/source.json.

Sponge tracking error above 1 cm stops the example. --check-penetration measures vertex intrusion and surface
crossings and checks robot self-collision, at the cost of extra geometry queries. --damping controls deformation
damping in 1/s; higher values settle faster but recover slower. With --vis, Space plays/pauses and Left/Right browse
time steps. T switches to solver iterations with colored collision edges; D cycles wireframe modes. The viewer
starts paused and retains visited steps and iterations in memory.
"""

import argparse
import math
import os
import sys
import time
from colorsys import hsv_to_rgb
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
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, Keybind


@dataclass(frozen=True)
class ContactMesh:
    geom_idx: int
    vertices: np.ndarray
    faces: np.ndarray
    bounds: np.ndarray


@dataclass(frozen=True)
class LiftFrame:
    step: int
    substep: int
    iteration: int
    buffers: tuple[RenderBuffer, ...]
    vertices: np.ndarray | None
    edges_idx: np.ndarray
    edge_positions: np.ndarray


class LiftPlayback:
    """Browse recorded render frames while retaining the latest physics and robot controller state."""

    def __init__(self, scene, sponge):
        self.scene = scene
        self.requests: SimpleQueue[Key] = SimpleQueue()
        self.frames: list[LiftFrame] = []
        self.step_frames: list[LiftFrame] = []
        self.markers = []
        self.frame = 0
        self.is_paused = True
        self.is_iteration_playback = False
        self.iterations_per_step = scene.sim.substeps * scene.pbd_options.max_solver_iterations
        self.sponge_node = scene.visualizer.context.static_nodes[(0, sponge.uid)]
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
            Keybind(name="toggle_iteration_playback", key=Key.T, callback=self.requests.put, args=(Key.T,)),
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
        step_frame = LiftFrame(step, 0, 0, buffers, None, edges_idx, edge_positions)
        self.step_frames.append(step_frame)
        if step:
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
                        LiftFrame(
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", dest="is_viewer_shown", action="store_true")
    parser.add_argument("-g", "--gpu", dest="is_gpu_enabled", action="store_true")
    parser.add_argument("--dt", type=float, default=0.002, help="Simulation step duration in seconds")
    parser.add_argument("--substeps", type=int, default=1, help="More substeps resolve contacts at extra cost")
    parser.add_argument(
        "--damping",
        type=float,
        default=80.0,
        help="Unified PBD damping rate in 1/s; higher values settle faster but recover slower",
    )
    parser.add_argument(
        "-t", "--seconds", type=float, default=2.9, help="Duration; the recorded pickup finishes at 2.4 seconds"
    )
    parser.add_argument(
        "--check-penetration",
        dest="is_penetration_checked",
        action="store_true",
        help="Check robot self-collision each step and sponge penetration every 0.1 seconds; adds collision queries",
    )
    args = parser.parse_args()
    if not math.isfinite(args.dt) or args.dt <= 0 or args.substeps <= 0:
        parser.error("--dt and --substeps must be positive and finite")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    if not math.isfinite(args.damping) or args.damping < 0:
        parser.error("--damping must be finite and nonnegative")
    if "PYTEST_VERSION" in os.environ:
        args.seconds = min(args.seconds, 0.02)

    gs.init(backend=gs.gpu if args.is_gpu_enabled else gs.cpu, seed=0, precision="32")
    asset_dir = Path(__file__).with_name("coffee_sponge")
    with np.load(asset_dir / "sponge.npz", allow_pickle=False) as archive:
        vertices, elements = archive["vertices"], archive["elements"]
    with np.load(asset_dir / "pickup.npz", allow_pickle=False) as archive:
        recorded_time = archive["time"]
        joint_names = tuple(archive["joint_names"])
        joint_targets = archive["joint_targets"]
        initial_qpos = archive["initial_qpos"]
        sponge_pos = archive["sponge_pos"]
        sponge_quat = archive["sponge_quat"]
    lift_start = 1.0
    # Rotation changes the centroid of the sponge whose authored origin lies on its bottom face.
    sponge_centers = sponge_pos + Rotation.from_quat(sponge_quat, scalar_first=True).apply(vertices.mean(axis=0))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
            gravity=(0.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.implicitfast,
            enable_self_collision=args.is_penetration_checked,
            noslip_iterations=10,
            enable_torsional_friction=True,
            enable_rolling_friction=True,
            constraint_timeconst=0.004,
        ),
        pbd_options=gs.options.PBDUnifiedOptions(
            particle_size=0.005,
            lower_bound=(-0.8, -0.8, 0.82),
            upper_bound=(0.9, 0.65, 1.8),
            max_solver_iterations=30,
            constraint_acceleration=0.0,
            collision_thickness=5e-5,
            contact_detection_distance=1.5e-4,
            is_recording_edge_contacts=args.is_viewer_shown,
        ),
        vis_options=gs.options.VisOptions(
            ambient_light=(0.4, 0.4, 0.4),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.90, -0.72, 1.48),
            camera_lookat=(0.36, -0.12, 1.0),
            camera_up=(0.0, 0.0, 1.0),
            camera_fov=50,
        ),
        show_viewer=args.is_viewer_shown,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(
            fixed=True,
            lower=(-0.7, -0.4, 0.88),
            upper=(0.7, 0.4, 0.9),
        ),
        material=gs.materials.Rigid(friction=3.0, coup_friction=0.0, is_coup_reaction_enabled=False),
        surface=gs.surfaces.Default(color=(0.36, 0.24, 0.14)),
        name="table",
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            pos=(0.0, -0.74, 0.052),
            quat=(math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)),
            file=str(Path(gs.utils.get_assets_dir()) / "urdf/lift_acone_x5a_coffee/urdf/lift_acone_x5a_pbd.urdf"),
            # Prepared closed collision surfaces retain well-defined contact normals through import.
            decimate=False,
            watertighten=None,
            convexify=False,
            fixed=True,
            merge_fixed_links=False,
        ),
        material=gs.materials.Rigid(coup_friction=1000.0, is_coup_reaction_enabled=False),
        name="coffee_robot",
    )
    sponge = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=sponge_pos[0],
            quat=sponge_quat[0],
            vertices=vertices,
            elements=elements,
        ),
        material=gs.materials.PBD.Elastic(
            rho=30.0,
            stretch_relaxation=3.0,
            volume_relaxation=3.0,
            damping=args.damping,
        ),
        surface=gs.surfaces.Default(color=(0.95, 0.68, 0.12), smooth=False),
        name="sponge",
    )
    scene.build()
    right_hand = robot.get_link(name="right_link6")
    arms_dofs_idx = tuple(
        tuple(robot.get_joint(f"{side}_joint{idx}").dofs_idx_local[0] for idx in range(1, 7))
        for side in ("right", "left")
    )
    fingers_dofs_idx = tuple(
        tuple(robot.get_joint(f"{side}_joint{idx}").dofs_idx_local[0] for idx in (7, 8)) for side in ("right", "left")
    )
    right_fingers = tuple(robot.get_link(f"right_link{idx}") for idx in (7, 8))
    controlled_dofs_idx = tuple(robot.get_joint(name).dofs_idx_local[0] for name in joint_names)
    right_arm_columns = tuple(joint_names.index(f"right_joint{idx}") for idx in range(1, 7))
    robot.set_dofs_kp(kp=2000.0)
    robot.set_dofs_kv(kv=100.0)
    robot.set_dofs_force_range(lower=-200.0, upper=200.0)
    for dofs in arms_dofs_idx:
        robot.set_dofs_kp(kp=1500.0, dofs_idx_local=dofs)
        robot.set_dofs_kv(kv=40.0, dofs_idx_local=dofs)
        robot.set_dofs_force_range(lower=-30.0, upper=30.0, dofs_idx_local=dofs)
    for dofs, kp, kv in zip(fingers_dofs_idx, (10000.0, 3000.0), (300.0, 30.0)):
        robot.set_dofs_kp(kp, dofs_idx_local=dofs)
        robot.set_dofs_kv(kv, dofs_idx_local=dofs)
        robot.set_dofs_force_range(lower=-20.0, upper=20.0, dofs_idx_local=dofs)
    lift_dofs_idx = robot.get_joint(name="joint_lift").dofs_idx_local
    robot.set_dofs_kp(kp=20000.0, dofs_idx_local=lift_dofs_idx)
    robot.set_dofs_kv(kv=500.0, dofs_idx_local=lift_dofs_idx)
    robot.set_dofs_force_range(lower=-5000.0, upper=5000.0, dofs_idx_local=lift_dofs_idx)
    for link in right_fingers:
        link.set_friction(friction=2.0)
        link.set_friction_torsional(friction_torsional=0.005)
    scene.rigid_solver.set_geoms_friction_rolling(friction_rolling=0.0)
    contact_geoms_idx = tuple(
        collision_geom.idx for link in (*right_fingers, table.base_link) for collision_geom in link.geoms
    )
    scene.rigid_solver.set_sol_params(
        sol_params=[0.004, 1.0, 0.99, 0.999, 0.001, 0.5, 2.0], geoms_idx=contact_geoms_idx
    )
    if len(initial_qpos) != robot.n_qs:
        raise RuntimeError("The bundled pickup coordinates must match the coffee robot's joints.")
    finger_opening = 0.04395019377660751
    for idx in (7, 8):
        initial_qpos[robot.get_joint(f"right_joint{idx}").qs_idx_local[0]] = finger_opening
    robot.set_qpos(initial_qpos)
    robot.control_dofs_position(position=0.0)
    robot.control_dofs_position(joint_targets[0], dofs_idx_local=controlled_dofs_idx)
    robot.control_dofs_position(finger_opening, dofs_idx_local=fingers_dofs_idx[0])

    steps = math.ceil(args.seconds / scene.dt)
    times = (np.arange(steps) + 1) * scene.dt
    elapsed = np.clip(times - lift_start, recorded_time[0], recorded_time[-1])
    indices = np.clip(np.searchsorted(recorded_time, elapsed, side="right"), 1, len(recorded_time) - 1)
    fraction = ((elapsed - recorded_time[indices - 1]) / (recorded_time[indices] - recorded_time[indices - 1]))[:, None]
    recorded_arm_targets = joint_targets[:, right_arm_columns]
    arm_delta = recorded_arm_targets[indices] - recorded_arm_targets[indices - 1]
    arm_targets = gs.tensor(recorded_arm_targets[indices - 1] + fraction * arm_delta)
    centers = sponge_centers[indices - 1] + fraction * (sponge_centers[indices] - sponge_centers[indices - 1])
    previous_target = gs.tensor(recorded_arm_targets[0])
    contact_meshes = []
    if args.is_penetration_checked:
        for link in (right_hand, *right_fingers):
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
    max_penetration = 0.0
    max_penetrating_vertices = 0
    max_rigid_penetration = 0.0
    max_penetrating_rigid_vertices = 0
    max_edge_penetration = 0.0
    max_edge_contacts = 0
    max_triangle_intersections = 0
    max_unpaired_intersections = 0
    sponge_lift_start = None
    max_tracking_error = 0.0
    lift_delta = np.zeros(3)
    playback = None
    if args.is_viewer_shown and "PYTEST_VERSION" not in os.environ:
        playback = LiftPlayback(scene, sponge)
        playback.record_frame(step=0)
    step = 0
    while step < steps or playback is not None:
        if playback is not None and not playback.advance(steps):
            break
        time = (step + 1) * scene.dt
        fraction = min(time / 0.6, 1.0)
        progress = fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
        robot.control_dofs_position(finger_opening - 0.01345 * progress, dofs_idx_local=fingers_dofs_idx[0])
        if time >= lift_start:
            if sponge_lift_start is None:
                sponge_lift_start = tensor_to_array(sponge.get_particles_pos().mean(dim=-2))
                gs.logger.info(f"{time:.3f}s: start lifting.")
            lift_delta = centers[step] - sponge_centers[0]
            target = arm_targets[step]
            velocity = (target - previous_target) / scene.dt
            robot.control_dofs_position_velocity(target, velocity, dofs_idx_local=arms_dofs_idx[0])
            previous_target = target
        scene.step(update_visualizer=playback is None)
        scene.rigid_solver.check_errno()
        scene.pbd_solver.check_errno()
        if args.is_penetration_checked:
            contacts = robot.get_contacts(with_entity=robot)
            if contacts["geom_a"].numel():
                links_a = tensor_to_array(contacts["link_a"])
                links_b = tensor_to_array(contacts["link_b"])
                link_pairs = tuple(
                    (scene.rigid_solver.links[i_a].name, scene.rigid_solver.links[i_b].name)
                    for i_a, i_b in np.unique(np.column_stack((links_a, links_b)), axis=0)
                )
                raise RuntimeError(f"Robot self-collision at {time:.3f}s: {link_pairs}.")
        if playback is not None:
            playback.record_frame(step + 1)
        step += 1
        if step % measurement_stride != 0 and step != steps:
            continue
        positions = tensor_to_array(
            sponge.get_particles_pos(), dtype=np.float64 if args.is_penetration_checked else None
        )
        if not np.isfinite(positions).all():
            raise RuntimeError("Sponge vertices contain non-finite values.")
        if sponge_lift_start is not None:
            displacement = positions.mean(axis=0) - sponge_lift_start
            tracking_error = np.linalg.norm(displacement - lift_delta)
            max_tracking_error = max(max_tracking_error, tracking_error)
            gs.logger.info(
                f"{time:.3f}s: sponge travel={1000 * displacement[2]:.3f}mm, "
                f"target={1000 * lift_delta[2]:.3f}mm, tracking error={1000 * tracking_error:.3f}mm."
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
    if sponge_lift_start is not None:
        gs.logger.info(f"Maximum sponge tracking error: {1000 * max_tracking_error:.3f}mm.")
    if args.is_penetration_checked:
        gs.logger.info(
            f"Peak vertex penetration: {1000 * max_penetration:.6f}mm; "
            f"maximum penetrating vertices: {max_penetrating_vertices}; "
            f"peak rigid vertex penetration: {1000 * max_rigid_penetration:.6f}mm; "
            f"maximum penetrating rigid vertices: {max_penetrating_rigid_vertices}."
        )
        gs.logger.info(
            f"Peak crossed-edge gap: {1000 * max_edge_penetration:.6f}mm; "
            f"maximum edge contacts: {max_edge_contacts}; "
            f"maximum surface triangle intersections: {max_triangle_intersections}; "
            f"maximum unpaired intersections: {max_unpaired_intersections}."
        )


if __name__ == "__main__":
    main()
