import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .config import GarmentFoldingScenarioConfig


@dataclass(frozen=True)
class GarmentRobotTrajectory:
    """Robot joint targets sampled at the task action rate."""

    joint_q: np.ndarray
    action_fps: float
    physics_steps_per_action: int

    @property
    def physics_steps(self) -> int:
        return len(self.joint_q) * self.physics_steps_per_action


@dataclass(frozen=True)
class GarmentFoldingRuntime:
    """Scene527 entities and recorded robot targets."""

    scene: gs.Scene
    config: GarmentFoldingScenarioConfig
    garment: object
    table: object
    robot: object
    robot_trajectory: GarmentRobotTrajectory
    camera: object | None
    initial_positions: np.ndarray
    default_steps: int


def build_scene(
    config: GarmentFoldingScenarioConfig | None = None,
    *,
    show_viewer: bool = False,
    add_camera: bool = False,
) -> GarmentFoldingRuntime:
    """Build the garment-folding scene without starting task motion or recording."""
    config = config or GarmentFoldingScenarioConfig()
    solver = config.solver
    material = config.material
    assets = config.assets
    package_root = Path(__file__).resolve().parent
    asset_root = package_root if assets.asset_root is None else Path(assets.asset_root).expanduser().resolve()
    garment_mesh = asset_root / assets.garment_mesh
    garment_texture = None if assets.garment_texture is None else asset_root / assets.garment_texture
    if not garment_mesh.is_file():
        raise FileNotFoundError(f"Garment mesh does not exist: {garment_mesh}")
    if garment_texture is not None and not garment_texture.is_file():
        raise FileNotFoundError(f"Garment texture does not exist: {garment_texture}")
    with garment_mesh.open("rb") as garment_file:
        if garment_file.read(42).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"Garment mesh is a Git LFS pointer; download the asset first: {garment_mesh}")

    robot_path = asset_root / assets.robot
    trajectory_path = asset_root / assets.trajectory
    if not robot_path.is_file():
        raise FileNotFoundError(f"Garment robot does not exist: {robot_path}")
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Garment trajectory does not exist: {trajectory_path}")
    with trajectory_path.open("rb") as trajectory_file:
        if trajectory_file.read(42).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"Garment trajectory is a Git LFS pointer; download the asset first: {trajectory_path}")
    if config.task.physics_steps_per_action < 1:
        raise ValueError("physics_steps_per_action must be positive.")
    if config.task.settle_steps < 0:
        raise ValueError("settle_steps must be non-negative.")
    if config.task.action_fps <= 0.0:
        raise ValueError("action_fps must be positive.")
    expected_dt = 1.0 / (config.task.action_fps * config.task.physics_steps_per_action)
    if not np.isclose(solver.dt, expected_dt):
        raise ValueError(f"Scene timestep must be {expected_dt} for the configured garment action rate.")
    with np.load(trajectory_path, allow_pickle=False) as trajectory_data:
        source_joint_q = trajectory_data["joint_q"].copy()
        source_joint_names = tuple(trajectory_data["joint_names"].tolist())
    if source_joint_q.ndim != 2 or source_joint_q.shape[1] != len(source_joint_names):
        raise ValueError("Garment robot trajectory joint data and names have incompatible shapes.")
    if not len(source_joint_q) or not np.all(np.isfinite(source_joint_q)):
        raise ValueError("Garment robot trajectory must contain finite joint targets.")
    if assets.expected_trajectory_frames is not None and len(source_joint_q) != assets.expected_trajectory_frames:
        raise ValueError(
            f"Garment robot trajectory has {len(source_joint_q)} frames; expected {assets.expected_trajectory_frames}."
        )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=solver.dt,
            gravity=solver.gravity,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=solver.contact_d_hat,
            contact_resistance=solver.contact_resistance,
            newton_semi_implicit_enable=solver.newton_semi_implicit_enable,
            newton_max_iterations=solver.newton_max_iterations,
            newton_min_iterations=solver.newton_min_iterations,
            n_linesearch_iterations=solver.n_linesearch_iterations,
            newton_tolerance=solver.newton_tolerance,
            newton_translation_tolerance=solver.newton_translation_tolerance,
            linear_system_tolerance=solver.linear_system_tolerance,
            contact_friction_enable=True,
            constraint_strength_translation=solver.constraint_strength_translation,
            constraint_strength_rotation=solver.constraint_strength_rotation,
            two_way_coupling=solver.two_way_coupling,
            enable_rigid_rigid_contact=solver.is_rigid_rigid_contact_enabled,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=solver.enable_genesis_rigid_collision,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=assets.camera_pos,
            camera_lookat=assets.camera_lookat,
        ),
        vis_options=gs.options.VisOptions(
            ambient_light=(0.26, 0.26, 0.26),
        ),
        show_viewer=show_viewer,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(
            size=assets.table_size,
            pos=assets.table_pos,
            fixed=True,
        ),
        material=gs.materials.Rigid(
            coup_type="ipc_only",
            coup_friction=material.support_friction,
            contact_resistance=solver.contact_resistance,
        ),
        surface=gs.surfaces.Plastic(color=(0.78, 0.82, 0.86, 1.0)),
    )
    robot_load_started = time.perf_counter()
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=str(robot_path),
            pos=assets.robot_pos,
            fixed=True,
            visualization=True,
            collision=True,
            convexify=False,
            decimate=False,
            watertighten=assets.robot_watertighten,
        ),
        material=gs.materials.Rigid(
            coup_type="two_way_soft_constraint",
            coup_friction=material.robot_friction,
            contact_resistance=solver.contact_resistance,
            coup_links=assets.robot_coupling_links,
        ),
        surface=gs.surfaces.Plastic(color=(0.56, 0.61, 0.66, 1.0)),
    )
    print(f"garment_folding timing: robot_load_seconds={time.perf_counter() - robot_load_started:.3f}", flush=True)
    genesis_joint_names = tuple(joint.name for joint in robot.joints if joint.n_qs)
    if set(genesis_joint_names) != set(source_joint_names):
        raise ValueError(
            f"Garment robot and trajectory joint names differ: robot={genesis_joint_names}, "
            f"trajectory={source_joint_names}."
        )
    source_index_by_name = {name: index for index, name in enumerate(source_joint_names)}
    joint_q = source_joint_q[:, [source_index_by_name[name] for name in genesis_joint_names]]
    robot_trajectory = GarmentRobotTrajectory(
        joint_q=joint_q,
        action_fps=config.task.action_fps,
        physics_steps_per_action=config.task.physics_steps_per_action,
    )
    movable_qs = 0
    for joint in robot.joints:
        if joint.n_qs:
            if np.any(joint.init_qpos != 0):
                raise ValueError("Scene527 requires neutral URDF joint defaults for IPC proxy construction.")
            movable_qs += joint.n_qs
    if movable_qs != joint_q.shape[1]:
        raise RuntimeError(
            f"Garment robot exposes {movable_qs} movable coordinates; trajectory has {joint_q.shape[1]}."
        )

    garment_surface = (
        gs.surfaces.Plastic(color=(0.10, 0.34, 0.68, 1.0), roughness=1.0, smooth=True, normal_diff_clamp=42.0)
        if garment_texture is None
        else gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(garment_texture)))
    )
    garment_load_started = time.perf_counter()
    garment = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(garment_mesh),
            pos=assets.garment_pos,
            scale=assets.garment_scale,
            euler=assets.garment_euler,
            decimate=False,
        ),
        material=gs.materials.FEM.Cloth(
            E=material.cloth_young_modulus,
            nu=material.cloth_poisson_ratio,
            rho=material.cloth_density,
            thickness=material.cloth_thickness,
            bending_stiffness=material.cloth_bending_stiffness,
            friction_mu=material.cloth_friction,
            self_friction_mu=material.cloth_self_friction,
        ),
        surface=garment_surface,
    )
    print(f"garment_folding timing: garment_load_seconds={time.perf_counter() - garment_load_started:.3f}", flush=True)
    if assets.expected_garment_vertices is not None and garment.n_vertices != assets.expected_garment_vertices:
        raise ValueError(
            f"Garment mesh has {garment.n_vertices} vertices; expected {assets.expected_garment_vertices}."
        )
    if assets.expected_garment_faces is not None and garment.n_elements != assets.expected_garment_faces:
        raise ValueError(f"Garment mesh has {garment.n_elements} faces; expected {assets.expected_garment_faces}.")
    camera = None
    if add_camera:
        camera = scene.add_camera(
            res=assets.camera_res,
            pos=assets.camera_pos,
            lookat=assets.camera_lookat,
            fov=assets.camera_fov,
            GUI=False,
        )
    scene_build_started = time.perf_counter()
    scene.build()
    print(f"garment_folding timing: scene_build_seconds={time.perf_counter() - scene_build_started:.3f}", flush=True)

    robot.set_qpos(robot_trajectory.joint_q[0], zero_velocity=True)

    initial_positions = tensor_to_array(garment.get_state().pos).reshape(-1, 3).copy()
    return GarmentFoldingRuntime(
        scene=scene,
        config=config,
        garment=garment,
        table=table,
        robot=robot,
        robot_trajectory=robot_trajectory,
        camera=camera,
        initial_positions=initial_positions,
        default_steps=robot_trajectory.physics_steps,
    )
