from dataclasses import dataclass
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .config import GarmentFoldingScenarioConfig


@dataclass(frozen=True)
class GarmentLandmarks:
    hem_left: tuple[int, ...]
    hem_right: tuple[int, ...]
    hem_center: tuple[int, ...]
    sleeve_left: tuple[int, ...]
    sleeve_right: tuple[int, ...]

    def indices(self, name: str) -> tuple[int, ...]:
        if name == "hem_left":
            return self.hem_left
        if name == "hem_right":
            return self.hem_right
        if name == "hem_center":
            return self.hem_center
        if name == "sleeve_left":
            return self.sleeve_left
        if name == "sleeve_right":
            return self.sleeve_right
        raise ValueError(f"Unknown garment landmark: {name}")


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
    """Named scene handles used by task controllers and downstream adapters."""

    scene: gs.Scene
    config: GarmentFoldingScenarioConfig
    garment: object
    table: object
    jaws: tuple[tuple[object, object], ...]
    robot: object | None
    robot_trajectory: GarmentRobotTrajectory | None
    camera: object | None
    initial_positions: np.ndarray
    landmarks: GarmentLandmarks | None
    garment_pos: tuple[float, float, float]
    garment_scale: float
    jaw_origins_local: tuple[tuple[float, float, float], ...]
    jaw_half_thickness: float
    default_steps: int

    @property
    def control_dt(self) -> float:
        return self.scene.dt

    def get_jaw_poses(self):
        """Return current jaw poses through the public rigid-entity state API."""
        return tuple(tuple((jaw.get_pos(), jaw.get_quat()) for jaw in pair) for pair in self.jaws)


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

    robot_path = None if assets.robot is None else asset_root / assets.robot
    trajectory_path = None if assets.trajectory is None else asset_root / assets.trajectory
    if config.task.uses_robot_trajectory:
        if robot_path is None or not robot_path.is_file():
            raise FileNotFoundError(f"Garment robot does not exist: {robot_path}")
        if trajectory_path is None or not trajectory_path.is_file():
            raise FileNotFoundError(f"Garment trajectory does not exist: {trajectory_path}")
        with trajectory_path.open("rb") as trajectory_file:
            if trajectory_file.read(42).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError(
                    f"Garment trajectory is a Git LFS pointer; download the asset first: {trajectory_path}"
                )
        if config.task.physics_steps_per_action < 1:
            raise ValueError("physics_steps_per_action must be positive.")
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
                f"Garment robot trajectory has {len(source_joint_q)} frames; "
                f"expected {assets.expected_trajectory_frames}."
            )
    else:
        source_joint_q = None
        source_joint_names = ()

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=solver.dt, gravity=solver.gravity),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=solver.contact_d_hat,
            contact_resistance=solver.contact_resistance,
            newton_semi_implicit_enable=solver.newton_semi_implicit_enable,
            newton_max_iterations=solver.newton_max_iterations,
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
        viewer_options=gs.options.ViewerOptions(camera_pos=assets.camera_pos, camera_lookat=assets.camera_lookat),
        show_viewer=show_viewer,
    )
    if not config.task.uses_robot_trajectory:
        scene.add_entity(
            morph=gs.morphs.Plane(),
            material=gs.materials.Rigid(coup_friction=material.support_friction),
            surface=gs.surfaces.Default(color=(0.28, 0.31, 0.34, 1.0)),
        )
    table_material = (
        gs.materials.Rigid(
            coup_type="ipc_only",
            coup_friction=material.support_friction,
            contact_resistance=solver.contact_resistance,
        )
        if config.task.uses_robot_trajectory
        else gs.materials.Rigid(coup_friction=material.support_friction)
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(size=assets.table_size, pos=assets.table_pos, fixed=True),
        material=table_material,
        surface=gs.surfaces.Default(color=(0.75, 0.72, 0.66, 1.0)),
    )
    robot = None
    if config.task.uses_robot_trajectory:
        robot = scene.add_entity(
            morph=gs.morphs.URDF(
                file=str(robot_path),
                pos=assets.robot_pos,
                fixed=True,
                visualization=True,
                collision=True,
                convexify=False,
                decimate=False,
            ),
            material=gs.materials.Rigid(
                coup_type="two_way_soft_constraint",
                coup_friction=material.robot_friction,
                contact_resistance=solver.contact_resistance,
                coup_links=assets.robot_coupling_links,
            ),
            surface=gs.surfaces.Plastic(color=(0.56, 0.61, 0.66, 1.0)),
        )
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
    else:
        robot_trajectory = None

    garment_surface = (
        gs.surfaces.Plastic(color=(0.10, 0.34, 0.68, 1.0), roughness=1.0, smooth=True, normal_diff_clamp=42.0)
        if garment_texture is None
        else gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(garment_texture)))
    )
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
    if assets.expected_garment_vertices is not None and garment.n_vertices != assets.expected_garment_vertices:
        raise ValueError(
            f"Garment mesh has {garment.n_vertices} vertices; expected {assets.expected_garment_vertices}."
        )
    if assets.expected_garment_faces is not None and garment.n_elements != assets.expected_garment_faces:
        raise ValueError(f"Garment mesh has {garment.n_elements} faces; expected {assets.expected_garment_faces}.")
    if material.cloth_table_friction is not None:
        scene.set_ipc_contact_pair_friction(garment, table, friction=material.cloth_table_friction)

    jaw_origins_local = () if config.task.uses_robot_trajectory else assets.jaw_origins(config.task.task)
    jaw_half_thickness = 0.5 * assets.jaw_size[2]
    jaws = []
    for origin_local in jaw_origins_local:
        origin = tuple(
            world_origin + assets.garment_scale * value for world_origin, value in zip(assets.garment_pos, origin_local)
        )
        pair = []
        for sign in (-1.0, 1.0):
            pair.append(
                scene.add_entity(
                    morph=gs.morphs.Box(
                        size=assets.jaw_size,
                        pos=(
                            origin[0],
                            origin[1],
                            origin[2] + sign * (0.5 * config.task.open_gap + jaw_half_thickness),
                        ),
                    ),
                    material=gs.materials.Rigid(
                        coup_type="two_way_soft_constraint",
                        rho=material.jaw_density,
                        coup_friction=material.jaw_friction,
                        gravity_compensation=material.jaw_gravity_compensation,
                    ),
                    surface=gs.surfaces.Default(color=(0.7, 0.74, 0.8, 1.0)),
                )
            )
        jaws.append(tuple(pair))

    camera = None
    if add_camera:
        camera = scene.add_camera(
            res=assets.camera_res,
            pos=assets.camera_pos,
            lookat=assets.camera_lookat,
            fov=assets.camera_fov,
            GUI=False,
        )
    scene.build()

    if robot is not None:
        robot.set_qpos(robot_trajectory.joint_q[0], zero_velocity=True)
        scene.reset(scene.get_state())

    initial_positions = tensor_to_array(garment.get_state().pos).reshape(-1, 3)
    if config.task.uses_robot_trajectory:
        landmarks = None
        default_steps = robot_trajectory.physics_steps
    else:
        landmark_specs = tuple(
            (name, assets.world_landmark(name))
            for name in ("hem_left", "hem_right", "hem_center", "sleeve_left", "sleeve_right")
        )
        landmark_indices = []
        for _, position in landmark_specs:
            distances = np.linalg.norm(initial_positions[:, :2] - position[:2], axis=1)
            landmark_indices.append(tuple(np.argsort(distances)[:6]))
        landmarks = GarmentLandmarks(*landmark_indices)
        default_steps = round(config.task.resolved_duration / config.solver.dt) + 1
    return GarmentFoldingRuntime(
        scene=scene,
        config=config,
        garment=garment,
        table=table,
        jaws=tuple(jaws),
        robot=robot,
        robot_trajectory=robot_trajectory,
        camera=camera,
        initial_positions=initial_positions,
        landmarks=landmarks,
        garment_pos=assets.garment_pos,
        garment_scale=assets.garment_scale,
        jaw_origins_local=jaw_origins_local,
        jaw_half_thickness=jaw_half_thickness,
        default_steps=default_steps,
    )
