from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class GarmentFoldingSolverConfig:
    """Scene527 finite element method (FEM) and Incremental Potential Contact (IPC) settings."""

    dt: float = 1.0 / 120.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    contact_d_hat: float = 0.0015
    contact_resistance: float = 1.0e7
    newton_semi_implicit_enable: bool = False
    newton_max_iterations: int = 50
    newton_min_iterations: int = 1
    n_linesearch_iterations: int = 8
    newton_tolerance: float = 1.0e-2
    newton_translation_tolerance: float = 1.0e-3
    linear_system_tolerance: float = 1.0e-3
    constraint_strength_translation: float = 300.0
    constraint_strength_rotation: float = 1000.0
    two_way_coupling: bool = False
    is_rigid_rigid_contact_enabled: bool = True
    # IPC owns contact for the fixed table and prescribed robot motion
    enable_genesis_rigid_collision: bool = False


@dataclass(frozen=True)
class GarmentFoldingMaterialConfig:
    """Scene527 cloth, table and robot contact properties."""

    cloth_young_modulus: float = 20000.0
    cloth_poisson_ratio: float = 0.49
    cloth_density: float = 800.0
    cloth_thickness: float = 0.0001
    cloth_bending_stiffness: float = 40.0
    cloth_friction: float = 1.0
    cloth_self_friction: float | None = 2.0
    support_friction: float = 1.0
    robot_friction: float = 2.0


@dataclass(frozen=True)
class GarmentFoldingAssets:
    """Portable Scene527 bundle paths, transforms and support geometry."""

    asset_root: str | None = None
    garment_mesh: str = "assets/cloth/short-shirt-55068f.obj"
    garment_texture: str | None = None
    garment_pos: tuple[float, float, float] = (0.667, 0.015, 0.93)
    garment_scale: float = 0.001
    garment_euler: tuple[float, float, float] = (-90.0, 0.0, 0.0)
    expected_garment_vertices: int | None = 27811
    expected_garment_faces: int | None = 55068
    robot: str = "assets/robots/dual_x5_2025_ipc_v1/dual_x5_2025_ipc.urdf"
    robot_pos: tuple[float, float, float] = (0.0, 0.0, 0.17)
    robot_coupling_links: tuple[str, ...] = (
        "left_link15",
        "left_link16",
        "left_link17",
        "left_link18",
        "right_link25",
        "right_link26",
        "right_link27",
        "right_link28",
    )
    # Alpha wrapping would change the authored exact finger collision geometry
    robot_watertighten: int | None = None
    trajectory: str = "controls/trajectory.npz"
    expected_trajectory_frames: int | None = 4373
    table_pos: tuple[float, float, float] = (0.65, 0.0, 0.4)
    table_size: tuple[float, float, float] = (1.0, 2.0, 0.8)
    camera_res: tuple[int, int] = (960, 720)
    camera_pos: tuple[float, float, float] = (1.55, -1.45, 1.65)
    camera_lookat: tuple[float, float, float] = (0.64, 0.0, 0.86)
    camera_fov: float = 42.0


@dataclass(frozen=True)
class GarmentFoldingTaskConfig:
    """Recorded robot action rate and explicit settling horizon."""

    action_fps: float = 60.0
    physics_steps_per_action: int = 2
    settle_steps: int = 60


@dataclass(frozen=True)
class GarmentFoldingScenarioConfig:
    solver: GarmentFoldingSolverConfig = field(default_factory=GarmentFoldingSolverConfig)
    material: GarmentFoldingMaterialConfig = field(default_factory=GarmentFoldingMaterialConfig)
    assets: GarmentFoldingAssets = field(default_factory=GarmentFoldingAssets)
    task: GarmentFoldingTaskConfig = field(default_factory=GarmentFoldingTaskConfig)


def create_scene527_config(asset_root: str | Path | None = None, *, mesh: str = "55k") -> GarmentFoldingScenarioConfig:
    """Select an 8k, 13k or 55k cloth mesh with the same material and trajectory settings."""
    if mesh == "8k":
        n_vertices, n_faces = 4090, 8000
    elif mesh == "13k":
        n_vertices, n_faces = 7021, 13767
    elif mesh == "55k":
        n_vertices, n_faces = 27811, 55068
    else:
        raise ValueError(f"Unknown garment mesh {mesh!r}; choose 8k, 13k or 55k.")
    return GarmentFoldingScenarioConfig(
        assets=replace(
            GarmentFoldingAssets(),
            asset_root=None if asset_root is None else str(Path(asset_root)),
            garment_mesh=f"assets/cloth/short-shirt-{n_faces}f.obj",
            expected_garment_vertices=n_vertices,
            expected_garment_faces=n_faces,
        )
    )
