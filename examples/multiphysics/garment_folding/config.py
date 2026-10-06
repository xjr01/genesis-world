from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

GarmentFoldingTask = Literal["grasp", "half", "quarter", "fold", "scene527"]


@dataclass(frozen=True)
class GarmentFoldingSolverConfig:
    """Scene-wide numerical settings for the finite element method (FEM) and IPC simulation."""

    dt: float = 0.02
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    contact_d_hat: float = 0.0008
    contact_resistance: float = 1.0e7
    newton_semi_implicit_enable: bool = False
    newton_max_iterations: int = 50
    newton_min_iterations: int = 1
    n_linesearch_iterations: int = 16
    newton_tolerance: float = 1.0e-2
    newton_translation_tolerance: float = 1.0e-3
    linear_system_tolerance: float = 1.0e-3
    constraint_strength_translation: float = 100.0
    constraint_strength_rotation: float = 100.0
    two_way_coupling: bool = False
    is_rigid_rigid_contact_enabled: bool = False
    enable_genesis_rigid_collision: bool = True


@dataclass(frozen=True)
class GarmentFoldingMaterialConfig:
    """Cloth, support, and jaw contact properties independent of task motion."""

    cloth_young_modulus: float = 6.0e4
    cloth_poisson_ratio: float = 0.49
    cloth_density: float = 200.0
    cloth_thickness: float = 0.001
    cloth_bending_stiffness: float = 1.0e-6
    cloth_friction: float = 0.6
    cloth_self_friction: float | None = None
    cloth_table_friction: float | None = None
    support_friction: float = 0.3
    jaw_friction: float = 0.8
    robot_friction: float = 2.0
    jaw_density: float = 8000.0
    jaw_gravity_compensation: float = 1.0


@dataclass(frozen=True)
class GarmentFoldingAssets:
    """Garment files and support geometry used by the folding scenario."""

    asset_root: str | None = None
    garment_mesh: str = "assets/shirt.obj"
    garment_texture: str | None = "assets/cotton-base.png"
    garment_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    garment_scale: float = 1.0
    garment_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    expected_garment_vertices: int | None = None
    expected_garment_faces: int | None = None
    robot: str | None = None
    robot_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_coupling_links: tuple[str, ...] = ()
    robot_watertighten: int | None = 5
    trajectory: str | None = None
    expected_trajectory_frames: int | None = None
    hem_left: tuple[float, float, float] = (-0.145, -0.22, 0.0625)
    hem_right: tuple[float, float, float] = (0.145, -0.22, 0.0625)
    hem_center: tuple[float, float, float] = (0.0, -0.211, 0.0625)
    sleeve_left: tuple[float, float, float] = (-0.289, 0.15, 0.0625)
    sleeve_right: tuple[float, float, float] = (0.289, 0.15, 0.0625)
    table_pos: tuple[float, float, float] = (0.0, 0.0, 0.025)
    table_size: tuple[float, float, float] = (0.54, 0.38, 0.05)
    jaw_size: tuple[float, float, float] = (0.022, 0.032, 0.004)
    camera_res: tuple[int, int] = (1280, 960)
    camera_pos: tuple[float, float, float] = (0.45, -0.65, 0.8)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.08)
    camera_fov: float = 42.0

    def world_landmark(self, name: str) -> tuple[float, float, float]:
        if name == "hem_left":
            local = self.hem_left
        elif name == "hem_right":
            local = self.hem_right
        elif name == "hem_center":
            local = self.hem_center
        elif name == "sleeve_left":
            local = self.sleeve_left
        elif name == "sleeve_right":
            local = self.sleeve_right
        else:
            raise ValueError(f"Unknown garment landmark: {name}")
        return tuple(origin + self.garment_scale * value for origin, value in zip(self.garment_pos, local))

    def jaw_origins(self, task: GarmentFoldingTask) -> tuple[tuple[float, float, float], ...]:
        if task == "fold":
            return (self.sleeve_left, self.sleeve_right)
        if task in ("half", "quarter"):
            return (self.sleeve_left, self.hem_left)
        return (self.sleeve_left,)


@dataclass(frozen=True)
class GarmentFoldingTaskConfig:
    """Prescribed jaw trajectory and phase timing for a garment-folding task.

    ``half`` is the validated default. The other task values expose experimental trajectories for contact and
    multi-stage folding studies.
    """

    task: GarmentFoldingTask = "half"
    duration: float | None = None
    open_gap: float = 0.020
    closed_gap: float = 0.003
    regrasp_gap: float = 0.003
    action_fps: float = 60.0
    physics_steps_per_action: int = 2
    settle_steps: int = 0

    @property
    def resolved_duration(self) -> float:
        if self.duration is not None:
            return self.duration
        return {"grasp": 6.0, "half": 16.0, "quarter": 32.0, "fold": 30.0, "scene527": 4373.0 / 60.0}[self.task]

    @property
    def is_validated(self) -> bool:
        return self.task == "half"

    @property
    def uses_robot_trajectory(self) -> bool:
        return self.task == "scene527"


@dataclass(frozen=True)
class GarmentFoldingScenarioConfig:
    solver: GarmentFoldingSolverConfig = field(default_factory=GarmentFoldingSolverConfig)
    material: GarmentFoldingMaterialConfig = field(default_factory=GarmentFoldingMaterialConfig)
    assets: GarmentFoldingAssets = field(default_factory=GarmentFoldingAssets)
    task: GarmentFoldingTaskConfig = field(default_factory=GarmentFoldingTaskConfig)


def create_scene527_config(asset_root: str | Path) -> GarmentFoldingScenarioConfig:
    """Create the high-fidelity dual-X5 Scene527 profile from a portable reproduction bundle."""
    return GarmentFoldingScenarioConfig(
        solver=GarmentFoldingSolverConfig(
            dt=1.0 / 120.0,
            contact_d_hat=0.0015,
            n_linesearch_iterations=8,
            constraint_strength_translation=300.0,
            constraint_strength_rotation=1000.0,
            is_rigid_rigid_contact_enabled=True,
            # Rigid geometry participates in IPC contact.  Genesis' separate rigid SDF collider has no dynamic
            # rigid-rigid work in this fixed/kinematic scene and only duplicates a costly preprocessing path.
            enable_genesis_rigid_collision=False,
        ),
        material=GarmentFoldingMaterialConfig(
            cloth_young_modulus=20000.0,
            cloth_density=800.0,
            cloth_thickness=0.0001,
            cloth_bending_stiffness=40.0,
            cloth_friction=1.0,
            cloth_self_friction=2.0,
            support_friction=1.0,
            robot_friction=2.0,
        ),
        assets=GarmentFoldingAssets(
            asset_root=str(Path(asset_root)),
            garment_mesh="assets/cloth/short-shirt-55068f.obj",
            garment_texture=None,
            garment_pos=(0.667, 0.015, 0.93),
            garment_scale=0.001,
            garment_euler=(-90.0, 0.0, 0.0),
            expected_garment_vertices=27811,
            expected_garment_faces=55068,
            robot="assets/robots/dual_x5_2025_ipc_v1/dual_x5_2025_ipc.urdf",
            robot_pos=(0.0, 0.0, 0.17),
            # The source scene uses its authored exact finger collision meshes.  Re-wrapping them changes contact
            # geometry, so opt out of Genesis' newer default alpha wrap.
            robot_watertighten=None,
            robot_coupling_links=(
                "left_link15",
                "left_link16",
                "left_link17",
                "left_link18",
                "right_link25",
                "right_link26",
                "right_link27",
                "right_link28",
            ),
            trajectory="controls/trajectory.npz",
            expected_trajectory_frames=4373,
            table_pos=(0.65, 0.0, 0.4),
            table_size=(1.0, 2.0, 0.8),
            camera_res=(960, 720),
            camera_pos=(1.55, -1.45, 1.65),
            camera_lookat=(0.64, 0.0, 0.86),
        ),
        task=GarmentFoldingTaskConfig(task="scene527", settle_steps=60),
    )
