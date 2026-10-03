from dataclasses import dataclass, field
from typing import Literal


GarmentFoldingTask = Literal["grasp", "half", "quarter", "fold"]


@dataclass(frozen=True)
class GarmentFoldingSolverConfig:
    """Scene-wide numerical settings for the finite element method (FEM) and IPC simulation."""

    dt: float = 0.02
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    contact_d_hat: float = 0.0008
    contact_resistance: float = 1.0e7
    newton_semi_implicit_enable: bool = False
    n_linesearch_iterations: int = 16
    constraint_strength_translation: float = 100.0
    constraint_strength_rotation: float = 100.0
    two_way_coupling: bool = False


@dataclass(frozen=True)
class GarmentFoldingMaterialConfig:
    """Cloth, support, and jaw contact properties independent of task motion."""

    cloth_young_modulus: float = 6.0e4
    cloth_poisson_ratio: float = 0.49
    cloth_density: float = 200.0
    cloth_thickness: float = 0.001
    cloth_bending_stiffness: float = 1.0e-6
    cloth_friction: float = 0.6
    support_friction: float = 0.3
    jaw_friction: float = 0.8
    jaw_density: float = 8000.0
    jaw_gravity_compensation: float = 1.0


@dataclass(frozen=True)
class GarmentFoldingAssets:
    """Garment files and support geometry used by the folding scenario."""

    garment_mesh: str = "assets/shirt.obj"
    garment_texture: str = "assets/cotton-base.png"
    garment_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    garment_scale: float = 1.0
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

    @property
    def resolved_duration(self) -> float:
        if self.duration is not None:
            return self.duration
        return {"grasp": 6.0, "half": 16.0, "quarter": 32.0, "fold": 30.0}[self.task]

    @property
    def is_validated(self) -> bool:
        return self.task == "half"


@dataclass(frozen=True)
class GarmentFoldingScenarioConfig:
    solver: GarmentFoldingSolverConfig = field(default_factory=GarmentFoldingSolverConfig)
    material: GarmentFoldingMaterialConfig = field(default_factory=GarmentFoldingMaterialConfig)
    assets: GarmentFoldingAssets = field(default_factory=GarmentFoldingAssets)
    task: GarmentFoldingTaskConfig = field(default_factory=GarmentFoldingTaskConfig)
