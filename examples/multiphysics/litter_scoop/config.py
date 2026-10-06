import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LitterScoopSolverConfig:
    """Scene-wide numerical settings for the coupled DEM/FLIP simulation."""

    dt: float = 1.0 / 60.0
    substeps: int = 1
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.8)
    particle_size: float = 0.8 / 128.0
    grid_res: int = 128
    ddt_safety: float = 0.5
    viscosity_coeff: float = 0.01
    lower_bound: tuple[float, float, float] = (-0.35, -0.30, 0.0)
    upper_bound: tuple[float, float, float] = (0.35, 0.30, 0.80)


@dataclass(frozen=True)
class LitterScoopMaterialConfig:
    """Physical properties of sand grains and water, independent of their geometry."""

    sand_density: float = 2.5
    sand_young_modulus: float = 1.0e6
    sand_poisson_ratio: float = 0.3
    sand_friction_angle: float = 0.5
    sand_max_water_ratio: float = 0.3
    water_density: float = 1000.0


@dataclass(frozen=True)
class LitterScoopAssets:
    """Geometry, poses and visualization choices belonging to the litter-scoop task."""

    shovel_mesh: str = "examples/sand_water_coupling/assets/litter_scoop_aligned.glb"
    shovel_sdf: str = "examples/sand_water_coupling/assets/litter_scoop_sdf_slots.npz"
    sand_pos: tuple[float, float, float] = (0.0, 0.0, 0.081)
    sand_size: tuple[float, float, float] = (0.68, 0.58, 0.16)
    droplet_pos: tuple[float, float, float] = (0.02, 0.0, 0.205)
    droplet_radius: float = 0.04
    wall_thickness: float = 0.01
    wall_height: float = 0.18
    blade_angle: float = math.radians(40.0)
    blade_initial_pos: tuple[float, float, float] = (-0.20, 0.0, 0.369)
    camera_res: tuple[int, int] = (1280, 720)
    camera_pos: tuple[float, float, float] = (1.05, -1.05, 0.75)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.18)
    camera_fov: float = 45.0


@dataclass(frozen=True)
class LitterScoopTaskConfig:
    """Task timing and shovel commands, independent of solver accuracy and assets."""

    settle_steps: int = 300
    descend_steps: int = 55
    insert_steps: int = 96
    rotate_steps: int = 105
    lift_steps: int = 150
    hold_steps: int = 60
    descend_velocity: tuple[float, float, float] = (0.0, 0.0, -0.12)
    insert_velocity: tuple[float, float, float] = (0.115, 0.0, -0.0964)
    rotate_velocity: tuple[float, float, float] = (0.0205, 0.0, 0.0563)
    rotate_angular_velocity: tuple[float, float, float] = (0.0, -math.radians(40.0) / 1.75, 0.0)
    lift_velocity: tuple[float, float, float] = (0.0, 0.0, 0.10)

    @property
    def total_steps(self):
        return (
            self.settle_steps
            + self.descend_steps
            + self.insert_steps
            + self.rotate_steps
            + self.lift_steps
            + self.hold_steps
        )


@dataclass(frozen=True)
class LitterScoopScenarioConfig:
    solver: LitterScoopSolverConfig = field(default_factory=LitterScoopSolverConfig)
    material: LitterScoopMaterialConfig = field(default_factory=LitterScoopMaterialConfig)
    assets: LitterScoopAssets = field(default_factory=LitterScoopAssets)
    task: LitterScoopTaskConfig = field(default_factory=LitterScoopTaskConfig)
