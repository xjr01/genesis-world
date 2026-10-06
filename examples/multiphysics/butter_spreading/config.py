from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ButterSpreadingSolverConfig:
    """MPM discretization and coupling settings independent of scene geometry."""

    dt: float = 3.5e-5
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    lower_bound: tuple[float, float, float] = (-0.12, -0.08, -0.02)
    upper_bound: tuple[float, float, float] = (0.12, 0.08, 0.11)
    grid_density: int = 512
    bread_particle_size: float = 0.001
    butter_particle_size: float = 0.0005
    is_cpic_enabled: bool = True
    butter_sampling: Literal["regular", "stratified"] = "stratified"
    butter_sampling_seed: int = 17


@dataclass(frozen=True)
class ButterSpreadingMaterialConfig:
    """Parameters for the migrated rate-dependent butter and crushable-bread models."""

    bread_youngs_modulus: float = 16000.0
    bread_poissons_ratio: float = 0.15
    bread_density: float = 280.0
    butter_shear_modulus: float = 20000.0
    butter_bulk_modulus: float = 150000.0
    butter_yield_stress: float = 80.0
    butter_consistency: float = 25.0
    butter_flow_exponent: float = 0.5
    butter_density: float = 900.0
    bread_compaction_yield_pressure: float = 650.0
    bread_compaction_hardening: float = 5200.0
    bread_densification_strain: float = 0.32
    bread_densification_hardening: float = 45000.0
    bread_min_plastic_volume_ratio: float = 0.35
    bread_shear_yield_stress: float = 700.0
    bread_shear_hardening: float = 7000.0
    support_density: float = 700.0
    support_friction: float = 0.9
    blade_density: float = 7800.0
    blade_friction: float = 0.65
    blade_coupling_friction: float = 0.12
    coupling_softness: float = 0.001

    @property
    def butter_youngs_modulus(self) -> float:
        shear = self.butter_shear_modulus
        bulk = self.butter_bulk_modulus
        return 9.0 * bulk * shear / (3.0 * bulk + shear)

    @property
    def butter_poissons_ratio(self) -> float:
        shear = self.butter_shear_modulus
        bulk = self.butter_bulk_modulus
        return (3.0 * bulk - 2.0 * shear) / (2.0 * (3.0 * bulk + shear))


@dataclass(frozen=True)
class ButterSpreadingContactConfig:
    """Calibrated finite-range butter/bread and butter/blade contact law."""

    bread_normal_stress: float = 450.0
    blade_normal_stress: float = 100.0
    bread_shear_stress: float = 650.0
    blade_shear_stress: float = 150.0
    bread_slip_time: float = 0.008
    blade_slip_time: float = 0.008
    butter_contact_range: float = 0.002916666666666667
    bread_contact_range: float = 0.0015
    blade_normal_relaxation_time: float = 0.005
    blade_max_separation_speed: float = 0.35
    blade_contact_margin: float = 0.001875
    is_equilibrium_adhesion: bool = True


@dataclass(frozen=True)
class ButterSpreadingAssets:
    """Primitive geometry, placement and collision resolution for the task."""

    plate_height: float = 0.010
    support_depth: float = 0.020
    support_size_xy: tuple[float, float] = (0.18, 0.14)
    bread_size: tuple[float, float, float] = (0.14, 0.114, 0.016)
    bread_center: tuple[float, float, float] = (0.0, 0.0, 0.018)
    butter_size: tuple[float, float, float] = (0.040, 0.030, 0.018)
    butter_center: tuple[float, float, float] = (-0.040, 0.0, 0.035)
    blade_size: tuple[float, float, float] = (0.020, 0.095, 0.005)
    blade_xy_offset: tuple[float, float] = (0.0021, 0.0035)
    support_sdf_cell_size: float = 0.0015
    blade_sdf_cell_size: float = 0.001
    camera_res: tuple[int, int] = (1280, 720)
    camera_pos: tuple[float, float, float] = (0.22, -0.25, 0.18)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.025)
    camera_fov: float = 40.0

    @property
    def task_origin(self) -> tuple[float, float, float]:
        return (
            self.bread_center[0],
            self.bread_center[1],
            self.bread_center[2] + 0.5 * self.bread_size[2],
        )


@dataclass(frozen=True)
class ButterSpreadingTaskConfig:
    """Continuous level-blade press, spread and lift trajectory."""

    press_end_time: float = 0.90
    acceleration_end_time: float = 1.65
    deceleration_start_time: float = 3.15
    spread_end_time: float = 3.50
    lift_end_time: float = 4.0
    press_end_x: float = -0.055
    spread_end_x: float = 0.049
    start_clearance: float = 0.0305
    press_clearance: float = 0.0185
    sweep_clearance: float = 0.0105
    sweep_end_clearance: float = 0.0088
    lift_clearance: float = 0.0315

    @property
    def spread_speed(self) -> float:
        distance = self.spread_end_x - self.press_end_x
        duration = (
            self.deceleration_start_time
            - self.acceleration_end_time
            + 0.5 * (self.acceleration_end_time - self.press_end_time)
            + 0.5 * (self.spread_end_time - self.deceleration_start_time)
        )
        return distance / duration


@dataclass(frozen=True)
class ButterSpreadingScenarioConfig:
    """Complete butter-spreading input with explicit parameter ownership."""

    solver: ButterSpreadingSolverConfig = field(default_factory=ButterSpreadingSolverConfig)
    material: ButterSpreadingMaterialConfig = field(default_factory=ButterSpreadingMaterialConfig)
    contact: ButterSpreadingContactConfig = field(default_factory=ButterSpreadingContactConfig)
    assets: ButterSpreadingAssets = field(default_factory=ButterSpreadingAssets)
    task: ButterSpreadingTaskConfig = field(default_factory=ButterSpreadingTaskConfig)
