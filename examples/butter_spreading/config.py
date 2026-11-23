"""Validated physical parameters for reproducible butter-spreading runs."""

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Positive = Annotated[float, Field(gt=0)]
NonNegative = Annotated[float, Field(ge=0)]


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ButterParameters(Parameters):
    shear_modulus: Positive = 20000.0
    bulk_modulus: Positive = 150000.0
    yield_stress: Positive = 80.0
    consistency: NonNegative = 25.0
    flow_exponent: Annotated[float, Field(ge=0.25, le=1.0)] = 0.5
    rho: Positive = 900.0


class BreadParameters(Parameters):
    E: Positive = 16000.0
    nu: Annotated[float, Field(gt=-1.0, lt=0.5)] = 0.15
    rho: Positive = 280.0
    compaction_yield_pressure: Positive = 650.0
    compaction_hardening: NonNegative = 5200.0
    densification_strain: Annotated[float, Field(gt=0.05, lt=0.8)] = 0.32
    densification_hardening: NonNegative = 45000.0
    min_plastic_volume_ratio: Annotated[float, Field(gt=0.1, lt=1.0)] = 0.35
    shear_yield_stress: Positive = 700.0
    shear_hardening: NonNegative = 7000.0


class ContactParameters(Parameters):
    bread_stress: NonNegative = 450.0
    blade_stress: NonNegative = 100.0
    bread_contact_range: Positive = 0.0015
    bread_slip_time: Positive = 0.008
    bread_shear_stress: NonNegative = 650.0
    blade_contact_range: Positive = 0.002916666666666667
    blade_slip_time: Positive = 0.008
    blade_shear_stress: NonNegative = 150.0
    blade_normal_time: Positive = 0.005
    blade_max_separation_speed: NonNegative = 0.35
    blade_contact_margin: NonNegative = 0.001875
    is_equilibrium_adhesion: bool = True


class MotionParameters(Parameters):
    press_end_x: float = -0.055
    spread_end_x: float = 0.049
    start_clearance: float = 0.0565
    press_clearance: float = 0.0445
    sweep_clearance: float = 0.0365
    sweep_end_clearance: float = 0.0348
    lift_clearance: float = 0.0575
    y: float = 0.0


class RunConfig(Parameters):
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)] = 17
    dt: Positive = 3.5e-5
    grid_density: Annotated[int, Field(strict=True, gt=0)] = 512
    bread_particle_size: Positive = 0.001
    butter_particle_size: Positive = 0.0005
    bread: BreadParameters = Field(default_factory=BreadParameters)
    butter: ButterParameters = Field(default_factory=ButterParameters)
    contact: ContactParameters = Field(default_factory=ContactParameters)
    motion: MotionParameters = Field(default_factory=MotionParameters)


def load_config(path: Path | None, *, seed=None, bread_particle_size=None, butter_particle_size=None) -> RunConfig:
    """Load a partial JSON configuration and apply explicit command-line overrides."""
    config = RunConfig.model_validate_json(path.read_text(encoding="utf-8-sig")) if path else RunConfig()
    values = config.model_dump()
    for name, value in (
        ("seed", seed),
        ("bread_particle_size", bread_particle_size),
        ("butter_particle_size", butter_particle_size),
    ):
        if value is not None:
            values[name] = value
    return RunConfig.model_validate(values)
