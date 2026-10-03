from dataclasses import dataclass, field

from .assets import CoffeeWaterAssets
from .boundaries import CoffeeWaterBoundaryConfig
from .config import CoffeeWaterMaterialConfig, CoffeeWaterSolverConfig
from .task import CoffeeWaterTaskConfig


@dataclass(frozen=True)
class CoffeeWaterScenarioConfig:
    """Complete coffee task input with explicit ownership of each parameter group."""

    solver: CoffeeWaterSolverConfig = field(default_factory=CoffeeWaterSolverConfig)
    material: CoffeeWaterMaterialConfig = field(default_factory=CoffeeWaterMaterialConfig)
    assets: CoffeeWaterAssets = field(default_factory=CoffeeWaterAssets)
    boundaries: CoffeeWaterBoundaryConfig = field(default_factory=CoffeeWaterBoundaryConfig)
    task: CoffeeWaterTaskConfig = field(default_factory=CoffeeWaterTaskConfig)
