from genesis.typing import PositiveFloat, UnitInterval

from .base import Base, SamplerType


class Liquid(Base):
    """Liquid for implicit position-based fluids (IPBF).

    ``rho`` is rest density in kg/m^3. The regular sampler gives reproducible particle spacing; random sampling adds
    spatial variation. ``surface_tension`` is the coefficient in N/m for the linear surface model; higher values
    produce stronger cohesion and a stiffer numerical response. The quadratic model uses the solver's area weight.
    ``c_init`` initializes a passive scalar in [0, 1], with zero representing water and one coffee in the color map.
    ``boundary_group=1`` retains particles in the movable pitcher until they exit; zero uses the primary boundary.
    """

    rho: PositiveFloat = 1000.0
    sampler: SamplerType = "regular"
    surface_tension: PositiveFloat = 15.7
    c_init: UnitInterval = 0.0
    boundary_group: int = 0
