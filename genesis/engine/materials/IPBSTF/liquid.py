from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import Base


class Liquid(Base):
    """Single-phase liquid simulated by the implicit position-based fluid solver.

    ``viscosity`` blends neighboring liquid velocities. Larger values suppress particle-scale velocity variation and
    produce more coherent flow at the cost of dissipating relative kinetic energy; zero preserves inviscid motion but
    can produce noisier impacts. The effective area-energy weight is ``IPBSTFOptions.alpha * surface_tension``;
    larger ``surface_tension`` values round droplets more strongly at the cost of local-mesh construction and a
    stiffer position solve, while zero disables that work.
    """

    rho: PositiveFloat = 1000.0
    viscosity: NonNegativeFloat = 0.0
    surface_tension: NonNegativeFloat = 0.0
