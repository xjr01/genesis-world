from genesis.typing import NonNegativeFloat, PositiveFloat, UnitInterval

from .base import Base


class Liquid(Base):
    """Single-phase liquid simulated by the implicit position-based fluid solver.

    ``viscosity`` blends neighboring liquid velocities after the pressure solve. Larger values suppress particle-scale
    spray and preserve coherent bulk flow but dissipate relative kinetic energy; zero preserves inviscid motion at the
    cost of noisier impacts. ``kinetic_smoothing`` performs the same neighborhood filtering while preserving linear
    momentum and total kinetic energy; it also preserves the covariance of relative velocities when the filtered
    covariance is well-conditioned. Larger values favor coherent sheets and jets but suppress physically meaningful
    small-scale velocity variation; zero leaves the pressure velocity spectrum unchanged.
    """

    rho: PositiveFloat = 1000.0
    viscosity: NonNegativeFloat = 0.0
    kinetic_smoothing: UnitInterval = 0.2
