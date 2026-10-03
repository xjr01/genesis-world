from genesis.typing import PositiveFloat

from .base import Base


class Liquid(Base):
    """Single-phase liquid simulated by the implicit position-based fluid solver."""

    rho: PositiveFloat = 1000.0
