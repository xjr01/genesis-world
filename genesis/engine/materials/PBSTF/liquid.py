from pydantic import StrictBool

from genesis.typing import NonNegativeFloat, PositiveFloat, UnitInterval

from .base import Base, SamplerType


class Liquid(Base):
    """
    Fluid material for the position-based surface-tension solver.

    The compliance values follow the non-time-scaled convention and world
    length units used by the C++ PBSTF implementation; uniformly rescaling a
    scene therefore requires retuning them. ``surface_viscosity`` and
    ``interior_viscosity`` are the reference XSPH velocity-filter coefficients.

    ``is_collider_adhesion_friction_enabled`` enables both wall effects. Adhesion keeps surface particles attached to
    nearby collider surfaces but can make detachment harder; lower compliance strengthens it. Friction damps tangential
    wall motion and reduces sliding at the cost of kinetic energy; zero preserves tangential speed and one removes it.
    """

    rho: PositiveFloat = 1000.0
    density_compliance: NonNegativeFloat = 500.0
    surface_tension_compliance: NonNegativeFloat = 2.0
    surface_distance_compliance: NonNegativeFloat = 40.0
    interior_distance_compliance: NonNegativeFloat = 90.0
    surface_viscosity: NonNegativeFloat = 0.05
    interior_viscosity: NonNegativeFloat = 0.05
    is_collider_adhesion_friction_enabled: StrictBool = False
    collider_adhesion_compliance: NonNegativeFloat = 10.0
    collider_friction: UnitInterval = 0.1
    sampler: SamplerType = "staggered"
