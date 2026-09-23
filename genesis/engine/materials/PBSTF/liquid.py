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
    nearby collider surfaces but can make detachment harder; lower compliance strengthens it. Friction affects surface
    particles whose centers lie within one particle diameter of a collider surface. It damps relative tangential motion
    and reduces sliding at the cost of kinetic energy; zero preserves tangential speed and one removes it.

    ``c_init`` and ``c_init_z_mid`` are the initial particle concentration for the multiflow demo (0 = water,
    1 = coffee). A per-entity constant ``c_init`` takes precedence; otherwise the two-phase ``c_init_z_mid``
    split assigns c = 1 below the given z and 0 above. Both None keeps the all-zero default. Concentration is a
    passive scalar: it never couples into density, pressure, or surface tension.
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
    c_init: float | None = None
    c_init_z_mid: float | None = None
