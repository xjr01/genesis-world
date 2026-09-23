import platform
import sys
from typing import TYPE_CHECKING, Literal

from genesis.typing import PositiveFloat, ValidFloat

from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.pbd_entity import PBDParticleEntity

SamplerType = Literal["pbs", "random", "regular"]
DEFAULT_SAMPLER: SamplerType = "pbs" if (sys.platform == "linux" and platform.machine() == "x86_64") else "random"


class Liquid(Base["PBDParticleEntity"]):
    """
    The liquid material class for PBD.

    Parameters
    ----------
    rho : float, optional
        The rest density of the fluid in kg/m³. Default is 1000.0.
    sampler : str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux x86 for now. Defaults
        to 'pbs' on supported platforms, 'random' otherwise.
    density_relaxation : float, optional
        Relaxation factor for solving the density constraint. Default is 0.2.
    viscosity_relaxation : float, optional
        Relaxation factor used in the viscosity solver. Default is 0.01.
    c_init_z_mid : float or None, optional
        If given, particles sampled below this world z are initialized with concentration c=1 ("coffee"),
        the rest with c=0 ("water"). None keeps the all-zero default. Default is None.
    c_init : float or None, optional
        If given, all particles of this entity are initialized with this constant concentration,
        overriding `c_init_z_mid`. Default is None.
    boundary_group : int, optional
        Clamp ownership group (multiflow pitcher demos): 0 = primary boundary, 1 = pitcher clamp
        (requires `PBDOptions.boundary_pitcher`). Group-1 particles transfer to group 0 permanently
        once they leave the pitcher's keep region. Default is 0.
    """

    rho: PositiveFloat = 1000.0
    sampler: SamplerType = DEFAULT_SAMPLER
    density_relaxation: ValidFloat = 0.2
    viscosity_relaxation: ValidFloat = 0.01
    c_init_z_mid: float | None = None
    c_init: float | None = None
    boundary_group: int = 0
