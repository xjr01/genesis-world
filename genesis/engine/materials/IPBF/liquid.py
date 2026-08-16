from genesis.typing import PositiveFloat

from .base import Base, SamplerType


class Liquid(Base):
    """
    The liquid material class for IPBF.

    Parameters
    ----------
    rho : float, optional
        The rest density (kg/m³). Default is 1000.0.
    sampler : str, optional
        Particle sampler. Defaults to 'regular' for numerical stability.
    """

    rho: PositiveFloat = 1000.0
    sampler: SamplerType = "regular"
