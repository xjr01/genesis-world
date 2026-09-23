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
    surface_tension : float, optional
        Physical surface tension coefficient sigma (N/m), used only by the linear ST model
        (`IPBFOptions.st_model="linear"`). Defaults to 15.7 (capillary length ~ 2 particle
        diameters at ps=0.02; derivation doc D3). The quadratic model instead takes its
        normalized weight from `IPBFOptions.st_stiffness`.
    c_init : float, optional
        Initial particle concentration for the multiflow demo (0 = water, 1 = coffee).
        Every particle of this entity starts with this constant value. Defaults to 0.0.
    boundary_group : int, optional
        Clamp ownership for the multiflow container demos (MF-11): 0 = the solver's primary
        boundary (big cup), 1 = the pitcher container (`IPBFOptions.boundary_pitcher`). Group-1
        particles are clamped by the pitcher only and permanently transfer to group 0 once they
        leave the pitcher's keep region (i.e. after pouring out). Defaults to 0.
    """

    rho: PositiveFloat = 1000.0
    sampler: SamplerType = "regular"
    surface_tension: PositiveFloat = 15.7
    c_init: float = 0.0
    boundary_group: int = 0
