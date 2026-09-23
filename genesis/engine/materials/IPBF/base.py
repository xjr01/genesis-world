from typing import TYPE_CHECKING, Literal

from ..base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.ipbf_entity import IPBFEntity

SamplerType = Literal["pbs", "random", "regular"]


class Base(Material["IPBFEntity"]):
    """
    The base class of IPBF materials.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    sampler : str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux x86 for now.
        Defaults to 'regular'.
    """

    sampler: SamplerType = "regular"
