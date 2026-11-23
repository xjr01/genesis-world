from typing import TYPE_CHECKING

from genesis.typing import PositiveFloat

from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.flip_entity import FLIPEntity


class Liquid(Base["FLIPEntity"]):
    """
    The liquid material class for FLIP.

    Parameters
    ----------
    rho : float, optional
        The rest density of the liquid. Defaults to 1000.0.
    """

    rho: PositiveFloat = 1000.0
