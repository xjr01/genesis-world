from typing import TYPE_CHECKING

from ..base import Material
from ..particle_fluid import DEFAULT_SAMPLER, SamplerType

if TYPE_CHECKING:
    from genesis.engine.entities.pbstf_entity import PBSTFEntity


class Base(Material["PBSTFEntity"]):
    """Base class for position-based surface-tension fluids."""

    sampler: SamplerType = DEFAULT_SAMPLER
