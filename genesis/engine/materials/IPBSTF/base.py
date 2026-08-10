from typing import TYPE_CHECKING

from ..base import Material
from ..particle_fluid import DEFAULT_SAMPLER, SamplerType

if TYPE_CHECKING:
    from genesis.engine.entities.ipbstf_entity import IPBSTFEntity


class Base(Material["IPBSTFEntity"]):
    """Base material for implicit position-based fluids."""

    sampler: SamplerType = DEFAULT_SAMPLER
