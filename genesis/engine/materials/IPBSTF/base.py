from typing import TYPE_CHECKING, ClassVar

from ..base import Material
from ..particle_fluid import DEFAULT_SAMPLER, SamplerType

if TYPE_CHECKING:
    from genesis.engine.entities.ipbstf_entity import IPBSTFEntity


class Base(Material["IPBSTFEntity"]):
    """Base material for implicit position-based fluids."""

    is_fixed: ClassVar[bool] = False
    sampler: SamplerType = DEFAULT_SAMPLER
