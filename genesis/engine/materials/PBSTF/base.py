from typing import TYPE_CHECKING

from ..base import Material
from ..particle_fluid import DEFAULT_SAMPLER, SamplerType

if TYPE_CHECKING:
    from genesis.engine.entities.pbstf_entity import PBSTFEntity
    from genesis.engine.entities.pbstf_porous_entity import PBSTFPorousEntity


class Base(Material["PBSTFEntity | PBSTFPorousEntity"]):
    """Base class for position-based surface-tension fluid (PBSTF) materials."""

    sampler: SamplerType = DEFAULT_SAMPLER
