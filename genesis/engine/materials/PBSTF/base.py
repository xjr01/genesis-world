import platform
import sys
from typing import TYPE_CHECKING, Literal

from ..base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.pbstf_entity import PBSTFEntity

SamplerType = Literal["pbs", "random", "regular", "staggered"]
DEFAULT_SAMPLER: SamplerType = "pbs" if (sys.platform == "linux" and platform.machine() == "x86_64") else "random"


class Base(Material["PBSTFEntity"]):
    """Base class for position-based surface-tension fluids."""

    sampler: SamplerType = DEFAULT_SAMPLER
