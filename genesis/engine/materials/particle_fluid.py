import platform
import sys
from typing import Literal

SamplerType = Literal["pbs", "random", "regular", "staggered"]
DEFAULT_SAMPLER: SamplerType = "pbs" if sys.platform == "linux" and platform.machine() == "x86_64" else "random"
