from pydantic import StrictBool

from genesis.typing import NonNegativeFloat

from .base import Base


class Elastic(Base):
    """Compressible Neo-Hookean tetrahedra with volume-preserving plastic flow.

    is_volume_plasticity_enabled retains deformation after unloading, while its default preserves elastic recovery.
    volume_yield_strain_percent sets the deviatoric logarithmic strain threshold through log(1 + percent/100). Smaller
    thresholds produce permanent deformation more readily. volume_plastic_hardening raises the threshold by this
    coefficient times accumulated equivalent plastic strain, making subsequent plastic flow harder.
    """

    is_volume_plasticity_enabled: StrictBool = False
    volume_yield_strain_percent: NonNegativeFloat = 10.0
    volume_plastic_hardening: NonNegativeFloat = 0.0
