from typing import Literal

from pydantic import StrictBool

from genesis.typing import NonNegativeFloat

from .base import Base


class Elastic(Base):
    """Neo-Hookean tetrahedra with volume-preserving plastic flow.

    model="stable_neohookean" keeps finite energy at zero and negative signed volume, permitting inversion recovery
    subject to constraints, loading and convergence. model="log_neohookean" increases compression resistance without
    bound near zero volume, but requires positive signed volume and halts on collapsed or inverted states. Both models
    use the same small-strain elastic moduli. Plastic flow updates at positive signed volume.

    is_volume_plasticity_enabled retains deformation after unloading, while its default preserves elastic recovery.
    volume_yield_strain_percent sets the deviatoric logarithmic strain threshold through log(1 + percent/100). Smaller
    thresholds produce permanent deformation more readily. volume_plastic_hardening raises the threshold by this
    coefficient times accumulated equivalent plastic strain, making subsequent plastic flow harder.
    """

    model: Literal["stable_neohookean", "log_neohookean"] = "stable_neohookean"
    is_volume_plasticity_enabled: StrictBool = False
    volume_yield_strain_percent: NonNegativeFloat = 10.0
    volume_plastic_hardening: NonNegativeFloat = 0.0
