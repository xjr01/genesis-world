from typing import Annotated, Any

from pydantic import Field

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, UnitInterval, ValidFloat

from .base import Base, SamplerType


class PorousElastic(Base):
    """Porous elastic material coupled to a position-based surface-tension fluid.

    The solid matrix uses compliant position-based elasticity. Lower compliance makes the matrix resist deformation
    more strongly but requires more solver work, while higher compliance makes it easier to deform. ``porosity``
    controls the available pore volume and therefore the amount of liquid the material can contain. Capillary
    attraction and drag retain liquid but also make separation and relative motion harder.

    Wet compliance scales interpolate from one in the dry state to the configured value at full saturation. Values
    below one stiffen the wet matrix and values above one soften it. ``bloating_volume_strain`` expands the fully
    saturated matrix and increases deformation around absorbed liquid.

    Parameters
    ----------
    rho : float
        Density of the solid matrix. Higher values increase inertia and the force needed to accelerate the sponge.
    porosity : float
        Dry pore-volume fraction. Higher values hold more liquid but leave less solid mass at a fixed outer volume.
    deviatoric_compliance : float
        Reciprocal shape modulus in inverse pressure units. Lower values preserve shape more strongly but require more
        solver iterations.
    volumetric_compliance : float
        Reciprocal bulk modulus in inverse pressure units. Lower values preserve volume but make aggressive compression
        less stable.
    pore_compliance : float
        Compliance of the pore-collapse limit. Lower values prevent over-compression more strongly.
    capillary_compliance : float | None
        Compliance of liquid attraction. Lower values retain liquid more strongly; ``None`` disables attraction.
    capillary_saturation_falloff : float
        Fraction of capillary attraction removed at full saturation. Larger values limit overfilling more strongly.
    drag : float
        Resistance to liquid-solid relative motion. Higher values retain moving liquid but dissipate more energy.
    wet_deviatoric_compliance_scale : float
        Full-saturation multiplier for shape compliance. Values below one stiffen and values above one soften.
    wet_volumetric_compliance_scale : float
        Full-saturation multiplier for volume compliance. Values below one stiffen and values above one soften.
    bloating_volume_strain : float
        Full-saturation target expansion. Higher values produce more swelling and stronger geometric displacement.
    """

    rho: PositiveFloat = 1000.0
    porosity: Annotated[ValidFloat, Field(gt=0.0, lt=1.0)] = 0.8
    deviatoric_compliance: NonNegativeFloat = 1e-5
    volumetric_compliance: NonNegativeFloat = 1e-5
    pore_compliance: NonNegativeFloat = 0.0
    capillary_compliance: NonNegativeFloat | None = None
    capillary_saturation_falloff: UnitInterval = 1.0
    drag: NonNegativeFloat = 0.0
    wet_deviatoric_compliance_scale: NonNegativeFloat = 1.0
    wet_volumetric_compliance_scale: NonNegativeFloat = 1.0
    bloating_volume_strain: NonNegativeFloat = 0.0
    sampler: SamplerType | None = None

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.sampler is not None and self.sampler != "staggered":
            gs.raise_exception("PBSTF porous elastic materials require staggered particle sampling.")
        self.sampler = "staggered"
