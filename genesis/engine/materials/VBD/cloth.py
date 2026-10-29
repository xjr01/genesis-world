from typing import Annotated, Literal

from pydantic import Field, StrictBool

from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .base import Base


class Cloth(Base):
    """Triangle shell with independent shear resistance, bending and stiffness-proportional damping.

    thickness is the full thickness in metres; increasing it increases mass and membrane stiffness. contact_thickness
    sets the full collision thickness independently; None uses thickness. Larger values maintain wider contact spacing.
    shear_stiffness adds in-plane deviatoric resistance in N/m independently of thickness and the elastic moduli;
    zero uses the shear resistance from E and nu alone, while larger values reduce shear deformation.
    bending_rigidity is the rigidity in N*m; larger values resist folds. DiscreteShellBending restores initial dihedral
    angles; QuadraticBending minimizes a rest-mesh cotangent curvature and favors flat cloth. None permits free hinges.
    membrane_damping and bending_damping are local stiffness-proportional damping times in seconds: larger values
    suppress vibration but also slow motion. is_bending_plasticity_enabled retains folds after unloading with
    DiscreteShellBending, while its default preserves elastic recovery. Lower bending_yield_angle_deg produces creases
    more readily; bending_plastic_hardening
    raises the angular yield threshold by this coefficient times the accumulated plastic angle in radians.
    """

    shear_stiffness: NonNegativeFloat = 8.0
    thickness: PositiveFloat = 0.00025
    contact_thickness: NonNegativeFloat | None = None
    membrane_damping: NonNegativeFloat = 0.0
    bending_damping: NonNegativeFloat = 0.0
    bending_model: Literal["DiscreteShellBending", "QuadraticBending"] | None = "DiscreteShellBending"
    bending_rigidity: NonNegativeFloat = 5.0e-8
    is_bending_plasticity_enabled: StrictBool = False
    bending_yield_angle_deg: Annotated[ValidFloat, Field(ge=0.0, le=180.0)] = 10.0
    bending_plastic_hardening: NonNegativeFloat = 0.0
