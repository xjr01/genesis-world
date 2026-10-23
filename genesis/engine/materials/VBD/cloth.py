from typing import Annotated, Literal

from pydantic import Field, StrictBool

from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .base import Base


class Cloth(Base):
    """Triangle shell with independent shear resistance and discrete hinge bending.

    thickness is the full thickness in metres; increasing it increases mass, membrane stiffness and contact spacing.
    shear_stiffness adds in-plane deviatoric resistance in N/m independently of thickness and the elastic moduli.
    bending_rigidity is the hinge rigidity in N*m; larger values resist folds. bending_model=None permits free hinges
    and disables bending plasticity. is_bending_plasticity_enabled retains folds after unloading, while its default
    preserves elastic recovery. Lower bending_yield_angle_deg produces creases more readily; bending_plastic_hardening
    raises the angular yield threshold by this coefficient times the accumulated plastic angle in radians.
    """

    shear_stiffness: PositiveFloat = 8.0
    thickness: PositiveFloat = 0.00025
    bending_model: Literal["DiscreteShellBending"] | None = "DiscreteShellBending"
    bending_rigidity: NonNegativeFloat = 5.0e-8
    is_bending_plasticity_enabled: StrictBool = False
    bending_yield_angle_deg: Annotated[ValidFloat, Field(ge=0.0, le=180.0)] = 10.0
    bending_plastic_hardening: NonNegativeFloat = 0.0
