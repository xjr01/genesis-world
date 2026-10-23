from typing import TYPE_CHECKING, Annotated

from pydantic import Field

from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from ..base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.vbd_entity import VBDEntity


class Base(Material["VBDEntity"]):
    """Material parameters for vertex block descent (VBD).

    E is Young's modulus in pascals, nu is Poisson's ratio, and rho is density in kg/m^3. Larger E resists deformation;
    larger nu resists volume change. Larger rho increases inertia and weight. friction_mu controls both ground and
    self-contact: larger values resist sliding, while zero permits frictionless motion.
    """

    E: PositiveFloat = 100000.0
    nu: Annotated[ValidFloat, Field(gt=0.0, lt=0.5)] = 0.25
    rho: PositiveFloat = 360.0
    friction_mu: NonNegativeFloat = 0.4
