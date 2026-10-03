from typing import TYPE_CHECKING, Literal

from genesis.typing import NonNegativeInt, PositiveFloat

from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.dem_entity import DEMEntity

SamplerType = Literal["fcc", "poisson"]
DEFAULT_SAMPLER: SamplerType = "fcc"


class Sand(Base["DEMEntity"]):
    """
    The sand material class for DEM.

    The defaults follow the reference implementation (`sand-water-coupling-PIC-DEM-3d`): the grain stiffness
    (`young_modulus`) and density (`rho`) enter the contact model only through the ratio `young / rho`, so the
    reference values are kept verbatim to reproduce its stiffness-to-gravity ratio exactly.

    Parameters
    ----------
    rho : float, optional
        The density of the sand grain. Defaults to 2.5, matching the reference implementation.
    young_modulus : float, optional
        Young's modulus of the sand grain. The normal contact stiffness is `young_modulus * radius`.
        Defaults to 1e6, matching the reference implementation.
    poisson_ratio : float, optional
        Poisson ratio of the sand grain. The tangential contact stiffness is `young * radius * poisson_ratio`.
        Defaults to 0.3, matching the reference implementation.
    friction_angle : float, optional
        Friction angle in radians. Tangential contact forces are clamped by the Coulomb limit
        `|f_shear| <= |f_normal| * tan(friction_angle)`. Defaults to 0.5, matching the reference implementation.
    max_ratio : float, optional
        Maximum water-mass fraction a grain can absorb (reference value 0.1). Larger values absorb more
        water per grain and faster (more absorption events per cell per water step). Defaults to 0.1.
    sampler : str, optional
        Initial particle placement, matching the two modes of the reference volume sampler: 'fcc' places grains on
        a face-centered cubic lattice, 'poisson' uses Poisson-disk sampling. Both enforce a nearest-neighbor
        distance of `2.01 * radius`. 'fcc' packs slightly denser and is deterministic; 'poisson' avoids lattice
        artifacts at the cost of a one-time sampling loop. Defaults to 'fcc'.
    sampler_seed : int, optional
        Random seed for the 'poisson' sampler (the reference implementation is non-deterministic; a fixed seed
        keeps runs reproducible). Ignored by the 'fcc' sampler. Defaults to 42.
    """

    rho: PositiveFloat = 2.5
    young_modulus: PositiveFloat = 1e6
    poisson_ratio: PositiveFloat = 0.3
    friction_angle: PositiveFloat = 0.5
    max_ratio: PositiveFloat = 0.1
    sampler: SamplerType = DEFAULT_SAMPLER
    sampler_seed: NonNegativeInt = 42
