from typing import TYPE_CHECKING

from genesis.typing import NonNegativeFloat, PositiveFloat, PositiveInt, UnitInterval, Vec3FArrayType

from .base import Base
from .liquid import DEFAULT_SAMPLER, SamplerType

if TYPE_CHECKING:
    from genesis.engine.entities.pbd_entity import PBD3DSolidEntity

# resolved defaults of the cohesive seam thresholds (None on the material): a seam starts overloading
# once its fragments slip tangentially by 20% of a bond length or hinge by 0.25 rad (~14 degrees),
# and dies once the accumulated overload fraction reaches 1.0
DEFAULT_SHEAR_THRESHOLD = 0.2
DEFAULT_ROTATION_THRESHOLD = 0.25
DEFAULT_SEAM_FAILURE_THRESHOLD = 1.0


class Solid(Base["PBD3DSolidEntity"]):
    """
    The rigid / fracturable solid material class for PBD, following the shape-matching model of
    Macklin et al. 2014, "Unified Particle Physics for Real-Time Applications" (SIGGRAPH).

    The body is sampled into particles grouped into shape-matching clusters that each act as a rigid
    region. With `fracture_threshold = 0` the body stays intact and supports plasticity. With
    `fracture_threshold > 0` the body is pre-split into rigid fragments joined by breakable bonds; a
    overstressed bond contributes to irreversible seam damage. Fragments connected by live
    bonds share one rigid shape fit, constraining relative translation and rotation; disconnected
    groups fit independently and collide as separate fragments. Failure is measured before this fit
    heals contact deformation. The mass-weighted connected-group projection also supplies reaction
    forces and moments to the entire interface graph, including seams away from the contact. A
    cached double-precision graph factor distributes these reactions through the live connections.
    Each sampled connection uses reduced particle mass and a spherical patch of radius half the
    particle spacing, with rotational inertia 2/5 times reduced mass times radius squared. This is a
    finite-volume particle connection model; its capacity is not inferred from the rendered cap area.

    Fully free, unit-stiffness groups without plastic creep also project their velocities onto a
    mass-weighted rigid translation and rotation. This removes internal velocity modes while
    preserving linear and angular momentum and never increasing kinetic energy. Fixed, attached,
    softer or plastically evolving groups retain their particle velocity path. Contact deformation
    and graph reactions are measured before this velocity projection.

    Parameters
    ----------
    rho : float, optional
        The density of the material in kg/m^3. Default is 1000.0.
    sampler : str, optional
        Particle sampler ('pbs', 'regular', 'random'). 'regular' gives the most uniform fragment behavior;
        'pbs' is only supported on Linux x86. Defaults to 'pbs' on supported platforms, 'random' otherwise.
    static_friction : float, optional
        Static friction coefficient. Default is 0.5.
    kinetic_friction : float, optional
        Kinetic friction coefficient. Default is 0.4.
    stiffness : float, optional
        Fraction of the cluster or connected-group shape correction per solver iteration, in [0, 1]. Lower values make
        the body softer and more deformable before breaking; higher values are more rigid but can jitter
        under heavy impact. Default is 1.0.
    fracture_threshold : float, optional
        Axial strain threshold: the larger of |d - d0| / d0 and the equivalent axial displacement
        transmitted by the connected-group reaction, divided by bond rest length. With
        `seam_failure_threshold=0`, only the distance criterion breaks individual bonds. Zero makes
        the body unbreakable. Strength depends on geometry, sampling and impact; calibrate it with
        an intact load control. Default is 0.0.
    shear_threshold : float, optional
        Tangential slip strain (slip / bond rest length) between two fragments beyond which their seam
        starts accumulating damage. Smaller values crack seams under glancing impacts that pure stretch
        would survive; larger values permit greater tangential projection residual before failure.
        The equivalent graph-reaction slip is included with the same bond-length normalization. Zero disables
        slip damage. Default is None (resolved to 0.2 when `fracture_threshold` > 0).
    rotation_threshold : float, optional
        Relative rotation in radians between two fragments beyond which their seam starts accumulating
        damage, measured by independent fragment fits before their connected-group projection and
        by the equivalent relative rotation transmitted through the reaction graph.
        Smaller values open seams under mild bending; larger values tolerate greater angular impact
        residuals before separation. Zero disables rotation damage. Default is None (resolved to 0.25 when
        `fracture_threshold` > 0).
    seam_failure_threshold : float, optional
        Accumulated damage (in units of overloaded bond fraction per substep) at which a whole seam dies
        at once, taking all of its bonds with it. At 1.0 a fully overloaded seam fails in a single
        substep while partial overload accumulates over several; larger values let cracked seams carry
        load longer before snapping; zero restricts failure to the last bond breaking. Default is None
        (resolved to 1.0 when `fracture_threshold` > 0).
    n_fragments : int, optional
        Number of rigid fragments the body is split into when `fracture_threshold` > 0, partitioned by
        k-means over the sampled particles. More fragments give finer shards at higher memory and compute
        cost. Ignored when `fragment_seeds` is given. Default is 8.
    fragment_seeds : array_like of shape (K, 3), optional
        Voronoi seed positions in the morph's local frame defining the fragment partition; every particle
        joins the fragment of its nearest seed. Use seeds to control the shard pattern, e.g. angular sectors
        for a vase. Default is None (k-means with `n_fragments`).
    bond_radius_factor : float, optional
        Search radius in units of `PBDOptions.particle_size` for bonds linking neighboring fragments. Larger
        values add more failure samples, discrete interface capacity and connectivity at higher memory
        and detection cost; smaller
        values may leave narrow interfaces unconnected. Live connected fragments share a rigid fit
        regardless of bond count. Default is 1.3.
    cluster_radius_factor : float, optional
        Cluster radius in units of `PBDOptions.particle_size`. Only used by the unbreakable (plastic) mode,
        where overlapping clusters let the body deform. Default is 2.5.
    cluster_spacing_factor : float, optional
        Spacing between cluster centers in units of `PBDOptions.particle_size`. Only used by the
        unbreakable (plastic) mode. Smaller values increase cluster overlap for smoother deformation at
        higher memory and compute cost. Default is 1.5.
    yield_threshold : float, optional
        Deviation in units of `PBDOptions.particle_size` beyond which deformation becomes permanent.
        Requires `plastic_creep` > 0 and `fracture_threshold` = 0. Default is 0.0.
    plastic_creep : float, optional
        Fraction of the excess deformation absorbed into the rest shape per substep once `yield_threshold`
        is exceeded, in [0, 1]. Higher values dent faster, but repeated violent impacts then drift the rest
        shape; keep at or below roughly 0.05 for clean dents. Default is 0.0.
    """

    rho: PositiveFloat = 1000.0
    sampler: SamplerType = DEFAULT_SAMPLER
    static_friction: NonNegativeFloat = 0.5
    kinetic_friction: NonNegativeFloat = 0.4
    stiffness: UnitInterval = 1.0
    fracture_threshold: NonNegativeFloat = 0.0
    shear_threshold: NonNegativeFloat | None = None
    rotation_threshold: NonNegativeFloat | None = None
    seam_failure_threshold: NonNegativeFloat | None = None
    n_fragments: PositiveInt = 8
    fragment_seeds: Vec3FArrayType | None = None
    bond_radius_factor: PositiveFloat = 1.3
    cluster_radius_factor: PositiveFloat = 2.5
    cluster_spacing_factor: PositiveFloat = 1.5
    yield_threshold: NonNegativeFloat = 0.0
    plastic_creep: UnitInterval = 0.0
