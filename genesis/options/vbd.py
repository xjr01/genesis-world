from pydantic import StrictBool

from genesis.typing import NonNegativeFloat, PositiveFloat, PositiveInt, Vec3FType

from .options import Options


class VBDOptions(Options):
    """Vertex block descent (VBD) settings shared by triangle shells and tetrahedral solids.

    dt and gravity inherit SimOptions. iterations bounds the work available for accurate stiffness and contact.
    convergence_tolerance stops when every accepted vertex correction is below this distance in metres: larger
    values save work at the cost of residual force imbalance. Zero runs every iteration.
    contact_d_hat is the contact activation gap outside the physical surface in metres; larger gaps improve clearance
    at the cost of a wider visible separation and more contact work. Smaller gaps can require more iterations during
    fast deformation. Self-contact uses the reference clearance as an upper bound to preserve densely sampled shapes.
    friction_epsilon is the regularization speed in m/s: smaller values approximate sticking more
    closely, while larger values make small sliding motions smoother.
    contact_stiffness scales the automatic inertia-based contact resistance; larger values reduce compression
    of the contact gap but require more iterations. is_line_search_enabled enables local energy backtracking for
    difficult deformations at the cost of extra energy evaluations. False accepts each contact-limited Newton step
    for faster iteration. line_search_iterations limits the backtracking attempts when enabled.
    """

    dt: PositiveFloat | None = None
    gravity: Vec3FType | None = None
    iterations: PositiveInt = 20
    convergence_tolerance: NonNegativeFloat = 1e-7
    contact_d_hat: PositiveFloat = 0.0005
    friction_epsilon: PositiveFloat = 0.001
    contact_stiffness: PositiveFloat = 10.0
    is_line_search_enabled: StrictBool = False
    line_search_iterations: PositiveInt = 12
