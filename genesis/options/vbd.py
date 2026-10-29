from typing import Annotated, Literal

from pydantic import Field, StrictBool

from genesis.typing import NonNegativeFloat, PositiveFloat, PositiveInt, ValidFloat, Vec3FType

from .options import Options


class VBDOptions(Options):
    """Vertex block descent (VBD) settings shared by triangle shells and tetrahedral solids.

    dt and gravity inherit SimOptions. iterations bounds the work available for accurate stiffness and contact.
    convergence_tolerance stops when every accepted vertex correction is below this distance in metres: larger
    values save work at the cost of residual force imbalance. The default zero runs every iteration and skips the
    convergence reduction; positive values can save iterations during simple motion at the cost of convergence checks.
    contact_d_hat is the contact activation gap outside the physical surface in metres. Shell contact accounts for
    contact_thickness before applying this gap.
    Larger gaps improve clearance under load at the cost of a wider visible separation and more contact work.
    Zero activates solid and plane contact at the physical surface. Finite penalties allow load-dependent penetration;
    is_contact_diagnostics_enabled checks for geometric penetration and reports it while simulation continues.
    Enable it when tuning contact clearance; disabling it saves the extra checks during normal simulation.
    contact_min_volume_ratio excludes tetrahedra from containment tests when their absolute current-to-rest volume ratio
    is smaller than this value. Larger values suppress unreliable contacts near collapse at the cost of missing contacts
    in compressed regions. Zero retains every nonzero-volume tetrahedron. Material forces remain active at every ratio.
    contact_search_radius is the shell contact search distance in metres, exceeding maximum thickness plus contact_d_hat.
    Larger values allow longer motion between searches at the cost of more candidate pairs; smaller values require
    more frequent searches.
    friction_epsilon is the regularization speed in m/s: smaller values approximate sticking more
    closely, while larger values make small sliding motions smoother.
    contact_stiffness and shell_contact_stiffness are the solid and shell contact resistances in N/m. Larger values
    reduce penetration and contact-gap compression but require more iterations. is_line_search_enabled enables local
    energy backtracking for difficult deformations at the cost of extra energy evaluations. False accepts each limited step
    for faster iteration. line_search_iterations limits the backtracking attempts when enabled.
    shell_initialization="previous" starts each substep at its previous positions, reducing prediction impulses at
    the cost of more iterations for free motion; "inertia" starts from a limited inertial prediction.
    shell_step_size scales each shell correction; smaller values stabilize simultaneous contacts but converge slower.
    contact_relaxation bounds shell motion as a fraction of clearance; smaller values increase search frequency.
    shell_ground_stiffness and shell_ground_friction_epsilon override the corresponding shell
    contact settings for planes. shell_ground_friction overrides material friction. None shares the existing setting.
    These controls let a floor use a softer penalty or different sliding response than cloth self-contact.
    shell_ground_d_hat sets the plane activation distance from the simulated vertices. None uses contact_d_hat outside
    the shell contact surface. Zero permits a ground penalty with no clearance; larger values separate cloth from planes.
    """

    dt: PositiveFloat | None = None
    gravity: Vec3FType | None = None
    iterations: PositiveInt = 20
    convergence_tolerance: NonNegativeFloat = 0.0
    contact_d_hat: NonNegativeFloat = 0.00001
    contact_min_volume_ratio: NonNegativeFloat = 0.01
    friction_epsilon: PositiveFloat = 0.001
    contact_stiffness: PositiveFloat = 100000.0
    shell_contact_stiffness: PositiveFloat = 10000.0
    contact_search_radius: PositiveFloat = 0.002
    contact_relaxation: Annotated[ValidFloat, Field(gt=0.0, lt=0.5)] = 0.45
    shell_initialization: Literal["inertia", "previous"] = "inertia"
    shell_step_size: Annotated[ValidFloat, Field(gt=0.0, le=1.0)] = 0.5
    shell_ground_stiffness: PositiveFloat | None = None
    shell_ground_d_hat: NonNegativeFloat | None = None
    shell_ground_friction: NonNegativeFloat | None = None
    shell_ground_friction_epsilon: PositiveFloat | None = None
    is_contact_diagnostics_enabled: StrictBool = False
    is_line_search_enabled: StrictBool = False
    line_search_iterations: PositiveInt = 12
