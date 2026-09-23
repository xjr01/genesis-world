from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, PrivateAttr, StrictBool, StrictInt, model_validator

import genesis as gs
from genesis.typing import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    UnitInterval,
    UnitVec4FType,
    Vec3FType,
)

from .options import Options

############################ Top level: simulator and coupler ############################
"""
Simulator options specifies the global settings for the simulator and the coupler options specifies whether the coupling between pairs of solvers is enabled.
"""


class SimOptions(Options):
    """
    Options configuring the top-level simulator.

    Note
    ----
    1. `SimOptions` specifies the global settings for the simulator. Some parameters exist both in `SimOptions` and `SolverOptions`. In this case, if such parameters are given in `SolverOptions`, it will override the one specified in `SimOptions` for this specific solver. For example, if `dt` is only given in `SimOptions`, it will be shared by all the solvers, but it's also possible to let a solver run at a different temporal speed by setting its own `dt` to be a different value.

    2. In differentiable mode, `substeps_local` must be divisible by `substeps`, as external command is input per `step`, but `substep`. If `requires_grad` is False, we can use arbitrary `substeps_local`.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. Defaults to 1e-2.
    substeps : int, optional
        Number of substeps per simulation step. Defaults to 1.
    substeps_local : int, optional
        Number of substeps stored in GPU memory. Defaults to None. This is used for differentiable mode.
    gravity : tuple, optional
        Gravity force in N/kg. Defaults to (0.0, 0.0, -9.81).
    floor_height : float, optional
        Height of the floor in meters. Defaults to 0.0.
    requires_grad : bool, optional
        Whether to enable differentiable mode. Defaults to False.
    use_hydroelastic_contact : bool, optional
        Whether to use hydroelastic contact. Defaults to False.
    """

    dt: PositiveFloat = 1e-2
    substeps: PositiveInt = 1
    substeps_local: PositiveInt | None = None  # number of substeps stored in GPU memory
    gravity: Vec3FType = (0.0, 0.0, -9.81)
    floor_height: float = 0.0
    requires_grad: StrictBool = False

    _steps_local: int | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_substeps(cls, data: dict) -> dict:
        if data.get("substeps_local") is None:
            # use 1 to save gpu memory when not in differentiable mode
            data["substeps_local"] = data.get("substeps", 1) if data.get("requires_grad", False) else 1
        return data

    def model_post_init(self, context: Any) -> None:
        if self.requires_grad:
            if self.substeps_local % self.substeps != 0:
                gs.raise_exception("`substeps_local` must be divisible by `substeps` when `requires_grad` is True.")
            else:
                self._steps_local = int(self.substeps_local / self.substeps)
        else:
            self._steps_local = None


class BaseCouplerOptions(Options):
    """
    Base class for all coupler options.
    """

    pass


class LegacyCouplerOptions(BaseCouplerOptions):
    """
    Options configuring the inter-solver coupling.

    Parameters
    ----------
    rigid_mpm : bool, optional
        Whether to enable coupling between rigid and MPM solvers. Defaults to True.
    rigid_sph : bool, optional
        Whether to enable coupling between rigid and SPH solvers. Defaults to True.
    rigid_pbd : bool, optional
        Whether to enable coupling between rigid and PBD solvers. Defaults to True.
    rigid_fem : bool, optional
        Whether to enable coupling between rigid and FEM solvers. Defaults to True.
    mpm_sph : bool, optional
        Whether to enable coupling between MPM and SPH solvers. Defaults to True.
    mpm_pbd : bool, optional
        Whether to enable coupling between MPM and PBD solvers. Defaults to True.
    fem_mpm : bool, optional
        Whether to enable coupling between FEM and MPM solvers. Defaults to True.
    fem_sph : bool, optional
        Whether to enable coupling between FEM and SPH solvers. Defaults to True.
    """

    rigid_mpm: StrictBool = True
    rigid_sph: StrictBool = True
    rigid_pbd: StrictBool = True
    rigid_fem: StrictBool = True
    mpm_sph: StrictBool = True
    mpm_pbd: StrictBool = True
    fem_mpm: StrictBool = True
    fem_sph: StrictBool = True


class SAPCouplerOptions(BaseCouplerOptions):
    """
    Options configuring the inter-solver coupling for the Semi-Analytic Primal (SAP) contact solver used in Drake.

    Note
    ----
    Paper reference: https://arxiv.org/abs/2110.10107
    Drake reference: https://drake.mit.edu/release_notes/v1.5.0.html

    Parameters
    ----------
    n_sap_iterations : int, optional
        Number of iterations for the SAP solver. Defaults to 5.
    n_pcg_iterations : int, optional
        Number of iterations for the Preconditioned Conjugate Gradient solver. Defaults to 100.
    n_linesearch_iterations : int, optional
        Max number of iterations for the line search solver. Defaults to 10.
    sap_convergence_atol : float, optional
        Absolute tolerance for SAP convergence. Defaults to 1e-6.
    sap_convergence_rtol : float, optional
        Relative tolerance for SAP convergence. Defaults to 1e-5.
    sap_taud : float, optional
        Dissipation time scale for SAP. Defaults to 0.1.
    sap_beta : float, optional
        Normal regularization parameter for SAP. Defaults to 1.0.
    sap_sigma : float, optional
        Friction regularization parameter for SAP. Defaults to 1e-3.
    pcg_threshold : float, optional
        Threshold for the Preconditioned Conjugate Gradient solver. Defaults to 1e-6.
    linesearch_ftol : float, optional
        Line search sufficient value close to zero for exact linesearch. Defaults to 1e-6.
    linesearch_max_step_size : float, optional
        Maximum step size for exact linesearch. Defaults to 1.5.
    hydroelastic_stiffness : float, optional
        Stiffness for hydroelastic contact. Defaults to 1e8.
    point_contact_stiffness : float, optional
        Stiffness for point contact. Defaults to 1e8.
    fem_floor_contact_type : str, optional
        Type of contact against the floor. Defaults to "tet". Can be "tet", "vert", or "none".
        TET would be the default choice for most cases.
        VERT would be preferable when the mesh is very coarse, such as a single cube or a tetrahedron.
    enable_fem_self_tet_contact : bool, optional
        Whether to use tetrahedral based self-contact. Defaults to True.
    rigid_rigid_type : str, optional
        Type of contact between rigid bodies. Defaults to "tet". Can be "tet", "vert", or "none".
    rigid_floor_contact_type : str, optional
        Type of contact against the floor. Defaults to "tet". Can be "tet", "vert", or "none".
        Tet would be the default choice for most cases.
        Vert would be preferable when the mesh is very coarse, such as a single cube or a tetrahedron.
    enable_rigid_fem_contact : bool, optional
        Whether to enable coupling between rigid and FEM solvers. Defaults to True.
    """

    n_sap_iterations: PositiveInt = 5
    n_pcg_iterations: PositiveInt = 100
    n_linesearch_iterations: PositiveInt = 10
    sap_convergence_atol: PositiveFloat = 1e-6
    sap_convergence_rtol: PositiveFloat = 1e-5
    sap_taud: PositiveFloat = 0.1
    sap_beta: PositiveFloat = 1.0
    sap_sigma: PositiveFloat = 1e-3
    pcg_threshold: PositiveFloat = 1e-6
    linesearch_ftol: PositiveFloat = 1e-6
    linesearch_max_step_size: PositiveFloat = 1.5
    hydroelastic_stiffness: PositiveFloat = 1e8
    point_contact_stiffness: PositiveFloat = 1e8
    fem_floor_contact_type: Literal["tet", "vert", "none"] = "tet"
    enable_fem_self_tet_contact: StrictBool = True
    rigid_floor_contact_type: Literal["tet", "vert", "none"] = "tet"
    enable_rigid_fem_contact: StrictBool = True
    rigid_rigid_contact_type: Literal["tet", "vert", "none"] = "tet"


class IPCCouplerOptions(BaseCouplerOptions):
    """
    Options configuring the Incremental Potential Contact (IPC) coupler.

    Time step, gravity, and differentiable simulation mode are derived from ``SimOptions``
    (``dt``, ``gravity``, ``requires_grad``) and should not be set here.

    Parameters
    ----------
    Newton Solver Options
    ---------------------
    newton_max_iterations : int, optional
        Maximum iterations for Newton solver. Defaults to None (use libuipc default: 1024).
    newton_min_iterations : int, optional
        Minimum iterations for Newton solver. Defaults to None (use libuipc default: 1).
    newton_tolerance : float, optional
        Velocity tolerance for Newton solver convergence. Defaults to None (use libuipc default: 0.05).
    newton_ccd_tolerance : float, optional
        CCD (Continuous Collision Detection) tolerance for Newton solver. Defaults to None (use libuipc default: 1.0).
    newton_use_adaptive_tolerance : bool, optional
        Whether Newton solver should use adaptive tolerance. Defaults to None (use libuipc default: False).
    newton_translation_tolerance : float, optional
        Translation rate tolerance for Newton solver. Defaults to None (use libuipc default: 0.1).
    newton_semi_implicit_enable : bool, optional
        Whether to enable semi-implicit Newton solver mode. Defaults to None (use libuipc default: False).
    newton_semi_implicit_beta_tolerance : float, optional
        Beta tolerance for semi-implicit Newton solver. Defaults to None (use libuipc default: 1e-3).

    Line Search Options
    -------------------
    n_linesearch_iterations : int, optional
        Maximum iterations for line search. Defaults to None (use libuipc default: 8).
    linesearch_report_energy : bool, optional
        Whether to report energy during line search. Defaults to None (use libuipc default: False).

    Linear System Options
    ---------------------
    linear_system_solver : str, optional
        Linear system solver type. Options: 'linear_pcg', 'direct', etc. Defaults to None (use libuipc default: 'linear_pcg').
    linear_system_tolerance : float, optional
        Tolerance for linear system solver. Defaults to None (use libuipc default: 1e-3).

    Contact Options
    ---------------
    contact_enable : bool, optional
        Whether to enable contact detection. Defaults to None (use libuipc default: True).
    contact_d_hat : float, optional
        Contact distance threshold. Defaults to None (use libuipc default: 0.01).
    contact_friction_enable : bool, optional
        Whether to enable friction in contact. Defaults to None (use libuipc default: True).
    contact_resistance : float, optional
        Ground/default contact resistance/stiffness. It is used for ground contact pairs and
        as the per-entity fallback when a material does not define ``contact_resistance``.
        For ground pairs, it is combined with entity ``material.contact_resistance`` via
        geometric mean. Defaults to 1e9.
    contact_eps_velocity : float, optional
        Epsilon velocity for contact. Defaults to None (use libuipc default: 0.01).
    contact_constitution : str, optional
        Contact constitution model. Options: 'ipc', 'isometric'. Defaults to None (use libuipc default: 'ipc').

    Collision Detection Options
    ---------------------------
    collision_detection_method : str, optional
        Collision detection method. Options: 'linear_bvh', 'spatial_hash', etc. Defaults to None (use libuipc default: 'linear_bvh').

    CFL Options
    -----------
    cfl_enable : bool, optional
        Whether to enable CFL (Courant-Friedrichs-Lewy) condition. Defaults to None (use libuipc default: False).

    Sanity Check Options
    --------------------
    sanity_check_enable : bool, optional
        Whether to enable sanity checks. Defaults to None (use libuipc default: True).

    Genesis Coupling Options
    ------------------------
    constraint_strength_translation : float, optional
        Translation strength for IPC soft transform constraint coupling.
        Higher values create stiffer position coupling between Genesis rigid bodies and IPC ABD objects.
        Defaults to 100.0.
    constraint_strength_rotation : float, optional
        Rotation strength for IPC soft transform constraint coupling.
        Higher values create stiffer orientation coupling between Genesis rigid bodies and IPC ABD objects.
        Defaults to 100.0.
    enable_rigid_ground_contact : bool, optional
        Whether to enable ground contact in IPC system. When False, objects in IPC will not collide
        with the ground plane. Defaults to True.
    enable_rigid_rigid_contact : bool, optional
        Whether to enable contact detection between rigid bodies (ABD objects) in the IPC system.
        When False, only soft-soft and soft-rigid collisions are detected by IPC; rigid-rigid
        collisions within IPC are skipped. Defaults to True.
    two_way_coupling : bool, optional
        Whether to apply coupling forces/torques from IPC back to Genesis rigid bodies. Defaults to True.
    enable_rigid_dofs_sync : bool, optional
        Whether to synchronize the IPC reference DOF state with Genesis each step for
        external_articulation entities. When True, IPC gets tighter coupling with Genesis joint
        state but may amplify small divergences. When False, IPC uses its own DOF reference
        without per-step updates. Defaults to False.
    free_base_driven_by_ipc : bool, optional
        For external_articulation with non-fixed base: whether base link is fully driven by IPC physics.
        When False, base link uses SoftTransformConstraint controlled by Genesis. When True, base link
        is fully driven by IPC physics. Defaults to False.
    _show_ipc_gui : bool, optional
        [Dev/debug] Enable the libuipc built-in polyscope GUI viewer for inspecting the IPC scene.
        Defaults to False.
    """

    # Newton solver options (None = use libuipc default)
    newton_max_iterations: PositiveInt | None = None
    newton_min_iterations: PositiveInt | None = None
    newton_tolerance: PositiveFloat | None = None
    newton_ccd_tolerance: PositiveFloat | None = None
    newton_use_adaptive_tolerance: StrictBool | None = None
    newton_translation_tolerance: PositiveFloat | None = None
    newton_semi_implicit_enable: StrictBool | None = None
    newton_semi_implicit_beta_tolerance: PositiveFloat | None = None

    # Line search options (None = use libuipc default)
    n_linesearch_iterations: PositiveInt | None = None
    linesearch_report_energy: StrictBool | None = None

    # Linear system options (None = use libuipc default)
    linear_system_solver: Literal["linear_pcg", "direct"] | None = None
    linear_system_tolerance: PositiveFloat | None = None

    # Contact options
    contact_enable: StrictBool | None = None
    contact_d_hat: PositiveFloat | None = None
    contact_friction_enable: StrictBool | None = None
    contact_resistance: PositiveFloat = 1e9
    contact_eps_velocity: PositiveFloat | None = None
    contact_constitution: Literal["ipc", "isometric"] | None = None

    # Collision detection options
    collision_detection_method: Literal["linear_bvh", "spatial_hash"] | None = None

    # CFL options
    cfl_enable: StrictBool | None = None

    # Sanity check options
    sanity_check_enable: StrictBool | None = None

    # Genesis coupling options
    constraint_strength_translation: PositiveFloat = 100.0
    constraint_strength_rotation: PositiveFloat = 100.0
    enable_rigid_ground_contact: StrictBool = True
    enable_rigid_rigid_contact: StrictBool = True
    two_way_coupling: StrictBool = True
    enable_rigid_dofs_sync: StrictBool = False
    free_base_driven_by_ipc: StrictBool = False

    _show_ipc_gui: bool = PrivateAttr(default=False)

    def __init__(self, *, _show_ipc_gui: StrictBool = False, **data) -> None:
        super().__init__(**data)
        self._show_ipc_gui = bool(_show_ipc_gui)


############################ Solvers inside simulator ############################
"""
Parameters in these solver-specific options will override SimOptions if available.
"""


class KinematicOptions(Options):
    """
    Options configuring the KinematicSolver (visualization-only solver).

    KinematicSolver is a lightweight solver for ghost/reference entities that only computes
    forward kinematics for visualization. No collision, physics integration, or constraint
    solving is performed.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. If none, it will inherit from `SimOptions`. Defaults to None.
    batch_links_info : bool, optional
        Whether to batch link info. Automatically enabled for heterogeneous simulation. Defaults to False.
    batch_dofs_info : bool, optional
        Whether to batch DOF info. Defaults to False.
    IK_max_targets : int, optional
        Maximum number of IK targets. Increasing this doesn't affect IK solving speed, but will increase memory usage.
        Defaults to 6.
    """

    dt: PositiveFloat | None = None
    batch_links_info: StrictBool = False
    batch_joints_info: StrictBool = False
    batch_dofs_info: StrictBool = False
    IK_max_targets: PositiveInt = 6


class ToolOptions(Options):
    """
    Options configuring the ToolSolver.

    Note
    ----
    ToolEntity is a simplified form of RigidEntity. It supports one way tool->other coupling, but has *no* internal dynamics and can only be created from a single mesh. This is a temporal workaround for differentiable rigid-soft interaction. This solver will be removed once differentiable mode is supported by the RigidSolver.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. Defaults to 1e-2.
    floor_height : float, optional
        Height of the floor in meters. Defaults to 0.0.
    """

    dt: PositiveFloat | None = None
    floor_height: float | None = None


class RigidOptions(Options):
    """
    Options configuring the RigidSolver.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. If none, it will inherit from `SimOptions`. Defaults to None.
    gravity : tuple, optional
        Gravity force in N/kg. If none, it will inherit from `SimOptions`. Defaults to None.
    enable_collision : bool, optional
        Whether to enable collision detection. Defaults to True.
    enable_joint_limit : bool, optional
        Whether to enable joint limit. Defaults to True.
    enable_self_collision : bool, optional
        Whether to enable self collision within each entity. Defaults to True.
    enable_neutral_collision : bool, optional
        Whether to enable self collision occurring in neutral configuration (qpos0) within each entity. Defaults to
        False.
    enable_adjacent_collision : bool, optional
        Whether to enable collision between successive parent-child body pairs within each entity. Defaults to False.
    disable_constraint: bool, optional
        Whether to disable all constraints. Defaults to False.
    max_collision_pairs : int, optional
        Maximum number of collision pairs. Defaults to 100.
    max_contacts : int, optional
        Maximum number of simultaneous contact points per environment that the constraint solver can handle, which
        determines the size of the contact constraint buffers (3 to 10 constraint rows per contact point depending on
        'friction_cone', 'enable_torsional_friction', and 'enable_rolling_friction'). Defaults to None.

        This limit applies to the final contact points after pruning, not to the candidate contact points that
        collision detection can emit (see 'max_collision_pairs'). Exceeding it at runtime halts the simulation with
        an error. None resolves it automatically: the pre-pruning worst case or, when contact pruning is enabled
        (see 'contact_pruning_tolerance'), 32 contact points per candidate link pair but no less than 512, whichever
        is smaller.
    integrator : gs.integrator, optional
        Integrator type. Current supported integrators are 'gs.integrator.Euler', 'gs.integrator.implicitfast' and
        'gs.integrator.approximate_implicitfast'. 'Euler' and 'implicitfast' are consistent with their Mujoco
        counterpart. 'approximate_implicitfast' is an even faster approximation of 'implicitfast', which avoid
        computing the inverse mass matrix twice by considering the first order correction terms of the implicit
        integration scheme systematically, including for computing the acceleration resulting from the constraints
        and external forces. Although this approximation is wrong in theory, it works reasonably well in practice.
        Defaults to 'approximate_implicitfast'.
    IK_max_targets : int, optional
        Maximum number of IK targets. Increasing this doesn't affect IK solving speed, but will increase memory usage.
        Defaults to 6.
    constraint_solver : gs.constraint_solver, optional
        Constraint solver type. Current supported constraint solvers are 'gs.constraint_solver.CG' (conjugate gradient)
        and 'gs.constraint_solver.Newton' (Newton's method). Defaults to 'Newton'.
    iterations : int, optional
        Maximum number of iterations for the constraint solver; the solve exits early once its convergence tolerance
        is met, so this bound only binds on hard steps. Defaults to 50.
    tolerance : float, optional
        Tolerance for the constraint solver. If None, resolved based on the floating-point precision selected via
        `gs.init(precision=...)`: 1e-5 for single precision ("32") and 1e-8 for double precision ("64"). Defaults
        to None.
    ls_iterations : int, optional
        Number of line search iterations for the constraint solver. Defaults to 50.
    ls_tolerance : float, optional
        Tolerance for the line search. Defaults to 1e-2.
    noslip_iterations : int, optional
        Number of iterations for the noslip solver. Defaults to 0 (disabled).
        noslip is a post-processing step after the main solver to suppress slip/drift.
        Recommended to set this value to 5 for manipulation tasks or when slip/drift is a big problem.
        This option should only be enabled if necessary because it is experimental and will slow down the simulation.
    noslip_tolerance : float, optional
        Tolerance for the noslip solver. Defaults to 1e-6.
    friction_cone : gs.friction_cone, optional
        Contact friction cone model, trading numerical robustness for physical accuracy. 'gs.friction_cone.pyramidal'
        (default) is robust and easy to solve; 'gs.friction_cone.elliptic' is the exact isotropic cone, harder to solve
        but paired with a high 'impratio' it holds resting stacks without slow tangential creep. See 'gs.friction_cone'
        for the description of each model. Unsupported with the noslip solver or differentiable simulation.
    contact_resolution : gs.contact_resolution, optional
        How a contact's normal force and friction force are resolved against each other.
        'gs.contact_resolution.signorini' bounds friction against the normal force the contact has developed, so sliding
        never inflates it and a body launched horizontally decelerates at mu * g instead of lifting off, at the cost of
        extra solver iterations. 'gs.contact_resolution.convex' poses the contact as a single convex program, which
        converges more predictably on stiff scenes but lets fast sliding buy normal force. See 'gs.contact_resolution'
        for the description of each model. Defaults to None, resolving to 'signorini' with the elliptic cone and the
        Newton solver, and 'convex' otherwise - the pyramidal cone's rows do not separate, and the conjugate gradient
        solver does not reach the fixed point. Always 'convex' when 'enable_mujoco_compatibility' is set.
    enable_torsional_friction : bool, optional
        Whether contacts also resist relative spin about their normal, with strength set per geometry by the material
        option 'friction_torsional' (see 'gs.materials.Rigid'). Enable it when spin resistance matters - a grasped
        object twisting in a gripper, a top spinning in place - motions a point contact transmits no torque against,
        so they persist indefinitely otherwise. The extra spin resistance slows down the constraint solve on every
        contact, including those where spin is irrelevant. Defaults to False.
    enable_rolling_friction : bool, optional
        Whether contacts also resist rolling, with strength set per geometry by the material option 'friction_rolling'
        (see 'gs.materials.Rigid'). Enable it when rolling resistance matters - a ball or wheel coasting to rest, a
        cylinder settling on a slope - motions a point contact otherwise never slows down. The extra rolling
        resistance slows down the constraint solve on every contact, more so than torsional friction (two extra axes),
        and requires 'enable_torsional_friction'. Defaults to False.
    impratio : float, optional
        Ratio of tangential (friction) to normal constraint impedance at contacts. Raising it above 1 stiffens
        friction so resting stacks and piles hold their pose under sustained shear, at the cost of a slower solve that
        turns numerically unstable once pushed too far - a stiffness-versus-stability tradeoff, so use the smallest
        value that holds the contacts. It matters mainly with the elliptic cone, which stiffens friction alone while
        leaving the normal contact response at its own impedance. Defaults to None, resolving to 100 with the elliptic
        cone (1 when 'enable_mujoco_compatibility' is set) and 1 otherwise.
    sparse_solve : bool, optional
        Whether to exploit sparsity (skyline-envelope Cholesky) in the constraint solver.

        Defaults to None, which resolves automatically: enabled on the CPU backend (and not under MuJoCo compatibility)
        when the scene has block structure - at least two DOF-carrying bodies or at least two free joints - so the
        Hessian band stays much tighter than its dimension. Never enabled on GPU, where the dense tiled factorization
        is faster. Set True or False to override the automatic choice; True is ignored with a warning on GPU.
    contact_resolve_time : float, optional
        Please note that this option will be deprecated in a future version. Use 'constraint_timeconst'
        instead.
    constraint_timeconst : float
        Lower-bound of the default time to resolve the constraint (2*dt). The smaller the value, the more stiff the
        constraint. This parameter is called 'timeconst' in Mujoco
        (https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters). Defaults to 0.01.
    use_contact_island : bool, optional
        Whether to partition the constraint solve into independent per-island blocks. It has no effect on a scene that
        is a single dense-coupled tree (one island) or is differentiable, where the dense whole-scene solve is used
        regardless. Defaults to True.
    use_hibernation : bool, optional
        Whether to put bodies that have come to rest to sleep, so the solver skips them until they are disturbed. It
        quietly has no effect on a body that is differentiable, prunable, or under no-slip friction. Defaults to False.
    hibernation_thresh_vel : float, optional
        Velocity tolerance for hibernation, in meters per second: a body sleeps once its maximum DOF speed stays below
        this for a few consecutive steps, and a whole island sleeps once all its bodies are ready. Each rotational DOF
        is weighted by the body's swept radius, so the tolerance is a single linear speed that applies uniformly to
        translation and rotation. If None, it is set to 1e-4 when MuJoCo compatibility is enabled (matching MuJoCo's
        default) and 2e-3 otherwise. Defaults to None.
    max_dynamic_constraints : int, optional
        Maximum number of dynamic constraints (like suction cup). Defaults to 8.
    use_gjk_collision: bool, optional
        Whether to use GJK for collision detection instead of MPR. More stable but much slower. Defaults to
        `sim_options.requires_grad`.
    enable_contact_patch: bool, optional
        Whether to recover the full contact patch from the touching faces inside GJK, in a single detection pass,
        instead of through perturbed re-detections. The contact patch is cheaper and reports the exact contact
        polygon, but it is discouraged: it is less reliable than the perturbation-based detection, which is extremely
        robust at the cost of extra detection passes. Requires GJK collision detection, and raises otherwise. If
        None, it is enabled when MuJoCo compatibility is enabled together with GJK and multi-contact, and disabled
        otherwise. Defaults to None.
    broadphase_traversal : gs.broadphase_traversal, optional
        Broadphase traversal strategy. ``SAP`` (sweep-and-prune) or ``ALL_VS_ALL`` (parallel pair iteration). Defaults
        to ``None`` (auto: ``SAP`` on CPU or when hibernation/heterogeneous entities are enabled, ``ALL_VS_ALL`` on GPU
        otherwise). See ``gs.broadphase_traversal`` for details on each strategy.

    Warning
    -------
    Hibernation hasn't been robustly tested and will be fully supported soon.
    """

    dt: PositiveFloat | None = None
    gravity: Vec3FType | None = None
    enable_collision: StrictBool = True
    enable_joint_limit: StrictBool = True
    enable_self_collision: StrictBool = True
    enable_neutral_collision: StrictBool = False
    enable_adjacent_collision: StrictBool = False
    disable_constraint: StrictBool = False
    max_collision_pairs: NonNegativeInt = 150
    max_contacts: PositiveInt | None = None
    multiplier_collision_broad_phase: PositiveInt = 8
    integrator: gs.integrator = gs.integrator.approximate_implicitfast
    IK_max_targets: PositiveInt = 6

    # batching info
    batch_links_info: StrictBool = False
    batch_joints_info: StrictBool = False
    batch_dofs_info: StrictBool = False

    # constraint solver
    constraint_solver: gs.constraint_solver = gs.constraint_solver.Newton
    iterations: PositiveInt = 50
    tolerance: PositiveFloat | None = None
    ls_iterations: PositiveInt = 50
    ls_tolerance: PositiveFloat = 1e-2
    noslip_iterations: NonNegativeInt = 0
    noslip_tolerance: PositiveFloat = 1e-6
    friction_cone: gs.friction_cone = gs.friction_cone.pyramidal
    contact_resolution: gs.contact_resolution | None = None
    enable_torsional_friction: StrictBool = False
    enable_rolling_friction: StrictBool = False
    impratio: PositiveFloat | None = None
    contact_pruning_tolerance: PositiveFloat | None = 0.02
    sparse_solve: StrictBool | None = None
    constraint_timeconst: PositiveFloat = 0.01
    use_contact_island: StrictBool = True
    box_box_detection: StrictBool = False

    # hibernation threshold
    use_hibernation: StrictBool = False
    hibernation_thresh_vel: PositiveFloat | None = None

    # for dynamic properties
    max_dynamic_constraints: NonNegativeInt = 8

    # Experimental options mainly intended for debug purpose and unit tests
    enable_multi_contact: StrictBool = True
    enable_mujoco_compatibility: StrictBool = False

    # GJK collision detection
    use_gjk_collision: StrictBool | None = None
    enable_contact_patch: StrictBool | None = None

    # broadphase configuration
    broadphase_traversal: gs.broadphase_traversal | None = None

    def __init__(self, *, contact_resolve_time: float | None = None, **data):
        super().__init__(**data)
        if contact_resolve_time is not None:
            gs.logger.warning("'contact_resolve_time' is deprecated. Use 'constraint_timeconst' instead.")

    def model_post_init(self, context):
        super().model_post_init(context)
        if self.contact_pruning_tolerance is not None and self.enable_mujoco_compatibility:
            if "contact_pruning_tolerance" in self.model_fields_set:
                gs.raise_exception(
                    "'contact_pruning_tolerance' is not supported when 'enable_mujoco_compatibility' is True"
                )
            # User did not explicitly request pruning, silently disable to guarantee mujoco compatibility
            self.contact_pruning_tolerance = None
        if self.friction_cone == gs.friction_cone.elliptic and self.noslip_iterations > 0:
            gs.raise_exception("The elliptic friction cone is not supported with the noslip solver.")
        if self.enable_rolling_friction and not self.enable_torsional_friction:
            gs.raise_exception("'enable_rolling_friction' requires 'enable_torsional_friction'.")


class MPMOptions(Options):
    """
    Options configuring the MPMSolver.

    Note
    ----
    MPM is a hybrid lagrangian-eulerian method for simulating soft materials. In the eulerian phase, it uses a grid representation. The `upper_bound` and `lower_bound` specify the simulation domain, but a safety padding will be added to the actual grid boundary. Therefore, the actual boundary could be slightly tighter than the specified one. Note that the size of the domain affects the performance of the simulation, hence you should set it as tight as possible.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. If none, it will inherit from `SimOptions`. Defaults to None.
    gravity : tuple, optional
        Gravity force in N/kg. If none, it will inherit from `SimOptions`. Defaults to None.
    particle_size : float, optional
        Particle diameter in meters. If not given, we will compute `particle_size` based on `grid_density`, where `particle_size` will be linearly proportional to the grid cell size. A reference value is `particle_size = 0.01` for `grid_density = 64`. Defaults to None.
    grid_density : float, optional
        Number of grid cells per meter. Defaults to 64.
    enable_CPIC : bool, optional
        Whether to enable CPIC (Compatible Particle-in-Cell) to support coupling with thin objects. Defaults to False.
    lower_bound : tuple, shape (3,), optional
        Lower bound of the simulation domain. Defaults to (-1.0, -1.0, 0.0).
    upper_bound : tuple, shape (3,), optional
        Upper bound of the simulation domain. Defaults to (1.0, 1.0, 1.0).
    use_sparse_grid : bool, optional
        This option is deprecated.
    leaf_block_size : int, optional
        This option is deprecated.
    """

    dt: PositiveFloat | None = None
    gravity: Vec3FType | None = None
    particle_size: PositiveFloat | None = None  # in meters. Will be computed automatically if it's None.
    grid_density: PositiveFloat = 64
    enable_CPIC: StrictBool = False

    # These will later be converted to discrete grid bound. The actual grid boundary could be slightly tighter.
    lower_bound: Vec3FType = (-1.0, -1.0, 0.0)
    upper_bound: Vec3FType = (1.0, 1.0, 1.0)

    def __init__(self, *, use_sparse_grid: bool = False, leaf_block_size: int = 8, **data):
        super().__init__(**data)
        if use_sparse_grid:
            gs.logger.warning("'use_sparse_grid' is deprecated and has no effect.")
        if leaf_block_size != 8:
            gs.logger.warning("'leaf_block_size' is deprecated and has no effect.")

    @model_validator(mode="before")
    @classmethod
    def _resolve_defaults(cls, data: dict) -> dict:
        if data.get("particle_size") is None:
            data["particle_size"] = 0.01 * 64.0 / data.get("grid_density", 64)
        return data

    def model_post_init(self, context: Any) -> None:
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            gs.raise_exception("Invalid pair of upper_bound and lower_bound.")


class SPHOptions(Options):
    """
    Options configuring the SPHSolver.

    Note
    ----
    If spatial hashing parameters are not given, we will compute them automatically this way: For `hash_grid_cell_size`, we will set it to be the `support_radius`, which is essentially 2 * `particle_size`. For `hash_grid_res`, if a small bound is given, it's used for the hash grid; otherwise, we use a default value of a 150^3 cube. Any grid bigger than that will results in too many cells hence not ideal.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. If none, it will inherit from `SimOptions`. Defaults to None.
    gravity : tuple, optional
        Gravity force in N/kg. If none, it will inherit from `SimOptions`. Defaults to None.
    particle_size : float, optional
        Particle diameter in meters. Defaults to 0.02.
    pressure_solver : str, optional
        Pressure solver type. Current supported pressure solvers are 'WCSPH' and 'DFSPH'. Defaults to 'WCSPH'.
    lower_bound : tuple, shape (3,), optional
        Lower bound of the simulation domain. Defaults to (-100.0, -100.0, 0.0).
    upper_bound : tuple, shape (3,), optional
        Upper bound of the simulation domain. Defaults to (100.0, 100.0, 100.0).
    hash_grid_res : tuple, optional
        Size of the spatially-repetitive spatial hashing grid in meters. If none, it will be computed automatically. Defaults to None.
    hash_grid_cell_size : float, optional
        Size of the lattic cell of the spatial hashing grid in meters. This should be at least 2 * `particle_size`. If none, it will be computed automatically. Defaults to None.
    max_divergence_error : float, optional
        Maximum divergence error for DFSPH. Defaults to 0.1.
    max_density_error_percent : float, optional
        Maximum density error *percent* for DFSPH, so 0.1 means 0.1%. Defaults to 0.05.
    max_divergence_solver_iterations : int, optional
        Maximum number of iterations for the divergence solver. Defaults to 100.
    max_density_solver_iterations : int, optional
        Maximum number of iterations for the density solver. Defaults to 100.
    """

    dt: PositiveFloat | None = None
    gravity: Vec3FType | None = None
    particle_size: PositiveFloat = 0.02
    pressure_solver: Literal["WCSPH", "DFSPH"] = "WCSPH"

    lower_bound: Vec3FType = (-100.0, -100.0, 0.0)
    upper_bound: Vec3FType = (100.0, 100.0, 100.0)

    # spatial hashing
    hash_grid_res: Vec3FType | None = None  # size of the spatially-repetitive hash grid in meters
    hash_grid_cell_size: PositiveFloat | None = None  # size of the cubic cell in meters

    # DFSPH parameters
    max_divergence_error: PositiveFloat = 0.1
    max_density_error_percent: PositiveFloat = 0.05  # This is percent
    max_divergence_solver_iterations: PositiveInt = 100
    max_density_solver_iterations: PositiveInt = 100

    _support_radius: float = PrivateAttr(default=0.0)
    _hash_grid_res: np.ndarray = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_defaults(cls, data: dict) -> dict:
        particle_size = data.get("particle_size", 0.02)
        support_radius = 2 * particle_size
        if data.get("hash_grid_cell_size") is None:
            data["hash_grid_cell_size"] = support_radius
        return data

    def model_post_init(self, context: Any) -> None:
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            gs.raise_exception("Invalid pair of upper_bound and lower_bound.")

        self._support_radius = 2 * self.particle_size

        if self.hash_grid_cell_size < self._support_radius:
            gs.raise_exception("`hash_grid_cell_size` should not be smaller than 2 * `particle_size`.")

        if self.hash_grid_res is None:
            max_hash_grid_res = np.ceil(
                (np.array(self.upper_bound) - np.array(self.lower_bound)) / self.hash_grid_cell_size
            ).astype(gs.np_int)
            self._hash_grid_res = np.minimum(max_hash_grid_res, np.array([150, 150, 150], dtype=gs.np_int))
        else:
            self._hash_grid_res = np.ceil(np.array(self.hash_grid_res) / self.hash_grid_cell_size).astype(gs.np_int)


class IPBFOptions(Options):
    """
    Options configuring the IPBFSolver (Implicit Position-Based Fluids, Diaz et al. 2025).

    Note
    ----
    If spatial hashing parameters are not given, we will compute them automatically this way: For `hash_grid_cell_size`, we will set it to be the `support_radius`, which is essentially 2 * `particle_size`. For `hash_grid_res`, if a small bound is given, it's used for the hash grid; otherwise, we use a default value of a 150^3 cube. Any grid bigger than that will results in too many cells hence not ideal.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. If none, it will inherit from `SimOptions`. Defaults to None.
    gravity : tuple, optional
        Gravity force in N/kg. If none, it will inherit from `SimOptions`. Defaults to None.
    particle_size : float, optional
        Particle diameter in meters. Defaults to 0.02.
    lower_bound : tuple, shape (3,), optional
        Lower bound of the simulation domain. Defaults to (-100.0, -100.0, 0.0).
    upper_bound : tuple, shape (3,), optional
        Upper bound of the simulation domain. Defaults to (100.0, 100.0, 100.0).
    hash_grid_res : tuple, optional
        Size of the spatially-repetitive spatial hashing grid in meters. If none, it will be computed automatically. Defaults to None.
    hash_grid_cell_size : float, optional
        Size of the lattic cell of the spatial hashing grid in meters. This should be at least 2 * `particle_size`. If none, it will be computed automatically. Defaults to None.
    ipbf_iterations : int, optional
        Number of per-particle Newton (relaxed Jacobi) iterations per substep. Defaults to 2.
    rebuild_neighbors_per_iter : bool, optional
        Non-paper experiment: re-run the neighbor search at the current guess before every Newton
        iteration, instead of once per substep on the inertial position (paper Algorithm 1 keeps
        the neighborhood fixed across iterations). Defaults to False.
    alpha : float, optional
        Compliance of the pressure energy (alpha = 1 / k). 0.0 means infinite stiffness (paper default). Defaults to 0.0.
    damping_beta : float, optional
        Beta coefficient of the artificial damping (in units of support radius). Defaults to 60.0.
    viscosity_xsph : float, optional
        Explicit XSPH viscosity coefficient epsilon (PLAN P7; NOT part of the IPBF paper).
        0.0 disables it (default); typical values 0.05-0.3. When > 0, a Jacobi velocity
        smoothing v_i += eps * sum_j V (v_j - v_i) W_ij over fluid-fluid neighbor pairs is
        applied after the velocity update of every substep.
    surface_viscosity_xsph : float, optional
        Interface-restricted XSPH viscosity coefficient (Boussinesq-Scriven-style surface
        dissipation): the same Jacobi velocity smoothing as `viscosity_xsph`, but only
        surface-surface neighbor pairs contribute, so tangential velocity differences along
        the free surface dissipate while the bulk keeps its own coefficient. Damps capillary
        waves, droplet oscillation and surface jitter without slowing the bulk pour. Requires
        `surface_tension_enabled=True` (the surface classification comes from the surface
        tension topology pass). 0.0 disables it (default); typical values 0.05-0.3.
    diffusion_coeff : float, optional
        Demo-level XSPH-style concentration diffusion strength (dimensionless, NOT a physical Fick
        diffusion coefficient). 0 disables the diffusion pass entirely. Defaults to 0.0.
    damping_enabled : bool, optional
        Whether to enable the artificial damping (paper section 3.6, eqs. 16-18; enabled in all paper tests).
        Defaults to True.
    damping_alpha_star : float, optional
        Compliance of the alternative (lower-stiffness) solution used by the artificial damping.
        The paper value 1e-3 is tuned for its own units (particle diameter 0.5, kernel radius R = 1):
        the inertia-to-constraint ratio in H scales as alpha * m * R^2 / h^2 with m ~ R^3 and the
        constraint gradient squared ~ 1 / R^2, i.e. as alpha * R^5 / h^2. Rescaling to a meter-scale
        scene with R = 0.02 (2 * particle_size = 0.01) gives 1e-3 * (1 / 0.02)^5 = 3.2e6.
        Defaults to 3.2e6.
    boundary_particles : bool, optional
        Whether to sample static Akinci-style boundary particles on the boundary box walls
        (bottom + 4 side faces) and include them in the density sum (PLAN P4.1). Defaults to True.
    boundary_cylinder : tuple, shape (4,), optional
        (center_x, center_y, radius, z_bottom) of a vertical-axis CylinderBoundary used as the
        position/velocity clamp instead of the box-shaped CubeBoundary (top left open). Intended
        for cylindrical container scenes; use together with `boundary_particles=False`, otherwise
        the box-wall boundary particles end up in the corners outside the cylinder (a warning is
        emitted in that case, not an error). Optional 5th element z_top = rim height (open-top
        cup), optional 6th element escape_band = band limit for the radial clamp and bottom
        plane (multiflow MF-14). Defaults to None (CubeBoundary, bit-identical).
    boundary_pitcher : tuple, shape (8,), optional
        (ox, oy, oz, ax, ay, az, radius, length) of a tilted open-mouth TiltedCylinderBoundary
        (multiflow MF-10 pitcher): origin = inner bottom center, axis = bottom->mouth direction,
        side wall clamped within a thin escape band for 0<=s<=length, mouth open past the rim so
        the container can pour. Applied to all fluid particles as a second clamp after `boundary`.
        Defaults to None (disabled, bit-identical).
    boundary_plane : tuple, optional
        Horizontal table plane z = z0 (multiflow MF-13 pattern, PBD parity): `(z0,)` for an
        infinite plane or `(z0, cx, cy, radius)` for a finite disk. Applied to every fluid
        particle after the container clamps, so liquid escaping a container above the plane
        lands on it. Defaults to None (disabled, bit-identical).
    boundary_layers : int, optional
        Number of boundary particle layers; layer 0 sits on the wall plane, further layers are
        shifted one `particle_size` outward each. Defaults to 2.
    surface_tension_enabled : bool, optional
        Enable the PBSTF-style local-mesh area constraint inside the IPBF Newton assembly
        (ST_IPBF_DERIVATION.md). Defaults to False (solver stays bit-identical to pre-ST).
    st_model : str, optional
        'quadratic' (default, energy 1/2 k~_st (C^A)^2, plan (b)) or 'linear' (sigma * A, plan (a);
        requires a finite density compliance, taken from `st_alpha_ref`).
    st_stiffness : float, optional
        Normalized quadratic area-penalty weight k~_st in m^-4. Defaults to 5.4e4 (calibration
        starting value of the derivation doc; per-scene calibration happens in ST-1).
    st_alpha_ref : float, optional
        Density compliance alpha used in linear mode (~1% compression). Defaults to 1.5.
    st_distance_enabled / st_distance_stiffness : optional
        One-sided distance constraint (default on, normalized weight 1e3 m^-2).
    st_ring_radius_factor : float, optional
        One-ring search radius in units of `particle_size`. Defaults to 3.0.
    st_topo_interval : int, optional
        Topology (surface detect + normals + local meshes) rebuild interval in substeps. Defaults to 1.
    st_lit_threshold / st_normal_compat : optional
        Spherical-illumination threshold (1/9) and normal-compatibility threshold (cos(pi/4)).
    st_max_surface_neighbors / st_max_localmesh_neighbors : int, optional
        Two-level one-ring capacities: 128 candidates / 64 final ring vertices.
    """

    dt: PositiveFloat | None = None
    gravity: Vec3FType | None = None
    particle_size: PositiveFloat = 0.02

    lower_bound: Vec3FType = (-100.0, -100.0, 0.0)
    upper_bound: Vec3FType = (100.0, 100.0, 100.0)

    # spatial hashing
    hash_grid_res: Vec3FType | None = None  # size of the spatially-repetitive hash grid in meters
    hash_grid_cell_size: PositiveFloat | None = None  # size of the cubic cell in meters

    # IPBF parameters
    ipbf_iterations: PositiveInt = 2
    alpha: NonNegativeFloat = 0.0
    # non-paper experiment: rebuild the spatial hash (neighbor search) inside the Newton loop at
    # every iteration, instead of once per substep on the inertial position y (paper Algorithm 1
    # fixes the neighborhood across iterations). Defaults to False (paper-faithful).
    rebuild_neighbors_per_iter: StrictBool = False

    # artificial damping (paper section 3.6, eqs. 16-18)
    # NOTE: disabled by default for now — alpha_star tuning is deferred (user decision 2026-08-17);
    # alpha_star >= 10 saturates (over-damped "sand-like" in 1 m scenes); effective range is [0.1, 10].
    damping_enabled: StrictBool = False
    damping_alpha_star: PositiveFloat = 3.2e6  # paper's 1e-3 rescaled from R=1 to R=0.02 (see docstring)
    damping_beta: NonNegativeFloat = 60.0

    # static boundary particles (PLAN P4.1, Akinci et al. 2012 style; the paper leaves boundaries open)
    boundary_particles: StrictBool = True
    boundary_layers: PositiveInt = 2

    # cylindrical clamp boundary (multiflow MF-7): (center_x, center_y, radius, z_bottom), top open.
    # When set, the solver clamps against a CylinderBoundary instead of the CubeBoundary box.
    # Optional 5th element z_top (multiflow MF-10): rim height — the radial wall clamp then applies
    # only below the rim (open-top cup; a pitcher hovering above the rim is not sucked onto the
    # wall). Without z_top the wall is infinitely tall (MF-7..9 behavior, bit-identical).
    # Optional 6th element escape_band (multiflow MF-14): the radial clamp and bottom plane then
    # only act within [radius, radius+escape_band], so droplets detaching off the outer wall
    # beyond the band escape to the table plane instead of being sucked back onto the wall.
    boundary_cylinder: (
        tuple[float, float, float, float]
        | tuple[float, float, float, float, float]
        | tuple[float, float, float, float, float, float]
        | None
    ) = None

    # tilted open-mouth cylinder clamp (multiflow MF-10/11 pitcher): (ox, oy, oz, ax, ay, az,
    # radius, length) — origin = inner bottom center, axis = bottom->mouth unit vector (need not
    # be normalized), side wall clamped for 0<=s<=length within a thin escape band, mouth open
    # past the rim so the container can pour. The pose is dynamic: `solver.set_pitcher_pose` can
    # animate it per frame (MF-11 progressive tilt). Clamp dispatch is per-particle by
    # `particles_ng.boundary_group` (material `boundary_group`): only group-1 particles see this
    # clamp, and they transfer to group 0 (primary boundary) once they leave the keep region.
    # None disables it (bit-identical).
    boundary_pitcher: tuple[float, float, float, float, float, float, float, float] | None = None

    # horizontal table plane (multiflow MF-13 pattern, PBD parity): (z0,) infinite or
    # (z0, cx, cy, radius) finite disk; applied to every fluid particle after the container
    # clamps, so milk that escapes a container above the plane lands on it. None disables it
    # (bit-identical).
    boundary_plane: tuple[float, ...] | None = None

    # explicit XSPH viscosity (PLAN P7; a user-requested feature experiment, NOT part of the IPBF
    # paper — the paper only allows viscosity inside a* and offers artificial damping instead).
    # Applied as a Jacobi velocity smoothing after the velocity update each substep:
    # v_i += eps * sum_j V (v_j - v_i) W_ij (fluid-fluid pairs only). 0.0 disables it entirely
    # (the kernel is never called, bit-identical behavior).
    viscosity_xsph: NonNegativeFloat = 0.0

    # interface-restricted XSPH viscosity: same Jacobi velocity smoothing as viscosity_xsph but
    # only surface-surface neighbor pairs contribute (the surface flags come from the ST topology
    # pass, so a positive value requires surface_tension_enabled). 0.0 disables it entirely (the
    # kernel is never called, bit-identical behavior).
    surface_viscosity_xsph: NonNegativeFloat = 0.0

    # demo-level XSPH-style concentration diffusion (dimensionless; 0 = off). Not a physical Fick coefficient.
    diffusion_coeff: NonNegativeFloat = 0.0

    # surface tension (ST_IPBF_DERIVATION.md section 4.1): PBSTF-style local-mesh area constraint
    # assembled into the IPBF per-particle Newton step. Defaults keep the solver bit-identical to
    # the pre-ST version (`surface_tension_enabled=False`).
    surface_tension_enabled: StrictBool = False
    st_model: Literal["quadratic", "linear"] = "quadratic"  # (b) 1/2 k_st (C^A)^2 / (a) sigma * A
    st_stiffness: NonNegativeFloat = 5.4e4  # normalized area-penalty weight k~_st [m^-4] (quadratic mode)
    st_alpha_ref: NonNegativeFloat = 1.5  # density compliance alpha used in linear mode (D2, ~1% compression)
    st_distance_enabled: StrictBool = True  # one-sided distance constraint (section 2.8, D8)
    st_distance_stiffness: NonNegativeFloat = 1.0e3  # normalized distance weight k~_d [m^-2]
    st_ring_radius_factor: PositiveFloat = 3.0  # one-ring search radius / particle_size (section 3.2)
    st_topo_interval: PositiveInt = 1  # topology rebuild interval in substeps (D6)
    st_lit_threshold: PositiveFloat = 1.0 / 9.0  # spherical-illumination threshold (Shibata 2015)
    st_normal_compat: NonNegativeFloat = 0.7071067811865476  # normal compatibility cos(pi/4) (film separation)
    st_max_surface_neighbors: PositiveInt = 128  # one-ring candidate capacity (pre-sort, two-level 1st)
    st_max_localmesh_neighbors: PositiveInt = 64  # final one-ring capacity (two-level 2nd)

    _support_radius: float = PrivateAttr(default=0.0)
    _hash_grid_res: np.ndarray = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_defaults(cls, data: dict) -> dict:
        particle_size = data.get("particle_size", 0.02)
        support_radius = 2 * particle_size
        # with surface tension the single hash grid must also cover the one-ring search radius
        # (section 3.2, option (i)); with ST off the cell size stays exactly 2 * particle_size
        st_enabled = data.get("surface_tension_enabled", False)
        if st_enabled:
            support_radius = max(support_radius, data.get("st_ring_radius_factor", 3.0) * particle_size)
        if data.get("hash_grid_cell_size") is None:
            data["hash_grid_cell_size"] = support_radius
        return data

    def model_post_init(self, context: Any) -> None:
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            gs.raise_exception("Invalid pair of upper_bound and lower_bound.")

        self._support_radius = 2 * self.particle_size

        if self.hash_grid_cell_size < self._support_radius:
            gs.raise_exception("`hash_grid_cell_size` should not be smaller than 2 * `particle_size`.")
        if self.surface_tension_enabled:
            ring_radius = self.st_ring_radius_factor * self.particle_size
            if self.hash_grid_cell_size < ring_radius:
                gs.raise_exception(
                    "`hash_grid_cell_size` must cover the surface one-ring radius "
                    "(`st_ring_radius_factor` * `particle_size`)."
                )
            if self.st_max_surface_neighbors < 3 or self.st_max_localmesh_neighbors < 3:
                gs.raise_exception("ST neighbor capacities must be at least 3.")
            if self.st_max_localmesh_neighbors > min(self.st_max_surface_neighbors, 128):
                gs.raise_exception(
                    "`st_max_localmesh_neighbors` must be at most min(`st_max_surface_neighbors`, 128) "
                    "(the reverse one-ring index packs the ring slot in 7 bits)."
                )

        if self.hash_grid_res is None:
            max_hash_grid_res = np.ceil(
                (np.array(self.upper_bound) - np.array(self.lower_bound)) / self.hash_grid_cell_size
            ).astype(gs.np_int)
            self._hash_grid_res = np.minimum(max_hash_grid_res, np.array([150, 150, 150], dtype=gs.np_int))
        else:
            self._hash_grid_res = np.ceil(np.array(self.hash_grid_res) / self.hash_grid_cell_size).astype(gs.np_int)


class PBSTFStaticColliderOptions(Options):
    """Base pose options for one-way PBSTF colliders.

    The pose can change after scene construction through
    :meth:`PBSTFSolver.set_static_colliders_pose`. The collider remains one-way: it affects the liquid and receives no
    force or velocity response from it.
    """

    pos: Vec3FType = (0.0, 0.0, 0.0)
    quat: UnitVec4FType = (1.0, 0.0, 0.0, 0.0)


class PBSTFBoxStaticColliderOptions(PBSTFStaticColliderOptions):
    """Finite analytic box collider.

    ``lower`` and ``upper`` are opposite corners in the collider's local frame. Analytic queries keep rectangular
    geometry exact and inexpensive; a mesh collider supports arbitrary geometry at preprocessing and field-memory cost.
    """

    type: Literal["box"] = "box"
    lower: Vec3FType
    upper: Vec3FType

    @model_validator(mode="after")
    def _validate_geometry(self):
        if not np.all(np.array(self.upper) > np.array(self.lower)):
            gs.raise_exception("PBSTF box collider `upper` must be greater than `lower` along every axis.")
        return self


class PBSTFAbsorbentStaticColliderOptionsMixin(Options):
    """Absorption controls for a position-based surface tension flow (PBSTF) static collider.

    ``absorption_rate`` limits sustained new captures per simulated second for each collider and environment, and sets
    the nearest-voxel exponential inward-motion rate. Voxels farther from the contact move liquid inward progressively
    more slowly. A higher value admits liquid faster but produces more abrupt local trajectories; a lower value reduces
    throughput and preserves gradual motion. ``absorption_capacity_fraction`` is the fraction of collider volume
    available for liquid at rest. A higher value stores more liquid before saturation, while a lower value resumes
    ordinary collision behavior sooner.
    """

    absorption_rate: PositiveFloat
    absorption_capacity_fraction: UnitInterval

    @model_validator(mode="after")
    def _validate_absorption_capacity(self):
        if self.absorption_capacity_fraction <= 0.0:
            gs.raise_exception("PBSTF absorbent collider `absorption_capacity_fraction` must be positive.")
        return self


class PBSTFAbsorbentBoxStaticColliderOptions(PBSTFAbsorbentStaticColliderOptionsMixin, PBSTFBoxStaticColliderOptions):
    """Finite position-based surface tension flow (PBSTF) box with rate-limited nearby-voxel capture.

    ``pbd_entity_name`` binds geometry to a volumetric position-based dynamics (PBD) entity using PBDUnifiedOptions.
    Binding lets the collider and its material-space absorption targets follow deformation, at the cost of updating a
    triangle surface and voxel search order whenever the PBD shape changes. ``sdf_res`` enables a signed distance field
    (SDF) that can be built after deformation stops. Higher resolutions preserve smaller surface features at cubic
    preprocessing and memory cost, while ``None`` keeps exact triangle queries. ``pbd_entity_name=None`` uses
    inexpensive analytic box geometry.
    """

    type: Literal["absorbent_box"] = "absorbent_box"
    pbd_entity_name: str | None = None
    sdf_res: StrictInt | None = Field(default=None, ge=16)

    @model_validator(mode="after")
    def _validate_sdf(self):
        if self.sdf_res is not None and self.pbd_entity_name is None:
            gs.raise_exception("PBSTF absorbent box collider `sdf_res` requires `pbd_entity_name`.")
        return self


class PBSTFConeStaticColliderOptions(PBSTFStaticColliderOptions):
    """Finite analytic cone collider.

    ``center`` is the local-frame center of the base disk, ``height`` points from the base center to the apex, and
    ``radius`` is the base radius.
    """

    type: Literal["cone"] = "cone"
    center: Vec3FType
    height: Vec3FType
    radius: PositiveFloat

    @model_validator(mode="after")
    def _validate_geometry(self):
        if np.linalg.norm(self.height) <= gs.EPS:
            gs.raise_exception("PBSTF cone collider `height` must be non-zero.")
        return self


class PBSTFMeshStaticColliderOptions(PBSTFStaticColliderOptions):
    """Signed-distance-field collider built from a watertight triangle mesh.

    Higher ``sdf_res`` resolves thinner walls and sharper features at the cost of cubic build memory and longer
    preprocessing. The cached field is expressed in the collider's local frame, so pose updates do not rebuild it.
    """

    type: Literal["mesh"] = "mesh"
    file: str
    scale: PositiveFloat = 1.0
    sdf_res: StrictInt = Field(default=150, ge=16)


PBSTFStaticColliderOptionsType = Annotated[
    PBSTFAbsorbentBoxStaticColliderOptions
    | PBSTFBoxStaticColliderOptions
    | PBSTFConeStaticColliderOptions
    | PBSTFMeshStaticColliderOptions,
    Field(discriminator="type"),
]


class PBSTFOptions(Options):
    """
    Options configuring the graphics processing unit (GPU) position-based surface tension flow (PBSTF) solver.

    ``particle_size`` is the particle diameter. The cubic-spline support radius is ``3 * particle_size`` (six particle
    radii). The hash-grid cell must cover that support radius so a 3x3x3 cell stencil contains every kernel neighbor.

    ``max_surface_neighbors`` bounds the projected candidates used to construct each local mesh. A higher value
    preserves topology in crowded regions at the cost of scene memory and topology rebuild time.
    ``max_localmesh_neighbors`` bounds the final one-ring vertices used by surface constraints. A higher value admits
    more irregular one-rings at the cost of constraint memory and time. Its default is the smaller of 64 and
    ``max_surface_neighbors``.

    ``static_colliders`` are ordered by support priority when their particle-radius-expanded regions overlap. Earlier
    entries preserve their exclusion region, while later entries yield at incompatible contacts. Put load-bearing
    surfaces before moving tools so particles remain on the support; reverse the order when the tool must take priority.

    ``diffusion_coeff`` is a demo-level XSPH-style concentration diffusion strength (dimensionless relaxation
    fraction per substep, NOT a physical Fick diffusion coefficient). 0 disables the diffusion pass entirely and
    the solver stays bit-identical to the pre-diffusion version. Defaults to 0.0.
    """

    dt: PositiveFloat | None = None
    gravity: Vec3FType | None = None

    particle_size: PositiveFloat = 0.02
    kernel_scale: PositiveFloat = 6.0
    max_solver_iterations: PositiveInt = 100
    topology_rebuild_interval: PositiveInt = 1
    max_surface_neighbors: PositiveInt = 128
    max_localmesh_neighbors: PositiveInt | None = None
    enable_pca_normals: bool = True
    diffusion_coeff: NonNegativeFloat = 0.0
    static_colliders: list[PBSTFStaticColliderOptionsType] = Field(default_factory=list)

    hash_grid_res: Vec3FType | None = None
    hash_grid_cell_size: PositiveFloat | None = None

    lower_bound: Vec3FType = (-100.0, -100.0, 0.0)
    upper_bound: Vec3FType = (100.0, 100.0, 100.0)

    _support_radius: float = PrivateAttr(default=0.0)
    _hash_grid_res: np.ndarray = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_defaults(cls, data: dict) -> dict:
        particle_size = data.get("particle_size", 0.02)
        kernel_scale = data.get("kernel_scale", 6.0)
        if not np.isclose(kernel_scale, 6.0):
            gs.raise_exception("PBSTF fixes `kernel_scale` to 6.0, matching the C++ reference implementation.")
        support_radius = 3.0 * particle_size
        if data.get("hash_grid_cell_size") is None:
            data["hash_grid_cell_size"] = support_radius
        if data.get("max_localmesh_neighbors") is None:
            data["max_localmesh_neighbors"] = min(data.get("max_surface_neighbors", 128), 64)
        return data

    def model_post_init(self, context: Any) -> None:
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            gs.raise_exception("Invalid pair of upper_bound and lower_bound.")
        if self.max_surface_neighbors < 3:
            gs.raise_exception("`max_surface_neighbors` must be at least 3.")
        if self.max_localmesh_neighbors < 3:
            gs.raise_exception("`max_localmesh_neighbors` must be at least 3.")
        if self.max_localmesh_neighbors > self.max_surface_neighbors:
            gs.raise_exception("`max_localmesh_neighbors` must be at most `max_surface_neighbors`.")

        self._support_radius = 3.0 * self.particle_size
        if self.hash_grid_cell_size < self._support_radius:
            gs.raise_exception("`hash_grid_cell_size` must not be smaller than the PBSTF cubic-spline support radius.")

        if self.hash_grid_res is None:
            max_hash_grid_res = np.ceil(
                (np.array(self.upper_bound) - np.array(self.lower_bound)) / self.hash_grid_cell_size
            ).astype(gs.np_int)
            self._hash_grid_res = np.minimum(max_hash_grid_res, np.array([150, 150, 150], dtype=gs.np_int))
        else:
            self._hash_grid_res = np.ceil(np.array(self.hash_grid_res) / self.hash_grid_cell_size).astype(gs.np_int)


class PBDUnifiedOptions(Options):
    """Options for unified position-based dynamics (PBD) of cloth and elastic solids.

    Each iteration combines elastic corrections and enforces one-way rigid contact. More iterations improve
    shape preservation under load at increased runtime cost. Collision iterations bound the work needed to
    resolve simultaneous contacts; exhausting this budget with intersections remaining raises an error.
    Constraint acceleration extrapolates consecutive elastic iterates within each time step. Larger values can
    improve shape recovery with fewer iterations, with a greater risk of overshoot under changing contacts.
    Zero uses the current correction alone.
    Particle size controls mesh sampling. Explicit tetrahedral meshes retain their supplied vertices.
    Recording constraint history retains three position samples per iteration for convergence diagnostics,
    using memory proportional to the particle count, environment count, and iteration count.
    """

    dt: PositiveFloat | None = None
    gravity: Vec3FType | None = None
    particle_size: PositiveFloat = 1e-2
    lower_bound: Vec3FType = (-100.0, -100.0, 0.0)
    upper_bound: Vec3FType = (100.0, 100.0, 100.0)
    max_solver_iterations: PositiveInt = 30
    max_collision_iterations: PositiveInt = 20
    constraint_acceleration: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.0
    is_recording_constraint_history: StrictBool = False

    def model_post_init(self, context: Any) -> None:
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            gs.raise_exception("Invalid pair of upper_bound and lower_bound.")


class PBDOptions(Options):
    """
    Options configuring the PBDSolver.

    Note
    ----
    If spatial hashing parameters are not given, we will compute them automatically this way: For `hash_grid_cell_size`, we will set it to be 1.25 * `particle_size`. For `hash_grid_res`, if a small bound is given, it's used for the hash grid; otherwise, we use a default value of a 150^3 cube. Any grid bigger than that will results in too many cells hence not ideal.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. If none, it will inherit from `SimOptions`. Defaults to None.
    gravity : tuple, optional
        Gravity force in N/kg. If none, it will inherit from `SimOptions`. Defaults to None.
    max_stretch_solver_iterations : int, optional
        Maximum number of iterations for the solving stretch constraints. Defaults to 4.
    max_bending_solver_iterations : int, optional
        Maximum number of iterations for the solving bending constraints. Defaults to 1.
    max_volume_solver_iterations : int, optional
        Maximum number of iterations for the solving volume constraints. Defaults to 1.
    max_density_solver_iterations : int, optional
        Maximum number of iterations for the solving density constraints. Defaults to 1.
    max_viscosity_solver_iterations : int, optional
        Maximum number of iterations for the solving viscosity constraints. Defaults to 1.
    particle_size : float, optional
        Particle diameter in meters. Defaults to 1e-2.
    hash_grid_res : tuple, optional
        Size of the spatially-repetitive spatial hashing grid in meters. If none, it will be computed automatically. Defaults to None.
    hash_grid_cell_size : float, optional
        Size of the lattic cell of the spatial hashing grid in meters. This should be at least 1.25 * `particle_size`. If none, it will be computed automatically. Defaults to None.
    lower_bound : tuple, shape (3,), optional
        Lower bound of the simulation domain. Defaults to (-100.0, -100.0, 0.0).
    upper_bound : tuple, shape (3,), optional
        Upper bound of the simulation domain. Defaults to (100.0, 100.0, 100.0).
    diffusion_coeff : float, optional
        Demo-level XSPH-style concentration diffusion strength (dimensionless, NOT a physical Fick diffusion
        coefficient). 0 disables the diffusion pass entirely. Defaults to 0.0.
    surface_tension_enabled : bool, optional
        PBSTF-style surface tension (local-mesh area constraint + one-sided surface distance constraint)
        projected inside the density constraint iteration loop (multiflow MF-12). Defaults to False
        (solver stays bit-identical to the pre-ST version).
    st_compliance : float, optional
        Area-constraint compliance (non-time-scaled convention, same role as PBSTF's
        `surface_tension_compliance`; smaller = stronger surface tension). Defaults to 2.0.
    st_distance_enabled : bool, optional
        One-sided distance constraint between surface particle pairs (prevents surface clumping),
        applied every second density iteration. Only meaningful when `surface_tension_enabled` is True.
        Defaults to True.
    st_distance_compliance : float, optional
        Distance-constraint compliance (same role as PBSTF's `surface_distance_compliance`).
        Defaults to 40.0.
    st_ring_radius_factor : float, optional
        One-ring search radius / particle_size for the surface topology (PBSTF uses 3.0). Note: the PBD
        hash cell stays 1.25 * particle_size — the topology kernels scan +-3 cells to cover the ring.
        Defaults to 3.0.
    st_topo_interval : int, optional
        Topology rebuild interval in substeps (1 = every substep). Defaults to 1.
    st_lit_threshold : float, optional
        Spherical-illumination threshold for surface detection (Shibata 2015). Defaults to 1/9.
    st_normal_compat : float, optional
        Normal-compatibility cosine threshold separating thin-film sides. Defaults to cos(pi/4).
    st_surface_density_factor : float, optional
        Surface-particle density target factor (PBSTF uses 0.7 — the under-density lets the area
        constraint bead). Default 1.0 = no modification. NOTE: 0.7 is calibrated for a fluid whose
        bulk equilibrium sits at rho_rest; for the stock PBF (whose emergent bulk density is ~2x
        the nominal rho_rest at stiff `density_lambda_epsilon`) 0.7 makes the surface constraint
        strongly POSITIVE and ejects surface particles — keep 1.0 there (multiflow MF-12).
    st_max_surface_neighbors : int, optional
        One-ring candidate capacity (pre-sort). Overflow raises. Defaults to 128.
    st_max_localmesh_neighbors : int, optional
        Final one-ring capacity. Overflow raises. Defaults to 64.
    velocity_damping : float, optional
        Global per-substep velocity decay factor (multiflow MF-12 demo damping; simplest possible
        settling aid). 1.0 disables it entirely (bit-identical). Defaults to 1.0.
    density_lambda_epsilon : float, optional
        Regularization epsilon in the PBF density-lambda denominator (lambda = -C / (grad_sum + eps)).
        The stock value 100.0 is kept as default (bit-identical); note it dominates the gradient terms
        (~1e-5..1e-3) at ps~0.008, making the projection extremely soft (tall columns collapse).
        Smaller values stiffen the fluid (multiflow MF-12). Defaults to 100.0.
    density_clamp_negative : bool, optional
        Clamp the density constraint at zero (tension-free liquid). A negative constraint at
        the free surface produces a positive lambda that pulls surface particles toward their
        neighbors and toward boundary balls — a numerical tensile cohesion that reads as
        stickiness and wall-cling; S_Corr only partially offsets it. With the clamp, the
        explicit surface-tension constraints remain the sole cohesion source. False keeps the
        classic two-sided constraint bit-identical (multiflow MF-18). Defaults to False.
    wall_adhesion_enabled : bool, optional
        Fluid-wall surface tension / wall adhesion (multiflow MF-12 rev7), ported from PBSTF's
        `_kernel_apply_static_collider_adhesion` (which follows the adhesion constraint of the
        Bender et al. 2017 PBD survey). Signed two-sided distance constraint C = (x - q).n to the
        analytic container wall (big cup / pitcher selected per-particle via `boundary_group`),
        projected every density Jacobi iteration into the shared dpos buffer. Gated to SURFACE
        particles within one particle radius of the wall — interior particles never stick (the
        key guard against near-wall clumping). Requires `surface_tension_enabled` (the on_surface
        gate). Defaults to False (bit-identical).
    wall_adhesion_compliance : float, optional
        Adhesion compliance in the NON-time-scaled convention (not divided by dt^2; same role as
        PBSTF's `collider_adhesion_compliance`). Smaller = stickier; PBSTF's teapot demo uses 20.0
        (safe starting point — too small over-stretches droplets on the wall because adhesion
        competes with the area constraint inside the same Jacobi iteration). Retune when changing
        dt / particle scale. Defaults to 20.0.
    wall_friction : float, optional
        Velocity-stage tangential damping factor for liquid surface particles within one particle
        radius of a wall (PBSTF's `collider_friction`, applied per substep after the velocity
        update: v -= f * v_tangential). Independent of wall_adhesion. 0.0 disables it entirely.
        Never put this into the position constraint. Defaults to 0.0.
    boundary_cylinder : tuple, optional
        Cylindrical clamp boundary (multiflow): (center_x, center_y, radius, z_bottom), top open.
        Optional 5th element z_top = rim height — the radial wall clamp then applies only below the
        rim (open-top cup). Optional 6th element escape_band — radial clamp and bottom plane then
        act only within [radius, radius+escape_band] (multiflow MF-14). When set, the solver
        clamps against a CylinderBoundary instead of the
        CubeBoundary box. Defaults to None.
    boundary_pitcher : tuple, optional
        Tilted open-mouth cylinder clamp (multiflow pitcher): (ox, oy, oz, ax, ay, az, radius, length)
        — origin = inner bottom center, axis = bottom->mouth unit vector. The pose is dynamic:
        `solver.set_pitcher_pose` animates it per frame. Clamp dispatch is per-particle by
        `particles_ng.boundary_group` (material `boundary_group`): only group-1 particles see this
        clamp, and they transfer to group 0 (primary boundary) once they leave the keep region.
        Defaults to None (bit-identical).
    boundary_plane : tuple, optional
        Horizontal table plane (multiflow MF-13): (z0,) = infinite plane z=z0, or
        (z0, center_x, center_y, radius) = finite disk. Clamped for ALL particles (in addition to
        the group-dispatched cup/pitcher clamp), and takes part in the wall-adhesion nearest-surface
        query. Defaults to None (bit-identical).
    boundary_pitcher_shell : tuple, optional
        Shell-model pitcher boundary (multiflow P2 teapot effect): (ox, oy, oz, ax, ay, az,
        radius, length, t_wall, t_bottom[, lip_round]) — same pose convention as
        `boundary_pitcher`, but the pitcher wall is a finite-thickness solid of revolution with
        an OUTER surface (exact 2D cross-section SDF). impose acts only inside the shell solid
        (pushing to the nearest surface); the exterior is free space, so milk can slide down
        the outer wall under wall adhesion. Mutually exclusive with `boundary_pitcher`; the new
        class coexists with the old one (bit-identical when unset). Defaults to None.
    boundary_cup_shell : tuple, optional
        Static upright cup shell boundary (multiflow: the "pitcher pours into the cup" scene needs
        a cup AND a pitcher shell at the same time): (center_x, center_y, z_bottom, radius, length,
        t_wall, t_bottom[, lip_round]) - center = cup axis horizontal position, z_bottom = inner
        cavity floor (the outer bottom sits at z_bottom - t_bottom), axis is always +z. Same
        finite-thickness shell semantics as `boundary_pitcher_shell`, applied as a world solid to
        EVERY particle with no `boundary_group` ownership. May coexist with `boundary_cylinder`
        (overlapping clamps are then both applied). Defaults to None (bit-identical).
    boundary_ball_sets : tuple, optional
        Frozen-fluid boundary ball walls (Akinci-style pressure walls sampled offline): each item
        is `(npz_path, kinematic)`, where the npz holds `pos_local (N, 3) float32` in the set's
        local frame. The set's particles are appended after every entity's particles, in list
        order; they carry the fluid material's mass, zero velocity and no constraint of their own,
        so they only add density. `solver.set_boundary_ball_pose` moves a `kinematic` set per frame.
        Wall adhesion treats the nearest ball as a solid surface, competing with the analytic
        boundaries for the nearest-surface arbitration. Fluid-ball pairs resolve as a contact at
        the ball spacing, so a ball container carries fluid; the contact is resolved once per
        substep against the current separation, so a ball lattice coarser than the particle size
        lets fluid tunnel through at speed. Keep the lattice inset into the solid it replaces: a
        fluid particle spawned on a ball center starts deep inside the contact distance and is
        ejected. Cost: one particle per ball in every hash lookup. Defaults to () (bit-identical).
    boundary_ball_initial_poses : tuple, optional
        Optional world poses corresponding one-for-one with `boundary_ball_sets`. Each item is
        `((px, py, pz), (qw, qx, qy, qz))`. A non-empty tuple must have exactly one pose per set;
        quaternions are normalized by the solver. These poses are installed before the build-time
        kernel compilation step, so separated containers never transiently overlap at their local
        origins. Defaults to (), meaning identity pose for every set.
    boundary_ball_radius : float, optional
        Collision radius of one boundary ball in the adhesion distance metric. Defaults to None,
        resolving to half the particle size (ball particles packed at the fluid particle spacing).
    boundary_ball_mass_weight_enabled : bool, optional
        Opt in to per-ball density quadrature weights stored as ``mass_weight (N,)`` in every
        ``boundary_ball_sets`` NPZ. When enabled, a boundary ball's density mass is the liquid
        material mass multiplied by its strictly-positive asset weight. Collision, broad phase,
        CCD and fluid particle mass are unchanged. Defaults to False: asset weights are ignored
        and every boundary ball keeps the historical unit mass.
    boundary_ball_smooth_barrel : tuple, optional
        Opt-in analytic narrow phase for one static, upright boundary-ball barrel. The tuple is
        ``(set_index, center_x, center_y, inner_visible_radius, outer_visible_radius,
        visible_z_min, visible_z_max, excluded_azimuth_rad, excluded_half_angle_rad,
        excluded_z_min, excluded_z_max, witness_margin)``. Boundary balls remain the density
        samples, broad phase and CCD trigger. A nearby selected-set ball activates an exact
        particle-radius offset of the visible annular-shell cross-section, including rounded
        top/bottom corners; this covers the sampled sphere union's inter-ball valleys without
        inventing a collar above the visible rim. The angular exclusion applies only while the
        swept segment intersects ``[excluded_z_min, excluded_z_max]``, so a handle attachment
        seam need not disable the smooth barrel at every height. The selected set must be static
        and have the identity initial pose. Defaults to None, keeping the boundary-ball contact
        path unchanged.
    boundary_ball_jug_history : tuple, optional
        History-only query for one KINEMATIC boundary-ball set (e.g. the MF-16 jug): per-accepted-
        substep exit classification of every entity particle in the set's substep-interpolated
        local frame. The tuple is ``(set_index, r_inner, r_outer, z_min, z_max,
        spout_azimuth_rad, spout_half_angle_rad, spout_dz, spout_dr, hist_tol)``; the spout-aware
        top contact surface is ``z_top(az) = z_max + spout_dz*w(az) + particle_radius`` and the
        outer contact radius ``r_out_contact(az) = r_outer + spout_dr*w(az) + particle_radius``
        with ``w(az)`` the cos^2 spout window of half angle ``spout_half_angle_rad`` centred at
        ``spout_azimuth_rad`` (w = 0 outside the window, so z_top is exactly z_max +
        particle_radius at non-spout azimuths). Three one-way OR-accumulating entity-indexed
        flags are maintained: legal interior residency (``r <= r_inner - particle_radius +
        hist_tol`` and ``z_min <= z <= z_max``), rim staging (accepted position at/above
        ``z_top(az)`` within the rim radial band), and irreversible direct tunnel (the
        accepted ``ipos -> pos`` segment crossing outward ``r_out_contact(az)`` below
        ``z_top(az_cross) - hist_tol`` after interior residency, unless staged). The frame-start
        pose for the substep interpolation is the set's pose at the most recent
        ``set_boundary_ball_pose`` call before the step. The selected set must be kinematic.
        This option performs NO projection and NO particle clamping -- it only records history.
        Defaults to None, keeping the solver bit-identical when disabled.
    """

    dt: PositiveFloat | None = None
    gravity: Vec3FType | None = None

    # constraints solving iterations
    max_stretch_solver_iterations: PositiveInt = 4
    max_bending_solver_iterations: PositiveInt = 1
    max_volume_solver_iterations: PositiveInt = 1
    max_density_solver_iterations: PositiveInt = 1
    max_viscosity_solver_iterations: PositiveInt = 1

    # self collision
    particle_size: PositiveFloat = 1e-2

    # spatial hashing
    hash_grid_res: Vec3FType | None = None  # size of the spatially-repetitive hash grid in meters
    hash_grid_cell_size: PositiveFloat | None = None  # size of the cubic cell in meters

    lower_bound: Vec3FType = (-100.0, -100.0, 0.0)
    upper_bound: Vec3FType = (100.0, 100.0, 100.0)

    # demo-level XSPH-style concentration diffusion (dimensionless; 0 = off). Not a physical Fick coefficient.
    diffusion_coeff: NonNegativeFloat = 0.0

    # surface tension (multiflow MF-12): PBSTF-style area + distance constraints projected inside the
    # density iteration loop. Defaults keep the solver bit-identical to the pre-ST version.
    surface_tension_enabled: StrictBool = False
    st_compliance: NonNegativeFloat = 2.0
    st_distance_enabled: StrictBool = True
    st_distance_compliance: NonNegativeFloat = 40.0
    st_ring_radius_factor: PositiveFloat = 3.0
    st_topo_interval: PositiveInt = 1
    st_lit_threshold: PositiveFloat = 1.0 / 9.0
    st_normal_compat: NonNegativeFloat = 0.7071067811865476
    st_surface_density_factor: PositiveFloat = 1.0
    st_max_surface_neighbors: PositiveInt = 128
    st_max_localmesh_neighbors: PositiveInt = 64

    # global per-substep velocity decay (multiflow MF-12 demo damping); 1.0 = off (bit-identical)
    velocity_damping: PositiveFloat = 1.0

    # density-lambda regularization (multiflow MF-12); stock value 100.0 = bit-identical
    density_lambda_epsilon: PositiveFloat = 100.0
    density_clamp_negative: StrictBool = False

    # wall adhesion (multiflow MF-12 rev7): PBSTF-style static-collider adhesion constraint
    # (Bender et al. 2017 PBD survey form, solved alongside the area/density constraints inside
    # every density Jacobi iteration). Requires surface_tension_enabled (the on_surface gate is
    # what prevents interior particles from sticking). Defaults keep the solver bit-identical.
    wall_adhesion_enabled: StrictBool = False
    wall_adhesion_compliance: NonNegativeFloat = 20.0
    wall_friction: NonNegativeFloat = 0.0

    # table-plane friction (multiflow MF-17): tangential velocity attenuation for liquid particles
    # resting on the boundary_plane; 0.0 keeps the solver bit-identical. Independent of
    # wall_adhesion/wall_friction; never touches boundary balls.
    plane_friction: NonNegativeFloat = 0.0

    # cylindrical clamp boundary (multiflow); None keeps the CubeBoundary box. Optional 5th element
    # z_top = rim height (open-top cup; radial clamp applies only below the rim). Optional 6th
    # element escape_band (MF-14): radial clamp + bottom plane act only within
    # [radius, radius+escape_band], letting detached droplets escape to the table plane.
    boundary_cylinder: (
        tuple[float, float, float, float]
        | tuple[float, float, float, float, float]
        | tuple[float, float, float, float, float, float]
        | None
    ) = None

    # tilted open-mouth pitcher clamp (multiflow); None disables it (bit-identical)
    boundary_pitcher: tuple[float, float, float, float, float, float, float, float] | None = None

    # horizontal table plane (multiflow MF-13): (z0,) infinite or (z0, cx, cy, radius) finite disk;
    # None disables it (bit-identical)
    boundary_plane: tuple[float, ...] | None = None

    # shell-model pitcher boundary (multiflow P2): (ox, oy, oz, ax, ay, az, radius, length,
    # t_wall, t_bottom[, lip_round]); mutually exclusive with boundary_pitcher; None disables
    # it (bit-identical)
    boundary_pitcher_shell: tuple[float, ...] | None = None

    # static upright cup shell boundary (multiflow): (cx, cy, z_bot, r_in, L, t_wall, t_bottom[,
    # lip_round]); coexists with boundary_pitcher_shell and boundary_cylinder; None disables it
    # (bit-identical)
    boundary_cup_shell: tuple[float, ...] | None = None

    boundary_ball_sets: tuple[tuple[str, bool], ...] = ()

    boundary_ball_initial_poses: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float, float]], ...
    ] = ()

    boundary_ball_radius: PositiveFloat | None = None

    boundary_ball_mass_weight_enabled: StrictBool = False

    boundary_ball_smooth_barrel: tuple[
        int, float, float, float, float, float, float, float, float, float, float, float
    ] | None = None

    # history-only query for one kinematic boundary-ball set (multiflow MF-16 jug); None keeps
    # the solver bit-identical (no projection, no particle clamping -- history recording only)
    boundary_ball_jug_history: (
        tuple[int, float, float, float, float, float, float, float, float, float] | None
    ) = None

    _hash_grid_res: np.ndarray = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_defaults(cls, data: dict) -> dict:
        particle_size = data.get("particle_size", 1e-2)
        # NOTE: 1.25 is a safety factor, as inside one single substep, multiple substages can change the position of
        # the particles but we only do spatial hashing once. The grid cell needs to be a bit bigger so that neighbours
        # are not missed.
        if data.get("hash_grid_cell_size") is None:
            data["hash_grid_cell_size"] = 1.25 * particle_size
        return data

    def model_post_init(self, context: Any) -> None:
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            gs.raise_exception("Invalid pair of upper_bound and lower_bound.")

        if self.hash_grid_cell_size < 1.25 * self.particle_size:
            gs.raise_exception("`hash_grid_cell_size` should not be smaller than 1.25 * `particle_size`.")

        if self.surface_tension_enabled:
            if self.st_max_surface_neighbors < 3 or self.st_max_localmesh_neighbors < 3:
                gs.raise_exception("ST neighbor capacities must be at least 3.")
            if self.st_max_localmesh_neighbors > self.st_max_surface_neighbors:
                gs.raise_exception("`st_max_localmesh_neighbors` must not exceed `st_max_surface_neighbors`.")

        if self.wall_adhesion_enabled and not self.surface_tension_enabled:
            gs.raise_exception("`wall_adhesion_enabled` requires `surface_tension_enabled` (on_surface gate).")

        if self.boundary_plane is not None and len(self.boundary_plane) not in (1, 4):
            gs.raise_exception("`boundary_plane` must be (z0,) or (z0, center_x, center_y, radius).")

        if self.plane_friction > 0 and self.boundary_plane is None:
            gs.raise_exception("`plane_friction` requires `boundary_plane` (nothing to friction without a plane).")

        if self.boundary_pitcher_shell is not None:
            if self.boundary_pitcher is not None:
                gs.raise_exception("`boundary_pitcher_shell` and `boundary_pitcher` are mutually exclusive.")
            if len(self.boundary_pitcher_shell) not in (10, 11):
                gs.raise_exception(
                    "`boundary_pitcher_shell` must be (ox, oy, oz, ax, ay, az, radius, length, "
                    "t_wall, t_bottom[, lip_round])."
                )

        if self.boundary_cup_shell is not None:
            if len(self.boundary_cup_shell) not in (7, 8):
                gs.raise_exception(
                    "`boundary_cup_shell` must be (cx, cy, z_bot, r_in, L, t_wall, t_bottom[, lip_round])."
                )
            if min(self.boundary_cup_shell[3:7]) <= 0.0:
                gs.raise_exception("`boundary_cup_shell` requires r_in, L, t_wall and t_bottom > 0.")
            if len(self.boundary_cup_shell) == 8 and self.boundary_cup_shell[7] < 0.0:
                gs.raise_exception("`boundary_cup_shell` requires lip_round >= 0.")

        for ball_set in self.boundary_ball_sets:
            if len(ball_set) != 2 or not isinstance(ball_set[0], str) or not isinstance(ball_set[1], bool):
                gs.raise_exception("`boundary_ball_sets` items must be (npz_path: str, kinematic: bool).")

        if self.boundary_ball_initial_poses:
            if len(self.boundary_ball_initial_poses) != len(self.boundary_ball_sets):
                gs.raise_exception(
                    "non-empty `boundary_ball_initial_poses` must have exactly one pose per "
                    "`boundary_ball_sets` item."
                )
            for pose in self.boundary_ball_initial_poses:
                if len(pose) != 2 or len(pose[0]) != 3 or len(pose[1]) != 4:
                    gs.raise_exception(
                        "`boundary_ball_initial_poses` items must be (pos(3), quat(4, wxyz))."
                    )
                pos = np.asarray(pose[0], dtype=np.float64)
                quat = np.asarray(pose[1], dtype=np.float64)
                if not np.isfinite(pos).all() or not np.isfinite(quat).all():
                    gs.raise_exception("`boundary_ball_initial_poses` requires finite pose values.")
                quat_norm = float(np.linalg.norm(quat))
                if not np.isfinite(quat_norm) or quat_norm <= np.finfo(gs.np_float).eps:
                    gs.raise_exception(
                        "`boundary_ball_initial_poses` requires non-zero finite quaternions."
                    )

        if self.boundary_ball_smooth_barrel is not None:
            barrel = self.boundary_ball_smooth_barrel
            if len(barrel) != 12:
                gs.raise_exception(
                    "`boundary_ball_smooth_barrel` must be (set_index, center_x, center_y, "
                    "inner_visible_radius, outer_visible_radius, visible_z_min, visible_z_max, "
                    "excluded_azimuth_rad, excluded_half_angle_rad, excluded_z_min, "
                    "excluded_z_max, witness_margin)."
                )
            i_set = barrel[0]
            values = np.asarray(barrel[1:], dtype=np.float64)
            if not np.isfinite(values).all():
                gs.raise_exception("`boundary_ball_smooth_barrel` requires finite values.")
            if not 0 <= i_set < len(self.boundary_ball_sets):
                gs.raise_exception("`boundary_ball_smooth_barrel` set_index is out of range.")
            if self.boundary_ball_sets[i_set][1]:
                gs.raise_exception("`boundary_ball_smooth_barrel` requires a static boundary-ball set.")
            inner_radius, outer_radius = barrel[3], barrel[4]
            z_min, z_max = barrel[5], barrel[6]
            excluded_half_angle = barrel[8]
            excluded_z_min, excluded_z_max, witness_margin = barrel[9], barrel[10], barrel[11]
            if inner_radius <= 0.0 or outer_radius <= inner_radius:
                gs.raise_exception(
                    "`boundary_ball_smooth_barrel` requires 0 < inner_contact_radius "
                    "< outer_contact_radius."
                )
            if z_max <= z_min:
                gs.raise_exception("`boundary_ball_smooth_barrel` requires z_max > z_min.")
            if excluded_z_max <= excluded_z_min:
                gs.raise_exception(
                    "`boundary_ball_smooth_barrel` requires excluded_z_max > excluded_z_min."
                )
            if witness_margin < 0.0:
                gs.raise_exception("`boundary_ball_smooth_barrel` requires witness_margin >= 0.")
            if excluded_half_angle < 0.0 or excluded_half_angle >= np.pi:
                gs.raise_exception(
                    "`boundary_ball_smooth_barrel` requires 0 <= excluded_half_angle_rad < pi."
                )
            if self.boundary_ball_initial_poses:
                pos = np.asarray(self.boundary_ball_initial_poses[i_set][0], dtype=np.float64)
                quat = np.asarray(self.boundary_ball_initial_poses[i_set][1], dtype=np.float64)
                quat /= np.linalg.norm(quat)
                identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                if not np.allclose(pos, 0.0, atol=1.0e-8, rtol=0.0) or not (
                    np.allclose(quat, identity_quat, atol=1.0e-8, rtol=0.0)
                    or np.allclose(quat, -identity_quat, atol=1.0e-8, rtol=0.0)
                ):
                    gs.raise_exception(
                        "`boundary_ball_smooth_barrel` requires the selected set's initial pose "
                        "to be identity (static upright barrel)."
                    )

        if self.boundary_ball_jug_history is not None:
            jug = self.boundary_ball_jug_history
            if len(jug) != 10:
                gs.raise_exception(
                    "`boundary_ball_jug_history` must be (set_index, r_inner, r_outer, "
                    "z_min, z_max, spout_azimuth_rad, spout_half_angle_rad, spout_dz, "
                    "spout_dr, hist_tol)."
                )
            i_set = jug[0]
            values = np.asarray(jug[1:], dtype=np.float64)
            if not np.isfinite(values).all():
                gs.raise_exception("`boundary_ball_jug_history` requires finite values.")
            if not 0 <= i_set < len(self.boundary_ball_sets):
                gs.raise_exception("`boundary_ball_jug_history` set_index is out of range.")
            if not self.boundary_ball_sets[i_set][1]:
                gs.raise_exception("`boundary_ball_jug_history` requires a kinematic boundary-ball set.")
            inner_radius, outer_radius = jug[1], jug[2]
            z_min, z_max = jug[3], jug[4]
            spout_half_angle = jug[6]
            hist_tol = jug[9]
            if inner_radius <= 0.0 or outer_radius <= inner_radius:
                gs.raise_exception(
                    "`boundary_ball_jug_history` requires 0 < r_inner < r_outer."
                )
            if z_max <= z_min:
                gs.raise_exception("`boundary_ball_jug_history` requires z_max > z_min.")
            if spout_half_angle <= 0.0:
                gs.raise_exception("`boundary_ball_jug_history` requires spout_half_angle_rad > 0.")
            if hist_tol <= 0.0:
                gs.raise_exception("`boundary_ball_jug_history` requires hist_tol > 0.")

        if self.hash_grid_res is None:
            max_hash_grid_res = np.ceil(
                (np.array(self.upper_bound) - np.array(self.lower_bound)) / self.hash_grid_cell_size
            ).astype(gs.np_int)
            self._hash_grid_res = np.minimum(max_hash_grid_res, np.array([150, 150, 150], dtype=gs.np_int))
        else:
            self._hash_grid_res = np.ceil(np.array(self.hash_grid_res) / self.hash_grid_cell_size).astype(gs.np_int)


class FEMOptions(Options):
    """
    Options configuring the FEMSolver.

    Note
    ----
    - Damping coefficients are used to control the damping effect in the simulation.
    They are used in the Rayleigh Damping model, which is a common damping model in FEM simulations.
    Reference: https://doc.comsol.com/5.5/doc/com.comsol.help.sme/sme_ug_modeling.05.083.html
    - TODO Move it to material parameters in the future instead of solver options.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. If none, it will inherit from `SimOptions`. Defaults to None.
    gravity : tuple, optional
        Gravity force in N/kg. If none, it will inherit from `SimOptions`. Defaults to None.
    damping : float, optional
        Damping factor. Defaults to 0.0.
    floor_height : float, optional
        Height of the floor in meters. If none, it will inherit from `SimOptions`. Defaults to None.
    use_implicit_solver : bool, optional
        Whether to use the implicit solver. Defaults to False.
        Implicit solver is a more stable solver for FEM. It can be used with a large time step.
    n_newton_iterations : int, optional
        Maximum number of Newton iterations. Defaults to 1. Only used when `use_implicit_solver` is True.
    n_pcg_iterations : int, optional
        Maximum number of PCG iterations. Defaults to 500. Only used when `use_implicit_solver` is True.
    n_linesearch_iterations : int, optional
        Maximum number of line search iterations. Defaults to 0. Only used when `use_implicit_solver` is True.
    newton_dx_threshold : float, optional
        Threshold for the Newton solver. Defaults to 1e-6. Only used when `use_implicit_solver` is True.
    pcg_threshold : float, optional
        Threshold for the PCG solver. Defaults to 1e-6. Only used when `use_implicit_solver` is True.
    linesearch_c : float, optional
        Line search sufficient decrease parameter. Defaults to 1e-4. Only used when `use_implicit_solver` is True.
    linesearch_tau : float, optional
        Line search step size reduction factor. Defaults to 0.5. Only used when `use_implicit_solver` is True.
    damping_alpha : float, optional
        Rayleigh Damping factor for the implicit solver. Defaults to 0.5. Only used when `use_implicit_solver` is True.
    damping_beta : float, optional
        Rayleigh Damping factor for the implicit solver. Defaults to 5e-4. Only used when `use_implicit_solver` is True.
    enable_vertex_constraints : bool, optional
        Whether to enable vertex constraints. Defaults to False.
    """

    dt: PositiveFloat | None = None
    gravity: Vec3FType | None = None
    damping: NonNegativeFloat = 0.0
    floor_height: float | None = None
    use_implicit_solver: StrictBool = False
    n_newton_iterations: PositiveInt = 1
    n_pcg_iterations: PositiveInt = 500
    n_linesearch_iterations: NonNegativeInt = 0
    newton_dx_threshold: PositiveFloat = 1e-6
    pcg_threshold: PositiveFloat = 1e-6
    linesearch_c: PositiveFloat = 1e-4
    linesearch_tau: PositiveFloat = 0.5
    damping_alpha: NonNegativeFloat = 0.5
    damping_beta: NonNegativeFloat = 5e-4
    enable_vertex_constraints: StrictBool = False


class SFOptions(Options):
    """
    Options configuring the SFSolver.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. If none, it will inherit from `SimOptions`. Defaults to None.
    """

    dt: PositiveFloat | None = None
    res: PositiveInt = 128
    solver_iters: PositiveInt = 500
    decay: PositiveFloat = 0.99

    T_low: float = 1.0
    T_high: float = 0.0

    inlet_pos: Vec3FType = (0.6, 0.0, 0.1)
    inlet_vel: Vec3FType = (0.0, 0.0, 1.0)
    inlet_quat: UnitVec4FType = (1.0, 0.0, 0.0, 0.0)
    inlet_s: PositiveFloat = 400.0
