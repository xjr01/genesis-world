from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableWipingSolverConfig:
    """PBSTF and sponge discretization settings independent of task geometry."""

    dt: float = 0.002
    gravity: tuple[float, float, float] = (0.0, -9.8, 0.0)
    particle_size: float = 2.0 / 300.0
    lower_bound: tuple[float, float, float] = (-0.4, -1.0 / 15.0, -4.0 / 15.0)
    upper_bound: tuple[float, float, float] = (0.4, 4.0 / 15.0, 4.0 / 15.0)
    max_solver_iterations: int = 10
    max_surface_neighbors: int = 128
    max_localmesh_neighbors: int = 64
    pbd_particle_size: float = 0.005
    pbd_solver_iterations: int = 100
    pbd_constraint_acceleration: float = 0.85


@dataclass(frozen=True)
class TableWipingMaterialConfig:
    """Liquid and absorbent contact properties independent of entity placement."""

    density: float = 1000.0
    density_compliance: float = 33750.0
    surface_tension_compliance: float = 1.0 / 225.0
    surface_distance_compliance: float = 40.0
    interior_distance_compliance: float = 180.0
    surface_viscosity: float = 0.5
    interior_viscosity: float = 0.5
    collider_adhesion_compliance: float = 50.0
    collider_friction: float = 0.5
    absorption_rate: float = 4000.0
    absorption_capacity_fraction: float = 1.0


@dataclass(frozen=True)
class TableWipingAssets:
    """Robot, sponge, table and liquid geometry used by the wiping scenario."""

    robot: str = "urdf/panda_bullet/panda.urdf"
    robot_base_pos: tuple[float, float, float] = (0.0, -1.0 / 60.0, -7.0 / 15.0)
    robot_base_quat: tuple[float, float, float, float] = (2**-0.5, -(2**-0.5), 0.0, 0.0)
    sponge_grid_resolution: tuple[int, int, int] = (10, 7, 20)
    sponge_density: float = 30.0
    sponge_lower: tuple[float, float, float] = (-0.04, 0.02 / 15.0, -0.08)
    sponge_upper: tuple[float, float, float] = (0.04, 0.85 / 15.0, 0.08)
    table_pos: tuple[float, float, float] = (0.0, -1.0 / 60.0, 0.0)
    table_size: tuple[float, float, float] = (0.8, 1.0 / 30.0, 8.0 / 15.0)
    liquid_lower: tuple[float, float, float] = (-1.0 / 6.0, 1.0 / 300.0, -7.0 / 150.0)
    liquid_upper: tuple[float, float, float] = (-1.0 / 30.0, 7.0 / 300.0, 7.0 / 150.0)
    camera_res: tuple[int, int] = (1280, 720)
    camera_pos: tuple[float, float, float] = (0.0, 0.3, 2.0 / 3.0)
    camera_lookat: tuple[float, float, float] = (0.0, 1.0 / 30.0, -0.1)
    camera_up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    camera_fov: float = 40.0


@dataclass(frozen=True)
class TableWipingTaskConfig:
    """Robot command and wiping trajectory parameters."""

    start_pos: tuple[float, float, float] = (-7.0 / 30.0, 0.0, 0.0)
    end_pos: tuple[float, float, float] = (7.0 / 30.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    settle_time: float = 0.2
    wipe_time: float = 2.5
    steps: int = 5000
    initial_qpos: tuple[float, ...] = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04)
    finger_open_qpos: float = 0.04
    finger_closed_qpos: float = 0.036
    tool_center_point: tuple[float, float, float] = (0.0, 0.0, 3.02 / 15.0)
    grasp_quat: tuple[float, float, float, float] = (
        0.6532814824,
        0.6532814824,
        0.2705980501,
        -0.2705980501,
    )


@dataclass(frozen=True)
class TableWipingScenarioConfig:
    solver: TableWipingSolverConfig = field(default_factory=TableWipingSolverConfig)
    material: TableWipingMaterialConfig = field(default_factory=TableWipingMaterialConfig)
    assets: TableWipingAssets = field(default_factory=TableWipingAssets)
    task: TableWipingTaskConfig = field(default_factory=TableWipingTaskConfig)
