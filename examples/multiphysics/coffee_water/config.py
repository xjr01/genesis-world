from dataclasses import dataclass


@dataclass(frozen=True)
class CoffeeWaterSolverConfig:
    """Scene-wide numerical parameters independent of the task's entity assets."""

    control_dt: float = 0.002
    contact_substeps: int = 2
    gravity: tuple[float, float, float] = (0.0, -9.8, 0.0)
    lower_bound: tuple[float, float, float] = (-0.4, -1.0 / 15.0, -4.0 / 15.0)
    upper_bound: tuple[float, float, float] = (0.4, 8.0 / 15.0, 4.0 / 15.0)
    pbstf_particle_size: float = 2.0 / 1500.0
    pbd_particle_size: float = 0.005
    pbd_iterations: int = 100
    pbd_constraint_acceleration: float = 0.85
    pbstf_diffusion_coeff: float = 0.005
    pbstf_iterations: int = 20
    pbstf_topology_rebuild_interval: int = 10
    pbstf_max_surface_neighbors: int = 128
    pbstf_max_localmesh_neighbors: int = 64
    is_pbstf_pca_normals_enabled: bool = False


@dataclass(frozen=True)
class CoffeeWaterMaterialConfig:
    """Per-liquid physical and numerical properties independent of entity geometry and pose."""

    density: float = 1000.0
    density_compliance: float = 843750.0
    surface_tension_compliance: float = 1.0 / 225.0
    surface_distance_compliance: float = 40.0
    interior_distance_compliance: float = 180.0
    surface_viscosity: float = 0.5
    interior_viscosity: float = 0.5
