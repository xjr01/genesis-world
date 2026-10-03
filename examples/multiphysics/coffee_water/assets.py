from dataclasses import dataclass


@dataclass(frozen=True)
class CoffeeWaterAssets:
    """Paths, poses, sizes and masses of entities used only by the coffee-water task."""

    cup: str = "meshes/drinking_glass/12-oz-glass.obj"
    cup_cavity: str = "meshes/drinking_glass/12-oz-glass-cavity.obj"
    robot: str = "urdf/sim1_acone/acone_collision.urdf"
    rod: str = "meshes/glass_stirring_rod/glass_rod.obj"
    coffee_cup_pos: tuple[float, float, float] = (-0.09, -0.045, 0.0)
    water_cup_pos: tuple[float, float, float] = (0.09, -0.045, 0.0)
    robot_pos: tuple[float, float, float] = (-0.06, -0.55, 0.52)
    robot_quat: tuple[float, float, float, float] = (0.5, -0.5, 0.5, 0.5)
    rod_park: tuple[float, float, float] = (-0.19, -0.025, 0.0)
    table_y: float = -0.045
    sponge_size: tuple[float, float, float] = (0.04, 0.06, 0.08)
    sponge_start: tuple[float, float, float] = (0.0, -0.044, -0.17)
    sponge_end: tuple[float, float, float] = (0.0, -0.044, 0.17)
    cup_mass: float = 0.02
    contact_sdf_cell_size: float = 0.0005
    coffee_fill_fraction: float = 0.5
    water_fill_fraction: float = 1.0
