import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.integrations import GranularFluidProperties, GranularFluidSetup, create_granular_fluid_setup

from .config import LitterScoopScenarioConfig


@dataclass(frozen=True)
class LitterScoopRuntime:
    """Named scene handles used by an adapter or task controller after construction."""

    scene: gs.Scene
    config: LitterScoopScenarioConfig
    setup: GranularFluidSetup
    sand: object
    water: object
    shovel: object
    camera: object | None

    @property
    def control_dt(self) -> float:
        return self.config.solver.dt

    def synchronize_shovel(self):
        """Align the visual shovel with the mesh obstacle across all environments."""
        self.shovel.set_pos(self.scene.dem_solver.get_sdf_obstacle_pos())
        self.shovel.set_quat(self.scene.dem_solver.get_sdf_obstacle_quat())


def build_scene(
    config: LitterScoopScenarioConfig | None = None,
    *,
    show_viewer: bool = False,
    add_camera: bool = False,
) -> LitterScoopRuntime:
    """Build the coupled litter-scoop scene without starting task motion or recording."""
    config = config or LitterScoopScenarioConfig()
    solver = config.solver
    material = config.material
    assets = config.assets
    properties = GranularFluidProperties(
        particle_size=solver.particle_size,
        sand_density=material.sand_density,
        liquid_density=material.water_density,
        lower_bound=solver.lower_bound,
        upper_bound=solver.upper_bound,
    )
    setup = create_granular_fluid_setup(
        properties,
        dem_options=gs.options.DEMOptions(
            particle_size=solver.particle_size,
            ddt_safety=solver.ddt_safety,
            lower_bound=solver.lower_bound,
            upper_bound=solver.upper_bound,
        ),
        flip_options=gs.options.FLIPOptions(
            grid_res=solver.grid_res,
            viscosity_coeff=solver.viscosity_coeff,
            lower_bound=solver.lower_bound,
            upper_bound=solver.upper_bound,
        ),
        sand_material=gs.materials.DEM.Sand(
            sampler="fcc",
            rho=material.sand_density,
            young_modulus=material.sand_young_modulus,
            poisson_ratio=material.sand_poisson_ratio,
            friction_angle=material.sand_friction_angle,
            max_ratio=material.sand_max_water_ratio,
        ),
        liquid_material=gs.materials.FLIP.Liquid(rho=material.water_density),
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=solver.dt, substeps=solver.substeps, gravity=solver.gravity),
        dem_options=setup.dem_options,
        flip_options=setup.flip_options,
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())

    box_x = solver.upper_bound[0] - solver.lower_bound[0]
    box_y = solver.upper_bound[1] - solver.lower_bound[1]
    wall_z = 0.5 * assets.wall_height
    wall_surface = gs.surfaces.Default(color=(0.55, 0.55, 0.6))
    wall_specs = (
        (
            (solver.lower_bound[0] - 0.5 * assets.wall_thickness, 0.0, wall_z),
            (assets.wall_thickness, box_y, assets.wall_height),
        ),
        (
            (solver.upper_bound[0] + 0.5 * assets.wall_thickness, 0.0, wall_z),
            (assets.wall_thickness, box_y, assets.wall_height),
        ),
        (
            (0.0, solver.lower_bound[1] - 0.5 * assets.wall_thickness, wall_z),
            (box_x + 2.0 * assets.wall_thickness, assets.wall_thickness, assets.wall_height),
        ),
        (
            (0.0, solver.upper_bound[1] + 0.5 * assets.wall_thickness, wall_z),
            (box_x + 2.0 * assets.wall_thickness, assets.wall_thickness, assets.wall_height),
        ),
    )
    for wall_pos, wall_size in wall_specs:
        scene.add_entity(
            morph=gs.morphs.Box(pos=wall_pos, size=wall_size),
            material=gs.materials.Kinematic(),
            surface=wall_surface,
        )

    blade_quat = (
        math.cos(0.5 * assets.blade_angle),
        0.0,
        math.sin(0.5 * assets.blade_angle),
        0.0,
    )
    repository_root = Path(__file__).resolve().parents[3]
    shovel = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(repository_root / assets.shovel_mesh),
            pos=assets.blade_initial_pos,
            quat=blade_quat,
        ),
        material=gs.materials.Kinematic(),
    )
    sand = scene.add_entity(
        morph=gs.morphs.Box(pos=assets.sand_pos, size=assets.sand_size),
        material=setup.sand_material,
        surface=gs.surfaces.Default(color=(0.87, 0.72, 0.53)),
    )
    water = scene.add_entity(
        morph=gs.morphs.Sphere(pos=assets.droplet_pos, radius=assets.droplet_radius),
        material=setup.liquid_material,
        surface=gs.surfaces.Default(color=(0.2, 0.5, 0.9), opacity=0.85),
    )
    camera = None
    if add_camera:
        camera = scene.add_camera(
            res=assets.camera_res,
            pos=assets.camera_pos,
            lookat=assets.camera_lookat,
            fov=assets.camera_fov,
        )

    with np.load(repository_root / assets.shovel_sdf) as sdf:
        mesh_sha256 = hashlib.sha256((repository_root / assets.shovel_mesh).read_bytes()).hexdigest()
        if sdf["mesh_sha256"].item() != mesh_sha256 or not sdf["has_open_slots"].item():
            raise ValueError("Shovel collision data must match the visual mesh and preserve its slots.")
        scene.build()
        scene.dem_solver.set_sdf_obstacle(
            sdf["sdf_val"],
            sdf["dims"],
            sdf["origin"],
            sdf["cell"],
            assets.blade_initial_pos,
            quat=blade_quat,
        )
    scene.reset(scene.get_state())
    runtime = LitterScoopRuntime(
        scene=scene,
        config=config,
        setup=setup,
        sand=sand,
        water=water,
        shovel=shovel,
        camera=camera,
    )
    scene.register_pre_step_callback(runtime.synchronize_shovel)
    return runtime
