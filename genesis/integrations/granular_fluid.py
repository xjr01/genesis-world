import math
from dataclasses import dataclass

import numpy as np

import genesis as gs
from genesis.engine import materials
from genesis.options.solvers import DEMOptions, FLIPOptions


@dataclass(frozen=True)
class GranularFluidProperties:
    """Physical quantities shared by an adapter and the coupled DEM/FLIP setup."""

    particle_size: float = 0.01
    sand_density: float = 2.5
    liquid_density: float = 1000.0
    lower_bound: tuple[float, float, float] = (-0.5, -0.5, 0.0)
    upper_bound: tuple[float, float, float] = (0.5, 0.5, 1.0)


@dataclass(frozen=True)
class GranularFluidSetup:
    """Validated options and materials for a coupled granular-liquid scene."""

    dem_options: DEMOptions
    flip_options: FLIPOptions
    sand_material: materials.DEM.Sand
    liquid_material: materials.FLIP.Liquid


def create_granular_fluid_setup(
    properties: GranularFluidProperties,
    dem_options: DEMOptions | None = None,
    flip_options: FLIPOptions | None = None,
    sand_material: materials.DEM.Sand | None = None,
    liquid_material: materials.FLIP.Liquid | None = None,
) -> GranularFluidSetup:
    """Create one consistent DEM/FLIP configuration while preserving solver-specific controls."""
    if not isinstance(properties, GranularFluidProperties):
        gs.raise_exception("`properties` requires `GranularFluidProperties`.")
    if properties.particle_size <= 0.0:
        gs.raise_exception("Granular `particle_size` must be positive.")
    if properties.sand_density <= 0.0 or properties.liquid_density <= 0.0:
        gs.raise_exception("Granular and liquid densities must be positive.")
    lower = np.asarray(properties.lower_bound, dtype=float)
    upper = np.asarray(properties.upper_bound, dtype=float)
    if not np.all(upper > lower):
        gs.raise_exception("Granular-fluid `upper_bound` must be greater than `lower_bound`.")

    if dem_options is None:
        dem_options = DEMOptions(
            particle_size=properties.particle_size,
            lower_bound=properties.lower_bound,
            upper_bound=properties.upper_bound,
        )
    if flip_options is None:
        flip_options = FLIPOptions(
            lower_bound=properties.lower_bound,
            upper_bound=properties.upper_bound,
        )
    if sand_material is None:
        sand_material = materials.DEM.Sand(rho=properties.sand_density)
    if liquid_material is None:
        liquid_material = materials.FLIP.Liquid(rho=properties.liquid_density)

    if not isinstance(dem_options, DEMOptions):
        gs.raise_exception("Coupled granular flow requires `DEMOptions`.")
    if not isinstance(flip_options, FLIPOptions):
        gs.raise_exception("Coupled granular flow requires `FLIPOptions`.")
    if not isinstance(sand_material, materials.DEM.Sand):
        gs.raise_exception("Coupled granular flow requires `materials.DEM.Sand`.")
    if not isinstance(liquid_material, materials.FLIP.Liquid):
        gs.raise_exception("Coupled granular flow requires `materials.FLIP.Liquid`.")
    if not math.isclose(dem_options.particle_size, properties.particle_size):
        gs.raise_exception("Common `particle_size` conflicts with `DEMOptions`.")
    if not np.allclose(dem_options.lower_bound, properties.lower_bound) or not np.allclose(
        dem_options.upper_bound, properties.upper_bound
    ):
        gs.raise_exception("Common bounds conflict with `DEMOptions`.")
    if not np.allclose(flip_options.lower_bound, properties.lower_bound) or not np.allclose(
        flip_options.upper_bound, properties.upper_bound
    ):
        gs.raise_exception("Common bounds conflict with `FLIPOptions`.")
    if not math.isclose(sand_material.rho, properties.sand_density):
        gs.raise_exception("Common `sand_density` conflicts with the DEM material.")
    if not math.isclose(liquid_material.rho, properties.liquid_density):
        gs.raise_exception("Common `liquid_density` conflicts with the FLIP material.")

    if flip_options.dem_coupling:
        cell_size = float(np.max(upper - lower) / flip_options.grid_res)
        grain_radius = 0.5 * properties.particle_size
        grain_volume = 4.0 / 3.0 * math.pi * grain_radius**3
        absorbable_particles = int(grain_volume / cell_size**3 * flip_options.seed_sub_factor**3)
        if absorbable_particles < 1:
            gs.raise_exception(
                "DEM grains are too small relative to the FLIP grid for absorption; increase `grid_res` or "
                "`particle_size`."
            )

    return GranularFluidSetup(
        dem_options=dem_options,
        flip_options=flip_options,
        sand_material=sand_material,
        liquid_material=liquid_material,
    )
