from dataclasses import dataclass
from pathlib import Path

import genesis as gs
from genesis.options.solvers import PBSTFStaticColliderOptions

from .assets import CoffeeWaterAssets


@dataclass(frozen=True)
class CoffeeWaterBoundaryConfig:
    """Interaction parameters attached to the task's table, cups, rod and sponge."""

    adhesion_compliance: float = 50.0
    table_friction: float = 0.5
    mesh_friction: float = 0.1
    cup_sdf_res: int = 128
    absorption_rate: float = 4000.0
    absorption_capacity_fraction: float = 1.0


@dataclass(frozen=True)
class CoffeeWaterColliders:
    """Object-bound PBSTF collider descriptions prepared before solver construction."""

    table: gs.options.PBSTFBoxStaticColliderOptions
    values: list[PBSTFStaticColliderOptions]


def create_pbstf_colliders(
    assets_dir: Path,
    assets: CoffeeWaterAssets,
    config: CoffeeWaterBoundaryConfig,
) -> CoffeeWaterColliders:
    """Create colliders from example assets while keeping numerical solver settings separate."""
    cup_path = assets_dir / assets.cup
    table = gs.options.PBSTFBoxStaticColliderOptions(
        is_collider_adhesion_friction_enabled=True,
        collider_adhesion_compliance=config.adhesion_compliance,
        collider_friction=config.table_friction,
        lower=(-0.4, assets.table_y - 0.02, -4.0 / 15.0),
        upper=(0.4, assets.table_y, 4.0 / 15.0),
    )
    cups = tuple(
        gs.options.PBSTFMeshStaticColliderOptions(
            pos=pos,
            is_collider_adhesion_friction_enabled=True,
            collider_adhesion_compliance=config.adhesion_compliance,
            collider_friction=config.mesh_friction,
            file=str(cup_path),
            sdf_res=config.cup_sdf_res,
        )
        for pos in (assets.coffee_cup_pos, assets.water_cup_pos)
    )
    rod = gs.options.PBSTFMeshStaticColliderOptions(
        pos=assets.rod_park,
        is_collider_adhesion_friction_enabled=True,
        collider_adhesion_compliance=config.adhesion_compliance,
        collider_friction=config.mesh_friction,
        file=str(assets_dir / assets.rod),
        sdf_res=config.cup_sdf_res,
    )
    sponge = gs.options.PBSTFAbsorbentBoxStaticColliderOptions(
        pos=assets.sponge_start,
        is_collider_adhesion_friction_enabled=True,
        collider_adhesion_compliance=config.adhesion_compliance,
        collider_friction=config.table_friction,
        lower=(-0.5 * assets.sponge_size[0], 0.0, -0.5 * assets.sponge_size[2]),
        upper=(0.5 * assets.sponge_size[0], assets.sponge_size[1], 0.5 * assets.sponge_size[2]),
        absorption_rate=config.absorption_rate,
        absorption_capacity_fraction=config.absorption_capacity_fraction,
        pbd_entity_name="sponge",
    )
    return CoffeeWaterColliders(table=table, values=[table, *cups, rod, sponge])
