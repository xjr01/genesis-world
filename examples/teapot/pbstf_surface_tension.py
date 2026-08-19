import argparse
import math
import os
from typing import NamedTuple

import numpy as np

import genesis as gs
import genesis.utils.geom as geom_utils
import genesis.utils.mesh as mesh_utils
from genesis.utils.misc import tensor_to_array

if __package__:
    from .fluid_helper import (
        TeapotSettings,
        add_teapot_entities,
        create_teapot_settings,
        initialize_teapot_manipulator,
        sample_teapot_particles,
        teapot_pose,
        update_rigid_teapot,
        update_teapot_case,
        update_teapot_manipulator,
    )
else:
    from fluid_helper import (
        TeapotSettings,
        add_teapot_entities,
        create_teapot_settings,
        initialize_teapot_manipulator,
        sample_teapot_particles,
        teapot_pose,
        update_rigid_teapot,
        update_teapot_case,
        update_teapot_manipulator,
    )

__all__ = (
    "CASES",
    "CASE_BOUNCE",
    "CASE_CONE",
    "CASE_CUBE",
    "CASE_MERGE",
    "CASE_MOP",
    "CASE_SPONGE_SQUEEZE",
    "CASE_SWEEP",
    "CASE_TAP",
    "CASE_TEAPOT",
    "MopSettings",
    "SpongeSqueezeSettings",
    "SweepSettings",
    "TeapotSettings",
    "add_case_entities",
    "build_scene",
    "case_settings",
    "draw_case_colliders",
    "mop_pose",
    "porous_top_particle_indices",
    "sample_teapot_particles",
    "start_viewer_recording",
    "stop_viewer",
    "teapot_pose",
    "update_mop_case",
    "update_rigid_teapot",
    "update_sponge_squeeze_case",
    "update_sweep_case",
    "update_teapot_manipulator",
)

CASE_CUBE = "cube"
CASE_MERGE = "merge"
CASE_BOUNCE = "bounce"
CASE_CONE = "cone"
CASE_MOP = "mop"
CASE_SPONGE_SQUEEZE = "sponge_squeeze"
CASE_SWEEP = "sweep"
CASE_TAP = "tap"
CASE_TEAPOT = "teapot"
CASES = (
    CASE_CUBE,
    CASE_MERGE,
    CASE_BOUNCE,
    CASE_CONE,
    CASE_MOP,
    CASE_SPONGE_SQUEEZE,
    CASE_SWEEP,
    CASE_TAP,
    CASE_TEAPOT,
)

DEFAULT_CASE_DT = 1.0 / 30.0


class TapEmitterSettings(NamedTuple):
    pos: tuple[float, float, float]
    direction: tuple[float, float, float]
    droplet_size: float
    generation_speed: float
    initial_speed: float
    max_particles: int


class MopSettings(NamedTuple):
    collider_lower: tuple[float, float, float]
    collider_upper: tuple[float, float, float]
    table_pos: tuple[float, float, float]
    table_size: tuple[float, float, float]
    liquid_lower: tuple[float, float, float]
    liquid_upper: tuple[float, float, float]
    start_pos: tuple[float, float, float]
    end_pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    settle_time: float
    wipe_time: float


class SweepSettings(NamedTuple):
    collider_idx: int
    collider_lower: tuple[float, float, float]
    collider_upper: tuple[float, float, float]
    table_pos: tuple[float, float, float]
    table_size: tuple[float, float, float]
    liquid_lower: tuple[float, float, float]
    liquid_upper: tuple[float, float, float]
    start_pos: tuple[float, float, float]
    end_pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    settle_time: float
    wipe_time: float


class SpongeSqueezeSettings(NamedTuple):
    table_pos: tuple[float, float, float]
    table_size: tuple[float, float, float]
    liquid_lower: tuple[float, float, float]
    liquid_upper: tuple[float, float, float]
    sponge_lower: tuple[float, float, float]
    sponge_upper: tuple[float, float, float]
    compression_distance: float
    absorb_time: float
    squeeze_time: float
    hold_time: float
    release_time: float


class CaseSettings(NamedTuple):
    scale: int
    dt: float
    gravity: tuple[float, float, float]
    lower_bound: tuple[float, float, float]
    upper_bound: tuple[float, float, float]
    camera_pos: tuple[float, float, float]
    camera_lookat: tuple[float, float, float]
    static_colliders: tuple[gs.options.PBSTFStaticColliderOptions, ...]
    max_solver_iterations: int
    max_surface_neighbors: int
    max_localmesh_neighbors: int
    enable_pca_normals: bool
    steps: int
    teapot: TeapotSettings | None
    emitter: TapEmitterSettings | None
    mop: MopSettings | None = None
    sponge: SpongeSqueezeSettings | None = None
    sweep: SweepSettings | None = None


def _liquid_material(
    sampler="staggered",
    rho=1000.0,
    density_compliance=500.0,
    surface_tension_compliance=0.8,
    surface_distance_compliance=40.0,
    interior_distance_compliance=180.0,
    surface_viscosity=0.3,
    interior_viscosity=0.3,
    is_collider_adhesion_friction_enabled=False,
    collider_adhesion_compliance=10.0,
    collider_friction=0.1,
):
    return gs.materials.PBSTF.Liquid(
        sampler=sampler,
        rho=rho,
        density_compliance=density_compliance,
        surface_tension_compliance=surface_tension_compliance,
        surface_distance_compliance=surface_distance_compliance,
        interior_distance_compliance=interior_distance_compliance,
        surface_viscosity=surface_viscosity,
        interior_viscosity=interior_viscosity,
        is_collider_adhesion_friction_enabled=is_collider_adhesion_friction_enabled,
        collider_adhesion_compliance=collider_adhesion_compliance,
        collider_friction=collider_friction,
    )


def _case_liquid_material(case):
    if case == CASE_MERGE:
        return _liquid_material(
            surface_tension_compliance=1.0,
            surface_viscosity=0.05,
            interior_viscosity=0.05,
        )
    if case == CASE_CONE:
        return _liquid_material(
            surface_tension_compliance=1.0,
            interior_distance_compliance=90.0,
            surface_viscosity=0.05,
            interior_viscosity=0.05,
        )
    if case == CASE_TAP:
        return _liquid_material(
            sampler="regular",
            surface_tension_compliance=0.21,
            interior_distance_compliance=90.0,
            surface_viscosity=0.2,
            interior_viscosity=0.2,
        )
    if case == CASE_TEAPOT:
        return _liquid_material(
            sampler="regular",
            density_compliance=150.0,
            surface_tension_compliance=3.0,
            surface_distance_compliance=40.0,
            interior_distance_compliance=180.0,
            surface_viscosity=0.2,
            interior_viscosity=0.05,
            is_collider_adhesion_friction_enabled=True,
            collider_adhesion_compliance=20.0,
            collider_friction=0.01,
        )
    if case in (CASE_MOP, CASE_SWEEP):
        return _liquid_material(
            sampler="regular",
            density_compliance=150.0,
            surface_tension_compliance=1.0,
            surface_distance_compliance=40.0,
            interior_distance_compliance=180.0,
            surface_viscosity=0.5,
            interior_viscosity=0.5,
            is_collider_adhesion_friction_enabled=True,
            collider_adhesion_compliance=20.0,
            collider_friction=0.5,
        )
    return _liquid_material()


def _case_porous_material(case):
    if case in (CASE_MOP, CASE_SPONGE_SQUEEZE):
        return gs.materials.PBSTF.PorousElastic(
            porosity=0.8,
            deviatoric_compliance=1e-6 if case == CASE_MOP else 1e-5,
            volumetric_compliance=1e-6 if case == CASE_MOP else 1e-5,
            capillary_compliance=10.0,
            drag=10.0,
            wet_deviatoric_compliance_scale=1.5,
            wet_volumetric_compliance_scale=1.5,
            bloating_volume_strain=0.05,
        )
    raise ValueError(f"Unknown PBSTF porous example case: {case}")


def case_settings(case):
    if case == CASE_CUBE:
        return CaseSettings(
            scale=10,
            dt=DEFAULT_CASE_DT,
            gravity=(0.0, 0.0, 0.0),
            lower_bound=(-2.0, -2.0, -2.0),
            upper_bound=(2.0, 2.0, 2.0),
            camera_pos=(4.0, 4.0, 3.0),
            camera_lookat=(0.0, 0.0, 0.0),
            static_colliders=(),
            max_solver_iterations=100,
            max_surface_neighbors=128,
            max_localmesh_neighbors=64,
            enable_pca_normals=False,
            steps=1500,
            teapot=None,
            emitter=None,
        )
    if case == CASE_MERGE:
        return CaseSettings(
            scale=10,
            dt=DEFAULT_CASE_DT,
            gravity=(0.0, 0.0, 0.0),
            lower_bound=(-10.0, -10.0, -10.0),
            upper_bound=(10.0, 10.0, 10.0),
            camera_pos=(4.0, 3.0, 4.0),
            camera_lookat=(0.0, -1.0, 0.0),
            static_colliders=(),
            max_solver_iterations=100,
            max_surface_neighbors=128,
            max_localmesh_neighbors=64,
            enable_pca_normals=False,
            steps=300,
            teapot=None,
            emitter=None,
        )
    if case == CASE_BOUNCE:
        return CaseSettings(
            scale=10,
            dt=DEFAULT_CASE_DT,
            gravity=(0.0, 0.0, 0.0),
            lower_bound=(-10.0, -2.0, -10.0),
            upper_bound=(10.0, 10.0, 10.0),
            camera_pos=(4.0, 2.0, 4.0),
            camera_lookat=(0.0, -1.0, 0.0),
            static_colliders=(),
            max_solver_iterations=100,
            max_surface_neighbors=128,
            max_localmesh_neighbors=64,
            enable_pca_normals=False,
            steps=300,
            teapot=None,
            emitter=None,
        )
    if case == CASE_CONE:
        return CaseSettings(
            scale=10,
            dt=DEFAULT_CASE_DT,
            gravity=(0.0, 0.0, 0.0),
            lower_bound=(-20.0, -20.0, -20.0),
            upper_bound=(20.0, 20.0, 20.0),
            camera_pos=(12.0, 2.0, 12.0),
            camera_lookat=(0.0, -3.0, 0.0),
            static_colliders=(
                gs.options.PBSTFConeStaticColliderOptions(
                    center=(0.0, -7.0, 0.0),
                    height=(0.0, 5.0, 0.0),
                    radius=5.0 * math.sqrt(3.0),
                ),
            ),
            max_solver_iterations=100,
            max_surface_neighbors=128,
            max_localmesh_neighbors=64,
            enable_pca_normals=False,
            steps=300,
            teapot=None,
            emitter=None,
        )
    if case in (CASE_MOP, CASE_SWEEP):
        table_pos = (0.0, -0.25, 0.0)
        table_size = (12.0, 0.5, 8.0)
        liquid_lower = (-2.5, 0.05, -0.7)
        liquid_upper = (-0.5, 0.35, 0.7)
        start_pos = (-3.5, 0.0, 0.0)
        end_pos = (3.5, 0.0, 0.0)
        collider_lower = (-0.6, 0.02, -1.2)
        collider_upper = (0.6, 0.85, 1.2)
        quat = (1.0, 0.0, 0.0, 0.0)
        settle_time = 1.0
        wipe_time = 5.0
        static_colliders = (
            gs.options.PBSTFBoxStaticColliderOptions(
                pos=table_pos,
                quat=quat,
                lower=tuple(-0.5 * size for size in table_size),
                upper=tuple(0.5 * size for size in table_size),
            ),
        )
        mop = None
        sweep = None
        if case == CASE_MOP:
            mop = MopSettings(
                collider_lower=collider_lower,
                collider_upper=collider_upper,
                table_pos=table_pos,
                table_size=table_size,
                liquid_lower=liquid_lower,
                liquid_upper=liquid_upper,
                start_pos=start_pos,
                end_pos=end_pos,
                quat=quat,
                settle_time=settle_time,
                wipe_time=wipe_time,
            )
        else:
            sweep = SweepSettings(
                collider_idx=1,
                collider_lower=collider_lower,
                collider_upper=collider_upper,
                table_pos=table_pos,
                table_size=table_size,
                liquid_lower=liquid_lower,
                liquid_upper=liquid_upper,
                start_pos=start_pos,
                end_pos=end_pos,
                quat=quat,
                settle_time=settle_time,
                wipe_time=wipe_time,
            )
            static_colliders += (
                gs.options.PBSTFBoxStaticColliderOptions(
                    pos=start_pos,
                    quat=quat,
                    lower=sweep.collider_lower,
                    upper=sweep.collider_upper,
                ),
            )
        return CaseSettings(
            scale=20,
            dt=0.01,
            gravity=(0.0, -9.8, 0.0),
            lower_bound=(-6.0, -1.0, -4.0),
            upper_bound=(6.0, 4.0, 4.0),
            camera_pos=(8.0, 6.0, 9.0),
            camera_lookat=(0.0, 0.4, 0.0),
            static_colliders=static_colliders,
            max_solver_iterations=10,
            max_surface_neighbors=128,
            max_localmesh_neighbors=64,
            enable_pca_normals=False,
            steps=1000,
            teapot=None,
            emitter=None,
            mop=mop,
            sweep=sweep,
        )
    if case == CASE_SPONGE_SQUEEZE:
        sponge_lower = (-0.6, 0.05, -0.6)
        sponge_upper = (0.6, 0.85, 0.6)
        sponge = SpongeSqueezeSettings(
            table_pos=(0.0, -0.25, 0.0),
            table_size=(5.0, 0.5, 4.0),
            liquid_lower=(-1.1, 0.05, -0.6),
            liquid_upper=(-0.55, 0.85, 0.6),
            sponge_lower=sponge_lower,
            sponge_upper=sponge_upper,
            compression_distance=0.45,
            absorb_time=1.0,
            squeeze_time=1.0,
            hold_time=0.5,
            release_time=1.0,
        )
        return CaseSettings(
            scale=20,
            dt=0.01,
            gravity=(0.0, -9.8, 0.0),
            lower_bound=(-2.5, -1.0, -2.0),
            upper_bound=(2.5, 3.0, 2.0),
            camera_pos=(4.0, 3.0, 5.0),
            camera_lookat=(0.0, 0.4, 0.0),
            static_colliders=(
                gs.options.PBSTFBoxStaticColliderOptions(
                    pos=sponge.table_pos,
                    lower=tuple(-0.5 * size for size in sponge.table_size),
                    upper=tuple(0.5 * size for size in sponge.table_size),
                ),
            ),
            max_solver_iterations=10,
            max_surface_neighbors=128,
            max_localmesh_neighbors=64,
            enable_pca_normals=False,
            steps=500,
            teapot=None,
            emitter=None,
            sponge=sponge,
        )
    if case == CASE_TAP:
        return CaseSettings(
            scale=20,
            dt=DEFAULT_CASE_DT,
            gravity=(0.0, -9.8, 0.0),
            lower_bound=(-500.0, -50.0, -500.0),
            upper_bound=(500.0, 500.0, 500.0),
            camera_pos=(12.0, 4.0, 12.0),
            camera_lookat=(0.0, 0.0, 0.0),
            static_colliders=(),
            max_solver_iterations=100,
            max_surface_neighbors=768,
            max_localmesh_neighbors=64,
            enable_pca_normals=True,
            steps=2000,
            teapot=None,
            emitter=TapEmitterSettings(
                pos=(0.0, 5.0, 0.0),
                direction=(0.0, -1.0, 0.0),
                droplet_size=2.0,
                generation_speed=3.0,
                initial_speed=0.0,
                max_particles=200000,
            ),
        )
    if case == CASE_TEAPOT:
        teapot = create_teapot_settings()
        return CaseSettings(
            scale=20,
            dt=0.01,
            gravity=(0.0, -9.8, 0.0),
            lower_bound=(-20.0, -6.04186, -20.0),
            upper_bound=(20.0, 15.0, 20.0),
            camera_pos=teapot.manipulator.camera_pos,
            camera_lookat=teapot.manipulator.camera_lookat,
            static_colliders=(
                gs.options.PBSTFMeshStaticColliderOptions(
                    file=teapot.asset,
                    scale=teapot.mesh_scale,
                    sdf_res=150,
                    pos=teapot.offset,
                    quat=teapot.quat,
                ),
            ),
            max_solver_iterations=5,
            max_surface_neighbors=128,
            max_localmesh_neighbors=64,
            enable_pca_normals=False,
            steps=10000,
            teapot=teapot,
            emitter=None,
        )
    raise ValueError(f"Unknown PBSTF example case: {case}")


def add_case_entities(scene, case, particle_size, settings, material_factory):
    if case == CASE_CUBE:
        liquid = scene.add_entity(
            morph=gs.morphs.Box(
                lower=(-1.0, -1.0, -1.0),
                upper=(1.0, 1.0, 1.0),
            ),
            material=material_factory(case),
        )
        return ((liquid, (0.0, 0.0, 0.0)),)

    if case == CASE_MERGE:
        left = scene.add_entity(
            morph=gs.morphs.Box(
                lower=(-1.5, -0.625, -0.5),
                upper=(-0.5, 0.375, 0.5),
            ),
            material=material_factory(case),
        )
        right = scene.add_entity(
            morph=gs.morphs.Box(
                lower=(0.5, -0.375, -0.5),
                upper=(1.5, 0.625, 0.5),
            ),
            material=material_factory(case),
        )
        return (
            (left, (1.0, 0.0, 0.0)),
            (right, (-1.0, 0.0, 0.0)),
        )

    if case == CASE_BOUNCE:
        liquid = scene.add_entity(
            morph=gs.morphs.Sphere(
                pos=(0.0, 0.0, 0.0),
                radius=1.0,
            ),
            material=material_factory(case),
        )
        return ((liquid, (0.0, -3.0, 0.0)),)

    if case == CASE_CONE:
        liquid = scene.add_entity(
            morph=gs.morphs.Sphere(
                pos=(0.0, 0.0, 0.0),
                radius=1.0,
            ),
            material=material_factory(case),
        )
        return ((liquid, (0.0, -4.0, 0.0)),)

    if case == CASE_MOP:
        mop = settings.mop
        if mop is None:
            gs.raise_exception("The mop case requires mop settings.")
        liquid = scene.add_entity(
            morph=gs.morphs.Box(
                lower=mop.liquid_lower,
                upper=mop.liquid_upper,
            ),
            material=material_factory(case),
        )
        sponge = scene.add_entity(
            morph=gs.morphs.Box(
                pos=mop.start_pos,
                quat=mop.quat,
                offset_pos=0.5 * np.add(mop.collider_lower, mop.collider_upper),
                size=np.subtract(mop.collider_upper, mop.collider_lower),
            ),
            material=_case_porous_material(case),
            surface=gs.surfaces.Default(
                color=(0.9, 0.45, 0.12),
                opacity=0.3,
            ),
        )
        return ((liquid, None), (sponge, None))

    if case == CASE_SWEEP:
        sweep = settings.sweep
        if sweep is None:
            gs.raise_exception("The sweep case requires sweep settings.")
        liquid = scene.add_entity(
            morph=gs.morphs.Box(
                lower=sweep.liquid_lower,
                upper=sweep.liquid_upper,
            ),
            material=material_factory(case),
        )
        return ((liquid, None),)

    if case == CASE_SPONGE_SQUEEZE:
        sponge_settings = settings.sponge
        if sponge_settings is None:
            gs.raise_exception("The sponge squeeze case requires sponge settings.")
        liquid = scene.add_entity(
            morph=gs.morphs.Box(
                lower=sponge_settings.liquid_lower,
                upper=sponge_settings.liquid_upper,
            ),
            material=material_factory(case),
        )
        sponge = scene.add_entity(
            morph=gs.morphs.Box(
                lower=sponge_settings.sponge_lower,
                upper=sponge_settings.sponge_upper,
            ),
            material=_case_porous_material(case),
            surface=gs.surfaces.Default(
                color=(0.95, 0.7, 0.15, 1.0),
            ),
        )
        return ((liquid, None), (sponge, None))

    if case == CASE_TAP:
        emitter_settings = settings.emitter
        if emitter_settings is None:
            gs.raise_exception("The tap case requires emitter settings.")
        emitter = scene.add_emitter(
            material=material_factory(case),
            max_particles=emitter_settings.max_particles,
        )
        return ((emitter.entity, None),)

    if case == CASE_TEAPOT:
        teapot = settings.teapot
        if teapot is None:
            gs.raise_exception("The teapot case requires teapot settings.")
        liquid = add_teapot_entities(
            scene,
            teapot,
            particle_size,
            material=material_factory(case),
        )
        return ((liquid, teapot.particles_vel),)

    raise ValueError(f"Unknown PBSTF example case: {case}")


def mop_pose(time, settings):
    """Return the translation along a single wiping stroke."""
    progress = min(max((time - settings.settle_time) / settings.wipe_time, 0.0), 1.0)
    return tuple(
        settings.start_pos[axis] + progress * (settings.end_pos[axis] - settings.start_pos[axis]) for axis in range(3)
    )


def porous_top_particle_indices(entity):
    """Return the local indices in the highest particle layer of a porous entity."""
    positions_y = tensor_to_array(entity.get_particles_pos())[..., 1]
    is_top_by_env = positions_y >= positions_y.max(axis=-1, keepdims=True) - 0.5 * entity.particle_size
    is_top = np.all(is_top_by_env, axis=tuple(range(is_top_by_env.ndim - 1)))
    return np.flatnonzero(is_top)


def update_mop_case(entity, anchor_positions, anchor_indices, time, settings):
    """Move the anchored particle layer along the mop's wiping stroke."""
    pos = mop_pose(time, settings)
    offset = np.subtract(pos, settings.start_pos)
    if settings.settle_time < time < settings.settle_time + settings.wipe_time:
        velocity = np.subtract(settings.end_pos, settings.start_pos) / settings.wipe_time
    else:
        velocity = (0.0, 0.0, 0.0)
    entity.set_particles_pos(anchor_positions + offset, particles_idx_local=anchor_indices)
    entity.set_particles_vel(velocity, particles_idx_local=anchor_indices)
    return pos


def update_sweep_case(solver, time, settings):
    """Move the one-way box collider along the sweeping stroke."""
    pos = mop_pose(time, settings)
    solver.set_static_colliders_pose(pos=pos, quat=settings.quat, colliders_idx=settings.collider_idx)
    return pos


def update_sponge_squeeze_case(entity, anchor_positions, anchor_indices, time, settings):
    """Compress and release the anchored particle layer of a porous sponge."""
    squeeze_end = settings.absorb_time + settings.squeeze_time
    hold_end = squeeze_end + settings.hold_time
    release_end = hold_end + settings.release_time
    if time <= settings.absorb_time:
        compression = 0.0
        velocity_y = 0.0
    elif time < squeeze_end:
        compression = settings.compression_distance * (time - settings.absorb_time) / settings.squeeze_time
        velocity_y = -settings.compression_distance / settings.squeeze_time
    elif time <= hold_end:
        compression = settings.compression_distance
        velocity_y = 0.0
    elif time < release_end:
        compression = settings.compression_distance * (release_end - time) / settings.release_time
        velocity_y = settings.compression_distance / settings.release_time
    else:
        compression = 0.0
        velocity_y = 0.0
    offset = (0.0, -compression, 0.0)
    entity.set_particles_pos(anchor_positions + offset, particles_idx_local=anchor_indices)
    entity.set_particles_vel((0.0, velocity_y, 0.0), particles_idx_local=anchor_indices)
    return offset


def draw_case_colliders(scene, case):
    if case == CASE_CONE:
        cone = mesh_utils.create_cone(
            radius=5.0 * math.sqrt(3.0),
            height=5.0,
            sections=96,
            color=(0.35, 0.38, 0.42, 1.0),
        )
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = np.array(
            (
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, -1.0, 0.0),
            ),
            dtype=np.float32,
        )
        transform[:3, 3] = (0.0, -7.0, 0.0)
        scene.draw_debug_mesh(cone, T=transform)
        return None

    if case in (CASE_MOP, CASE_SPONGE_SQUEEZE, CASE_SWEEP):
        settings = case_settings(case)
        if settings.mop is not None:
            table_pos = settings.mop.table_pos
            table_size = settings.mop.table_size
            table_quat = settings.mop.quat
        elif settings.sponge is not None:
            table_pos = settings.sponge.table_pos
            table_size = settings.sponge.table_size
            table_quat = (1.0, 0.0, 0.0, 0.0)
        elif settings.sweep is not None:
            table_pos = settings.sweep.table_pos
            table_size = settings.sweep.table_size
            table_quat = settings.sweep.quat
        else:
            gs.raise_exception("The table cases require table settings.")
        table_mesh = mesh_utils.create_box(extents=table_size, color=(0.36, 0.24, 0.14, 1.0))
        table_transform = geom_utils.trans_quat_to_T(np.array(table_pos), np.array(table_quat))
        scene.draw_debug_mesh(table_mesh, T=table_transform)
        if settings.sweep is not None:
            sweep_mesh = mesh_utils.create_box(
                bounds=(settings.sweep.collider_lower, settings.sweep.collider_upper),
                color=(0.85, 0.2, 0.08, 0.35),
            )
            sweep_transform = geom_utils.trans_quat_to_T(
                np.array(mop_pose(0.0, settings.sweep)), np.array(settings.sweep.quat)
            )
            return scene.draw_debug_mesh(sweep_mesh, T=sweep_transform)
        return None

    return None


def start_viewer_recording(scene, is_recording):
    if is_recording:
        scene.viewer.toggle_recording()


def stop_viewer(scene, is_viewer_shown):
    if not is_viewer_shown:
        return
    if scene.viewer.recording:
        scene.viewer.toggle_recording()
    scene.viewer.stop()


def build_scene(case=CASE_CUBE, scale=None, show_viewer=False, dt=None, n_envs=0):
    """Build a position-based surface-tension flow example scene."""
    settings = case_settings(case)
    if scale is None:
        scale = settings.scale
    if scale <= 0:
        raise ValueError("PBSTF particle scale must be positive")
    if case == CASE_TEAPOT and scale < settings.scale:
        raise ValueError(
            f"The teapot case requires scale >= {settings.scale} to keep more than 50,000 initial particles"
        )

    if dt is None:
        dt = settings.dt
    if dt <= 0.0:
        raise ValueError("PBSTF time step must be positive")
    particle_size = 2.0 / scale
    static_colliders = list(settings.static_colliders)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=settings.gravity,
        ),
        rigid_options=(
            gs.options.RigidOptions(
                enable_collision=False,
                disable_constraint=True,
            )
            if case == CASE_TEAPOT
            else None
        ),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=particle_size,
            lower_bound=settings.lower_bound,
            upper_bound=settings.upper_bound,
            max_solver_iterations=settings.max_solver_iterations,
            topology_rebuild_interval=10,
            max_surface_neighbors=settings.max_surface_neighbors,
            max_localmesh_neighbors=settings.max_localmesh_neighbors,
            enable_pca_normals=settings.enable_pca_normals,
            static_colliders=static_colliders,
        ),
        viewer_options=gs.options.ViewerOptions(
            refresh_rate=round(1.0 / dt),
            camera_pos=settings.camera_pos,
            camera_lookat=settings.camera_lookat,
            camera_up=(0.0, 1.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    entities_and_velocities = add_case_entities(scene, case, particle_size, settings, _case_liquid_material)
    scene.build(n_envs=n_envs)
    for entity, velocity in entities_and_velocities:
        if velocity is not None:
            entity.set_particles_vel(velocity)
        if isinstance(entity.material, gs.materials.PBSTF.PorousElastic):
            entity.fix_particles(porous_top_particle_indices(entity))
    if settings.teapot is not None:
        initialize_teapot_manipulator(scene, settings.teapot)
    if show_viewer:
        draw_case_colliders(scene, case)

    return scene, tuple(entity for entity, _ in entities_and_velocities)


def main():
    parser = argparse.ArgumentParser(description="GPU Position-Based Surface Tension Flow examples")
    parser.add_argument("--case", choices=CASES, default=CASE_CUBE)
    parser.add_argument("--scale", type=int, default=None, help="C++ particle scale (radius = 1 / scale)")
    parser.add_argument("--dt", type=float, default=None, help="Override the case's default time step")
    parser.add_argument("--steps", type=int, default=None, help="Override the case's default simulation horizon")
    parser.add_argument("-v", "--vis", dest="show_viewer", action="store_true", default=False)
    parser.add_argument(
        "--record",
        dest="is_recording",
        action="store_true",
        default=False,
        help="Record the viewer and prompt for the output path on exit",
    )
    args = parser.parse_args()
    if args.scale is not None and args.scale <= 0:
        parser.error("--scale must be positive")
    if args.dt is not None and args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")

    gs.init(backend=gs.cuda, precision="32", logging_level="info")
    settings = case_settings(args.case)
    is_viewer_shown = args.show_viewer or args.is_recording
    scene, entities = build_scene(case=args.case, scale=args.scale, show_viewer=is_viewer_shown, dt=args.dt)
    teapot_settings = settings.teapot
    emitter_settings = settings.emitter
    mop_settings = settings.mop
    sponge_settings = settings.sponge
    sweep_settings = settings.sweep
    emitter = scene.emitters[0] if emitter_settings is not None else None
    teapot = scene.get_entity(name=teapot_settings.entity_name) if teapot_settings is not None else None
    porous = next(
        (entity for entity in entities if isinstance(entity.material, gs.materials.PBSTF.PorousElastic)),
        None,
    )
    if porous is None:
        porous_anchor_indices = None
        porous_anchor_positions = None
    else:
        porous_anchor_indices = porous_top_particle_indices(porous)
        porous_anchor_positions = tensor_to_array(porous.get_particles_pos())[..., porous_anchor_indices, :].copy()
    if sweep_settings is not None and is_viewer_shown:
        scene.clear_debug_objects()
        sweep_debug_object = draw_case_colliders(scene, args.case)
        sweep_debug_transform = geom_utils.trans_quat_to_T(np.zeros(3), np.array(sweep_settings.quat))
    else:
        sweep_debug_object = None
        sweep_debug_transform = None
    if teapot_settings is None:
        kuka = None
        kuka_qpos = None
    else:
        kuka = scene.get_entity(name=teapot_settings.manipulator.kuka_entity_name)
        kuka_qpos = kuka.get_qpos()

    steps = settings.steps if args.steps is None else args.steps
    if "PYTEST_VERSION" in os.environ:
        steps = min(steps, 2)
    start_viewer_recording(scene, args.is_recording)
    try:
        for step_idx in range(steps):
            if emitter is not None:
                emitter.emit(
                    droplet_shape="circle",
                    droplet_size=emitter_settings.droplet_size,
                    droplet_length=emitter.entity.particle_size if step_idx == 0 else None,
                    pos=emitter_settings.pos,
                    direction=emitter_settings.direction,
                    speed=emitter_settings.initial_speed,
                    generation_speed=emitter_settings.generation_speed,
                )
            if teapot is not None:
                kuka_qpos = update_teapot_case(
                    scene.pbstf_solver,
                    teapot,
                    kuka,
                    scene.cur_t,
                    teapot_settings,
                    kuka_qpos,
                )
            if mop_settings is not None:
                update_mop_case(porous, porous_anchor_positions, porous_anchor_indices, scene.cur_t, mop_settings)
            if sponge_settings is not None:
                update_sponge_squeeze_case(
                    porous,
                    porous_anchor_positions,
                    porous_anchor_indices,
                    scene.cur_t,
                    sponge_settings,
                )
            if sweep_settings is not None:
                sweep_pos = update_sweep_case(scene.pbstf_solver, scene.cur_t, sweep_settings)
                if sweep_debug_object is not None:
                    sweep_debug_transform[:3, 3] = sweep_pos
                    scene.update_debug_objects((sweep_debug_object,), (sweep_debug_transform,))
            scene.step()
    finally:
        stop_viewer(scene, is_viewer_shown)


if __name__ == "__main__":
    main()
