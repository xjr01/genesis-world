import argparse
import math
import os
from typing import NamedTuple

import numpy as np

import genesis as gs
import genesis.utils.geom as geom_utils
import genesis.utils.mesh as mesh_utils

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
    "CASE_SWEEP",
    "CASE_TAP",
    "CASE_TEAPOT",
    "TeapotSettings",
    "WipeSettings",
    "add_case_entities",
    "build_scene",
    "case_settings",
    "draw_case_colliders",
    "get_wipe_settings",
    "sample_teapot_particles",
    "start_viewer_recording",
    "stop_viewer",
    "teapot_pose",
    "update_rigid_teapot",
    "update_teapot_manipulator",
    "update_wipe_case",
    "wipe_pose",
)

CASE_CUBE = "cube"
CASE_MERGE = "merge"
CASE_BOUNCE = "bounce"
CASE_CONE = "cone"
CASE_MOP = "mop"
CASE_SWEEP = "sweep"
CASE_TAP = "tap"
CASE_TEAPOT = "teapot"
CASES = (CASE_CUBE, CASE_MERGE, CASE_BOUNCE, CASE_CONE, CASE_MOP, CASE_SWEEP, CASE_TAP, CASE_TEAPOT)

DEFAULT_CASE_DT = 1.0 / 30.0


class TapEmitterSettings(NamedTuple):
    pos: tuple[float, float, float]
    direction: tuple[float, float, float]
    droplet_size: float
    generation_speed: float
    initial_speed: float
    max_particles: int


class WipeSettings(NamedTuple):
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
    mop: WipeSettings | None = None
    sweep: WipeSettings | None = None


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
        wipe = WipeSettings(
            collider_idx=1,
            collider_lower=(-0.6, 0.02, -1.2),
            collider_upper=(0.6, 0.85, 1.2),
            table_pos=(0.0, -0.25, 0.0),
            table_size=(12.0, 0.5, 8.0),
            liquid_lower=(-2.5, 0.05, -0.7),
            liquid_upper=(-0.5, 0.35, 0.7),
            start_pos=(-3.5, 0.0, 0.0),
            end_pos=(3.5, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            settle_time=1.0,
            wipe_time=5.0,
        )
        if case == CASE_MOP:
            wipe_collider = gs.options.PBSTFAbsorbentBoxStaticColliderOptions(
                pos=wipe.start_pos,
                quat=wipe.quat,
                lower=wipe.collider_lower,
                upper=wipe.collider_upper,
                absorption_rate=800.0,
                absorption_capacity_fraction=1.0,
            )
        else:
            wipe_collider = gs.options.PBSTFBoxStaticColliderOptions(
                pos=wipe.start_pos,
                quat=wipe.quat,
                lower=wipe.collider_lower,
                upper=wipe.collider_upper,
            )
        return CaseSettings(
            scale=20,
            dt=0.01,
            gravity=(0.0, -9.8, 0.0),
            lower_bound=(-6.0, -1.0, -4.0),
            upper_bound=(6.0, 4.0, 4.0),
            camera_pos=(8.0, 6.0, 9.0),
            camera_lookat=(0.0, 0.4, 0.0),
            static_colliders=(
                gs.options.PBSTFBoxStaticColliderOptions(
                    pos=wipe.table_pos,
                    quat=wipe.quat,
                    lower=tuple(-0.5 * size for size in wipe.table_size),
                    upper=tuple(0.5 * size for size in wipe.table_size),
                ),
                wipe_collider,
            ),
            max_solver_iterations=10,
            max_surface_neighbors=128,
            max_localmesh_neighbors=64,
            enable_pca_normals=False,
            steps=1000,
            teapot=None,
            emitter=None,
            mop=wipe if case == CASE_MOP else None,
            sweep=wipe if case == CASE_SWEEP else None,
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


def get_wipe_settings(settings):
    """Return the active MOP or SWEEP trajectory settings for one case."""
    if settings.mop is not None:
        return settings.mop
    if settings.sweep is not None:
        return settings.sweep
    gs.raise_exception("The selected case requires wipe settings.")


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

    if case in (CASE_MOP, CASE_SWEEP):
        wipe = get_wipe_settings(settings)
        liquid = scene.add_entity(
            morph=gs.morphs.Box(
                lower=wipe.liquid_lower,
                upper=wipe.liquid_upper,
            ),
            material=material_factory(case),
        )
        return ((liquid, None),)

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


def wipe_pose(time, settings):
    """Return the collider translation along its single wiping stroke."""
    progress = min(max((time - settings.settle_time) / settings.wipe_time, 0.0), 1.0)
    return tuple(
        settings.start_pos[axis] + progress * (settings.end_pos[axis] - settings.start_pos[axis]) for axis in range(3)
    )


def update_wipe_case(solver, time, settings):
    """Move the active MOP or SWEEP collider and return its current position."""
    pos = wipe_pose(time, settings)
    solver.set_static_colliders_pose(pos=pos, quat=settings.quat, colliders_idx=settings.collider_idx)
    return pos


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

    if case in (CASE_MOP, CASE_SWEEP):
        wipe = get_wipe_settings(case_settings(case))
        table_mesh = mesh_utils.create_box(extents=wipe.table_size, color=(0.36, 0.24, 0.14, 1.0))
        table_transform = geom_utils.trans_quat_to_T(np.array(wipe.table_pos), np.array(wipe.quat))
        scene.draw_debug_mesh(table_mesh, T=table_transform)
        wipe_mesh = mesh_utils.create_box(
            bounds=(wipe.collider_lower, wipe.collider_upper),
            color=(0.85, 0.2, 0.08, 0.35),
        )
        wipe_transform = geom_utils.trans_quat_to_T(np.array(wipe_pose(0.0, wipe)), np.array(wipe.quat))
        return scene.draw_debug_mesh(wipe_mesh, T=wipe_transform)

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


def build_scene(case=CASE_CUBE, scale=None, show_viewer=False, dt=None):
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
    scene.build()
    for entity, velocity in entities_and_velocities:
        if velocity is not None:
            entity.set_particles_vel(velocity)
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
    scene, _ = build_scene(case=args.case, scale=args.scale, show_viewer=is_viewer_shown, dt=args.dt)
    teapot_settings = settings.teapot
    emitter_settings = settings.emitter
    wipe_settings = get_wipe_settings(settings) if args.case in (CASE_MOP, CASE_SWEEP) else None
    emitter = scene.emitters[0] if emitter_settings is not None else None
    teapot = scene.get_entity(name=teapot_settings.entity_name) if teapot_settings is not None else None
    if wipe_settings is not None and is_viewer_shown:
        scene.clear_debug_objects()
        wipe_debug_object = draw_case_colliders(scene, args.case)
        wipe_debug_transform = geom_utils.trans_quat_to_T(np.zeros(3), np.array(wipe_settings.quat))
    else:
        wipe_debug_object = None
        wipe_debug_transform = None
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
            if wipe_settings is not None:
                wipe_pos = update_wipe_case(scene.pbstf_solver, scene.cur_t, wipe_settings)
                if wipe_debug_object is not None:
                    wipe_debug_transform[:3, 3] = wipe_pos
                    scene.update_debug_objects((wipe_debug_object,), (wipe_debug_transform,))
            scene.step()
    finally:
        stop_viewer(scene, is_viewer_shown)


if __name__ == "__main__":
    main()
