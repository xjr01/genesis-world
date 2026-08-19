import argparse
import os

import genesis as gs

if __package__:
    from .teapot.fluid_helper import initialize_teapot_manipulator, update_teapot_case
    from .teapot.pbstf_surface_tension import (
        CASE_BOUNCE,
        CASE_CONE,
        CASE_CUBE,
        CASE_MERGE,
        CASE_TAP,
        CASE_TEAPOT,
        CASES as PBSTF_CASES,
        add_case_entities,
        case_settings,
        draw_case_colliders,
        start_viewer_recording,
        stop_viewer,
    )
else:
    from teapot.fluid_helper import initialize_teapot_manipulator, update_teapot_case
    from teapot.pbstf_surface_tension import (
        CASE_BOUNCE,
        CASE_CONE,
        CASE_CUBE,
        CASE_MERGE,
        CASE_TAP,
        CASE_TEAPOT,
        CASES as PBSTF_CASES,
        add_case_entities,
        case_settings,
        draw_case_colliders,
        start_viewer_recording,
        stop_viewer,
    )

CASE_DAM_BREAK = "dam_break"
CASES = (*PBSTF_CASES, CASE_DAM_BREAK)


def _liquid_material(case):
    sampler = "regular" if case in (CASE_TAP, CASE_TEAPOT) else "staggered"
    return gs.materials.IPBSTF.Liquid(sampler=sampler)


def build_scene(
    case=CASE_CUBE,
    scale=None,
    show_viewer=False,
    dt=None,
    alpha=1e-6,
    max_solver_iterations=10,
    show_boundary=True,
):
    """Build an implicit position-based fluid example scene."""
    if case == CASE_DAM_BREAK:
        if scale is None:
            scale = 20
        if scale <= 0:
            raise ValueError("IPBSTF particle scale must be positive")
        if dt is None:
            dt = 0.005
        if dt <= 0.0:
            raise ValueError("IPBSTF time step must be positive")

        particle_size = 2.0 / scale
        boundary_thickness = 2.0 * particle_size
        # The padded hash-grid domain keeps its projection boundary beyond the solid particle layers.
        domain_padding = 3.0 * boundary_thickness
        # Each solid particle layer is paired with an analytic box collider of identical extent so the boundary
        # combines the frozen-fluid density response with a strict project-out.
        solid_bounds = (
            (
                (-2.0 - boundary_thickness, -boundary_thickness, -2.0 - boundary_thickness),
                (2.0 + boundary_thickness, 0.0, 2.0 + boundary_thickness),
            ),
            (
                (-2.0 - boundary_thickness, 0.0, -2.0 - boundary_thickness),
                (-2.0, 4.0, 2.0 + boundary_thickness),
            ),
            (
                (2.0, 0.0, -2.0 - boundary_thickness),
                (2.0 + boundary_thickness, 4.0, 2.0 + boundary_thickness),
            ),
            (
                (-2.0, 0.0, -2.0 - boundary_thickness),
                (2.0, 4.0, -2.0),
            ),
            (
                (-2.0, 0.0, 2.0),
                (2.0, 4.0, 2.0 + boundary_thickness),
            ),
            (
                (-2.0 - boundary_thickness, 4.0, -2.0 - boundary_thickness),
                (2.0 + boundary_thickness, 4.0 + boundary_thickness, 2.0 + boundary_thickness),
            ),
        )
        boundary_colliders = [
            gs.options.PBSTFBoxStaticColliderOptions(
                lower=lower,
                upper=upper,
                is_density_blocking=False,
            )
            for lower, upper in solid_bounds
        ]
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=dt,
                gravity=(0.0, -9.81, 0.0),
            ),
            ipbstf_options=gs.options.IPBSTFOptions(
                alpha=alpha,
                particle_size=particle_size,
                max_solver_iterations=max_solver_iterations,
                static_colliders=boundary_colliders,
                lower_bound=(
                    -2.0 - domain_padding,
                    -domain_padding,
                    -2.0 - domain_padding,
                ),
                upper_bound=(
                    2.0 + domain_padding,
                    4.0 + domain_padding,
                    2.0 + domain_padding,
                ),
            ),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=round(1.0 / dt),
                camera_pos=(3.2, 2.2, 4.0),
                camera_lookat=(0.4, 0.6, 0.0),
                camera_up=(0.0, 1.0, 0.0),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )
        liquid = scene.add_entity(
            morph=gs.morphs.Box(
                lower=(-1.5, 0.1, -1.5),
                upper=(0.0, 2.0, 1.5),
            ),
            material=_liquid_material(CASE_DAM_BREAK),
        )
        solid_material = gs.materials.IPBSTF.Solid(
            sampler="staggered",
        )
        solid_surface = gs.surfaces.Default(
            color=(0.7, 0.75, 0.8, 0.35 if show_boundary else 0.0),
            vis_mode="particle",
        )
        for lower, upper in solid_bounds:
            scene.add_entity(
                morph=gs.morphs.Box(
                    lower=lower,
                    upper=upper,
                ),
                material=solid_material,
                surface=solid_surface,
            )
        scene.build()
        return scene, (liquid,)

    settings = case_settings(case)
    if scale is None:
        scale = settings.scale
    if scale <= 0:
        raise ValueError("IPBSTF particle scale must be positive")
    if case == CASE_TEAPOT and scale < settings.scale:
        raise ValueError(
            f"The teapot case requires scale >= {settings.scale} to keep more than 50,000 initial particles"
        )
    if dt is None:
        dt = settings.dt
    if dt <= 0.0:
        raise ValueError("IPBSTF time step must be positive")

    particle_size = 2.0 / scale
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
        ipbstf_options=gs.options.IPBSTFOptions(
            alpha=alpha,
            particle_size=particle_size,
            max_solver_iterations=max_solver_iterations,
            static_colliders=list(settings.static_colliders),
            lower_bound=settings.lower_bound,
            upper_bound=settings.upper_bound,
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
    entities_and_velocities = add_case_entities(scene, case, particle_size, settings, _liquid_material)
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
    parser = argparse.ArgumentParser(description="Implicit Position-Based Fluid examples")
    parser.add_argument("--case", choices=CASES, default=CASE_DAM_BREAK)
    parser.add_argument("--scale", type=int, default=None, help="Particle scale (radius = 1 / scale)")
    parser.add_argument("--dt", type=float, default=None, help="Override the case's default time step")
    parser.add_argument("--steps", type=int, default=None, help="Override the case's default simulation horizon")
    parser.add_argument("--alpha", type=float, default=1e-6, help="Inertial energy weight")
    parser.add_argument("--iterations", type=int, default=10, help="Local Newton iterations per step")
    parser.add_argument("-v", "--vis", dest="show_viewer", action="store_true", default=False)
    parser.add_argument(
        "--record",
        dest="is_recording",
        action="store_true",
        default=False,
        help="Record the viewer and prompt for the output path on exit",
    )
    parser.add_argument(
        "--show-boundary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Semi-transparently display the solid boundary particles; the box colliders are never shown",
    )
    args = parser.parse_args()
    if args.scale is not None and args.scale <= 0:
        parser.error("--scale must be positive")
    if args.dt is not None and args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.alpha <= 0.0:
        parser.error("--alpha must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    gs.init(backend=gs.cuda, precision="64", logging_level="info")
    scene, _ = build_scene(
        case=args.case,
        scale=args.scale,
        show_viewer=args.show_viewer or args.is_recording,
        dt=args.dt,
        alpha=args.alpha,
        max_solver_iterations=args.iterations,
        show_boundary=args.show_boundary,
    )
    if args.case == CASE_DAM_BREAK:
        settings = None
        steps = 240 if args.steps is None else args.steps
    else:
        settings = case_settings(args.case)
        steps = settings.steps if args.steps is None else args.steps
    if "PYTEST_VERSION" in os.environ:
        steps = min(steps, 2)

    emitter_settings = settings.emitter if settings is not None else None
    teapot_settings = settings.teapot if settings is not None else None
    emitter = scene.emitters[0] if emitter_settings is not None else None
    teapot = scene.get_entity(name=teapot_settings.entity_name) if teapot_settings is not None else None
    if teapot_settings is None:
        kuka = None
        kuka_qpos = None
    else:
        kuka = scene.get_entity(name=teapot_settings.manipulator.kuka_entity_name)
        kuka_qpos = kuka.get_qpos()

    is_viewer_shown = args.show_viewer or args.is_recording
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
                    scene.ipbstf_solver,
                    teapot,
                    kuka,
                    scene.cur_t,
                    teapot_settings,
                    kuka_qpos,
                )
            scene.step()
            # while True: pass
    finally:
        stop_viewer(scene, is_viewer_shown)


if __name__ == "__main__":
    main()
