import argparse
import functools
import os

import genesis as gs
from genesis.utils import particle

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
CASE_DOUBLE_DAM_BREAK = "double_dam_break"
CASE_FREE_TRANSLATION = "free_translation"
CASES = (*PBSTF_CASES, CASE_DAM_BREAK, CASE_DOUBLE_DAM_BREAK, CASE_FREE_TRANSLATION)


def _liquid_material(case, kinetic_smoothing=0.2):
    sampler = "regular" if case in (CASE_TAP, CASE_TEAPOT) else "staggered"
    return gs.materials.IPBSTF.Liquid(
        sampler=sampler,
        kinetic_smoothing=kinetic_smoothing,
    )


def build_scene(
    case=CASE_CUBE,
    scale=None,
    show_viewer=False,
    dt=None,
    alpha=0.0,
    is_damping_enabled=True,
    damping_alpha=1e-3,
    density_update_fraction=0.25,
    surface_update_scale=0.002,
    kinetic_smoothing=0.2,
    support_radius_scale=2.0,
    max_solver_iterations=4,
):
    """Build an implicit position-based fluid example scene."""
    if case in (CASE_DAM_BREAK, CASE_DOUBLE_DAM_BREAK, CASE_FREE_TRANSLATION):
        if scale is None:
            scale = 48 if case == CASE_DOUBLE_DAM_BREAK else 20
        if scale <= 0:
            raise ValueError("IPBSTF particle scale must be positive")
        if dt is None:
            dt = 1.0 / 240.0 if case == CASE_DOUBLE_DAM_BREAK else 0.005
        if dt <= 0.0:
            raise ValueError("IPBSTF time step must be positive")
        if density_update_fraction <= 0.0 or density_update_fraction > 1.0:
            raise ValueError("IPBSTF density update fraction must lie in (0, 1]")
        if surface_update_scale <= 0.0:
            raise ValueError("IPBSTF surface update scale must be positive")
        if kinetic_smoothing < 0.0 or kinetic_smoothing > 1.0:
            raise ValueError("IPBSTF kinetic smoothing must lie in [0, 1]")
        if support_radius_scale <= 0.0:
            raise ValueError("IPBSTF support radius scale must be positive")

        particle_size = 2.0 / scale
        particle_radius = 0.5 * particle_size
        support_radius = support_radius_scale * particle_size
        boundary_thickness = 2.0 * particle_size
        is_free_translation = case == CASE_FREE_TRANSLATION
        bottom_y = -10.0 if is_free_translation else 0.0
        container_height = 5.8 if case == CASE_DOUBLE_DAM_BREAK else 4.0
        container_half_depth = 0.5 if case == CASE_DOUBLE_DAM_BREAK else 2.0
        liquid_half_depth = container_half_depth - particle_size if case == CASE_DOUBLE_DAM_BREAK else 1.5
        camera_pos = (0.0, 2.9, 9.0) if case == CASE_DOUBLE_DAM_BREAK else (3.2, 2.2, 4.0)
        camera_lookat = (0.0, 2.9, 0.0) if case == CASE_DOUBLE_DAM_BREAK else (0.4, 0.6, 0.0)
        # The padded sampling domain contains the solid particle layers outside the collision bounds.
        domain_padding = 3.0 * boundary_thickness
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=dt,
                gravity=(0.0, 0.0, 0.0) if is_free_translation else (0.0, -9.81, 0.0),
            ),
            ipbstf_options=gs.options.IPBSTFOptions(
                alpha=alpha,
                is_damping_enabled=is_damping_enabled,
                damping_alpha=damping_alpha,
                density_update_fraction=density_update_fraction,
                surface_update_scale=surface_update_scale,
                particle_size=particle_size,
                support_radius=support_radius,
                max_solver_iterations=max_solver_iterations,
                lower_bound=(
                    -2.0 - domain_padding,
                    bottom_y - domain_padding,
                    -container_half_depth - domain_padding,
                ),
                upper_bound=(
                    2.0 + domain_padding,
                    container_height + domain_padding,
                    container_half_depth + domain_padding,
                ),
                collision_lower_bound=(
                    -2.0 + particle_radius,
                    bottom_y + particle_radius,
                    -container_half_depth + particle_radius,
                ),
                collision_upper_bound=(
                    2.0 - particle_radius,
                    container_height - particle_radius,
                    container_half_depth - particle_radius,
                ),
            ),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=round(1.0 / dt),
                camera_pos=camera_pos,
                camera_lookat=camera_lookat,
                camera_up=(0.0, 1.0, 0.0),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )
        if case in (CASE_DAM_BREAK, CASE_FREE_TRANSLATION):
            liquid_bounds = (((-1.5, 0.1, -liquid_half_depth), (0.0, 2.0, liquid_half_depth)),)
        else:
            liquid_bounds = (
                ((-2.0 + particle_radius, 0.1, -liquid_half_depth), (-2.0 / 3.0, 5.3, liquid_half_depth)),
                ((2.0 / 3.0, 0.1, -liquid_half_depth), (2.0 - particle_radius, 5.3, liquid_half_depth)),
            )
        entities = []
        if case == CASE_DOUBLE_DAM_BREAK:
            lattice = particle.box_to_particles(
                p_size=particle_size,
                pos=(0.0, 0.5 * (bottom_y - boundary_thickness + container_height + boundary_thickness), 0.0),
                size=(
                    4.0 + 2.0 * boundary_thickness,
                    container_height - bottom_y + 2.0 * boundary_thickness,
                    2.0 * (container_half_depth + boundary_thickness),
                ),
                sampler="staggered",
            )
            lattice += (
                -2.0 - boundary_thickness,
                bottom_y - boundary_thickness,
                -container_half_depth - boundary_thickness,
            ) - lattice.min(axis=0)
            for lower, upper in liquid_bounds:
                is_liquid = (
                    (lattice[:, 0] >= lower[0])
                    & (lattice[:, 0] <= upper[0])
                    & (lattice[:, 1] >= lower[1])
                    & (lattice[:, 1] <= upper[1])
                    & (lattice[:, 2] >= lower[2])
                    & (lattice[:, 2] <= upper[2])
                )
                entities.append(
                    scene.add_entity(
                        morph=gs.morphs.Particles(
                            positions=lattice[is_liquid],
                        ),
                        material=_liquid_material(case, kinetic_smoothing),
                    )
                )
        else:
            for lower, upper in liquid_bounds:
                entities.append(
                    scene.add_entity(
                        morph=gs.morphs.Box(
                            lower=lower,
                            upper=upper,
                        ),
                        material=_liquid_material(case, kinetic_smoothing),
                    )
                )
        solid_material = gs.materials.IPBSTF.Solid(
            sampler="staggered",
        )
        solid_surface = gs.surfaces.Default(
            color=(0.7, 0.75, 0.8),
            opacity=0.0,
            vis_mode="particle",
        )
        solid_bounds = (
            (
                (
                    -2.0 - boundary_thickness,
                    bottom_y - boundary_thickness,
                    -container_half_depth - boundary_thickness,
                ),
                (2.0 + boundary_thickness, bottom_y, container_half_depth + boundary_thickness),
            ),
            (
                (-2.0 - boundary_thickness, 0.0, -container_half_depth - boundary_thickness),
                (-2.0, container_height, container_half_depth + boundary_thickness),
            ),
            (
                (2.0, 0.0, -container_half_depth - boundary_thickness),
                (2.0 + boundary_thickness, container_height, container_half_depth + boundary_thickness),
            ),
            (
                (-2.0, 0.0, -container_half_depth - boundary_thickness),
                (2.0, container_height, -container_half_depth),
            ),
            (
                (-2.0, 0.0, container_half_depth),
                (2.0, container_height, container_half_depth + boundary_thickness),
            ),
        )
        if case == CASE_DOUBLE_DAM_BREAK:
            is_solid = (
                (lattice[:, 0] <= -2.0)
                | (lattice[:, 0] >= 2.0)
                | (lattice[:, 1] <= bottom_y)
                | (lattice[:, 1] >= container_height)
                | (lattice[:, 2] <= -container_half_depth)
                | (lattice[:, 2] >= container_half_depth)
            )
            scene.add_entity(
                morph=gs.morphs.Particles(
                    positions=lattice[is_solid],
                ),
                material=solid_material,
                surface=solid_surface,
            )
        else:
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
        if is_free_translation:
            for entity in entities:
                entity.set_particles_vel((0.0, -9.81 * dt, 0.0))
        return scene, tuple(entities)

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
    if density_update_fraction <= 0.0 or density_update_fraction > 1.0:
        raise ValueError("IPBSTF density update fraction must lie in (0, 1]")
    if surface_update_scale <= 0.0:
        raise ValueError("IPBSTF surface update scale must be positive")
    if kinetic_smoothing < 0.0 or kinetic_smoothing > 1.0:
        raise ValueError("IPBSTF kinetic smoothing must lie in [0, 1]")
    if support_radius_scale <= 0.0:
        raise ValueError("IPBSTF support radius scale must be positive")

    particle_size = 2.0 / scale
    support_radius = support_radius_scale * particle_size
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
            is_damping_enabled=is_damping_enabled,
            damping_alpha=damping_alpha,
            density_update_fraction=density_update_fraction,
            surface_update_scale=surface_update_scale,
            particle_size=particle_size,
            support_radius=support_radius,
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
    material_factory = functools.partial(_liquid_material, kinetic_smoothing=kinetic_smoothing)
    entities_and_velocities = add_case_entities(scene, case, particle_size, settings, material_factory)
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
    parser.add_argument("--alpha", type=float, default=0.0, help="Inertial energy weight")
    parser.add_argument("--damping-alpha", type=float, default=1e-3, help="Reference compliance used by damping")
    parser.add_argument(
        "--density-update-fraction",
        type=float,
        default=0.25,
        help="Maximum local compression fraction removed by one parallel update",
    )
    parser.add_argument(
        "--surface-update-scale",
        type=float,
        default=0.002,
        help="Surface correction limit divided by the cubic-spline support radius",
    )
    parser.add_argument(
        "--support-radius-scale",
        type=float,
        default=2.0,
        help="Cubic-spline support radius divided by particle size",
    )
    parser.add_argument(
        "--kinetic-smoothing",
        type=float,
        default=0.2,
        help="Energy-preserving neighborhood velocity smoothing",
    )
    parser.add_argument("--iterations", type=int, default=4, help="Local Newton iterations per step")
    parser.add_argument(
        "--no-damping",
        dest="is_damping_enabled",
        action="store_false",
        default=True,
        help="Disable the alternative-compliance kinetic-energy limiter",
    )
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
    if args.alpha < 0.0:
        parser.error("--alpha must be non-negative")
    if args.damping_alpha <= 0.0:
        parser.error("--damping-alpha must be positive")
    if args.density_update_fraction <= 0.0 or args.density_update_fraction > 1.0:
        parser.error("--density-update-fraction must lie in (0, 1]")
    if args.surface_update_scale <= 0.0:
        parser.error("--surface-update-scale must be positive")
    if args.support_radius_scale <= 0.0:
        parser.error("--support-radius-scale must be positive")
    if args.kinetic_smoothing < 0.0 or args.kinetic_smoothing > 1.0:
        parser.error("--kinetic-smoothing must lie in [0, 1]")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    gs.init(backend=gs.cuda, precision="64", logging_level="info")
    scene, _ = build_scene(
        case=args.case,
        scale=args.scale,
        show_viewer=args.show_viewer or args.is_recording,
        dt=args.dt,
        alpha=args.alpha,
        is_damping_enabled=args.is_damping_enabled,
        damping_alpha=args.damping_alpha,
        density_update_fraction=args.density_update_fraction,
        surface_update_scale=args.surface_update_scale,
        kinetic_smoothing=args.kinetic_smoothing,
        support_radius_scale=args.support_radius_scale,
        max_solver_iterations=args.iterations,
    )
    if args.case in (CASE_DAM_BREAK, CASE_DOUBLE_DAM_BREAK, CASE_FREE_TRANSLATION):
        settings = None
        steps = (360 if args.case == CASE_DOUBLE_DAM_BREAK else 240) if args.steps is None else args.steps
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
    finally:
        stop_viewer(scene, is_viewer_shown)


if __name__ == "__main__":
    main()
