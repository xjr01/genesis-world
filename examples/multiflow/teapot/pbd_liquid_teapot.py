import argparse
import os

import genesis as gs

from pbstf_surface_tension import (
    CASE_TEAPOT,
    case_settings,
    sample_teapot_particles,
    start_viewer_recording,
    stop_viewer,
    update_rigid_teapot,
)


def build_scene(scale=None, show_viewer=False, dt=None):
    settings = case_settings(CASE_TEAPOT)
    teapot_settings = settings.teapot
    if teapot_settings is None:
        gs.raise_exception("The teapot case requires teapot settings.")
    if scale is None:
        scale = settings.scale
    if scale < settings.scale:
        raise ValueError(f"The teapot case requires scale >= {settings.scale}.")
    if dt is None:
        dt = settings.dt
    if dt <= 0.0:
        raise ValueError("The Position-Based Dynamics (PBD) time step must be positive.")

    particle_size = 2.0 / scale
    particles = sample_teapot_particles(teapot_settings, particle_size)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=settings.gravity,
        ),
        pbd_options=gs.options.PBDOptions(
            max_density_solver_iterations=10,
            max_viscosity_solver_iterations=1,
            particle_size=particle_size,
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
    teapot = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=teapot_settings.asset,
            scale=teapot_settings.mesh_scale,
            pos=teapot_settings.offset,
            quat=teapot_settings.quat,
            decimate=False,
            watertighten=None,
            convexify=False,
            align=False,
            fixed=True,
        ),
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=0.01,
            coup_restitution=0.0,
            sdf_min_res=150,
            sdf_max_res=150,
        ),
        surface=gs.surfaces.Default(
            color=(0.66, 0.66, 0.66),
            opacity=0.3,
        ),
        name=teapot_settings.entity_name,
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Particles(
            positions=particles,
        ),
        material=gs.materials.PBD.Liquid(
            sampler="regular",
            rho=1.0,
            density_relaxation=1.0,
            viscosity_relaxation=0.0,
        ),
        surface=gs.surfaces.Default(
            color=(0.25, 0.55, 0.95),
        ),
    )
    scene.build(n_envs=0)
    liquid.set_particles_vel(teapot_settings.particles_vel)
    return scene, teapot, liquid, teapot_settings


def main():
    parser = argparse.ArgumentParser(description="PBD liquid in the moving teapot reference case")
    parser.add_argument("--scale", type=int, default=None, help="Particle scale (diameter = 2 / scale)")
    parser.add_argument("--dt", type=float, default=None, help="Override the PBD time step")
    parser.add_argument("--steps", type=int, default=None, help="Override the reference simulation horizon")
    parser.add_argument("-v", "--vis", dest="show_viewer", action="store_true", default=False)
    parser.add_argument(
        "--record",
        dest="is_recording",
        action="store_true",
        default=False,
        help="Record the viewer and prompt for the output path on exit",
    )
    args = parser.parse_args()
    settings = case_settings(CASE_TEAPOT)
    if args.scale is not None and args.scale < settings.scale:
        parser.error(f"--scale must be at least {settings.scale}")
    if args.dt is not None and args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")

    gs.init(backend=gs.cuda, precision="32", logging_level="info")
    is_viewer_shown = args.show_viewer or args.is_recording
    scene, teapot, _, teapot_settings = build_scene(
        scale=args.scale,
        show_viewer=is_viewer_shown,
        dt=args.dt,
    )
    steps = settings.steps if args.steps is None else args.steps
    if "PYTEST_VERSION" in os.environ:
        steps = min(steps, 2)

    start_viewer_recording(scene, args.is_recording)
    try:
        for _ in range(steps):
            update_rigid_teapot(teapot, scene.cur_t, teapot_settings)
            scene.step()
    finally:
        stop_viewer(scene, is_viewer_shown)


if __name__ == "__main__":
    main()
