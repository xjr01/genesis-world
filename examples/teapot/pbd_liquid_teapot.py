import argparse

import genesis as gs

if __package__:
    from .fluid_helper import create_teapot_settings
    from .teapot_checkpoint import prepare_teapot_run
    from .teapot_render import add_teapot_arguments, build_teapot_scene, run_teapot
else:
    from fluid_helper import create_teapot_settings
    from teapot_checkpoint import prepare_teapot_run
    from teapot_render import add_teapot_arguments, build_teapot_scene, run_teapot


def build_scene(scale=None, show_viewer=False, dt=None, n_envs=0, preview="particles", spp=32, playback_speed=1.0):
    settings = create_teapot_settings(scale, dt)
    material = gs.materials.PBD.Liquid(
        sampler="regular",
        rho=1.0,
        density_relaxation=1.0,
        viscosity_relaxation=0.0,
    )
    solver_options = gs.options.PBDOptions(
        max_density_solver_iterations=10,
        max_viscosity_solver_iterations=1,
        particle_size=settings.particle_size,
        lower_bound=settings.lower_bound,
        upper_bound=settings.upper_bound,
    )
    return build_teapot_scene(settings, material, solver_options, show_viewer, n_envs, preview, spp, playback_speed)


def main():
    parser = argparse.ArgumentParser(description="PBD liquid in the shared moving teapot scene.")
    add_teapot_arguments(parser)
    args = parser.parse_args()
    checkpoint_info = prepare_teapot_run(args, method="PBD")
    gs.init(backend=gs.cuda, precision="32", logging_level="info")
    simulation = build_scene(
        scale=args.scale,
        show_viewer=args.vis,
        dt=args.dt,
        preview=args.preview,
        spp=args.spp,
        playback_speed=args.playback_speed,
    )
    run_teapot(simulation, args, method="PBD", checkpoint_info=checkpoint_info)


if __name__ == "__main__":
    main()
