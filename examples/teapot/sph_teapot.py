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


def build_scene(
    pressure_solver="DFSPH",
    scale=None,
    show_viewer=False,
    dt=None,
    n_envs=0,
    preview="particles",
    spp=32,
    playback_speed=1.0,
):
    settings = create_teapot_settings(scale, dt)
    if pressure_solver not in ("DFSPH", "WCSPH"):
        raise ValueError(f"Unknown pressure solver: {pressure_solver}")
    material = gs.materials.SPH.Liquid(
        rho=1000.0,
        sampler="regular",
    )
    solver_options = gs.options.SPHOptions(
        particle_size=settings.particle_size,
        pressure_solver=pressure_solver,
        lower_bound=settings.lower_bound,
        upper_bound=settings.upper_bound,
    )
    return build_teapot_scene(settings, material, solver_options, show_viewer, n_envs, preview, spp, playback_speed)


def main():
    parser = argparse.ArgumentParser(description="SPH liquid in the shared moving teapot scene.")
    add_teapot_arguments(parser)
    parser.add_argument(
        "--pressure-solver",
        type=str.upper,
        choices=("DFSPH", "WCSPH"),
        default="DFSPH",
        help="Pressure solver used by the fluid.",
    )
    args = parser.parse_args()
    checkpoint_info = prepare_teapot_run(args, args.pressure_solver)
    gs.init(backend=gs.cuda, precision="32", logging_level="info")
    simulation = build_scene(
        pressure_solver=args.pressure_solver,
        scale=args.scale,
        show_viewer=args.vis,
        dt=args.dt,
        preview=args.preview,
        spp=args.spp,
        playback_speed=args.playback_speed,
    )
    run_teapot(simulation, args, args.pressure_solver, checkpoint_info)


if __name__ == "__main__":
    main()
