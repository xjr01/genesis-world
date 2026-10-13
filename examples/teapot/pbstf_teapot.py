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
    material = gs.materials.PBSTF.Liquid(
        sampler="regular",
        rho=1000.0,
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
    solver_options = gs.options.PBSTFOptions(
        particle_size=settings.particle_size,
        lower_bound=settings.lower_bound,
        upper_bound=settings.upper_bound,
        max_solver_iterations=5,
        topology_rebuild_interval=10,
        max_surface_neighbors=128,
        max_localmesh_neighbors=64,
        enable_pca_normals=False,
        static_colliders=[
            gs.options.PBSTFMeshStaticColliderOptions(
                file=settings.asset,
                scale=settings.mesh_scale,
                sdf_res=150,
                pos=settings.offset,
                quat=settings.quat,
            ),
        ],
    )
    return build_teapot_scene(settings, material, solver_options, show_viewer, n_envs, preview, spp, playback_speed)


def main():
    parser = argparse.ArgumentParser(description="PBSTF liquid in the shared moving teapot scene.")
    add_teapot_arguments(parser)
    args = parser.parse_args()
    checkpoint_info = prepare_teapot_run(args, method="PBSTF")
    gs.init(backend=gs.cuda, precision="32", logging_level="info")
    simulation = build_scene(
        scale=args.scale,
        show_viewer=args.vis,
        dt=args.dt,
        preview=args.preview,
        spp=args.spp,
        playback_speed=args.playback_speed,
    )
    run_teapot(simulation, args, method="PBSTF", checkpoint_info=checkpoint_info)


if __name__ == "__main__":
    main()
