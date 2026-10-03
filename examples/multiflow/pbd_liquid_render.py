import argparse
import os


import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument(
        "--raytrace",
        action="store_true",
        default=False,
        help="Render the reconstructed liquid with a refractive glass material.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=10,
        help="Update the reconstructed surface every N simulation steps.",
    )
    args = parser.parse_args()
    if args.render_every < 1:
        parser.error("--render-every must be at least 1.")

    ########################## init ##########################
    gs.init(precision="32", logging_level="info")

    if args.raytrace:
        renderer = gs.renderers.RayTracer(
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path="textures/indoor_bright.png",
                ),
            ),
            env_radius=15.0,
            env_euler=(0.0, 0.0, 180.0),
            lights=[
                {"pos": (0.5, 0.5, 4.0), "radius": 1.0, "color": (10.0, 10.0, 10.0)},
            ],
        )
        liquid_surface = gs.surfaces.Glass(
            color=(0.92, 0.97, 1.0),
            ior=1.333,
            roughness=0.02,
            vis_mode="recon",
            recon_backend="splashsurf",
        )
    else:
        renderer = gs.renderers.Rasterizer()
        liquid_surface = gs.surfaces.Default(
            color=(0.08, 0.35, 0.75, 0.65),
            roughness=0.08,
            vis_mode="recon",
            recon_backend="splashsurf",
        )

    ########################## scene ##########################
    camera_pos = (2.2, 1.8, 1.5)
    camera_lookat = (0.4, 0.3, 0.2)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
        ),
        pbd_options=gs.options.PBDOptions(
            lower_bound=(0.0, 0.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
            max_density_solver_iterations=10,
            max_viscosity_solver_iterations=1,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            ambient_light=(0.2, 0.2, 0.2),
            plane_reflection=True,
        ),
        renderer=renderer,
        # The interactive viewer always uses the rasterizer. In ray-tracing mode,
        # the camera GUI below displays the ray-traced frames instead.
        show_viewer=args.vis and not args.raytrace,
    )

    ########################## entities ##########################
    scene.add_entity(
        morph=gs.morphs.Plane(
            plane_size=(2.0, 2.0),
        ),
        surface=gs.surfaces.Default(
            color=(0.18, 0.2, 0.24, 1.0),
            roughness=0.65,
        ),
    )
    scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular",
            rho=1.0,
            density_relaxation=1.0,
            viscosity_relaxation=0.0,
        ),
        morph=gs.morphs.Box(lower=(0.2, 0.1, 0.1), upper=(0.6, 0.5, 0.5)),
        surface=liquid_surface,
    )

    camera = None
    if args.raytrace:
        camera = scene.add_camera(
            res=(960, 540),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=40,
            GUI=args.vis,
            spp=64,
            denoise=True,
        )

    scene.build(n_envs=0)

    horizon = 4000 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        should_render = args.vis and i % args.render_every == 0
        scene.step(update_visualizer=should_render and not args.raytrace)
        if camera is not None and should_render:
            camera.render()


if __name__ == "__main__":
    main()
