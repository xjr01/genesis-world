import argparse
import os

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(backend=gs.cuda, precision="32", logging_level="info")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 30.0, gravity=(0.0, 0.0, 0.0)),
        pbstf_options=gs.options.PBSTFOptions(
            # Match PositionBasedLiquidBuilder::buildCase0 (scale=10): the
            # C++ particle radius is 0.1, hence the diameter is 0.2 and the
            # cubic-spline kernel radius is 3 * 0.2 = 0.6.
            particle_size=0.1,
            lower_bound=(-2.0, -2.0, -2.0),
            upper_bound=(2.0, 2.0, 2.0),
            max_solver_iterations=100,
            topology_rebuild_interval=10,
            max_surface_neighbors=128,
            enable_pca_normals=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(4.0, 4.0, 3.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(
        material=gs.materials.PBSTF.Liquid(
            sampler="staggered",
            rho=1000.0,
            density_compliance=500.0,
            surface_tension_compliance=0.8,
            surface_distance_compliance=40.0,
            interior_distance_compliance=180.0,
            surface_viscosity=0.3,
            interior_viscosity=0.3,
        ),
        morph=gs.morphs.Box(lower=(-1.0, -1.0, -1.0), upper=(1.0, 1.0, 1.0)),
    )
    scene.build()

    horizon = 1500 if "PYTEST_VERSION" not in os.environ else 2
    for _ in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
