"""Compile and exercise DEM tilt-obstacle checkpoint fields in a minimal CPU scene."""

import numpy as np

import genesis as gs


def main():
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.8)),
        dem_options=gs.options.DEMOptions(
            particle_size=0.01,
            lower_bound=(-0.1, -0.1, 0.0),
            upper_bound=(0.1, 0.1, 0.2),
        ),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.04), size=(0.02, 0.02, 0.02)),
        material=gs.materials.DEM.Sand(sampler="fcc", rho=2.5),
    )
    scene.build()

    initial_pos = np.array((0.06, 0.0, 0.08), dtype=np.float32)
    initial_quat = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    initial_vel = np.array((0.01, 0.0, 0.0), dtype=np.float32)
    initial_omega = np.array((0.0, 0.2, 0.0), dtype=np.float32)
    scene.dem_solver.set_tilt_box_obstacle(
        half_extents=(0.005, 0.01, 0.01),
        pos=initial_pos,
        quat=initial_quat,
        vel=initial_vel,
    )
    scene.dem_solver.set_tilt_box_vel(initial_vel, initial_omega)
    checkpoint = scene.get_state()

    scene.dem_solver.set_tilt_box_vel((0.2, 0.0, 0.0), (0.0, 1.0, 0.0))
    scene.step()
    scene.restore(checkpoint)

    np.testing.assert_allclose(scene.dem_solver.get_tilt_box_pos()[0], initial_pos, atol=1.0e-6)
    np.testing.assert_allclose(scene.dem_solver.get_tilt_box_quat()[0], initial_quat, atol=1.0e-6)
    np.testing.assert_allclose(scene.dem_solver.get_tilt_box_vel()[0], initial_vel, atol=1.0e-6)
    np.testing.assert_allclose(scene.dem_solver.get_tilt_box_omega()[0], initial_omega, atol=1.0e-6)
    print("DEM tilt-obstacle checkpoint smoke passed")


if __name__ == "__main__":
    main()
