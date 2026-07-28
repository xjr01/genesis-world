import argparse
import os

import genesis as gs

CASE_CUBE = "cube"
CASE_MERGE = "merge"
CASE_BOUNCE = "bounce"
CASES = (CASE_CUBE, CASE_MERGE, CASE_BOUNCE)


def _liquid_material(**overrides):
    parameters = dict(
        sampler="staggered",
        rho=1000.0,
        density_compliance=500.0,
        surface_tension_compliance=0.8,
        surface_distance_compliance=40.0,
        interior_distance_compliance=180.0,
        surface_viscosity=0.3,
        interior_viscosity=0.3,
    )
    parameters.update(overrides)
    return gs.materials.PBSTF.Liquid(**parameters)


def _case_settings(case):
    if case == CASE_CUBE:
        return dict(
            gravity=(0.0, 0.0, 0.0),
            lower_bound=(-2.0, -2.0, -2.0),
            upper_bound=(2.0, 2.0, 2.0),
            camera_pos=(4.0, 4.0, 3.0),
            camera_lookat=(0.0, 0.0, 0.0),
            steps=1500,
        )
    if case == CASE_MERGE:
        return dict(
            gravity=(0.0, 0.0, 0.0),
            lower_bound=(-10.0, -10.0, -10.0),
            upper_bound=(10.0, 10.0, 10.0),
            camera_pos=(4.0, 3.0, 4.0),
            camera_lookat=(0.0, -1.0, 0.0),
            steps=300,
        )
    if case == CASE_BOUNCE:
        return dict(
            gravity=(0.0, 0.0, 0.0),
            # The C++ case uses an infinite plane at y=-2. The other five
            # AABB faces are kept far enough away to be inactive.
            lower_bound=(-10.0, -2.0, -10.0),
            upper_bound=(10.0, 10.0, 10.0),
            camera_pos=(4.0, 2.0, 4.0),
            camera_lookat=(0.0, -1.0, 0.0),
            steps=300,
        )
    raise ValueError(f"Unknown PBSTF example case: {case}")


def _add_case_entities(scene, case):
    if case == CASE_CUBE:
        liquid = scene.add_entity(
            material=_liquid_material(),
            morph=gs.morphs.Box(lower=(-1.0, -1.0, -1.0), upper=(1.0, 1.0, 1.0)),
        )
        return ((liquid, (0.0, 0.0, 0.0)),)

    if case == CASE_MERGE:
        material_overrides = dict(
            surface_tension_compliance=1.0,
            surface_viscosity=0.05,
            interior_viscosity=0.05,
        )
        left = scene.add_entity(
            material=_liquid_material(**material_overrides),
            morph=gs.morphs.Box(lower=(-1.5, -0.625, -0.5), upper=(-0.5, 0.375, 0.5)),
        )
        right = scene.add_entity(
            material=_liquid_material(**material_overrides),
            morph=gs.morphs.Box(lower=(0.5, -0.375, -0.5), upper=(1.5, 0.625, 0.5)),
        )
        return (
            (left, (1.0, 0.0, 0.0)),
            (right, (-1.0, 0.0, 0.0)),
        )

    if case == CASE_BOUNCE:
        liquid = scene.add_entity(
            material=_liquid_material(),
            morph=gs.morphs.Sphere(pos=(0.0, 0.0, 0.0), radius=1.0),
        )
        return ((liquid, (0.0, -3.0, 0.0)),)

    raise ValueError(f"Unknown PBSTF example case: {case}")


def build_scene(case=CASE_CUBE, scale=10, show_viewer=False):
    """Build one of the C++ PBSTF reference cases.

    The C++ ``scale`` controls particle radius as ``1 / scale``. Genesis uses
    particle diameter, hence ``particle_size = 2 / scale``.
    """
    if scale <= 0:
        raise ValueError("PBSTF particle scale must be positive")
    settings = _case_settings(case)
    particle_size = 2.0 / scale

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 30.0, gravity=settings["gravity"]),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=particle_size,
            lower_bound=settings["lower_bound"],
            upper_bound=settings["upper_bound"],
            max_solver_iterations=100,
            topology_rebuild_interval=10,
            max_surface_neighbors=128,
            enable_pca_normals=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=settings["camera_pos"],
            camera_lookat=settings["camera_lookat"],
            camera_up=(0.0, 1.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    entities_and_velocities = _add_case_entities(scene, case)
    scene.build()
    for entity, velocity in entities_and_velocities:
        entity.set_particles_vel(velocity)

    return scene, tuple(entity for entity, _ in entities_and_velocities)


def main():
    parser = argparse.ArgumentParser(description="GPU Position-Based Surface Tension Flow examples")
    parser.add_argument("--case", choices=CASES, default=CASE_CUBE)
    parser.add_argument("--scale", type=int, default=10, help="C++ particle scale (radius = 1 / scale)")
    parser.add_argument("--steps", type=int, default=None, help="Override the case's default simulation horizon")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error("--scale must be positive")
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")

    gs.init(backend=gs.cuda, precision="32", logging_level="info")
    scene, _ = build_scene(case=args.case, scale=args.scale, show_viewer=args.vis)

    steps = _case_settings(args.case)["steps"] if args.steps is None else args.steps
    if "PYTEST_VERSION" in os.environ:
        steps = min(steps, 2)
    for _ in range(steps):
        scene.step()


if __name__ == "__main__":
    main()
