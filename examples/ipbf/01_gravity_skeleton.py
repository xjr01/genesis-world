"""IPBF example 01: simplest IPBF scene — a liquid block drops into a box (hello world).

A 0.4 m liquid cube falls under gravity and splashes onto the floor of the domain box.
Config: h = 1/480 s (dt=4e-3, substeps=10 -> substep h=4e-4 here for a quick look; use
dt=1/60, substeps=8 for the paper's canonical h=1/480), ipbf_iterations=2, alpha=0,
damping off (default), boundary particles on (default).
Originally the Phase-1 skeleton validation of the IPBF solver integration.
Run: python examples/ipbf/01_gravity_skeleton.py
"""

import argparse
import os

import torch

import genesis as gs

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None, choices=["cpu", "gpu"])
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "gravity_skeleton.mp4"))
    args = parser.parse_args()

    test_mode = "PYTEST_VERSION" in os.environ  # quick CPU smoke for tests/test_examples.py
    backend = args.backend or ("cpu" if test_mode else "gpu")
    if test_mode:
        args.horizon = min(args.horizon, 3)

    gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
        ipbf_options=gs.options.IPBFOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1.0),
            particle_size=0.01,
        ),
        show_viewer=False,
    )

    scene.add_entity(morph=gs.morphs.Plane())
    liquid = scene.add_entity(
        material=gs.materials.IPBF.Liquid(),
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.65), size=(0.4, 0.4, 0.4)),
        surface=gs.surfaces.Default(color=(0.4, 0.8, 1.0), vis_mode="particle"),
    )

    cam = scene.add_camera(res=(1280, 960), pos=(2.0, -2.0, 1.5), lookat=(0, 0, 0.4), fov=35, GUI=False)
    scene.build()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cam.start_recording(save_to_filename=args.out, fps=60)
    for _ in range(args.horizon):
        scene.step()
    cam.stop_recording()

    pos = liquid.get_particles_pos()
    assert torch.isfinite(pos).all(), "Non-finite particle positions detected!"
    print("final particles pos stats: min", pos.min(axis=0), "max", pos.max(axis=0))
    print("video saved to", args.out)


if __name__ == "__main__":
    main()
