"""IPBF example 02: single-column dam break.

A water column occupying the left half of the box collapses and sloshes against the walls.
Config: h = 1/480 s (dt=1/60, substeps=8), ipbf_iterations=2, alpha=0, damping off (default),
boundary particles on (default). --bigstep runs h = 1/30 s (dt=1/30, substeps=1), the paper's
unconditional-stability regime — note the known limitation that the fluid over-compresses at
such steps in this implementation (see README, "Known limitations").
Paper reference: dam-break examples in Figs. 2-3 (Diaz et al., SIGGRAPH Asia 2025).
Run: python examples/ipbf/02_dam_break.py [--bigstep] [--iters 2] [--damping]
"""

import argparse
import os

import torch

import genesis as gs

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None, choices=["cpu", "gpu"])
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "dam_break.mp4"))
    parser.add_argument("--bigstep", action="store_true", help="dt=1/30, substeps=1 (h=1/30 s)")
    parser.add_argument("--dt", type=float, default=None, help="override SimOptions dt")
    parser.add_argument("--substeps", type=int, default=None, help="override SimOptions substeps")
    parser.add_argument("--iters", type=int, default=2, help="IPBF Newton iterations per substep")
    parser.add_argument("--damping", dest="damping", action="store_true", default=None,
                        help="enable artificial damping (default: follow IPBFOptions, i.e. off)")
    parser.add_argument("--no-damping", dest="damping", action="store_false",
                        help="disable artificial damping")
    args = parser.parse_args()

    test_mode = "PYTEST_VERSION" in os.environ  # quick CPU smoke for tests/test_examples.py
    backend = args.backend or ("cpu" if test_mode else "gpu")
    if test_mode:
        args.horizon = min(args.horizon, 3)

    gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu)

    if args.bigstep:
        dt, substeps, fps = 1.0 / 30, 1, 30
    else:
        dt, substeps, fps = 1.0 / 60, 8, 60
    if args.dt is not None:
        dt = args.dt
    if args.substeps is not None:
        substeps = args.substeps
    print(f"dt={dt}, substeps={substeps}, substep h={dt / substeps}")

    ipbf_kwargs = dict(
        lower_bound=(-0.5, -0.5, 0.0),
        upper_bound=(0.5, 0.5, 1.0),
        particle_size=0.01,
        ipbf_iterations=args.iters,
        alpha=0.0,
    )
    if args.damping is not None:
        ipbf_kwargs["damping_enabled"] = args.damping

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
        ipbf_options=gs.options.IPBFOptions(**ipbf_kwargs),
        show_viewer=False,
    )

    scene.add_entity(morph=gs.morphs.Plane())
    liquid = scene.add_entity(
        material=gs.materials.IPBF.Liquid(),
        morph=gs.morphs.Box(pos=(-0.25, 0.0, 0.25), size=(0.4, 0.8, 0.5)),
        surface=gs.surfaces.Default(color=(0.4, 0.8, 1.0), vis_mode="particle"),
    )

    cam = scene.add_camera(res=(1280, 960), pos=(1.8, -1.8, 1.2), lookat=(0, 0, 0.3), fov=35, GUI=False)
    scene.build()

    solver = scene.sim.ipbf_solver
    print(f"particles: {solver._n_fluid_particles} fluid + {solver._n_boundary_particles} boundary")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cam.start_recording(save_to_filename=args.out, fps=fps)
    for _ in range(args.horizon):
        scene.step()
    cam.stop_recording()

    pos = liquid.get_particles_pos()
    assert torch.isfinite(pos).all(), "Non-finite particle positions detected!"
    print("final fluid particles pos stats: min", pos.min(axis=0), "max", pos.max(axis=0))
    print("video saved to", args.out)


if __name__ == "__main__":
    main()
