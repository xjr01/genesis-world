"""IPBF example 03: resting-block density-error metrics (no rendering).

A 0.2^3 m liquid block resting on the box floor; logs the per-frame avg/max of the clamped
density constraint C = max(rho~ - 1, 0) to CSV. Config: h = 1/480 s (dt=1/60, substeps=8),
ipbf_iterations=2, alpha=0, damping off (default), boundary particles on (default).
Paper reference: density-error metric used in Table 1 / Fig. 10 (Diaz et al., SIGGRAPH Asia 2025).
Run: python examples/ipbf/03_density_check.py [--horizon 180] [--damping]
"""

import argparse
import csv
import os

import torch

import genesis as gs

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None, choices=["cpu", "gpu"])
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "density_error.csv"))
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

    ipbf_kwargs = dict(
        lower_bound=(-0.5, -0.5, 0.0),
        upper_bound=(0.5, 0.5, 1.0),
        particle_size=0.01,
        ipbf_iterations=2,
        alpha=0.0,
    )
    if args.damping is not None:
        ipbf_kwargs["damping_enabled"] = args.damping

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60, substeps=8),
        ipbf_options=gs.options.IPBFOptions(**ipbf_kwargs),
        show_viewer=False,
    )

    liquid = scene.add_entity(
        material=gs.materials.IPBF.Liquid(),
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.1), size=(0.2, 0.2, 0.2)),
        surface=gs.surfaces.Default(color=(0.4, 0.8, 1.0), vis_mode="particle"),
    )

    scene.build()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["frame", "avg_C", "max_C"])
        for frame in range(args.horizon):
            scene.step()
            rho = liquid.get_particles_rho()
            assert torch.isfinite(rho).all(), f"Non-finite density at frame {frame}!"
            C = torch.clamp(rho - 1.0, min=0.0)
            writer.writerow([frame, C.mean().item(), C.max().item()])

    pos = liquid.get_particles_pos()
    assert torch.isfinite(pos).all(), "Non-finite particle positions detected!"
    print("final particles pos stats: min", pos.min(axis=0), "max", pos.max(axis=0))
    print("density error log saved to", args.out)


if __name__ == "__main__":
    main()
