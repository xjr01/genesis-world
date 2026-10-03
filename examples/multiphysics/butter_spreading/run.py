import argparse

import genesis as gs

from .config import ButterSpreadingScenarioConfig
from .scene import build_scene
from .task import ButterSpreadingController


def main():
    parser = argparse.ArgumentParser(description="Self-contained Genesis MPM butter-spreading scenario")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be non-negative")

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu, precision="32", logging_level="info")
    config = ButterSpreadingScenarioConfig()
    runtime = build_scene(config, show_viewer=args.vis)
    controller = ButterSpreadingController(config.task, config.solver.dt)
    for _ in range(args.steps):
        controller.step(runtime)


if __name__ == "__main__":
    main()
