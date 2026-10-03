import argparse

import genesis as gs

from .config import LitterScoopScenarioConfig
from .scene import build_scene
from .task import LitterScoopController


def main():
    parser = argparse.ArgumentParser(description="Coupled DEM/FLIP litter-scoop scenario")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be non-negative")

    gs.init(backend=gs.gpu, logging_level="info")
    config = LitterScoopScenarioConfig()
    runtime = build_scene(config, show_viewer=args.vis)
    controller = LitterScoopController(config.task)
    for _ in range(args.steps):
        controller.step(runtime)


if __name__ == "__main__":
    main()
