import argparse

import genesis as gs

from .config import TableWipingScenarioConfig
from .scene import build_scene
from .task import TableWipingController


def main():
    parser = argparse.ArgumentParser(description="PBSTF table-wiping scenario")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be non-negative")

    gs.init(backend=gs.gpu, precision="32", logging_level="info")
    config = TableWipingScenarioConfig()
    runtime = build_scene(config, show_viewer=args.vis)
    controller = TableWipingController(config.task)
    for _ in range(args.steps):
        controller.step(runtime)


if __name__ == "__main__":
    main()
