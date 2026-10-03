import argparse
from dataclasses import replace

import genesis as gs

from .config import GarmentFoldingScenarioConfig
from .scene import build_scene
from .task import GarmentFoldingController


def main():
    parser = argparse.ArgumentParser(description="FEM/IPC garment-folding scenario")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--task", choices=("grasp", "half", "quarter", "fold"), default="half")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")

    gs.init(backend=gs.gpu, logging_level="info", seed=1)
    config = GarmentFoldingScenarioConfig()
    config = replace(config, task=replace(config.task, task=args.task))
    runtime = build_scene(config, show_viewer=args.vis)
    controller = GarmentFoldingController(config.task)
    steps = args.steps
    if steps is None:
        steps = round(config.task.resolved_duration / config.solver.dt) + 1
    for _ in range(steps):
        controller.step(runtime)


if __name__ == "__main__":
    main()
