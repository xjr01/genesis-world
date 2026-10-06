import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch

import genesis as gs

from .config import GarmentFoldingScenarioConfig, create_scene527_config
from .scene import build_scene
from .task import GarmentFoldingController


def main():
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="FEM/IPC garment-folding scenario")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--profile", choices=("lightweight", "scene527"), default="lightweight")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--task", choices=("grasp", "half", "quarter", "fold"))
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("out/garment_folding"))
    args = parser.parse_args()
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.profile == "scene527" and args.asset_root is None:
        parser.error("--asset-root is required for the scene527 profile")
    if args.profile == "scene527" and args.task is not None:
        parser.error("--task cannot be combined with the scene527 profile")

    gs.init(backend=gs.gpu, logging_level="warning", seed=1)
    if args.profile == "scene527":
        config = create_scene527_config(args.asset_root)
    else:
        config = GarmentFoldingScenarioConfig()
        config = replace(config, task=replace(config.task, task=args.task or "half"))
    runtime = build_scene(config, show_viewer=args.vis, add_camera=args.record)
    build_seconds = time.perf_counter() - process_started
    print(f"garment_folding timing: build_seconds={build_seconds:.3f}", flush=True)
    controller = GarmentFoldingController(config.task)
    steps = args.steps
    if steps is None:
        steps = runtime.default_steps
    if runtime.camera is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        runtime.camera.start_recording(save_to_filename=str(args.output / "garment-folding.mp4"), fps=30)
    try:
        simulation_started = time.perf_counter()
        for _ in range(steps):
            controller.step(runtime)
    finally:
        if runtime.camera is not None:
            runtime.camera.stop_recording()
    simulation_seconds = time.perf_counter() - simulation_started
    garment_state = runtime.garment.get_state()
    if not torch.isfinite(garment_state.pos).all() or not torch.isfinite(garment_state.vel).all():
        raise RuntimeError("Garment state contains non-finite values.")
    print(f"garment_folding timing: simulation_seconds={simulation_seconds:.3f}", flush=True)
    print(
        f"garment_folding completed: profile={args.profile}, task={config.task.task}, steps={steps}, "
        f"simulated_time={runtime.scene.cur_t:.6f}s"
    )


if __name__ == "__main__":
    main()
