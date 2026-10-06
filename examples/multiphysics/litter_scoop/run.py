import argparse
import time
from pathlib import Path

import torch

import genesis as gs

from .config import LitterScoopScenarioConfig
from .scene import build_scene
from .task import LitterScoopController


def main():
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Coupled DEM/FLIP litter-scoop scenario")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("out/litter_scoop"))
    args = parser.parse_args()
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")

    gs.init(backend=gs.gpu, logging_level="warning")
    config = LitterScoopScenarioConfig()
    runtime = build_scene(config, show_viewer=args.vis, add_camera=args.record)
    build_seconds = time.perf_counter() - process_started
    print(f"litter_scoop timing: build_seconds={build_seconds:.3f}", flush=True)
    controller = LitterScoopController(config.task)
    steps = config.task.total_steps if args.steps is None else args.steps
    if runtime.camera is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        runtime.camera.start_recording(save_to_filename=str(args.output / "litter-scoop.mp4"), fps=30)
    try:
        simulation_started = time.perf_counter()
        for step in range(steps):
            controller.step(runtime)
            if args.progress_every and ((step + 1) % args.progress_every == 0 or step + 1 == steps):
                print(
                    f"litter_scoop progress: step={step + 1}/{steps}, simulated_time={runtime.scene.cur_t:.6f}s",
                    flush=True,
                )
    finally:
        if runtime.camera is not None:
            runtime.camera.stop_recording()
    simulation_seconds = time.perf_counter() - simulation_started
    for name, entity in (("sand", runtime.sand), ("water", runtime.water)):
        if not torch.isfinite(entity.get_particles_pos()).all():
            raise RuntimeError(f"Litter-scoop {name} positions contain non-finite values.")
        if not torch.isfinite(entity.get_particles_vel()).all():
            raise RuntimeError(f"Litter-scoop {name} velocities contain non-finite values.")
    print(f"litter_scoop timing: simulation_seconds={simulation_seconds:.3f}", flush=True)
    print(f"litter_scoop completed: steps={steps}, simulated_time={runtime.scene.cur_t:.6f}s")


if __name__ == "__main__":
    main()
