import argparse
import csv
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
    parser.add_argument(
        "--record-segment-steps", type=int, default=60, help="Native steps per finalized video segment."
    )
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("out/litter_scoop"))
    args = parser.parse_args()
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.record_segment_steps <= 0:
        parser.error("--record-segment-steps must be positive")

    gs.init(backend=gs.gpu, logging_level="warning")
    config = LitterScoopScenarioConfig()
    runtime = build_scene(config, show_viewer=args.vis, add_camera=args.record)
    build_seconds = time.perf_counter() - process_started
    print(f"litter_scoop timing: build_seconds={build_seconds:.3f}", flush=True)
    controller = LitterScoopController(config.task)
    steps = config.task.total_steps if args.steps is None else args.steps
    args.output.mkdir(parents=True, exist_ok=True)
    if runtime.camera is not None:
        runtime.camera.start_recording(save_to_filename=str(args.output / "litter-scoop-segment-000000.mp4"), fps=30)
    try:
        simulation_started = time.perf_counter()
        with (args.output / "litter-scoop-metrics.csv").open("w", newline="", encoding="ascii") as metrics:
            writer = csv.writer(metrics)
            writer.writerow(("step", "time_s", "wall_s", "sand_min_z", "sand_max_z", "sand_rms_speed", "water_max_z"))
            for step in range(steps):
                controller.step(runtime)
                if (step + 1) % 50 == 0 or step == 0 or step + 1 == steps:
                    sand_pos = runtime.sand.get_particles_pos()
                    sand_vel = runtime.sand.get_particles_vel()
                    water_pos = runtime.water.get_particles_pos()
                    water_vel = runtime.water.get_particles_vel()
                    if not all(torch.isfinite(values).all() for values in (sand_pos, sand_vel, water_pos, water_vel)):
                        raise RuntimeError(f"Litter-scoop state contains non-finite values at step {step + 1}.")
                    writer.writerow(
                        (
                            step + 1,
                            runtime.scene.cur_t,
                            time.perf_counter() - simulation_started,
                            sand_pos[..., 2].min().item(),
                            sand_pos[..., 2].max().item(),
                            torch.sqrt(torch.square(sand_vel).sum(dim=-1).mean()).item(),
                            water_pos[..., 2].max().item(),
                        )
                    )
                    metrics.flush()
                if runtime.camera is not None and (step + 1) % args.record_segment_steps == 0 and step + 1 < steps:
                    runtime.camera.stop_recording()
                    runtime.camera.start_recording(
                        save_to_filename=str(args.output / f"litter-scoop-segment-{step + 1:06d}.mp4"), fps=30
                    )
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
    try:
        main()
    finally:
        gs.destroy()
