import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .config import create_scene527_config
from .scene import build_scene
from .task import GarmentFoldingController, step_settling


def run_settle_only(runtime, config, output: Path, settle_steps: int):
    """Record an explicit settling pass and save the resulting resumable scene state."""
    output.mkdir(parents=True, exist_ok=True)
    video_path = output / "garment-folding-settle.mp4"
    metrics_path = output / "settle-metrics.csv"
    checkpoint_path = output / "settled-scene-state.pt"
    summary_path = output / "settle-summary.json"
    table_top_z = config.assets.table_pos[2] + 0.5 * config.assets.table_size[2]
    fieldnames = (
        "step",
        "simulated_time_s",
        "step_wall_seconds",
        "elapsed_wall_seconds",
        "height_min_mm",
        "height_median_mm",
        "height_p90_mm",
        "height_max_mm",
        "speed_mean_m_per_s",
        "speed_rms_m_per_s",
        "speed_max_m_per_s",
        "centroid_x_m",
        "centroid_y_m",
        "centroid_z_m",
        "extent_x_m",
        "extent_y_m",
        "extent_z_m",
    )
    rows = []
    settle_started = time.perf_counter()
    if runtime.camera is None:
        raise RuntimeError("Settling recording requires a camera.")
    runtime.camera.start_recording(save_to_filename=str(video_path), fps=round(1.0 / config.solver.dt))
    try:
        with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
            writer.writeheader()
            for step_index in range(settle_steps):
                step_started = time.perf_counter()
                step_settling(runtime)
                step_finished = time.perf_counter()
                garment_state = runtime.garment.get_state()
                if not torch.isfinite(garment_state.pos).all() or not torch.isfinite(garment_state.vel).all():
                    raise RuntimeError(f"Garment state contains non-finite values at settling step {step_index + 1}.")
                positions = tensor_to_array(garment_state.pos).reshape(-1, 3)
                velocities = tensor_to_array(garment_state.vel).reshape(-1, 3)
                heights_mm = (positions[:, 2] - table_top_z) * 1000.0
                speeds = np.linalg.norm(velocities, axis=1)
                centroid = positions.mean(axis=0)
                extent = np.ptp(positions, axis=0)
                row = {
                    "step": step_index + 1,
                    "simulated_time_s": runtime.scene.cur_t,
                    "step_wall_seconds": step_finished - step_started,
                    "elapsed_wall_seconds": step_finished - settle_started,
                    "height_min_mm": float(np.min(heights_mm)),
                    "height_median_mm": float(np.median(heights_mm)),
                    "height_p90_mm": float(np.quantile(heights_mm, 0.9)),
                    "height_max_mm": float(np.max(heights_mm)),
                    "speed_mean_m_per_s": float(np.mean(speeds)),
                    "speed_rms_m_per_s": float(np.sqrt(np.mean(np.square(speeds)))),
                    "speed_max_m_per_s": float(np.max(speeds)),
                    "centroid_x_m": float(centroid[0]),
                    "centroid_y_m": float(centroid[1]),
                    "centroid_z_m": float(centroid[2]),
                    "extent_x_m": float(extent[0]),
                    "extent_y_m": float(extent[1]),
                    "extent_z_m": float(extent[2]),
                }
                writer.writerow(row)
                metrics_file.flush()
                rows.append(row)
                print(
                    "garment_folding settle: "
                    f"step={row['step']}/{settle_steps}, "
                    f"simulated_time={row['simulated_time_s']:.6f}s, "
                    f"wall_seconds={row['step_wall_seconds']:.3f}, "
                    f"height_median_mm={row['height_median_mm']:.3f}, "
                    f"height_max_mm={row['height_max_mm']:.3f}, "
                    f"speed_rms={row['speed_rms_m_per_s']:.6f}m/s, "
                    f"speed_max={row['speed_max_m_per_s']:.6f}m/s",
                    flush=True,
                )
    finally:
        runtime.camera.stop_recording()

    checkpoint = runtime.scene.get_state()
    checkpoint.serializable()
    torch.save(checkpoint, checkpoint_path)
    settle_finished = time.perf_counter()
    summary = {
        "profile": "scene527",
        "settle_steps": settle_steps,
        "physics_dt_s": config.solver.dt,
        "simulated_time_s": settle_steps * config.solver.dt,
        "wall_seconds": settle_finished - settle_started,
        "rigid_rigid_contact_enabled": config.solver.is_rigid_rigid_contact_enabled,
        "configuration": asdict(config),
        "video": video_path.name,
        "metrics": metrics_path.name,
        "checkpoint": checkpoint_path.name,
        "first_step": rows[0],
        "final_step": rows[-1],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"garment_folding settle completed: steps={settle_steps}, "
        f"simulated_time={summary['simulated_time_s']:.6f}s, wall_seconds={summary['wall_seconds']:.3f}",
        flush=True,
    )


def main():
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="FEM/IPC garment-folding scenario")
    parser.add_argument("-s", "--steps", type=int, help="Physics steps to run; defaults to the remaining trajectory.")
    parser.add_argument("--asset-root", type=Path, help="External Scene527 bundle; defaults to the packaged assets.")
    parser.add_argument("--mesh", choices=("8k", "13k", "55k"), default="55k", help="Cloth mesh resolution.")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the scene viewer.")
    parser.add_argument("-r", "--record", action="store_true", help="Record each trajectory segment.")
    parser.add_argument("--settle-only", action="store_true", help="Record settling and save its checkpoint.")
    parser.add_argument("--settle-steps", type=int, help="Override Scene527 settling steps.")
    parser.add_argument("--checkpoint", type=Path, help="Resume a saved scene or scene/controller checkpoint.")
    parser.add_argument("--segment-steps", type=int, default=120, help="Physics steps per saved segment.")
    parser.add_argument("--disable-robot-ipc-proxies", action="store_true", help="Exclude robot links from IPC.")
    parser.add_argument(
        "--progress-every", type=int, default=10, help="Print progress every N steps; zero disables it."
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("out/garment_folding"), help="Metrics, videos and checkpoints."
    )
    args = parser.parse_args()
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.segment_steps <= 0:
        parser.error("--segment-steps must be positive")
    if args.settle_only and args.steps is not None:
        parser.error("--settle-only cannot be combined with --steps")
    if args.settle_only and args.checkpoint is not None:
        parser.error("--settle-only cannot be combined with --checkpoint")
    if args.settle_steps is not None and not args.settle_only:
        parser.error("--settle-steps requires --settle-only")
    if args.settle_steps is not None and args.settle_steps <= 0:
        parser.error("--settle-steps must be positive")

    # The Incremental Potential Contact (IPC) backend executes its solve on CUDA
    gs.init(backend=gs.gpu, logging_level="warning", seed=0)
    config = create_scene527_config(args.asset_root, mesh=args.mesh)
    if args.disable_robot_ipc_proxies:
        config = replace(
            config,
            assets=replace(config.assets, robot_coupling_links=()),
        )
    runtime = build_scene(
        config,
        show_viewer=args.vis,
        add_camera=args.record or args.settle_only,
    )
    build_seconds = time.perf_counter() - process_started
    print(f"garment_folding timing: build_seconds={build_seconds:.3f}", flush=True)
    table_top_z = config.assets.table_pos[2] + 0.5 * config.assets.table_size[2]
    garment_heights = runtime.initial_positions[:, 2] - table_top_z
    print(
        "garment_folding initial_height: "
        f"min_mm={garment_heights.min() * 1000.0:.3f}, "
        f"median_mm={np.median(garment_heights) * 1000.0:.3f}, "
        f"p90_mm={np.quantile(garment_heights, 0.9) * 1000.0:.3f}, "
        f"max_mm={garment_heights.max() * 1000.0:.3f}",
        flush=True,
    )
    if args.settle_only:
        run_settle_only(runtime, config, args.output_dir, args.settle_steps or config.task.settle_steps)
        return
    controller = GarmentFoldingController(config.task)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint is None:
        for settle_index in range(config.task.settle_steps):
            step_settling(runtime)
            if settle_index == 0 or (settle_index + 1) % 10 == 0:
                print(f"garment_folding settling: step={settle_index + 1}/{config.task.settle_steps}", flush=True)
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, tuple):
            runtime.scene.restore(checkpoint[0])
            controller.set_state(runtime, checkpoint[1])
        else:
            runtime.scene.restore(checkpoint)
        print(
            f"garment_folding restored: checkpoint={args.checkpoint}, "
            f"scene_cur_t={runtime.scene.cur_t:.6f}s, controller_step={controller.step_index}",
            flush=True,
        )
    steps = args.steps if args.steps is not None else max(0, runtime.default_steps - controller.step_index)
    metrics_path = args.output_dir / "garment-folding-metrics.csv"
    fields = (
        "step",
        "trajectory_time_s",
        "scene_cur_t_s",
        "step_wall_seconds",
        "elapsed_wall_seconds",
        "height_median_mm",
        "height_p90_mm",
        "height_max_mm",
        "speed_rms_m_per_s",
        "speed_max_m_per_s",
    )
    simulation_started = time.perf_counter()
    progress_started = simulation_started
    with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fields)
        if metrics_file.tell() == 0:
            writer.writeheader()
        completed_steps = 0
        segment_index = 0
        while completed_steps < steps:
            segment_steps = min(args.segment_steps, steps - completed_steps)
            segment_start = controller.step_index
            video_path = args.output_dir / f"garment-folding-segment-{segment_start:06d}.mp4"
            if runtime.camera is not None:
                runtime.camera.start_recording(save_to_filename=str(video_path), fps=30)
            try:
                for _ in range(segment_steps):
                    step_started = time.perf_counter()
                    controller.step(runtime)
                    step_finished = time.perf_counter()
                    completed_steps += 1
                    garment_state = runtime.garment.get_state()
                    if not torch.isfinite(garment_state.pos).all() or not torch.isfinite(garment_state.vel).all():
                        raise RuntimeError(f"Garment state contains non-finite values at step {controller.step_index}.")
                    positions = tensor_to_array(garment_state.pos).reshape(-1, 3)
                    velocities = tensor_to_array(garment_state.vel).reshape(-1, 3)
                    speeds = np.linalg.norm(velocities, axis=1)
                    heights_mm = (positions[:, 2] - table_top_z) * 1000.0
                    row = {
                        "step": controller.step_index,
                        "trajectory_time_s": controller.step_index * config.solver.dt,
                        "scene_cur_t_s": runtime.scene.cur_t,
                        "step_wall_seconds": step_finished - step_started,
                        "elapsed_wall_seconds": step_finished - simulation_started,
                        "height_median_mm": float(np.median(heights_mm)),
                        "height_p90_mm": float(np.quantile(heights_mm, 0.9)),
                        "height_max_mm": float(np.max(heights_mm)),
                        "speed_rms_m_per_s": float(np.sqrt(np.mean(np.square(speeds)))),
                        "speed_max_m_per_s": float(np.max(speeds)),
                    }
                    writer.writerow(row)
                    metrics_file.flush()
                    if args.progress_every and (completed_steps % args.progress_every == 0 or completed_steps == steps):
                        progress_finished = time.perf_counter()
                        print(
                            f"garment_folding progress: step={completed_steps}/{steps}, "
                            f"controller_step={controller.step_index}, "
                            f"trajectory_time={row['trajectory_time_s']:.6f}s, "
                            f"scene_cur_t={row['scene_cur_t_s']:.6f}s, "
                            f"step_wall_seconds={row['step_wall_seconds']:.3f}, "
                            f"wall_seconds={progress_finished - progress_started:.3f}, "
                            f"height_median_mm={row['height_median_mm']:.3f}, "
                            f"height_max_mm={row['height_max_mm']:.3f}, "
                            f"speed_rms={row['speed_rms_m_per_s']:.6f}m/s, "
                            f"speed_max={row['speed_max_m_per_s']:.6f}m/s",
                            flush=True,
                        )
                        progress_started = progress_finished
            finally:
                if runtime.camera is not None:
                    runtime.camera.stop_recording()
            scene_state = runtime.scene.get_state()
            scene_state.serializable()
            torch.save(
                (scene_state, controller.get_state()),
                args.output_dir / f"garment-folding-checkpoint-{controller.step_index:06d}.pt",
            )
            print(
                f"garment_folding segment finalized: index={segment_index}, steps={segment_steps}, "
                f"video={video_path.name if runtime.camera is not None else 'disabled'}, "
                f"checkpoint_step={controller.step_index}",
                flush=True,
            )
            segment_index += 1
    simulation_seconds = time.perf_counter() - simulation_started
    garment_state = runtime.garment.get_state()
    if not torch.isfinite(garment_state.pos).all() or not torch.isfinite(garment_state.vel).all():
        raise RuntimeError("Garment state contains non-finite values.")
    print(f"garment_folding timing: simulation_seconds={simulation_seconds:.3f}", flush=True)
    print(
        f"garment_folding completed: profile=scene527, steps={steps}, "
        f"scene_cur_t={runtime.scene.cur_t:.6f}s, controller_step={controller.step_index}, "
        f"trajectory_time={controller.step_index * config.solver.dt:.6f}s"
    )


if __name__ == "__main__":
    main()
