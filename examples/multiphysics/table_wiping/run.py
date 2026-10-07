import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .config import TableWipingScenarioConfig
from .scene import build_scene
from .task import TableWipingController


@dataclass(frozen=True)
class TableWipingMetrics:
    step: int
    time_s: float
    free_particles: int
    absorbed_particles: int
    free_p90_height_m: float
    free_max_height_m: float
    free_extent_x_m: float
    free_extent_z_m: float
    free_rms_speed_m_s: float
    absorbed_progress_mean: float


def main():
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="PBSTF table-wiping scenario")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--metrics-every", type=int, default=100, help="Native steps between liquid measurements")
    parser.add_argument(
        "--record-segment-seconds", type=float, default=1.0, help="Simulation seconds per video segment"
    )
    parser.add_argument("--output", type=Path, default=Path("out/table_wiping"))
    args = parser.parse_args()
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.metrics_every <= 0:
        parser.error("--metrics-every must be positive")
    if args.record_segment_seconds <= 0.0:
        parser.error("--record-segment-seconds must be positive")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    config = TableWipingScenarioConfig()
    runtime = build_scene(config, show_viewer=args.vis, add_camera=args.record)
    build_seconds = time.perf_counter() - process_started
    print(f"table_wiping timing: build_seconds={build_seconds:.3f}", flush=True)
    controller = TableWipingController(config.task)
    steps = config.task.steps if args.steps is None else args.steps
    args.output.mkdir(parents=True, exist_ok=True)
    segment_steps = max(1, round(args.record_segment_seconds / config.solver.dt))
    rows = []
    table_top = config.assets.table_pos[1] + 0.5 * config.assets.table_size[1]
    initial_particle_count = runtime.liquid.n_particles
    is_recording = False
    simulation_started = time.perf_counter()
    try:
        for step in range(steps + 1):
            if step % args.metrics_every == 0 or step == steps:
                state = runtime.scene.pbstf_solver.get_state(0)
                if not torch.isfinite(state.pos).all() or not torch.isfinite(state.vel).all():
                    raise RuntimeError("Table-wiping liquid state contains non-finite values.")
                is_absorbed = tensor_to_array(state.absorbed_collider_idx[0]) >= 0
                positions = tensor_to_array(state.pos[0])[~is_absorbed]
                velocities = tensor_to_array(state.vel[0])[~is_absorbed]
                progress = tensor_to_array(state.absorption_progress[0])[is_absorbed]
                heights = positions[:, 1] - table_top
                extent = np.ptp(positions, axis=0) if len(positions) else np.zeros(3)
                if not state.active.all() or len(positions) + is_absorbed.sum() != initial_particle_count:
                    raise RuntimeError("Table-wiping particle conservation check failed.")
                row = TableWipingMetrics(
                    step=step,
                    time_s=step * config.solver.dt,
                    free_particles=len(positions),
                    absorbed_particles=int(is_absorbed.sum()),
                    free_p90_height_m=float(np.quantile(heights, 0.9)) if len(heights) else 0.0,
                    free_max_height_m=float(heights.max()) if len(heights) else 0.0,
                    free_extent_x_m=float(extent[0]),
                    free_extent_z_m=float(extent[2]),
                    free_rms_speed_m_s=float(np.sqrt(np.mean(np.sum(velocities**2, axis=1))))
                    if len(positions)
                    else 0.0,
                    absorbed_progress_mean=float(progress.mean()) if len(progress) else 0.0,
                )
                rows.append(row)
            if args.progress_every and step and (step % args.progress_every == 0 or step == steps):
                print(
                    f"table_wiping progress: step={step}/{steps}, simulated_time={runtime.scene.cur_t:.6f}s, "
                    f"free_particles={rows[-1].free_particles}, absorbed_particles={rows[-1].absorbed_particles}",
                    flush=True,
                )
            if step == steps:
                break
            if runtime.camera is not None and step % segment_steps == 0:
                if is_recording:
                    runtime.camera.stop_recording()
                video_path = args.output / f"table-wiping-segment-{step:06d}.mp4"
                runtime.camera.start_recording(save_to_filename=str(video_path), fps=30)
                is_recording = True
            controller.step(runtime)
    finally:
        if is_recording:
            runtime.camera.stop_recording()
        with (args.output / "table-wiping-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=asdict(rows[0]))
                writer.writeheader()
                writer.writerows(asdict(row) for row in rows)
    simulation_seconds = time.perf_counter() - simulation_started
    if not torch.isfinite(runtime.liquid.get_particles_pos()).all():
        raise RuntimeError("Table-wiping liquid positions contain non-finite values.")
    if not torch.isfinite(runtime.sponge.get_particles_pos()).all():
        raise RuntimeError("Table-wiping sponge positions contain non-finite values.")
    runtime.scene.pbstf_solver.check_errno()
    print(f"table_wiping timing: simulation_seconds={simulation_seconds:.3f}", flush=True)
    print(f"table_wiping completed: steps={steps}, simulated_time={runtime.scene.cur_t:.6f}s")
    gs.destroy()


if __name__ == "__main__":
    main()
