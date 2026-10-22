"""Encode a central particle cross-section without a graphics context or surface reconstruction."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from scipy.spatial.transform import Rotation

from examples.butter_cut_spread.config import PHASES, Config


@dataclass(frozen=True)
class VideoReport:
    frames: int
    fps: int
    elapsed_seconds: float
    state_sha256: str
    video_sha256: str
    scope: str


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Directory containing config.json and state.npz.")
    args = parser.parse_args()
    started = time.perf_counter()
    run = args.run.resolve()
    config = Config(**json.loads((run / "config.json").read_text()))
    simulation = json.loads((run / "simulation.json").read_text())
    state_hash = hashlib.sha256((run / "state.npz").read_bytes()).hexdigest()
    if state_hash != simulation["state_sha256"]:
        raise ValueError("Particle archive differs from the simulation report.")
    with np.load(run / "state.npz") as archive:
        particles, bread, positions, quaternions, times = (
            archive[key] for key in ("particles", "bread_particles", "knife_positions", "knife_quaternions", "time_s")
        )
    is_slice = particles[0, :, 0] > config.butter_center[0] + config.cut_offset + config.blade_size[2] / 2
    figure = Figure(figsize=(12.8, 4.8), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.set(xlim=(-165, 120), ylim=(-5, 95), xlabel="X (mm)", ylabel="Z (mm)", aspect="equal")
    axes.grid(alpha=0.15)
    bulk_plot = axes.scatter([], [], s=0.5, c="#d0a342", rasterized=True)
    slice_plot = axes.scatter([], [], s=0.6, c="#ee922a", rasterized=True)
    bread_plot = axes.scatter([], [], s=0.3, c="#b99a70", rasterized=True)
    for center, size in zip(config.board_centers, config.board_sizes):
        axes.plot(
            [(center[0] - size[0] / 2) * 1000, (center[0] + size[0] / 2) * 1000],
            [(center[2] + size[2] / 2) * 1000] * 2,
            color="#72543a",
            linewidth=2,
        )
    blade_plot = Polygon(np.zeros((4, 2)), closed=True, color="#687b8c")
    axes.add_patch(blade_plot)
    half = np.array(config.blade_size) / 2
    section = np.array(
        [
            [-half[0], 0, -half[2]],
            [half[0], 0, -half[2]],
            [half[0], 0, half[2]],
            [-half[0], 0, half[2]],
        ]
    )
    figure.tight_layout()
    output = run / "cut-spread-particles.mp4"
    with av.open(str(output), "w") as movie:
        stream = movie.add_stream("libx264", rate=config.fps)
        stream.width, stream.height, stream.pix_fmt = 1280, 480, "yuv420p"
        stream.options = {"crf": "19"}
        for i, time_s in enumerate(times):
            points = particles[i]
            is_section = abs(points[:, 1]) < 0.00035
            bulk_plot.set_offsets(points[is_section & ~is_slice][:, (0, 2)] * 1000)
            slice_plot.set_offsets(points[is_section & is_slice][:, (0, 2)] * 1000)
            is_section = abs(bread[i, :, 1]) < 0.0005
            bread_plot.set_offsets(bread[i, is_section][:, (0, 2)] * 1000)
            world = Rotation.from_quat(quaternions[i, [1, 2, 3, 0]]).apply(section) + positions[i]
            blade_plot.set_xy(world[:, (0, 2)] * 1000)
            phase = next((name for name, end in PHASES if time_s <= end), "withdraw")
            axes.set_title(f"Continuous particle simulation | {phase} | {time_s:.3f} s | central Y section")
            canvas.draw()
            frame = av.VideoFrame.from_ndarray(np.array(canvas.buffer_rgba())[:, :, :3], format="rgb24")
            for packet in stream.encode(frame):
                movie.mux(packet)
        for packet in stream.encode():
            movie.mux(packet)
    report = VideoReport(
        len(times),
        config.fps,
        time.perf_counter() - started,
        state_hash,
        hashlib.sha256(output.read_bytes()).hexdigest(),
        "Central Y cross-section of actual particle positions; orange identifies the initial end slice. "
        "The blade outline shows its rectangular envelope, not its microscopic bevel.",
    )
    (run / "particles-video.json").write_text(json.dumps(asdict(report), indent=2))
    print(json.dumps(asdict(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
