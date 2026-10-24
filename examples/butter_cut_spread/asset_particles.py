"""Encode actual particle motion against the authored dish and blade cross-sections."""

import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
import trimesh
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    with np.load(run / "state.npz") as data:
        particles, times, centers, quats = (
            data[k] for k in ("particles", "time_s", "knife_positions", "knife_quaternions")
        )
    dish = trimesh.load_mesh(run / "dish.obj", process=False)
    blade = trimesh.load_mesh(run / "blade.obj", process=False)
    dish_section = dish.section((0, 1, 0), (0, 0, 0)).discrete
    figure = Figure(figsize=(12.8, 7.2), dpi=100)
    canvas = FigureCanvasAgg(figure)
    side, top = figure.subplots(2, 1)
    side.set(xlim=(-80, 85), ylim=(-3, 115), xlabel="X (mm)", ylabel="Z (mm)", aspect="equal")
    top.set(xlim=(-80, 85), ylim=(-25, 25), xlabel="X (mm)", ylabel="Y (mm)", aspect="equal")
    for line in dish_section:
        side.plot(line[:, 0] * 1000, line[:, 2] * 1000, color="#8a939b", linewidth=1)
    plots = [side.scatter([], [], s=1, color=color) for color in ("#d5ac45", "#ed852b")]
    overview = top.scatter([], [], s=0.5, color="#d5ac45")
    blade_plot = LineCollection([], colors="#50667a", linewidths=1.5)
    side.add_collection(blade_plot)
    slice_ids = particles[0, :, 0] > config["butter_center"][0] + config["cut_offset"] + config["blade_size"][2] / 2
    static_ids = np.arange(0, particles.shape[1], 80)
    core_ids = (abs(particles[0, :, 0] - config["butter_center"][0]) < 0.02) & (
        abs(particles[0, :, 1] - config["butter_center"][1]) < 0.004
    )
    metrics = []
    side.set_title("Authored assets | central Y section | active MPM particles")
    top.set_title("Top view | fixed particle-ID subsample")
    figure.tight_layout()
    with av.open(str(run / "asset-cut-particles.mp4"), "w") as movie:
        stream = movie.add_stream("libx264", rate=config["fps"])
        stream.width, stream.height, stream.pix_fmt = 1280, 720, "yuv420p"
        stream.options = {"crf": "18"}
        for i, points in enumerate(particles):
            section = abs(points[:, 1]) < 0.0004
            for plot, mask in zip(plots, (~slice_ids, slice_ids)):
                plot.set_offsets(points[section & mask][:, (0, 2)] * 1000)
            overview.set_offsets(points[static_ids, :2] * 1000)
            rotation = Rotation.from_quat(quats[i, [1, 2, 3, 0]])
            world_blade = blade.copy()
            world_blade.vertices = rotation.apply(blade.vertices) + centers[i]
            section_mesh = world_blade.section((0, 1, 0), (0, 0, 0))
            blade_plot.set_segments(
                [] if section_mesh is None else [line[:, (0, 2)] * 1000 for line in section_mesh.discrete]
            )
            side.set_title(f"Authored assets | central Y section | t={times[i]:.3f}s | active MPM particles")
            top.set_title("Top view | fixed particle-ID subsample")
            canvas.draw()
            pixels = np.array(canvas.buffer_rgba())[:, :, :3]
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                movie.mux(packet)
            low, high = np.quantile(points[~slice_ids], (0.01, 0.99), axis=0)
            core_bottom, core_top = np.quantile(points[core_ids, 2], (0.01, 0.99))
            metrics.append(
                {
                    "time_s": float(times[i]),
                    "bulk_height_p01_p99_m": float(high[2] - low[2]),
                    "bulk_width_p01_p99_m": float(high[1] - low[1]),
                    "core_height_p01_p99_m": float(core_top - core_bottom),
                    "core_top_p99_m": float(core_top),
                    "center_m": points.mean(0).tolist(),
                    "minimum_z_m": float(points[:, 2].min()),
                    "slice_above_initial_top_fraction": float(
                        np.mean(points[slice_ids, 2] > config["butter_center"][2] + config["butter_size"][2] / 2)
                    ),
                }
            )
        for packet in stream.encode():
            movie.mux(packet)
    (run / "particle-metrics.json").write_text(
        json.dumps({"frames": metrics, "elapsed_seconds": time.perf_counter() - started}, indent=2)
    )
    print(json.dumps({"first": metrics[0], "last": metrics[-1]}, indent=2))


if __name__ == "__main__":
    main()
