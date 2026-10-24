"""Render a recorded spreading trajectory with Isaac Sim."""

import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Output of prepare_isaac.")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--background-fill", choices=("off", "on"), default="off")
    parser.add_argument(
        "--frozen-frame", type=int, help="Repeat a source frame eight times for preview and stability checks."
    )
    parser.add_argument("--back-view", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("Samples must be positive.")
    started = time.perf_counter()
    # Kit must initialize before importing scene modules that depend on its extensions.
    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": True, "width": 1280, "height": 720, "renderer": "PathTracing", "samples_per_pixel_per_frame": 2}
    )
    startup = time.perf_counter() - started
    try:
        from examples.butter_spreading.isaac_scene import render

        render(
            app,
            args.run.resolve(),
            args.samples,
            args.background_fill == "on",
            args.frozen_frame,
            args.back_view,
            started,
            startup,
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
