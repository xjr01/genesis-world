"""Render a recorded asset cutting case in Isaac with the approved butter appearance."""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--camera-target", type=float, nargs=3, help="World-space look-at point in meters.")
    parser.add_argument("--focal-length", type=float, help="Camera focal length in millimeters.")
    parser.add_argument("--camera-eye", type=float, nargs=3, help="World-space camera position in meters.")
    parser.add_argument("--frame-indices", type=int, nargs="+", help="Recorded frames for close-up inspection.")
    parser.add_argument("--output-name", help="Separate filename stem for an inspection render.")
    args = parser.parse_args()
    if args.output_name is not None and Path(args.output_name).name != args.output_name:
        parser.error("Output name must be a filename stem.")
    started = time.perf_counter()
    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": True, "width": 1280, "height": 720, "renderer": "PathTracing", "disable_viewport_updates": True}
    )
    from examples.butter_cut_spread.asset_scene import render

    completed = False
    try:
        render(
            app,
            args.run.resolve(),
            args.samples,
            args.preview,
            started,
            args.camera_target,
            args.focal_length,
            args.camera_eye,
            args.frame_indices,
            args.output_name,
        )
        completed = True
    finally:
        if not completed:
            traceback.print_exc()
            sys.stderr.flush()
            sys.stdout.flush()
            os._exit(1)
        app.close()


if __name__ == "__main__":
    main()
