import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import av

if __package__:
    from .teapot_checkpoint import read_checkpoint_info
else:
    from teapot_checkpoint import read_checkpoint_info


def main():
    parser = argparse.ArgumentParser(description="Render four synchronized teapot methods with Genesis ray tracing.")
    parser.add_argument(
        "-t", "--seconds", type=float, help="Simulated duration (default: 32, or the saved replay duration)."
    )
    parser.add_argument("--spp", type=int, default=32, help="Ray-tracing samples per pixel.")
    parser.add_argument("--fps", type=int, help="Output frames per second (default: 25, or the saved replay rate).")
    parser.add_argument("--playback-speed", type=float, help="Playback speed (default: 2, or the saved replay speed).")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("out/teapot_comparison"), help="Videos, images and run metadata."
    )
    parser.add_argument("--compose-only", action="store_true", help="Compose previously rendered method videos.")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent methods; higher values need more GPU memory.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("PBSTF", "PBD", "DFSPH", "WCSPH"),
        default=("PBSTF", "PBD", "DFSPH", "WCSPH"),
        help="Methods to record and compare, in display order.",
    )
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--save-checkpoints", type=Path, help="Save reusable simulation frames under DIR/METHOD.")
    checkpoints.add_argument("--replay-checkpoints", type=Path, help="Render cached simulation frames from DIR/METHOD.")
    parser.add_argument("--checkpoint-every", type=int, help="Save every N simulation steps (default: video cadence).")
    parser.add_argument(
        "--simulate-only", action="store_true", help="Save checkpoints with particle preview and no video."
    )
    parser.add_argument(
        "--timings", action="store_true", help="Synchronize GPU stage timing and save timing CSV files."
    )
    args = parser.parse_args()
    methods = tuple(args.methods)
    if len(set(methods)) != len(methods):
        parser.error("Each method may appear once.")
    if args.simulate_only and (args.save_checkpoints is None or args.compose_only):
        parser.error("--simulate-only requires --save-checkpoints and a simulation run.")
    info = None
    if args.replay_checkpoints is not None:
        info = read_checkpoint_info(args.replay_checkpoints, methods[0])
    if args.seconds is None:
        args.seconds = 32.0 if info is None else info.steps * info.dt
    if args.fps is None:
        args.fps = 25 if info is None else info.fps
    if args.playback_speed is None:
        args.playback_speed = 2.0 if info is None else info.playback_speed
    if args.seconds <= 0.0 or args.spp < 1 or args.fps < 1 or args.playback_speed <= 0.0 or args.jobs < 1:
        parser.error("Duration, samples, frame rate, playback speed and jobs must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scripts = tuple(
        "pbstf_teapot.py" if method == "PBSTF" else "pbd_liquid_teapot.py" if method == "PBD" else "sph_teapot.py"
        for method in methods
    )
    if not args.compose_only:
        with ExitStack() as stack:
            logs = [
                stack.enter_context((args.output_dir / f"{method}.log").open("w", encoding="utf-8"))
                for method in methods
            ]
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=args.jobs))
            runs = []
            for method, script, log in zip(methods, scripts, logs):
                command = [
                    sys.executable,
                    str(Path(__file__).with_name(script)),
                    "--preview",
                    "particles" if args.simulate_only else "render",
                    "--seconds",
                    str(args.seconds),
                    "--spp",
                    str(args.spp),
                    "--fps",
                    str(args.fps),
                    "--playback-speed",
                    str(args.playback_speed),
                    "--output-dir",
                    str(args.output_dir),
                ]
                if not args.simulate_only:
                    command.append("--record")
                if args.save_checkpoints is not None:
                    command.extend(("--save-checkpoints", str(args.save_checkpoints)))
                if args.replay_checkpoints is not None:
                    command.extend(("--replay-checkpoints", str(args.replay_checkpoints)))
                if args.checkpoint_every is not None:
                    command.extend(("--checkpoint-every", str(args.checkpoint_every)))
                if args.timings:
                    command.append("--timings")
                if method in ("DFSPH", "WCSPH"):
                    command.extend(("--pressure-solver", method))
                print(f"Queued {method}; log: {args.output_dir / (method + '.log')}", flush=True)
                runs.append(pool.submit(subprocess.run, command, stdout=log, stderr=subprocess.STDOUT, check=True))
            for run in runs:
                run.result()

    with np.load(args.output_dir / f"{methods[0]}_initial.npz") as reference:
        for method in methods[1:]:
            with np.load(args.output_dir / f"{method}_initial.npz") as initial:
                for key in ("positions", "velocities"):
                    if not np.array_equal(initial[key], reference[key]):
                        raise ValueError(f"{method} initial {key} differs from {methods[0]}.")
    if args.simulate_only:
        print(f"Saved simulation checkpoints to {args.save_checkpoints}.")
        return
    is_unstable = tuple(
        (args.output_dir / f"{method}_status.txt").is_file()
        and (args.output_dir / f"{method}_status.txt").read_text(encoding="utf-8").strip() == "unstable"
        for method in methods
    )
    font = ImageFont.truetype(
        str(Path(__file__).parents[2] / "genesis/ext/pyrender/fonts/OpenSans-Semibold.ttf"), size=22
    )
    with ExitStack() as stack:
        sources = [stack.enter_context(av.open(str(args.output_dir / f"{method}.mp4"))) for method in methods]
        streams = [source.streams.video[0] for source in sources]
        if any(stream.average_rate != streams[0].average_rate for stream in streams[1:]):
            raise ValueError("Method video frame rates differ.")
        width, height = streams[0].width, streams[0].height
        if any((stream.width, stream.height) != (width, height) for stream in streams[1:]):
            raise ValueError("Method video resolutions differ.")
        destination = stack.enter_context(av.open(str(args.output_dir / "teapot_comparison.mp4"), mode="w"))
        stream = destination.add_stream("libx264", rate=streams[0].average_rate)
        columns = min(2, len(methods))
        rows = (len(methods) + columns - 1) // columns
        stream.width = columns * width
        stream.height = rows * height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        frame_count = 0
        for frame_idx, frames in enumerate(zip(*(source.decode(video=0) for source in sources), strict=True)):
            canvas = Image.new("RGB", (columns * width, rows * height))
            for method_idx, (method, frame) in enumerate(zip(methods, frames)):
                canvas.paste(frame.to_image(), ((method_idx % columns) * width, (method_idx // columns) * height))
            draw = ImageDraw.Draw(canvas)
            simulated_time = (frame_idx + 1) * args.playback_speed / streams[0].average_rate
            for method_idx, method in enumerate(methods):
                x, y = (method_idx % columns) * width, (method_idx // columns) * height
                draw.rectangle((x, y, x + width, y + 38), fill=(18, 23, 30))
                label = f"{method} | unstable" if is_unstable[method_idx] else method
                draw.text((x + 16, y + 6), label, fill=(240, 245, 250), font=font)
                draw.text(
                    (x + width - 275, y + 6),
                    f"t = {float(simulated_time):05.2f}s  |  {args.playback_speed:g}x",
                    fill=(183, 203, 218),
                    font=font,
                )
            if frame_idx == 0 or (frame_idx + 1) % (args.fps * 4) == 0:
                canvas.save(args.output_dir / f"preview_{frame_idx + 1:04d}.jpg", quality=95)
            encoded_frame = av.VideoFrame.from_image(canvas)
            encoded_frame.pts = frame_idx
            for packet in stream.encode(encoded_frame):
                destination.mux(packet)
            frame_count += 1
        for packet in stream.encode():
            destination.mux(packet)
        expected_frames = round(args.seconds * streams[0].average_rate / args.playback_speed)
        if frame_count != expected_frames:
            raise ValueError(f"Expected {expected_frames} frames, received {frame_count}.")
    print(f"Saved {args.output_dir / 'teapot_comparison.mp4'} ({frame_count} synchronized frames).")


if __name__ == "__main__":
    main()
