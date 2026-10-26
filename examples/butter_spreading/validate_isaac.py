"""Validate complete tabletop video coverage and geometry provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--camera", choices=("scene", "bread"), default="scene")
    args = parser.parse_args()
    run = args.run.resolve()
    name = f"isaac-{args.camera}"
    inputs = json.loads((run / "isaac-inputs.json").read_text())
    report = json.loads((run / f"{name}.json").read_text())
    timing = json.loads((run / f"{name}.timing.json").read_text())
    for path, checksum in inputs["input_sha256"].items():
        if hashlib.sha256((run / path).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Modified input: {path}")
    if report["input_sha256"] != inputs["input_sha256"] or not timing["completed"]:
        raise ValueError("Render provenance or completion mismatch.")
    with np.load(run / "mpm-state.npz") as data:
        times, fps = data["time_s"], int(data["fps"])
    if report["frames"] != len(times) or report["start_frame"] != 0 or not np.array_equal(report["time_s"], times):
        raise ValueError("Render does not cover the complete source trajectory.")
    if len(timing["frames"]) != len(times):
        raise ValueError("Incomplete frame timing.")
    for i, frame in enumerate(timing["frames"]):
        if frame["source_frame"] != i or max(frame["butter_error_m"], frame["blade_error_m"]) > 1e-8:
            raise ValueError("Butter or blade replay differs from its source.")
    if report["effective_render_settings"]["/rtx/pathtracing/totalSpp"] != report["path_tracing_samples"]:
        raise ValueError("Render sample count changed.")
    audit = report["scene_audit"]
    for key in (
        "no_synthetic_contact_film",
        "no_hand_model",
        "flat_blade_present",
        "bread_library_unchanged",
        "task_bread_is_independently_deformable",
        "background_bread_remains_instanced",
    ):
        if not audit[key]:
            raise ValueError(f"Scene invariant failed: {key}")
    frames = report["bread_deformation"]["frames"]
    if len(frames) != len(times) or report["bread_deformation"]["usd_readback_max_error_m"] > 1e-8:
        raise ValueError("Incomplete or inconsistent bread deformation.")
    for i, frame in enumerate(frames):
        if frame["source_frame"] != i or frame["time_s"] != times[i] or frame["minimum_deformation_jacobian"] <= 0:
            raise ValueError("Invalid bread deformation or frame ordering.")
    hashes = set()
    count = 0
    with av.open(str(run / f"{name}.mp4")) as video:
        stream = video.streams.video[0]
        if float(stream.average_rate) != fps:
            raise ValueError("Video frame rate differs from the source.")
        for i, frame in enumerate(video.decode(video=0)):
            pixels = frame.to_ndarray(format="rgb24")
            if pixels.shape[:2] != (report["height"], report["width"]) or pixels.std() < 1:
                raise ValueError("Invalid video image.")
            if abs(frame.time - i / fps) > 1e-5:
                raise ValueError("Video timestamps are discontinuous.")
            hashes.add(hashlib.sha256(pixels.tobytes()).hexdigest())
            count += 1
    if count != len(times) or len(hashes) < 10:
        raise ValueError("Incomplete or static video.")
    validation = {
        "is_passed": True,
        "frames": count,
        "fps": fps,
        "distinct_frames": len(hashes),
        "elapsed_seconds": timing["elapsed_wall_s"],
        "source_sha256": report["source_sha256"],
        "scope": "Recorded trajectory replay; no physics rerun. Bread deformation and scene/video checks.",
    }
    (run / f"{name}-validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
