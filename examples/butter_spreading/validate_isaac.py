"""Check spreading replay provenance, frame coverage, scene coordinates and bundled appearance."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np


@dataclass(frozen=True)
class VideoResult:
    filename: str
    frames: int
    elapsed_seconds: float
    max_scene_error_m: float
    distinct_images: int


@dataclass(frozen=True)
class Validation:
    is_passed: bool
    state_sha256: str
    surface_sha256: str
    bread_sha256: str
    appearance_profile_sha256: str
    matches_reference: bool | None
    videos: list[VideoResult]
    code_sha256: dict[str, str]
    scope: str


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Completed render report for material and lighting comparison.")
    args = parser.parse_args()
    run = args.run.resolve()
    inputs = json.loads((run / "isaac-inputs.json").read_text(encoding="utf-8"))
    for name, key in (
        ("mpm-state.npz", "state_sha256"),
        ("mpm-surface.npz", "surface_sha256"),
        ("bread-surface.npz", "bread_sha256"),
    ):
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != inputs[key]:
            raise ValueError(f"Modified input: {name}")
    profile = Path(__file__).parent / "materials/appearance.json"
    profile_hash = hashlib.sha256(profile.read_bytes()).hexdigest()
    material = json.loads(profile.read_bytes())["material"]
    with np.load(run / "mpm-state.npz") as state:
        times, fps = state["time_s"], int(state["fps"])
    reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
    results = []
    main_report = None
    for suffix in ("", "-back"):
        name = f"isaac-spreading{suffix}"
        report_path = run / f"{name}.json"
        if suffix and not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report["is_complete"] or not report["is_full_trajectory"] or len(report["frames"]) != len(times):
            raise ValueError("Render does not cover the complete trajectory.")
        for key in ("state_sha256", "surface_sha256", "bread_sha256"):
            if report[key] != inputs[key]:
                raise ValueError(f"Render provenance mismatch: {key}")
        if report["appearance_profile_sha256"] != profile_hash or report["butter_appearance"] != material:
            raise ValueError("Butter material differs from the bundled appearance profile.")
        appearance_keys = (
            "butter_appearance",
            "renderer",
            "render_metres_per_unit",
            "max_volume_bounces",
            "ray_offset_m",
            "samples",
            "auto_exposure",
            "tone_mapping_operator",
            "background_fill",
            "dome_intensity",
            "key_intensity",
            "fill_intensity",
            "front_intensity",
        )
        for compared in (reference, main_report):
            if compared is not None and any(report[key] != compared[key] for key in appearance_keys):
                raise ValueError("Appearance or rendering settings differ between the compared scenes.")
        if main_report is None:
            main_report = report
        error = 0.0
        for index, frame in enumerate(report["frames"]):
            if frame["source_frame"] != index or frame["time_s"] != times[index]:
                raise ValueError("Source frame order or time changed.")
            error = max(error, frame["butter_error_m"], frame["bread_error_m"], frame["blade_error_m"])
        if error > 1e-8:
            raise ValueError("Rendered geometry differs from recorded geometry.")
        hashes = set()
        count = 0
        with av.open(str(run / f"{name}.mp4")) as container:
            if float(container.streams.video[0].average_rate) != fps:
                raise ValueError("Video frame rate differs from trajectory export.")
            for index, frame in enumerate(container.decode(video=0)):
                pixels = frame.to_ndarray(format="rgb24")
                if pixels.shape != (720, 1280, 3) or pixels[:686].std() < 1:
                    raise ValueError("Invalid image buffer in video.")
                if abs(frame.time - index / fps) > 1e-5:
                    raise ValueError("Encoded frame timestamps are discontinuous.")
                hashes.add(hashlib.sha256(pixels[:686].tobytes()).digest())
                count += 1
        if count != len(times) or len(hashes) < 10:
            raise ValueError("Incomplete or static video.")
        results.append(VideoResult(f"{name}.mp4", count, report["elapsed_seconds"], error, len(hashes)))
    source = Path(__file__).resolve().parent
    code_files = [
        source / name
        for name in ("prepare_isaac.py", "render_geometry.py", "render_isaac.py", "isaac_scene.py", "validate_isaac.py")
    ]
    code_files.extend(source / name for name in ("isaac_materials.py", "appearance.py"))
    validation = Validation(
        True,
        inputs["state_sha256"],
        inputs["surface_sha256"],
        inputs["bread_sha256"],
        profile_hash,
        True if reference else None,
        results,
        {str(path.relative_to(source.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files},
        "Accepted spreading trajectory replay. Original butter surface, material-coordinate "
        "deformed bread, unchanged blade dimensions/poses. Thin blade coverage is inactive because "
        "the source has no near-face particles within one particle spacing. No physics rerun.",
    )
    (run / "isaac-validation.json").write_text(json.dumps(asdict(validation), indent=2), encoding="utf-8")
    print(json.dumps(asdict(validation), indent=2))


if __name__ == "__main__":
    main()
