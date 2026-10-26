"""Run the bundled authored scene with Genesis physics and optional Isaac replay."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def unpack_scene(output):
    """Verify and unpack portable scene inputs into a new preparation directory."""
    assets = Path(__file__).parent / "assets"
    manifest = json.loads((assets / "manifest.json").read_text())
    archive_path = assets / manifest["file"]
    with archive_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["sha256"]:
            raise ValueError("Bundled scene checksum failed; restore assets/scene.zip from the repository.")
    if output.exists():
        raise ValueError("Preparation requires a new directory; choose a new --prepared path.")
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if not (output / member.filename).resolve().is_relative_to(output.resolve()):
                raise ValueError("Scene archive contains a path outside its destination.")
        output.mkdir(parents=True)
        archive.extractall(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="New simulation output directory.")
    parser.add_argument("--prepared", type=Path, help="Preparation directory; defaults to RUN-prepared.")
    parser.add_argument("--stage", choices=("all", "prepare", "simulate", "render"), default="all")
    parser.add_argument("--isaac-python", type=Path, help="Python executable in the separate Isaac Sim environment.")
    parser.add_argument(
        "--physics-only", action="store_true", help="Run without Isaac surface reconstruction or replay."
    )
    parser.add_argument(
        "--preview", action="store_true", help="Reconstruct and render five states; physics stays complete."
    )
    parser.add_argument("--material", type=Path, default=Path(__file__).parent / "configs/asset_firm_smooth.json")
    parser.add_argument("--particle-size", type=float, default=0.0006, help="Preparation sampling distance in meters.")
    parser.add_argument("--dt", type=float, default=0.00005, help="Preparation integration step in seconds.")
    parser.add_argument("--workers", type=int, default=3, help="Surface workers; more workers require more RAM.")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--duration", type=float, help="Short simulation smoke check; skips full-trajectory audits.")
    args = parser.parse_args()
    has_render = args.stage in ("all", "render") and not args.physics_only
    if has_render and (args.isaac_python is None or not args.isaac_python.is_file()):
        parser.error("Rendering requires --isaac-python pointing to an installed Isaac Sim Python executable.")
    if args.stage == "render" and args.physics_only:
        parser.error("--physics-only cannot be combined with --stage render.")
    if not args.material.is_file() or args.workers < 1 or args.samples < 1:
        parser.error("Material must exist, and workers and samples must be positive.")
    if any(not math.isfinite(value) or value <= 0 for value in (args.dt, args.particle_size)):
        parser.error("Preparation time step and particle spacing must be finite and positive.")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("Smoke-check duration must be finite and positive.")
    run = args.run.resolve()
    if args.stage in ("all", "simulate") and run.exists() and (not run.is_dir() or any(run.iterdir())):
        parser.error("Simulation requires a new or empty --run directory.")
    prepared = args.prepared.resolve() if args.prepared else run.with_name(run.name + "-prepared")
    material = args.material.resolve()
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    commands = []
    if args.stage in ("all", "prepare"):
        unpack_scene(prepared)
        commands.append(
            [
                sys.executable,
                "-m",
                "examples.butter_cut_spread.asset_demo",
                "--assets",
                str(prepared),
                "--run",
                str(prepared),
                "--prepare-only",
                "--material",
                str(material),
                "--particle-size",
                str(args.particle_size),
                "--dt",
                str(args.dt),
            ]
        )
    if args.stage in ("all", "simulate"):
        command = [
            sys.executable,
            "-m",
            "examples.butter_cut_spread.asset_demo",
            "--assets",
            str(prepared),
            "--run",
            str(run),
            "--material",
            str(material),
        ]
        if args.duration is not None:
            command += ["--duration", str(args.duration)]
        commands.append(command)
        commands.append([sys.executable, "-m", "examples.butter_cut_spread.asset_particles", "--run", str(run)])
        if args.duration is None:
            for module in ("asset_audit", "aligned_audit", "shape_audit"):
                commands.append([sys.executable, "-m", f"examples.butter_cut_spread.{module}", "--run", str(run)])
    if has_render:
        preview = ["--preview"] if args.preview else []
        commands.append(
            [
                sys.executable,
                "-m",
                "examples.butter_cut_spread.asset_prepare",
                "--run",
                str(run),
                "--workers",
                str(args.workers),
                "--coating-voxel",
                "0.0001",
                *preview,
            ]
        )
        commands.append(
            [
                str(args.isaac_python.resolve()),
                "-m",
                "examples.butter_cut_spread.asset_render",
                "--run",
                str(run),
                "--samples",
                str(args.samples),
                *preview,
            ]
        )
        if not args.preview and args.duration is None:
            commands.append([sys.executable, "-m", "examples.butter_cut_spread.aligned_validate", "--run", str(run)])
    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == "__main__":
    main()
