"""Run simulation, physics audit, surface reconstruction, and ripple diagnostics."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import ValidationError

from examples.butter_spreading.config import load_config


@dataclass
class PipelineStage:
    name: str
    command: list[str]
    seconds: float
    exit_code: int


@dataclass
class PipelineTiming:
    stages: list[PipelineStage]
    elapsed_wall_s: float


@dataclass(frozen=True)
class PipelineCompletion:
    is_complete: bool
    is_full_trajectory: bool
    is_accepted: bool
    configuration_sha256: str
    state_sha256: str
    source_sha256: dict[str, str]
    artifact_sha256: dict[str, str]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("out/butter-pipeline"), help="New run directory.")
    parser.add_argument("--config", type=Path, help="Partial physical configuration JSON.")
    parser.add_argument("--seed", type=int, help="Override the initial particle sampling seed.")
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu", help="GPU suits the dense particle scene.")
    parser.add_argument("--bread-particle-size", type=float, help="Override bread spacing in metres.")
    parser.add_argument("--butter-particle-size", type=float, help="Override butter spacing in metres.")
    parser.add_argument(
        "-s", "--steps", type=int, help="Short preview step count; omit for a complete dataset episode."
    )
    parser.add_argument("-r", "--record", action="store_true", help="Record the Genesis camera during simulation.")
    args = parser.parse_args()
    try:
        config = load_config(
            args.config,
            seed=args.seed,
            bread_particle_size=args.bread_particle_size,
            butter_particle_size=args.butter_particle_size,
        )
    except (OSError, ValidationError) as error:
        parser.error(str(error))
    if not 0 < config.butter_particle_size <= config.bread_particle_size <= 0.002:
        parser.error("Particle spacings must satisfy 0 < butter <= bread <= 0.002 metres.")
    full_steps = round(4.0 / config.dt)
    if full_steps < 96:
        parser.error("The time step must resolve all 97 trajectory samples.")
    if args.steps is not None and not 1 <= args.steps <= full_steps:
        parser.error("Preview steps must be positive and fit within the four-second motion.")
    is_preview = args.steps is not None and args.steps < full_steps
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty for a new pipeline run.")
    output.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository)
    commands = [
        (
            "simulation",
            [
                sys.executable,
                "-u",
                "-m",
                "examples.butter_spreading.demo",
                "--save-trajectory",
                "--save-state",
                "--progress-every",
                "10000",
                "--backend",
                args.backend,
                "--output-dir",
                str(output),
                *(["--config", str(args.config.resolve())] if args.config is not None else []),
                "--seed",
                str(config.seed),
                "--bread-particle-size",
                str(config.bread_particle_size),
                "--butter-particle-size",
                str(config.butter_particle_size),
                *(["--steps", str(args.steps)] if args.steps is not None else []),
                *(["--record"] if args.record else []),
            ],
        ),
        (
            "audit",
            [
                sys.executable,
                "-m",
                "examples.butter_spreading.audit",
                "--state",
                str(output / "mpm-state.npz"),
                "--output",
                str(output / "physics-audit.json"),
                *(["--preview"] if is_preview else []),
            ],
        ),
        (
            "surface",
            [
                sys.executable,
                "-m",
                "examples.butter_spreading.surface",
                "--state",
                str(output / "mpm-state.npz"),
                "--output",
                str(output / "mpm-surface.npz"),
            ],
        ),
    ]
    if not is_preview:
        commands.append(
            (
                "ripples",
                [
                    sys.executable,
                    "-m",
                    "examples.butter_spreading.ripples",
                    "--state",
                    str(output / "mpm-state.npz"),
                    "--surface",
                    str(output / "mpm-surface.npz"),
                    "--output",
                    str(output / "surface-ripples.json"),
                ],
            )
        )
    started = time.perf_counter()
    timing = PipelineTiming([], 0.0)
    for name, command in commands:
        stage_started = time.perf_counter()
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=repository, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        timing.stages.append(PipelineStage(name, command, time.perf_counter() - stage_started, result.returncode))
        timing.elapsed_wall_s = time.perf_counter() - started
        (output / "full-pipeline-timing.json").write_text(json.dumps(asdict(timing), indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {timing.stages[-1].seconds:.3f}s, exit_code={result.returncode}", flush=True)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
    artifacts = ["completion.json", "physics-audit.json", "mpm-surface.npz", "mpm-surface.json"]
    if not is_preview:
        artifacts.append("surface-ripples.json")
    if args.record:
        artifacts.append("butter-spreading.mp4")
    completion = PipelineCompletion(
        is_complete=True,
        is_full_trajectory=not is_preview,
        is_accepted=not is_preview,
        configuration_sha256=hashlib.sha256((output / "run-config.json").read_bytes()).hexdigest(),
        state_sha256=hashlib.sha256((output / "mpm-state.npz").read_bytes()).hexdigest(),
        source_sha256={
            path.relative_to(repository).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.py")
        },
        artifact_sha256={name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in artifacts},
    )
    (output / "pipeline-completion.json").write_text(json.dumps(asdict(completion), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
