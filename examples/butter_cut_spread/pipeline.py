"""Run continuous simulation, particle-video encoding and physical checks in one Python environment."""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class StageTime:
    stage: str
    seconds: float


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="New output directory for the complete operation.")
    parser.add_argument("--config", type=Path, help="Partial JSON configuration; defaults to the end-slice scenario.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run = args.run.resolve()
    if run.exists() and any(run.iterdir()):
        parser.error("Use a new output directory.")
    run.parent.mkdir(parents=True, exist_ok=True)
    stages = []
    for name in ("demo", "particles", "audit"):
        command = [sys.executable, "-u", "-m", f"examples.butter_cut_spread.{name}", "--run", str(run)]
        if name == "demo" and args.config is not None:
            command.extend(["--config", str(args.config.resolve())])
        started = time.perf_counter()
        log = run.parent / f"{run.name}-{name}.log"
        print(f"Starting {name}: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
        stages.append(StageTime(name, time.perf_counter() - started))
        (run / "pipeline.json").write_text(json.dumps([asdict(stage) for stage in stages], indent=2))
    print(f"Complete particle video: {run / 'cut-spread-particles.mp4'}", flush=True)


if __name__ == "__main__":
    main()
