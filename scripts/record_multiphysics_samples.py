"""Record one simulated second from each normalized multiphysics scenario.

Each scenario runs in a fresh process because Genesis initialization is process-global. A
per-scenario wall-clock timeout includes imports, kernel compilation, scene construction,
simulation, rendering and video finalization.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCENARIOS = (
    ("table_wiping", "examples.multiphysics.table_wiping.run", 500),
    ("coffee_water", "examples.multiphysics.coffee_water.run", 500),
    ("litter_scoop", "examples.multiphysics.litter_scoop.run", 60),
    ("garment_folding", "examples.multiphysics.garment_folding.run", 50),
    ("butter_spreading", "examples.multiphysics.butter_spreading.run", 28_572),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("out/one_second_recordings"))
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--scenario", choices=tuple(item[0] for item in SCENARIOS), action="append")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    repository = Path(__file__).resolve().parents[1]
    output = (repository / args.output).resolve() if not args.output.is_absolute() else args.output
    output.mkdir(parents=True, exist_ok=True)
    cache = repository / ".runtime-cache"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(repository),
            "QD_OFFLINE_CACHE_FILE_PATH": str(cache / "one-second-recordings" / "quadrants"),
            "GS_CACHE_FILE_PATH": str(cache / "genesis"),
            "TEMP": str(cache / "tmp"),
            "TMP": str(cache / "tmp"),
        }
    )
    results = []
    results_path = output / "results.json"
    selected = SCENARIOS if args.scenario is None else tuple(item for item in SCENARIOS if item[0] in args.scenario)
    for name, module, steps in selected:
        scenario_output = output / name
        scenario_output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-u",
            "-m",
            module,
            "--steps",
            str(steps),
            "--record",
            "--output",
            str(scenario_output),
        ]
        log_path = output / f"{name}.log"
        print(f"START {name}: steps={steps}, timeout={args.timeout_seconds:.0f}s", flush=True)
        started = time.perf_counter()
        timed_out = False
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=repository,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                returncode = process.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                returncode = process.wait()
        elapsed = time.perf_counter() - started
        video_paths = list(scenario_output.glob("*.mp4"))
        if timed_out:
            for path in video_paths:
                path.unlink(missing_ok=True)
            video_paths = []
        videos = [str(path) for path in video_paths]
        result = {
            "scenario": name,
            "steps": steps,
            "target_simulated_seconds": 1.0,
            "timeout_seconds": args.timeout_seconds,
            "wall_seconds": elapsed,
            "status": "TIMEOUT" if timed_out else ("PASS" if returncode == 0 and videos else "FAIL"),
            "returncode": returncode,
            "log": str(log_path),
            "videos": videos,
        }
        results.append(result)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"END {name}: status={result['status']}, wall_seconds={elapsed:.3f}", flush=True)

    failed = [result for result in results if result["status"] != "PASS"]
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
