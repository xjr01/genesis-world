"""Prepare portable tabletop assets and an accepted Genesis trajectory."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from examples.butter_spreading.prepare_bread import prepare_bread


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, run = args.source.resolve(), args.output_dir.resolve()
    if run.exists() and any(run.iterdir()):
        parser.error("Use an empty output directory.")
    started = time.perf_counter()
    completion = json.loads((source / "pipeline-completion.json").read_text(encoding="utf-8"))
    audit = json.loads((source / "physics-audit.json").read_text(encoding="utf-8"))
    if not all(completion[k] for k in ("is_complete", "is_full_trajectory", "is_accepted")) or not audit["is_passed"]:
        raise ValueError("Rendering requires an accepted complete trajectory.")
    state_hash = hashlib.sha256((source / "mpm-state.npz").read_bytes()).hexdigest()
    surface_hash = hashlib.sha256((source / "mpm-surface.npz").read_bytes()).hexdigest()
    if state_hash != completion["state_sha256"] or state_hash != audit["state_sha256"]:
        raise ValueError("State hashes disagree.")
    if surface_hash != completion["artifact_sha256"]["mpm-surface.npz"]:
        raise ValueError("Surface hash differs from the accepted pipeline.")
    assets = Path(__file__).parent / "render_assets"
    manifest = json.loads((assets / "manifest.json").read_text())
    for name, checksum in manifest.items():
        if hashlib.sha256((assets / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Bundled asset hash mismatch: {name}")
    run.mkdir(parents=True, exist_ok=True)
    names = (
        "mpm-state.npz",
        "mpm-state.json",
        "mpm-surface.npz",
        "mpm-surface.json",
        "run-config.json",
        "completion.json",
        "pipeline-completion.json",
        "physics-audit.json",
    )
    for name in names:
        shutil.copy2(source / name, run / name)
    target = run / "render-assets"
    shutil.copytree(assets / "textures", target / "textures")
    shutil.copy2(assets / "BREAD_LICENSE.txt", target / "BREAD_LICENSE.txt")
    prepare_bread(assets / "bread-raw.npz", target / "cdmpm-bread-hero-ready.npz")
    files = [run / name for name in names] + [path for path in target.rglob("*") if path.is_file()]
    report = {
        "input_sha256": {
            path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
        "preparation_seconds": time.perf_counter() - started,
    }
    (run / "isaac-inputs.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Prepared tabletop assets and trajectory in {report['preparation_seconds']:.2f} s: {run}")


if __name__ == "__main__":
    main()
