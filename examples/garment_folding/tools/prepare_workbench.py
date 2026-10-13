"""Prepare a cloth snapshot and robot start state for viewing or IK."""

import argparse
from pathlib import Path

from .core.imports import import_observation, import_run_manifest
from .core.preparation import prepare_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Native garment-folding .pt checkpoint; omit for initial pose.")
    parser.add_argument("--trajectory", type=Path, help="Trajectory used to produce the checkpoint.")
    parser.add_argument("--asset-root", type=Path, help="External garment-folding asset root.")
    parser.add_argument("--mesh", choices=("8k", "13k", "55k"), default="55k")
    parser.add_argument("--output-dir", type=Path, required=True, help="New saved-state directory.")
    imports = parser.add_mutually_exclusive_group()
    imports.add_argument("--import-config", type=Path, help="Upstream observation workbench JSON.")
    imports.add_argument(
        "--run-manifest", type=Path, help="Run manifest; native execution imports require --checkpoint."
    )
    parser.add_argument("--source-root", type=Path, help="Root for portable upstream config asset paths.")
    parser.add_argument(
        "--start-frame", type=int, help="Observation import anchor; use replay first frame for editing."
    )
    args = parser.parse_args()
    if args.import_config is not None:
        if args.checkpoint is not None or args.trajectory is not None:
            parser.error("Observation import uses the config replay; use native preparation for checkpoints.")
        import_observation(args.import_config, args.output_dir, args.source_root, args.start_frame)
    elif args.run_manifest is not None:
        import_run_manifest(args.run_manifest, args.checkpoint, args.output_dir, args.source_root)
    else:
        prepare_workspace(args.output_dir, args.checkpoint, args.trajectory, args.asset_root, args.mesh)


if __name__ == "__main__":
    main()
