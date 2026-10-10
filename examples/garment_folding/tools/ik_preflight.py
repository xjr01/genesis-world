"""Check a TCP path against the actual robot start state."""

import argparse
from pathlib import Path

from .core.ik_options import IKOptions
from .core.preflight import run_preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Saved state from prepare_workbench.")
    parser.add_argument("--compiled", type=Path, required=True, help="TCP output from compile_plan or edit_trajectory.")
    parser.add_argument("--options", type=Path, help="JSON hand, orientation and table-clearance settings.")
    args = parser.parse_args()
    options = None if args.options is None else IKOptions.model_validate_json(args.options.read_text(encoding="utf-8"))
    run_preflight(args.workspace, args.compiled, options)
    print(f"IK accepted: {args.compiled.resolve()}")


if __name__ == "__main__":
    main()
