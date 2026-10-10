"""Export accepted IK commands for the existing folding entry point."""

import argparse
from pathlib import Path

from .core.runtime import export_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Prepared state used by IK.")
    parser.add_argument("--compiled", type=Path, required=True, help="Directory with accepted IK output.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New trajectory and launcher directory.")
    args = parser.parse_args()
    export_trajectory(args.workspace, args.compiled, args.output_dir)


if __name__ == "__main__":
    main()
