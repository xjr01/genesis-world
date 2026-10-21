"""Export shirt material labels and semantic landmarks."""

import argparse
from pathlib import Path

from .core.material import save_atlas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positive-x-side", choices=("unknown", "wearer_left", "wearer_right"), default="unknown")
    args = parser.parse_args()
    save_atlas(args.mesh, args.output_dir, args.positive_x_side)
    print(f"Saved {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
