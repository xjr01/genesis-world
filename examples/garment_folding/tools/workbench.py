"""Open the local cloth viewer and grasp-point picker."""

import argparse
from pathlib import Path

from .core.browser import WorkbenchData, WorkbenchServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Directory created by prepare_workbench.")
    parser.add_argument("--replay", type=Path, help="Matching native garment-folding replay for frame inspection.")
    parser.add_argument("--port", type=int, default=0, help="Local HTTP port; default selects a free port.")
    args = parser.parse_args()
    data = WorkbenchData.load(args.workspace.resolve(), args.replay)
    with WorkbenchServer(("127.0.0.1", args.port), data) as server:
        print(f"Workbench: http://127.0.0.1:{server.server_address[1]}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
