"""Prepare compact support collision, initial particles, trajectories and native SDF caches offline."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import igl
import numpy as np
import trimesh

from examples.butter_cut_spread.asset_init import support_heights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--dish-faces", type=int, default=2000)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    if output.exists():
        parser.error("Use a new prepared-asset directory.")
    if args.dish_faces < 100:
        parser.error("Dish triangle budget must be at least 100.")
    shutil.copytree(args.assets, output)
    source = trimesh.load_mesh(output / "dish.obj", process=False)
    work = output / "offline"
    work.mkdir()
    np.savez(work / "dish-source.npz", vertices=source.vertices, faces=source.faces)
    subprocess.run(
        [
            str(args.blender),
            "--background",
            "--python",
            str(Path(__file__).with_name("simplify_collision.py")),
            "--",
            "--source",
            str(work / "dish-source.npz"),
            "--output",
            str(work / "dish-collision.npz"),
            "--faces",
            str(args.dish_faces),
        ],
        check=True,
    )
    with np.load(work / "dish-collision.npz") as data:
        proxy = trimesh.Trimesh(data["vertices"], data["faces"], process=False)
    if not proxy.is_volume:
        raise ValueError("Simplified tray must remain a closed positive-volume surface.")
    config = json.loads((output / "config.json").read_text())
    center, size = np.array(config["butter_center"]), np.array(config["butter_size"])
    x, y = np.meshgrid(
        np.linspace(center[0] - size[0] / 2 - 0.004, center[0] + size[0] / 2 + 0.004, 101),
        np.linspace(center[1] - size[1] / 2 - 0.004, center[1] + size[1] / 2 + 0.004, 41),
    )
    # Avoid rays exactly on triangulation edges; sub-micron offset is fixed and shared by both meshes.
    xy = np.column_stack((x.reshape(-1), y.reshape(-1))) + (1.237e-7, 3.931e-7)
    error = support_heights(proxy, xy) - support_heights(source, xy)
    squared_distance, _, _ = igl.point_mesh_squared_distance(source.vertices, proxy.vertices, proxy.faces)
    reverse_squared, _, _ = igl.point_mesh_squared_distance(proxy.vertices, source.vertices, source.faces)
    surface_error = max(np.sqrt(squared_distance.max()), np.sqrt(reverse_squared.max()))
    if abs(error).max() > 0.00005 or surface_error > 0.0002:
        raise ValueError("Tray proxy exceeds the 0.05 mm support or 0.2 mm sampled surface tolerance.")
    proxy.export(output / "dish-collision.obj")
    report = {
        "source_triangles": len(source.faces),
        "collision_triangles": len(proxy.faces),
        "support_max_error_m": float(abs(error).max()),
        "support_rms_error_m": float(np.sqrt(np.mean(error**2))),
        "sampled_surface_max_error_m": float(surface_error),
        "scope": "Offline collision proxy; original display mesh and actual support elevation retained.",
    }
    (output / "collision-validation.json").write_text(json.dumps(report, indent=2))
    assets = json.loads((output / "assets.json").read_text())
    assets.update(
        dish_collision_file="dish-collision.obj",
        dish_sdf_cell_size_m=0.001,
        dish_sdf_max_res=256,
        board_sdf_cell_size_m=0.001,
        board_sdf_max_res=384,
    )
    (output / "assets.json").write_text(json.dumps(assets, indent=2))
    offline_seconds = time.perf_counter() - started
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.butter_cut_spread.asset_demo",
            "--assets",
            str(output),
            "--run",
            str(output),
            "--prepare-only",
        ],
        check=True,
    )
    path = output / "runtime-preparation.json"
    prepared = json.loads(path.read_text())
    prepared["collision_preparation_seconds"] = offline_seconds
    prepared["total_preparation_seconds"] = time.perf_counter() - started
    path.write_text(json.dumps(prepared, indent=2))
    print(
        json.dumps({"collision": report, "total_preparation_seconds": prepared["total_preparation_seconds"]}, indent=2)
    )


if __name__ == "__main__":
    main()
