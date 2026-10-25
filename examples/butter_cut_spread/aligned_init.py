"""Align annotated ZIP assets with a measured butter/dish initialization in meters."""

import argparse
import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import trimesh
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation

from examples.butter_cut_spread.asset_init import support_heights
from examples.butter_cut_spread.config import Config
from examples.butter_cut_spread.geometry import rigid_fit


def unpack_asset(archive_path, folder):
    """Extract one annotated asset while retaining its relative texture references."""
    folder.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        for record in archive.infolist():
            parts = Path(record.filename).parts
            if record.is_dir() or parts[0] == "__MACOSX":
                continue
            path = folder.joinpath(*parts[1:]).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError("Archive member escapes the asset folder.")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(record))
    return json.loads((folder / "geometry.json").read_text(encoding="utf-8"))


def triangulated_mesh(folder, label, index):
    """Load Blender's triangulation of the corresponding authored polygon mesh."""
    with np.load(folder / f"{label}-{index}.npz") as archive:
        mesh = trimesh.Trimesh(archive["vertices"], archive["faces"], process=False)
    if not mesh.is_volume:
        raise ValueError(f"Triangulated asset must be watertight: {label}-{index}")
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Measured butter/dish/tool initialization directory.")
    parser.add_argument(
        "--archives", type=Path, required=True, help="Directory containing board-, bread-, knife-*.zip."
    )
    parser.add_argument("--knife-export", type=Path, required=True, help="Evaluated source knife export directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New aligned initialization directory.")
    parser.add_argument("--triangulated", type=Path, required=True, help="Blender triangulations of the ZIP meshes.")
    parser.add_argument(
        "--slice-thickness", type=float, default=0.010, help="Butter beyond the outer blade face, in meters."
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    for path in args.base.iterdir():
        if path.is_file():
            shutil.copy2(path, output / path.name)
    assets = json.loads((output / "assets.json").read_text())
    assets["cutting_roll_degrees"] = -15.0
    assets["cutting_direction"] = -1
    assets["lift_outward_clearance_m"] = 0.003
    config = Config(**json.loads((output / "config.json").read_text()))
    if not 0 < args.slice_thickness < config.butter_size[0] / 2:
        parser.error("Slice thickness must be positive and smaller than half the butter length.")
    geometry = {}
    archives = {}
    for label in ("board", "bread", "knife"):
        (path,) = args.archives.glob(label + "-*.zip")
        geometry[label] = unpack_asset(path, output / "annotated" / label)
        archives[label] = {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        provenance = json.loads((args.triangulated / f"{label}.json").read_text())
        if provenance["archive_sha256"] != archives[label]["sha256"]:
            raise ValueError("Triangulation was generated from another asset archive.")
    board = triangulated_mesh(args.triangulated, "board", 0)
    board_translation = np.array((0.205, 0, 0))
    board.apply_translation(board_translation)
    board.export(output / "board.obj")
    board_stage = Usd.Stage.Open(str(output / "annotated/board/asset.usda"))
    collider = UsdGeom.Mesh(board_stage.GetPrimAtPath("/Asset/collision_0"))
    if not collider or set(collider.GetFaceVertexCountsAttr().Get()) != {3}:
        raise ValueError("The annotated board must provide its triangulated support collider.")
    collider_vertices = np.array(collider.GetPointsAttr().Get())
    collider_transform = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(collider.GetPrim())).T
    board_collision = trimesh.Trimesh(
        collider_vertices @ collider_transform[:3, :3].T + collider_transform[:3, 3],
        np.array(collider.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3),
        process=False,
    )
    board_collision.apply_translation(board_translation)
    if not board_collision.is_volume:
        raise ValueError("Annotated support collider must be a closed positive volume.")
    board_collision.export(output / "board-collision.obj")
    bread = triangulated_mesh(args.triangulated, "bread", 0)
    bread_transform = trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0))
    bread.apply_transform(bread_transform)
    planar_bounds, _ = trimesh.bounds.oriented_bounds_2D(bread.vertices[:, :2])
    yaw = (np.arctan2(planar_bounds[1, 0], planar_bounds[0, 0]) + np.pi / 4) % (np.pi / 2) - np.pi / 4
    straighten = trimesh.transformations.rotation_matrix(yaw, (0, 0, 1))
    bread.apply_transform(straighten)
    bread_transform = straighten @ bread_transform
    bread_translation = np.array((board_translation[0], 0, 0))
    bread_translation[:2] -= bread.bounds.mean(0)[:2]
    support_top = float(support_heights(board_collision, board_translation[None, :2])[0])
    bread_translation[2] = support_top - bread.bounds[0, 2] + 0.00005
    bread.apply_translation(bread_translation)
    bread_transform[:3, 3] = bread_translation
    if not bread.is_volume:
        raise ValueError("Bread must be a closed positive-volume authored mesh.")
    bread.export(output / "bread.obj")
    source_records = json.loads((args.knife_export / "export.json").read_text(encoding="utf-8"))["meshes"]
    source_steel = next(record for record in source_records if record["name"].startswith("Steel |"))
    target_steel = next(record for record in geometry["knife"]["meshes"] if "|Steel |" in record["name"])
    with np.load(args.knife_export / source_steel["file"]) as archive:
        source_vertices = archive["vertices"]
    source_to_zip, fit_error = rigid_fit(source_vertices, np.array(target_steel["vertices"]))
    source_to_local = np.eye(4)
    source_to_local[:3, :3] = np.diag((-1, 1, -1)) @ np.array(assets["knife_source_to_local_rotation"])
    source_to_local[:3, 3] = -source_to_local[:3, :3] @ assets["knife_source_origin_m"]
    zip_to_local = source_to_local @ np.linalg.inv(source_to_zip)
    tool_files = []
    for index, record in enumerate(geometry["knife"]["meshes"]):
        mesh = triangulated_mesh(args.triangulated, "knife", index)
        mesh.apply_transform(zip_to_local)
        filename = f"tool-{index}.obj"
        mesh.export(output / filename)
        tool_files.append(filename)
        if "|Steel |" in record["name"]:
            mesh.export(output / "blade.obj")
            orientation = Rotation.from_euler("x", assets["cutting_roll_degrees"], degrees=True) * Rotation.from_euler(
                "y", -90, degrees=True
            )
            assets["blade_min_z_when_cutting_m"] = float(orientation.apply(mesh.vertices)[:, 2].min())
    top_xy = np.array([(board_translation[0] + x, y) for x in (-0.035, 0, 0.035) for y in (-0.015, 0, 0.015)])
    top_heights = support_heights(bread, top_xy)
    config = replace(
        config,
        duration=8.4,
        cut_end_time=2.9,
        board_normal_stress=0.0,
        board_shear_stress=0.0,
        normal_stress=2000.0,
        shear_stress=1500.0,
        adhesion_range=0.0016,
        adhesion_capture_range=0.0008,
        cut_offset=config.butter_size[0] / 2 - args.slice_thickness - config.blade_size[2] / 2,
        bread_size=tuple(bread.extents),
        bread_center=tuple(bread.bounds.mean(0)),
        lower_bound=(-0.09, -0.08, -0.015),
        upper_bound=(0.29, 0.08, 0.145),
        spread_transition_time=5.65,
        spread_start_offset=-0.031,
        spread_end_offset=0.023,
        spread_start_gap=0.0006,
        spread_end_gap=0.0,
        spread_slip_time=0.00005,
        spread_shear_stress=450.0,
        spread_normal_stress=40.0,
        spread_pitch_degrees=10.0,
        spread_yaw_degrees=180.0,
    )
    assets.update(
        {
            "annotated_archives": archives,
            "tool_meshes": tool_files,
            "knife_zip_to_local": zip_to_local.tolist(),
            "knife_identity_max_error_m": fit_error,
            "board_translation_m": board_translation.tolist(),
            "board_support_top_m": support_top,
            "bread_zip_to_world": bread_transform.tolist(),
            "bread_straightening_degrees": float(np.rad2deg(yaw)),
            "slice_thickness_m": args.slice_thickness,
            "bread_bounds_m": bread.bounds.tolist(),
            "board_bounds_m": board.bounds.tolist(),
            "bread_top_m": float(np.median(top_heights)),
            "bread_top_range_m": [float(top_heights.min()), float(top_heights.max())],
            "is_cut_and_spread": True,
            "scope": "ZIP-authored geometry and textures, rigid transforms in meters, one continuous MPM trajectory.",
        }
    )
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    (output / "assets.json").write_text(json.dumps(assets, indent=2))
    print(json.dumps(assets, indent=2), flush=True)


if __name__ == "__main__":
    main()
