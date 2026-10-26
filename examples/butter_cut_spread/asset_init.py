"""Place meter-scale evaluated assets on their measured support surface."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from examples.butter_cut_spread.config import Config


def load_export(folder):
    records = json.loads((folder / "export.json").read_text(encoding="utf-8"))["meshes"]
    meshes = []
    for record in records:
        with np.load(folder / record["file"]) as data:
            meshes.append(trimesh.Trimesh(data["vertices"], data["faces"], process=True))
    return meshes


def support_heights(mesh, xy):
    origins = np.column_stack((xy, np.full(len(xy), mesh.bounds[1, 2] + 0.1)))
    locations, indices, _ = mesh.ray.intersects_location(
        origins, np.tile((0.0, 0.0, -1.0), (len(xy), 1)), multiple_hits=True
    )
    heights = np.full(len(xy), -np.inf)
    np.maximum.at(heights, indices, locations[:, 2])
    if not np.isfinite(heights).all():
        raise ValueError("Support sampling extends beyond the dish.")
    return heights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        label: json.loads((args.exports / label / "export.json").read_text(encoding="utf-8"))
        for label in ("butter", "dish", "knife")
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    butter = load_export(args.exports / "butter")[0]
    dish = load_export(args.exports / "dish")[0]
    knife = load_export(args.exports / "knife")
    # Long X and shortest Z become the footprint; the middle Y dimension becomes height.
    butter.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
    butter.vertices[:, :2] -= butter.bounds.mean(0)[:2]
    butter.vertices[:, 2] -= butter.bounds[0, 2]
    low, high = butter.bounds[:, :2]
    x, y = np.meshgrid(np.linspace(low[0], high[0], 129), np.linspace(low[1], high[1], 33), indexing="ij")
    top = support_heights(dish, np.column_stack((x.ravel(), y.ravel())))
    bottom_vertices = butter.vertices[butter.vertices[:, 2] < 0.003]
    translation = np.max(support_heights(dish, bottom_vertices[:, :2]) - bottom_vertices[:, 2]) + 0.00005
    butter.apply_translation((0, 0, translation))
    steel = knife[-1]
    patch = steel.vertices[(steel.vertices[:, 0] > 0.140) & (steel.vertices[:, 0] < 0.178)]
    slope, intercept = np.polyfit(patch[:, 0], patch[:, 2], 1)
    tangent = np.array((1.0, 0, slope)) / np.sqrt(1 + slope**2)
    across = np.array((0.0, 1.0, 0.0))
    normal = np.cross(across, tangent)
    basis = np.stack((across, tangent, normal))
    origin = np.array((0.160, 0, slope * 0.160 + intercept))
    for mesh in knife:
        mesh.vertices = (mesh.vertices - origin) @ basis.T
    patch = (patch - origin) @ basis.T
    thickness = np.ptp(patch[:, 2])
    for label, mesh in (("butter", butter), ("dish", dish), ("blade", knife[-1])):
        if not mesh.is_volume:
            raise ValueError(f"{label} must be a closed positive-volume mesh.")
        mesh.export(args.output / f"{label}.obj")
    for index, mesh in enumerate(knife):
        mesh.export(args.output / f"tool-{index}.obj")
    support_origin = butter.bounds[0, :2] - 0.004
    end = butter.bounds[1, :2] + 0.004
    shape = np.ceil((end - support_origin) / 0.001).astype(int) + 1
    spacing = (end - support_origin) / (shape - 1)
    gx, gy = np.meshgrid(*(support_origin[i] + np.arange(shape[i]) * spacing[i] for i in range(2)), indexing="ij")
    heights = support_heights(dish, np.column_stack((gx.ravel(), gy.ravel()))).reshape(tuple(shape))
    np.savez(args.output / "support.npz", origin=support_origin, spacing=spacing, heights=heights)
    config = replace(
        Config(),
        duration=4.0,
        butter_size=tuple(butter.extents),
        butter_center=tuple(butter.bounds.mean(0)),
        blade_size=(0.022, 0.035, float(thickness)),
        blade_bevel_height=0.0,
        cut_offset=float(butter.extents[0] / 2 - 0.005),
        cut_end_time=2.9,
        board_normal_stress=0.0,
        board_shear_stress=0.0,
        lower_bound=(*tuple(dish.bounds[0, :2] - 0.025), -0.015),
        upper_bound=(*tuple(dish.bounds[1, :2] + 0.025), 0.14),
    )
    (args.output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    report = {
        "butter_bounds_m": butter.bounds.tolist(),
        "butter_volume_m3": butter.volume,
        "support_height_range_m": [float(top.min()), float(top.max())],
        "knife_source_origin_m": origin.tolist(),
        "knife_source_to_local_rotation": basis.tolist(),
        "blade_thickness_m": float(thickness),
        "cutting_roll_degrees": -15.0,
        "cutting_direction": -1,
        "lift_outward_clearance_m": 0.003,
        "blade_min_z_when_cutting_m": float(
            (Rotation.from_euler("x", -15, degrees=True) * Rotation.from_euler("y", -90, degrees=True))
            .apply(knife[-1].vertices)[:, 2]
            .min()
        ),
        "tool_meshes": [f"tool-{i}.obj" for i in range(len(knife))],
        "source_exports": str(args.exports.resolve()),
        "scope": "Rigid transforms only; no asset rescaling.",
    }
    (args.output / "assets.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
