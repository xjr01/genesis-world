"""Stable material labels and semantic landmarks for the authored shirt mesh."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .plan_io import write_json
from .trajectory import file_hash


@dataclass(frozen=True)
class MaterialAtlas:
    canonical_uv: np.ndarray
    width_band: np.ndarray
    length_zone: np.ndarray
    surface_layer: np.ndarray

    def payload(self):
        """Serialize label arrays for the browser protocol."""
        return {
            "canonical_uv": self.canonical_uv.tolist(),
            "width_band": self.width_band.tolist(),
            "length_zone": self.length_zone.tolist(),
            "surface_layer": self.surface_layer.tolist(),
        }


def material_atlas(vertices: np.ndarray) -> MaterialAtlas:
    """Label width, hem-to-collar length and authored surface identity."""
    minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-12)
    uv = (vertices[:, (0, 2)] - minimum[[0, 2]]) / span[[0, 2]]
    low, high = np.quantile(vertices[:, 0], (0.35, 0.65))
    width = np.where(vertices[:, 0] <= low, 0, np.where(vertices[:, 0] >= high, 2, 1))
    length = np.where(uv[:, 1] <= 0.25, 0, np.where(uv[:, 1] >= 0.72, 2, 1))
    surface = np.where(vertices[:, 1] <= np.median(vertices[:, 1]), 0, 1)
    return MaterialAtlas(uv, width, length, surface)


def save_atlas(mesh_path: Path, output: Path, positive_x_side="unknown"):
    """Export material labels and paired hem, collar and sleeve-tip landmarks."""
    if positive_x_side not in ("unknown", "wearer_left", "wearer_right"):
        raise ValueError("Expected a calibrated wearer side or unknown.")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    vertices = mesh.vertices
    atlas = material_atlas(vertices)
    minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-12)
    hem = vertices[atlas.canonical_uv[:, 1] <= 0.25]
    low, high = np.quantile(hem[:, 0], (0.05, 0.95))
    targets = [
        ("hem_negative", (low, minimum[2])),
        ("hem_center", ((low + high) / 2, minimum[2])),
        ("hem_positive", (high, minimum[2])),
        ("collar_center", (0, maximum[2])),
    ]
    for name, mask in (
        ("negative_x_sleeve_tip", vertices[:, 0] <= np.quantile(vertices[:, 0], 0.02)),
        ("positive_x_sleeve_tip", vertices[:, 0] >= np.quantile(vertices[:, 0], 0.98)),
    ):
        targets.append((name, tuple(np.median(vertices[mask][:, (0, 2)], axis=0))))
    keypoints = {}
    for name, target in targets:
        distances = np.linalg.norm((vertices[:, (0, 2)] - target) / span[[0, 2]], axis=1)
        record = {"target_raw_xz": list(target)}
        for layer, surface_name in ((0, "table_up"), (1, "table_facing")):
            candidates = np.flatnonzero(atlas.surface_layer == layer)
            if not len(candidates):
                raise ValueError("Semantic surface pairing requires both authored surfaces.")
            vertex = candidates[np.argmin(distances[candidates])]
            record[f"{surface_name}_vertex"] = vertex.item()
            record[f"{surface_name}_raw_xyz"] = vertices[vertex].tolist()
        keypoints[name] = record
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "atlas.npz",
        rest_vertex_id=np.arange(len(vertices)),
        rest_raw_xyz=vertices,
        canonical_uv=atlas.canonical_uv,
        width_band=atlas.width_band,
        length_zone=atlas.length_zone,
        surface_layer=atlas.surface_layer,
    )
    negative = (
        "unknown"
        if positive_x_side == "unknown"
        else "wearer_right"
        if positive_x_side == "wearer_left"
        else "wearer_left"
    )
    write_json(
        output / "atlas.json",
        {
            "mesh_sha256": file_hash(mesh_path),
            "vertices": len(vertices),
            "faces": len(mesh.faces),
            "raw_x_mapping": {"positive_x": positive_x_side, "negative_x": negative},
            "integer_labels": {
                "width_band": ["negative_x_outer", "center", "positive_x_outer"],
                "length_zone": ["hem", "torso", "shoulder_collar"],
                "surface_layer": ["table_up", "table_facing"],
            },
            "keypoints": keypoints,
            "surface_semantics": "Authored material identity; inspect current folded layers in world space",
        },
    )
