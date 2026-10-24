"""Export selected evaluated Blender meshes in meters, including object and scene transforms."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    bpy.ops.wm.open_mainfile(filepath=str(args.file.resolve()))
    args.output.mkdir(parents=True, exist_ok=True)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scale = bpy.context.scene.unit_settings.scale_length
    records = []
    for index, name in enumerate(args.names):
        obj = bpy.data.objects[name].evaluated_get(depsgraph)
        mesh = obj.to_mesh()
        mesh.calc_loop_triangles()
        vertices = np.array([tuple(obj.matrix_world @ vertex.co) for vertex in mesh.vertices]) * scale
        faces = np.array([tuple(face.vertices) for face in mesh.loop_triangles])
        filename = f"mesh-{index}.npz"
        np.savez(args.output / filename, vertices=vertices, faces=faces)
        records.append(
            {"name": name, "file": filename, "bounds_m": [vertices.min(0).tolist(), vertices.max(0).tolist()]}
        )
        obj.to_mesh_clear()
    (args.output / "export.json").write_text(
        json.dumps(
            {
                "source": str(args.file.resolve()),
                "source_sha256": hashlib.sha256(args.file.read_bytes()).hexdigest(),
                "meters_per_unit": scale,
                "meshes": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
