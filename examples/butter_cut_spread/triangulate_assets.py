"""Export exact Blender triangulations of annotated ZIP polygon meshes."""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True, help="ZIP asset directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Triangulated archive directory.")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label in ("board", "bread", "knife"):
        (source,) = args.archives.glob(label + "-*.zip")
        with zipfile.ZipFile(source) as archive:
            (member,) = (name for name in archive.namelist() if name.endswith("/geometry.json"))
            geometry = json.loads(archive.read(member))
        records = []
        for index, record in enumerate(geometry["meshes"]):
            mesh = bpy.data.meshes.new(record["name"])
            mesh.from_pydata(record["vertices"], [], record["faces"])
            mesh.calc_loop_triangles()
            faces = np.array([tuple(face.vertices) for face in mesh.loop_triangles])
            filename = f"{label}-{index}.npz"
            np.savez_compressed(args.output_dir / filename, vertices=record["vertices"], faces=faces)
            records.append({"name": record["name"], "file": filename})
            bpy.data.meshes.remove(mesh)
        (args.output_dir / f"{label}.json").write_text(
            json.dumps({"archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "meshes": records}, indent=2)
        )


if __name__ == "__main__":
    main()
