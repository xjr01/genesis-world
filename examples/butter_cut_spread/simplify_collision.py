"""Offline Blender decimation in the supplied meter-scale frame; no scene-axis conversion."""

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--faces", type=int, default=2000)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with np.load(args.source) as archive:
        mesh = bpy.data.meshes.new("Collision")
        mesh.from_pydata(archive["vertices"].tolist(), [], archive["faces"].tolist())
    obj = bpy.data.objects.new("Collision", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    modifier = obj.modifiers.new("Offline collision reduction", "DECIMATE")
    modifier.ratio = min(1.0, args.faces / len(mesh.polygons))
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    mesh = obj.data
    mesh.calc_loop_triangles()
    np.savez(
        args.output,
        vertices=np.array([tuple(vertex.co) for vertex in mesh.vertices]),
        faces=np.array([tuple(face.vertices) for face in mesh.loop_triangles]),
    )
    print(f"Prepared {len(mesh.loop_triangles)} collision triangles.", flush=True)


if __name__ == "__main__":
    main()
