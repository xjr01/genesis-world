"""Replay annotated textured assets with bread motion driven by persistent MPM points."""

import json
from dataclasses import dataclass

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdLux, Vt
from scipy.spatial import cKDTree


def reference_asset(stage, path, source, transform):
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(str(source))
    preview_prims = [
        child for child in Usd.PrimRange(prim) if child.HasAPI(UsdLux.LightAPI) or child.IsA(UsdGeom.Camera)
    ]
    for child in preview_prims:
        child.SetActive(False)
    UsdGeom.Xformable(prim).AddTransformOp().Set(Gf.Matrix4d(transform.T.tolist()))
    return prim


@dataclass
class BreadReplay:
    mesh: UsdGeom.Mesh
    initial: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    world_vertices: np.ndarray
    local_from_world: np.ndarray
    faces: np.ndarray

    def update(self, positions):
        displacement = ((positions[self.indices] - self.initial[self.indices]) * self.weights[:, :, None]).sum(axis=1)
        world = self.world_vertices + displacement
        local = world @ self.local_from_world[:3, :3].T + self.local_from_world[:3, 3]
        geometry = trimesh.Trimesh(local, self.faces, process=False)
        self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(local.astype(np.float32)))
        self.mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(geometry.vertex_normals.astype(np.float32)))
        self.mesh.SetNormalsInterpolation("vertex")


def build_aligned_assets(stage, run, manifest, initial_bread):
    board_transform = np.eye(4)
    board_transform[:3, 3] = manifest["board_translation_m"]
    board = reference_asset(stage, "/World/Board", run / "annotated/board/visual.usda", board_transform)
    tool = reference_asset(
        stage, "/World/Tool/Annotated", run / "annotated/knife/visual.usda", np.array(manifest["knife_zip_to_local"])
    )
    display_check = {}
    for name, prim, filenames in (("board", board, ("board.obj",)), ("tool", tool, manifest["tool_meshes"])):
        authored = np.concatenate([trimesh.load_mesh(run / filename, process=False).vertices for filename in filenames])
        displayed = []
        cache = UsdGeom.XformCache()
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                vertices = np.array(UsdGeom.Mesh(child).GetPointsAttr().Get())
                matrix = np.array(cache.GetLocalToWorldTransform(child)).T
                displayed.append(vertices @ matrix[:3, :3].T + matrix[:3, 3])
        displayed = np.concatenate(displayed)
        error = float(cKDTree(authored).query(displayed)[0].max())
        if error > 1e-6 or len(displayed) != len(authored):
            raise ValueError(f"Rendered {name} differs from the physical asset initialization.")
        display_check[f"{name}_display_to_physics_max_error_m"] = error
    bread = reference_asset(
        stage, "/World/Bread", run / "annotated/bread/visual.usda", np.array(manifest["bread_zip_to_world"])
    )
    meshes = [UsdGeom.Mesh(prim) for prim in Usd.PrimRange(bread) if prim.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise ValueError("Annotated bread must contain exactly one visual mesh.")
    mesh = meshes[0]
    local_vertices = np.array(mesh.GetPointsAttr().Get())
    transform = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())).T
    world_vertices = local_vertices @ transform[:3, :3].T + transform[:3, 3]
    authored_bread = trimesh.load_mesh(run / "bread.obj", process=False)
    alignment_error = float(cKDTree(authored_bread.vertices).query(world_vertices)[0].max())
    if alignment_error > 1e-6:
        raise ValueError("Rendered bread coordinates differ from the physical asset initialization.")
    display_check["bread_display_to_physics_max_error_m"] = alignment_error
    (run / "asset-display-check.json").write_text(json.dumps(display_check, indent=2))
    distances, indices = cKDTree(initial_bread).query(world_vertices, k=12)
    weights = 1 / np.maximum(distances, 0.0002) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    raw = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    offsets = np.r_[0, np.cumsum(counts)]
    faces = np.concatenate(
        [
            np.column_stack(
                (
                    np.repeat(raw[offsets[i]], count - 2),
                    raw[offsets[i] + 1 : offsets[i + 1] - 1],
                    raw[offsets[i] + 2 : offsets[i + 1]],
                )
            )
            for i, count in enumerate(counts)
        ]
    )
    return BreadReplay(mesh, initial_bread, indices, weights, world_vertices, np.linalg.inv(transform), faces)
