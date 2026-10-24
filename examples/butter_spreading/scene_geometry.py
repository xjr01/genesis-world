"""Geometry and textured USD materials for the tabletop scene."""

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt


def _material(stage, name, color, roughness=0.65, metallic=0):
    mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _textured_material(stage, name, image_path, color, roughness):
    mat = _material(stage, name, color, roughness)
    shader = UsdShade.Shader.Get(stage, f"/World/Looks/{name}/Preview")
    uv = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/UV")
    uv.CreateIdAttr("UsdPrimvarReader_float2")
    uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(image_path)))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
    return mat


def _mesh(stage, path, points, faces, material, uvs=None):
    shape = UsdGeom.Mesh.Define(stage, path)
    shape.CreateSubdivisionSchemeAttr("none")
    shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points, np.float32)))
    shape.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.asarray([len(f) for f in faces], np.int32)))
    shape.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray([i for f in faces for i in f], np.int32)))
    if uvs is not None:
        p = UsdGeom.PrimvarsAPI(shape.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        p.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uvs, np.float32)))
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material)
    return shape


def _smooth_outline(control, samples=5):
    result = []
    count = len(control)
    for i in range(count):
        p0, p1, p2, p3 = [np.asarray(control[j % count], float) for j in (i - 1, i, i + 1, i + 2)]
        for s in range(samples):
            t = s / samples
            result.append(
                tuple(
                    0.5
                    * (
                        2 * p1
                        + (-p0 + p2) * t
                        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
                    )
                )
            )
    return result


def _rounded_rect(width, height, radius, n=9):
    pts = []
    for cx, cy, start in [
        (width / 2 - radius, height / 2 - radius, 0),
        (-width / 2 + radius, height / 2 - radius, 90),
        (-width / 2 + radius, -height / 2 + radius, 180),
        (width / 2 - radius, -height / 2 + radius, 270),
    ]:
        for t in np.linspace(start, start + 90, n, endpoint=False):
            # winding is immaterial for preview surface, but continuous outline matters
            rad = np.deg2rad(t)
            pts.append((cx + radius * np.cos(rad), cy + radius * np.sin(rad)))
    return pts
