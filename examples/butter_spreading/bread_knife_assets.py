"""CD-MPM porous bread asset and a thin, flat table-knife reconstruction."""

from pathlib import Path
import carb.tokens
import numpy as np
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter, map_coordinates
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
from examples.butter_spreading.scene_geometry import _material, _mesh, _smooth_outline


def crumb_material(stage, asset_root):
    mat = _material(stage, "CDMPM_Crumb", (0.70, 0.66, 0.56), 0.91)
    shader = UsdShade.Shader.Define(stage, "/World/Looks/CDMPM_Crumb/Scattering")
    # Resolve through the running Kit application, including standalone Isaac
    # installs and Linux environments; never assume a sibling Python folder.
    mdl = Path(carb.tokens.get_tokens_interface().resolve("${kit}")) / "mdl/core/Base/OmniSurface.mdl"
    if not mdl.is_file():
        raise RuntimeError(f"Isaac Kit OmniSurface material is missing: {mdl}")
    shader.CreateImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(str(mdl)), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")
    for name, value in {
        "diffuse_reflection_weight": 1.0,
        "diffuse_reflection_roughness": 0.85,
        "specular_reflection_weight": 0.15,
        "specular_reflection_roughness": 0.72,
        "subsurface_weight": 0.20,
        "subsurface_scale": 0.00065,
    }.items():
        shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
    for name, value in {
        "diffuse_reflection_color": (0.70, 0.66, 0.56),
        "subsurface_transmission_color": (0.86, 0.81, 0.68),
        "subsurface_scattering_color": (0.75, 0.65, 0.5),
    }.items():
        shader.CreateInput(name, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*value))
    shader.CreateInput("enable_diffuse_transmission", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("diffuse_reflection_color_image", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(asset_root / "textures/bread-crumb-v2.png"))
    )
    shader.CreateInput("geometry_normal_image", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(asset_root / "textures/bread-micro-normal.png"))
    )
    shader.CreateInput("geometry_normal_strength", Sdf.ValueTypeNames.Float).Set(0.38)
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return mat


def bread_prototype(stage, asset_root):
    path = "/World/BreadLibrary/PorousSlice"
    if stage.GetPrimAtPath(path):
        return path
    library = UsdGeom.Xform.Define(stage, "/World/BreadLibrary")
    # Reference children without inheriting this off-camera library transform.
    library.AddTranslateOp().Set(Gf.Vec3d(0, 0, -20))
    UsdGeom.Xform.Define(stage, path)
    data = np.load(asset_root / "cdmpm-bread-hero-ready.npz")
    v, f = data["vertices"], data["faces"]
    crumb = crumb_material(stage, asset_root)
    crust = _material(stage, "CDMPM_Crust", (0.54, 0.32, 0.13), 0.83)
    mesh = UsdGeom.Mesh.Define(stage, path + "/Mesh")
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.concatenate(f)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(data["normals"]))
    mesh.SetNormalsInterpolation("vertex")
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
    uv.Set(Vt.Vec2fArray.FromNumpy(data["uv"]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(crumb)
    subset = UsdGeom.Subset.Define(stage, path + "/Mesh/Crust")
    subset.CreateElementTypeAttr("face")
    subset.CreateFamilyNameAttr("materialBind")
    subset.CreateIndicesAttr(Vt.IntArray.FromNumpy(data["crust_faces"]))
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(crust)
    UsdGeom.Subset.SetFamilyType(mesh, "materialBind", "nonOverlapping")
    # Slight baked-crust variation is a vertex colour on the actual shell.
    noise = gaussian_filter(np.random.default_rng(19).normal(size=(64, 64, 32)), 1.4)
    noise /= noise.std()
    coords = ((v - v.min(0)) / (v.max(0) - v.min(0)) * [63, 63, 31]).T
    mottling = 0.045 * map_coordinates(noise, coords, order=1, mode="nearest")
    colors = np.stack((0.54 + mottling, 0.32 + mottling * 0.7, 0.13 + mottling * 0.4), axis=-1)
    color = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("crustColor", Sdf.ValueTypeNames.Color3fArray, "vertex")
    color.Set(Vt.Vec3fArray.FromNumpy(np.clip(colors, 0, 1).astype("float32")))
    reader = UsdShade.Shader.Define(stage, "/World/Looks/CDMPM_Crust/Colour")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("crustColor")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    UsdShade.Shader.Get(stage, "/World/Looks/CDMPM_Crust/Preview").GetInput("diffuseColor").ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    return path


def add_bread(stage, path, center, angle, asset_root, seed=0):
    prototype = bread_prototype(stage, asset_root)
    root = UsdGeom.Xform.Define(stage, path)
    root.GetPrim().GetReferences().AddInternalReference(prototype)
    root.GetPrim().SetInstanceable(True)
    root.AddTranslateOp().Set(Gf.Vec3d(*center))
    root.AddRotateZOp().Set(float(np.rad2deg(angle)))
    if seed:
        root.AddRotateXOp().Set(float((seed % 3 - 1) * 1.3))
        root.AddScaleOp().Set(Gf.Vec3f(1 + (seed % 2) * 0.008, 1 - (seed % 3) * 0.006, 1))
    return root


def _inside(points, polygon):
    x, y = points.T
    hit = np.zeros(len(points), bool)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        hit ^= ((a[1] > y) != (b[1] > y)) & (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1] + 1e-15) + a[0])
    return hit


def _solid_flat_profile(stage, path, outline, thickness, bevel, material):
    outline = np.asarray(outline, dtype=np.float32)
    lo = outline.min(0)
    hi = outline.max(0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 43), np.linspace(lo[1], hi[1], 180))
    q = np.c_[np.concatenate(gx), np.concatenate(gy)]
    q = q[_inside(q, outline)]
    q = np.vstack((q, outline))
    distance = np.full(len(q), np.inf)
    for a, b in zip(outline, np.roll(outline, -1, axis=0)):
        ab = b - a
        t = np.clip(np.sum((q - a) * ab, axis=1) / np.sum(ab * ab), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(q - a - t[:, None] * ab, axis=1))
    # Broad planar face plus a narrow ground bevel: no spoon-like convex crown.
    height = thickness * 0.5 * (0.14 + 0.86 * np.minimum(distance / bevel, 1))
    height += 0.000025 * np.sin(q[:, 1] * 26) * np.minimum(distance / bevel, 1)
    tri = Delaunay(q).simplices
    tri = tri[_inside(q[tri].mean(1), outline)]
    n = len(q)
    boundary = np.arange(n - len(outline), n)
    points = np.vstack((np.c_[q, height], np.c_[q, -height]))
    faces = np.vstack((tri, tri[:, [0, 2, 1]] + n))
    sides = []
    for a, b in zip(boundary, np.roll(boundary, -1)):
        sides.extend([(a, b, b + n), (a, b + n, a + n)])
    faces = np.vstack((faces, np.array(sides)))
    uv = np.c_[(q[:, 0] + 0.013) / 0.026, (q[:, 1] + 0.155) / 0.21]
    mesh = _mesh(stage, path, points, faces, material, np.vstack((uv, uv)))
    normal = np.zeros_like(points)
    fn = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    for k in range(3):
        np.add.at(normal, faces[:, k], fn)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normal.astype("float32")))
    mesh.SetNormalsInterpolation("vertex")
    return mesh


def add_knife(stage, path, steel, initial_pos, collider_thickness=0.012):
    root = UsdGeom.Xform.Define(stage, path)
    blade = _smooth_outline(
        [
            (-0.0068, -0.034),
            (-0.0078, 0.012),
            (-0.0072, 0.037),
            (-0.0053, 0.047),
            (-0.001, 0.051),
            (0.005, 0.049),
            (0.010, 0.041),
            (0.012, 0.023),
            (0.011, -0.014),
            (0.008, -0.031),
            (0.004, -0.042),
            (-0.003, -0.044),
        ],
        9,
    )
    _solid_flat_profile(stage, path + "/FlatBlade", blade, 0.0012, 0.0015, steel)
    handle = _smooth_outline(
        [
            (-0.003, -0.036),
            (-0.0038, -0.055),
            (-0.0057, -0.077),
            (-0.0083, -0.130),
            (-0.0076, -0.145),
            (-0.003, -0.151),
            (0.0035, -0.151),
            (0.008, -0.145),
            (0.0088, -0.129),
            (0.006, -0.077),
            (0.0038, -0.055),
            (0.003, -0.036),
        ],
        8,
    )
    _solid_flat_profile(stage, path + "/RoundedHandle", handle, 0.0036, 0.0025, steel)
    # Keep the original trajectory, aligning the visible lower face to the
    # calibrated MPM proxy's lower face for each saved physics state.
    move = root.AddTranslateOp()

    def set_pose(pos):
        x, y, z = map(float, pos)
        move.Set(Gf.Vec3d(x, y, z + 0.0006 - collider_thickness / 2))

    set_pose(initial_pos)
    return set_pose
