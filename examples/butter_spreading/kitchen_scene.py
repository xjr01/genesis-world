"""Three-dimensional tabletop with patterned plates, bread, sugar and a butter tub.

All props are geometry with materials. No reference-photo background or
painted butter trail is used. Physics states are rotated as one task group.
"""

from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
from examples.butter_spreading.scene_geometry import _material, _textured_material, _mesh, _rounded_rect

from examples.butter_spreading.bread_knife_assets import add_bread, add_knife

TASK_ANGLE = -24.0


def smooth_mesh(stage, path, points, faces, material, uvs=None):
    mesh = _mesh(stage, path, points, faces, material, uvs)
    p = np.asarray(points, dtype=np.float32)
    normals = np.zeros_like(p)
    for face in faces:
        a, b, c = face[:3]
        normal = np.cross(p[b] - p[a], p[c] - p[a])
        for i in face:
            normals[i] += normal
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateDoubleSidedAttr(True)
    return mesh


def normal_texture(stage, material, path, scale=1):
    shader = UsdShade.Shader.Get(stage, material.GetPath().AppendChild("Preview"))
    base = str(material.GetPath())
    uv = UsdShade.Shader.Define(stage, base + "/NormalUV")
    uv.CreateIdAttr("UsdPrimvarReader_float2")
    uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, base + "/NormalMap")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(path)))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
    tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(tex.ConnectableAPI(), "rgb")


def lathe(stage, path, profile, center, mat, uv_radius=0.12, segments=160):
    pts = []
    uv = []
    faces = []
    for r, z in profile:
        for t in np.linspace(0, 2 * np.pi, segments, endpoint=False):
            x, y = r * np.cos(t), r * np.sin(t)
            pts.append((x + center[0], y + center[1], z + center[2]))
            uv.append((0.5 + x / (2 * uv_radius), 0.5 + y / (2 * uv_radius)))
    for k in range(len(profile) - 1):
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((k * segments + i, (k + 1) * segments + i, (k + 1) * segments + j, k * segments + j))
    return smooth_mesh(stage, path, pts, faces, mat, uv)


def plate(stage, path, center, pattern, rim):
    profile = [
        (0.00001, 0.010),
        (0.030, 0.010),
        (0.062, 0.010),
        (0.077, 0.0105),
        (0.084, 0.0115),
        (0.094, 0.014),
        (0.104, 0.018),
        (0.115, 0.023),
        (0.121, 0.024),
        (0.123, 0.0235),
        (0.124, 0.022),
        (0.123, 0.020),
        (0.115, 0.016),
        (0.095, 0.009),
        (0.073, 0.003),
        (0.04, 0.002),
        (0.00001, 0.002),
    ]
    lathe(stage, path + "/Ceramic", profile, center, pattern, 0.124)
    lathe(
        stage,
        path + "/InkRim",
        [(0.1227, 0.022), (0.124, 0.0225), (0.124, 0.0237), (0.1227, 0.024)],
        center,
        rim,
        0.124,
    )
    lathe(stage, path + "/Foot", [(0.047, 0.003), (0.047, -0.006), (0.051, -0.006), (0.051, 0.003)], center, rim)


def inside(points, polygon):
    x, y = points.T
    hit = np.zeros(len(points), bool)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        hit ^= ((a[1] > y) != (b[1] > y)) & (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1] + 1e-15) + a[0])
    return hit


def tub(stage, base, center, angle, white, blue, label, butter):
    root = UsdGeom.Xform.Define(stage, base)
    root.AddTranslateOp().Set(Gf.Vec3d(*center))
    root.AddRotateZOp().Set(angle)
    outline = np.array(_rounded_rect(0.12, 0.145, 0.019, 16))
    n = len(outline)
    pts = []
    faces = []
    layers = [(0.86, -0.003), (0.87, 0.004), (1.0, 0.046), (1.005, 0.05), (0.975, 0.052), (0.94, 0.049), (0.90, 0.036)]
    for scale, z in layers:
        pts.extend([(x * scale, y * scale, z) for x, y in outline])
    for k in range(len(layers) - 1):
        for i in range(n):
            faces.append((k * n + i, k * n + (i + 1) % n, (k + 1) * n + (i + 1) % n, (k + 1) * n + i))
    smooth_mesh(stage, base + "/PlasticRim", pts, faces, white)
    # Blue outer sleeve, with text oriented towards the camera on the front.
    sleeve = []
    uv = []
    for k, (scale, z) in enumerate([(0.877, 0.003), (0.986, 0.043)]):
        for i, (x, y) in enumerate(outline):
            sleeve.append((x * scale, y * scale, z))
            uv.append((i / n, k))
    smooth_mesh(
        stage, base + "/BlueSleeve", sleeve, [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)], blue, uv
    )
    _mesh(
        stage,
        base + "/FrontLabel",
        [(-0.047, -0.067, 0.007), (0.047, -0.067, 0.007), (0.053, -0.074, 0.042), (-0.053, -0.074, 0.042)],
        [(0, 1, 2, 3)],
        label,
        [(0, 0), (1, 0), (1, 1), (0, 1)],
    )
    # Scooped, folded margarine surface is a static prop inside the tub.
    gx, gy = np.meshgrid(np.linspace(-0.053, 0.053, 95), np.linspace(-0.064, 0.064, 113))
    q = np.c_[np.concatenate(gx), np.concatenate(gy)]
    q = q[inside(q, outline * 0.90)]
    x, y = q.T
    z = np.full_like(x, 0.040)
    for cx, cy, depth in [(-0.017, 0.028, 0.009), (0.018, -0.025, 0.008), (0.019, 0.047, 0.005)]:
        dx = x - cx - 0.20 * (y - cy)
        g = np.exp(-((dx / 0.026) ** 2) - ((y - cy) / 0.035) ** 2)
        z -= depth * g
        z += depth * 0.65 * np.exp(-(((dx + 0.024) / 0.0045) ** 2) - ((y - cy) / 0.033) ** 2)
    z += 0.00045 * np.sin(780 * x + 90 * y) * np.sin(y * 73)
    tri = Delaunay(q).simplices
    tri = tri[inside(q[tri].mean(axis=1), outline * 0.90)]
    smooth_mesh(stage, base + "/ScoopedMargarine", np.c_[q, z], tri, butter)
    for i, cy in enumerate([-0.045, -0.019, 0.010, 0.033, 0.050]):
        points = []
        faces = []
        for a, u in enumerate(np.linspace(0, 1, 20)):
            for b, v in enumerate(np.linspace(-1, 1, 12)):
                px = -0.047 + 0.024 * u + 0.003 * np.sin(v * 3 + i)
                py = cy + 0.011 * v + 0.007 * u
                pz = 0.039 + (0.009 + 0.002 * np.sin(i)) * np.sin(u * np.pi * 0.9) * (1 - 0.25 * v * v)
                points.append((px, py, pz))
        for a in range(19):
            for b in range(11):
                k = a * 12 + b
                faces.append((k, k + 12, k + 13, k + 1))
        smooth_mesh(stage, f"{base}/ButterCurl{i}", points, faces, butter)


def make_scene_assets(stage, asset_root: Path, initial_knife_pose, knife_thickness=0.012):
    tex = asset_root / "textures"
    table = _textured_material(stage, "Stone", tex / "stone-gray.png", (0.2, 0.2, 0.2), 0.72)
    normal_texture(stage, table, tex / "stone-normal.png")
    pattern = _textured_material(stage, "Seigaiha", tex / "seigaiha.png", (0.7, 0.7, 0.7), 0.22)
    rim = _material(stage, "NavyGlaze", (0.009, 0.018, 0.03), 0.20)
    white = _material(stage, "IvoryPlastic", (0.78, 0.79, 0.70), 0.26)
    blue = _material(stage, "BlueTub", (0.025, 0.35, 0.47), 0.34)
    label = _textured_material(stage, "BlueBandLabel", tex / "tub-label.png", (0.7, 0.7, 0.7), 0.38)
    steel = _material(stage, "BrushedSteel", (0.65, 0.67, 0.69), 0.16, 0.98)
    sugar = _material(stage, "SugarCrystals", (0.78, 0.78, 0.75), 0.55)
    butter = _material(stage, "TubMargarine", (0.87, 0.74, 0.24), 0.35)
    _mesh(
        stage,
        "/World/Countertop",
        [(-0.6, -0.6, -0.006), (0.6, -0.6, -0.006), (0.6, 0.6, -0.006), (-0.6, 0.6, -0.006)],
        [(0, 1, 2, 3)],
        table,
        [(0, 0), (1, 0), (1, 1), (0, 1)],
    )
    plate(stage, "/World/MainPlate", (0, -0.006, 0), pattern, rim)
    plate(stage, "/World/StackPlate", (-0.197, 0.139, 0), pattern, rim)
    task = UsdGeom.Xform.Define(stage, "/World/Task")
    task.AddRotateZOp().Set(TASK_ANGLE)
    add_bread(stage, "/World/Task/Bread", (0, 0, 0.026), 0, asset_root)
    for i in range(4):
        add_bread(
            stage,
            f"/World/Stack/Slice{i}",
            (-0.243 + i * 0.023, 0.115 + i * 0.014, 0.025 + i * 0.012),
            -0.35 + i * 0.10,
            asset_root,
            i + 1,
        )
    lathe(
        stage,
        "/World/SugarBowl/Steel",
        [
            (0.0001, 0),
            (0.028, 0),
            (0.041, 0.006),
            (0.049, 0.022),
            (0.051, 0.041),
            (0.052, 0.049),
            (0.051, 0.050),
            (0.049, 0.049),
            (0.047, 0.035),
            (0.039, 0.012),
            (0.026, 0.004),
            (0.0001, 0.004),
        ],
        (-0.048, 0.209, -0.004),
        steel,
        0.052,
    )
    lathe(
        stage,
        "/World/SugarBowl/Sugar",
        [(0.00001, 0.026), (0.018, 0.025), (0.035, 0.022), (0.043, 0.018)],
        (-0.048, 0.209, -0.004),
        sugar,
        0.045,
    )
    # Thousands of small facets give sugar actual highlights and self-shadow.
    rng = np.random.default_rng(7)
    p = []
    f = []
    for _ in range(2200):
        x, y = rng.uniform(-0.042, 0.042, 2)
        r = np.hypot(x, y)
        if r > 0.042:
            continue
        z = 0.022 - 0.007 * (r / 0.043) ** 2
        r0 = rng.uniform(0.00025, 0.00065)
        k = len(p)
        p.extend(
            [
                (x - 0.048 - r0, y + 0.209, z),
                (x - 0.048 + r0, y + 0.209, z),
                (x - 0.048, y + 0.209 + r0, z),
                (x - 0.048, y + 0.209, z + r0),
            ]
        )
        f.extend([(k, k + 1, k + 3), (k + 1, k + 2, k + 3), (k + 2, k, k + 3)])
    _mesh(stage, "/World/SugarBowl/Granules", p, f, sugar)
    tub(stage, "/World/ButterTub", (0.123, 0.172, -0.003), -17, white, blue, label, butter)
    knife_steel = _textured_material(stage, "KnifeSteel", tex / "stone-gray.png", (0.6, 0.62, 0.65), 0.19)
    # Use a constant metal colour; the only mapped detail is fine brushed relief.
    knife_shader = UsdShade.Shader.Get(stage, "/World/Looks/KnifeSteel/Preview")
    knife_shader.GetInput("diffuseColor").DisconnectSource()
    knife_shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.6, 0.62, 0.65))
    knife_shader.GetInput("metallic").Set(1.0)
    normal_texture(stage, knife_steel, tex / "knife-brushed-normal.png")
    return add_knife(stage, "/World/Task/Knife", knife_steel, initial_knife_pose, collider_thickness=knife_thickness)
