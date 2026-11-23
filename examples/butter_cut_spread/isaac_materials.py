"""Materials and lighting for recorded butter cutting and spreading."""

from dataclasses import asdict, dataclass

import carb
import omni.replicator.core as rep
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade


def material(stage, name, color, roughness, metallic=0.0):
    """Create a physically based preview material in the supplied stage."""
    result = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return result


def box(stage, path, size, position, surface_material):
    """Create a box with a separately editable translation."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    transform = UsdGeom.Xformable(cube)
    translation = transform.AddTranslateOp()
    translation.Set(Gf.Vec3d(*map(float, position)))
    transform.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(surface_material)
    return translation


def apply_butter_appearance(stage, butter_surface, appearance):
    """Apply the cutting demo scattering profile without changing its parameters."""
    if appearance is not None:
        shader = UsdShade.Shader.Define(stage, "/World/Looks/Butter/Scattering")
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath("OmniSurface.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        butter_surface.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        butter_surface.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        for name, value in asdict(appearance).items():
            if isinstance(value, (tuple, list)):
                shader.CreateInput(name, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*value))
            elif isinstance(value, bool):
                shader.CreateInput(name, Sdf.ValueTypeNames.Bool).Set(value)
            else:
                shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)


def create_lights(stage, board_top, background_fill=False):
    """Use the cutting demo's neutral lights, powers, sizes and aim."""
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(750.0 if background_fill else 650.0)
    for name, position, intensity, width in (
        ("Key", (-0.12, -0.08, 0.25), 2000.0, 0.18),
        ("Fill", (0.15, 0.10, 0.20), 1300.0, 0.16),
        ("Front", (0.02, -0.18, 0.14), 1050.0 if background_fill else 900.0, 0.22),
    ):
        light = UsdLux.RectLight.Define(stage, f"/World/Lights/{name}")
        light.CreateIntensityAttr(intensity)
        light.CreateWidthAttr(width)
        light.CreateHeightAttr(width)
        light_pose = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(position[0], position[1], position[2] + board_top - 0.01),
            Gf.Vec3d(0.006, 0.0, board_top + 0.024),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
        UsdGeom.Xformable(light).AddTransformOp().Set(light_pose.GetInverse())
    return dome


@dataclass(frozen=True)
class SceneMaterials:
    butter: UsdShade.Material
    steel: UsdShade.Material
    board: UsdShade.Material
    counter: UsdShade.Material


def create_scene_materials(stage, appearance):
    """Keep butter, steel, board and counter appearance shared by both tasks."""
    butter = material(stage, "Butter", (0.88, 0.66, 0.23), roughness=0.4)
    apply_butter_appearance(stage, butter, appearance)
    return SceneMaterials(
        butter,
        material(stage, "Steel", (0.63, 0.67, 0.72), roughness=0.25, metallic=1.0),
        material(stage, "Board", (0.23, 0.105, 0.035), roughness=0.6),
        material(stage, "Counter", (0.18, 0.21, 0.23), roughness=0.75),
    )


def configure_pathtracing(samples, volume_bounces=63):
    """Use metre units, resolved contact rays and the calibrated fixed exposure."""
    settings = carb.settings.get_settings()
    rep.settings.set_render_pathtraced(samples_per_pixel=samples)
    # Optical distances and the renderer's internal length unit must agree.
    settings.set("/rtx/scene/renderMeterPerUnit", 1.0)
    settings.set("/rtx/raytracing/rayOffset", 0.000001)
    settings.set("/rtx/pathtracing/maxBounces", 12)
    settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 12)
    settings.set("/rtx/pathtracing/maxVolumeBounces", volume_bounces)
    settings.set("/rtx/post/histogram/enabled", False)
    settings.set("/rtx/post/tonemap/op", 6)
