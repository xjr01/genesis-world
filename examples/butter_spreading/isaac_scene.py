"""Isaac scene adapter for the recorded, unmodified spreading trajectory."""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import carb
import imageio_ffmpeg
import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

from examples.butter_spreading.appearance import ButterAppearance
from examples.butter_spreading.isaac_materials import (
    box,
    configure_pathtracing,
    create_lights,
    create_scene_materials,
    material,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameReport:
    source_frame: int
    time_s: float
    seconds: float
    butter_error_m: float
    bread_error_m: float
    blade_error_m: float
    capture_retries: int


@dataclass(frozen=True)
class RenderReport:
    is_complete: bool
    is_full_trajectory: bool
    state_sha256: str
    surface_sha256: str
    bread_sha256: str
    appearance_profile_sha256: str
    butter_appearance: ButterAppearance
    width: int
    height: int
    fps: int
    samples: int
    renderer: str
    render_metres_per_unit: float
    max_volume_bounces: int
    ray_offset_m: float
    auto_exposure: bool
    tone_mapping_operator: int
    background_fill: bool
    dome_intensity: float
    key_intensity: float
    fill_intensity: float
    front_intensity: float
    camera_position_m: tuple[float, float, float]
    camera_target_m: tuple[float, float, float]
    focal_length_mm: float
    horizontal_aperture_mm: float
    blade_film_status: str
    bread_geometry: str
    startup_seconds: float
    scene_seconds: float
    rendering_seconds: float
    elapsed_seconds: float
    frames: list[FrameReport]


def mesh(stage, name, surface):
    result = UsdGeom.Mesh.Define(stage, f"/World/Task/{name}")
    result.CreateSubdivisionSchemeAttr("none")
    result.SetNormalsInterpolation("vertex")
    UsdShade.MaterialBindingAPI.Apply(result.GetPrim()).Bind(surface)
    return result


def upload(target, points, normals, faces):
    target.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    target.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
    target.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    target.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.concatenate(faces).astype(np.int32)))
    target.GetExtentAttr().Set(
        Vt.Vec3fArray.FromNumpy(np.stack((points.min(axis=0), points.max(axis=0))).astype(np.float32))
    )


def render(app, run, samples, background_fill, frozen_frame, back_view, started, startup):
    inputs = json.loads((run / "isaac-inputs.json").read_text(encoding="utf-8"))
    for name, key in (
        ("mpm-state.npz", "state_sha256"),
        ("mpm-surface.npz", "surface_sha256"),
        ("bread-surface.npz", "bread_sha256"),
    ):
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != inputs[key]:
            raise ValueError(f"Render input hash mismatch: {name}")
    with np.load(run / "mpm-state.npz") as state:
        knives, times = state["knife_positions"], state["time_s"]
    with np.load(run / "mpm-surface.npz") as surface:
        vertices, normals, faces = (surface[key] for key in ("vertices", "normals", "faces"))
        vo, fo = surface["vertex_offsets"], surface["face_offsets"]
    with np.load(run / "bread-surface.npz") as bread:
        bread_vertices, bread_normals, bread_faces, bread_colors = (
            bread[key] for key in ("vertices", "normals", "faces", "colors")
        )
        if not np.array_equal(bread["time_s"], times) or str(bread["state_sha256"]) != inputs["state_sha256"]:
            raise ValueError("Bread mesh does not match the physical trajectory.")
    if frozen_frame is not None and not 0 <= frozen_frame < len(times):
        raise ValueError("Preview frame is outside the recorded trajectory.")
    indices = range(len(times)) if frozen_frame is None else [frozen_frame] * 8
    output = "isaac-spreading" if frozen_frame is None else f"frozen-{frozen_frame:03d}-{samples}spp"
    if back_view:
        output += "-back"
    if any((run / f"{output}{suffix}").exists() for suffix in (".mp4", ".json", ".usdc")):
        raise ValueError("Render output already exists; prepare a new output directory for another render.")
    profile_path = Path(__file__).parent / "materials/appearance.json"
    profile_bytes = profile_path.read_bytes()
    appearance = ButterAppearance(**json.loads(profile_bytes)["material"])
    scene_started = time.perf_counter()
    settings = carb.settings.get_settings()
    settings.set("/omni/replicator/captureOnPlay", False)
    settings.set("/omni/replicator/captureMotionBlur", False)
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    materials = create_scene_materials(stage, appearance)
    bread_material = material(stage, "Bread", (0.72, 0.53, 0.29), 0.88)
    bread_shader = UsdShade.Shader.Get(stage, "/World/Looks/Bread/Shader")
    reader = UsdShade.Shader.Define(stage, "/World/Looks/Bread/Color")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    bread_shader.GetInput("diffuseColor").ConnectToSource(reader.ConnectableAPI(), "result")
    top, size = inputs["support_top_m"], inputs["support_size_m"]
    box(stage, "/World/Board", size, (0, 0, top - size[2] / 2), materials.board)
    box(stage, "/World/Counter", (2, 2, 0.02), (0, 0, top - size[2] - 0.01), materials.counter)
    task = UsdGeom.Xform.Define(stage, "/World/Task")
    translation = task.AddTranslateOp()
    translation.Set(Gf.Vec3d(*map(float, knives[0])))
    box(stage, "/World/Task/Blade", inputs["blade_size_m"], inputs["blade_offset_m"], materials.steel)
    butter_mesh, bread_mesh = mesh(stage, "Butter", materials.butter), mesh(stage, "Bread", bread_material)
    bread_mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(bread_colors))
    dome = create_lights(stage, top, background_fill)
    camera_position = (-0.19 if back_view else 0.19, -0.28, 0.21)
    camera_target = (0.0, 0.0, 0.03)
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    pose = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*camera_position), Gf.Vec3d(*camera_target), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(camera).AddTransformOp().Set(pose.GetInverse())
    camera.CreateFocalLengthAttr(35)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateClippingRangeAttr((0.001, 10))
    product = rep.create.render_product("/World/Camera", (1280, 720))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    for _ in range(24):
        app.update()
    configure_pathtracing(samples)
    scene_seconds = time.perf_counter() - scene_started
    rendering_started = time.perf_counter()
    encoder = imageio_ffmpeg.write_frames(
        str(run / f"{output}.mp4"), (1280, 720), fps=inputs["fps"], codec="libx264", quality=8
    )
    encoder.send(None)
    font = ImageFont.load_default(size=18)
    frames = []
    try:
        for output_index, index in enumerate(indices):
            tick = time.perf_counter()
            points = vertices[vo[index] : vo[index + 1]]
            upload(
                butter_mesh,
                points - knives[index],
                normals[vo[index] : vo[index + 1]],
                faces[fo[index] : fo[index + 1]],
            )
            upload(bread_mesh, bread_vertices[index] - knives[index], bread_normals[index], bread_faces)
            translation.Set(Gf.Vec3d(*map(float, knives[index])))
            pose_readback = np.array(translation.Get())
            butter_error = float(np.max(np.abs(np.array(butter_mesh.GetPointsAttr().Get()) + pose_readback - points)))
            bread_error = float(
                np.max(np.abs(np.array(bread_mesh.GetPointsAttr().Get()) + pose_readback - bread_vertices[index]))
            )
            blade_error = float(np.max(np.abs(pose_readback - knives[index])))
            if max(butter_error, bread_error, blade_error) > 1e-8:
                raise RuntimeError(f"Scene pose differs from the simulation at frame {index}.")
            rep.orchestrator.step(rt_subframes=8, delta_time=0.0, wait_for_render=True)
            for retry in range(8):
                rep.orchestrator.step(rt_subframes=8, delta_time=0.0, wait_for_render=True)
                pixels = np.asarray(rgb.get_data())
                if pixels.shape[:2] == (720, 1280) and pixels[:, :, :3].max() > 0:
                    break
                logger.warning("Invalid RGB at source %s; retrying the same frozen state", index)
                app.update()
            else:
                raise RuntimeError(f"Invalid RGB buffer at source {index}.")
            frame = Image.fromarray(pixels[:, :, :3].copy())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 686, 1280, 720), fill=(18, 18, 18))
            draw.text((12, 693), f"Butter spreading | t = {times[index]:.3f} s", font=font, fill="white")
            encoder.send(np.array(frame))
            if frozen_frame is not None or index in (0, 12, 24, 36, 48, 60, 72, 84, 96):
                frame.save(run / f"{output}-{output_index:03d}.png")
            frames.append(
                FrameReport(
                    index,
                    float(times[index]),
                    time.perf_counter() - tick,
                    butter_error,
                    bread_error,
                    blade_error,
                    retry,
                )
            )
            print(f"Rendered {output_index + 1}/{len(indices)} source={index}: {frames[-1].seconds:.3f}s", flush=True)
    finally:
        encoder.close()
    if frozen_frame is None:
        stage.GetRootLayer().Export(str(run / f"{output}.usdc"))
    report = RenderReport(
        True,
        frozen_frame is None,
        inputs["state_sha256"],
        inputs["surface_sha256"],
        inputs["bread_sha256"],
        hashlib.sha256(profile_bytes).hexdigest(),
        appearance,
        1280,
        720,
        inputs["fps"],
        samples,
        settings.get("/rtx/rendermode"),
        settings.get("/rtx/scene/renderMeterPerUnit"),
        settings.get("/rtx/pathtracing/maxVolumeBounces"),
        settings.get("/rtx/raytracing/rayOffset"),
        settings.get("/rtx/post/histogram/enabled"),
        settings.get("/rtx/post/tonemap/op"),
        background_fill,
        dome.GetIntensityAttr().Get(),
        2000.0,
        1300.0,
        1050.0 if background_fill else 900.0,
        camera_position,
        camera_target,
        35.0,
        36.0,
        inputs["blade_film_status"],
        "MPM-driven material boundary; procedural crumb/crust color; no added pore geometry",
        startup,
        scene_seconds,
        time.perf_counter() - rendering_started,
        time.perf_counter() - started,
        frames,
    )
    (run / f"{output}.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
