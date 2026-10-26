"""Display the same measured dish, tool and particle-derived butter geometry used by the simulation."""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import carb
import imageio_ffmpeg
import numpy as np
import omni.replicator.core as rep
import omni.usd
import trimesh
from PIL import Image, ImageDraw
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

from examples.butter_cut_spread.aligned_scene import build_aligned_assets
from examples.butter_cut_spread.appearance import ButterAppearance
from examples.butter_cut_spread.film_settings import FilmSettings
from examples.butter_cut_spread.isaac_materials import (
    box,
    configure_pathtracing,
    create_lights,
    create_scene_materials,
    material,
)
from examples.butter_cut_spread.mesh_upload import upload_triangle_mesh
from examples.butter_cut_spread.replay_frames import replay_indices


def render(
    app,
    run,
    samples,
    preview,
    started,
    camera_target=None,
    focal_length=None,
    camera_eye=None,
    frame_indices=None,
    output_name=None,
):
    config = json.loads((run / "config.json").read_text())
    manifest = json.loads((run / "assets.json").read_text())
    with (run / "state.npz").open("rb") as source:
        state_hash = hashlib.file_digest(source, "sha256").hexdigest()
    report = json.loads((run / "surface-report.json").read_text())
    if state_hash != report["state_sha256"]:
        raise ValueError("Reconstructed surfaces refer to another particle trajectory.")
    with np.load(run / "state.npz") as data:
        times, centers, quats = (data[k] for k in ("time_s", "knife_positions", "knife_quaternions"))
        bread_particles = data["bread_particles"] if manifest.get("is_cut_and_spread", False) else None
    indices = replay_indices(len(times), preview) if frame_indices is None else frame_indices
    if not indices or min(indices) < 0 or max(indices) >= len(times):
        raise ValueError("Frame indices must address the recorded trajectory.")
    available = set(report.get("frame_indices", range(report["frames"])))
    if not set(indices).issubset(available):
        raise ValueError("Required surfaces are missing; run asset_prepare without --preview for a complete video.")
    profile = Path(__file__).parent / "materials/approved_yellow/appearance.json"
    appearance = ButterAppearance(**json.loads(profile.read_text())["material"])
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
    porcelain = material(stage, "Porcelain", (0.8, 0.8, 0.78), 0.22)
    handle = material(stage, "Handle", (0.06, 0.009, 0.012), 0.32)
    box(stage, "/World/Counter", (2, 2, 0.02), (0, 0, -0.01), materials.counter)
    dish = trimesh.load_mesh(run / "dish.obj", process=False)
    dish_prim = UsdGeom.Mesh.Define(stage, "/World/Dish")
    upload_triangle_mesh(dish_prim, dish)
    dish_prim.CreateSubdivisionSchemeAttr("none")
    dish_prim.SetNormalsInterpolation("vertex")
    UsdShade.MaterialBindingAPI.Apply(dish_prim.GetPrim()).Bind(porcelain)
    tool = UsdGeom.Xform.Define(stage, "/World/Tool")
    translation, orientation = tool.AddTranslateOp(), tool.AddOrientOp()
    bread_replay = None
    if bread_particles is not None:
        bread_replay = build_aligned_assets(stage, run, manifest, bread_particles[0])
    for i, filename in enumerate(manifest["tool_meshes"] if bread_replay is None else []):
        mesh = trimesh.load_mesh(run / filename, process=False)
        prim = UsdGeom.Mesh.Define(stage, f"/World/Tool/Mesh{i}")
        upload_triangle_mesh(prim, mesh)
        prim.CreateSubdivisionSchemeAttr("none")
        prim.SetNormalsInterpolation("vertex")
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(handle if i < 3 else materials.steel)
    butter = UsdGeom.Mesh.Define(stage, "/World/Butter")
    butter.CreateSubdivisionSchemeAttr("none")
    butter.SetNormalsInterpolation("vertex")
    UsdShade.MaterialBindingAPI.Apply(butter.GetPrim()).Bind(materials.butter)
    geometry_path = run / "blade-film" / "geometry.npz"
    if hashlib.sha256(geometry_path.read_bytes()).hexdigest() != report["film_files_sha256"]["geometry.npz"]:
        raise ValueError("Blade film geometry differs from its reconstruction manifest.")
    with np.load(geometry_path) as geometry:
        film_vertices, film_faces = geometry["vertices"], geometry["faces"]
    film_settings = FilmSettings(**report["film_settings"])
    film_material = UsdShade.Material.Define(stage, "/World/Materials/BladeFilm")
    shader = UsdShade.Shader.Define(stage, "/World/Materials/BladeFilm/Surface")
    film_shader_path = Path(__file__).parent / "materials/blade_film.mdl"
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(str(film_shader_path.resolve())), "mdl")
    shader.SetSourceAssetSubIdentifier("BladeFilm", "mdl")
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    shader.CreateInput("tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*film_settings.color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(film_settings.roughness)
    film_material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    film_opacity = []
    film_meshes = []
    for side_index in range(2):
        prim = UsdGeom.Mesh.Define(stage, f"/World/Tool/ButterFilm{side_index}")
        vertices = film_vertices[side_index]
        prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        prim.GetExtentAttr().Set(
            Vt.Vec3fArray.FromNumpy(np.stack((vertices.min(axis=0), vertices.max(axis=0))).astype(np.float32))
        )
        prim.GetFaceVertexCountsAttr().Set(Vt.IntArray())
        prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray())
        prim.GetVisibilityAttr().Set("invisible")
        prim.CreateSubdivisionSchemeAttr("none")
        prim.CreateDoubleSidedAttr(False)
        prim.SetNormalsInterpolation("uniform")
        prim.GetNormalsAttr().Set(Vt.Vec3fArray())
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(film_material)
        opacity = prim.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.vertex)
        opacity.Set(np.zeros(len(film_vertices[side_index]), dtype=np.float32))
        film_opacity.append(opacity)
        film_meshes.append(prim)
    create_lights(stage, 0, False)
    if bread_replay is not None:
        # Center the unchanged neutral light rig on the two-workspace layout.
        UsdGeom.Xform.Define(stage, "/World/Lights").AddTranslateOp().Set((0.10, 0, 0))
    active_lights = [str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdLux.LightAPI)]
    if len(active_lights) != 4 or any(not path.startswith("/World/Lights/") for path in active_lights):
        raise ValueError("Only the four fixed scene lights may illuminate the assembled assets.")
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    if camera_eye is None:
        camera_eye = (0.28, -0.50, 0.38) if bread_replay is not None else (0.22, -0.36, 0.27)
    if camera_target is None:
        camera_target = (0.11, -0.025, 0.035) if bread_replay is not None else (0, -0.035, 0.035)
    if focal_length is None:
        focal_length = 34.0 if bread_replay is not None else 40.0
    pose = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*camera_eye),
        Gf.Vec3d(*camera_target),
        Gf.Vec3d(0, 0, 1),
    )
    UsdGeom.Xformable(camera).AddTransformOp().Set(pose.GetInverse())
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(36.0 * 720 / 1280)
    camera.CreateClippingRangeAttr((0.001, 10))
    product = rep.create.render_product("/World/Camera", (1280, 720))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    for _ in range(24):
        app.update()
    # Stage initialization can restore renderer defaults. Apply the capture
    # settings after initialization and verify the effective sampling budget.
    configure_pathtracing(samples)
    name = "asset-cut-preview" if preview else "asset-cut-isaac"
    if bread_replay is not None:
        name = "aligned-cut-spread-preview" if preview else "aligned-cut-spread-isaac"
    if output_name is not None:
        name = output_name
    (run / f"{name}.json").unlink(missing_ok=True)
    encoder = imageio_ffmpeg.write_frames(
        str(run / f"{name}.mp4"), (1280, 720), fps=config["fps"], codec="libx264", quality=8
    )
    encoder.send(None)
    frame_times = []
    try:
        for i in indices:
            tick = time.perf_counter()
            mesh = trimesh.load_mesh(run / "surfaces" / f"{i:04d}.ply", process=False)
            upload_triangle_mesh(butter, mesh)
            film_path = run / "blade-film" / f"{i:04d}.npz"
            if hashlib.sha256(film_path.read_bytes()).hexdigest() != report["film_files_sha256"][film_path.name]:
                raise ValueError("Blade film frame differs from its reconstruction manifest.")
            visible_film_triangles = []
            with np.load(film_path) as coating:
                for side_index, opacity in enumerate(film_opacity):
                    values = coating["opacity"][side_index]
                    # Match standalone cutting: zero-coverage triangles must not
                    # enter RTX as transparent geometry, where they can still
                    # produce reflections and a rectangular moving silhouette.
                    faces = film_faces[side_index]
                    visible = faces[np.max(values[faces], axis=1) > 1e-6]
                    prim = film_meshes[side_index]
                    prim.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(visible), 3, dtype=np.int32)))
                    prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(visible.reshape(-1).astype(np.int32)))
                    normals = np.zeros((len(visible), 3), dtype=np.float32)
                    normals[:, 2] = 2 * side_index - 1
                    prim.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
                    prim.GetVisibilityAttr().Set("inherited" if len(visible) else "invisible")
                    opacity.Set(Vt.FloatArray.FromNumpy(values.astype(np.float32)))
                    visible_film_triangles.append(len(visible))
            translation.Set(Gf.Vec3d(*map(float, centers[i])))
            q = quats[i]
            orientation.Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:]))))
            if bread_replay is not None:
                bread_replay.update(bread_particles[i])
            error = np.max(np.abs(np.array(butter.GetPointsAttr().Get()) - mesh.vertices))
            if error > 1e-8:
                raise ValueError("Isaac mesh upload differs from the recorded surface.")
            geometry_seconds = time.perf_counter() - tick
            capture_started = time.perf_counter()
            for _ in range(8):
                rep.orchestrator.step(rt_subframes=8, delta_time=0, wait_for_render=True)
                if (
                    settings.get("/rtx/rendermode") != "PathTracing"
                    or settings.get_as_int("/rtx/pathtracing/totalSpp") != samples
                ):
                    raise RuntimeError("Effective path-tracing settings differ from the requested capture settings.")
                pixels = np.array(rgb.get_data())
                if pixels.shape[:2] == (720, 1280) and pixels[:, :, :3].any():
                    pixels = pixels[:, :, :3].copy()
                    break
                app.update()
            else:
                raise RuntimeError(f"Isaac returned an empty RGB frame at recorded index {i}.")
            capture_seconds = time.perf_counter() - capture_started
            frame = Image.fromarray(pixels)
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 690, 1280, 720), fill=(22, 22, 22))
            draw.text(
                (16, 700),
                f"{'ALIGNED CUT AND SPREAD' if bread_replay is not None else 'END-SLICE CUT AND LIFT'} | t={times[i]:.3f}s",
                fill="white",
            )
            encoder.send(np.array(frame))
            if preview or frame_indices is not None or i % 24 == 0 or i == len(times) - 1:
                frame.save(run / f"{name}-{i:04d}.png")
            frame_times.append(
                {
                    "index": int(i),
                    "seconds": time.perf_counter() - tick,
                    "geometry_seconds": geometry_seconds,
                    "capture_seconds": capture_seconds,
                    "upload_error_m": float(error),
                    "effective_samples": settings.get_as_int("/rtx/pathtracing/totalSpp"),
                    "visible_film_triangles_negative_positive": visible_film_triangles,
                }
            )
            print(f"Rendered {i + 1}/{len(times)}", flush=True)
    finally:
        encoder.close()
    stage.GetRootLayer().Export(str(run / f"{name}.usdc"))
    (run / f"{name}.json").write_text(
        json.dumps(
            {
                "is_complete": len(indices) == len(times),
                "blade_film": "Two rotating particle-driven opacity sheets; complementary bulk weights; approximate thin coating BRDF.",
                "film_shader_sha256": hashlib.sha256(film_shader_path.read_bytes()).hexdigest(),
                "film_visibility": "Whole-BSDF stochastic coverage; zero-coverage triangles omitted; constant face normals.",
                "film_settings": asdict(film_settings),
                "state_sha256": state_hash,
                "appearance_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
                "samples": samples,
                "effective_samples": settings.get_as_int("/rtx/pathtracing/totalSpp"),
                "render_mode": settings.get("/rtx/rendermode"),
                "viewport_updates": not app.config["disable_viewport_updates"],
                "active_lights": active_lights,
                "camera_eye_m": camera_eye,
                "camera_target_m": camera_target,
                "camera_focal_length_mm": focal_length,
                "camera_aperture_mm": (36.0, 36.0 * 720 / 1280),
                "frames": frame_times,
                "elapsed_seconds": time.perf_counter() - started,
                "scope": "Actual asset geometry; approved yellow butter appearance; no render-time pose or motion edits.",
            },
            indent=2,
        )
    )
