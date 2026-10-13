import csv
import json
import os
import urllib.request
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image

import filelock
import trimesh

import genesis as gs
from genesis.engine.entities import KinematicEntity, PBDFreeParticleEntity, PBSTFEntity, RigidEntity, SPHEntity
from genesis.utils.misc import tensor_to_array
from genesis.utils.video_encoder import VideoEncoder
from genesis.vis.camera import Camera

if __package__:
    from .fluid_helper import TeapotSettings, sample_teapot_particles, update_rigid_teapot, update_teapot_case
    from .teapot_checkpoint import CheckpointInfo, load_teapot_checkpoint, save_teapot_checkpoint
else:
    from fluid_helper import TeapotSettings, sample_teapot_particles, update_rigid_teapot, update_teapot_case
    from teapot_checkpoint import CheckpointInfo, load_teapot_checkpoint, save_teapot_checkpoint


class TeapotScene(NamedTuple):
    scene: gs.Scene
    teapot: KinematicEntity | RigidEntity
    liquid: PBDFreeParticleEntity | PBSTFEntity | SPHEntity
    settings: TeapotSettings
    camera: Camera


class RenderAsset(NamedTuple):
    name: str
    filename: str
    url: str
    source: str


def prepare_render_assets():
    """Cache the indoor lighting and surface maps alongside their sources and licenses."""
    directory = Path("out/teapot_assets").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    assets = (
        RenderAsset(
            "Brown Photostudio 05",
            "brown_photostudio_05_2k.hdr",
            "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/brown_photostudio_05_2k.hdr",
            "https://polyhaven.com/a/brown_photostudio_05",
        ),
        RenderAsset(
            "Wood Table Diffuse",
            "wood_table_diff_1k.jpg",
            "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/wood_table/wood_table_diff_1k.jpg",
            "https://polyhaven.com/a/wood_table",
        ),
        RenderAsset(
            "Wood Table Roughness",
            "wood_table_rough_1k.jpg",
            "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/wood_table/wood_table_rough_1k.jpg",
            "https://polyhaven.com/a/wood_table",
        ),
    )
    with filelock.FileLock(directory / "assets.lock", timeout=600):
        for asset in assets:
            destination = directory / asset.filename
            if not destination.exists():
                gs.logger.info(f"Downloading {asset.name} from {asset.source}")
                temporary = destination.with_suffix(".download")
                request = urllib.request.Request(asset.url, headers={"User-Agent": "Genesis-teapot-render"})
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                temporary.replace(destination)
        with (directory / "sources.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(("name", "filename", "download", "source", "license"))
            writer.writerows((*asset, "CC0") for asset in assets)
    return directory


def build_teapot_scene(
    settings, material, solver_options, show_viewer=False, n_envs=0, preview="particles", spp=32, playback_speed=1.0
):
    """Build the common scene with the selected fluid solver and preview quality."""
    if preview not in ("particles", "render"):
        raise ValueError("preview must be 'particles' or 'render'.")
    if spp < 1 or playback_speed <= 0.0:
        raise ValueError("Samples per pixel and playback speed must be positive.")
    is_pbstf = isinstance(solver_options, gs.options.PBSTFOptions)
    is_rendered = preview == "render"
    if is_rendered:
        assets = prepare_render_assets()
        renderer = gs.renderers.RayTracer(
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path=str(assets / "brown_photostudio_05_2k.hdr"),
                    image_color=(0.6, 0.5, 0.38),
                    encoding="linear",
                ),
            ),
            env_euler=(90.0, 120.0, 0.0),
            lights=[
                gs.renderers.SphereLight(
                    pos=(12.0, 18.0, 0.0),
                    color=(1.0, 0.9, 0.75),
                    intensity=4.0,
                    radius=5.0,
                ),
                gs.renderers.SphereLight(
                    pos=(16.0, 3.0, -28.0),
                    color=(1.0, 0.9, 0.75),
                    intensity=6.0,
                    radius=5.0,
                ),
            ],
        )
        teapot_surface = gs.surfaces.Rough(
            color=(0.19, 0.06, 0.032),
            roughness=0.8,
            ior=1.45,
        )
        table_surface = gs.surfaces.Rough(
            roughness=None,
            ior=1.08,
            diffuse_texture=gs.textures.ImageTexture(
                image_path=str(assets / "wood_table_diff_1k.jpg"),
                image_color=(0.4, 0.34, 0.28),
            ),
            roughness_texture=gs.textures.ImageTexture(
                image_path=str(assets / "wood_table_rough_1k.jpg"),
                encoding="linear",
            ),
        )
        ground_surface = gs.surfaces.Rough(color=(0.14, 0.115, 0.09), roughness=0.9)
        liquid_surface = gs.surfaces.Glass(
            color=(1.0, 1.0, 1.0),
            roughness=0.025,
            ior=1.333,
            transmission_texture=gs.textures.ColorTexture(
                color=(0.99, 0.975, 0.94),
            ),
            vis_mode="recon",
            recon_backend="splashsurf",
        )
    else:
        renderer = gs.renderers.Rasterizer()
        teapot_surface = gs.surfaces.Default(
            color=(0.66, 0.66, 0.66),
            opacity=0.3,
        )
        ground_surface = gs.surfaces.Rough(
            color=(0.35, 0.37, 0.4),
        )
        liquid_surface = gs.surfaces.Default(
            color=(0.25, 0.55, 0.95),
            vis_mode="particle",
        )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=settings.dt,
            substeps=10 if isinstance(solver_options, gs.options.SPHOptions) else 1,
            gravity=settings.gravity,
        ),
        sph_options=solver_options if isinstance(solver_options, gs.options.SPHOptions) else None,
        pbd_options=solver_options if isinstance(solver_options, gs.options.PBDOptions) else None,
        pbstf_options=solver_options if is_pbstf else None,
        viewer_options=gs.options.ViewerOptions(
            refresh_rate=round(1.0 / settings.dt),
            realtime_factor=playback_speed,
            camera_pos=settings.camera_pos,
            camera_lookat=settings.camera_lookat,
            camera_up=(0.0, 1.0, 0.0),
            camera_fov=40,
        ),
        renderer=renderer,
        show_viewer=show_viewer and not is_rendered,
    )
    if is_rendered:
        # A vertical emitter outside the camera view gives the falling stream a continuous reflection highlight.
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(12.0, -5.0, 28.0),
                size=(0.2, 26.0, 12.0),
                euler=(0.0, 121.0, 0.0),
                collision=False,
            ),
            material=gs.materials.Kinematic(),
            surface=gs.surfaces.Emission(
                color=(18.0, 16.0, 13.5),
            ),
            name="water_rim_light",
        )
        wall_surface = gs.surfaces.Rough(color=(0.9, 0.9, 0.9), roughness=0.95, ior=1.1)
        for name, pos, size in (
            ("back_wall", (-12.0, 10.0, -43.0), (72.0, 60.0, 0.6)),
            ("side_wall", (-48.0, 10.0, -9.0), (0.6, 60.0, 68.0)),
        ):
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=pos,
                    size=size,
                    collision=False,
                ),
                material=gs.materials.Kinematic(),
                surface=wall_surface,
                name=name,
            )
        table_offset = np.array((-12.0, 0.0, -8.0))
        for name, pos, size in (
            ("tea_table_top", (-17.0, -12.6, -7.0), (30.0, 1.2, 48.0)),
            ("tea_table_apron_front", (-3.7, -14.0, -7.0), (0.8, 1.6, 44.0)),
            ("tea_table_apron_back", (-30.3, -14.0, -7.0), (0.8, 1.6, 44.0)),
            ("tea_table_apron_left", (-17.0, -14.0, 14.3), (26.0, 1.6, 0.8)),
            ("tea_table_apron_right", (-17.0, -14.0, -28.3), (26.0, 1.6, 0.8)),
            ("tea_table_leg_front_left", (-4.0, -16.6, 14.0), (1.6, 6.8, 1.6)),
            ("tea_table_leg_front_right", (-4.0, -16.6, -28.0), (1.6, 6.8, 1.6)),
            ("tea_table_leg_back_left", (-30.0, -16.6, 14.0), (1.6, 6.8, 1.6)),
            ("tea_table_leg_back_right", (-30.0, -16.6, -28.0), (1.6, 6.8, 1.6)),
        ):
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=table_offset + pos,
                    size=size,
                    collision=False,
                ),
                material=gs.materials.Kinematic(),
                surface=table_surface,
                name=name,
            )
        tray_surface = gs.surfaces.Rough(color=(0.055, 0.028, 0.016), roughness=0.7)
        for name, pos, size in (
            ("tea_tray", (-18.5, -11.55, 5.5), (23.0, 0.9, 17.0)),
            ("tea_tray_back", (-30.0, -10.85, 5.5), (0.5, 0.5, 17.0)),
            ("tea_tray_front", (-7.0, -10.85, 5.5), (0.5, 0.5, 17.0)),
            ("tea_tray_left", (-18.5, -10.85, 14.0), (23.0, 0.5, 0.5)),
            ("tea_tray_right", (-18.5, -10.85, -3.0), (23.0, 0.5, 0.5)),
        ):
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=table_offset + pos,
                    size=size,
                    collision=False,
                ),
                material=gs.materials.Kinematic(),
                surface=tray_surface,
                name=name,
            )
        for index, x in enumerate(np.linspace(-28.5, -8.5, 18)):
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=table_offset + (x, -11.02, 5.5),
                    size=(0.95, 0.16, 15.8),
                    collision=False,
                ),
                material=gs.materials.Kinematic(),
                surface=gs.surfaces.Rough(color=(0.09, 0.045, 0.022), roughness=0.65),
                name=f"tea_tray_slat_{index}",
            )
        cup_profile = np.array(
            (
                (0.0, 0.08),
                (0.8, 0.08),
                (0.85, 0.3),
                (1.0, 0.4),
                (1.2, 0.7),
                (1.45, 1.25),
                (1.72, 2.1),
                (1.78, 2.3),
                (1.75, 2.36),
                (1.66, 2.36),
                (1.62, 2.27),
                (1.55, 2.05),
                (1.3, 1.25),
                (1.05, 0.75),
                (0.8, 0.5),
                (0.0, 0.5),
            )
        )
        saucer_profile = np.array(
            (
                (0.0, 0.0),
                (1.1, 0.0),
                (1.2, 0.1),
                (2.0, 0.18),
                (2.65, 0.4),
                (2.75, 0.48),
                (2.72, 0.56),
                (2.6, 0.58),
                (1.9, 0.34),
                (1.15, 0.25),
                (0.0, 0.25),
            )
        )
        jar_profile = np.array(
            (
                (0.0, 0.0),
                (1.8, 0.0),
                (2.15, 0.25),
                (2.45, 0.9),
                (2.5, 3.4),
                (2.4, 4.15),
                (2.1, 4.65),
                (1.8, 4.85),
                (1.8, 5.05),
                (0.0, 5.05),
            )
        )
        lid_profile = np.array(
            (
                (0.0, 0.0),
                (1.9, 0.0),
                (2.05, 0.1),
                (2.05, 0.25),
                (1.9, 0.4),
                (1.3, 0.6),
                (0.45, 0.7),
                (0.35, 0.9),
                (0.45, 1.05),
                (0.4, 1.2),
                (0.0, 1.25),
            )
        )
        porcelain_surface = gs.surfaces.Rough(color=(0.5, 0.46, 0.37), roughness=0.5, ior=1.35)
        celadon_surface = gs.surfaces.Rough(color=(0.21, 0.29, 0.25), roughness=0.5, ior=1.35)
        for name, profile, positions, surface in (
            ("tea_cup", cup_profile, ((-19.0, -10.69, 8.5), (-13.0, -10.69, 0.5)), porcelain_surface),
            ("tea_saucer", saucer_profile, ((-19.0, -10.94, 8.5), (-13.0, -10.94, 0.5)), porcelain_surface),
            ("tea_jar", jar_profile, ((-26.0, -10.94, 5.0),), teapot_surface),
            ("tea_jar_lid", lid_profile, ((-26.0, -5.89, 5.0),), teapot_surface),
            ("gaiwan", cup_profile * (1.35, 1.4), ((-10.0, -11.65, -25.0),), celadon_surface),
            ("gaiwan_saucer", saucer_profile * 1.4, ((-10.0, -12.0, -25.0),), celadon_surface),
            ("gaiwan_lid", lid_profile * (1.15, 0.8), ((-10.0, -8.4, -25.0),), celadon_surface),
        ):
            mesh = trimesh.creation.revolve(profile, sections=96)
            for index, pos in enumerate(positions):
                scene.add_entity(
                    morph=gs.morphs.MeshSet(
                        files=(mesh,),
                        pos=table_offset + pos,
                        euler=(-90.0, 0.0, 0.0),
                        collision=False,
                    ),
                    material=gs.materials.Kinematic(),
                    surface=surface,
                    name=f"{name}_{index}",
                )
    scene.add_entity(
        morph=gs.morphs.Plane(
            pos=(-12.0, settings.lower_bound[1], -8.0),
            euler=(-90.0, 0.0, 0.0),
            collision=False,
            plane_size=(72.0, 70.0),
            tile_size=(16.0, 16.0),
        ),
        material=gs.materials.Kinematic(),
        surface=ground_surface,
        name="ground",
    )
    teapot = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=settings.asset,
            scale=settings.mesh_scale,
            pos=settings.offset,
            quat=settings.quat,
            collision=not is_pbstf,
            decimate=False,
            watertighten=None,
            convexify=False,
            align=False,
            fixed=True,
        ),
        material=(
            gs.materials.Kinematic()
            if is_pbstf
            else gs.materials.Rigid(
                needs_coup=True,
                coup_friction=0.01,
                coup_restitution=0.0,
                sdf_min_res=150,
                sdf_max_res=150,
            )
        ),
        surface=teapot_surface,
        name=settings.entity_name,
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Particles(
            positions=sample_teapot_particles(settings, settings.particle_size),
        ),
        material=material,
        surface=liquid_surface,
        name="water",
    )
    camera = scene.add_camera(
        res=(960, 540),
        pos=settings.camera_pos,
        lookat=settings.camera_lookat,
        up=(0.0, 1.0, 0.0),
        fov=40,
        GUI=show_viewer and is_rendered,
        spp=spp,
        far=100.0,
    )
    scene.build(n_envs=n_envs)
    liquid.set_particles_vel(settings.particles_vel)
    return TeapotScene(scene, teapot, liquid, settings, camera)


def add_teapot_arguments(parser):
    parser.add_argument("--scale", type=int, help="Particle scale; diameter is 2 / scale (default and minimum: 20).")
    parser.add_argument("--dt", type=float, help="Simulation time step in seconds (default: 0.01).")
    horizon = parser.add_mutually_exclusive_group()
    horizon.add_argument("-s", "--steps", type=int, help="Number of simulation steps.")
    horizon.add_argument("-t", "--seconds", type=float, help="Simulated duration in seconds (default: 100).")
    parser.add_argument("-v", "--vis", action="store_true", help="Display the selected preview in a window.")
    parser.add_argument(
        "--preview", choices=("particles", "render"), default="particles", help="Fast particles or ray-traced water."
    )
    parser.add_argument("-r", "--record", action="store_true", help="Save the selected camera preview to MP4.")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("out/teapot"), help="Video and metadata directory."
    )
    parser.add_argument("--fps", type=float, help="Output frames per second (default: 25, or the saved replay rate).")
    parser.add_argument(
        "--spp", type=int, default=32, help="Ray-tracing samples per pixel; higher values reduce noise."
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        help="Simulated seconds per video second (default: 1, or the saved replay speed).",
    )
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument(
        "--save-checkpoints", type=Path, help="Save simulation frames under DIR/METHOD for later rendering."
    )
    checkpoints.add_argument(
        "--replay-checkpoints", type=Path, help="Render saved frames from DIR/METHOD without advancing physics."
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        help="Steps between saved frames; smaller values use more disk and preserve finer motion (default: video cadence).",
    )
    parser.add_argument(
        "--replay-time", type=float, help="Render one PNG at the closest saved time at or before this simulated second."
    )
    parser.add_argument(
        "--timings", action="store_true", help="Synchronize GPU work for stage timing; this reduces execution overlap."
    )


def run_teapot(simulation, args, method, checkpoint_info=None):
    """Simulate or replay recorded particle states, with optional video and checkpoint output."""
    scene, teapot, liquid, settings, camera = simulation
    if args.steps is not None and args.steps < 0:
        raise ValueError("--steps must be non-negative.")
    if args.seconds is not None and args.seconds < 0.0:
        raise ValueError("--seconds must be non-negative.")
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive.")
    frames_every = max(1, round(args.playback_speed / (args.fps * settings.dt)))
    progress_every = max(1, round(1.0 / settings.dt))
    checkpoint_every = frames_every if args.checkpoint_every is None else args.checkpoint_every
    checkpoint_paths = None
    replay_indices = None
    if checkpoint_info is not None:
        checkpoint_paths = sorted((args.replay_checkpoints / method).glob("frame_*.npz"))
        if not checkpoint_paths or checkpoint_info.particles != liquid.n_particles:
            raise ValueError("Checkpoint frames are missing or their particle count differs from the scene.")
        saved_steps = np.array([int(path.stem.removeprefix("frame_")) for path in checkpoint_paths])
        if saved_steps[0] != 0:
            raise ValueError("Replay requires the initial checkpoint at step zero.")
        steps = saved_steps[-1].item()
        if args.steps is not None:
            steps = args.steps
        elif args.seconds is not None:
            steps = round(args.seconds / settings.dt)
        if args.replay_time is not None:
            if args.replay_time > saved_steps[-1] * settings.dt:
                raise ValueError(f"Replay ends at {saved_steps[-1] * settings.dt:g} simulated seconds.")
            saved_index = np.searchsorted(saved_steps * settings.dt, args.replay_time, side="right") - 1
            steps = saved_steps[saved_index].item()
            step_indices = (steps,)
        else:
            step_indices = range(frames_every, steps + 1, frames_every)
        if steps > saved_steps[-1]:
            raise ValueError(f"Replay ends at {saved_steps[-1] * settings.dt:g} simulated seconds.")
        replay_indices = np.searchsorted(saved_steps, step_indices, side="right") - 1
        load_teapot_checkpoint(simulation, checkpoint_paths[0])
    else:
        seconds = 100.0 if args.seconds is None else args.seconds
        steps = round(seconds / settings.dt) if args.steps is None else args.steps
        if "PYTEST_VERSION" in os.environ:
            steps = min(steps, 2)
        step_indices = range(1, steps + 1)
    has_output = args.record or args.save_checkpoints is not None or checkpoint_info is not None
    if has_output:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / f"{method}_settings.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(("setting", "value"))
            writer.writerows(zip(settings._fields, settings))
            writer.writerows(
                (
                    ("method", method),
                    ("particles", liquid.n_particles),
                    ("steps", steps),
                    ("substeps", scene.sim_options.substeps),
                    ("rho", liquid.material.rho),
                    ("playback_speed", args.playback_speed),
                    ("fps", args.playback_speed / (frames_every * settings.dt)),
                    ("spp", args.spp),
                    ("material", liquid.material.model_dump_json()),
                )
            )
        np.savez_compressed(
            args.output_dir / f"{method}_initial.npz",
            positions=tensor_to_array(liquid.get_particles_pos()),
            velocities=tensor_to_array(liquid.get_particles_vel()),
        )
        (args.output_dir / f"{method}_status.txt").write_text("running\n", encoding="utf-8")
    checkpoint_directory = None
    if args.save_checkpoints is not None:
        checkpoint_directory = args.save_checkpoints / method
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        info = CheckpointInfo(
            version=1,
            method=method,
            scale=settings.scale,
            dt=settings.dt,
            particles=liquid.n_particles,
            steps=steps,
            checkpoint_every=checkpoint_every,
            fps=args.fps,
            playback_speed=args.playback_speed,
        )
        with (checkpoint_directory / "metadata.json").open("w", encoding="utf-8") as output:
            json.dump(info._asdict(), output, indent=2)
        save_teapot_checkpoint(simulation, checkpoint_directory, step=0)
    encoder = None
    if args.record and args.replay_time is None:
        encoder = VideoEncoder(str(args.output_dir / f"{method}.mp4"), fps=args.fps)
    physics_seconds = 0.0
    checkpoint_write_seconds = 0.0
    checkpoint_read_seconds = 0.0
    render_seconds = 0.0
    encode_seconds = 0.0
    if args.timings:
        torch.cuda.synchronize()
    try:
        for frame_idx, step in enumerate(step_indices):
            start = perf_counter()
            if checkpoint_info is not None:
                restored_step = load_teapot_checkpoint(simulation, checkpoint_paths[replay_indices[frame_idx]])
                if args.timings:
                    torch.cuda.synchronize()
                checkpoint_read_seconds += perf_counter() - start
            else:
                if isinstance(liquid, PBSTFEntity):
                    update_teapot_case(scene.pbstf_solver, teapot, scene.cur_t, settings)
                else:
                    update_rigid_teapot(teapot, scene.cur_t, settings)
                scene.step(update_visualizer=args.vis and args.preview == "particles")
                if args.timings:
                    torch.cuda.synchronize()
                physics_seconds += perf_counter() - start
            if checkpoint_directory is not None and (step % checkpoint_every == 0 or step == steps):
                start = perf_counter()
                save_teapot_checkpoint(simulation, checkpoint_directory, step)
                checkpoint_write_seconds += perf_counter() - start
            is_frame = checkpoint_info is not None or step % frames_every == 0
            has_render_preview = args.vis and (args.preview == "render" or checkpoint_info is not None)
            if is_frame and (encoder is not None or has_render_preview or args.replay_time is not None):
                start = perf_counter()
                rgb, *_ = camera.render(force_render=checkpoint_info is not None)
                render_seconds += perf_counter() - start
                if encoder is not None:
                    start = perf_counter()
                    encoder.write(rgb)
                    encode_seconds += perf_counter() - start
                if args.replay_time is not None:
                    path = args.output_dir / f"{method}_step_{restored_step:08d}.png"
                    Image.fromarray(rgb).save(path)
                    gs.logger.info(f"Saved checkpoint preview at {restored_step * settings.dt:g} s to {path}")
            if step % progress_every == 0:
                operation = "replayed" if checkpoint_info is not None else "simulated"
                gs.logger.info(f"{method}: {operation} {step * settings.dt:.2f} / {steps * settings.dt:.2f} s")
        positions = tensor_to_array(liquid.get_particles_pos())
        velocities = tensor_to_array(liquid.get_particles_vel())
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            if has_output:
                (args.output_dir / f"{method}_status.txt").write_text("unstable\n", encoding="utf-8")
            raise FloatingPointError(f"{method} produced non-finite particle state.")
        if has_output:
            np.savez_compressed(args.output_dir / f"{method}_final.npz", positions=positions, velocities=velocities)
            (args.output_dir / f"{method}_status.txt").write_text("complete\n", encoding="utf-8")
    finally:
        if encoder is not None:
            start = perf_counter()
            encoder.close()
            encode_seconds += perf_counter() - start
        if args.timings:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with (args.output_dir / f"{method}_timings.csv").open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(("stage", "seconds"))
                writer.writerows(
                    (
                        ("physics", physics_seconds),
                        ("checkpoint_write", checkpoint_write_seconds),
                        ("checkpoint_read", checkpoint_read_seconds),
                        ("render_including_reconstruction", render_seconds),
                        ("encode", encode_seconds),
                    )
                )
            gs.logger.info(
                f"{method} timing: physics={physics_seconds:.3f}s, checkpoint write={checkpoint_write_seconds:.3f}s, "
                f"checkpoint read={checkpoint_read_seconds:.3f}s, render={render_seconds:.3f}s, encode={encode_seconds:.3f}s"
            )
