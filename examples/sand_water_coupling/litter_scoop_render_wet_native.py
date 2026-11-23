"""
Phase 8d wet-sand NATIVE renderer for the litter-scoop SDF demo: wet grains darken through
Genesis's own rendering pipeline -- no projection compositing, no shader/source changes.
(Arm-less sibling of phase7_gripper_render_wet_native.py.)

Mechanism (per-frame color via the render-only particle buffers):
  - DEM particles are drawn as ONE instanced mesh per entity -> one color per entity.
    So besides the real sand entity (dry tan), we add N-1 "ghost" DEM entities with the
    same morph (same particle count/layout) and progressively darker wet colors.
  - The rasterizer rebuilds instance transforms every frame from
    `dem.particles_render.pos/active` (inactive grains get a zeroed transform = hidden).
    This script never calls scene.step(): it drives those render buffers DIRECTLY each
    frame -- every grain's recorded position is written into its wetness bucket's entity
    slice and marked active there (and inactive in all other buckets). The physics fields
    are untouched and the ghost particles never participate in any simulation.
  - Bucket per grain: round((ratio / MAX_RATIO) * (N_BUCKETS-1)); ratio comes from the
    recorded run (experiments/recordings/phase5_shovel_wet_sdf/frame_*.npz).

The litter-scoop mesh is kinematic and driven by the recorded blade pose. Frames are written
as PNGs and encoded with ffmpeg (the built-in recorder does not update without scene.step()).

Usage: python phase8d_render_wet_native.py [max_frames]
Env:   PHASE8D_N_BUCKETS (default 6), PHASE8D_ONLY_FRAME=<n> (render a single recorded frame)
Output: experiments/videos/phase5_shovel_wet_sdf.mp4 (overwrites the plain inline render)
        experiments/frames/phase8d_sandwet_native_frame_*.png (keyframes)
"""

import json
import math
import os
import subprocess
import sys

import imageio
import numpy as np

import genesis as gs

MAX_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else None

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(EXPERIMENTS_DIR, "frames")
REC_DIR = os.path.join(EXPERIMENTS_DIR, "recordings", "phase5_shovel_wet_sdf")
VIDEO_PATH = os.path.join(EXPERIMENTS_DIR, "videos", "phase5_shovel_wet_sdf.mp4")

DT = 1.0 / 60.0
KEYFRAME_STEPS = [0, 299, 410, 522, 592, 740, 872]

PARTICLE_RADIUS = 3.125e-3
BOX_LOWER = (-0.35, -0.30, 0.0)
BOX_UPPER = (0.35, 0.30, 0.80)
WALL_THICK = 0.01
WALL_HEIGHT = 0.24

DROPLET_POS = (-0.02, 0.0, 0.245)
DROPLET_RADIUS = 0.04
MAX_RATIO = 0.3

PARK_POS = (0.0, 0.0, -10.0)  # absorbed water particles are parked below the floor

# wetness buckets: bucket 0 = the real sand entity (DRY_COLOR), buckets 1..N-1 = ghosts.
# NOTE: 5-6 ghost copies of the deep bed (5-6 x 466k grains stacked in the same box) crash the
# first DEM substep at build with CUDA_ERROR_LAUNCH_FAILED on the RTX 4060; 4 buckets is the
# largest count that builds and renders cleanly.
N_BUCKETS = int(os.environ.get("PHASE8D_N_BUCKETS", "4"))
DRY_COLOR = np.array([0.87, 0.72, 0.53])
WET_COLOR = np.array([0.25, 0.13, 0.06])


def main():
    with open(os.path.join(REC_DIR, "meta.json")) as f:
        meta = json.load(f)
    n_sand = meta["n_sand"]
    n_water = meta["n_water"]
    frame_files = sorted(f for f in os.listdir(REC_DIR) if f.startswith("frame_"))
    only_frame = os.environ.get("PHASE8D_ONLY_FRAME")  # debug: render a single recorded frame
    if only_frame is not None:
        frame_files = [f"frame_{int(only_frame):04d}.npz"]
    elif MAX_FRAMES is not None:
        frame_files = frame_files[:MAX_FRAMES]
    n_frames = len(frame_files)
    print(f"replaying {n_frames} frames from {REC_DIR}, {N_BUCKETS} wetness buckets", flush=True)

    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=1, gravity=(0.0, 0.0, -9.8)),
        dem_options=gs.options.DEMOptions(
            particle_size=2.0 * PARTICLE_RADIUS,
            ddt_safety=0.5,
            lower_bound=BOX_LOWER,
            upper_bound=BOX_UPPER,
        ),
        flip_options=gs.options.FLIPOptions(
            grid_res=128,
            viscosity_coeff=0.01,
            lower_bound=BOX_LOWER,
            upper_bound=BOX_UPPER,
        ),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())

    wall_surface = gs.surfaces.Default(color=(0.55, 0.55, 0.6))
    box_x = BOX_UPPER[0] - BOX_LOWER[0]
    box_y = BOX_UPPER[1] - BOX_LOWER[1]
    wall_z = 0.5 * WALL_HEIGHT
    for wall_pos, wall_size in [
        ((BOX_LOWER[0] - 0.5 * WALL_THICK, 0.0, wall_z), (WALL_THICK, box_y, WALL_HEIGHT)),
        ((BOX_UPPER[0] + 0.5 * WALL_THICK, 0.0, wall_z), (WALL_THICK, box_y, WALL_HEIGHT)),
        ((0.0, BOX_LOWER[1] - 0.5 * WALL_THICK, wall_z), (box_x + 2.0 * WALL_THICK, WALL_THICK, WALL_HEIGHT)),
        ((0.0, BOX_UPPER[1] + 0.5 * WALL_THICK, wall_z), (box_x + 2.0 * WALL_THICK, WALL_THICK, WALL_HEIGHT)),
    ]:
        scene.add_entity(
            morph=gs.morphs.Box(pos=wall_pos, size=wall_size),
            material=gs.materials.Kinematic(),
            surface=wall_surface,
        )

    # real litter-scoop asset (objaverse 'Litter scoop/貓砂鏟', aligned to the blade frame by
    # phase8b_litter_align.py); driven by the recorded blade pose
    shovel = scene.add_entity(
        morph=gs.morphs.Mesh(file=os.path.join(EXPERIMENTS_DIR, "assets", "litter_scoop_aligned.glb")),
        material=gs.materials.Kinematic(),
    )

    # the real sand stays bucket 0 (dry color); the ghosts only differ in surface color --
    # their per-frame instance transforms are overwritten from the recording below
    sand_morph = gs.morphs.Box(pos=(0.0, 0.0, 0.1005), size=(0.68, 0.58, 0.20))
    sand = scene.add_entity(
        morph=sand_morph,
        material=gs.materials.DEM.Sand(sampler="fcc", rho=2.5, max_ratio=MAX_RATIO),
        surface=gs.surfaces.Default(color=tuple(DRY_COLOR)),
    )
    dem_ents = [sand]
    for k in range(1, N_BUCKETS):
        t = k / (N_BUCKETS - 1)
        color = DRY_COLOR * (1.0 - t) + WET_COLOR * t
        ghost = scene.add_entity(
            morph=sand_morph,
            material=gs.materials.DEM.Sand(sampler="fcc", rho=2.5, max_ratio=MAX_RATIO),
            surface=gs.surfaces.Default(color=tuple(color)),
        )
        dem_ents.append(ghost)

    water = scene.add_entity(
        morph=gs.morphs.Sphere(pos=DROPLET_POS, radius=DROPLET_RADIUS),
        material=gs.materials.FLIP.Liquid(),
        surface=gs.surfaces.Default(color=(0.2, 0.5, 0.9), opacity=0.85),
    )
    cam = scene.add_camera(
        res=(1280, 720),
        pos=(1.05, -1.05, 1.00),  # user request (2026-08-21): raise the viewpoint (was z=0.75)
        lookat=(0.0, 0.0, 0.18),
        fov=45,
    )

    scene.build()
    dem = scene.sim.dem_solver
    flip = scene.sim.flip_solver
    assert sand.n_particles == n_sand and water.n_particles == n_water, (
        f"particle count mismatch: sand {sand.n_particles} vs {n_sand}, "
        f"water {water.n_particles} vs {n_water}"
    )
    n_total = dem.n_particles
    slices = []
    for ent in dem_ents:
        assert ent.n_particles == n_sand, f"ghost particle count mismatch: {ent.n_particles} vs {n_sand}"
        slices.append((ent.particle_start, ent.particle_end))

    raw_dir = os.path.join(FRAMES_DIR, "phase8d_sandwet_native_raw")
    # wipe stale frames: ffmpeg globs frame_*.png, so leftovers from a longer earlier run
    # would be appended to the video (the 2026-08-21 "wrong last frames" bug)
    if os.path.isdir(raw_dir):
        for stale in os.listdir(raw_dir):
            os.remove(os.path.join(raw_dir, stale))
    os.makedirs(raw_dir, exist_ok=True)

    keyframe_idx = 0
    park_block = np.tile(np.array(PARK_POS, dtype=np.float64), (n_water, 1))
    pos_all = np.zeros((n_total, 1, 3), dtype=np.float32)
    act_all = np.zeros((n_total, 1), dtype=np.bool_)
    for i, fname in enumerate(frame_files):
        frame = np.load(os.path.join(REC_DIR, fname))

        # water is still driven through the physics-side fields + render-field sync
        water_pos = frame["water_pos"].astype(np.float64)
        buf = park_block.copy()
        buf[: len(water_pos)] = water_pos
        water.set_particles_pos(buf[None])

        shovel.set_pos(frame["blade_pos"].astype(np.float64))
        shovel.set_quat(frame["blade_quat"].astype(np.float64))

        # wetness buckets: every grain is active in exactly one entity slice, which picks
        # its color; the rasterizer rebuilds instance transforms from these buffers
        sand_pos = frame["sand_pos"].astype(np.float32)
        level = np.clip(frame["sand_ratio"].astype(np.float64) / MAX_RATIO, 0.0, 1.0)
        bucket = np.minimum((level * (N_BUCKETS - 1) + 0.5).astype(np.int64), N_BUCKETS - 1)
        act_all[:] = False
        for k, (s, e) in enumerate(slices):
            pos_all[s:e, 0] = sand_pos
            act_all[s:e, 0] = bucket == k
        dem.particles_render.pos.from_numpy(pos_all)
        dem.particles_render.active.from_numpy(act_all)
        flip.update_render_fields()

        rgb, *_ = cam.render(rgb=True, force_render=True)
        rgb = rgb[0] if isinstance(rgb, list) else rgb
        imageio.imwrite(os.path.join(raw_dir, f"frame_{i:04d}.png"), rgb)

        if i in KEYFRAME_STEPS:
            imageio.imwrite(
                os.path.join(FRAMES_DIR, f"phase8d_sandwet_native_frame_{keyframe_idx:04d}.png"),
                rgb,
            )
            keyframe_idx += 1

        if (i + 1) % 60 == 0:
            print(f"frame {i + 1}/{n_frames}", flush=True)

    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", "60",
            "-i", os.path.join(raw_dir, "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", VIDEO_PATH,
        ],
        check=True,
        capture_output=True,
    )
    print(f"video saved to {VIDEO_PATH}", flush=True)


if __name__ == "__main__":
    main()
