"""
Phase 7 wet-sand NATIVE renderer: wet grains darken through Genesis's own rendering
pipeline -- no projection compositing, no shader/source changes.

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
    recorded run (experiments/recordings/phase7_gripper_shovel/frame_*.npz).

The arm uses the same continuous-IK scheme as gripper_shovel_render.py. Frames are written
as PNGs and encoded with ffmpeg (the built-in recorder does not update without scene.step()).

Usage: python phase7_gripper_render_wet_native.py [max_frames]
Env:   PHASE7_N_BUCKETS (default 6), PHASE7_ONLY_FRAME=<n> (render a single recorded frame)
Output: experiments/videos/phase7_gripper_shovel_sandwet_native.mp4
        experiments/frames/phase7_gripper_shovel_sandwet_native_frame_*.png (keyframes)
"""

import math
import os
import subprocess
import sys

import imageio
import json
import numpy as np

import genesis as gs

MAX_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else None

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(EXPERIMENTS_DIR, "frames")
REC_DIR = os.path.join(EXPERIMENTS_DIR, "recordings", "phase7_gripper_shovel")
VIDEO_PATH = os.path.join(EXPERIMENTS_DIR, "videos", "phase7_gripper_shovel_sandwet_native.mp4")

DT = 1.0 / 60.0
KEYFRAME_STEPS = [0, 299, 354, 450, 555, 705, 765]

PARTICLE_RADIUS = 3.125e-3
BOX_LOWER = (-0.35, -0.30, 0.0)
BOX_UPPER = (0.35, 0.30, 0.80)
WALL_THICK = 0.01
WALL_HEIGHT = 0.18

BLADE_HALF = (0.15, 0.12, 0.01)
HANDLE_LEN = 0.3
HANDLE_HALF_THICK = 0.015
HANDLE_ANGLE = math.radians(40.0)

DROPLET_POS = (0.02, 0.0, 0.205)
DROPLET_RADIUS = 0.04
MAX_RATIO = 0.3

FRANKA_BASE_POS = (-0.25, 0.38, 0.0)
FINGER_GRIP = 0.015
GRASP_DIST = 0.09
GRIP_ALONG = 0.08  # grip point shifted toward the handle's free end (kinematically hard pose)

# continuity scheme
DQ_MAX = 0.02  # rad per frame joint increment clamp
PARK_POS = (0.0, 0.0, -10.0)  # absorbed water particles are parked below the floor

# wetness buckets: bucket 0 = the real sand entity (DRY_COLOR), buckets 1..N-1 = ghosts
N_BUCKETS = int(os.environ.get("PHASE7_N_BUCKETS", "6"))
DRY_COLOR = np.array([0.87, 0.72, 0.53])
WET_COLOR = np.array([0.25, 0.13, 0.06])


def to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def quat_to_R(q):
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def R_to_quat(R):
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def blade_to_handle(blade_pos, blade_quat):
    handle_quat_rel = np.array(
        [math.cos((math.pi + HANDLE_ANGLE) / 2.0), 0.0, math.sin((math.pi + HANDLE_ANGLE) / 2.0), 0.0]
    )
    handle_dir_local = np.array([-math.cos(HANDLE_ANGLE), 0.0, math.sin(HANDLE_ANGLE)])
    handle_offset_local = np.array([-BLADE_HALF[0], 0.0, 0.0]) + handle_dir_local * (HANDLE_LEN / 2.0)
    handle_pos = blade_pos + quat_to_R(blade_quat) @ handle_offset_local
    handle_quat = quat_mul(blade_quat, handle_quat_rel)
    return handle_pos, handle_quat


def hand_target_from_handle(handle_pos, handle_quat):
    x_h = quat_to_R(handle_quat) @ np.array([1.0, 0.0, 0.0])
    x_h = x_h / np.linalg.norm(x_h)
    grip_pos = handle_pos + x_h * GRIP_ALONG
    z_h = np.array([0.0, -1.0, 0.0])
    z_h = z_h - np.dot(z_h, x_h) * x_h
    z_h = z_h / np.linalg.norm(z_h)
    y_h = np.cross(z_h, x_h)
    R_hand = np.column_stack([x_h, y_h, z_h])
    hand_pos = grip_pos - R_hand @ np.array([0.0, 0.0, GRASP_DIST])
    return hand_pos, R_to_quat(R_hand)


def solve_ik(franka, hand_link, pos, quat, q_prev):
    """Continuous IK: warm-start from the previous solution on the same branch (no random
    restarts), then clamp the per-frame joint increment; retry with more iterations from the
    SAME starting point when the warm-started solve does not converge well."""
    if q_prev is None:
        q = franka.inverse_kinematics(
            hand_link, pos=pos, quat=quat,
            max_samples=100, max_solver_iters=100, damping=0.005, pos_tol=1e-4, rot_tol=1e-3,
        )
        return to_np(q).astype(np.float64).reshape(-1)
    q, err = franka.inverse_kinematics(
        hand_link, pos=pos, quat=quat, init_qpos=q_prev, return_error=True,
        max_samples=1, max_solver_iters=30, damping=0.01, pos_tol=1e-4, rot_tol=1e-3,
    )
    err = to_np(err).astype(np.float64).reshape(-1)[:6]
    if np.linalg.norm(err[:3]) > 5e-3 or np.linalg.norm(err[3:]) > 5e-2:
        q, err = franka.inverse_kinematics(
            hand_link, pos=pos, quat=quat, init_qpos=q_prev, return_error=True,
            max_samples=1, max_solver_iters=300, damping=0.005, pos_tol=1e-4, rot_tol=1e-3,
        )
    q = to_np(q).astype(np.float64).reshape(-1)
    dq = np.clip(q - q_prev, -DQ_MAX, DQ_MAX)
    return q_prev + dq


def main():
    with open(os.path.join(REC_DIR, "meta.json")) as f:
        meta = json.load(f)
    n_sand = meta["n_sand"]
    n_water = meta["n_water"]
    frame_files = sorted(f for f in os.listdir(REC_DIR) if f.startswith("frame_"))
    only_frame = os.environ.get("PHASE7_ONLY_FRAME")  # debug: render a single recorded frame
    if only_frame is not None:
        frame_files = [f"frame_{int(only_frame):04d}.npz"]
    elif MAX_FRAMES is not None:
        frame_files = frame_files[:MAX_FRAMES]
    n_frames = len(frame_files)
    print(f"replaying {n_frames} frames from {REC_DIR}, {N_BUCKETS} wetness buckets", flush=True)

    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=1, gravity=(0.0, 0.0, -9.8)),
        rigid_options=gs.options.RigidOptions(
            gravity=(0.0, 0.0, 0.0),
            enable_collision=False,
            enable_self_collision=False,
        ),
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

    # real litter-scoop asset (objaverse 'Litter scoop/貓砂鏟', aligned to the tilt-box frame by
    # phase8b_litter_align.py); its frame origin coincides with the physical blade box
    shovel = scene.add_entity(
        morph=gs.morphs.Mesh(file=os.path.join(EXPERIMENTS_DIR, "assets", "litter_scoop_aligned.glb")),
        material=gs.materials.Kinematic(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=FRANKA_BASE_POS),
    )

    # the real sand stays bucket 0 (dry color); the ghosts only differ in surface color --
    # their per-frame instance transforms are overwritten from the recording below
    sand_morph = gs.morphs.Box(pos=(0.0, 0.0, 0.081), size=(0.68, 0.58, 0.16))
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
        pos=(1.05, -1.05, 0.75),
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
    hand_link = franka.get_link("hand")

    raw_dir = os.path.join(FRAMES_DIR, "phase7_gripper_shovel_sandwet_native_raw")
    os.makedirs(raw_dir, exist_ok=True)

    q_prev = None
    keyframe_idx = 0
    park_block = np.tile(np.array(PARK_POS, dtype=np.float64), (n_water, 1))
    pos_all = np.zeros((n_total, 1, 3), dtype=np.float32)
    act_all = np.zeros((n_total, 1), dtype=np.bool_)
    for i, fname in enumerate(frame_files):
        frame = np.load(os.path.join(REC_DIR, fname))
        blade_pos = frame["blade_pos"].astype(np.float64)
        blade_quat = frame["blade_quat"].astype(np.float64)

        # water is still driven through the physics-side fields + render-field sync
        water_pos = frame["water_pos"].astype(np.float64)
        buf = park_block.copy()
        buf[: len(water_pos)] = water_pos
        water.set_particles_pos(buf[None])

        shovel.set_pos(blade_pos)
        shovel.set_quat(blade_quat)
        handle_pos, handle_quat = blade_to_handle(blade_pos, blade_quat)

        # continuous arm tracking
        target_pos, target_quat = hand_target_from_handle(handle_pos, handle_quat)
        q = solve_ik(franka, hand_link, target_pos, target_quat, q_prev)
        q[-2:] = FINGER_GRIP
        franka.set_qpos(q)
        q_prev = q

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
                os.path.join(FRAMES_DIR, f"phase7_gripper_shovel_sandwet_native_frame_{keyframe_idx:04d}.png"),
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
