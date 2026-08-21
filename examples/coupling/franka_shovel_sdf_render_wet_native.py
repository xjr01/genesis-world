"""
Phase 9 wet-sand NATIVE renderer for the half-scale Franka + litter-scoop demo: wet grains darken
through Genesis's own rendering pipeline -- no projection compositing, no shader/source changes.
(Armed, half-scale sibling of phase8d_render_wet_native.py.)

Mechanism: identical to phase8d (N wetness-bucket DEM entities, per-frame overwrite of
`dem.particles_render.pos/active` from the recorded npz, no scene.step()). The Franka is re-driven
every frame with the same continuous-IK follower as the phase9 run (warm start + joint increment
clamp) from the recorded blade pose, so the rendered grasp matches the run's ik_log.

Usage: python phase9_render_wet_native.py [max_frames] [tag]
Env:   PHASE9_N_BUCKETS (default 4), PHASE9_ONLY_FRAME=<n> (render a single recorded frame)
Output: experiments/videos/phase9_franka_shovel_sdf{tag}.mp4 (overwrites the plain inline render)
        experiments/frames/phase9_sandwet_native{tag}_frame_*.png (keyframes)
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
TAG = sys.argv[2] if len(sys.argv) > 2 else ""

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(EXPERIMENTS_DIR, "frames")
REC_DIR = os.path.join(EXPERIMENTS_DIR, "recordings", f"phase9_franka_shovel_sdf{TAG}")
VIDEO_PATH = os.path.join(EXPERIMENTS_DIR, "videos", f"phase9_franka_shovel_sdf{TAG}.mp4")

DT = 1.0 / 60.0
KEYFRAME_STEPS = [0, 299, 410, 522, 592, 740, 872]

SC = 0.5
PARTICLE_RADIUS = 3.125e-3 * SC
BOX_LOWER = (-0.35 * SC, -0.30 * SC, 0.0)
BOX_UPPER = (0.35 * SC, 0.30 * SC, 0.80 * SC)
WALL_THICK = 0.01 * SC
WALL_HEIGHT = 0.24 * SC

BLADE_HALF = (0.15 * SC, 0.117 * SC, 0.01 * SC)
HANDLE_LEN = 0.3 * SC
HANDLE_ANGLE = math.radians(40.0)

DROPLET_POS = (-0.02 * SC, 0.0, 0.245 * SC)
DROPLET_RADIUS = 0.04 * SC
MAX_RATIO = 0.3

FRANKA_BASE_POS = (-0.25, 0.45, 0.0)
FINGER_GRIP = 0.0075
GRASP_DIST = 0.09
DQ_MAX = 0.02

PARK_POS = (0.0, 0.0, -10.0)  # absorbed water particles are parked below the floor

# wetness buckets: bucket 0 = the real sand entity (DRY_COLOR), buckets 1..N-1 = ghosts.
# 4 buckets of the 464k-grain bed is the RTX 4060 limit (see phase8d_render_wet_native.py).
N_BUCKETS = int(os.environ.get("PHASE9_N_BUCKETS", "4"))
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
    z_h = np.array([0.0, -1.0, 0.0])
    z_h = z_h - np.dot(z_h, x_h) * x_h
    z_h = z_h / np.linalg.norm(z_h)
    y_h = np.cross(z_h, x_h)
    R_hand = np.column_stack([x_h, y_h, z_h])
    hand_pos = handle_pos - R_hand @ np.array([0.0, 0.0, GRASP_DIST])
    return hand_pos, R_to_quat(R_hand)


def solve_ik(franka, hand_link, pos, quat, q_prev):
    """Same continuous-IK follower as the phase9 run (warm start, clamped increments)."""
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
    only_frame = os.environ.get("PHASE9_ONLY_FRAME")  # debug: render a single recorded frame
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

    # half-scale litter-scoop asset, driven by the recorded blade pose
    shovel = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=os.path.join(EXPERIMENTS_DIR, "assets", "litter_scoop_aligned.glb"),
            scale=SC,
        ),
        material=gs.materials.Kinematic(),
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=FRANKA_BASE_POS),
    )

    # the real sand stays bucket 0 (dry color); the ghosts only differ in surface color --
    # their per-frame instance transforms are overwritten from the recording below
    sand_morph = gs.morphs.Box(pos=(0.0, 0.0, 0.1005 * SC), size=(0.68 * SC, 0.58 * SC, 0.20 * SC))
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
        pos=(0.60, -0.65, 0.55),  # same raised viewpoint as the phase9 run
        lookat=(-0.05, 0.0, 0.10),
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

    raw_dir = os.path.join(FRAMES_DIR, f"phase9_sandwet_native{TAG}_raw")
    # wipe stale frames: ffmpeg globs frame_*.png, leftovers from a longer earlier run would be
    # appended to the video (the 2026-08-21 "wrong last frames" bug)
    if os.path.isdir(raw_dir):
        for stale in os.listdir(raw_dir):
            os.remove(os.path.join(raw_dir, stale))
    os.makedirs(raw_dir, exist_ok=True)

    keyframe_idx = 0
    q_prev = None
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

        blade_pos = frame["blade_pos"].astype(np.float64)
        blade_quat = frame["blade_quat"].astype(np.float64)
        shovel.set_pos(blade_pos)
        shovel.set_quat(blade_quat)

        # re-drive the arm with the same continuous-IK follower as the run
        handle_pos, handle_quat = blade_to_handle(blade_pos, blade_quat)
        target_pos, target_quat = hand_target_from_handle(handle_pos, handle_quat)
        q = solve_ik(franka, hand_link, target_pos, target_quat, q_prev)
        q_prev = q
        q = q.copy()
        q[-2:] = FINGER_GRIP
        franka.set_qpos(q)

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
                os.path.join(FRAMES_DIR, f"phase9_sandwet_native{TAG}_frame_{keyframe_idx:04d}.png"),
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
    sys.exit(main())
