"""
Phase 7 offline re-render: replay the recorded Phase 7 run (experiments/recordings/
phase7_gripper_shovel/frame_*.npz) and re-render the native video with a CONTINUOUS arm.

The user found the live-run video's arm motion too jumpy frame-to-frame: each frame's IK was a
full multi-restart solve (max_samples=100), so consecutive frames could land on different
solution branches. This renderer fixes that without re-running the sand/water physics:

  - sand/water particles and the shovel are driven straight from the recorded npz frames
    (set_particles_pos writes the solver fields directly; no scene.step(), no physics),
    so the blade trajectory is bit-identical to the accepted run;
  - the arm IK trades exactness for continuity: frame 0 gets a full solve to establish a good
    branch, afterwards every frame warm-starts from the previous solution with
    max_samples=1 (no random restarts -> stays on the same branch), few iterations, and a
    per-frame joint increment clamp (DQ_MAX). If the grasp error ever exceeds REGRASP_ERR the
    tracker re-solves harder from the current pose (never triggered in practice).

Output overwrites experiments/videos/phase7_gripper_shovel.mp4 and the keyframes, and writes
ik_log_smooth.csv next to the recording for follow-error stats.

Usage: python phase7_gripper_render.py [max_frames]
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
VIDEO_PATH = os.path.join(EXPERIMENTS_DIR, "videos", "phase7_gripper_shovel.mp4")

DT = 1.0 / 60.0
KEYFRAME_STEPS = [0, 299, 354, 450, 555, 705, 765]

PARTICLE_RADIUS = 3.125e-3
BOX_LOWER = (-0.35, -0.30, 0.0)
BOX_UPPER = (0.35, 0.30, 0.80)
WALL_THICK = 0.01
WALL_HEIGHT = 0.18

BLADE_HALF = (0.15, 0.12, 0.01)
BLADE_ANGLE = math.radians(40.0)
HANDLE_LEN = 0.3
HANDLE_HALF_THICK = 0.015
HANDLE_ANGLE = math.radians(40.0)

DROPLET_POS = (0.02, 0.0, 0.205)
DROPLET_RADIUS = 0.04
MAX_RATIO = 0.3

FRANKA_BASE_POS = (-0.25, 0.38, 0.0)  # closer to the box: the rotate-end handle pose is the
# hardest to reach; a nearer base keeps the warm-started IK branch feasible there
FINGER_GRIP = 0.015
GRASP_DIST = 0.09
GRIP_ALONG = 0.08  # grip point shifted this far along the handle toward its free end (the
# handle's local +x): raises the hand at the low rotate-end pose, easing the wrist

# continuity scheme
DQ_MAX = 0.02  # rad per frame joint increment clamp (~69 deg/s, far above the needed rates)
REGRASP_ERR = 0.03  # m; re-solve harder only if the tracker drifts past this
PARK_POS = (0.0, 0.0, -10.0)  # absorbed water particles are parked below the floor


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
    restarts), then clamp the per-frame joint increment. If the warm-started solve does not
    converge well (kinematically hard region), retry with more iterations from the SAME
    starting point -- better convergence, never a branch jump."""
    if q_prev is None:
        # first frame: full multi-restart solve to establish a good branch
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
    if MAX_FRAMES is not None:
        frame_files = frame_files[:MAX_FRAMES]
    n_frames = len(frame_files)
    print(f"replaying {n_frames} frames from {REC_DIR}", flush=True)

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
    # phase8b_litter_align.py); its frame origin/center coincides with the physical blade box
    shovel = scene.add_entity(
        morph=gs.morphs.Mesh(file=os.path.join(EXPERIMENTS_DIR, "assets", "litter_scoop_aligned.glb")),
        material=gs.materials.Kinematic(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=FRANKA_BASE_POS),
    )
    sand = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.081), size=(0.68, 0.58, 0.16)),
        material=gs.materials.DEM.Sand(sampler="fcc", rho=2.5, max_ratio=MAX_RATIO),
        surface=gs.surfaces.Default(color=(0.87, 0.72, 0.53)),
    )
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
    hand_link = franka.get_link("hand")

    # PNG dump + external ffmpeg encode: the built-in recorder's renders early-return without
    # scene.step() (rasterizer skips updates when scene._t does not advance), and its video
    # write can silently no-op, so we render every frame ourselves with force_render=True
    raw_dir = os.path.join(FRAMES_DIR, "phase7_gripper_shovel_raw")
    os.makedirs(raw_dir, exist_ok=True)

    ik_log = open(os.path.join(REC_DIR, "ik_log_smooth.csv"), "w")
    ik_log.write("step,grasp_err,dq_max\n")

    q_prev = None
    keyframe_idx = 0
    park_block = np.tile(np.array(PARK_POS, dtype=np.float64), (n_water, 1))
    for i, fname in enumerate(frame_files):
        frame = np.load(os.path.join(REC_DIR, fname))
        blade_pos = frame["blade_pos"].astype(np.float64)
        blade_quat = frame["blade_quat"].astype(np.float64)

        # drive everything straight from the recording (no physics)
        sand.set_particles_pos(frame["sand_pos"].astype(np.float64)[None])
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
        dq_max = 0.0 if q_prev is None else float(np.abs(q - q_prev).max())
        q_prev = q

        achieved_pos = to_np(hand_link.get_pos()).astype(np.float64).reshape(-1)[:3]
        achieved_quat = to_np(hand_link.get_quat()).astype(np.float64).reshape(-1)[:4]
        grasp = achieved_pos + quat_to_R(achieved_quat) @ np.array([0.0, 0.0, GRASP_DIST])
        x_h = quat_to_R(handle_quat) @ np.array([1.0, 0.0, 0.0])
        grip_pos = handle_pos + x_h / np.linalg.norm(x_h) * GRIP_ALONG
        grasp_err = float(np.linalg.norm(grasp - grip_pos))
        ik_log.write(f"{i},{grasp_err:.6f},{dq_max:.6f}\n")

        # refresh the render-only particle buffers (normally done inside scene.step())
        dem.update_render_fields()
        flip.update_render_fields()
        rgb, *_ = cam.render(rgb=True, force_render=True)
        rgb = rgb[0] if isinstance(rgb, list) else rgb
        imageio.imwrite(os.path.join(raw_dir, f"frame_{i:04d}.png"), rgb)

        if i in KEYFRAME_STEPS:
            imageio.imwrite(
                os.path.join(FRAMES_DIR, f"phase7_gripper_shovel_frame_{keyframe_idx:04d}.png"),
                rgb,
            )
            keyframe_idx += 1

        if (i + 1) % 60 == 0:
            print(f"frame {i + 1}/{n_frames}  grasp_err = {grasp_err:.5f}  dq_max = {dq_max:.5f}", flush=True)

    ik_log.close()
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
