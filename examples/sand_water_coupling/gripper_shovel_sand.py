"""
Phase 7 gripper-arm shovel demo: a Franka Panda grips the shovel handle and follows it through
the exact same scoop motion as the best wet-sand demo (examples/sand_water_coupling/shovel_wet_sand.py,
viscosity_coeff = 0.01 + contact release, n_wet_lifted ~ 5666).

The shovel dynamics are replicated at the code level: the same scene parameters, the same
action constants (descend / insert / pivot / lift / hold velocities and step counts) drive the
same DEM tilt-box obstacle, so the blade trajectory is bit-identical to the best demo. The arm
is a pure kinematic follower: every step the handle midpoint world pose is recomputed from the
solver's actual blade pose (same blade->handle transform as the demo), the hand link target is
built with the convention calibrated in phase7_franka_smoke.py (hand x-axis || handle axis,
hand z-axis = world -y, grasp point 0.09 m along +z pinned to the handle midpoint), IK is
solved with warm start, the finger joints are overwritten to the 0.015 m grip, and the result
is hard-set with set_qpos (zero_velocity). Rigid gravity and collisions are off, and the
LegacyCoupler has no rigid<->dem / rigid<->flip channel, so the arm has zero effect on the
sand/water physics.

Usage: python phase7_gripper_shovel.py [max_steps] [tag]
  max_steps  cap on the 766-step run (for short scaffold checks)
  tag        suffix for output paths (video / recordings / frames), default "" (full run)

Outputs (tagged):
- experiments/videos/phase7_gripper_shovel{tag}.mp4          (native render)
- experiments/videos/phase7_gripper_shovel{tag}_sandwet.mp4  (per-grain wetness render)
- experiments/recordings/phase7_gripper_shovel{tag}/frame_*.npz + meta.json
- experiments/recordings/phase7_gripper_shovel{tag}/ik_log.csv (per-step IK follow error)
- experiments/frames/phase7_gripper_shovel{tag}_frame_*.png  (keyframes)
"""

import math
import os
import subprocess
import sys
import time

import imageio
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import genesis as gs

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(EXPERIMENTS_DIR, "frames")

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
TAG = sys.argv[2] if len(sys.argv) > 2 else ""

VIDEO_PATH = os.path.join(EXPERIMENTS_DIR, "videos", f"phase7_gripper_shovel{TAG}.mp4")
VIDEO_WET_PATH = os.path.join(EXPERIMENTS_DIR, "videos", f"phase7_gripper_shovel{TAG}_sandwet.mp4")
WET_RAW_DIR = os.path.join(FRAMES_DIR, f"phase7_gripper_shovel{TAG}_sandwet_raw")
REC_DIR = os.path.join(EXPERIMENTS_DIR, "recordings", f"phase7_gripper_shovel{TAG}")

# --- identical scene/action constants to the best demo (shovel_wet_sand.py) ---
DT = 1.0 / 60.0
N_SETTLE_STEPS = 300  # 5 s: droplet falls and is absorbed
N_DESCEND_STEPS = 55  # 0.92 s
N_INSERT_STEPS = 96  # 1.6 s
N_ROTATE_STEPS = 105  # 1.75 s
N_LIFT_STEPS = 150  # 2.5 s
N_HOLD_STEPS = 60  # 1 s
DESCEND_VEL = (0.0, 0.0, -0.12)
INSERT_VEL = (0.115, 0.0, -0.0964)
ROTATE_VEL = (0.0205, 0.0, 0.0563)
ROTATE_OMEGA = (0.0, -math.radians(40.0) / 1.75, 0.0)
LIFT_VEL = (0.0, 0.0, 0.10)
KEYFRAME_STEPS = [0, 299, 354, 450, 555, 705, 765]
WET_EVERY = 3

PARTICLE_RADIUS = 3.125e-3
BOX_LOWER = (-0.35, -0.30, 0.0)
BOX_UPPER = (0.35, 0.30, 0.80)
WALL_THICK = 0.01
WALL_HEIGHT = 0.18

BLADE_HALF = (0.15, 0.12, 0.01)
BLADE_ANGLE = math.radians(40.0)
BLADE_POS0 = (-0.20, 0.0, 0.369)
HANDLE_LEN = 0.3
HANDLE_HALF_THICK = 0.015
HANDLE_ANGLE = math.radians(40.0)

DROPLET_POS = (0.02, 0.0, 0.205)
DROPLET_RADIUS = 0.04

MAX_RATIO = 0.3

# --- Phase 7 arm constants (calibrated in phase7_franka_smoke.py) ---
FRANKA_BASE_POS = (-0.25, 0.45, 0.0)  # 0.15 m behind the +y wall, reaching toward -y
FINGER_GRIP = 0.015  # 3 cm gap = handle cross-section (visual grip only, no contact physics)
GRASP_DIST = 0.09  # hand origin -> grasp point along hand local +z


def to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def quat_to_R(q):
    # rotation matrix of a (w, x, y, z) unit quaternion
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def quat_mul(q1, q2):
    # hamilton product of (w, x, y, z) quaternions
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
    # (w, x, y, z) quaternion of a rotation matrix
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
    """World pose of the handle midpoint, same transform as the demo script."""
    handle_quat_rel = np.array(
        [math.cos((math.pi + HANDLE_ANGLE) / 2.0), 0.0, math.sin((math.pi + HANDLE_ANGLE) / 2.0), 0.0]
    )
    handle_dir_local = np.array([-math.cos(HANDLE_ANGLE), 0.0, math.sin(HANDLE_ANGLE)])
    handle_offset_local = np.array([-BLADE_HALF[0], 0.0, 0.0]) + handle_dir_local * (HANDLE_LEN / 2.0)
    handle_pos = blade_pos + quat_to_R(blade_quat) @ handle_offset_local
    handle_quat = quat_mul(blade_quat, handle_quat_rel)
    return handle_pos, handle_quat


def hand_target_from_handle(handle_pos, handle_quat):
    """IK target (pos, quat wxyz) for the franka 'hand' link: x-axis || handle axis, z-axis
    = world -y (fingers reach from the robot side across the handle), grasp point pinned at
    the handle midpoint."""
    x_h = quat_to_R(handle_quat) @ np.array([1.0, 0.0, 0.0])
    x_h = x_h / np.linalg.norm(x_h)
    z_h = np.array([0.0, -1.0, 0.0])
    z_h = z_h - np.dot(z_h, x_h) * x_h
    z_h = z_h / np.linalg.norm(z_h)
    y_h = np.cross(z_h, x_h)
    R_hand = np.column_stack([x_h, y_h, z_h])
    hand_pos = handle_pos - R_hand @ np.array([0.0, 0.0, GRASP_DIST])
    return hand_pos, R_to_quat(R_hand)


def render_wet_frame(pos, ratio, path):
    fig = plt.figure(figsize=(9.6, 7.2), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    t = np.clip(ratio / MAX_RATIO, 0.0, 1.0)
    dry = np.array([0.87, 0.72, 0.53])
    wet = np.array([0.25, 0.13, 0.06])
    colors = dry[None, :] * (1.0 - t[:, None]) + wet[None, :] * t[:, None]
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=2, c=colors, alpha=0.9, edgecolors="none")
    ax.set_xlim(BOX_LOWER[0], BOX_UPPER[0])
    ax.set_ylim(BOX_LOWER[1], BOX_UPPER[1])
    ax.set_zlim(0.0, 0.6)
    ax.set_box_aspect((0.7, 0.6, 0.6))
    ax.view_init(elev=18, azim=-60)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=1, gravity=(0.0, 0.0, -9.8)),
        # the arm is a pure kinematic follower: no rigid gravity, no rigid-rigid collisions
        # (the LegacyCoupler has no rigid<->dem / rigid<->flip channel, so the arm can never
        # touch the sand/water physics)
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

    blade_quat0 = np.array(
        [math.cos(BLADE_ANGLE / 2.0), 0.0, math.sin(BLADE_ANGLE / 2.0), 0.0]
    )
    # real litter-scoop asset (objaverse 'Litter scoop/貓砂鏟', aligned to the tilt-box frame by
    # phase8b_litter_align.py): pure visualization; physics stays with the DEM tilt-box obstacle
    shovel = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "litter_scoop_aligned.glb"),
            pos=BLADE_POS0,
            quat=tuple(blade_quat0),
        ),
        material=gs.materials.Kinematic(),
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=FRANKA_BASE_POS),
    )

    sand = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.081),
            size=(0.68, 0.58, 0.16),
        ),
        material=gs.materials.DEM.Sand(sampler="fcc", rho=2.5, max_ratio=MAX_RATIO),
        surface=gs.surfaces.Default(color=(0.87, 0.72, 0.53)),
    )
    water = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=DROPLET_POS,
            radius=DROPLET_RADIUS,
        ),
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
    dem.set_tilt_box_obstacle(
        BLADE_HALF,
        BLADE_POS0,
        quat=tuple(blade_quat0),
        handle=(HANDLE_LEN, HANDLE_HALF_THICK, HANDLE_ANGLE),
    )
    hand_link = franka.get_link("hand")
    print(
        f"n_water = {water.n_particles}, n_sand = {sand.n_particles}, "
        f"single_ratio = {flip._single_ratio:.4f}",
        flush=True,
    )

    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(WET_RAW_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    os.makedirs(REC_DIR, exist_ok=True)

    with open(os.path.join(REC_DIR, "meta.json"), "w") as f_meta:
        json.dump(
            {
                "dt": DT,
                "particle_radius": PARTICLE_RADIUS,
                "grid_res": 128,
                "max_ratio": MAX_RATIO,
                "viscosity_coeff": 0.01,
                "box_lower": BOX_LOWER,
                "box_upper": BOX_UPPER,
                "wall_thick": WALL_THICK,
                "wall_height": WALL_HEIGHT,
                "blade_half": BLADE_HALF,
                "handle": {"len": HANDLE_LEN, "half_thick": HANDLE_HALF_THICK, "angle": HANDLE_ANGLE},
                "droplet_pos": DROPLET_POS,
                "droplet_radius": DROPLET_RADIUS,
                "n_sand": sand.n_particles,
                "n_water": water.n_particles,
                "franka_base_pos": FRANKA_BASE_POS,
                "finger_grip": FINGER_GRIP,
                "grasp_dist": GRASP_DIST,
            },
            f_meta,
            indent=2,
        )
    ik_log = open(os.path.join(REC_DIR, "ik_log.csv"), "w")
    ik_log.write("step,ik_pos_err,ik_rot_err,grasp_err\n")

    cam.start_recording(save_to_filename=VIDEO_PATH, fps=60)

    t0 = time.time()
    i_descend = N_SETTLE_STEPS
    i_insert = i_descend + N_DESCEND_STEPS
    i_rotate = i_insert + N_INSERT_STEPS
    i_lift = i_rotate + N_ROTATE_STEPS
    i_hold = i_lift + N_LIFT_STEPS
    n_steps = min(i_hold + N_HOLD_STEPS, MAX_STEPS)
    keyframe_idx = 0
    wet_idx = 0
    n_water0 = water.n_particles
    for i in range(n_steps):
        if i == i_descend:
            dem.set_tilt_box_vel(DESCEND_VEL)
            print(f"step {i + 1}: blade starts descending vertically at {DESCEND_VEL} m/s", flush=True)
        elif i == i_insert:
            dem.set_tilt_box_vel(INSERT_VEL)
            print(f"step {i + 1}: blade starts inserting along the blade direction at {INSERT_VEL} m/s", flush=True)
        elif i == i_rotate:
            dem.set_tilt_box_vel(ROTATE_VEL, omega=ROTATE_OMEGA)
            print(
                f"step {i + 1}: blade starts pivoting up about the trailing edge "
                f"(omega_y = {ROTATE_OMEGA[1]:.4f} rad/s)",
                flush=True,
            )
        elif i == i_lift:
            dem.set_tilt_box_vel(LIFT_VEL)
            print(f"step {i + 1}: blade starts lifting vertically at {LIFT_VEL} m/s", flush=True)
        elif i == i_hold:
            dem.set_tilt_box_vel((0.0, 0.0, 0.0))
            print(f"step {i + 1}: blade holds", flush=True)

        # sync the visualization-only kinematic shovel with the solver's obstacle
        blade_pos = to_np(dem.get_tilt_box_pos()[0]).astype(np.float64)
        blade_quat = to_np(dem.get_tilt_box_quat()[0]).astype(np.float64)
        shovel.set_pos(blade_pos)
        shovel.set_quat(blade_quat)
        handle_pos, handle_quat = blade_to_handle(blade_pos, blade_quat)

        # arm follows the handle: IK to the calibrated hand target, grip hard-set, zero velocity
        target_pos, target_quat = hand_target_from_handle(handle_pos, handle_quat)
        q, err = franka.inverse_kinematics(
            hand_link,
            pos=target_pos,
            quat=target_quat,
            return_error=True,
            max_samples=100,
            max_solver_iters=100,
            damping=0.005,
            pos_tol=1e-4,
            rot_tol=1e-3,
        )
        q = to_np(q).astype(np.float64).reshape(-1).copy()
        err = to_np(err).astype(np.float64).reshape(-1)[:6]
        q[-2:] = FINGER_GRIP
        franka.set_qpos(q)

        scene.step()

        achieved_pos = to_np(hand_link.get_pos()).astype(np.float64).reshape(-1)[:3]
        achieved_quat = to_np(hand_link.get_quat()).astype(np.float64).reshape(-1)[:4]
        grasp = achieved_pos + quat_to_R(achieved_quat) @ np.array([0.0, 0.0, GRASP_DIST])
        grasp_err = float(np.linalg.norm(grasp - handle_pos))
        ik_log.write(f"{i},{np.linalg.norm(err[:3]):.6f},{np.linalg.norm(err[3:]):.6f},{grasp_err:.6f}\n")

        # per-frame recording for offline re-rendering (blade pose = the one just synced above)
        sand_pos = sand.get_particles_pos().cpu().numpy()[0].astype(np.float32)
        sand_ratio = dem.particles.ratio.to_numpy()[:, 0].astype(np.float32)
        water_active_mask = flip.particles.active.to_numpy()[:, 0].astype(bool)
        water_pos = water.get_particles_pos().cpu().numpy()[0][water_active_mask].astype(np.float32)
        np.savez(
            os.path.join(REC_DIR, f"frame_{i:04d}.npz"),
            sand_pos=sand_pos,
            sand_ratio=sand_ratio,
            water_pos=water_pos,
            blade_pos=np.asarray(blade_pos, dtype=np.float32),
            blade_quat=np.asarray(blade_quat, dtype=np.float32),
        )

        if i % WET_EVERY == 0:
            render_wet_frame(sand_pos, sand_ratio, os.path.join(WET_RAW_DIR, f"frame_{wet_idx:04d}.png"))
            wet_idx += 1

        if i in KEYFRAME_STEPS:
            rgb, *_ = cam.render(rgb=True)
            imageio.imwrite(
                os.path.join(FRAMES_DIR, f"phase7_gripper_shovel{TAG}_frame_{keyframe_idx:04d}.png"),
                rgb[0] if isinstance(rgb, list) else rgb,
            )
            keyframe_idx += 1

        if (i + 1) % 30 == 0:
            pos = sand.get_particles_pos().cpu().numpy()[0]
            vel = sand.get_particles_vel().cpu().numpy()[0]
            ratio = dem.particles.ratio.to_numpy()[:, 0]
            n_active = int(flip.particles.active.to_numpy()[:, 0].sum())
            theta = math.degrees(2.0 * math.atan2(blade_quat[2], blade_quat[0]))
            print(
                f"step {i + 1:4d}/{n_steps}  max|v| = {np.linalg.norm(vel, axis=1).max():8.4f}  "
                f"min_z = {pos[:, 2].min():+.6f}  n_lifted = {(pos[:, 2] > 0.2).sum():5d}  "
                f"n_wet_lifted = {((pos[:, 2] > 0.2) & (ratio > 0.02)).sum():5d}  "
                f"water_active = {n_active}/{n_water0}  ratio_mean = {ratio.mean():.4f}  "
                f"blade = ({blade_pos[0]:+.3f}, {blade_pos[2]:.3f})  theta = {theta:+6.2f}  "
                f"grasp_err = {grasp_err:.5f}  "
                f"nan = {np.isnan(pos).any()}  elapsed = {time.time() - t0:.1f}s",
                flush=True,
            )

    ik_log.close()
    cam.stop_recording()

    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", "20",
            "-i", os.path.join(WET_RAW_DIR, "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", VIDEO_WET_PATH,
        ],
        check=True,
        capture_output=True,
    )
    print(f"videos saved to {VIDEO_PATH} and {VIDEO_WET_PATH}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
