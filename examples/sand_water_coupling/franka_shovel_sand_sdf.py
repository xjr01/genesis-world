"""
Phase 9: half-scale wet-sand scoop demo with a Franka Panda gripping the litter scoop.

User request (2026-08-21): the Franka looked too small next to the full-size litter scoop, so the
WHOLE scene is scaled down by 2x (sand bed, grains, droplet, scoop, trajectory -- everything) and
the real-size Franka grips the now 15 cm pan / 15 mm handle. The physics is the validated v4
trajectory (50 deg initial tilt, insert parallel to the blade face, pivot up to flat) with the
scoop's true slotted SDF as the DEM boundary; the arm is the phase7 continuous-IK follower
(warm start + per-frame joint increment clamp), a pure kinematic entity with zero effect on the
sand/water (LegacyCoupler has no rigid<->dem / rigid<->flip channel).

Scale convention: SC = 0.5 multiplies every length of the phase5_shovel_wet_sdf.py v4 setup;
angles, step counts, dt and the DEM/FLIP material coefficients are unchanged. The DEM substep
ddt ∝ radius halves automatically (velocity-adaptive subcycling), so the run costs ~2x v4.

Usage: python phase9_franka_shovel_sdf.py [max_steps] [sdf_npz] [tag]
  max_steps  cap on the 873-step run (for short scaffold checks)
  sdf_npz    SDF grid file under experiments/assets (default: half-scale slotted pan)
  tag        suffix for output paths, default "" (full run)

Outputs (tagged):
- experiments/videos/phase9_franka_shovel_sdf{tag}.mp4          (native render)
- experiments/videos/phase9_franka_shovel_sdf{tag}_sandwet.mp4  (per-grain wetness, matplotlib)
- experiments/recordings/phase9_franka_shovel_sdf{tag}/frame_*.npz + meta.json + ik_log.csv
- experiments/frames/phase9_franka_shovel_sdf{tag}_frame_*.png  (keyframes)
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
SDF_NPZ = os.path.join(
    EXPERIMENTS_DIR, "assets", sys.argv[2] if len(sys.argv) > 2 else "litter_scoop_sdf_slots_s0.5.npz"
)
TAG = sys.argv[3] if len(sys.argv) > 3 else ""

VIDEO_PATH = os.path.join(EXPERIMENTS_DIR, "videos", f"phase9_franka_shovel_sdf{TAG}.mp4")
VIDEO_WET_PATH = os.path.join(EXPERIMENTS_DIR, "videos", f"phase9_franka_shovel_sdf{TAG}_sandwet.mp4")
WET_RAW_DIR = os.path.join(FRAMES_DIR, f"phase9_franka_shovel_sdf{TAG}_sandwet_raw")
REC_DIR = os.path.join(EXPERIMENTS_DIR, "recordings", f"phase9_franka_shovel_sdf{TAG}")

SC = 0.5  # whole-scene scale factor (user request 2026-08-21)

# --- v4 trajectory (phase5_shovel_wet_sdf.py), all lengths x SC, angles/steps unchanged ---
DT = 1.0 / 60.0
N_SETTLE_STEPS = 300  # 5 s: droplet falls and is absorbed
N_DESCEND_STEPS = 60  # 1.00 s: tip descends until it is 2 cm/SC = 1 cm into the bed
N_ROTATE_STEPS = 140  # 2.33 s
N_LIFT_STEPS = 200  # 3.33 s
N_HOLD_STEPS = 80  # 1.33 s
DESCEND_VEL = (0.0, 0.0, -0.0731 * SC)  # straight down: tip (-0.0718, 0.1270) -> (-0.0718, 0.0905)
INSERT_VEL = (0.0723 * SC, 0.0, -0.0862 * SC)  # 50 deg below horizontal, parallel to the blade face
ROTATE_VEL = (0.0237 * SC, 0.0, 0.0508 * SC)  # trailing-edge-pivot emulation (mid-rotation 25 deg)
ROTATE_OMEGA = (0.0, -math.radians(50.0) / (140.0 / 60.0), 0.0)  # 50 deg -> flat about the blade center
LIFT_VEL = (0.0, 0.0, 0.075 * SC)
# --- trajectory v5 "wincombo" (user request 2026-08-25: visually ZERO splash in the first-touch
# window, abs ~360-420). Segment-vehicle experiments (phase9_segment.py) showed the splash is a
# bow-wave force-chain pop-out at the leading edge, robust to any single lever; the validated
# combination is: (a) windowed contact parameters -- grain-grain restitution 0.5 -> 0.1 and sand
# Young's modulus x 0.25 from abs 330 until WINCOMBO_END (both are re-read by
# dem_solver._material_constants() on every advance, plain runtime switches, no rebuild), then a
# 30-step linear stiffness ramp-back; (b) a slower insert start -- path-units 0-30 at 1/3 speed
# (90 steps, the tip transits the surface layer where the pops launch), a 30-step linear ramp
# 1/3 -> 1.0 over units 30-50, then full speed for units 50-93 (43 steps). The speed-up happens
# only with the wedge fully submerged (tip z 0.069 -> 0.054), which the segment runs proved does
# not re-splash. Insert 93 -> 163 steps; total 873 -> 943 steps. All other phases unchanged.
N_INSERT_STEPS = 163  # 2.72 s: 90 slow (1/3) + 30 ramp + 43 full = 93 original path-units
INSERT_SLOW_STEPS = 90  # path-units 0-30 at 1/3 speed
INSERT_SLOW_RATIO = 1.0 / 3.0
INSERT_RAMP_STEPS = 30  # linear 1/3 -> 1.0, mean 2/3 -> 20 path-units
WINCOMBO_START = 330  # abs step: restitution/Young window opens (first touch ~344)
WINCOMBO_REST = 0.1  # windowed grain-grain restitution
WINCOMBO_YOUNG_F = 0.25  # windowed Young's modulus factor
WINCOMBO_END = 530  # insert ends at abs 523; params ramp back 530 -> 560
WINCOMBO_RAMP = 30
KEYFRAME_STEPS = [0, 299, 410, 593, 663, 810, 942]  # settle-end, splash window, rotate-mid/end, lift, final
WET_EVERY = 3

PARTICLE_RADIUS = 3.125e-3 * SC  # 1.5625 mm: grain diameter 3.125 mm = 0.4 / 128, still dx = 2r
BOX_LOWER = (-0.35 * SC, -0.30 * SC, 0.0)
BOX_UPPER = (0.35 * SC, 0.30 * SC, 0.80 * SC)
WALL_THICK = 0.01 * SC
WALL_HEIGHT = 0.24 * SC  # taller than the 0.10 m sand bed

BLADE_HALF = (0.15 * SC, 0.117 * SC, 0.01 * SC)  # meta only; physics is the SDF obstacle
BLADE_ANGLE = math.radians(50.0)  # v4: initial tilt 50 deg, ending flat
BLADE_POS0 = (-0.24 * SC, 0.0, 0.369 * SC)  # tip starts at (-0.0718, 0.1270)
HANDLE_LEN = 0.3 * SC
HANDLE_HALF_THICK = 0.015 * SC
HANDLE_ANGLE = math.radians(40.0)  # asset property: handle rises 40 deg above the pan plane

DROPLET_POS = (-0.02 * SC, 0.0, 0.245 * SC)  # contact release at the 0.10 m sand surface
DROPLET_RADIUS = 0.04 * SC

MAX_RATIO = 0.3
SURFACE_TENSION = 0.02  # ~3x the reference 0.007 (user-selected): the wet clump survives the slots
VISCOSITY = 0.01

# --- arm constants (phase7, calibrated in phase7_franka_smoke.py; lengths NOT scaled: the
# robot itself stays real-size, only GRASP_DIST/FINGER_GRIP are robot/handle geometry) ---
FRANKA_BASE_POS = (-0.25, 0.45, 0.0)  # phase9_ik_probe.py: worst grasp_err 9e-5 over all key
# handle poses (nearer bases fail the vertical-handle 50-deg initial / lift-end poses)
FINGER_GRIP = 0.0075  # 15 mm gap = half-scale handle cross-section (visual grip only)
GRASP_DIST = 0.09  # hand origin -> grasp point along hand local +z (Franka hand geometry)
DQ_MAX = 0.02  # rad per frame joint increment clamp (continuous IK, phase7_gripper_render.py)


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
    """World pose of the handle midpoint (same convention as phase7, lengths x SC)."""
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


def solve_ik(franka, hand_link, pos, quat, q_prev):
    """Continuous IK (phase7_gripper_render.py): warm-start from the previous solution on the same
    branch (no random restarts), then clamp the per-frame joint increment. If the warm-started
    solve does not converge well, retry with more iterations from the SAME starting point --
    better convergence, never a branch jump."""
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


def render_wet_frame(pos, ratio, path):
    fig = plt.figure(figsize=(9.6, 7.2), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    # ratio in [0, MAX_RATIO] -> light tan (dry) to dark brown (wet)
    t = np.clip(ratio / MAX_RATIO, 0.0, 1.0)
    dry = np.array([0.87, 0.72, 0.53])
    wet = np.array([0.25, 0.13, 0.06])
    colors = dry[None, :] * (1.0 - t[:, None]) + wet[None, :] * t[:, None]
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=2, c=colors, alpha=0.9, edgecolors="none")
    ax.set_xlim(BOX_LOWER[0], BOX_UPPER[0])
    ax.set_ylim(BOX_LOWER[1], BOX_UPPER[1])
    ax.set_zlim(0.0, 0.3)
    ax.set_box_aspect((0.35, 0.3, 0.3))
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
            surface_tension_coeff=SURFACE_TENSION,
            # user request (2026-08-22): grain-grain restitution 0.5 (spring-dashpot normal damping;
            # recorded deviation from the reference's strictly-elastic contact) to calm the
            # first-touch splash at ORIGINAL trajectory speed
            restitution=0.5,
            lower_bound=BOX_LOWER,
            upper_bound=BOX_UPPER,
        ),
        flip_options=gs.options.FLIPOptions(
            grid_res=128,  # dx ~ 3.125 mm over the 0.4 m extent = the sand particle diameter (dx = 2r,
            # same relative resolution as the validated full-scale run -> absorption stays active)
            viscosity_coeff=VISCOSITY,
            lower_bound=BOX_LOWER,
            upper_bound=BOX_UPPER,
        ),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())

    # visualization-only walls; physical containment is the DEM domain bounds themselves
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
    # the same real litter-scoop asset at 0.5x: pure visualization; physics is the DEM SDF obstacle
    shovel = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=os.path.join(EXPERIMENTS_DIR, "assets", "litter_scoop_aligned.glb"),
            pos=BLADE_POS0,
            quat=tuple(blade_quat0),
            scale=SC,
        ),
        material=gs.materials.Kinematic(),
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=FRANKA_BASE_POS),
    )

    sand = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.1005 * SC),
            size=(0.68 * SC, 0.58 * SC, 0.20 * SC),
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
        pos=(0.60, -0.65, 0.55),  # raised viewpoint (user request 2026-08-21) on the half-scale box
        lookat=(-0.05, 0.0, 0.10),
        fov=45,
    )

    scene.build()
    dem = scene.sim.dem_solver
    flip = scene.sim.flip_solver
    # physical boundary = the half-scale litter-scoop SDF (same mesh x 0.5, true slotted pan floor)
    sdf_data = np.load(SDF_NPZ)
    dem.set_sdf_obstacle(
        sdf_data["sdf_val"],
        sdf_data["dims"],
        sdf_data["origin"],
        sdf_data["cell"],
        BLADE_POS0,
        quat=tuple(blade_quat0),
    )
    hand_link = franka.get_link("hand")
    print(
        f"n_water = {water.n_particles}, n_sand = {sand.n_particles}, "
        f"single_ratio = {flip._single_ratio:.4f}",
        flush=True,
    )

    os.makedirs(FRAMES_DIR, exist_ok=True)
    # wipe stale wet frames: ffmpeg globs frame_*.png, leftovers from a longer earlier run would
    # be appended to the video (the 2026-08-21 "wrong last frames" bug)
    if os.path.isdir(WET_RAW_DIR):
        for stale in os.listdir(WET_RAW_DIR):
            os.remove(os.path.join(WET_RAW_DIR, stale))
    os.makedirs(WET_RAW_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    os.makedirs(REC_DIR, exist_ok=True)

    # static scene parameters for offline re-rendering of the per-frame npz recordings
    with open(os.path.join(REC_DIR, "meta.json"), "w") as f_meta:
        json.dump(
            {
                "dt": DT,
                "scale": SC,
                "particle_radius": PARTICLE_RADIUS,
                "grid_res": 128,
                "max_ratio": MAX_RATIO,
                "viscosity_coeff": VISCOSITY,
                "surface_tension_coeff": SURFACE_TENSION,
                "restitution": 0.5,
                "trajectory": "v5 wincombo: insert 163 steps (90 @1/3 + 30 ramp + 43 full); "
                "restitution 0.1 + young x0.25 window abs 330-530, 30-step ramp-back",
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
    ik_log.write("step,grasp_err,dq_max\n")

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
    q_prev = None
    n_water0 = water.n_particles
    young0 = float(dem._entities[0].material.young_modulus)  # for the wincombo window ramp-back
    for i in range(n_steps):
        # wincombo window (trajectory v5): soft + damped contacts during the first-touch/insert
        # window, then a 30-step linear stiffness ramp-back (see the constants block above)
        if i == WINCOMBO_START:
            dem._restitution = WINCOMBO_REST
            dem._entities[0].material.young_modulus = young0 * WINCOMBO_YOUNG_F
            print(
                f"step {i + 1}: wincombo window opens (restitution {WINCOMBO_REST}, "
                f"young x{WINCOMBO_YOUNG_F}) until step {WINCOMBO_END}",
                flush=True,
            )
        if i == WINCOMBO_END:
            dem._restitution = 0.5
            print(f"step {i + 1}: wincombo restitution restored to 0.5", flush=True)
        if i >= WINCOMBO_END:
            g = min(1.0, (i - WINCOMBO_END + 1) / WINCOMBO_RAMP)
            dem._entities[0].material.young_modulus = young0 * (WINCOMBO_YOUNG_F + (1.0 - WINCOMBO_YOUNG_F) * g)
        if i == i_descend:
            dem.set_sdf_obstacle_vel(DESCEND_VEL)
            print(f"step {i + 1}: blade starts descending vertically at {DESCEND_VEL} m/s", flush=True)
        elif i_insert <= i < i_rotate:
            # v5 insert speed profile along the (unchanged) 93-unit blade path
            j = i - i_insert
            if j < INSERT_SLOW_STEPS:
                f = INSERT_SLOW_RATIO
            elif j < INSERT_SLOW_STEPS + INSERT_RAMP_STEPS:
                k = j - INSERT_SLOW_STEPS
                f = INSERT_SLOW_RATIO + (1.0 - INSERT_SLOW_RATIO) * (k + 0.5) / INSERT_RAMP_STEPS
            else:
                f = 1.0
            dem.set_sdf_obstacle_vel(tuple(v * f for v in INSERT_VEL))
            if i == i_insert:
                print(
                    f"step {i + 1}: blade starts inserting along the blade direction "
                    f"(v5 wincombo profile: 1/3 speed for the first {INSERT_SLOW_STEPS} steps)",
                    flush=True,
                )
        elif i == i_rotate:
            dem.set_sdf_obstacle_vel(ROTATE_VEL, omega=ROTATE_OMEGA)
            print(
                f"step {i + 1}: blade starts pivoting up about the trailing edge "
                f"(omega_y = {ROTATE_OMEGA[1]:.4f} rad/s)",
                flush=True,
            )
        elif i == i_lift:
            dem.set_sdf_obstacle_vel(LIFT_VEL)
            print(f"step {i + 1}: blade starts lifting vertically at {LIFT_VEL} m/s", flush=True)
        elif i == i_hold:
            dem.set_sdf_obstacle_vel((0.0, 0.0, 0.0))
            print(f"step {i + 1}: blade holds", flush=True)

        # sync the visualization-only kinematic shovel with the solver's obstacle
        blade_pos = to_np(dem.get_sdf_obstacle_pos()[0]).astype(np.float64)
        blade_quat = to_np(dem.get_sdf_obstacle_quat()[0]).astype(np.float64)
        shovel.set_pos(blade_pos)
        shovel.set_quat(blade_quat)
        handle_pos, handle_quat = blade_to_handle(blade_pos, blade_quat)

        # arm follows the handle: continuous IK (same branch, clamped increments), grip hard-set
        target_pos, target_quat = hand_target_from_handle(handle_pos, handle_quat)
        q = solve_ik(franka, hand_link, target_pos, target_quat, q_prev)
        dq_max = 0.0 if q_prev is None else float(np.abs(q - q_prev).max())
        q_prev = q
        q = q.copy()
        q[-2:] = FINGER_GRIP
        franka.set_qpos(q)

        scene.step()

        achieved_pos = to_np(hand_link.get_pos()).astype(np.float64).reshape(-1)[:3]
        achieved_quat = to_np(hand_link.get_quat()).astype(np.float64).reshape(-1)[:4]
        grasp = achieved_pos + quat_to_R(achieved_quat) @ np.array([0.0, 0.0, GRASP_DIST])
        grasp_err = float(np.linalg.norm(grasp - handle_pos))
        ik_log.write(f"{i},{grasp_err:.6f},{dq_max:.6f}\n")

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
                os.path.join(FRAMES_DIR, f"phase9_franka_shovel_sdf{TAG}_frame_{keyframe_idx:04d}.png"),
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
                f"min_z = {pos[:, 2].min():+.6f}  n_lifted = {(pos[:, 2] > 0.1).sum():5d}  "
                f"n_wet_lifted = {((pos[:, 2] > 0.1) & (ratio > 0.02)).sum():5d}  "
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
