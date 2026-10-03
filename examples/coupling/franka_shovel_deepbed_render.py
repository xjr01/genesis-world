"""Renderer for the phase9_deepbed_full run (1.8x poisson bed, droplet, full v5 trajectory +
Franka follower).

Wetness-bucket DEM rendering + water replay as in phase9_probe_deepbed_render.py, plus the
Franka re-driven every frame with the same continuous-IK follower as the run (from the recorded
blade pose, warm start + joint increment clamp) as in phase9_render_wet_native.py. Camera is the
fixed close-up framing validated on phase9_probe_deepbed. Keyframes at the phase boundaries
from meta.json: start / settle end / absorb end / insert start / insert end / rotate end /
lift end / hold end.

Usage: python phase9_deepbed_full_render.py [tag] [max_frames]
Output: experiments/videos/phase9_deepbed_full{tag}.mp4
        experiments/frames/phase9_deepbed_full{tag}_frame_*.png (keyframes)
"""

import json
import math
import os
import shutil
import subprocess
import sys

import imageio
import numpy as np

import genesis as gs

import fast_poisson

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
MAX_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else None

# the 4 wetness-bucket DEM entities share the SAME morph/material/seed: sample once (with the
# vectorized sampler) and reuse for the ghosts
fast_poisson.install()
from genesis.engine.entities.dem_entity import DEMEntity

_orig_sample_poisson = DEMEntity._sample_poisson
_poisson_cache = {}


def _cached_sample_poisson(trimesh, min_dist, rng, is_inside):
    key = (round(float(min_dist), 9), tuple(np.round(trimesh.bounds.ravel(), 6)))
    if key not in _poisson_cache:
        _poisson_cache[key] = _orig_sample_poisson(trimesh, min_dist, rng, is_inside)
    return _poisson_cache[key]


DEMEntity._sample_poisson = staticmethod(_cached_sample_poisson)

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(EXPERIMENTS_DIR, "frames")
REC_DIR = os.path.join(EXPERIMENTS_DIR, "recordings", f"phase9_deepbed_full{TAG}")
VIDEO_PATH = os.path.join(EXPERIMENTS_DIR, "videos", f"phase9_deepbed_full{TAG}.mp4")

DT = 1.0 / 60.0
SC = 0.5
PARTICLE_RADIUS = 3.125e-3 * SC
BOX_LOWER = (-0.35 * SC, -0.30 * SC, 0.0)
BOX_UPPER = (0.35 * SC, 0.30 * SC, 0.80 * SC)
WALL_THICK = 0.01 * SC
WALL_HEIGHT = 0.24 * SC  # just above the settled surface; taller walls blocked the camera
MAX_RATIO = 0.3

N_BUCKETS = 4
DRY_COLOR = np.array([0.87, 0.72, 0.53])
WET_COLOR = np.array([0.25, 0.13, 0.06])
PARK_POS = (0.0, 0.0, -10.0)  # absorbed water particles are parked below the floor

# Franka follower constants (identical to the run / phase9_franka_shovel_sdf.py)
FRANKA_BASE_POS = (-0.25, 0.45, 0.0)
FINGER_GRIP = 0.0042  # measured rod diameter ~8.4 mm (phase9_handle_geometry.py, 2026-09-02);
                      # was 0.0075 = 15 mm gap -> fingers visibly missed the 8.4 mm rod
GRASP_DIST = 0.09
DQ_MAX = 0.02
# Grip target on the blue scoop's handle ROD (measured via phase9_handle_geometry.py,
# 2026-09-02; MUST stay identical to phase9_deepbed_full.py). The old Phase-5 parametric
# value [-0.13245, 0, 0.04821] sat ~1.7 cm below the rod axis -- gripper pinched air.
HANDLE_OFFSET_LOCAL = np.array([-0.17407, 0.0, 0.09749])
HANDLE_ANGLE = math.radians(40.39)


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
    """World pose of the handle-rod grip point (measured on the blue scoop mesh, 2026-09-02)."""
    handle_quat_rel = np.array(
        [math.cos((math.pi + HANDLE_ANGLE) / 2.0), 0.0, math.sin((math.pi + HANDLE_ANGLE) / 2.0), 0.0]
    )
    handle_pos = blade_pos + quat_to_R(blade_quat) @ HANDLE_OFFSET_LOCAL
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
    """Same continuous-IK follower as the run (warm start, clamped increments)."""
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


# --- render-side IK recovery (2026-08-28): the run's continuous follower drifted to 15 mm
# grasp_err during the fast -0.1 m/s approach; the offline re-drive must not reproduce it.
# Strategy (per review): measure the GEOMETRIC grasp err after applying the clamped q; if it
# exceeds ERR_MAX, retry up to 3 times with a relaxed increment clamp (DQ_MAX x 5, more solver
# iters) from the SAME warm start; only if still over the limit do a full-restart IK
# (phase9_ik_probe.py style: multi-restart, worst 9e-5 on the trajectory key poses) and adopt
# the new branch unclamped. Full restarts should fire only on a handful of approach frames.
ERR_MAX = 2e-3
RELAX_F = 5.0
N_RELAX = 3


def full_restart_ik(franka, hand_link, pos, quat):
    q = franka.inverse_kinematics(
        hand_link, pos=pos, quat=quat,
        max_samples=100, max_solver_iters=200, damping=0.005, pos_tol=1e-5, rot_tol=1e-4,
    )
    return to_np(q).astype(np.float64).reshape(-1)


def grasp_err_of(franka, hand_link, handle_pos):
    achieved_pos = to_np(hand_link.get_pos()).astype(np.float64).reshape(-1)[:3]
    achieved_quat = to_np(hand_link.get_quat()).astype(np.float64).reshape(-1)[:4]
    grasp = achieved_pos + quat_to_R(achieved_quat) @ np.array([0.0, 0.0, GRASP_DIST])
    return float(np.linalg.norm(grasp - handle_pos))


def drive_arm(franka, hand_link, blade_pos, blade_quat, q_prev):
    """Drive the arm to the recorded handle pose with drift recovery.
    Returns (q_applied, grasp_err, n_relaxed_used, restarted)."""
    handle_pos, handle_quat = blade_to_handle(blade_pos, blade_quat)
    target_pos, target_quat = hand_target_from_handle(handle_pos, handle_quat)

    def apply(q):
        qq = q.copy()
        qq[-2:] = FINGER_GRIP
        franka.set_qpos(qq)
        return grasp_err_of(franka, hand_link, handle_pos)

    if q_prev is None:
        q = full_restart_ik(franka, hand_link, target_pos, target_quat)
        return q, apply(q), 0, True

    q = solve_ik(franka, hand_link, target_pos, target_quat, q_prev)
    err = apply(q)
    n_relaxed = 0
    while err > ERR_MAX and n_relaxed < N_RELAX:
        # retry from the same warm start with a relaxed increment clamp and more iterations
        n_relaxed += 1
        q_try, ierr = franka.inverse_kinematics(
            hand_link, pos=target_pos, quat=target_quat, init_qpos=q_prev, return_error=True,
            max_samples=1, max_solver_iters=300, damping=0.005, pos_tol=1e-5, rot_tol=1e-4,
        )
        q_try = to_np(q_try).astype(np.float64).reshape(-1)
        q_try = q_prev + np.clip(q_try - q_prev, -DQ_MAX * RELAX_F, DQ_MAX * RELAX_F)
        err_try = apply(q_try)
        if err_try < err:
            q, err = q_try, err_try
    restarted = False
    if err > ERR_MAX:
        q = full_restart_ik(franka, hand_link, target_pos, target_quat)
        err = apply(q)
        restarted = True
    return q, err, n_relaxed, restarted


def main():
    with open(os.path.join(REC_DIR, "meta.json")) as f:
        meta = json.load(f)
    n_sand = meta["n_sand"]
    n_water = meta["n_water"]
    frame_files = sorted(f for f in os.listdir(REC_DIR) if f.startswith("frame_"))
    # drop stale frames left by earlier runs (the sim does not wipe REC_DIR): keep only
    # frames up to this run's final_step. Without this, 262 old-run frames (frame_1225..)
    # would render after the new run's end -- and their water_pos (old droplet, up to
    # 19064 particles) would overflow the n_water-sized replay buffer.
    final_step = meta.get("final_step")
    if final_step is not None:
        frame_files = [f for f in frame_files if int(f[6:-4]) <= final_step]
    if MAX_FRAMES is not None:
        frame_files = frame_files[:MAX_FRAMES]
    n_frames = len(frame_files)
    # keyframes: start / settle end / absorb end / insert start / insert end / rotate / lift / hold
    boundary_steps = [0]
    for key in ("settle_end", "absorb_end", "descend_end", "insert_end",
                "rotate_end", "lift_end", "hold_end"):
        if key in meta:
            s = meta[key]
            boundary_steps.append(s + 1 if key == "descend_end" else s)  # insert starts after descend
    rec_steps = np.array([int(f[6:10]) for f in frame_files])
    keyframe_idx_set = set()
    for s in boundary_steps:
        keyframe_idx_set.add(int(np.argmin(np.abs(rec_steps - s))))
    print(f"replaying {n_frames} frames from {REC_DIR}, keyframes at steps {boundary_steps}",
          flush=True)

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

    # same fill box as the run so the entity particle count matches the recording
    sand_morph = gs.morphs.Box(
        pos=(0.0, 0.0, 0.00025 + meta["bed_fill_z"] / 2.0),
        size=(0.34, 0.29, meta["bed_fill_z"]),
    )
    sand = scene.add_entity(
        morph=sand_morph,
        material=gs.materials.DEM.Sand(sampler="poisson", rho=2.5, max_ratio=MAX_RATIO),
        surface=gs.surfaces.Default(color=tuple(DRY_COLOR)),
    )
    dem_ents = [sand]
    for k in range(1, N_BUCKETS):
        t = k / (N_BUCKETS - 1)
        color = DRY_COLOR * (1.0 - t) + WET_COLOR * t
        ghost = scene.add_entity(
            morph=sand_morph,
            material=gs.materials.DEM.Sand(sampler="poisson", rho=2.5, max_ratio=MAX_RATIO),
            surface=gs.surfaces.Default(color=tuple(color)),
        )
        dem_ents.append(ghost)

    water = scene.add_entity(
        # droplet radius mirrors phase9_deepbed_full.py DROPLET_RADIUS (0.016 = 0.02 * 0.8,
        # user decision 2026-09-01); meta.json has no droplet_radius field, so keep this in
        # sync by hand -- the assert below catches a mismatch against meta["n_water"]
        morph=gs.morphs.Sphere(pos=(meta.get("droplet_x", -0.01), 0.0, 0.35), radius=0.016),
        material=gs.materials.FLIP.Liquid(),
        surface=gs.surfaces.Default(color=(0.2, 0.5, 0.9), opacity=0.85),
    )
    cam = scene.add_camera(
        res=(1280, 720),
        pos=(0.42, -0.48, 0.44),  # z +0.06 vs validated probe framing: keep lifted shovel (z~0.25) in frame
        lookat=(-0.06, 0.0, 0.12),
        fov=40,
    )

    scene.build()
    dem = scene.sim.dem_solver
    flip = scene.sim.flip_solver
    assert sand.n_particles == n_sand, f"sand {sand.n_particles} vs {n_sand}"
    assert water.n_particles == n_water, f"water {water.n_particles} vs {n_water}"
    n_total = dem.n_particles
    slices = []
    for ent in dem_ents:
        assert ent.n_particles == n_sand, f"ghost {ent.n_particles} vs {n_sand}"
        slices.append((ent.particle_start, ent.particle_end))
    hand_link = franka.get_link("hand")

    raw_dir = os.path.join(FRAMES_DIR, f"phase9_deepbed_full{TAG}_raw")
    # wipe stale frames: ffmpeg globs frame_*.png, leftovers would corrupt the video
    if os.path.isdir(raw_dir):
        for stale in os.listdir(raw_dir):
            os.remove(os.path.join(raw_dir, stale))
    os.makedirs(raw_dir, exist_ok=True)

    keyframe_idx = 0
    q_prev = None
    err_hist = []
    ik_log = open(os.path.join(REC_DIR, f"ik_render_log{TAG}.csv"), "w")
    ik_log.write("step,grasp_err,n_relaxed,full_restart\n")
    park_block = np.tile(np.array(PARK_POS, dtype=np.float64), (n_water, 1))
    pos_all = np.zeros((n_total, 1, 3), dtype=np.float32)
    act_all = np.zeros((n_total, 1), dtype=np.bool_)
    for i, fname in enumerate(frame_files):
        frame = np.load(os.path.join(REC_DIR, fname))

        water_pos = frame["water_pos"].astype(np.float64)
        buf = park_block.copy()
        buf[: len(water_pos)] = water_pos
        water.set_particles_pos(buf[None])

        blade_pos = frame["blade_pos"].astype(np.float64)
        blade_quat = frame["blade_quat"].astype(np.float64)
        shovel.set_pos(blade_pos)
        shovel.set_quat(blade_quat)

        # re-drive the arm: continuous IK follower + drift recovery (see drive_arm above)
        q, grasp_err, n_relaxed, restarted = drive_arm(franka, hand_link, blade_pos, blade_quat, q_prev)
        q_prev = q
        ik_log.write(f"{rec_steps[i]},{grasp_err:.6f},{n_relaxed},{int(restarted)}\n")
        err_hist.append(grasp_err)
        if restarted:
            print(f"frame {i} (step {rec_steps[i]}): full-restart IK, grasp_err = {grasp_err:.2e}",
                  flush=True)

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
        # name raw frames by SIM STEP (not replay index) so the video timeline can be
        # aligned 1:1 with sim time below
        imageio.imwrite(os.path.join(raw_dir, f"frame_{rec_steps[i]:04d}.png"), rgb)

        if i in keyframe_idx_set:
            imageio.imwrite(
                os.path.join(FRAMES_DIR, f"phase9_deepbed_full{TAG}_frame_{keyframe_idx:04d}.png"),
                rgb,
            )
            keyframe_idx += 1

        if (i + 1) % 30 == 0:
            print(f"frame {i + 1}/{n_frames}", flush=True)

    ik_log.close()
    err_hist = np.asarray(err_hist)
    print(
        f"render IK grasp_err: max = {err_hist.max():.2e}  mean = {err_hist.mean():.2e}  "
        f"frames>2mm = {(err_hist > ERR_MAX).sum()}/{len(err_hist)}",
        flush=True,
    )
    # timeline alignment (2026-09-03, user: "水滴应该第四秒才落下，在这之前沙子不应该是湿的";
    # main-agent acceptance oversight): the settle/wait phase is recorded every 5 steps, so
    # replaying the recordings in order at 60 fps compressed the first 4 s of sim time into
    # ~0.8 s of video. Hold each sparsely-recorded frame for its gap (image reuse, NO
    # re-render) so 1 video frame = 1 sim step: the droplet then starts falling at exactly
    # video t = 4.0 s (frame 240) and the sand stays dry before that.
    steps_present = sorted(
        int(f[6:-4]) for f in os.listdir(raw_dir) if f.startswith("frame_")
    )
    filled = 0
    prev = None
    for s in range(steps_present[0], steps_present[-1] + 1):
        p = os.path.join(raw_dir, f"frame_{s:04d}.png")
        if os.path.exists(p):
            prev = p
        elif prev is not None:
            shutil.copyfile(prev, p)
            filled += 1
    print(
        f"timeline alignment: duplicated {filled} held frame(s) over sparse recordings; "
        f"video length = {steps_present[-1] - steps_present[0] + 1} frames "
        f"(= {(steps_present[-1] - steps_present[0] + 1) / 60.0:.2f} s @ 60 fps)",
        flush=True,
    )
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
