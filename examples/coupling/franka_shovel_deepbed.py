"""Phase 9 deep-bed FULL run: 1.8x poisson-sampled bed, settle-until-stable, droplet, then the
complete v5 wincombo scoop trajectory with the Franka follower (user request 2026-08-28:
"初始采样改到1.8倍 ... 等沙子不再沉降了再滴水 然后完整仿真铲子").

Evolution of experiments/phase9_probe_deepbed.py (2 s probe) into the full demo:

  Phase A  sand bed filled to 1.8x the current height (0.10 m -> 0.18 m) with the poisson-disk
           sampler (via fast_poisson.install() -- the reference pure-python sampler is a
           ~2 h build here). poisson_ratio untouched (library default 0.3). Blade SDF parked high
           (tip z = 0.3025), water deactivated. SETTLE-UNTIL-STABLE criterion (user: "等沙子不再
           沉降了再滴水"): the central surface height S is measured every 10 steps; settling is
           done when 3 consecutive measurements differ by < 0.3 mm (min 150, cap 600 steps;
           every measurement is logged).
  Phase B  droplet (r = 0.02, x = -0.01) teleported to S1 + 0.0225, zeroed, re-activated;
           runs until water_active == 0 for 5 consecutive steps (min 60, cap 240). Surface S2
           re-measured afterwards.
  Phase C  blade descends fast (-0.1 m/s, in air) until the tip is 8 mm above S2. The Franka
           continuous-IK follower starts tracking the handle here (full-restart solve first).
  Phase D  wincombo window opens (restitution 0.5 -> 0.1, Young x 0.25); blade descends at the
           original DESCEND_VEL until the tip is 2.8 cm INTO the bed (the user-approved
           phase9_probe_deepbed.py depth: tip S2 - 0.028).
  Phase E  50-deg insert, the FULL wincombo 163-step profile (90 steps 1/3 speed + 30-step
           linear ramp + 43 steps full speed = the original 93 path-units), same constants as
           phase9_franka_shovel_sdf.py v5.
  Phase F  rotate 140 steps (omega -50 deg / 2.333 s, ROTATE_VEL trailing-edge pivot), then
  Phase G  lift 200 steps at 0.0375 m/s (same lift distance as the main run), then
  Phase H  hold 80 steps. Wincombo params ramp back over 30 steps starting 7 steps after the
           insert ends (mirrors the main run's abs 523 -> 530 -> 560 alignment).

Recording: npz every 5 steps in phase A, every step from phase B on; ik_log.csv from phase C on.
Outputs (do NOT collide with the phase9_franka_shovel_sdf final products):
  experiments/recordings/phase9_deepbed_full{tag}/frame_*.npz + meta.json + ik_log.csv
Render: experiments/phase9_deepbed_full_render.py [tag] -> videos/phase9_deepbed_full{tag}.mp4

Usage: python phase9_deepbed_full.py [mode] [tag]
  mode  "smoke": build + 30 settle steps + one full-restart IK solve, then exit
        "full" (default): the whole A-H sequence
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

import genesis as gs

import fast_poisson

HERE = os.path.dirname(os.path.abspath(__file__))
SDF_NPZ = os.path.join(HERE, "assets", "litter_scoop_sdf_slots_s0.5.npz")

MODE = sys.argv[1] if len(sys.argv) > 1 else "full"
TAG = sys.argv[2] if len(sys.argv) > 2 else ("_smoke" if MODE == "smoke" else "")
OUT_DIR = os.path.join(HERE, "recordings", f"phase9_deepbed_full{TAG}")

DT = 1.0 / 60.0
PARTICLE_RADIUS = 3.125e-3 * 0.5  # 1.5625 mm
BOX_LOWER = (-0.175, -0.15, 0.0)
BOX_UPPER = (0.175, 0.15, 0.40)
MAX_RATIO = 0.3
SURFACE_TENSION = 0.02
VISCOSITY = 0.8  # sand<->water quadratic drag (C++ m_ViscosityCoeff, flip._viscosity_coeff;
                 # ONLY used by _kernel_cal_coupling's cf += dv*|dv|*12*pi*r^2*visc -- not the
                 # water's own viscosity). 0.4 -> 0.8 (user decision 2026-09-01), combined with
                 # the 0.8x droplet radius below. History: 0.01 -> 2.0 (2026-08-29) fixed the
                 # percolation (probes: 0.01 -> p1 depth 0.0826, 0.05 -> 0.0683, 0.25 -> 0.0532,
                 # 2.0 -> 0.0359 PASS; log fit depth = 0.0407 - 0.00914*ln(visc)); 2.0 -> 0.4
                 # with r0.8 droplet measured 0.0424 FAIL, with r0.7 droplet 0.0384 FAIL +
                 # too-small wet clump (n_wet 1621); the r0.8 measured/log-model ratio ~0.865
                 # extrapolates visc = 0.8 + r0.8 to p1 ~0.037 (borderline band, user-aware).
                 # C++ reference default is m_ViscosityCoeff = 1 (flip_solver.py comment),
                 # our original 0.01 was the outlier.
RESTITUTION = 0.5
SAFETY = 0.5
# user request 2026-08-27/28: poisson_ratio stays at the library default 0.3 -- never touched

BED_FILL_Z = 0.18  # 1.8 x the current 0.10 m bed (user request 2026-08-28)
DROPLET_X = -0.01
DROPLET_RADIUS = 0.016  # 0.02 * 0.8 (user decision 2026-09-01: back to r0.8 after the r0.7
                        # probe gave n_wet 1621 = too-small wet clump). Volume ~0.512x the
                        # original, n_water ~19064 -> ~9700 (read dynamically via
                        # water.n_particles)
DROPLET_INIT = (DROPLET_X, 0.0, 0.35)  # parked high; deactivated until phase B
DROPLET_GAP = 0.05  # center height above the measured surface. 0.0225 -> 0.05 (user request
                    # 2026-09-02, "水滴更高"): free fall from ~S1+0.05-0.016 gives impact
                    # ~0.8 m/s (was ~0.5) -- a bit more splash on entry, still mild.
RELEASE_STEP = int(round(4.0 / DT))  # = 240: user request 2026-09-02 ("第 4 秒再落") --
                    # after the settle goes stable, keep idling until t = 4 s so the bed is
                    # fully at rest, THEN release the droplet (no-op if settle ends later).

# --- blade / trajectory (main-script v5 constants, lengths already x0.5) ---
BLADE_ANGLE = math.radians(50.0)
BLADE_QUAT0 = (math.cos(BLADE_ANGLE / 2.0), 0.0, math.sin(BLADE_ANGLE / 2.0), 0.0)
BLADE_PARK_POS = (-0.12, 0.0, 0.36)  # tip parked at z = 0.3025, clear of everything
TIP_OFF = np.array([0.0482, 0.0, -0.0575])  # world offset blade pos -> tip (constant 50-deg quat)
APPROACH_VEL = (0.0, 0.0, -0.1)  # fast air descent to just above the surface
DESCEND_VEL = (0.0, 0.0, -0.03655)
DESCEND_DEPTH = 0.028  # tip target below S2: 2.8 cm into the bed, same as the user-approved
                       # phase9_probe_deepbed.py ("下降的低一点然后再斜着插入")
INSERT_VEL = (0.03615, 0.0, -0.0431)  # 50 deg below horizontal, parallel to the blade face
N_INSERT_STEPS = 163  # wincombo: 90 slow + 30 ramp + 43 full = 93 original path-units
INSERT_SLOW_STEPS = 90
INSERT_SLOW_RATIO = 1.0 / 3.0
INSERT_RAMP_STEPS = 30
ROTATE_VEL = (0.01185, 0.0, 0.0254)  # trailing-edge-pivot emulation (mid-rotation 25 deg)
N_ROTATE_STEPS = 140
ROTATE_OMEGA = (0.0, -math.radians(50.0) / (140.0 / 60.0), 0.0)
LIFT_VEL = (0.0, 0.0, 0.0375)
N_LIFT_STEPS = 200
N_HOLD_STEPS = 80

WINCOMBO_REST = 0.1
WINCOMBO_YOUNG_F = 0.25
WINCOMBO_RAMP = 30  # stiffness ramp-back after the window ends (insert end + 7, as in the main run)

# --- Franka follower (identical to phase9_franka_shovel_sdf.py) ---
FRANKA_BASE_POS = (-0.25, 0.45, 0.0)
FINGER_GRIP = 0.0042  # measured rod diameter ~8.4 mm (phase9_handle_geometry.py, 2026-09-02);
                      # was 0.0075 = 15 mm gap -> fingers visibly missed the 8.4 mm rod
GRASP_DIST = 0.09
DQ_MAX = 0.02
# Grip target on the blue scoop's handle ROD, measured on litter_scoop_aligned.glb with the
# voxelizer's Y-up->z-up transform (phase9_handle_geometry.py, 2026-09-02): rod centerline
# runs from the junction (x ~ -0.132) to the tip (x = -0.2144) at 40.39 deg, diameter ~8.4 mm;
# the grip point is the centerline midpoint. The old Phase-5 parametric constants
# ([-0.2649, 0, 0.09642] x SC = [-0.13245, 0, 0.04821]) sat ~1.7 cm BELOW the rod axis near
# the junction -- the gripper pinched air beside the rod (user report 2026-09-02). Keep
# HANDLE_OFFSET_LOCAL identical in phase9_deepbed_full_render.py.
HANDLE_OFFSET_LOCAL = np.array([-0.17407, 0.0, 0.09749])
HANDLE_ANGLE = math.radians(40.39)

# phase caps / criteria
SETTLE_MIN, SETTLE_MAX = 150, 600
SETTLE_MEASURE_EVERY = 10
SETTLE_STABLE_DS = 3e-4  # 0.3 mm
SETTLE_STABLE_N = 3  # consecutive stable measurements
ABSORB_MIN, ABSORB_MAX, ABSORB_DRY_N = 60, 700, 5
# ABSORB_MAX 240 -> 700 (2026-08-29): with VISCOSITY = 2.0 the visc2.0 probe showed absorption
# is rate-limited to ~23 water particles/step in the tail (decaying to ~14), zero at ~step
# 470-630 after release -- the old 240 cap (set for visc = 0.01, ~60 steps) would have started
# the blade descent with ~5000 water particles still floating on the bed.
APPROACH_MAX = 200
DESCEND_MAX = 120


def gpu_mem_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip().splitlines()[0] + " MB"
    except Exception:
        return "n/a"


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
    """Continuous IK (phase7/phase9): warm start on the same branch, clamped increments."""
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
    t_build = time.time()
    fast_poisson.install()  # vectorized poisson-disk sampler (reference one is ~2 h here)
    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=1, gravity=(0.0, 0.0, -9.8)),
        rigid_options=gs.options.RigidOptions(
            gravity=(0.0, 0.0, 0.0),  # kinematic follower: no rigid-rigid collisions
            enable_collision=False,
            enable_self_collision=False,
        ),
        dem_options=gs.options.DEMOptions(
            particle_size=2.0 * PARTICLE_RADIUS,
            ddt_safety=SAFETY,
            surface_tension_coeff=SURFACE_TENSION,
            restitution=RESTITUTION,
            lower_bound=BOX_LOWER,
            upper_bound=BOX_UPPER,
        ),
        flip_options=gs.options.FLIPOptions(
            grid_res=128,
            viscosity_coeff=VISCOSITY,
            lower_bound=BOX_LOWER,
            upper_bound=BOX_UPPER,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    sand = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.00025 + BED_FILL_Z / 2.0),
            size=(0.34, 0.29, BED_FILL_Z),
        ),
        material=gs.materials.DEM.Sand(sampler="poisson", rho=2.5, max_ratio=MAX_RATIO),
    )
    water = scene.add_entity(
        morph=gs.morphs.Sphere(pos=DROPLET_INIT, radius=DROPLET_RADIUS),
        material=gs.materials.FLIP.Liquid(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=FRANKA_BASE_POS),
    )
    scene.build()
    dem = scene.sim.dem_solver
    flip = scene.sim.flip_solver
    hand_link = franka.get_link("hand")
    print(
        f"build done in {time.time() - t_build:.1f}s: n_sand = {sand.n_particles}, "
        f"n_water = {water.n_particles}, poisson_ratio = {dem._entities[0].material.poisson_ratio}, "
        f"gpu mem = {gpu_mem_mb()}",
        flush=True,
    )

    sdf = np.load(SDF_NPZ)
    dem.set_sdf_obstacle(sdf["sdf_val"], sdf["dims"], sdf["origin"], sdf["cell"],
                         BLADE_PARK_POS, quat=BLADE_QUAT0)
    print(f"sdf obstacle parked at {BLADE_PARK_POS} (tip z = {BLADE_PARK_POS[2] + TIP_OFF[2]:.4f})",
          flush=True)

    water_pos0 = water.get_particles_pos().cpu().numpy()[0].astype(np.float64)
    water_offsets = water_pos0 - np.array(DROPLET_INIT)
    n_water = water.n_particles
    flip.particles.active.from_numpy(np.zeros((n_water, 1), dtype=np.bool_))
    print(f"water deactivated for the settle ({n_water} particles)", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    if MODE != "smoke":
        # wipe stale frame_*.npz from earlier runs BEFORE any record() call (2026-09-02):
        # the visc0.8+r0.8 run (1225 steps) left 398 old visc2.0 frames (frame_1225..1622)
        # mixed into this directory, which would corrupt the render. OUT_DIR is the single
        # recordings-path constant (see line ~57); smoke mode uses a separate _smoke dir.
        stale = [f for f in os.listdir(OUT_DIR) if f.startswith("frame_") and f.endswith(".npz")]
        for f in stale:
            os.remove(os.path.join(OUT_DIR, f))
        if stale:
            print(f"wiped {len(stale)} stale frame(s) from {OUT_DIR}", flush=True)
    ik_log = open(os.path.join(OUT_DIR, "ik_log.csv"), "w")
    ik_log.write("step,grasp_err,dq_max\n")

    def record(i):
        sand_pos = sand.get_particles_pos().cpu().numpy()[0].astype(np.float32)
        active = flip.particles.active.to_numpy()[:, 0].astype(bool)
        water_pos = water.get_particles_pos().cpu().numpy()[0][active].astype(np.float32)
        np.savez(
            os.path.join(OUT_DIR, f"frame_{i:04d}.npz"),
            sand_pos=sand_pos,
            sand_ratio=dem.particles.ratio.to_numpy()[:, 0].astype(np.float32),
            water_pos=water_pos,
            blade_pos=np.asarray(dem.get_sdf_obstacle_pos()[0], dtype=np.float32),
            blade_quat=np.asarray(dem.get_sdf_obstacle_quat()[0], dtype=np.float32),
        )
        return sand_pos

    def sand_vmax():
        vel = sand.get_particles_vel().cpu().numpy()[0]
        return float(np.linalg.norm(vel, axis=1).max())

    def surface_z():
        pos = sand.get_particles_pos().cpu().numpy()[0]
        central = pos[(np.abs(pos[:, 0]) < 0.10) & (np.abs(pos[:, 1]) < 0.08)]
        return float(np.percentile(central[:, 2], 99))

    def blade_pose():
        pos = to_np(dem.get_sdf_obstacle_pos()[0]).astype(np.float64)
        quat = to_np(dem.get_sdf_obstacle_quat()[0]).astype(np.float64)
        return pos, quat

    def tip_z():
        return float(blade_pose()[0][2] + TIP_OFF[2])

    q_prev = None

    def follow_handle(i):
        """Drive the arm to the current handle pose; returns (grasp_err, dq_max)."""
        nonlocal q_prev
        bpos, bquat = blade_pose()
        handle_pos, handle_quat = blade_to_handle(bpos, bquat)
        target_pos, target_quat = hand_target_from_handle(handle_pos, handle_quat)
        q = solve_ik(franka, hand_link, target_pos, target_quat, q_prev)
        dq_max = 0.0 if q_prev is None else float(np.abs(q - q_prev).max())
        q_prev = q
        q = q.copy()
        q[-2:] = FINGER_GRIP
        franka.set_qpos(q)
        achieved_pos = to_np(hand_link.get_pos()).astype(np.float64).reshape(-1)[:3]
        achieved_quat = to_np(hand_link.get_quat()).astype(np.float64).reshape(-1)[:4]
        grasp = achieved_pos + quat_to_R(achieved_quat) @ np.array([0.0, 0.0, GRASP_DIST])
        grasp_err = float(np.linalg.norm(grasp - handle_pos))
        ik_log.write(f"{i},{grasp_err:.6f},{dq_max:.6f}\n")
        return grasp_err, dq_max

    t0 = time.time()
    meta = {
        "dt": DT, "particle_radius": PARTICLE_RADIUS, "max_ratio": MAX_RATIO,
        "n_sand": int(sand.n_particles), "n_water": int(n_water),
        "sampler": "poisson", "poisson_ratio": 0.3, "bed_fill_z": BED_FILL_Z,
        "droplet_x": DROPLET_X,
        "trajectory": "full v5 wincombo anchored to measured surface S2: approach -0.1 m/s to "
        "S2+8mm, descend to S2-2.8cm, insert 163 steps (90@1/3 + 30 ramp + 43 full), rotate 140, "
        "lift 200, hold 80; wincombo window restitution 0.1 + young x0.25 from descend start "
        "until insert end + 7, 30-step stiffness ramp-back",
    }

    # ---------------- phase A: settle until the surface is stable ----------------
    s_hist = []
    i = -1
    for i in range(SETTLE_MAX):
        scene.step()
        if i % 5 == 0:
            record(i)
        if i % SETTLE_MEASURE_EVERY == 0:
            s = surface_z()
            s_hist.append(s)
            vm = sand_vmax()
            ds = abs(s_hist[-1] - s_hist[-2]) if len(s_hist) > 1 else float("nan")
            print(f"A settle step {i:4d}: S = {s:.5f}  dS = {ds:.5f}  vmax = {vm:7.3f}  "
                  f"nan = {np.isnan(sand.get_particles_pos().cpu().numpy()[0]).any()}  "
                  f"elapsed = {time.time() - t0:.0f}s", flush=True)
            stable = (
                len(s_hist) > SETTLE_STABLE_N
                and all(
                    abs(s_hist[k + 1] - s_hist[k]) < SETTLE_STABLE_DS
                    for k in range(len(s_hist) - SETTLE_STABLE_N - 1, len(s_hist) - 1)
                )
            )
            if MODE != "smoke" and i + 1 >= SETTLE_MIN and stable:
                break
        if MODE == "smoke" and i >= 29:
            break
    settle_end = i
    s1 = surface_z()
    print(f"phase A done at step {settle_end}: settled surface z = {s1:.4f} "
          f"(fill top was {0.00025 + BED_FILL_Z:.4f})", flush=True)
    if MODE == "smoke":
        ge, dq = follow_handle(0)
        print(f"SMOKE done: n_sand = {sand.n_particles}, n_water = {n_water}, "
              f"first-IK grasp_err = {ge:.2e} (dq_max {dq:.3f}), gpu mem = {gpu_mem_mb()}",
              flush=True)
        ik_log.close()
        return
    meta["settle_end"] = int(settle_end)
    meta["surface_after_settle"] = s1

    # ---------------- phase B: droplet ----------------
    # user request 2026-09-02: idle until t = 4 s (RELEASE_STEP) before releasing, so the
    # bed is fully at rest; no-op when the settle itself ended at/after RELEASE_STEP.
    release_step = max(settle_end + 1, RELEASE_STEP)
    for i in range(settle_end + 1, release_step):
        scene.step()
        if i % 5 == 0:
            record(i)
    if release_step > settle_end + 1:
        print(f"phase A2: idled from step {settle_end + 1} to {release_step} "
              f"(t = {release_step * DT:.2f} s) before the droplet release", flush=True)
    meta["release_step"] = int(release_step)
    droplet_center = np.array([DROPLET_X, 0.0, s1 + DROPLET_GAP])
    water.set_particles_pos((water_offsets + droplet_center).astype(np.float32)[None])
    water.set_particles_vel(np.zeros((1, n_water, 3), dtype=np.float32))
    flip.particles.active.from_numpy(np.ones((n_water, 1), dtype=np.bool_))
    print(f"phase B: droplet released at {droplet_center} (gap {DROPLET_GAP} above z = {s1:.4f})",
          flush=True)
    dry = 0
    for j in range(ABSORB_MAX):
        i = release_step + j
        scene.step()
        record(i)
        n_active = int(flip.particles.active.to_numpy()[:, 0].sum())
        dry = dry + 1 if n_active == 0 else 0
        if j % 10 == 0 or dry == ABSORB_DRY_N:
            print(f"B absorb step {i:4d}: water_active = {n_active}/{n_water}  "
                  f"vmax = {sand_vmax():7.3f}  elapsed = {time.time() - t0:.0f}s", flush=True)
        if j + 1 >= ABSORB_MIN and dry >= ABSORB_DRY_N:
            break
    absorb_end = i
    s2 = surface_z()
    print(f"phase B done at step {absorb_end}: surface after absorption z = {s2:.4f}", flush=True)
    meta["absorb_end"] = int(absorb_end)
    meta["surface_after_absorb"] = s2

    # ---------------- phase C: fast approach (+ IK follower starts) ----------------
    dem.set_sdf_obstacle_vel(APPROACH_VEL)
    target = s2 + 0.008
    for j in range(APPROACH_MAX):
        i = absorb_end + 1 + j
        ge, _ = follow_handle(i)
        scene.step()
        record(i)
        if j % 20 == 0:
            print(f"C approach step {i:4d}: tip z = {tip_z():.4f}  grasp_err = {ge:.2e}", flush=True)
        if tip_z() <= target:
            break
    approach_end = i
    dem.set_sdf_obstacle_vel((0.0, 0.0, 0.0))
    print(f"phase C done at step {approach_end}: tip z = {tip_z():.4f} (target {target:.4f})",
          flush=True)
    meta["approach_end"] = int(approach_end)

    # ---------------- phase D: descend to 1 cm into the bed ----------------
    dem._restitution = WINCOMBO_REST
    young0 = float(dem._entities[0].material.young_modulus)
    dem._entities[0].material.young_modulus = young0 * WINCOMBO_YOUNG_F
    print(f"phase D: wincombo window opens (restitution {WINCOMBO_REST}, young x{WINCOMBO_YOUNG_F})",
          flush=True)
    dem.set_sdf_obstacle_vel(DESCEND_VEL)
    target = s2 - DESCEND_DEPTH
    for j in range(DESCEND_MAX):
        i = approach_end + 1 + j
        follow_handle(i)
        scene.step()
        record(i)
        if j % 10 == 0 or tip_z() <= target:
            print(f"D descend step {i:4d}: tip z = {tip_z():.4f}  vmax = {sand_vmax():7.3f}  "
                  f"nan = {np.isnan(sand.get_particles_pos().cpu().numpy()[0]).any()}", flush=True)
        if tip_z() <= target:
            break
    descend_end = i
    print(f"phase D done at step {descend_end}: tip z = {tip_z():.4f} "
          f"(target {target:.4f} = S2 {s2:.4f} - {DESCEND_DEPTH})", flush=True)
    meta["descend_end"] = int(descend_end)

    # ---------------- phase E: insert, full wincombo 163-step profile ----------------
    wincombo_end = descend_end + N_INSERT_STEPS + 7  # params restore starts here (main: abs 530)
    for j in range(N_INSERT_STEPS):
        i = descend_end + 1 + j
        if j < INSERT_SLOW_STEPS:
            f = INSERT_SLOW_RATIO
        elif j < INSERT_SLOW_STEPS + INSERT_RAMP_STEPS:
            k = j - INSERT_SLOW_STEPS
            f = INSERT_SLOW_RATIO + (1.0 - INSERT_SLOW_RATIO) * (k + 0.5) / INSERT_RAMP_STEPS
        else:
            f = 1.0
        dem.set_sdf_obstacle_vel(tuple(v * f for v in INSERT_VEL))
        if j == 0:
            print(f"phase E: insert starts at step {i} (wincombo 163-step profile)", flush=True)
        follow_handle(i)
        scene.step()
        pos = record(i)
        if j % 20 == 0 or j == N_INSERT_STEPS - 1:
            print(f"E insert step {i:4d} ({j + 1}/{N_INSERT_STEPS}, f={f:.3f}): tip z = {tip_z():.4f}  "
                  f"vmax = {sand_vmax():7.3f}  nan = {np.isnan(pos).any()}  "
                  f"elapsed = {time.time() - t0:.0f}s", flush=True)
    insert_end = i
    meta["insert_end"] = int(insert_end)

    # ---------------- phases F/G/H: rotate, lift, hold (wincombo ramp-back inside) ----
    for phase, phase_key, n_steps, vel, omega in [
        ("F rotate", "rotate_end", N_ROTATE_STEPS, ROTATE_VEL, ROTATE_OMEGA),
        ("G lift", "lift_end", N_LIFT_STEPS, LIFT_VEL, None),
        ("H hold", "hold_end", N_HOLD_STEPS, (0.0, 0.0, 0.0), None),
    ]:
        dem.set_sdf_obstacle_vel(vel, omega=omega) if omega is not None else dem.set_sdf_obstacle_vel(vel)
        print(f"{phase}: starts at step {i + 1}", flush=True)
        for j in range(n_steps):
            i += 1
            # wincombo parameter restore: restitution back to 0.5 at insert_end + 7,
            # Young ramped back linearly over WINCOMBO_RAMP steps (mirrors the main run)
            if i == wincombo_end:
                dem._restitution = RESTITUTION
                print(f"wincombo restitution restored to {RESTITUTION} at step {i}", flush=True)
            if i >= wincombo_end:
                g = min(1.0, (i - wincombo_end + 1) / WINCOMBO_RAMP)
                dem._entities[0].material.young_modulus = young0 * (
                    WINCOMBO_YOUNG_F + (1.0 - WINCOMBO_YOUNG_F) * g
                )
            follow_handle(i)
            scene.step()
            record(i)
            if j % 20 == 0 or j == n_steps - 1:
                bpos, bquat = blade_pose()
                theta = math.degrees(2.0 * math.atan2(bquat[2], bquat[0]))
                print(f"{phase} step {i:4d} ({j + 1}/{n_steps}): blade z = {bpos[2]:.4f}  "
                      f"theta = {theta:+6.2f}  vmax = {sand_vmax():7.3f}  "
                      f"elapsed = {time.time() - t0:.0f}s", flush=True)
        meta[phase_key] = int(i)

    meta["final_step"] = int(i)
    meta["final_tip_z"] = tip_z()
    ik_log.close()
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"deep-bed full run done: {i + 1} steps in {time.time() - t0:.0f}s, "
          f"frames in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
