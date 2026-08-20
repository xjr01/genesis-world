"""
Wet-sand scoop demo, MESH-SDF obstacle variant: identical scene/motion as phase5_shovel_wet.py,
but the physical boundary is the voxelized SDF of the real litter-scoop mesh
(assets/litter_scoop_sdf.npz, built by phase8c_voxelize_scoop.py from the SAME aligned glb that
is rendered), driven through dem.set_sdf_obstacle / set_sdf_obstacle_vel instead of the
analytic tilt box. The scoop's raised walls and curved pan now participate in the physics.

Usage: python phase5_shovel_wet_sdf.py [max_steps]
Outputs (separate from the tilt-box run):
- experiments/videos/phase5_shovel_wet_sdf.mp4           (sand + water + scoop, native rendering)
- experiments/videos/phase5_shovel_wet_sdf_sandwet.mp4   (sand only, per-grain color by water ratio)
- experiments/recordings/phase5_shovel_wet_sdf/frame_*.npz

Wet-sand scoop demo: drip water onto sand, then descend, insert, pivot up, and lift the wet clump.

Scene: a walled box (interior = DEM domain bounds, x in [-0.35, 0.35], y in [-0.30, 0.30]) with a
sand layer (0.68 x 0.58 x 0.20 m, fcc) covering its floor; the four walls are kinematic
visualization-only entities. At t = 0 a water droplet (radius 0.04 m) is released at (0.02, 0)
right at the sand surface (z = 0.205, contact release) -- centered over the horizontal blade's
span after the pivot, clear of the vertical descent path -- and is absorbed during the 5 s settle
(max_ratio = 0.3, grid dx ~ 6.25 mm keeps absorption active), forming a cohesive wet clump via
the capillary (liquid-bridge) forces.

The shovel (0.3 x 0.24 x 0.02 m blade + 0.3 m handle, one unioned SDF obstacle, blade at 40 deg)
then:
  1. descends vertically (vel = (0, 0, -0.12) m/s) for 0.92 s until the leading tip touches the
     sand surface at (-0.085, 0.201),
  2. inserts diagonally along the blade direction (vel = (0.115, 0, -0.0964) m/s, i.e. 40 deg
     down) for 1.6 s, ending with the tip at (0.099, 0.008), nearly grazing the floor,
     underneath the wet clump,
  3. pivots up about the blade's TRAILING edge (handle side): a rotation from 40 deg to flat
     (omega_y = -0.399 rad/s about the blade center) with a compensating translation
     (vel = (0.0205, 0, 0.0563) m/s, mid-rotation value of v_c = -omega x r_pivot) that keeps the
     trailing edge near (-0.131, 0.201) fixed -- the tip sweeps up along a circular arc through
     the sand, scooping the wet clump onto the blade for 1.75 s, ending horizontal at z ~ 0.20,
  4. lifts vertically (vel = (0, 0, 0.10) m/s) for 2.5 s up to z ~ 0.46, carrying the wet clump,
  5. holds still for 1 s.

Note: the FLIP water does not see the shovel obstacle (DEM-only); the script relies on the droplet
being fully absorbed during the settle phase (monitored via water_active).

Per-frame recording for offline re-rendering: every step saves sand particle positions + water
ratios, active water particle positions, and the shovel (blade) position + quaternion to
experiments/recordings/phase5_shovel_wet/frame_%04d.npz; static scene parameters (domain bounds,
walls, blade/handle geometry, droplet, grid/particle sizes) go to meta.json in the same directory.

Videos:
- experiments/videos/phase5_shovel_wet.mp4           (sand + water + shovel, native rendering)
- experiments/videos/phase5_shovel_wet_sandwet.mp4   (sand only, per-grain color by water ratio)
Keyframes: experiments/frames/phase5_shovel_wet_frame_*.png.
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
VIDEO_PATH = os.path.join(EXPERIMENTS_DIR, "videos", "phase5_shovel_wet_sdf.mp4")
VIDEO_WET_PATH = os.path.join(EXPERIMENTS_DIR, "videos", "phase5_shovel_wet_sdf_sandwet.mp4")
WET_RAW_DIR = os.path.join(FRAMES_DIR, "phase5_shovel_wet_sdf_sandwet_raw")
REC_DIR = os.path.join(EXPERIMENTS_DIR, "recordings", "phase5_shovel_wet_sdf")
SDF_NPZ = os.path.join(
    EXPERIMENTS_DIR, "assets", sys.argv[2] if len(sys.argv) > 2 else "litter_scoop_sdf_slots.npz"
)  # default: true slotted pan floor (exact mesh geometry); argv[2] litter_scoop_sdf.npz = plate-filled variant

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else None  # debug: cap the total steps

DT = 1.0 / 60.0
N_SETTLE_STEPS = 300  # 5 s: droplet falls and is absorbed
# user request (2026-08-20): same geometric path, but 0.75x speed (steps x4/3) and shifted 0.04 m
# toward -x ("scoop further back"); the phase displacements are unchanged from the validated run
N_DESCEND_STEPS = 48  # 0.80 s: tip descends 0.072 m until it touches the deeper bed's surface (0.201)
N_INSERT_STEPS = 128  # 2.13 s
N_ROTATE_STEPS = 140  # 2.33 s
N_LIFT_STEPS = 200  # 3.33 s
N_HOLD_STEPS = 80  # 1.33 s
DESCEND_VEL = (0.0, 0.0, -0.09)  # straight down until the leading tip touches the sand surface
# (tip (-0.125, 0.273) -> (-0.125, 0.201), surface of the 0.20 m bed)
INSERT_VEL = (0.08625, 0.0, -0.0723)  # 0.1125 m/s along the 40 deg blade direction: the tip slides
# into the sand to (0.059, 0.008), nearly grazing the floor, underneath the wet clump
ROTATE_VEL = (0.015375, 0.0, 0.042225)  # trailing-edge-pivot emulation: v_center = -omega x r_pivot
# (mid-rotation value), keeps the trailing (handle-side) edge near (-0.171, 0.201) fixed so the tip
# sweeps up along a circular arc through the sand, scooping the wet clump onto the blade
ROTATE_OMEGA = (0.0, -math.radians(40.0) / (140.0 / 60.0), 0.0)  # 40 deg -> flat about the blade center
LIFT_VEL = (0.0, 0.0, 0.075)
KEYFRAME_STEPS = [0, 299, 390, 546, 616, 760, 895]
WET_EVERY = 3

PARTICLE_RADIUS = 3.125e-3  # sand particle diameter 6.25 mm = 0.8 / 128, matching the FLIP grid dx
BOX_LOWER = (-0.35, -0.30, 0.0)
BOX_UPPER = (0.35, 0.30, 0.80)
WALL_THICK = 0.01
WALL_HEIGHT = 0.24  # taller than the 0.20 m sand bed

# blade box matched to the litter-scoop pan (objaverse 'Litter scoop/貓砂鏟'):
# pan depth 0.30 m along x, pan width 0.234 m along y
BLADE_HALF = (0.15, 0.117, 0.01)
BLADE_ANGLE = math.radians(40.0)  # about +y: local +x edge descends toward +x (leading edge low)
BLADE_POS0 = (-0.24, 0.0, 0.369)  # tip starts at (-0.125, 0.273) above the descent target point
HANDLE_LEN = 0.3
HANDLE_HALF_THICK = 0.015
HANDLE_ANGLE = math.radians(40.0)

DROPLET_POS = (-0.02, 0.0, 0.245)  # contact release at the (deeper, 0.20 m) sand surface, shifted
# -0.04 m with the scoop trajectory so the wet clump stays centered on the scoop path
DROPLET_RADIUS = 0.04

# 0.5 was tried and is marginally unstable at this fine particle size (NaN in the full run, 10x
# velocity spikes in the absorb probe); the demo uses the previously validated 0.3
MAX_RATIO = 0.3


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
        # user request (2026-08-20): deeper sand bed (0.20 m) so the scoop no longer grazes the
        # domain floor at the insert end (tip ends ~0.047 m above it, ~7 grain layers) --
        # the tilt-box-era trajectory pinched grains against the floor and erupted
        dem_options=gs.options.DEMOptions(
            particle_size=2.0 * PARTICLE_RADIUS,
            ddt_safety=0.5,
            # user request (2026-08-20): stronger wet-sand cohesion (~3x the reference 0.007) so the
            # wet clump sticks together instead of trickling through the litter scoop's slots
            surface_tension_coeff=0.02,
            lower_bound=BOX_LOWER,
            upper_bound=BOX_UPPER,
        ),
        flip_options=gs.options.FLIPOptions(
            grid_res=128,  # dx ~ 6.25 mm over the 0.8 m extent = the sand particle diameter; absorption
            # needs dx < 15 mm at r = 5 mm (floor truncation), so dx = 2r here keeps it well active
            viscosity_coeff=0.01,  # weaker sand<->water drag than the reference default 1.0: water
            # infiltrates deeper into the bed instead of spreading on the surface (user-selected
            # after the 0.3/0.1/0.04/0.01 infiltration comparison, see phase5_absorb_test.py)
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
    sand = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.1005),
            size=(0.68, 0.58, 0.20),
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
    # physical boundary = the litter scoop's voxelized SDF (the same mesh as the visual asset),
    # not an analytic tilt box; the pose/motion constants are unchanged (mesh frame = blade frame)
    sdf_data = np.load(SDF_NPZ)
    dem.set_sdf_obstacle(
        sdf_data["sdf_val"],
        sdf_data["dims"],
        sdf_data["origin"],
        sdf_data["cell"],
        BLADE_POS0,
        quat=tuple(blade_quat0),
    )
    print(
        f"n_water = {water.n_particles}, n_sand = {sand.n_particles}, "
        f"single_ratio = {flip._single_ratio:.4f}",
        flush=True,
    )

    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(WET_RAW_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    os.makedirs(REC_DIR, exist_ok=True)

    # static scene parameters for offline re-rendering of the per-frame npz recordings
    with open(os.path.join(REC_DIR, "meta.json"), "w") as f_meta:
        json.dump(
            {
                "dt": DT,
                "particle_radius": PARTICLE_RADIUS,
                "grid_res": 128,
                "max_ratio": MAX_RATIO,
                "viscosity_coeff": 0.01,
                "surface_tension_coeff": 0.02,
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
            },
            f_meta,
            indent=2,
        )

    cam.start_recording(save_to_filename=VIDEO_PATH, fps=60)

    t0 = time.time()
    i_descend = N_SETTLE_STEPS
    i_insert = i_descend + N_DESCEND_STEPS
    i_rotate = i_insert + N_INSERT_STEPS
    i_lift = i_rotate + N_ROTATE_STEPS
    i_hold = i_lift + N_LIFT_STEPS
    n_steps = i_hold + N_HOLD_STEPS
    if MAX_STEPS is not None:
        n_steps = min(n_steps, MAX_STEPS)
    keyframe_idx = 0
    wet_idx = 0
    n_water0 = water.n_particles
    for i in range(n_steps):
        if i == i_descend:
            dem.set_sdf_obstacle_vel(DESCEND_VEL)
            print(f"step {i + 1}: blade starts descending vertically at {DESCEND_VEL} m/s", flush=True)
        elif i == i_insert:
            dem.set_sdf_obstacle_vel(INSERT_VEL)
            print(f"step {i + 1}: blade starts inserting along the blade direction at {INSERT_VEL} m/s", flush=True)
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
        blade_pos = dem.get_sdf_obstacle_pos()[0]
        blade_quat = dem.get_sdf_obstacle_quat()[0]
        shovel.set_pos(blade_pos)
        shovel.set_quat(blade_quat)

        scene.step()

        # per-frame recording for offline re-rendering (blade pose = the one just synced to the
        # kinematic visualization entities above)
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
                os.path.join(FRAMES_DIR, f"phase5_shovel_wet_frame_{keyframe_idx:04d}.png"),
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
                f"nan = {np.isnan(pos).any()}  elapsed = {time.time() - t0:.1f}s",
                flush=True,
            )

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
