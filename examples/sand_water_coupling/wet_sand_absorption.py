"""
Absorption test: drip water onto fine sand with max_ratio = 0.3 and stop as soon as the water is
fully absorbed -- no shovel, just the wetting process.

Scene matches the fine-grain wet-sand shovel demo: a walled box (interior = DEM domain bounds,
x in [-0.35, 0.35], y in [-0.30, 0.30]) with a sand layer (0.68 x 0.58 x 0.16 m, fcc,
particle diameter 6.25 mm = the FLIP grid dx at grid_res = 128); the four walls are kinematic
visualization-only entities. At t = 0 a water droplet (radius 0.04 m) is released at (0.02, 0)
right at the surface (z = 0.205, contact release -- no falling impact) and is absorbed
(max_ratio = 0.3, dx ~ 6.25 mm keeps absorption active), with the sand-water drag coefficient
given by the command line (viscosity_coeff, reference default 1.0; lower = deeper infiltration
instead of surface spreading).

The loop runs at most N_MAX_STEPS steps and stops early once water_active hits 0 (checked every
5 steps, starting after the droplet has landed). With max_ratio = 0.3 at the coarse resolution the
same scene absorbed fully in ~390 steps.

Videos:
- experiments/videos/phase5_absorb_test{TAG}.mp4                 (sand + water, native rendering)
- experiments/videos/phase5_absorb_test{TAG}_sandwet.mp4         (3D, per-grain color by ratio)
- experiments/videos/phase5_absorb_test{TAG}_sandwet_slice.mp4   (y = 0 cross-section: the 3D
  view hides wet grains below the surface; the slice shows the actual infiltration depth)
Infiltration log: experiments/recordings/phase5_absorb{TAG}_infiltration.csv (per-frame wet-grain
stats: wet z min/mean, local surface z, depth = surface - wet z min, wet-disc centroid and
diameter = 2 * 95th-percentile horizontal radius).
Keyframes: experiments/frames/phase5_absorb_test{TAG}_frame_*.png.
"""

import os
import subprocess
import sys
import time

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import genesis as gs

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(EXPERIMENTS_DIR, "frames")

DT = 1.0 / 60.0
N_MAX_STEPS = 480  # 8 s cap; stops early once water_active == 0
CHECK_EVERY = 5
CHECK_START = 30  # contact release: absorption starts immediately
MAX_RATIO = 0.3
DRAG_COEFF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3  # viscosity_coeff: reference default 1.0
TAG = sys.argv[2] if len(sys.argv) > 2 else ""  # output filename suffix for comparison runs
KEYFRAME_STEPS = [0, 90, 180, 270, 360]  # plus one final frame at the stop step
WET_EVERY = 3

VIDEO_PATH = os.path.join(EXPERIMENTS_DIR, "videos", f"phase5_absorb_test{TAG}.mp4")
VIDEO_WET_PATH = os.path.join(EXPERIMENTS_DIR, "videos", f"phase5_absorb_test{TAG}_sandwet.mp4")
VIDEO_SLICE_PATH = os.path.join(EXPERIMENTS_DIR, "videos", f"phase5_absorb_test{TAG}_sandwet_slice.mp4")
WET_RAW_DIR = os.path.join(FRAMES_DIR, f"phase5_absorb_test{TAG}_sandwet_raw")
SLICE_RAW_DIR = os.path.join(FRAMES_DIR, f"phase5_absorb_test{TAG}_sandwet_slice_raw")
CSV_PATH = os.path.join(EXPERIMENTS_DIR, "recordings", f"phase5_absorb{TAG}_infiltration.csv")

PARTICLE_RADIUS = 3.125e-3  # sand particle diameter 6.25 mm = 0.8 / 128, matching the FLIP grid dx
BOX_LOWER = (-0.35, -0.30, 0.0)
BOX_UPPER = (0.35, 0.30, 0.80)
WALL_THICK = 0.01
WALL_HEIGHT = 0.18

DROPLET_Z = float(sys.argv[3]) if len(sys.argv) > 3 else 0.205  # 0.205 = contact release; 0.55 = falling
DROPLET_POS = (0.02, 0.0, DROPLET_Z)
DROPLET_RADIUS = 0.04


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
    ax.set_box_aspect((0.7, 0.6, 0.3))
    ax.view_init(elev=18, azim=-60)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_slice_frame(pos, ratio, water_pos, surf_z, step, path):
    # y = 0 cross-section: the 3D view hides wet grains below the surface, so the infiltration
    # depth is only visible in a slice
    fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=150)
    mask = np.abs(pos[:, 1]) <= 0.02
    t = np.clip(ratio[mask] / MAX_RATIO, 0.0, 1.0)
    dry = np.array([0.87, 0.72, 0.53])
    wet = np.array([0.25, 0.13, 0.06])
    colors = dry[None, :] * (1.0 - t[:, None]) + wet[None, :] * t[:, None]
    ax.scatter(pos[mask, 0], pos[mask, 2], s=12, c=colors, edgecolors="none")
    if len(water_pos) > 0:
        wmask = np.abs(water_pos[:, 1]) <= 0.03
        ax.scatter(
            water_pos[wmask, 0], water_pos[wmask, 2],
            s=8, c=[(0.2, 0.5, 0.9)], alpha=0.8, edgecolors="none",
        )
    ax.axhline(surf_z, color="gray", linewidth=1, linestyle="--", label=f"surface z = {surf_z:.3f}")
    ax.set_xlim(-0.2, 0.25)
    ax.set_ylim(0.0, 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"y = 0 slice, step {step} (t = {step * DT:.2f} s)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
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
            grid_res=128,  # dx ~ 6.25 mm over the 0.8 m extent = the sand particle diameter
            viscosity_coeff=DRAG_COEFF,
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
    print(
        f"n_water = {water.n_particles}, n_sand = {sand.n_particles}, "
        f"single_ratio = {flip._single_ratio:.4f}",
        flush=True,
    )

    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(WET_RAW_DIR, exist_ok=True)
    os.makedirs(SLICE_RAW_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # per-frame infiltration log: wet grains are ratio > 0.02; the local surface is the 99.5th z
    # percentile of dry grains in the column under the droplet; the wet-disc diameter is twice the
    # 95th percentile horizontal distance of wet grains from their centroid
    with open(CSV_PATH, "w") as f_csv:
        f_csv.write(
            "step,time,n_water_active,n_wet,ratio_max,wet_z_min,wet_z_mean,surface_z,depth,"
            "wet_cx,wet_cy,wet_diameter\n"
        )

    cam.start_recording(save_to_filename=VIDEO_PATH, fps=60)

    t0 = time.time()
    keyframe_idx = 0
    wet_idx = 0
    n_water0 = water.n_particles
    stop_step = N_MAX_STEPS
    for i in range(N_MAX_STEPS):
        scene.step()

        n_active = int(flip.particles.active.to_numpy()[:, 0].sum()) if i % CHECK_EVERY == 0 else -1

        if i % WET_EVERY == 0:
            pos = sand.get_particles_pos().cpu().numpy()[0]
            ratio = dem.particles.ratio.to_numpy()[:, 0].copy()
            active_mask = flip.particles.active.to_numpy()[:, 0].astype(bool)
            water_pos = water.get_particles_pos().cpu().numpy()[0][active_mask]
            wet = ratio > 0.02
            column = (np.abs(pos[:, 0] - DROPLET_POS[0]) < 0.08) & (np.abs(pos[:, 1]) < 0.08) & ~wet
            surf_z = float(np.percentile(pos[column, 2], 99.5)) if column.any() else float("nan")
            wet_z_min = float(pos[wet, 2].min()) if wet.any() else float("nan")
            wet_z_mean = float(pos[wet, 2].mean()) if wet.any() else float("nan")
            depth = surf_z - wet_z_min
            if wet.any():
                wet_cx = float(pos[wet, 0].mean())
                wet_cy = float(pos[wet, 1].mean())
                r_xy = np.hypot(pos[wet, 0] - wet_cx, pos[wet, 1] - wet_cy)
                wet_diameter = 2.0 * float(np.percentile(r_xy, 95))
            else:
                wet_cx = wet_cy = wet_diameter = float("nan")
            n_active_frame = int(flip.particles.active.to_numpy()[:, 0].sum())
            with open(CSV_PATH, "a") as f_csv:
                f_csv.write(
                    f"{i + 1},{(i + 1) * DT:.3f},{n_active_frame},{int(wet.sum())},"
                    f"{ratio.max():.4f},{wet_z_min:.4f},{wet_z_mean:.4f},{surf_z:.4f},{depth:.4f},"
                    f"{wet_cx:.4f},{wet_cy:.4f},{wet_diameter:.4f}\n"
                )
            render_wet_frame(pos, ratio, os.path.join(WET_RAW_DIR, f"frame_{wet_idx:04d}.png"))
            render_slice_frame(
                pos, ratio, water_pos, surf_z, i + 1,
                os.path.join(SLICE_RAW_DIR, f"frame_{wet_idx:04d}.png"),
            )
            wet_idx += 1

        if i in KEYFRAME_STEPS:
            rgb, *_ = cam.render(rgb=True)
            imageio.imwrite(
                os.path.join(FRAMES_DIR, f"phase5_absorb_test{TAG}_frame_{keyframe_idx:04d}.png"),
                rgb[0] if isinstance(rgb, list) else rgb,
            )
            keyframe_idx += 1

        if (i + 1) % 30 == 0:
            pos = sand.get_particles_pos().cpu().numpy()[0]
            vel = sand.get_particles_vel().cpu().numpy()[0]
            ratio = dem.particles.ratio.to_numpy()[:, 0]
            if n_active < 0:
                n_active = int(flip.particles.active.to_numpy()[:, 0].sum())
            print(
                f"step {i + 1:4d}/{N_MAX_STEPS}  max|v| = {np.linalg.norm(vel, axis=1).max():8.4f}  "
                f"min_z = {pos[:, 2].min():+.6f}  n_wet(ratio>0.02) = {(ratio > 0.02).sum():6d}  "
                f"ratio_max = {ratio.max():.4f}  ratio_mean = {ratio.mean():.4f}  "
                f"water_active = {n_active}/{n_water0}  "
                f"nan = {np.isnan(pos).any()}  elapsed = {time.time() - t0:.1f}s",
                flush=True,
            )

        if i >= CHECK_START and i % CHECK_EVERY == 0 and n_active == 0:
            stop_step = i + 1
            print(f"step {i + 1}: water fully absorbed, stopping", flush=True)
            break

    # final keyframe at the stop step
    rgb, *_ = cam.render(rgb=True)
    imageio.imwrite(
        os.path.join(FRAMES_DIR, f"phase5_absorb_test{TAG}_frame_{keyframe_idx:04d}.png"),
        rgb[0] if isinstance(rgb, list) else rgb,
    )
    pos = sand.get_particles_pos().cpu().numpy()[0]
    ratio = dem.particles.ratio.to_numpy()[:, 0].copy()
    active_mask = flip.particles.active.to_numpy()[:, 0].astype(bool)
    water_pos = water.get_particles_pos().cpu().numpy()[0][active_mask]
    wet = ratio > 0.02
    column = (np.abs(pos[:, 0] - DROPLET_POS[0]) < 0.08) & (np.abs(pos[:, 1]) < 0.08) & ~wet
    surf_z = float(np.percentile(pos[column, 2], 99.5)) if column.any() else float("nan")
    render_wet_frame(pos, ratio, os.path.join(WET_RAW_DIR, f"frame_{wet_idx:04d}.png"))
    render_slice_frame(
        pos, ratio, water_pos, surf_z, stop_step,
        os.path.join(SLICE_RAW_DIR, f"frame_{wet_idx:04d}.png"),
    )

    cam.stop_recording()

    for raw_dir, out_path in [(WET_RAW_DIR, VIDEO_WET_PATH), (SLICE_RAW_DIR, VIDEO_SLICE_PATH)]:
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", "20",
                "-i", os.path.join(raw_dir, "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path,
            ],
            check=True,
            capture_output=True,
        )
    print(
        f"stopped at step {stop_step}; videos saved to {VIDEO_PATH}, {VIDEO_WET_PATH}, "
        f"{VIDEO_SLICE_PATH}; infiltration log at {CSV_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
