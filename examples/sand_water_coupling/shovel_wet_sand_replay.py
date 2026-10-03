"""
Offline re-render of the per-frame recordings produced by phase5_shovel_wet.py.

Reads experiments/recordings/phase5_shovel_wet/meta.json + frame_%04d.npz (sand positions +
water ratios, active water positions, blade pose per step) and renders a matplotlib video:
sand colored by water ratio (light tan -> dark brown), water as blue dots, the blade as a
semi-transparent box reconstructed from its recorded position + quaternion.

Usage: python experiments/phase5_shovel_wet_replay.py [stride] [max_frames]
Output: experiments/videos/phase5_shovel_wet_replay.mp4
"""

import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.path.join(EXPERIMENTS_DIR, "recordings", "phase5_shovel_wet")
OUT_PATH = os.path.join(EXPERIMENTS_DIR, "videos", "phase5_shovel_wet_replay.mp4")
RAW_DIR = os.path.join(EXPERIMENTS_DIR, "frames", "phase5_shovel_wet_replay_raw")

STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MAX_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9


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


def main():
    with open(os.path.join(REC_DIR, "meta.json")) as f_meta:
        meta = json.load(f_meta)
    box_lower = meta["box_lower"]
    box_upper = meta["box_upper"]
    blade_half = np.array(meta["blade_half"])
    max_ratio = meta["max_ratio"]

    frame_files = sorted(f for f in os.listdir(REC_DIR) if f.startswith("frame_") and f.endswith(".npz"))
    frame_files = frame_files[::STRIDE][:MAX_FRAMES]
    print(f"re-rendering {len(frame_files)} frames (stride {STRIDE})", flush=True)

    os.makedirs(RAW_DIR, exist_ok=True)
    dry = np.array([0.87, 0.72, 0.53])
    wet = np.array([0.25, 0.13, 0.06])

    # blade box edges in the local frame
    hx, hy, hz = blade_half
    corners_local = np.array(
        [[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for out_idx, fname in enumerate(frame_files):
        data = np.load(os.path.join(REC_DIR, fname))
        sand_pos = data["sand_pos"]
        sand_ratio = data["sand_ratio"]
        water_pos = data["water_pos"]
        blade_pos = data["blade_pos"]
        blade_quat = data["blade_quat"]

        fig = plt.figure(figsize=(9.6, 7.2), dpi=150)
        ax = fig.add_subplot(111, projection="3d")
        t = np.clip(sand_ratio / max_ratio, 0.0, 1.0)
        colors = dry[None, :] * (1.0 - t[:, None]) + wet[None, :] * t[:, None]
        ax.scatter(sand_pos[:, 0], sand_pos[:, 1], sand_pos[:, 2], s=2, c=colors, alpha=0.9, edgecolors="none")
        if len(water_pos) > 0:
            ax.scatter(
                water_pos[:, 0], water_pos[:, 1], water_pos[:, 2],
                s=3, c=(0.2, 0.5, 0.9), alpha=0.7, edgecolors="none",
            )
        corners = blade_pos[None, :] + corners_local @ quat_to_R(blade_quat).T
        for a, b in edges:
            ax.plot(
                [corners[a, 0], corners[b, 0]],
                [corners[a, 1], corners[b, 1]],
                [corners[a, 2], corners[b, 2]],
                color=(0.4, 0.7, 1.0), linewidth=2,
            )
        ax.set_xlim(box_lower[0], box_upper[0])
        ax.set_ylim(box_lower[1], box_upper[1])
        ax.set_zlim(0.0, 0.6)
        ax.set_box_aspect((0.7, 0.6, 0.6))
        ax.view_init(elev=18, azim=-60)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        fig.tight_layout()
        fig.savefig(os.path.join(RAW_DIR, f"frame_{out_idx:04d}.png"))
        plt.close(fig)
        if (out_idx + 1) % 20 == 0:
            print(f"{out_idx + 1}/{len(frame_files)}", flush=True)

    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", "20",
            "-i", os.path.join(RAW_DIR, "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", OUT_PATH,
        ],
        check=True,
        capture_output=True,
    )
    print(f"re-rendered video saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
