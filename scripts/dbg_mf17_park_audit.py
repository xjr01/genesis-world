"""MF-17 park-segment containment audit (offline, numpy-only).

Reads the frame dumps written by mf17_pour_overflow.py (--dump-frames during the
park window) and measures, per dumped frame:
  1. the distribution of minimum fluid-particle-to-jug-ball centre distances
     (contact band = one particle radius 0.004; penetration = negative gap);
  2. overlap events between the jug's outer-bottom ball band (lifted, world
     z ~ Z_LIFT) and milk particles riding on the table (z <= 2.5 ps).

The MF-16 park-crush signature (jug outer bottom at z=0 crushing table milk,
peak ~38 particles at gap d~1e-4) should be structurally absent now that the
jug rides at Z_LIFT=0.030 for the whole run.

Usage:
  python dbg_mf17_park_audit.py <dumps.npz> [--times 17.5,18.0,...]
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # multiflow/
sys.path.insert(0, HERE)

import mf17_pour_overflow as m  # noqa: E402  (pure numpy pose/transform helpers)

PS = m.PS_FULL
PR = 0.5 * PS


def transform(local, pos, quat):
    return m.transform_points(local, np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dumps")
    parser.add_argument("--times", type=str, default="",
                        help="comma-separated dump times to audit (default: all in file)")
    args = parser.parse_args()

    with np.load(os.path.join(ROOT, "videos", "mf17_balls_cup_small.npz")) as d:
        jug_local = np.asarray(d["pos_local"], dtype=np.float64)
    jug_tree_local = cKDTree(jug_local)

    with np.load(os.path.join(ROOT, "videos", "mf17_settled.npz")) as d:
        n_coffee = int(d["n_coffee"])
        n_milk = int(d["n_milk"])

    with np.load(args.dumps) as d:
        keys = [k for k in d.files if not k.startswith(
            ("vel_", "c_", "pose_", "valid_", "mug_resident_", "not_jug_",
             "jug_wall_entered_", "jug_bottom_entered_", "mug_wall_entered_",
             "mug_bottom_entered_", "mug_rim_crossed_", "mug_cleared_",
             "mug_wall_tunnel_", "mug_bottom_tunnel_", "solver_rim_staged_",
             "solver_direct_tunnel_"))]
        times = [float(v) for v in args.times.split(",") if v.strip()] or [float(k) for k in keys]
        print(f"park audit: n_coffee={n_coffee} n_milk={n_milk} jug balls={len(jug_local)} "
              f"times={times}")
        worst = None
        for t in times:
            k = f"{t:g}"
            if k not in d.files:
                cand = min(keys, key=lambda kk: abs(float(kk) - t))
                k = cand
                t = float(k)
            pos = np.asarray(d[k], dtype=np.float64)  # (n_fluid, 3)
            pose = np.asarray(d[f"pose_{k}"], dtype=np.float64)
            origin, quat = pose[:3], pose[3:]
            jug_world = transform(jug_local, origin, quat)
            tree = cKDTree(jug_world)
            dist, idx = tree.query(pos, k=1, workers=-1)
            gap = dist - 2.0 * PR  # centre distance minus contact distance
            milk = pos[n_coffee:]
            milk_gap = gap[n_coffee:]
            coffee_gap = gap[:n_coffee]
            # jug outer-bottom band: lowest 5% of jug ball centres in world z
            z_lo = np.quantile(jug_world[:, 2], 0.05)
            bottom_band = jug_world[:, 2] <= z_lo + 0.01
            # milk riding on the table under/near the jug footprint
            table_milk = milk[milk[:, 2] <= 2.5 * PS]
            # overlap event: a table milk particle whose centre is within one
            # particle diameter of ANY jug ball (i.e. gap < -PS => true overlap)
            near_bottom = 0
            overlap = 0
            if len(table_milk):
                dm, _ = tree.query(table_milk, k=1, workers=-1)
                gm = dm - 2.0 * PR
                near_bottom = int((gm < PS).sum())          # contact-ish band
                overlap = int((gm < -0.25 * PS).sum())      # genuine penetration
            line = (
                f"t={t:5.2f}s  jug bottom band z<={z_lo + 0.01:.3f} (min ball z {jug_world[:,2].min():.4f}) | "
                f"fluid->jug gap: min {gap.min():+.6f}  milk p1 {np.percentile(milk_gap, 1):+.6f} "
                f"p50 {np.percentile(milk_gap, 50):+.6f}  coffee p1 {np.percentile(coffee_gap, 1):+.6f} | "
                f"gap<0: all {int((gap < 0).sum())} milk {int((milk_gap < 0).sum())} | "
                f"table milk {len(table_milk)}: near(gap<ps) {near_bottom} overlap(gap<-0.25ps) {overlap}"
            )
            print(line)
            if worst is None or gap.min() < worst[0]:
                worst = (float(gap.min()), t)
        print(f"WORST fluid->jug gap {worst[0]:+.6f} at t={worst[1]:.2f}s "
              f"(contact at 0; MF-16 crush signature was gap~+1e-4 at jug bottom z=0)")


if __name__ == "__main__":
    main()
