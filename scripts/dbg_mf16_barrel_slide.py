#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Two-second MF16 mug-barrel slide fixture for smooth ball narrow-phase A/B.

The same curved two-layer fluid sheet is initialized on the +x barrel, far from the floor, rim,
and -x handle seam.  Boundary balls retain density, broad phase and CCD triggering in both runs;
`--smooth` only enables the opt-in analytic-offset barrel narrow phase.  No video is rendered.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree


WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENESIS = os.path.join(WORKSPACE, "genesis-world")
VIDEOS = os.path.join(WORKSPACE, "videos")
MUG_NPZ = os.path.join(VIDEOS, "mf16_balls_mug.npz")
if GENESIS not in sys.path:
    sys.path.insert(0, GENESIS)

import genesis as gs  # noqa: E402


PS = 0.008
RF = 0.5 * PS
DT = 1.0 / 60.0
SUBSTEPS = 8
R_IN_CONTACT = 0.146
R_OUT_CONTACT = 0.170
Z_MIN = 0.040
Z_MAX = 0.292
HANDLE_AZIMUTH = math.pi
HANDLE_EXCLUDE_HALF = math.radians(35.0)


def curved_initial_positions(raw):
    """Map the regular Box lattice onto a deterministic +x cylindrical sheet."""
    raw = np.asarray(raw, dtype=np.float32)
    x_levels = np.unique(np.round(raw[:, 0], 6))
    x_rank = np.searchsorted(x_levels, np.round(raw[:, 0], 6))
    radial = 0.168 + x_rank * PS
    theta = raw[:, 1] / R_OUT_CONTACT
    out = raw.copy()
    out[:, 0] = radial * np.cos(theta)
    out[:, 1] = radial * np.sin(theta)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    if args.seconds <= 0.0:
        raise SystemExit("--seconds must be > 0")
    tag = args.tag or ("smooth" if args.smooth else "sphere")
    os.makedirs(VIDEOS, exist_ok=True)

    balls = np.load(MUG_NPZ)["pos_local"].astype(np.float64)
    ball_tree = cKDTree(balls)
    smooth_spec = (
        (0, 0.0, 0.0, R_IN_CONTACT, R_OUT_CONTACT, Z_MIN, Z_MAX,
         HANDLE_AZIMUTH, HANDLE_EXCLUDE_HALF)
        if args.smooth else None
    )

    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, -9.81)
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=PS,
            lower_bound=(-0.5, -0.5, -30.0),
            upper_bound=(0.5, 0.5, 0.6),
            boundary_ball_sets=((MUG_NPZ, False),),
            boundary_ball_initial_poses=(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),),
            boundary_ball_smooth_barrel=smooth_spec,
            max_density_solver_iterations=20,
            max_viscosity_solver_iterations=1,
            density_lambda_epsilon=0.1,
            surface_tension_enabled=True,
            st_compliance=0.7,
            st_surface_density_factor=1.0,
            st_distance_enabled=True,
            st_distance_compliance=40.0,
            st_max_surface_neighbors=256,
            velocity_damping=1.0,
            wall_adhesion_enabled=False,
            wall_friction=0.0,
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )
    sheet = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=0.0,
            density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Box(
            pos=(0.176, 0.0, 0.166), size=(0.016, 0.096, 0.192),
        ),
    )
    scene.build()
    solver = scene.sim.pbd_solver
    n_fluid = solver._n_fluid_particles
    assert n_fluid == sheet.n_particles
    assert solver.n_boundary_balls == len(balls)

    raw = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
    initial = curved_initial_positions(raw)
    sheet.set_particles_pos(initial)
    sheet.set_particles_vel(np.zeros_like(initial))
    solver.update_render_fields()

    n_frames = int(round(args.seconds / DT))
    current_run = np.zeros(n_fluid, dtype=np.int32)
    maximum_run = np.zeros(n_fluid, dtype=np.int32)
    tunnel = np.zeros(n_fluid, dtype=bool)
    nan_max = 0
    rows = []
    wall_start = time.time()
    for frame in range(n_frames + 1):
        t = frame * DT
        if frame > 0:
            scene.step()
        pos = solver.particles.pos.to_numpy()[:n_fluid, 0, :].astype(np.float64)
        vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :].astype(np.float64)
        finite = np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1)
        nan_max = max(nan_max, int((~finite).sum()))
        safe = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        safe_vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        radius = np.linalg.norm(safe[:, :2], axis=1)
        speed = np.linalg.norm(safe_vel, axis=1)
        angle = np.arctan2(safe[:, 1], safe[:, 0])
        away_handle = np.abs(np.arctan2(np.sin(angle - HANDLE_AZIMUTH),
                                        np.cos(angle - HANDLE_AZIMUTH))) >= HANDLE_EXCLUDE_HALF
        barrel_z = (safe[:, 2] >= Z_MIN) & (safe[:, 2] <= Z_MAX)
        analytic_contact = (
            finite & away_handle & barrel_z
            & (np.abs(radius - R_OUT_CONTACT) <= 0.5 * PS)
        )
        nearest_distance = ball_tree.query(safe, k=1, workers=-1)[0]
        sphere_contact = finite & barrel_z & (nearest_distance <= PS + 1.0e-5)
        current_run = np.where(analytic_contact, current_run + 1, 0)
        maximum_run = np.maximum(maximum_run, current_run)
        # Outside-sheet particles are a true tunnel only after entering the cavity beyond the
        # complete 16 mm wall.  Radial motion below/above the safe barrel band is irrelevant.
        tunnel |= finite & barrel_z & (radius < R_IN_CONTACT)
        rows.append((
            frame, t, int(analytic_contact.sum()), int(sphere_contact.sum()),
            int((speed < 1.0e-4).sum()), int(tunnel.sum()),
            float(np.percentile(speed, 50)), float(np.percentile(speed, 90)), nan_max,
        ))
        if frame % 15 == 0 or frame == n_frames:
            print(
                f"t={t:.3f} contact={int(analytic_contact.sum())}/{n_fluid} "
                f"sphere={int(sphere_contact.sum())} zero={int((speed < 1e-4).sum())} "
                f"tunnel={int(tunnel.sum())} v50/90={np.percentile(speed,50):.4f}/"
                f"{np.percentile(speed,90):.4f} nan={nan_max}"
            )

    max_residence = maximum_run.astype(np.float64) * DT
    final_contact = rows[-1][2]
    final_zero = rows[-1][4]
    p90_residence = float(np.percentile(max_residence, 90))
    final_contact_frac = final_contact / n_fluid
    final_zero_frac = final_zero / n_fluid
    tunnel_count = int(tunnel.sum())
    contacts = solver.boundary_ball_collision_stats()
    passed = (
        nan_max == 0
        and p90_residence < 0.5
        and final_contact_frac < 0.01
        and final_zero_frac < 0.01
        and tunnel_count == 0
    )
    summary = {
        "tag": tag,
        "smooth": args.smooth,
        "seconds": args.seconds,
        "fluid": n_fluid,
        "balls": len(balls),
        "st_compliance": 0.7,
        "wall_adhesion": False,
        "wall_friction": 0.0,
        "p90_max_contact_residence_s": p90_residence,
        "final_contact": final_contact,
        "final_contact_fraction": final_contact_frac,
        "final_speed_zero": final_zero,
        "final_speed_zero_fraction": final_zero_frac,
        "tunnel": tunnel_count,
        "nan_max": nan_max,
        "ball_contacts_overlap_swept": [int(contacts[0]), int(contacts[1])],
        "wall_seconds": time.time() - wall_start,
        "pass": passed,
    }
    csv_path = os.path.join(VIDEOS, f"mf16_dbg_barrel_slide_{tag}_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("frame", "t", "n_contact", "n_sphere_contact", "n_speed_zero",
                         "n_tunnel", "speed_p50", "speed_p90", "nan"))
        writer.writerows(rows)
    json_path = os.path.join(VIDEOS, f"mf16_dbg_barrel_slide_{tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("=" * 78)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"MF16 BARREL SLIDE {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
