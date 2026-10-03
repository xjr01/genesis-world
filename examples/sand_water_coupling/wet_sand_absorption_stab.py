"""
Quick stability probe for the absorption max_ratio parameter (no video, coarse scene).

Usage: python experiments/phase5_absorb_stab.py <max_ratio> [n_steps] [particle_radius] [grid_res]

Scene: sand layer + FLIP grid + droplet, mirroring the wet-sand demos. Defaults (r = 5 mm,
grid_res = 80) are ~6x faster per step. Prints max|v| / NaN status every 10 steps. Used to check
the claim that max_ratio >~ 0.35 makes the C++ fraction lower bound 1 - 0.74 * (1 + max_ratio)
non-positive, which destabilizes the projection (A = diag(1/f) * M).
"""

import sys
import time

import numpy as np

import genesis as gs

MAX_RATIO = float(sys.argv[1])
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 180
PARTICLE_RADIUS = float(sys.argv[3]) if len(sys.argv) > 3 else 5e-3
GRID_RES = int(sys.argv[4]) if len(sys.argv) > 4 else 80
BOX_LOWER = (-0.35, -0.30, 0.0)
BOX_UPPER = (0.35, 0.30, 0.80)


def main():
    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0, substeps=1, gravity=(0.0, 0.0, -9.8)),
        dem_options=gs.options.DEMOptions(
            particle_size=2.0 * PARTICLE_RADIUS,
            ddt_safety=0.5,
            lower_bound=BOX_LOWER,
            upper_bound=BOX_UPPER,
        ),
        flip_options=gs.options.FLIPOptions(
            grid_res=GRID_RES,
            lower_bound=BOX_LOWER,
            upper_bound=BOX_UPPER,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    sand = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.081), size=(0.68, 0.58, 0.16)),
        material=gs.materials.DEM.Sand(sampler="fcc", rho=2.5, max_ratio=MAX_RATIO),
    )
    water = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.12, 0.0, 0.55), radius=0.04),
        material=gs.materials.FLIP.Liquid(),
    )
    scene.build()
    dem = scene.sim.dem_solver
    flip = scene.sim.flip_solver
    print(
        f"max_ratio = {MAX_RATIO}, fraction_floor = {flip._fraction_floor:.4f}, "
        f"n_water = {water.n_particles}, n_sand = {sand.n_particles}",
        flush=True,
    )

    t0 = time.time()
    for i in range(N_STEPS):
        scene.step()
        if (i + 1) % 10 == 0:
            pos = sand.get_particles_pos().cpu().numpy()[0]
            vel = sand.get_particles_vel().cpu().numpy()[0]
            ratio = dem.particles.ratio.to_numpy()[:, 0]
            n_active = int(flip.particles.active.to_numpy()[:, 0].sum())
            fmin = flip.target_fraction.to_numpy().min()
            print(
                f"step {i + 1:4d}  max|v| = {np.linalg.norm(vel, axis=1).max():9.4f}  "
                f"ratio_max = {ratio.max():.4f}  water_active = {n_active}  "
                f"target_fraction_min = {fmin:+.4f}  nan = {np.isnan(pos).any()}  "
                f"elapsed = {time.time() - t0:.1f}s",
                flush=True,
            )
            if np.isnan(pos).any():
                print("NaN detected, aborting probe", flush=True)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
