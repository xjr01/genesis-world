"""MF-8 final demo: a jet of water free-falls into a cup of coffee (IPBF + diffusion).

Scene: tall open-top cup (multiflow/assets/cup_tall.obj, inner R=0.12, inner H=0.75,
cavity floor z=0.02) shown as a semi-transparent rigid mesh (visualization only --
rigid<->IPBF coupling does not exist; the physical constraint is the CylinderBoundary
clamp at R=0.12, z_bottom=0.02). Inside the cup: a coffee column (c_init=1.0, brown)
from z=0.03 to z~0.35. Above the rim: a thin water column (c_init=0.0, blue) starting
at z~0.85 that free-falls into the coffee, splashes, and the XSPH-style concentration
diffusion (IPBFOptions.diffusion_coeff) mixes the two phases. Rendered as a
per-particle colored point cloud (VisOptions render_particle_as="points").

Per-frame CSV: multiflow/videos/mf8{tag}_metrics.csv
  (frame, t, sum_c, std_c, zc_c, r_max, z_min, ke, nan_count)
Video: multiflow/videos/mf8_ipbf_pour{tag}.mp4

Pass criteria (printed at the end):
  1. r_max  <= 0.13 (R_CYL + ps) for every frame; z_min >= 0.01 (z_bottom - ps); nan == 0
  2. sum_c relative drift < 1%
  3. std_c drops below 60% of its initial value by the end (mixing happened)

Run (smoke first!):
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf8_ipbf_pour.py --seconds 3 --tag _smoke
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf8_ipbf_pour.py --epsd 0.02 --seconds 30 --tag ""
"""

import argparse
import csv
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np  # noqa: E402

import genesis as gs  # noqa: E402

print("genesis loaded from:", gs.__file__)
assert os.path.join(WORKSPACE, "genesis-world") in gs.__file__, "Wrong genesis! Must be the multiflow copy."

CUP_OBJ = os.path.join(WORKSPACE, "assets", "cup_tall.obj")
VIDEOS_DIR = os.path.join(WORKSPACE, "videos")

# cup geometry (see assets/gen_cup.py --r-in 0.12 --h-in 0.75): cavity floor z=0.02, rim z=0.77
R_IN = 0.12
T_BOTTOM = 0.02
PS = 0.01  # particle_size
DT = 1.0 / 60.0
SUBSTEPS = 8

# coffee column inside the cup: r=0.10 (< R_IN - ps), z in [0.03, 0.35] (~half the cup)
R_COFFEE = 0.10
Z_COFFEE0 = T_BOTTOM + PS  # 0.03
Z_COFFEE1 = 0.35
# water jet above the rim: r=0.035, h=0.3, bottom face at z=0.85 (free-falls ~0.5m, impact ~0.35s)
R_WATER = 0.035
Z_WATER0 = 0.85
H_WATER = 0.30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsd", type=float, default=0.02, help="diffusion_coeff (0 = off)")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--tag", default="", help="suffix for output file names, e.g. '_e002'")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--point-size", type=float, default=6.0, help="GL point size in pixels")
    args = parser.parse_args()

    csv_path = os.path.join(VIDEOS_DIR, f"mf8{args.tag}_metrics.csv")
    mp4_path = os.path.join(VIDEOS_DIR, f"mf8_ipbf_pour{args.tag}.mp4")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        ipbf_options=gs.options.IPBFOptions(
            particle_size=PS,
            lower_bound=(-0.15, -0.15, 0.0),
            upper_bound=(0.15, 0.15, 1.25),
            boundary_particles=False,
            boundary_cylinder=(0.0, 0.0, R_IN, T_BOTTOM),  # open-top cylinder clamp
            ipbf_iterations=2,
            alpha=1e-8,
            viscosity_xsph=0.1,
            damping_enabled=False,
            diffusion_coeff=args.epsd,
        ),
        vis_options=gs.options.VisOptions(
            render_particle_as="points",  # per-particle concentration coloring (IPBF points branch)
        ),
        show_viewer=False,
    )

    # cup: visualization only (no rigid<->IPBF coupling; CylinderBoundary does the physics)
    cup = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=CUP_OBJ,
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            decimate=False,  # keep the exact 96-segment cylinder
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.8, 0.8, 0.85, 0.35), opacity=0.35),
    )

    coffee = scene.add_entity(
        material=gs.materials.IPBF.Liquid(sampler="regular", rho=1000.0, c_init=1.0),
        morph=gs.morphs.Cylinder(
            radius=R_COFFEE,
            height=Z_COFFEE1 - Z_COFFEE0,
            pos=(0.0, 0.0, 0.5 * (Z_COFFEE0 + Z_COFFEE1)),
        ),
    )
    water = scene.add_entity(
        material=gs.materials.IPBF.Liquid(sampler="regular", rho=1000.0, c_init=0.0),
        morph=gs.morphs.Cylinder(
            radius=R_WATER,
            height=H_WATER,
            pos=(0.0, 0.0, Z_WATER0 + H_WATER / 2.0),
        ),
    )

    # oblique top view: free surface, cup side, and the falling jet above the rim
    cam = scene.add_camera(res=(960, 1280), pos=(0.95, -0.95, 1.05), lookat=(0.0, 0.0, 0.48), fov=35, GUI=False)
    scene.build()
    print(f"cup geoms: {cup.n_geoms}, coffee particles: {coffee.n_particles}, water particles: {water.n_particles}")
    print(f"epsd={args.epsd}, seconds={args.seconds}, tag='{args.tag}'")

    # GL point size for the point-cloud rendering (consumed in Rasterizer.render_camera)
    scene.visualizer.rasterizer._camera_targets[cam.uid].point_size = args.point_size

    solver = scene.sim.ipbf_solver
    assert type(solver.boundary).__name__ == "CylinderBoundary", f"unexpected boundary: {type(solver.boundary)}"
    n_fluid = solver._n_fluid_particles
    n_frames = int(round(args.seconds / DT))
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    rows = []
    cam.start_recording(save_to_filename=mp4_path, fps=args.fps)
    for i in range(n_frames + 1):
        if i > 0:
            scene.step()
        pos = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
        vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
        c = solver.particles.c.to_numpy()[:n_fluid, 0]
        nan_count = int((~np.isfinite(pos)).sum() + (~np.isfinite(vel)).sum() + (~np.isfinite(c)).sum())
        safe = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        safe_c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        sum_c = float(safe_c.sum())
        std_c = float(safe_c.std())
        zc_c = float((safe_c * safe[:, 2]).sum() / sum_c) if sum_c > 0 else float("nan")
        r_max = float(np.sqrt(safe[:, 0] ** 2 + safe[:, 1] ** 2).max())
        z_min = float(safe[:, 2].min())
        ke = float(0.5 * np.nansum((vel**2).sum(axis=1)))
        rows.append((i, i * DT, sum_c, std_c, zc_c, r_max, z_min, ke, nan_count))
        if i % 60 == 0:
            print(
                f"t={i * DT:6.2f}s  sum_c={sum_c:9.2f}  std_c={std_c:.5f}  zc_c={zc_c:.4f}"
                f"  r_max={r_max:.4f}  z_min={z_min:.4f}  KE={ke:.4e}  nan={nan_count}"
            )
    cam.stop_recording()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "sum_c", "std_c", "zc_c", "r_max", "z_min", "ke", "nan_count"])
        w.writerows(rows)

    # pass/fail summary
    cols = ["frame", "t", "sum_c", "std_c", "zc_c", "r_max", "z_min", "ke", "nan"]
    arr = {k: np.array([r[j] for r in rows], dtype=float) for j, k in enumerate(cols)}
    sum_c0 = arr["sum_c"][0]
    drift = float(np.max(np.abs(arr["sum_c"] - sum_c0)) / sum_c0)
    p1 = bool(arr["r_max"].max() <= R_IN + PS + 1e-6)
    p2 = bool(arr["z_min"].min() >= T_BOTTOM - PS - 1e-6)
    p3 = bool(arr["nan"].max() == 0)
    p4 = bool(drift < 0.01)
    p5 = bool(arr["std_c"][-1] < 0.6 * arr["std_c"][0])
    print("=" * 70)
    print(f"[1] r_max  max = {arr['r_max'].max():.4f}  (limit {R_IN + PS:.2f})     -> {'PASS' if p1 else 'FAIL'}")
    print(f"[1] z_min  min = {arr['z_min'].min():.4f}  (limit {T_BOTTOM - PS:.2f})   -> {'PASS' if p2 else 'FAIL'}")
    print(f"[1] nan    max = {int(arr['nan'].max())}                              -> {'PASS' if p3 else 'FAIL'}")
    print(f"[2] sum_c drift = {drift:.3e}  (limit 1e-2, sum_c0={sum_c0:.2f})   -> {'PASS' if p4 else 'FAIL'}")
    print(
        f"[3] std_c {arr['std_c'][0]:.5f} -> {arr['std_c'][-1]:.5f}"
        f"  (target < {0.6 * arr['std_c'][0]:.5f})       -> {'PASS' if p5 else 'FAIL'}"
    )
    print(f"csv: {csv_path}")
    print(f"mp4: {mp4_path}")


if __name__ == "__main__":
    main()
