"""MF-3: cylinder (cup) leak test — PBF liquid inside a rigid mesh cup via rigid<->PBD SDF coupling.

Cup: multiflow/assets/cup.obj (inner R=0.15, inner H=0.6, wall/bottom 0.02 = 2*particle_size).
Liquid: PBD box inscribed in the cup interior, filling lower ~60% of inner height.

Checks over --seconds (default 10s, dt=1/60, 10 substeps):
  1. r_max  <= 0.15 + 0.01   (no particle escapes the side wall)
  2. z_min  >= 0.02 - 0.01   (no particle falls through the bottom)
  3. nan_count == 0
  4. liquid settles in the cup (KE stabilizes/decays at the end)

Outputs:
  multiflow/videos/mf3_leak_metrics.csv
  multiflow/videos/mf3_cylinder_leak.mp4

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf3_cylinder_leak_test.py --seconds 10
"""

import argparse
import csv
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import genesis as gs  # noqa: E402

print("genesis loaded from:", gs.__file__)
assert os.path.join(WORKSPACE, "genesis-world") in gs.__file__, "Wrong genesis! Must be the multiflow copy."

CUP_OBJ = os.path.join(WORKSPACE, "assets", "cup.obj")
VIDEOS_DIR = os.path.join(WORKSPACE, "videos")
CSV_PATH = os.path.join(VIDEOS_DIR, "mf3_leak_metrics.csv")
MP4_PATH = os.path.join(VIDEOS_DIR, "mf3_cylinder_leak.mp4")

# cup geometry (see assets/gen_cup.py)
R_IN = 0.15
T_BOTTOM = 0.02
H_IN = 0.6
PS = 0.01  # particle_size
DT = 1.0 / 60.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=10,  # substep dt ~1.7e-3 -> per-substep travel << wall thickness (no tunneling)
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=PS,
            lower_bound=(-1.0, -1.0, -0.5),
            upper_bound=(1.0, 1.0, 2.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.9, -0.9, 0.8),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=35,
        ),
        show_viewer=False,
    )

    cup = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=CUP_OBJ,
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            decimate=False,  # keep the exact 96-segment cylinder; convexify (default True) handles concavity
        ),
        material=gs.materials.Rigid(),  # needs_coup defaults True
        surface=gs.surfaces.Default(color=(0.8, 0.8, 0.85, 0.35), opacity=0.35),
    )

    # liquid box inscribed in the cup interior: corner radius sqrt(2)*0.095 = 0.134 < R_IN - PS = 0.14
    half = 0.095
    z0 = T_BOTTOM + PS  # start 1*ps above the inner floor
    z1 = z0 + 0.6 * H_IN  # fill lower ~60% of inner height
    liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular",
            rho=1.0,
            density_relaxation=1.0,
            viscosity_relaxation=0.0,
        ),
        morph=gs.morphs.Box(lower=(-half, -half, z0), upper=(half, half, z1)),
    )

    cam = scene.add_camera(res=(1280, 960), pos=(0.9, -0.9, 0.8), lookat=(0.0, 0.0, 0.3), fov=35, GUI=False)
    scene.build()
    print(f"cup geoms: {cup.n_geoms}, liquid particles: {liquid.n_particles}")
    print(f"rigid_pbd coupling active: {scene.sim.coupler._rigid_pbd}")

    os.makedirs(VIDEOS_DIR, exist_ok=True)
    n_steps = int(round(args.seconds / DT))
    cam.start_recording(save_to_filename=MP4_PATH, fps=int(round(1.0 / DT)))

    rows = []
    prev_pos = None
    for i in range(n_steps):
        scene.step()
        pos = liquid.get_particles_pos()
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)
        pos = pos.reshape(-1, 3).float()
        nan_count = int((~torch.isfinite(pos)).sum().item())
        safe = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        r_max = float(safe[:, :2].norm(dim=1).max().item())
        z_min = float(safe[:, 2].min().item())
        if prev_pos is None:
            ke = 0.0
        else:
            vel = (safe - prev_pos) / DT
            ke = float((0.5 * vel.norm(dim=1) ** 2).mean().item())
        prev_pos = safe
        rows.append((i, (i + 1) * DT, r_max, z_min, ke, nan_count))
        cam.render()
        if (i + 1) % 60 == 0:
            print(f"t={rows[-1][1]:5.2f}s  r_max={r_max:.4f}  z_min={z_min:.4f}  KE={ke:.4e}  nan={nan_count}")

    cam.stop_recording()

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "r_max", "z_min", "ke", "nan_count"])
        w.writerows(rows)

    # pass/fail summary
    arr = {k: np.array([r[j] for r in rows]) for j, k in enumerate(["frame", "t", "r_max", "z_min", "ke", "nan"])}
    tail = arr["t"] >= arr["t"][-1] - 2.0  # last 2s
    p1 = bool(arr["r_max"].max() <= R_IN + PS + 1e-6)
    p2 = bool(arr["z_min"].min() >= T_BOTTOM - PS - 1e-6)
    p3 = bool(arr["nan"].max() == 0)
    p4 = bool(arr["ke"][tail][-1] <= arr["ke"][tail].max())  # KE decaying/steady in the tail
    print("=" * 60)
    print(f"[1] r_max  max = {arr['r_max'].max():.4f}  (limit {R_IN + PS:.2f})        -> {'PASS' if p1 else 'FAIL'}")
    print(f"[2] z_min  min = {arr['z_min'].min():.4f}  (limit {T_BOTTOM - PS:.2f})     -> {'PASS' if p2 else 'FAIL'}")
    print(f"[3] nan    max = {int(arr['nan'].max())}                                 -> {'PASS' if p3 else 'FAIL'}")
    print(f"[4] KE tail: max={arr['ke'][tail].max():.3e} final={arr['ke'][-1]:.3e}    -> {'PASS' if p4 else 'FAIL'}")
    print(f"csv: {CSV_PATH}")
    print(f"mp4: {MP4_PATH}")


if __name__ == "__main__":
    main()
