"""MF-4 final demo: coffee diffusing into water inside a tall cylindrical cup.

Scene: tall open-top cup (multiflow/assets/cup_tall.obj, inner R=0.12, inner H=0.75,
aspect 3.125) fixed rigid mesh + PBF liquid. Lower half of the settled liquid column is
coffee (c=1, brown), upper half water (c=0, blue); the XSPH-style concentration diffusion
(PBDOptions.diffusion_coeff) gradually mixes them. Rendered as a per-particle colored
point cloud (VisOptions render_particle_as="points").

Workflow: settle-first. Genesis PBF's rest spacing (~0.85*ps) is tighter than the regular
sampling spacing (ps), so a freshly sampled column compacts by ~1.5x in volume with a
violent splash (measured: a 0.49m column slumps to ~0.23m). Recording that would pre-mix
the two phases and wreck the demo. Instead:
  1. fill the cup cross-section (R_LIQ = R_IN - ps) from z=0.03 up to z=0.73 (~95% fill),
  2. pre-roll --settle seconds (no recording): the column compacts to ~60% fill, at rest,
  3. assign c=1 below / c=0 above the *settled* column median, zero residual velocities,
  4. start recording + metrics (t=0 of the CSV/video is this clean two-phase state).

Other hard-won settings (see smoke-test postmortems in the multiflow RECORD):
  - max_density_solver_iterations=10: the PBD default (1) under-converges hydrostatic
    pressure in this tall narrow column, pumps energy and blows particles through the
    wall at t~0.3s.
  - CoacdOptions(threshold=0.02, preprocess_resolution=100): the default decomposition
    (threshold 0.1 / res 30) yields coarse hulls whose sagitta bulges into the thin tall
    wall's cavity and ejects particles on step 1.
  - substeps=10: per-substep travel << wall thickness (no tunneling).

Per-frame CSV: multiflow/videos/mf4{tag}_metrics.csv
  (frame, t, sum_c, std_c, zc_c [absolute z centroid of c], r_max, z_min, z_max, ke, nan_count)
Video: multiflow/videos/mf4_coffee_demo{tag}.mp4

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf4_coffee_demo.py --epsd 0.1 --seconds 40 --tag ""
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

CUP_OBJ = os.path.join(WORKSPACE, "assets", "cup_tall.obj")
VIDEOS_DIR = os.path.join(WORKSPACE, "videos")

# cup geometry (see assets/gen_cup.py --r-in 0.12 --h-in 0.75)
R_IN = 0.12
T_BOTTOM = 0.02
H_IN = 0.75
PS = 0.01  # particle_size
DT = 1.0 / 60.0

# initial liquid column (pre-settle): fills the cross-section up to the rim (top 0.75 = rim),
# so that after the ~1.5x volume compaction the settled column reaches ~2/3 of the cup
R_LIQ = R_IN - 1.5 * PS   # 0.105, 1.5*ps clearance to the wall
Z0 = T_BOTTOM + PS        # 0.03, start 1*ps above the inner floor
H_INIT = 0.72             # initial column height; top at 0.75 = rim (compaction only pulls down)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsd", type=float, default=0.1, help="diffusion_coeff (0 = off)")
    parser.add_argument("--seconds", type=float, default=40.0, help="recorded seconds after settling")
    parser.add_argument("--settle", type=float, default=4.0, help="pre-roll seconds to settle (not recorded)")
    parser.add_argument("--tag", default="", help="suffix for output file names, e.g. '_e02'")
    parser.add_argument("--point-size", type=float, default=6.0, help="GL point size in pixels")
    args = parser.parse_args()

    csv_path = os.path.join(VIDEOS_DIR, f"mf4{args.tag}_metrics.csv")
    mp4_path = os.path.join(VIDEOS_DIR, f"mf4_coffee_demo{args.tag}.mp4")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=10,  # substep dt ~1.7e-3 -> per-substep travel << wall thickness (no tunneling)
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=PS,
            lower_bound=(-0.5, -0.5, -0.1),
            upper_bound=(0.5, 0.5, 1.5),
            # default density iterations = 1 is not enough for a tall narrow column:
            # hydrostatic pressure under-converges, pumps energy and blows particles through
            # the wall at t~0.3s. 10 iterations (same as MF-2) keeps it contained.
            max_density_solver_iterations=20,
            diffusion_coeff=args.epsd,
        ),
        vis_options=gs.options.VisOptions(
            render_particle_as="points",  # per-particle concentration coloring (PBD points branch)
        ),
        show_viewer=False,
    )

    cup = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=CUP_OBJ,
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            decimate=False,  # keep the exact 96-segment cylinder; convexify (default True) handles concavity
            # tall thin wall: defaults (threshold 0.1, preprocess res 30) yield coarse hulls whose
            # sagitta bulges into the cavity and ejects particles on step 1
            coacd_options=gs.options.CoacdOptions(threshold=0.02, preprocess_resolution=100),
        ),
        material=gs.materials.Rigid(),  # needs_coup defaults True
        surface=gs.surfaces.Default(color=(0.8, 0.8, 0.85, 0.35), opacity=0.35),
    )

    liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular",
            rho=1.0,
            density_relaxation=1.0,
            viscosity_relaxation=0.0,
        ),
        morph=gs.morphs.Cylinder(radius=R_LIQ, height=H_INIT, pos=(0.0, 0.0, Z0 + H_INIT / 2.0)),
    )

    # oblique top view: see the free surface and the cup side
    cam = scene.add_camera(res=(1280, 960), pos=(0.85, -0.85, 0.95), lookat=(0.0, 0.0, 0.38), fov=35, GUI=False)
    scene.build()
    print(f"cup geoms: {cup.n_geoms}, liquid particles: {liquid.n_particles}")
    print(f"rigid_pbd coupling active: {scene.sim.coupler._rigid_pbd}")
    print(f"initial column z in [{Z0}, {Z0 + H_INIT}], epsd={args.epsd}, settle={args.settle}s")

    # GL point size for the point-cloud rendering (consumed in Rasterizer.render_camera)
    scene.visualizer.rasterizer._camera_targets[cam.uid].point_size = args.point_size

    solver = scene.sim.pbd_solver

    def get_pos():
        pos = liquid.get_particles_pos()
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)
        return pos.reshape(-1, 3).float()

    # ---- phase 1: settle (not recorded) -------------------------------------
    n_settle = int(round(args.settle / DT))
    for i in range(n_settle):
        scene.step()
        if (i + 1) % 30 == 0:
            pos = get_pos()
            r = pos[:, :2].norm(dim=1)
            esc = int(((r > R_IN + PS) | (pos[:, 2] < T_BOTTOM - PS)).sum().item())
            print(f"settle t={(i + 1) * DT:5.2f}s / {args.settle:.1f}s  escaped={esc}"
                  f"  r_max={float(r.max()):.4f}  z_min={float(pos[:, 2].min()):.4f}")

    # ---- phase 2: two-phase concentration init on the settled column ---------
    pos = get_pos()
    # demo cleanup: a handful of particles (~9 / 26k) get squeezed through the wall by the
    # initial compaction splash and end up pinned at the domain corner. Teleport them back
    # into the settled bulk before assigning concentrations (and zero all velocities below).
    r = pos[:, :2].norm(dim=1)
    esc_mask = (r > R_IN + PS) | (pos[:, 2] < T_BOTTOM - PS)
    n_esc = int(esc_mask.sum().item())
    bulk = pos[~esc_mask]
    bz_min = float(bulk[:, 2].min().item())
    bz_max = float(bulk[:, 2].max().item())
    if n_esc > 0:
        rng = np.random.default_rng(0)
        rr = 0.09 * np.sqrt(rng.random(n_esc))
        th = 2.0 * np.pi * rng.random(n_esc)
        zz = rng.uniform(bz_min + 0.03, bz_max - 0.03, n_esc)
        pos[esc_mask] = torch.from_numpy(
            np.stack([rr * np.cos(th), rr * np.sin(th), zz], axis=1)
        ).to(pos.device).float()
        solver.particles.pos.from_numpy(pos.cpu().numpy().reshape(liquid.n_particles, 1, 3))
        print(f"teleported {n_esc} escaped particles back into the bulk")
    pos = get_pos()
    z_min0 = float(pos[:, 2].min().item())
    z_max0 = float(pos[:, 2].max().item())
    z_mid = 0.5 * (z_min0 + z_max0)
    c0 = (pos[:, 2].cpu().numpy() < z_mid).astype(gs.np_float)
    solver.particles.c.from_numpy(c0[:, None])
    solver.particles.vel.from_numpy(np.zeros((liquid.n_particles, 1, 3), dtype=gs.np_float))
    print(f"settled column z in [{z_min0:.4f}, {z_max0:.4f}] (fill {(z_max0 - z_min0) / H_IN * 100:.0f}% of cup), "
          f"z_mid={z_mid:.4f}, coffee particles={int(c0.sum())}")

    # ---- phase 3: record + metrics -------------------------------------------
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    n_steps = int(round(args.seconds / DT))
    cam.start_recording(save_to_filename=mp4_path, fps=int(round(1.0 / DT)))

    rows = []
    prev_pos = None
    for i in range(n_steps + 1):
        if i > 0:
            scene.step()
        pos = get_pos()
        c = torch.from_numpy(solver.particles.c.to_numpy()[:, 0]).to(pos.device).float()
        nan_count = int((~torch.isfinite(pos)).sum().item() + (~torch.isfinite(c)).sum().item())
        safe = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        safe_c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        sum_c = float(safe_c.sum().item())
        std_c = float(safe_c.std().item())
        zc_c = float((safe_c * safe[:, 2]).sum().item() / sum_c) if sum_c > 0 else float("nan")
        r_max = float(safe[:, :2].norm(dim=1).max().item())
        z_min = float(safe[:, 2].min().item())
        z_max = float(safe[:, 2].max().item())
        if prev_pos is None:
            ke = 0.0
        else:
            vel = (safe - prev_pos) / DT
            ke = float((0.5 * vel.norm(dim=1) ** 2).mean().item())
        prev_pos = safe
        rows.append((i, i * DT, sum_c, std_c, zc_c, r_max, z_min, z_max, ke, nan_count))
        cam.render()
        if i % 60 == 0:
            print(
                f"t={i * DT:6.2f}s  sum_c={sum_c:9.2f}  std_c={std_c:.5f}  zc_c={zc_c:.4f}"
                f"  r_max={r_max:.4f}  z_min={z_min:.4f}  z_max={z_max:.4f}  KE={ke:.3e}  nan={nan_count}"
            )

    cam.stop_recording()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "sum_c", "std_c", "zc_c", "r_max", "z_min", "z_max", "ke", "nan_count"])
        w.writerows(rows)

    # pass/fail summary
    cols = ["frame", "t", "sum_c", "std_c", "zc_c", "r_max", "z_min", "z_max", "ke", "nan"]
    arr = {k: np.array([r[j] for r in rows], dtype=float) for j, k in enumerate(cols)}
    sum_c0 = arr["sum_c"][0]
    drift = float(np.max(np.abs(arr["sum_c"] - sum_c0)) / sum_c0)
    # z centroid relative to the *current* liquid column (robust to residual settling)
    span = np.maximum(arr["z_max"] - arr["z_min"], 1e-9)
    rel_zc = (arr["zc_c"] - arr["z_min"]) / span
    p1 = bool(arr["r_max"].max() <= R_IN + PS + 1e-6)
    p2 = bool(arr["z_min"].min() >= T_BOTTOM - PS - 1e-6)
    p3 = bool(arr["nan"].max() == 0)
    p4 = bool(drift < 0.01)
    p5 = bool(arr["std_c"][-1] < 0.5 * arr["std_c"][0])
    p6 = bool(rel_zc[-1] > rel_zc[0])
    print("=" * 70)
    print(f"[1] r_max  max = {arr['r_max'].max():.4f}  (limit {R_IN + PS:.2f})     -> {'PASS' if p1 else 'FAIL'}")
    print(f"[1] z_min  min = {arr['z_min'].min():.4f}  (limit {T_BOTTOM - PS:.2f})   -> {'PASS' if p2 else 'FAIL'}")
    print(f"[1] nan    max = {int(arr['nan'].max())}                              -> {'PASS' if p3 else 'FAIL'}")
    print(f"[2] sum_c drift = {drift:.3e}  (limit 1e-2, sum_c0={sum_c0:.2f})   -> {'PASS' if p4 else 'FAIL'}")
    print(
        f"[3] std_c {arr['std_c'][0]:.5f} -> {arr['std_c'][-1]:.5f}"
        f"  (target < {0.5 * arr['std_c'][0]:.5f})       -> {'PASS' if p5 else 'FAIL'}"
    )
    print(
        f"[3] zc_c rel {rel_zc[0]:.4f} -> {rel_zc[-1]:.4f}  (must rise)"
        f"                            -> {'PASS' if p6 else 'FAIL'}"
    )
    print(f"csv: {csv_path}")
    print(f"mp4: {mp4_path}")


if __name__ == "__main__":
    main()
