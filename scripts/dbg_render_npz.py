"""dbg: reproduce the missing-particle gray band at the mug wall in the mf15 top-down render.

Minimal scene - boundary_cup_shell + boundary_plane, ONE PBD Liquid entity loaded with the settled
mf15_settled.npz coffee positions (42840 particles, c=1), the mug.obj static visual mesh and the table
box. Built, the settled state loaded, then rendered frame by frame so the solver state and the drawn
pixels can be compared frame by frame.

Measurements per frame
  projected edge - image radius of the outermost SOLVER particle projected through the live camera
  drawn edge     - image radius of the outermost coffee-coloured pixel
  band ratio     - fraction of the reported gray band (radius 294..341 px around (x=255, y=685))
                   whose colour matches the coffee colour |RGB-(72,40,8)|_1 < 70
If the projected edge sits at ~319-335 px while the band is wall grey, the particles are skipped at
draw time; if the projected edge itself retreats below the band, the particles moved (solver/timing).

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \\
    "$PY" multiflow/scripts/dbg_render_npz.py
"""

import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import genesis as gs  # noqa: E402

NPZ_PATH = os.path.join(WORKSPACE, "videos", "mf15_settled.npz")
MUG_OBJ = os.path.join(WORKSPACE, "assets", "mug.obj")
OUT_PATH = os.path.join(WORKSPACE, "videos", "mf15_dbgrep_f0.png")

# mf15 constants (mf15_pour_overflow.py)
DT = 1.0 / 60.0
SUBSTEPS = 8
PS = 0.008
R_IN = 0.15
L_MUG = 0.30
T_WALL = 0.016
T_BOTTOM = 0.016
LIP_ROUND = 0.008
Z_FLOOR = T_BOTTOM
Z_TABLE = 0.0
S0 = 0.002

# reported gray band (mf15 top-down frame, 1280 x 960)
BAND_CENTER = (255.0, 685.0)  # (x, y) px
BAND_R_IN, BAND_R_OUT = 294.0, 341.0
COFFEE_RGB = np.array([72.0, 40.0, 8.0])
COFFEE_TOL = 70.0
FRAMES = (0, 1, 2, 5, 10, 30)


def band_coffee_ratio(img):
    """Fraction of band pixels whose colour matches the coffee colour (L1 distance < COFFEE_TOL)."""
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    radius = np.hypot(xx - BAND_CENTER[0], yy - BAND_CENTER[1])
    band = (radius >= BAND_R_IN) & (radius <= BAND_R_OUT)
    delta = np.abs(img.astype(np.int32) - COFFEE_RGB.astype(np.int32)).sum(axis=2)
    n_band = int(band.sum())
    n_coffee = int((band & (delta < COFFEE_TOL)).sum())
    return n_coffee / max(n_band, 1), n_band, n_coffee


def drawn_edge_radius(img, angle_lo, angle_hi):
    delta = np.abs(img.astype(np.int32) - COFFEE_RGB.astype(np.int32)).sum(axis=2)
    ys, xs = np.nonzero(delta < COFFEE_TOL)
    ang = np.degrees(np.arctan2(ys - BAND_CENTER[1], xs - BAND_CENTER[0])) % 360.0
    rad = np.hypot(xs - BAND_CENTER[0], ys - BAND_CENTER[1])
    sel = (ang >= angle_lo) & (ang < angle_hi)
    return float(rad[sel].max()) if sel.any() else float("nan")


def main():
    dat = np.load(NPZ_PATH)
    n_coffee = int(dat["n_coffee"])
    settled = dat["pos"].astype(np.float32)[:n_coffee]
    r_coffee, h_coffee = float(dat["coffee_r"]), float(dat["coffee_h0"])
    print(f"settled npz: coffee {n_coffee} particles, morph cylinder r={r_coffee} h={h_coffee}")

    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
        pbd_options=gs.options.PBDOptions(
            particle_size=PS,
            lower_bound=(-1.0, -0.9, 0.0),
            upper_bound=(1.7, 0.9, 1.1),
            boundary_cup_shell=(0.0, 0.0, Z_FLOOR, R_IN, L_MUG, T_WALL, T_BOTTOM, LIP_ROUND),
            boundary_plane=(Z_TABLE,),
            max_density_solver_iterations=20,
            max_viscosity_solver_iterations=1,
            density_lambda_epsilon=0.1,
            diffusion_coeff=0.01,
            surface_tension_enabled=True,
            st_compliance=1.0,
            st_surface_density_factor=1.0,
            st_distance_enabled=True,
            st_distance_compliance=40.0,
            st_max_surface_neighbors=256,
            velocity_damping=1.0,
            wall_adhesion_enabled=True,
            wall_adhesion_compliance=20.0,
            wall_friction=0.01,
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )

    scene.add_entity(
        morph=gs.morphs.Mesh(file=MUG_OBJ, pos=(0.0, 0.0, 0.0), fixed=True, decimate=False),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.85, 0.85, 0.9, 0.35), opacity=0.35),
    )
    scene.add_entity(
        morph=gs.morphs.Box(size=(2.6, 1.6, 0.04), pos=(0.35, 0.0, Z_TABLE - 0.02), fixed=True),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.55, 0.55, 0.58, 1.0), opacity=1.0),
    )
    coffee = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=1.0,
            density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Cylinder(radius=r_coffee, height=h_coffee, pos=(0.0, 0.0, Z_FLOOR + S0 + 0.5 * h_coffee)),
    )

    cam = scene.add_camera(res=(960, 1280), pos=(0.10, 0.02, 0.85), lookat=(0.10, 0.02, 0.25),
                           up=(0.0, 1.0, 0.0), fov=50, GUI=False)
    scene.build()
    assert coffee.n_particles == n_coffee, f"coffee morph sampled {coffee.n_particles} != npz {n_coffee}"
    solver = scene.sim.ipbf_solver if scene.sim.ipbf_solver.is_active else scene.sim.pbd_solver

    solver.particles.pos.from_numpy(settled)
    solver.particles.vel.from_numpy(np.zeros_like(settled))
    scene.visualizer.rasterizer._camera_targets[cam.uid].point_size = 18.0

    node = scene.visualizer.rasterizer._camera_nodes[cam.uid]
    proj = node.camera.get_projection_matrix()
    view = np.linalg.inv(scene.visualizer.rasterizer._context._scene.get_pose(node))
    width, height = 960, 1280

    def projected_edge(points, angle_lo, angle_hi):
        world_h = np.c_[points, np.ones(len(points))]
        cam_pts = (view @ world_h.T).T[:, :3]
        clip = np.c_[cam_pts, np.ones(len(cam_pts))] @ proj.T
        ndc = clip[:, :2] / clip[:, 3:4]
        px = (ndc[:, 0] + 1.0) * 0.5 * width
        py = (1.0 - ndc[:, 1]) * 0.5 * height
        ang = np.degrees(np.arctan2(py - BAND_CENTER[1], px - BAND_CENTER[0])) % 360.0
        rad = np.hypot(px - BAND_CENTER[0], py - BAND_CENTER[1])
        sel = (ang >= angle_lo) & (ang < angle_hi)
        return float(rad[sel].max()) if sel.any() else float("nan")

    def report(tag, img):
        sectors = np.arange(0.0, 360.0, 10.0)
        pos_now = solver.particles.pos.to_numpy()[:n_coffee, 0, :]
        rend_pos = solver.particles_render.pos.to_numpy()[:n_coffee, 0, :]
        rend_active = solver.particles_render.active.to_numpy()[:n_coffee, 0]
        projected = np.array([projected_edge(pos_now, s, s + 10.0) for s in sectors])
        drawn = np.array([drawn_edge_radius(img, s, s + 10.0) for s in sectors])
        gap = projected - drawn
        ratio, n_band, n_coffee_px = band_coffee_ratio(img)
        world_r = np.hypot(pos_now[:, 0], pos_now[:, 1])
        surface_z = float(np.percentile(pos_now[:, 2], 99))
        print(f"[{tag:>9s}] band coffee ratio {ratio:.3f} | surface z p99 {surface_z:.4f} m | "
              f"solver particles with world r in (0.131,0.145]: "
              f"{int(((world_r > 0.131) & (world_r <= 0.145)).sum()):5d} | render field active "
              f"{int(rend_active.sum())}/{n_coffee}, parked {int((np.abs(rend_pos).max(axis=1) > 1.0e3).sum())}")
        print(f"            projected particle edge: mean {np.nanmean(projected):5.1f} px "
              f"(min {np.nanmin(projected):.0f} max {np.nanmax(projected):.0f}) | drawn coffee edge: "
              f"mean {np.nanmean(drawn):5.1f} px (min {np.nanmin(drawn):.0f} max {np.nanmax(drawn):.0f}) | "
              f"gap mean {np.nanmean(gap):5.1f} px max {np.nanmax(gap):5.1f} px")
        return ratio

    print("[1] render f0 with the settled state loaded and no simulation step")
    rgb = np.asarray(cam.render(rgb=True)[0])
    report("f0 no-step", rgb)
    Image.fromarray(rgb).save(OUT_PATH)
    print(f"saved {OUT_PATH}")

    pos_f0 = solver.particles.pos.to_numpy()[:n_coffee, 0, :]
    r_f0 = np.hypot(pos_f0[:, 0], pos_f0[:, 1])
    shell = (r_f0 > 0.131) & (r_f0 <= 0.145)
    print(f"[1b] f0 wall-shell particles: {int(shell.sum())}")

    done = 0
    for target in FRAMES[1:]:
        while done < target:
            scene.step()
            done += 1
        rgb = np.asarray(cam.render(rgb=True)[0])
        report(f"f{target}", rgb)
        Image.fromarray(rgb).save(OUT_PATH.replace(".png", f"_f{target}.png"))
        if target == 1:
            pos_f1 = solver.particles.pos.to_numpy()[:n_coffee, 0, :]
            r_f1 = np.hypot(pos_f1[:, 0], pos_f1[:, 1])
            dr = r_f1[shell] - r_f0[shell]
            print(f"[1c] where the f0 wall-shell particles went after 1 step: inward(<-4mm) "
                  f"{int((dr < -0.004).sum())}, stayed {int((np.abs(dr) <= 0.004).sum())}, "
                  f"outward(>4mm) {int((dr > 0.004).sum())}; dr mean {dr.mean() * 1000:.2f} mm")
            print(f"[1c] their z: {pos_f0[shell, 2].mean():.4f} -> {pos_f1[shell, 2].mean():.4f} m; "
                  f"their r: {r_f0[shell].mean():.4f} -> {r_f1[shell].mean():.4f} m")
            edges = np.arange(0.10, 0.1701, 0.005)
            h0, _ = np.histogram(r_f0, edges)
            h1, _ = np.histogram(r_f1, edges)
            print("[1c] radial histogram (particles per 5 mm bin, r from 0.100):")
            print("      f0:", " ".join(f"{v:5d}" for v in h0))
            print("      f1:", " ".join(f"{v:5d}" for v in h1))
            print(f"[1c] particles outside the mug inner radius r>{R_IN}: {int((r_f1 > R_IN).sum())} "
                  f"(max r {r_f1.max():.4f})")

    print("[2] verdicts: a retreating PROJECTED edge means the particles moved (solver/timing); a stable "
          "projected edge with a retreating DRAWN edge means they are skipped at draw time (renderer)")


if __name__ == "__main__":
    main()
