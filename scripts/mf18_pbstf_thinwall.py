"""MF-18 plan section 6-C, phase C1: thin-wall unit tests for the ported PBSTF path.

Recipe fixed by phase B (do not retune): ps=0.002, support=0.006, rho0=1000,
h=1/960 (dt=1/240 x 4 substeps), 20 solver iterations, topology_rebuild_interval=2,
PCA normals on, material rho=1000, density_compliance=375000,
surface_tension_compliance=0.0048, surface/interior distance compliance 40/180,
regular sampler, no adhesion/friction.

Tests (all zero-gravity except T3/T4):
  t1  : static 4mm box wall at x=0, one 16mm water cube per side (8^3=512 each),
        cube faces 1mm from the wall faces. Per-side interior density quantiles,
        offline cross-wall neighbor-pair census vs density uplift.
  t1v : t1 + surface/interior viscosity 0.05, left block gets +x 0.05 m/s initial
        velocity (via entity.set_particles_vel). Watches whether the right block
        gets dragged through the wall by the (unfiltered) viscosity link.
  t2  : same geometry, left block only; wall translates +x by 10mm over 1s, then
        holds 1s. Tunneling / drag / leftover-particle checks.
  t3  : watertight cylindrical cup (R_in=20mm, wall 4mm, H=40mm, closed bottom)
        as PBSTFMeshStaticColliderOptions(sdf_res=150) on a box table, filled with
        a 32mm x 20mm water cylinder under gravity -9.8. Leak / tunneling /
        wall-layer density / free-surface flatness checks.

  T4 (gravity -9.8, pressurized; fixes C1's zero cross-wall-pair evidence gap).
  Open blocks bead up and detach from the wall at this scale (Bond < 1, verified with
  12/16/24mm open blocks), so each side is a confined channel: thin wall at x=0,
  outer wall at |x|=13..15mm, end caps at y=+/-(10..12)mm, water column pressed
  against the thin wall by hydrostatic pressure. Wall thickness is 2mm (not 4mm):
  collider contact anchors particles at surface + particle_radius (1mm, hard-coded),
  so a 4mm wall puts settled cross-wall centers exactly at the 6mm support
  truncation -> zero pairs by construction; 2mm gives 4mm < 6mm.
  t4ctrl: left channel only, 12mm water column, visc=0.
  t4a :   both channels with 12mm columns, visc=0.
          Per-frame cross-wall neighbor-pair census (must be > 0), wall-layer
          density vs t4ctrl, zero tunneling.
  t4b :   t4a + surface/interior viscosity 0.05. Quantifies the unfiltered
          _task_viscosity link across the wall (KE/com trajectory vs t4a).
  t4c :   left 24mm column + right 12mm column, visc=0.05, 2.5s.
          Asymmetric hydrostatic head: cross-wall drag / tunneling check.

Outputs (multiflow/videos/, prefix mf18_pbstf_tw_): per-frame CSV, final npz,
mp4 (30fps, 960x1280, point particles, fixed camera).
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import genesis as gs  # noqa: E402
from genesis.utils.misc import qd_to_numpy  # noqa: E402

assert "multiflow" in gs.__file__, gs.__file__

VIDEOS_DIR = os.path.join(WORKSPACE, "videos")
ASSETS_DIR = os.path.join(WORKSPACE, "assets")
OUT_PREFIX = "mf18_pbstf_tw_"

DT = 1.0 / 240.0
SUBSTEPS = 4
PS = 0.002
SUPPORT = 0.006
RHO0 = 1000.0
WALL_HALF = 0.002  # 4mm wall at x=0

CUBE_L = 0.016  # 8^3 = 512 particles per side
CUBE_HALF = 0.008
CUBE_CX = 0.011  # |x| of cube centers: face at 0.003, 1mm from wall face at 0.002
CUBE_CZ = 0.030

PARTICLE_RADIUS = 0.5 * PS


def cubic_w(d, h=SUPPORT):
    """Reference cubic-spline kernel (same convention as pbstf_solver.cubic_kernel)."""
    q = d / h
    coef = 8.0 / (np.pi * h**3)
    w = np.zeros_like(d)
    inner = q < 0.5
    outer = (q >= 0.5) & (q < 1.0)
    w[inner] = coef * (6.0 * q[inner] ** 2 * (q[inner] - 1.0) + 1.0)
    w[outer] = 2.0 * coef * (1.0 - q[outer]) ** 3
    return w


def liquid_material(visc=0.0):
    return gs.materials.PBSTF.Liquid(
        rho=RHO0,
        density_compliance=375000.0,
        surface_tension_compliance=0.0048,
        surface_distance_compliance=40.0,
        interior_distance_compliance=180.0,
        surface_viscosity=visc,
        interior_viscosity=visc,
        is_collider_adhesion_friction_enabled=False,
        sampler="regular",
    )


def wall_box_options(pos_x=0.0):
    return gs.options.PBSTFBoxStaticColliderOptions(
        pos=(pos_x, 0.0, CUBE_CZ),
        quat=(1.0, 0.0, 0.0, 0.0),
        lower=(-WALL_HALF, -0.020, -0.020),
        upper=(WALL_HALF, 0.020, 0.020),
    )


def make_cup_obj(path):
    """Watertight cylindrical cup shell: R_out=22mm, R_in=20mm, H=40mm, 4mm bottom."""
    import trimesh

    section = np.array(
        [
            [0.0, 0.0],
            [0.022, 0.0],
            [0.022, 0.040],
            [0.020, 0.040],
            [0.020, 0.004],
            [0.0, 0.004],
        ]
    )
    mesh = trimesh.creation.revolve(section, sections=128)
    mesh.export(path)
    reloaded = trimesh.load(path)
    print(
        f"[cup] obj -> {path}: {len(reloaded.vertices)}v/{len(reloaded.faces)}f "
        f"watertight={reloaded.is_watertight} volume={reloaded.volume:.6e}"
    )
    if not reloaded.is_watertight:
        gs.raise_exception("generated cup mesh is not watertight")
    return path


class Recorder:
    def __init__(self, scene, solver, entities):
        self.scene = scene
        self.solver = solver
        self.entities = entities

    def snapshot_user_order(self):
        """Per-particle state in user (entity) order. substep end state: particles.pos is
        fresh (copied back before the final XSPH); vel/density/on_surface are read from the
        reordered view (fresh after the final reorder) and mapped back via reordered_idx."""
        sv = self.solver
        idx = qd_to_numpy(sv.particles_ng.reordered_idx, transpose=True)[0]  # user i -> reordered slot
        pos = qd_to_numpy(sv.particles.pos, transpose=True)[0].copy()
        vel_r = qd_to_numpy(sv.particles_reordered.vel, transpose=True)[0]
        rho_r = qd_to_numpy(sv.particles_reordered.density, transpose=True)[0]
        surf_r = qd_to_numpy(sv.on_surface, transpose=True)[0].astype(bool)
        return pos, vel_r[idx], rho_r[idx], surf_r[idx]

    def sync_t0(self):
        """Work around the stale reordered views left by scene.build()'s compile step+reset
        (phase-B finding): one explicit reorder + topology rebuild; does not move particles."""
        self.solver._kernel_reorder_particles(0)
        self.solver._rebuild_topology(0)


def run_test(args):
    test = args.test
    dur = args.dur

    gs.init(backend=gs.gpu, precision="32", seed=0)

    # ---------------- scene ----------------
    if test in ("t1", "t1v", "t2"):
        bounds = ((-0.04, -0.03, 0.0), (0.05, 0.03, 0.06))
        gravity = (0.0, 0.0, 0.0)
        colliders = [wall_box_options()]
        cam_pos, cam_lookat = (0.0, -0.095, CUBE_CZ), (0.0, 0.0, CUBE_CZ)
    else:
        bounds = ((-0.045, -0.045, -0.006), (0.045, 0.045, 0.075))
        gravity = (0.0, 0.0, -9.8)
        colliders = []  # filled below
        cam_pos, cam_lookat = (0.085, -0.085, 0.080), (0.0, 0.0, 0.015)

    visc = 0.05 if test == "t1v" else 0.0

    if test == "t3":
        os.makedirs(ASSETS_DIR, exist_ok=True)
        cup_path = make_cup_obj(os.path.join(ASSETS_DIR, OUT_PREFIX + "cup.obj"))
        colliders = [
            # load-bearing table first (support priority), cup second
            gs.options.PBSTFBoxStaticColliderOptions(
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
                lower=(-0.045, -0.045, -0.004),
                upper=(0.045, 0.045, 0.0),
            ),
            gs.options.PBSTFMeshStaticColliderOptions(file=cup_path, scale=1.0, sdf_res=150),
        ]

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=gravity),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=PS,
            lower_bound=bounds[0],
            upper_bound=bounds[1],
            max_solver_iterations=20,
            topology_rebuild_interval=2,
            static_colliders=colliders,
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )

    entities = []
    if test in ("t1", "t1v"):
        left = scene.add_entity(
            material=liquid_material(visc),
            morph=gs.morphs.Box(
                lower=(-CUBE_CX - CUBE_HALF, -CUBE_HALF, CUBE_CZ - CUBE_HALF),
                upper=(-CUBE_CX + CUBE_HALF, CUBE_HALF, CUBE_CZ + CUBE_HALF),
            ),
            surface=gs.surfaces.Default(color=(0.35, 0.55, 0.95, 1.0), opacity=1.0),
        )
        right = scene.add_entity(
            material=liquid_material(visc),
            morph=gs.morphs.Box(
                lower=(CUBE_CX - CUBE_HALF, -CUBE_HALF, CUBE_CZ - CUBE_HALF),
                upper=(CUBE_CX + CUBE_HALF, CUBE_HALF, CUBE_CZ + CUBE_HALF),
            ),
            surface=gs.surfaces.Default(color=(0.95, 0.45, 0.30, 1.0), opacity=1.0),
        )
        entities = [left, right]
    elif test == "t2":
        left = scene.add_entity(
            material=liquid_material(visc),
            morph=gs.morphs.Box(
                lower=(-CUBE_CX - CUBE_HALF, -CUBE_HALF, CUBE_CZ - CUBE_HALF),
                upper=(-CUBE_CX + CUBE_HALF, CUBE_HALF, CUBE_CZ + CUBE_HALF),
            ),
            surface=gs.surfaces.Default(color=(0.35, 0.55, 0.95, 1.0), opacity=1.0),
        )
        entities = [left]
    else:
        water = scene.add_entity(
            material=liquid_material(0.0),
            morph=gs.morphs.Cylinder(radius=0.016, height=0.020, pos=(0.0, 0.0, 0.014)),
            surface=gs.surfaces.Default(color=(0.35, 0.55, 0.95, 1.0), opacity=1.0),
        )
        entities = [water]

    cam = scene.add_camera(res=(960, 1280), pos=cam_pos, lookat=cam_lookat, fov=40, GUI=False)

    t_build0 = time.perf_counter()
    scene.build()
    build_s = time.perf_counter() - t_build0

    solver = scene.sim.pbstf_solver
    rec = Recorder(scene, solver, entities)
    n_total = sum(e.n_particles for e in entities)
    mass = solver._default_mass
    print(
        f"THINWALL [{test}]: n={n_total} ps={PS} h={solver._substep_dt:.6e} iters=20 "
        f"gravity={solver._gravity.to_numpy()} mass={mass:.10e} build={build_s:.1f}s"
    )
    if test == "t3":
        from genesis.utils.misc import get_gsd_cache_dir

        print(f"[cup] sdf cache dir: {get_gsd_cache_dir()} (build wall time above includes first SDF build)")

    rec.sync_t0()
    pos0, vel0, rho0, surf0 = rec.snapshot_user_order()
    print(f"[t0] on_surface={surf0.sum()}/{len(surf0)} rho50={np.median(rho0[~surf0])/RHO0:.4f}")

    if test == "t1v":
        entities[0].set_particles_vel((0.05, 0.0, 0.0))
        print("[t1v] left block initial velocity +x 0.05 m/s applied via set_particles_vel")

    # ---------------- per-frame metric ----------------
    groups = []
    if test in ("t1", "t1v"):
        groups = [(entities[0].particle_start, entities[0].particle_end), (entities[1].particle_start, entities[1].particle_end)]
    elif test == "t2":
        groups = [(entities[0].particle_start, entities[0].particle_end)]

    def frame_metrics(t, pos, vel, rho, surf, wall_x, ms):
        nan = int((~np.isfinite(pos).all(axis=1) | ~np.isfinite(vel).all(axis=1)).sum())
        ke = float(0.5 * mass * np.sum(vel * vel))
        vmax = float(np.linalg.norm(vel, axis=1).max())
        row = dict(t=t, nan=nan, ke=ke, vmax=vmax, n_surf=int(surf.sum()), ms=ms)
        if test in ("t1", "t1v", "t2"):
            for gi, (s, e) in enumerate(groups):
                mask = np.zeros(len(pos), dtype=bool)
                mask[s:e] = True
                interior = mask & ~surf
                com = pos[mask].mean(axis=0)
                for a, ax in enumerate("xyz"):
                    row[f"com{gi}_{ax}"] = float(com[a])
                rr = rho[interior] / RHO0
                for q, v in zip(("p25", "p50", "p75"), np.percentile(rr, (25, 50, 75))):
                    row[f"rho{gi}_{q}"] = float(v)
                row[f"ke{gi}"] = float(0.5 * mass * np.sum(vel[mask] ** 2))
            if test == "t2":
                row["wall_x"] = wall_x
                row["n_beyond"] = int((pos[:, 0] > wall_x + WALL_HALF).sum())
                row["xmin"] = float(pos[:, 0].min())
                row["xmax"] = float(pos[:, 0].max())
        else:
            r_xy = np.linalg.norm(pos[:, :2], axis=1)
            row["com_z"] = float(pos[:, 2].mean())
            # leak: outside the outer wall, or below the table top
            row["n_leak"] = int(((r_xy > 0.022) | (pos[:, 2] < 0.0)).sum())
            wall_layer = ((r_xy > 0.017) & (pos[:, 2] > 0.004)) | (pos[:, 2] < 0.007)
            deep = ~wall_layer & ~surf
            wall_int = wall_layer & ~surf
            for name, msk in (("wall", wall_int), ("deep", deep)):
                rr = rho[msk] / RHO0
                if len(rr):
                    p25, p50, p75 = np.percentile(rr, (25, 50, 75))
                else:
                    p25 = p50 = p75 = float("nan")
                row[f"rho_{name}_p25"], row[f"rho_{name}_p50"], row[f"rho_{name}_p75"] = float(p25), float(p50), float(p75)
            top = pos[:, 2] >= np.percentile(pos[:, 2], 90)
            row["ztop_p95_p5"] = float(np.percentile(pos[top, 2], 95) - np.percentile(pos[top, 2], 5))
        return row

    # ---------------- loop ----------------
    mp4 = os.path.join(VIDEOS_DIR, f"{OUT_PREFIX}{test}.mp4")
    csv_path = os.path.join(VIDEOS_DIR, f"{OUT_PREFIX}{test}_metrics.csv")
    if not args.no_video:
        cam.start_recording(save_to_filename=mp4, fps=30)

    n_frames = int(round(dur / DT))
    move_frames = int(round(1.0 / DT))  # t2: 1s translation, then hold
    rows = [frame_metrics(0.0, pos0, vel0, rho0, surf0, 0.0, 0.0)]
    t_start = time.perf_counter()
    abort = ""
    for i in range(1, n_frames + 1):
        wall_x = 0.0
        if test == "t2":
            progress = min(i / move_frames, 1.0)
            wall_x = progress * 0.010
            solver.set_static_colliders_pose(pos=(wall_x, 0.0, CUBE_CZ), quat=(1.0, 0.0, 0.0, 0.0), colliders_idx=0)
        t0 = time.perf_counter()
        try:
            scene.step()
        except Exception as e:
            abort = f"{type(e).__name__}: {e}"
            print(f"[abort] frame {i}: {abort}")
            break
        ms = (time.perf_counter() - t0) * 1e3
        pos, vel, rho, surf = rec.snapshot_user_order()
        rows.append(frame_metrics(i * DT, pos, vel, rho, surf, wall_x, ms))
        if i % 60 == 0 or i == n_frames:
            r = rows[-1]
            extra = ""
            if test in ("t1", "t1v"):
                extra = f"rhoL50={r['rho0_p50']:.4f} rhoR50={r['rho1_p50']:.4f} comLx={r['com0_x']:+.4f} comRx={r['com1_x']:+.4f}"
            elif test == "t2":
                extra = f"wall_x={r['wall_x']:.4f} n_beyond={r['n_beyond']} xmax={r['xmax']:+.4f} rho50={r['rho0_p50']:.4f}"
            else:
                extra = f"com_z={r['com_z']:.4f} leak={r['n_leak']} rhoW50={r['rho_wall_p50']:.4f} rhoD50={r['rho_deep_p50']:.4f} ztop={r['ztop_p95_p5']:.4f}"
            print(f"t={r['t']:5.2f}s nan={r['nan']} KE={r['ke']:.3e} vmax={r['vmax']:.3f} n_surf={r['n_surf']} {extra}")

    if not args.no_video:
        cam.stop_recording()
    wall = time.perf_counter() - t_start
    n_done = len(rows) - 1
    print(f"[perf] {n_done} frames in {wall:.1f}s -> {1e3*wall/max(n_done,1):.2f} ms/frame (incl. render+IO)")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------- final npz + offline analysis ----------------
    final_npz = os.path.join(VIDEOS_DIR, f"{OUT_PREFIX}{test}_final.npz")
    np.savez(final_npz, pos=pos, vel=vel, density=rho, on_surface=surf, abort=abort)
    print(f"final npz: {final_npz}")

    if test in ("t1", "t1v") and n_done > 0:
        # Offline cross-wall neighbor census: pairs with one particle clearly left
        # (x < -2mm) and one clearly right (x > +2mm) within the support radius.
        pl, pr = pos[pos[:, 0] < -WALL_HALF], pos[pos[:, 0] > WALL_HALF]
        n_pairs = 0
        uplift = 0.0
        if len(pl) and len(pr):
            d2 = ((pl[:, None, :] - pr[None, :, :]) ** 2).sum(axis=2)
            near = (d2 < SUPPORT**2)
            n_pairs = int(near.sum())
            if n_pairs:
                dij = np.sqrt(d2[near])
                uplift = float(mass * cubic_w(dij).sum() / n_pairs)  # mean density contribution per pair
        print(f"[xwall] geometric cross-wall pairs within support: {n_pairs}")
        print(f"[xwall] mean hypothetical per-pair density contribution (if unfiltered): {uplift:.2f} kg/m^3")
        # wall-adjacent layer vs deep interior density, per side
        for gi, (s, e) in enumerate(groups):
            p_g = pos[s:e]
            rho_g = rho[s:e]
            surf_g = surf[s:e]
            dist_wall = np.abs(np.abs(p_g[:, 0]) - WALL_HALF)  # distance to the side's wall face
            layer = (dist_wall <= 2 * PS) & ~surf_g
            deep = (dist_wall > 2 * PS) & ~surf_g
            for name, msk in (("wall2ps", layer), ("deep", deep)):
                rr = rho_g[msk] / RHO0
                if len(rr):
                    print(
                        f"[xwall] side{gi} {name}: n={msk.sum()} rho p25/50/75 = "
                        f"{np.percentile(rr,25):.4f}/{np.percentile(rr,50):.4f}/{np.percentile(rr,75):.4f}"
                    )
    print(f"csv: {csv_path}")
    if not args.no_video:
        print(f"mp4: {mp4}")


# ----------------------------------------------------------------------------
# T4: pressurized thin wall (gravity, water adhered on BOTH sides of the wall)
# ----------------------------------------------------------------------------

# Compartment design (open blocks bead up and detach from the wall at this scale —
# Bond number < 1, verified with 12/16/24mm open blocks: 12/16mm detach, 24mm splashes
# over the wall top). Each side gets a channel: thin wall at x=0, outer wall at
# |x|=13..15mm, end caps at y=±(10..12mm), all 25mm tall, on a desktop.
#
# Wall thickness is 2mm (same as the T3 cup shell), NOT the 4mm originally planned:
# query_static_collider_contact anchors particles at surface + particle_radius (1mm,
# hard-coded), so settled centers rest 1mm off each face -> cross-wall distance
# = 2*1mm + thickness. A 4mm wall gives exactly 6mm = support truncation
# (cubic_w(1)=0) -> zero cross-wall pairs by construction; 2mm gives 4mm < 6mm.
# The settled 1mm standoff also sits exactly on static_collider_separates'
# particle_radius threshold, so the separator filter genuinely engages.
# Water column pressed into the channel -> cross-wall neighbor pairs > 0.
T4_WALL_HALF = 0.001  # 2mm thin wall
T4_WALL_LO = (-0.001, -0.020, 0.0)
T4_WALL_HI = (0.001, 0.020, 0.025)
T4_DESK_LO = (-0.030, -0.030, -0.004)
T4_DESK_HI = (0.030, 0.030, 0.0)
T4_OUT_X0 = 0.013  # outer wall inner face
T4_OUT_X1 = 0.015  # outer wall outer face
T4_CAP_Y0 = 0.010  # end cap inner face
T4_CAP_Y1 = 0.012
# water: 12mm wide (x 0.001..0.013, faces 0.5mm into both walls), 16mm deep (y ±8mm)
T4_WAT_X0 = 0.0005
T4_WAT_X1 = 0.0125
T4_WAT_Y = 0.008
T4_Z_LOW = 0.012  # 6x8x6 = 288 particles per side
T4_Z_HIGH = 0.024  # 6x8x12 = 576 particles (t4c left column)
T4_BOUNDS = ((-0.032, -0.032, -0.006), (0.032, 0.032, 0.045))


def run_t4(args):
    """t4ctrl: left low block only, visc=0 (single-side adhered baseline).
    t4a:    symmetric low blocks on both sides, visc=0.
    t4b:    same as t4a but surface/interior viscosity 0.05.
    t4c:    left tall column (24mm) + right low block, visc=0.05, 2.5s.
    Hard requirement: per-frame cross-wall neighbor pairs must be > 0.
    """
    test = args.test
    double = test in ("t4a", "t4b", "t4c")
    visc = 0.05 if test in ("t4b", "t4c") else 0.0
    dur = args.dur if args.dur > 0 else (2.5 if test == "t4c" else 2.0)
    z_left = T4_Z_HIGH if test == "t4c" else T4_Z_LOW

    gs.init(backend=gs.gpu, precision="32", seed=0)

    colliders = [
        # desktop first (load bearing, keeps its exclusion region), thin wall second,
        # then outer walls and end caps (they only touch at edges, no volume overlaps)
        gs.options.PBSTFBoxStaticColliderOptions(
            pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0),
            lower=T4_DESK_LO, upper=T4_DESK_HI,
        ),
        gs.options.PBSTFBoxStaticColliderOptions(
            pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0),
            lower=T4_WALL_LO, upper=T4_WALL_HI,
        ),
    ]
    for sx in (-1.0, 1.0):
        # outer wall of this side's channel
        xa, xb = sorted((sx * T4_OUT_X0, sx * T4_OUT_X1))
        colliders.append(
            gs.options.PBSTFBoxStaticColliderOptions(
                pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0),
                lower=(xa, -0.020, 0.0), upper=(xb, 0.020, 0.025),
            )
        )
        # end caps at y = +/- (10..12mm), spanning this side's channel (x 0.001..0.015)
        for sy in (-1.0, 1.0):
            ya, yb = sorted((sy * T4_CAP_Y0, sy * T4_CAP_Y1))
            colliders.append(
                gs.options.PBSTFBoxStaticColliderOptions(
                    pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0),
                    lower=(min(sx * T4_WALL_HALF, xa), ya, 0.0),
                    upper=(max(sx * T4_WALL_HALF, xb), yb, 0.025),
                )
            )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, -9.8)),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=PS,
            lower_bound=T4_BOUNDS[0],
            upper_bound=T4_BOUNDS[1],
            max_solver_iterations=20,
            topology_rebuild_interval=2,
            static_colliders=colliders,
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )

    left = scene.add_entity(
        material=liquid_material(visc),
        morph=gs.morphs.Box(lower=(-T4_WAT_X1, -T4_WAT_Y, 0.0), upper=(-T4_WAT_X0, T4_WAT_Y, z_left)),
        surface=gs.surfaces.Default(color=(0.35, 0.55, 0.95, 1.0), opacity=1.0),
    )
    entities = [left]
    if double:
        right = scene.add_entity(
            material=liquid_material(visc),
            morph=gs.morphs.Box(lower=(T4_WAT_X0, -T4_WAT_Y, 0.0), upper=(T4_WAT_X1, T4_WAT_Y, T4_Z_LOW)),
            surface=gs.surfaces.Default(color=(0.95, 0.45, 0.30, 1.0), opacity=1.0),
        )
        entities.append(right)

    # visualization-only rigid meshes so the video shows the wall, outer walls, desktop
    scene.add_entity(
        morph=gs.morphs.Box(size=(0.002, 0.040, 0.025), pos=(0.0, 0.0, 0.0125), collision=False, fixed=True),
        material=gs.materials.Rigid(needs_coup=False),
        surface=gs.surfaces.Default(color=(0.55, 0.55, 0.58, 1.0), opacity=0.35),
    )
    for sx in (-1.0, 1.0):
        scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.002, 0.040, 0.025), pos=(sx * 0.014, 0.0, 0.0125), collision=False, fixed=True
            ),
            material=gs.materials.Rigid(needs_coup=False),
            surface=gs.surfaces.Default(color=(0.55, 0.55, 0.58, 1.0), opacity=0.35),
        )
    scene.add_entity(
        morph=gs.morphs.Box(size=(0.060, 0.060, 0.004), pos=(0.0, 0.0, -0.002), collision=False, fixed=True),
        material=gs.materials.Rigid(needs_coup=False),
        surface=gs.surfaces.Default(color=(0.40, 0.40, 0.42, 1.0), opacity=0.5),
    )

    cam = scene.add_camera(res=(960, 1280), pos=(0.0, -0.12, 0.015), lookat=(0.0, 0.0, 0.012), fov=40, GUI=False)

    t_build0 = time.perf_counter()
    scene.build()
    build_s = time.perf_counter() - t_build0

    solver = scene.sim.pbstf_solver
    rec = Recorder(scene, solver, entities)
    n_total = sum(e.n_particles for e in entities)
    mass = solver._default_mass
    groups = [(e.particle_start, e.particle_end) for e in entities]
    print(
        f"THINWALL [{test}]: n={n_total} groups={[e - s for s, e in groups]} ps={PS} "
        f"h={solver._substep_dt:.6e} iters=20 visc={visc} z_left={z_left} build={build_s:.1f}s"
    )

    rec.sync_t0()
    pos0, vel0, rho0, surf0 = rec.snapshot_user_order()
    print(f"[t0] on_surface={surf0.sum()}/{len(surf0)} rho50={np.median(rho0[~surf0])/RHO0:.4f}")

    def cross_wall_pairs(pos):
        """Pairs (one from each entity) on opposite sides of the wall within the support."""
        if not double:
            return 0
        s0, e0 = groups[0]
        s1, e1 = groups[1]
        a = pos[s0:e0][pos[s0:e0, 0] < -T4_WALL_HALF]
        b = pos[s1:e1][pos[s1:e1, 0] > T4_WALL_HALF]
        if not len(a) or not len(b):
            return 0
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        return int((d2 < SUPPORT**2).sum())

    def frame_metrics(t, pos, vel, rho, surf, ms):
        nan = int((~np.isfinite(pos).all(axis=1) | ~np.isfinite(vel).all(axis=1)).sum())
        ke = float(0.5 * mass * np.sum(vel * vel))
        vmax = float(np.linalg.norm(vel, axis=1).max())
        row = dict(t=t, nan=nan, ke=ke, vmax=vmax, n_surf=int(surf.sum()), ms=ms)
        # leak: off the desktop or sunk below its top surface
        row["n_leak"] = int(
            ((np.abs(pos[:, 0]) > 0.030) | (np.abs(pos[:, 1]) > 0.030) | (pos[:, 2] < -0.0005)).sum()
        )
        row["xwall_pairs"] = cross_wall_pairs(pos)
        for gi, (s, e) in enumerate(groups):
            mask = np.zeros(len(pos), dtype=bool)
            mask[s:e] = True
            interior = mask & ~surf
            com = pos[mask].mean(axis=0)
            for a, ax in enumerate("xyz"):
                row[f"com{gi}_{ax}"] = float(com[a])
            rr = rho[interior] / RHO0
            for q, v in zip(("p25", "p50", "p75"), np.percentile(rr, (25, 50, 75))):
                row[f"rho{gi}_{q}"] = float(v)
            row[f"ke{gi}"] = float(0.5 * mass * np.sum(vel[mask] ** 2))
            # wall-adhered layer (within 2ps of this side's wall face) density
            dist_wall = np.abs(np.abs(pos[s:e, 0]) - T4_WALL_HALF)
            wall_mask = (dist_wall <= 2 * PS) & ~surf[s:e]
            rw = rho[s:e][wall_mask] / RHO0
            row[f"rhoW{gi}_p50"] = float(np.median(rw)) if len(rw) else float("nan")
            row[f"n_wall{gi}"] = int(wall_mask.sum())
            if double:
                # tunneling: this entity's particle found on the opposite side of the wall
                # (left entity gi=0 sits at x<0 -> tunneled when x > +T4_WALL_HALF)
                if gi == 0:
                    row[f"tun{gi}"] = int((pos[s:e, 0] > T4_WALL_HALF).sum())
                else:
                    row[f"tun{gi}"] = int((pos[s:e, 0] < -T4_WALL_HALF).sum())
        return row

    mp4 = os.path.join(VIDEOS_DIR, f"{OUT_PREFIX}{test}.mp4")
    csv_path = os.path.join(VIDEOS_DIR, f"{OUT_PREFIX}{test}_metrics.csv")
    if not args.no_video:
        cam.start_recording(save_to_filename=mp4, fps=30)

    n_frames = int(round(dur / DT))
    rows = [frame_metrics(0.0, pos0, vel0, rho0, surf0, 0.0)]
    t_start = time.perf_counter()
    abort = ""
    for i in range(1, n_frames + 1):
        t0 = time.perf_counter()
        try:
            scene.step()
        except Exception as e:
            abort = f"{type(e).__name__}: {e}"
            print(f"[abort] frame {i}: {abort}")
            break
        ms = (time.perf_counter() - t0) * 1e3
        pos, vel, rho, surf = rec.snapshot_user_order()
        rows.append(frame_metrics(i * DT, pos, vel, rho, surf, ms))
        if i % 60 == 0 or i == n_frames:
            r = rows[-1]
            extra = f"xwall={r['xwall_pairs']} rhoL50={r['rho0_p50']:.4f} rhoWL50={r['rhoW0_p50']:.4f} comLz={r['com0_z']:.4f}"
            if double:
                extra += (
                    f" rhoR50={r['rho1_p50']:.4f} rhoWR50={r['rhoW1_p50']:.4f}"
                    f" tun={r['tun0']}/{r['tun1']} keR={r['ke1']:.3e}"
                )
            print(f"t={r['t']:5.2f}s nan={r['nan']} KE={r['ke']:.3e} vmax={r['vmax']:.3f} leak={r['n_leak']} {extra}")

    if not args.no_video:
        cam.stop_recording()
    wall = time.perf_counter() - t_start
    n_done = len(rows) - 1
    print(f"[perf] {n_done} frames in {wall:.1f}s -> {1e3*wall/max(n_done,1):.2f} ms/frame (incl. render+IO)")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    final_npz = os.path.join(VIDEOS_DIR, f"{OUT_PREFIX}{test}_final.npz")
    np.savez(final_npz, pos=pos, vel=vel, density=rho, on_surface=surf, abort=abort)
    print(f"final npz: {final_npz}")

    # ---------------- final offline analysis ----------------
    xwall_series = np.array([r["xwall_pairs"] for r in rows])
    first_nz = next((r["t"] for r in rows if r["xwall_pairs"] > 0), None)
    print(
        f"[xwall] per-frame pairs: max={xwall_series.max()} mean={xwall_series.mean():.1f} "
        f"last={xwall_series[-1]} first_nz_t={first_nz}"
    )
    pl, pr = pos[pos[:, 0] < -T4_WALL_HALF], pos[pos[:, 0] > T4_WALL_HALF]
    n_pairs = 0
    uplift = 0.0
    if len(pl) and len(pr):
        d2 = ((pl[:, None, :] - pr[None, :, :]) ** 2).sum(axis=2)
        near = d2 < SUPPORT**2
        n_pairs = int(near.sum())
        if n_pairs:
            dij = np.sqrt(d2[near])
            uplift = float(mass * cubic_w(dij).sum() / n_pairs)
    print(f"[xwall] final-state geometric cross-wall pairs within support: {n_pairs}")
    print(f"[xwall] mean hypothetical per-pair density contribution (if unfiltered): {uplift:.2f} kg/m^3")
    for gi, (s, e) in enumerate(groups):
        p_g = pos[s:e]
        rho_g = rho[s:e]
        surf_g = surf[s:e]
        dist_wall = np.abs(np.abs(p_g[:, 0]) - T4_WALL_HALF)
        layer = (dist_wall <= 2 * PS) & ~surf_g
        deep = (np.abs(p_g[:, 0]) > 0.008) & ~surf_g
        for name, msk in (("wall2ps", layer), ("deep", deep)):
            rr = rho_g[msk] / RHO0
            if len(rr):
                print(
                    f"[xwall] side{gi} {name}: n={msk.sum()} rho p25/50/75 = "
                    f"{np.percentile(rr,25):.4f}/{np.percentile(rr,50):.4f}/{np.percentile(rr,75):.4f}"
                )
    summary = dict(
        test=test, visc=visc, dur=dur, n_particles=int(n_total), double=double, z_left=z_left,
        aborted=abort,
        xwall_max=int(xwall_series.max()), xwall_mean=float(xwall_series.mean()),
        xwall_last=int(xwall_series[-1]), xwall_first_nz_t=first_nz, xwall_final_pairs=n_pairs,
        uplift_per_pair=uplift,
        ms_per_frame=1e3 * wall / max(n_done, 1),
        final=rows[-1],
    )
    with open(os.path.join(VIDEOS_DIR, f"{OUT_PREFIX}{test}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"csv: {csv_path}")
    if not args.no_video:
        print(f"mp4: {mp4}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", choices=("t1", "t1v", "t2", "t3", "t4ctrl", "t4a", "t4b", "t4c"), required=True)
    parser.add_argument("--dur", type=float, default=-1.0, help="<0 means the test default (2.0s, 2.5s for t4c)")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.test.startswith("t4"):
        run_t4(args)
    else:
        if args.dur < 0:
            args.dur = 2.0
        run_test(args)


if __name__ == "__main__":
    main()
