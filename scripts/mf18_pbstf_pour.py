"""MF-18 plan section 6-C, phase C2b: the production MF-18 pour scene on the PBSTF path.

Geometry, liquid levels and the motion timeline are mirrored item-by-item from the PBD
production script `mf18_realscale_pbd.py` (do not retune here):

  * mug  : 7 cm mouth diameter, 10 cm tall, 4 mm walls -> `assets/mf18_cup_big.obj`
           (watertight, 386v/768f, local origin = outer-bottom centre, static mesh collider)
  * jug  : 5.6 cm diameter, 10 cm tall, 4 mm walls -> `assets/mf18_cup_small.obj`
           (kinematic mesh collider, per-frame set_static_colliders_pose over the timeline)
  * table: analytic box collider FIRST in the collider list (load-bearing priority), same
           geometry as the PBD visual table box (0.87 x 0.53 x 0.013 at (0.117, 0, -0.0067))
  * coffee: R=0.0345, H=0.090 cylinder at the mug cavity floor; milk: R=0.0265, H=0.082
           cylinder inside the upright jug at its rest pose (PBD sampling convention)
  * domain: plan section 6-D "medium" tier [-0.10, -0.12, 0] -> [0.34, 0.12, 0.30]
  * recipe (phase-B fix): ps=0.002, support=0.006, rho0=1000, h=1/960 (dt=1/240 x 4
           substeps), 20 iterations, topology_rebuild_interval=2, PCA normals on,
           density_compliance=375000, surface_tension_compliance=0.0048, distance 40/180,
           regular sampler, wall friction 0.1 with adhesion effectively off (stage-D ruling:
           is_collider_adhesion_friction_enabled=True + collider_adhesion_compliance=1e12,
           keeps the "no wall sticking" rule), diffusion_coeff=0.005 (C2a calibration:
           the old PBD demo's visible rate is eps ~= 0.0044; production takes 0.005)
  * cameras: main (0.093, -0.473, 0.210)/(0.073, 0, 0.077) fov 42 and closeup
           (0.147, -0.133, 0.143)/(0.020, 0, 0.107) fov 32, 960x1280, point sizes 5/10

Modes:
  --presettle SECONDS   settle-only stage: full recipe with diffusion on, jug collider
                        frozen at its rest pose; saves the settled state npz (velocities
                        ZEROED, PBD precedent) plus a strict sidecar json, then exits.
  --test static         2 s hold from the settled state, both vessels motionless.
  --test dyn2s          2 s motion window from the settled state: timeline t in [5, 7] s
                        (the complete lift; chosen over [6, 8] because the pose must be
                        continuous with the settled state at frame 0 - a [6, 8] window
                        would teleport the jug to mid-lift).
  --test d1             12 s stage-D1 sensitivity window from the settled state: timeline
                        t in [0, 12] s (lift 5-7 -> tilt 7-11 -> first pour), requires
                        --label A/B/C; hard gates = common set only, the dome/spread and
                        pour metrics are informational.
  --substeps/--iters    stage-D1 knobs; defaults 4/20 reproduce tier A (current recipe).
                        solver h = dt/substeps (asserted at startup).

Settled state integrity (plan section 6-C: "refuse to silently load old PBD states"):
the sidecar json records solver=PBSTF, kernel=cubic, ps, support, the mass calibration,
the full recipe, collider specs, sha256 of both OBJ assets and the creation date.
--init-from refuses to load when the sidecar is missing (every old PBD npz), when
solver/kernel/ps/support mismatch, when asset hashes mismatch, when particle counts
mismatch, or when the saved velocities are a high-energy state (PBD 2026-09-21 guard).

Usage:
  PYTHONIOENCODING=utf-8 "$PY" mf18_pbstf_pour.py --presettle 6.0
  PYTHONIOENCODING=utf-8 "$PY" mf18_pbstf_pour.py --test static
  PYTHONIOENCODING=utf-8 "$PY" mf18_pbstf_pour.py --test dyn2s
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import genesis as gs  # noqa: E402
from genesis.utils.misc import qd_to_numpy  # noqa: E402

assert "multiflow" in os.path.normpath(gs.__file__), f"Wrong genesis: {gs.__file__}"

VIDEOS_DIR = os.path.join(WORKSPACE, "videos")
ASSETS_DIR = os.path.join(WORKSPACE, "assets")
OUT_PREFIX = "mf18_pbstf_pour_"
MUG_OBJ = os.path.join(ASSETS_DIR, "mf18_cup_big.obj")
JUG_OBJ = os.path.join(ASSETS_DIR, "mf18_cup_small.obj")
SETTLED_DEFAULT = os.path.join(ASSETS_DIR, "mf18_pbstf_settled.npz")

DT = 1.0 / 240.0
SUBSTEPS = 4
PS = 0.002
SUPPORT = 0.006
RHO0 = 1000.0
DIFFUSION_EPS = 0.005
GRAVITY = (0.0, 0.0, -9.81)

# ---- mirrored from mf18_realscale_pbd.py (geometry) -------------------------------------
R_IN = 0.035
L_MUG = 0.10
T_WALL = 0.004
T_BOTTOM = 0.004
R_OUT = R_IN + T_WALL  # 0.039
Z_FLOOR = T_BOTTOM  # 0.004 cavity floor
Z_RIM = Z_FLOOR + L_MUG  # 0.104 rim
Z_TABLE = 0.0

R_JUG = 0.028
L_JUG = 0.10
T_JUG_WALL = 0.004
T_JUG_BOTTOM = 0.004
Z_JUG_RIM = T_JUG_BOTTOM + L_JUG  # 0.104 jug-local rim

R_COFFEE = R_IN - 0.5 * PS  # 0.0345
H_COFFEE = 0.090
R_MILK = R_JUG - 0.75 * PS  # 0.0265
H_MILK = 0.082
S0 = 0.5 * PS  # sample margin above the cavity floor

# ---- mirrored from mf18_realscale_pbd.py (timeline) --------------------------------------
PX, PZ = 0.14, 0.0
PIVOT_X, PIVOT_Z = 0.030, 0.125
PARK_X = 0.217
T_LIFT0, T_LIFT1 = 5.0, 7.0
T_TILT0, T_TILT1 = 7.0, 11.0
T_TRICKLE = 26.0
TRICKLE_RAMP = 1.5
T_DOME_STOP = 30.0
STOP_RAMP = 1.0
STOP_RETREAT_X = 0.04 / 3.0
STOP_LIFT_Z = 0.02 / 3.0
T_OVERFILL = 34.0
OVERFILL_RAMP = 0.75
T_BACK = 37.0
T_PARK0, T_PARK1 = 42.0, 46.0
TILT_MAX_DEG = 72.0
TILT_TRICKLE_DEG = 86.0
TILT_DOME_HOLD_DEG = 65.0
TILT_OVERFILL_DEG = 112.0

# plan section 6-D "medium" domain tier
LOWER_BOUND = np.array((-0.10, -0.12, 0.0), dtype=np.float32)
UPPER_BOUND = np.array((0.34, 0.12, 0.30), dtype=np.float32)

# collider list order (support priority: load-bearing first)
IDX_TABLE = 0
IDX_MUG = 1
IDX_JUG = 2

DYN_WINDOW = (T_LIFT0, T_LIFT1)  # dyn2s timeline window [5, 7] s (see module docstring)

RECIPE = dict(
    dt=DT,
    substeps=SUBSTEPS,
    max_solver_iterations=20,
    topology_rebuild_interval=2,
    density_compliance=375000.0,
    surface_tension_compliance=0.0048,
    surface_distance_compliance=40.0,
    interior_distance_compliance=180.0,
    surface_viscosity=0.0,
    interior_viscosity=0.0,
    sampler="regular",
    rho=RHO0,
    diffusion_coeff=DIFFUSION_EPS,
    gravity=list(GRAVITY),
    # stage-D ruling (2026-09-22): wall friction on, adhesion present but compliance 1e12
    # = effectively off (keeps the "no wall sticking" rule); friction coefficient 0.1
    is_collider_adhesion_friction_enabled=True,
    collider_adhesion_compliance=1e12,
    collider_friction=0.1,
)


def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _ramp(t, t0, t1):
    return smoothstep((t - t0) / max(t1 - t0, 1e-9))


def quat_roty_neg(theta):
    return np.array([np.cos(theta / 2.0), 0.0, -np.sin(theta / 2.0), 0.0])


def jug_origin_p():
    return np.array([PIVOT_X + R_JUG, 0.0, PIVOT_Z - Z_RIM])


def jug_park():
    return np.array([PARK_X, 0.0, PZ])


def jug_theta(t):
    th_max = np.deg2rad(TILT_MAX_DEG)
    th_tr = np.deg2rad(TILT_TRICKLE_DEG)
    th_hold = np.deg2rad(TILT_DOME_HOLD_DEG)
    if t < T_TILT0:
        return 0.0
    if t < T_TILT1:
        return th_max * _ramp(t, T_TILT0, T_TILT1)
    if t < T_TRICKLE:
        return th_max
    if t < T_DOME_STOP:
        return th_max + (th_tr - th_max) * _ramp(t, T_TRICKLE, min(T_TRICKLE + TRICKLE_RAMP, T_DOME_STOP))
    if t < T_OVERFILL:
        angle_start = T_DOME_STOP + 0.5 * STOP_RAMP
        return th_tr + (th_hold - th_tr) * _ramp(t, angle_start, min(T_DOME_STOP + STOP_RAMP, T_OVERFILL))
    th_over = np.deg2rad(TILT_OVERFILL_DEG)
    if t < T_BACK:
        return th_hold + (th_over - th_hold) * _ramp(t, T_OVERFILL, min(T_OVERFILL + OVERFILL_RAMP, T_BACK))
    return th_over * (1.0 - _ramp(t, T_BACK, T_PARK1))


def jug_pose(t):
    """MF-17 jug pose (mirrored): (origin=outer-bottom centre, quat, axis, theta, u)."""
    th = jug_theta(t)
    a = np.array([-np.sin(th), 0.0, np.cos(th)])
    e1 = np.array([np.cos(th), 0.0, np.sin(th)])
    origin_0 = np.array([PX, 0.0, PZ])
    lift_top = np.array([PX, 0.0, PIVOT_Z - T_JUG_BOTTOM])
    origin_p = jug_origin_p()
    pivot = np.array([PIVOT_X, 0.0, PIVOT_Z])
    rot = (pivot - Z_RIM * a + R_JUG * e1) - origin_p
    u_lift = _ramp(t, T_LIFT0, T_LIFT1)
    u_tilt = _ramp(t, T_TILT0, T_TILT1)
    u_park = _ramp(t, T_BACK, T_PARK1)
    stop_mid = T_DOME_STOP + 0.5 * STOP_RAMP
    if t < T_DOME_STOP:
        u_stop = 0.0
    elif t < stop_mid:
        u_stop = _ramp(t, T_DOME_STOP, stop_mid)
    elif t < T_OVERFILL:
        u_stop = 1.0
    elif t < T_OVERFILL + OVERFILL_RAMP:
        u_stop = 1.0 - _ramp(t, T_OVERFILL, T_OVERFILL + OVERFILL_RAMP)
    else:
        u_stop = 0.0
    origin = (origin_0
              + u_lift * (lift_top - origin_0)
              + u_tilt * (origin_p - lift_top)
              + rot
              + u_stop * np.array([STOP_RETREAT_X, 0.0, STOP_LIFT_Z])
              + u_park * (jug_park() - origin_p))
    quat = quat_roty_neg(th).astype(np.float32)
    return origin.astype(np.float32), quat, a.astype(np.float32), th, u_tilt


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_assets():
    import trimesh

    for path in (MUG_OBJ, JUG_OBJ):
        if not os.path.exists(path):
            raise SystemExit(f"collider asset missing: {path}")
        mesh = trimesh.load(path)
        print(
            f"[asset] {os.path.basename(path)}: {len(mesh.vertices)}v/{len(mesh.faces)}f "
            f"watertight={mesh.is_watertight} winding={mesh.is_winding_consistent} "
            f"bounds=({mesh.bounds[0].round(4).tolist()} -> {mesh.bounds[1].round(4).tolist()})"
        )
        if not (mesh.is_watertight and mesh.is_winding_consistent):
            raise SystemExit(f"{path} is not watertight/winding-consistent - fix the asset first")


def liquid_material(c_init):
    return gs.materials.PBSTF.Liquid(
        rho=RHO0,
        density_compliance=RECIPE["density_compliance"],
        surface_tension_compliance=RECIPE["surface_tension_compliance"],
        surface_distance_compliance=RECIPE["surface_distance_compliance"],
        interior_distance_compliance=RECIPE["interior_distance_compliance"],
        surface_viscosity=RECIPE["surface_viscosity"],
        interior_viscosity=RECIPE["interior_viscosity"],
        is_collider_adhesion_friction_enabled=RECIPE["is_collider_adhesion_friction_enabled"],
        collider_adhesion_compliance=RECIPE["collider_adhesion_compliance"],
        collider_friction=RECIPE["collider_friction"],
        sampler="regular",
        c_init=c_init,
    )


def build_scene(substeps=SUBSTEPS, iters=RECIPE["max_solver_iterations"], surface_recon=False):
    origin0, quat0, _, _, _ = jug_pose(0.0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=substeps, gravity=GRAVITY),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=PS,
            lower_bound=tuple(LOWER_BOUND),
            upper_bound=tuple(UPPER_BOUND),
            max_solver_iterations=iters,
            topology_rebuild_interval=RECIPE["topology_rebuild_interval"],
            diffusion_coeff=RECIPE["diffusion_coeff"],
            static_colliders=[
                gs.options.PBSTFBoxStaticColliderOptions(
                    pos=(0.0, 0.0, 0.0),
                    quat=(1.0, 0.0, 0.0, 0.0),
                    lower=(0.117 - 0.435, -0.265, -0.0132),
                    upper=(0.117 + 0.435, 0.265, 0.0),
                ),
                gs.options.PBSTFMeshStaticColliderOptions(file=MUG_OBJ, scale=1.0, sdf_res=150),
                gs.options.PBSTFMeshStaticColliderOptions(
                    file=JUG_OBJ, scale=1.0, sdf_res=150,
                    pos=tuple(origin0), quat=(1.0, 0.0, 0.0, 0.0),
                ),
            ],
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )

    # visualization-only rigid entities (T4 pattern): the OBJ origins are the outer-bottom
    # centres, so the jug visual lands on the collider pose with set_pos/set_quat directly
    mug_vis = scene.add_entity(
        morph=gs.morphs.Mesh(file=MUG_OBJ, pos=(0.0, 0.0, 0.0), fixed=True, collision=False),
        material=gs.materials.Rigid(needs_coup=False),
        surface=gs.surfaces.Default(color=(0.85, 0.85, 0.9, 0.35), opacity=0.35),
    )
    jug_vis = scene.add_entity(
        morph=gs.morphs.Mesh(file=JUG_OBJ, pos=tuple(origin0), fixed=True, collision=False),
        material=gs.materials.Rigid(needs_coup=False),
        surface=gs.surfaces.Default(color=(0.9, 0.92, 0.95, 0.45), opacity=0.45),
    )
    scene.add_entity(
        morph=gs.morphs.Box(size=(0.87, 0.53, 0.013), pos=(0.117, 0.0, Z_TABLE - 0.0067),
                            fixed=True, collision=False),
        material=gs.materials.Rigid(needs_coup=False),
        surface=gs.surfaces.Default(color=(0.55, 0.55, 0.58, 1.0), opacity=1.0),
    )

    # --surface: both liquids share one merged recon mesh in the rasterizer, so both entities
    # must be in vis_mode="recon" (only recon-mode entities enter the merged particle set).
    fluid_surface = (
        gs.surfaces.Default(vis_mode="recon", recon_backend="splashsurf") if surface_recon else None
    )
    coffee = scene.add_entity(
        material=liquid_material(1.0),
        morph=gs.morphs.Cylinder(
            radius=R_COFFEE, height=H_COFFEE,
            pos=(0.0, 0.0, Z_FLOOR + S0 + 0.5 * H_COFFEE),
        ),
        surface=fluid_surface or gs.surfaces.Default(color=(0.35, 0.55, 0.95, 1.0), opacity=1.0),
    )
    milk_pos = origin0 + np.array([0.0, 0.0, T_JUG_BOTTOM + S0 + 0.5 * H_MILK])
    milk = scene.add_entity(
        material=liquid_material(0.0),
        morph=gs.morphs.Cylinder(radius=R_MILK, height=H_MILK, pos=tuple(milk_pos)),
        surface=fluid_surface or gs.surfaces.Default(color=(0.95, 0.95, 0.95, 1.0), opacity=1.0),
    )

    cam = scene.add_camera(res=(960, 1280), pos=(0.093, -0.473, 0.210),
                           lookat=(0.073, 0.0, 0.077), fov=42, GUI=False)
    cam_close = scene.add_camera(res=(960, 1280), pos=(0.147, -0.133, 0.143),
                                 lookat=(0.020, 0.0, 0.107), fov=32, GUI=False)

    t0 = time.perf_counter()
    scene.build()
    build_s = time.perf_counter() - t0
    scene.visualizer.rasterizer._camera_targets[cam.uid].point_size = 5.0
    scene.visualizer.rasterizer._camera_targets[cam_close.uid].point_size = 10.0
    return scene, coffee, milk, jug_vis, cam, cam_close, build_s


def snapshot_user_order(solver):
    """Substep-end state in user (entity) order; thinwall Recorder convention."""
    idx = qd_to_numpy(solver.particles_ng.reordered_idx, transpose=True)[0]
    pos = qd_to_numpy(solver.particles.pos, transpose=True)[0].copy()
    vel_r = qd_to_numpy(solver.particles_reordered.vel, transpose=True)[0]
    rho_r = qd_to_numpy(solver.particles_reordered.density, transpose=True)[0]
    surf_r = qd_to_numpy(solver.on_surface, transpose=True)[0].astype(bool)
    c = qd_to_numpy(solver.particles.c, transpose=True)[0].copy()
    active = qd_to_numpy(solver.particles_ng.active, transpose=True)[0].astype(bool)
    return pos, vel_r[idx], rho_r[idx], surf_r[idx], c, active


def sidecar_path(npz_path):
    return os.path.splitext(npz_path)[0] + ".json"


def save_settled(npz_path, solver, coffee, milk, settle_dur, end_stats):
    n_coffee, n_milk = coffee.n_particles, milk.n_particles
    n_fluid = n_coffee + n_milk
    pos, vel, _, _, c, _ = snapshot_user_order(solver)
    np.savez(
        npz_path,
        pos=pos[:, None, :].astype(np.float32),
        vel=np.zeros((n_fluid, 1, 3), dtype=np.float32),  # velocities zeroed (PBD precedent)
        c=c[:, None].astype(np.float32),
        n_coffee=n_coffee,
        n_milk=n_milk,
        ps=PS,
    )
    sidecar = dict(
        solver="PBSTF",
        kernel="cubic",
        particle_size=PS,
        support_radius=SUPPORT,
        default_mass=float(solver._default_mass),
        recipe=RECIPE,
        colliders=[
            dict(type="box", lower=[-0.318, -0.265, -0.0132], upper=[0.552, 0.265, 0.0], role="table"),
            dict(type="mesh", file=os.path.basename(MUG_OBJ), sdf_res=150, role="mug", pose="identity"),
            dict(type="mesh", file=os.path.basename(JUG_OBJ), sdf_res=150, role="jug",
                 pose="rest " + np.array2string(jug_pose(0.0)[0], precision=6)),
        ],
        asset_hashes={os.path.basename(MUG_OBJ): sha256_file(MUG_OBJ),
                      os.path.basename(JUG_OBJ): sha256_file(JUG_OBJ)},
        n_coffee=int(n_coffee),
        n_milk=int(n_milk),
        settle=dict(duration_s=settle_dur, vel_saved="zeroed", **end_stats),
        created=datetime.now(timezone.utc).isoformat(),
    )
    with open(sidecar_path(npz_path), "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[settled] npz -> {npz_path}")
    print(f"[settled] sidecar -> {sidecar_path(npz_path)}")


def refuse(msg):
    print(f"[settled-refuse] {msg}")
    raise SystemExit(2)


def load_settled(npz_path, solver, coffee, milk):
    """Strict settled-state load. Any violation refuses the run (exit code 2)."""
    n_coffee, n_milk = coffee.n_particles, milk.n_particles
    sc_path = sidecar_path(npz_path)
    if not os.path.exists(sc_path):
        refuse(f"{npz_path}: missing sidecar json ({sc_path}); old PBD settled states have "
               f"no sidecar and must not be loaded silently - regenerate with --presettle")
    with open(sc_path) as f:
        sc = json.load(f)
    if sc.get("solver") != "PBSTF":
        refuse(f"{npz_path}: sidecar solver={sc.get('solver')!r}, want 'PBSTF'")
    if sc.get("kernel") != "cubic":
        refuse(f"{npz_path}: sidecar kernel={sc.get('kernel')!r}, want 'cubic'")
    if abs(float(sc.get("particle_size", -1)) - PS) > 1e-12:
        refuse(f"{npz_path}: sidecar ps={sc.get('particle_size')}, want {PS}")
    if abs(float(sc.get("support_radius", -1)) - SUPPORT) > 1e-12:
        refuse(f"{npz_path}: sidecar support={sc.get('support_radius')}, want {SUPPORT}")
    for path in (MUG_OBJ, JUG_OBJ):
        want = sc.get("asset_hashes", {}).get(os.path.basename(path))
        got = sha256_file(path)
        if want != got:
            refuse(f"{npz_path}: asset hash mismatch for {os.path.basename(path)} "
                   f"(sidecar {want}, current {got}) - the collider assets changed")
    if int(sc.get("n_coffee", -1)) != n_coffee or int(sc.get("n_milk", -1)) != n_milk:
        refuse(f"{npz_path}: sidecar counts {sc.get('n_coffee')}/{sc.get('n_milk')} != "
               f"built scene {n_coffee}/{n_milk}")
    with np.load(npz_path) as dat:
        pos = np.asarray(dat["pos"], dtype=np.float32)
        vel = np.asarray(dat["vel"], dtype=np.float32)
        c = np.asarray(dat["c"], dtype=np.float32)
        if int(dat["n_coffee"]) != n_coffee or int(dat["n_milk"]) != n_milk:
            refuse(f"{npz_path}: npz counts {int(dat['n_coffee'])}/{int(dat['n_milk'])} != "
                   f"built scene {n_coffee}/{n_milk}")
        if abs(float(dat["ps"]) - PS) > 1e-12:
            refuse(f"{npz_path}: npz ps={float(dat['ps'])}, want {PS}")
    n_fluid = n_coffee + n_milk
    if pos.shape != (n_fluid, 1, 3) or vel.shape != (n_fluid, 1, 3) or c.shape != (n_fluid, 1):
        refuse(f"{npz_path}: unexpected array shapes pos{pos.shape} vel{vel.shape} c{c.shape}")
    speed = np.linalg.norm(vel[:, 0, :], axis=1)
    ke = 0.5 * float(solver._default_mass) * float((vel.astype(np.float64) ** 2).sum())
    edge = ((np.abs(pos[:, 0, :] - LOWER_BOUND[None, :]) < 1e-4)
            | (np.abs(pos[:, 0, :] - UPPER_BOUND[None, :]) < 1e-4)).any(axis=1)
    print(f"[settled] load check: KE={ke:.3e} speed max={speed.max():.4f} "
          f"domain-edge particles={int(edge.sum())}")
    if speed.max() > 0.5:
        refuse(f"{npz_path}: high-energy state (speed max {speed.max():.2f} m/s) - re-run --presettle")
    coffee.set_particles_pos(pos[:n_coffee, 0])
    milk.set_particles_pos(pos[n_coffee:, 0])
    coffee.set_particles_vel(vel[:n_coffee, 0])
    milk.set_particles_vel(vel[n_coffee:, 0])
    c_full = qd_to_numpy(solver.particles.c, transpose=True)
    c_full[0, :n_fluid] = c[:, 0]
    solver.particles.c.from_numpy(np.ascontiguousarray(c_full.transpose(1, 0)))
    print(f"[settled] loaded {npz_path} (sidecar validated: solver=PBSTF kernel=cubic "
          f"ps={PS} support={SUPPORT} hashes ok)")


def jug_cavity_mask(pos, origin, axis):
    """s (along axis from inner bottom centre) and r (radial) for the jug at (origin, axis)."""
    rel = pos - (origin + T_JUG_BOTTOM * axis)[None, :]
    s = rel @ axis
    r = np.linalg.norm(rel - s[:, None] * axis[None, :], axis=1)
    return s, r


def frame_metrics(t, theta, solver, coffee, milk, pos, vel, rho, surf, c, active,
                  jug_origin, jug_quat, jug_axis, mass, sum_c0, ms):
    n_coffee = coffee.n_particles
    n_fluid = n_coffee + milk.n_particles
    nan = int((~np.isfinite(pos).all(axis=1) | ~np.isfinite(vel).all(axis=1)
               | ~np.isfinite(c)).sum())
    safe = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    safe_v = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
    safe_c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    row = dict(t=t, theta_deg=float(np.rad2deg(theta)), nan=nan,
               ke=float(0.5 * mass * np.sum(safe_v * safe_v)),
               vmax=float(np.linalg.norm(safe_v, axis=1).max()),
               n_active=int(active.sum()), sum_c=float(safe_c.sum()),
               drift_c=abs(float(safe_c.sum()) - sum_c0) / max(sum_c0, 1e-30),
               z_min=float(safe[:, 2].min()),
               n_edge=int(((np.abs(safe - LOWER_BOUND[None, :]) < 1e-4)
                           | (np.abs(safe - UPPER_BOUND[None, :]) < 1e-4)).any(axis=1).sum()),
               ms=ms)
    # mug containment (world frame, static)
    r_all = np.linalg.norm(safe[:, :2], axis=1)
    in_mug = (r_all <= R_IN + PS) & (safe[:, 2] >= Z_FLOOR - PS) & (safe[:, 2] <= Z_RIM + PS)
    # jug containment (current pose)
    s_j, r_j = jug_cavity_mask(safe, jug_origin, jug_axis)
    in_jug = (s_j > -PS) & (s_j < L_JUG + PS) & (r_j <= R_JUG + PS)
    row["n_coffee_mug"] = int(in_mug[:n_coffee].sum())
    row["n_coffee_out"] = n_coffee - row["n_coffee_mug"]
    row["n_milk_jug"] = int(in_jug[n_coffee:].sum())
    row["n_milk_mug"] = int((in_mug & ~in_jug)[n_coffee:].sum())
    row["n_below_table"] = int((safe[:, 2] < Z_TABLE - 1e-4).sum())
    for gi, (s0, e0) in enumerate(((0, n_coffee), (n_coffee, n_fluid))):
        com = safe[s0:e0].mean(axis=0)
        for a, ax in enumerate("xyz"):
            row[f"com{gi}_{ax}"] = float(com[a])
        interior = np.zeros(n_fluid, dtype=bool)
        interior[s0:e0] = ~surf[s0:e0]
        rr = rho[interior] / RHO0
        for q, v in zip(("p25", "p50", "p75"), np.percentile(rr, (25, 50, 75)) if len(rr) else (np.nan,) * 3):
            row[f"rho{gi}_{q}"] = float(v)
        row[f"ke{gi}"] = float(0.5 * mass * np.sum(safe_v[s0:e0] ** 2))
    # wall-layer density (within 2ps of a vessel surface, non-surface particles)
    mug_wall = ((r_all > R_IN - 2 * PS) | (safe[:, 2] < Z_FLOOR + 2 * PS)) & in_mug
    mug_wall[:n_coffee] &= ~surf[:n_coffee]
    rw = rho[:n_coffee][mug_wall[:n_coffee]] / RHO0
    row["rho0_wall_p50"] = float(np.median(rw)) if len(rw) else float("nan")
    jug_wall = ((r_j > R_JUG - 2 * PS) | (s_j < 2 * PS)) & in_jug
    jug_wall[n_coffee:] &= ~surf[n_coffee:]
    rw = rho[n_coffee:][jug_wall[n_coffee:]] / RHO0
    row["rho1_wall_p50"] = float(np.median(rw)) if len(rw) else float("nan")
    # geometric tunneling: fluid centers inside the solid wall/bottom shells
    # (contact anchors particles at particle_radius = 1mm off each surface)
    eps_in = 0.5 * PS
    tun_jug = ((r_j > R_JUG + eps_in) & (r_j < R_JUG + T_JUG_WALL - eps_in)
               & (s_j > 0.0) & (s_j < L_JUG))
    tun_jug |= ((s_j > -T_JUG_BOTTOM + eps_in) & (s_j < -eps_in) & (r_j < R_JUG))
    row["n_tun_jug"] = int(tun_jug[n_coffee:].sum())
    tun_mug = ((r_all > R_IN + eps_in) & (r_all < R_OUT - eps_in)
               & (safe[:, 2] > Z_FLOOR) & (safe[:, 2] < Z_RIM))
    tun_mug |= ((safe[:, 2] > Z_FLOOR - T_BOTTOM + eps_in) & (safe[:, 2] < Z_FLOOR - eps_in)
                & (r_all < R_IN))
    row["n_tun_mug_coffee"] = int(tun_mug[:n_coffee].sum())
    row["n_tun_mug_milk"] = int(tun_mug[n_coffee:].sum())
    # free-surface flatness: z spread (p95 - p5) of the top-10% particles per entity
    for gi, (s0, e0) in enumerate(((0, n_coffee), (n_coffee, n_fluid))):
        z_g = safe[s0:e0, 2]
        top = z_g >= np.percentile(z_g, 90)
        row[f"ztop{gi}_p95_p5"] = float(np.percentile(z_g[top], 95) - np.percentile(z_g[top], 5))
    # collider pose readback (jug), w-x-y-z convention
    pos_rb = qd_to_numpy(solver._static_colliders_pos, transpose=True)[0, IDX_JUG]
    quat_rb = qd_to_numpy(solver._static_colliders_quat, transpose=True)[0, IDX_JUG]
    row["pose_err_pos"] = float(np.linalg.norm(pos_rb - jug_origin))
    row["pose_err_quat"] = float(min(np.linalg.norm(quat_rb - jug_quat),
                                     np.linalg.norm(quat_rb + jug_quat)))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--presettle", type=float, default=0.0, help="settle-only duration, s")
    parser.add_argument("--test", choices=("static", "dyn2s", "d1"), default=None)
    parser.add_argument("--init-from", type=str, default=SETTLED_DEFAULT)
    parser.add_argument("--settled-out", type=str, default=SETTLED_DEFAULT)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--surface", action="store_true",
                        help="render liquids as one merged splashsurf iso-surface with concentration "
                             "vertex colors instead of point clouds (render-only, physics unchanged)")
    parser.add_argument("--fps", type=int, default=30)
    # stage-D1 sensitivity knobs (defaults reproduce tier A = current recipe)
    parser.add_argument("--substeps", type=int, default=SUBSTEPS, help="sim substeps; solver h = dt/substeps")
    parser.add_argument("--iters", type=int, default=RECIPE["max_solver_iterations"])
    parser.add_argument("--surface-visc", type=float, default=None,
                        help="override RECIPE surface_viscosity (default keeps RECIPE value)")
    parser.add_argument("--interior-visc", type=float, default=None,
                        help="override RECIPE interior_viscosity (default keeps RECIPE value)")
    parser.add_argument("--diffusion", type=float, default=None,
                        help="override RECIPE diffusion_coeff (concentration smoothing; 0 disables)")
    parser.add_argument("--timeline", choices=("v1", "v2"), default="v1",
                        help="v2 = keep pouring as the jug empties: ramp 72->82 deg over [14,18] s, "
                             "dome hold 22-24 s, overfill pulse at 24 s (ramp 1.75 s), return at 28 s")
    parser.add_argument("--label", type=str, default="", help="required for --test d1 (output file tag)")
    parser.add_argument("--dur", type=float, default=0.0,
                        help="duration override s (0 = mode default: presettle value / 2 s / d1 12 s)")
    parser.add_argument("--out-prefix", type=str, default="",
                        help="output filename prefix override (product naming only, e.g. mf18_pbstf_pour_)")
    args = parser.parse_args()
    if (args.presettle > 0.0) == (args.test is not None):
        raise SystemExit("exactly one of --presettle / --test must be given")
    if args.test == "d1" and not args.label:
        raise SystemExit("--test d1 requires --label (A/B/C)")
    if args.substeps < 1 or args.iters < 1:
        raise SystemExit("--substeps/--iters must be >= 1")
    if args.dur < 0.0:
        raise SystemExit("--dur must be >= 0")
    if args.surface_visc is not None:
        RECIPE["surface_viscosity"] = args.surface_visc
    if args.interior_visc is not None:
        RECIPE["interior_viscosity"] = args.interior_visc
    if args.diffusion is not None:
        if args.diffusion < 0.0:
            raise SystemExit("--diffusion must be >= 0")
        RECIPE["diffusion_coeff"] = args.diffusion
    if args.timeline == "v2":
        # rim/mix review (2026-09-23): continue tilting as the jug empties instead of holding
        # 72 deg until 26 s (accumulate-and-drip); slow the final pulse ramp 0.75 -> 1.75 s.
        globals().update(T_TRICKLE=14.0, TRICKLE_RAMP=4.0, TILT_TRICKLE_DEG=82.0,
                         T_DOME_STOP=22.0, T_OVERFILL=24.0, OVERFILL_RAMP=1.75, T_BACK=28.0)

    check_assets()
    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene, coffee, milk, jug_vis, cam, cam_close, build_s = build_scene(args.substeps, args.iters, args.surface)
    solver = scene.sim.pbstf_solver
    n_coffee, n_milk = coffee.n_particles, milk.n_particles
    n_fluid = n_coffee + n_milk
    mass = float(solver._default_mass)
    h_expect = DT / args.substeps
    assert abs(solver._substep_dt - h_expect) < 1e-9, (
        f"solver h {solver._substep_dt:.6e} != dt/substeps {h_expect:.6e}")
    print(
        f"MF18 PBSTF POUR: coffee={n_coffee} milk={n_milk} ps={PS} support={solver._support_radius} "
        f"h={solver._substep_dt:.6e} (assert dt/{args.substeps}) iters={args.iters} "
        f"topo_rebuild={RECIPE['topology_rebuild_interval']} eps_d={solver._diffusion_coeff} "
        f"visc_s={RECIPE['surface_viscosity']} visc_i={RECIPE['interior_viscosity']} "
        f"friction={RECIPE['collider_friction']} adhesion_c={RECIPE['collider_adhesion_compliance']:.0e} "
        f"mass={mass:.10e} colliders={solver._n_static_colliders} build={build_s:.1f}s"
    )
    print(f"[domain] {LOWER_BOUND.tolist()} -> {UPPER_BOUND.tolist()} "
          f"hash res x cell {solver._support_radius:.4f}")
    print(f"[timeline] {args.timeline}: tilt {TILT_MAX_DEG}deg [{T_TILT0},{T_TILT1}] "
          f"trickle {TILT_TRICKLE_DEG}deg @{T_TRICKLE}+{TRICKLE_RAMP}s dome_stop {T_DOME_STOP} "
          f"hold {TILT_DOME_HOLD_DEG}deg pulse {TILT_OVERFILL_DEG}deg @{T_OVERFILL}+{OVERFILL_RAMP}s "
          f"back {T_BACK} eps_d={solver._diffusion_coeff}")

    tag = "preset" if args.presettle > 0.0 else (f"d1_{args.label}" if args.test == "d1" else args.test)
    out_prefix = "mf18_pbstf_d1_" if args.test == "d1" else OUT_PREFIX
    tag = f"{args.label}" if args.test == "d1" else tag
    if args.out_prefix:
        out_prefix, tag = args.out_prefix, args.label or tag
    mp4_main = os.path.join(VIDEOS_DIR, f"{out_prefix}{tag}_main.mp4")
    mp4_close = os.path.join(VIDEOS_DIR, f"{out_prefix}{tag}_closeup.mp4")
    csv_path = os.path.join(VIDEOS_DIR, f"{out_prefix}{tag}_metrics.csv")

    if args.test is not None:
        load_settled(args.init_from, solver, coffee, milk)

    # post-build sync (stage-B finding) then t=0 snapshot
    solver._kernel_reorder_particles(0)
    solver._rebuild_topology(0)
    pos, vel, rho, surf, c, active = snapshot_user_order(solver)
    sum_c0 = float(c.sum())
    print(f"[t0] on_surface={surf.sum()}/{len(surf)} sum_c={sum_c0:.3f} "
          f"rho50={(np.median(rho[~surf]) / RHO0):.4f}")

    dur = args.presettle if args.presettle > 0.0 else (12.0 if args.test == "d1" else 2.0)
    if args.dur > 0.0:
        dur = args.dur
    n_frames = int(round(dur / DT))
    t_win0 = 0.0 if args.presettle > 0.0 or args.test in ("static", "d1") else DYN_WINDOW[0]

    if not args.no_video:
        cam.start_recording(save_to_filename=mp4_main, fps=args.fps)
        cam_close.start_recording(save_to_filename=mp4_close, fps=args.fps)

    origin, quat, axis, theta, _ = jug_pose(t_win0)
    solver.set_static_colliders_pose(pos=tuple(origin), quat=tuple(quat), colliders_idx=IDX_JUG)
    jug_vis.set_pos(origin, relative=True, zero_velocity=True)
    jug_vis.set_quat(quat, relative=True, zero_velocity=True)

    rows = [frame_metrics(0.0, theta, solver, coffee, milk, pos, vel, rho, surf, c, active,
                          origin, quat, axis, mass, sum_c0, 0.0)]
    print_every = max(1, int(round(0.25 / DT)))
    t_start = time.perf_counter()
    abort = ""
    for i in range(1, n_frames + 1):
        t_tl = t_win0 + i * DT
        # presettle pins the jug at its rest pose (module docstring contract) instead of
        # advancing the timeline - the 2026-09-22 presettle ran jug_pose(t) and lifted the
        # jug to mid-lift over t in [5, 6] s, saving a floating-milk settled state
        origin, quat, axis, theta, _ = jug_pose(0.0 if args.presettle > 0.0 else t_tl)
        solver.set_static_colliders_pose(pos=tuple(origin), quat=tuple(quat), colliders_idx=IDX_JUG)
        jug_vis.set_pos(origin, relative=True, zero_velocity=True)
        jug_vis.set_quat(quat, relative=True, zero_velocity=True)
        t0 = time.perf_counter()
        try:
            scene.step()
        except Exception as e:
            abort = f"{type(e).__name__}: {e}"
            print(f"[abort] frame {i}: {abort}")
            break
        ms = (time.perf_counter() - t0) * 1e3
        pos, vel, rho, surf, c, active = snapshot_user_order(solver)
        row = frame_metrics(t_tl - t_win0, theta, solver, coffee, milk, pos, vel, rho, surf, c,
                            active, origin, quat, axis, mass, sum_c0, ms)
        if i % print_every == 0 or i == n_frames:
            print(
                f"t={row['t']:5.2f}s th={row['theta_deg']:5.1f} nan={row['nan']} KE={row['ke']:.3e} "
                f"vmax={row['vmax']:.3f} act={row['n_active']} cM={row['n_coffee_mug']}/{n_coffee} "
                f"mJ={row['n_milk_jug']}/{n_milk} mM={row['n_milk_mug']} tbl={row['n_below_table']} "
                f"tun={row['n_tun_jug']}/{row['n_tun_mug_coffee'] + row['n_tun_mug_milk']} "
                f"drift={row['drift_c']:.1e} pose_err={row['pose_err_pos']:.1e}"
            )
        rows.append(row)

    if not args.no_video:
        cam.stop_recording()
        cam_close.stop_recording()
    wall = time.perf_counter() - t_start
    n_done = len(rows) - 1
    print(f"[perf] {n_done} frames in {wall:.1f}s -> {1e3 * wall / max(n_done, 1):.2f} ms/frame "
          f"(incl. render+IO)")
    if abort:
        print(f"[abort] run stopped early: {abort}")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"csv: {csv_path}")
    if not args.no_video:
        print(f"mp4: {mp4_main}")
        print(f"mp4: {mp4_close}")

    if args.presettle > 0.0:
        speed_end = np.linalg.norm(vel, axis=1)
        end_stats = dict(
            end_ke=float(rows[-1]["ke"]),
            speed_median=float(np.median(speed_end)),
            speed_p99=float(np.percentile(speed_end, 99)),
            speed_max=float(speed_end.max()),
            n_coffee_mug=int(rows[-1]["n_coffee_mug"]),
            n_milk_jug=int(rows[-1]["n_milk_jug"]),
            n_below_table=int(rows[-1]["n_below_table"]),
            nan=int(rows[-1]["nan"]),
        )
        print(f"[preset-end] {json.dumps(end_stats)}")
        os.makedirs(os.path.dirname(args.settled_out), exist_ok=True)
        save_settled(args.settled_out, solver, coffee, milk, args.presettle, end_stats)
        return

    # ---- test summary gates ----------------------------------------------------------------
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    fails = []

    def gate(name, ok, detail):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            fails.append(name)

    gate("nan", arr["nan"].max() == 0, f"nan max={int(arr['nan'].max())}")
    gate("active", arr["n_active"].min() == n_fluid and arr["n_active"].max() == n_fluid,
         f"n_active in [{int(arr['n_active'].min())}, {int(arr['n_active'].max())}] (n_fluid={n_fluid})")
    gate("c_conservation", arr["drift_c"].max() < 0.01,
         f"sum(c) drift max={arr['drift_c'].max():.3e} (<1%)")
    gate("below_table", arr["n_below_table"].max() == 0,
         f"n_below_table max={int(arr['n_below_table'].max())}")
    gate("domain_edge", arr["n_edge"].max() == 0, f"domain-edge particles max={int(arr['n_edge'].max())}")
    gate("tunnel_jug", arr["n_tun_jug"].max() == 0, f"milk-in-jug-wall max={int(arr['n_tun_jug'].max())}")
    gate("tunnel_mug", (arr["n_tun_mug_coffee"] + arr["n_tun_mug_milk"]).max() == 0,
         f"fluid-in-mug-wall max={int((arr['n_tun_mug_coffee'] + arr['n_tun_mug_milk']).max())}")
    gate("pose_readback", np.nanmax(arr["pose_err_pos"]) < 1e-6 and np.nanmax(arr["pose_err_quat"]) < 1e-6,
         f"pose err pos={np.nanmax(arr['pose_err_pos']):.2e} quat={np.nanmax(arr['pose_err_quat']):.2e}")
    if args.test == "static":
        leak_c = n_coffee - arr["n_coffee_mug"].min()
        leak_m = n_milk - arr["n_milk_jug"].min()
        gate("coffee_contained", leak_c == 0, f"coffee outside mug max={int(leak_c)} / {n_coffee}")
        gate("milk_contained", leak_m == 0, f"milk outside jug max={int(leak_m)} / {n_milk}")
        com0 = np.array([rows[0]["com0_x"], rows[0]["com0_y"], rows[0]["com0_z"]])
        com1 = np.array([rows[-1]["com0_x"], rows[-1]["com0_y"], rows[-1]["com0_z"]])
        gate("com_drift", abs(com1[2] - com0[2]) < 2 * PS,
             f"coffee com_z {com0[2]:.5f} -> {com1[2]:.5f}")
        # settled acceptance: calm KE band within 1 s (velocities load as zero, so KE can
        # only rise from 0; stage-B band per-particle scaled to this scene is ~3e-4)
        i_1s = min(int(round(1.0 / DT)), len(rows) - 1)
        gate("ke_calm_1s", arr["ke"][i_1s] < 5e-3 and rows[-1]["ke"] < 5e-3,
             f"KE t=1s {arr['ke'][i_1s]:.3e}, t=end {rows[-1]['ke']:.3e} (limit 5e-3)")
        # stage-D ruling: the static central dome (~9.4 mm spread) is accepted - record only
        print(f"[INFO] surface_calm: coffee z-top spread t=1s {arr['ztop0_p95_p5'][i_1s]:.4f}, "
              f"t=end {arr['ztop0_p95_p5'][-1]:.4f} (informational, dome accepted)")
    elif args.test == "dyn2s":
        gate("milk_rides_jug", arr["n_milk_jug"].min() >= 0.999 * n_milk,
             f"n_milk_jug min={int(arr['n_milk_jug'].min())} / {n_milk} (lift window, jug upright)")
        gate("theta_swept", abs(arr["theta_deg"].max() - 0.0) < 1e-6,
             f"theta max={arr['theta_deg'].max():.3f} deg (lift-only window, tilt starts at t=7)")
        gate("ke_bounded", arr["ke"].max() < 10.0, f"KE max={arr['ke'].max():.3e}")
    else:  # d1 sensitivity window: hard gates above only; the rest is reported, not gated
        print(f"[INFO] d1 summary: n_milk_mug end={int(arr['n_milk_mug'][-1])} "
              f"first>0 at t={next((r['t'] for r in rows if r['n_milk_mug'] > 0), float('nan')):.2f}s "
              f"KE max={arr['ke'].max():.3e} ztop0 end={arr['ztop0_p95_p5'][-1]:.4f} "
              f"milk outside jug max={int(n_milk - arr['n_milk_jug'].min())}")
    if fails:
        print(f"RESULT: FAIL ({fails})")
        raise SystemExit(1)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
