#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dbg_mf16_jug_event_forensic.py -- mechanism forensics for the two diag17 historical-tunnel events.

Run ``mf16_smoke_diag17_jug_statefix_h20`` (20 s no-video smoke) left exactly two events in
``videos/mf16_smoke_diag17_jug_statefix_h20_tunnel_events.csv``:

  1. jug pid 58375  t=8.65 s  kind=wall          (over the spout rim, below the 0.320 clearance plane)
  2. mug pid 64992  t=16.90 s kind=wall_inbound  (cleared overflow particle grazing the r=0.146 proxy)

This script decides, per event, TRUE collision tunnelling (particle centre strictly inside a solid
boundary ball, d < R_ball - 1e-4, R_ball = 0.5*ps = 0.004 per the engine default
``boundary_ball_radius = 0.5*particle_size``) vs classifier FALSE POSITIVE, with numeric evidence.

Sub-modes (argv[1]):
  static : no simulation.  Geometry-only audit of the two recorded events from the official CSV
           + the official ball assets / pose helpers (same-source).  Writes
           videos/dbg_mf16_jug_event_forensic_static.json.
  main   : full 20 s rerun at the exact diag17 configuration (dt=1/60, substeps=8, smoke schedule,
           smooth_a2=True, adh off, no video).  Replicates the official tunnel classifiers
           verbatim, tracks both target pids every frame, and runs a classifier-INDEPENDENT
           containment audit of ALL 86088 fluid particles against BOTH boundary-ball unions every
           frame.  Writes videos/dbg_mf16_jug_event_forensic_main_{log,audit.csv,targets.csv,events.csv}.
  mainrep: identical configuration to main (a non-determinism replica); additionally logs the
           position / nearest-ball detail of every contained particle.
           Writes videos/dbg_mf16_jug_event_forensic_mainrep_*.{log,csv,json}.
  fine   : substep-granular replay: dt=1/480, substeps=1 (every solver substep observable),
           0 -> 19 s.  NOTE: pose updates then happen at 480 Hz (diag17 staged one pose per 60 Hz
           frame and interpolated across its 8 substeps), so trajectories DIVERGE from diag17 --
           this is mechanism evidence, not bit-reproduction.  Logs the same per-pid quantities per
           substep and runs the containment audit inside t in [8.0,9.2], [16.3,17.5],
           [18.25,18.62] (the diag16 jug return-sweep window).  Writes
           videos/dbg_mf16_jug_event_forensic_fine_{log,audit.csv,targets.csv,events.csv}.
  all    : run static, then main, then fine (separate processes would be needed for main+fine
           together because gs.init runs once per process; use subcommands instead).

Nothing under multiflow/genesis-world/ and no existing script is modified.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(SCRIPT_DIR)  # multiflow/
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import mf16_pour_overflow as M  # noqa: E402  (asserts the multiflow genesis copy; prints gs.__file__)

gs = M.gs
VIDEOS_DIR = M.VIDEOS_DIR
TAG = "dbg_mf16_jug_event_forensic"
PS = M.PS_FULL
R_BALL = 0.5 * PS  # engine pbd_solver: boundary_ball_radius = 0.5*particle_size when unset
CONTAIN_TOL = R_BALL - 1e-4

JUG_PID = 58375   # milk-range global fluid index (n_coffee = 49980 -> milk local 8395)
MUG_PID = 64992   # global fluid index (milk local 15012)

OFFICIAL_EVENTS_CSV = os.path.join(VIDEOS_DIR, "mf16_smoke_diag17_jug_statefix_h20_tunnel_events.csv")

FINE_WINDOWS = [(8.0, 9.2), (16.3, 17.5), (18.25, 18.62)]  # diag16 return sweep in the last one


def out_path(suffix):
    return os.path.join(VIDEOS_DIR, f"{TAG}_{suffix}")


# --------------------------------------------------------------------------- shared helpers
def load_balls():
    """Official same-source ball assets + per-ball face/layer provenance."""
    mug_local, mug_meta = M.load_ball_asset(M.MUG_NPZ, "mug", False)
    jug_local, jug_meta = M.load_ball_asset(M.JUG_NPZ, "jug", True)
    with np.load(M.MUG_NPZ) as d:
        mug_face, mug_layer = d["face"].astype(str), d["layer"].astype(np.int8)
    with np.load(M.JUG_NPZ) as d:
        jug_face, jug_layer = d["face"].astype(str), d["layer"].astype(np.int8)
    return (mug_local, mug_face, mug_layer), (jug_local, jug_face, jug_layer)


def set_smoke_timeline():
    """Reproduce the module-global timeline that the official --smoke branch installs."""
    M.T_LIFT0, M.T_LIFT1 = 2.0, 3.5
    M.T_TILT0, M.T_TILT1 = 3.5, 6.0
    M.T_TRICKLE, M.TRICKLE_RAMP = 8.0, 1.0
    M.T_DOME_STOP, M.STOP_RAMP = 11.0, 1.0
    M.STOP_RETREAT_X, M.STOP_LIFT_Z = 0.04, 0.02
    M.T_OVERFILL, M.OVERFILL_RAMP = 14.25, 0.75
    M.T_BACK, M.T_PARK0, M.T_PARK1 = 17.5, 17.5, 19.0
    M.TILT_MAX_DEG, M.TILT_TRICKLE_DEG = 72.0, 78.0
    M.TILT_DOME_HOLD_DEG, M.TILT_OVERFILL_DEG = 65.0, 112.0
    M.PIVOT_X, M.PIVOT_Z = 0.09, 0.38
    M.DUR = 20.0


def containment_counts(pos, mug_tree, jug_tree):
    """Class-independent strict-solid-union containment for all fluid rows.

    Containment = centre distance < R_BALL - 1e-4 inside ANY boundary ball of either set.
    Returns (n_mug, n_jug, mug_pids, jug_pids, d_mug_min, d_jug_min).
    """
    d_mug = mug_tree.query(pos, k=1, workers=-1)[0]
    d_jug = jug_tree.query(pos, k=1, workers=-1)[0]
    cm, cj = d_mug < CONTAIN_TOL, d_jug < CONTAIN_TOL
    return int(cm.sum()), int(cj.sum()), np.flatnonzero(cm), np.flatnonzero(cj), \
        float(d_mug.min()), float(d_jug.min())


def track_row(t, i, pid, pos, vel, jl, mug_rz, mug_ball, jug_ball, mug_face, jug_face,
              mug_layer, jug_layer, contained_mug, contained_jug):
    az_j = float(np.degrees(np.arctan2(jl[pid, 1], jl[pid, 0])))
    r_j = float(np.linalg.norm(jl[pid, :2]))
    dj, ij = jug_ball
    dm, im = mug_ball
    return [
        i, t, int(pid),
        float(pos[pid, 0]), float(pos[pid, 1]), float(pos[pid, 2]),
        float(vel[pid, 0]), float(vel[pid, 1]), float(vel[pid, 2]),
        float(np.linalg.norm(vel[pid])),
        r_j, float(jl[pid, 2]), az_j,
        dj, int(ij), str(jug_face[ij]), int(jug_layer[ij]),
        float(mug_rz[pid, 0]), float(mug_rz[pid, 1]),
        float(np.degrees(np.arctan2(pos[pid, 1], pos[pid, 0]))),
        dm, int(im), str(mug_face[im]), int(mug_layer[im]),
        bool(contained_mug), bool(contained_jug),
    ]


TRACK_HEADER = [
    "frame", "t", "pid", "x", "y", "z", "vx", "vy", "vz", "speed",
    "jug_r", "jug_z", "jug_az_deg", "jug_nearest_d", "jug_nearest_idx", "jug_nearest_face",
    "jug_nearest_layer", "mug_r", "mug_z", "mug_az_deg", "mug_nearest_d", "mug_nearest_idx",
    "mug_nearest_face", "mug_nearest_layer", "contained_mug", "contained_jug",
]

EVENT_FIELDS = [
    "body", "pid", "t", "phase", "kind", "prev_r", "prev_z", "prev_az_deg", "cur_r", "cur_z",
    "cur_az_deg", "cross_z", "speed", "cleared_before", "resident", "entered",
    "nearest_ball_d", "nearest_ball_idx", "nearest_face", "nearest_layer",
    "nearest_ball_r", "nearest_ball_z", "nearest_ball_az_deg", "spout_sector",
]


# --------------------------------------------------------------------------- static mode
def run_static():
    (mug_local, mug_face, mug_layer), (jug_local, jug_face, jug_layer) = load_balls()
    jug_tree = cKDTree(jug_local.astype(np.float64))
    mug_tree = cKDTree(mug_local.astype(np.float64))

    rows = list(csv.DictReader(open(OFFICIAL_EVENTS_CSV)))
    assert len(rows) == 2, f"expected 2 official events, got {len(rows)}"
    report = {"official_events_csv": OFFICIAL_EVENTS_CSV, "R_ball": R_BALL,
              "contain_tol": CONTAIN_TOL, "events": []}

    def rzaz(r, z, az_deg):
        a = np.radians(az_deg)
        return np.array([r * np.cos(a), r * np.sin(a), z])

    for row in rows:
        body = row["body"]
        pid = int(row["pid"])
        prev = rzaz(float(row["prev_r"]), float(row["prev_z"]), float(row["prev_az_deg"]))
        cur = rzaz(float(row["cur_r"]), float(row["cur_z"]), float(row["cur_az_deg"]))
        tree, local, face, layer = (
            (jug_tree, jug_local, jug_face, jug_layer) if body == "jug"
            else (mug_tree, mug_local, mug_face, mug_layer))
        # dense chord scan for strict containment + min centre-to-ball distance
        us = np.linspace(0.0, 1.0, 401)
        chord = prev[None, :] + us[:, None] * (cur - prev)[None, :]
        d = tree.query(chord, k=1, workers=-1)[0]
        imin = int(np.argmin(d))
        dp, ip = tree.query(prev, k=1)
        dc, ic = tree.query(cur, k=1)
        ev = {
            "body": body, "pid": pid, "kind": row["kind"], "t": float(row["t"]),
            "prev_d_nearest": float(dp), "cur_d_nearest": float(dc),
            "chord_min_d": float(d.min()), "chord_min_u": float(us[imin]),
            "ever_contained_d<%g" % CONTAIN_TOL: bool(d.min() < CONTAIN_TOL),
            "prev_ball_idx": int(ip), "cur_ball_idx": int(ic),
            "prev_ball": local[ip].tolist(), "prev_ball_face": str(face[ip]),
            "prev_ball_layer": int(layer[ip]),
            "cross_z": float(row["cross_z"]),
            "clearance_plane": M.Z_RIM + 0.5 * PS,
            "shortfall_below_plane": (M.Z_RIM + 0.5 * PS) - float(row["cross_z"]),
            "speed": float(row["speed"]),
            "cleared_before": row["cleared_before"] == "True",
            "resident": row["resident"] == "True",
            "entered": row["entered"] == "True",
        }
        if body == "jug":
            # rim / outer-wall top profile around the particle azimuth (spout sector)
            azb = np.degrees(np.arctan2(local[:, 1], local[:, 0]))
            near = (np.abs(((azb - 180.0 + 180.0) % 360.0) - 180.0) <= 3.0)
            for f in ("rim_face", "outer_wall", "inner_wall"):
                sel = near & (face == f) & (layer == 0)
                ev[f"max_solid_top_{f}_L0_spout_sector"] = float(local[sel, 2].max() + R_BALL)
            sel = (face == "rim_face") & (layer == 0)
            ev["max_solid_top_rim_L0_any_az"] = float(local[sel, 2].max() + R_BALL)
            ev["min_particle_z_chord"] = float(chord[:, 2].min())
            ev["vertical_gap_above_local_solid_top"] = \
                float(chord[:, 2].min() - ev["max_solid_top_rim_face_L0_spout_sector"])
            # spout asset facts (jug spout params come from gen_jug defaults, mirrored in npz meta)
            jug_meta = json.loads(str(np.load(M.JUG_NPZ)["meta"].item()))
            ev["spout_params"] = jug_meta["geometry"]["spout"]
            ev["proxy_outer_r"] = M.R_JUG + M.T_JUG_WALL + 0.5 * PS
        else:
            ev["inner_proxy_r"] = M.R_IN - 0.5 * PS
            ev["inward_cross_m"] = float(row["prev_r"]) - float(row["cur_r"])
            ev["max_solid_top_rim_L0"] = float(
                local[(face == "rim_face") & (layer == 0), 2].max() + R_BALL)
            ev["max_solid_top_inner_wall_L0"] = float(
                local[(face == "inner_wall") & (layer == 0), 2].max() + R_BALL)
            ev["particle_z_vs_rim_top"] = float(row["cur_z"]) - ev["max_solid_top_rim_L0"]
            ev["classifier_note"] = (
                "mug_wall_inbound_entered |= (mug_resident & ~mug_any_tunnel & outer_in & "
                "(cross_outer_in_z < legal_outer_z)) and frame_mug_inbound_tunnel_now both LACK "
                "~mug_cleared (mf16_pour_overflow.py:~1100,~1107), unlike the jug side which has "
                "~jug_cleared (~967,~974): a legally cleared particle re-entering the r=0.146 "
                "inner proxy below 0.320 is flagged wall_inbound.")
        report["events"].append(ev)

    with open(out_path("static.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"static report: {out_path('static.json')}")


# --------------------------------------------------------------------------- simulation modes
def build_scene(dt, substeps, settled_path=None):
    """Mirror the official scene construction (diag17 configuration), cameras excluded."""
    ps = PS
    origin_0 = np.array([M.PX, 0.0, M.PZ])
    settled_path = settled_path or os.path.join(VIDEOS_DIR, "mf16_settled.npz")
    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, gravity=(0.0, 0.0, -9.81)),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_pbd=False),
        rigid_options=gs.options.RigidOptions(enable_collision=False, gravity=(0.0, 0.0, 0.0)),
        pbd_options=gs.options.PBDOptions(
            particle_size=ps,
            lower_bound=tuple(M.LOWER_BOUND),
            upper_bound=tuple(M.UPPER_BOUND),
            boundary_ball_sets=((M.MUG_NPZ, False), (M.JUG_NPZ, True)),
            boundary_ball_initial_poses=(
                ((0.0, 0.0, 0.0), tuple(M.IDENTITY_QUAT)),
                (tuple(origin_0), tuple(M.IDENTITY_QUAT)),
            ),
            boundary_ball_mass_weight_enabled=False,
            boundary_ball_smooth_barrel=(
                0, 0.0, 0.0, M.R_IN, M.R_OUT, 0.0, M.Z_RIM,
                np.pi, np.deg2rad(35.0), 0.08, 0.24, 0.002,
            ),
            boundary_plane=(M.Z_TABLE,),
            max_density_solver_iterations=20,
            max_viscosity_solver_iterations=1,
            density_lambda_epsilon=0.1,
            diffusion_coeff=0.01,
            surface_tension_enabled=True,
            st_compliance=0.7,
            st_surface_density_factor=1.0,
            st_distance_enabled=True,
            st_distance_compliance=40.0,
            st_max_surface_neighbors=256,
            velocity_damping=1.0,
            wall_adhesion_enabled=False,   # diag17: --no-adh -> engine wall_friction forced to 0
            wall_adhesion_compliance=20.0,
            wall_friction=0.0,             # diag17: --wfric 0
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(file=M.MUG_OBJ, pos=(0.0, 0.0, 0.0), fixed=True,
                             collision=False, decimate=False),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.85, 0.85, 0.9, 0.35), opacity=0.35),
    )
    jug = scene.add_entity(
        morph=gs.morphs.Mesh(file=M.JUG_OBJ, pos=tuple(origin_0), euler=(0.0, 0.0, 0.0),
                             fixed=True, collision=False, decimate=False),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.9, 0.92, 0.95, 0.45), opacity=0.45),
    )
    scene.add_entity(
        morph=gs.morphs.Box(size=(2.6, 1.6, 0.04), pos=(0.35, 0.0, M.Z_TABLE - 0.02), fixed=True),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.55, 0.55, 0.58, 1.0), opacity=1.0),
    )
    coffee = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=1.0,
            density_relaxation=0.2, viscosity_relaxation=0.01),
        morph=gs.morphs.Cylinder(
            radius=float(np.load(settled_path)["coffee_r"]),
            height=float(np.load(settled_path)["coffee_h0"]),
            pos=(0.0, 0.0, M.Z_FLOOR + M.S0 + 0.5 * float(np.load(settled_path)["coffee_h0"]))),
    )
    milk = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=0.0,
            density_relaxation=0.2, viscosity_relaxation=0.01),
        morph=gs.morphs.Cylinder(
            radius=float(np.load(settled_path)["milk_r"]),
            height=float(np.load(settled_path)["milk_h0"]),
            pos=(M.PX, 0.0, M.PZ + M.Z_FLOOR + M.S0 + 0.5 * float(np.load(settled_path)["milk_h0"]))),
    )
    scene.build()
    return scene, coffee, milk, jug


def run_sim(mode):
    """mode: 'main' (diag17 replica, per-frame audit) or 'fine' (480 Hz, window audits)."""
    set_smoke_timeline()
    if mode in ("main", "mainrep"):
        dt, substeps, dur = M.DT, 8, 20.0
        audit_every_frame = True
    else:
        dt, substeps, dur = 1.0 / 480.0, 1, 19.0
        audit_every_frame = False

    log_path = out_path(f"{mode}.log")
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"== dbg_mf16_jug_event_forensic mode={mode} ==")
    log(f"genesis loaded from: {gs.__file__}")
    assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), "Wrong genesis!"
    log(f"dt={dt:.7f} substeps={substeps} dur={dur} R_ball={R_BALL} contain_tol={CONTAIN_TOL:.6f}")
    if mode == "fine":
        log("NOTE: pose updates occur at 480 Hz here (diag17 staged one pose per 60 Hz frame); "
            "trajectories diverge from diag17.  Mechanism evidence, not bit-reproduction.")

    (mug_local, mug_face, mug_layer), (jug_local, jug_face, jug_layer) = load_balls()
    mug_ball_tree = cKDTree(mug_local.astype(np.float64))
    jug_ball_tree = cKDTree(jug_local.astype(np.float64))

    scene, coffee, milk, jug = build_scene(dt, substeps)
    solver = scene.pbd_solver
    n_fluid = solver._n_entity_particles
    n_coffee = coffee.n_particles
    n_milk = milk.n_particles
    assert n_coffee + n_milk == n_fluid == 86088
    assert solver._has_boundary_ball_smooth_barrel
    assert solver._boundary_ball_mass_weight_enabled is False
    solver.reset_boundary_ball_smooth_barrel_history()

    # ---- settled-state install (official procedure) ----
    settled_path = os.path.join(VIDEOS_DIR, "mf16_settled.npz")
    dat = np.load(settled_path)
    settled_pos = dat["pos"].astype(np.float32)
    settled_vel = dat["vel"].astype(np.float32)
    settled_c = dat["c"].astype(np.float32)
    coffee.set_particles_pos(settled_pos[:n_coffee, 0])
    milk.set_particles_pos(settled_pos[n_coffee:, 0])
    coffee.set_particles_vel(settled_vel[:n_coffee, 0])
    milk.set_particles_vel(settled_vel[n_coffee:, 0])
    c_full = solver.particles.c.to_numpy()
    c_full[:n_fluid] = settled_c
    solver.particles.c.from_numpy(c_full)
    log(f"installed settled state: n_fluid={n_fluid} n_coffee={n_coffee} n_milk={n_milk}")

    # ---- classifier initial state (verbatim from mf16_pour_overflow.py main loop) ----
    prev_pos = solver.particles.pos.to_numpy()[:n_fluid, 0, :].copy()
    prev_origin, prev_quat, _, _, _ = M.jug_pose(0.0)
    prev_jlocal = M.inverse_transform_points(prev_pos, prev_origin, prev_quat)
    jug_resident = np.zeros(n_fluid, dtype=bool)
    jug_resident[n_coffee:] = (
        (np.linalg.norm(prev_jlocal[n_coffee:, :2], axis=1) <= M.R_JUG + PS)
        & (prev_jlocal[n_coffee:, 2] >= M.Z_FLOOR - PS)
        & (prev_jlocal[n_coffee:, 2] <= M.Z_RIM + PS)
    )
    jug_cleared = np.zeros(n_fluid, dtype=bool)
    jug_wall_entered = (
        jug_resident & (np.linalg.norm(prev_jlocal[:, :2], axis=1) > M.R_JUG - 0.5 * PS)
        & (prev_jlocal[:, 2] <= M.Z_RIM - 0.5 * PS)
    )
    jug_wall_inbound_entered = np.zeros(n_fluid, dtype=bool)
    jug_bottom_entered = jug_resident & (prev_jlocal[:, 2] < M.Z_FLOOR + 0.5 * PS)
    jug_wall_tunnel = np.zeros(n_fluid, dtype=bool)
    jug_bottom_tunnel = np.zeros(n_fluid, dtype=bool)
    mug_resident = np.zeros(n_fluid, dtype=bool)
    mug_resident[:n_coffee] = True
    mug_rim_crossed = np.zeros(n_fluid, dtype=bool)
    mug_cleared = np.zeros(n_fluid, dtype=bool)
    prev_r_mug0 = np.linalg.norm(prev_pos[:, :2], axis=1)
    mug_wall_entered = (
        mug_resident & (prev_r_mug0 > M.R_IN - 0.5 * PS)
        & (prev_pos[:, 2] <= M.Z_RIM - 0.5 * PS)
    )
    mug_wall_inbound_entered = np.zeros(n_fluid, dtype=bool)
    mug_bottom_entered = mug_resident & (prev_pos[:, 2] < M.Z_FLOOR + 0.5 * PS)
    mug_wall_tunnel = np.zeros(n_fluid, dtype=bool)
    mug_bottom_tunnel = np.zeros(n_fluid, dtype=bool)

    n_frames = int(round(dur / dt))
    event_records = []
    target_pids = [JUG_PID, MUG_PID]

    audit_csv = open(out_path(f"{mode}_audit.csv"), "w", newline="")
    audit_w = csv.writer(audit_csv)
    audit_w.writerow(["frame", "t", "n_contained_mug", "n_contained_jug", "min_d_mug",
                      "min_d_jug", "contained_pids_mug", "contained_pids_jug"])
    tgt_csv = open(out_path(f"{mode}_targets.csv"), "w", newline="")
    tgt_w = csv.writer(tgt_csv)
    tgt_w.writerow(TRACK_HEADER)
    detail_csv = open(out_path(f"{mode}_contain_detail.csv"), "w", newline="")
    detail_w = csv.writer(detail_csv)
    detail_w.writerow(["frame", "t", "body", "pid", "x", "y", "z",
                       "ball_idx", "ball_lx", "ball_ly", "ball_lz", "d"])

    wall0 = time.time()
    for i in range(n_frames + 1):
        t = i * dt
        origin, quat, axis, th, _ = M.jug_pose(t)
        jug_world_expected = M.transform_points(jug_local, origin, quat)
        solver.set_boundary_ball_pose(1, origin, quat)
        jug.set_pos(origin, relative=True, zero_velocity=True)
        jug.set_quat(quat, relative=True, zero_velocity=True)

        frame_t0 = time.time()
        if i > 0:
            scene.step()

        pos_all = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
        vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
        nan_count = int((~np.isfinite(pos_all)).sum() + (~np.isfinite(vel)).sum())
        assert nan_count == 0, f"NaN at t={t:.3f}"
        safe = np.nan_to_num(pos_all, nan=0.0, posinf=0.0, neginf=0.0)
        safe_v = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        active_fluid = int(solver.particles_ng.active.to_numpy()[:n_fluid, 0].sum())
        assert active_fluid == n_fluid, f"active fluid changed at t={t:.3f}"

        # pose proof (same assertion as the official loop)
        jug_start, jug_end, _ = solver.boundary_ball_ranges[1]
        jug_ball_now = solver.particles.pos.to_numpy()[jug_start:jug_end, 0, :]
        ball_pose_err = float(np.max(np.abs(jug_ball_now - jug_world_expected)))
        assert ball_pose_err < 5e-7, f"jug ball pose mismatch at t={t:.3f}"

        solver_rim_staged, solver_direct_tunnel = solver.boundary_ball_smooth_barrel_history()

        # ================= jug classifier (verbatim logic) =================
        cur_jlocal = M.inverse_transform_points(safe, origin, quat)
        pr, cr = np.linalg.norm(prev_jlocal[:, :2], axis=1), np.linalg.norm(cur_jlocal[:, :2], axis=1)
        pzj, czj = prev_jlocal[:, 2], cur_jlocal[:, 2]
        jug_clear_z = M.Z_RIM + 0.5 * PS
        jug_wall_z_lo = -0.5 * PS
        jug_wall_z_hi = M.Z_RIM + 0.5 * PS
        up = (czj > pzj) & (pzj < jug_clear_z) & (czj >= jug_clear_z)
        u = np.clip((jug_clear_z - pzj) / np.maximum(czj - pzj, 1e-12), 0.0, 1.0)
        cross_xy = prev_jlocal[:, :2] + u[:, None] * (cur_jlocal[:, :2] - prev_jlocal[:, :2])
        legal_jug_cross = up & (np.linalg.norm(cross_xy, axis=1) <= M.R_JUG + 0.5 * PS)
        jug_any_tunnel = jug_wall_tunnel | jug_bottom_tunnel
        jug_open = jug_resident & ~jug_cleared & ~jug_any_tunnel
        jug_wall_entered |= (
            jug_open & (cr > M.R_JUG - 0.5 * PS)
            & (czj >= jug_wall_z_lo) & (czj < jug_clear_z)
        )
        jug_bottom_entered |= (
            jug_open & (czj < M.Z_FLOOR + 0.5 * PS)
            & (cr <= M.R_JUG + M.T_JUG_WALL + PS)
        )
        jug_outer_r = M.R_JUG + M.T_JUG_WALL + 0.5 * PS
        jug_outer_cross, _, jug_outer_z = M.segment_outward_cylinder_crossing(
            prev_jlocal, cur_jlocal, jug_outer_r)
        jug_outer_in, _, jug_outer_in_z = M.segment_inward_cylinder_crossing(
            prev_jlocal, cur_jlocal, jug_outer_r)
        jug_wall_inbound_entered |= (
            jug_resident & ~jug_cleared & ~jug_any_tunnel & jug_outer_in
            & (jug_outer_in_z >= jug_wall_z_lo) & (jug_outer_in_z <= jug_wall_z_hi)
        )
        jug_inner_in, _, jug_inner_in_z = M.segment_inward_cylinder_crossing(
            prev_jlocal, cur_jlocal, M.R_JUG - 0.5 * PS)
        jug_inbound_tunnel_now = (
            jug_resident & ~jug_cleared & ~jug_any_tunnel & jug_wall_inbound_entered
            & jug_inner_in & (jug_inner_in_z >= jug_wall_z_lo) & (jug_inner_in_z < jug_clear_z)
        )
        jug_high_clear_now = jug_open & (
            legal_jug_cross
            | (jug_outer_cross & (jug_outer_z >= jug_clear_z))
            | ((pr >= jug_outer_r) & (pzj >= jug_clear_z))
        )
        jug_wall_tunnel_now = (
            jug_open & ~jug_high_clear_now & jug_wall_entered
            & (
                (jug_outer_cross & (jug_outer_z >= jug_wall_z_lo) & (jug_outer_z < jug_clear_z))
                | (~jug_outer_cross & (cr >= jug_outer_r)
                   & (czj >= jug_wall_z_lo) & (czj < jug_clear_z))
            )
        )
        jug_bottom_cross, _, jug_bottom_point = M.segment_downward_plane_crossing(
            prev_jlocal, cur_jlocal, -0.5 * PS)
        jug_bottom_cross_r = np.linalg.norm(jug_bottom_point[:, :2], axis=1)
        jug_bottom_tunnel_now = (
            jug_open & jug_bottom_entered
            & (
                (jug_bottom_cross & (jug_bottom_cross_r <= M.R_JUG + M.T_JUG_WALL + PS))
                | (~jug_bottom_cross & (czj < -0.5 * PS)
                   & (cr <= M.R_JUG + M.T_JUG_WALL + PS))
            )
        )
        new_jug_wall_tunnel = (jug_wall_tunnel_now | jug_inbound_tunnel_now) & ~jug_wall_tunnel
        for pid in np.flatnonzero(new_jug_wall_tunnel):
            is_inbound = bool(jug_inbound_tunnel_now[pid])
            query_pt = cur_jlocal[pid]
            ball_d, ball_idx = jug_ball_tree.query(query_pt, k=1, workers=-1)
            ball_d, ball_idx = float(ball_d), int(ball_idx)
            prev_az = float(np.degrees(np.arctan2(prev_jlocal[pid, 1], prev_jlocal[pid, 0])))
            cur_az = float(np.degrees(np.arctan2(cur_jlocal[pid, 1], cur_jlocal[pid, 0])))
            ball_az = float(np.degrees(np.arctan2(jug_local[ball_idx, 1], jug_local[ball_idx, 0])))
            event = {
                "body": "jug", "pid": int(pid), "t": float(t), "phase": int(M.phase_of(t)),
                "kind": "wall_inbound" if is_inbound else "wall",
                "prev_r": float(pr[pid]), "prev_z": float(pzj[pid]), "prev_az_deg": prev_az,
                "cur_r": float(cr[pid]), "cur_z": float(czj[pid]), "cur_az_deg": cur_az,
                "cross_z": float(jug_outer_in_z[pid] if is_inbound else jug_outer_z[pid]),
                "speed": float(np.linalg.norm(safe_v[pid])),
                "cleared_before": bool(jug_cleared[pid]), "resident": bool(jug_resident[pid]),
                "entered": bool(jug_wall_entered[pid]), "nearest_ball_d": ball_d,
                "nearest_ball_idx": ball_idx, "nearest_face": str(jug_face[ball_idx]),
                "nearest_layer": int(jug_layer[ball_idx]),
                "nearest_ball_r": float(np.linalg.norm(jug_local[ball_idx, :2])),
                "nearest_ball_z": float(jug_local[ball_idx, 2]), "nearest_ball_az_deg": ball_az,
                "spout_sector": bool(abs(((cur_az - 180.0 + 180.0) % 360.0) - 180.0) <= 20.0),
                "run_mode": mode,
            }
            event_records.append(event)
            log(f"TUNNEL_JUG id={pid} t={t:.6f} kind={event['kind']} "
                f"cross_z={event['cross_z']:.6f} prev_rz=({event['prev_r']:.6f},{event['prev_z']:.6f}) "
                f"cur_rz=({event['cur_r']:.6f},{event['cur_z']:.6f}) az=({prev_az:.2f},{cur_az:.2f}) "
                f"speed={event['speed']:.6f} cleared_before={event['cleared_before']} "
                f"ball_d={ball_d:.6f} ball={ball_idx}:{event['nearest_face']}:{event['nearest_layer']} "
                f"entered={event['entered']} spout_sector={event['spout_sector']}")
        for pid in np.flatnonzero(jug_bottom_tunnel_now & ~jug_bottom_tunnel):
            ball_d, ball_idx = jug_ball_tree.query(cur_jlocal[pid], k=1, workers=-1)
            ball_d, ball_idx = float(ball_d), int(ball_idx)
            event_records.append({
                "body": "jug", "pid": int(pid), "t": float(t), "phase": int(M.phase_of(t)),
                "kind": "bottom",
                "prev_r": float(pr[pid]), "prev_z": float(pzj[pid]),
                "prev_az_deg": float(np.degrees(np.arctan2(prev_jlocal[pid, 1], prev_jlocal[pid, 0]))),
                "cur_r": float(cr[pid]), "cur_z": float(czj[pid]),
                "cur_az_deg": float(np.degrees(np.arctan2(cur_jlocal[pid, 1], cur_jlocal[pid, 0]))),
                "cross_z": float("nan"), "speed": float(np.linalg.norm(safe_v[pid])),
                "cleared_before": bool(jug_cleared[pid]), "resident": bool(jug_resident[pid]),
                "entered": bool(jug_bottom_entered[pid]), "nearest_ball_d": float(ball_d),
                "nearest_ball_idx": ball_idx, "nearest_face": str(jug_face[ball_idx]),
                "nearest_layer": int(jug_layer[ball_idx]),
                "nearest_ball_r": float(np.linalg.norm(jug_local[ball_idx, :2])),
                "nearest_ball_z": float(jug_local[ball_idx, 2]),
                "nearest_ball_az_deg": float(np.degrees(np.arctan2(jug_local[ball_idx, 1], jug_local[ball_idx, 0]))),
                "spout_sector": False, "run_mode": mode,
            })
            log(f"TUNNEL_JUG id={pid} t={t:.6f} kind=bottom (recorded)")
        jug_wall_tunnel |= jug_wall_tunnel_now | jug_inbound_tunnel_now
        jug_bottom_tunnel |= jug_bottom_tunnel_now
        jug_any_tunnel = jug_wall_tunnel | jug_bottom_tunnel
        jug_cleared &= ~jug_any_tunnel
        jug_cleared |= jug_resident & ~jug_any_tunnel & jug_high_clear_now
        assert not np.any(jug_cleared & jug_any_tunnel), "jug clear/tunnel histories must be exclusive"

        # ================= mug classifier (verbatim logic) =================
        prev_r_mug = np.linalg.norm(prev_pos[:, :2], axis=1)
        cur_r_mug = np.linalg.norm(safe[:, :2], axis=1)
        z_all = safe[:, 2]
        inside_mug_now = (
            (cur_r_mug <= M.R_IN - 0.5 * PS) & (z_all >= M.Z_FLOOR) & (z_all <= M.Z_RIM)
        )
        mug_resident |= inside_mug_now
        mug_any_tunnel = mug_wall_tunnel | mug_bottom_tunnel
        mug_open = mug_resident & ~mug_cleared & ~mug_any_tunnel
        mug_wall_entered |= (
            mug_open & (cur_r_mug > M.R_IN - 0.5 * PS)
            & (z_all >= 0.0) & (z_all < M.Z_RIM + M.MUG_OUTER_CLEAR_Z_PS * PS)
        )
        mug_bottom_entered |= (
            mug_open & (z_all < M.Z_FLOOR + 0.5 * PS) & (cur_r_mug <= M.R_OUT + PS)
        )
        outer_clear_r = M.R_OUT + M.MUG_OUTER_CLEAR_PS * PS
        outer_cross, outer_u, cross_outer_z = M.segment_outward_cylinder_crossing(
            prev_pos, safe, outer_clear_r)
        legal_outer_z = M.Z_RIM + M.MUG_OUTER_CLEAR_Z_PS * PS
        outer_in, _, cross_outer_in_z = M.segment_inward_cylinder_crossing(
            prev_pos, safe, outer_clear_r)
        mug_wall_inbound_entered |= (
            mug_resident & ~mug_any_tunnel & outer_in & (cross_outer_in_z < legal_outer_z)
        )
        inner_safe_r = M.R_IN - 0.5 * PS
        inner_in, inner_in_u, cross_inner_in_z = M.segment_inward_cylinder_crossing(
            prev_pos, safe, inner_safe_r)
        frame_mug_inbound_tunnel_now = (
            mug_resident & ~mug_any_tunnel & mug_wall_inbound_entered
            & inner_in & (cross_inner_in_z < legal_outer_z)
        )
        frame_mug_wall_tunnel_now = (
            mug_open & mug_wall_entered
            & (
                (outer_cross & (cross_outer_z >= 0.0) & (cross_outer_z < legal_outer_z))
                | (~outer_cross & (cur_r_mug >= outer_clear_r)
                   & (z_all >= 0.0) & (z_all < legal_outer_z))
            )
        )
        staged_top_reentry = solver_rim_staged & (cross_inner_in_z >= M.Z_RIM - 1e-6)
        mug_inbound_tunnel_now = frame_mug_inbound_tunnel_now & ~staged_top_reentry
        mug_wall_tunnel_now = (frame_mug_wall_tunnel_now & ~solver_rim_staged) | solver_direct_tunnel
        mug_bottom_cross, _, mug_bottom_point = M.segment_downward_plane_crossing(
            prev_pos, safe, -0.5 * PS)
        mug_bottom_cross_r = np.linalg.norm(mug_bottom_point[:, :2], axis=1)
        mug_bottom_tunnel_now = (
            mug_open & mug_bottom_entered
            & (
                (mug_bottom_cross & (mug_bottom_cross_r <= M.R_OUT + PS))
                | (~mug_bottom_cross & (z_all < -0.5 * PS) & (cur_r_mug <= M.R_OUT + PS))
            )
        )
        new_mug_wall_tunnel = (mug_wall_tunnel_now | mug_inbound_tunnel_now) & ~mug_wall_tunnel
        for pid in np.flatnonzero(new_mug_wall_tunnel):
            is_solver_direct = bool(solver_direct_tunnel[pid])
            is_inbound = bool(mug_inbound_tunnel_now[pid]) and not is_solver_direct
            cp = (prev_pos[pid] + inner_in_u[pid] * (safe[pid] - prev_pos[pid])) \
                if is_inbound else ((prev_pos[pid] + outer_u[pid] * (safe[pid] - prev_pos[pid]))
                                    if outer_cross[pid] else safe[pid])
            ball_d, ball_idx = mug_ball_tree.query(cp, k=1, workers=-1)
            ball_d, ball_idx = float(ball_d), int(ball_idx)
            prev_az = float(np.degrees(np.arctan2(prev_pos[pid, 1], prev_pos[pid, 0])))
            cur_az = float(np.degrees(np.arctan2(safe[pid, 1], safe[pid, 0])))
            event = {
                "body": "mug", "pid": int(pid), "t": float(t), "phase": int(M.phase_of(t)),
                "kind": "solver_direct" if is_solver_direct else ("wall_inbound" if is_inbound else "wall"),
                "prev_r": float(prev_r_mug[pid]), "prev_z": float(prev_pos[pid, 2]),
                "prev_az_deg": prev_az,
                "cur_r": float(cur_r_mug[pid]), "cur_z": float(z_all[pid]), "cur_az_deg": cur_az,
                "cross_z": float(cross_inner_in_z[pid] if is_inbound else cross_outer_z[pid]),
                "speed": float(np.linalg.norm(safe_v[pid])),
                "cleared_before": bool(mug_cleared[pid]), "resident": bool(mug_resident[pid]),
                "entered": bool(mug_wall_entered[pid]), "nearest_ball_d": ball_d,
                "nearest_ball_idx": ball_idx, "nearest_face": str(mug_face[ball_idx]),
                "nearest_layer": int(mug_layer[ball_idx]),
                "nearest_ball_r": float(np.linalg.norm(mug_local[ball_idx, :2])),
                "nearest_ball_z": float(mug_local[ball_idx, 2]),
                "nearest_ball_az_deg": float(np.degrees(np.arctan2(mug_local[ball_idx, 1], mug_local[ball_idx, 0]))),
                "spout_sector": False, "run_mode": mode,
            }
            event_records.append(event)
            log(f"TUNNEL_MUG id={pid} t={t:.6f} kind={event['kind']} "
                f"cross_z={event['cross_z']:.6f} prev_rz=({event['prev_r']:.6f},{event['prev_z']:.6f}) "
                f"cur_rz=({event['cur_r']:.6f},{event['cur_z']:.6f}) speed={event['speed']:.6f} "
                f"ball_d={ball_d:.6f} cleared_before={event['cleared_before']} "
                f"entered={event['entered']} rim_path={bool(mug_rim_crossed[pid])}")
        for pid in np.flatnonzero(mug_bottom_tunnel_now & ~mug_bottom_tunnel):
            ball_d, ball_idx = mug_ball_tree.query(safe[pid], k=1, workers=-1)
            event_records.append({
                "body": "mug", "pid": int(pid), "t": float(t), "phase": int(M.phase_of(t)),
                "kind": "bottom",
                "prev_r": float(prev_r_mug[pid]), "prev_z": float(prev_pos[pid, 2]),
                "prev_az_deg": float(np.degrees(np.arctan2(prev_pos[pid, 1], prev_pos[pid, 0]))),
                "cur_r": float(cur_r_mug[pid]), "cur_z": float(z_all[pid]),
                "cur_az_deg": float(np.degrees(np.arctan2(safe[pid, 1], safe[pid, 0]))),
                "cross_z": float("nan"), "speed": float(np.linalg.norm(safe_v[pid])),
                "cleared_before": bool(mug_cleared[pid]), "resident": bool(mug_resident[pid]),
                "entered": bool(mug_bottom_entered[pid]), "nearest_ball_d": float(ball_d),
                "nearest_ball_idx": int(ball_idx), "nearest_face": str(mug_face[ball_idx]),
                "nearest_layer": int(mug_layer[ball_idx]),
                "nearest_ball_r": float(np.linalg.norm(mug_local[ball_idx, :2])),
                "nearest_ball_z": float(mug_local[ball_idx, 2]),
                "nearest_ball_az_deg": float(np.degrees(np.arctan2(mug_local[ball_idx, 1], mug_local[ball_idx, 0]))),
                "spout_sector": False, "run_mode": mode,
            })
            log(f"TUNNEL_MUG id={pid} t={t:.6f} kind=bottom (recorded)")
        mug_wall_tunnel |= mug_wall_tunnel_now | mug_inbound_tunnel_now
        mug_bottom_tunnel |= mug_bottom_tunnel_now
        mug_any_tunnel = mug_wall_tunnel | mug_bottom_tunnel
        mug_cleared &= ~mug_any_tunnel

        rim_path_r = M.R_IN - M.MUG_RIM_PATH_R_MIN_PS * PS
        inner_cross, _, cross_inner_z = M.segment_outward_cylinder_crossing(prev_pos, safe, rim_path_r)
        upm = (z_all > prev_pos[:, 2]) & (prev_pos[:, 2] < M.Z_RIM) & (z_all >= M.Z_RIM)
        uz = np.clip((M.Z_RIM - prev_pos[:, 2]) / np.maximum(z_all - prev_pos[:, 2], 1e-12), 0.0, 1.0)
        cross_rim_xy = prev_pos[:, :2] + uz[:, None] * (safe[:, :2] - prev_pos[:, :2])
        cross_rim_r = np.linalg.norm(cross_rim_xy, axis=1)
        near_inner_lip_now = (
            (cur_r_mug >= rim_path_r)
            & (cur_r_mug <= M.R_OUT + M.MUG_RIM_PATH_R_MAX_PS * PS)
            & (z_all >= M.Z_RIM)
        )
        rim_plane_event = (
            upm & (cross_rim_r >= rim_path_r)
            & (cross_rim_r <= M.R_OUT + M.MUG_RIM_PATH_R_MAX_PS * PS)
        )
        rim_path_event = (inner_cross & (cross_inner_z >= M.Z_RIM)) | rim_plane_event
        frame_rim_probe = near_inner_lip_now | rim_path_event
        authoritative_rim_event = solver_rim_staged  # smooth_a2 is always on here
        mug_rim_crossed |= mug_resident & ~mug_cleared & ~mug_any_tunnel & authoritative_rim_event
        mug_rim_crossed &= ~mug_any_tunnel
        legal_outer_clear = (
            mug_rim_crossed & ~mug_cleared & ~mug_any_tunnel
            & (
                (outer_cross & (cross_outer_z >= legal_outer_z))
                | (solver_rim_staged & (cur_r_mug >= outer_clear_r))
            )
        )
        mug_cleared |= legal_outer_clear
        assert not np.any(mug_cleared & mug_any_tunnel), "mug clear/tunnel histories must be exclusive"

        # ================= containment audit =================
        in_window = any(w0 - 1e-9 <= t <= w1 + 1e-9 for w0, w1 in FINE_WINDOWS)
        do_audit = audit_every_frame or in_window
        if do_audit:
            jug_tree_now = cKDTree(jug_world_expected)
            n_cm, n_cj, pids_m, pids_j, dmin_m, dmin_j = containment_counts(
                safe, mug_ball_tree, jug_tree_now)
            audit_w.writerow([i, f"{t:.6f}", n_cm, n_cj, f"{dmin_m:.6f}", f"{dmin_j:.6f}",
                              ";".join(str(int(p)) for p in pids_m[:60]),
                              ";".join(str(int(p)) for p in pids_j[:60])])
            if n_cm or n_cj:
                log(f"AUDIT_CONTAIN t={t:.6f} mug={n_cm} jug={n_cj} "
                    f"pids_mug={pids_m[:10].tolist()} pids_jug={pids_j[:10].tolist()}")
                for body, pids, tree_now, local in (
                        ("mug", pids_m, mug_ball_tree, mug_local),
                        ("jug", pids_j, jug_tree_now, jug_local)):
                    if len(pids) == 0:
                        continue
                    dd, ii = tree_now.query(safe[pids], k=1, workers=-1)
                    for k, pid in enumerate(pids):
                        b = local[ii[k]]
                        detail_w.writerow([i, f"{t:.6f}", body, int(pid),
                                           f"{safe[pid,0]:.6f}", f"{safe[pid,1]:.6f}",
                                           f"{safe[pid,2]:.6f}", int(ii[k]),
                                           f"{b[0]:.6f}", f"{b[1]:.6f}", f"{b[2]:.6f}",
                                           f"{dd[k]:.6f}"])

        # ================= target-pid tracking =================
        jl = cur_jlocal
        mug_rz = np.stack([cur_r_mug, z_all], axis=1)
        d_j_t = jug_ball_tree.query(jl[target_pids], k=1, workers=-1)
        d_m_t = mug_ball_tree.query(safe[target_pids], k=1, workers=-1)
        for k, pid in enumerate(target_pids):
            tgt_w.writerow(track_row(
                t, i, pid, safe, safe_v, jl, mug_rz,
                (float(d_m_t[0][k]), int(d_m_t[1][k])),
                (float(d_j_t[0][k]), int(d_j_t[1][k])),
                mug_face, jug_face, mug_layer, jug_layer,
                d_m_t[0][k] < CONTAIN_TOL, d_j_t[0][k] < CONTAIN_TOL))

        prev_pos = safe.copy()
        prev_origin, prev_quat = origin.copy(), quat.copy()
        prev_jlocal = cur_jlocal.copy()

        if mode in ("main", "mainrep") and i % 60 == 0:
            wall = time.time() - wall0
            log(f"t={t:6.2f}s th={np.rad2deg(th):5.1f} "
                f"tunnelJ/M={int(jug_wall_tunnel.sum() + jug_bottom_tunnel.sum())}/"
                f"{int(mug_wall_tunnel.sum() + mug_bottom_tunnel.sum())} "
                f"({wall / max(i, 1) * 1000:.0f} ms/frame avg)")
        elif mode == "fine" and i % 960 == 0:
            wall = time.time() - wall0
            log(f"t={t:6.2f}s ({wall / max(i, 1) * 1000:.1f} ms/step avg)")

    audit_csv.close()
    tgt_csv.close()
    detail_csv.close()

    if event_records:
        with open(out_path(f"{mode}_events.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(event_records[0]))
            w.writeheader()
            w.writerows(event_records)
    summary = {
        "mode": mode, "dt": dt, "substeps": substeps, "dur": dur,
        "n_events": len(event_records),
        "events": event_records,
        "final_tunnel_counts": {
            "jug_wall": int(jug_wall_tunnel.sum()), "jug_bottom": int(jug_bottom_tunnel.sum()),
            "mug_wall": int(mug_wall_tunnel.sum()), "mug_bottom": int(mug_bottom_tunnel.sum()),
        },
        "target_pids": target_pids,
        "target_pid_was_contained_ever": {
            str(JUG_PID): None,  # filled below from targets csv is overkill; computed in analysis
            str(MUG_PID): None,
        },
    }
    with open(out_path(f"{mode}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"mode={mode} done: {len(event_records)} events; "
        f"final tunnel counts J={summary['final_tunnel_counts']['jug_wall']}+"
        f"{summary['final_tunnel_counts']['jug_bottom']} M="
        f"{summary['final_tunnel_counts']['mug_wall']}+"
        f"{summary['final_tunnel_counts']['mug_bottom']}")
    log(f"artifacts: {out_path(mode + '_audit.csv')}, {out_path(mode + '_targets.csv')}, "
        f"{out_path(mode + '_events.csv') if event_records else '(no events csv)'}, "
        f"{out_path(mode + '_summary.json')}")
    logf.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["static", "main", "mainrep", "fine"])
    args = ap.parse_args()
    if args.mode == "static":
        run_static()
    else:
        run_sim(args.mode)


if __name__ == "__main__":
    main()
