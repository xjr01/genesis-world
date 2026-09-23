"""dbg_mf16_jug_history.py -- validation harness S1-S5 for PBDOptions.boundary_ball_jug_history.

The kinematic jug ball set (multiflow MF-16) gains a per-accepted-substep, same-source exit
history (legal staged exits over the rim/spout vs irreversible direct tunnel penetration),
mirroring the static-mug "A2" machinery (``boundary_ball_smooth_barrel``).  This script is the
approval harness for that option.  Modes:

  s3dump  : run a short default-OFF scene (variant ``default`` or ``a2``) and dump the full
            particle state + hashes.  Run with ``--phase pre`` on the pre-change engine and
            ``--phase post`` on the changed engine (plus one extra post run saved as
            ``s3_<variant>_replica.npz``); ``s3cmp`` then proves zero regression by replica
            equivalence (cross-process bit-identity is unattainable on this GPU engine:
            identical runs diverge chaotically from substep 1 -- see the forensic mainrep
            replica -- so pre-vs-post divergence is compared against replica-vs-replica
            divergence, and the deterministic c-field must stay bit-identical).
  s3cmp   : compare the pre/post dumps of both variants (bit identity + hashes) and check the
            disabled-option API surface.
  s1      : static pour stages legally (diag15 pose16 fixture, jug held fixed; milk drains over
            the spout).  Asserts staged >= threshold, zero direct, and spot-checks the diag17
            false-positive class (spout-sector crossings with local z in [0.318, 0.320)).
  s2      : forced penetration flags direct at substep resolution (seeded particle inside the
            jug, outward velocity below the rim); direct is irreversible, staged never washes it.
  s4      : sweep regression -- replay the jug return window 18.0 -> 19.2 s (smoke schedule;
            diag16 false-positive window 18.25-18.62 s inside it) from the t=18 dump state.
            Solver side must show ZERO direct.
  s5      : pose-interpolation unit -- one probe particle held just above the rim in world space
            during the max tilt-rate phase (stop ramp, ~39 deg/s peak).  The solver classification
            must match the SUBSTEP-interpolated pose prediction and differ from the frame-frozen
            pose prediction (the check fails if the kernel used a frame-frozen pose).

Nothing under multiflow/genesis-world/ is modified by this script; engine changes land there
separately.  Artifacts: videos/mf16_dbg_jug_history_*.{json,csv,log,npz}.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(SCRIPT_DIR)  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np  # noqa: E402

import genesis as gs  # noqa: E402
import mf16_pour_overflow as h  # noqa: E402  (asserts the multiflow genesis copy; prints gs path)

VIDEOS_DIR = os.path.join(WORKSPACE, "videos")
TAG = "mf16_dbg_jug_history"

DUMPS = os.path.join(VIDEOS_DIR, "mf16_smoke_diag15_noadh_fric0_dumps.npz")
SETTLED = os.path.join(VIDEOS_DIR, "mf16_settled.npz")

N_EXPECTED = 86088
N_COFFEE = 49980

# --- mf16 jug geometry (same-source: multiflow/videos/mf16_balls_jug.npz meta geometry.spout,
#     generator multiflow/scripts/mf16_balls.py gen_jug deformation rules) ---
PS = h.PS_FULL  # 0.008
PR = 0.5 * PS  # particle_radius 0.004
JUG_R_IN = 0.125
JUG_R_OUT = 0.141
JUG_Z_MIN = 0.0
JUG_Z_MAX = 0.316  # -> non-spout top contact surface exactly 0.320 with w = 0
SPOUT_AZ = math.pi  # spout centre azimuth (180 deg)
SPOUT_HALF = math.radians(20.0)  # spout half angle
SPOUT_DZ = -0.002
SPOUT_DR = 0.030
HIST_TOL = 2.0e-4 * PS  # same tolerance semantics as the mug A2 kernel (1.6e-6 m)

# mug A2 tuple, identical to dbg_mf16_barrel_resume.py --smooth-a2
SMOOTH_BARREL_A2 = (
    0, 0.0, 0.0, h.R_IN, h.R_OUT, 0.0, h.Z_RIM,
    math.pi, math.radians(35.0), 0.080, 0.240, 0.002,
)

# S1 stated threshold (calibrated 2026-09-14 against the reference run; see s1 result json:
# measured staged_max was ~34k while draining, threshold set two orders below)
S1_STAGED_MIN = 100

# S5 probe-margin floor: gate margins must dominate the f32 kernel/prediction noise (~1e-7 m)
# and the sub-support ball-density nudge (~1e-5 m) by an order of magnitude
S5_MIN_MARGIN = 5.0e-5
# S5 probe clearance from every jug boundary ball across the frame: above the ball CONTACT
# distance (R_ball + particle_radius = ps = 0.008 m; below it the swept clamp projects the
# probe).  Between 8 and 10 mm the kernel weights are ~(support-r)^2 <= 4e-6 of peak, i.e. the
# density/viscosity nudge stays ~1e-5 m, an order below S5_MIN_MARGIN.
S5_MIN_BALL_CLEARANCE = 0.0086


def out_path(suffix):
    return os.path.join(VIDEOS_DIR, f"{TAG}_{suffix}")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


# --------------------------------------------------------------------------- shared geometry
def spout_w_az(az):
    """cos^2 spout window weight for local azimuths (vectorized, same-source as mf16_balls.py)."""
    d = np.arctan2(np.sin(az - SPOUT_AZ), np.cos(az - SPOUT_AZ))
    w = np.zeros_like(az)
    m = np.abs(d) <= SPOUT_HALF
    w[m] = np.cos(0.5 * math.pi * d[m] / SPOUT_HALF) ** 2
    return d, w


def z_top_of_w(w):
    return JUG_Z_MAX + SPOUT_DZ * w + PR


def r_out_of_w(w):
    return JUG_R_OUT + SPOUT_DR * w + PR


def spout_aware_crossing(prev_l, cur_l):
    """Outward crossing of the spout-aware outer contact cylinder r_out_contact(az).

    Mirrors the solver kernel's tunnel geometry: fire when r crosses from below the
    azimuth-dependent contact radius at the segment start to at/above the radius at the segment
    end; interpolate the crossing point linearly (same method as the mug kernel's crossing-z).
    Returns (crossed, cross_z, cross_az).
    """
    p = np.asarray(prev_l, dtype=np.float64)
    q = np.asarray(cur_l, dtype=np.float64)
    r_old = np.linalg.norm(p[:, :2], axis=1)
    r_new = np.linalg.norm(q[:, :2], axis=1)
    _, w_old = spout_w_az(np.arctan2(p[:, 1], p[:, 0]))
    _, w_new = spout_w_az(np.arctan2(q[:, 1], q[:, 0]))
    ro_old = r_out_of_w(w_old)
    ro_new = r_out_of_w(w_new)
    crossed = (r_old < ro_old) & (r_new >= ro_new) & (r_new > r_old)
    t = np.where(crossed, (ro_old - r_old) / np.maximum(r_new - r_old, 1.0e-20), 0.0)
    cross_xy = p[:, :2] + t[:, None] * (q[:, :2] - p[:, :2])
    cross_z = p[:, 2] + t * (q[:, 2] - p[:, 2])
    cross_az = np.arctan2(cross_xy[:, 1], cross_xy[:, 0])
    return crossed, cross_z, cross_az


def set_smoke_timeline():
    """Reproduce the module-global timeline that the official --smoke branch installs."""
    h.T_LIFT0, h.T_LIFT1 = 2.0, 3.5
    h.T_TILT0, h.T_TILT1 = 3.5, 6.0
    h.T_TRICKLE, h.TRICKLE_RAMP = 8.0, 1.0
    h.T_DOME_STOP, h.STOP_RAMP = 11.0, 1.0
    h.STOP_RETREAT_X, h.STOP_LIFT_Z = 0.04, 0.02
    h.T_OVERFILL, h.OVERFILL_RAMP = 14.25, 0.75
    h.T_BACK, h.T_PARK0, h.T_PARK1 = 17.5, 17.5, 19.0
    h.TILT_MAX_DEG, h.TILT_TRICKLE_DEG = 72.0, 78.0
    h.TILT_DOME_HOLD_DEG, h.TILT_OVERFILL_DEG = 65.0, 112.0
    h.PIVOT_X, h.PIVOT_Z = 0.09, 0.38
    h.DUR = 20.0


def jug_history_option(set_index):
    return (
        set_index, JUG_R_IN, JUG_R_OUT, JUG_Z_MIN, JUG_Z_MAX,
        SPOUT_AZ, SPOUT_HALF, SPOUT_DZ, SPOUT_DR, HIST_TOL,
    )


# --------------------------------------------------------------------------- scene builders
def install_fluid(scene, solver, coffee, milk, pos0, vel0, c0):
    """Install an external fluid state and explicitly reset both contact histories."""
    n_coffee = coffee.n_particles
    coffee.set_particles_pos(pos0[:n_coffee])
    milk.set_particles_pos(pos0[n_coffee:])
    coffee.set_particles_vel(vel0[:n_coffee])
    milk.set_particles_vel(vel0[n_coffee:])
    c_full = solver.particles.c.to_numpy()
    c_full[:N_EXPECTED, 0] = c0
    solver.particles.c.from_numpy(c_full)
    # The histories are solver-owned contact facts; an externally installed fluid state must
    # start them clean (mirrors dbg_mf16_barrel_resume.py:216).
    if getattr(solver, "_has_boundary_ball_jug_history", False):
        solver.reset_boundary_ball_jug_history()
    if solver._has_boundary_ball_smooth_barrel:
        solver.reset_boundary_ball_smooth_barrel_history()


def build_scene(dump_state=None, jug_history=None, smooth_barrel=None):
    """Mirror dbg_mf16_barrel_resume.build_scene (no rigid meshes), plus the jug history option.

    dump_state: None -> install the settled state with the jug upright at (PX, 0, PZ);
    else a (pos, vel, c, pose7) tuple from the diag15 dumps (jug frozen at that pose).
    """
    settled = np.load(SETTLED)
    n_coffee = int(settled["n_coffee"])
    n_milk = int(settled["n_milk"])
    assert (n_coffee, n_milk, n_coffee + n_milk) == (N_COFFEE, 36108, N_EXPECTED)
    r_coffee, h_coffee = float(settled["coffee_r"]), float(settled["coffee_h0"])
    r_milk, h_milk = float(settled["milk_r"]), float(settled["milk_h0"])
    mug_local, _ = h.load_ball_asset(h.MUG_NPZ, "mug", False)
    jug_local, _ = h.load_ball_asset(h.JUG_NPZ, "jug", True)
    assert int(settled["mug_ball_count"]) == len(mug_local)
    assert int(settled["jug_ball_count"]) == len(jug_local)

    if dump_state is None:
        pos0 = settled["pos"][:, 0, :].astype(np.float32)
        vel0 = settled["vel"][:, 0, :].astype(np.float32)
        c0 = settled["c"][:, 0].astype(np.float32)
        origin0 = np.array([h.PX, 0.0, h.PZ], dtype=np.float32)
        quat0 = h.IDENTITY_QUAT.copy()
    else:
        pos0, vel0, c0, pose0 = dump_state
        origin0, quat0 = pose0[:3].astype(np.float32), pose0[3:].astype(np.float32)

    gs.init(backend=gs.gpu, precision="32", seed=0)
    pbd_kwargs = dict(
        particle_size=h.PS_FULL,
        lower_bound=tuple(h.LOWER_BOUND),
        upper_bound=tuple(h.UPPER_BOUND),
        boundary_ball_sets=((h.MUG_NPZ, False), (h.JUG_NPZ, True)),
        boundary_ball_initial_poses=(
            ((0.0, 0.0, 0.0), tuple(h.IDENTITY_QUAT)),
            (tuple(origin0), tuple(quat0)),
        ),
        boundary_ball_mass_weight_enabled=False,
        boundary_ball_smooth_barrel=smooth_barrel,
        boundary_plane=(h.Z_TABLE,),
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
        wall_adhesion_enabled=False,
        wall_adhesion_compliance=20.0,
        wall_friction=0.0,
    )
    if jug_history is not None:
        pbd_kwargs["boundary_ball_jug_history"] = jug_history
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=h.DT, substeps=h.SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_pbd=False),
        rigid_options=gs.options.RigidOptions(enable_collision=False, gravity=(0.0, 0.0, 0.0)),
        pbd_options=gs.options.PBDOptions(**pbd_kwargs),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )
    coffee = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=1.0,
            density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Cylinder(
            radius=r_coffee, height=h_coffee,
            pos=(0.0, 0.0, h.Z_FLOOR + h.S0 + 0.5 * h_coffee),
        ),
    )
    milk = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=0.0,
            density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Cylinder(
            radius=r_milk, height=h_milk,
            pos=(h.PX, 0.0, h.PZ + h.Z_FLOOR + h.S0 + 0.5 * h_milk),
        ),
    )
    scene.build()
    solver = scene.sim.pbd_solver
    assert coffee.n_particles == n_coffee and milk.n_particles == n_milk
    assert solver._n_entity_particles == N_EXPECTED
    assert solver.boundary_ball_ranges == [
        (N_EXPECTED, N_EXPECTED + len(mug_local), False),
        (N_EXPECTED + len(mug_local), N_EXPECTED + len(mug_local) + len(jug_local), True),
    ]

    # load proof (mirror dbg_mf16_barrel_resume.py:207-221): boundary balls must be untouched by
    # the fluid-state install, and the jug balls must match the transform of the asset locals.
    ball_slice = slice(N_EXPECTED, solver.n_particles)
    ball_before = solver.particles.pos.to_numpy()[ball_slice].copy()
    install_fluid(scene, solver, coffee, milk, pos0, vel0, c0)
    assert np.array_equal(solver.particles.pos.to_numpy()[ball_slice], ball_before)
    expected_jug = h.transform_points(jug_local, origin0, quat0)
    lo, hi = solver.boundary_ball_ranges[1][0], solver.boundary_ball_ranges[1][1]
    actual_jug = solver.particles.pos.to_numpy()[lo:hi, 0]
    pose_error = float(np.max(np.abs(actual_jug - expected_jug)))
    assert pose_error < 5.0e-7
    return scene, solver, coffee, milk, origin0, quat0, pose_error


def load_dump_state(t):
    dump = np.load(DUMPS)
    return (
        np.asarray(dump[f"{t}"], dtype=np.float32),
        np.asarray(dump[f"vel_{t}"], dtype=np.float32),
        np.asarray(dump[f"c_{t}"], dtype=np.float32),
        np.asarray(dump[f"pose_{t}"], dtype=np.float32),
    )


# --------------------------------------------------------------------------- S3: zero regression
def run_s3dump(args):
    """Short default-OFF run; dump full particle state + hashes for pre/post comparison."""
    variant = args.variant
    smooth_barrel = SMOOTH_BARREL_A2 if variant == "a2" else None
    scene, solver, coffee, milk, origin0, quat0, pose_error = build_scene(
        dump_state=None, jug_history=None, smooth_barrel=smooth_barrel
    )
    # disabled-option API surface (attributes exist only on the new engine; getattr keeps the
    # pre-change baseline phase runnable with the identical script)
    has_jug_hist = getattr(solver, "_has_boundary_ball_jug_history", False)
    assert has_jug_hist is False, "jug history must default off"
    api = getattr(solver, "boundary_ball_jug_history", None)
    api_check = "skipped (pre-change engine)"
    if api is not None:
        staged, tunnel = api()
        assert staged.shape == (N_EXPECTED,) and tunnel.shape == (N_EXPECTED,)
        assert not staged.any() and not tunnel.any()
        api_check = "empty arrays, all False"

    n_frames = int(round(args.seconds / h.DT))
    wall0 = time.time()
    for frame in range(1, n_frames + 1):
        origin, quat, _, _, _ = h.jug_pose(frame * h.DT)
        solver.set_boundary_ball_pose(1, origin, quat)  # exercise the touched kernel every frame
        scene.step()
    pos = solver.particles.pos.to_numpy()[:, 0, :].copy()
    vel = solver.particles.vel.to_numpy()[:, 0, :].copy()
    conc = solver.particles.c.to_numpy()[:, 0].copy()

    def _sha(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

    hashes = {"pos": _sha(pos), "vel": _sha(vel), "c": _sha(conc)}
    npz_path = out_path(f"s3_{variant}_{args.phase}.npz")
    json_path = out_path(f"s3_{variant}_{args.phase}.json")
    np.savez(npz_path, pos=pos, vel=vel, c=conc)
    meta = {
        "test": "S3", "variant": variant, "phase": args.phase,
        "genesis_file": gs.__file__, "n_frames": n_frames,
        "seconds": args.seconds, "hashes": hashes, "api_check": api_check,
        "wall_seconds": time.time() - wall0,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    print(f"S3DUMP {variant} {args.phase}: " + json.dumps(hashes))
    print(f"  npz={npz_path}\n  json={json_path}")


def _diff_stats(a, b):
    diff = a != b
    return {
        "bit_identical": bool(diff.sum() == 0),
        "diff_fraction": float(diff.mean()),
        "max_abs_diff": float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()),
    }


def run_s3cmp(_args):
    """Zero-regression verdict for the default-OFF engine.

    Cross-process bit-identity is unattainable on this engine: two identical post-change runs
    diverge chaotically from substep 1 (GPU neighbour-order nondeterminism; same conclusion as
    the forensic mainrep replica).  The regression check is therefore REPLICA EQUIVALENCE: the
    pre-change vs post-change divergence must be statistically the same as run-to-run replica
    divergence, plus the deterministic c-field must stay bit-identical and the disabled-option
    API surface must report all-False empty arrays (asserted in s3dump).
    """
    results = {}
    ok_all = True
    for variant in ("default", "a2"):
        pre = np.load(out_path(f"s3_{variant}_pre.npz"))
        post = np.load(out_path(f"s3_{variant}_post.npz"))
        replica = np.load(out_path(f"s3_{variant}_replica.npz"))
        with open(out_path(f"s3_{variant}_post.json"), encoding="utf-8") as fh:
            meta_post = json.load(fh)

        changes = {}
        for key in ("pos", "vel"):
            pre_post = _diff_stats(pre[key], post[key])
            replica_rep = _diff_stats(post[key], replica[key])
            # replica equivalence: the engine change may not diverge from the old engine by
            # more than the engine's own run-to-run replica divergence (2x tolerance)
            equiv = (
                pre_post["max_abs_diff"] <= 2.0 * max(replica_rep["max_abs_diff"], 1.0e-12)
                and pre_post["diff_fraction"] <= 2.0 * max(replica_rep["diff_fraction"], 1.0e-12)
            )
            changes[key] = {"pre_vs_post": pre_post, "replica_vs_replica": replica_rep,
                            "replica_equivalent": bool(equiv)}
            ok_all &= equiv
        c_identical = bool(np.array_equal(pre["c"], post["c"]))
        changes["c_bit_identical"] = c_identical  # deterministic field incl. the diffusion path
        ok_all &= c_identical
        results[variant] = {
            "checks": changes,
            "api_check_post": meta_post["api_check"],
            "genesis_post": meta_post["genesis_file"],
        }
        print(f"S3CMP {variant}: " + json.dumps(changes, sort_keys=True))
    results["pass"] = ok_all
    results["methodology"] = (
        "cross-process bit-identity is unattainable on this GPU engine (identical runs diverge "
        "chaotically from substep 1; see forensic mainrep replica). Zero regression = replica "
        "equivalence of pre-vs-post divergence + bit-identical deterministic c-field + "
        "disabled-option API surface."
    )
    with open(out_path("s3_result.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")
    print(f"S3 {'PASS' if ok_all else 'FAIL'} (replica-equivalent default-off, both variants)")
    return ok_all


# --------------------------------------------------------------------------- S1: static pour
def run_s1(args):
    """Jug fixed in the diag15 pose16 pour pose; milk drains over the spout -> all staged.

    Phase A (the specced pose16 fixture): jug held fixed at the diag15 t=16 pour pose; the
    draining milk must accumulate rim_staged and never direct-tunnel.
    Phase B (spot-check fixture): the diag17 false-positive class (spout-sector crossing with
    local z in [0.318, 0.320)) arises during ACTIVE spout flow at the 78 deg trickle angle,
    which the pose16 dribble does not reproduce (all of its spout-sector crossings land
    outside [0.316, 0.322) -- the mouth points mostly downward at 112 deg tilt).  Phase B
    rebuilds the settled state and tilts the jug to 78 deg with the official jug_pose phase
    C/D form; the spot class then occurs and every such particle must be staged, never direct.  Phase B runs as the separate ``s1b`` mode (one Scene per process).
    """
    run_s1_phase_a(args)


def run_s1_phase_a(args):
    """Phase A: pose16 fixture -- staged threshold + zero direct."""
    scene, solver, coffee, milk, origin0, quat0, pose_error = build_scene(
        dump_state=load_dump_state(16), jug_history=jug_history_option(1), smooth_barrel=None
    )
    n_fluid = N_EXPECTED
    n_frames = int(round(args.seconds / h.DT))
    staged_max = 0
    spot_events = []  # diag17 false-positive class: spout-sector crossing with z in [0.318, 0.320)
    rows = []
    prev = h.inverse_transform_points(
        solver.particles.pos.to_numpy()[:n_fluid, 0], origin0, quat0
    )
    wall0 = time.time()
    spout_z_hist = {"[0.316,0.318)": 0, "[0.318,0.320)": 0, "[0.320,0.322)": 0, "other": 0}
    for frame in range(n_frames + 1):
        # kinematic API path with a fixed pose: pose_from == target every frame
        solver.set_boundary_ball_pose(1, origin0, quat0)
        if frame > 0:
            scene.step()
        staged, direct = solver.boundary_ball_jug_history()
        n_direct = int(direct.sum())
        assert n_direct == 0, f"S1: direct_tunnel={n_direct} at frame {frame} (static pour)"
        pos = solver.particles.pos.to_numpy()[:n_fluid, 0]
        cur = h.inverse_transform_points(pos, origin0, quat0)
        crossed, cross_z, cross_az = spout_aware_crossing(prev, cur)
        _, cross_w = spout_w_az(cross_az)
        spout_cross = crossed & (cross_w > 0.0)
        for zc in cross_z[spout_cross]:
            if 0.316 <= zc < 0.318:
                spout_z_hist["[0.316,0.318)"] += 1
            elif 0.318 <= zc < 0.320:
                spout_z_hist["[0.318,0.320)"] += 1
            elif 0.320 <= zc < 0.322:
                spout_z_hist["[0.320,0.322)"] += 1
            else:
                spout_z_hist["other"] += 1
        spot = crossed & (cross_w > 0.0) & (cross_z >= 0.318) & (cross_z < 0.320)
        for pid in np.flatnonzero(spot):
            spot_events.append({
                "frame": int(frame), "pid": int(pid), "cross_z": float(cross_z[pid]),
                "az_deg": float(math.degrees(math.atan2(cur[pid, 1], cur[pid, 0]))),
                "staged_ever": bool(staged[pid]),
            })
        staged_max = max(staged_max, int(staged.sum()))
        rows.append({
            "frame": frame, "t": frame * h.DT, "staged": int(staged.sum()),
            "direct": n_direct, "spot_events": int(spot.sum()),
        })
        if frame % 120 == 0:
            print(f"S1 t=+{frame * h.DT:.2f}s staged={int(staged.sum())} direct={n_direct} "
                  f"spot={len(spot_events)} spout_hist={spout_z_hist}")
        prev = cur

    assert staged_max >= args.s1_min, f"S1: staged_max {staged_max} < threshold {args.s1_min}"
    # the pose16 dribble does not produce the diag17 spot class (see run_s1 docstring):
    # phase A asserts staging + zero direct only; the combined spot assert runs in phase B.
    staged_final, direct_final = solver.boundary_ball_jug_history()
    spot_pids = sorted({ev["pid"] for ev in spot_events})
    for pid in spot_pids:
        assert staged_final[pid], f"S1: spot-check pid {pid} never staged"
        assert not direct_final[pid], f"S1: spot-check pid {pid} direct-tunnelled"
    result = {
        "test": "S1", "pass": True, "seconds": args.seconds,
        "staged_max": staged_max, "staged_threshold": args.s1_min,
        "direct_total": int(direct_final.sum()),
        "spot_events": len(spot_events), "spot_pids": spot_pids,
        "spout_crossing_z_hist": spout_z_hist,
        "staged_final": int(staged_final.sum()),
        "wall_seconds": time.time() - wall0,
    }
    with open(out_path("s1.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    with open(out_path("s1_frames.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"S1 phase A PASS staged_max={staged_max} (>= {args.s1_min}), direct=0, "
          f"phase_a_spot_events={len(spot_events)}")
    return spot_events


def run_s1_phase_b(args):
    """Phase B: active spout pour at the 78 deg trickle angle (diag17 class reproduction)."""
    spot_events = []
    set_smoke_timeline()
    # mini-timeline: approach+tilt to the 78 deg trickle angle over 2.5 s, then hold; every
    # other phase pushed out of reach so jug_pose(t) stays in the official phase C/D form
    h.T_LIFT0, h.T_LIFT1 = 1.0e9, 1.0e9
    h.T_TILT0, h.T_TILT1 = 0.0, 2.5
    h.T_TRICKLE, h.TRICKLE_RAMP = 1.0e9, 1.0
    h.T_DOME_STOP, h.STOP_RAMP = 1.0e9, 1.0
    h.T_OVERFILL, h.OVERFILL_RAMP = 1.0e9, 0.75
    h.T_BACK, h.T_PARK0, h.T_PARK1 = 1.0e9, 1.0e9, 1.0e9
    h.TILT_MAX_DEG = 78.0
    scene, solver, coffee, milk, origin0, quat0, pose_error = build_scene(
        dump_state=None, jug_history=jug_history_option(1), smooth_barrel=None
    )
    n_fluid = N_EXPECTED
    n_frames = int(round(args.seconds_b / h.DT))
    rows = []
    prev = h.inverse_transform_points(
        solver.particles.pos.to_numpy()[:n_fluid, 0], origin0, quat0
    )
    wall0 = time.time()
    for frame in range(n_frames + 1):
        t = frame * h.DT
        origin, quat, _, th, _ = h.jug_pose(t)
        solver.set_boundary_ball_pose(1, origin, quat)
        if frame > 0:
            scene.step()
        staged, direct = solver.boundary_ball_jug_history()
        n_direct = int(direct.sum())
        assert n_direct == 0, f"S1B: direct_tunnel={n_direct} at t={t:.3f} (spout pour)"
        pos = solver.particles.pos.to_numpy()[:n_fluid, 0]
        cur = h.inverse_transform_points(pos, origin, quat)
        crossed, cross_z, cross_az = spout_aware_crossing(prev, cur)
        _, cross_w = spout_w_az(cross_az)
        spot = crossed & (cross_w > 0.0) & (cross_z >= 0.318) & (cross_z < 0.320)
        for pid in np.flatnonzero(spot):
            spot_events.append({
                "frame": int(frame), "pid": int(pid), "cross_z": float(cross_z[pid]),
                "az_deg": float(math.degrees(math.atan2(cur[pid, 1], cur[pid, 0]))),
                "staged_ever": bool(staged[pid]),
                "phase": "B",
            })
        rows.append({
            "frame": frame, "t": t, "th_deg": float(math.degrees(th)),
            "staged": int(staged.sum()), "direct": n_direct, "spot_events": int(spot.sum()),
        })
        if frame % 60 == 0:
            print(f"S1B t={t:.2f}s th={math.degrees(th):.1f} staged={int(staged.sum())} "
                  f"direct={n_direct} spot={len(spot_events)}")
        prev = cur

    assert spot_events, "S1B: no spout-sector [0.318,0.320) crossings observed (spot check void)"
    staged_final, direct_final = solver.boundary_ball_jug_history()
    spot_pids = sorted({ev["pid"] for ev in spot_events})
    for pid in spot_pids:
        assert staged_final[pid], f"S1: spot-check pid {pid} never staged"
        assert not direct_final[pid], f"S1: spot-check pid {pid} direct-tunnelled"
    result = {
        "test": "S1", "phase": "B", "pass": True, "seconds": args.seconds_b,
        "direct_total": int(direct_final.sum()),
        "spot_events": len(spot_events), "spot_pids": spot_pids,
        "staged_final": int(staged_final.sum()),
        "wall_seconds": time.time() - wall0,
    }
    with open(out_path("s1b.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    with open(out_path("s1b_frames.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"S1 PASS (A+B): spot_events={len(spot_events)} pids={spot_pids[:8]}, "
          f"all staged, direct=0")


# --------------------------------------------------------------------------- S2: forced direct
def run_s2(args):
    """Seed one milk particle inside the jug with an outward velocity below the rim -> direct.

    Two-phase seeding: the jug's swept ball CCD scans the substep-start cell neighbourhood, so
    no constant-velocity seed can both (a) register interior residency and (b) jump past the
    wall within one neighbourhood-missing substep.  Phase 1 rests the particle inside the jug
    for one frame (inner_seen accumulates naturally); phase 2 re-seeds it at r=0.09 with
    v=30 m/s (62.5 mm/substep): the crossing substep starts at r=0.09 whose neighbourhood
    reaches r<=0.11 and therefore misses the wall balls (influence starts at r>=~0.118), the
    segment lands at r=0.1525 >= r_out_contact(az=0)=0.145, and the history kernel flags the
    below-rim outward crossing at substep resolution.  This is exactly the diag17 event class
    (swept-CCD blind spot), now recorded same-source instead of by a 60 Hz frame chord.
    """
    scene, solver, coffee, milk, origin0, quat0, pose_error = build_scene(
        dump_state=None, jug_history=jug_history_option(1), smooth_barrel=None
    )
    n_fluid = N_EXPECTED
    pid = N_COFFEE + 1234  # any milk row; the state below is fully synthetic
    seed1_pos = np.array([h.PX + 0.05, 0.0, 0.15], dtype=np.float64)   # interior: r=0.05, z=0.15
    seed2_pos = np.array([h.PX + 0.09, 0.0, 0.15], dtype=np.float64)   # r=0.09, still interior
    seed2_vel = np.array([30.0, 0.0, 0.0], dtype=np.float64)           # 62.5 mm/substep

    def install(pos_row, vel_row):
        pos = solver.particles.pos.to_numpy()[:n_fluid, 0].copy()
        vel = solver.particles.vel.to_numpy()[:n_fluid, 0].copy()
        pos[pid] = pos_row
        vel[pid] = vel_row
        coffee.set_particles_pos(pos[:N_COFFEE])
        milk.set_particles_pos(pos[N_COFFEE:])
        coffee.set_particles_vel(vel[:N_COFFEE])
        milk.set_particles_vel(vel[N_COFFEE:])

    install(seed1_pos, np.zeros(3))
    solver.reset_boundary_ball_jug_history()
    # phase 1: one frame at rest inside the jug -> jug_inner_seen[pid] accumulates at substep 1
    solver.set_boundary_ball_pose(1, origin0, quat0)
    scene.step()
    # phase 2: forced penetration (history NOT reset: inner_seen must carry over)
    install(seed2_pos, seed2_vel)
    n_frames = int(round(args.seconds / h.DT))
    pred_fire_substep = 1  # crossing happens in the first substep after the phase-2 install
    fire_frame = None
    rows = []
    wall0 = time.time()
    for frame in range(2, 2 + n_frames):
        # jug stays upright at its rest pose (kinematic API still exercised every frame)
        solver.set_boundary_ball_pose(1, origin0, quat0)
        scene.step()
        staged, direct = solver.boundary_ball_jug_history()
        pos_now = solver.particles.pos.to_numpy()[:n_fluid, 0]
        cur = h.inverse_transform_points(pos_now, origin0, quat0)
        # exclusivity for the seeded particle (mf16_pour_overflow.py:1006-1007 style):
        # tunnel history is locked first; a later rim event can never wash it into "cleared".
        assert not (direct[pid] and staged[pid]), (
            f"S2: seeded pid {pid} holds direct & staged simultaneously at frame {frame}"
        )
        if fire_frame is None and direct[pid]:
            fire_frame = frame
        if fire_frame is not None:
            assert direct[pid], f"S2: direct washed at frame {frame} (irreversibility violated)"
        rows.append({
            "frame": frame, "t": frame * h.DT,
            "pid_r": float(np.linalg.norm(cur[pid, :2])), "pid_z": float(cur[pid, 2]),
            "pid_direct": bool(direct[pid]), "pid_staged": bool(staged[pid]),
            "direct_total": int(direct.sum()), "staged_total": int(staged.sum()),
        })
        print(f"S2 frame {frame}: pid_r={rows[-1]['pid_r']:.4f} z={rows[-1]['pid_z']:.4f} "
              f"direct={rows[-1]['pid_direct']} staged={rows[-1]['pid_staged']}")

    assert fire_frame is not None, (
        "S2: seeded particle never direct-tunnelled "
        f"(trajectory: {[(r['pid_r'], r['pid_direct']) for r in rows[:6]]})"
    )
    fire = rows[fire_frame - 2]
    assert not fire["pid_staged"], "S2: seeded particle staged before/instead of direct"
    assert fire_frame == 2, (
        f"S2: direct fired at frame {fire_frame}, expected frame 2 (substep {pred_fire_substep})"
    )
    result = {
        "test": "S2", "pass": True,
        "pid": int(pid), "seed1_pos": seed1_pos.tolist(),
        "seed2_pos": seed2_pos.tolist(), "seed2_vel": seed2_vel.tolist(),
        "predicted_fire_substep": int(pred_fire_substep),
        "fire_frame": int(fire_frame), "fire_t": fire_frame * h.DT,
        "fire_r": fire["pid_r"], "fire_z": fire["pid_z"],
        "substep_resolution": "kernel evaluates the accepted ipos->pos segment every substep; "
                              "the crossing landed in substep %d of frame %d"
                              % (pred_fire_substep, fire_frame),
        "direct_final": int(solver.boundary_ball_jug_history()[1].sum()),
        "wall_seconds": time.time() - wall0,
    }
    with open(out_path("s2.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    with open(out_path("s2_frames.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"S2 PASS pid={pid} fired at frame {fire_frame} substep {pred_fire_substep} "
          f"(frame-end r={fire['pid_r']:.4f}, z={fire['pid_z']:.4f}), "
          f"direct_final={result['direct_final']}")


# --------------------------------------------------------------------------- S4: return sweep
def run_s4(args):
    """Replay the jug return window 18.0 -> 19.2 s from the t=18 dump state; solver direct == 0."""
    set_smoke_timeline()
    scene, solver, coffee, milk, origin0, quat0, pose_error = build_scene(
        dump_state=load_dump_state(18), jug_history=jug_history_option(1), smooth_barrel=None
    )
    n_fluid = N_EXPECTED
    t0 = 18.0
    n_frames = int(round(args.seconds / h.DT))
    rows = []
    staged_max_window = 0
    wall0 = time.time()
    for frame in range(n_frames + 1):
        t = t0 + frame * h.DT
        origin, quat, _, th, _ = h.jug_pose(t)
        solver.set_boundary_ball_pose(1, origin, quat)
        if frame > 0:
            scene.step()
        staged, direct = solver.boundary_ball_jug_history()
        n_direct = int(direct.sum())
        assert n_direct == 0, f"S4: solver direct_tunnel={n_direct} at t={t:.4f}"
        in_window = 18.25 - 1e-9 <= t <= 18.62 + 1e-9
        staged_max_window = max(staged_max_window, int(staged.sum()) if in_window else 0)
        rows.append({
            "frame": frame, "t": t, "th_deg": float(math.degrees(th)),
            "staged": int(staged.sum()), "direct": n_direct, "in_window": in_window,
        })
        if frame % 12 == 0:
            print(f"S4 t={t:.3f}s th={math.degrees(th):6.1f}deg staged={int(staged.sum())} "
                  f"direct={n_direct}")
    result = {
        "test": "S4", "pass": True, "t0": t0, "seconds": args.seconds,
        "window_s": [18.25, 18.62],
        "solver_direct_total": 0,
        "staged_max_window": staged_max_window,
        "staged_final": int(solver.boundary_ball_jug_history()[0].sum()),
        "wall_seconds": time.time() - wall0,
    }
    with open(out_path("s4.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    with open(out_path("s4_frames.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"S4 PASS solver_direct=0 over {args.seconds:.2f}s replay "
          f"(staged_max_window={staged_max_window})")


# --------------------------------------------------------------------------- S5: pose interp unit
def nlerp_quat(q0, q1, a):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    s = -1.0 if float(q0 @ q1) < 0.0 else 1.0
    q = (1.0 - a) * q0 + a * s * q1
    return q / np.linalg.norm(q)


def pose_at(pose_from, pose_to, a):
    p = pose_from[:3] + a * (pose_to[:3] - pose_from[:3])
    q = nlerp_quat(pose_from[3:], pose_to[3:], a)
    return p, q


def predict_probe(X, pose_from, pose_to, substeps):
    """Per-hypothesis cumulative (inner, staged, margin) for a free-falling probe particle.

    Hypotheses: 'interp' (pose at a=(f+1)/n per substep, the specified kernel behavior),
    'frozen_start' (a=0 for every substep) and 'frozen_end' (a=1 for every substep).
    Mirrors the kernel gates exactly; the probe has no neighbours so its accepted substep
    positions are the exact semi-implicit Euler free fall z_k = X_z - g*h_sub^2*k*(k+1)/2.
    margin = decision robustness: for a positive result, the weakest staging substep's
    signed distance to the nearest gate boundary; for a negative result, the closest
    approach of any substep to staging.  A large margin makes f32/integration noise
    irrelevant even though a decisive sweep crosses the boundary by construction.
    """
    g = 9.81
    h_sub = h.DT / substeps
    out = {}
    for hyp in ("interp", "frozen_start", "frozen_end"):
        inner = False
        staged = False
        best_signed = -float("inf")  # max over substeps of the signed staging margin
        for k in range(1, substeps + 1):
            xk = np.array([X[0], X[1], X[2] - g * h_sub * h_sub * k * (k + 1) / 2.0])
            a = k / substeps if hyp == "interp" else (0.0 if hyp == "frozen_start" else 1.0)
            p, q = pose_at(pose_from, pose_to, a)
            loc = h.inverse_transform_points(xk[None, :], p, q)[0]
            r = float(np.linalg.norm(loc[:2]))
            z = float(loc[2])
            az = math.atan2(loc[1], loc[0])
            _, w = spout_w_az(np.array([az]))
            z_top = float(z_top_of_w(w)[0])
            r_out = float(r_out_of_w(w)[0])
            inner |= (r <= JUG_R_IN - PR + HIST_TOL) and (JUG_Z_MIN <= z <= JUG_Z_MAX)
            signed = min(
                z - (z_top - HIST_TOL),
                r - (JUG_R_IN - 0.5 * PS - HIST_TOL),
                (r_out + HIST_TOL) - r,
            )
            staged |= signed >= 0.0
            best_signed = max(best_signed, signed)
        margin = best_signed if staged else -best_signed
        out[hyp] = (inner, staged, margin)
    return out


def search_probe_point(pose_from, pose_to, substeps):
    """Find a world point just above the rim where the interp and frame-frozen hypotheses
    disagree on staging, with comfortable gate margins AND ball clearance (the probe must
    free-fall: no jug ball within S5_MIN_BALL_CLEARANCE at any substep pose)."""
    from scipy.spatial import cKDTree

    jug_local, _ = h.load_ball_asset(h.JUG_NPZ, "jug", True)
    # the ball trajectory is the true interpolation in every hypothesis (the pose-advance
    # kernel is unchanged), so one set of per-substep world-space ball trees serves all
    substep_trees = []
    for k in range(1, substeps + 1):
        p, q = pose_at(pose_from, pose_to, k / substeps)
        substep_trees.append(cKDTree(h.transform_points(jug_local, p, q)))

    def clearance(X):
        g = 9.81
        h_sub = h.DT / substeps
        worst = float("inf")
        for k, tree in enumerate(substep_trees, start=1):
            xk = np.array([X[0], X[1], X[2] - g * h_sub * h_sub * k * (k + 1) / 2.0])
            worst = min(worst, float(tree.query(xk, k=1)[0]))
        return worst

    best = None
    stats = {"clearance": 0, "inner": 0, "margin": 0, "agree": 0, "ok": 0}
    az_grid = np.linspace(SPOUT_AZ - 0.35, SPOUT_AZ + 0.35, 15)
    r_grid = np.array([0.128, 0.137, 0.146, 0.150, 0.155, 0.160, 0.166])
    z_grid = np.linspace(0.312, 0.332, 41)
    for az in az_grid:
        for r_b in r_grid:
            local = np.stack(
                [r_b * np.cos(az) * np.ones_like(z_grid),
                 r_b * np.sin(az) * np.ones_like(z_grid), z_grid], axis=1
            )
            world = h.transform_points(local, pose_from[:3], pose_from[3:])
            for X in world:
                if clearance(X) < S5_MIN_BALL_CLEARANCE:
                    stats["clearance"] += 1
                    continue  # ball contact/density would break the free-fall prediction
                pred = predict_probe(X, pose_from, pose_to, substeps)
                if pred["interp"][0]:
                    stats["inner"] += 1
                    continue  # probe must never satisfy the interior gate (direct stays 0)
                if min(p[2] for p in pred.values()) < S5_MIN_MARGIN:
                    stats["margin"] += 1
                    continue
                differs = (pred["interp"][1] != pred["frozen_end"][1]
                           or pred["interp"][1] != pred["frozen_start"][1])
                if not differs:
                    stats["agree"] += 1
                    continue  # hypotheses agree here: cannot pin the interpolation
                stats["ok"] += 1
                # prefer the strongest disagreement + margin
                score = min(p[2] for p in pred.values())
                if best is None or score > best[1]:
                    best = (X, score, pred)
    print(f"S5 search: {stats}")
    assert best is not None, "S5: no probe point found where hypotheses disagree with margin"
    X, score, pred = best
    return X, pred


def build_probe_scene(origin0, quat0, X):
    """Minimal scene: one probe particle + the kinematic jug ball set (set index 0)."""
    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=h.DT, substeps=h.SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_pbd=False),
        rigid_options=gs.options.RigidOptions(enable_collision=False, gravity=(0.0, 0.0, 0.0)),
        pbd_options=gs.options.PBDOptions(
            particle_size=h.PS_FULL,
            lower_bound=tuple(h.LOWER_BOUND),
            upper_bound=tuple(h.UPPER_BOUND),
            boundary_ball_sets=((h.JUG_NPZ, True),),
            boundary_ball_initial_poses=((tuple(origin0), tuple(quat0)),),
            boundary_ball_jug_history=jug_history_option(0),
            boundary_plane=(h.Z_TABLE,),
            max_density_solver_iterations=20,
            max_viscosity_solver_iterations=1,
            density_lambda_epsilon=0.1,
            diffusion_coeff=0.0,  # lone probe: keep every optional transport off
            surface_tension_enabled=False,  # option under test is independent of ST
            velocity_damping=1.0,
            wall_adhesion_enabled=False,
            wall_friction=0.0,
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )
    probe = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=0.0,
            density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Sphere(radius=0.5 * h.PS_FULL, pos=tuple(X)),
    )
    scene.build()
    return scene, probe


def run_s5(args):
    """Pose-interpolation unit test at the max tilt-rate phase (stop ramp)."""
    set_smoke_timeline()
    # frame i: upload pose(t_i+dt) after build with pose(t_i); the solver records pose_from =
    # pose(t_i) at the upload (frame-start pose) and interpolates over the 8 substeps.
    t_i = args.t_frame
    origin_f, quat_f, _, th_f, _ = h.jug_pose(t_i)
    origin_t, quat_t, _, th_t, _ = h.jug_pose(t_i + h.DT)
    pose_from = np.concatenate([origin_f, quat_f]).astype(np.float64)
    pose_to = np.concatenate([origin_t, quat_t]).astype(np.float64)
    X, pred = search_probe_point(pose_from, pose_to, h.SUBSTEPS)

    scene, probe = build_probe_scene(origin_f, quat_f, X)
    solver = scene.sim.pbd_solver
    assert solver._has_boundary_ball_jug_history is True
    assert probe.n_particles == 1, f"probe entity must hold exactly 1 particle, got {probe.n_particles}"
    solver.set_boundary_ball_pose(0, origin_t, quat_t)  # records pose_from = pose(t_i) first
    # hold the probe at X at rest (entity index 0)
    probe.set_particles_pos(X[None, :].astype(np.float32))
    probe.set_particles_vel(np.zeros((1, 3), dtype=np.float32))
    solver.reset_boundary_ball_jug_history()

    wall0 = time.time()
    scene.step()  # one frame = 8 substeps, pose interpolated pose(t_i) -> pose(t_i+dt)
    staged, direct = solver.boundary_ball_jug_history()
    pos_end = solver.particles.pos.to_numpy()[0, 0, :]

    solver_staged = bool(staged[0])
    solver_direct = bool(direct[0])
    pred_interp = pred["interp"][1]
    pred_frozen_end = pred["frozen_end"][1]
    pred_frozen_start = pred["frozen_start"][1]

    assert solver_staged == pred_interp, (
        f"S5: solver staged={solver_staged} != interp prediction {pred_interp}"
    )
    assert solver_direct is False and pred["interp"][0] is False
    pinned = (pred_interp != pred_frozen_end) or (pred_interp != pred_frozen_start)
    assert pinned, "S5: hypotheses agree -- check cannot pin the pose interpolation"

    result = {
        "test": "S5", "pass": True,
        "t_frame": t_i,
        "tilt_rate_peak_deg_s": 1.5 * abs(math.degrees(th_t - th_f)) / h.DT,
        "probe_X": [float(v) for v in X],
        "probe_pos_end": [float(v) for v in pos_end],
        "predictions": {
            hyp: {"inner": bool(v[0]), "staged": bool(v[1]), "margin": float(v[2])}
            for hyp, v in pred.items()
        },
        "solver_staged": solver_staged,
        "solver_direct": solver_direct,
        "pins": {
            "differs_from_frozen_end": pred_interp != pred_frozen_end,
            "differs_from_frozen_start": pred_interp != pred_frozen_start,
        },
        "wall_seconds": time.time() - wall0,
    }
    with open(out_path("s5.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print(f"S5 PASS t={t_i:.3f}s (peak tilt rate {result['tilt_rate_peak_deg_s']:.1f} deg/s): "
          f"solver staged={solver_staged} == interp {pred_interp}, "
          f"frozen_end {pred_frozen_end}, frozen_start {pred_frozen_start}")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("s3dump", help="S3 state dump (run pre-change AND post-change)")
    p.add_argument("--variant", choices=["default", "a2"], required=True)
    p.add_argument("--phase", choices=["pre", "post"], required=True)
    p.add_argument("--seconds", type=float, default=0.5)
    p.set_defaults(func=run_s3dump)

    p = sub.add_parser("s3cmp", help="S3 bit-identity comparison")
    p.set_defaults(func=run_s3cmp)

    p = sub.add_parser("s1", help="S1 static pour stages legally")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--seconds-b", type=float, default=6.0,

                   help="phase B active-spout-pour duration (diag17 spot class)")
    p.add_argument("--s1-min", type=int, default=S1_STAGED_MIN,
                   help="stated staged-count threshold (calibrated, see S1_STAGED_MIN)")
    p.set_defaults(func=run_s1)

    p = sub.add_parser("s1b", help="S1 phase B: active spout pour, diag17 spot class")
    p.add_argument("--seconds-b", type=float, default=6.0)
    p.set_defaults(func=run_s1_phase_b)

    p = sub.add_parser("s2", help="S2 forced penetration flags direct")
    p.add_argument("--seconds", type=float, default=0.5)
    p.set_defaults(func=run_s2)

    p = sub.add_parser("s4", help="S4 return-sweep replay, zero solver direct")
    p.add_argument("--seconds", type=float, default=1.2)
    p.set_defaults(func=run_s4)

    p = sub.add_parser("s5", help="S5 pose-interpolation unit")
    p.add_argument("--t-frame", type=float, default=11.75,
                   help="frame start time at the stop-ramp peak tilt rate (~39 deg/s)")
    p.set_defaults(func=run_s5)

    args = ap.parse_args()
    print(f"genesis loaded from: {gs.__file__}")
    assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), "Wrong genesis!"

    log_path = out_path(f"{args.mode}.log")
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_fh:
        stdout = sys.stdout
        sys.stdout = Tee(stdout, log_fh)
        try:
            ok = args.func(args)
            gs.destroy()
        finally:
            sys.stdout = stdout
    if ok is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
