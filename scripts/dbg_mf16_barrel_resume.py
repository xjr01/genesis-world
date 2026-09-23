"""Resume the MF-16 diag15 t=16 state and measure outer-mug pinning for four seconds.

This is an A/B fixture, not a replacement for ``mf16_pour_overflow.py``.  It rebuilds the
diag15 no-adhesion scene, installs the dumped jug pose before build (there is no first-step
pose sweep), restores all fluid positions/velocities/concentrations, and then holds the jug
fixed.  ``--smooth`` enables the opt-in analytic mug-barrel narrow phase and ``--mass-weight``
enables the asset-authored boundary density quadrature weights; the flags can be combined.
"""

import argparse
import csv
import json
import math
import os
import sys
import time

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import genesis as gs  # noqa: E402
import mf16_pour_overflow as h  # noqa: E402


DEFAULT_DUMPS = os.path.join(WORKSPACE, "videos", "mf16_smoke_diag15_noadh_fric0_dumps.npz")
DEFAULT_SETTLED = os.path.join(WORKSPACE, "videos", "mf16_settled.npz")
N_EXPECTED = 86088
N_COFFEE_EXPECTED = 49980
RESUME_T = 16.0


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


def dump_outer_mask(dump, t, audit):
    """Reproduce the two frozen evidence cohorts directly from the diag15 dumps."""
    key = f"{t:g}"
    pos = np.asarray(dump[key], dtype=np.float64)
    radius = np.linalg.norm(pos[:, :2], axis=1)
    r_lo, r_hi = ((0.166, 0.190) if audit else (h.R_OUT - 0.5 * h.PS_FULL, h.R_OUT + 2.5 * h.PS_FULL))
    return (
        np.asarray(dump[f"valid_{key}"], dtype=bool)
        & np.asarray(dump[f"not_jug_{key}"], dtype=bool)
        & np.asarray(dump[f"mug_cleared_{key}"], dtype=bool)
        & ~np.asarray(dump[f"mug_wall_tunnel_{key}"], dtype=bool)
        & ~np.asarray(dump[f"mug_bottom_tunnel_{key}"], dtype=bool)
        & (radius >= r_lo)
        & (radius <= r_hi)
        & (pos[:, 2] >= 0.020)
        & (pos[:, 2] <= h.Z_RIM)
    )


def live_outer_mask(pos, valid, not_jug, cleared, wall_tunnel, bottom_tunnel, audit):
    radius = np.linalg.norm(pos[:, :2], axis=1)
    r_lo, r_hi = ((0.166, 0.190) if audit else (h.R_OUT - 0.5 * h.PS_FULL, h.R_OUT + 2.5 * h.PS_FULL))
    return (
        valid
        & not_jug
        & cleared
        & ~wall_tunnel
        & ~bottom_tunnel
        & (radius >= r_lo)
        & (radius <= r_hi)
        & (pos[:, 2] >= 0.020)
        & (pos[:, 2] <= h.Z_RIM)
    )


def percentile_runs(max_run, cohort, q=90):
    values = max_run[cohort].astype(np.float64) * h.DT
    return float(np.percentile(values, q)) if len(values) else 0.0


def build_scene(dump, settled, smooth, smooth_a2, mass_weight):
    pos16 = np.asarray(dump["16"], dtype=np.float32)
    vel16 = np.asarray(dump["vel_16"], dtype=np.float32)
    c16 = np.asarray(dump["c_16"], dtype=np.float32)
    pose16 = np.asarray(dump["pose_16"], dtype=np.float32)
    assert pos16.shape == vel16.shape == (N_EXPECTED, 3)
    assert c16.shape == (N_EXPECTED,) and pose16.shape == (7,)
    assert np.isfinite(pos16).all() and np.isfinite(vel16).all() and np.isfinite(c16).all()

    n_coffee = int(settled["n_coffee"])
    n_milk = int(settled["n_milk"])
    assert (n_coffee, n_milk, n_coffee + n_milk) == (N_COFFEE_EXPECTED, 36108, N_EXPECTED)
    r_coffee, h_coffee = float(settled["coffee_r"]), float(settled["coffee_h0"])
    r_milk, h_milk = float(settled["milk_r"]), float(settled["milk_h0"])
    mug_local, _ = h.load_ball_asset(h.MUG_NPZ, "mug", False)
    jug_local, _ = h.load_ball_asset(h.JUG_NPZ, "jug", True)
    assert int(settled["mug_ball_count"]) == len(mug_local)
    assert int(settled["jug_ball_count"]) == len(jug_local)

    smooth_barrel = None
    if smooth:
        smooth_barrel = (
            0,
            0.0,
            0.0,
            0.150,
            0.166,
            0.000,
            h.Z_RIM,
            math.pi,
            math.radians(35.0),
            -1.0,
            1.0,
            0.002,
        )
    elif smooth_a2:
        smooth_barrel = (
            0,
            0.0,
            0.0,
            h.R_IN,
            h.R_OUT,
            0.0,
            h.Z_RIM,
            math.pi,
            math.radians(35.0),
            0.080,
            0.240,
            0.002,
        )

    origin16, quat16 = pose16[:3], pose16[3:]
    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=h.DT, substeps=h.SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_pbd=False),
        rigid_options=gs.options.RigidOptions(enable_collision=False, gravity=(0.0, 0.0, 0.0)),
        pbd_options=gs.options.PBDOptions(
            particle_size=h.PS_FULL,
            lower_bound=tuple(h.LOWER_BOUND),
            upper_bound=tuple(h.UPPER_BOUND),
            boundary_ball_sets=((h.MUG_NPZ, False), (h.JUG_NPZ, True)),
            boundary_ball_initial_poses=(
                ((0.0, 0.0, 0.0), tuple(h.IDENTITY_QUAT)),
                (tuple(origin16), tuple(quat16)),
            ),
            boundary_ball_mass_weight_enabled=mass_weight,
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
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )
    coffee = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=1.0,
            density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Cylinder(
            radius=r_coffee,
            height=h_coffee,
            pos=(0.0, 0.0, h.Z_FLOOR + h.S0 + 0.5 * h_coffee),
        ),
    )
    milk = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=0.0,
            density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Cylinder(
            radius=r_milk,
            height=h_milk,
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

    ball_slice = slice(N_EXPECTED, solver.n_particles)
    ball_before = solver.particles.pos.to_numpy()[ball_slice].copy()
    coffee.set_particles_pos(pos16[:n_coffee])
    milk.set_particles_pos(pos16[n_coffee:])
    coffee.set_particles_vel(vel16[:n_coffee])
    milk.set_particles_vel(vel16[n_coffee:])
    c_full = solver.particles.c.to_numpy()
    c_full[:N_EXPECTED, 0] = c16
    solver.particles.c.from_numpy(c_full)
    solver.reset_boundary_ball_smooth_barrel_history()
    assert np.array_equal(solver.particles.pos.to_numpy()[ball_slice], ball_before)
    expected_jug = h.transform_points(jug_local, origin16, quat16)
    actual_jug = solver.particles.pos.to_numpy()[solver.boundary_ball_ranges[1][0]:solver.boundary_ball_ranges[1][1], 0]
    pose_error = float(np.max(np.abs(actual_jug - expected_jug)))
    assert pose_error < 5.0e-7
    return scene, solver, origin16, quat16, pose_error


def run(args):
    if args.seconds <= 0.0:
        raise SystemExit("--seconds must be > 0")
    if not os.path.isfile(args.dumps) or not os.path.isfile(args.settled):
        raise SystemExit("missing diag15 dumps or mf16_settled.npz")
    dump = np.load(args.dumps)
    settled = np.load(args.settled)
    times = (16, 17, 18, 19, 20)
    audit_dump_masks = [dump_outer_mask(dump, t, True) for t in times]
    online_dump_masks = [dump_outer_mask(dump, t, False) for t in times]
    audit_t16 = audit_dump_masks[0]
    online_t16 = online_dump_masks[0]
    audit_persistent = np.logical_and.reduce(audit_dump_masks)
    online_persistent = np.logical_and.reduce(online_dump_masks)
    assert int(audit_persistent.sum()) == 1848, int(audit_persistent.sum())
    assert int(online_persistent.sum()) == 1841, int(online_persistent.sum())

    scene, solver, jug_origin, jug_quat, pose_error = build_scene(
        dump, settled, args.smooth, args.smooth_a2, args.mass_weight
    )
    n_fluid = N_EXPECTED
    valid = np.asarray(dump["valid_16"], dtype=bool).copy()
    mug_resident = np.asarray(dump["mug_resident_16"], dtype=bool).copy()
    mug_wall_entered = np.asarray(dump["mug_wall_entered_16"], dtype=bool).copy()
    mug_bottom_entered = np.asarray(dump["mug_bottom_entered_16"], dtype=bool).copy()
    mug_rim_crossed = np.asarray(dump["mug_rim_crossed_16"], dtype=bool).copy()
    mug_cleared = np.asarray(dump["mug_cleared_16"], dtype=bool).copy()
    mug_wall_tunnel = np.asarray(dump["mug_wall_tunnel_16"], dtype=bool).copy()
    mug_bottom_tunnel = np.asarray(dump["mug_bottom_tunnel_16"], dtype=bool).copy()
    wall_tunnel_0, bottom_tunnel_0 = int(mug_wall_tunnel.sum()), int(mug_bottom_tunnel.sum())
    mug_wall_inbound_entered = np.zeros(n_fluid, dtype=bool)

    prev = np.asarray(dump["16"], dtype=np.float64).copy()
    audit_run = audit_t16.astype(np.int32)
    online_run = online_t16.astype(np.int32)
    audit_max_run = audit_run.copy()
    online_max_run = online_run.copy()
    threshold_frames = int(round(1.0 / h.DT))
    rows = []
    tunnel_events = []
    nan_max = 0
    active_error_max = 0
    n_frames = int(round(args.seconds / h.DT))
    wall0 = time.time()

    for frame in range(n_frames + 1):
        if frame > 0:
            scene.step()
        pos_raw = solver.particles.pos.to_numpy()[:n_fluid, 0]
        vel_raw = solver.particles.vel.to_numpy()[:n_fluid, 0]
        c_raw = solver.particles.c.to_numpy()[:n_fluid, 0]
        nan = int((~np.isfinite(pos_raw)).sum() + (~np.isfinite(vel_raw)).sum() + (~np.isfinite(c_raw)).sum())
        nan_max = max(nan_max, nan)
        pos = np.nan_to_num(pos_raw, nan=0.0, posinf=0.0, neginf=0.0)
        r = np.linalg.norm(pos[:, :2], axis=1)
        z = pos[:, 2]
        active = int(solver.particles_ng.active.to_numpy()[:n_fluid, 0].sum())
        active_error_max = max(active_error_max, abs(active - n_fluid))

        jug_local = h.inverse_transform_points(pos, jug_origin, jug_quat)
        jug_s = jug_local[:, 2]
        jug_r = np.linalg.norm(jug_local[:, :2], axis=1)
        in_jug_footprint = (
            (jug_s > h.Z_FLOOR - 3.0 * h.PS_FULL)
            & (jug_s < h.Z_RIM + 3.0 * h.PS_FULL)
            & (jug_r <= h.R_JUG + h.T_JUG_WALL + 3.0 * h.PS_FULL)
        )
        # Match H exactly: only the milk population can still belong to the jug.  Coffee is
        # always counted in the mug/free-fluid census even when the parked jug footprint
        # overlaps it in world space.
        not_jug = np.ones(n_fluid, dtype=bool)
        not_jug[N_COFFEE_EXPECTED:] = ~in_jug_footprint[N_COFFEE_EXPECTED:]

        inside_mug = (r <= h.R_IN - 0.5 * h.PS_FULL) & (z >= h.Z_FLOOR) & (z <= h.Z_RIM)
        mug_resident |= inside_mug
        if args.smooth_a2:
            solver_rim_staged, solver_direct_tunnel = solver.boundary_ball_smooth_barrel_history()
        else:
            solver_rim_staged = np.zeros(n_fluid, dtype=bool)
            solver_direct_tunnel = np.zeros(n_fluid, dtype=bool)
        solver_new_direct = solver_direct_tunnel & ~mug_wall_tunnel
        mug_wall_tunnel |= solver_direct_tunnel
        any_tunnel = mug_wall_tunnel | mug_bottom_tunnel
        mug_open = mug_resident & ~mug_cleared & ~any_tunnel
        mug_wall_entered |= (
            mug_open & (r > h.R_IN - 0.5 * h.PS_FULL)
            & (z >= 0.0) & (z < h.Z_RIM + h.MUG_OUTER_CLEAR_Z_PS * h.PS_FULL)
        )
        mug_bottom_entered |= mug_open & (z < h.Z_FLOOR + 0.5 * h.PS_FULL) & (r <= h.R_OUT + h.PS_FULL)
        outer_r = h.R_OUT + h.MUG_OUTER_CLEAR_PS * h.PS_FULL
        outer_cross, _, outer_z = h.segment_outward_cylinder_crossing(prev, pos, outer_r)
        outer_in, _, outer_in_z = h.segment_inward_cylinder_crossing(prev, pos, outer_r)
        legal_outer_z = h.Z_RIM + h.MUG_OUTER_CLEAR_Z_PS * h.PS_FULL
        mug_wall_inbound_entered |= mug_resident & ~any_tunnel & outer_in & (outer_in_z < legal_outer_z)
        inner_in, _, inner_in_z = h.segment_inward_cylinder_crossing(prev, pos, h.R_IN - 0.5 * h.PS_FULL)
        inbound_now = (
            mug_resident & ~any_tunnel & mug_wall_inbound_entered
            & inner_in & (inner_in_z < legal_outer_z) & ~solver_rim_staged
        )
        wall_now = (
            mug_open & mug_wall_entered
            & ((outer_cross & (outer_z >= 0.0) & (outer_z < legal_outer_z))
               | (~outer_cross & (r >= outer_r) & (z >= 0.0) & (z < legal_outer_z)))
            & ~solver_rim_staged
        )
        bottom_cross, _, bottom_point = h.segment_downward_plane_crossing(prev, pos, -0.5 * h.PS_FULL)
        bottom_now = (
            mug_open & mug_bottom_entered
            & ((bottom_cross & (np.linalg.norm(bottom_point[:, :2], axis=1) <= h.R_OUT + h.PS_FULL))
               | (~bottom_cross & (z < -0.5 * h.PS_FULL) & (r <= h.R_OUT + h.PS_FULL)))
        )
        new_wall_tunnel = ((wall_now | inbound_now) & ~mug_wall_tunnel) | solver_new_direct
        if np.any(new_wall_tunnel):
            for idx in np.flatnonzero(new_wall_tunnel):
                theta = float(math.atan2(pos[idx, 1], pos[idx, 0]))
                dtheta_handle = float(abs(math.atan2(math.sin(theta - math.pi), math.cos(theta - math.pi))))
                tunnel_events.append({
                    "frame": int(frame),
                    "t_resume": float(frame * h.DT),
                    "particle": int(idx),
                    "kind": (
                        "solver_direct" if bool(solver_new_direct[idx]) else
                        "inbound" if bool(inbound_now[idx]) else "outbound"
                    ),
                    "prev_r": float(np.linalg.norm(prev[idx, :2])),
                    "prev_z": float(prev[idx, 2]),
                    "r": float(r[idx]),
                    "z": float(z[idx]),
                    "theta_deg": float(math.degrees(theta)),
                    "handle_delta_deg": float(math.degrees(dtheta_handle)),
                    "in_handle_sector": bool(dtheta_handle <= math.radians(35.0)),
                    "segment_hits_handle_z": bool(
                        min(float(prev[idx, 2]), float(z[idx])) <= 0.240
                        and max(float(prev[idx, 2]), float(z[idx])) >= 0.080
                    ),
                    "solver_rim_staged": bool(solver_rim_staged[idx]),
                    "outer_cross_z": float(outer_z[idx]),
                    "speed": float(np.linalg.norm(vel_raw[idx])),
                })
        mug_wall_tunnel |= new_wall_tunnel
        mug_bottom_tunnel |= bottom_now
        any_tunnel = mug_wall_tunnel | mug_bottom_tunnel
        mug_cleared &= ~any_tunnel

        # Existing cleared particles are the target population. Updating rim history keeps the
        # global outer census faithful if any other resident legally exits during the resume.
        rim_path_r = h.R_IN - h.MUG_RIM_PATH_R_MIN_PS * h.PS_FULL
        inner_cross, _, inner_cross_z = h.segment_outward_cylinder_crossing(prev, pos, rim_path_r)
        up = (z > prev[:, 2]) & (prev[:, 2] < h.Z_RIM) & (z >= h.Z_RIM)
        uz = np.clip((h.Z_RIM - prev[:, 2]) / np.maximum(z - prev[:, 2], 1.0e-12), 0.0, 1.0)
        rim_xy = prev[:, :2] + uz[:, None] * (pos[:, :2] - prev[:, :2])
        rim_r = np.linalg.norm(rim_xy, axis=1)
        near_lip = (r >= rim_path_r) & (r <= h.R_OUT + h.MUG_RIM_PATH_R_MAX_PS * h.PS_FULL) & (z >= h.Z_RIM)
        rim_event = (
            (inner_cross & (inner_cross_z >= h.Z_RIM))
            | (up & (rim_r >= rim_path_r) & (rim_r <= h.R_OUT + h.MUG_RIM_PATH_R_MAX_PS * h.PS_FULL))
        )
        mug_rim_crossed |= (
            mug_resident & ~mug_cleared & ~any_tunnel
            & (near_lip | rim_event | solver_rim_staged)
        )
        mug_rim_crossed &= ~any_tunnel
        mug_cleared |= (
            mug_rim_crossed & ~mug_cleared & ~any_tunnel
            & ((outer_cross & (outer_z >= legal_outer_z))
               | (solver_rim_staged & (r >= outer_r)))
        )

        audit_outer = live_outer_mask(pos, valid, not_jug, mug_cleared, mug_wall_tunnel, mug_bottom_tunnel, True)
        online_outer = live_outer_mask(pos, valid, not_jug, mug_cleared, mug_wall_tunnel, mug_bottom_tunnel, False)
        audit_run = np.where(audit_t16 & audit_outer, audit_run + (frame > 0), 0)
        online_run = np.where(online_t16 & online_outer, online_run + (frame > 0), 0)
        audit_max_run = np.maximum(audit_max_run, audit_run)
        online_max_run = np.maximum(online_max_run, online_run)
        row = {
            "frame": frame,
            "t_resume": frame * h.DT,
            "t_source": RESUME_T + frame * h.DT,
            "audit_outer_all": int(audit_outer.sum()),
            "audit_t16_remaining": int((audit_t16 & audit_outer).sum()),
            "audit_persistent_remaining": int((audit_persistent & audit_outer).sum()),
            "audit_t16_ever_ge_1s": int((audit_max_run[audit_t16] >= threshold_frames).sum()),
            "audit_t16_residence_p90_s": percentile_runs(audit_max_run, audit_t16),
            "online_outer_all": int(online_outer.sum()),
            "online_t16_remaining": int((online_t16 & online_outer).sum()),
            "online_persistent_remaining": int((online_persistent & online_outer).sum()),
            "online_t16_ever_ge_1s": int((online_max_run[online_t16] >= threshold_frames).sum()),
            "online_t16_residence_p90_s": percentile_runs(online_max_run, online_t16),
            "mug_wall_tunnel_increment": int(mug_wall_tunnel.sum()) - wall_tunnel_0,
            "mug_bottom_tunnel_increment": int(mug_bottom_tunnel.sum()) - bottom_tunnel_0,
            "solver_rim_staged": int(solver_rim_staged.sum()),
            "solver_direct_tunnel": int(solver_direct_tunnel.sum()),
            "nan": nan,
            "active_fluid": active,
        }
        rows.append(row)
        if frame % int(round(0.5 / h.DT)) == 0 or frame == n_frames:
            print(
                f"t+={row['t_resume']:.2f}s audit_outer={row['audit_outer_all']} "
                f"persistent={row['audit_persistent_remaining']}/1848 "
                f"ever1s={row['audit_t16_ever_ge_1s']} p90={row['audit_t16_residence_p90_s']:.3f}s "
                f"online_outer={row['online_outer_all']} tunnels="
                f"{row['mug_wall_tunnel_increment']}+{row['mug_bottom_tunnel_increment']} "
                f"nan={nan} active={active}"
            )
        prev = pos.copy()

    final = rows[-1]
    # Explain the residual population without changing any gate.  These three regions are
    # mutually exclusive and exactly partition each final outer cohort according to the
    # opt-in barrel patch: eligible analytic barrel, the full-height handle-exclusion sector,
    # and the low/high edge bands outside that sector.
    angle = np.arctan2(pos[:, 1], pos[:, 0])
    handle_delta = np.abs(np.arctan2(np.sin(angle - math.pi), np.cos(angle - math.pi)))
    handle_sector = handle_delta < math.radians(35.0)
    barrel_z = (pos[:, 2] >= 0.040) & (pos[:, 2] <= 0.292)

    def spatial_partition(cohort):
        return {
            "total": int(cohort.sum()),
            "smooth_eligible": int((cohort & ~handle_sector & barrel_z).sum()),
            "handle_excluded_full_height": int((cohort & handle_sector).sum()),
            "edge_outside_handle": int((cohort & ~handle_sector & ~barrel_z).sum()),
            "edge_low_z_lt_0p040": int((cohort & ~handle_sector & (pos[:, 2] < 0.040)).sum()),
            "edge_high_z_gt_0p292": int((cohort & ~handle_sector & (pos[:, 2] > 0.292)).sum()),
        }

    spatial = {
        "audit_outer_all": spatial_partition(audit_outer),
        "audit_t16_remaining": spatial_partition(audit_t16 & audit_outer),
        "audit_persistent_remaining": spatial_partition(audit_persistent & audit_outer),
    }
    mode = (
        "smooth_a2" if args.smooth_a2 else
        "smooth_mass" if args.smooth and args.mass_weight else
        "smooth" if args.smooth else
        "mass" if args.mass_weight else
        "baseline"
    )
    result = {
        "mode": mode,
        "smooth_barrel_enabled": bool(args.smooth),
        "smooth_barrel_a2_enabled": bool(args.smooth_a2),
        "boundary_ball_mass_weight_enabled": bool(args.mass_weight),
        "source_dumps": os.path.abspath(args.dumps),
        "resume_source_time_s": RESUME_T,
        "duration_s": args.seconds,
        "n_fluid": n_fluid,
        "n_coffee": N_COFFEE_EXPECTED,
        "jug_pose": np.concatenate((jug_origin, jug_quat)).tolist(),
        "initial_jug_pose_error": pose_error,
        "dump_cohorts": {
            "audit_t16": int(audit_t16.sum()),
            "audit_persistent_16_to_20": int(audit_persistent.sum()),
            "online_t16": int(online_t16.sum()),
            "online_persistent_16_to_20": int(online_persistent.sum()),
        },
        "final": final,
        "nan_max": nan_max,
        "active_fluid_error_max": active_error_max,
        "final_spatial_partition": spatial,
        "new_tunnel_events": tunnel_events,
        "smooth_gates": {
            "outer_end_lt_200": final["audit_outer_all"] < 200,
            "ever_ge_1s_lt_200": final["audit_t16_ever_ge_1s"] < 200,
            "residence_p90_lt_1s": final["audit_t16_residence_p90_s"] < 1.0,
            "tunnel_unchanged": (
                final["mug_wall_tunnel_increment"] == 0
                and final["mug_bottom_tunnel_increment"] == 0
            ),
            "finite_and_active": nan_max == 0 and active_error_max == 0,
        },
        "wall_seconds": time.time() - wall0,
    }
    result["smooth_pass"] = bool((args.smooth or args.smooth_a2) and all(result["smooth_gates"].values()))

    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print("RESULT", json.dumps(result, sort_keys=True))
    print(f"csv={args.csv}\njson={args.json}")


def main():
    parser = argparse.ArgumentParser()
    smooth_group = parser.add_mutually_exclusive_group()
    smooth_group.add_argument("--smooth", action="store_true", help="legacy full-height seam A")
    smooth_group.add_argument("--smooth-a2", action="store_true", help="localized handle seam + exact rim offset")
    parser.add_argument(
        "--mass-weight", action="store_true",
        help="enable the opt-in per-ball mass_weight arrays stored in the boundary assets",
    )
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--dumps", default=DEFAULT_DUMPS)
    parser.add_argument("--settled", default=DEFAULT_SETTLED)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()
    mode = (
        "smooth_a2" if args.smooth_a2 else
        "smooth_mass" if args.smooth and args.mass_weight else
        "smooth" if args.smooth else
        "mass" if args.mass_weight else
        "baseline"
    )
    stem = os.path.join(WORKSPACE, "videos", f"mf16_barrel_resume_{mode}")
    args.csv = args.csv or stem + ".csv"
    args.json = args.json or stem + ".json"
    args.log = args.log or stem + ".log"
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    with open(args.log, "w", encoding="utf-8", buffering=1) as log_fh:
        stdout = sys.stdout
        sys.stdout = Tee(stdout, log_fh)
        try:
            run(args)
            # Genesis' logger captures the current stdout stream at init.  Destroy while the
            # tee file is still open so its final cache message cannot write to a closed handle.
            gs.destroy()
        finally:
            sys.stdout = stdout


if __name__ == "__main__":
    main()
