"""dbg: PBD frozen-fluid boundary ball sets (PBDOptions.boundary_ball_sets, Akinci pressure walls).

The ball sets are frozen fluid particles: they add density and adhesion, own no constraint and are
skipped by every other pass. This solver's density projection is a soft cohesion term, not a
contact constraint, so the numbers below quantify what such a wall does and does not buy:
the mass calibration is exact (wall-layer density == interior density on the undeformed lattice)
while a fluid column under gravity compresses through the wall band. The analytic-plane control
(`plate --control-plane`) isolates the wall mechanism as the only difference.

Subcommands
  plate     ball plate under a fluid block: t=0 density calibration, penetration, nan.
            `--kinematic` additionally oscillates the plate through set_boundary_ball_pose.
  cup       double-layer ball cup + water blob: leak, supported free surface, nan.
  negative  same cup WITHOUT ball sets: the blob collapses through the bottom (leak > 0),
            proving the cup metric can actually fail.
  adhesion  plate + wall adhesion resolved against the nearest ball: n_adh > 0, no nan.
  jug       kinematic ball cup swung to --tilt-deg and back: nothing driven through the solid,
            the wall carries the liquid (pouring over the rim is allowed), no nan.
  double-set
            real MF16 static mug set0 + kinematic jug set1: every transformed jug point matches
            a NumPy rigid-transform reference while the mug remains unchanged; bad quaternions fail.
  regress   archived mf15 selftest (npz-free) for the zero-regression comparison; extra flags are
            forwarded verbatim (e.g. `regress --no-video`).

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \\
    "$PY" multiflow/scripts/dbg_ball_boundary.py plate
"""

import argparse
import math
import os
import sys
import time

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np  # noqa: E402

import genesis as gs  # noqa: E402

print("genesis loaded from:", gs.__file__)
assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), "Wrong genesis! Must be the multiflow copy."

DT = 1.0 / 60.0
SUBSTEPS = 8
PS = 0.008
RHO = 1000.0
OUT_DIR = os.path.join(WORKSPACE, "scripts", "_out_ball_boundary")
ARCHIVE = os.path.join(WORKSPACE, "scripts", "_archive_20260913", "mf15_pour_overflow.py")
MF16_MUG = os.path.join(WORKSPACE, "videos", "mf16_balls_mug.npz")
MF16_JUG = os.path.join(WORKSPACE, "videos", "mf16_balls_jug.npz")

# solver kernel constants (PBDSolver.poly6): support 1.25 * particle_size, self term excluded
DIST_SCALE = 0.5 * PS / 0.4
POLY6_COE = 315.0 / (64.0 * math.pi)


# ---------------------------------------------------------------------------------------------
# ball set sampling (what an offline sampler would put in the npz)
# ---------------------------------------------------------------------------------------------


def save_ball_set(name, pos_local):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    np.savez(path, pos_local=np.asarray(pos_local, dtype=np.float32))
    return path


def load_mf16_ball_set(path):
    """Load one checked-in MF16 sampler product through the engine's public npz contract."""
    assert os.path.isfile(path), f"missing MF16 ball product: {path}; run mf16_balls.py first"
    with np.load(path) as data:
        assert "pos_local" in data.files, f"{path} lacks pos_local"
        pos = np.asarray(data["pos_local"], dtype=np.float32)
    assert pos.ndim == 2 and pos.shape[1] == 3 and len(pos) > 0, (path, pos.shape)
    assert np.isfinite(pos).all(), f"non-finite MF16 balls in {path}"
    return pos


def transform_wxyz(points, trans, quat):
    """NumPy reference for gu.qd_transform_by_trans_quat_fast (wxyz quaternion)."""
    q = np.asarray(quat, dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rot = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return np.asarray(points, dtype=np.float64) @ rot.T + np.asarray(trans, dtype=np.float64)


def plate_balls(half_xy, n_layers, fluid_half, n_fluid_cells, edge_center=None, skirt_height=None):
    """Ball plate aligned with the fluid lattice, optionally sealed against the box side walls.

    ``edge_center`` adds four one-ball-wide strips whose inflated collision spheres overlap the
    analytic side walls.  This removes the zero-width tangent seam between a finite plate and the
    box boundary without changing the regular central lattice used by the density calibration.
    """
    x0 = -fluid_half + 0.5 * PS
    x1 = fluid_half - 0.5 * PS
    n_pad = max(0, int(math.ceil((half_xy - x1) / PS)))
    xs = x0 + PS * np.arange(-n_pad, n_fluid_cells + n_pad)
    grid = np.stack(np.meshgrid(xs, xs, indexing="ij"), -1).reshape((-1, 2))
    if edge_center is not None:
        n_edge = int(math.ceil(2.0 * edge_center / PS))
        edge_axis = np.linspace(-edge_center, edge_center, n_edge + 1)
        strips = np.concatenate(
            [
                np.column_stack([np.full(len(edge_axis), -edge_center), edge_axis]),
                np.column_stack([np.full(len(edge_axis), +edge_center), edge_axis]),
                np.column_stack([edge_axis, np.full(len(edge_axis), -edge_center)]),
                np.column_stack([edge_axis, np.full(len(edge_axis), +edge_center)]),
            ],
            axis=0,
        )
        grid = np.unique(np.round(np.concatenate([grid, strips], axis=0), decimals=9), axis=0)
    layers = [np.column_stack([grid, np.full(len(grid), -l * PS)]) for l in range(n_layers)]
    chunks = list(layers)
    if skirt_height is not None:
        # A moving infinite plate cannot both seal against a static box wall and remain inside the
        # hash domain.  Give the diagnostic plate its own two-layer ball skirt instead, producing a
        # rigid translating tray whose swept extent is finite and fully in-domain.
        edge_axis = np.arange(-half_xy, half_xy + 0.5 * PS, PS)
        zs = np.arange(0.0, skirt_height + 0.5 * PS, PS)
        for outward in (0.0, PS):
            h = half_xy + outward
            for sign in (-1.0, 1.0):
                yy, zz = np.meshgrid(edge_axis, zs, indexing="ij")
                chunks.append(np.column_stack([np.full(yy.size, sign * h), yy.ravel(), zz.ravel()]))
                xx, zz = np.meshgrid(edge_axis, zs, indexing="ij")
                chunks.append(np.column_stack([xx.ravel(), np.full(xx.size, sign * h), zz.ravel()]))
    pos = np.concatenate(chunks, axis=0)
    return np.unique(np.round(pos, decimals=9), axis=0)


def cup_balls(r_in, r_out, length):
    """Vertical double-layer ring (inset 0.5 ps into the wall) plus a two-layer bottom disc."""
    ring_pts = []
    for radius in (r_in + 0.5 * PS, r_out - 0.5 * PS):
        n_theta = int(round(2.0 * math.pi * radius / PS))
        theta, zs = np.meshgrid(
            np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False),
            np.arange(0.0, length + 0.5 * PS, PS),
            indexing="ij",
        )
        ring_pts.append(np.stack([radius * np.cos(theta), radius * np.sin(theta), zs], axis=-1).reshape((-1, 3)))
    # bottom disc: two layers, kept one spacing clear of the ring's lowest row
    n_half = int(round((r_in - 0.5 * PS) / PS))
    axis = (np.arange(-n_half, n_half + 1) + 0.5) * PS
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    keep = np.hypot(gx, gy) <= r_in - 0.5 * PS
    disc_xy = np.stack([gx[keep], gy[keep]], -1)
    discs = [np.column_stack([disc_xy, np.full(len(disc_xy), -l * PS)]) for l in range(2)]

    pos = np.concatenate(ring_pts + discs, axis=0)
    _, uniq = np.unique(np.round(pos / PS).astype(np.int64), axis=0, return_index=True)
    return pos[np.sort(uniq)]


# ---------------------------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------------------------


def density_ratio(queries, sources):
    """Solver-convention rho/rho_rest for every query particle (mass == rho_rest => ratio = sum W)."""
    out = np.zeros(len(queries))
    for s in range(0, len(sources), 2048):
        delta = queries[:, None, :] - sources[None, s : s + 2048, :]
        d = np.linalg.norm(delta, axis=-1) / DIST_SCALE
        w = np.where((d > 0.0) & (d < 1.0), POLY6_COE * np.maximum(1.0 - d * d, 0.0) ** 3, 0.0)
        out += w.sum(axis=1)
    return out


def read_particles(solver, n_fluid):
    pos = solver.particles.pos.to_numpy()[:, 0, :]
    vel = solver.particles.vel.to_numpy()[:, 0, :]
    nan = int((~np.isfinite(pos)).sum() + (~np.isfinite(vel)).sum())
    return pos[:n_fluid], pos[n_fluid:], nan


def surface_height(z, r, r_in):
    """98th-percentile height of the particles well inside the wall (the free surface)."""
    inner = r < 0.8 * r_in
    return float(np.percentile(z[inner], 98)) if inner.sum() > 4 else float(z.min())


# ---------------------------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------------------------


def make_scene(extra_pbd):
    bounds = {k: extra_pbd.pop(k) for k in ("lower_bound", "upper_bound") if k in extra_pbd}
    gs.init(backend=gs.gpu, precision="32", seed=0)
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
        pbd_options=gs.options.PBDOptions(
            particle_size=PS,
            lower_bound=bounds.get("lower_bound", (-0.30, -0.30, -0.05)),
            upper_bound=bounds.get("upper_bound", (0.30, 0.30, 0.60)),
            max_density_solver_iterations=20,
            max_viscosity_solver_iterations=1,
            velocity_damping=1.0,
            **extra_pbd,
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )


def run_plate(n_layers, kinematic, seconds, depth, relax, lam_eps, control_plane):
    # Leave a swept-width margin between the translating plate and the hash/box bounds.  The old
    # +/-0.296 edge row translated by +/-0.02 inside a +/-0.30 domain: one side wrapped in the hash
    # while the other opened a 1.5 ps chute against the static side clamp, so its "full leak" was
    # a fixture-edge failure rather than evidence about CCD through the plate face.
    fluid_half, plate_half, n_cells = 0.088, 0.34, depth
    domain_half = 0.40
    npz = save_ball_set(
        f"plate_{n_layers}l.npz",
        plate_balls(plate_half, n_layers, fluid_half, 22, skirt_height=0.24),
    )
    extra = dict(
        boundary_ball_sets=((npz, kinematic),),
        boundary_ball_radius=None,
        density_lambda_epsilon=lam_eps,
        lower_bound=(-domain_half, -domain_half, -0.05),
        upper_bound=(domain_half, domain_half, 0.60),
    )
    if control_plane:
        # control: the same block on an analytic plane (impose) instead of the ball wall
        extra["boundary_plane"] = (0.0,)
    scene = make_scene(extra)
    water = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=RHO, c_init=0.0, density_relaxation=relax, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Box(pos=(0.0, 0.0, PS + 0.5 * (n_cells - 1) * PS), size=(2 * fluid_half, 2 * fluid_half, n_cells * PS)),
    )
    scene.build()
    solver = scene.sim.pbd_solver
    n_fluid = solver._n_fluid_particles
    print(f"fluid={n_fluid} balls={solver.n_boundary_balls} ranges={solver.boundary_ball_ranges}")
    assert solver.boundary_ball_ranges[0][2] == kinematic
    # Scene snapshots are entity-only; boundary balls are solver-owned geometry.
    state = solver.get_state(0)
    assert state.pos.shape == (1, solver.n_entity_particles, 3), state.pos.shape
    assert state.free.shape == (1, solver.n_entity_particles), state.free.shape
    ball_ref = solver.particles.pos.to_numpy()[n_fluid:, 0, :].copy()
    solver.set_state(0, state)
    ball_state_err = float(np.max(np.abs(solver.particles.pos.to_numpy()[n_fluid:, 0, :] - ball_ref)))
    assert ball_state_err == 0.0, f"state round-trip overwrote boundary balls: {ball_state_err}"
    print(
        f"entity-only state snapshot: pos {tuple(state.pos.shape)} free {tuple(state.free.shape)}; "
        f"ball round-trip error={ball_state_err:.3e}"
    )

    n_frames = int(round(seconds / DT))
    wall0, nan_max, z_min_min, ratio_min = time.time(), 0, 1e9, 1e9
    below_global_max, below_central_max = 0, 0
    central_half = fluid_half + PS
    x_follow = 0.0
    # mass-semantics calibration on the UNDEFORMED lattice: the wall layer (5 fluid neighbours +
    # 1 ball one spacing below) must reproduce the interior density exactly when ball mass == fluid mass
    pos0 = solver.particles.pos.to_numpy()[:, 0, :]
    src0 = np.concatenate([pos0[:n_fluid], pos0[n_fluid:]], axis=0)
    wall0_mask = (pos0[:n_fluid, 2] > PS) & (pos0[:n_fluid, 2] < 2.0 * PS)
    bulk0_mask = (pos0[:n_fluid, 2] > 0.4 * (PS + (n_cells - 1) * PS)) & (
        pos0[:n_fluid, 2] < 0.7 * (PS + (n_cells - 1) * PS)
    )
    ratio_t0 = float(
        np.median(density_ratio(pos0[:n_fluid][wall0_mask], src0))
        / max(np.median(density_ratio(pos0[:n_fluid][bulk0_mask], src0)), 1e-9)
    )
    print(
        f"t=0 lattice calibration: wall rho_tilde = {np.median(density_ratio(pos0[:n_fluid][wall0_mask], src0)):.4f}"
        f" (n={int(wall0_mask.sum())})  bulk = {np.median(density_ratio(pos0[:n_fluid][bulk0_mask], src0)):.4f}"
        f" (n={int(bulk0_mask.sum())})  ratio = {ratio_t0:.4f}"
    )
    for i in range(n_frames + 1):
        t = i * DT
        if i > 0:
            if kinematic and t >= 1.0:
                amp, om = 0.02, 2.0 * math.pi / 1.0
                solver.set_boundary_ball_pose(0, (amp * math.sin(om * (t - 1.0)), 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
            scene.step()

        pos_f, pos_b, nan = read_particles(solver, n_fluid)
        nan_max = max(nan_max, nan)
        safe_now = np.nan_to_num(pos_f, nan=0.0, posinf=0.0, neginf=0.0)
        z_min_min = min(z_min_min, float(safe_now[:, 2].min()))
        below_now = safe_now[:, 2] < -0.5 * PS
        central_now = np.maximum(np.abs(safe_now[:, 0]), np.abs(safe_now[:, 1])) <= central_half
        below_global_max = max(below_global_max, int(below_now.sum()))
        below_central_max = max(below_central_max, int((below_now & central_now).sum()))
        if i % 2 == 0 or i == n_frames:
            vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
            ke = float(0.5 * RHO * np.nansum((vel**2).sum(axis=1)))
            print(
                f"t={t:5.2f}s  z_min={float(safe_now[:, 2].min()):+.5f}  z_max={float(safe_now[:, 2].max()):+.5f}  "
                f"n_below={int((safe_now[:, 2] < -0.5 * PS).sum())}  "
                f"n_high={int((safe_now[:, 2] > 3.0 * PS + n_cells * PS).sum())}  KE={ke:.3e}  nan={nan}"
            )
        if i == n_frames:
            safe = np.nan_to_num(pos_f, nan=0.0, posinf=0.0, neginf=0.0)
            below = safe[:, 2] < -0.5 * PS
            if np.any(below):
                outside_plate = np.maximum(np.abs(safe[below, 0]), np.abs(safe[below, 1])) > plate_half - PS
                print(
                    f"final below-plane locality: n={int(below.sum())}, "
                    f"outside plate interior={int(outside_plate.sum())}, "
                    f"max|xy|={float(np.max(np.abs(safe[below, :2]))):.5f}"
                )
            wall = safe[:, 2] < 3.0 * PS
            bulk = (safe[:, 2] > 0.4 * (PS + (n_cells - 1) * PS)) & (safe[:, 2] < 0.7 * (PS + (n_cells - 1) * PS))
            src = np.concatenate([safe, pos_b], axis=0)
            r_wall = density_ratio(safe[wall], src)
            r_bulk = density_ratio(safe[bulk], src)
            wall_median = float(np.median(r_wall)) if len(r_wall) else float("nan")
            wall_mean = float(np.mean(r_wall)) if len(r_wall) else float("nan")
            bulk_median = float(np.median(r_bulk)) if len(r_bulk) else float("nan")
            bulk_mean = float(np.mean(r_bulk)) if len(r_bulk) else float("nan")
            ratio_min = wall_median / max(bulk_median, 1e-9) if np.isfinite(bulk_median) else float("nan")
            x_follow = float(np.mean(safe[:, 0]) - np.mean(pos_b[:, 0]))
            print(
                f"final: wall rho_tilde median/mean = {wall_median:.4f}/{wall_mean:.4f} (n={wall.sum()})  "
                f"bulk = {bulk_median:.4f}/{bulk_mean:.4f} (n={bulk.sum()})"
            )
            print(
                f"final: z_min={z_min_min:+.5f}  wall/bulk median ratio = {ratio_min:.4f}  "
                f"fluid-vs-plate mean dx = {x_follow:+.5f}  balls mean x = {float(np.mean(pos_b[:, 0])):+.5f}"
            )
        if i % int(round(0.5 / DT)) == 0 or i == n_frames:
            print(f"t={t:5.2f}s  z_min={float(safe_now[:, 2].min()):+.5f}  z_max={float(safe_now[:, 2].max()):+.5f}  nan={nan}")

    p_nan = nan_max == 0
    p_pen_central = below_central_max == 0
    p_pen_global = below_global_max == 0
    p_rho = 0.85 <= ratio_t0 <= 1.20
    print("=" * 78)
    print(f"[1] nan max = {nan_max}                                                     -> {'PASS' if p_nan else 'FAIL'}")
    print(
        f"[2a] central-footprint leak max = {below_central_max} "
        f"(|x|,|y| <= {central_half:.3f}; z < {-0.5 * PS:+.4f}) -> {'PASS' if p_pen_central else 'FAIL'}"
    )
    print(
        f"[2b] full-domain leak max = {below_global_max}; z_min = {z_min_min:+.5f} "
        f"(z < {-0.5 * PS:+.4f}) -> {'PASS' if p_pen_global else 'FAIL'}"
    )
    print(f"[3] wall/bulk rho_tilde ratio at t=0 = {ratio_t0:.4f} (limit 0.85..1.20)   -> {'PASS' if p_rho else 'FAIL'}")
    print(f"ball contacts overlap/swept = {solver.boundary_ball_collision_stats()}")
    print(f"wall={time.time() - wall0:.1f}s")
    if not (p_nan and p_pen_central and p_pen_global and p_rho):
        sys.exit(1)


def run_cup(seconds, with_balls):
    r_in, r_out, length = 0.15, 0.166, 0.316
    extra = {}
    if with_balls:
        load_mf16_ball_set(MF16_MUG)
        extra = dict(boundary_ball_sets=((MF16_MUG, False),))
    scene = make_scene(extra)
    water = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=RHO, c_init=0.0, density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Cylinder(pos=(0.0, 0.0, 2.0 * PS + 0.06), radius=r_in - 2.0 * PS, height=0.12),
    )
    scene.build()
    solver = scene.sim.pbd_solver
    n_fluid = solver._n_fluid_particles
    print(f"fluid={n_fluid} balls={solver.n_boundary_balls} ranges={solver.boundary_ball_ranges}")
    if with_balls:
        assert solver.n_boundary_balls > 0 and len(solver.boundary_ball_ranges) == 1

    n_frames = int(round(seconds / DT))
    wall0, nan_max, leak_max = time.time(), 0, 0
    z_surf, z_surf_prev, z_surf_drift = None, None, 0.0
    for i in range(n_frames + 1):
        t = i * DT
        if i > 0:
            scene.step()
        pos_f, _, nan = read_particles(solver, n_fluid)
        nan_max = max(nan_max, nan)
        safe = np.nan_to_num(pos_f, nan=0.0, posinf=0.0, neginf=0.0)
        r = np.hypot(safe[:, 0], safe[:, 1])
        z = safe[:, 2]
        # a fluid particle at or beyond the inner ball ring plane and below the rim has entered
        # the wall solid - the legitimate wall layer sits at r_in - 0.5 ps
        leak = int(((r >= r_in + 0.5 * PS) & (z < length - PS)).sum())
        leak_max = max(leak_max, leak)
        z_surf = surface_height(z, r, r_in)
        if i == n_frames - int(round(0.5 / DT)):
            z_surf_prev = z_surf
        if i % int(round(0.25 / DT)) == 0 or i == n_frames:
            print(f"t={t:5.2f}s  z_surf={z_surf:+.4f}  z_min={float(z.min()):+.4f}  r_max={float(r.max()):.4f}  leak={leak}  nan={nan}")
    z_surf_drift = abs(z_surf - z_surf_prev) if z_surf_prev is not None else float("inf")

    p_nan = nan_max == 0
    p_leak = leak_max == 0 if with_balls else leak_max > 0
    # PLAN's acceptance target is support/no leak.  Keep a weaker sanity check that the free
    # surface remains above the bottom and has settled, without imposing an arbitrary fill height
    # on this deliberately short, compressible PBF test slug.
    p_surf = z_surf is not None and z_surf > PS and z_surf_drift <= 2.0 * PS
    print("=" * 78)
    print(f"[1] nan max = {nan_max}                                                     -> {'PASS' if p_nan else 'FAIL'}")
    print(f"[2] leak max = {leak_max} (r >= {r_in + 0.5 * PS:.4f}, z < {length - PS:.3f}; expect {'0' if with_balls else '>0'})"
          f"        -> {'PASS' if p_leak else 'FAIL'}")
    if with_balls:
        print(f"[3] z_surf final = {z_surf:+.4f} (> 1 ps), drift last 0.5 s = {z_surf_drift:.4f} "
              f"(limit 2 ps = {2.0 * PS:.4f}) -> {'PASS' if p_surf else 'FAIL'}")
    else:
        print(f"[3] no-wall control surface = {z_surf:+.4f} (collapse/spill is expected; not a support PASS)")
    print(f"wall={time.time() - wall0:.1f}s")
    if not (p_nan and p_leak and (p_surf if with_balls else True)):
        sys.exit(1)


def run_adhesion(seconds):
    fluid_half, n_cells = 0.060, 8
    npz = save_ball_set("plate_adhesion.npz", plate_balls(0.20, 2, fluid_half, n_cells))
    scene = make_scene(
        dict(
            boundary_ball_sets=((npz, False),),
            surface_tension_enabled=True,
            st_compliance=1.0,
            st_surface_density_factor=1.0,
            st_max_surface_neighbors=256,
            wall_adhesion_enabled=True,
            wall_adhesion_compliance=20.0,
            wall_friction=0.15,
        )
    )
    water = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=RHO, c_init=0.0, density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Box(pos=(0.0, 0.0, PS + 0.5 * (n_cells - 1) * PS), size=(2 * fluid_half, 2 * fluid_half, n_cells * PS)),
    )
    scene.build()
    solver = scene.sim.pbd_solver
    n_fluid = solver._n_fluid_particles
    print(f"fluid={n_fluid} balls={solver.n_boundary_balls} ranges={solver.boundary_ball_ranges}")

    n_frames = int(round(seconds / DT))
    wall0, nan_max, adh_max, frames_adh, z_min_min = time.time(), 0, 0, 0, 1e9
    for i in range(n_frames + 1):
        t = i * DT
        if i > 0:
            scene.step()
        pos_f, _, nan = read_particles(solver, n_fluid)
        nan_max = max(nan_max, nan)
        z_min_min = min(z_min_min, float(np.nan_to_num(pos_f[:, 2], neginf=0.0).min()))
        n_adh = solver.wall_adhesion_stats()
        adh_max = max(adh_max, n_adh)
        frames_adh += 1 if n_adh > 0 else 0
        if i % int(round(0.25 / DT)) == 0 or i == n_frames:
            print(f"t={t:5.2f}s  n_adh={n_adh}  z_min={float(np.nan_to_num(pos_f[:, 2], neginf=0.0).min()):+.5f}  nan={nan}")

    p_nan = nan_max == 0
    p_adh = adh_max > 0
    p_support = z_min_min >= -0.5 * PS
    print("=" * 78)
    print(f"[1] nan max = {nan_max}                                                     -> {'PASS' if p_nan else 'FAIL'}")
    print(f"[2] n_adh max = {adh_max} over {frames_adh}/{n_frames + 1} frames (expect > 0)       -> {'PASS' if p_adh else 'FAIL'}")
    print(f"[3] z_min = {z_min_min:+.5f} (limit {-0.5 * PS:+.5f})                 -> {'PASS' if p_support else 'FAIL'}")
    print(f"wall={time.time() - wall0:.1f}s")
    if not (p_nan and p_adh and p_support):
        sys.exit(1)


def run_jug(seconds, tilt_deg):
    """Kinematic ball cup (bottom disc + double-layer wall) swung upright -> tilt -> upright.

    Escape is history-classified in the moving set's local frame.  A particle is permanently
    marked `cleared_rim` only after its observed relative trajectory clears the physical lip
    (including the jug's 2 mm lowered beak).  A particle that has never cleared the rim is a
    confirmed wall/bottom tunnel only after its *whole fluid sphere* reaches the far side of the
    visible solid.  Merely occupying the legal contact band is reported but is not called a leak.

    This is intentionally one-way bookkeeping: a confirmed tunnel can never be laundered into a
    legal pour by later rising above the rim.  At shallow tilt no particle should clear.  At 72
    degrees an egress is permitted but not manufactured as a requirement: this deliberately soft
    PBF fixture may compress below the lip, but any egress that does occur must first clear the rim.
    """

    r_in, r_out, length = 0.125, 0.141, 0.316
    fluid_radius = 0.5 * PS
    spout_dr, spout_dz, spout_half = 0.030, -0.002, math.radians(20.0)
    metric_tol = 0.10 * PS
    load_mf16_ball_set(MF16_JUG)
    scene = make_scene(
        dict(
            boundary_ball_sets=((MF16_JUG, True),),
            lower_bound=(-0.45, -0.45, -0.10),
            upper_bound=(0.45, 0.45, 0.70),
        )
    )
    scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=RHO, c_init=0.0, density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Cylinder(pos=(0.0, 0.0, 2.0 * PS + 0.06), radius=r_in - 2.0 * PS, height=0.12),
    )
    scene.build()
    solver = scene.sim.pbd_solver
    n_fluid = solver._n_fluid_particles
    print(f"fluid={n_fluid} balls={solver.n_boundary_balls} ranges={solver.boundary_ball_ranges}")
    assert solver.boundary_ball_ranges[0][2], "the jug set must be kinematic"

    # Tip about +y through the jug base so its +x beak/spout is the low side.  Tilting about x
    # would ignore the authored pour feature and reduce this to a plain circular-lip cup test.
    tilt = math.radians(tilt_deg)

    # The old 0.5 s hold only measured rigid following immediately after the ramp.  Keep the jug
    # at 72 degrees for 2 s so gravity has an independent interval in which to drive a real pour.
    ramp_start, ramp_end, hold_end, return_end = 0.5, 2.0, 4.0, 5.5

    def pose(t):
        if t < ramp_start or t > return_end:
            return 0.0
        if t < ramp_end:
            return tilt * (t - ramp_start) / (ramp_end - ramp_start)
        if t < hold_end:
            return tilt
        return tilt * (1.0 - (t - hold_end) / (return_end - hold_end))

    def to_local(p_world, angle):
        # The uploaded pose is Ry(+angle), so world -> jug-local is Ry(-angle).
        c, s = math.cos(angle), math.sin(angle)
        return np.stack(
            [c * p_world[:, 0] - s * p_world[:, 2], p_world[:, 1], s * p_world[:, 0] + c * p_world[:, 2]],
            axis=1,
        )

    n_frames = int(round(seconds / DT))
    wall0, nan_max = time.time(), 0
    fill_0, fill_end = None, None
    cleared_rim = np.zeros(n_fluid, dtype=bool)
    wall_tunnel = np.zeros(n_fluid, dtype=bool)
    bottom_tunnel = np.zeros(n_fluid, dtype=bool)
    prev_loc = None
    contact_max = 0
    for i in range(n_frames + 1):
        t = i * DT
        angle = pose(t)
        if i > 0:
            solver.set_boundary_ball_pose(
                0, (0.0, 0.0, 0.0), (math.cos(0.5 * angle), 0.0, math.sin(0.5 * angle), 0.0)
            )
            scene.step()
        pos, _, nan = read_particles(solver, n_fluid)
        nan_max = max(nan_max, nan)
        safe = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        loc = to_local(safe, angle)
        r_loc = np.hypot(loc[:, 0], loc[:, 1])

        theta = np.arctan2(loc[:, 1], loc[:, 0])
        w = np.where(np.abs(theta) <= spout_half, np.cos(0.5 * np.pi * theta / spout_half) ** 2, 0.0)
        rim_top = length + spout_dz * w
        rim_extent = r_out + spout_dr * w + fluid_radius
        rim_signed = loc[:, 2] - (rim_top + fluid_radius)

        # Legal rim history follows the authored lip itself: the +x beak is 2 mm lower and 30 mm
        # farther out than the circular rim.  A single global plane/extent would falsely legalize
        # wall crossings beside the spout.  The signed-crossing interpolation also catches a fast
        # particle whose frame endpoints straddle the local lip.
        clear_now = (rim_signed >= 0.0) & (r_loc <= rim_extent)
        if prev_loc is not None:
            theta0 = np.arctan2(prev_loc[:, 1], prev_loc[:, 0])
            w0 = np.where(
                np.abs(theta0) <= spout_half,
                np.cos(0.5 * np.pi * theta0 / spout_half) ** 2,
                0.0,
            )
            signed0 = prev_loc[:, 2] - (length + spout_dz * w0 + fluid_radius)
            crosses = (signed0 < 0.0) & (rim_signed >= 0.0) & (np.abs(rim_signed - signed0) > 1e-12)
            u = np.zeros(n_fluid, dtype=np.float64)
            u[crosses] = -signed0[crosses] / (rim_signed[crosses] - signed0[crosses])
            xy_cross = prev_loc[:, :2] + u[:, None] * (loc[:, :2] - prev_loc[:, :2])
            theta_cross = np.arctan2(xy_cross[:, 1], xy_cross[:, 0])
            w_cross = np.where(
                np.abs(theta_cross) <= spout_half,
                np.cos(0.5 * np.pi * theta_cross / spout_half) ** 2,
                0.0,
            )
            extent_cross = r_out + spout_dr * w_cross + fluid_radius
            clear_now |= crosses & (np.linalg.norm(xy_cross, axis=1) <= extent_cross)

        # Visible outer-wall radius at this (theta,z): gen_jug's outer top ring has a cos^2
        # beak and the wall is the ruled interpolation to the circular outer-bottom ring.
        qz = np.clip(loc[:, 2] / length, 0.0, 1.0)
        outer_surface = r_out + spout_dr * w * qz

        # "Contact" includes centres in/near the solid band but not fully through it.  Reporting
        # it prevents the new far-side criterion from hiding a pathological pile-up in the wall.
        contact = (
            (r_loc > r_in - fluid_radius - metric_tol)
            & (r_loc < outer_surface + fluid_radius + metric_tol)
            & (loc[:, 2] > -fluid_radius - metric_tol)
            & (loc[:, 2] < rim_top + fluid_radius + metric_tol)
        )
        contact_max = max(contact_max, int(contact.sum()))

        already_classified = cleared_rim | wall_tunnel | bottom_tunnel
        candidate_clear = clear_now & ~already_classified
        # Far-side thresholds include the fluid radius: these are complete crossings of the
        # visible solid, not legal centre positions in the hard-contact layer.
        new_bottom = (
            ~already_classified
            & ~candidate_clear
            & (loc[:, 2] <= -fluid_radius - metric_tol)
            & (r_loc <= r_out + fluid_radius + metric_tol)
        )
        new_wall = (
            ~already_classified
            & ~candidate_clear
            & (loc[:, 2] > -fluid_radius - metric_tol)
            & (loc[:, 2] <= rim_top - fluid_radius - metric_tol)
            & (r_loc >= outer_surface + fluid_radius + metric_tol)
        )
        cleared_rim |= candidate_clear
        bottom_tunnel |= new_bottom
        wall_tunnel |= new_wall

        inside = int(((r_loc < r_in) & (loc[:, 2] > 0.0) & (loc[:, 2] < length)).sum())
        fill_0 = inside if fill_0 is None else fill_0
        fill_end = inside
        prev_loc = loc.copy()
        if i % int(round(0.25 / DT)) == 0 or i == n_frames:
            print(
                f"t={t:5.2f}s  tilt={math.degrees(angle):5.1f}deg  fill={inside:6d}  z_loc_min={float(loc[:, 2].min()):+.4f}"
                f"  cleared={int(cleared_rim.sum()):5d}  tunnel(wall/bottom)="
                f"{int(wall_tunnel.sum())}/{int(bottom_tunnel.sum())}  contact={int(contact.sum()):4d}  nan={nan}"
            )

    p_nan = nan_max == 0
    p_tunnel = not wall_tunnel.any() and not bottom_tunnel.any()
    shallow = abs(tilt_deg) <= 10.0
    collision_overlap, collision_swept = solver.boundary_ball_collision_stats()
    p_flow = (not cleared_rim.any() and fill_end >= 0.99 * fill_0) if shallow else (collision_swept > 0)
    classified = int(cleared_rim.sum() + wall_tunnel.sum() + bottom_tunnel.sum())
    print("=" * 78)
    print(f"[1] nan max = {nan_max}                                                     -> {'PASS' if p_nan else 'FAIL'}")
    print(
        f"[2] confirmed tunnel history wall={int(wall_tunnel.sum())}, bottom={int(bottom_tunnel.sum())} "
        f"(expect 0/0; legal contact max={contact_max}) -> {'PASS' if p_tunnel else 'FAIL'}"
    )
    if shallow:
        print(
            f"[3] shallow retention fill {fill_0}->{fill_end}, cleared_rim={int(cleared_rim.sum())} "
            f"(expect >=99%, 0 cleared) -> {'PASS' if p_flow else 'FAIL'}"
        )
    else:
        print(
            f"[3] 72-deg moving-contact exercise overlap/swept={collision_overlap}/{collision_swept}; "
            f"cleared_rim={int(cleared_rim.sum())}/{n_fluid}, final strict-cavity fill={fill_end} "
            f"-> {'PASS' if p_flow else 'FAIL'}"
        )
    print(f"    terminal history census: resident={n_fluid - classified}, cleared={int(cleared_rim.sum())}, "
          f"wall_tunnel={int(wall_tunnel.sum())}, bottom_tunnel={int(bottom_tunnel.sum())}, total={n_fluid}")
    print(f"wall={time.time() - wall0:.1f}s")
    if not (p_nan and p_tunnel and p_flow):
        sys.exit(1)


def run_double_set():
    """Catch initial-pose and concatenated-local-index regressions with real MF16 products."""
    mug_local = load_mf16_ball_set(MF16_MUG)
    jug_local = load_mf16_ball_set(MF16_JUG)
    mug_initial_pos = np.array((-0.42, 0.0, 0.0), dtype=np.float32)
    jug_initial_pos = np.array((+0.42, 0.0, 0.0), dtype=np.float32)
    identity_quat = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    scene = make_scene(
        dict(
            boundary_ball_sets=((MF16_MUG, False), (MF16_JUG, True)),
            boundary_ball_initial_poses=(
                (tuple(mug_initial_pos), tuple(identity_quat)),
                (tuple(jug_initial_pos), tuple(identity_quat)),
            ),
            lower_bound=(-0.75, -0.35, -0.10),
            upper_bound=(0.75, 0.35, 0.65),
        )
    )
    scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=RHO, c_init=0.0, density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        # This probe sits at the origin. If build initializes both local-origin ball sets before
        # applying their configured poses, the automatic compile step records hard contacts here.
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.10), size=(2.0 * PS, 2.0 * PS, 2.0 * PS)),
    )
    scene.build()
    solver = scene.sim.pbd_solver
    assert solver.n_boundary_balls == len(mug_local) + len(jug_local)
    assert len(solver.boundary_ball_ranges) == 2
    m0, m1, mkin = solver.boundary_ball_ranges[0]
    j0, j1, jkin = solver.boundary_ball_ranges[1]
    assert not mkin and jkin
    assert m1 - m0 == len(mug_local) and j1 - j0 == len(jug_local) and j0 == m1

    pos0 = solver.particles.pos.to_numpy()[:, 0, :]
    mug_initial_expected = transform_wxyz(mug_local, mug_initial_pos, identity_quat)
    jug_initial_expected = transform_wxyz(jug_local, jug_initial_pos, identity_quat)
    mug_initial_err = float(np.max(np.abs(pos0[m0:m1].astype(np.float64) - mug_initial_expected)))
    jug_initial_err = float(np.max(np.abs(pos0[j0:j1].astype(np.float64) - jug_initial_expected)))
    jug_x_min = float(pos0[j0:j1, 0].min())
    build_contacts = solver.boundary_ball_collision_stats()

    state = solver.get_state(0)
    state_shape_ok = tuple(state.pos.shape) == (1, solver.n_entity_particles, 3)
    ball_state_ref = pos0[m0:j1].copy()
    solver.set_state(0, state)
    ball_state_err = float(
        np.max(np.abs(solver.particles.pos.to_numpy()[m0:j1, 0, :] - ball_state_ref))
    )

    angle = math.radians(31.0)
    trans = np.array((0.445, -0.018, 0.045), dtype=np.float32)
    quat = np.array((math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)), dtype=np.float32)
    solver.set_boundary_ball_pose(1, trans, quat)
    pos1 = solver.particles.pos.to_numpy()[:, 0, :]
    mug_static_err = float(np.max(np.abs(pos1[m0:m1].astype(np.float64) - mug_initial_expected)))
    jug_expected = transform_wxyz(jug_local, trans, quat)
    jug_pose_err = float(np.max(np.abs(pos1[j0:j1].astype(np.float64) - jug_expected)))

    guard_pass = True
    for label, bad_quat in (
        ("zero", np.zeros(4, dtype=np.float32)),
        ("nan", np.array((1.0, np.nan, 0.0, 0.0), dtype=np.float32)),
    ):
        try:
            solver.set_boundary_ball_pose(1, trans, bad_quat)
        except Exception as exc:
            print(f"  quaternion guard {label}: rejected ({type(exc).__name__})")
        else:
            print(f"  quaternion guard {label}: NOT rejected")
            guard_pass = False

    initial_guard_pass = True
    bad_initial_poses = (
        ("length", (((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),)),
        (
            "nan",
            (
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
                ((float("nan"), 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            ),
        ),
        (
            "zero-quat",
            (
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
                ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
            ),
        ),
    )
    for label, bad_poses in bad_initial_poses:
        try:
            gs.options.PBDOptions(
                boundary_ball_sets=((MF16_MUG, False), (MF16_JUG, True)),
                boundary_ball_initial_poses=bad_poses,
            )
        except Exception as exc:
            print(f"  initial-pose guard {label}: rejected ({type(exc).__name__})")
        else:
            print(f"  initial-pose guard {label}: NOT rejected")
            initial_guard_pass = False

    bounds_guard_pass = False
    try:
        bad_scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
            pbd_options=gs.options.PBDOptions(
                particle_size=PS,
                lower_bound=(-0.35, -0.35, -0.10),
                upper_bound=(0.35, 0.35, 0.65),
                boundary_ball_sets=((MF16_JUG, True),),
                boundary_ball_initial_poses=(((0.42, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),),
            ),
            show_viewer=False,
        )
        bad_scene.add_entity(
            material=gs.materials.PBD.Liquid(sampler="regular", rho=RHO),
            morph=gs.morphs.Box(pos=(0.0, 0.0, 0.10), size=(2.0 * PS, 2.0 * PS, 2.0 * PS)),
        )
        bad_scene.build()
    except Exception as exc:
        print(f"  initial-pose transformed-bounds guard: rejected ({type(exc).__name__})")
        bounds_guard_pass = True
    else:
        print("  initial-pose transformed-bounds guard: NOT rejected")

    tol_identity = 2.0e-7
    tol_pose = 3.0e-6
    p_initial = mug_initial_err <= tol_identity and jug_initial_err <= tol_identity
    p_build_separated = jug_x_min > 0.15 and build_contacts == (0, 0)
    p_static = mug_static_err <= tol_identity
    p_pose = jug_pose_err <= tol_pose
    p_finite = bool(np.isfinite(pos1[m0:j1]).all())
    print("=" * 78)
    print(
        f"[1] identity errors mug={mug_initial_err:.3e}, jug={jug_initial_err:.3e} "
        f"(limit {tol_identity:.1e}) -> {'PASS' if p_initial else 'FAIL'}"
    )
    print(
        f"[1b] build initial pose: jug x_min={jug_x_min:.3f}, compile-step contacts={build_contacts} "
        f"(expect x_min>0.15 and 0/0) -> {'PASS' if p_build_separated else 'FAIL'}"
    )
    print(
        f"[2] static set0 error after moving set1 = {mug_static_err:.3e} "
        f"(limit {tol_identity:.1e}) -> {'PASS' if p_static else 'FAIL'}"
    )
    print(
        f"[3] kinematic set1 NumPy pose max error = {jug_pose_err:.3e} "
        f"(limit {tol_pose:.1e}) -> {'PASS' if p_pose else 'FAIL'}"
    )
    print(f"[4] transformed positions finite = {p_finite} -> {'PASS' if p_finite else 'FAIL'}")
    print(f"[5] zero/NaN quaternion guards -> {'PASS' if guard_pass else 'FAIL'}")
    print(f"[6] initial-pose length/finite/nonzero guards -> {'PASS' if initial_guard_pass else 'FAIL'}")
    print(f"[7] initial-pose transformed hash-bounds guard -> {'PASS' if bounds_guard_pass else 'FAIL'}")
    p_state = state_shape_ok and ball_state_err == 0.0
    print(
        f"[8] entity-only state {tuple(state.pos.shape)}, ball round-trip error={ball_state_err:.3e} "
        f"-> {'PASS' if p_state else 'FAIL'}"
    )
    print(
        f"ranges={solver.boundary_ball_ranges}, entity_particles={solver._n_entity_particles}, "
        f"mug={len(mug_local)}, jug={len(jug_local)}"
    )
    if not (
        p_initial
        and p_build_separated
        and p_static
        and p_pose
        and p_finite
        and guard_pass
        and initial_guard_pass
        and bounds_guard_pass
        and p_state
    ):
        sys.exit(1)


def run_regress(forward):
    """Exec the archived selftest with __file__ faked to its original location (WORKSPACE = multiflow)."""
    src = open(ARCHIVE, encoding="utf-8").read()
    fake = os.path.join(WORKSPACE, "scripts", "mf15_pour_overflow.py")
    sys.argv = [fake] + forward
    exec(compile(src, ARCHIVE, "exec"), {"__name__": "__main__", "__file__": fake})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "case", choices=["plate", "cup", "negative", "adhesion", "jug", "double-set", "regress"],
    )
    parser.add_argument("--tilt-deg", type=float, default=72.0, help="jug: peak tilt angle in degrees")
    parser.add_argument("--dur", type=float, default=None, help="simulated seconds (case default otherwise)")
    parser.add_argument("--layers", type=int, default=2, choices=[1, 2], help="plate: ball layers")
    parser.add_argument("--depth", type=int, default=22, help="plate: fluid block layers (22 ~ 1e4 particles)")
    parser.add_argument("--relax", type=float, default=0.2, help="plate: material density_relaxation")
    parser.add_argument("--lam-eps", type=float, default=0.1, help="plate: density_lambda_epsilon")
    parser.add_argument("--kinematic", action="store_true", help="plate: oscillate the set through set_boundary_ball_pose")
    parser.add_argument("--control-plane", action="store_true", help="plate: add an analytic plane at z=0 as a load-bearing control")
    args, forward = parser.parse_known_args()

    if args.case == "plate":
        run_plate(
            args.layers, args.kinematic, args.dur if args.dur is not None else 2.0,
            args.depth, args.relax, args.lam_eps, args.control_plane,
        )
    elif args.case == "cup":
        run_cup(args.dur if args.dur is not None else 3.0, with_balls=True)
    elif args.case == "negative":
        run_cup(args.dur if args.dur is not None else 3.0, with_balls=False)
    elif args.case == "adhesion":
        run_adhesion(args.dur if args.dur is not None else 1.5)
    elif args.case == "jug":
        run_jug(args.dur if args.dur is not None else 6.0, args.tilt_deg)
    elif args.case == "double-set":
        run_double_set()
    else:
        run_regress(forward)


if __name__ == "__main__":
    main()
