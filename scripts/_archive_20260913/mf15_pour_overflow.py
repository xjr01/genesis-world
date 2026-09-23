"""MF-15: the flagship demo = handled mug of coffee + milk jug pour + surface-tension dome above
the rim + collapse and overflow running down the outer wall onto the table.

Narrative (all times CLI-tunable, defaults below):
  phase A (0-5s)    : rest - coffee settled in the mug (boundary_cup_shell), milk settled in the
                      upright jug (boundary_pitcher_shell) standing on the table
  phase B (5-7s)    : pure vertical lift to the pivot height - the upright body clears the mug
                      rim before anything approaches it
  phase C (7-11s)   : approach + descent + tilt on one smoothstep - the jug arrives at the pour
                      stance already at --tilt-max, so no upright pose hangs over the mug
  phase D (11-26s)  : hold 52 deg; the stream pours and the mug level climbs toward the rim
  phase E (26-34s)  : theta eases back to --tilt-trickle deg (38); the stream thins while the
                      level reaches the rim and the surface-tension dome grows above it
  phase F (34-46s)  : theta returns to 0 while the jug glides aside to origin_park - one shared
                      smoothstep window, because returning upright while still over the mug
                      would sink the jug wall into the mug rim; the thin stream keeps the flux
                      until the dome loses stability and spills, running down the outer wall
  phase G (46-52s)  : the parked jug and the full mug settle

Boundaries: static upright mug shell `boundary_cup_shell`, dynamic jug shell
`boundary_pitcher_shell` (driven per frame by `solver.set_pitcher_pose`), infinite table
`boundary_plane`. No boundary_cylinder, no boundary_group surgery beyond the milk entity's
constant group 1 (a milk particle leaving the jug keep region is transferred to group 0 by the
solver itself, i.e. poured milk is free).

Per-frame CSV: multiflow/videos/mf15_metrics.csv
  (frame, t, phase, theta_deg, sum_c, std_c, ke, nan, n_coffee, n_milk_in_jug, z_surf_p99,
   n_dome, n_over_rim, n_outer_wall, n_table, n_adh, first_over_flag, frame_ms)
Videos: multiflow/videos/mf15_pour_overflow_main.mp4 + mf15_pour_overflow_closeup.mp4

Run:
  # self-test (no npz, coarse ps=0.012, kinematic hold, compressed timeline, 8 s, no video;
  # tilts 72 deg with the lip pivot over the mug mouth so the pour actually clears the jug lip
  # and lands inside the mug - the narrative 52 deg pose cannot pour this jug's settled fill)
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf15_pour_overflow.py --selftest
  # full run (needs the presettle state first: multiflow/scripts/mf15_presettle.py)
  PYTHONIOENCODING=utf-8 "$PY" multiflow/scripts/mf15_pour_overflow.py \
    --settled multiflow/videos/mf15_settled.npz
"""

import argparse
import csv
import os
import sys
import time

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np  # noqa: E402

import genesis as gs  # noqa: E402

print("genesis loaded from:", gs.__file__)
assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), "Wrong genesis! Must be the multiflow copy."

MUG_OBJ = os.path.join(WORKSPACE, "assets", "mug.obj")
JUG_OBJ = os.path.join(WORKSPACE, "assets", "jug.obj")
VIDEOS_DIR = os.path.join(WORKSPACE, "videos")
CSV_PATH = os.path.join(VIDEOS_DIR, "mf15_metrics.csv")
MP4_MAIN = os.path.join(VIDEOS_DIR, "mf15_pour_overflow_main.mp4")
MP4_CLOSEUP = os.path.join(VIDEOS_DIR, "mf15_pour_overflow_closeup.mp4")
MP4_TOP = os.path.join(VIDEOS_DIR, "mf15_pour_overflow_top.mp4")

DT = 1.0 / 60.0
SUBSTEPS = 8

# mug: static upright shell (assets/gen_mug.py r_in=0.15 h_in=0.30 t=0.016/0.016, z=0 at the
# outer bottom) - analytic twin of the visual mesh
R_IN = 0.15
L_MUG = 0.30
T_WALL = 0.016
T_BOTTOM = 0.016
LIP_ROUND = 0.008
R_OUT = R_IN + T_WALL  # 0.166
Z_FLOOR = T_BOTTOM  # 0.016 cavity floor (= boundary z_bot; the outer bottom sits at z=0)
Z_RIM = Z_FLOOR + L_MUG  # 0.316
Z_TABLE = 0.0

# jug: dynamic shell (assets/gen_jug.py r_in=0.125 h_in=0.30 t=0.016/0.016; mesh spout at
# local +x, handle at local -x)
R_JUG = 0.125
L_JUG = 0.30
T_JUG_WALL = 0.016
T_JUG_BOTTOM = 0.016
# the mesh is authored with the spout at local +x, the pour side is world -x (the mug), so the
# visual body carries a fixed rotZ(180) on top of the tilt: local +x -> -e1 (down toward the
# mug), local -x (handle) -> +e1 (up, away)
QUAT_Z_PI = np.array([0.0, 0.0, 0.0, 1.0])

# jug placement and timeline defaults. The upright rest pose (PX, PZ) is what the presettle npz
# is bound to; the pour stance is defined by the lip PIVOT the jug rotates about. A pivot
# hovering inside/above the mug mouth keeps the stream landing in the cup - a pivot far outside
# it with a slow tilt lets the milk wrap the lip and run down the outer wall (teapot effect).
PX, PZ = 0.42, 0.016  # origin_0: standing on the table
PIVOT_X, PIVOT_Z = 0.215, 0.42  # origin_p = (PIVOT_X + R_JUG, 0, PIVOT_Z - L_JUG) = (0.34, 0.12)
PARK_X = 0.60  # origin_park x (z back on the table)
T_LIFT0, T_LIFT1 = 5.0, 7.0  # phase B: pure vertical lift
T_TILT0, T_TILT1 = 7.0, 11.0  # phase C: approach + descent + tilt together
T_TRICKLE = 26.0  # phase D holds until here, then theta -> trickle
T_BACK = 34.0  # phase E ends here (theta = trickle from here on)
T_PARK0, T_PARK1 = 42.0, 46.0  # phase F ends / phase G glide aside
TILT_MAX_DEG = 52.0
TILT_TRICKLE_DEG = 38.0
DUR = 52.0

PS_FULL = 0.008
PS_SELFTEST = 0.012
R_COFFEE = 0.14
H_COFFEE = 0.20
R_MILK_SELFTEST = 0.115
H_MILK_SELFTEST = 0.24
S0 = 0.002  # sample margin above the cavity floor
HOLD = 1.0  # selftest kinematic pin seconds (ends exactly when the compressed lift starts)
FIRST_OVER_STREAK = 8  # frames (0.133 s) of sustained n_over_rim >= 5 before first_over_t is
                       # recorded: single-frame hits are load-blast droplets and pour splash
T_FIRST_OVER_MIN = 2.5  # the load-blast splash rain dies by t~2 (KE floor); nothing before this
                        # counts as overflow, whatever n_over_rim says
T_CLEAN_STRAYS = 2.0  # deactivate the blast strays that landed on the table (they render as
                      # specks for the rest of the shot); same window as n_table, jug excluded


def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _ramp(t, t0, t1):
    return smoothstep((t - t0) / max(t1 - t0, 1e-9))


def quat_mul(q1, q2):
    """Hamilton product q1 * q2 (q2 is applied to a local vector first)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_roty_neg(theta):
    """Quaternion (w, x, y, z) of rotY(-theta): maps the mesh local +z to the jug axis."""
    return np.array([np.cos(theta / 2.0), 0.0, -np.sin(theta / 2.0), 0.0])


def jug_origin_p():
    return np.array([PIVOT_X + R_JUG, 0.0, PIVOT_Z - L_JUG])


def jug_park():
    return np.array([PARK_X, 0.0, PZ])


def jug_theta(t):
    """Jug tilt angle (rad) at time t: pour stance -> trickle stance -> upright.

    The return to upright shares the glide window [T_BACK, T_PARK1] with the park translation:
    coming upright while still at the pour stance puts the jug's outer wall through the mug rim.
    """
    th_max = np.deg2rad(TILT_MAX_DEG)
    th_tr = np.deg2rad(TILT_TRICKLE_DEG)
    if t < T_TILT0:
        return 0.0
    if t < T_TILT1:
        return th_max * _ramp(t, T_TILT0, T_TILT1)
    if t < T_TRICKLE:
        return th_max
    if t < T_BACK:
        return th_tr + (th_max - th_tr) * (1.0 - _ramp(t, T_TRICKLE, T_BACK))
    return th_tr * (1.0 - _ramp(t, T_BACK, T_PARK1))


def jug_pose(t):
    """Jug clamp pose at time t: (origin O, unit axis a, theta rad, tilt blend u).

    Pure function of t. Phase B is a pure vertical lift to the pivot height, so the upright
    body clears the mug rim before the approach starts; phase C combines the approach, the
    descent and the tilt on one smoothstep, so the jug arrives at the pour stance already at
    tilt-max and no upright pose hangs over the mug. D onward rotates about the pour-side lip
    pivot PIVOT = (pivot_x, 0, pivot_z) - the pour-side lip point of the pour stance
    (origin_p + L*a0 - R_JUG*e1 with a0 = +z, e1 = +x) - which stays fixed in the world while
    the body tilts. The theta return and the park glide share [T_BACK, T_PARK1]: coming upright
    while still over the mug would sink the jug wall into the mug rim.
    """
    th = jug_theta(t)
    a = np.array([-np.sin(th), 0.0, np.cos(th)])  # bottom -> mouth
    e1 = np.array([np.cos(th), 0.0, np.sin(th)])  # in-plane, toward the pour side
    origin_0 = np.array([PX, 0.0, PZ])
    lift_top = np.array([PX, 0.0, PIVOT_Z])
    origin_p = jug_origin_p()
    pivot = np.array([PIVOT_X, 0.0, PIVOT_Z])
    rot = (pivot - L_JUG * a + R_JUG * e1) - origin_p  # vanishes at theta = 0
    u_lift = _ramp(t, T_LIFT0, T_LIFT1)
    u_tilt = _ramp(t, T_TILT0, T_TILT1)
    u_park = _ramp(t, T_BACK, T_PARK1)
    origin = (origin_0
              + u_lift * (lift_top - origin_0)
              + u_tilt * (origin_p - lift_top)
              + rot
              + u_park * (jug_park() - origin_p))
    return origin, a, th, u_tilt


def _assert_no_overlap(t0, t1):
    """Startup guard over the whole timeline: every jug outer-wall point sitting at or below
    the mug rim (+4 mm) must clear the mug's outer surface by a margin. Catches an approach,
    upright hold, trickle or return pose that sinks the jug wall through the mug rim, and
    reports the offending time instead of leaving the contact in the render."""
    r_out_jug = R_JUG + T_JUG_WALL
    limit = R_OUT + 0.02
    worst_d, worst_t = float("inf"), None
    n = max(int(round((t1 - t0) / 0.05)) + 1, 2)
    for t in np.linspace(t0, t1, n):
        origin, axis, _, _ = jug_pose(t)
        sin_t, cos_t = -axis[0], axis[2]  # a = (-sin th, 0, cos th)
        for s in np.linspace(0.0, L_JUG, 61):
            pz = origin[2] + s * axis[2]
            if pz - r_out_jug * sin_t > Z_RIM + 0.004:
                break  # stations rise along the axis, so the rest is past the rim
            d = (origin[0] + s * axis[0]) - r_out_jug * cos_t
            if d < worst_d:
                worst_d, worst_t = d, t
            if d <= limit:
                print(f"jug/mug overlap: t={t:.2f}s wall station s={s:.3f} clearance "
                      f"{d:.4f} <= mug r_out + margin {limit:.4f} (raise --pivot-x / "
                      f"--pivot-z, or lower --tilt-max / --tilt-trickle)")
                sys.exit(3)
    print(f"jug/mug clearance over [{t0:g},{t1:g}]s: min horizontal "
          f"{worst_d:.4f} at t={worst_t:.2f}s (limit {limit:.4f}) OK")


def phase_of(t):
    if t < T_LIFT0:
        return 0  # A rest
    if t < T_TILT0:
        return 1  # B lift
    if t < T_TILT1:
        return 2  # C tilt up
    if t < T_TRICKLE:
        return 3  # D pour at tilt-max
    if t < T_BACK:
        return 4  # E trickle + dome
    return 5 if t < T_PARK1 else 6  # F overflow + return/glide, G settle


def longest_true_run(mask):
    """Length (frames) of the longest run of True in a 1-D bool array."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def best_window_frac(t_arr, ok, win_dur):
    """Max over start frames of the fraction of `ok` inside the ~`win_dur` window from it."""
    n = len(t_arr)
    if n == 0:
        return 0.0, 0, 0
    best_frac, bi, bj = 0.0, 0, 0
    j = 0
    csum = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    for i in range(n):
        while j + 1 < n and t_arr[j + 1] - t_arr[i] <= win_dur + 0.5 * DT:
            j += 1
        if t_arr[j] - t_arr[i] >= win_dur:
            frac = (csum[j + 1] - csum[i]) / (j + 1 - i)
            if frac > best_frac:
                best_frac, bi, bj = frac, i, j
    return best_frac, bi, bj


def main():
    global PX, PZ, PARK_X, PIVOT_X, PIVOT_Z, T_LIFT0, T_LIFT1, T_TILT0, T_TILT1
    global T_TRICKLE, T_BACK, T_PARK0, T_PARK1
    global TILT_MAX_DEG, TILT_TRICKLE_DEG, DUR

    parser = argparse.ArgumentParser()
    parser.add_argument("--dur", type=float, default=52.0, help="recorded seconds (default 52)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--settled", type=str, default=None,
                        help="npz from mf15_presettle.py: morphs are rebuilt from its metadata and "
                             "the settled positions are loaded (required unless --selftest)")
    parser.add_argument("--st-comp", type=float, default=1.0, help="st_compliance (smaller = stronger ST)")
    parser.add_argument("--adh-comp", type=float, default=20.0, help="wall_adhesion_compliance (smaller = stickier)")
    parser.add_argument("--wfric", type=float, default=0.05,
                        help="wall_friction: tangential damping for adhering particles (resists the "
                             "mercury-bead contraction of the table puddle; needs adhesion on)")
    parser.add_argument("--tilt-max", type=float, default=TILT_MAX_DEG)
    parser.add_argument("--tilt-trickle", type=float, default=TILT_TRICKLE_DEG)
    parser.add_argument("--t-lift0", type=float, default=T_LIFT0)
    parser.add_argument("--t-lift1", type=float, default=T_LIFT1)
    parser.add_argument("--t-tilt0", type=float, default=T_TILT0)
    parser.add_argument("--t-tilt1", type=float, default=T_TILT1)
    parser.add_argument("--t-trickle", type=float, default=T_TRICKLE)
    parser.add_argument("--t-back", type=float, default=T_BACK)
    parser.add_argument("--t-park0", type=float, default=T_PARK0,
                        help="unused: the theta return and the glide share [t-back, t-park1] "
                             "(kept so old command lines still parse)")
    parser.add_argument("--t-park1", type=float, default=T_PARK1)
    parser.add_argument("--px", type=float, default=PX, help="upright jug axis x (rest pose, npz-bound)")
    parser.add_argument("--pz", type=float, default=PZ, help="upright jug cavity-floor z")
    parser.add_argument("--pivot-x", type=float, default=PIVOT_X,
                        help="pour-side lip pivot x the jug rotates about (inside the mug mouth -> clean pour)")
    parser.add_argument("--pivot-z", type=float, default=PIVOT_Z, help="pour-side lip pivot z")
    parser.add_argument("--park-x", type=float, default=PARK_X)
    parser.add_argument("--point-size", type=float, default=9.0,
                        help="main-camera particle sprite size in px (spacing there is ~8 px)")
    parser.add_argument("--point-size-cup", type=float, default=29.0,
                        help="closeup-camera particle sprite size in px (spacing there is ~28 px)")
    parser.add_argument("--cam-top", action=argparse.BooleanOptionalAction, default=False,
                        help="add a straight-down diagnostic camera on the pour column")
    parser.add_argument("--dump-slabs", type=str, default="",
                        help="comma-separated times: dump the milk particles in the stream slab "
                             "z=[0.315,0.345] (mug rim -> jug lip) at the nearest frame of each, "
                             "with per-time N/centroid/eigenvalues/ellipticity/effective diameter")
    parser.add_argument("--dump-frames", type=str, default="",
                        help="comma-separated times: full pos+c+pose snapshots at the nearest "
                             "frame, saved to videos/mf15_dumps.npz for the offline renderer")
    parser.add_argument("--no-st", action="store_true",
                        help="A/B experiment: disable surface tension (+wall adhesion/friction, "
                             "which depend on it) entirely")
    parser.add_argument("--warmup", type=int, default=None,
                        help="off-camera re-settle frames before the first recorded frame: the "
                             "settled lattice is radially inhomogeneous, so the first substeps "
                             "drag wall-layer particles into the gaps and the mug surface "
                             "collapses briefly; 0 disables (defaults to 60, or 0 in --selftest "
                             "where the kinematic hold already covers it)")
    parser.add_argument("--clean-strays", action=argparse.BooleanOptionalAction, default=True,
                        help="deactivate the load-blast droplets stranded on the table at "
                             f"t={T_CLEAN_STRAYS:g}s (they render as specks); full runs only")
    parser.add_argument("--selftest", action="store_true",
                        help="npz-free quick mode: ps=0.012 inline fill, kinematic hold, compressed "
                             "timeline, 8 s, no video, 0.5 s prints, exit 1 on failure")
    args = parser.parse_args()

    PX, PZ, PARK_X = args.px, args.pz, args.park_x
    PIVOT_X, PIVOT_Z = args.pivot_x, args.pivot_z
    TILT_MAX_DEG, TILT_TRICKLE_DEG = args.tilt_max, args.tilt_trickle
    T_LIFT0, T_LIFT1 = args.t_lift0, args.t_lift1
    T_TILT0, T_TILT1 = args.t_tilt0, args.t_tilt1
    T_TRICKLE, T_BACK = args.t_trickle, args.t_back
    T_PARK0, T_PARK1 = args.t_park0, args.t_park1
    DUR = 8.0 if args.selftest else args.dur
    if args.selftest:
        # compressed timeline: A (held) -> B lift -> C tilt ramp -> D hold until t=8. The trickle
        # / park times are pushed beyond the window so theta holds at --tilt-max to the end (a
        # zero-length theta ramp would teleport the clamp and blast the milk). The selftest tilts
        # further and stands closer than the narrative defaults: this jug (L=0.30, R=0.125) only
        # brings its surface to the lip above ~63% fill at 52 deg, while the settled lattice fill
        # is ~57% - at 72 deg the spill threshold is ~29% fill, so the pour actually happens.
        # pivot_x 0.09 keeps the lip well inside the cavity (a wider stance sheds the stream on
        # the rim and out to the table); the overlap guard samples only this window's t=T_BACK
        # pose, which at 72 deg clears the mug by a wide margin.
        T_LIFT0, T_LIFT1, T_TILT0, T_TILT1 = HOLD, 2.5, 2.5, 5.5
        T_TRICKLE = T_BACK = T_PARK0 = T_PARK1 = DUR + 5.0
        # the trickle angle never runs inside the 8 s window, but the overlap guard samples the
        # t=T_BACK pose (the trickle stance by construction), so it is pinned to tilt-max here.
        # 72 deg is the lowest angle that pours in every run (65 sits right at the spill
        # threshold of the ~57% settled fill and sometimes does not flow at all); the impact
        # ejecta it throws onto the table is why S3 is a ratio, not a count. Friction is pinned
        # to 0 for the same reason (the full run keeps the --wfric default).
        TILT_MAX_DEG = TILT_TRICKLE_DEG = 72.0
        PX, PIVOT_X = 0.36, 0.09
        args.wfric = 0.0
        if args.warmup is None:
            args.warmup = 0  # the kinematic hold covers the blast; pass --warmup N to override
        args.no_video = True
    if args.warmup is None:
        args.warmup = 60
    ps = PS_SELFTEST if args.selftest else PS_FULL

    # ---- settled state: fail BEFORE the engine is initialized ----------------------------
    r_coffee, h_coffee, r_milk, h_milk = R_COFFEE, H_COFFEE, R_MILK_SELFTEST, H_MILK_SELFTEST
    settled_pos = None
    if args.selftest:
        pass
    elif args.settled is None:
        raise SystemExit(
            "mf15 needs the presettle state: run multiflow/scripts/mf15_presettle.py and pass "
            "--settled multiflow/videos/mf15_settled.npz (or use --selftest for the npz-free check)."
        )
    elif not os.path.isfile(args.settled):
        raise SystemExit(f"settled npz not found: {args.settled} - run mf15_presettle.py first.")
    else:
        dat = np.load(args.settled)
        need = ["pos", "n_coffee", "n_milk", "coffee_r", "coffee_h0", "milk_r", "milk_h0"]
        missing = [k for k in need if k not in dat.files]
        if missing:
            raise SystemExit(f"settled npz {args.settled} is missing keys {missing} (has {sorted(dat.files)})")
        r_coffee, h_coffee = float(dat["coffee_r"]), float(dat["coffee_h0"])
        r_milk, h_milk = float(dat["milk_r"]), float(dat["milk_h0"])
        settled_pos = dat["pos"].astype(np.float32)

    origin_0 = np.array([PX, 0.0, PZ])
    mesh_pos_0 = origin_0 - T_JUG_BOTTOM * np.array([0.0, 0.0, 1.0])
    mp4_paths = [] if args.no_video else [MP4_MAIN, MP4_CLOSEUP]
    _assert_no_overlap(0.0, DUR)

    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
        pbd_options=gs.options.PBDOptions(
            particle_size=ps,
            lower_bound=(-1.0, -0.9, 0.0),
            upper_bound=(1.7, 0.9, 1.1),
            boundary_cup_shell=(0.0, 0.0, Z_FLOOR, R_IN, L_MUG, T_WALL, T_BOTTOM, LIP_ROUND),
            boundary_pitcher_shell=(PX, 0.0, PZ, 0.0, 0.0, 1.0, R_JUG, L_JUG, T_JUG_WALL,
                                    T_JUG_BOTTOM, LIP_ROUND),
            boundary_plane=(Z_TABLE,),
            max_density_solver_iterations=20,
            max_viscosity_solver_iterations=1,
            density_lambda_epsilon=0.1,
            diffusion_coeff=0.01,
            surface_tension_enabled=not args.no_st,
            st_compliance=args.st_comp,
            st_surface_density_factor=1.0,
            st_distance_enabled=True,
            st_distance_compliance=40.0,
            st_max_surface_neighbors=256,
            velocity_damping=1.0,
            wall_adhesion_enabled=not args.no_st,
            wall_adhesion_compliance=args.adh_comp,
            wall_friction=0.0 if args.no_st else args.wfric,
        ),
        vis_options=gs.options.VisOptions(
            render_particle_as="points",  # per-particle concentration coloring
        ),
        show_viewer=False,
    )

    # mug: visualization only (boundary_cup_shell does the physics); handle on -x
    mug = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=MUG_OBJ,
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            decimate=False,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.85, 0.85, 0.9, 0.35), opacity=0.35),
    )
    # jug: visualization only (dynamic shell boundary does the physics). FIXED body teleported
    # every frame - a free body drifts under the PBF t=0 lattice blast (mf12 lesson). euler 0:
    # the pour-side flip lives in the per-frame quaternion.
    jug = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=JUG_OBJ,
            pos=tuple(mesh_pos_0),
            euler=(0.0, 0.0, 0.0),
            fixed=True,
            decimate=False,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.9, 0.92, 0.95, 0.45), opacity=0.45),
    )
    # table visual: top face exactly at the PlaneBoundary height (mf14 convention); sized so the
    # splash (r p90 ~ 0.7) stays on the visible deck instead of hovering past its edge
    table = scene.add_entity(
        morph=gs.morphs.Box(size=(2.6, 1.6, 0.04), pos=(0.35, 0.0, Z_TABLE - 0.02), fixed=True),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.55, 0.55, 0.58, 1.0), opacity=1.0),
    )

    coffee = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=1.0,
            density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Cylinder(
            radius=r_coffee,
            height=h_coffee,
            pos=(0.0, 0.0, Z_FLOOR + S0 + 0.5 * h_coffee),
        ),
    )
    # milk sampled upright INSIDE the jug cavity at the initial pose, so the build warm-up step
    # never sees it outside the keep region (boundary_group stays 1 without any surgery)
    milk = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=0.0, boundary_group=1,
            density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Cylinder(
            radius=r_milk,
            height=h_milk,
            pos=(PX, 0.0, PZ + S0 + 0.5 * h_milk),
        ),
    )

    # front view (-y): the mug handle (-x side) reads on the left of frame and the jug (+x) on
    # the right - the old three-quarter view hid the handle behind the mug body
    cam = scene.add_camera(res=(960, 1280), pos=(0.28, -1.42, 0.60), lookat=(0.22, 0.0, 0.20), fov=42, GUI=False)
    cam_close = scene.add_camera(res=(960, 1280), pos=(0.44, -0.40, 0.40), lookat=(0.06, 0.0, 0.29), fov=32, GUI=False)
    # straight-down diagnostic view of the pour column; the explicit horizontal up vector keeps
    # the look-at build non-degenerate for a -z view direction
    cam_top = None
    if args.cam_top:
        cam_top = scene.add_camera(res=(960, 1280), pos=(0.10, 0.02, 0.85), lookat=(0.10, 0.02, 0.25),
                                   up=(0.0, 1.0, 0.0), fov=50, GUI=False)
    scene.build()
    print(f"mug geoms: {mug.n_geoms}, jug geoms: {jug.n_geoms}, table geoms: {table.n_geoms}, "
          f"coffee particles: {coffee.n_particles}, milk particles: {milk.n_particles}")
    print(
        f"ps={ps} dt={DT:.5f} substeps={SUBSTEPS} st_comp={args.st_comp:g} adh_comp={args.adh_comp:g} "
        f"wfric={args.wfric:g} "
        f"tilt_max={TILT_MAX_DEG:g} tilt_trickle={TILT_TRICKLE_DEG:g} "
        f"lift=({T_LIFT0:g},{T_LIFT1:g}) tilt=({T_TILT0:g},{T_TILT1:g}) trickle={T_TRICKLE:g} "
        f"back={T_BACK:g} park=({T_PARK0:g},{T_PARK1:g}) dur={DUR:g} selftest={args.selftest} "
        f"cam_top={args.cam_top} slabs='{args.dump_slabs}' warmup={args.warmup}"
    )
    print(f"jug: origin_0=({PX:g},0,{PZ:g}) pivot=({PIVOT_X:g},0,{PIVOT_Z:g}) "
          f"origin_p={np.round(jug_origin_p(), 3).tolist()} "
          f"park={np.round(jug_park(), 3).tolist()} | mug r_in={R_IN} rim z={Z_RIM:.3f} "
          f"| coffee r={r_coffee:.3f} h0={h_coffee:.3f} | milk r={r_milk:.3f} h0={h_milk:.3f}")

    # per-camera sprite size: the two viewpoints sit at very different distances, so one value
    # cannot close the gaps between particles in both frames
    scene.visualizer.rasterizer._camera_targets[cam.uid].point_size = args.point_size
    scene.visualizer.rasterizer._camera_targets[cam_close.uid].point_size = args.point_size_cup
    if cam_top is not None:
        scene.visualizer.rasterizer._camera_targets[cam_top.uid].point_size = 18.0

    solver = scene.sim.pbd_solver
    assert type(solver.boundary).__name__ == "CubeBoundary", f"unexpected boundary: {type(solver.boundary)}"
    assert type(solver.boundary2).__name__ == "TiltedCylinderShellBoundary", (
        f"unexpected boundary2: {type(solver.boundary2)}"
    )
    assert type(solver.boundary_cup).__name__ == "TiltedCylinderShellBoundary", (
        f"unexpected boundary_cup: {type(solver.boundary_cup)}"
    )
    assert type(solver.boundary3).__name__ == "PlaneBoundary", f"unexpected boundary3: {type(solver.boundary3)}"
    n_fluid = solver._n_fluid_particles
    n_coffee = coffee.n_particles
    n_milk = milk.n_particles
    assert n_coffee + n_milk == n_fluid
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    # boundary_group sanity: coffee 0 (mug-free), milk 1 (jug-owned until it leaves the keep region)
    gid = solver.particles_ng.boundary_group.to_numpy()[:n_fluid, 0]
    uniq_c, cnt_c = np.unique(gid[:n_coffee], return_counts=True)
    _bg = solver.particles_ng.boundary_group.to_numpy()
    _bg[n_coffee:n_fluid, :] = 1
    solver.particles_ng.boundary_group.from_numpy(_bg)
    print(f"boundary_group: coffee={dict(zip(uniq_c.tolist(), cnt_c.tolist()))} -> milk set to 1 ({n_milk})")

    # ---- initial state -------------------------------------------------------------------
    pos_np = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
    cent0 = np.stack([pos_np[:n_coffee].mean(axis=0), pos_np[n_coffee:n_fluid].mean(axis=0)])

    if settled_pos is not None:
        assert settled_pos.shape == (n_fluid, 1, 3), (
            f"settled pos shape {settled_pos.shape} != scene ({n_fluid}, 1, 3)"
        )
        assert int(dat["n_coffee"]) == n_coffee and int(dat["n_milk"]) == n_milk, (
            f"settled npz counts ({int(dat['n_coffee'])}, {int(dat['n_milk'])}) != scene "
            f"({n_coffee}, {n_milk}) - rebuild the npz with the same morphs/ps"
        )
        solver.particles.pos.from_numpy(settled_pos)
        solver.particles.vel.from_numpy(np.zeros_like(settled_pos))
        print(f"loaded settled state from {args.settled} (keys: {sorted(dat.files)})")
        chk = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
        r_c = np.sqrt(chk[:n_coffee, 0] ** 2 + chk[:n_coffee, 1] ** 2)
        out_c = int(((r_c > R_IN + ps) | (chk[:n_coffee, 2] < Z_FLOOR - ps)).sum())
        rel_m = chk[n_coffee:n_fluid] - origin_0
        s_m = rel_m @ np.array([0.0, 0.0, 1.0])
        r_m = np.linalg.norm(rel_m - s_m[:, None] * np.array([0.0, 0.0, 1.0]), axis=1)
        out_m = int(((r_m > R_JUG + ps) | (s_m < -ps) | (s_m > L_JUG + ps)).sum())
        print(f"settled check: coffee outside mug = {out_c}/{n_coffee}, milk outside jug = {out_m}/{n_milk}")
        assert out_c <= 0.01 * n_coffee, "settled coffee is not inside the mug"
        assert out_m <= 0.01 * n_milk, "settled milk is not inside the jug"

    n_frames = int(round(DUR / DT))
    rows = []

    # ---- off-camera warm-up (mf14 resettle precedent) --------------------------------------
    # the settled lattice is radially inhomogeneous (an under-filled near-wall shell), so the
    # first substeps drag wall-layer particles into the gaps and the mug surface collapses
    # briefly; letting that happen before the first recorded frame keeps the shot clean. The
    # jug holds the phase-A pose; nothing is recorded and no CSV row is produced. The main
    # loop's t still counts from 0, so the t=2 stray clean-up simply fires one warm-up later
    # in physical time.
    if args.warmup > 0:
        origin, axis, _, _ = jug_pose(0.0)
        solver.set_pitcher_pose(origin, axis)
        jug.set_pos(origin - T_JUG_BOTTOM * axis, relative=True, zero_velocity=True)
        jug.set_quat(quat_mul(quat_roty_neg(0.0), QUAT_Z_PI), relative=True, zero_velocity=True)

        def warm_stats():
            pos_np = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
            vel_np = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
            safe_np = np.nan_to_num(pos_np, nan=0.0, posinf=0.0, neginf=0.0)
            r_np = np.sqrt(safe_np[:, 0] ** 2 + safe_np[:, 1] ** 2)
            in_mug = (r_np <= R_IN + ps) & (safe_np[:, 2] >= Z_FLOOR - ps)
            z_s = float(np.percentile(safe_np[in_mug, 2], 99)) if in_mug.any() else float("nan")
            return z_s, float(0.5 * np.nansum((vel_np**2).sum(axis=1)))

        zs0, ke0 = warm_stats()
        for _ in range(args.warmup):
            scene.step()
        zs1, ke1 = warm_stats()
        vel_np = solver.particles.vel.to_numpy()  # absorb the collapse residual
        vel_np[:n_fluid, 0, :] = 0.0
        solver.particles.vel.from_numpy(vel_np)
        print(f"warm-up {args.warmup} frames: z_surf {zs0:.3f} -> {zs1:.3f}, "
              f"KE {ke0:.3e} -> {ke1:.3e} (residual velocities zeroed)")

    first_over_t = None
    over_streak = 0
    strays_done = False
    act_mask = None
    all_valid = np.ones(n_fluid, dtype=bool)
    z_min_run = float("inf")
    wall0 = time.time()
    if not args.no_video:
        cam.start_recording(save_to_filename=MP4_MAIN, fps=args.fps)
        cam_close.start_recording(save_to_filename=MP4_CLOSEUP, fps=args.fps)
        if cam_top is not None:
            cam_top.start_recording(save_to_filename=MP4_TOP, fps=args.fps)
    print_every = int(round((0.5 if args.selftest else 1.0) / DT))
    slab_frames = {}
    if args.dump_slabs:
        for ts in [float(v) for v in args.dump_slabs.split(",") if v.strip()]:
            slab_frames[min(max(int(round(ts / DT)), 0), n_frames)] = ts  # nearest frame
    slab_records = []
    # offline-renderer frame dumps: full pos+c snapshots at chosen times (mf15_dumps.npz)
    dump_frames = {}
    if args.dump_frames:
        for ts in [float(v) for v in args.dump_frames.split(",") if v.strip()]:
            dump_frames[min(max(int(round(ts / DT)), 0), n_frames)] = ts
    frame_dumps = {}

    for i in range(n_frames + 1):
        # ---- animation: physics clamp pose + visual mesh pose -----------------------------
        t = i * DT
        origin, axis, th, _ = jug_pose(t)
        solver.set_pitcher_pose(origin, axis)
        # relative=True: user (morph) frame, same convention as the morph's own pos/euler -
        # relative=False would target the solver's inertial link frame (mf11 lesson)
        jug.set_pos(origin - T_JUG_BOTTOM * axis, relative=True, zero_velocity=True)
        jug.set_quat(quat_mul(quat_roty_neg(th), QUAT_Z_PI), relative=True, zero_velocity=True)

        frame_t0 = time.time()
        if i > 0:
            scene.step()
            # the hold and --warmup are two devices for the same job (dissipate the t=0 lattice
            # blast), so exactly one of them runs: a warm-up already relaxed the state off-camera
            if args.selftest and args.warmup <= 0 and t <= HOLD:
                pos_np = solver.particles.pos.to_numpy()
                blk = pos_np[:n_fluid, 0, :]
                blk[:n_coffee] += cent0[0] - blk[:n_coffee].mean(axis=0)
                blk[n_coffee:n_fluid] += cent0[1] - blk[n_coffee:n_fluid].mean(axis=0)
                solver.particles.pos.from_numpy(pos_np)
                vel_np = solver.particles.vel.to_numpy()
                vel_np[:n_fluid, 0, :] *= 0.3
                solver.particles.vel.from_numpy(vel_np)
            elif args.clean_strays and not args.selftest and not strays_done and t >= T_CLEAN_STRAYS:
                # the t=0 lattice blast strands droplets on the table (mostly coffee) where they
                # render as specks; retire them with the solver's particle-active flag (they stay
                # in the arrays but leave the hash/neighbor loops and the render)
                pos_np = solver.particles.pos.to_numpy()
                blk = pos_np[:n_fluid, 0, :]
                r_w = np.sqrt(blk[:, 0] ** 2 + blk[:, 1] ** 2)
                rel_s = blk - origin
                s_w = rel_s @ axis
                rj_w = np.linalg.norm(rel_s - s_w[:, None] * axis[None, :], axis=1)
                in_jug = (s_w > -(T_JUG_BOTTOM + 3.0 * ps)) & (s_w < L_JUG + 3.0 * ps) \
                    & (rj_w <= R_JUG + T_JUG_WALL + 3.0 * ps)
                strays = (blk[:, 2] < 2.5 * ps) & (r_w > R_OUT + 3.0 * ps) & ~in_jug
                n_strays = int(strays.sum())
                if n_strays:
                    coffee.set_particles_active(~strays[:n_coffee])
                    milk.set_particles_active(~strays[n_coffee:])
                    # set_particles_active() applied exactly ~strays; reading it back via
                    # get_particles_active() returns a torch cuda tensor (np.array fails on it),
                    # and would be redundant anyway -- rebuild the mask from what we set.
                    act_mask = ~strays.copy()
                    print(f"clean-strays @t={t:.2f}s: deactivated {n_strays} table strays "
                          f"(coffee {int(strays[:n_coffee].sum())}, milk {int(strays[n_coffee:].sum())}); "
                          f"metrics use the active mask from here on")
                strays_done = True

        # ---- metrics ----------------------------------------------------------------------
        phase = phase_of(t)
        pos_all = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
        vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
        c = solver.particles.c.to_numpy()[:n_fluid, 0]
        nan_count = int((~np.isfinite(pos_all)).sum() + (~np.isfinite(vel)).sum() + (~np.isfinite(c)).sum())
        safe = np.nan_to_num(pos_all, nan=0.0, posinf=0.0, neginf=0.0)
        safe_v = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        safe_c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        valid = act_mask if act_mask is not None else all_valid
        sum_c = float(safe_c[valid].sum())
        std_c = float(safe_c[valid].std())
        ke = float(0.5 * np.nansum((safe_v[valid] ** 2).sum(axis=1)))
        r_all = np.sqrt(safe[:, 0] ** 2 + safe[:, 1] ** 2)
        z_all = safe[:, 2]
        rel_j = safe[n_coffee:n_fluid] - origin
        s_j = rel_j @ axis
        r_j = np.linalg.norm(rel_j - s_j[:, None] * axis[None, :], axis=1)
        # particles still inside the jug's own footprint (cavity + wall + escape band): the jug
        # stands on the table with its cavity floor at z=Z_FLOOR > 2.5 ps, and its carried milk
        # rides above the mug rim, so without this mask the literal n_table / n_over_rim windows
        # count the jug's own bottom layers and its milk instead of spilled milk
        in_jug_fp = (s_j > -(T_JUG_BOTTOM + 3.0 * ps)) & (s_j < L_JUG + 3.0 * ps) \
            & (r_j <= R_JUG + T_JUG_WALL + 3.0 * ps)
        not_jug = np.ones(n_fluid, dtype=bool)
        not_jug[n_coffee:] = ~in_jug_fp
        # slow particles inside the mug cavity: the falling stream is excluded from the surface
        # estimate, the dome above the rim is deliberately NOT (no z < Z_RIM gate here)
        slow = (np.abs(safe_v[:, 2]) <= 0.2) & valid
        in_mug = (r_all <= R_IN + ps) & (z_all >= Z_FLOOR - ps) & valid
        in_mug_slow = in_mug & slow
        z_surf = float(np.percentile(z_all[in_mug_slow], 99)) if in_mug_slow.any() else float("nan")
        n_coffee_mug = int(in_mug[:n_coffee].sum())
        n_dome = int((slow & (z_all > Z_RIM) & (r_all <= R_IN + ps)).sum())
        n_over_rim = int(((z_all > Z_RIM + 0.5 * ps) & (r_all > R_OUT - ps) & not_jug & valid).sum())
        n_outer_wall = int(((r_all >= R_OUT - 0.5 * ps) & (r_all <= R_OUT + 2.5 * ps)
                            & (z_all >= 0.02) & (z_all <= Z_RIM) & not_jug & valid).sum())
        n_table = int(((z_all <= 2.5 * ps) & (r_all > R_OUT + 3.0 * ps) & not_jug & valid).sum())
        n_milk_jug = int(((s_j > -ps) & (s_j < L_JUG + ps) & (r_j <= R_JUG + ps)).sum())
        n_milk_mug = int(in_mug[n_coffee:].sum())
        if i in dump_frames:
            # full-state snapshot for the offline (non-genesis) renderer: pos + concentration
            frame_dumps[f"{dump_frames[i]:g}"] = safe[:n_fluid].copy()
            frame_dumps[f"c_{dump_frames[i]:g}"] = solver.particles.c.to_numpy()[:n_fluid, 0].copy()
            frame_dumps[f"pose_{dump_frames[i]:g}"] = np.concatenate([origin, axis, [th]])
            frame_dumps[f"valid_{dump_frames[i]:g}"] = valid[:n_fluid].copy()
        if i in slab_frames:
            # cross-section of the pour column in the z band between the mug rim and the jug
            # lip, in two variants: every milk particle in the band (stream plus whatever clings
            # to the jug wall), and the free-falling part alone (jug footprint excluded) - with
            # adhesion the column partly hugs the wall, so the two answers differ by design
            mk_xy = safe[n_coffee:n_fluid, :2]
            mk_z = z_all[n_coffee:n_fluid]
            in_band = (mk_z >= 0.315) & (mk_z <= 0.345)
            selections = (
                ("all", valid[n_coffee:] & in_band),
                ("free", valid[n_coffee:] & not_jug[n_coffee:] & in_band),
            )
            rec = [slab_frames[i]]
            line = f"slab t={slab_frames[i]:.2f}s:"
            for name, msk in selections:
                pts = mk_xy[msk]
                n_pts = int(pts.shape[0])
                cen = pts.mean(axis=0) if n_pts else np.array([np.nan, np.nan])
                if n_pts >= 2:
                    ev = np.linalg.eigvalsh(np.cov(pts.T, ddof=0))
                    lam1, lam2 = float(ev[1]), float(ev[0])
                    ellip = float(np.sqrt(max(lam2, 0.0) / lam1)) if lam1 > 0.0 else float("nan")
                else:
                    lam1 = lam2 = ellip = float("nan")
                d_eff = 2.0 * np.sqrt(n_pts * 0.8 * ps * ps / np.pi)
                line += (f" | {name}: N={n_pts} cen=({cen[0]:.4f},{cen[1]:.4f}) "
                         f"lam=({lam1:.6f},{lam2:.6f}) e={ellip:.3f} d_eff={d_eff:.4f}")
                rec.extend([n_pts, pts, cen, (lam1, lam2), ellip, d_eff])
            print(line)
            slab_records.append(rec)
        over_streak = over_streak + 1 if (n_over_rim >= 5 and t >= T_FIRST_OVER_MIN) else 0
        if first_over_t is None and over_streak >= FIRST_OVER_STREAK:
            first_over_t = t - (FIRST_OVER_STREAK - 1) * DT  # the streak's first frame
        z_min_run = min(z_min_run, float(z_all[valid].min()))
        safe_last = safe
        n_adh = solver.wall_adhesion_stats()
        frame_ms = (time.time() - frame_t0) * 1000.0
        rows.append((i, t, phase, float(np.rad2deg(th)), sum_c, std_c, ke, nan_count, n_coffee_mug,
                     n_milk_jug, z_surf, n_dome, n_over_rim, n_outer_wall, n_table, n_adh,
                     0 if first_over_t is None else 1, frame_ms))
        if i % print_every == 0 or i == n_frames:
            wall = time.time() - wall0
            print(
                f"t={t:6.2f}s  ph={phase}  th={np.rad2deg(th):5.1f}  jug={n_milk_jug:6d}  "
                f"mugM={n_milk_mug:5d}  z_surf={z_surf:6.3f}  dome={n_dome:5d}  over={n_over_rim:4d}  "
                f"outer={n_outer_wall:5d}  table={n_table:5d}  KE={ke:.3e}  nan={nan_count}  "
                f"({wall / max(i, 1) * 1000:.0f} ms/frame avg)"
            )

    if not args.no_video:
        cam.stop_recording()
        cam_close.stop_recording()
        if cam_top is not None:
            cam_top.stop_recording()

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["frame", "t", "phase", "theta_deg", "sum_c", "std_c", "ke", "nan", "n_coffee",
             "n_milk_in_jug", "z_surf_p99", "n_dome", "n_over_rim", "n_outer_wall", "n_table",
             "n_adh", "first_over_flag", "frame_ms"]
        )
        w.writerows(rows)

    if slab_records:
        # per record: t, N, xy, centroid, (lam1, lam2), ellipticity, effective diameter - the
        # first six for the raw band, the last six for the jug-footprint-free part
        slab_path = os.path.join(VIDEOS_DIR, "mf15_stream_slab.npz")
        payload = {"times": np.array([r[0] for r in slab_records])}
        for name, off in (("all", 1), ("free", 7)):
            payload[f"Ns_{name}"] = np.array([r[off] for r in slab_records], dtype=np.int64)
            payload[f"cents_{name}"] = np.array([r[off + 2] for r in slab_records])
            payload[f"eigvals_{name}"] = np.array([r[off + 3] for r in slab_records])
            payload[f"ellip_{name}"] = np.array([r[off + 4] for r in slab_records])
            payload[f"d_eff_{name}"] = np.array([r[off + 5] for r in slab_records])
        for k, rec in enumerate(slab_records):
            payload[f"xy_all_{k}"] = rec[2]
            payload[f"xy_free_{k}"] = rec[8]
        np.savez(slab_path, **payload)
        print(f"stream slabs: {slab_path} ({len(slab_records)} snapshots)")
    if frame_dumps:
        np.savez(os.path.join(VIDEOS_DIR, "mf15_dumps.npz"), **frame_dumps)
        print(f"frame dumps: {os.path.join(VIDEOS_DIR, 'mf15_dumps.npz')} "
              f"({len([k for k in frame_dumps if not k.startswith(('c_', 'pose_', 'valid_'))])} snapshots)")

    # ---- pass/fail summary -------------------------------------------------------------------
    cols = ["frame", "t", "phase", "theta", "sum_c", "std_c", "ke", "nan", "n_coffee",
            "n_milk_jug", "z_surf", "n_dome", "n_over_rim", "n_outer_wall", "n_table", "n_adh",
            "first_over", "frame_ms"]
    arr = {k: np.array([r_[j] for r_ in rows], dtype=float) for j, k in enumerate(cols)}
    sum_c0 = arr["sum_c"][0]
    drift = float(np.max(np.abs(arr["sum_c"] - sum_c0)) / sum_c0) if sum_c0 > 0 else float("nan")
    t_end = arr["t"][-1]
    print("=" * 78)
    print(f"[1] nan    max = {int(arr['nan'].max())}                                        "
          f"-> {'PASS' if arr['nan'].max() == 0 else 'FAIL'}")
    p2 = bool(z_min_run >= -ps - 1e-6)
    print(f"[2] z_min  min = {z_min_run:+.4f}  (table {Z_TABLE:.2f}, limit {Z_TABLE - ps:.3f})     "
          f"-> {'PASS' if p2 else 'FAIL'}")
    p3 = bool(drift < 0.01)
    print(f"[3] sum_c drift = {drift:.3e}  (limit 1e-2, sum_c0={sum_c0:.2f})      -> {'PASS' if p3 else 'FAIL'}")
    print(f"    first_over_t = {first_over_t}  (n_over_rim >= 5 for {FIRST_OVER_STREAK} consecutive "
          f"frames, from t>={T_FIRST_OVER_MIN:g}s)")
    # where did the milk end up? (final-frame census, mutually exclusive buckets in priority
    # order: jug cavity > table > mug > rest of the domain; the landing site is the pour knob)
    mkr = np.sqrt(safe_last[n_coffee:n_fluid, 0] ** 2 + safe_last[n_coffee:n_fluid, 1] ** 2)
    mkz = safe_last[n_coffee:n_fluid, 2]
    mvalid = act_mask[n_coffee:] if act_mask is not None else all_valid[n_coffee:]
    mk_jug = (s_j > -ps) & (s_j < L_JUG + ps) & (r_j <= R_JUG + ps) & mvalid
    mk_rest = mvalid & ~mk_jug
    mk_tab = mk_rest & (mkz <= 2.5 * ps) & (mkr > R_OUT + 3.0 * ps)
    mk_mug = mk_rest & ~mk_tab & (mkr <= R_IN + ps) & (mkz >= Z_FLOOR - ps)
    cen_jug, cen_tab, cen_mug = int(mk_jug.sum()), int(mk_tab.sum()), int(mk_mug.sum())
    print(f"    milk census: in jug={cen_jug}, in mug={cen_mug}, on table={cen_tab}, "
          f"other={n_milk - cen_jug - cen_mug - cen_tab} (total {n_milk}) | "
          f"milk r p50/p90 = {np.percentile(mkr, 50):.3f}/{np.percentile(mkr, 90):.3f}, "
          f"z p50/p90 = {np.percentile(mkz, 50):.3f}/{np.percentile(mkz, 90):.3f}")
    if not args.selftest:
        # narrative criteria: the 8 s selftest window cannot reach the rim, let alone overflow
        if first_over_t is not None:
            pre = arr["t"] < first_over_t
            dur_above = float(longest_true_run(pre & (arr["z_surf"] >= Z_RIM + 0.5 * ps))) * DT
            frac, _, _ = best_window_frac(arr["t"], (arr["z_surf"] >= Z_RIM + 0.5 * ps) & pre, 2.0)
            p_dome = bool(frac >= 0.9 and dur_above >= 2.0)
            print(f"[4] dome: longest above-rim run {dur_above:.2f}s (need >=2s), "
                  f"best 2s-window fraction {frac:.2f} (need >=0.9) -> {'PASS' if p_dome else 'FAIL'}")
        else:
            print(f"[4] dome: never overflowed (n_over_rim < 5 the whole run)          -> FAIL")
        z_surf_max = float(np.nanmax(arr["z_surf"]))
        p_fill = bool(z_surf_max >= Z_RIM - ps)
        print(f"[5] z_surf_p99 max = {z_surf_max:.4f}  (rim {Z_RIM:.3f}, need >= {Z_RIM - ps:.3f})  "
              f"-> {'PASS' if p_fill else 'FAIL'}")
        if t_end >= T_TRICKLE:
            n_tab_end = arr["n_table"][-1]
            p_ov_a = bool(n_tab_end >= 3000)
            i0 = 0
            p_ov_b = False
            if n_tab_end > 0:
                i0 = int(np.nonzero(arr["n_table"] >= 0.05 * n_tab_end)[0][0])
                p_ov_b = bool(t_end - arr["t"][i0] >= 1.0)
            film_run = longest_true_run((arr["t"] >= arr["t"][i0]) & (arr["n_outer_wall"] > 0))
            p_ov = bool(p_ov_a and p_ov_b and film_run >= 60)
            print(f"[6] overflow: n_table end {int(n_tab_end)} (need >=3000), growth "
                  f"{t_end - arr['t'][i0]:.2f}s (need >=1s), outer-wall film {film_run} frames "
                  f"(need >=60) -> {'PASS' if p_ov else 'FAIL'}")
    mask_a = (arr["t"] >= 1.0) & (arr["t"] <= min(T_LIFT0 - 0.5, t_end))
    if mask_a.any():
        leak_c = float(np.max(np.abs(arr["n_coffee"][mask_a] - arr["n_coffee"][mask_a][0]))
                       / max(arr["n_coffee"][mask_a][0], 1))
        leak_m = float(np.max(np.abs(arr["n_milk_jug"][mask_a] - arr["n_milk_jug"][mask_a][0]))
                       / max(arr["n_milk_jug"][mask_a][0], 1))
        p_rest = bool(leak_c < 0.01 and leak_m < 0.01)
        print(f"[7] phase-A rest: coffee leak {leak_c:.4%}, milk leak {leak_m:.4%} (limit 1%) "
              f"-> {'PASS' if p_rest else 'FAIL'}")
    ke_med = float(np.median(arr["ke"][arr["t"] >= t_end - 2.0])) if t_end > 2.0 else float("nan")
    print(f"[8] KE last-2s median = {ke_med:.3e}  (warn limit 2.0)              "
          f"-> {'PASS' if ke_med <= 2.0 else 'WARN'}")
    ms_p95 = float(np.percentile(arr["frame_ms"], 95))
    print(f"[9] frame_ms p95 = {ms_p95:.0f}  (warn limit 1000)                    "
          f"-> {'PASS' if ms_p95 < 1000 else 'WARN'}")

    if args.selftest:
        ok_nan = bool(arr["nan"].max() == 0)
        ok_pour = bool(arr["n_milk_jug"][-1] < arr["n_milk_jug"][0])
        # a table count is splash noise unless it rivals the pour itself: the failure mode this
        # catches is the stream missing the cup (clamp/pivot wrong), not a few ejected droplets.
        # the ratio, not the count, is the stable quantity here - the ejecta count swings by
        # hundreds run-to-run (neighbor-loop atomics make the splash chaotic), observed 0-45%
        # of the pour across identical-config runs
        poured = arr["n_milk_jug"][0] - arr["n_milk_jug"][-1]
        tab_limit = max(10.0, 0.6 * poured)
        ok_table = bool(arr["n_table"].max() <= tab_limit)
        print("-" * 78)
        print(f"[S1] selftest nan = {int(arr['nan'].max())}                                  "
              f"-> {'PASS' if ok_nan else 'FAIL'}")
        print(f"[S2] selftest milk left jug: {int(arr['n_milk_jug'][0])} -> {int(arr['n_milk_jug'][-1])} "
              f"-> {'PASS' if ok_pour else 'FAIL'}")
        print(f"[S3] selftest milk on table: max {int(arr['n_table'].max())} "
              f"(limit 60% of poured {int(poured)} = {tab_limit:.0f})   "
              f"-> {'PASS' if ok_table else 'FAIL'}")
        if not (ok_nan and ok_pour and ok_table):
            sys.exit(1)

    print(f"csv: {CSV_PATH}")
    for p in mp4_paths:
        print(f"mp4: {p}")


if __name__ == "__main__":
    main()
