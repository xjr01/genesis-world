"""camcheck armframe: pick the pulled-back deepbed camera that frames the WHOLE Franka arm.

Two arm envelopes are projected, besides the bed/shovel/sand/water sets:
  - arm_kin : kinematic envelope = convex hull of the base column, the shoulder point, the
    swept elbow locus (circle where sphere(shoulder, upper-arm 0.316) meets sphere(TCP, 0.55)
    for every recorded hand-target), and 0.30-padded hand-target boxes (wrist + fingers +
    IK lag). While the gripper holds the handle, every arm configuration consistent with the
    recorded TCP path lies inside this set. HARD GATE: >= 5% frame margin on all edges.
  - arm_task: the conservative task envelope (cylinder r=0.6 about the base, z<=0.85,
    y<=0.60). REPORTED for reference -- parts of it (e.g. z=0.85) cannot be occupied by any
    configuration of this run and it does not fit any sanctioned candidate; its margins are
    printed so the trade-off is on record.
Scoop pixels use a blade-local box whose half-extents are calibrated so its projection
covers the blue-mask bbox observed in the wide20 raw frames (steps 1275-1695, water gone).
Occlusion gate: camera->arm_kin sightlines vs the four render walls; the sand heap is
checked against the bed far rim (strip y>0.10) and reported.

Usage: python camcheck_armframe.py                 # candidate sweep table
       python camcheck_armframe.py report B L F    # key-point dump for back=B lookatz=L fov=F
Output: redirect to frames/camcheck_armframe.log
"""

import math
import os
import sys

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

EXPT = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(EXPT, "recordings", "phase9_deepbed_full")
WIDE20_RAW = os.path.join(EXPT, "frames", "phase9_deepbed_full_raw")
W, H = 1280, 720
MARGIN_X, MARGIN_Y = 0.05 * W, 0.05 * H  # 64 / 36 px

POS_WIDE20 = np.array([0.516, -0.576, 0.504])
LOOKAT_BASE = np.array([-0.06, 0.0, 0.12])
FOV_WIDE20 = 40.0

FRANKA_BASE = np.array([-0.25, 0.45, 0.0])
SHOULDER = FRANKA_BASE + np.array([0.0, 0.0, 0.333])
UPPER_ARM, ELBOW_TO_TCP = 0.316, 0.60  # panda link lengths (elbow-to-TCP padded for wrist+hand)
HAND_PAD = 0.15  # wrist/finger ball around the driven TCP target
TASK_R, TASK_ZMAX, TASK_YMAX = 0.6, 0.85, 0.60

HANDLE_OFFSET_LOCAL = np.array([-0.17407, 0.0, 0.09749])
HANDLE_ANGLE = math.radians(40.39)
GRASP_DIST = 0.09

# render walls (WALL_THICK=0.005 centered on the bed faces, WALL_HEIGHT=0.12)
WALLS = [
    (np.array([0.175, -0.15, 0.0]), np.array([0.180, 0.15, 0.12])),
    (np.array([-0.180, -0.15, 0.0]), np.array([-0.175, 0.15, 0.12])),
    (np.array([-0.18, -0.155, 0.0]), np.array([0.18, -0.150, 0.12])),
    (np.array([-0.18, 0.150, 0.0]), np.array([0.18, 0.155, 0.12])),
]
BED_BOX = (np.array([-0.175, -0.15, 0.0]), np.array([0.175, 0.15, 0.14]))


def quat_to_R(q):
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def quat_mul(q1, q2):
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


def hand_target(blade_pos, blade_quat):
    """Driven hand-target pose (identical chain to franka_shovel_deepbed_render.py)."""
    R = quat_to_R(blade_quat)
    handle_quat_rel = np.array(
        [math.cos((math.pi + HANDLE_ANGLE) / 2.0), 0.0, math.sin((math.pi + HANDLE_ANGLE) / 2.0), 0.0]
    )
    handle_pos = blade_pos + R @ HANDLE_OFFSET_LOCAL
    handle_quat = quat_mul(blade_quat, handle_quat_rel)
    x_h = quat_to_R(handle_quat) @ np.array([1.0, 0.0, 0.0])
    x_h = x_h / np.linalg.norm(x_h)
    z_h = np.array([0.0, -1.0, 0.0])
    z_h = z_h - np.dot(z_h, x_h) * x_h
    z_h = z_h / np.linalg.norm(z_h)
    y_h = np.cross(z_h, x_h)
    R_hand = np.column_stack([x_h, y_h, z_h])
    return handle_pos - R_hand @ np.array([0.0, 0.0, GRASP_DIST])


def cam_basis(pos, lookat, fov):
    fwd = lookat - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    cup = np.cross(right, fwd)
    f = (H / 2.0) / np.tan(np.radians(fov) / 2.0)
    return right, cup, fwd, f


def project(P, pos, basis):
    right, cup, fwd, f = basis
    rel = np.atleast_2d(P) - pos
    d = rel @ fwd
    u = rel @ right
    v = rel @ cup
    x = W / 2.0 + f * u / d
    y = H / 2.0 - f * v / d
    return np.column_stack([x, y]), d


def seg_hits_box(p0, p1, lo, hi):
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if p0[i] < lo[i] or p0[i] > hi[i]:
                return False
        else:
            a = (lo[i] - p0[i]) / d[i]
            b = (hi[i] - p0[i]) / d[i]
            if a > b:
                a, b = b, a
            t0 = max(t0, a)
            t1 = min(t1, b)
            if t0 > t1:
                return False
    return True


def load_trajectory():
    files = sorted(f for f in os.listdir(REC) if f.startswith("frame_"))
    blade, quats, tcp = [], [], []
    sand_pts, water_pts = [], []
    sand_rim_z = 0.0
    for i, fname in enumerate(files):
        fr = np.load(os.path.join(REC, fname))
        b = fr["blade_pos"].astype(np.float64)
        q = fr["blade_quat"].astype(np.float64)
        blade.append(b)
        quats.append(q)
        tcp.append(hand_target(b, q))
        if i % 10 == 0:
            sp = fr["sand_pos"].astype(np.float64)
            sand_pts.append(sp)
            rim = sp[sp[:, 1] > 0.10, 2]
            if len(rim):
                sand_rim_z = max(sand_rim_z, rim.max())
        wp = fr["water_pos"].astype(np.float64)
        if len(wp):
            water_pts.append(wp)
    return (
        np.array(blade), quats, np.array(tcp),
        np.vstack(sand_pts), np.vstack(water_pts), sand_rim_z,
    )


def arm_envelope_kinematic(tcp):
    """Convex hull of base column + shoulder + swept elbow circles + padded TCP boxes."""
    pts = [SHOULDER]
    bx, by = FRANKA_BASE[:2]
    for z in np.linspace(0.0, 0.40, 5):
        for a in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
            pts.append(np.array([bx + 0.15 * np.cos(a), by + 0.15 * np.sin(a), z]))
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * HAND_PAD
    for t in tcp:
        pts.extend(t + corners)
        c = t - SHOULDER
        dc = np.linalg.norm(c)
        if abs(dc - ELBOW_TO_TCP) > UPPER_ARM or dc < 1e-9:
            continue  # elbow circle empty: TCP out of arm reach (padded), skip
        a = (dc * dc + UPPER_ARM * UPPER_ARM - ELBOW_TO_TCP * ELBOW_TO_TCP) / (2.0 * dc)
        h = math.sqrt(max(UPPER_ARM * UPPER_ARM - a * a, 0.0))
        ce = SHOULDER + a * c / dc
        # orthonormal basis perpendicular to c
        u = np.array([0.0, 0.0, 1.0])
        if abs(u @ c / dc) > 0.9:
            u = np.array([1.0, 0.0, 0.0])
        u = u - (u @ c / dc) * c / dc
        u /= np.linalg.norm(u)
        v = np.cross(c / dc, u)
        for ang in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
            pts.append(ce + h * (math.cos(ang) * u + math.sin(ang) * v))
    pts = np.array(pts)
    return pts[ConvexHull(pts).vertices]


def arm_envelope_task():
    """Reference cylinder r=TASK_R about the base, z<=TASK_ZMAX, y<=TASK_YMAX."""
    bx, by = FRANKA_BASE[:2]
    pts = []
    for z in np.linspace(0.0, TASK_ZMAX, 6):
        for a in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False):
            p = np.array([bx + TASK_R * np.cos(a), by + TASK_R * np.sin(a), z])
            if p[1] <= TASK_YMAX:
                pts.append(p)
    for dx in (-TASK_R, TASK_R):
        for z in np.linspace(0.0, TASK_ZMAX, 6):
            pts.append(np.array([bx + dx, TASK_YMAX, z]))
    return np.array(pts)


def blue_bbox(png_path):
    img = np.asarray(Image.open(png_path).convert("RGB")).astype(np.int32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mask = (b > r + 30) & (b > g + 30) & (b > 60)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()


def scoop_corners(ext):
    xlo, xhi, hy, hz = ext
    return np.array(
        [[-xlo if sx else xhi, sy * hy, sz * hz] for sx in (1, 0) for sy in (-1, 1) for sz in (-1, 1)]
    )


def calibrate_scoop(blade, quats):
    """Fit blade-local box half-extents so the projected box covers the wide20 blue bbox."""
    basis = cam_basis(POS_WIDE20, LOOKAT_BASE, FOV_WIDE20)
    obs = []
    for s in (1275, 1415, 1550, 1695):  # water fully absorbed from step 1275 on
        png = os.path.join(WIDE20_RAW, f"frame_{s:04d}.png")
        if not os.path.exists(png):
            continue
        fname = f"frame_{s:04d}.npz"
        fr = np.load(os.path.join(REC, fname))
        obs.append((fr["blade_pos"].astype(np.float64), quat_to_R(fr["blade_quat"].astype(np.float64)), blue_bbox(png)))
    best = None
    for xlo in np.arange(0.16, 0.32, 0.02):
        for xhi in np.arange(0.06, 0.18, 0.02):
            for hy in np.arange(0.05, 0.16, 0.01):
                for hz in np.arange(0.05, 0.14, 0.01):
                    corners = scoop_corners((xlo, xhi, hy, hz))
                    area, ok = 0.0, True
                    for b, R, bb in obs:
                        px, _ = project(b + corners @ R.T, POS_WIDE20, basis)
                        x0, y0, x1, y1 = px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()
                        if x0 > bb[0] - 1 or y0 > bb[1] - 1 or x1 < bb[2] + 1 or y1 < bb[3] + 1:
                            ok = False
                            break
                        area += (x1 - x0) * (y1 - y0)
                    if ok and (best is None or area < best[0]):
                        best = (area, xlo, xhi, hy, hz)
    return best


def scoop_points(blade, quats, ext):
    corners = scoop_corners(ext)
    return np.vstack([b + corners @ quat_to_R(q).T for b, q in zip(blade, quats)])


def margin_stats(px):
    x0, y0 = px[:, 0].min(), px[:, 1].min()
    x1, y1 = px[:, 0].max(), px[:, 1].max()
    return {"l": x0, "r": W - 1 - x1, "t": y0, "b": H - 1 - y1}


def occluded_count(pos, pts, boxes):
    return sum(any(seg_hits_box(pos, p, lo, hi) for lo, hi in boxes) for p in pts)


def in_box(p, box):
    lo, hi = box
    return bool(np.all(p >= lo - 1e-9) and np.all(p <= hi + 1e-9))


def evaluate(pos, lookat, fov, sets, sand_rim_z):
    """Margins for every gated set + occlusion counts. Returns (rows, pass, worst_ratio)."""
    basis = cam_basis(pos, lookat, fov)
    rows, ok, worst = {}, True, 1e9
    for name, pts in sets.items():
        px, _ = project(pts, pos, basis)
        m = margin_stats(px)
        rows[name] = (m, px)
        if m["l"] < MARGIN_X or m["r"] < MARGIN_X or m["t"] < MARGIN_Y or m["b"] < MARGIN_Y:
            ok = False
        worst = min(worst, m["l"] / MARGIN_X, m["r"] / MARGIN_X, m["t"] / MARGIN_Y, m["b"] / MARGIN_Y)
    arm = sets["arm_kin"]
    # the 0.12 m render walls always shadow the ground-contact ring of the pedestal behind the
    # bed (z < 0.05, occluded in every accepted render so far); gate on the arm body above it
    arm_body = np.array([p for p in arm if p[2] >= 0.05 and not in_box(p, BED_BOX)])
    rows["occl_walls"] = occluded_count(pos, arm_body, WALLS)
    arm_ground = np.array([p for p in arm if p[2] < 0.05])
    rows["occl_ground"] = occluded_count(pos, arm_ground, WALLS)
    rows["occl_rim"] = occluded_count(
        pos, arm_body, [(np.array([-0.175, 0.10, 0.0]), np.array([0.175, 0.155, sand_rim_z]))]
    )
    return rows, ok and rows["occl_walls"] == 0, worst


def main():
    print("loading recordings ...", flush=True)
    blade, quats, tcp, sand, water, sand_rim_z = load_trajectory()
    arm_kin = arm_envelope_kinematic(tcp)
    arm_task = arm_envelope_task()
    print(
        f"frames={len(blade)} tcp z [{tcp[:,2].min():.3f},{tcp[:,2].max():.3f}] "
        f"tcp y [{tcp[:,1].min():.3f},{tcp[:,1].max():.3f}] sand rim zmax(y>0.10)={sand_rim_z:.3f} "
        f"water zmax={water[:,2].max():.3f}"
    )
    print(
        f"arm_kin hull: {len(arm_kin)} verts x [{arm_kin[:,0].min():.3f},{arm_kin[:,0].max():.3f}] "
        f"y [{arm_kin[:,1].min():.3f},{arm_kin[:,1].max():.3f}] z [{arm_kin[:,2].min():.3f},{arm_kin[:,2].max():.3f}]"
    )

    fit = calibrate_scoop(blade, quats)
    print(f"scoop envelope fit on wide20 blue bbox (steps 1275/1415/1550/1695): "
          f"xlo={fit[1]:.2f} xhi={fit[2]:.2f} hy={fit[3]:.2f} hz={fit[4]:.2f} (m, blade-local)")
    scoop = scoop_points(blade, quats, fit[1:])
    bed = np.array([[sx * 0.18, sy * 0.155, sz * 0.12] for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)])
    sets = {
        "arm_kin": arm_kin, "scoop": scoop, "bed": bed, "sand": sand, "water": water,
    }

    if len(sys.argv) > 1 and sys.argv[1] == "report":
        hback, dz, lz, fov = (float(a) for a in sys.argv[2:6])
        lookat = np.array([LOOKAT_BASE[0], LOOKAT_BASE[1], lz])
        view_h = np.array([POS_WIDE20[0] - LOOKAT_BASE[0], POS_WIDE20[1] - LOOKAT_BASE[1], 0.0])
        pos = lookat + hback * view_h
        pos = np.array([pos[0], pos[1], POS_WIDE20[2] + dz])
        rows, ok, worst = evaluate(pos, lookat, fov, sets, sand_rim_z)
        print(f"\nREPORT camera pos={pos} lookat={lookat} fov={fov} pass={ok} worst_ratio={worst:.2f}")
        for name in ("arm_kin", "scoop", "bed", "sand", "water"):
            m, px = rows[name]
            print(f"  {name:<8} l={m['l']:7.1f} r={m['r']:7.1f} t={m['t']:7.1f} b={m['b']:7.1f}")
        print(f"  occl_walls={rows['occl_walls']} occl_ground(accepted)={rows['occl_ground']} "
              f"occl_sandrim={rows['occl_rim']}")
        task_px, _ = project(arm_task, pos, cam_basis(pos, lookat, fov))
        tm = margin_stats(task_px)
        print(f"  arm_task(ref, not gated) l={tm['l']:.1f} r={tm['r']:.1f} t={tm['t']:.1f} b={tm['b']:.1f}")
        basis = cam_basis(pos, lookat, fov)
        key_pts = {
            "base_center_bottom": FRANKA_BASE,
            "base_column_top": np.array([FRANKA_BASE[0], FRANKA_BASE[1], 0.40]),
            "shoulder": SHOULDER,
            "arm_kin_top": arm_kin[np.argmax(arm_kin[:, 2])],
            "arm_kin_left": arm_kin[np.argmin(arm_kin[:, 0])],
            "arm_kin_back": arm_kin[np.argmax(arm_kin[:, 1])],
        }
        steps = [int(f[6:-4]) for f in sorted(os.listdir(REC)) if f.startswith("frame_")]
        for s in (0, 939, 1275, 1415, 1615, 1695):
            if s in steps:
                i = steps.index(s)
                key_pts[f"tcp_step{s}"] = tcp[i]
                key_pts[f"blade_step{s}"] = blade[i]
        key_pts["water_top"] = water[np.argmax(water[:, 2])]
        key_pts["sand_rim_top"] = np.array([0.0, 0.12, sand_rim_z])
        print("  key points (world -> pixel):")
        for name, p in key_pts.items():
            px, d = project(np.atleast_2d(p), pos, basis)
            x, y = px[0, 0], px[0, 1]
            ml = min(x, W - 1 - x) / MARGIN_X
            mt = min(y, H - 1 - y) / MARGIN_Y
            print(f"    {name:<20} ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) -> "
                  f"({x:7.1f},{y:7.1f}) d={d[0]:.2f} margin_x={ml:5.2f} margin_y={mt:5.2f}")
        return 0

    view = POS_WIDE20 - LOOKAT_BASE
    view_h = np.array([view[0], view[1], 0.0])  # horizontal pull-back keeps the same azimuth
    print(
        f"\nsweep: pos_xy = lookat_xy + hback * view_xy, pos_z = 0.504 + dz (pure pull-back along"
        f" the full view axis cannot fit the arm top: see camcheck_armframe_pure.log)\n"
        f"{'candidate':<34}{'arm_kin l/r/t/b (px)':<26}{'scoop l/r/t/b':<24}{'bed l/r/t/b':<24}"
        f"{'sand':>5}{'water':>7}{'occl':>10}  verdict"
    )
    for hback in (1.35, 1.40, 1.45):
        for dz in (0.0, 0.08, 0.16, 0.24, 0.32):
            for lz in (0.18, 0.22, 0.26, 0.30):
                for fov in (42, 45):
                    lookat = np.array([LOOKAT_BASE[0], LOOKAT_BASE[1], lz])
                    pos = lookat + hback * view_h
                    pos = np.array([pos[0], pos[1], POS_WIDE20[2] + dz])
                    rows, ok, worst = evaluate(pos, lookat, fov, sets, sand_rim_z)
                    if not ok and worst < 0.55:
                        continue  # far-from-passing rows would only bury the winners
                    a, sc, bd = rows["arm_kin"][0], rows["scoop"][0], rows["bed"][0]
                    task_m = margin_stats(project(arm_task, pos, cam_basis(pos, lookat, fov))[0])
                    print(
                        f"h{hback:.2f} dz{dz:.2f} lz{lz:.2f} f{fov:.0f}   "
                        f"{a['l']:5.0f}/{a['r']:5.0f}/{a['t']:5.0f}/{a['b']:5.0f}{'':<4}"
                        f"{sc['l']:5.0f}/{sc['r']:5.0f}/{sc['t']:5.0f}/{sc['b']:5.0f}{'':<2}"
                        f"{bd['l']:5.0f}/{bd['r']:5.0f}/{bd['t']:5.0f}/{bd['b']:5.0f}{'':<2}"
                        f"{rows['sand'][0]['b']:5.0f}{rows['water'][0]['b']:7.0f}"
                        f"w{rows['occl_walls']:3d}/r{rows['occl_rim']:<3d}  "
                        f"{'PASS' if ok else 'fail'} wrst={worst:.2f}"
                        f" task_t={task_m['t']:.0f},task_r={task_m['r']:.0f}"
                    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
