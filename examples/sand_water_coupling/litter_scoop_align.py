"""
Phase 8-bis v3: align the objaverse 'Litter scoop/貓砂鏟' (uid 411d1075691141fbb52f644602c6d5d3)
to the physical tilt-box blade frame.

Measured geometry (raw units): handle axis h = (0.644, 0.702, -0.305) [PCA of the y>10 verts],
pan floor normal n0 = (-0.236, 0.564, 0.792) [smallest PCA dir of the y<2 verts], h . n0 ~= 0
(the handle lies IN the pan plane, like a small shovel). Pan span along h: 34.78 (u in
[-29.87, 4.91]); handle verts u in [15.04, 29.03]; pan width 27.09; pan width-center is offset
~5 units from the handle axis (asymmetric model); handle cross-section 6.26.

Pipeline:
  1. uniform scale s = 0.30 / 34.78 (pan length -> physical blade depth 0.30 m; width lands
     at 0.234 m ~ the 0.24 m box width);
  2. rigid map: h -> d = (-cos40, 0, sin40), w -> (0,-1,0), n0 -> (sin40, 0, cos40)
     (right-handed);
  3. bend the pan by exactly -40 deg about the width axis through the junction pivot (handle
     in pan plane in the model -> pan flat, handle keeps the physical 40 deg);
  4. lateral S-correction: ramp the handle's width offset so both the pan center and the
     handle end up on the blade-frame y = 0 line (junction stays continuous);
  5. handle surgery: shrink the cross-section around its axis to 0.028 m (3 cm gripper gap)
     and stretch its length past the grip point (0.23 m from the blade rear);
  6. translate: pan tip -> x = +0.145, pivot -> onto the physical handle axis line,
     pan center -> y = 0, pan floor bottom -> z = +0.006.

Smoke: physical boxes semi-transparent + aligned mesh at recorded poses 0/299/555/765.
"""

import math
import os

import imageio
import numpy as np
import trimesh

import genesis as gs

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_GLB = os.path.join(HERE, "assets", "raw", "litter_scoop.glb")
OUT_GLB = os.path.join(HERE, "assets", "litter_scoop_aligned.glb")
OUT_DIR = os.path.join(HERE, "frames", "phase8b_align_smoke")
os.makedirs(OUT_DIR, exist_ok=True)

REC_DIR = os.path.join(HERE, "recordings", "phase7_gripper_shovel")
SMOKE_FRAMES = [0, 299, 555, 765]

HANDLE_ANGLE = math.radians(40.0)
BLADE_HALF = (0.15, 0.12, 0.01)
TIP_X = 0.145
Z_NUDGE = 0.006

U_JOINT = 4.91    # raw; pan rear along the handle axis
PAN_U_MIN = -29.87
PAN_LEN_TARGET = 0.30
HANDLE_THICK_TARGET = 0.028
HANDLE_LEN_TARGET = 0.36
NECK_U = 15.04    # raw; handle verts start here (neck between U_JOINT and NECK_U)


def align():
    m = trimesh.load(SRC_GLB, force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64)

    # measured frame (recomputed, not hardcoded)
    Hmask = V[:, 1] > 10.0
    Hc = V[Hmask] - V[Hmask].mean(axis=0)
    h = np.linalg.eigh(Hc.T @ Hc)[1][:, 2]
    if h[1] < 0:
        h = -h
    Pmask = V[:, 1] < 2.0
    Pc = V[Pmask] - V[Pmask].mean(axis=0)
    n0 = np.linalg.eigh(Pc.T @ Pc)[1][:, 0]
    n0 = n0 - (n0 @ h) * h
    n0 /= np.linalg.norm(n0)
    # sign of n0: the pan centroid sits on the OPENING side of the floor (walls add mass
    # there); pick n0 so that it points from the floor-side verts toward the centroid
    t_n = V[Pmask] @ n0
    floor_c = V[Pmask][t_n < np.percentile(t_n, 15)].mean(axis=0)
    if (V[Pmask].mean(axis=0) - floor_c) @ n0 < 0:
        n0 = -n0
    w = np.cross(n0, h)
    w /= np.linalg.norm(w)
    S = np.column_stack([h, w, n0])  # right-handed
    print(f"h={np.round(h,3)} w={np.round(w,3)} n0={np.round(n0,3)} det={np.linalg.det(S):.2f}")

    # 1. scale
    s = PAN_LEN_TARGET / (U_JOINT - PAN_U_MIN)
    V = V * s
    u_joint = U_JOINT * s
    print(f"scale {s:.5f}; pan width {np.ptp((V[Pmask] @ w)):.3f} m")

    # 2. rigid map [h, w, n0] -> [d, (0,-1,0), (sin40,0,cos40)]
    c, si = math.cos(HANDLE_ANGLE), math.sin(HANDLE_ANGLE)
    T = np.column_stack([np.array([-c, 0.0, si]), np.array([0.0, -1.0, 0.0]), np.array([si, 0.0, c])])
    R = T @ S.T
    V = V @ R.T

    # junction pivot on the handle axis (handle centroid line)
    CH = V[V[:, 1] > 10.0 * s * 0 + 0]  # placeholder; recompute mask in new coords below
    # NOTE: masks must be recomputed in blade-frame coords via u = V @ d
    d = np.array([-c, 0.0, si])
    u = V @ d
    shaft0 = u > NECK_U * s
    CH = V[shaft0].mean(axis=0)
    pivot = CH + d * (u_joint - CH @ d)

    # 3. bend the pan flat: measured floor normal after the rigid map -> rotate to +z.
    #    n0 from the pan PCA is contaminated by the walls (~9 deg residual), so iterate:
    #    each round re-measures the floor normal from the lowest pan verts and corrects.
    pan = u < u_joint
    for _ in range(3):
        n_b = R @ n0 if _ == 0 else None
        if _ > 0:
            floor_mask = V[pan][:, 2] < np.percentile(V[pan][:, 2], 30)
            F = V[pan][floor_mask]
            Fc = F - F.mean(axis=0)
            n_b = np.linalg.eigh(Fc.T @ Fc)[1][:, 0]
            if n_b[2] < 0:
                n_b = -n_b
        theta = -math.atan2(n_b[0], n_b[2])
        print(f"pan floor normal {np.round(n_b, 3)}, bend {math.degrees(theta):.2f} deg")
        Rb = trimesh.transformations.rotation_matrix(theta, [0.0, 1.0, 0.0], point=pivot)
        V[pan] = trimesh.transform_points(V[pan], Rb)
        if abs(theta) < math.radians(0.5):
            break

    # 4. lateral S-correction: pan center -> y=0, handle end -> y=0, smooth ramp on the neck
    u = V @ d
    pan = u < u_joint
    pan_cy = (V[pan][:, 1].min() + V[pan][:, 1].max()) / 2.0
    V[:, 1] -= pan_cy                       # pan centered on y=0
    shaft = u > u_joint
    handle_cy = V[u > NECK_U * s][:, 1].mean()
    ramp = np.clip((u[shaft] - u_joint) / max(NECK_U * s - u_joint, 1e-6), 0.0, 1.0)
    V[shaft, 1] -= handle_cy * ramp          # handle bends smoothly onto y=0
    print(f"pan center y was {pan_cy:.3f}; handle end y offset {handle_cy:.3f} corrected")

    # 5. handle surgery: shrink cross-section around its own axis, stretch length
    u = V @ d
    shaft = u > u_joint
    far = u > NECK_U * s
    Cf = V[far].mean(axis=0)
    Cf -= d * (Cf @ d)                       # perpendicular center of the far handle
    Hu = u[shaft] - u_joint
    Hp = V[shaft] - np.outer(u[shaft], d)    # offset from the pivot-line through origin
    Hp_c = Hp - (Cf - d * 0)                 # center on the far-handle axis
    thick = np.ptp(Hp_c, axis=0).max()
    k_thick = HANDLE_THICK_TARGET / thick
    k_len = HANDLE_LEN_TARGET / Hu.max()
    print(f"handle: thick {thick:.3f} -> x{k_thick:.2f}; len {Hu.max():.3f} -> x{k_len:.2f}")
    # handle axis forced onto the pivot line: drop the constant offset, scale the rest
    V[shaft] = np.outer(u_joint + Hu * k_len, d) + Hp_c * k_thick

    # 6. translate: tip -> +x, pivot -> physical handle axis, floor -> z=+Z_NUDGE
    pan = (V @ d) < u_joint * 0.999  # pan verts (they moved; recompute loosely by x side)
    pan = V[:, 0] > -0.30  # fallback: everything right of the far handle end is pan... refine below
    # pan verts = those NOT stretched past the junction
    pan = (V @ d) < u_joint
    tip = V[pan][np.argmax(V[pan][:, 0])]
    n_perp = np.array([si, 0.0, c])
    tr = np.zeros(3)
    tr[0] = TIP_X - tip[0]
    # pivot (currently at blade-frame pos of the junction) -> onto the line through (-0.15,0,0) along d
    tr[2] = (-0.15 * si - (pivot[0] + tr[0]) * si - pivot[2] * c) / c
    V = V + tr
    floor_z = V[pan][:, 2].min()
    V = V + np.array([0.0, 0.0, -floor_z + Z_NUDGE])
    m.vertices = V

    lo, hi = m.bounds
    print(f"aligned bounds: {np.round(lo, 3).tolist()} .. {np.round(hi, 3).tolist()}")
    print(f"pan tip x={tip[0] + tr[0]:.3f} (target {TIP_X}); handle end at {np.round(V[np.argmax(V @ d)], 3).tolist()}")

    # export as glTF Y-up (Genesis assumes Y-up for glb)
    m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(-90.0), [1.0, 0.0, 0.0]))
    m.export(OUT_GLB)
    print(f"saved {OUT_GLB}")


def blade_to_handle(blade_pos, blade_quat):
    def quat_to_R(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    handle_quat_rel = np.array([
        math.cos((math.pi + HANDLE_ANGLE) / 2.0), 0.0, math.sin((math.pi + HANDLE_ANGLE) / 2.0), 0.0
    ])
    handle_dir_local = np.array([-math.cos(HANDLE_ANGLE), 0.0, math.sin(HANDLE_ANGLE)])
    handle_offset_local = np.array([-BLADE_HALF[0], 0.0, 0.0]) + handle_dir_local * (0.3 / 2.0)
    handle_pos = blade_pos + quat_to_R(blade_quat) @ handle_offset_local
    handle_quat = quat_mul(blade_quat, handle_quat_rel)
    return handle_pos, handle_quat


def smoke():
    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    blade = scene.add_entity(
        morph=gs.morphs.Box(size=tuple(2.0 * np.array(BLADE_HALF))),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(color=(0.4, 0.7, 1.0), opacity=0.35),
    )
    handle = scene.add_entity(
        morph=gs.morphs.Box(size=(0.3, 0.03, 0.03)),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(color=(0.6, 0.4, 0.2), opacity=0.35),
    )
    shovel = scene.add_entity(
        morph=gs.morphs.Mesh(file=OUT_GLB),
        material=gs.materials.Kinematic(),
    )
    cam = scene.add_camera(res=(1280, 720), pos=(1.05, -1.05, 0.75), lookat=(0.0, 0.0, 0.18), fov=45)
    scene.build()

    for fi in SMOKE_FRAMES:
        frame = np.load(os.path.join(REC_DIR, f"frame_{fi:04d}.npz"))
        bp = frame["blade_pos"].astype(np.float64)
        bq = frame["blade_quat"].astype(np.float64)
        blade.set_pos(bp)
        blade.set_quat(bq)
        shovel.set_pos(bp)
        shovel.set_quat(bq)
        hp, hq = blade_to_handle(bp, bq)
        handle.set_pos(hp)
        handle.set_quat(hq)
        for tag, eye, lk in [
            ("main", (1.05, -1.05, 0.75), (0.0, 0.0, 0.18)),
            ("close", (bp[0] + 0.55, bp[1] - 0.55, bp[2] + 0.35), tuple(bp)),
            ("top", (bp[0] + 0.05, bp[1] - 0.15, bp[2] + 0.8), tuple(bp)),
        ]:
            cam.set_pose(pos=eye, lookat=lk)
            rgb, *_ = cam.render(rgb=True, force_render=True)
            imageio.imwrite(
                os.path.join(OUT_DIR, f"frame_{fi:04d}_{tag}.png"),
                rgb[0] if isinstance(rgb, list) else rgb,
            )
        print(f"frame {fi} rendered", flush=True)


if __name__ == "__main__":
    align()
    smoke()
