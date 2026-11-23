"""
Phase 8-1: align the objaverse 'Shovel vintage' asset (uid b4c2a8dc099a4e02ac40cff7f6490c27)
to the physical tilt-box blade frame, then smoke-render it against the REAL physical boxes
(semi-transparent) at several recorded blade poses for verification.

Alignment pipeline (all in trimesh, then exported as glTF Y-up so Genesis loads it 1:1):
  1. uniform scale: blade width 0.328 m -> 0.24 m (matches the physical box width);
  2. rigid map: model +y (blade->grip) -> shaft dir d = (-cos40, 0, sin40) in blade frame,
     model +x (width) -> +y, model +z -> e_x' x e_y' (right-handed);
  3. bend: the real spade blade is nearly collinear with the shaft, but the physical
     shovel has the shaft 40 deg above the blade plane -- rotate the blade vertices
     (u < u_joint along the shaft axis) about the width axis through the socket pivot so
     the blade plane normal becomes +z (blade flat, shaft at 40 deg);
  4. translate: blade tip to x = +0.145 (physical leading edge 0.15), pivot onto the
     physical handle axis line (through (-0.15, 0, 0) along d), width centered at y = 0,
     small +z nudge so the blade top face nearly touches the physical box top (z = 0.01).

Smoke: Genesis scene with the physical blade/handle boxes (opacity 0.35) + aligned mesh,
posed at recorded frames (0 / 299 / 555 / 765), two camera views each.
"""

import json
import math
import os

import imageio
import numpy as np
import trimesh

import genesis as gs

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_GLB = os.path.join(HERE, "assets", "baked", "shovel_vintage.glb")
OUT_GLB = os.path.join(HERE, "assets", "shovel_vintage_aligned.glb")
OUT_DIR = os.path.join(HERE, "frames", "phase8_align_smoke")
os.makedirs(OUT_DIR, exist_ok=True)

REC_DIR = os.path.join(HERE, "recordings", "phase7_gripper_shovel")
SMOKE_FRAMES = [0, 299, 555, 765]

HANDLE_ANGLE = math.radians(40.0)
BLADE_HALF = (0.15, 0.12, 0.01)

Y_JOINT = -0.15   # model-space shaft coordinate of the blade/socket junction
TIP_X = 0.145     # blade-frame x of the visual blade tip (physical leading edge = 0.15)
Z_NUDGE = 0.006   # raise the visual blade toward the physical box top face (z = 0.01)


def align():
    m = trimesh.load(SRC_GLB, force="mesh")
    lo, hi = m.bounds
    print(f"src bounds: {np.round(lo, 3).tolist()} .. {np.round(hi, 3).tolist()}")

    # 1. uniform scale: blade width (model x span of the blade part) -> 0.24 m
    blade_mask0 = m.vertices[:, 1] < Y_JOINT
    blade_w = np.ptp(m.vertices[blade_mask0][:, 0])
    s = 0.24 / blade_w
    m.apply_scale(s)
    print(f"blade width {blade_w:.3f} -> scale {s:.4f}; total len {(hi[1]-lo[1])*s:.3f} m")

    # 2. rigid map into the blade frame (pre-bend: shaft along d, blade in the shaft plane)
    c, si = math.cos(HANDLE_ANGLE), math.sin(HANDLE_ANGLE)
    e_x = np.array([0.0, 1.0, 0.0])        # model x (width)  -> blade-frame +y
    e_y = np.array([-c, 0.0, si])          # model y (shaft)  -> shaft dir d
    e_z = np.cross(e_x, e_y)               # model z (normal) -> right-handed completion
    R = np.eye(4)
    R[:3, :3] = np.column_stack([e_x, e_y, e_z])
    m.apply_transform(R)

    # 3. bend the blade flat: rotate blade verts about the width axis through the pivot
    V = m.vertices
    u = V @ e_y                            # shaft-axis coordinate
    pivot = e_y * (Y_JOINT * s)            # pivot point on the shaft axis
    blade = u < (Y_JOINT * s)
    # current blade plane normal via PCA of blade verts
    B = V[blade] - V[blade].mean(axis=0)
    n_b = np.linalg.svd(B.T @ B)[2][:, -1]
    n_b = n_b / np.linalg.norm(n_b)
    if n_b @ e_z < 0:
        n_b = -n_b
    # rotate about +y (blade-frame width axis) so n_b -> +z
    phi = math.atan2(n_b[0], n_b[2])       # current tilt of the blade normal toward +x
    theta = -phi                           # R_y(theta) @ n_b = +z
    print(f"blade normal pre-bend {np.round(n_b, 3).tolist()}, bend angle {math.degrees(theta):.2f} deg")
    Rb = trimesh.transformations.rotation_matrix(theta, [0.0, 1.0, 0.0], point=pivot)
    Vb = trimesh.transform_points(V[blade], Rb)
    V[blade] = Vb
    m.vertices = V

    # 4. translate: tip to +x, pivot onto the physical handle axis, width centered
    T = V[blade][np.argmax(V[blade][:, 0])]          # blade tip (max x)
    n_perp = np.array([si, 0.0, c])                  # in-plane perpendicular of d
    t = np.zeros(3)
    t[0] = TIP_X - T[0]
    t[1] = -pivot[1]
    # (pivot + t) . n_perp = handle axis point (-0.15, 0, 0) . n_perp
    t[2] = (-0.15 * si - (pivot[0] + t[0]) * si - pivot[2] * c) / c
    m.apply_translation(t)
    # blade plane should sit at z ~ 0 (box mid); nudge toward the box top face
    blade_z = m.vertices[blade][:, 2].mean()
    m.apply_translation([0.0, 0.0, -blade_z + Z_NUDGE])

    lo, hi = m.bounds
    print(f"aligned bounds: {np.round(lo, 3).tolist()} .. {np.round(hi, 3).tolist()}")
    tip = m.vertices[blade][np.argmax(m.vertices[blade][:, 0])]
    print(f"blade tip at {np.round(tip, 3).tolist()} (target x={TIP_X})")

    # export as glTF Y-up (Genesis assumes Y-up for glb): rotate -90 deg about x
    m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(-90.0), [1.0, 0.0, 0.0]))
    m.export(OUT_GLB)
    print(f"saved {OUT_GLB}")
    return s


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
            ("side", (bp[0] + 0.1, bp[1] - 0.8, bp[2] + 0.15), tuple(bp)),
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
