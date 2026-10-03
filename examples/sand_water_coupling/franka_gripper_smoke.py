"""
Phase 7-1 smoke: load a Franka Panda (gripper arm), verify its DOF layout, and IK-track the
shovel-handle poses at the key steps of the recorded best shovel demo trajectory.

Zero-physics sanity check (gravity off, collision off) before wiring the arm into the full
Phase 7 demo. The handle world poses are recomputed from the recorded blade poses
(experiments/recordings/phase5_shovel_wet/frame_*.npz) with the same blade->handle transform
as the demo, so the IK targets here are exactly the points the gripper must hit later.

Hand target frame convention (calibration under test):
  - hand local x  := parallel to the handle axis (the rod lies across the palm)
  - hand local z  := world +y (fingers point away from the sand box, toward the robot base)
  - grasp point   := hand_pos + R_hand @ (0, 0, GRASP_DIST) placed at the handle midpoint
Outputs phase7_franka_smoke_step*.png keyframes for visual inspection.
"""

import json
import math
import os

import imageio
import numpy as np

import genesis as gs

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(EXPERIMENTS_DIR, "frames")
REC_DIR = os.path.join(EXPERIMENTS_DIR, "recordings", "phase5_shovel_wet")

# --- shovel geometry (identical to examples/sand_water_coupling/shovel_wet_sand.py) ---
BLADE_HALF = (0.15, 0.12, 0.01)
HANDLE_LEN = 0.3
HANDLE_HALF_THICK = 0.015
HANDLE_ANGLE = math.radians(40.0)

DT = 1.0 / 60.0
# phase-boundary steps of the best demo (frame_i = pose before step i is integrated)
KEY_STEPS = [0, 354, 450, 555, 705, 765]

FINGER_OPEN = 0.04
FINGER_GRIP = 0.015  # 3 cm gap = handle cross-section (visual grip only, no contact physics)
GRASP_DIST = 0.09  # hand origin -> grasp point along hand local +z (to be tuned visually)

FRANKA_BASE_POS = (-0.25, 0.45, 0.0)  # behind the +y wall, reaching toward -y


def quat_to_R(q):
    # rotation matrix of a (w, x, y, z) unit quaternion
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def quat_mul(q1, q2):
    # hamilton product of (w, x, y, z) quaternions
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


def R_to_quat(R):
    # (w, x, y, z) quaternion of a rotation matrix
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def blade_to_handle(blade_pos, blade_quat):
    """World pose of the handle midpoint, same transform as the demo script."""
    handle_quat_rel = np.array(
        [math.cos((math.pi + HANDLE_ANGLE) / 2.0), 0.0, math.sin((math.pi + HANDLE_ANGLE) / 2.0), 0.0]
    )
    handle_dir_local = np.array([-math.cos(HANDLE_ANGLE), 0.0, math.sin(HANDLE_ANGLE)])
    handle_offset_local = np.array([-BLADE_HALF[0], 0.0, 0.0]) + handle_dir_local * (HANDLE_LEN / 2.0)
    handle_pos = blade_pos + quat_to_R(blade_quat) @ handle_offset_local
    handle_quat = quat_mul(blade_quat, handle_quat_rel)
    return handle_pos, handle_quat


def hand_target_from_handle(handle_pos, handle_quat):
    """IK target (pos, quat wxyz) for the franka 'hand' link.

    hand x-axis || handle axis (rod across the palm), hand z-axis = world -y (fingers point
    from the robot side across the rod toward the sand), grasp point pinned at the handle midpoint.
    """
    x_h = quat_to_R(handle_quat) @ np.array([1.0, 0.0, 0.0])
    x_h = x_h / np.linalg.norm(x_h)
    z_h = np.array([0.0, -1.0, 0.0])
    z_h = z_h - np.dot(z_h, x_h) * x_h
    z_h = z_h / np.linalg.norm(z_h)
    y_h = np.cross(z_h, x_h)
    R_hand = np.column_stack([x_h, y_h, z_h])
    hand_pos = handle_pos - R_hand @ np.array([0.0, 0.0, GRASP_DIST])
    return hand_pos, R_to_quat(R_hand)


def save_render(cam, path):
    rgb, *_ = cam.render(rgb=True)
    rgb = rgb[0] if isinstance(rgb, list) else rgb
    imageio.imwrite(path, rgb)


def main():
    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(
            gravity=(0.0, 0.0, 0.0),
            enable_collision=False,
            enable_self_collision=False,
        ),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=FRANKA_BASE_POS),
    )

    # visualization-only shovel (identical to the demo) so gripper/handle alignment can be checked
    blade_quat0 = np.array([math.cos(math.radians(40.0) / 2.0), 0.0, math.sin(math.radians(40.0) / 2.0), 0.0])
    blade = scene.add_entity(
        morph=gs.morphs.Box(pos=(-0.20, 0.0, 0.369), quat=tuple(blade_quat0), size=tuple(2.0 * np.array(BLADE_HALF))),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(color=(0.4, 0.7, 1.0)),
    )
    handle_pos0, handle_quat0 = blade_to_handle(np.array([-0.20, 0.0, 0.369]), blade_quat0)
    handle = scene.add_entity(
        morph=gs.morphs.Box(
            pos=tuple(handle_pos0),
            quat=tuple(handle_quat0),
            size=(HANDLE_LEN, 2.0 * HANDLE_HALF_THICK, 2.0 * HANDLE_HALF_THICK),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(color=(0.6, 0.4, 0.2)),
    )

    cam = scene.add_camera(
        res=(1280, 720),
        pos=(0.45, -0.75, 0.72),
        lookat=(-0.25, 0.05, 0.42),
        fov=45,
    )

    scene.build()

    print(f"n_dofs = {franka.n_dofs}", flush=True)
    for j in franka.joints:
        print(f"  joint {j.name:20s} dofs_idx_local = {j.dofs_idx_local}", flush=True)
    hand_link = franka.get_link("hand")
    fingers_dof = [franka.get_joint(n).dofs_idx_local[0] for n in ("finger_joint1", "finger_joint2")]
    print(f"fingers_dof = {fingers_dof}", flush=True)

    os.makedirs(FRAMES_DIR, exist_ok=True)
    results = {}
    for step in KEY_STEPS:
        frame = np.load(os.path.join(REC_DIR, f"frame_{step:04d}.npz"))
        blade_pos = frame["blade_pos"].astype(np.float64)
        blade_quat = frame["blade_quat"].astype(np.float64)
        blade.set_pos(blade_pos)
        blade.set_quat(blade_quat)
        handle_pos, handle_quat = blade_to_handle(blade_pos, blade_quat)
        handle.set_pos(handle_pos)
        handle.set_quat(handle_quat)

        target_pos, target_quat = hand_target_from_handle(handle_pos, handle_quat)
        q, err = franka.inverse_kinematics(
            hand_link,
            pos=target_pos,
            quat=target_quat,
            return_error=True,
            max_samples=100,
            max_solver_iters=100,
            damping=0.005,
            pos_tol=1e-4,
            rot_tol=1e-3,
        )
        q = q.detach().cpu().numpy().astype(np.float64).reshape(-1).copy()
        err = err.detach().cpu().numpy().astype(np.float64).reshape(-1)[:6]
        q[-2:] = FINGER_GRIP
        franka.set_qpos(q)
        scene.step()

        # achieved grasp point vs handle midpoint
        achieved_pos = hand_link.get_pos().cpu().numpy().reshape(-1)[:3]
        achieved_quat = hand_link.get_quat().cpu().numpy().reshape(-1)[:4]
        grasp = achieved_pos + quat_to_R(achieved_quat) @ np.array([0.0, 0.0, GRASP_DIST])
        grasp_err = float(np.linalg.norm(grasp - handle_pos))
        print(
            f"step {step:4d}: handle=({handle_pos[0]:+.3f},{handle_pos[1]:+.3f},{handle_pos[2]:+.3f}) "
            f"ik_err={np.asarray(err).reshape(-1)[:6].round(5).tolist()} grasp_err={grasp_err:.5f}",
            flush=True,
        )
        results[step] = {"ik_err": np.asarray(err).reshape(-1)[:6].tolist(), "grasp_err": grasp_err}
        save_render(cam, os.path.join(FRAMES_DIR, f"phase7_franka_smoke_step{step:04d}.png"))

    # open vs closed gripper at the final pose
    q_open = franka.get_qpos().detach().cpu().numpy().astype(np.float64).reshape(-1).copy()
    q_open[-2:] = FINGER_OPEN
    franka.set_qpos(q_open)
    scene.step()
    save_render(cam, os.path.join(FRAMES_DIR, "phase7_franka_smoke_gripper_open.png"))

    with open(os.path.join(FRAMES_DIR, "phase7_franka_smoke_ik.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("smoke done", flush=True)


if __name__ == "__main__":
    main()
