"""Squeeze a free elastic box between two slowly approaching rigid links."""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

# Select this checkout when the editable installation points elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

import genesis as gs
from genesis.utils.element import create_tetrahedral_grid
from genesis.utils.misc import tensor_to_array


class SqueezeSample(NamedTuple):
    time_s: float
    gap_m: float
    compression: float
    penetration_m: float
    speed_rms_m_s: float
    speed_max_m_s: float
    jacobian_min: float


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation step duration in seconds")
    parser.add_argument(
        "--substeps", type=int, default=10, help="More substeps resolve contact at increased runtime cost"
    )
    parser.add_argument(
        "-t", "--seconds", type=float, default=4.0, help="Simulation duration; a full squeeze takes 4 seconds"
    )
    args = parser.parse_args()
    if "PYTEST_VERSION" in os.environ:
        args.seconds = min(args.seconds, 0.02)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, seed=0)

    camera_pos = (0.12, -0.30, 0.67)
    camera_lookat = (0.0, 0.0, 0.5)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
            gravity=(0.0, 0.0, 0.0),
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        vis_options=gs.options.VisOptions(
            ambient_light=(0.5, 0.5, 0.5),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=30,
        ),
        show_viewer=args.vis,
    )

    vertices, elements = create_tetrahedral_grid(
        lower=(-0.05, -0.05, -0.05), upper=(0.05, 0.05, 0.05), resolution=(8, 8, 8)
    )
    soft_entity = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=camera_lookat,
            vertices=vertices,
            elements=elements,
        ),
        material=gs.materials.FEM.Elastic(
            E=1e4,
            nu=0.3,
            rho=1000.0,
            model="linear_corotated",
        ),
        surface=gs.surfaces.Default(
            color=(0.95, 0.65, 0.15),
            smooth=False,
        ),
        name="soft_box",
    )

    rigid_entities = []
    for direction, name, color in ((1, "left", (0.2, 0.45, 0.8)), (-1, "right", (0.25, 0.7, 0.65))):
        mjcf = ET.Element("mujoco", model=f"{name}_slider")
        world = ET.SubElement(mjcf, "worldbody")
        link = ET.SubElement(world, "body", name="jaw", pos=f"{-direction * 0.06} 0 0.5")
        ET.SubElement(link, "joint", name="slide", type="slide", axis="1 0 0", armature="0")
        ET.SubElement(link, "geom", name="box", type="box", size="0.01 0.02 0.02", mass="0.1")
        rigid_entities.append(
            scene.add_entity(
                morph=gs.morphs.MJCF(
                    file=ET.tostring(mjcf, encoding="unicode"),
                ),
                material=gs.materials.Rigid(
                    is_coup_reaction_enabled=False,
                ),
                surface=gs.surfaces.Default(
                    color=color,
                ),
                vis_mode="collision",
                name=name,
            )
        )

    scene.build()
    rigid_links = tuple(entity.get_link(name="jaw") for entity in rigid_entities)
    position_gain = 1e5
    # The position term is explicit; damping must also stabilize its discrete update at the chosen step size
    velocity_gain = 1000.0
    for entity in rigid_entities:
        entity.set_dofs_kp(kp=position_gain)
        entity.set_dofs_kv(kv=velocity_gain)

    el2v = tensor_to_array(soft_entity.get_el2v())
    rest_pos = tensor_to_array(soft_entity.init_positions)
    rest_tets = rest_pos[el2v]
    rest_det = np.linalg.det(rest_tets[:, 1:] - rest_tets[:, :1])
    is_patch = (np.abs(rest_pos[:, 1]) < 0.031) & (np.abs(rest_pos[:, 2] - 0.5) < 0.031)
    is_left_patch = is_patch & (np.abs(rest_pos[:, 0] + 0.05) < 1e-6)
    is_right_patch = is_patch & (np.abs(rest_pos[:, 0] - 0.05) < 1e-6)
    if not is_left_patch.any() or not is_right_patch.any() or (rest_det == 0).any():
        raise RuntimeError("The input mesh must have contact-patch vertices and nondegenerate tetrahedra.")
    patch_width_rest = rest_pos[is_right_patch, 0].mean() - rest_pos[is_left_patch, 0].mean()

    samples = []
    n_steps = math.ceil(args.seconds / scene.dt)
    for i_step in range(n_steps + 1):
        if i_step:
            time = (i_step - 1) * scene.dt
            progress = min(max((time - 0.5) / 2.5, 0.0), 1.0)
            target_pos = 0.025 * progress**2 * (3.0 - 2.0 * progress)
            target_vel = 0.025 * 6.0 * progress * (1.0 - progress) / 2.5
            for direction, entity in zip((1, -1), rigid_entities):
                entity.control_dofs_position_velocity(position=direction * target_pos, velocity=direction * target_vel)
            scene.step()

        state = soft_entity.get_state()
        pos = tensor_to_array(state.pos)[0]
        vel = tensor_to_array(state.vel)[0]
        rigid_pos = np.stack([tensor_to_array(link.get_pos()) for link in rigid_links])
        rigid_vel = np.stack([tensor_to_array(link.get_vel()) for link in rigid_links])
        if not (
            np.isfinite(pos).all()
            and np.isfinite(vel).all()
            and np.isfinite(rigid_pos).all()
            and np.isfinite(rigid_vel).all()
        ):
            raise FloatingPointError(f"Non-finite state at step {i_step}, t={i_step * scene.dt} s.")

        deformed_tets = pos[el2v]
        jacobian = np.linalg.det(deformed_tets[:, 1:] - deformed_tets[:, :1]) / rest_det
        box_dist = np.abs(pos[None] - rigid_pos[:, None]) - (0.01, 0.04, 0.04)
        signed_dist = np.linalg.norm(np.maximum(box_dist, 0.0), axis=-1)
        signed_dist += np.minimum(box_dist.max(axis=-1), 0.0)
        patch_width = pos[is_right_patch, 0].mean() - pos[is_left_patch, 0].mean()
        speed_sq = (vel**2).sum(axis=-1)
        sample = SqueezeSample(
            time_s=i_step * scene.dt,
            gap_m=rigid_pos[1, 0] - rigid_pos[0, 0] - 0.02,
            compression=1.0 - patch_width / patch_width_rest,
            penetration_m=max(0.0, -signed_dist.min()),
            speed_rms_m_s=np.sqrt(speed_sq.mean()),
            speed_max_m_s=np.sqrt(speed_sq.max()),
            jacobian_min=jacobian.min(),
        )
        samples.append(sample)
        if np.abs(pos - camera_lookat).max() > 1.0 or sample.speed_max_m_s > 100.0:
            raise FloatingPointError(f"Unbounded elastic state at step {i_step}: {sample}")
        if np.abs(rigid_pos - camera_lookat).max() > 1.0 or np.linalg.norm(rigid_vel, axis=-1).max() > 100.0:
            raise FloatingPointError(f"Unbounded rigid state at step {i_step}: {sample}")

    if samples[-1].time_s < 4.0:
        return

    failures = []
    if abs(samples[-1].gap_m - 0.07) > 0.001:
        failures.append("Final jaw gap error exceeds 1 mm.")
    if samples[-1].compression < 0.2:
        failures.append("Contact-patch compression is below 20 percent.")
    if max(sample.penetration_m for sample in samples) > 0.001:
        failures.append("Vertex penetration exceeds 1 mm.")
    if min(sample.jacobian_min for sample in samples) <= 0.0:
        failures.append("At least one tetrahedron degenerates or inverts.")
    if max(sample.speed_rms_m_s for sample in samples if sample.time_s >= 3.8) > 0.001:
        failures.append("Root mean square speed exceeds 1 mm/s in the final 0.2 seconds.")
    if failures:
        raise RuntimeError("Squeeze acceptance failed: " + " ".join(failures))


if __name__ == "__main__":
    main()
