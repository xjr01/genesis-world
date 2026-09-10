"""Squeeze a position-based dynamics (PBD) elastic box with one-way rigid collision coupling."""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Select this checkout when the editable installation points elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import genesis as gs
from genesis.utils.element import create_tetrahedral_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation step duration in seconds")
    parser.add_argument(
        "--substeps", type=int, default=1, help="More substeps resolve contact at increased runtime cost"
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
        pbd_options=gs.options.PBDOptions(
            particle_size=0.005,
            lower_bound=(-0.2, -0.2, 0.3),
            upper_bound=(0.2, 0.2, 0.7),
            max_stretch_solver_iterations=10,
            max_volume_solver_iterations=10,
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
        lower=(-0.05, -0.05, -0.05), upper=(0.05, 0.05, 0.05), resolution=(20, 20, 20)
    )
    scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=camera_lookat,
            vertices=vertices,
            elements=elements,
        ),
        material=gs.materials.PBD.Elastic(),
        surface=gs.surfaces.Default(
            color=(0.95, 0.65, 0.15),
            smooth=False,
        ),
        # The displayed surface must follow the same triangles that contact the jaws.
        vis_mode="collision",
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

    # Convert exponential velocity decay to the explicit force update so drag stays dissipative across substep sizes.
    substep_dt = scene.pbd_solver.substep_dt
    drag_coefficient = -math.expm1(-500.0 * substep_dt) / substep_dt
    scene.add_force_field(gs.force_fields.Drag(linear=drag_coefficient)).activate()
    scene.build()
    position_gain = 1e5
    # The position term is explicit; damping must also stabilize its discrete update at the chosen step size.
    velocity_gain = 1000.0
    for entity in rigid_entities:
        entity.set_dofs_kp(kp=position_gain)
        entity.set_dofs_kv(kv=velocity_gain)

    for i_step in range(math.ceil(args.seconds / scene.dt)):
        time = i_step * scene.dt
        progress = min(max((time - 0.5) / 2.5, 0.0), 1.0)
        target_pos = 0.025 * progress**2 * (3.0 - 2.0 * progress)
        target_vel = 0.025 * 6.0 * progress * (1.0 - progress) / 2.5
        for direction, entity in zip((1, -1), rigid_entities):
            entity.control_dofs_position_velocity(position=direction * target_pos, velocity=direction * target_vel)
        scene.step()


if __name__ == "__main__":
    main()
