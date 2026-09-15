"""Simulate a position-based dynamics (PBD) elastic box resting on rigid ground under gravity."""

import argparse
import math
import os
import sys
from pathlib import Path

# Select this checkout when the editable installation points elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import genesis as gs
from genesis.utils.element import create_tetrahedral_grid

SPONGE_RENDER_SKELETON = "skeleton"
SPONGE_RENDER_STYLES = (SPONGE_RENDER_SKELETON, "solid")


def build_scene(show_viewer=False, sponge_render_style=SPONGE_RENDER_SKELETON):
    """Build the tetrahedral box and fixed ground with Y-up gravity."""
    if sponge_render_style not in SPONGE_RENDER_STYLES:
        raise ValueError(f"Unknown sponge render style: {sponge_render_style}")
    dt = 0.01
    box_lower = (-0.04, 0.02 / 15.0, -0.08)
    box_upper = (0.04, 0.85 / 15.0, 0.08)
    box_offset = (-7.0 / 30.0, 0.0, 0.0)
    box_size = tuple(box_upper[axis] - box_lower[axis] for axis in range(3))
    center = tuple(box_offset[axis] + 0.5 * (box_lower[axis] + box_upper[axis]) for axis in range(3))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, -9.8, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            disable_constraint=True,
        ),
        pbd_options=gs.options.PBDUnifiedOptions(
            particle_size=0.005,
            lower_bound=(-0.4, -1.0 / 15.0, -4.0 / 15.0),
            upper_bound=(0.4, 4.0 / 15.0, 4.0 / 15.0),
            max_solver_iterations=30,
            constraint_acceleration=0.85,
        ),
        viewer_options=gs.options.ViewerOptions(
            refresh_rate=round(1.0 / dt),
            camera_pos=(center[0] + 0.2, center[1] + 0.15, center[2] + 0.3),
            camera_lookat=center,
            camera_up=(0.0, 1.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    vertices, elements = create_tetrahedral_grid(
        lower=tuple(-0.5 * size for size in box_size),
        upper=tuple(0.5 * size for size in box_size),
        resolution=(15, 10, 30),
    )
    scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=center,
            vertices=vertices,
            elements=elements,
        ),
        material=gs.materials.PBD.Elastic(
            rho=30.0,
            stretch_relaxation=0.1,
            volume_relaxation=0.15,
        ),
        surface=gs.surfaces.Default(
            color=(0.95, 0.68, 0.12),
            vis_mode="tetrahedral" if sponge_render_style == SPONGE_RENDER_SKELETON else "visual",
        ),
        name="sponge",
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, -1.0 / 60.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            fixed=True,
            size=(0.8, 1.0 / 30.0, 8.0 / 15.0),
        ),
        material=gs.materials.Rigid(
            coup_friction=0.0,
            is_coup_reaction_enabled=False,
        ),
        surface=gs.surfaces.Default(
            color=(0.36, 0.24, 0.14),
        ),
        name="wipe_table",
    )
    scene.build()
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-t", "--seconds", type=float, default=10.0, help="Simulation duration in seconds")
    parser.add_argument(
        "--sponge-render-style",
        choices=SPONGE_RENDER_STYLES,
        default=SPONGE_RENDER_SKELETON,
        help="Render all tetrahedron edges or the solid surface",
    )
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0.0:
        parser.error("--seconds must be finite and positive")
    if "PYTEST_VERSION" in os.environ:
        args.seconds = min(args.seconds, 0.02)

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, seed=0)
    scene = build_scene(show_viewer=args.vis, sponge_render_style=args.sponge_render_style)
    for _ in range(math.ceil(args.seconds / scene.dt)):
        scene.step()


if __name__ == "__main__":
    main()
