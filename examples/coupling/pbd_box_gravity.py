"""Simulate a position-based dynamics (PBD) elastic box with keyboard playback.

With --vis, Space toggles playback and Left/Right move one frame while paused. The viewer starts paused and keeps
each visited scene state in memory for backward playback; --seconds bounds the history length.
"""

import argparse
import math
import os
import sys
from pathlib import Path
from queue import SimpleQueue

# Select this checkout when the editable installation points elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import genesis as gs
from genesis.utils.element import create_tetrahedral_grid
from genesis.vis.keybindings import Key, Keybind

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
            max_solver_iterations=100,
            constraint_acceleration=0.0,
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
    sponge = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=center,
            vertices=vertices,
            elements=elements,
        ),
        material=gs.materials.PBD.Elastic(
            rho=30.0,
            stretch_relaxation=0.25,
            volume_relaxation=0.15,
            # stretch_compliance=5.0,
            # volume_compliance=1.0,
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
    parser.add_argument(
        "-v", "--vis", action="store_true", help="Show viewer: Space plays/pauses, Left/Right step frames"
    )
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
    n_steps = math.ceil(args.seconds / scene.dt)
    if not args.vis or "PYTEST_VERSION" in os.environ:
        for _ in range(n_steps):
            scene.step()
        return

    # Queue viewer callbacks so state restoration and simulation stay on the stepping thread.
    frame_requests = SimpleQueue()
    scene.viewer.register_keybinds(
        Keybind(name="play_pause", key=Key.SPACE, callback=frame_requests.put, args=(0,)),
        Keybind(name="previous_frame", key=Key.LEFT, callback=frame_requests.put, args=(-1,)),
        Keybind(name="next_frame", key=Key.RIGHT, callback=frame_requests.put, args=(1,)),
    )
    states = [scene.get_state()]
    frame = 0
    is_paused = True
    gs.logger.info("Space: play/pause. Left/Right: previous/next frame. Close the viewer to exit.")
    while scene.viewer.is_alive():
        frame_step = 0 if is_paused else 1
        if not frame_requests.empty():
            frame_step = frame_requests.get()
            is_paused = not is_paused if frame_step == 0 else True
        next_frame = min(max(frame + frame_step, 0), n_steps)
        if next_frame == len(states):
            scene.step()
            states.append(scene.get_state())
        elif next_frame != frame:
            scene.reset(states[next_frame])
        else:
            scene.viewer.update()
        frame = next_frame
        if frame == n_steps:
            is_paused = True


if __name__ == "__main__":
    main()
