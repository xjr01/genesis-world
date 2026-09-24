"""Pour water into coffee using moving mesh cups and passive concentration diffusion.

Run headlessly with ``python examples/pbstf_coffee_water.py``. Add ``--vis`` for the viewer,
``--record`` for a video and stage images, or ``--surface`` for a reconstructed liquid surface.
``--steps`` extends the final resting phase beyond the default ten-second motion sequence.
"""

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from PIL import Image
import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array


@dataclass
class CoffeeWaterScene:
    scene: gs.Scene
    coffee: gs.engine.entities.PBSTFEntity
    water: gs.engine.entities.PBSTFEntity
    water_cup: gs.engine.entities.RigidEntity
    camera: gs.vis.camera.Camera | None


def cup_mesh(radius, height, thickness):
    """Create a watertight cup shell with an open mouth along positive y."""
    profile = np.array(
        [
            (0.0, 0.0),
            (radius + thickness, 0.0),
            (radius + thickness, height + thickness),
            (radius, height + thickness),
            (radius, thickness),
            (0.0, thickness),
        ]
    )
    mesh = trimesh.creation.revolve(profile, sections=96)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2.0, (1.0, 0.0, 0.0)))
    return mesh


def water_cup_pose(time):
    """Return the water cup pose through settling, lifting, pouring and returning, in seconds."""
    times = np.array((0.0, 1.0, 2.5, 3.5, 5.0, 6.5, 7.5, 8.5))
    poses = np.array(
        [
            (0.07, -0.045, 0.0),
            (0.07, -0.045, 0.0),
            (0.07, 0.10, 0.0),
            (0.0, 0.10, 0.0),
            (0.0, 0.10, 115.0),
            (0.0, 0.10, 115.0),
            (0.0, 0.10, 0.0),
            (0.07, -0.045, 0.0),
        ]
    )
    interval = np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 2)
    fraction = np.clip((time - times[interval]) / (times[interval + 1] - times[interval]), 0.0, 1.0)
    fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    pose = poses[interval] + fraction * (poses[interval + 1] - poses[interval])
    angle = math.radians(pose[2])
    return (pose[0], pose[1], 0.0), (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def build_scene(asset_dir, scale=750, is_viewer_shown=False, is_recording=False, is_surface=False):
    """Build two equal-volume liquids with the mop solver and material parameters."""
    if scale <= 0:
        raise ValueError("Particle scale must be positive.")
    particle_size = 2.0 / scale
    cup_path = Path(asset_dir) / "cup.obj"
    cup_mesh(radius=0.03, height=0.07, thickness=0.004).export(cup_path)
    cup_positions = ((-0.065, -0.045, 0.0), (0.07, -0.045, 0.0))
    colliders = [
        gs.options.PBSTFMeshStaticColliderOptions(
            pos=pos,
            file=str(cup_path),
            sdf_res=128,
        )
        for pos in cup_positions
    ]
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.002,
            gravity=(0.0, -9.8, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            disable_constraint=True,
        ),
        pbstf_options=gs.options.PBSTFOptions(
            diffusion_coeff=0.005,
            particle_size=particle_size,
            max_solver_iterations=10,
            topology_rebuild_interval=10,
            max_surface_neighbors=128,
            max_localmesh_neighbors=64,
            enable_pca_normals=False,
            static_colliders=colliders,
            lower_bound=(-0.4, -1.0 / 15.0, -4.0 / 15.0),
            upper_bound=(0.4, 4.0 / 15.0, 4.0 / 15.0),
        ),
        vis_options=gs.options.VisOptions(
            render_particle_as="points",
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.20, 0.25, 0.38),
            camera_lookat=(-0.015, 0.025, 0.0),
            camera_up=(0.0, 1.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=is_viewer_shown,
    )
    liquids = []
    cups = []
    for pos, concentration, name in zip(cup_positions, (1.0, 0.0), ("coffee", "water")):
        cups.append(
            scene.add_entity(
                morph=gs.morphs.Mesh(
                    pos=pos,
                    collision=False,
                    file=str(cup_path),
                    convexify=False,
                    fixed=True,
                ),
                material=gs.materials.Rigid(
                    needs_coup=False,
                ),
                surface=gs.surfaces.Default(
                    color=(0.65, 0.75, 0.85, 0.25),
                ),
                name=f"{name}_cup",
            )
        )
        liquids.append(
            scene.add_entity(
                morph=gs.morphs.Cylinder(
                    pos=(pos[0], pos[1] + 0.004 + particle_size + 0.011, pos[2]),
                    euler=(-90.0, 0.0, 0.0),
                    height=0.022,
                    radius=0.03 - particle_size,
                ),
                material=gs.materials.PBSTF.Liquid(
                    sampler="regular",
                    rho=1000.0,
                    density_compliance=33750.0,
                    surface_tension_compliance=1.0 / 225.0,
                    surface_distance_compliance=40.0,
                    interior_distance_compliance=180.0,
                    surface_viscosity=0.5,
                    interior_viscosity=0.5,
                    is_collider_adhesion_friction_enabled=True,
                    collider_adhesion_compliance=50.0,
                    collider_friction=0.5,
                    c_init=concentration,
                ),
                surface=gs.surfaces.Default(
                    vis_mode="recon" if is_surface else "particle",
                ),
                name=name,
            )
        )
    camera = None
    if is_recording:
        camera = scene.add_camera(
            res=(960, 720),
            pos=(0.20, 0.25, 0.38),
            lookat=(-0.015, 0.025, 0.0),
            up=(0.0, 1.0, 0.0),
            fov=40,
            GUI=False,
        )
    scene.build()
    return CoffeeWaterScene(scene, liquids[0], liquids[1], cups[1], camera)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, default=750)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--vis", dest="is_viewer_shown", action="store_true")
    parser.add_argument("--record", dest="is_recording", action="store_true")
    parser.add_argument("--surface", dest="is_surface", action="store_true")
    args = parser.parse_args()
    if args.scale <= 0 or args.steps <= 0:
        parser.error("--scale and --steps must be positive")
    gs.init(backend=gs.cuda, logging_level="warning")
    output = Path("out/pbstf_coffee_water")
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="coffee-water-") as asset_dir:
        demo = build_scene(asset_dir, args.scale, args.is_viewer_shown, args.is_recording, args.is_surface)
        initial_mass = tensor_to_array(demo.coffee.get_mass() + demo.water.get_mass()).sum()
        if demo.camera is not None:
            demo.camera.start_recording(save_to_filename=str(output / "pour.mp4"), fps=50)
        try:
            with (output / "metrics.csv").open("w", newline="", encoding="ascii") as metrics:
                writer = csv.writer(metrics)
                writer.writerow(
                    ("step", "time", "active_particles", "mass_kg", "coffee_amount", "variance", "water_in_coffee_cup")
                )
                for step in range(args.steps):
                    pos, quat = water_cup_pose((step + 1) * 0.002)
                    demo.scene.pbstf_solver.set_static_colliders_pose(pos, quat, colliders_idx=[1])
                    demo.water_cup.set_pos(pos)
                    demo.water_cup.set_quat(quat)
                    demo.scene.step()
                    is_checkpoint = step in (0, 500, 1750, 2500, 3250, 4250, 4990)
                    if demo.camera is not None and step % 10 == 0:
                        rgb, *_ = demo.camera.render()
                        if is_checkpoint:
                            Image.fromarray(rgb).save(output / f"stage-{step:04d}.png")
                    if (step + 1) % 100 == 0 or step + 1 == args.steps or is_checkpoint:
                        state = demo.scene.pbstf_solver.get_state(demo.scene.sim.cur_substep_local)
                        concentrations = tensor_to_array(state.c)
                        positions = tensor_to_array(state.pos)
                        velocities = tensor_to_array(state.vel)
                        if not all(np.isfinite(values).all() for values in (positions, velocities, concentrations)):
                            raise RuntimeError("Liquid state contains non-finite values.")
                        mass = tensor_to_array(demo.coffee.get_mass() + demo.water.get_mass()).sum()
                        if abs(mass - initial_mass) > initial_mass * 2e-6:
                            raise RuntimeError("Liquid mass changed during pouring.")
                        demo.scene.pbstf_solver.check_errno()
                        if args.is_recording and is_checkpoint:
                            np.savez_compressed(
                                output / f"state-{step:04d}.npz",
                                pos=positions,
                                vel=velocities,
                                c=concentrations,
                                cup_pos=pos,
                                cup_quat=quat,
                            )
                        water_pos = positions[:, demo.water.particle_start : demo.water.particle_end]
                        is_in_cup = (water_pos[..., 0] + 0.065) ** 2 + water_pos[..., 2] ** 2 < 0.03**2
                        is_in_cup &= (water_pos[..., 1] > -0.041) & (water_pos[..., 1] < 0.029)
                        row = (
                            step + 1,
                            (step + 1) * 0.002,
                            tensor_to_array(state.active).sum(),
                            mass,
                            concentrations.sum(),
                            concentrations.var(),
                            is_in_cup.mean(),
                        )
                        writer.writerow(row)
                        metrics.flush()
                        print(row, flush=True)
        finally:
            if demo.camera is not None:
                demo.camera.stop_recording()


if __name__ == "__main__":
    main()
