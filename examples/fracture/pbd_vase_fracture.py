import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.couplers.pbd_rigid_fragments import PBDRigidFragmentBridge
from genesis.utils.misc import tensor_to_array


def sector_seeds(n_sectors=6, bands=((0.087, 0.09), (0.044, 0.19))):
    """Voronoi seeds on the wall midline: angular sectors in two height bands."""
    return [
        (r * np.cos(theta), r * np.sin(theta), z)
        for r, z in bands
        for theta in np.arange(n_sectors) * (2.0 * np.pi / n_sectors)
    ]


def fragment_layout(n_fragments):
    """Pick (n_sectors, bands) and per-band (radius, height) for a requested fragment count.

    Prefers cell shapes that are roughly square on the wall: the arc span per sector and the band
    height should be comparable. Only exact factorizations within sectors [4, 16] and bands [1, 6]
    are used; the caller validates the requested count.
    """
    best = None
    for n_bands in range(1, 7):
        for n_sectors in range(4, 17):
            if n_sectors * n_bands != n_fragments:
                continue
            arc = 2.0 * np.pi * 0.065 / n_sectors
            height = 0.24 / n_bands
            err = abs(arc - height)
            if best is None or err < best[2]:
                best = (n_sectors, n_bands, err)
    if best is None:
        return None, None
    n_sectors, n_bands = best[0], best[1]
    if n_bands == 1:
        bands = ((0.065, 0.14),)
    elif n_bands == 2:
        bands = ((0.087, 0.09), (0.044, 0.19))
    else:
        z_lo, z_hi = 0.05, 0.22
        bands = []
        for i in range(n_bands):
            z = z_lo + (z_hi - z_lo) * i / (n_bands - 1)
            r = 0.087 + (0.044 - 0.087) * (z - 0.09) / (0.19 - 0.09)
            bands.append((max(r, 0.02), z))
        bands = tuple(bands)
    return n_sectors, bands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-r", "--record", action="store_true", default=False, help="record an mp4 of the shatter")
    parser.add_argument("--duration", type=float, default=8.0, help="simulation duration in seconds")
    parser.add_argument("--output", type=Path, default=Path("pbd_vase_fracture.mp4"), help="recording output path")
    parser.add_argument("--fragments", type=int, default=12,
                        help="number of pre-seeded fragments (4-64); the actual count equals sectors x bands" )
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if not 4 <= args.fragments <= 64:
        parser.error("--fragments must be in [4, 64]")

    gs.init(backend=gs.gpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
        ),
        pbd_options=gs.options.PBDOptions(
            lower_bound=(-2.0, -2.0, 0.0),
            upper_bound=(2.0, 2.0, 2.0),
            particle_size=0.005,
        ),
        viewer_options=gs.options.ViewerOptions(
            # slow motion makes the shatter readable; the recording follows this pace
            realtime_factor=0.24,
            camera_pos=(-1.0, -1.6, 1.55),
            camera_lookat=(-0.1, -0.25, 0.3),
            camera_fov=50,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            lower=(0.15, -0.15, 0.0),
            upper=(0.45, 0.15, 0.75),
            fixed=True,
        ),
        material=gs.materials.Rigid(),
    )
    # vase center of mass overhangs the table edge, so it tips over once nudged
    vase = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=os.path.join(gs.utils.get_assets_dir(), "meshes", "vase.obj"),
            pos=(0.14, 0.0, 0.75),
        ),
        material=gs.materials.PBD.Solid(
            sampler="regular",
            static_friction=0.8,
            kinetic_friction=0.7,
            stiffness=1.0,
            fracture_threshold=0.11,
            shear_threshold=0.2,
            rotation_threshold=0.25,
            seam_failure_threshold=1.0,
            # The seed count controls the partition granularity; denser layouts produce smaller shards
            # that may rock on their hull facets before resting (condim=3 carries no rolling friction).
            fragment_seeds=sector_seeds(*fragment_layout(args.fragments)),
            bond_radius_factor=1.15,
        ),
        surface=gs.surfaces.Default(
            color=(0.35, 0.6, 0.8, 1.0),
        ),
    )

    cam = None
    if args.record:
        cam = scene.add_camera(
            res=(960, 720),
            pos=(-1.0, -1.6, 1.55),
            lookat=(-0.1, -0.25, 0.3),
            fov=50,
        )

    # Once a fragment's seams are fully broken, the bridge hands it to the native rigid solver: the fragment
    # becomes a FREE link carrying the mass and full inertia of its particles plus a convex collision proxy,
    # while the PBD render mesh stays the single visual representation of the shards.
    bridge = PBDRigidFragmentBridge(scene.sim)
    scene.sim._pbd_rigid_fragment_bridge = bridge
    bridge.register_solid_fragments(vase, proxy_dir=Path(tempfile.mkdtemp(prefix="pbd_vase_fracture_")))
    scene.build(n_envs=0)

    # The parsed contact time constants are far softer than the impact timescale of a m/s-scale slam at this
    # substep dt, so a slammed fragment would sink visibly into the floor before the contact force ramps.
    # 4e-4 is the solver's own stability floor (2 * substep dt); the remaining values keep the defaults.
    scene.rigid_solver.set_global_sol_params([4e-4, 1.0, 0.9, 0.95, 1.0e-3, 0.5, 2.0])

    # knocked sideways: translate plus spin about y, so it topples over the edge instead of creeping down the side
    pos0 = tensor_to_array(vase.get_particles_pos())
    vel = np.cross((0.0, -4.0, 0.0), pos0 - pos0.mean(axis=0)) + (-0.5, 0.0, 0.0)
    vase.set_particles_vel(vel)

    if cam is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cam.start_recording(save_to_filename=str(args.output), fps=60)
    # recording is driven by scene.step itself; a manual render here would be duplicate work
    for _ in range(math.ceil(args.duration / scene.dt)):
        scene.step()
    if cam is not None:
        cam.stop_recording()


if __name__ == "__main__":
    main()
