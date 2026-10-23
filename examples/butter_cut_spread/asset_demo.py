"""Cut and lift an end slice using evaluated, meter-scale butter, dish and knife assets."""

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

import genesis as gs
from examples.butter_cut_spread.asset_contact import MeshProjection
from examples.butter_cut_spread.config import Config, angular_velocities, ease
from examples.butter_cut_spread.contact import ToolContact
from examples.butter_cut_spread.cutting_scene import CuttingScene
from examples.butter_spreading.config import ButterParameters
from examples.butter_spreading.runtime import physics_only_visual_updates
from genesis.utils.misc import tensor_to_array


def asset_motion(times, config, assets, blade_mesh, stationary=False):
    orientation = Rotation.from_euler("x", assets["cutting_roll_degrees"], degrees=True) * Rotation.from_euler(
        "y", -90, degrees=True
    )
    cutting_mesh = blade_mesh.copy()
    cutting_mesh.vertices = orientation.apply(blade_mesh.vertices)
    section = cutting_mesh.section(plane_origin=(0, 0, 0), plane_normal=(1, 0, 0))
    if section is None:
        raise ValueError("The authored blade must intersect its cutting plane.")
    contour = np.concatenate(section.discrete)
    tip_y = contour[np.argmin(contour[:, 2]), 1]
    # The lowest cutting edge traverses both short-side boundaries. Starting
    # over the butter also sweeps the upper section during the downward stroke.
    margin = max(0.001, 2 * config.particle_size)
    entry_y = config.butter_center[1] + config.butter_size[1] / 2 + margin - tip_y
    exit_y = config.butter_center[1] - config.butter_size[1] / 2 - margin - tip_y
    if (
        contour[:, 1].min() + entry_y > config.butter_center[1] - config.butter_size[1] / 2
        or contour[:, 1].max() + entry_y < config.butter_center[1] + config.butter_size[1] / 2
    ):
        raise ValueError("The blade section must span the butter short side during entry.")
    floor = config.butter_center[2] - config.butter_size[2] / 2
    x = config.butter_center[0] + config.cut_offset
    bottom = floor - assets["blade_min_z_when_cutting_m"] + 0.00015
    start = bottom + config.butter_size[2] + 0.003
    ends = np.array((0, 0.6, 1.5, 2.6, 2.9, 3.7, 4.0))
    centers = np.array(
        (
            (x, entry_y, start),
            (x, entry_y, start),
            (x, entry_y, bottom),
            (x, exit_y, bottom),
            (x + 0.001, config.butter_center[1], bottom + 0.002),
            (x + 0.002, config.butter_center[1], start + 0.018),
            (x + 0.002, config.butter_center[1], start + 0.018),
        )
    )
    index = np.clip(np.searchsorted(ends, times, side="right") - 1, 0, len(ends) - 2)
    blend = ease((times - ends[index]) / (ends[index + 1] - ends[index]))
    positions = centers[index] + blend[:, None] * (centers[index + 1] - centers[index])
    if stationary:
        positions[:] = centers[0]
    angles = np.tile((-np.pi / 2, 0), (len(times), 1))
    quats = np.tile(orientation.as_quat()[[3, 0, 1, 2]], (len(times), 1))
    return positions, angles, quats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--stationary", action="store_true")
    parser.add_argument(
        "--material",
        type=Path,
        help="ButterParameters JSON overrides; omitted fields retain the calibrated real-asset preset.",
    )
    parser.add_argument("--fps", type=int, help="Particle recording frequency; does not change integration dt.")
    args = parser.parse_args()
    run = args.run.resolve()
    if run.exists() and any(run.iterdir()):
        parser.error("Use a new output directory.")
    run.mkdir(parents=True, exist_ok=True)
    for file in args.assets.iterdir():
        if file.is_file():
            shutil.copy2(file, run / file.name)
    snapshot = run / "source"
    snapshot.mkdir()
    for file in Path(__file__).parent.glob("*.py"):
        shutil.copy2(file, snapshot / file.name)
    config = Config(**json.loads((run / "config.json").read_text()))
    if args.fps is not None:
        if args.fps <= 0 or args.fps * config.dt > 1:
            parser.error("Recording frequency must be positive and no greater than the integration frequency.")
        config = replace(config, fps=args.fps)
        (run / "config.json").write_text(json.dumps(asdict(config), indent=2))
    material = ButterParameters.model_validate_json(Path(__file__).with_name("asset_material.json").read_text())
    if args.material is not None:
        material = ButterParameters.model_validate(material.model_dump() | json.loads(args.material.read_text()))
    (run / "material.json").write_text(material.model_dump_json(indent=2))
    assets = json.loads((run / "assets.json").read_text())
    duration = config.duration if args.duration is None else args.duration
    if not math.isfinite(duration) or duration <= 0 or (not args.stationary and duration > 4.0):
        parser.error("Duration must be positive; the moving asset trajectory ends at 4 seconds.")
    steps = math.ceil(duration / config.dt)
    started = time.perf_counter()
    gs.init(backend=gs.gpu, precision="32", seed=config.seed, logging_level="warning")
    scene = CuttingScene(
        sim_options=gs.options.SimOptions(dt=config.dt, substeps=1, gravity=(0, 0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, disable_constraint=True),
        mpm_options=gs.options.MPMOptions(
            lower_bound=config.lower_bound,
            upper_bound=config.upper_bound,
            grid_density=config.grid_density,
            particle_size=config.particle_size,
            enable_CPIC=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Box(size=(2, 2, 0.02), pos=(0, 0, -0.01), fixed=True),
        material=gs.materials.Rigid(coup_friction=0.35, coup_softness=0.0),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(run / "dish.obj"), fixed=True, convexify=False, decimate=False, watertighten=None
        ),
        material=gs.materials.Rigid(coup_friction=0.35, coup_softness=0.0, sdf_cell_size=0.0003, sdf_max_res=768),
    )
    butter = scene.add_entity(
        morph=gs.morphs.Mesh(file=str(run / "butter.obj")),
        material=gs.materials.MPM.HerschelBulkleyButter(
            **material.model_dump(), dt=config.dt, sampler="regular", particle_size=config.particle_size
        ),
    )
    times = np.arange(steps + 1) * config.dt
    blade_mesh = trimesh.load_mesh(run / "blade.obj", process=False)
    positions, angles, quats = asset_motion(times, config, assets, blade_mesh, args.stationary)
    velocities = np.diff(positions, axis=0) / config.dt
    joint_angular, angular = angular_velocities(quats, config.dt)
    blade = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(run / "blade.obj"),
            pos=positions[0],
            quat=quats[0],
            align=False,
            convexify=False,
            decimate=False,
            watertighten=None,
        ),
        material=gs.materials.Rigid(
            rho=7800,
            coup_friction=0.02,
            coup_softness=0.0,
            is_coup_reaction_enabled=False,
            gravity_compensation=1.0,
            sdf_cell_size=0.00015,
            sdf_max_res=768,
        ),
    )
    scene.build(cutting_tools=(blade,))
    # Prescribed trajectories use the authored mesh frame, not an automatically
    # recentered COM/principal-axis frame of an asymmetric imported tool.
    geom_index = blade.geoms[0].idx
    geom_position = tensor_to_array(blade.solver.get_geoms_pos([geom_index]))[0]
    geom_quaternion = tensor_to_array(blade.solver.get_geoms_quat([geom_index]))[0]
    if np.linalg.norm(geom_position - positions[0]) > 1e-7 or abs(np.dot(geom_quaternion, quats[0])) < 1 - 1e-6:
        raise RuntimeError("Native collision geometry does not match the authored tool trajectory frame.")
    with np.load(run / "support.npz") as archive:
        support = archive["origin"], archive["spacing"], archive["heights"]
    contact = ToolContact(butter, config, support=support)
    projection = MeshProjection(
        butter, blade, Rotation.from_quat(quats[0, [1, 2, 3, 0]]).as_matrix(), config.blade_size
    )
    initial = tensor_to_array(butter.get_particles_pos())
    if np.max(np.abs(initial.mean(0) - np.array(config.butter_center))) > 0.002:
        raise RuntimeError("Sampled particles do not match the authored asset placement.")
    samples = sorted({0, steps, *(round(i / config.fps / config.dt) for i in range(1, int(duration * config.fps) + 1))})
    frames, faces = [initial], [contact.faces()]
    knife_frames, quaternion_frames = [positions[0]], [quats[0]]
    target_positions, target_quats = gs.tensor(positions), gs.tensor(quats)
    target_rates = gs.tensor(np.column_stack((velocities, joint_angular)))
    centers, speeds, thetas, omegas = positions.tolist(), velocities.tolist(), angles.tolist(), angular.tolist()
    init_seconds = time.perf_counter() - started
    print(
        f"Built {butter.n_particles} butter particles; init={init_seconds:.2f}s; bounds={np.stack((initial.min(0), initial.max(0))).tolist()}",
        flush=True,
    )
    phase_times = []
    separation_checks = []
    step_started = phase_started = time.perf_counter()
    phase_index = 0
    phases = (
        (("stationary", duration),)
        if args.stationary
        else (("settle", 0.6), ("press", 1.5), ("draw_cut", 2.6), ("separate", 2.9), ("lift", 3.7), ("hold", 4.0))
    )
    with physics_only_visual_updates(scene):
        for i in range(steps):
            blade.set_pos(target_positions[i], zero_velocity=False, relative=False)
            blade.set_quat(target_quats[i], zero_velocity=False, relative=False)
            blade.set_dofs_velocity(target_rates[i])
            contact.apply(
                centers[i],
                speeds[i],
                thetas[i],
                omegas[i],
                float(times[i]),
                roll=math.radians(assets["cutting_roll_degrees"]),
            )
            projection.record(centers[i])
            scene.step(update_visualizer=False)
            projection.apply(centers[i + 1], speeds[i])
            phase_finished = phase_index < len(phases) and times[i + 1] >= phases[phase_index][1]
            if i == 0 or phase_finished:
                counts = scene.get_cpic_transfer_counts()
                if counts.sum() != counts[[geom.idx for geom in blade.geoms]].sum():
                    raise RuntimeError("A non-cutting rigid geometry separated MPM transfers.")
                separation_checks.append({"time_s": float(times[i + 1]), "counts": counts.tolist()})
                print(f"CPIC t={times[i + 1]:.5f}: {counts.tolist()}", flush=True)
            if i + 1 == samples[len(frames)]:
                points = tensor_to_array(butter.get_particles_pos())
                if not np.isfinite(points).all():
                    raise RuntimeError(f"Nonfinite particles at step {i + 1}")
                frames.append(points)
                faces.append(contact.faces())
                knife_frames.append(tensor_to_array(blade.solver.get_geoms_pos([geom_index]))[0])
                quaternion_frames.append(tensor_to_array(blade.solver.get_geoms_quat([geom_index]))[0])
            if phase_finished:
                gs.qd.sync()
                phase_times.append({"name": phases[phase_index][0], "seconds": time.perf_counter() - phase_started})
                phase_started = time.perf_counter()
                phase_index += 1
            if (i + 1) % 5000 == 0:
                print(
                    f"Step {i + 1}/{steps} t={times[i + 1]:.3f} elapsed={time.perf_counter() - step_started:.1f}s",
                    flush=True,
                )
    gs.qd.sync()
    stepping_seconds = time.perf_counter() - step_started
    np.savez_compressed(
        run / "state.npz",
        particles=np.stack(frames),
        tool_faces=np.stack(faces),
        knife_positions=np.stack(knife_frames),
        knife_quaternions=np.stack(quaternion_frames),
        time_s=times[samples],
        particle_size=config.particle_size,
        knife_size=config.blade_size,
        fps=config.fps,
    )
    report = {
        "is_complete": True,
        "steps": steps,
        "frames": len(frames),
        "butter_particles": butter.n_particles,
        "all_particles_active": bool(tensor_to_array(butter.get_particles_active()).all()),
        "initialization_seconds": init_seconds,
        "stepping_seconds": stepping_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "phases": phase_times,
        "cpic_cutting_geom_indices": [geom.idx for geom in blade.geoms],
        "cpic_separation_checks": separation_checks,
        "material": material.model_dump(),
        "max_contact_correction_m": projection.maximum_correction(),
        "max_tool_pose_error_m": float(np.linalg.norm(np.stack(knife_frames) - positions[samples], axis=1).max()),
        "stationary": args.stationary,
        "initial_bounds_m": [initial.min(0).tolist(), initial.max(0).tolist()],
        "final_bounds_m": [frames[-1].min(0).tolist(), frames[-1].max(0).tolist()],
        "state_sha256": hashlib.sha256((run / "state.npz").read_bytes()).hexdigest(),
    }
    (run / "simulation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
