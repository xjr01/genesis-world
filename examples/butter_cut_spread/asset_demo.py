"""Cut and lift an end slice using evaluated, meter-scale butter, dish and knife assets."""

import argparse
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
from examples.butter_cut_spread.aligned_contact import AuthoredBreadContact
from examples.butter_cut_spread.aligned_motion import PHASES as ALIGNED_PHASES, continue_motion
from examples.butter_cut_spread.asset_contact import MeshProjection
from examples.butter_cut_spread.asset_runtime import (
    StartupTimings,
    file_digest,
    particle_morph,
    sdf_records,
    verify_prepared,
    write_preparation,
)
from examples.butter_cut_spread.config import Config, angular_velocities, ease
from examples.butter_cut_spread.contact import ToolContact
from examples.butter_cut_spread.cutting_scene import CuttingScene
from examples.butter_cut_spread.face_footprint import BladeFace, face_samples
from examples.butter_cut_spread.geometry import cutting_sweep_limits
from examples.butter_spreading.config import BreadParameters, ButterParameters, ContactParameters
from examples.butter_spreading.runtime import physics_only_visual_updates
from genesis.utils.misc import tensor_to_array


def asset_motion(times, config, assets, blade_mesh, stationary=False):
    orientation = Rotation.from_euler("x", assets["cutting_roll_degrees"], degrees=True) * Rotation.from_euler(
        "y", -90, degrees=True
    )
    cutting_mesh = blade_mesh.copy()
    cutting_mesh.vertices = orientation.apply(blade_mesh.vertices)
    near_y, far_y = cutting_sweep_limits(
        cutting_mesh,
        config.butter_size[1],
        config.butter_size[2],
        config.particle_size,
        direction=assets.get("cutting_direction", 1),
    )
    near_y += config.butter_center[1]
    far_y += config.butter_center[1]
    floor = config.butter_center[2] - config.butter_size[2] / 2
    x = config.butter_center[0] + config.cut_offset
    bottom = floor - cutting_mesh.bounds[0, 2] + 0.00002
    start = bottom + config.butter_size[2] + 0.003
    ends = np.array((0, 0.6, 1.5, 2.9, 3.7, 4.0))
    centers = np.array(
        (
            (x, near_y, start),
            (x, near_y, start),
            (x, near_y, bottom),
            (x, far_y, bottom),
            (x, far_y, start + 0.018),
            (x, far_y, start + 0.018),
        )
    )
    index = np.clip(np.searchsorted(ends, times, side="right") - 1, 0, len(ends) - 2)
    blend = ease((times - ends[index]) / (ends[index + 1] - ends[index]))
    positions = centers[index] + blend[:, None] * (centers[index + 1] - centers[index])
    # Open the fresh cut while rising; a single outward peel avoids recontact
    # without an additional return stroke or a pause at the bottom.
    positions[:, 0] += assets["lift_outward_clearance_m"] * ease((times - config.cut_end_time) / 0.2)
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
        "--prepare-only",
        action="store_true",
        help="Offline sampling, trajectory, SDF and one discarded compile warm-up step.",
    )
    parser.add_argument(
        "--material",
        type=Path,
        help="ButterParameters JSON overrides; omitted fields retain the calibrated real-asset preset.",
    )
    parser.add_argument("--fps", type=int, help="Particle recording frequency; does not change integration dt.")
    parser.add_argument("--dt", type=float, help="Integration step in seconds; requires --prepare-only.")
    parser.add_argument(
        "--particle-size",
        type=float,
        help="Butter spacing in meters during preparation; coarser sampling saves work but resolves less detail.",
    )
    args = parser.parse_args()
    if args.prepare_only and (args.duration is not None or args.stationary):
        parser.error("Offline preparation must use the complete configured trajectory.")
    if args.particle_size is not None and not args.prepare_only:
        parser.error("Particle spacing changes require --prepare-only to rebuild particles and their manifest.")
    if args.dt is not None and (not args.prepare_only or not math.isfinite(args.dt) or args.dt <= 0):
        parser.error("A finite positive --dt requires --prepare-only to rebuild the trajectory.")
    started = time.perf_counter()
    startup = StartupTimings()
    if not args.prepare_only:
        verify_prepared(args.assets.resolve())
    startup.mark("verify_prepared")
    run = args.run.resolve()
    prepare_in_place = args.prepare_only and run == args.assets.resolve()
    if run.exists() and any(run.iterdir()) and not prepare_in_place:
        parser.error("Use a new output directory.")
    run.mkdir(parents=True, exist_ok=True)
    for file in () if prepare_in_place else args.assets.iterdir():
        if file.is_file():
            shutil.copy2(file, run / file.name)
        elif file.name == "annotated":
            shutil.copytree(file, run / file.name)
    snapshot = run / "source"
    snapshot.mkdir(exist_ok=prepare_in_place)
    for file in Path(__file__).parent.glob("*.py"):
        shutil.copy2(file, snapshot / file.name)
    startup.mark("copy_assets_and_source")
    config = Config(**json.loads((run / "config.json").read_text()))
    if args.dt is not None:
        config = replace(config, dt=args.dt)
        (run / "config.json").write_text(json.dumps(asdict(config), indent=2))
    if args.particle_size is not None:
        if not math.isfinite(args.particle_size) or not 0 < args.particle_size <= 0.5 / config.grid_density:
            parser.error("Butter spacing must be positive and at most half a grid cell (eight samples per cell).")
        config = replace(
            config,
            particle_size=args.particle_size,
            contact_particle_size=config.contact_particle_size or config.particle_size,
        )
        (run / "config.json").write_text(json.dumps(asdict(config), indent=2))
    contact_config = replace(config, particle_size=config.contact_particle_size or config.particle_size)
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
    has_bread = assets.get("is_cut_and_spread", False)
    is_spreading = has_bread and not args.stationary
    duration = config.duration if args.duration is None else args.duration
    if not math.isfinite(duration) or duration <= 0 or (not args.stationary and duration > config.duration):
        parser.error("Duration must be positive and within the configured trajectory.")
    steps = math.ceil(duration / config.dt)
    gs.init(backend=gs.gpu, precision="32", seed=config.seed, logging_level="warning")
    startup.mark("engine_initialization")
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
            file=str(run / assets.get("dish_collision_file", "dish.obj")),
            fixed=True,
            convexify=False,
            decimate=False,
            watertighten=None,
        ),
        material=gs.materials.Rigid(
            coup_friction=0.35,
            coup_softness=0.0,
            sdf_cell_size=assets.get("dish_sdf_cell_size_m", 0.0003),
            sdf_max_res=assets.get("dish_sdf_max_res", 768),
        ),
    )
    bread = None
    if has_bread:
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(run / "board-collision.obj"), fixed=True, convexify=False, decimate=False, watertighten=None
            ),
            material=gs.materials.Rigid(
                coup_friction=0.6,
                coup_softness=0.0,
                sdf_cell_size=assets.get("board_sdf_cell_size_m", 0.0005),
                sdf_max_res=assets.get("board_sdf_max_res", 768),
            ),
        )
    startup.mark("static_collision_entities")
    if has_bread:
        bread = scene.add_entity(
            morph=particle_morph(run, "bread", config.bread_particle_size, args.prepare_only),
            material=gs.materials.MPM.PorousBread(
                **BreadParameters().model_dump(), sampler="regular", particle_size=config.bread_particle_size
            ),
        )
    butter = scene.add_entity(
        morph=particle_morph(run, "butter", config.particle_size, args.prepare_only),
        material=gs.materials.MPM.HerschelBulkleyButter(
            **material.model_dump(), dt=config.dt, sampler="regular", particle_size=config.particle_size
        ),
    )
    startup.mark("particle_entities")
    times = np.arange(steps + 1) * config.dt
    blade_mesh = trimesh.load_mesh(run / "blade.obj", process=False) if args.prepare_only else None
    footprint = face_samples(run, blade_mesh, -config.blade_size[1] / 2, args.prepare_only)
    if args.prepare_only:
        positions, angles, quats = asset_motion(times, contact_config, assets, blade_mesh)
        rolls = np.full(len(times), math.radians(assets["cutting_roll_degrees"]))
        if has_bread:
            positions, angles, quats, rolls = continue_motion(times, config, assets, positions, angles, quats)
        np.savez(run / "motion.npz", positions=positions, angles=angles, quats=quats, rolls=rolls)
    else:
        with np.load(run / "motion.npz") as motion:
            positions, angles, quats, rolls = (
                motion[name][: steps + 1] for name in ("positions", "angles", "quats", "rolls")
            )
        if args.stationary:
            positions[:] = positions[0]
            angles[:] = angles[0]
            quats[:] = quats[0]
            rolls[:] = rolls[0]
    velocities = np.diff(positions, axis=0) / config.dt
    joint_angular, angular = angular_velocities(quats, config.dt)
    startup.mark("trajectory")
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
    caches_before_build = sdf_records(scene)
    if not args.prepare_only and any(not row["cache_exists"] for row in caches_before_build):
        raise RuntimeError("Current engine selected a missing SDF cache; run offline preparation again.")
    startup.mark("tool_entity")
    scene.build(cutting_tools=(blade,))
    startup.mark("build_and_load_native_caches")
    # Prescribed trajectories use the authored mesh frame, not an automatically
    # recentered COM/principal-axis frame of an asymmetric imported tool.
    geom_index = blade.geoms[0].idx
    geom_position = tensor_to_array(blade.solver.get_geoms_pos([geom_index]))[0]
    geom_quaternion = tensor_to_array(blade.solver.get_geoms_quat([geom_index]))[0]
    if np.linalg.norm(geom_position - positions[0]) > 1e-7 or abs(np.dot(geom_quaternion, quats[0])) < 1 - 1e-6:
        raise RuntimeError("Native collision geometry does not match the authored tool trajectory frame.")
    with np.load(run / "support.npz") as archive:
        support = archive["origin"], archive["spacing"], archive["heights"]
    face = BladeFace(footprint)
    bread_contact = None
    bread_frames = []
    if bread is not None:
        wet = ContactParameters(blade_stress=0, blade_shear_stress=0)
        bread_contact = AuthoredBreadContact(
            bread,
            butter,
            blade,
            spacing=contact_config.particle_size,
            dt=config.dt,
            blade_size=config.blade_size,
            lower_bound=config.lower_bound,
            upper_bound=config.upper_bound,
            **wet.model_dump(exclude={"blade_normal_time"}),
            blade_normal_relaxation=0.0,
        )
        bread_frames.append(tensor_to_array(bread.get_particles_pos()))
    contact = ToolContact(butter, contact_config, support=support, face=face, bread_contact=bread_contact)
    projection = MeshProjection(
        butter,
        blade,
        Rotation.from_quat(quats[0, [1, 2, 3, 0]]).as_matrix(),
        config.blade_size,
        face=face,
        contact=contact,
    )
    initial = tensor_to_array(butter.get_particles_pos())
    slice_ids = initial[:, 0] > config.butter_center[0] + config.cut_offset + config.blade_size[2] / 2
    if np.max(np.abs(initial.mean(0) - np.array(config.butter_center))) > 0.002:
        raise RuntimeError("Sampled particles do not match the authored asset placement.")
    samples = sorted({0, steps, *(round(i / config.fps / config.dt) for i in range(1, int(duration * config.fps) + 1))})
    frames, faces = [initial], [contact.faces()]
    residue_frames = [contact.residue_faces()]
    knife_frames, quaternion_frames = [positions[0]], [quats[0]]
    target_positions, target_quats = gs.tensor(positions), gs.tensor(quats)
    target_rates = gs.tensor(np.column_stack((velocities, joint_angular)))
    centers, speeds, thetas, omegas = positions.tolist(), velocities.tolist(), angles.tolist(), angular.tolist()
    rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix().tolist()
    startup.mark("contact_and_recording_buffers")
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
        else ALIGNED_PHASES
        if is_spreading
        else (("settle", 0.6), ("cut_mid", 1.5), ("cut", 2.9), ("lift_start", 3.05), ("lift", 3.7), ("hold", 4.0))
    )
    with physics_only_visual_updates(scene):
        for i in range(1 if args.prepare_only else steps):
            # Publish geometry once, after both the velocity and position writes
            blade.set_dofs_velocity(target_rates[i], skip_forward=True)
            blade.set_pos(target_positions[i], zero_velocity=False, relative=False, skip_forward=True)
            blade.set_quat(target_quats[i], zero_velocity=False, relative=False)
            if bread_contact is not None:
                bread_contact.apply(center=centers[i], velocity=speeds[i])
            contact.apply(
                centers[i],
                speeds[i],
                thetas[i],
                omegas[i],
                float(times[i]),
                roll=float(rolls[i]),
            )
            projection.record(centers[i], rotations[i])
            scene.step(update_visualizer=False)
            projection.apply(centers[i + 1], speeds[i], rotations[i + 1], omegas[i])
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
                if bread is not None:
                    bread_points = tensor_to_array(bread.get_particles_pos())
                    if not np.isfinite(bread_points).all():
                        raise RuntimeError(f"Nonfinite bread particles at step {i + 1}")
                    bread_frames.append(bread_points)
                faces.append(contact.faces())
                residue_frames.append(contact.residue_faces())
                knife_frames.append(tensor_to_array(blade.solver.get_geoms_pos([geom_index]))[0])
                quaternion_frames.append(tensor_to_array(blade.solver.get_geoms_quat([geom_index]))[0])
            if phase_finished:
                gs.qd.sync()
                phase_times.append({"name": phases[phase_index][0], "seconds": time.perf_counter() - phase_started})
                if is_spreading:
                    phase_points = tensor_to_array(butter.get_particles_pos())
                    np.savez(
                        run / f"checkpoint-{phases[phase_index][0]}.npz",
                        particles=phase_points,
                        bread_particles=tensor_to_array(bread.get_particles_pos()),
                        knife_positions=positions[i + 1],
                        knife_quaternions=quats[i + 1],
                        time_s=times[i + 1],
                    )
                    print(
                        f"Phase {phases[phase_index][0]}: slice_center={phase_points[slice_ids].mean(0).tolist()}",
                        flush=True,
                    )
                phase_started = time.perf_counter()
                phase_index += 1
            if (i + 1) % 5000 == 0:
                print(
                    f"Step {i + 1}/{steps} t={times[i + 1]:.3f} elapsed={time.perf_counter() - step_started:.1f}s",
                    flush=True,
                )
    gs.qd.sync()
    stepping_seconds = time.perf_counter() - step_started
    if args.prepare_only:
        report = write_preparation(run, scene, time.perf_counter() - started)
        report.update(
            startup_seconds=startup.seconds,
            sdf_before_build=caches_before_build,
            discarded_warmup_steps=1,
            warmup_seconds=stepping_seconds,
        )
        (run / "runtime-preparation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return
    export_started = time.perf_counter()
    np.savez(
        run / "state.npz",
        particles=np.stack(frames),
        tool_faces=np.stack(faces),
        residue_faces=np.stack(residue_frames),
        knife_positions=np.stack(knife_frames),
        knife_quaternions=np.stack(quaternion_frames),
        time_s=times[samples],
        particle_size=config.particle_size,
        bread_particles=np.stack(bread_frames) if bread_frames else np.empty((0, 0, 3)),
        bread_particle_size=config.bread_particle_size,
        knife_size=config.blade_size,
        fps=config.fps,
    )
    export_seconds = time.perf_counter() - export_started
    hash_started = time.perf_counter()
    state_hash = file_digest(run / "state.npz")
    hash_seconds = time.perf_counter() - hash_started
    report = {
        "is_complete": args.stationary or math.isclose(duration, config.duration),
        "steps": steps,
        "frames": len(frames),
        "butter_particles": butter.n_particles,
        "bread_particles": 0 if bread is None else bread.n_particles,
        "all_particles_active": bool(tensor_to_array(butter.get_particles_active()).all())
        and (bread is None or bool(tensor_to_array(bread.get_particles_active()).all())),
        "initialization_seconds": init_seconds,
        "startup_seconds": startup.seconds,
        "native_sdf": caches_before_build,
        "stepping_seconds": stepping_seconds,
        "export_seconds": export_seconds,
        "hash_seconds": hash_seconds,
        "state_storage": "uncompressed_npz",
        "elapsed_seconds": time.perf_counter() - started,
        "phases": phase_times,
        "cpic_cutting_geom_indices": [geom.idx for geom in blade.geoms],
        "cpic_separation_checks": separation_checks,
        "material": material.model_dump(),
        "adhesion": asdict(contact.parameters),
        "residue_peak_particles": int(np.max(np.count_nonzero(np.stack(residue_frames), axis=1))),
        "residue_peak_scope": "Maximum over saved frames, not every integration step.",
        "residue_scope": "Sparse existing active particles constrained to local anchors; external kinematic adhesion with bread-contact and finite-distance release.",
        "max_contact_correction_m": projection.maximum_correction(),
        "max_tool_pose_error_m": float(np.linalg.norm(np.stack(knife_frames) - positions[samples], axis=1).max()),
        "stationary": args.stationary,
        "initial_bounds_m": [initial.min(0).tolist(), initial.max(0).tolist()],
        "final_bounds_m": [frames[-1].min(0).tolist(), frames[-1].max(0).tolist()],
        "state_sha256": state_hash,
    }
    (run / "simulation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
