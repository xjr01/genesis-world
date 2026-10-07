"""Record and validate the normalized coffee manipulation task."""

import argparse
import csv
import math
import time as wall_time
from pathlib import Path

import numpy as np
from PIL import Image

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .implementation import (
    CONTROL_DT,
    MOTION_END,
    SOLVER_CONFIG,
    Phase,
    check_contacts,
    measure_motion,
    motion_target,
    observe_scene,
    step_scene,
    surface_clearance,
    update_motion,
)
from .scene import build_scene


def main():
    process_started = wall_time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particle-size", type=float, default=SOLVER_CONFIG.pbstf_particle_size)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--record-segment-seconds", type=float, default=1.0)
    parser.add_argument("--vis", dest="is_viewer_shown", action="store_true")
    parser.add_argument("--record", dest="is_recording", action="store_true")
    parser.add_argument("--surface", dest="is_surface", action="store_true")
    parser.add_argument("--check-motion", dest="is_motion_only", action="store_true")
    parser.add_argument("--no-liquid", dest="is_liquid_enabled", action="store_false")
    parser.add_argument(
        "--profile-build",
        action="store_true",
        help="Show Genesis scene-build and kernel-compilation timers.",
    )
    parser.add_argument("--output", type=Path, default=Path("out/pbstf_coffee_water"))
    args = parser.parse_args()
    if args.particle_size <= 0 or (args.steps is not None and args.steps <= 0) or args.record_segment_seconds <= 0:
        parser.error("--particle-size, --steps, and --record-segment-seconds must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    gs.init(
        backend=gs.cuda,
        precision="32",
        logging_level="info" if args.profile_build else "warning",
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    demo = build_scene(
        particle_size=args.particle_size,
        is_viewer_shown=args.is_viewer_shown,
        is_recording=args.is_recording,
        is_surface=args.is_surface,
        is_motion_only=args.is_motion_only,
        is_liquid_enabled=args.is_liquid_enabled,
    )
    build_seconds = wall_time.perf_counter() - process_started
    print(f"coffee_water timing: build_seconds={build_seconds:.3f}", flush=True)
    dt = CONTROL_DT
    steps = args.steps if args.steps is not None else round(MOTION_END / dt)
    record_segment_steps = max(1, round(args.record_segment_seconds / dt))
    initial_mass = None
    if demo.coffee is not None:
        initial_mass = tensor_to_array(demo.coffee.get_mass() + demo.water.get_mass()).sum()
    if demo.camera is not None:
        demo.camera.start_recording(save_to_filename=str(output / "coffee-water-segment-00000.mp4"), fps=50)
    record_segment_index = 0
    previous_phases = None
    observation = observe_scene(demo)
    checkpoint_steps = {round(time / dt) - 1 for time in (3.0, 4.0, 6.0, 6.4, 14.0, 20.0, MOTION_END)}
    absorbed_particles = 0
    absorbed_water_particles = 0
    gripper_clearance = math.inf
    previous_stir_angle = None
    measured_stir_angle = 0.0
    gripper_geoms_idx = tuple(field.geom_idx for field in demo.gripper_distance_fields)
    target = motion_target(0.0, demo.motion, observation, demo.config)
    executed_steps = 0
    simulation_started = wall_time.perf_counter()
    try:
        with (output / "metrics.csv").open("w", newline="", encoding="ascii") as metrics:
            writer = csv.writer(metrics)
            writer.writerow(
                (
                    "step",
                    "time",
                    "right_phase",
                    "left_phase",
                    "ik_error_m",
                    "right_hand_error_m",
                    "left_hand_error_m",
                    "cup_position_error_m",
                    "cup_rotation_error_deg",
                    "cup_tilt_deg",
                    "knock_spilled_particles",
                    "water_particles_in_cup",
                    "water_particles_before_knock",
                    "max_knock_tilt_deg",
                    "cup_grasp_span_m",
                    "right_finger_opening_m",
                    "left_finger_opening_m",
                    "pour_liquid_gripper_clearance_m",
                    "stir_turns",
                    "measured_stir_turns",
                    "sponge_position_error_m",
                    "active_particles",
                    "mass_kg",
                    "coffee_amount",
                    "variance",
                    "water_in_coffee_cup",
                    "absorbed_particles",
                    "absorbed_water_particles",
                    "sponge_wetness",
                )
            )
            for step in range(steps):
                if args.steps is None and step > 0 and target.right_phase == target.left_phase == Phase.REST:
                    break
                time = (step + 1) * dt
                target, ik_error = update_motion(demo, time, observation)
                step_scene(demo)
                executed_steps += 1
                observation = observe_scene(demo)
                if (
                    demo.camera is not None
                    and args.is_recording
                    and (step + 1) % record_segment_steps == 0
                    and step + 1 < steps
                ):
                    demo.camera.stop_recording()
                    record_segment_index += 1
                    demo.camera.start_recording(
                        save_to_filename=str(output / f"coffee-water-segment-{record_segment_index:05d}.mp4"),
                        fps=50,
                    )
                measurement = measure_motion(time, target, observation, demo.config)
                if args.progress_every and ((step + 1) % args.progress_every == 0 or step + 1 == steps):
                    print(
                        f"coffee_water progress: step={step + 1}/{steps}, simulated_time={demo.scene.cur_t:.6f}s, "
                        f"wall_seconds={wall_time.perf_counter() - simulation_started:.3f}, "
                        f"right={target.right_phase.name}, left={target.left_phase.name}",
                        flush=True,
                    )
                if measurement.hands_position_error[1] > 0.005 or (
                    target.right_phase not in (Phase.REACH_SPONGE, Phase.CATCH)
                    and measurement.hands_position_error[0] > 0.005
                ):
                    raise RuntimeError(f"Arm tracking failed at {time:.3f}s: {measurement.hands_position_error}m.")
                if target.left_phase == Phase.STIR:
                    rod_offset = observation.rod.pos - demo.config.assets.coffee_cup_pos
                    angle = math.atan2(rod_offset[2], rod_offset[0])
                    if previous_stir_angle is not None:
                        difference = angle - previous_stir_angle
                        measured_stir_angle += math.atan2(math.sin(difference), math.cos(difference))
                    previous_stir_angle = angle
                check_contacts(demo, time)
                demo.scene.rigid_solver.check_errno()
                if demo.water is not None and time <= 6.0:
                    clearance = surface_clearance(
                        demo.water.get_particles_pos(),
                        demo.gripper_distance_fields,
                        demo.scene.rigid_solver.get_geoms_pos(gripper_geoms_idx),
                        demo.scene.rigid_solver.get_geoms_quat(gripper_geoms_idx),
                    ).item()
                    gripper_clearance = min(gripper_clearance, clearance)
                    if clearance < 0.5 * args.particle_size:
                        raise RuntimeError(
                            f"Pouring liquid reached the right gripper at {time:.3f}s: {clearance:.6f}m."
                        )
                if time <= 6.0 and (measurement.cup_position_error > 0.002 or measurement.cup_rotation_error > 2.0):
                    raise RuntimeError(
                        f"Pour tracking failed at {time:.3f}s: position={measurement.cup_position_error:.6f}m, "
                        f"rotation={measurement.cup_rotation_error:.3f}deg."
                    )
                if target.right_phase >= Phase.CLEAR_CUP and (
                    measurement.cup_position_error > 0.002 or observation.cup_tilt > 2.0
                ):
                    raise RuntimeError(
                        f"Cup recovery failed at {time:.3f}s: position={measurement.cup_position_error:.6f}m, "
                        f"tilt={observation.cup_tilt:.3f}deg."
                    )
                if target.right_phase in (Phase.WIPE, Phase.REST) and measurement.sponge_position_error > 0.01:
                    raise RuntimeError(
                        f"Sponge tracking failed at {time:.3f}s: {measurement.sponge_position_error:.6f}m."
                    )
                phases = (target.right_phase, target.left_phase)
                is_checkpoint = phases != previous_phases or step + 1 == steps or step in checkpoint_steps
                if is_checkpoint:
                    gs.logger.info(f"{time:.3f}s: right={target.right_phase.name}, left={target.left_phase.name}")
                previous_phases = phases
                if demo.camera is not None and is_checkpoint:
                    rgb, *_ = demo.camera.render()
                    stage = f"{target.right_phase.name.lower()}-{target.left_phase.name.lower()}"
                    Image.fromarray(rgb).save(output / f"stage-{step:05d}-{stage}.png")
                if (step + 1) % 100 == 0 or step + 1 == steps or is_checkpoint:
                    motion_row = (
                        step + 1,
                        time,
                        target.right_phase.name,
                        target.left_phase.name,
                        ik_error,
                        *measurement.hands_position_error,
                        measurement.cup_position_error,
                        measurement.cup_rotation_error,
                        observation.cup_tilt,
                        demo.motion.spilled_particles,
                        observation.water_in_cup,
                        demo.motion.water_before_reach,
                        demo.motion.max_knock_tilt,
                        observation.cup_grasp_span,
                        target.right_opening,
                        target.left_opening,
                        gripper_clearance if demo.water is not None else None,
                        target.stir_angle / (2.0 * math.pi),
                        measured_stir_angle / (2.0 * math.pi),
                        measurement.sponge_position_error,
                    )
                    if demo.water is None:
                        writer.writerow(motion_row + (None,) * 8)
                        metrics.flush()
                        continue
                    coffee_positions = tensor_to_array(demo.coffee.get_particles_pos())
                    water_positions = tensor_to_array(demo.water.get_particles_pos())
                    positions = np.concatenate((coffee_positions, water_positions), axis=-2)
                    velocities = np.concatenate(
                        (
                            tensor_to_array(demo.coffee.get_particles_vel()),
                            tensor_to_array(demo.water.get_particles_vel()),
                        ),
                        axis=-2,
                    )
                    concentrations = np.concatenate(
                        (
                            tensor_to_array(demo.coffee.get_particles_concentration()),
                            tensor_to_array(demo.water.get_particles_concentration()),
                        ),
                        axis=-1,
                    )
                    if not all(np.isfinite(values).all() for values in (positions, velocities, concentrations)):
                        raise RuntimeError("Liquid state contains non-finite values.")
                    mass = tensor_to_array(demo.coffee.get_mass() + demo.water.get_mass()).sum()
                    if abs(mass - initial_mass) > initial_mass * 2e-6:
                        raise RuntimeError("Liquid mass changed during the simulation.")
                    demo.scene.pbstf_solver.check_errno()
                    coffee_absorbed_idx = tensor_to_array(demo.coffee.get_particles_absorbed_collider_idx())
                    water_absorbed_idx = tensor_to_array(demo.water.get_particles_absorbed_collider_idx())
                    is_absorbed = np.concatenate((coffee_absorbed_idx, water_absorbed_idx), axis=-1) >= 0
                    absorbed_particles = is_absorbed.sum()
                    absorbed_water_particles = (water_absorbed_idx >= 0).sum()
                    wetness = tensor_to_array(demo.scene.pbstf_solver.get_static_collider_wetness(4))
                    if args.is_recording and is_checkpoint:
                        np.savez_compressed(
                            output / f"state-{step:05d}.npz",
                            pos=positions,
                            vel=velocities,
                            c=concentrations,
                            cup_pos=observation.cup.pos,
                            cup_quat=observation.cup.quat,
                            robot_qpos=tensor_to_array(demo.robot.get_qpos()),
                            rod_pos=observation.rod.pos,
                            rod_quat=observation.rod.quat,
                            sponge_pos=tensor_to_array(demo.sponge.get_particles_pos()),
                            absorbed=is_absorbed,
                            wetness=wetness,
                        )
                    water_pos = water_positions - demo.config.assets.coffee_cup_pos
                    is_in_cup = demo.cup_cavity.contains(water_pos.reshape((-1, 3)))
                    writer.writerow(
                        motion_row
                        + (
                            tensor_to_array(demo.coffee.get_particles_active()).sum()
                            + tensor_to_array(demo.water.get_particles_active()).sum(),
                            mass,
                            concentrations.sum(),
                            concentrations.var(),
                            is_in_cup.mean(),
                            absorbed_particles,
                            absorbed_water_particles,
                            wetness.mean(),
                        )
                    )
                    metrics.flush()
            if steps * dt >= MOTION_END:
                if (
                    target.right_phase != Phase.REST
                    or target.left_phase != Phase.REST
                    or not demo.motion.has_caught_cup
                ):
                    raise RuntimeError("The manipulation sequence did not complete.")
                if abs(measured_stir_angle - 6.0 * math.pi) > 0.05:
                    raise RuntimeError(f"The rod completed {measured_stir_angle / (2.0 * math.pi):.6f} stirring turns.")
                if np.max(np.abs(observation.rod.pos - demo.config.assets.rod_park)) > 0.002:
                    raise RuntimeError(f"The rod missed its parking position: {observation.rod.pos}.")
                if demo.water is not None and absorbed_water_particles == 0:
                    raise RuntimeError("The wiping stroke absorbed no water; inspect the spill and sponge path.")
                if demo.water is not None and not 0 < observation.water_in_cup < demo.motion.water_before_reach:
                    raise RuntimeError("Cup recovery must spill some water and retain some in the cup.")
    finally:
        if demo.camera is not None:
            demo.camera.stop_recording()
    simulation_seconds = wall_time.perf_counter() - simulation_started
    print(f"coffee_water timing: simulation_seconds={simulation_seconds:.3f}", flush=True)
    print(f"coffee_water completed: steps={executed_steps}, simulated_time={demo.scene.cur_t:.6f}s")


if __name__ == "__main__":
    try:
        main()
    finally:
        gs.destroy()
