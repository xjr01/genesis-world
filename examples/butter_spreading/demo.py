"""Calibrated MPM butter press/spread/lift demo.

Run from the repository root with:
    python -m examples.butter_spreading.demo --backend gpu --steps 1
Omit --steps for the full four-second trajectory (114,286 steps).
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import genesis as gs
from examples.butter_spreading.contact import ButterContact

DT = 3.5e-5


def _ease(value: float) -> float:
    return value * value * value * (10.0 + value * (-15.0 + 6.0 * value))


def _ease_integral(value: float) -> float:
    return value**4 * (2.5 + value * (-3.0 + value))


def knife_pose(time_s: float) -> tuple[float, float, float]:
    """Evaluate the continuous level-blade trajectory in metres."""
    acceleration_end_time = 1.65
    deceleration_start_time = 3.15
    lift_clearance = 0.0315
    lift_end_time = 4.0
    press_clearance = 0.0185
    press_end_time = 0.9
    press_end_x = -0.055
    spread_end_time = 3.5
    spread_end_x = 0.049
    spread_speed = (spread_end_x - press_end_x) / (
        deceleration_start_time
        - acceleration_end_time
        + 0.5 * (acceleration_end_time - press_end_time)
        + 0.5 * (spread_end_time - deceleration_start_time)
    )
    start_clearance = 0.0305
    sweep_clearance = 0.0105
    sweep_end_clearance = 0.0088
    time_s = min(max(time_s, 0.0), lift_end_time)
    if time_s < press_end_time:
        phase = time_s / press_end_time
        x_pos = press_end_x
        z_pos = start_clearance + (press_clearance - start_clearance) * _ease(phase)
    elif time_s < acceleration_end_time:
        duration = acceleration_end_time - press_end_time
        phase = (time_s - press_end_time) / duration
        x_pos = press_end_x + spread_speed * duration * _ease_integral(phase)
        z_pos = press_clearance + (sweep_clearance - press_clearance) * _ease(phase)
    elif time_s < deceleration_start_time:
        x_pos = (
            press_end_x
            + 0.5 * spread_speed * (acceleration_end_time - press_end_time)
            + spread_speed * (time_s - acceleration_end_time)
        )
        phase = (time_s - acceleration_end_time) / (spread_end_time - acceleration_end_time)
        z_pos = sweep_clearance + (sweep_end_clearance - sweep_clearance) * _ease(phase)
    elif time_s < spread_end_time:
        duration = spread_end_time - deceleration_start_time
        phase = (time_s - deceleration_start_time) / duration
        x_pos = (
            press_end_x
            + 0.5 * spread_speed * (acceleration_end_time - press_end_time)
            + spread_speed * (deceleration_start_time - acceleration_end_time)
            + spread_speed * duration * (phase - _ease_integral(phase))
        )
        sweep_phase = (time_s - acceleration_end_time) / (spread_end_time - acceleration_end_time)
        z_pos = sweep_clearance + (sweep_end_clearance - sweep_clearance) * _ease(sweep_phase)
    else:
        phase = (time_s - spread_end_time) / (lift_end_time - spread_end_time)
        x_pos = spread_end_x
        z_pos = sweep_end_clearance + (lift_clearance - sweep_end_clearance) * _ease(phase)
    return x_pos, 0.0, z_pos


@dataclass(frozen=True)
class ButterScene:
    scene: gs.Scene
    bread: object
    butter: object
    blade: object
    blade_xy_offset: tuple[float, float]
    task_origin: tuple[float, float, float]
    camera: object | None


def _rigid_material(*, is_blade: bool):
    return gs.materials.Rigid(
        rho=7800.0 if is_blade else 700.0,
        friction=0.65 if is_blade else 0.9,
        coup_friction=0.12 if is_blade else 1.0,
        coup_restitution=0.0,
        coup_softness=0.001,
        needs_coup=True,
        sdf_cell_size=0.001 if is_blade else 0.0015,
        sdf_min_res=32,
        sdf_max_res=128,
    )


def build_scene(
    *,
    show_viewer: bool = False,
    add_camera: bool = False,
) -> ButterScene:
    """Build the calibrated, self-contained butter-spreading MPM task."""
    plate_height = 0.010
    support_depth = 0.020
    support_size_xy = (0.18, 0.14)
    camera_pos = (0.22, -0.25, 0.18)
    camera_lookat = (0.0, 0.0, 0.025)
    bread_center = (0.0, 0.0, 0.018)
    bread_size = (0.14, 0.114, 0.016)
    task_origin = (bread_center[0], bread_center[1], bread_center[2] + 0.5 * bread_size[2])
    blade_size = (0.020, 0.095, 0.005)
    blade_xy_offset = (0.0021, 0.0035)
    bread_particle_size = 0.001
    butter_particle_size = 0.0005
    lower_bound = (-0.12, -0.08, -0.02)
    upper_bound = (0.12, 0.08, 0.11)
    initial_local_blade = knife_pose(0.0)
    initial_blade = tuple(origin + value for origin, value in zip(task_origin, initial_local_blade))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=1,
            gravity=(0.0, 0.0, -9.81),
        ),
        mpm_options=gs.options.MPMOptions(
            dt=DT,
            gravity=(0.0, 0.0, -9.81),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            particle_size=bread_particle_size,
            grid_density=512,
            enable_CPIC=True,
        ),
        coupler_options=gs.options.LegacyCouplerOptions(
            rigid_mpm=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40.0,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            lower=(
                -0.5 * support_size_xy[0],
                -0.5 * support_size_xy[1],
                plate_height - support_depth,
            ),
            upper=(0.5 * support_size_xy[0], 0.5 * support_size_xy[1], plate_height),
            fixed=True,
        ),
        material=_rigid_material(is_blade=False),
    )
    bread = scene.add_entity(
        morph=gs.morphs.Box(
            pos=bread_center,
            size=bread_size,
        ),
        material=gs.materials.MPM.PorousBread(
            E=16000.0,
            nu=0.15,
            rho=280.0,
            compaction_yield_pressure=650.0,
            compaction_hardening=5200.0,
            densification_strain=0.32,
            densification_hardening=45000.0,
            min_plastic_volume_ratio=0.35,
            shear_yield_stress=700.0,
            shear_hardening=7000.0,
            sampler="regular",
            particle_size=bread_particle_size,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.72, 0.55, 0.35),
            roughness=0.9,
        ),
        vis_mode="particle",
    )
    blade = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(
                initial_blade[0] + blade_xy_offset[0],
                initial_blade[1] + blade_xy_offset[1],
                initial_blade[2],
            ),
            size=blade_size,
        ),
        material=_rigid_material(is_blade=True),
    )
    butter = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.04, 0.0, 0.035),
            size=(0.04, 0.03, 0.018),
        ),
        material=gs.materials.MPM.HerschelBulkleyButter(
            shear_modulus=20000.0,
            bulk_modulus=150000.0,
            yield_stress=80.0,
            consistency=25.0,
            flow_exponent=0.5,
            rho=900.0,
            dt=DT,
            sampler="regular",
            particle_size=butter_particle_size,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.97, 0.78, 0.30),
            roughness=0.3,
        ),
        vis_mode="particle",
    )
    camera = None
    if add_camera:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=40.0,
            GUI=False,
        )
    scene.build()

    bread_pos = bread.get_particles_pos()
    bread_pos[:, 2] += plate_height + 0.5 * bread_particle_size - bread_pos[:, 2].min()
    bread.set_particles_pos(bread_pos)
    bread.set_particles_vel(gs.zeros_like(bread_pos))
    butter_pos = butter.get_particles_pos()
    material_gap = 0.5 * (bread_particle_size + butter_particle_size)
    butter_pos[:, 2] += bread.get_particles_pos()[:, 2].max() + material_gap - butter_pos[:, 2].min()
    generator = torch.Generator(device=butter_pos.device)
    generator.manual_seed(17)
    offsets = torch.empty_like(butter_pos).uniform_(generator=generator)
    butter_pos += (offsets - 0.5) * 0.9 * butter_particle_size
    butter.set_particles_pos(butter_pos)
    butter.set_particles_vel(gs.zeros_like(butter_pos))

    contact = ButterContact(
        bread,
        butter,
        blade,
        spacing=butter_particle_size,
        dt=DT,
        blade_size=blade_size,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        bread_stress=450.0,
        blade_stress=100.0,
        bread_contact_range=0.0015,
        bread_slip_time=0.008,
        bread_shear_stress=650.0,
        blade_contact_range=0.002916666666666667,
        blade_slip_time=0.008,
        blade_shear_stress=150.0,
        blade_normal_relaxation=DT / 0.005,
        blade_max_separation_speed=0.35,
        blade_contact_margin=0.001875,
        is_equilibrium_adhesion=True,
    )
    scene.register_pre_step_callback(contact)
    scene.reset(scene.get_state())

    return ButterScene(
        scene=scene,
        bread=bread,
        butter=butter,
        blade=blade,
        blade_xy_offset=blade_xy_offset,
        task_origin=task_origin,
        camera=camera,
    )


def main():
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Self-contained Genesis MPM butter-spreading scenario")
    parser.add_argument("-s", "--steps", type=int, help="Number of steps; defaults to the full four-second trajectory.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Report every N steps; zero disables reports.")
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu", help="Simulation backend.")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the scene viewer.")
    parser.add_argument("-r", "--record", action="store_true", help="Record the configured camera to MP4.")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("out/butter_spreading"), help="Video output directory."
    )
    args = parser.parse_args()
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu, precision="32", logging_level="warning")
    runtime = build_scene(show_viewer=args.vis, add_camera=args.record)
    build_seconds = time.perf_counter() - process_started
    print(f"butter_spreading timing: build_seconds={build_seconds:.3f}", flush=True)
    print(
        f"butter_spreading particles: bread={runtime.bread.n_particles}, butter={runtime.butter.n_particles}, dt={DT}",
        flush=True,
    )
    steps = round(4.0 / DT) if args.steps is None else args.steps
    if runtime.camera is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        runtime.camera.start_recording(save_to_filename=str(args.output / "butter-spreading.mp4"), fps=30)
    try:
        simulation_started = time.perf_counter()
        for step in range(steps):
            task_time = step * DT
            current_local_pose = knife_pose(task_time)
            next_local_pose = knife_pose(task_time + DT)
            current_pose = tuple(origin + value for origin, value in zip(runtime.task_origin, current_local_pose))
            next_pose = tuple(origin + value for origin, value in zip(runtime.task_origin, next_local_pose))
            center = (
                current_pose[0] + runtime.blade_xy_offset[0],
                current_pose[1] + runtime.blade_xy_offset[1],
                current_pose[2],
            )
            velocity = tuple((next_value - value) / DT for value, next_value in zip(current_pose, next_pose))
            runtime.blade.set_pos(center, zero_velocity=False, relative=False, skip_forward=True)
            runtime.blade.set_dofs_velocity((*velocity, 0.0, 0.0, 0.0))
            # Prepare display buffers only when a new video frame is due.
            update_visualizer = int((step + 1) * DT * 30) > int(step * DT * 30) or step + 1 == steps
            runtime.scene.step(update_visualizer=update_visualizer)
            if args.progress_every and ((step + 1) % args.progress_every == 0 or step + 1 == steps):
                print(
                    f"butter_spreading progress: step={step + 1}/{steps}, simulated_time={runtime.scene.cur_t:.6f}s, "
                    f"steps_per_second={(step + 1) / (time.perf_counter() - simulation_started):.2f}",
                    flush=True,
                )
    finally:
        if runtime.camera is not None:
            runtime.camera.stop_recording()
    simulation_seconds = time.perf_counter() - simulation_started
    for name, entity in (("bread", runtime.bread), ("butter", runtime.butter)):
        if not torch.isfinite(entity.get_particles_pos()).all():
            raise RuntimeError(f"Butter-spreading {name} positions contain non-finite values.")
        if not torch.isfinite(entity.get_particles_vel()).all():
            raise RuntimeError(f"Butter-spreading {name} velocities contain non-finite values.")
    print(f"butter_spreading timing: simulation_seconds={simulation_seconds:.3f}", flush=True)
    print(f"butter_spreading completed: steps={steps}, simulated_time={runtime.scene.cur_t:.6f}s")


if __name__ == "__main__":
    main()
