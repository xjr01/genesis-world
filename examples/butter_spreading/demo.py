"""Calibrated MPM butter press/spread/lift demo.

Run from the repository root with:
    python -m examples.butter_spreading.demo --backend gpu --steps 1
Omit --steps for the full four-second trajectory (114,286 steps).
"""

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

PROCESS_STARTED = time.perf_counter()

import numpy as np
import torch
from pydantic import ValidationError

import genesis as gs
from examples.butter_spreading import motion
from examples.butter_spreading.config import MotionParameters, RunConfig, load_config
from examples.butter_spreading.contact import ButterContact
from examples.butter_spreading.runtime import physics_only_visual_updates
from genesis.utils.misc import tensor_to_array

DT = 3.5e-5
DEFAULT_CONFIG = RunConfig()

# Tool-frame API uses the undeformed bread top as z=0; simulation config specifies world coordinates.
knife_pose = partial(
    motion.knife_pose,
    motion=MotionParameters(
        start_clearance=0.0305,
        press_clearance=0.0185,
        sweep_clearance=0.0105,
        sweep_end_clearance=0.0088,
        lift_clearance=0.0315,
    ),
)


@dataclass(frozen=True)
class ButterScene:
    scene: gs.Scene
    bread: object
    butter: object
    blade: object
    blade_size: tuple[float, float, float]
    blade_xy_offset: tuple[float, float]
    plate_height: float
    task_origin: tuple[float, float, float]
    camera: object | None
    contact: ButterContact


@dataclass(frozen=True)
class KnifeMotion:
    positions: torch.Tensor
    velocities: torch.Tensor
    contact_centers: list[tuple[float, float, float]]
    contact_velocities: list[tuple[float, float, float]]


def prepare_knife_motion(steps, task_origin, blade_xy_offset, config: RunConfig = DEFAULT_CONFIG):
    """Upload the prescribed world-frame positions and finite-difference velocities once for the entire run."""
    positions = []
    velocities = []
    # Quantize the collider offset to simulation precision before adding it to the continuous trajectory.
    offset = np.array((*blade_xy_offset, 0.0), dtype=np.float32).tolist()
    translation = (task_origin[0], task_origin[1], task_origin[2] - 0.026)
    for step in range(steps):
        task_time = step * config.dt
        current_local = motion.knife_pose(task_time, config.motion)
        # Keep t + dt to preserve rounding of the finite-difference velocity
        next_local = motion.knife_pose(task_time + config.dt, config.motion)
        current = tuple(origin + value for origin, value in zip(translation, current_local))
        following = tuple(origin + value for origin, value in zip(translation, next_local))
        positions.append(tuple(value + delta for value, delta in zip(current, offset)))
        velocities.append(
            (*((next_value - value) / config.dt for value, next_value in zip(current, following)), 0.0, 0.0, 0.0)
        )
    return KnifeMotion(
        gs.tensor(positions).reshape(steps, 3),
        gs.tensor(velocities).reshape(steps, 6),
        positions,
        [velocity[:3] for velocity in velocities],
    )


@dataclass
class PhaseTiming:
    name: str
    start_step: int
    end_step: int
    seconds: float


@dataclass(frozen=True)
class ContactMetadata:
    range_m: float
    bread_range_m: float
    equilibrium_adhesion: bool


@dataclass(frozen=True)
class TrajectoryMetadata:
    genesis_source: str
    frames: int
    total_steps: int
    particle_spacing_m: float
    bread_particle_spacing_m: float
    butter_particle_count: int
    bread_particle_count: int
    knife_collider_size_m: tuple[float, float, float]
    knife_collider_xy_offset_m: tuple[float, float]
    contact_parameters: ContactMetadata
    solver: str = "Genesis MPM"
    solver_backend: str = "genesis"
    schema_version: int = 1
    dt_s: float = DT
    fps: int = 24
    grid_density: int = 512
    reconstruction_kernel_width_m: float = 1.0 / 768
    particle_constraints: bool = False
    material_point_deletion: bool = False
    butter_sampling: str = "stratified"
    enable_CPIC: bool = True
    fast_rigid: bool = True


@dataclass
class RunTiming:
    genesis_source: str
    backend: str
    steps: int
    simulated_seconds: float
    bread_particle_size: float
    butter_particle_size: float
    bread_particles: int
    butter_particles: int
    is_recording: bool
    is_viewer_enabled: bool
    initialization_seconds: float
    motion_preparation_seconds: float
    stepping_seconds: float
    recording_finalize_seconds: float
    validation_seconds: float
    state_export_seconds: float
    total_seconds: float
    milliseconds_per_step: float
    frame_readback_seconds: float
    trajectory_export_seconds: float
    phases: list[PhaseTiming]


@dataclass(frozen=True)
class RunCompletion:
    configuration_sha256: str
    source_sha256: dict[str, str]
    genesis_version: str
    quadrants_version: tuple[int, ...]
    torch_version: str
    numpy_version: str
    command: list[str]
    is_complete: bool = True


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
    bread_particle_size: float | None = None,
    butter_particle_size: float | None = None,
    config: RunConfig = DEFAULT_CONFIG,
) -> ButterScene:
    """Build the calibrated, self-contained butter-spreading MPM task."""
    bread_particle_size = config.bread_particle_size if bread_particle_size is None else bread_particle_size
    butter_particle_size = config.butter_particle_size if butter_particle_size is None else butter_particle_size
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
    if not all(math.isfinite(value) and value > 0.0 for value in (bread_particle_size, butter_particle_size)):
        raise ValueError("Particle spacings must be finite and positive.")
    lower_bound = (-0.12, -0.08, -0.02)
    upper_bound = (0.12, 0.08, 0.11)
    initial_local_blade = motion.knife_pose(0.0, config.motion)
    initial_blade = initial_local_blade
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=config.dt,
            substeps=1,
            gravity=(0.0, 0.0, -9.81),
        ),
        # The prescribed blade stays above the support; particle coupling remains active
        rigid_options=gs.options.RigidOptions(enable_collision=False, disable_constraint=True),
        mpm_options=gs.options.MPMOptions(
            dt=config.dt,
            gravity=(0.0, 0.0, -9.81),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            particle_size=bread_particle_size,
            grid_density=config.grid_density,
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
            **config.bread.model_dump(),
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
            **config.butter.model_dump(),
            dt=config.dt,
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
    if scene.sim.rigid_solver.n_entities != 2:
        raise ValueError("The collision-free rigid configuration requires only the support and the prescribed blade.")

    bread_pos = bread.get_particles_pos()
    bread_pos[:, 2] += plate_height + 0.5 * bread_particle_size - bread_pos[:, 2].min()
    bread.set_particles_pos(bread_pos)
    bread.set_particles_vel(gs.zeros_like(bread_pos))
    butter_pos = butter.get_particles_pos()
    material_gap = 0.5 * (bread_particle_size + butter_particle_size)
    butter_pos[:, 2] += bread.get_particles_pos()[:, 2].max() + material_gap - butter_pos[:, 2].min()
    generator = np.random.default_rng(config.seed)
    offsets = generator.uniform(-0.45, 0.45, size=tuple(butter_pos.shape)) * butter_particle_size
    butter_pos += gs.tensor(offsets.astype(np.float32))
    butter.set_particles_pos(butter_pos)
    butter.set_particles_vel(gs.zeros_like(butter_pos))

    contact = ButterContact(
        bread,
        butter,
        blade,
        spacing=butter_particle_size,
        dt=config.dt,
        blade_size=blade_size,
        lower_bound=(-0.25, -0.28, lower_bound[2]),
        upper_bound=(0.25, 0.28, upper_bound[2]),
        **config.contact.model_dump(exclude={"blade_normal_time"}),
        blade_normal_relaxation=config.dt / config.contact.blade_normal_time,
    )
    scene.reset(scene.get_state())

    return ButterScene(
        scene=scene,
        bread=bread,
        butter=butter,
        blade=blade,
        blade_size=blade_size,
        blade_xy_offset=blade_xy_offset,
        plate_height=plate_height,
        task_origin=task_origin,
        camera=camera,
        contact=contact,
    )


def main():
    parser = argparse.ArgumentParser(description="Self-contained Genesis MPM butter-spreading scenario")
    parser.add_argument("-s", "--steps", type=int, help="Number of steps; defaults to the full four-second trajectory.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Report every N steps; zero disables reports.")
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu", help="Simulation backend.")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the scene viewer.")
    parser.add_argument("-r", "--record", action="store_true", help="Record the configured camera to MP4.")
    parser.add_argument("--config", type=Path, help="Partial JSON physical configuration; omitted values use defaults.")
    parser.add_argument("--seed", type=int, help="Override the configuration's initial particle sampling seed.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("out/butter_spreading"),
        help="Timing, particle state and video output directory.",
    )
    parser.add_argument(
        "--bread-particle-size",
        type=float,
        help="Override bread spacing in metres; larger spacing reduces cost and spatial resolution.",
    )
    parser.add_argument(
        "--butter-particle-size",
        type=float,
        help="Override butter spacing in metres; larger spacing reduces cost and thin-layer resolution.",
    )
    parser.add_argument(
        "--save-state", action="store_true", help="Save initial/final particle states for offline inspection."
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Save all particles at 24 fps for surface and bread reconstruction.",
    )
    args = parser.parse_args()
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    try:
        config = load_config(
            args.config,
            seed=args.seed,
            bread_particle_size=args.bread_particle_size,
            butter_particle_size=args.butter_particle_size,
        )
    except (OSError, ValidationError) as error:
        parser.error(str(error))
    if args.steps is not None and args.steps > round(4.0 / config.dt):
        parser.error("--steps must fit within the four-second motion")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in ("run-config.json", "mpm-state.npz", "timing.json")):
        parser.error("Output already contains a run; choose a new output directory.")
    config_text = config.model_dump_json(indent=2) + "\n"
    (args.output_dir / "run-config.json").write_text(config_text, encoding="utf-8")

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu, precision="32", logging_level="warning")
    runtime = build_scene(
        show_viewer=args.vis,
        add_camera=args.record,
        config=config,
    )
    gs.qd.sync()
    build_seconds = time.perf_counter() - PROCESS_STARTED
    print(f"butter_spreading timing: build_seconds={build_seconds:.3f}", flush=True)
    print(
        f"butter_spreading particles: bread={runtime.bread.n_particles}, butter={runtime.butter.n_particles}, dt={config.dt}",
        flush=True,
    )
    steps = round(4.0 / config.dt) if args.steps is None else args.steps
    preparation_started = time.perf_counter()
    knife_motion = prepare_knife_motion(steps, runtime.task_origin, runtime.blade_xy_offset, config)
    # An edited trajectory must keep the blade's lower face above the rigid support
    if steps and knife_motion.positions[:, 2].min() - 0.5 * runtime.blade_size[2] <= runtime.plate_height:
        raise ValueError("Blade intersects the support; rigid collision must be enabled for this trajectory.")
    gs.qd.sync()
    preparation_seconds = time.perf_counter() - preparation_started
    initial_bread = runtime.bread.get_particles_pos().clone() if args.save_state else None
    initial_butter = runtime.butter.get_particles_pos().clone() if args.save_state else None
    if runtime.camera is not None:
        runtime.camera.start_recording(save_to_filename=str(args.output_dir / "butter-spreading.mp4"), fps=30)
    gs.qd.sync()
    recording_finalize_seconds = 0.0
    frame_readback_seconds = 0.0
    frame_steps = sorted(
        {0, steps, *(round(index / 24 / config.dt) for index in range(97) if round(index / 24 / config.dt) <= steps)}
    )
    butter_frames = []
    velocity_frames = []
    bread_frames = []
    if args.save_trajectory:
        readback_started = time.perf_counter()
        butter_frames.append(tensor_to_array(runtime.butter.get_particles_pos()))
        velocity_frames.append(tensor_to_array(runtime.butter.get_particles_vel()))
        bread_frames.append(tensor_to_array(runtime.bread.get_particles_pos()))
        frame_readback_seconds += time.perf_counter() - readback_started
    phase_ends = [
        (name, min(steps, math.ceil(end / config.dt)))
        for name, end in (("press", 0.9), ("accelerate", 1.65), ("spread", 3.15), ("decelerate", 3.5), ("lift", 4.0))
    ]
    phases = []
    phase_start_step = 0
    try:
        with physics_only_visual_updates(runtime.scene):
            simulation_started = time.perf_counter()
            phase_started = simulation_started
            for step in range(steps):
                runtime.blade.set_pos(
                    knife_motion.positions[step], zero_velocity=False, relative=False, skip_forward=True
                )
                runtime.blade.set_dofs_velocity(knife_motion.velocities[step])
                runtime.contact.apply(
                    center=knife_motion.contact_centers[step], velocity=knife_motion.contact_velocities[step]
                )
                # Prepare display buffers only when a new video frame is due.
                update_visualizer = (args.vis or args.record) and (
                    int((step + 1) * config.dt * 30) > int(step * config.dt * 30) or step + 1 == steps
                )
                runtime.scene.step(update_visualizer=update_visualizer)
                if args.save_trajectory and step + 1 == frame_steps[len(butter_frames)]:
                    readback_started = time.perf_counter()
                    butter_frames.append(tensor_to_array(runtime.butter.get_particles_pos()))
                    velocity_frames.append(tensor_to_array(runtime.butter.get_particles_vel()))
                    bread_frames.append(tensor_to_array(runtime.bread.get_particles_pos()))
                    frame_readback_seconds += time.perf_counter() - readback_started
                if len(phases) < len(phase_ends) and step + 1 == phase_ends[len(phases)][1]:
                    gs.qd.sync()
                    now = time.perf_counter()
                    phases.append(
                        PhaseTiming(phase_ends[len(phases)][0], phase_start_step, step + 1, now - phase_started)
                    )
                    phase_started = now
                    phase_start_step = step + 1
                if args.progress_every and ((step + 1) % args.progress_every == 0 or step + 1 == steps):
                    print(
                        f"butter_spreading progress: step={step + 1}/{steps}, simulated_time={runtime.scene.cur_t:.6f}s, "
                        f"steps_per_second={(step + 1) / (time.perf_counter() - simulation_started):.2f}",
                        flush=True,
                    )
            gs.qd.sync()
            simulation_seconds = time.perf_counter() - simulation_started
    finally:
        if runtime.camera is not None:
            finalize_started = time.perf_counter()
            runtime.camera.stop_recording()
            recording_finalize_seconds = time.perf_counter() - finalize_started
    validation_started = time.perf_counter()
    for name, entity in (("bread", runtime.bread), ("butter", runtime.butter)):
        if not torch.isfinite(entity.get_particles_pos()).all():
            raise RuntimeError(f"Butter-spreading {name} positions contain non-finite values.")
        if not torch.isfinite(entity.get_particles_vel()).all():
            raise RuntimeError(f"Butter-spreading {name} velocities contain non-finite values.")
    gs.qd.sync()
    validation_seconds = time.perf_counter() - validation_started
    export_started = time.perf_counter()
    if args.save_state:
        np.savez(
            args.output_dir / "particle-state.npz",
            bread_initial=tensor_to_array(initial_bread),
            butter_initial=tensor_to_array(initial_butter),
            bread_position=tensor_to_array(runtime.bread.get_particles_pos()),
            butter_position=tensor_to_array(runtime.butter.get_particles_pos()),
            bread_velocity=tensor_to_array(runtime.bread.get_particles_vel()),
            butter_velocity=tensor_to_array(runtime.butter.get_particles_vel()),
        )
    export_seconds = time.perf_counter() - export_started
    trajectory_export_started = time.perf_counter()
    if args.save_trajectory:
        particles = np.stack(butter_frames)
        velocities = np.stack(velocity_frames)
        bread_particles = np.stack(bread_frames)
        if not all(np.isfinite(values).all() for values in (particles, velocities, bread_particles)):
            raise RuntimeError("Non-finite particle trajectory.")
        sample_times = np.array(frame_steps) * config.dt
        np.savez_compressed(
            args.output_dir / "mpm-state.npz",
            particles=particles,
            particle_velocities=velocities,
            bread_particles=bread_particles,
            knife_positions=np.array([motion.knife_pose(sample_time, config.motion) for sample_time in sample_times]),
            time_s=sample_times,
            fps=24,
            bread_center=np.array((0.0, 0.0, 0.018)),
            bread_size=np.array((0.14, 0.114, 0.016)),
            knife_size=np.array(runtime.blade_size),
        )
        metadata = TrajectoryMetadata(
            genesis_source=str(Path(gs.__file__).resolve()),
            frames=len(frame_steps),
            total_steps=steps,
            particle_spacing_m=config.butter_particle_size,
            bread_particle_spacing_m=config.bread_particle_size,
            butter_particle_count=runtime.butter.n_particles,
            bread_particle_count=runtime.bread.n_particles,
            knife_collider_size_m=runtime.blade_size,
            knife_collider_xy_offset_m=runtime.blade_xy_offset,
            contact_parameters=ContactMetadata(
                config.contact.blade_contact_range,
                config.contact.bread_contact_range,
                config.contact.is_equilibrium_adhesion,
            ),
            dt_s=config.dt,
            grid_density=config.grid_density,
        )
        (args.output_dir / "mpm-state.json").write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")
    trajectory_export_seconds = time.perf_counter() - trajectory_export_started
    timing = RunTiming(
        genesis_source=str(Path(gs.__file__).resolve()),
        backend=gs.backend.name,
        steps=steps,
        simulated_seconds=runtime.scene.cur_t,
        bread_particle_size=config.bread_particle_size,
        butter_particle_size=config.butter_particle_size,
        bread_particles=runtime.bread.n_particles,
        butter_particles=runtime.butter.n_particles,
        is_recording=args.record,
        is_viewer_enabled=args.vis,
        initialization_seconds=build_seconds,
        motion_preparation_seconds=preparation_seconds,
        stepping_seconds=simulation_seconds,
        recording_finalize_seconds=recording_finalize_seconds,
        validation_seconds=validation_seconds,
        state_export_seconds=export_seconds,
        total_seconds=time.perf_counter() - PROCESS_STARTED,
        milliseconds_per_step=1000.0 * simulation_seconds / steps if steps else 0.0,
        frame_readback_seconds=frame_readback_seconds,
        trajectory_export_seconds=trajectory_export_seconds,
        phases=phases,
    )
    (args.output_dir / "timing.json").write_text(json.dumps(asdict(timing), indent=2) + "\n", encoding="utf-8")
    source_files = (
        Path(__file__),
        Path(__file__).with_name("contact.py"),
        Path(__file__).with_name("config.py"),
        Path(__file__).with_name("motion.py"),
        Path(__file__).with_name("runtime.py"),
        Path(gs.__file__).parent / "engine/materials/MPM/butter.py",
        Path(gs.__file__).parent / "vis/visualizer.py",
    )
    completion = RunCompletion(
        configuration_sha256=hashlib.sha256((args.output_dir / "run-config.json").read_bytes()).hexdigest(),
        source_sha256={
            path.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in source_files
        },
        genesis_version=gs.__version__,
        quadrants_version=gs.qd.__version__,
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        command=sys.argv,
    )
    (args.output_dir / "completion.json").write_text(json.dumps(asdict(completion), indent=2) + "\n", encoding="utf-8")
    print(f"butter_spreading timing: simulation_seconds={simulation_seconds:.3f}", flush=True)
    print(
        f"butter_spreading timing: total_seconds={timing.total_seconds:.3f}, ms_per_step={timing.milliseconds_per_step:.3f}"
    )
    print(f"butter_spreading completed: steps={steps}, simulated_time={runtime.scene.cur_t:.6f}s")


if __name__ == "__main__":
    main()
