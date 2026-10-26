"""Simulate cutting and spreading without resetting or replacing material points."""

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import genesis as gs
from examples.butter_cut_spread.config import PHASES, Config, angular_velocities, motion
from examples.butter_cut_spread.contact import ToolContact
from examples.butter_cut_spread.cutting_scene import CuttingScene
from examples.butter_cut_spread.geometry import tool_mesh
from examples.butter_spreading.config import BreadParameters, ButterParameters, ContactParameters
from examples.butter_spreading.contact import ButterContact
from examples.butter_spreading.runtime import physics_only_visual_updates
from genesis.utils.misc import tensor_to_array


@dataclass(frozen=True)
class PhaseTime:
    name: str
    seconds: float


@dataclass(frozen=True)
class SimulationReport:
    is_complete: bool
    frames: int
    steps: int
    butter_particles: int
    bread_particles: int
    all_particles_active: bool
    initialization_seconds: float
    stepping_seconds: float
    elapsed_seconds: float
    max_contact_correction_m: float
    phases: list[PhaseTime]
    state_sha256: str
    scope: str


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="New output directory for this simulation.")
    parser.add_argument("--steps", type=int, help="Limit integration steps for a smoke check.")
    parser.add_argument("--material", type=Path, help="ButterParameters JSON; omitted fields retain material defaults.")
    parser.add_argument(
        "--config", type=Path, help="Partial JSON configuration; omitted fields use end-slice defaults."
    )
    args = parser.parse_args()
    config = Config() if args.config is None else Config(**json.loads(args.config.read_text()))
    material = (
        ButterParameters() if args.material is None else ButterParameters.model_validate_json(args.material.read_text())
    )
    steps = math.ceil(config.duration / config.dt) if args.steps is None else args.steps
    if not 0 < steps <= math.ceil(config.duration / config.dt):
        parser.error("Step count outside trajectory.")
    run = args.run.resolve()
    if run.exists() and any(run.iterdir()):
        parser.error("Use a new output directory.")
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(config), indent=2))
    (run / "material.json").write_text(material.model_dump_json(indent=2))
    snapshot = run / "source"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        (snapshot / path.name).write_bytes(path.read_bytes())
    started = time.perf_counter()
    # The complete scene integrates over one million material points for 230,000 steps.
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
    for center, size in zip(config.board_centers, config.board_sizes):
        scene.add_entity(
            morph=gs.morphs.Box(pos=center, size=size, fixed=True),
            material=gs.materials.Rigid(coup_friction=0.35, coup_softness=0.0),
        )
    bread = scene.add_entity(
        morph=gs.morphs.Box(pos=config.bread_center, size=config.bread_size),
        material=gs.materials.MPM.PorousBread(
            **BreadParameters().model_dump(), sampler="regular", particle_size=config.bread_particle_size
        ),
    )
    butter = scene.add_entity(
        morph=gs.morphs.Box(pos=config.butter_center, size=config.butter_size),
        material=gs.materials.MPM.HerschelBulkleyButter(
            **material.model_dump(), dt=config.dt, sampler="regular", particle_size=config.particle_size
        ),
    )
    times = np.arange(steps + 1) * config.dt
    positions, angles, quaternions = motion(times, config)
    velocities = np.diff(positions, axis=0) / config.dt
    joint_angular, angular = angular_velocities(quaternions, config.dt)
    blade_path = run / "blade.obj"
    tool_mesh(config).export(blade_path)
    blade = scene.add_entity(
        morph=gs.morphs.Mesh(
            pos=positions[0],
            quat=quaternions[0],
            file=str(blade_path),
            decimate=False,
            convexify=False,
            watertighten=None,
            align=False,
        ),
        material=gs.materials.Rigid(
            rho=7800.0,
            coup_friction=0.02,
            coup_softness=0.0,
            is_coup_reaction_enabled=False,
            gravity_compensation=1.0,
            sdf_cell_size=0.0002,
        ),
    )
    scene.build(cutting_tools=(blade,))
    for entity, spacing, board_index in ((bread, config.bread_particle_size, 1), (butter, config.particle_size, 0)):
        initial = entity.get_particles_pos()
        support_top = config.board_centers[board_index][2] + config.board_sizes[board_index][2] / 2
        initial[:, 2] += support_top + spacing / 2 - initial[:, 2].min()
        entity.set_particles_pos(initial)
    initial = butter.get_particles_pos()
    jitter = np.random.default_rng(config.seed).uniform(-0.1, 0.1, size=tuple(initial.shape)) * config.particle_size
    butter.set_particles_pos(initial + gs.tensor(jitter))
    wet = ContactParameters(blade_stress=0, blade_shear_stress=0)
    bread_contact = ButterContact(
        bread,
        butter,
        blade,
        spacing=config.particle_size,
        dt=config.dt,
        blade_size=config.blade_size,
        lower_bound=config.lower_bound,
        upper_bound=config.upper_bound,
        **wet.model_dump(exclude={"blade_normal_time"}),
        blade_normal_relaxation=0.0,
    )
    contact = ToolContact(butter, config, bread_contact=bread_contact)
    target_positions, target_quats = gs.tensor(positions), gs.tensor(quaternions)
    target_velocity = gs.tensor(np.column_stack((velocities, joint_angular)))
    centers, speeds, thetas, omegas = positions.tolist(), velocities.tolist(), angles.tolist(), angular.tolist()
    samples = sorted(
        {0, steps, *(round(i / config.fps / config.dt) for i in range(1, int(times[-1] * config.fps) + 1))}
    )
    frames, bread_frames, contacts = (
        [tensor_to_array(butter.get_particles_pos())],
        [tensor_to_array(bread.get_particles_pos())],
        [contact.faces()],
    )
    knife_frames, quat_frames = [positions[0]], [quaternions[0]]
    init_seconds = time.perf_counter() - started
    print(f"Built butter={butter.n_particles}, bread={bread.n_particles}, init={init_seconds:.2f}s", flush=True)
    phase_times = []
    phase_started = step_started = time.perf_counter()
    phase_index = 0
    with physics_only_visual_updates(scene):
        for i in range(steps):
            blade.set_pos(target_positions[i], zero_velocity=False, relative=False)
            blade.set_quat(target_quats[i], zero_velocity=False, relative=False)
            blade.set_dofs_velocity(target_velocity[i])
            bread_contact.apply(center=centers[i], velocity=speeds[i])
            contact.apply(centers[i], speeds[i], thetas[i], omegas[i], float(times[i]))
            scene.step(update_visualizer=False)
            contact.apply(centers[i + 1], speeds[i], thetas[i + 1], omegas[i], float(times[i + 1]), project=True)
            if i + 1 == samples[len(frames)]:
                xyz = tensor_to_array(butter.get_particles_pos())
                bread_xyz = tensor_to_array(bread.get_particles_pos())
                if not np.isfinite(xyz).all() or not np.isfinite(bread_xyz).all():
                    raise RuntimeError(f"Nonfinite particles at {i + 1}")
                frames.append(xyz)
                bread_frames.append(bread_xyz)
                contacts.append(contact.faces())
                knife_frames.append(tensor_to_array(blade.get_pos(relative=False)))
                quat_frames.append(tensor_to_array(blade.get_quat(relative=False)))
            if phase_index < len(PHASES) and times[i + 1] >= PHASES[phase_index][1]:
                gs.qd.sync()
                now = time.perf_counter()
                phase_times.append(PhaseTime(PHASES[phase_index][0], now - phase_started))
                np.savez_compressed(
                    run / f"checkpoint-{PHASES[phase_index][0]}.npz",
                    particles=tensor_to_array(butter.get_particles_pos()),
                    knife_position=tensor_to_array(blade.get_pos(relative=False)),
                    knife_quaternion=tensor_to_array(blade.get_quat(relative=False)),
                    time_s=times[i + 1],
                )
                phase_started = time.perf_counter()
                phase_index += 1
            if (i + 1) % 5000 == 0 or i + 1 == steps:
                print(
                    f"Step {i + 1}/{steps} t={times[i + 1]:.3f} elapsed={time.perf_counter() - step_started:.1f}s",
                    flush=True,
                )
    gs.qd.sync()
    stepping = time.perf_counter() - step_started
    active = bool(
        tensor_to_array(butter.get_particles_active()).all() and tensor_to_array(bread.get_particles_active()).all()
    )
    np.savez_compressed(
        run / "state.npz",
        particles=np.stack(frames),
        bread_particles=np.stack(bread_frames),
        knife_positions=np.stack(knife_frames),
        knife_quaternions=np.stack(quat_frames),
        tool_faces=np.stack(contacts),
        time_s=times[samples],
        particle_size=config.particle_size,
        bread_particle_size=config.bread_particle_size,
        knife_size=config.blade_size,
        fps=config.fps,
    )
    report = SimulationReport(
        steps == math.ceil(config.duration / config.dt),
        len(frames),
        steps,
        butter.n_particles,
        bread.n_particles,
        active,
        init_seconds,
        stepping,
        time.perf_counter() - started,
        contact.max_correction(),
        phase_times,
        hashlib.sha256((run / "state.npz").read_bytes()).hexdigest(),
        "One Genesis CPIC MPM scene and persistent particle IDs throughout. All butter remains active in MPM. "
        "Finite cohesive tractions and unilateral blade projection; no trajectory reset or replacement butter.",
    )
    (run / "simulation.json").write_text(json.dumps(asdict(report), indent=2))
    print(json.dumps(asdict(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
