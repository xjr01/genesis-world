import json
from typing import NamedTuple

import numpy as np

from genesis.utils.misc import tensor_to_array


class CheckpointInfo(NamedTuple):
    version: int
    method: str
    scale: int
    dt: float
    particles: int
    steps: int
    checkpoint_every: int
    fps: float
    playback_speed: float


def read_checkpoint_info(directory, method):
    with (directory / method / "metadata.json").open(encoding="utf-8") as source:
        info = CheckpointInfo(**json.load(source))
    if info.version != 1 or info.method != method:
        raise ValueError(f"Unsupported checkpoint metadata for {method}.")
    return info


def prepare_teapot_run(args, method):
    """Resolve replay defaults from the saved run and validate checkpoint arguments."""
    info = None
    if args.checkpoint_every is not None and args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive.")
    if args.replay_time is not None and (args.replay_checkpoints is None or args.replay_time < 0.0):
        raise ValueError("--replay-time requires --replay-checkpoints and a non-negative time.")
    if args.replay_checkpoints is not None:
        info = read_checkpoint_info(args.replay_checkpoints, method)
        if args.scale is not None and args.scale != info.scale:
            raise ValueError("Replay particle scale must match the saved checkpoint.")
        if args.dt is not None and args.dt != info.dt:
            raise ValueError("Replay time step must match the saved checkpoint.")
        args.scale = info.scale
        args.dt = info.dt
        args.fps = info.fps if args.fps is None else args.fps
        args.playback_speed = info.playback_speed if args.playback_speed is None else args.playback_speed
        if args.replay_time is None and not args.vis:
            args.record = True
    if args.save_checkpoints is not None:
        directory = args.save_checkpoints / method
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Checkpoint directory must be empty: {directory}")
    args.fps = 25.0 if args.fps is None else args.fps
    args.playback_speed = 1.0 if args.playback_speed is None else args.playback_speed
    return info


def save_teapot_checkpoint(simulation, directory, step):
    """Atomically save particle state and teapot pose for rendering at this simulation step."""
    positions = tensor_to_array(simulation.liquid.get_particles_pos())
    velocities = tensor_to_array(simulation.liquid.get_particles_vel())
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise FloatingPointError(f"Cannot save non-finite particle state at step {step}.")
    path = directory / f"frame_{step:08d}.npz"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output,
            step=step,
            positions=positions,
            velocities=velocities,
            active=tensor_to_array(simulation.liquid.get_particles_active()),
            teapot_pos=tensor_to_array(simulation.teapot.get_pos()),
            teapot_quat=tensor_to_array(simulation.teapot.get_quat()),
        )
    temporary.replace(path)


def load_teapot_checkpoint(simulation, path):
    """Restore a recorded frame through entity setters for rendering with the current appearance."""
    with np.load(path, allow_pickle=False) as checkpoint:
        if checkpoint["positions"].shape != (simulation.liquid.n_particles, 3):
            raise ValueError(f"Checkpoint particle count differs from the scene: {path}")
        for field in ("positions", "velocities", "teapot_pos", "teapot_quat"):
            if not np.isfinite(checkpoint[field]).all():
                raise FloatingPointError(f"Checkpoint contains non-finite {field}: {path}")
        simulation.liquid.set_particles_pos(checkpoint["positions"])
        simulation.liquid.set_particles_vel(checkpoint["velocities"])
        simulation.liquid.set_particles_active(checkpoint["active"])
        simulation.teapot.set_pos(checkpoint["teapot_pos"])
        simulation.teapot.set_quat(checkpoint["teapot_quat"])
        return checkpoint["step"].item()
