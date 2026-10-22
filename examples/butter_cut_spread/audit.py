"""Check the continuous particle trajectory and its decoded diagnostic video."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np
from scipy.spatial.transform import Rotation

from examples.butter_cut_spread.config import Config, motion


@dataclass(frozen=True)
class ParticleMetrics:
    pickup_fraction: float
    coating_zone_deposited_fraction: float
    max_bread_displacement_m: float
    max_prescribed_pose_error_m: float
    max_particle_blade_penetration_m: float
    sampled_interior_side_switches: int


@dataclass(frozen=True)
class AuditReport:
    is_accepted: bool
    frames: int
    butter_particles: int
    bread_particles: int
    state_sha256: str
    metrics: ParticleMetrics
    scope: str


def measure(config, particles, bread, positions, quaternions, times):
    """Measure pickup, deposition, deformation, prescribed poses and flat-face crossings."""
    if not np.isfinite(particles).all() or not np.isfinite(bread).all():
        raise ValueError("Nonfinite particle state.")
    if particles.ndim != 3 or bread.ndim != 3 or len(particles) != len(times) or len(bread) != len(times):
        raise ValueError("Particle frame counts differ.")
    if times[0] != 0 or np.any(np.diff(times) <= 0) or abs(times[-1] - config.duration) > config.dt:
        raise ValueError("A complete chronological trajectory is required.")
    is_slice = particles[0, :, 0] > config.butter_center[0] + config.cut_offset + config.blade_size[2] / 2
    if not is_slice.any():
        raise ValueError("The configured cut produces no slice particles.")
    pickup = np.mean(particles[np.argmin(abs(times - 3.25)), is_slice, 2] > 0.035)
    final = particles[-1]
    top = config.bread_center[2] + config.bread_size[2] / 2
    is_deposited = np.all(abs(final[:, :2] - config.bread_center[:2]) < np.array(config.bread_size[:2]) / 2, axis=1)
    is_deposited &= (final[:, 2] > top - 0.004) & (final[:, 2] < top + 0.004)
    deposited = np.mean(is_deposited[is_slice])
    targets, _, target_quaternions = motion(times, config)
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
    target_rotations = Rotation.from_quat(target_quaternions[:, [1, 2, 3, 0]])
    pose_error = np.max(
        np.linalg.norm(positions - targets, axis=1)
        + (rotations.inv() * target_rotations).magnitude() * np.linalg.norm(config.blade_size) / 2
    )
    half = np.array(config.blade_size) / 2
    previous = None
    switches = 0
    penetration = 0.0
    deformation = 0.0
    for i, points in enumerate(particles):
        deformation = max(deformation, np.linalg.norm(bread[i] - bread[0], axis=1).max())
        local = rotations[i].inv().apply(points - positions[i])
        is_interior = (abs(local[:, 0]) < half[0] - 0.003) & (abs(local[:, 1]) < half[1] - 0.003)
        if is_interior.any():
            penetration = max(penetration, max(0.0, np.max(half[2] - abs(local[is_interior, 2]))))
        if previous is not None:
            is_interior &= (abs(previous[:, 0]) < half[0] - 0.003) & (abs(previous[:, 1]) < half[1] - 0.003)
            switches += np.count_nonzero(is_interior & (local[:, 2] * previous[:, 2] < 0))
        previous = local
    return ParticleMetrics(
        float(pickup), float(deposited), float(deformation), float(pose_error), float(penetration), int(switches)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed simulation and particle-video directory.")
    args = parser.parse_args()
    run = args.run.resolve()
    config = Config(**json.loads((run / "config.json").read_text()))
    simulation = json.loads((run / "simulation.json").read_text())
    video = json.loads((run / "particles-video.json").read_text())
    state_hash = hashlib.sha256((run / "state.npz").read_bytes()).hexdigest()
    if not simulation["is_complete"] or not simulation["all_particles_active"]:
        raise ValueError("Simulation is incomplete or particles were deactivated.")
    if simulation["state_sha256"] != state_hash or video["state_sha256"] != state_hash:
        raise ValueError("Simulation and video refer to different particle archives.")
    if hashlib.sha256((run / "cut-spread-particles.mp4").read_bytes()).hexdigest() != video["video_sha256"]:
        raise ValueError("Particle video changed after encoding.")
    with np.load(run / "state.npz") as archive:
        particles, bread, positions, quaternions, times = (
            archive[key] for key in ("particles", "bread_particles", "knife_positions", "knife_quaternions", "time_s")
        )
    metrics = measure(config, particles, bread, positions, quaternions, times)
    frame_hashes = set()
    with av.open(str(run / "cut-spread-particles.mp4")) as movie:
        if float(movie.streams.video[0].average_rate) != config.fps:
            raise ValueError("Video frame rate differs from the simulation sampling rate.")
        count = 0
        for frame in movie.decode(video=0):
            pixels = frame.to_ndarray(format="rgb24")
            if pixels.shape != (480, 1280, 3) or pixels.max() == pixels.min():
                raise ValueError("Invalid particle-video frame.")
            frame_hashes.add(hashlib.sha256(pixels.tobytes()).digest())
            count += 1
    if count != len(times) or video["frames"] != count or len(frame_hashes) < count / 2:
        raise ValueError("Particle video is incomplete or frozen.")
    is_accepted = bool(
        metrics.pickup_fraction > 0.95
        and metrics.coating_zone_deposited_fraction > 0.5
        and metrics.max_prescribed_pose_error_m < config.particle_size / 20
        and metrics.max_particle_blade_penetration_m < 1e-7
        and metrics.sampled_interior_side_switches == 0
        and simulation["max_contact_correction_m"] < config.particle_size / 4
    )
    report = AuditReport(
        is_accepted,
        count,
        particles.shape[1],
        bread.shape[1],
        state_hash,
        metrics,
        "Particle-center checks at recorded frames. Deposition is within the bread footprint and +/-4 mm of its "
        "nominal top. Interior crossings exclude edge flow; this is not a continuous-time collision proof. "
        "No reconstructed-surface or photorealistic-rendering claims are made.",
    )
    (run / "audit.json").write_text(json.dumps(asdict(report), indent=2))
    print(json.dumps(asdict(report), indent=2), flush=True)
    if not is_accepted:
        raise RuntimeError("The particle trajectory failed physical acceptance; inspect audit.json.")


if __name__ == "__main__":
    main()
