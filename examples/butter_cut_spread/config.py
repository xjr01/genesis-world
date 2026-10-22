"""Continuous two-board cutting, transport, and spreading scenario."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Config:
    dt: float = 2.5e-5
    duration: float = 5.75
    fps: int = 24
    seed: int = 17
    particle_size: float = 0.0004
    bread_particle_size: float = 0.0008
    grid_density: int = 768
    blade_size: tuple[float, float, float] = (0.020, 0.048, 0.0008)
    blade_bevel_height: float = 0.0015
    butter_size: tuple[float, float, float] = (0.080, 0.032, 0.018)
    butter_center: tuple[float, float, float] = (-0.115, 0.0, 0.009)
    bread_size: tuple[float, float, float] = (0.110, 0.105, 0.014)
    bread_center: tuple[float, float, float] = (0.055, 0.0, 0.007)
    board_centers: tuple[tuple[float, float, float], ...] = ((-0.115, 0, -0.005), (0.055, 0, -0.005))
    board_sizes: tuple[tuple[float, float, float], ...] = ((0.11, 0.14, 0.01), (0.17, 0.14, 0.01))
    lower_bound: tuple[float, float, float] = (-0.18, -0.08, -0.015)
    upper_bound: tuple[float, float, float] = (0.15, 0.08, 0.115)
    cut_offset: float = 0.0354
    normal_stress: float = 450.0
    shear_stress: float = 1000.0
    spread_normal_stress: float = 100.0
    spread_shear_stress: float = 150.0
    spread_adhesion_range: float = 0.0008
    adhesion_range: float = 0.0008
    board_normal_stress: float = 40.0
    board_shear_stress: float = 80.0
    spread_start_offset: float = -0.037
    spread_end_offset: float = -0.0196
    spread_start_gap: float = 0.0028
    spread_end_gap: float = 0.00254
    spread_pitch_degrees: float = 0.0


PHASES = (
    ("settle", 0.15),
    ("cut", 0.95),
    ("separate", 1.35),
    ("lift", 1.95),
    ("transfer", 3.25),
    ("approach", 4.0),
    ("spread", 5.0),
    ("withdraw", 5.75),
)


def ease(t):
    t = np.clip(t, 0, 1)
    return t**3 * (10 + t * (-15 + 6 * t))


def stroke_progress(t):
    """Accelerate into a constant-speed stroke; deceleration occurs during peeling."""
    t = np.clip(t, 0, 1)
    ramp = 0.3375
    s = t / ramp
    distance = ramp * (s**3 - 0.5 * s**4) / (1 - ramp / 2)
    return np.where(t < ramp, distance, (t - ramp / 2) / (1 - ramp / 2))


def motion(times, config):
    """Prescribe one rigid pose; the loaded negative-local-Z face turns downward."""
    cut_x = config.butter_center[0] + config.cut_offset
    x, y, z = config.bread_center
    top = z + config.bread_size[2] / 2
    ends = np.array([0, *(end for _, end in PHASES)])
    centers = np.array(
        [
            [cut_x, 0, 0.030],
            [cut_x, 0, 0.030],
            [cut_x, 0, 0.010],
            [cut_x + 0.0015, 0, 0.014],
            [cut_x + 0.003, 0, 0.065],
            [x + config.spread_start_offset, y, 0.060],
            [x + config.spread_start_offset, y, top + config.blade_size[2] / 2 + config.spread_start_gap],
            [x + config.spread_end_offset, y, top + config.blade_size[2] / 2 + config.spread_end_gap],
            [x + config.spread_end_offset + 0.008, y, 0.060],
        ]
    )
    pitch = np.deg2rad(config.spread_pitch_degrees)
    angles = np.array([-np.pi / 2] * 6 + [pitch] * 2 + [pitch - 0.25])
    index = np.clip(np.searchsorted(ends, times, side="right") - 1, 0, len(ends) - 2)
    u = ease((times - ends[index]) / (ends[index + 1] - ends[index]))
    positions = centers[index] + u[:, None] * (centers[index + 1] - centers[index])
    is_spreading = (times >= 4.0) & (times <= 5.0)
    travel = stroke_progress(times[is_spreading] - 4.0)
    positions[is_spreading] = centers[6] + travel[:, None] * (centers[7] - centers[6])
    theta = angles[index] + u * (angles[index + 1] - angles[index])
    is_withdrawing = times > 5.0
    peel_time = np.clip((times[is_withdrawing] - 5.0) / 0.75, 0, 1)
    # Quintic Hermite interpolation preserves the incoming velocity and zero acceleration.
    incoming = (centers[7] - centers[6]) / (1 - 0.3375 / 2)
    velocity_basis = peel_time - 6 * peel_time**3 + 8 * peel_time**4 - 3 * peel_time**5
    positions[is_withdrawing] = (
        centers[7] + ease(peel_time)[:, None] * (centers[8] - centers[7]) + 0.75 * velocity_basis[:, None] * incoming
    )
    theta[is_withdrawing] = pitch - 0.25 * ease((times[is_withdrawing] - 5.0) / 0.25)
    yaw = np.zeros_like(times)
    ct, st, cy, sy = np.cos(theta / 2), np.sin(theta / 2), np.cos(yaw / 2), np.sin(yaw / 2)
    quaternion = np.column_stack((cy * ct, -sy * st, cy * st, sy * ct))
    return positions, np.column_stack((theta, yaw)), quaternion


def rotation(angles):
    """Local blade frame under pitch then world-Z yaw."""
    theta, yaw = angles
    c, s, cy, sy = np.cos(theta), np.sin(theta), np.cos(yaw), np.sin(yaw)
    return np.array(((cy * c, -sy, cy * s), (sy * c, cy, sy * s), (-s, 0, c)))


def angular_velocities(quaternions, dt):
    """Return body-frame joint rates and world-frame contact rates for each finite pose increment."""
    orientations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
    local = (orientations[:-1].inv() * orientations[1:]).as_rotvec() / dt
    return local, orientations[:-1].apply(local)
