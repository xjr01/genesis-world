"""Continue the measured end cut through transport and left-to-right spreading."""

import numpy as np
from scipy.spatial.transform import Rotation

from examples.butter_cut_spread.config import ease

PHASES = (
    ("settle", 0.6),
    ("cut_mid", 1.5),
    ("cut", 2.9),
    ("lift_start", 3.05),
    ("lift", 3.7),
    ("transfer", 5.2),
    ("approach", 6.0),
    ("spread", 7.5),
    ("withdraw", 8.1),
    ("hold", 8.4),
)


def continue_motion(times, config, assets, positions, angles, quaternions):
    """Turn the loaded broad face downward while the handle clears both supports."""
    ends = np.array((3.7, 5.2, 6.0, 7.5, 8.1, 8.4))
    start = positions[np.searchsorted(times, 3.7)]
    top = assets["bread_top_m"]
    pitch = np.deg2rad(config.spread_pitch_degrees)
    bottom_offset = config.blade_size[2] / 2 * np.cos(pitch) + config.blade_size[0] / 2 * abs(np.sin(pitch))
    x, y, _ = config.bread_center
    centers = np.array(
        (
            start,
            (x + config.spread_start_offset, y, 0.115),
            (x + config.spread_start_offset, y, top + bottom_offset + config.spread_start_gap),
            (x + config.spread_end_offset, y, top + bottom_offset + config.spread_end_gap),
            (x + config.spread_end_offset + 0.012, y, 0.105),
            (x + config.spread_end_offset + 0.012, y, 0.105),
        )
    )
    peel = pitch - np.deg2rad(20) * np.cos(np.deg2rad(config.spread_yaw_degrees))
    pitches = np.array((-np.pi / 2, 0, pitch, pitch, peel, peel))
    yaws = np.deg2rad((0, *(config.spread_yaw_degrees,) * 5))
    rolls = np.deg2rad((assets["cutting_roll_degrees"], 0, 0, 0, 0, 0))
    index = np.clip(np.searchsorted(ends, times, side="right") - 1, 0, len(ends) - 2)
    progress = ease((times - ends[index]) / (ends[index + 1] - ends[index]))
    active = times >= ends[0]
    positions[active] = (centers[index] + progress[:, None] * (centers[index + 1] - centers[index]))[active]
    theta = pitches[index] + progress * (pitches[index + 1] - pitches[index])
    yaw = yaws[index] + progress * (yaws[index + 1] - yaws[index])
    roll = rolls[index] + progress * (rolls[index + 1] - rolls[index])
    angles[active, 0] = theta[active]
    angles[active, 1] = yaw[active]
    rotations = (
        Rotation.from_euler("x", roll[:, None])
        * Rotation.from_euler("z", angles[:, 1:2])
        * Rotation.from_euler("y", angles[:, :1])
    )
    quaternions[active] = rotations.as_quat()[active][:, [3, 0, 1, 2]]
    roll[~active] = np.deg2rad(assets["cutting_roll_degrees"])
    return positions, angles, quaternions, roll
