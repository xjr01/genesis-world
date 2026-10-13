import math
import os
from typing import NamedTuple

import numpy as np

import genesis as gs
import genesis.utils.geom as geom_utils
import genesis.utils.mesh as mesh_utils
import genesis.utils.particle as particle_utils


class TeapotSettings(NamedTuple):
    asset: str
    entity_name: str
    mesh_scale: float
    offset: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    turning_axis_pos: tuple[float, float, float]
    particles_seed: tuple[float, float, float]
    particles_max_height: float
    particles_vel: tuple[float, float, float]
    scale: int
    dt: float
    gravity: tuple[float, float, float]
    lower_bound: tuple[float, float, float]
    upper_bound: tuple[float, float, float]
    camera_pos: tuple[float, float, float]
    camera_lookat: tuple[float, float, float]

    @property
    def particle_size(self):
        return 2.0 / self.scale


class TeapotPose(NamedTuple):
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    vel: tuple[float, float, float]
    ang: tuple[float, float, float]


def create_teapot_settings(scale=None, dt=None):
    scale = 20 if scale is None else scale
    dt = 0.01 if dt is None else dt
    if scale < 20:
        raise ValueError("The teapot requires scale >= 20.")
    if dt <= 0.0:
        raise ValueError("The time step must be positive.")
    mesh_scale = 2.25
    offset = (0.0, -3.79, 0.0)
    quat = (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)
    turning_axis_pos = tuple(
        geom_utils.transform_by_trans_quat(
            np.array((-2.6035435064512424, 1.692335, 0.2395262797782305)) * mesh_scale,
            np.array(offset),
            np.array(quat),
        ).tolist()
    )
    return TeapotSettings(
        asset="meshes/utah_teapot_modified.obj",
        entity_name="teapot",
        mesh_scale=mesh_scale,
        offset=offset,
        quat=quat,
        turning_axis_pos=turning_axis_pos,
        particles_seed=(0.0, -3.15, 0.0),
        particles_max_height=0.7,
        particles_vel=(0.0, 0.0, 0.0),
        scale=scale,
        dt=dt,
        gravity=(0.0, -9.8, 0.0),
        lower_bound=(-20.0, -20.0, -20.0),
        upper_bound=(20.0, 15.0, 20.0),
        camera_pos=(26.0, 8.0, 17.0),
        camera_lookat=(0.0, -5.0, 1.5),
    )


def teapot_pose(time, settings):
    """Return the teapot pose for a rotation about the world-X line through its handle."""
    turning_rate_initial = math.radians(1.5)
    stop_angle_initial = math.radians(27.0)
    hold_end = stop_angle_initial / turning_rate_initial + 10.0
    turning_rate_final = math.radians(5.0)
    stop_angle_final = math.radians(19.0)
    turn_end = hold_end + abs(stop_angle_initial - stop_angle_final) / turning_rate_final

    if time <= stop_angle_initial / turning_rate_initial:
        angle = turning_rate_initial * time
        turning_rate = turning_rate_initial
    elif time <= hold_end:
        angle = stop_angle_initial
        turning_rate = 0.0
    elif time <= turn_end:
        angle = stop_angle_initial - turning_rate_final * (time - hold_end)
        turning_rate = -turning_rate_final
    else:
        angle = stop_angle_final
        turning_rate = 0.0

    half_angle = 0.5 * angle
    cos_half_angle = math.cos(half_angle)
    sin_half_angle = math.sin(half_angle)
    quat = settings.quat
    turning_axis_pos = settings.turning_axis_pos
    offset_y = settings.offset[1] - turning_axis_pos[1]
    offset_z = settings.offset[2] - turning_axis_pos[2]
    pos = (
        settings.offset[0],
        turning_axis_pos[1] + offset_y * math.cos(angle) - offset_z * math.sin(angle),
        turning_axis_pos[2] + offset_y * math.sin(angle) + offset_z * math.cos(angle),
    )
    return TeapotPose(
        pos=pos,
        quat=(
            cos_half_angle * quat[0] - sin_half_angle * quat[1],
            cos_half_angle * quat[1] + sin_half_angle * quat[0],
            cos_half_angle * quat[2] - sin_half_angle * quat[3],
            cos_half_angle * quat[3] + sin_half_angle * quat[2],
        ),
        vel=(
            0.0,
            -turning_rate * (pos[2] - turning_axis_pos[2]),
            turning_rate * (pos[1] - turning_axis_pos[1]),
        ),
        ang=(turning_rate, 0.0, 0.0),
    )


def sample_teapot_particles(settings, particle_size):
    teapot_mesh = mesh_utils.load_mesh(os.path.join(gs.utils.get_assets_dir(), settings.asset)).copy()
    teapot_mesh.merge_vertices(merge_tex=True, merge_norm=True)
    teapot_mesh.vertices = geom_utils.transform_by_quat(
        teapot_mesh.vertices * settings.mesh_scale,
        np.array(settings.quat),
    ) + np.array(settings.offset)
    return particle_utils.mesh_cavity_to_particles(
        teapot_mesh,
        p_size=particle_size,
        seed=settings.particles_seed,
        max_height=settings.particles_max_height,
        clearance=0.5 * particle_size,
    )


def update_rigid_teapot(teapot, time, settings):
    pose = teapot_pose(time, settings)
    teapot.set_pos(
        pose.pos,
        zero_velocity=False,
        relative=False,
        skip_forward=True,
    )
    teapot.set_quat(
        pose.quat,
        zero_velocity=False,
        relative=False,
    )
    teapot.set_velocity(
        vel=pose.vel,
        ang=pose.ang,
    )


def update_teapot_case(solver, teapot, time, settings):
    pose = teapot_pose(time, settings)
    solver.set_static_colliders_pose(pos=pose.pos, quat=pose.quat, colliders_idx=0)
    teapot.set_pos(pose.pos, relative=False, skip_forward=True)
    teapot.set_quat(pose.quat, relative=False)
