"""Resolve particle penetration against the same mesh distance field used by native rigid contact."""

import numpy as np

import genesis as gs
import genesis.utils.array_class as arrays
from examples.butter_cut_spread.contact import release_residue
from examples.butter_cut_spread.face_footprint import face_distance
from genesis.utils import sdf
from genesis.utils.misc import qd_to_numpy

qd = gs.qd


class MeshProjection:
    def __init__(self, butter, blade, rotation, blade_size, face=None, contact=None):
        self.contact = contact
        self.has_residue = contact is not None
        self.mpm = butter.solver
        self.rigid = blade.solver
        self.start, self.count = butter.particle_start, butter.n_particles
        self.geom_index = blade.geoms[0].idx
        self.rotation = rotation.tolist()
        self.half = qd.Vector((np.array(blade_size) / 2).tolist())
        self.face = face
        self.has_face = face is not None
        self.skin = 0.00001
        self.previous = qd.Vector.field(3, gs.qd_float, shape=self.count)
        self.maximum = qd.field(gs.qd_float, shape=1)
        self.maximum.fill(0)

    def record(self, center, rotation=None):
        if rotation is None:
            rotation = self.rotation
        record_local(self.mpm.sim.cur_substep_local, *center, *rotation[0], *rotation[1], *rotation[2], self)

    def apply(self, center, velocity, rotation=None, angular=(0.0, 0.0, 0.0)):
        if rotation is None:
            rotation = self.rotation
        project_mesh(
            self.mpm.sim.cur_substep_local,
            self.geom_index,
            *center,
            *velocity,
            *angular,
            *rotation[0],
            *rotation[1],
            *rotation[2],
            self.rigid.dyn_state.geoms,
            self.rigid.dyn_info.geoms,
            self.rigid.rigid_info,
            self.rigid.collider._sdf._sdf_info,
            self.rigid.collider._collider_static_config,
            self,
        )

    def maximum_correction(self):
        return float(qd_to_numpy(self.maximum, transpose=True, copy=True)[0])


@qd.kernel
def record_local(
    f: qd.i32,
    cx: float,
    cy: float,
    cz: float,
    r00: float,
    r01: float,
    r02: float,
    r10: float,
    r11: float,
    r12: float,
    r20: float,
    r21: float,
    r22: float,
    self: qd.template(),
):
    center = qd.Vector([cx, cy, cz])
    rotation = qd.Matrix([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    for i in range(self.count):
        self.previous[i] = rotation.transpose() @ (self.mpm.particles[f, self.start + i, 0].pos - center)


@qd.kernel
def project_mesh(
    f: qd.i32,
    geom_index: qd.i32,
    cx: float,
    cy: float,
    cz: float,
    vx: float,
    vy: float,
    vz: float,
    wx: float,
    wy: float,
    wz: float,
    r00: float,
    r01: float,
    r02: float,
    r10: float,
    r11: float,
    r12: float,
    r20: float,
    r21: float,
    r22: float,
    geoms_state: arrays.GeomsState,
    geoms_info: arrays.GeomsInfo,
    rigid_info: arrays.RigidInfo,
    sdf_info: arrays.SDFInfo,
    collider_config: qd.template(),
    self: qd.template(),
):
    center, velocity = qd.Vector([cx, cy, cz]), qd.Vector([vx, vy, vz])
    rotation = qd.Matrix([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    for i in range(self.count):
        p = self.start + i
        if qd.static(self.has_residue):  # noqa: SIM102
            if self.contact.film_side[i] != 0:
                target = center + rotation @ self.contact.film_anchor[i]
                if (target - self.mpm.particles[f, p, 0].pos).norm() < self.contact.parameters.residue_release_distance:
                    self.mpm.particles[f, p, 0].pos = target
                    self.mpm.particles[f, p, 0].vel = velocity + qd.Vector([wx, wy, wz]).cross(target - center)
                    continue
                release_residue(f, i, self.contact)
        original = self.mpm.particles[f, p, 0].pos
        point = original
        local = rotation.transpose() @ (point - center)
        previous = self.previous[i]
        previous_sign = qd.select(previous[2] >= 0, 1.0, -1.0)
        crossed = local[2] * previous_sign < 0
        inside_patch = qd.abs(local[0]) < self.half[0] - 0.001 and qd.abs(local[1]) < self.half[1] - 0.001
        was_inside_patch = qd.abs(previous[0]) < self.half[0] - 0.001 and qd.abs(previous[1]) < self.half[1] - 0.001
        # Eliminate the optional field at compile time before testing dynamic crossings.
        if qd.static(self.has_face):  # noqa: SIM102
            if crossed and qd.abs(previous[2]) < self.half[2] + 0.001:
                inside_patch = face_distance(local[0], local[1], self.face) > 0.001
                was_inside_patch = face_distance(previous[0], previous[1], self.face) > 0.001
        normal = qd.Vector([0.0, 0.0, 0.0])
        if crossed and inside_patch and was_inside_patch and qd.abs(previous[2]) < self.half[2] + 0.001:
            midplane = center + rotation @ qd.Vector([local[0], local[1], 0.0])
            if sdf.sdf_func_world(geom_index, 0, midplane, geoms_state, geoms_info, sdf_info) < 0:
                local[2] = previous_sign * (self.half[2] + self.skin)
                point = center + rotation @ local
                normal = rotation @ qd.Vector([0.0, 0.0, previous_sign])
        for _ in range(3):
            distance = sdf.sdf_func_world(geom_index, 0, point, geoms_state, geoms_info, sdf_info)
            if distance < self.skin:
                n = sdf.sdf_func_normal_world(
                    geom_index, 0, point, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
                )
                if n.norm() > 1e-6:
                    normal = n.normalized()
                    point += (self.skin - distance) * normal
            else:
                break
        correction = (point - original).norm()
        if correction > 0:
            surface_velocity = velocity + qd.Vector([wx, wy, wz]).cross(point - center)
            relative = self.mpm.particles[f, p, 0].vel - surface_velocity
            relative -= qd.min(0.0, relative.dot(normal)) * normal
            self.mpm.particles[f, p, 0].pos = point
            self.mpm.particles[f, p, 0].vel = surface_velocity + relative
            qd.atomic_max(self.maximum[0], correction)
