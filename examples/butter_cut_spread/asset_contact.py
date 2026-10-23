"""Resolve particle penetration against the same mesh distance field used by native rigid contact."""

import numpy as np

import genesis as gs
import genesis.utils.array_class as arrays
from genesis.utils import sdf
from genesis.utils.misc import qd_to_numpy

qd = gs.qd


class MeshProjection:
    def __init__(self, butter, blade, rotation, blade_size):
        self.mpm = butter.solver
        self.rigid = blade.solver
        self.start, self.count = butter.particle_start, butter.n_particles
        self.geom_index = blade.geoms[0].idx
        self.rotation = qd.Matrix(rotation.tolist())
        self.half = qd.Vector((np.array(blade_size) / 2).tolist())
        self.skin = 0.00001
        self.previous = qd.Vector.field(3, gs.qd_float, shape=self.count)
        self.maximum = qd.field(gs.qd_float, shape=1)
        self.maximum.fill(0)

    def record(self, center):
        record_local(self.mpm.sim.cur_substep_local, *center, self)

    def apply(self, center, velocity):
        project_mesh(
            self.mpm.sim.cur_substep_local,
            self.geom_index,
            *center,
            *velocity,
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
def record_local(f: qd.i32, cx: float, cy: float, cz: float, self: qd.template()):
    center = qd.Vector([cx, cy, cz])
    for i in range(self.count):
        self.previous[i] = self.rotation.transpose() @ (self.mpm.particles[f, self.start + i, 0].pos - center)


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
    geoms_state: arrays.GeomsState,
    geoms_info: arrays.GeomsInfo,
    rigid_info: arrays.RigidInfo,
    sdf_info: arrays.SDFInfo,
    collider_config: qd.template(),
    self: qd.template(),
):
    center, velocity = qd.Vector([cx, cy, cz]), qd.Vector([vx, vy, vz])
    for i in range(self.count):
        p = self.start + i
        original = self.mpm.particles[f, p, 0].pos
        point = original
        local = self.rotation.transpose() @ (point - center)
        previous = self.previous[i]
        previous_sign = qd.select(previous[2] >= 0, 1.0, -1.0)
        crossed = local[2] * previous_sign < 0
        inside_patch = qd.abs(local[0]) < self.half[0] - 0.001 and qd.abs(local[1]) < self.half[1] - 0.001
        was_inside_patch = qd.abs(previous[0]) < self.half[0] - 0.001 and qd.abs(previous[1]) < self.half[1] - 0.001
        normal = qd.Vector([0.0, 0.0, 0.0])
        if crossed and inside_patch and was_inside_patch and qd.abs(previous[2]) < self.half[2] + 0.001:
            midplane = center + self.rotation @ qd.Vector([local[0], local[1], 0.0])
            if sdf.sdf_func_world(geom_index, 0, midplane, geoms_state, geoms_info, sdf_info) < 0:
                local[2] = previous_sign * (self.half[2] + self.skin)
                point = center + self.rotation @ local
                normal = self.rotation @ qd.Vector([0.0, 0.0, previous_sign])
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
            relative = self.mpm.particles[f, p, 0].vel - velocity
            relative -= qd.min(0.0, relative.dot(normal)) * normal
            self.mpm.particles[f, p, 0].pos = point
            self.mpm.particles[f, p, 0].vel = velocity + relative
            qd.atomic_max(self.maximum[0], correction)
