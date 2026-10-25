"""Bounded cohesive tool contact in the rotating blade's material coordinates."""

import numpy as np

import genesis as gs
from examples.butter_cut_spread.face_footprint import face_distance
from examples.butter_cut_spread.geometry import convex_planes, tool_mesh
from examples.butter_spreading.config import ContactParameters
from examples.butter_spreading.contact import _impulse_velocity
from genesis.utils.misc import qd_to_numpy

qd = gs.qd


class ToolContact:
    def __init__(self, butter, config, support=None, face=None):
        self.solver = butter.solver
        self.particles = butter.solver.particles
        self.start = butter.particle_start
        self.count = butter.n_particles
        self.dt = config.dt
        self.face = face
        self.has_face = face is not None
        self.spacing = config.particle_size
        self.half = qd.Vector([v / 2 for v in config.blade_size])
        self.bevel = config.blade_bevel_height
        mesh = tool_mesh(config)
        planes = convex_planes(mesh.vertices, mesh.faces)
        self.plane_count = len(planes)
        self.planes = qd.Vector.field(4, gs.qd_float, shape=self.plane_count)
        self.planes.from_numpy(planes.astype(gs.np_float))
        self.rho = butter.material.rho
        self.range = config.adhesion_range
        self.capture_range = (
            config.particle_size if config.adhesion_capture_range is None else config.adhesion_capture_range
        )
        if not 0 < self.capture_range <= self.range:
            raise ValueError("Adhesion capture range must be positive and within the release range.")
        self.normal_stress = config.normal_stress
        self.shear_stress = config.shear_stress
        self.spread_normal = config.spread_normal_stress
        self.spread_shear = config.spread_shear_stress
        wet = ContactParameters()
        self.spread_range = config.spread_adhesion_range
        self.spread_slip_time = config.spread_slip_time
        self.spread_normal_time = wet.blade_normal_time
        self.spread_max_separation_speed = wet.blade_max_separation_speed
        self.equilibrium_adhesion = wet.is_equilibrium_adhesion
        self.board_normal = config.board_normal_stress
        self.board_shear = config.board_shear_stress
        self.cut_end_time = config.cut_end_time
        self.cutting_shear = config.cutting_shear_stress
        self.spread_transition_time = config.spread_transition_time
        if support is None:
            center, size = config.board_centers[0], config.board_sizes[0]
            origin = np.array(center[:2]) - np.array(size[:2]) / 2
            spacing = np.array(size[:2])
            heights = np.full((2, 2), center[2] + size[2] / 2)
        else:
            origin, spacing, heights = support
        self.support_origin = qd.Vector(origin.tolist())
        self.support_spacing = qd.Vector(spacing.tolist())
        self.support_shape = qd.Vector(list(heights.shape))
        self.support = qd.field(gs.qd_float, shape=heights.shape)
        self.support.from_numpy(heights.astype(gs.np_float))
        self.anchors = qd.Vector.field(2, gs.qd_float, shape=self.count)
        self.age = qd.field(gs.qd_float, shape=self.count)
        self.side = qd.field(gs.qd_int, shape=self.count)
        self.maximum_correction = qd.field(gs.qd_float, shape=1)
        self.anchors.fill(0)
        self.age.fill(0)
        self.side.fill(0)
        self.maximum_correction.fill(0)

    def apply(self, center, velocity, angles, omega, time_s, project=False, roll=0.0):
        contact(self.solver.sim.cur_substep_local, *center, *velocity, *angles, roll, *omega, time_s, project, self)

    def faces(self):
        return qd_to_numpy(self.side, transpose=True, copy=True)

    def max_correction(self):
        return float(qd_to_numpy(self.maximum_correction, transpose=True, copy=True)[0])


@qd.kernel
def contact(
    f: qd.i32,
    cx: float,
    cy: float,
    cz: float,
    vx: float,
    vy: float,
    vz: float,
    theta: float,
    yaw: float,
    roll: float,
    wx: float,
    wy: float,
    wz: float,
    time_s: float,
    project: qd.i32,
    self: qd.template(),
):
    c, s, cyaw, syaw = qd.cos(theta), qd.sin(theta), qd.cos(yaw), qd.sin(yaw)
    R = qd.Matrix([[cyaw * c, -syaw, cyaw * s], [syaw * c, cyaw, syaw * s], [-s, 0.0, c]])
    cr, sr = qd.cos(roll), qd.sin(roll)
    R = qd.Matrix([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]) @ R
    center, velocity = qd.Vector([cx, cy, cz]), qd.Vector([vx, vy, vz])
    for i in range(self.count):
        p = self.start + i
        world = self.particles[f, p, 0].pos
        local = R.transpose() @ (world - center)
        surface_velocity = velocity + qd.Vector([wx, wy, wz]).cross(world - center)
        relative = R.transpose() @ (self.particles[f, p, 0].vel - surface_velocity)
        if project:
            distance = -1e6
            normal = qd.Vector([0.0, 0.0, 0.0])
            for a in range(self.plane_count):
                plane = self.planes[a]
                n = qd.Vector([plane[0], plane[1], plane[2]])
                d = n.dot(local) + plane[3]
                if d > distance:
                    distance, normal = d, n
            if distance < 0:
                local -= distance * normal
                relative -= qd.min(0.0, relative.dot(normal)) * normal
                self.particles[f, p, 0].pos = center + R @ local
                self.particles[f, p, 0].vel = surface_velocity + R @ relative
                qd.atomic_max(self.maximum_correction[0], -distance)
        else:
            initial_relative = relative
            transition = qd.min(1.0, qd.max(0.0, (time_s - self.spread_transition_time) / 0.4))
            transition = transition**2 * (3 - 2 * transition)
            sign = qd.select(local[2] >= 0, 1, -1)
            gap = qd.abs(local[2]) - self.half[2] - self.spacing / 2
            edge = qd.min(self.half[0] - qd.abs(local[0]), self.half[1] - qd.abs(local[1]))
            edge = qd.min(edge, local[0] + self.half[0] - self.bevel)
            if qd.static(self.has_face):
                edge = -self.spacing
                if gap < qd.max(self.range, self.spread_range):
                    edge = face_distance(local[0], local[1], self.face)
            if edge <= 0 or gap >= self.range or gap < -self.spacing or sign * self.side[i] < 0:
                self.side[i] = 0
                self.age[i] = 0.0
            elif self.side[i] == 0 and gap < self.capture_range:
                self.side[i] = sign
                self.anchors[i] = qd.Vector([local[0], local[1]])
            if self.side[i] != 0:
                self.age[i] += self.dt
                maturity = qd.min(1.0, self.age[i] / 0.002)
                maturity = maturity * maturity * (3 - 2 * maturity)
                normal_stress = self.normal_stress
                shear = self.shear_stress
                if sign > 0:
                    normal_stress = 0.0
                    shear = 0.0
                weight = maturity * qd.min(1.0, edge / self.spacing)
                separation = qd.min(1.0, qd.max(0.0, gap / self.range))
                relative[2] -= (
                    sign
                    * weight
                    * 4
                    * separation
                    * (1 - separation)
                    * normal_stress
                    * self.dt
                    / (self.rho * (2 * self.range / 3))
                )
                tangent = qd.Vector([relative[0], relative[1]])
                displacement = qd.Vector([local[0], local[1]]) - self.anchors[i]
                slip = 0.0003
                if displacement.norm() > slip:
                    displacement *= slip / displacement.norm()
                    self.anchors[i] = qd.Vector([local[0], local[1]]) - displacement
                depth = self.range / 3 + self.spacing / 2
                stiffness = shear / slip
                damping = 2 * qd.sqrt(stiffness * self.rho * depth)
                weight *= qd.max(0.0, 1 - qd.max(gap, 0.0) / self.range) ** 2
                impulse = -self.dt * weight * (stiffness * displacement + (damping + self.dt * stiffness) * tangent)
                impulse /= self.rho * depth + weight * (self.dt * damping + self.dt**2 * stiffness)
                limit = shear * self.dt * weight / (self.rho * depth)
                if time_s < self.cut_end_time:
                    limit = qd.min(limit, self.cutting_shear * self.dt * weight / (self.rho * depth))
                    self.anchors[i] = qd.Vector([local[0], local[1]])
                impulse *= qd.min(1.0, limit / qd.max(impulse.norm(), 1e-12))
                relative[0] += impulse[0]
                relative[1] += impulse[1]
            spreading_impulse = qd.Vector([0.0, 0.0, 0.0])
            # Material flowing around an edge sees the same wet contact on either face.
            # The outward normal follows its current side; no adhesion pulls through the blade.
            if transition > 0 and edge > 0 and -self.spacing < gap < self.spread_range:
                normal = qd.Vector([0.0, 0.0, sign])
                spreading_impulse = _impulse_velocity(
                    initial_relative,
                    normal,
                    gap,
                    self.spread_normal,
                    self.spread_range,
                    self.spread_slip_time,
                    self.spread_shear,
                    self,
                )
                spreading_impulse *= qd.min(1.0, edge / self.spacing)
            relative = (1 - transition) * relative + transition * (initial_relative + spreading_impulse)
            if transition > 0 and edge > 0 and gap < 0:
                target_speed = qd.min(-gap / self.spread_normal_time, self.spread_max_separation_speed)
                relative[2] += transition * sign * qd.max(0.0, target_speed - sign * relative[2])
            self.particles[f, p, 0].vel = surface_velocity + R @ relative
            # Finite stationary board adhesion acts only on the cutting board.
            uv = (qd.Vector([world[0], world[1]]) - self.support_origin) / self.support_spacing
            if 0 <= uv[0] < self.support_shape[0] - 1 and 0 <= uv[1] < self.support_shape[1] - 1:
                node = qd.cast(qd.floor(uv), qd.i32)
                weight = uv - node
                height = 0.0
                for a, b in qd.static(qd.ndrange(2, 2)):
                    w = qd.select(a == 0, 1 - weight[0], weight[0])
                    w *= qd.select(b == 0, 1 - weight[1], weight[1])
                    height += w * self.support[node[0] + a, node[1] + b]
                gap_board = world[2] - height - self.spacing / 2
                if 0 < gap_board < 0.0009:
                    u = qd.min(1.0, gap_board / 0.0009)
                    self.particles[f, p, 0].vel[2] -= (
                        self.board_normal * 4 * u * (1 - u) * self.dt / (self.rho * 0.0006)
                    )
                    tangent = qd.Vector([self.particles[f, p, 0].vel[0], self.particles[f, p, 0].vel[1]])
                    limit = self.board_shear * (1 - u) ** 2 * self.dt / (self.rho * 0.0006)
                    tangent *= qd.min(1.0, limit / qd.max(tangent.norm(), 1e-12))
                    self.particles[f, p, 0].vel[0] -= tangent[0]
                    self.particles[f, p, 0].vel[1] -= tangent[1]
