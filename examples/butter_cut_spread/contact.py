"""Bounded cohesive tool contact in the rotating blade's material coordinates."""

from dataclasses import dataclass

import numpy as np

import genesis as gs
from examples.butter_cut_spread.face_footprint import face_distance
from examples.butter_cut_spread.geometry import convex_planes, tool_mesh
from examples.butter_spreading.contact import _impulse_velocity
from genesis.utils.misc import qd_to_numpy

qd = gs.qd


@dataclass(frozen=True)
class AdhesionParameters:
    """Fixed, symmetric cohesive contact with velocity-triggered sliding and rebonding."""

    normal_stress: float = 1500.0
    bulk_normal_stress: float = 60.0
    thin_column_depth: float = 0.0112
    bulk_column_depth: float = 0.0118
    transferred_normal_stress: float = 40.0
    transferred_shear_stress: float = 450.0
    transferred_range: float = 0.0008
    transferred_slip_time: float = 0.00005
    normal_relaxation_time: float = 0.005
    max_separation_speed: float = 0.35
    substrate_capture_distance: float = 0.0016
    static_shear_stress: float = 1500.0
    cross_shear_stress: float = 50.0
    sliding_shear_stress: float = 50.0
    slip_speed: float = 0.003
    rebond_time: float = 0.006
    elastic_slip: float = 0.0003
    residue_per_face: int = 512
    residue_capture_time: float = 0.004
    residue_separation_speed: float = 0.01
    residue_release_distance: float = 0.0012
    residue_cooldown: float = 0.08


class ToolContact:
    def __init__(self, butter, config, support=None, face=None, bread_contact=None):
        self.solver = butter.solver
        self.particles = butter.solver.particles
        self.start = butter.particle_start
        self.count = butter.n_particles
        self.dt = config.dt
        self.bread_contact = bread_contact
        self.has_bread = bread_contact is not None
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
        self.parameters = AdhesionParameters()
        self.normal_stress = self.parameters.normal_stress
        self.shear_stress = self.parameters.static_shear_stress
        self.sliding_shear = self.parameters.sliding_shear_stress
        self.slip_speed = self.parameters.slip_speed
        self.rebond_time = self.parameters.rebond_time
        self.elastic_slip = self.parameters.elastic_slip
        self.board_normal = config.board_normal_stress
        self.board_shear = config.board_shear_stress
        self.equilibrium_adhesion = True
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
        self.material_spacing = butter.particle_size
        self.coating_cell = 2 * self.material_spacing
        if face is None:
            origin = -np.array(config.blade_size[:2]) / 2
            extent = np.array(config.blade_size[:2])
        else:
            origin = np.array(list(face.origin))
            extent = (np.array(list(face.shape)) - 1) * face.spacing
        self.coating_origin = qd.Vector(origin.tolist())
        self.coating_shape = tuple(np.ceil(extent / self.coating_cell).astype(int) + 1)
        self.coating_depth = qd.field(gs.qd_float, shape=(2, *self.coating_shape))
        self.coating_depth.fill(0)
        self.measure_step = 0
        self.transferred = qd.field(gs.qd_int, shape=self.count)
        self.transferred.fill(0)
        self.film_count = qd.field(gs.qd_int, shape=2)
        self.film_count.fill(0)
        self.film_side = qd.field(gs.qd_int, shape=self.count)
        self.film_anchor = qd.Vector.field(3, gs.qd_float, shape=self.count)
        self.film_cell = 2 * self.material_spacing
        self.film_shape = tuple(np.ceil(extent / self.film_cell).astype(int) + 1)
        self.film_owner = qd.field(gs.qd_int, shape=(2, *self.film_shape))
        self.film_candidate = qd.field(gs.qd_int, shape=(2, *self.film_shape))
        self.film_owner.fill(self.count)
        self.film_candidate.fill(self.count)
        self.cooldown = qd.field(gs.qd_float, shape=self.count)
        self.film_side.fill(0)
        self.film_anchor.fill(0)
        self.cooldown.fill(0)
        self.anchors = qd.Vector.field(2, gs.qd_float, shape=self.count)
        self.age = qd.field(gs.qd_float, shape=self.count)
        self.side = qd.field(gs.qd_int, shape=self.count)
        self.sliding = qd.field(gs.qd_int, shape=self.count)
        self.quiet = qd.field(gs.qd_float, shape=self.count)
        self.sliding.fill(0)
        self.quiet.fill(0)
        self.maximum_correction = qd.field(gs.qd_float, shape=1)
        self.anchors.fill(0)
        self.age.fill(0)
        self.side.fill(0)
        self.maximum_correction.fill(0)

    def apply(self, center, velocity, angles, omega, time_s, project=False, roll=0.0):
        if not project:
            if self.measure_step % 10 == 0:
                measure_coating(self.solver.sim.cur_substep_local, *center, *angles, roll, self)
            self.measure_step += 1
        contact(self.solver.sim.cur_substep_local, *center, *velocity, *angles, roll, *omega, time_s, project, self)
        if not project and self.parameters.residue_per_face > 0:
            prepare_residue(self.solver.sim.cur_substep_local, *center, *velocity, *angles, roll, *omega, self)
            choose_residue(self.solver.sim.cur_substep_local, *center, *velocity, *angles, roll, *omega, self)
            capture_residue(self.solver.sim.cur_substep_local, *center, *velocity, *angles, roll, *omega, self)

    def faces(self):
        return qd_to_numpy(self.side, transpose=True, copy=True)

    def residue_faces(self):
        return qd_to_numpy(self.film_side, transpose=True, copy=True)

    def max_correction(self):
        return float(qd_to_numpy(self.maximum_correction, transpose=True, copy=True)[0])


@qd.func
def residue_edge(local, self: qd.template()):
    edge = qd.min(self.half[0] - qd.abs(local[0]), self.half[1] - qd.abs(local[1]))
    edge = qd.min(edge, local[0] + self.half[0] - self.bevel)
    if qd.static(self.has_face):
        edge = face_distance(local[0], local[1], self.face)
    return edge


@qd.func
def release_residue(f, i, self: qd.template()):
    side = qd.select(self.film_side[i] > 0, 1, 0)
    qd.atomic_add(self.film_count[side], -1)
    self.film_side[i] = 0
    self.side[i] = 0
    self.age[i] = 0.0
    self.cooldown[i] = self.parameters.residue_cooldown
    self.solver.particles_ng[f, self.start + i, 0].active = True
    # Deposited film has relaxed while attached; release without stored elastic or affine impulses.
    self.particles[f, self.start + i, 0].F = qd.Matrix.identity(gs.qd_float, 3)
    self.particles[f, self.start + i, 0].C = qd.Matrix.zero(gs.qd_float, 3, 3)


@qd.kernel
def prepare_residue(
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
    self: qd.template(),
):
    for node in qd.grouped(self.film_owner):
        self.film_owner[node] = self.count
        self.film_candidate[node] = self.count
    R = blade_rotation(theta, yaw, roll)
    center, velocity = qd.Vector([cx, cy, cz]), qd.Vector([vx, vy, vz])
    for i in range(self.count):
        self.cooldown[i] = qd.max(0.0, self.cooldown[i] - self.dt)
        if self.film_side[i] != 0:
            anchor = self.film_anchor[i]
            world = center + R @ anchor
            surface_velocity = velocity + qd.Vector([wx, wy, wz]).cross(world - center)
            if local_coating_depth(anchor, self) > 3 * self.material_spacing:
                slip = R.transpose() @ qd.Vector([0.0, 0.0, -qd.min(0.0, surface_velocity[2])])
                anchor[0] += slip[0] * self.dt
                anchor[1] += slip[1] * self.dt
            world = center + R @ anchor
            bread_gap = 1.0
            if qd.static(self.has_bread):
                bread_gap = supported_gap(world, self.bread_contact)
            if residue_edge(anchor, self) < self.material_spacing / 2 or bread_gap < self.material_spacing:
                release_residue(f, i, self)
            else:
                self.film_anchor[i] = anchor
                cell = qd.cast(
                    qd.floor((qd.Vector([anchor[0], anchor[1]]) - self.coating_origin) / self.film_cell), qd.i32
                )
                side = qd.select(self.film_side[i] > 0, 1, 0)
                if 0 <= cell[0] < self.film_shape[0] and 0 <= cell[1] < self.film_shape[1]:
                    qd.atomic_min(self.film_owner[side, cell[0], cell[1]], i)


@qd.kernel
def choose_residue(
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
    self: qd.template(),
):
    R = blade_rotation(theta, yaw, roll)
    center = qd.Vector([cx, cy, cz])
    for i in range(self.count):
        local = R.transpose() @ (self.particles[f, self.start + i, 0].pos - center)
        if self.film_side[i] != 0:
            local = self.film_anchor[i]
        cell = qd.cast(qd.floor((qd.Vector([local[0], local[1]]) - self.coating_origin) / self.film_cell), qd.i32)
        side = qd.select(local[2] >= 0, 1, 0)
        if 0 <= cell[0] < self.film_shape[0] and 0 <= cell[1] < self.film_shape[1]:
            if self.film_side[i] != 0:
                if self.film_owner[side, cell[0], cell[1]] != i:
                    release_residue(f, i, self)
            elif self.cooldown[i] == 0 and self.side[i] != 0 and self.age[i] >= self.parameters.residue_capture_time:
                gap = qd.abs(local[2]) - self.half[2]
                surface_velocity = qd.Vector([vx, vy, vz]) + qd.Vector([wx, wy, wz]).cross(R @ local)
                relative = R.transpose() @ (self.particles[f, self.start + i, 0].vel - surface_velocity)
                separating_speed = (2 * side - 1) * relative[2]
                bread_gap = 1.0
                if qd.static(self.has_bread):
                    bread_gap = supported_gap(self.particles[f, self.start + i, 0].pos, self.bread_contact)
                if (
                    0 <= gap <= self.material_spacing
                    and residue_edge(local, self) > self.material_spacing / 2
                    and bread_gap > 2 * self.material_spacing
                    and separating_speed > self.parameters.residue_separation_speed
                    and self.film_owner[side, cell[0], cell[1]] == self.count
                ):
                    qd.atomic_min(self.film_candidate[side, cell[0], cell[1]], i)


@qd.kernel
def capture_residue(
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
    self: qd.template(),
):
    R = blade_rotation(theta, yaw, roll)
    center, velocity = qd.Vector([cx, cy, cz]), qd.Vector([vx, vy, vz])
    for side, x, y in self.film_candidate:
        i = self.film_candidate[side, x, y]
        if i < self.count:
            slot = qd.atomic_add(self.film_count[side], 1)
            if slot < self.parameters.residue_per_face:
                p = self.start + i
                local = R.transpose() @ (self.particles[f, p, 0].pos - center)
                sign = 2 * side - 1
                local[2] = sign * (self.half[2] + self.material_spacing / 2)
                self.film_side[i] = sign
                self.film_anchor[i] = local
                self.solver.particles_ng[f, p, 0].active = False
                self.particles[f, p, 0].pos = center + R @ local
                self.particles[f, p, 0].vel = velocity + qd.Vector([wx, wy, wz]).cross(R @ local)
            else:
                qd.atomic_add(self.film_count[side], -1)


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
    R = blade_rotation(theta, yaw, roll)
    center, velocity = qd.Vector([cx, cy, cz]), qd.Vector([vx, vy, vz])
    for i in range(self.count):
        p = self.start + i
        if self.film_side[i] != 0:
            continue
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
            bread_gap = 1.0
            if qd.static(self.has_bread):
                bread_gap = supported_gap(world, self.bread_contact)
            sign = qd.select(local[2] >= 0, 1, -1)
            gap = qd.abs(local[2]) - self.half[2] - self.spacing / 2
            edge = qd.min(self.half[0] - qd.abs(local[0]), self.half[1] - qd.abs(local[1]))
            edge = qd.min(edge, local[0] + self.half[0] - self.bevel)
            if qd.static(self.has_face):
                edge = -self.spacing
                if gap < self.range:
                    edge = face_distance(local[0], local[1], self.face)
            if edge <= 0 or gap >= self.range or gap < -self.spacing or sign * self.side[i] < 0:
                self.side[i] = 0
                self.age[i] = 0.0
                self.sliding[i] = 0
                self.quiet[i] = 0.0
            elif self.side[i] == 0 and gap < self.capture_range:
                self.side[i] = sign
                self.anchors[i] = qd.Vector([local[0], local[1]])
                self.sliding[i] = qd.cast(qd.Vector([relative[0], relative[1]]).norm() > self.slip_speed, gs.qd_int)
            coating_depth = 0.0
            if self.side[i] != 0 or self.film_side[i] != 0:
                coating_depth = local_coating_depth(local, self)
            column_reach = qd.max(0.0, coating_depth - (qd.abs(local[2]) - self.half[2]))
            column_reach *= qd.max(0.0, -sign * R[2, 2])
            if bread_gap < self.parameters.substrate_capture_distance + column_reach:
                self.transferred[i] = 1
            elif bread_gap > 0.006 and self.side[i] == 0:
                self.transferred[i] = 0
            if self.side[i] != 0:
                self.age[i] += self.dt
                maturity = qd.min(1.0, self.age[i] / 0.002)
                maturity = maturity * maturity * (3 - 2 * maturity)
                bulk_weight = qd.min(
                    1.0,
                    qd.max(
                        0.0,
                        (coating_depth - self.parameters.thin_column_depth)
                        / (self.parameters.bulk_column_depth - self.parameters.thin_column_depth),
                    ),
                )
                normal_stress = self.normal_stress + bulk_weight * (
                    self.parameters.bulk_normal_stress - self.normal_stress
                )
                shear = self.shear_stress
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
                slip = self.elastic_slip
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
                if self.sliding[i] == 0 and impulse.norm() > limit:
                    self.sliding[i] = 1
                    self.quiet[i] = 0.0
                if self.sliding[i] != 0:
                    if tangent.norm() < self.slip_speed:
                        self.quiet[i] += self.dt
                    else:
                        self.quiet[i] = 0.0
                    self.anchors[i] = qd.Vector([local[0], local[1]])
                    limit = self.sliding_shear * self.dt * weight / (self.rho * depth)
                    impulse = -tangent
                    if self.quiet[i] >= self.rebond_time:
                        self.sliding[i] = 0
                cross_limit = self.parameters.cross_shear_stress * self.dt * weight / (self.rho * depth)
                impulse[1] = qd.max(-cross_limit, qd.min(cross_limit, impulse[1]))
                impulse *= qd.min(1.0, limit / qd.max(impulse.norm(), 1e-12))
                relative[0] += impulse[0]
                relative[1] += impulse[1]
            if self.transferred[i] != 0:
                relative = initial_relative
                if edge > 0 and -self.spacing < gap < self.parameters.transferred_range:
                    normal = qd.Vector([0.0, 0.0, sign])
                    relative += qd.min(1.0, edge / self.spacing) * _impulse_velocity(
                        initial_relative,
                        normal,
                        gap,
                        self.parameters.transferred_normal_stress,
                        self.parameters.transferred_range,
                        self.parameters.transferred_slip_time,
                        self.parameters.transferred_shear_stress,
                        self,
                    )
                    if gap < 0:
                        target_speed = qd.min(
                            -gap / self.parameters.normal_relaxation_time, self.parameters.max_separation_speed
                        )
                        relative[2] += sign * qd.max(0.0, target_speed - sign * relative[2])
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


@qd.func
def supported_gap(world, bread: qd.template()):
    uv = (qd.Vector([world[0], world[1]]) - bread.origin) / bread.grid_spacing
    cell = qd.cast(qd.floor(uv), qd.i32)
    fraction = uv - cell
    height, support_weight = 0.0, 0.0
    for a, b in qd.static(qd.ndrange(2, 2)):
        node = cell + qd.Vector([a, b])
        if 0 <= node[0] < bread.res[0] and 0 <= node[1] < bread.res[1]:
            mass = bread.mass[0, node[0], node[1]]
            if mass > 0:
                weight = qd.select(a == 0, 1 - fraction[0], fraction[0])
                weight *= qd.select(b == 0, 1 - fraction[1], fraction[1])
                height += weight * bread.height_mass[0, node[0], node[1]] / mass
                support_weight += weight
    gap = 1.0
    if support_weight > 0.5:
        gap = world[2] - height / support_weight - (bread.spacing + bread.bread_spacing) / 2
    return gap


@qd.func
def blade_rotation(theta, yaw, roll):
    c, s, cyaw, syaw = qd.cos(theta), qd.sin(theta), qd.cos(yaw), qd.sin(yaw)
    R = qd.Matrix([[cyaw * c, -syaw, cyaw * s], [syaw * c, cyaw, syaw * s], [-s, 0.0, c]])
    cr, sr = qd.cos(roll), qd.sin(roll)
    R = qd.Matrix([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]) @ R
    return R


@qd.func
def local_coating_depth(local, self: qd.template()):
    uv = (qd.Vector([local[0], local[1]]) - self.coating_origin) / self.coating_cell
    cell = qd.cast(qd.floor(uv), qd.i32)
    side = qd.select(local[2] >= 0, 1, 0)
    depth = 0.0
    for x, y in qd.static(qd.ndrange((-1, 2), (-1, 2))):
        node = cell + qd.Vector([x, y])
        if 0 <= node[0] < self.coating_shape[0] and 0 <= node[1] < self.coating_shape[1]:
            depth = qd.max(depth, self.coating_depth[side, node[0], node[1]])
    return depth


@qd.kernel
def measure_coating(
    f: qd.i32, cx: float, cy: float, cz: float, theta: float, yaw: float, roll: float, self: qd.template()
):
    for node in qd.grouped(self.coating_depth):
        self.coating_depth[node] = 0.0
    R = blade_rotation(theta, yaw, roll)
    center = qd.Vector([cx, cy, cz])
    for i in range(self.count):
        local = R.transpose() @ (self.particles[f, self.start + i, 0].pos - center)
        if self.half[2] - self.spacing / 2 < qd.abs(local[2]) < self.half[2] + 0.012:
            uv = (qd.Vector([local[0], local[1]]) - self.coating_origin) / self.coating_cell
            cell = qd.cast(qd.floor(uv), qd.i32)
            fraction = uv - cell
            side = qd.select(local[2] >= 0, 1, 0)
            for x, y in qd.static(qd.ndrange(2, 2)):
                node = cell + qd.Vector([x, y])
                if 0 <= node[0] < self.coating_shape[0] and 0 <= node[1] < self.coating_shape[1]:
                    weight = qd.select(x == 0, 1 - fraction[0], fraction[0])
                    weight *= qd.select(y == 0, 1 - fraction[1], fraction[1])
                    qd.atomic_add(
                        self.coating_depth[side, node[0], node[1]],
                        weight * self.material_spacing**3 / self.coating_cell**2,
                    )
