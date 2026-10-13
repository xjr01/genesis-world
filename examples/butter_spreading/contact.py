import math

import torch

import genesis as gs
from genesis.utils import geom as gu
from genesis.utils.misc import tensor_to_array

qd = gs.qd


def _with_batch(tensor):
    return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor


class ButterContact:
    """Apply finite-range contact with a fused GPU kernel and per-environment surface grids."""

    def __init__(
        self,
        bread,
        butter,
        blade,
        *,
        spacing,
        dt,
        blade_size,
        lower_bound,
        upper_bound,
        bread_stress,
        blade_stress,
        bread_contact_range,
        bread_slip_time,
        bread_shear_stress,
        blade_contact_range,
        blade_slip_time,
        blade_shear_stress,
        blade_normal_relaxation,
        blade_max_separation_speed,
        blade_contact_margin,
        is_equilibrium_adhesion,
    ):
        self._blade = blade
        self.dt = dt
        self.spacing = spacing
        self.bread_spacing = bread.particle_size
        self.bread_stress = bread_stress
        self.blade_stress = blade_stress
        self.bread_contact_range = bread_contact_range
        self.blade_contact_range = blade_contact_range
        self.bread_slip_time = bread_slip_time
        self.blade_slip_time = blade_slip_time
        self.bread_shear_stress = bread_shear_stress
        self.blade_shear_stress = blade_shear_stress
        self.blade_normal_relaxation = blade_normal_relaxation
        self.blade_max_separation_speed = blade_max_separation_speed
        self.blade_contact_margin = blade_contact_margin
        self.equilibrium_adhesion = is_equilibrium_adhesion

        bread_positions = _with_batch(bread.get_particles_pos())
        top = bread_positions[0, :, 2].max()
        self._surface_indices = torch.nonzero(
            bread_positions[0, :, 2] > top - self.bread_spacing * 0.55,
            as_tuple=False,
        ).squeeze(-1)
        self._res = tuple(
            math.ceil((upper - lower) / self.bread_spacing) + 2
            for lower, upper in zip(lower_bound[:2], upper_bound[:2])
        )
        self._solver = butter.solver
        self.particles = self._solver.particles
        self.info = self._solver.particles_info
        self.butter_start = butter.particle_start
        self.butter_end = butter.particle_end
        self.n_envs = bread_positions.shape[0]
        self.surface_ids = qd.field(gs.qd_int, shape=len(self._surface_indices))
        self.surface_ids.from_numpy(tensor_to_array(self._surface_indices + bread.particle_start, dtype=gs.np_int))
        shape = (self.n_envs, *self._res)
        self.mass = qd.field(gs.qd_float, shape=shape)
        self.height_mass = qd.field(gs.qd_float, shape=shape)
        self.momentum = qd.Vector.field(3, gs.qd_float, shape=shape)
        self.reaction = qd.Vector.field(3, gs.qd_float, shape=shape)
        self.origin = qd.Vector(lower_bound[:2])
        self.res = self._res
        self.grid_spacing = self.bread_spacing
        self.half_blade = qd.Vector([value / 2 for value in blade_size])
        self.rho = butter.material.rho

    def apply(self, *, center=None, velocity=None):
        """Use actual blade state, or Python xyz tuples prescribing one level-blade motion across environments."""
        if (center is None) != (velocity is None):
            raise ValueError("Prescribed blade contact requires both center and velocity.")
        if center is not None:
            _apply_prescribed_contact(self._solver.sim.cur_substep_local, *center, *velocity, self)
            return
        center = self._blade.get_pos(relative=False).reshape(self.n_envs, 3)
        quaternion = self._blade.get_quat(relative=False).reshape(self.n_envs, 4)
        velocity = self._blade.get_vel().reshape(self.n_envs, 3)
        angular_velocity = self._blade.get_ang().reshape(self.n_envs, 3)
        _apply_dynamic_contact(self._solver.sim.cur_substep_local, center, quaternion, velocity, angular_velocity, self)

    def __call__(self):
        self.apply()
        return False


@qd.func
def _cell(pos, self: qd.template()):
    uv = (qd.Vector([pos[0], pos[1]]) - self.origin) / self.grid_spacing
    base = qd.cast(qd.floor(uv), qd.i32)
    return base, uv - base


@qd.func
def _weight(frac, a, b):
    return (frac[0] if a == 1 else 1 - frac[0]) * (frac[1] if b == 1 else 1 - frac[1])


@qd.func
def _valid(node, self: qd.template()):
    return 0 <= node[0] < self.res[0] and 0 <= node[1] < self.res[1]


@qd.func
def _impulse_velocity(
    relative_velocity, normal, gap, stress, contact_range, slip_time, shear_stress, self: qd.template()
):
    # A finite separation distance permits peel-off; there are no permanent bonds.
    weight = qd.max(0.0, 1.0 - qd.max(gap, 0.0) / contact_range) ** 2
    normal_weight = weight
    normal_depth = self.spacing
    tangent_depth = self.spacing
    if qd.static(self.equilibrium_adhesion):
        # A cohesive zone pulls separated surfaces together, but supplies
        # no inward traction once the material is already in contact.
        # Constant attraction at negative gaps otherwise acts like an
        # artificial compressive load inside the shared MPM grid.
        separation = qd.min(1.0, qd.max(0.0, gap / contact_range))
        normal_weight = 4.0 * separation * (1.0 - separation)
        # Integrals of 4*s*(1-s) and (1-s)^2 over the physical contact
        # layer. Refining quadrature must not increase total traction.
        normal_depth = 2.0 * contact_range / 3.0
        tangent_depth = contact_range / 3.0 + 0.5 * self.spacing
    dv_normal = -normal * stress * self.dt / (self.rho * normal_depth) * normal_weight
    tangent = relative_velocity - relative_velocity.dot(normal) * normal
    speed = tangent.norm(1.0e-12)
    drag = qd.min(
        speed * (1.0 - qd.exp(-self.dt / slip_time)),
        shear_stress * self.dt / (self.rho * tangent_depth),
    )
    return dv_normal - tangent / speed * drag * weight


@qd.kernel
def _apply_dynamic_contact(
    f: qd.i32,
    center: qd.types.ndarray(ndim=2),
    quaternion: qd.types.ndarray(ndim=2),
    velocity: qd.types.ndarray(ndim=2),
    angular_velocity: qd.types.ndarray(ndim=2),
    self: qd.template(),
):
    _apply_contact(f, center, quaternion, velocity, angular_velocity, self, is_prescribed=False)


@qd.kernel
def _apply_prescribed_contact(
    f: qd.i32,
    center_x: float,
    center_y: float,
    center_z: float,
    velocity_x: float,
    velocity_y: float,
    velocity_z: float,
    self: qd.template(),
):
    center = qd.Vector([center_x, center_y, center_z])
    velocity = qd.Vector([velocity_x, velocity_y, velocity_z])
    _apply_contact(f, center, 0, velocity, 0, self, is_prescribed=True)


@qd.func
def _apply_contact(
    f,
    center: qd.template(),
    quaternion: qd.template(),
    velocity: qd.template(),
    angular_velocity: qd.template(),
    self: qd.template(),
    is_prescribed: qd.template(),
):
    for node in qd.grouped(self.mass):
        self.mass[node] = 0.0
        self.height_mass[node] = 0.0
        self.momentum[node] = qd.Vector.zero(gs.qd_float, 3)
        self.reaction[node] = qd.Vector.zero(gs.qd_float, 3)

    # Conservative bilinear scatter of the deforming bread surface.
    for b, a in qd.ndrange(self.n_envs, self.surface_ids.shape[0]):
        i = self.surface_ids[a]
        pos = self.particles[f, i, b].pos
        mass = self.info[i].mass
        base, frac = _cell(pos, self)
        for u, v in qd.static(qd.ndrange(2, 2)):
            node = base + qd.Vector([u, v])
            if _valid(node, self):
                wm = _weight(frac, u, v) * mass
                self.mass[b, node[0], node[1]] += wm
                self.height_mass[b, node[0], node[1]] += wm * pos[2]
                self.momentum[b, node[0], node[1]] += wm * self.particles[f, i, b].vel

    for b, i in qd.ndrange(self.n_envs, (self.butter_start, self.butter_end)):
        pos = self.particles[f, i, b].pos
        vel = self.particles[f, i, b].vel
        base, frac = _cell(pos, self)
        height = 0.0
        surface_velocity = qd.Vector.zero(gs.qd_float, 3)
        weight_sum = 0.0
        for u, v in qd.static(qd.ndrange(2, 2)):
            node = base + qd.Vector([u, v])
            if _valid(node, self) and self.mass[b, node[0], node[1]] > 1.0e-12:
                weight = _weight(frac, u, v)
                height += weight * self.height_mass[b, node[0], node[1]] / self.mass[b, node[0], node[1]]
                surface_velocity += weight * self.momentum[b, node[0], node[1]] / self.mass[b, node[0], node[1]]
                weight_sum += weight
        if weight_sum > 0.5 and self.bread_stress > 0.0:
            height /= weight_sum
            surface_velocity /= weight_sum
            gap = pos[2] - height - 0.5 * (self.spacing + self.bread_spacing)
            if -self.spacing < gap < self.bread_contact_range:
                dv = _impulse_velocity(
                    vel - surface_velocity,
                    qd.Vector([0.0, 0.0, 1.0]),
                    gap,
                    self.bread_stress,
                    self.bread_contact_range,
                    self.bread_slip_time,
                    self.bread_shear_stress,
                    self,
                )
                vel += dv
                impulse = self.info[i].mass * dv
                for u, v in qd.static(qd.ndrange(2, 2)):
                    node = base + qd.Vector([u, v])
                    if _valid(node, self) and self.mass[b, node[0], node[1]] > 1.0e-12:
                        self.reaction[b, node[0], node[1]] -= _weight(frac, u, v) / weight_sum * impulse

        # Exact closest point on the oriented rectangular blade; only butter
        # receives this wet-contact law, not the bread or wooden board.
        blade_rotation = qd.Matrix.identity(gs.qd_float, 3)
        local = qd.Vector.zero(gs.qd_float, 3)
        blade_velocity = qd.Vector.zero(gs.qd_float, 3)
        if qd.static(is_prescribed):
            local = pos - center
            blade_velocity = velocity
        else:
            blade_rotation = gu.qd_quat_to_R(qd.Vector([quaternion[b, axis] for axis in range(4)]), gs.EPS)
            offset = pos - qd.Vector([center[b, axis] for axis in range(3)])
            local = blade_rotation.transpose() @ offset
            blade_velocity = qd.Vector([velocity[b, axis] for axis in range(3)])
            blade_velocity += qd.Vector([angular_velocity[b, axis] for axis in range(3)]).cross(offset)
        closest = qd.min(qd.max(local, -self.half_blade), self.half_blade)
        delta = local - closest
        distance = delta.norm(1.0e-12)
        blade_gap = distance - 0.5 * self.spacing
        if distance > 1.0e-6 and blade_gap < self.blade_contact_range and self.blade_stress > 0.0:
            n = delta / distance
            normal = blade_rotation @ n
            vel += _impulse_velocity(
                vel - blade_velocity,
                normal,
                blade_gap,
                self.blade_stress,
                self.blade_contact_range,
                self.blade_slip_time,
                self.blade_shear_stress,
                self,
            )
        # The MPM grid contact can admit particle centers into a thin,
        # moving rigid box. Enforce a one-sided normal velocity at the
        # blade surface, including for centers already inside the box.
        # This is a contact response, not a particle position edit.
        if self.blade_normal_relaxation > 0.0:
            signed_distance = distance
            contact_normal_local = delta / qd.max(distance, 1.0e-12)
            if distance <= 1.0e-6:
                face_distance = self.half_blade - qd.abs(local)
                signed_distance = -qd.min(qd.min(face_distance[0], face_distance[1]), face_distance[2])
                if face_distance[2] <= face_distance[0] and face_distance[2] <= face_distance[1]:
                    contact_normal_local = qd.Vector([0.0, 0.0, qd.select(local[2] >= 0.0, 1.0, -1.0)])
                elif face_distance[0] <= face_distance[1]:
                    contact_normal_local = qd.Vector([qd.select(local[0] >= 0.0, 1.0, -1.0), 0.0, 0.0])
                else:
                    contact_normal_local = qd.Vector([0.0, qd.select(local[1] >= 0.0, 1.0, -1.0), 0.0])
            center_gap = signed_distance - self.blade_contact_margin
            if center_gap < 0.0:
                contact_normal = blade_rotation @ contact_normal_local
                target_speed = qd.min(
                    -center_gap * self.blade_normal_relaxation / self.dt,
                    self.blade_max_separation_speed,
                )
                normal_speed = (vel - blade_velocity).dot(contact_normal)
                if normal_speed < target_speed:
                    vel += (target_speed - normal_speed) * contact_normal
        self.particles[f, i, b].vel = vel

    # The opposite bread impulse uses the same weights and masses as the
    # scatter, preserving the pair's total linear momentum.
    for b, a in qd.ndrange(self.n_envs, self.surface_ids.shape[0]):
        i = self.surface_ids[a]
        base, frac = _cell(self.particles[f, i, b].pos, self)
        dv = qd.Vector.zero(gs.qd_float, 3)
        for u, v in qd.static(qd.ndrange(2, 2)):
            node = base + qd.Vector([u, v])
            if _valid(node, self) and self.mass[b, node[0], node[1]] > 1.0e-12:
                dv += _weight(frac, u, v) * self.reaction[b, node[0], node[1]] / self.mass[b, node[0], node[1]]
        self.particles[f, i, b].vel += dv
