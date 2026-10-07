import math
from dataclasses import dataclass

import torch

from genesis.utils import geom as gu


def _with_batch(tensor):
    return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor


@dataclass(frozen=True)
class _SurfaceGrid:
    mass: torch.Tensor
    height: torch.Tensor
    velocity: torch.Tensor
    indices: tuple[torch.Tensor, ...]
    weights: tuple[torch.Tensor, ...]
    valid: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _SurfaceSample:
    height: torch.Tensor
    velocity: torch.Tensor
    weight: torch.Tensor
    indices: tuple[torch.Tensor, ...]
    weights: tuple[torch.Tensor, ...]
    valid: tuple[torch.Tensor, ...]


class ButterContact:
    """Apply finite-range butter contact through public particle and rigid-entity APIs."""

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
        self._bread = bread
        self._butter = butter
        self._blade = blade
        self._dt = dt
        self._spacing = spacing
        self._bread_spacing = bread.particle_size
        self._bread_stress = bread_stress
        self._blade_stress = blade_stress
        self._bread_contact_range = bread_contact_range
        self._blade_contact_range = blade_contact_range
        self._bread_slip_time = bread_slip_time
        self._blade_slip_time = blade_slip_time
        self._bread_shear_stress = bread_shear_stress
        self._blade_shear_stress = blade_shear_stress
        self._blade_normal_relaxation = blade_normal_relaxation
        self._blade_max_separation_speed = blade_max_separation_speed
        self._blade_contact_margin = blade_contact_margin
        self._is_equilibrium_adhesion = is_equilibrium_adhesion

        bread_positions = _with_batch(bread.get_particles_pos())
        top = bread_positions[0, :, 2].max()
        self._surface_indices = torch.nonzero(
            bread_positions[0, :, 2] > top - self._bread_spacing * 0.55,
            as_tuple=False,
        ).squeeze(-1)
        self._origin = bread_positions.new_tensor(lower_bound[:2])
        self._res = tuple(
            math.ceil((upper - lower) / self._bread_spacing) + 2
            for lower, upper in zip(lower_bound[:2], upper_bound[:2])
        )
        grid_x_indices = torch.arange(self._res[0], device=bread_positions.device)
        grid_y_indices = torch.arange(self._res[1], device=bread_positions.device)
        self._grid_x = self._origin[0] + grid_x_indices * self._bread_spacing
        self._grid_y = self._origin[1] + grid_y_indices * self._bread_spacing
        self._n_grid_cells = self._res[0] * self._res[1]
        self._half_blade = bread_positions.new_tensor(blade_size) * 0.5
        self._bread_particle_mass = bread.get_mass() / bread.n_particles
        self._butter_particle_mass = butter.get_mass() / butter.n_particles

    def _corners(self, positions):
        uv = (positions[..., :2] - self._origin) / self._bread_spacing
        base_x = torch.searchsorted(self._grid_x, positions[..., 0].contiguous(), right=True) - 1
        base_y = torch.searchsorted(self._grid_y, positions[..., 1].contiguous(), right=True) - 1
        base = torch.stack((base_x, base_y), dim=-1)
        fraction = uv - base
        env_offset = torch.arange(positions.shape[0], device=positions.device)[:, None] * self._n_grid_cells
        indices = []
        weights = []
        valid = []
        for offset_x, offset_y in ((0, 0), (0, 1), (1, 0), (1, 1)):
            node_x = base[..., 0] + offset_x
            node_y = base[..., 1] + offset_y
            is_valid = (node_x >= 0) & (node_x < self._res[0]) & (node_y >= 0) & (node_y < self._res[1])
            node_x = node_x.clamp(0, self._res[0] - 1)
            node_y = node_y.clamp(0, self._res[1] - 1)
            indices.append(env_offset + node_x * self._res[1] + node_y)
            weight_x = fraction[..., 0] if offset_x else 1.0 - fraction[..., 0]
            weight_y = fraction[..., 1] if offset_y else 1.0 - fraction[..., 1]
            weights.append(weight_x * weight_y)
            valid.append(is_valid)
        return tuple(indices), tuple(weights), tuple(valid)

    def _build_surface_grid(self, bread_positions, bread_velocities):
        surface_positions = bread_positions[:, self._surface_indices]
        surface_velocities = bread_velocities[:, self._surface_indices]
        indices, weights, valid = self._corners(surface_positions)
        shape = (bread_positions.shape[0] * self._n_grid_cells,)
        mass = bread_positions.new_zeros(shape)
        height_mass = bread_positions.new_zeros(shape)
        momentum = bread_positions.new_zeros((*shape, 3))
        particle_mass = self._bread_particle_mass[:, None]
        for node_indices, node_weights, is_valid in zip(indices, weights, valid):
            weighted_mass = node_weights * particle_mass
            flat_indices = node_indices[is_valid]
            mass.index_add_(0, flat_indices, weighted_mass[is_valid])
            height_mass.index_add_(0, flat_indices, (weighted_mass * surface_positions[..., 2])[is_valid])
            momentum.index_add_(0, flat_indices, (weighted_mass[..., None] * surface_velocities)[is_valid])
        has_mass = mass > 1.0e-12
        height = torch.where(has_mass, height_mass / mass.clamp_min(1.0e-12), 0.0)
        velocity = torch.where(has_mass[:, None], momentum / mass.clamp_min(1.0e-12)[:, None], 0.0)
        return _SurfaceGrid(mass, height, velocity, indices, weights, valid)

    def _sample_surface(self, positions, grid):
        indices, weights, valid = self._corners(positions)
        height = positions.new_zeros(positions.shape[:-1])
        velocity = positions.new_zeros(positions.shape)
        weight_sum = positions.new_zeros(positions.shape[:-1])
        sample_valid = []
        sample_weights = []
        for node_indices, node_weights, is_valid in zip(indices, weights, valid):
            has_mass = is_valid & (grid.mass[node_indices] > 1.0e-12)
            weight = torch.where(has_mass, node_weights, 0.0)
            height += weight * grid.height[node_indices]
            velocity += weight[..., None] * grid.velocity[node_indices]
            weight_sum += weight
            sample_valid.append(has_mass)
            sample_weights.append(weight)
        denominator = weight_sum.clamp_min(1.0e-12)
        return _SurfaceSample(
            height / denominator,
            velocity / denominator[..., None],
            weight_sum,
            indices,
            tuple(sample_weights),
            tuple(sample_valid),
        )

    def _impulse_velocity(self, relative_velocity, normal, gap, stress, contact_range, slip_time, shear_stress):
        weight = (1.0 - torch.clamp_min(gap, 0.0) / contact_range).clamp_min(0.0).square()
        normal_weight = weight
        normal_depth = self._spacing
        tangent_depth = self._spacing
        if self._is_equilibrium_adhesion:
            separation = (gap / contact_range).clamp(0.0, 1.0)
            normal_weight = 4.0 * separation * (1.0 - separation)
            normal_depth = 2.0 * contact_range / 3.0
            tangent_depth = contact_range / 3.0 + 0.5 * self._spacing
        density = self._butter.material.rho
        normal_velocity = -normal * (stress * self._dt / (density * normal_depth) * normal_weight)[..., None]
        tangent = relative_velocity - (relative_velocity * normal).sum(dim=-1, keepdim=True) * normal
        speed = torch.linalg.vector_norm(tangent, dim=-1).clamp_min(1.0e-12)
        drag = torch.minimum(
            speed * (1.0 - math.exp(-self._dt / slip_time)),
            speed.new_full(speed.shape, shear_stress * self._dt / (density * tangent_depth)),
        )
        return normal_velocity - tangent / speed[..., None] * (drag * weight)[..., None]

    def _apply_blade_contact(self, positions, velocities):
        center = self._blade.get_pos(relative=False)
        quaternion = self._blade.get_quat(relative=False)
        linear_velocity = self._blade.get_vel()
        angular_velocity = self._blade.get_ang()
        if center.ndim == 1:
            center = center.unsqueeze(0)
            quaternion = quaternion.unsqueeze(0)
            linear_velocity = linear_velocity.unsqueeze(0)
            angular_velocity = angular_velocity.unsqueeze(0)
        rotation = gu.quat_to_R(quaternion)
        offset = positions - center[:, None]
        local = torch.einsum("bij,bnj->bni", rotation.transpose(-1, -2), offset)
        closest = torch.minimum(torch.maximum(local, -self._half_blade), self._half_blade)
        delta = local - closest
        distance = torch.linalg.vector_norm(delta, dim=-1)
        outside = distance > 1.0e-6
        normal_local = delta / distance.clamp_min(1.0e-12)[..., None]
        face_distance = self._half_blade - local.abs()
        inside_axis = face_distance.argmin(dim=-1)
        inside_normal = torch.zeros_like(local)
        inside_sign = torch.where(
            torch.gather(local, -1, inside_axis[..., None]).squeeze(-1) >= 0.0,
            1.0,
            -1.0,
        )
        inside_normal.scatter_(-1, inside_axis[..., None], inside_sign[..., None])
        normal_local = torch.where(outside[..., None], normal_local, inside_normal)
        signed_distance = torch.where(outside, distance, -face_distance.min(dim=-1).values)
        normal = torch.einsum("bij,bnj->bni", rotation, normal_local)
        blade_velocity = linear_velocity[:, None] + torch.linalg.cross(
            angular_velocity[:, None].expand_as(offset),
            offset,
            dim=-1,
        )
        gap = distance - 0.5 * self._spacing
        is_near = outside & (gap < self._blade_contact_range) & (self._blade_stress > 0.0)
        impulse_velocity = self._impulse_velocity(
            velocities - blade_velocity,
            normal,
            gap,
            self._blade_stress,
            self._blade_contact_range,
            self._blade_slip_time,
            self._blade_shear_stress,
        )
        velocities += torch.where(is_near[..., None], impulse_velocity, 0.0)

        center_gap = signed_distance - self._blade_contact_margin
        is_penetrating = center_gap < 0.0
        target_speed = torch.minimum(
            -center_gap * self._blade_normal_relaxation / self._dt,
            center_gap.new_full(center_gap.shape, self._blade_max_separation_speed),
        )
        normal_speed = ((velocities - blade_velocity) * normal).sum(dim=-1)
        correction = (target_speed - normal_speed).clamp_min(0.0)
        velocities += torch.where(is_penetrating[..., None], correction[..., None] * normal, 0.0)

    def apply(self):
        """Apply bread and blade contact to all environments before the next scene step."""
        bread_positions_raw = self._bread.get_particles_pos()
        bread_velocities_raw = self._bread.get_particles_vel()
        butter_positions_raw = self._butter.get_particles_pos()
        butter_velocities_raw = self._butter.get_particles_vel()
        bread_positions = _with_batch(bread_positions_raw)
        bread_velocities = _with_batch(bread_velocities_raw)
        butter_positions = _with_batch(butter_positions_raw)
        butter_velocities = _with_batch(butter_velocities_raw)

        grid = self._build_surface_grid(bread_positions, bread_velocities)
        sample = self._sample_surface(butter_positions, grid)
        gap = butter_positions[..., 2] - sample.height - 0.5 * (self._spacing + self._bread_spacing)
        normal = torch.zeros_like(butter_positions)
        normal[..., 2] = 1.0
        has_bread_contact = (
            (sample.weight > 0.5)
            & (gap > -self._spacing)
            & (gap < self._bread_contact_range)
            & (self._bread_stress > 0.0)
        )
        impulse_velocity = self._impulse_velocity(
            butter_velocities - sample.velocity,
            normal,
            gap,
            self._bread_stress,
            self._bread_contact_range,
            self._bread_slip_time,
            self._bread_shear_stress,
        )
        impulse_velocity = torch.where(has_bread_contact[..., None], impulse_velocity, 0.0)
        butter_velocities += impulse_velocity

        reaction = butter_positions.new_zeros((butter_positions.shape[0] * self._n_grid_cells, 3))
        impulse = self._butter_particle_mass[:, None, None] * impulse_velocity
        denominator = sample.weight.clamp_min(1.0e-12)
        for node_indices, node_weights, is_valid in zip(sample.indices, sample.weights, sample.valid):
            share = node_weights / denominator
            reaction.index_add_(0, node_indices[is_valid], (-share[..., None] * impulse)[is_valid])

        surface_velocity_change = bread_positions.new_zeros(
            (bread_positions.shape[0], self._surface_indices.shape[0], 3)
        )
        for node_indices, node_weights, is_valid in zip(grid.indices, grid.weights, grid.valid):
            has_mass = is_valid & (grid.mass[node_indices] > 1.0e-12)
            contribution = (
                node_weights[..., None] * reaction[node_indices] / grid.mass[node_indices].clamp_min(1.0e-12)[..., None]
            )
            surface_velocity_change += torch.where(has_mass[..., None], contribution, 0.0)
        bread_velocities[:, self._surface_indices] += surface_velocity_change

        self._apply_blade_contact(butter_positions, butter_velocities)
        self._bread.set_particles_vel(
            bread_velocities.squeeze(0) if bread_velocities_raw.ndim == 2 else bread_velocities
        )
        self._butter.set_particles_vel(
            butter_velocities.squeeze(0) if butter_velocities_raw.ndim == 2 else butter_velocities
        )

    def __call__(self):
        self.apply()
        return False
