from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from examples.multiphysics.butter_spreading import (
    ButterSpreadingContactConfig,
    ButterSpreadingMaterialConfig,
    ButterSpreadingScenarioConfig,
    knife_pose,
)
from examples.multiphysics.butter_spreading.contact import ButterContact


class _Particles:
    def __init__(self, positions, velocities, particle_size, density):
        self.positions = positions
        self.velocities = velocities
        self.particle_size = particle_size
        self.material = SimpleNamespace(rho=density)
        self.n_particles = positions.shape[-2]

    def get_particles_pos(self):
        return self.positions

    def get_particles_vel(self):
        return self.velocities

    def set_particles_vel(self, velocities):
        self.velocities = velocities

    def get_mass(self):
        return self.positions.new_full((self.positions.shape[0],), self.n_particles * self.particle_size**3)


class _Blade:
    def __init__(self, n_envs):
        self.pos = torch.tensor([[0.0, 0.0, 1.0]]).repeat(n_envs, 1)
        self.quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n_envs, 1)
        self.vel = torch.zeros((n_envs, 3))
        self.ang = torch.zeros((n_envs, 3))

    def get_pos(self, *, relative):
        return self.pos

    def get_quat(self, *, relative):
        return self.quat

    def get_vel(self):
        return self.vel

    def get_ang(self):
        return self.ang


def test_butter_spreading_material_conversion():
    material = ButterSpreadingMaterialConfig()

    assert material.butter_youngs_modulus == pytest.approx(57446.808510638296)
    assert material.butter_poissons_ratio == pytest.approx(0.43617021276595747)
    assert material.butter_consistency == 25.0
    assert material.butter_flow_exponent == 0.5
    assert material.bread_compaction_yield_pressure == 650.0
    assert material.bread_shear_hardening == 7000.0


def test_butter_spreading_contact_calibration():
    contact = ButterSpreadingContactConfig()

    assert contact.butter_contact_range == pytest.approx(0.002916666666666667)
    assert contact.bread_contact_range == pytest.approx(0.0015)
    assert contact.blade_contact_margin == pytest.approx(0.001875)
    assert contact.is_equilibrium_adhesion is True


def test_butter_asset_changes_are_independent_from_solver_parameters():
    config = ButterSpreadingScenarioConfig()
    resized_blade = replace(config, assets=replace(config.assets, blade_size=(0.03, 0.095, 0.005)))

    assert resized_blade.solver == config.solver
    assert resized_blade.material == config.material
    assert resized_blade.assets.blade_size == (0.03, 0.095, 0.005)


def test_butter_spreading_knife_trajectory_is_continuous():
    config = ButterSpreadingScenarioConfig()
    task = config.task
    epsilon = config.solver.dt

    for transition in (
        task.press_end_time,
        task.acceleration_end_time,
        task.deceleration_start_time,
        task.spread_end_time,
    ):
        before = knife_pose(task, transition - epsilon)
        after = knife_pose(task, transition + epsilon)
        assert before == pytest.approx(after, abs=1e-4)

    assert knife_pose(task, 0.0) == (task.press_end_x, 0.0, task.start_clearance)
    assert knife_pose(task, task.lift_end_time) == (task.spread_end_x, 0.0, task.lift_clearance)


def test_butter_task_origin_tracks_bread_geometry():
    config = ButterSpreadingScenarioConfig()
    moved = replace(
        config.assets,
        bread_center=(0.03, -0.02, 0.04),
        bread_size=(0.16, 0.12, 0.02),
    )

    assert moved.task_origin == (0.03, -0.02, 0.05)


def test_butter_contact_uses_public_batched_entity_state():
    bread_positions = torch.tensor(
        [
            [[-0.001, -0.001, 0.0], [-0.001, 0.001, 0.0], [0.001, -0.001, 0.0], [0.001, 0.001, 0.0]],
            [[-0.001, -0.001, 0.0], [-0.001, 0.001, 0.0], [0.001, -0.001, 0.0], [0.001, 0.001, 0.0]],
        ]
    )
    bread = _Particles(bread_positions, torch.zeros_like(bread_positions), 0.002, 1.0)
    butter_positions = torch.tensor([[[0.0, 0.0, 0.002]], [[0.0, 0.0, 0.002]]])
    butter_velocities = torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    butter = _Particles(butter_positions, butter_velocities, 0.001, 1.0)
    contact = ButterContact(
        bread,
        butter,
        _Blade(n_envs=2),
        spacing=0.001,
        dt=0.0001,
        blade_size=(0.02, 0.02, 0.002),
        lower_bound=(-0.01, -0.01, -0.01),
        upper_bound=(0.01, 0.01, 0.02),
        bread_stress=1.0,
        blade_stress=1.0,
        bread_contact_range=0.003,
        bread_slip_time=0.001,
        bread_shear_stress=1.0,
        blade_contact_range=0.003,
        blade_slip_time=0.001,
        blade_shear_stress=1.0,
        blade_normal_relaxation=0.0,
        blade_max_separation_speed=0.35,
        blade_contact_margin=0.001,
        is_equilibrium_adhesion=True,
    )

    contact.apply()

    assert (butter.velocities[..., 0] < 1.0).all()
    assert (butter.velocities[..., 2] < 0.0).all()
    assert (torch.linalg.vector_norm(bread.velocities, dim=-1).sum(dim=-1) > 0.0).all()
