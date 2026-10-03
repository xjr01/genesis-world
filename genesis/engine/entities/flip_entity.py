import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.particle_entity import ParticleEntity


class FLIPEntity(ParticleEntity):
    """
    Entity for FLIP (Fluid-Implicit-Particle) liquid particles.

    Particles are seeded on a stratified pattern, following the reference implementation's `SeedParticles`:
    the domain cell grid is refined by `seed_sub_factor` per axis, one jittered sample is placed per sub-cell,
    and samples are kept only when they lie inside the liquid morph with a margin of
    `0.874 * dx` (`1.01 * sqrt(3) / 2 * dx`), so the initial particle distribution is uniform and
    one particle radius away from the free surface.

    Parameters
    ----------
    scene : Scene
        The scene object that this entity belongs to.
    solver : FLIPSolver
        The FLIP solver responsible for simulating the entity's particles.
    material : FLIP.Base
        The material definition.
    morph : Morph
        Geometry or volumetric shape used for seeding particles. Only primitive morphs are supported.
    surface : Surface
        Surface material or texture information associated with the entity.
    particle_size : float
        Nominal size of each FLIP particle (the MAC grid spacing `dx`), used for the seeding margin.
    idx : int
        Index of this entity in the simulation.
    particle_start : int
        Global index offset for this entity's particles in the solver.
    name : str, optional
        Name of the entity.
    """

    def __init__(
        self,
        scene,
        solver,
        material,
        morph,
        surface,
        particle_size,
        idx,
        particle_start,
        name=None,
    ):
        super().__init__(
            scene=scene,
            solver=solver,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=particle_size,
            idx=idx,
            particle_start=particle_start,
            need_skinning=False,
            name=name,
        )

    def sample(self):
        """
        Stratified seeding inside the primitive morph, following the reference `SeedParticles`.
        """
        if not isinstance(self._morph, gs.options.morphs.Primitive):
            gs.raise_exception(f"FLIPEntity only supports primitive morphs. Got: {self._morph}.")

        dx = self._solver.dx
        sub = self._solver.seed_sub_factor
        # reference: m_ParticleRadFactor = 1.01 * sqrt(3) / 2
        margin = 1.01 * np.sqrt(3.0) / 2.0 * dx

        lower, upper = self._vmesh.trimesh.bounds
        n_sub = np.maximum(np.ceil((upper - lower) / (dx / sub)).astype(gs.np_int), 1)
        sub_idx = np.indices(tuple(n_sub)).reshape(3, -1).T
        # stratified: sub-cell center + uniform jitter of quarter sub-cell spacing (C++: Random() * spacing / 4)
        rng = np.random.default_rng(42)
        candidates = (
            lower
            + (sub_idx + 0.5) * (dx / sub)
            + (rng.random(sub_idx.shape[0] * 3).reshape(-1, 3) - 0.5) * (dx / sub) / 2.0
        )
        particles = candidates[self._inside_with_margin(candidates, margin)]

        if particles.shape[0] == 0:
            gs.raise_exception("Entity has zero particles.")

        # transform vmesh and particles by the morph pose (offset composed onto the morph pose)
        pos, quat = gu.transform_pos_quat_by_trans_quat(
            np.array(self._morph.offset_pos, dtype=gs.np_float),
            np.array(self._morph.offset_quat, dtype=gs.np_float),
            np.array(self._morph.pos, dtype=gs.np_float),
            np.array(self._morph.quat, dtype=gs.np_float),
        )
        self._vmesh.apply_transform(gu.trans_quat_to_T(pos, quat))
        particles = gu.transform_by_trans_quat(particles, pos, quat)

        if not self._solver.boundary.is_inside(particles):
            gs.raise_exception(
                "Entity has particles outside solver boundary.\n\nCurrent boundary:\n"
                f"{self._solver.boundary}\n\nEntity to be added:\nmin: {particles.min(0)}\nmax: {particles.max(0)}\n"
            )

        self._vverts = np.zeros((0, 3), dtype=gs.np_float)
        self._vfaces = np.zeros((0, 3), dtype=gs.np_int)

        self._particles = np.asarray(particles, dtype=gs.np_float, order="C")
        self._init_particles_offset = gs.tensor(self._particles) - gs.tensor(pos)
        self._n_particles = len(self._particles)

        gs.logger.info(f"Sampled ~~<{self._n_particles:,}>~~ FLIP particles (stratified {sub}^3 per cell).")

    def _inside_with_margin(self, points, margin):
        # analytic inside tests with a surface margin, matching the C++ analytic implicit surfaces
        lower, upper = self._vmesh.trimesh.bounds
        center = (lower + upper) * 0.5

        if isinstance(self._morph, gs.options.morphs.Box):
            half = np.maximum((upper - lower) * 0.5 - margin, 0.0)
            return (np.abs(points - center) <= half).all(axis=1)
        elif isinstance(self._morph, gs.options.morphs.Sphere):
            radius = max(float(self._morph.radius) - margin, 0.0)
            return np.linalg.norm(points - center, axis=1) <= radius
        elif isinstance(self._morph, gs.options.morphs.Cylinder):
            radius = max(float(self._morph.radius) - margin, 0.0)
            half_height = max((upper[2] - lower[2]) * 0.5 - margin, 0.0)
            radial = np.linalg.norm(points[:, :2] - center[:2], axis=1)
            return (radial <= radius) & (np.abs(points[:, 2] - center[2]) <= half_height)
        else:
            gs.raise_exception(f"Unsupported morph for FLIP seeding: {self._morph}.")

    def _add_particles_to_solver(self):
        self.solver._kernel_add_particles(
            particle_start=self._particle_start,
            n_particles=self._n_particles,
            positions=self._particles,
        )

    @gs.assert_built
    def get_particles_pos(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        poss = torch.zeros((len(envs_idx), self._n_particles, 3), dtype=gs.tc_float, device=gs.device)
        self.solver._kernel_get_particles_pos(self._particle_start, self._n_particles, envs_idx, poss)
        return poss

    @gs.assert_built
    def set_particles_pos(self, poss, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        particles_idx_local_ = particles_idx_local + self._particle_start
        self.solver._kernel_set_particles_pos(particles_idx_local_, envs_idx, poss)

    @gs.assert_built
    def _set_particles_pos_grad(self, poss_grad):
        pass

    @gs.assert_built
    def get_particles_vel(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        vels = torch.zeros((len(envs_idx), self._n_particles, 3), dtype=gs.tc_float, device=gs.device)
        self.solver._kernel_get_particles_vel(self._particle_start, self._n_particles, envs_idx, vels)
        return vels

    @gs.assert_built
    def set_particles_vel(self, vels, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        particles_idx_local_ = particles_idx_local + self._particle_start
        self.solver._kernel_set_particles_vel(particles_idx_local_, envs_idx, vels)

    @gs.assert_built
    def _set_particles_vel_grad(self, vels_grad):
        pass

    @gs.assert_built
    def set_particles_active(self, actives, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        particles_idx_local_ = particles_idx_local + self._particle_start
        self.solver._kernel_set_particles_active(particles_idx_local_, envs_idx, actives)

    @gs.assert_built
    def get_particles_active(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        actives = torch.zeros((len(envs_idx), self._n_particles), dtype=gs.tc_bool, device=gs.device)
        self.solver._kernel_get_particles_active(self._particle_start, self._n_particles, envs_idx, actives)
        return actives
