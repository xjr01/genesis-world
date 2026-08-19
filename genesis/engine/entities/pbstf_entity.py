import quadrants as qd

import genesis as gs
from genesis.engine.states.entities import PBSTFEntityState

from .particle_entity import ParticleEntity
from .sph_entity import SPHEntity


@qd.data_oriented
class PBSTFEntity(SPHEntity):
    """Particle entity simulated by :class:`PBSTFSolver`."""

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
        fluid_particle_start=None,
        name: str | None = None,
    ):
        self._fluid_particle_start = particle_start if fluid_particle_start is None else fluid_particle_start
        super().__init__(
            scene,
            solver,
            material,
            morph,
            surface,
            particle_size,
            idx,
            particle_start,
            name=name,
        )

    def init_sampler(self):
        if self._material.sampler == "staggered":
            self.sampler = "staggered"
        else:
            super().init_sampler()

    def _add_particles_to_solver(self):
        self._solver._kernel_add_particles(
            self._sim.cur_substep_local,
            self.active,
            self._fluid_particle_start,
            self._n_particles,
            self._material.rho,
            self._particles,
        )

    @gs.assert_built
    def get_state(self):
        state = PBSTFEntityState(self, self.sim.cur_step_global)
        envs_idx = self._scene._sanitize_envs_idx(None)
        self._solver._kernel_get_particles_pos(self._fluid_particle_start, self._n_particles, envs_idx, state.pos)
        self._solver._kernel_get_particles_vel(self._fluid_particle_start, self._n_particles, envs_idx, state.vel)
        self._queried_states.append(state)
        return state

    @gs.assert_built
    def set_particles_pos(self, poss, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        poss = self._sanitize_particles_tensor(poss, gs.tc_float, particles_idx_local, envs_idx, (3,))
        self._solver._kernel_set_particles_pos(particles_idx_local + self._fluid_particle_start, envs_idx, poss)

    def get_particles_pos(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        poss = self._sanitize_particles_tensor(None, gs.tc_float, None, envs_idx, (3,))
        self._solver._kernel_get_particles_pos(self._fluid_particle_start, self._n_particles, envs_idx, poss)
        return poss[0] if self._scene.n_envs == 0 else poss

    def get_position(self, envs_idx=None):
        return self.get_particles_pos(envs_idx)

    @gs.assert_built
    def set_particles_vel(self, vels, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        vels = self._sanitize_particles_tensor(vels, gs.tc_float, particles_idx_local, envs_idx, (3,))
        self._solver._kernel_set_particles_vel(particles_idx_local + self._fluid_particle_start, envs_idx, vels)

    def get_particles_vel(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        vels = self._sanitize_particles_tensor(None, gs.tc_float, None, envs_idx, (3,))
        self._solver._kernel_get_particles_vel(self._fluid_particle_start, self._n_particles, envs_idx, vels)
        return vels[0] if self._scene.n_envs == 0 else vels

    @gs.assert_built
    def set_particles_active(self, actives, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        actives = self._sanitize_particles_tensor(actives, gs.tc_bool, particles_idx_local, envs_idx)
        self._solver._kernel_set_particles_active(particles_idx_local + self._fluid_particle_start, envs_idx, actives)

    def get_particles_active(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        actives = self._sanitize_particles_tensor(None, gs.tc_bool, None, envs_idx)
        self._solver._kernel_get_particles_active(self._fluid_particle_start, self._n_particles, envs_idx, actives)
        return actives[0] if self._scene.n_envs == 0 else actives

    @gs.assert_built
    def get_mass(self, envs_idx=None):
        """Return the active liquid mass in each selected environment."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        mass = gs.zeros((len(envs_idx),), dtype=gs.tc_float, requires_grad=False, scene=self._scene)
        self._solver._kernel_get_mass(self._fluid_particle_start, self._n_particles, mass, envs_idx)
        return mass

    def _get_morph_identifier(self) -> str:
        return f"pbstf_{ParticleEntity._get_morph_identifier(self)}"

    @property
    def fluid_particle_start(self):
        """Starting index in the solver's liquid-particle phase."""
        return self._fluid_particle_start
