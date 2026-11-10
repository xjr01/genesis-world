import genesis as gs
from genesis.engine.states.entities import PBSTFEntityState

from .particle_concentration import initial_concentration
from .particle_entity import ParticleEntity
from .sph_entity import SPHEntity


class PBSTFEntity(SPHEntity):
    """Particle entity simulated by :class:`PBSTFSolver`."""

    def init_sampler(self):
        if self._material.sampler == "staggered":
            self.sampler = "staggered"
        else:
            super().init_sampler()

    def _add_particles_to_solver(self):
        c_init = initial_concentration(self._particles, self.material.c_init, self.material.c_init_z_mid)
        self._solver._kernel_add_particles(
            self._sim.cur_substep_local,
            self.active,
            self._particle_start,
            self._n_particles,
            self._material.rho,
            c_init,
            self._particles,
        )

    @gs.assert_built
    def fix_particles(self, particles_idx_local=None, envs_idx=None, zero_velocity=True):
        """Prescribe particle motion using their velocities, optionally zeroing them to hold positions.

        ``particles_idx_local`` selects particles within this entity and ``envs_idx`` selects environments;
        each defaults to all. Use ``set_particles_vel`` to move fixed particles and ``release_particle`` to free them.
        """
        if zero_velocity:
            self.set_particles_vel(0.0, particles_idx_local, envs_idx)
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        self.solver.set_particles_fixed(particles_idx_local + self._particle_start, envs_idx, is_fixed=True)

    @gs.assert_built
    def release_particle(self, particles_idx_local=None, envs_idx=None):
        """Restore free motion for the selected entity-local particles and environments, defaulting to all."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        self.solver.set_particles_fixed(particles_idx_local + self._particle_start, envs_idx, is_fixed=False)

    @gs.assert_built
    def get_state(self):
        state = PBSTFEntityState(self, self.sim.cur_step_global)
        self.get_frame(self.sim.cur_substep_local, state.pos, state.vel)
        state.c[:] = self.get_particles_concentration()
        self._queried_states.append(state)
        return state

    def _get_morph_identifier(self) -> str:
        return f"pbstf_{ParticleEntity._get_morph_identifier(self)}"
