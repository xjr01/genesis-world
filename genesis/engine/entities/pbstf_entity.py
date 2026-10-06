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
    def get_state(self):
        state = PBSTFEntityState(self, self.sim.cur_step_global)
        self.get_frame(self.sim.cur_substep_local, state.pos, state.vel)
        state.c[:] = self.get_particles_concentration()
        self._queried_states.append(state)
        return state

    @gs.assert_built
    def get_particles_absorbed_collider_idx(self):
        """Return the absorbing static-collider index for each particle, or -1 while unabsorbed."""
        state = self.solver.get_state(self.sim.cur_substep_local)
        if state.absorbed_collider_idx is None:
            return None
        indices = state.absorbed_collider_idx[..., self.particle_start : self.particle_end]
        return indices if self.scene.n_envs else indices[0]

    def _get_morph_identifier(self) -> str:
        return f"pbstf_{ParticleEntity._get_morph_identifier(self)}"
