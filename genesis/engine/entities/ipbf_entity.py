import genesis as gs
from genesis.engine.states.entities import IPBFEntityState

from .particle_entity import ParticleEntity
from .sph_entity import SPHEntity


class IPBFEntity(SPHEntity):
    """Particle entity simulated by the implicit position-based fluid solver."""

    def _add_particles_to_solver(self):
        self._solver._kernel_add_particles(
            self._sim.cur_substep_local,
            self.active,
            self._particle_start,
            self._n_particles,
            self._material.rho,
            self._material.c_init,
            self._material.boundary_group,
            self._particles,
        )

    @gs.assert_built
    def get_state(self):
        state = IPBFEntityState(self, self.sim.cur_step_global)
        self.get_frame(self.sim.cur_substep_local, state.pos, state.vel)
        state.c[:] = self.get_particles_concentration()
        self._queried_states.append(state)
        return state

    def _get_morph_identifier(self):
        return f"ipbf_{ParticleEntity._get_morph_identifier(self)}"
