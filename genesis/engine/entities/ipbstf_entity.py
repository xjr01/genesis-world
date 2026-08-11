import genesis as gs
from genesis.engine.states.entities import IPBSTFEntityState

from .particle_entity import ParticleEntity
from .pbstf_entity import PBSTFEntity


class IPBSTFEntity(PBSTFEntity):
    """Particle entity simulated by the implicit position-based surface-tension fluid (IPBSTF) solver."""

    def _add_particles_to_solver(self):
        self._solver._kernel_add_particles(
            self._sim.cur_substep_local,
            self.active,
            self._particle_start,
            self._n_particles,
            self._material.is_fixed,
            self._particles,
        )

    @gs.assert_built
    def get_state(self):
        state = IPBSTFEntityState(self, self.sim.cur_step_global)
        self.get_frame(self.sim.cur_substep_local, state.pos, state.vel)
        self._queried_states.append(state)
        return state

    def _get_morph_identifier(self) -> str:
        return f"ipbstf_{ParticleEntity._get_morph_identifier(self)}"
