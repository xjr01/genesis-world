import quadrants as qd

import genesis as gs
from genesis.engine.states.entities import PBSTFEntityState

from .particle_entity import ParticleEntity
from .sph_entity import SPHEntity


@qd.data_oriented
class PBSTFEntity(SPHEntity):
    """Particle entity simulated by :class:`PBSTFSolver`."""

    def init_sampler(self):
        if self._material.sampler == "staggered":
            self.sampler = "staggered"
        else:
            super().init_sampler()

    def _add_particles_to_solver(self):
        self._solver._kernel_add_particles(
            self._sim.cur_substep_local,
            self.active,
            self._particle_start,
            self._n_particles,
            self._material.rho,
            self._particles,
        )

    @gs.assert_built
    def get_state(self):
        state = PBSTFEntityState(self, self.sim.cur_step_global)
        self.get_frame(self.sim.cur_substep_local, state.pos, state.vel)
        self._queried_states.append(state)
        return state

    def _get_morph_identifier(self) -> str:
        return f"pbstf_{ParticleEntity._get_morph_identifier(self)}"
