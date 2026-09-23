import numpy as np
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
        # optional concentration init (multiflow demo): a per-entity constant `c_init` takes
        # precedence; otherwise the two-phase `c_init_z_mid` split (c = 1 below z_mid, 0 above);
        # material without either keeps the previous all-zero behavior.
        if self._material.c_init is not None:
            c_init = np.full(self._n_particles, self._material.c_init, dtype=gs.np_float)
        elif self._material.c_init_z_mid is None:
            c_init = np.zeros(self._n_particles, dtype=gs.np_float)
        else:
            c_init = (self._particles[:, 2] < self._material.c_init_z_mid).astype(gs.np_float)
        self._solver._kernel_add_particles(
            self._sim.cur_substep_local,
            self.active,
            self._particle_start,
            self._n_particles,
            self._material.rho,
            c_init,
            self._particles,
        )

    @qd.kernel
    def get_frame_c(
        self,
        c: qd.types.ndarray(),  # shape [B, n_particles]
    ):
        """Retrieve particle concentrations for the given frame."""
        for i_p_, i_b in qd.ndrange(self.n_particles, self._sim._B):
            i_p = i_p_ + self._particle_start
            c[i_b, i_p_] = self.solver.particles[i_p, i_b].c

    @gs.assert_built
    def get_state(self):
        state = PBSTFEntityState(self, self.sim.cur_step_global)
        self.get_frame(self.sim.cur_substep_local, state.pos, state.vel)
        self.get_frame_c(state.c)
        self._queried_states.append(state)
        return state

    def _get_morph_identifier(self) -> str:
        return f"pbstf_{ParticleEntity._get_morph_identifier(self)}"
