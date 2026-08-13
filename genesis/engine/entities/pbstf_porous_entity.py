import genesis as gs
from genesis.engine.states.entities import PBSTFPorousEntityState

from .particle_entity import ParticleEntity


class PBSTFPorousEntity(ParticleEntity):
    """Porous solid particle entity simulated by the position-based surface-tension fluid solver."""

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
        porous_particle_start,
        material_idx,
        vvert_start,
        name: str | None = None,
    ):
        self._porous_particle_start = porous_particle_start
        self._material_idx = material_idx
        self._rest_volume = 0.0
        is_skinning_enabled = surface.vis_mode == "visual"
        super().__init__(
            scene,
            solver,
            material,
            morph,
            surface,
            particle_size,
            idx,
            particle_start,
            vvert_start=vvert_start if is_skinning_enabled else -1,
            vface_start=-1,
            need_skinning=is_skinning_enabled,
            name=name,
        )
        if self._vmesh is None or not self._vmesh.is_watertight or self._vmesh.volume <= 0.0:
            gs.raise_exception("PBSTF porous elastic entities require a watertight volumetric morph.")
        self._rest_volume = float(self._vmesh.volume / self._n_particles)
        if is_skinning_enabled and self._n_particles < self._solver._n_vvert_supports:
            gs.raise_exception(
                "PBSTF porous visual skinning requires at least `VisOptions.n_support_neighbors` particles."
            )

    def init_sampler(self):
        self.sampler = "staggered"

    def _add_particles_to_solver(self):
        self._solver._add_porous_particles(
            self.active,
            self._porous_particle_start,
            self._n_particles,
            self._material_idx,
            self._material.rho,
            self._material.porosity,
            self._rest_volume,
            self._particles,
        )

    def _reset_grad(self):
        pass

    def add_grad_from_state(self, state):
        pass

    @gs.assert_built
    def get_state(self):
        state = PBSTFPorousEntityState(self, self.sim.cur_step_global)
        envs_idx = self._scene._sanitize_envs_idx(None)
        self._solver._get_porous_particles_frame(
            self._porous_particle_start,
            self._n_particles,
            envs_idx,
            state.pos,
            state.vel,
            state.active,
            state.is_fixed,
        )
        self._queried_states.append(state)
        return state

    @gs.assert_built
    def set_state(self, state, envs_idx=None):
        """Restore position, velocity, activity, and fixed status from a porous entity state."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        self._solver._set_porous_particles_frame(
            self._porous_particle_start,
            self._n_particles,
            envs_idx,
            state.pos,
            state.vel,
            state.active,
            state.is_fixed,
        )

    @gs.assert_built
    def set_particles_pos(self, poss, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        poss = self._sanitize_particles_tensor(poss, gs.tc_float, particles_idx_local, envs_idx, (3,))
        self._solver._set_porous_particles_pos(particles_idx_local + self._porous_particle_start, envs_idx, poss)

    def get_particles_pos(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        poss = self._sanitize_particles_tensor(None, gs.tc_float, None, envs_idx, (3,))
        self._solver._get_porous_particles_pos(self._porous_particle_start, self._n_particles, envs_idx, poss)
        return poss[0] if self._scene.n_envs == 0 else poss

    def get_position(self, envs_idx=None):
        return self.get_particles_pos(envs_idx)

    @gs.assert_built
    def set_particles_vel(self, vels, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        vels = self._sanitize_particles_tensor(vels, gs.tc_float, particles_idx_local, envs_idx, (3,))
        self._solver._set_porous_particles_vel(particles_idx_local + self._porous_particle_start, envs_idx, vels)

    def get_particles_vel(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        vels = self._sanitize_particles_tensor(None, gs.tc_float, None, envs_idx, (3,))
        self._solver._get_porous_particles_vel(self._porous_particle_start, self._n_particles, envs_idx, vels)
        return vels[0] if self._scene.n_envs == 0 else vels

    @gs.assert_built
    def set_particles_active(self, actives, particles_idx_local=None, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        actives = self._sanitize_particles_tensor(actives, gs.tc_bool, particles_idx_local, envs_idx)
        self._solver._set_porous_particles_active(particles_idx_local + self._porous_particle_start, envs_idx, actives)

    def get_particles_active(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        actives = self._sanitize_particles_tensor(None, gs.tc_bool, None, envs_idx)
        self._solver._get_porous_particles_active(self._porous_particle_start, self._n_particles, envs_idx, actives)
        return actives[0] if self._scene.n_envs == 0 else actives

    @gs.assert_built
    def fix_particles(self, particles_idx_local=None, envs_idx=None, is_velocity_zeroed=True):
        """Fix selected porous particles at positions supplied through subsequent position setters."""
        if is_velocity_zeroed:
            self.set_particles_vel(0.0, particles_idx_local, envs_idx)
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        self._solver._fix_porous_particles(particles_idx_local + self._porous_particle_start, envs_idx)

    @gs.assert_built
    def release_particle(self, particles_idx_local=None, envs_idx=None):
        """Release selected porous particles so elastic and porous forces can move them."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        particles_idx_local = self._sanitize_particles_idx_local(particles_idx_local, envs_idx)
        self._solver._release_porous_particles(particles_idx_local + self._porous_particle_start, envs_idx)

    def get_particles_is_fixed(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        is_fixed = self._sanitize_particles_tensor(None, gs.tc_bool, None, envs_idx)
        self._solver._get_porous_particles_fixed(self._porous_particle_start, self._n_particles, envs_idx, is_fixed)
        return is_fixed[0] if self._scene.n_envs == 0 else is_fixed

    def get_saturation(self, envs_idx=None):
        """Return current liquid saturation per porous particle."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        saturation = self._sanitize_particles_tensor(None, gs.tc_float, None, envs_idx)
        self._solver.get_porous_particles_saturation(
            self._porous_particle_start, self._n_particles, envs_idx, saturation
        )
        return saturation[0] if self._scene.n_envs == 0 else saturation

    def get_porosity(self, envs_idx=None):
        """Return current pore-volume fraction per porous particle."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        porosity = self._sanitize_particles_tensor(None, gs.tc_float, None, envs_idx)
        self._solver.get_porous_particles_porosity(self._porous_particle_start, self._n_particles, envs_idx, porosity)
        return porosity[0] if self._scene.n_envs == 0 else porosity

    def get_absorbed_fluid_volume(self, envs_idx=None):
        """Return the fluid volume represented by local saturation in each selected environment."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        volume = gs.zeros((len(envs_idx),), dtype=gs.tc_float, requires_grad=False, scene=self._scene)
        self._solver.get_absorbed_fluid_volume(self._porous_particle_start, self._n_particles, envs_idx, volume)
        return volume[0] if self._scene.n_envs == 0 else volume

    @gs.assert_built
    def get_mass(self, envs_idx=None):
        """Return the active solid-matrix mass in each selected environment."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        active = self._sanitize_particles_tensor(None, gs.tc_bool, None, envs_idx)
        self._solver._get_porous_particles_active(self._porous_particle_start, self._n_particles, envs_idx, active)
        particle_mass = (1.0 - self._material.porosity) * self._material.rho * self._rest_volume
        return active.sum(axis=-1) * particle_mass

    def _get_morph_identifier(self) -> str:
        return f"pbstf_porous_{ParticleEntity._get_morph_identifier(self)}"

    @property
    def porous_particle_start(self):
        """Starting index in the solver's porous-particle phase."""
        return self._porous_particle_start

    @property
    def rest_volume(self):
        """Reference volume represented by each uniformly sampled porous particle."""
        return self._rest_volume
