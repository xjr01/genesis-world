import math

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.particle_entity import ParticleEntity


class DEMEntity(ParticleEntity):
    """
    Entity for DEM (Discrete Element Method) particles.

    Particles are sampled on a face-centered cubic (FCC) lattice whose nearest-neighbor distance is
    `2.01 * radius`, matching the reference implementation's volume sampler. This guarantees zero initial
    overlap between grains, which the strictly elastic (undamped) contact model requires for stability.

    Parameters
    ----------
    scene : Scene
        The scene object that this entity belongs to.
    solver : DEMSolver
        The DEM solver responsible for simulating the entity's particles.
    material : DEM.Base
        The material definition, including density and contact properties.
    morph : Morph
        Geometry or volumetric shape used for sampling particles. Only primitive morphs are supported.
    surface : Surface
        Surface material or texture information associated with the entity.
    particle_size : float
        Size of each DEM particle (diameter).
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
        Sample particles inside the primitive morph, following the reference volume sampler. Both modes enforce
        a nearest-neighbor distance of `2.01 * radius`, guaranteeing zero initial overlap between grains, which
        the strictly elastic (undamped) contact model requires for stability:

        - 'fcc': face-centered cubic lattice, cell size `2.01 * radius * sqrt(2)`, each cell contributing its
          lower corner plus its three lower face centers.
        - 'poisson': Poisson-disk sampling (Bridson-style), ported from the reference PoissonDiskSampler.
        """
        if not isinstance(self._morph, gs.options.morphs.Primitive):
            gs.raise_exception(f"DEMEntity only supports primitive morphs. Got: {self._morph}.")

        radius = self._particle_size / 2.0
        # C++ VolumeSampler / PoissonDiskSampler both use latticeConstant = 2.01 * radius as the min distance
        min_dist = 2.01 * radius

        is_inside = self._make_inside_test()
        if self._material.sampler == "fcc":
            particles = self._sample_fcc(self._vmesh.trimesh, min_dist, is_inside)
        else:
            particles = self._sample_poisson(
                self._vmesh.trimesh, min_dist, np.random.default_rng(self._material.sampler_seed), is_inside
            )
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

        gs.logger.info(f"Sampled ~~<{self._n_particles:,}>~~ DEM particles ({self._material.sampler} sampler).")

    def _make_inside_test(self):
        """
        Build a vectorized inside-surface test on local-frame points. Analytic tests are used for Box, Sphere
        and Cylinder morphs, matching the analytic implicit surfaces of the reference implementation; other
        primitives fall back to `trimesh.contains`.
        """
        lower, upper = self._vmesh.trimesh.bounds
        center = (lower + upper) * 0.5

        if isinstance(self._morph, gs.options.morphs.Box):
            half = (upper - lower) * 0.5

            def is_inside(points):
                return (np.abs(points - center) <= half).all(axis=1)

        elif isinstance(self._morph, gs.options.morphs.Sphere):
            radius = float(self._morph.radius)

            def is_inside(points):
                return np.linalg.norm(points - center, axis=1) <= radius

        elif isinstance(self._morph, gs.options.morphs.Cylinder):
            radius = float(self._morph.radius)
            half_height = (upper[2] - lower[2]) * 0.5

            def is_inside(points):
                radial = np.linalg.norm(points[:, :2] - center[:2], axis=1)
                return (radial <= radius) & (np.abs(points[:, 2] - center[2]) <= half_height)

        else:
            trimesh = self._vmesh.trimesh

            def is_inside(points):
                return trimesh.contains(points)

        return is_inside

    @staticmethod
    def _sample_fcc(trimesh, min_dist, is_inside):
        # C++ VolumeSampler: cell size = latticeConstant * sqrt(2); the four sites per cell form an FCC lattice
        cell_size = min_dist * math.sqrt(2.0)
        half_cell = 0.5 * cell_size

        lower, upper = trimesh.bounds
        n_cells = np.maximum(np.ceil((upper - lower) / cell_size).astype(gs.np_int), 1) + 1
        cell_idx = np.indices(tuple(n_cells)).reshape(3, -1).T
        cell_origins = lower + cell_idx * cell_size
        fcc_offsets = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, half_cell, half_cell],
                [half_cell, 0.0, half_cell],
                [half_cell, half_cell, 0.0],
            ],
            dtype=gs.np_float,
        )
        candidates = (cell_origins[:, None, :] + fcc_offsets[None, :, :]).reshape(-1, 3)
        return candidates[is_inside(candidates)]

    @staticmethod
    def _sample_poisson(trimesh, min_dist, rng, is_inside):
        # C++ PoissonDiskSampler: background grid with cell = min_dist / sqrt(3), one point per cell; candidates
        # are drawn on the spherical shell [min_dist, 2 * min_dist] around an active point, up to 300 tries each
        lower, upper = trimesh.bounds
        cell_size = min_dist / math.sqrt(3.0)
        grid_size = np.maximum(((upper - lower) / cell_size).astype(gs.np_int), 1)
        grid = np.full(tuple(grid_size[::-1]), -1, dtype=np.int64)  # z, y, x indexing as in C++
        grid_points = []
        process_list = []

        def grid_idx(p):
            return ((p - lower) / cell_size).astype(gs.np_int)

        def add_to_grid(p):
            gi = grid_idx(p)
            grid[gi[2], gi[1], gi[0]] = len(grid_points)
            grid_points.append(p)

        def is_far_enough(p):
            gi = grid_idx(p)
            lo = np.maximum(gi - 2, 0)
            hi = np.minimum(gi + 3, grid_size)
            idxs = np.unique(grid[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]])
            idxs = idxs[idxs != -1]
            if idxs.size == 0:
                return True
            neighbors = np.array([grid_points[i] for i in idxs])
            return np.all(np.linalg.norm(neighbors - p, axis=1) >= min_dist)

        p0 = (lower + upper) * 0.5
        # C++ adds the AABB center unconditionally; requiring it inside the surface avoids a spurious grain
        if not is_inside(p0[None, :])[0]:
            gs.raise_exception("Poisson sampler initial point (morph AABB center) is outside the morph surface.")
        process_list.append(p0)
        add_to_grid(p0)

        while process_list:
            idx = int(rng.integers(len(process_list)))
            point = process_list[idx]
            found = False
            for _ in range(300):
                angle1 = rng.uniform() * 2.0 * math.pi
                angle2 = math.acos(1.0 - 2.0 * rng.uniform())
                dist = min_dist + rng.uniform() * min_dist
                candidate = point + dist * np.array(
                    [
                        math.cos(angle1) * math.sin(angle2),
                        math.sin(angle1) * math.sin(angle2),
                        math.cos(angle2),
                    ]
                )
                gi = grid_idx(candidate)
                if np.any(gi < 0) or np.any(gi >= grid_size):
                    continue
                if not is_inside(candidate[None, :])[0]:
                    continue
                if not is_far_enough(candidate):
                    continue
                process_list.append(candidate)
                add_to_grid(candidate)
                found = True
                break
            if not found:
                process_list.pop(idx)

        # C++ collects the points referenced by the grid (at most one per cell)
        return np.array([grid_points[idx] for idx in grid.flat if idx != -1], dtype=gs.np_float)

    def _add_particles_to_solver(self):
        self.solver._kernel_add_particles(
            particle_start=self._particle_start,
            n_particles=self._n_particles,
            positions=self._particles,
            radius=self._particle_size / 2.0,
            density=self._material.rho,
        )

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
    def get_particles_pos(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        poss = torch.zeros((len(envs_idx), self._n_particles, 3), dtype=gs.tc_float, device=gs.device)
        self.solver._kernel_get_particles_pos(self._particle_start, self._n_particles, envs_idx, poss)
        return poss

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
    def get_particles_vel(self, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        vels = torch.zeros((len(envs_idx), self._n_particles, 3), dtype=gs.tc_float, device=gs.device)
        self.solver._kernel_get_particles_vel(self._particle_start, self._n_particles, envs_idx, vels)
        return vels

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
