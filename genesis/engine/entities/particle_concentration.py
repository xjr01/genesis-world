import numpy as np

import quadrants as qd

from genesis.utils.array_class import V_ANNOTATION


def initial_concentration(positions, c_init, c_init_z_mid):
    """Return constant concentrations, or a binary split along world z when the constant is None."""
    if c_init is not None:
        return np.full(len(positions), c_init)
    if c_init_z_mid is None:
        return np.zeros(len(positions))
    return np.where(positions[:, 2] < c_init_z_mid, 1.0, 0.0)


@qd.kernel
def kernel_get_concentration(
    envs_idx: qd.types.ndarray(),
    particle_start: int,
    concentrations: qd.types.ndarray(),
    particles: V_ANNOTATION,
):
    for i_b, i_p in qd.ndrange(envs_idx.shape[0], concentrations.shape[1]):
        concentrations[i_b, i_p] = particles[particle_start + i_p, envs_idx[i_b]].c


@qd.kernel
def kernel_set_concentration(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    concentrations: qd.types.ndarray(),
    particles: V_ANNOTATION,
):
    for i_b, i_p in qd.ndrange(envs_idx.shape[0], particles_idx.shape[1]):
        particles[particles_idx[i_b, i_p], envs_idx[i_b]].c = concentrations[i_b, i_p]
