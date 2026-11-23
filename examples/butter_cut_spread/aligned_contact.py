"""Cohesive contact on the column-wise top surface of an authored porous bread mesh."""

import numpy as np

import genesis as gs
from examples.butter_spreading.contact import ButterContact
from genesis.utils.misc import tensor_to_array


class AuthoredBreadContact(ButterContact):
    def __init__(self, bread, butter, blade, **options):
        super().__init__(bread, butter, blade, **options)
        positions = tensor_to_array(bread.get_particles_pos())
        columns = np.round((positions[:, :2] - positions[:, :2].min(0)) / bread.particle_size).astype(np.int64)
        _, labels = np.unique(columns, axis=0, return_inverse=True)
        order = np.lexsort((positions[:, 2], labels))
        ends = np.r_[labels[order][1:] != labels[order][:-1], True]
        indices = order[ends]
        self.surface_ids = gs.qd.field(gs.qd_int, shape=len(indices))
        self.surface_ids.from_numpy((indices + bread.particle_start).astype(gs.np_int))
