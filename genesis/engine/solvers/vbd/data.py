import numpy as np

import numba

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import AutoInitMeta, V


class VBDConfig(metaclass=AutoInitMeta):
    contact_d_hat: float
    friction_epsilon: float
    contact_stiffness: float
    line_search_iterations: int


def array_to_field(values, dtype, element_shape=()):
    """Upload an exactly sized array; empty components are represented by None."""
    shape = values.shape[: values.ndim - len(element_shape)]
    if not values.size:
        return None
    field_type = dtype
    if len(element_shape) == 1:
        field_type = qd.types.vector(*element_shape, dtype)
    elif len(element_shape) == 2:
        field_type = qd.types.matrix(*element_shape, dtype)
    field = V(field_type, shape)
    field.from_numpy(values)
    return field


def incidence(indices, n_vertices):
    """Return compressed incidence offsets and sorted entries for a connectivity array."""
    entries = np.argsort(indices, axis=None, kind="stable")
    offsets = np.concatenate((np.zeros(1, dtype=gs.np_int), np.cumsum(np.bincount(indices.flat, minlength=n_vertices))))
    return offsets, entries


@numba.njit(cache=True)
def color_vertices(offsets, neighbors):
    """Greedily color a vertex adjacency graph on a single thread."""
    colors = np.full(len(offsets) - 1, -1)
    for vertex in range(len(colors)):
        color = 0
        while True:
            is_used = False
            for adjacent in range(offsets[vertex], offsets[vertex + 1]):
                if colors[neighbors[adjacent]] == color:
                    is_used = True
                    break
            if not is_used:
                break
            color += 1
        colors[vertex] = color
    return colors


class VertexData(metaclass=AutoInitMeta):
    pos: qd.Tensor
    rest_pos: qd.Tensor
    vel: qd.Tensor
    prev: qd.Tensor
    collision_origin: qd.Tensor
    target: qd.Tensor
    next_pos: qd.Tensor
    forces: qd.Tensor
    is_fixed: qd.Tensor
    fixed_pos: qd.Tensor
    bound: qd.Tensor
    motion_budget: qd.Tensor
    mass: qd.Tensor
    radius: qd.Tensor
    edge_length: qd.Tensor
    friction: qd.Tensor
    entity_idx: qd.Tensor
    elem_offsets: qd.Tensor
    elem_entries: qd.Tensor
    hinge_offsets: qd.Tensor
    hinge_entries: qd.Tensor | None
    neighbor_offsets: qd.Tensor
    neighbors: qd.Tensor
    entity_mass: qd.Tensor


class ElementData(metaclass=AutoInitMeta):
    indices: qd.Tensor
    gradient: qd.Tensor
    measure: qd.Tensor
    mu: qd.Tensor
    lam: qd.Tensor
    is_volume: qd.Tensor
    is_plastic: qd.Tensor
    yield_strain: qd.Tensor
    hardening: qd.Tensor
    plastic_inverse: qd.Tensor
    plastic_strain: qd.Tensor
    has_shell: bool
    has_plastic: bool


class HingeData(metaclass=AutoInitMeta):
    indices: qd.Tensor | None
    stiffness: qd.Tensor | None
    rest: qd.Tensor | None
    strain: qd.Tensor | None
    is_plastic: qd.Tensor | None
    yield_angle: qd.Tensor | None
    hardening: qd.Tensor | None
    n_hinges: int


class PlaneData(metaclass=AutoInitMeta):
    normal: qd.Tensor | None
    offset: qd.Tensor | None
    n_planes: int
