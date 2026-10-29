import numpy as np

import numba

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import AutoInitMeta, V


class VBDConfig(metaclass=AutoInitMeta):
    contact_d_hat: float
    contact_search_radius: float
    friction_epsilon: float
    contact_stiffness: float
    shell_contact_stiffness: float
    convergence_tolerance: float
    is_contact_diagnostics_enabled: bool
    is_line_search_enabled: bool
    line_search_iterations: int
    n_vertex_threads: int
    contact_relaxation: float
    is_shell_previous: bool
    shell_step_size: float
    shell_ground_stiffness: float
    shell_ground_d_hat: float
    shell_ground_friction: float
    shell_ground_friction_epsilon: float
    has_shell: bool = False
    max_contact_thickness: float = 0.0


class ColorData(metaclass=AutoInitMeta):
    indices: qd.Tensor
    offsets: tuple[int, ...]


class IterationData(metaclass=AutoInitMeta):
    # Status entries hold the rebuild request, completed sweep count and active flag for a submission batch.
    status: qd.Tensor
    max_displacement: qd.Tensor | None


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
    vel: qd.Tensor
    prev: qd.Tensor
    collision_origin: qd.Tensor
    target: qd.Tensor
    next_pos: qd.Tensor
    forces: qd.Tensor
    is_fixed: qd.Tensor
    fixed_pos: qd.Tensor
    bound: qd.Tensor
    mass: qd.Tensor
    radius: qd.Tensor
    is_volume: qd.Tensor
    friction: qd.Tensor
    entity_idx: qd.Tensor
    elem_offsets: qd.Tensor
    elem_entries: qd.Tensor
    hinge_offsets: qd.Tensor
    hinge_entries: qd.Tensor | None


class ElementData(metaclass=AutoInitMeta):
    indices: qd.Tensor
    gradient: qd.Tensor
    measure: qd.Tensor
    mu: qd.Tensor
    lam: qd.Tensor
    damping: qd.Tensor
    is_volume: qd.Tensor
    is_log_neohookean: qd.Tensor | None
    is_plastic: qd.Tensor
    yield_strain: qd.Tensor
    hardening: qd.Tensor
    plastic_inverse: qd.Tensor
    plastic_strain: qd.Tensor
    has_shell: bool
    has_plastic: bool
    has_log_neohookean: bool


class HingeData(metaclass=AutoInitMeta):
    indices: qd.Tensor | None
    stiffness: qd.Tensor | None
    cotangent: qd.Tensor | None
    is_quadratic: qd.Tensor | None
    damping: qd.Tensor | None
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


class CollisionData(metaclass=AutoInitMeta):
    surface_indices: qd.Tensor
    faces: qd.Tensor
    edges: qd.Tensor
    face_edges: qd.Tensor
    edge_faces: qd.Tensor
    vertex_faces: qd.Tensor
    neighbor_offsets: qd.Tensor
    neighbors: qd.Tensor
    neighbor_edges: qd.Tensor | None
    # Components hold minimum clearance bounds over every pair and over cross-entity pairs, respectively.
    vertices_bound: qd.Tensor | None
    faces_bound: qd.Tensor | None
    edges_bound: qd.Tensor | None
    face_tets: qd.Tensor
    tet_indices: qd.Tensor | None
    # Reference volume scales the containment determinant threshold with each tetrahedron's size.
    tet_min_determinant: qd.Tensor | None
    tet_neighbors: qd.Tensor | None
    intersected_tets: qd.Tensor
    contact_faces: qd.Tensor
    translation: qd.Tensor | None
    translation_bound: qd.Tensor | None
    plane_clearance: qd.Tensor | None
    # Bits mark candidate pairs within an entity (1) and between entities (2).
    contact_flags: qd.Tensor | None
    entity_mass: qd.Tensor | None
    n_faces: int
    n_edges: int
    n_tets: int
    has_shell: bool
