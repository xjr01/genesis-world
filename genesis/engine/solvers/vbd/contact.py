import quadrants as qd

import genesis as gs
from genesis.utils.array_class import ErrorCode, V_ANNOTATION

from .physics import barrier, contact_response, local_response


@qd.func
def segment_parameter(point, first, second):
    edge = second - first
    return qd.min(1.0, qd.max(0.0, (point - first).dot(edge) / qd.max(edge.norm_sqr(), 1e-30)))


@qd.func
def triangle_weights(point, a, b, c):
    e0, e1, delta = b - a, c - a, point - a
    d00, d01, d11 = e0.dot(e0), e0.dot(e1), e1.dot(e1)
    denominator = qd.max(d00 * d11 - d01**2, 1e-30)
    v = (d11 * delta.dot(e0) - d01 * delta.dot(e1)) / denominator
    w = (d00 * delta.dot(e1) - d01 * delta.dot(e0)) / denominator
    weights = qd.Vector([1.0 - v - w, v, w])
    if weights.min() < 0.0:
        t0, t1, t2 = segment_parameter(point, a, b), segment_parameter(point, b, c), segment_parameter(point, c, a)
        r0 = point - a - t0 * (b - a)
        r1 = point - b - t1 * (c - b)
        r2 = point - c - t2 * (a - c)
        weights = qd.Vector([1.0 - t0, t0, 0.0])
        if r1.norm_sqr() < r0.norm_sqr() and r1.norm_sqr() <= r2.norm_sqr():
            weights = qd.Vector([0.0, 1.0 - t1, t1])
        elif r2.norm_sqr() < r0.norm_sqr():
            weights = qd.Vector([t2, 0.0, 1.0 - t2])
    return weights


@qd.func
def edge_weights(a, b, c, d):
    e0, e1, delta = b - a, d - c, a - c
    aa, bb, cc = e0.dot(e0), e0.dot(e1), e1.dot(e1)
    dd, ee = e0.dot(delta), e1.dot(delta)
    determinant = aa * cc - bb**2
    s = qd.cast(0.0, gs.qd_float)
    if determinant > 1e-12 * aa * cc:
        s = qd.min(1.0, qd.max(0.0, (bb * ee - cc * dd) / determinant))
    t = (bb * s + ee) / qd.max(cc, 1e-30)
    if t < 0.0:
        t = 0.0
        s = qd.min(1.0, qd.max(0.0, -dd / qd.max(aa, 1e-30)))
    elif t > 1.0:
        t = 1.0
        s = qd.min(1.0, qd.max(0.0, (bb - dd) / qd.max(aa, 1e-30)))
    return qd.Vector([1.0 - s, s, t - 1.0, -t])


@qd.func
def next_node(node, batch, root, bvh: V_ANNOTATION):
    parent = bvh.nodes[batch, node].parent
    while node != root and parent >= 0 and bvh.nodes[batch, parent].right == node:
        node = parent
        parent = bvh.nodes[batch, node].parent
    following = -1
    if node != root and parent >= 0:
        following = bvh.nodes[batch, parent].right
    return following


@qd.kernel
def update_bounds(
    primitives: qd.Tensor,
    mean_displacement: qd.Tensor,
    vertices: V_ANNOTATION,
    bvh: V_ANNOTATION,
    d_hat: float,
    is_initial: qd.template(),
    has_multiple_entities: qd.template(),
):
    # Independent entity translations require absolute swept bounds for cross-entity candidate completeness.
    if qd.static(has_multiple_entities):
        for i_entity, i_b in mean_displacement:
            mean_displacement[i_entity, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_v, i_b in vertices.pos:
        translation = mean_displacement[vertices.entity_idx[i_v], i_b]
        displacement = vertices.target[i_v, i_b] - vertices.pos[i_v, i_b] - translation
        if qd.static(is_initial):
            displacement = vertices.next_pos[i_v, i_b] - vertices.pos[i_v, i_b] - translation
        vertices.motion_budget[i_v, i_b] = qd.max(0.5 * d_hat, 1.25 * displacement.norm())
        # Bounded free-vertex searches keep large force predictions from expanding every primitive's neighborhood.
        if not is_initial and not vertices.is_fixed[i_v, i_b]:
            limit = qd.max(2.0 * d_hat, 0.25 * vertices.edge_length[i_v])
            vertices.motion_budget[i_v, i_b] = qd.min(vertices.motion_budget[i_v, i_b], limit)
        vertices.bound[i_v, i_b] = 0.9 * vertices.motion_budget[i_v, i_b]
        vertices.collision_origin[i_v, i_b] = vertices.pos[i_v, i_b]
    for i_p, i_b in qd.ndrange(primitives.shape[0], bvh.n_batches):
        indices = primitives[i_p]
        lower = vertices.pos[indices[0], i_b]
        upper = lower
        margin = vertices.motion_budget[indices[0], i_b] + vertices.radius[indices[0]]
        for corner in qd.static(range(1, 3)):
            pos = vertices.pos[indices[corner], i_b]
            lower = qd.min(lower, pos)
            upper = qd.max(upper, pos)
            margin = qd.max(margin, vertices.motion_budget[indices[corner], i_b] + vertices.radius[indices[corner]])
        bvh.aabbs[i_b, i_p].min = lower - margin
        bvh.aabbs[i_b, i_p].max = upper + margin


@qd.kernel
def find_contacts(
    surface_indices: qd.Tensor,
    primitives: qd.Tensor,
    offsets: qd.types.ndarray(),
    pairs: qd.types.ndarray(),
    batches: qd.types.ndarray(),
    kinds: qd.types.ndarray(),
    weights_out: qd.types.ndarray(),
    normals: qd.types.ndarray(),
    rest_gaps: qd.types.ndarray(),
    counts: qd.types.ndarray(),
    initial_scale: qd.Tensor,
    vertices: V_ANNOTATION,
    bvh: V_ANNOTATION,
    n_faces: int,
    is_fill: qd.template(),
    is_initial: qd.template(),
    errno: qd.Tensor,
):
    for i_q, i_b in qd.ndrange(counts.shape[0] // bvh.n_batches, bvh.n_batches):
        n_surface = surface_indices.shape[0]
        i_primitive = n_faces + i_q - n_surface
        lower = qd.Vector.zero(gs.qd_float, 3)
        upper = lower
        root = bvh.roots_idx[0]
        margin = qd.cast(0.0, gs.qd_float)
        if i_q < n_surface:
            lower = vertices.pos[surface_indices[i_q], i_b]
            upper = lower
            margin = vertices.motion_budget[surface_indices[i_q], i_b] + vertices.radius[surface_indices[i_q]]
        else:
            lower = bvh.aabbs[i_b, i_primitive].min
            upper = bvh.aabbs[i_b, i_primitive].max
            root = bvh.roots_idx[1]
        node = root
        if errno[None] != 0:
            node = -1
        count = 0
        while node >= 0:
            bound = bvh.nodes[i_b, node].bound
            has_overlap = (lower - margin <= bound.max).all() and (upper + margin >= bound.min).all()
            left = bvh.nodes[i_b, node].left
            if has_overlap and left >= 0:
                node = left
            else:
                if has_overlap:
                    candidate = bvh.leaves_idx[i_b, node - bvh.n_aabbs + 1]
                    ids = qd.Vector.zero(gs.qd_int, 4)
                    is_eligible = True
                    if i_q < n_surface:
                        ids[0] = surface_indices[i_q]
                        for corner in qd.static(range(3)):
                            ids[corner + 1] = primitives[candidate][corner]
                        for neighbor in range(vertices.neighbor_offsets[ids[0]], vertices.neighbor_offsets[ids[0] + 1]):
                            for corner in qd.static(range(1, 4)):
                                if vertices.neighbors[neighbor] == ids[corner]:
                                    is_eligible = False
                        for corner in qd.static(range(1, 4)):
                            if ids[0] == ids[corner]:
                                is_eligible = False
                    else:
                        ids = qd.Vector(
                            [
                                primitives[i_primitive][0],
                                primitives[i_primitive][1],
                                primitives[candidate][0],
                                primitives[candidate][1],
                            ]
                        )
                        is_eligible = candidate > i_primitive
                        for corner in qd.static(range(2)):
                            for neighbor in range(
                                vertices.neighbor_offsets[ids[corner]], vertices.neighbor_offsets[ids[corner] + 1]
                            ):
                                if vertices.neighbors[neighbor] == ids[2] or vertices.neighbors[neighbor] == ids[3]:
                                    is_eligible = False
                            if ids[corner] == ids[2] or ids[corner] == ids[3]:
                                is_eligible = False
                    if is_eligible:
                        a, b = vertices.pos[ids[0], i_b], vertices.pos[ids[1], i_b]
                        c, d = vertices.pos[ids[2], i_b], vertices.pos[ids[3], i_b]
                        weights = edge_weights(a, b, c, d)
                        if i_q < n_surface:
                            bary = triangle_weights(a, b, c, d)
                            weights = qd.Vector([1.0, -bary[0], -bary[1], -bary[2]])
                        delta = weights[0] * a + weights[1] * b + weights[2] * c + weights[3] * d
                        radius = qd.cast(0.0, gs.qd_float)
                        for corner in qd.static(range(4)):
                            radius += qd.abs(weights[corner]) * vertices.radius[ids[corner]]
                        gap = delta.norm() - radius
                        reach = qd.max(vertices.motion_budget[ids[0], i_b], vertices.motion_budget[ids[1], i_b])
                        reach += qd.max(vertices.motion_budget[ids[2], i_b], vertices.motion_budget[ids[3], i_b])
                        if i_q < n_surface:
                            reach = vertices.motion_budget[ids[0], i_b] + qd.max(
                                vertices.motion_budget[ids[1], i_b],
                                vertices.motion_budget[ids[2], i_b],
                                vertices.motion_budget[ids[3], i_b],
                            )
                        if gap < reach:
                            if gap <= 0.0:
                                qd.atomic_or(errno[None], ErrorCode.INVALID_VBD_CONTACT)
                            for corner in qd.static(range(4)):
                                qd.atomic_min(vertices.bound[ids[corner], i_b], 0.45 * qd.max(gap, 0.0))
                            if qd.static(is_initial):
                                motion = qd.Matrix.zero(gs.qd_float, 4, 3)
                                for corner in qd.static(range(4)):
                                    motion[corner, :] = (
                                        vertices.next_pos[ids[corner], i_b] - vertices.pos[ids[corner], i_b]
                                    )
                                relative_speed = qd.cast(0.0, gs.qd_float)
                                for first in qd.static(range(4)):
                                    for second in qd.static(range(first + 1, 4)):
                                        relative_speed = qd.max(
                                            relative_speed, (motion[first, :] - motion[second, :]).norm()
                                        )
                                for corner in qd.static(range(4)):
                                    qd.atomic_min(
                                        initial_scale[vertices.entity_idx[ids[corner]], i_b],
                                        0.9 * qd.max(gap, 0.0) / qd.max(relative_speed, 1e-30),
                                    )
                            if qd.static(is_fill):
                                index = offsets[i_q * bvh.n_batches + i_b] + count
                                batches[index] = i_b
                                kinds[index] = i_q < n_surface
                                rest_gaps[index] = 1e30
                                has_same_entity = True
                                for corner in qd.static(range(1, 4)):
                                    if vertices.entity_idx[ids[corner]] != vertices.entity_idx[ids[0]]:
                                        has_same_entity = False
                                if has_same_entity:
                                    a, b = vertices.rest_pos[ids[0]], vertices.rest_pos[ids[1]]
                                    c, d = vertices.rest_pos[ids[2]], vertices.rest_pos[ids[3]]
                                    rest_weights = edge_weights(a, b, c, d)
                                    if i_q < n_surface:
                                        bary = triangle_weights(a, b, c, d)
                                        rest_weights = qd.Vector([1.0, -bary[0], -bary[1], -bary[2]])
                                    rest_delta = rest_weights[0] * a + rest_weights[1] * b
                                    rest_delta += rest_weights[2] * c + rest_weights[3] * d
                                    rest_radius = qd.cast(0.0, gs.qd_float)
                                    for corner in qd.static(range(4)):
                                        rest_radius += qd.abs(rest_weights[corner]) * vertices.radius[ids[corner]]
                                    rest_gaps[index] = qd.max(rest_delta.norm() - rest_radius, 1e-30)
                                for corner in qd.static(range(4)):
                                    pairs[index, corner] = ids[corner]
                                    weights_out[index, corner] = weights[corner]
                                for axis in qd.static(range(3)):
                                    normals[index, axis] = delta[axis] / qd.max(delta.norm(), 1e-30)
                            count += 1
                node = next_node(node, i_b, root, bvh)
        counts[i_q * bvh.n_batches + i_b] = count


@qd.kernel
def propagate_initialization(
    pairs: qd.types.ndarray(),
    batches: qd.types.ndarray(),
    scale: qd.Tensor,
    has_change: qd.Tensor,
    vertices: V_ANNOTATION,
):
    for index in range(pairs.shape[0]):
        i_b = batches[index]
        fraction = qd.cast(1.0, gs.qd_float)
        for corner in qd.static(range(4)):
            fraction = qd.min(fraction, scale[vertices.entity_idx[pairs[index, corner]], i_b])
        for corner in qd.static(range(4)):
            i_entity = vertices.entity_idx[pairs[index, corner]]
            previous = qd.atomic_min(scale[i_entity, i_b], fraction)
            if fraction < previous:
                has_change[None] = 1


@qd.func(requires_top_level=True)
def refresh_contacts(
    pairs: qd.types.ndarray(),
    batches: qd.types.ndarray(),
    kinds: qd.types.ndarray(),
    weights: qd.types.ndarray(),
    normals: qd.types.ndarray(),
    vertices: V_ANNOTATION,
    errno: qd.Tensor,
):
    for i_v, i_b in vertices.pos:
        # This displacement budget keeps omitted pairs separated throughout the substep.
        budget = (
            0.9 * vertices.motion_budget[i_v, i_b]
            - (vertices.pos[i_v, i_b] - vertices.collision_origin[i_v, i_b]).norm()
        )
        vertices.bound[i_v, i_b] = qd.max(0.0, budget)
    for index in range(pairs.shape[0]):
        i_b = batches[index]
        a, b = vertices.pos[pairs[index, 0], i_b], vertices.pos[pairs[index, 1], i_b]
        c, d = vertices.pos[pairs[index, 2], i_b], vertices.pos[pairs[index, 3], i_b]
        bary = edge_weights(a, b, c, d)
        if kinds[index]:
            face_bary = triangle_weights(a, b, c, d)
            bary = qd.Vector([1.0, -face_bary[0], -face_bary[1], -face_bary[2]])
        delta = bary[0] * a + bary[1] * b + bary[2] * c + bary[3] * d
        radius = qd.cast(0.0, gs.qd_float)
        for corner in qd.static(range(4)):
            radius += qd.abs(bary[corner]) * vertices.radius[pairs[index, corner]]
            weights[index, corner] = bary[corner]
        gap = delta.norm() - radius
        if gap <= 0.0:
            qd.atomic_or(errno[None], ErrorCode.INVALID_VBD_CONTACT)
        for corner in qd.static(range(4)):
            qd.atomic_min(vertices.bound[pairs[index, corner], i_b], 0.45 * qd.max(gap, 0.0))
        for axis in qd.static(range(3)):
            normals[index, axis] = delta[axis] / qd.max(delta.norm(), 1e-30)


@qd.func
def self_response(
    i_v,
    i_b,
    i_t,
    offsets: qd.template(),
    entries: qd.template(),
    pairs: qd.template(),
    trial,
    dt,
    weights: qd.template(),
    normals: qd.template(),
    rest_gaps: qd.template(),
    vertices: V_ANNOTATION,
    options: V_ANNOTATION,
):
    energy = qd.cast(0.0, gs.qd_float)
    gradient = qd.Vector.zero(gs.qd_float, 3)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    vertex_slot = i_v * vertices.pos.shape[1] + i_b
    i_a = offsets[vertex_slot] + i_t
    while i_a < offsets[vertex_slot + 1]:
        entry = entries[i_a]
        index, local_idx = entry // 4, entry % 4
        separation = qd.Vector.zero(gs.qd_float, 3)
        displacement = separation
        normal = qd.Vector([normals[index, 0], normals[index, 1], normals[index, 2]])
        radius, stiffness, friction = 0.0, 0.0, 0.0
        initial_separation = separation
        for corner in qd.static(range(4)):
            vertex = pairs[index, corner]
            weight = weights[index, corner]
            pos = vertices.pos[vertex, i_b]
            if corner == local_idx:
                pos = trial
            separation += weight * pos
            initial_separation += weight * vertices.prev[vertex, i_b]
            displacement += weight * (pos - vertices.prev[vertex, i_b])
            radius += qd.abs(weight) * vertices.radius[vertex]
            stiffness = qd.max(stiffness, vertices.mass[vertex] / dt**2)
            friction += 0.25 * vertices.friction[vertex]
        stiffness *= options.contact_stiffness
        # Reference clearances keep densely sampled surfaces free of artificial repulsion at rest.
        d_hat = qd.min(options.contact_d_hat, rest_gaps[index])
        _, slope, _ = barrier(initial_separation.dot(normal) - radius, stiffness, d_hat)
        value, force, matrix = contact_response(
            displacement,
            normal,
            separation.dot(normal) - radius,
            stiffness,
            friction,
            -slope,
            d_hat,
            options.friction_epsilon * dt,
        )
        weight = weights[index, local_idx]
        energy += value
        gradient += weight * force
        hessian += weight**2 * matrix
        i_a += options.n_vertex_threads
    return energy, gradient, hessian


@qd.func
def vertex_response(
    i_v,
    i_b,
    i_t,
    offsets: qd.template(),
    entries: qd.template(),
    pairs: qd.template(),
    trial,
    dt,
    is_active,
    weights: qd.template(),
    normals: qd.template(),
    rest_gaps: qd.template(),
    vertices: V_ANNOTATION,
    elements: V_ANNOTATION,
    hinges: V_ANNOTATION,
    planes: V_ANNOTATION,
    options: V_ANNOTATION,
):
    energy = qd.cast(0.0, gs.qd_float)
    gradient = qd.Vector.zero(gs.qd_float, 3)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    if is_active:
        energy, gradient, hessian = local_response(
            i_v, i_b, i_t, trial, dt, vertices, elements, hinges, planes, options
        )
        value, force, matrix = self_response(
            i_v, i_b, i_t, offsets, entries, pairs, trial, dt, weights, normals, rest_gaps, vertices, options
        )
        energy += value
        gradient += force
        hessian += matrix
    if qd.static(options.n_vertex_threads > 1):
        # Broadcasting the leader's sums gives every lane identical backtracking decisions.
        log2_size = qd.static(options.n_vertex_threads.bit_length() - 1)
        i_l = qd.u32(qd.simt.subgroup.invocation_id() // options.n_vertex_threads * options.n_vertex_threads)
        energy = qd.simt.subgroup.shuffle(qd.simt.subgroup.reduce_add_tiled(energy, log2_size), i_l)
        for i_d in qd.static(range(3)):
            gradient[i_d] = qd.simt.subgroup.shuffle(qd.simt.subgroup.reduce_add_tiled(gradient[i_d], log2_size), i_l)
        for i_d, j_d in qd.static(qd.ndrange(3, 3)):
            hessian[i_d, j_d] = qd.simt.subgroup.shuffle(
                qd.simt.subgroup.reduce_add_tiled(hessian[i_d, j_d], log2_size), i_l
            )
    return energy, gradient, hessian


@qd.func(requires_top_level=True)
def solve_color(
    color_indices: qd.Tensor,
    color_start: int,
    color_end: int,
    offsets: qd.types.ndarray(),
    entries: qd.types.ndarray(),
    pairs: qd.types.ndarray(),
    dt: float,
    weights: qd.types.ndarray(),
    normals: qd.types.ndarray(),
    rest_gaps: qd.types.ndarray(),
    iteration_stats: qd.Tensor,
    vertices: V_ANNOTATION,
    elements: V_ANNOTATION,
    hinges: V_ANNOTATION,
    planes: V_ANNOTATION,
    options: V_ANNOTATION,
    errno: qd.Tensor,
):
    n_threads = qd.static(options.n_vertex_threads)
    n_lanes = qd.static(32 if n_threads > 1 else 1)
    n_slots = (color_end - color_start) * vertices.pos.shape[1]
    qd.loop_config(name="vbd_local_solve", block_dim=128)
    for i_s in range((n_slots * n_threads + n_lanes - 1) // n_lanes * n_lanes):
        i_t = i_s % n_threads
        i_s_ = qd.min(i_s // n_threads, n_slots - 1)
        i_b = i_s_ % vertices.pos.shape[1]
        i_v = color_indices[color_start + i_s_ // vertices.pos.shape[1]]
        is_in_range = i_s // n_threads < n_slots
        is_valid = is_in_range and errno[None] == 0
        is_free = is_valid and not vertices.is_fixed[i_v, i_b]
        pos = vertices.pos[i_v, i_b]
        next_pos = pos
        budget = qd.max(
            0.0, 0.9 * vertices.motion_budget[i_v, i_b] - (pos - vertices.collision_origin[i_v, i_b]).norm()
        )
        direction = qd.Vector.zero(gs.qd_float, 3)
        scale = qd.cast(1.0, gs.qd_float)
        is_searching = False
        # Fixed vertices and padding keep their lanes active for every subgroup collective.
        energy, gradient, hessian = vertex_response(
            i_v,
            i_b,
            i_t,
            offsets,
            entries,
            pairs,
            pos,
            dt,
            is_free,
            weights,
            normals,
            rest_gaps,
            vertices,
            elements,
            hinges,
            planes,
            options,
        )
        if is_valid:
            if vertices.is_fixed[i_v, i_b]:
                direction = vertices.fixed_pos[i_v, i_b] - pos
            else:
                direction = -(hessian / hessian.trace()).inverse() @ (gradient / hessian.trace())
                if qd.math.isnan(direction).any() or qd.math.isinf(direction).any():
                    if i_t == 0:
                        qd.atomic_or(errno[None], ErrorCode.INVALID_VBD_STATE)
                else:
                    is_searching = True
            if vertices.is_fixed[i_v, i_b] or is_searching:
                scale = qd.min(1.0, vertices.bound[i_v, i_b] / qd.max(direction.norm(), 1e-30))
                if qd.static(planes.n_planes > 0):
                    for i_p in range(planes.n_planes):
                        normal = planes.normal[i_p]
                        gap = pos.dot(normal) - planes.offset[i_p] - vertices.radius[i_v]
                        if direction.dot(normal) < 0.0:
                            scale = qd.min(scale, 0.9 * qd.max(gap, 0.0) / -direction.dot(normal))
                if vertices.is_fixed[i_v, i_b] or qd.static(not options.is_line_search_enabled):
                    next_pos = pos + scale * direction
        if qd.static(options.is_line_search_enabled):
            for iteration in range(options.line_search_iterations):
                has_searching = is_searching
                if qd.static(n_threads > 1):
                    has_searching = qd.simt.subgroup.any_true(is_searching)
                if not has_searching:
                    break
                trial = pos + scale * direction
                trial_energy, _trial_gradient, _trial_hessian = vertex_response(
                    i_v,
                    i_b,
                    i_t,
                    offsets,
                    entries,
                    pairs,
                    trial,
                    dt,
                    is_searching,
                    weights,
                    normals,
                    rest_gaps,
                    vertices,
                    elements,
                    hinges,
                    planes,
                    options,
                )
                if is_searching:
                    if trial_energy <= energy + 1e-4 * scale * gradient.dot(direction):
                        next_pos = trial
                        is_searching = False
                    else:
                        scale *= 0.5
        if is_in_range and i_t == 0:
            if (
                direction.norm() > budget
                and vertices.bound[i_v, i_b] >= budget
                and budget < 0.2 * vertices.motion_budget[i_v, i_b]
            ):
                qd.atomic_max(iteration_stats[0], 1.0)
            vertices.next_pos[i_v, i_b] = next_pos
            qd.atomic_max(iteration_stats[1], (next_pos - pos).norm())


@qd.func(requires_top_level=True)
def apply_color(color_indices: qd.Tensor, color_start: int, color_end: int, vertices: V_ANNOTATION):
    for slot, i_b in qd.ndrange((color_start, color_end), vertices.pos.shape[1]):
        i_v = color_indices[slot]
        vertices.pos[i_v, i_b] = vertices.next_pos[i_v, i_b]


@qd.kernel(graph=True)
def solve_colors(
    offsets: qd.types.ndarray(),
    entries: qd.types.ndarray(),
    pairs: qd.types.ndarray(),
    batches: qd.types.ndarray(),
    kinds: qd.types.ndarray(),
    dt: float,
    weights: qd.types.ndarray(),
    normals: qd.types.ndarray(),
    rest_gaps: qd.types.ndarray(),
    iteration_stats: qd.Tensor,
    vertices: V_ANNOTATION,
    elements: V_ANNOTATION,
    hinges: V_ANNOTATION,
    planes: V_ANNOTATION,
    colors: V_ANNOTATION,
    options: V_ANNOTATION,
    errno: qd.Tensor,
):
    """Replay an ordered color sweep with contact geometry frozen for the sweep."""
    refresh_contacts(pairs, batches, kinds, weights, normals, vertices, errno)
    for i in range(2):
        iteration_stats[i] = 0.0
    # Each phase retains its global barrier before subsequent colors read the updated positions.
    for i_c in qd.static(range(len(colors.offsets) - 1)):
        solve_color(
            colors.indices,
            colors.offsets[i_c],
            colors.offsets[i_c + 1],
            offsets,
            entries,
            pairs,
            dt,
            weights,
            normals,
            rest_gaps,
            iteration_stats,
            vertices,
            elements,
            hinges,
            planes,
            options,
            errno,
        )
        apply_color(colors.indices, colors.offsets[i_c], colors.offsets[i_c + 1], vertices)
