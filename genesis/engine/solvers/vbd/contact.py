import quadrants as qd

import genesis as gs
from genesis.utils.array_class import ErrorCode, V_ANNOTATION

from .collision import contact_geometry
from .physics import contact_response, local_response


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
    kinds: qd.template(),
    vertices: V_ANNOTATION,
    options: V_ANNOTATION,
):
    energy = qd.cast(0.0, gs.qd_float)
    gradient = qd.Vector.zero(gs.qd_float, 3)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    slot = i_v * vertices.pos.shape[1] + i_b
    i_a = offsets[slot] + i_t
    while i_a < offsets[slot + 1]:
        entry = qd.cast(entries[i_a], gs.qd_int)
        index, local_idx = entry // 4, entry % 4
        weights, normal, gap, displacement, is_active = contact_geometry(
            index, i_b, local_idx, trial, pairs, kinds, vertices
        )
        friction = 0.5 * (vertices.friction[pairs[index, 0]] + vertices.friction[pairs[index, 2]])
        is_shell = kinds[index] != 2
        stiffness = options.contact_stiffness
        if is_shell:
            stiffness = options.shell_contact_stiffness
        if is_active:
            value, force, matrix = contact_response(
                displacement,
                normal,
                gap,
                stiffness,
                friction,
                options.contact_d_hat,
                options.friction_epsilon * dt,
                is_shell,
            )
            weight = weights[local_idx]
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
    kinds: qd.template(),
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
            i_v, i_b, i_t, offsets, entries, pairs, trial, dt, kinds, vertices, options
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
    kinds: qd.types.ndarray(),
    dt: float,
    is_active: bool,
    iteration: V_ANNOTATION,
    vertices: V_ANNOTATION,
    elements: V_ANNOTATION,
    hinges: V_ANNOTATION,
    planes: V_ANNOTATION,
    options: V_ANNOTATION,
    errno: qd.Tensor,
):
    n_threads = qd.static(options.n_vertex_threads)
    n_lanes = qd.static(32 if n_threads > 1 else 1)
    n_slots = (color_end - color_start) * vertices.pos.shape[1] * is_active
    qd.loop_config(name="vbd_local_solve", block_dim=128)
    for i_s in range((n_slots * n_threads + n_lanes - 1) // n_lanes * n_lanes):
        i_t = i_s % n_threads
        i_s_ = qd.min(i_s // n_threads, n_slots - 1)
        i_b = i_s_ % vertices.pos.shape[1]
        i_v = color_indices[color_start + i_s_ // vertices.pos.shape[1]]
        is_in_range = i_s // n_threads < n_slots
        is_valid = is_in_range and (errno[None] & ErrorCode.INVALID_VBD_STATE) == 0
        is_free = is_valid and not vertices.is_fixed[i_v, i_b]
        pos = vertices.pos[i_v, i_b]
        next_pos = pos
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
            kinds,
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
                matrix = hessian / hessian.trace()
                terms = matrix[:, 0] * matrix[:, 1].cross(matrix[:, 2])
                determinant = terms.sum()
                # A relative determinant check rejects cancellation before inverting the local block.
                if determinant > 1e-5 * qd.abs(terms).sum():
                    direction = -matrix.inverse() @ (gradient / hessian.trace())
                    if qd.static(options.has_shell) and not vertices.is_volume[i_v]:
                        # Contact connects equal-colored vertices, so relax simultaneous shell corrections.
                        direction *= 0.5
                    is_searching = True
                if (
                    qd.math.isnan(determinant)
                    or qd.math.isinf(determinant)
                    or qd.math.isnan(gradient).any()
                    or qd.math.isinf(gradient).any()
                    or qd.math.isnan(direction).any()
                    or qd.math.isinf(direction).any()
                ):
                    if i_t == 0:
                        qd.atomic_or(errno[None], ErrorCode.INVALID_VBD_STATE)
                    is_searching = False
            if vertices.is_fixed[i_v, i_b] or is_searching:
                if qd.static(options.has_shell) and not vertices.is_fixed[i_v, i_b]:
                    accumulated = pos + direction - vertices.collision_origin[i_v, i_b]
                    if accumulated.norm() > vertices.bound[i_v, i_b]:
                        direction = (
                            vertices.collision_origin[i_v, i_b]
                            - pos
                            + accumulated * (vertices.bound[i_v, i_b] / qd.max(accumulated.norm(), 1e-30))
                        )
                        if i_t == 0:
                            qd.atomic_or(iteration.status[0], 1)
                if vertices.is_fixed[i_v, i_b] or qd.static(not options.is_line_search_enabled):
                    next_pos = pos + scale * direction
        if qd.static(options.is_line_search_enabled):
            for i_l in range(options.line_search_iterations):
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
                    kinds,
                    vertices,
                    elements,
                    hinges,
                    planes,
                    options,
                )
                if is_searching:
                    # Rounded geometric queries limit energy resolution near a stationary point.
                    roundoff = 64.0 * gs.EPS * qd.max(qd.abs(energy), qd.abs(trial_energy))
                    if trial_energy <= energy + 1e-4 * scale * gradient.dot(direction) + roundoff:
                        next_pos = trial
                        is_searching = False
                    else:
                        scale *= 0.5
        if is_in_range and i_t == 0:
            vertices.next_pos[i_v, i_b] = next_pos


@qd.func(requires_top_level=True)
def apply_color(
    color_indices: qd.Tensor,
    color_start: int,
    color_end: int,
    is_active: bool,
    iteration: V_ANNOTATION,
    vertices: V_ANNOTATION,
    options: V_ANNOTATION,
):
    n_lanes = qd.static(32 if options.n_vertex_threads > 1 else 1)
    n_slots = (color_end - color_start) * vertices.pos.shape[1] * is_active
    for i_s in range((n_slots + n_lanes - 1) // n_lanes * n_lanes):
        i_s_ = qd.min(i_s, n_slots - 1)
        i_b = i_s_ % vertices.pos.shape[1]
        i_v = color_indices[color_start + i_s_ // vertices.pos.shape[1]]
        is_in_range = i_s < n_slots
        if qd.static(options.convergence_tolerance > 0):
            displacement = qd.cast(0.0, gs.qd_float)
            if is_in_range:
                displacement = (vertices.next_pos[i_v, i_b] - vertices.pos[i_v, i_b]).norm()
            if qd.static(n_lanes > 1):
                displacement = qd.simt.subgroup.reduce_max(displacement)
            if i_s % n_lanes == 0:
                qd.atomic_max(iteration.max_displacement[None], displacement)
        if is_in_range:
            vertices.pos[i_v, i_b] = vertices.next_pos[i_v, i_b]


@qd.kernel(graph=True)
def solve_colors(
    offsets: qd.types.ndarray(),
    entries: qd.types.ndarray(),
    pairs: qd.types.ndarray(),
    kinds: qd.types.ndarray(),
    dt: float,
    is_reset: bool,
    iteration: V_ANNOTATION,
    vertices: V_ANNOTATION,
    elements: V_ANNOTATION,
    hinges: V_ANNOTATION,
    planes: V_ANNOTATION,
    colors: V_ANNOTATION,
    options: V_ANNOTATION,
    errno: qd.Tensor,
):
    """Replay a color sweep, stopping device updates when convergence or contact rebuilding requires host attention."""
    if is_reset:
        iteration.status[0] = 0
        iteration.status[1] = 0
        iteration.status[2] = 1
    is_active = iteration.status[2] != 0 and (errno[None] & ErrorCode.INVALID_VBD_STATE) == 0
    iteration.status[2] = is_active
    if qd.static(options.convergence_tolerance > 0):
        iteration.max_displacement[None] = 0.0
    # Each phase retains its global barrier before subsequent colors read the updated positions.
    for i_c in qd.static(range(len(colors.offsets) - 1)):
        solve_color(
            colors.indices,
            colors.offsets[i_c],
            colors.offsets[i_c + 1],
            offsets,
            entries,
            pairs,
            kinds,
            dt,
            is_active,
            iteration,
            vertices,
            elements,
            hinges,
            planes,
            options,
            errno,
        )
        apply_color(
            colors.indices, colors.offsets[i_c], colors.offsets[i_c + 1], is_active, iteration, vertices, options
        )
    is_converged = False
    if qd.static(options.convergence_tolerance > 0):
        is_converged = iteration.max_displacement[None] <= options.convergence_tolerance
    if is_active:
        iteration.status[1] += 1
        if iteration.status[0] or errno[None] & ErrorCode.INVALID_VBD_STATE or is_converged:
            iteration.status[2] = 0
