import quadrants as qd

import genesis as gs
from genesis.engine.bvh import point_aabb_distance_sqr
from genesis.utils.array_class import ErrorCode, V_ANNOTATION


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
def is_edge_contact_active(a, b, c, d):
    """Suppress the ill-conditioned closest-point direction of nearly parallel segments."""
    first, second = b - a, d - c
    return first.cross(second).norm_sqr() > 1e-9 * first.norm() * second.norm()


@qd.func
def contact_geometry(i_c, i_b, i_local, trial, pairs: qd.template(), kinds: qd.template(), vertices: V_ANNOTATION):
    """Evaluate contact geometry, relative motion and the tangent projection at a trial position."""
    positions = qd.Matrix.zero(gs.qd_float, 4, 3)
    for i in qd.static(range(4)):
        positions[i, :] = vertices.pos[pairs[i_c, i], i_b]
        if i == i_local:
            positions[i, :] = trial
    a, b, c, d = positions[0, :], positions[1, :], positions[2, :], positions[3, :]
    weights = edge_weights(a, b, c, d)
    if kinds[i_c] != 0:
        bary = triangle_weights(a, b, c, d)
        weights = qd.Vector([1.0, -bary[0], -bary[1], -bary[2]])
    separation, displacement = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
    radius = qd.cast(0.0, gs.qd_float)
    for i in qd.static(range(4)):
        i_v = pairs[i_c, i]
        separation += weights[i] * positions[i, :]
        displacement += weights[i] * (positions[i, :] - vertices.prev[i_v, i_b])
        radius += qd.abs(weights[i]) * vertices.radius[i_v]
    normal = separation / qd.max(separation.norm(), 1e-30)
    if kinds[i_c] == 2:
        face_normal = (c - b).cross(d - b).normalized(1e-30)
        if qd.min(-weights[1], -weights[2], -weights[3]) > 0.0 or separation.norm_sqr() < 1e-30:
            normal = face_normal
        elif normal.dot(face_normal) < 0.0:
            normal = -normal
    is_active = kinds[i_c] != 0 or is_edge_contact_active(a, b, c, d)
    tangent_projection = qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal)
    if kinds[i_c] != 2:
        edge = b - a
        if kinds[i_c] == 1:
            edge = c - b
        tangent0 = edge.normalized(1e-30)
        tangent1 = tangent0.cross(normal).normalized(1e-30)
        tangent_projection = tangent0.outer_product(tangent0) + tangent1.outer_product(tangent1)
    return weights, normal, separation.dot(normal) - radius, displacement, tangent_projection, is_active


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


@qd.func
def tetrahedron_weights(point, a, b, c, d, min_determinant):
    matrix = qd.Matrix.cols([b - a, c - a, d - a])
    determinant = matrix.determinant()
    # Zero-volume tetrahedra have no interior for containment queries.
    weights = qd.Vector([-1.0, -1.0, -1.0, -1.0])
    if determinant != 0.0 and qd.abs(determinant) >= min_determinant:
        bary = matrix.inverse() @ (point - a)
        weights = qd.Vector([1.0 - bary.sum(), bary[0], bary[1], bary[2]])
    return weights


@qd.func
def cubic_roots(coefficients):
    """Find roots on [0, 1] by bisecting the monotone intervals separated by stationary points."""
    scale = qd.abs(coefficients).max()
    roots = qd.Vector([2.0, 2.0, 2.0])
    if scale > 0.0:
        c = coefficients / scale
        cuts = qd.Vector([0.0, 1.0, 1.0, 1.0])
        a, b, d = 3.0 * c[3], 2.0 * c[2], c[1]
        if qd.abs(a) > 1e-12:
            discriminant = b * b - 4.0 * a * d
            if discriminant >= 0.0:
                root = qd.sqrt(discriminant)
                t0, t1 = (-b - root) / (2.0 * a), (-b + root) / (2.0 * a)
                cuts[1] = qd.min(1.0, qd.max(0.0, qd.min(t0, t1)))
                cuts[2] = qd.min(1.0, qd.max(0.0, qd.max(t0, t1)))
        elif qd.abs(b) > 1e-12:
            cuts[1] = qd.min(1.0, qd.max(0.0, -d / b))
        for i in range(3):
            lower, upper = cuts[i], cuts[i + 1]
            f_lower = ((c[3] * lower + c[2]) * lower + c[1]) * lower + c[0]
            f_upper = ((c[3] * upper + c[2]) * upper + c[1]) * upper + c[0]
            if upper > lower and f_lower * f_upper <= 0.0:
                for _ in range(24):
                    middle = 0.5 * (lower + upper)
                    value = ((c[3] * middle + c[2]) * middle + c[1]) * middle + c[0]
                    if f_lower * value <= 0.0:
                        upper = middle
                    else:
                        lower, f_lower = middle, value
                roots[i] = 0.5 * (lower + upper)
    return roots


@qd.func
def triangle_time_of_impact(previous, current):
    e1, e2, e3 = previous[1, :] - previous[0, :], previous[2, :] - previous[0, :], previous[3, :] - previous[0, :]
    motion = current - previous
    de1, de2, de3 = motion[1, :] - motion[0, :], motion[2, :] - motion[0, :], motion[3, :] - motion[0, :]
    normal, cross_motion = e1.cross(e2), e1.cross(de2) + de1.cross(e2)
    roots = cubic_roots(
        qd.Vector(
            [
                e3.dot(normal),
                de3.dot(normal) + e3.dot(cross_motion),
                e3.dot(de1.cross(de2)) + de3.dot(cross_motion),
                de3.dot(de1.cross(de2)),
            ]
        )
    )
    time = qd.cast(2.0, gs.qd_float)
    for i in range(3):
        if roots[i] <= 1.0:
            points = previous + roots[i] * motion
            a, b, c, p = points[0, :], points[1, :], points[2, :], points[3, :]
            normal = (b - a).cross(c - a)
            area = normal.norm_sqr()
            if area > 1e-30:
                u = (b - p).cross(c - p).dot(normal) / area
                v = (c - p).cross(a - p).dot(normal) / area
                if qd.min(u, v, 1.0 - u - v) >= -1e-6:
                    time = qd.min(time, roots[i])
    return time


@qd.kernel
def update_bounds(
    is_previous: bool, is_swept: bool, vertices: V_ANNOTATION, collision: V_ANNOTATION, bvh: V_ANNOTATION
):
    for i_p, i_b in qd.ndrange(bvh.n_aabbs, bvh.n_batches):
        indices = qd.Vector.zero(gs.qd_int, 4)
        n_corners = 3
        if i_p < collision.n_faces:
            for i in qd.static(range(3)):
                indices[i] = collision.faces[i_p][i]
        elif i_p < collision.n_faces + collision.n_edges:
            if qd.static(collision.n_edges > 0):
                for i in qd.static(range(2)):
                    indices[i] = collision.edges[i_p - collision.n_faces][i]
                n_corners = 2
        elif qd.static(collision.n_tets > 0):
            indices = collision.tet_indices[i_p - collision.n_faces - collision.n_edges]
            n_corners = 4
        lower, upper = qd.Vector([1e30, 1e30, 1e30]), qd.Vector([-1e30, -1e30, -1e30])
        for i in range(n_corners):
            position = vertices.pos[indices[i], i_b]
            if is_previous:
                position = vertices.prev[indices[i], i_b]
            lower, upper = qd.min(lower, position), qd.max(upper, position)
            if is_swept:
                lower = qd.min(lower, vertices.prev[indices[i], i_b])
                upper = qd.max(upper, vertices.prev[indices[i], i_b])
        bvh.aabbs[i_b, i_p].min, bvh.aabbs[i_b, i_p].max = lower, upper


@qd.func
def traverse_tetrahedra(i_b, start, target, origin, end, vertices: V_ANNOTATION, collision: V_ANNOTATION):
    """Validate a surface-to-interior segment by walking face-adjacent tetrahedra."""
    current, previous = start, -1
    progress = qd.cast(0.0, gs.qd_float)
    is_valid = False
    for _ in range(collision.n_tets):
        if current < 0:
            break
        ids = collision.tet_indices[current]
        a, b = vertices.prev[ids[0], i_b], vertices.prev[ids[1], i_b]
        c, d = vertices.prev[ids[2], i_b], vertices.prev[ids[3], i_b]
        finish = tetrahedron_weights(end, a, b, c, d, 0.0)
        if current == target and finish.min() >= -1e-5:
            is_valid = True
            break
        begin = tetrahedron_weights(origin, a, b, c, d, 0.0)
        crossing, next_tet = qd.cast(2.0, gs.qd_float), -1
        for i in qd.static(range(4)):
            neighbor = collision.tet_neighbors[current][i]
            if begin[i] > finish[i] and neighbor != previous:
                parameter = begin[i] / (begin[i] - finish[i])
                if parameter >= progress - 1e-6 and parameter < crossing:
                    crossing, next_tet = parameter, neighbor
        previous, current, progress = current, next_tet, crossing
        if progress > 1.0:
            break
    return is_valid


@qd.kernel
def find_volume_contacts(
    is_ccd: bool,
    vertices: V_ANNOTATION,
    collision: V_ANNOTATION,
    bvh: V_ANNOTATION,
    options: V_ANNOTATION,
    errno: qd.Tensor,
):
    for i_q, i_b in collision.contact_faces:
        i_v = collision.surface_indices[i_q]
        if not is_ccd:
            collision.contact_faces[i_q, i_b] = -1
            collision.intersected_tets[i_q, i_b] = -1
        if qd.static(collision.n_tets > 0):  # noqa: SIM102
            if vertices.is_volume[i_v]:
                point = vertices.prev[i_v, i_b]
                if not is_ccd:
                    root = bvh.roots_idx[1 + (collision.n_edges > 0)]
                    node, intersected = root, -1
                    while node >= 0 and intersected < 0:
                        box = bvh.nodes[i_b, node].bound
                        is_overlap = (point >= box.min).all() and (point <= box.max).all()
                        left = bvh.nodes[i_b, node].left
                        if is_overlap and left >= 0:
                            node = left
                        else:
                            if is_overlap:
                                i_e = (
                                    bvh.leaves_idx[i_b, node - bvh.n_aabbs + 1] - collision.n_faces - collision.n_edges
                                )
                                ids = collision.tet_indices[i_e]
                                if (ids != i_v).all():
                                    weights = tetrahedron_weights(
                                        point,
                                        vertices.prev[ids[0], i_b],
                                        vertices.prev[ids[1], i_b],
                                        vertices.prev[ids[2], i_b],
                                        vertices.prev[ids[3], i_b],
                                        collision.tet_min_determinant[i_e],
                                    )
                                    if weights.min() > 1e-6:
                                        intersected = i_e
                            node = next_node(node, i_b, root, bvh)
                    collision.intersected_tets[i_q, i_b] = intersected
                    if intersected >= 0:
                        if qd.static(options.is_contact_diagnostics_enabled):
                            qd.atomic_or(errno[None], ErrorCode.VBD_CONTACT_PENETRATION)
                        target_ids = collision.tet_indices[intersected]
                        target_entity = vertices.entity_idx[target_ids[0]]
                        center = qd.Vector.zero(gs.qd_float, 3)
                        for i in qd.static(range(4)):
                            center += 0.25 * vertices.prev[target_ids[i], i_b]
                        root, best, closest = bvh.roots_idx[0], qd.cast(1e30, gs.qd_float), -1
                        node = root
                        n_candidates = 0
                        while node >= 0:
                            box = bvh.nodes[i_b, node].bound
                            is_overlap = point_aabb_distance_sqr(point, box.min, box.max) < best
                            left = bvh.nodes[i_b, node].left
                            if is_overlap and left >= 0:
                                node = left
                            else:
                                if is_overlap:
                                    i_f = bvh.leaves_idx[i_b, node - bvh.n_aabbs + 1]
                                    ids = collision.faces[i_f]
                                    is_eligible = vertices.entity_idx[ids[0]] == target_entity
                                    if is_eligible:
                                        n_candidates += 1
                                        if n_candidates > 500:
                                            # An incomplete search can retain a distant match with excessive penetration.
                                            closest = -1
                                            break
                                        # A vertex can cross its incident tetrahedron's opposite face during recovery.
                                        is_eligible = (collision.tet_indices[collision.face_tets[i_f]] != i_v).all()
                                    if is_eligible:
                                        a, b, c = (
                                            vertices.prev[ids[0], i_b],
                                            vertices.prev[ids[1], i_b],
                                            vertices.prev[ids[2], i_b],
                                        )
                                        bary = triangle_weights(point, a, b, c)
                                        nearest = bary[0] * a + bary[1] * b + bary[2] * c
                                        distance = (point - nearest).norm_sqr()
                                        if (
                                            distance < best
                                            and feasible_face(
                                                i_f, i_b, point, bary, vertices, collision, is_unique=False
                                            )
                                            and traverse_tetrahedra(
                                                i_b,
                                                collision.face_tets[i_f],
                                                intersected,
                                                0.99 * nearest + 0.01 * (a + b + c) / 3.0,
                                                0.99 * point + 0.01 * center,
                                                vertices,
                                                collision,
                                            )
                                        ):
                                            best, closest = distance, i_f
                                node = next_node(node, i_b, root, bvh)
                        collision.contact_faces[i_q, i_b] = closest
                elif collision.intersected_tets[i_q, i_b] < 0:
                    root, node = bvh.roots_idx[0], bvh.roots_idx[0]
                    lower, upper = qd.min(point, vertices.pos[i_v, i_b]), qd.max(point, vertices.pos[i_v, i_b])
                    best, closest = qd.cast(2.0, gs.qd_float), -1
                    best_distance, nearby = options.contact_d_hat**2, -1
                    while node >= 0:
                        box = bvh.nodes[i_b, node].bound
                        is_overlap = (lower - options.contact_d_hat <= box.max).all() and (
                            upper + options.contact_d_hat >= box.min
                        ).all()
                        left = bvh.nodes[i_b, node].left
                        if is_overlap and left >= 0:
                            node = left
                        else:
                            if is_overlap:
                                i_f = bvh.leaves_idx[i_b, node - bvh.n_aabbs + 1]
                                ids = collision.faces[i_f]
                                is_eligible = vertices.is_volume[ids[0]]
                                if is_eligible:
                                    is_eligible = (collision.tet_indices[collision.face_tets[i_f]] != i_v).all()
                                if is_eligible:
                                    previous, current = (
                                        qd.Matrix.zero(gs.qd_float, 4, 3),
                                        qd.Matrix.zero(gs.qd_float, 4, 3),
                                    )
                                    for i in qd.static(range(3)):
                                        previous[i, :], current[i, :] = (
                                            vertices.prev[ids[i], i_b],
                                            vertices.pos[ids[i], i_b],
                                        )
                                    previous[3, :], current[3, :] = point, vertices.pos[i_v, i_b]
                                    time = triangle_time_of_impact(previous, current)
                                    if time < best:
                                        best, closest = time, i_f
                                    if qd.static(options.contact_d_hat > 0.0):
                                        a, b, c, p = current[0, :], current[1, :], current[2, :], current[3, :]
                                        bary = triangle_weights(p, a, b, c)
                                        separation = p - bary[0] * a - bary[1] * b - bary[2] * c
                                        distance = separation.norm_sqr()
                                        if (
                                            distance < best_distance
                                            and separation.dot((b - a).cross(c - a)) >= 0.0
                                            and feasible_face(i_f, i_b, p, bary, vertices, collision, is_unique=False)
                                        ):
                                            best_distance, nearby = distance, i_f
                            node = next_node(node, i_b, root, bvh)
                    # A swept crossing takes priority over proximity to another surface.
                    collision.contact_faces[i_q, i_b] = closest if closest >= 0 else nearby


@qd.func
def feasible_vertex(i_v, i_b, point, vertices: V_ANNOTATION, collision: V_ANNOTATION):
    is_valid = True
    delta = point - vertices.pos[i_v, i_b]
    for i_a in range(collision.neighbor_offsets[i_v], collision.neighbor_offsets[i_v + 1]):
        edge = vertices.pos[collision.neighbors[i_a], i_b] - vertices.pos[i_v, i_b]
        if delta.dot(edge) > 1e-6 * qd.sqrt(delta.norm_sqr() * edge.norm_sqr()):
            is_valid = False
    return is_valid


@qd.func
def feasible_edge(i_e, i_b, point, vertices: V_ANNOTATION, collision: V_ANNOTATION):
    ids = collision.edges[i_e]
    a, b = vertices.pos[ids[0], i_b], vertices.pos[ids[1], i_b]
    edge = b - a
    is_valid = True
    for i in qd.static(range(2)):
        i_f = collision.edge_faces[i_e][i]
        if i_f >= 0:
            for j in qd.static(range(3)):
                i_v = collision.faces[i_f][j]
                if i_v != ids[0] and i_v != ids[1]:
                    opposite = vertices.pos[i_v, i_b]
                    foot = a + (opposite - a).dot(edge) / qd.max(edge.norm_sqr(), 1e-30) * edge
                    delta, divider = point - foot, opposite - foot
                    if delta.dot(divider) > 1e-6 * qd.sqrt(delta.norm_sqr() * divider.norm_sqr()):
                        is_valid = False
    return is_valid


@qd.func
def feasible_face(i_f, i_b, point, bary, vertices: V_ANNOTATION, collision: V_ANNOTATION, is_unique):
    is_valid = True
    zeros = 0
    for i in qd.static(range(3)):
        zeros += bary[i] <= 0.0
    if zeros == 2:
        for i in qd.static(range(3)):
            if bary[i] > 0.0:
                i_v = collision.faces[i_f][i]
                is_valid = (not is_unique or collision.vertex_faces[i_v] == i_f) and feasible_vertex(
                    i_v, i_b, point, vertices, collision
                )
    elif zeros == 1:
        for i in qd.static(range(3)):
            if bary[i] <= 0.0:
                i_e = collision.face_edges[i_f][(i + 1) % 3]
                is_valid = (not is_unique or collision.edge_faces[i_e][0] == i_f) and feasible_edge(
                    i_e, i_b, point, vertices, collision
                )
    return is_valid


@qd.kernel
def initialize_shell_bounds(
    vertices: V_ANNOTATION, collision: V_ANNOTATION, planes: V_ANNOTATION, options: V_ANNOTATION
):
    for i_f, i_b in collision.faces_bound:
        collision.faces_bound[i_f, i_b] = qd.Vector([1e30, 1e30])
    for i_e, i_b in collision.edges_bound:
        collision.edges_bound[i_e, i_b] = qd.Vector([1e30, 1e30])
    for i_entity, i_b in collision.translation:
        collision.translation[i_entity, i_b] = qd.Vector.zero(gs.qd_float, 4)
        collision.contact_flags[i_entity, i_b] = 0
        collision.translation_bound[i_entity, i_b] = options.contact_relaxation * (
            options.contact_search_radius - options.max_contact_thickness
        )
        # See apply_prediction for the common-translation bound.
        if collision.entity_mass.shape[0] == 1:
            collision.translation_bound[i_entity, i_b] = 1e30
    if qd.static(planes.n_planes > 0):
        for i_entity, i_p, i_b in collision.plane_clearance:
            collision.plane_clearance[i_entity, i_p, i_b] = 1e30
    for i_v, i_b in vertices.pos:
        vertices.bound[i_v, i_b] = options.contact_relaxation * (
            options.contact_search_radius - options.max_contact_thickness
        )
        if vertices.is_volume[i_v]:
            vertices.bound[i_v, i_b] = 1e30
        vertices.collision_origin[i_v, i_b] = vertices.pos[i_v, i_b]
        i_entity = vertices.entity_idx[i_v]
        motion = vertices.next_pos[i_v, i_b] - vertices.pos[i_v, i_b]
        weight = vertices.mass[i_v] / collision.entity_mass[i_entity]
        collision.translation[i_entity, i_b] += weight * qd.Vector(
            [motion[0], motion[1], motion[2], qd.cast(vertices.is_fixed[i_v, i_b], gs.qd_float)]
        )
        if qd.static(planes.n_planes > 0):
            for i_p in range(planes.n_planes):
                gap = vertices.pos[i_v, i_b].dot(planes.normal[i_p]) - planes.offset[i_p] - vertices.radius[i_v]
                qd.atomic_min(collision.plane_clearance[i_entity, i_p, i_b], qd.max(0.0, gap))
    if qd.static(planes.n_planes > 0):
        for i_entity, i_b in collision.translation:
            motion = collision.translation[i_entity, i_b][:3]
            # Limiting the normal component preserves frictionless tangential motion.
            for i_p in range(planes.n_planes):
                normal = planes.normal[i_p]
                approach = motion.dot(normal)
                gap = collision.plane_clearance[i_entity, i_p, i_b]
                motion -= qd.min(0.0, approach + gap) * normal
            for i_d in qd.static(range(3)):
                collision.translation[i_entity, i_b][i_d] = motion[i_d]


@qd.kernel
def gather_shell_bounds(vertices: V_ANNOTATION, collision: V_ANNOTATION):
    for i_q, i_b in collision.vertices_bound:
        i_v = collision.surface_indices[i_q]
        bound = collision.vertices_bound[i_q, i_b]
        for i_a in range(collision.neighbor_offsets[i_v], collision.neighbor_offsets[i_v + 1]):
            i_e = collision.neighbor_edges[i_a]
            bound = qd.min(bound, collision.edges_bound[i_e, i_b])
            for i in qd.static(range(2)):
                i_f = collision.edge_faces[i_e][i]
                if i_f >= 0:
                    bound = qd.min(bound, collision.faces_bound[i_f, i_b])
        vertices.bound[i_v, i_b] = qd.min(vertices.bound[i_v, i_b], bound[0])
        if bound[1] < 1e30:
            qd.atomic_min(collision.translation_bound[vertices.entity_idx[i_v], i_b], bound[1])


@qd.kernel
def find_shell_contacts(
    offsets: qd.types.ndarray(),
    pairs: qd.types.ndarray(),
    batches: qd.types.ndarray(),
    kinds: qd.types.ndarray(),
    counts: qd.types.ndarray(),
    is_fill: bool,
    vertices: V_ANNOTATION,
    collision: V_ANNOTATION,
    bvh: V_ANNOTATION,
    options: V_ANNOTATION,
    errno: qd.Tensor,
):
    for i_q, i_b in qd.ndrange(counts.shape[0] // bvh.n_batches, bvh.n_batches):
        n_surface = collision.surface_indices.shape[0]
        is_vertex = i_q < n_surface
        i_e = i_q - n_surface
        point = qd.Vector.zero(gs.qd_float, 3)
        lower, upper = point, point
        root = bvh.roots_idx[0]
        if is_vertex:
            point = vertices.pos[collision.surface_indices[i_q], i_b]
            lower, upper = point, point
        elif qd.static(collision.n_edges > 0):
            box = bvh.aabbs[i_b, collision.n_faces + i_e]
            lower, upper = box.min, box.max
            root = bvh.roots_idx[1]
        radius = options.contact_search_radius
        node, count = root, 0
        bound = qd.Vector([1e30, 1e30])
        while node >= 0:
            box = bvh.nodes[i_b, node].bound
            is_overlap = (lower - radius <= box.max).all() and (upper + radius >= box.min).all()
            left = bvh.nodes[i_b, node].left
            if is_overlap and left >= 0:
                node = left
            else:
                if is_overlap:
                    candidate = bvh.leaves_idx[i_b, node - bvh.n_aabbs + 1]
                    ids = qd.Vector.zero(gs.qd_int, 4)
                    is_eligible = True
                    if is_vertex:
                        ids[0] = collision.surface_indices[i_q]
                        for i in qd.static(range(3)):
                            ids[i + 1] = collision.faces[candidate][i]
                            if ids[0] == ids[i + 1]:
                                is_eligible = False
                    elif qd.static(collision.n_edges > 0):
                        candidate -= collision.n_faces
                        ids = qd.Vector(
                            [
                                collision.edges[i_e][0],
                                collision.edges[i_e][1],
                                collision.edges[candidate][0],
                                collision.edges[candidate][1],
                            ]
                        )
                        is_eligible = candidate > i_e
                        for i, j in qd.static(qd.ndrange(2, 2)):
                            if ids[i] == ids[j + 2]:
                                is_eligible = False
                    has_shell = False
                    for i in qd.static(range(4)):
                        has_shell = has_shell or not vertices.is_volume[ids[i]]
                    if is_eligible and has_shell:
                        a, b, c, d = (
                            vertices.pos[ids[0], i_b],
                            vertices.pos[ids[1], i_b],
                            vertices.pos[ids[2], i_b],
                            vertices.pos[ids[3], i_b],
                        )
                        weights = qd.Vector.zero(gs.qd_float, 4)
                        if is_vertex:
                            bary = triangle_weights(a, b, c, d)
                            weights = qd.Vector([1.0, -bary[0], -bary[1], -bary[2]])
                        else:
                            weights = edge_weights(a, b, c, d)
                        separation = weights[0] * a + weights[1] * b + weights[2] * c + weights[3] * d
                        distance = separation.norm()
                        thickness = qd.cast(0.0, gs.qd_float)
                        for i in qd.static(range(4)):
                            thickness += qd.abs(weights[i]) * vertices.radius[ids[i]]
                        if distance < radius:
                            if not is_fill:
                                clearance = options.contact_relaxation * qd.max(0.0, distance - thickness)
                                bound[0] = qd.min(bound[0], clearance)
                                # Bounds only decrease during the query, so a stale smaller value also permits skipping.
                                if is_vertex:
                                    if clearance < collision.faces_bound[candidate, i_b][0]:
                                        qd.atomic_min(collision.faces_bound[candidate, i_b][0], clearance)
                                else:
                                    if clearance < collision.edges_bound[candidate, i_b][0]:
                                        qd.atomic_min(collision.edges_bound[candidate, i_b][0], clearance)
                                if vertices.entity_idx[ids[0]] != vertices.entity_idx[ids[2]]:
                                    bound[1] = qd.min(bound[1], clearance)
                                    if is_vertex:
                                        if clearance < collision.faces_bound[candidate, i_b][1]:
                                            qd.atomic_min(collision.faces_bound[candidate, i_b][1], clearance)
                                    else:
                                        if clearance < collision.edges_bound[candidate, i_b][1]:
                                            qd.atomic_min(collision.edges_bound[candidate, i_b][1], clearance)
                            if is_vertex:
                                is_eligible = feasible_face(
                                    candidate, i_b, a, -weights[1:], vertices, collision, is_unique=True
                                )
                            else:
                                first = weights[0] * a + weights[1] * b
                                second = -weights[2] * c - weights[3] * d
                                first_edge, second_edge = b - a, d - c
                                is_eligible = (
                                    (second - a).dot(first_edge) >= 0.0 and (b - second).dot(first_edge) >= 0.0
                                ) or ((first - c).dot(second_edge) >= 0.0 and (d - first).dot(second_edge) >= 0.0)
                                # Retain parallel candidates: their angle can change before the next search.
                            if is_eligible:
                                if not is_fill:
                                    flag = 1
                                    if vertices.entity_idx[ids[0]] != vertices.entity_idx[ids[2]]:
                                        flag = 2
                                    for i in qd.static(range(4)):
                                        qd.atomic_or(collision.contact_flags[vertices.entity_idx[ids[i]], i_b], flag)
                                if qd.static(options.is_contact_diagnostics_enabled) and distance <= thickness:
                                    qd.atomic_or(errno[None], ErrorCode.VBD_CONTACT_PENETRATION)
                                if is_fill:
                                    index = offsets[i_q * bvh.n_batches + i_b] + count
                                    for i in qd.static(range(4)):
                                        pairs[index, i] = ids[i]
                                    batches[index] = i_b
                                    kinds[index] = 1 if is_vertex else 0
                                count += 1
                node = next_node(node, i_b, root, bvh)
        counts[i_q * bvh.n_batches + i_b] = count
        if not is_fill:
            if is_vertex:
                collision.vertices_bound[i_q, i_b] = bound
            else:
                qd.atomic_min(collision.edges_bound[i_e, i_b][0], bound[0])
                if bound[1] < 1e30:
                    qd.atomic_min(collision.edges_bound[i_e, i_b][1], bound[1])
