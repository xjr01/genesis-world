import quadrants as qd

import genesis as gs


@qd.func
def point_triangle_constraint(point, v0, v1, v2):
    """Return signed plane distance, its four vertex gradients, and triangle validity."""
    edge0 = v1 - v0
    edge1 = v2 - v0
    normal = edge0.cross(edge1)
    area = normal.norm()
    is_valid = area > gs.EPS * edge0.norm() * edge1.norm()
    residual = gs.qd_float(0.0)
    gradients = qd.Matrix.zero(gs.qd_float, 3, 4)
    if is_valid:
        normal /= area
        offset = point - v0
        residual = offset.dot(normal)
        tangent = (offset - residual * normal) / area
        gradients[:, 0] = normal
        gradients[:, 2] = edge1.cross(tangent)
        gradients[:, 3] = tangent.cross(edge0)
        gradients[:, 1] = -normal - gradients[:, 2] - gradients[:, 3]
    return residual, gradients, is_valid


@qd.func
def point_triangle_entry_time(previous_point, point, previous_triangle, triangle):
    """Return the last outside-to-inside crossing time for linearly moving vertices, or -1."""
    offset = previous_point - previous_triangle[:, 0]
    edge0 = previous_triangle[:, 1] - previous_triangle[:, 0]
    edge1 = previous_triangle[:, 2] - previous_triangle[:, 0]
    delta = point - triangle[:, 0] - offset
    delta0 = triangle[:, 1] - triangle[:, 0] - edge0
    delta1 = triangle[:, 2] - triangle[:, 0] - edge1
    normal0 = edge0.cross(edge1)
    normal1 = delta0.cross(edge1) + edge0.cross(delta1)
    normal2 = delta0.cross(delta1)
    coefficients = qd.Vector(
        [
            offset.dot(normal0),
            delta.dot(normal0) + offset.dot(normal1),
            delta.dot(normal1) + offset.dot(normal2),
            delta.dot(normal2),
        ]
    )
    scale = qd.abs(coefficients).max()
    intervals = qd.Vector([0.0, 1.0, 1.0, 1.0])
    if qd.abs(coefficients[3]) > gs.EPS * scale:
        discriminant = coefficients[2] ** 2 - 3.0 * coefficients[3] * coefficients[1]
        if discriminant > 0.0:
            root0 = (-coefficients[2] - qd.sqrt(discriminant)) / (3.0 * coefficients[3])
            root1 = (-coefficients[2] + qd.sqrt(discriminant)) / (3.0 * coefficients[3])
            intervals[1] = qd.min(1.0, qd.max(0.0, qd.min(root0, root1)))
            intervals[2] = qd.min(1.0, qd.max(0.0, qd.max(root0, root1)))
    elif qd.abs(coefficients[2]) > gs.EPS * scale:
        intervals[1] = qd.min(1.0, qd.max(0.0, -coefficients[1] / (2.0 * coefficients[2])))
    entry_time = gs.qd_float(-1.0)
    for i_interval in range(3):
        lower = intervals[i_interval]
        upper = intervals[i_interval + 1]
        value_lower = ((coefficients[3] * lower + coefficients[2]) * lower + coefficients[1]) * lower + coefficients[0]
        value_upper = ((coefficients[3] * upper + coefficients[2]) * upper + coefficients[1]) * upper + coefficients[0]
        if value_lower >= 0.0 and value_upper <= 0.0 and value_lower > value_upper:
            while upper - lower > 4.0 * gs.EPS:
                middle = 0.5 * (lower + upper)
                value = (
                    (coefficients[3] * middle + coefficients[2]) * middle + coefficients[1]
                ) * middle + coefficients[0]
                if value > 0.0:
                    lower = middle
                else:
                    upper = middle
            time = 0.5 * (lower + upper)
            vertices = previous_triangle + time * (triangle - previous_triangle)
            position = previous_point + time * (point - previous_point)
            _, gradients, is_valid = point_triangle_constraint(position, vertices[:, 0], vertices[:, 1], vertices[:, 2])
            if is_valid:
                is_inside = True
                for i_v in qd.static(range(3)):
                    is_inside = is_inside and -gradients[:, i_v + 1].dot(gradients[:, 0]) >= -32.0 * gs.EPS
                if is_inside:
                    entry_time = qd.max(entry_time, time)
    return entry_time


@qd.func
def closest_point_on_triangle(
    point: qd.types.vector(3), v0: qd.types.vector(3), v1: qd.types.vector(3), v2: qd.types.vector(3)
) -> qd.types.vector(3):
    """Return the closest point on a triangle to a query point."""
    ab = v1 - v0
    ac = v2 - v0
    ap = point - v0

    d1 = ab.dot(ap)
    d2 = ac.dot(ap)

    closest = v0
    if not (d1 <= 0.0 and d2 <= 0.0):
        bp = point - v1
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)

        if d3 >= 0.0 and d4 <= d3:
            closest = v1
        else:
            cp = point - v2
            d5 = ab.dot(cp)
            d6 = ac.dot(cp)

            if d6 >= 0.0 and d5 <= d6:
                closest = v2
            else:
                vc = d1 * d4 - d3 * d2
                if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                    w = d1 / (d1 - d3)
                    closest = v0 + w * ab
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        w = d2 / (d2 - d6)
                        closest = v0 + w * ac
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                            closest = v1 + w * (v2 - v1)
                        else:
                            denom = 1.0 / (va + vb + vc)
                            v = vb * denom
                            w = vc * denom
                            closest = v0 + v * ab + w * ac
    return closest


@qd.func
def segment_triangle_intersection(
    segment_start: qd.types.vector(3),
    segment_end: qd.types.vector(3),
    v0: qd.types.vector(3),
    v1: qd.types.vector(3),
    v2: qd.types.vector(3),
    eps: float,
):
    """Return an inclusive segment-triangle hit and its position."""
    direction = segment_end - segment_start
    edge0 = v1 - v0
    edge1 = v2 - v0
    cross_direction = direction.cross(edge1)
    determinant = edge0.dot(cross_direction)
    determinant_tolerance = 16.0 * eps * direction.norm() * edge0.norm() * edge1.norm()
    is_hit = False
    parameter = gs.qd_float(0.0)
    hit_position = segment_start
    if qd.abs(determinant) > determinant_tolerance:
        determinant_inv = 1.0 / determinant
        offset = segment_start - v0
        barycentric1 = offset.dot(cross_direction) * determinant_inv
        cross_offset = offset.cross(edge0)
        barycentric2 = direction.dot(cross_offset) * determinant_inv
        parameter = edge1.dot(cross_offset) * determinant_inv
        coordinate_tolerance = 32.0 * eps
        is_hit = (
            barycentric1 >= -coordinate_tolerance
            and barycentric2 >= -coordinate_tolerance
            and barycentric1 + barycentric2 <= 1.0 + coordinate_tolerance
            and parameter >= -coordinate_tolerance
            and parameter <= 1.0 + coordinate_tolerance
        )
        if is_hit:
            parameter = qd.min(1.0, qd.max(0.0, parameter))
            hit_position = segment_start + parameter * direction
    return is_hit, hit_position


@qd.func
def segment_triangle_crossing(segment_start, segment_end, vertices):
    """Return an inclusive segment crossing through a triangle, excluding coplanar contact."""
    edge0 = vertices[:, 1] - vertices[:, 0]
    edge1 = vertices[:, 2] - vertices[:, 0]
    normal = edge0.cross(edge1)
    distance0 = (segment_start - vertices[:, 0]).dot(normal)
    distance1 = (segment_end - vertices[:, 0]).dot(normal)
    is_crossing = distance0 * distance1 <= 0.0
    position = segment_start
    if is_crossing:
        is_crossing, position = segment_triangle_intersection(
            segment_start, segment_end, vertices[:, 0], vertices[:, 1], vertices[:, 2], gs.EPS
        )
    return is_crossing, position


@qd.func
def triangle_crossing_edges(vertices0, vertices1):
    """Return transverse crossing-edge masks and validity; isolated touching points have empty masks."""
    edges0 = qd.Vector.zero(gs.qd_int, 3)
    edges1 = qd.Vector.zero(gs.qd_int, 3)
    is_valid = True
    scale = gs.qd_float(0.0)
    for side in qd.static(range(2)):
        vertices = vertices0
        if qd.static(side == 1):
            vertices = vertices1
        edge0 = vertices[:, 1] - vertices[:, 0]
        edge1 = vertices[:, 2] - vertices[:, 0]
        is_valid = is_valid and edge0.cross(edge1).norm_sqr() > gs.EPS**2 * edge0.norm_sqr() * edge1.norm_sqr()
        scale = qd.max(scale, edge0.norm(), edge1.norm(), (vertices[:, 2] - vertices[:, 1]).norm())
    lower = qd.Vector([1.0e30, 1.0e30, 1.0e30])
    upper = -lower
    if is_valid:
        for edge_idx in qd.static(range(3)):
            is_hit0, position0 = segment_triangle_crossing(
                vertices0[:, edge_idx], vertices0[:, (edge_idx + 1) % 3], vertices1
            )
            is_hit1, position1 = segment_triangle_crossing(
                vertices1[:, edge_idx], vertices1[:, (edge_idx + 1) % 3], vertices0
            )
            edges0[edge_idx] = is_hit0
            edges1[edge_idx] = is_hit1
            if is_hit0:
                lower = qd.min(lower, position0)
                upper = qd.max(upper, position0)
            if is_hit1:
                lower = qd.min(lower, position1)
                upper = qd.max(upper, position1)
        if (upper - lower).max() <= 32.0 * gs.EPS * scale:
            edges0.fill(0)
            edges1.fill(0)
    return edges0, edges1, is_valid


@qd.func
def edge_edge_contact(vertices0, vertices1, reference_normal):
    """Return finite-edge signed distance and endpoint gradients, oriented toward the reference normal."""
    edge0 = vertices0[:, 1] - vertices0[:, 0]
    edge1 = vertices1[:, 1] - vertices1[:, 0]
    offset = vertices0[:, 0] - vertices1[:, 0]
    length0_sqr = edge0.norm_sqr()
    length1_sqr = edge1.norm_sqr()
    is_valid = length0_sqr > 0.0 and length1_sqr > 0.0
    residual = gs.qd_float(0.0)
    gradients = qd.Matrix.zero(gs.qd_float, 3, 2)
    if is_valid:
        alignment = edge0.dot(edge1)
        offset0 = edge0.dot(offset)
        offset1 = edge1.dot(offset)
        denominator = edge0.cross(edge1).norm_sqr()
        parameter0 = gs.qd_float(0.0)
        if denominator > gs.EPS**2 * length0_sqr * length1_sqr:
            parameter0 = qd.min(1.0, qd.max(0.0, (alignment * offset1 - offset0 * length1_sqr) / denominator))
        parameter1 = (alignment * parameter0 + offset1) / length1_sqr
        if parameter1 < 0.0:
            parameter1 = 0.0
            parameter0 = qd.min(1.0, qd.max(0.0, -offset0 / length0_sqr))
        elif parameter1 > 1.0:
            parameter1 = 1.0
            parameter0 = qd.min(1.0, qd.max(0.0, (alignment - offset0) / length0_sqr))
        separation = offset + parameter0 * edge0 - parameter1 * edge1
        distance = separation.norm()
        normal = edge0.cross(edge1)
        if distance > 32.0 * gs.EPS * qd.sqrt(qd.max(length0_sqr, length1_sqr)):
            normal = separation / distance
        elif denominator > gs.EPS**2 * length0_sqr * length1_sqr:
            normal /= qd.sqrt(denominator)
        else:
            normal = reference_normal
        is_valid = normal.norm_sqr() > 0.0
        if is_valid:
            normal = normal.normalized()
            if normal.dot(reference_normal) < 0.0:
                normal = -normal
            # Freezing the closest-point coordinates and normal gives a linear contact projection.
            residual = separation.dot(normal)
            gradients[:, 0] = (1.0 - parameter0) * normal
            gradients[:, 1] = parameter0 * normal
    return residual, gradients, is_valid


@qd.func
def _triangle_interval(axis, origin, vertices):
    projection0 = (vertices[:, 0] - origin).dot(axis)
    projection1 = (vertices[:, 1] - origin).dot(axis)
    projection2 = (vertices[:, 2] - origin).dot(axis)
    return qd.min(projection0, projection1, projection2), qd.max(projection0, projection1, projection2)


@qd.func
def _triangle_intervals_overlap(axis, origin, vertices0, vertices1, length_scale, eps):
    axis_length = axis.norm()
    is_overlapping = True
    if axis_length > 0.0:
        lower0, upper0 = _triangle_interval(axis, origin, vertices0)
        lower1, upper1 = _triangle_interval(axis, origin, vertices1)
        tolerance = 32.0 * eps * axis_length * qd.max(1.0, length_scale)
        is_overlapping = lower0 <= upper1 + tolerance and lower1 <= upper0 + tolerance
    return is_overlapping


@qd.func
def _triangle_previous_separating_axis_correction(
    axis,
    current_vertices,
    previous_vertices,
    rigid_vertices,
    clearance,
    length_scale,
    eps,
    best_distance,
    best_correction,
    has_correction,
):
    axis_length = axis.norm()
    if axis_length > 0.0:
        origin = rigid_vertices[:, 0]
        current_lower, current_upper = _triangle_interval(axis, origin, current_vertices)
        previous_lower, previous_upper = _triangle_interval(axis, origin, previous_vertices)
        rigid_lower, rigid_upper = _triangle_interval(axis, origin, rigid_vertices)
        tolerance = 32.0 * eps * axis_length * qd.max(1.0, length_scale)
        correction_depth = gs.qd_float(0.0)
        correction_sign = gs.qd_float(0.0)
        if previous_upper < rigid_lower - tolerance:
            correction_depth = current_upper - rigid_lower + clearance * axis_length
            correction_sign = -1.0
        elif previous_lower > rigid_upper + tolerance:
            correction_depth = rigid_upper - current_lower + clearance * axis_length
            correction_sign = 1.0
        if correction_depth > 0.0:
            correction_distance = correction_depth / axis_length
            if correction_distance < best_distance:
                best_distance = correction_distance
                best_correction = correction_sign * correction_depth / axis_length**2 * axis
                has_correction = True
    return best_distance, best_correction, has_correction


@qd.func
def triangle_triangle_previous_separating_correction(
    current_vertices,
    previous_vertices,
    rigid_vertices,
    clearance,
    eps,
):
    """Return the smallest displacement restoring an axis that separated the previous geom-relative triangle."""
    previous_edges = qd.Matrix.cols(
        [
            previous_vertices[:, 1] - previous_vertices[:, 0],
            previous_vertices[:, 2] - previous_vertices[:, 1],
            previous_vertices[:, 0] - previous_vertices[:, 2],
        ]
    )
    rigid_edges = qd.Matrix.cols(
        [
            rigid_vertices[:, 1] - rigid_vertices[:, 0],
            rigid_vertices[:, 2] - rigid_vertices[:, 1],
            rigid_vertices[:, 0] - rigid_vertices[:, 2],
        ]
    )
    previous_normal = previous_edges[:, 0].cross(-previous_edges[:, 2])
    rigid_normal = rigid_edges[:, 0].cross(-rigid_edges[:, 2])
    length_scale = qd.max(
        previous_edges[:, 0].norm(),
        previous_edges[:, 1].norm(),
        previous_edges[:, 2].norm(),
        rigid_edges[:, 0].norm(),
        rigid_edges[:, 1].norm(),
        rigid_edges[:, 2].norm(),
    )
    best_distance = gs.qd_float(1.0e30)
    best_correction = qd.Vector.zero(gs.qd_float, 3)
    has_correction = False
    best_distance, best_correction, has_correction = _triangle_previous_separating_axis_correction(
        previous_normal,
        current_vertices,
        previous_vertices,
        rigid_vertices,
        clearance,
        length_scale,
        eps,
        best_distance,
        best_correction,
        has_correction,
    )
    best_distance, best_correction, has_correction = _triangle_previous_separating_axis_correction(
        rigid_normal,
        current_vertices,
        previous_vertices,
        rigid_vertices,
        clearance,
        length_scale,
        eps,
        best_distance,
        best_correction,
        has_correction,
    )
    for previous_edge_idx, rigid_edge_idx in qd.static(qd.ndrange(3, 3)):
        best_distance, best_correction, has_correction = _triangle_previous_separating_axis_correction(
            previous_edges[:, previous_edge_idx].cross(rigid_edges[:, rigid_edge_idx]),
            current_vertices,
            previous_vertices,
            rigid_vertices,
            clearance,
            length_scale,
            eps,
            best_distance,
            best_correction,
            has_correction,
        )
    for edge_idx in qd.static(range(3)):
        best_distance, best_correction, has_correction = _triangle_previous_separating_axis_correction(
            previous_normal.cross(previous_edges[:, edge_idx]),
            current_vertices,
            previous_vertices,
            rigid_vertices,
            clearance,
            length_scale,
            eps,
            best_distance,
            best_correction,
            has_correction,
        )
        best_distance, best_correction, has_correction = _triangle_previous_separating_axis_correction(
            rigid_normal.cross(rigid_edges[:, edge_idx]),
            current_vertices,
            previous_vertices,
            rigid_vertices,
            clearance,
            length_scale,
            eps,
            best_distance,
            best_correction,
            has_correction,
        )
    return has_correction, best_correction


@qd.func
def triangle_separating_corrections(vertices, separation):
    """Project each vertex onto the halfspace defined by the triangle's minimum separating translation."""
    corrections = qd.Matrix.zero(gs.qd_float, 3, 3)
    distance = separation.norm()
    if distance > 0.0:
        normal = separation / distance
        projections = qd.Vector.zero(gs.qd_float, 3)
        for i_v in qd.static(range(3)):
            projections[i_v] = (vertices[:, i_v] - vertices[:, 0]).dot(normal)
        target_projection = projections.min() + distance
        for i_v in qd.static(range(3)):
            corrections[:, i_v] = qd.max(target_projection - projections[i_v], 0.0) * normal
    return corrections


@qd.func
def triangle_triangle_intersection(vertices0, vertices1, eps):
    """Return an exact discrete triangle-triangle overlap and one point on its intersection.

    Separating axes handle proper, edge, vertex, containment, and coplanar overlaps. Segment tests locate a surface
    point for the response ray; containment falls back to the first triangle's centroid.
    """
    edges0 = qd.Matrix.cols(
        [
            vertices0[:, 1] - vertices0[:, 0],
            vertices0[:, 2] - vertices0[:, 1],
            vertices0[:, 0] - vertices0[:, 2],
        ]
    )
    edges1 = qd.Matrix.cols(
        [
            vertices1[:, 1] - vertices1[:, 0],
            vertices1[:, 2] - vertices1[:, 1],
            vertices1[:, 0] - vertices1[:, 2],
        ]
    )
    normal0_raw = edges0[:, 0].cross(-edges0[:, 2])
    normal1_raw = edges1[:, 0].cross(-edges1[:, 2])
    length_scale = qd.max(
        edges0[:, 0].norm(),
        edges0[:, 1].norm(),
        edges0[:, 2].norm(),
        edges1[:, 0].norm(),
        edges1[:, 1].norm(),
        edges1[:, 2].norm(),
    )
    # Relative area tests retain the same classification under uniform changes of length units.
    is_intersecting = (
        normal0_raw.norm_sqr() > eps**2 * edges0[:, 0].norm_sqr() * edges0[:, 2].norm_sqr()
        and normal1_raw.norm_sqr() > eps**2 * edges1[:, 0].norm_sqr() * edges1[:, 2].norm_sqr()
    )
    if is_intersecting:
        is_intersecting = _triangle_intervals_overlap(
            normal0_raw, vertices1[:, 0], vertices0, vertices1, length_scale, eps
        ) and _triangle_intervals_overlap(normal1_raw, vertices1[:, 0], vertices0, vertices1, length_scale, eps)

    for edge0_idx, edge1_idx in qd.static(qd.ndrange(3, 3)):
        if is_intersecting:
            is_intersecting = _triangle_intervals_overlap(
                edges0[:, edge0_idx].cross(edges1[:, edge1_idx]),
                vertices1[:, 0],
                vertices0,
                vertices1,
                length_scale,
                eps,
            )

    # Coplanar triangles need their in-plane edge normals in addition to the standard 3D triangle axes.
    for edge_idx in qd.static(range(3)):
        if is_intersecting:
            is_intersecting = _triangle_intervals_overlap(
                normal0_raw.cross(edges0[:, edge_idx]),
                vertices1[:, 0],
                vertices0,
                vertices1,
                length_scale,
                eps,
            ) and _triangle_intervals_overlap(
                normal1_raw.cross(edges1[:, edge_idx]),
                vertices1[:, 0],
                vertices0,
                vertices1,
                length_scale,
                eps,
            )

    hit_position = (vertices0[:, 0] + vertices0[:, 1] + vertices0[:, 2]) / 3.0
    has_segment_hit = False
    for edge_idx in qd.static(range(3)):
        if is_intersecting and not has_segment_hit:
            next_edge_idx = (edge_idx + 1) % 3
            edge_hit, edge_position = segment_triangle_intersection(
                vertices0[:, edge_idx],
                vertices0[:, next_edge_idx],
                vertices1[:, 0],
                vertices1[:, 1],
                vertices1[:, 2],
                eps,
            )
            if edge_hit:
                hit_position = edge_position
                has_segment_hit = True

    for edge_idx in qd.static(range(3)):
        if is_intersecting and not has_segment_hit:
            next_edge_idx = (edge_idx + 1) % 3
            edge_hit, edge_position = segment_triangle_intersection(
                vertices1[:, edge_idx],
                vertices1[:, next_edge_idx],
                vertices0[:, 0],
                vertices0[:, 1],
                vertices0[:, 2],
                eps,
            )
            if edge_hit:
                hit_position = edge_position
                has_segment_hit = True

    return is_intersecting, hit_position


@qd.func
def triangle_face_normal(v0: qd.types.vector(3), v1: qd.types.vector(3), v2: qd.types.vector(3)) -> qd.types.vector(3):
    """Return the outward unit normal under right-hand triangle winding."""
    return (v1 - v0).cross(v2 - v0).normalized()


@qd.func
def ray_projection(ray_dir: qd.types.vector(3), eps: float):
    """Map a ray onto +Z for watertight triangle intersection tests.

    The transform depends only on the ray, so triangles sharing an edge project its vertices to bit-identical
    coordinates. Permuting the largest absolute direction component last bounds the shear, and swapping the first two
    axes for a negative component preserves triangle winding.

    Returns the permuted axes, the two shear factors plus reciprocal ray scale, and whether the direction defines a
    ray.
    """
    dir_abs = qd.abs(ray_dir)
    kz = 0
    if dir_abs[1] > dir_abs[0]:
        kz = 1
    if dir_abs[2] > dir_abs[kz]:
        kz = 2
    kx = (kz + 1) % 3
    ky = (kx + 1) % 3
    if ray_dir[kz] < 0.0:
        k_swap = kx
        kx = ky
        ky = k_swap

    shear = qd.math.vec3(0.0, 0.0, 0.0)
    is_valid = dir_abs[kz] > eps
    if is_valid:
        shear = qd.math.vec3(ray_dir[kx], ray_dir[ky], 1.0) / ray_dir[kz]

    return gs.qd_ivec3(kx, ky, kz), shear, is_valid


@qd.func
def ray_triangle_intersection(
    axes: qd.types.vector(3),
    ray_start: qd.types.vector(3),
    shear: qd.types.vector(3),
    v0: qd.types.vector(3),
    v1: qd.types.vector(3),
    v2: qd.types.vector(3),
    eps: float,
):
    """Return the positive distance to a watertight ray-triangle hit, or -1 for a miss.

    The ray is inside the triangle when its three edge tests agree in sign, with values inside each test's rounding
    bound accepted by either side. Both triangles sharing a crossed edge therefore count the hit consistently.
    """
    hit_distance = gs.qd_float(-1.0)

    # Shearing makes the ray +Z and reduces the edge tests to two dimensions in a shared ray frame.
    verts_rel = qd.Matrix.cols([v0 - ray_start, v1 - ray_start, v2 - ray_start])
    verts_along = verts_rel[axes[2], :]
    verts_x = verts_rel[axes[0], :] - shear[0] * verts_along
    verts_y = verts_rel[axes[1], :] - shear[1] * verts_along
    verts_z = shear[2] * verts_along

    edge_areas = verts_y.cross(verts_x)
    # This bound covers cancellation in each signed edge-area test, including backend-dependent fused operations.
    edge_bound = 2.0 * eps * verts_x.norm() * verts_y.norm()

    is_crossing = not ((edge_areas < -edge_bound).any() and (edge_areas > edge_bound).any())
    det = edge_areas.sum()
    if is_crossing and det != 0.0:
        t = edge_areas.dot(verts_z) / det
        if t > eps:
            hit_distance = t

    return hit_distance


@qd.func
def ray_aabb_intersection(
    ray_start: qd.types.vector(3),
    ray_dir: qd.types.vector(3),
    aabb_min: qd.types.vector(3),
    aabb_max: qd.types.vector(3),
    eps: float,
):
    """Return the nonnegative entry distance for a conservative ray-axis-aligned bounding box (AABB) hit.

    A miss returns -1. The conservative interval ensures every surface crossing reaches the watertight triangle test.
    """
    result = -1.0
    # Flooring only keeps the reciprocal finite; a larger floor could reject a real hit on a near-parallel axis.
    sign = qd.select(ray_dir >= 0.0, 1.0, -1.0)
    inv_dir = sign / qd.max(qd.abs(ray_dir), eps * eps)

    t1 = (aabb_min - ray_start) * inv_dir
    t2 = (aabb_max - ray_start) * inv_dir
    tmin = qd.min(t1, t2)
    tmax = qd.max(t1, t2)
    t_near = qd.max(tmin.x, tmin.y, tmin.z, gs.qd_float(0.0))
    t_far = qd.min(tmax.x, tmax.y, tmax.z)

    # Inverted bounds are an unhittable sentinel, while the widened slab interval covers its own rounding error.
    is_non_empty = aabb_min.x <= aabb_max.x and aabb_min.y <= aabb_max.y and aabb_min.z <= aabb_max.z
    if is_non_empty and t_near * (1.0 - 2.0 * eps) <= t_far * (1.0 + 2.0 * eps):
        result = t_near

    return result
