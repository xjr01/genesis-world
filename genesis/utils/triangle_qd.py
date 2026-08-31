import quadrants as qd

import genesis as gs


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
    determinant_tolerance = 16.0 * eps * qd.max(1.0, direction.norm() * edge0.norm() * edge1.norm())
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
def _triangle_interval(axis, origin, vertices):
    projection0 = (vertices[:, 0] - origin).dot(axis)
    projection1 = (vertices[:, 1] - origin).dot(axis)
    projection2 = (vertices[:, 2] - origin).dot(axis)
    return qd.min(projection0, projection1, projection2), qd.max(projection0, projection1, projection2)


@qd.func
def _triangle_intervals_overlap(axis, origin, vertices0, vertices1, length_scale, eps):
    axis_length = axis.norm()
    is_overlapping = True
    if axis_length > eps:
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
    if axis_length > eps:
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
    is_intersecting = normal0_raw.norm_sqr() > eps**2 and normal1_raw.norm_sqr() > eps**2
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
