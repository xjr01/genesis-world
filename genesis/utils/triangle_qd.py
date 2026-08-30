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
