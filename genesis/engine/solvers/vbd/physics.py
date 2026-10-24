import quadrants as qd

import genesis as gs
from genesis.utils.array_class import ErrorCode, V_ANNOTATION

from .collision import contact_geometry


@qd.func
def wrapped_angle(angle):
    return qd.atan2(qd.sin(angle), qd.cos(angle))


@qd.func
def hinge_angle_gradient(local_idx, p0, p1, p2, p3):
    edge = p1 - p0
    length = edge.norm()
    n0 = edge.cross(p2 - p0)
    n1 = (p3 - p0).cross(edge)
    n0_sq = qd.max(n0.norm_sqr(), 1e-30)
    n1_sq = qd.max(n1.norm_sqr(), 1e-30)
    angle = qd.atan2(edge.dot(n0.cross(n1)) / qd.max(length, 1e-15), n0.dot(n1))
    q2 = -length / n0_sq * n0
    q3 = -length / n1_sq * n1
    gradient = q2
    if local_idx == 0:
        gradient = -((p1 - p2).dot(edge) * q2 + (p1 - p3).dot(edge) * q3) / length**2
    elif local_idx == 1:
        gradient = -((p2 - p0).dot(edge) * q2 + (p3 - p0).dot(edge) * q3) / length**2
    elif local_idx == 3:
        gradient = q3
    return angle, gradient


@qd.func
def element_response(i_e, i_local, i_b, trial, vertices: V_ANNOTATION, elements: V_ANNOTATION):
    F = qd.Matrix.zero(gs.qd_float, 3, 3)
    gradient = elements.gradient[i_e]
    if qd.static(elements.has_plastic) and elements.is_plastic[i_e]:
        gradient = gradient @ elements.plastic_inverse[i_e, i_b]
    n_corners = 4
    if qd.static(elements.has_shell):
        n_corners = 3 + elements.is_volume[i_e]
    origin = vertices.pos[elements.indices[i_e][0], i_b]
    if i_local == 0:
        origin = trial
    for corner in qd.static(range(1, 4)):
        if corner < n_corners:
            pos = vertices.pos[elements.indices[i_e][corner], i_b]
            if corner == i_local:
                pos = trial
            F += (pos - origin).outer_product(gradient[corner, :])
    g = gradient[i_local, :]
    mu, lam = elements.mu[i_e], elements.lam[i_e]
    identity = qd.Matrix.identity(gs.qd_float, 3)
    energy = qd.cast(0.0, gs.qd_float)
    force = qd.Vector.zero(gs.qd_float, 3)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    if qd.static(not elements.has_shell) or elements.is_volume[i_e]:
        cofactor = qd.Matrix.cols([F[:, 1].cross(F[:, 2]), F[:, 2].cross(F[:, 0]), F[:, 0].cross(F[:, 1])])
        J = F[:, 0].dot(cofactor[:, 0])
        a = cofactor @ g
        if J > 0.0:
            logarithm = qd.log(J)
            energy = 0.5 * mu * (F.norm_sqr() - 3.0) - mu * logarithm + 0.5 * lam * logarithm**2
            force = mu * (F @ g) + (lam * logarithm - mu) / J * a
            curvature = qd.max(0.0, mu + lam * (1.0 - logarithm)) / J**2
            hessian = mu * g.norm_sqr() * identity + curvature * a.outer_product(a)
        else:
            energy = 1e30
    else:
        G = 0.5 * (F.transpose() @ F - identity)
        G[2, 2] = 0.0
        trace = G.trace()
        S = 2.0 * mu * G + lam * trace * identity
        u = F @ g
        energy = mu * G.norm_sqr() + 0.5 * lam * trace**2
        force = F @ S @ g
        hessian = (lam + mu) * u.outer_product(u) + mu * g.norm_sqr() * F @ F.transpose()
        hessian += qd.max(g.dot(S @ g), 0.0) * identity
    measure = elements.measure[i_e]
    return measure * energy, measure * force, measure * hessian


@qd.func
def contact_potential(gap, stiffness, distance, is_shell):
    energy, slope, curvature = 0.0, 0.0, 0.0
    if gap < distance:
        penetration = distance - gap
        energy = 0.5 * stiffness * penetration**2
        slope, curvature = -stiffness * penetration, stiffness
        if is_shell and 1e-5 < gap < 0.5 * distance:
            tau = 0.5 * distance
            coefficient = 0.5 * tau**2 * stiffness
            energy = coefficient * (qd.log(tau / gap) + 1.0 / distance)
            slope, curvature = -coefficient / gap, coefficient / gap**2
    return energy, slope, curvature


@qd.func
def contact_response(displacement, normal, gap, stiffness, friction, distance, epsilon, is_shell):
    energy, slope, curvature = contact_potential(gap, stiffness, distance, is_shell)
    gradient = slope * normal
    hessian = curvature * normal.outer_product(normal)
    if slope < 0.0 and friction > 0.0:
        tangent = displacement - normal * displacement.dot(normal)
        length = tangent.norm()
        if length > 0.0:
            factor = 1.0 / length
            potential = length - epsilon / 3.0
            if length < epsilon:
                factor = (2.0 - length / epsilon) / epsilon
                potential = length**2 / epsilon - length**3 / (3.0 * epsilon**2)
            load = -friction * slope
            energy += load * potential
            gradient += load * factor * tangent
            # Lagging the tangent direction gives a positive semi-definite (PSD) friction approximation.
            hessian += load * factor * (qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal))
    return energy, gradient, hessian


@qd.kernel
def predict(dt: float, gravity: qd.Tensor, vertices: V_ANNOTATION, errno: qd.Tensor):
    for i_v, i_b in vertices.pos:
        vertices.prev[i_v, i_b] = vertices.pos[i_v, i_b]
        vertices.target[i_v, i_b] = vertices.pos[i_v, i_b] + dt * vertices.vel[i_v, i_b]
        vertices.target[i_v, i_b] += dt**2 * (gravity[i_b] + vertices.forces[i_v, i_b] / vertices.mass[i_v])
        if vertices.is_fixed[i_v, i_b]:
            vertices.target[i_v, i_b] = vertices.fixed_pos[i_v, i_b]
        if qd.math.isnan(vertices.target[i_v, i_b]).any() or qd.math.isinf(vertices.target[i_v, i_b]).any():
            qd.atomic_or(errno[None], ErrorCode.INVALID_VBD_STATE)


@qd.kernel
def prepare_prediction(vertices: V_ANNOTATION):
    for i_v, i_b in vertices.pos:
        vertices.next_pos[i_v, i_b] = vertices.target[i_v, i_b]


@qd.func
def prediction_fraction(i_e, i_b, vertices: V_ANNOTATION, elements: V_ANNOTATION, errno: qd.Tensor):
    F = qd.Matrix.zero(gs.qd_float, 3, 3)
    delta_F = F
    first = elements.indices[i_e][0]
    origin = vertices.pos[first, i_b]
    motion = vertices.next_pos[first, i_b] - origin
    is_prescribed = vertices.is_fixed[first, i_b]
    for corner in range(1, 3 + elements.is_volume[i_e]):
        i_v = elements.indices[i_e][corner]
        is_prescribed = is_prescribed and vertices.is_fixed[i_v, i_b]
        gradient = elements.gradient[i_e][corner, :]
        F += (vertices.pos[i_v, i_b] - origin).outer_product(gradient)
        delta_F += (vertices.next_pos[i_v, i_b] - vertices.pos[i_v, i_b] - motion).outer_product(gradient)
    fraction = qd.cast(1.0, gs.qd_float)
    if elements.is_volume[i_e]:
        if is_prescribed and (F + delta_F).determinant() <= 0.0:
            qd.atomic_or(errno[None], ErrorCode.INVALID_VBD_STATE)
        fraction = qd.min(1.0, 0.9 / qd.max((F.inverse() @ delta_F).norm(), 1e-30))
    else:
        gram = F.transpose() @ F
        eigen_min = 0.5 * (gram[0, 0] + gram[1, 1] - qd.sqrt((gram[0, 0] - gram[1, 1]) ** 2 + 4.0 * gram[0, 1] ** 2))
        fraction = qd.min(1.0, 0.9 * qd.sqrt(qd.max(eigen_min, 0.0)) / qd.max(delta_F.norm(), 1e-30))
    return fraction


@qd.kernel
def limit_prediction(
    scales: qd.Tensor,
    has_correction: qd.Tensor,
    vertices: V_ANNOTATION,
    elements: V_ANNOTATION,
    errno: qd.Tensor,
):
    for i_v, i_b in vertices.pos:
        scales[i_v, i_b] = 1.0
    for i_e, i_b in elements.plastic_strain:
        if prediction_fraction(i_e, i_b, vertices, elements, errno) < 1.0:
            has_correction[None] = 1
            for corner in range(3 + elements.is_volume[i_e]):
                qd.atomic_min(scales[elements.indices[i_e][corner], i_b], 0.5)
    for i_v, i_b in vertices.pos:
        vertices.next_pos[i_v, i_b] = vertices.pos[i_v, i_b] + scales[i_v, i_b] * (
            vertices.next_pos[i_v, i_b] - vertices.pos[i_v, i_b]
        )


@qd.kernel
def limit_initialization(scale: qd.Tensor, vertices: V_ANNOTATION, elements: V_ANNOTATION, errno: qd.Tensor):
    for i_e, i_b in elements.plastic_strain:
        first = elements.indices[i_e][0]
        fraction = prediction_fraction(i_e, i_b, vertices, elements, errno)
        qd.atomic_min(scale[vertices.entity_idx[first], i_b], fraction)


@qd.kernel
def apply_prediction(
    is_shell: bool, scale: qd.Tensor, vertices: V_ANNOTATION, collision: V_ANNOTATION, errno: qd.Tensor
):
    for i_v, i_b in vertices.pos:
        if (errno[None] & ErrorCode.INVALID_VBD_STATE) == 0 and vertices.is_volume[i_v] != is_shell:
            fraction = scale[vertices.entity_idx[i_v], i_b]
            displacement = fraction * (vertices.next_pos[i_v, i_b] - vertices.pos[i_v, i_b])
            if qd.static(collision.has_shell):  # noqa: SIM102
                if is_shell:
                    translation = qd.Vector.zero(gs.qd_float, 3)
                    i_entity = vertices.entity_idx[i_v]
                    if collision.translation[i_entity, i_b][3] == 0.0 and not collision.has_contact[i_entity, i_b]:
                        translation = collision.translation[i_entity, i_b][:3] * fraction
                        translation *= qd.min(
                            1.0, collision.translation_bound[i_entity, i_b] / qd.max(translation.norm(), 1e-30)
                        )
                    # Common translation preserves self-distances; external clearance bounds the moving frame.
                    vertices.collision_origin[i_v, i_b] += translation
                    vertices.bound[i_v, i_b] = qd.min(
                        vertices.bound[i_v, i_b],
                        qd.max(0.0, collision.translation_bound[i_entity, i_b] - translation.norm()),
                    )
                    if not vertices.is_fixed[i_v, i_b]:
                        if collision.has_contact[i_entity, i_b]:
                            displacement = qd.Vector.zero(gs.qd_float, 3)
                        else:
                            displacement -= translation
                            displacement *= qd.min(1.0, vertices.bound[i_v, i_b] / qd.max(displacement.norm(), 1e-30))
                            displacement += translation
            vertices.pos[i_v, i_b] += displacement


@qd.func
def local_response(
    i_v,
    i_b,
    i_t,
    trial,
    dt,
    vertices: V_ANNOTATION,
    elements: V_ANNOTATION,
    hinges: V_ANNOTATION,
    planes: V_ANNOTATION,
    options: V_ANNOTATION,
):
    mass = vertices.mass[i_v] / dt**2
    delta = trial - vertices.target[i_v, i_b]
    energy = qd.cast(0.0, gs.qd_float)
    gradient = qd.Vector.zero(gs.qd_float, 3)
    hessian = qd.Matrix.zero(gs.qd_float, 3, 3)
    if i_t == 0:
        energy = 0.5 * mass * delta.norm_sqr()
        gradient = mass * delta
        hessian = mass * qd.Matrix.identity(gs.qd_float, 3)
    i_a = vertices.elem_offsets[i_v] + i_t
    while i_a < vertices.elem_offsets[i_v + 1]:
        entry = vertices.elem_entries[i_a]
        value, force, matrix = element_response(entry // 4, entry % 4, i_b, trial, vertices, elements)
        energy += value
        gradient += force
        hessian += matrix
        i_a += options.n_vertex_threads
    if qd.static(hinges.n_hinges > 0):
        i_a = vertices.hinge_offsets[i_v] + i_t
        while i_a < vertices.hinge_offsets[i_v + 1]:
            entry = vertices.hinge_entries[i_a]
            i_h, i_local = entry // 4, entry % 4
            positions = qd.Matrix.zero(gs.qd_float, 4, 3)
            for corner in qd.static(range(4)):
                positions[corner, :] = vertices.pos[hinges.indices[i_h][corner], i_b]
                if corner == i_local:
                    positions[corner, :] = trial
            angle, q = hinge_angle_gradient(i_local, positions[0, :], positions[1, :], positions[2, :], positions[3, :])
            residual = wrapped_angle(angle - hinges.rest[i_h, i_b])
            stiffness = hinges.stiffness[i_h]
            energy += 0.5 * stiffness * residual**2
            gradient += stiffness * residual * q
            hessian += stiffness * q.outer_product(q)
            i_a += options.n_vertex_threads
    if qd.static(planes.n_planes > 0):
        i_p = i_t
        while i_p < planes.n_planes:
            normal = planes.normal[i_p]
            gap = trial.dot(normal) - planes.offset[i_p] - vertices.radius[i_v]
            stiffness = options.contact_stiffness
            if qd.static(elements.has_shell) and not vertices.is_volume[i_v]:
                stiffness = options.shell_contact_stiffness
            value, force, matrix = contact_response(
                trial - vertices.prev[i_v, i_b],
                normal,
                gap,
                stiffness,
                vertices.friction[i_v],
                distance=options.contact_d_hat,
                epsilon=options.friction_epsilon * dt,
                is_shell=False,
            )
            energy += value
            gradient += force
            hessian += matrix
            i_p += options.n_vertex_threads
    return energy, gradient, hessian


@qd.kernel
def finish(
    pairs: qd.types.ndarray(),
    kinds: qd.types.ndarray(),
    batches: qd.types.ndarray(),
    dt: float,
    gravity: qd.Tensor,
    vertices: V_ANNOTATION,
    elements: V_ANNOTATION,
    hinges: V_ANNOTATION,
    planes: V_ANNOTATION,
    errno: qd.Tensor,
):
    for i_v, i_b in vertices.pos:
        # Updating the predicted velocity with the solve correction avoids amplifying position rounding by 1 / dt.
        vertices.vel[i_v, i_b] += dt * (gravity[i_b] + vertices.forces[i_v, i_b] / vertices.mass[i_v])
        vertices.vel[i_v, i_b] += (vertices.pos[i_v, i_b] - vertices.target[i_v, i_b]) / dt
        if vertices.is_fixed[i_v, i_b]:
            vertices.vel[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        if qd.math.isnan(vertices.pos[i_v, i_b]).any() or qd.math.isinf(vertices.pos[i_v, i_b]).any():
            qd.atomic_or(errno[None], ErrorCode.INVALID_VBD_STATE)
        if qd.static(planes.n_planes > 0):
            for i_p in range(planes.n_planes):
                gap = vertices.pos[i_v, i_b].dot(planes.normal[i_p]) - planes.offset[i_p] - vertices.radius[i_v]
                if gap < 0.0:
                    qd.atomic_or(errno[None], ErrorCode.VBD_CONTACT_PENETRATION)
    for i_c in range(pairs.shape[0]):
        _weights, _normal, gap, _displacement, is_active = contact_geometry(
            i_c, batches[i_c], -1, qd.Vector.zero(gs.qd_float, 3), pairs, kinds, vertices
        )
        if is_active and gap < 0.0:
            qd.atomic_or(errno[None], ErrorCode.VBD_CONTACT_PENETRATION)
    for i_e, i_b in elements.plastic_strain:
        if elements.is_volume[i_e]:
            F = qd.Matrix.zero(gs.qd_float, 3, 3)
            origin = vertices.pos[elements.indices[i_e][0], i_b]
            for corner in qd.static(range(1, 4)):
                F += (vertices.pos[elements.indices[i_e][corner], i_b] - origin).outer_product(
                    elements.gradient[i_e][corner, :]
                )
            if F.determinant() <= 0.0 or qd.math.isnan(F).any() or qd.math.isinf(F).any():
                qd.atomic_or(errno[None], ErrorCode.INVALID_VBD_STATE)
            elif elements.is_plastic[i_e] and (errno[None] & ErrorCode.INVALID_VBD_STATE) == 0:
                Fe = F @ elements.plastic_inverse[i_e, i_b]
                _rotation, sigma, V = qd.svd(Fe, gs.qd_float)
                logarithm = qd.Vector([qd.log(sigma[0, 0]), qd.log(sigma[1, 1]), qd.log(sigma[2, 2])])
                deviator = logarithm - logarithm.sum() / 3.0
                magnitude = deviator.norm()
                limit = elements.yield_strain[i_e] + elements.hardening[i_e] * elements.plastic_strain[i_e, i_b]
                flow = qd.max(0.0, magnitude - limit) / (1.0 + elements.hardening[i_e])
                if flow > 0.0:
                    scale = qd.exp(-flow / magnitude * deviator)
                    correction = (
                        V
                        @ qd.Matrix([[scale[0], 0.0, 0.0], [0.0, scale[1], 0.0], [0.0, 0.0, scale[2]]])
                        @ V.transpose()
                    )
                    inverse = elements.plastic_inverse[i_e, i_b] @ correction
                    elements.plastic_inverse[i_e, i_b] = inverse / inverse.determinant() ** (1.0 / 3.0)
                    elements.plastic_strain[i_e, i_b] += flow
    if qd.static(hinges.n_hinges > 0):
        for i_h, i_b in hinges.rest:
            if hinges.is_plastic[i_h] and (errno[None] & ErrorCode.INVALID_VBD_STATE) == 0:
                ids = hinges.indices[i_h]
                angle, _angle_gradient = hinge_angle_gradient(
                    0,
                    vertices.pos[ids[0], i_b],
                    vertices.pos[ids[1], i_b],
                    vertices.pos[ids[2], i_b],
                    vertices.pos[ids[3], i_b],
                )
                delta = wrapped_angle(angle - hinges.rest[i_h, i_b])
                limit = hinges.yield_angle[i_h] + hinges.hardening[i_h] * hinges.strain[i_h, i_b]
                flow = qd.max(0.0, qd.abs(delta) - limit) / (1.0 + hinges.hardening[i_h])
                hinges.rest[i_h, i_b] += qd.math.sign(delta) * flow
                hinges.strain[i_h, i_b] += flow


@qd.kernel
def validate_state(vertices: V_ANNOTATION, elements: V_ANNOTATION, hinges: V_ANNOTATION, errno: qd.Tensor):
    for i_v, i_b in vertices.pos:
        pos, vel = vertices.pos[i_v, i_b], vertices.vel[i_v, i_b]
        if qd.math.isnan(pos).any() or qd.math.isinf(pos).any() or qd.math.isnan(vel).any() or qd.math.isinf(vel).any():
            qd.atomic_or(errno[None], ErrorCode.INVALID_VBD_STATE)
    for i_e, i_b in elements.plastic_strain:
        F = qd.Matrix.zero(gs.qd_float, 3, 3)
        origin = vertices.pos[elements.indices[i_e][0], i_b]
        for corner in range(1, 3 + elements.is_volume[i_e]):
            F += (vertices.pos[elements.indices[i_e][corner], i_b] - origin).outer_product(
                elements.gradient[i_e][corner, :]
            )
        measure = F[:, 0].cross(F[:, 1]).norm_sqr()
        if elements.is_volume[i_e]:
            measure = F.determinant()
        inverse = elements.plastic_inverse[i_e, i_b]
        strain = elements.plastic_strain[i_e, i_b]
        if (
            measure <= 0.0
            or qd.math.isnan(F).any()
            or qd.math.isinf(F).any()
            or inverse.determinant() <= 0.0
            or qd.math.isnan(inverse).any()
            or qd.math.isinf(inverse).any()
            or strain < 0.0
            or qd.math.isnan(strain)
            or qd.math.isinf(strain)
        ):
            qd.atomic_or(errno[None], ErrorCode.INVALID_VBD_STATE)
    if qd.static(hinges.n_hinges > 0):
        for i_h, i_b in hinges.rest:
            rest, strain = hinges.rest[i_h, i_b], hinges.strain[i_h, i_b]
            if (
                qd.math.isnan(rest)
                or qd.math.isinf(rest)
                or strain < 0.0
                or qd.math.isnan(strain)
                or qd.math.isinf(strain)
            ):
                qd.atomic_or(errno[None], ErrorCode.INVALID_VBD_STATE)
