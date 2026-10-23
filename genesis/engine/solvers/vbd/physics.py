import quadrants as qd

import genesis as gs
from genesis.utils.array_class import ErrorCode, V_ANNOTATION


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
def barrier(gap, stiffness, d_hat):
    energy, gradient, hessian = 0.0, 0.0, 0.0
    if gap <= 0.0:
        energy = 1e30
    elif gap < d_hat:
        delta = gap - d_hat
        log_gap = qd.log(gap / d_hat)
        energy = -stiffness * delta**2 * log_gap
        gradient = -stiffness * (2.0 * delta * log_gap + delta**2 / gap)
        hessian = -stiffness * (2.0 * log_gap + 4.0 * delta / gap - (delta / gap) ** 2)
    return energy, gradient, hessian


@qd.func
def contact_response(displacement, normal, gap, stiffness, friction, normal_load, d_hat, epsilon):
    energy, slope, curvature = barrier(gap, stiffness, d_hat)
    tangent = displacement - normal * displacement.dot(normal)
    distance = qd.sqrt(tangent.norm_sqr() + epsilon**2)
    projector = qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal)
    energy += friction * normal_load * (distance - epsilon)
    gradient = slope * normal + friction * normal_load / distance * tangent
    hessian = curvature * normal.outer_product(normal)
    hessian += friction * normal_load * (projector / distance - tangent.outer_product(tangent) / distance**3)
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
def update_translation(
    mean_displacement: qd.Tensor,
    fixed_translation: qd.Tensor,
    vertices: V_ANNOTATION,
    is_prediction: qd.template(),
):
    for i_entity, i_b in mean_displacement:
        mean_displacement[i_entity, i_b] = qd.Vector.zero(gs.qd_float, 3)
        fixed_translation[i_entity, i_b] = qd.Vector.zero(gs.qd_float, 4)
    for i_v, i_b in vertices.pos:
        i_entity = vertices.entity_idx[i_v]
        weight = vertices.mass[i_v] / vertices.entity_mass[i_entity]
        displacement = vertices.target[i_v, i_b] - vertices.pos[i_v, i_b]
        if qd.static(is_prediction):
            displacement = vertices.next_pos[i_v, i_b] - vertices.pos[i_v, i_b]
        mean_displacement[i_entity, i_b] += weight * displacement
        if vertices.is_fixed[i_v, i_b]:
            fixed_translation[i_entity, i_b] += weight * qd.Vector(
                [displacement[0], displacement[1], displacement[2], 1.0]
            )
    for i_entity, i_b in mean_displacement:
        if fixed_translation[i_entity, i_b][3] > 0.0:
            mean_displacement[i_entity, i_b] = (
                fixed_translation[i_entity, i_b][:3] / fixed_translation[i_entity, i_b][3]
            )


@qd.kernel
def prepare_prediction(
    mean_displacement: qd.Tensor,
    vertices: V_ANNOTATION,
    planes: V_ANNOTATION,
    d_hat: float,
    errno: qd.Tensor,
):
    for i_v, i_b in vertices.pos:
        pos = vertices.pos[i_v, i_b]
        translation = mean_displacement[vertices.entity_idx[i_v], i_b]
        relative = vertices.target[i_v, i_b] - pos - translation
        fraction = qd.cast(1.0, gs.qd_float)
        if not vertices.is_fixed[i_v, i_b]:
            limit = qd.max(2.0 * d_hat, 0.25 * vertices.edge_length[i_v])
            fraction = qd.min(1.0, 0.9 * limit / qd.max(relative.norm(), 1e-30))
        guess = pos + translation + fraction * relative
        if qd.static(planes.n_planes > 0):
            for i_p in range(planes.n_planes):
                normal = planes.normal[i_p]
                gap = pos.dot(normal) - planes.offset[i_p] - vertices.radius[i_v]
                if gap <= 0.0:
                    qd.atomic_or(errno[None], ErrorCode.INVALID_VBD_CONTACT)
                clearance = qd.min(0.1 * d_hat, 0.1 * gap)
                predicted_gap = guess.dot(normal) - planes.offset[i_p] - vertices.radius[i_v]
                guess += qd.max(0.0, clearance - predicted_gap) * normal
        vertices.next_pos[i_v, i_b] = guess


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
def limit_initialization(
    scale: qd.Tensor,
    mean_displacement: qd.Tensor,
    vertices: V_ANNOTATION,
    elements: V_ANNOTATION,
    planes: V_ANNOTATION,
    errno: qd.Tensor,
):
    for i_v, i_b in vertices.pos:
        pos = vertices.pos[i_v, i_b]
        direction = vertices.next_pos[i_v, i_b] - pos
        relative_motion = (direction - mean_displacement[vertices.entity_idx[i_v], i_b]).norm()
        fraction = qd.min(1.0, 0.9 * vertices.motion_budget[i_v, i_b] / qd.max(relative_motion, 1e-30))
        if qd.static(planes.n_planes > 0):
            for i_p in range(planes.n_planes):
                normal = planes.normal[i_p]
                gap = pos.dot(normal) - planes.offset[i_p] - vertices.radius[i_v]
                if gap <= 0.0:
                    qd.atomic_or(errno[None], ErrorCode.INVALID_VBD_CONTACT)
                if direction.dot(normal) < 0.0:
                    fraction = qd.min(fraction, 0.9 * qd.max(gap, 0.0) / -direction.dot(normal))
        qd.atomic_min(scale[vertices.entity_idx[i_v], i_b], fraction)
    for i_e, i_b in elements.plastic_strain:
        first = elements.indices[i_e][0]
        fraction = prediction_fraction(i_e, i_b, vertices, elements, errno)
        qd.atomic_min(scale[vertices.entity_idx[first], i_b], fraction)


@qd.kernel
def apply_prediction(scale: qd.Tensor, mean_displacement: qd.Tensor, vertices: V_ANNOTATION, errno: qd.Tensor):
    for i_v, i_b in vertices.pos:
        if errno[None] == 0:
            fraction = scale[vertices.entity_idx[i_v], i_b]
            vertices.pos[i_v, i_b] += fraction * (vertices.next_pos[i_v, i_b] - vertices.pos[i_v, i_b])
            # A common translation preserves every self-contact distance and consumes no relative motion budget.
            vertices.collision_origin[i_v, i_b] += fraction * mean_displacement[vertices.entity_idx[i_v], i_b]


@qd.func
def local_response(
    i_v,
    i_b,
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
    energy = 0.5 * mass * delta.norm_sqr()
    gradient = mass * delta
    hessian = mass * qd.Matrix.identity(gs.qd_float, 3)
    for entry_idx in range(vertices.elem_offsets[i_v], vertices.elem_offsets[i_v + 1]):
        entry = vertices.elem_entries[entry_idx]
        value, force, matrix = element_response(entry // 4, entry % 4, i_b, trial, vertices, elements)
        energy += value
        gradient += force
        hessian += matrix
    if qd.static(hinges.n_hinges > 0):
        for entry_idx in range(vertices.hinge_offsets[i_v], vertices.hinge_offsets[i_v + 1]):
            entry = vertices.hinge_entries[entry_idx]
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
    if qd.static(planes.n_planes > 0):
        for i_p in range(planes.n_planes):
            normal = planes.normal[i_p]
            gap = trial.dot(normal) - planes.offset[i_p] - vertices.radius[i_v]
            initial_gap = vertices.prev[i_v, i_b].dot(normal) - planes.offset[i_p] - vertices.radius[i_v]
            stiffness = options.contact_stiffness * mass
            _, slope, _ = barrier(initial_gap, stiffness, options.contact_d_hat)
            value, force, matrix = contact_response(
                trial - vertices.prev[i_v, i_b],
                normal,
                gap,
                stiffness,
                vertices.friction[i_v],
                -slope,
                options.contact_d_hat,
                options.friction_epsilon * dt,
            )
            energy += value
            gradient += force
            hessian += matrix
    return energy, gradient, hessian


@qd.kernel
def finish(dt: float, vertices: V_ANNOTATION, elements: V_ANNOTATION, hinges: V_ANNOTATION, errno: qd.Tensor):
    for i_v, i_b in vertices.pos:
        vertices.vel[i_v, i_b] = (vertices.pos[i_v, i_b] - vertices.prev[i_v, i_b]) / dt
        if vertices.is_fixed[i_v, i_b]:
            vertices.vel[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        if qd.math.isnan(vertices.pos[i_v, i_b]).any() or qd.math.isinf(vertices.pos[i_v, i_b]).any():
            qd.atomic_or(errno[None], ErrorCode.INVALID_VBD_STATE)
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
            elif elements.is_plastic[i_e] and errno[None] == 0:
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
            if hinges.is_plastic[i_h] and errno[None] == 0:
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
