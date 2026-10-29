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
def element_response(i_e, i_local, i_b, trial, dt, vertices: V_ANNOTATION, elements: V_ANNOTATION):
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
        is_log = False
        if qd.static(elements.has_log_neohookean):
            is_log = elements.is_log_neohookean[i_e]
        if is_log:
            if J > 0.0:
                logarithm = qd.log(J)
                # Retain the rest-energy offset so backtracking accounts for cancellation at the elastic energy scale.
                energy = 0.5 * mu * F.norm_sqr() - mu * logarithm + 0.5 * lam * logarithm**2
                force = mu * (F @ g) + (lam * logarithm - mu) / J * a
                # Project the volume curvature to keep the local Hessian positive semi-definite (PSD).
                curvature = qd.max(0.0, mu + lam * (1.0 - logarithm)) / J**2
                hessian = mu * g.norm_sqr() * identity + curvature * a.outer_product(a)
            else:
                # A finite sentinel rejects out-of-domain trials during energy backtracking.
                energy = 1e30
        else:
            # This volumetric coefficient preserves the small-strain Young's modulus and Poisson's ratio.
            volume_stiffness = lam + mu
            delta_J = J - 1.0
            energy = 0.5 * mu * (F.norm_sqr() - 3.0) - mu * delta_J + 0.5 * volume_stiffness * delta_J**2
            force = mu * (F @ g) + (volume_stiffness * delta_J - mu) * a
            # The determinant is affine in one vertex, so its second derivative vanishes in this block.
            hessian = mu * g.norm_sqr() * identity + volume_stiffness * a.outer_product(a)
    else:
        F_shell = F[:, :2]
        g_shell = g[:2]
        G = 0.5 * (F_shell.transpose() @ F_shell - qd.Matrix.identity(gs.qd_float, 2))
        trace = G.trace()
        S = 2.0 * mu * G + lam * trace * qd.Matrix.identity(gs.qd_float, 2)
        u = F_shell @ g_shell
        energy = mu * G.norm_sqr() + 0.5 * lam * trace**2
        force = F_shell @ S @ g_shell
        hessian = (lam + mu) * u.outer_product(u) + mu * g_shell.norm_sqr() * (F_shell @ F_shell.transpose())
        hessian += g_shell.dot(S @ g_shell) * identity
        displacement = trial - vertices.prev[elements.indices[i_e][i_local], i_b]
        damping_hessian = (elements.damping[i_e] / dt) * hessian
        force += damping_hessian @ displacement
        hessian += damping_hessian
        energy += 0.5 * displacement.dot(damping_hessian @ displacement)
    measure = elements.measure[i_e]
    return measure * energy, measure * force, measure * hessian


@qd.func
def contact_potential(gap, stiffness, distance, is_shell):
    energy, slope, curvature = 0.0, 0.0, 0.0
    if gap < distance:
        penetration = distance - gap
        energy = 0.5 * stiffness * penetration**2
        slope, curvature = -stiffness * penetration, stiffness
        if is_shell and 1e-7 < gap < 0.5 * distance:
            tau = 0.5 * distance
            coefficient = 0.5 * tau**2 * stiffness
            # The additive energy offset uses a centimetre reference length, expressed here in metres.
            energy = coefficient * (qd.log(tau / gap) + 0.01 / distance)
            slope, curvature = -coefficient / gap, coefficient / gap**2
    return energy, slope, curvature


@qd.func
def contact_response(displacement, normal, gap, stiffness, friction, distance, epsilon, tangent_projection, is_shell):
    energy, slope, curvature = contact_potential(gap, stiffness, distance, is_shell)
    gradient = slope * normal
    hessian = curvature * normal.outer_product(normal)
    if slope < 0.0 and friction > 0.0:
        tangent = tangent_projection @ displacement
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
            hessian += load * factor * tangent_projection
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
def prepare_prediction(vertices: V_ANNOTATION, options: V_ANNOTATION):
    for i_v, i_b in vertices.pos:
        vertices.next_pos[i_v, i_b] = vertices.target[i_v, i_b]
        if qd.static(options.is_shell_previous) and not vertices.is_volume[i_v] and not vertices.is_fixed[i_v, i_b]:
            vertices.next_pos[i_v, i_b] = vertices.prev[i_v, i_b]


@qd.func
def prediction_fraction(i_e, i_b, vertices: V_ANNOTATION, elements: V_ANNOTATION):
    F = qd.Matrix.zero(gs.qd_float, 3, 3)
    delta_F = F
    first = elements.indices[i_e][0]
    origin = vertices.pos[first, i_b]
    motion = vertices.next_pos[first, i_b] - origin
    for corner in range(1, 3 + elements.is_volume[i_e]):
        i_v = elements.indices[i_e][corner]
        gradient = elements.gradient[i_e][corner, :]
        F += (vertices.pos[i_v, i_b] - origin).outer_product(gradient)
        delta_F += (vertices.next_pos[i_v, i_b] - vertices.pos[i_v, i_b] - motion).outer_product(gradient)
    fraction = qd.cast(1.0, gs.qd_float)
    if elements.is_volume[i_e]:
        # Limit the initial guess under concentrated loads; local solves can cross zero volume during recovery.
        if F.determinant() != 0.0:
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
):
    for i_v, i_b in vertices.pos:
        scales[i_v, i_b] = 1.0
    for i_e, i_b in elements.plastic_strain:
        if prediction_fraction(i_e, i_b, vertices, elements) < 1.0:
            has_correction[None] = 1
            for corner in range(3 + elements.is_volume[i_e]):
                qd.atomic_min(scales[elements.indices[i_e][corner], i_b], 0.5)
    for i_v, i_b in vertices.pos:
        vertices.next_pos[i_v, i_b] = vertices.pos[i_v, i_b] + scales[i_v, i_b] * (
            vertices.next_pos[i_v, i_b] - vertices.pos[i_v, i_b]
        )


@qd.kernel
def limit_initialization(scale: qd.Tensor, vertices: V_ANNOTATION, elements: V_ANNOTATION):
    for i_e, i_b in elements.plastic_strain:
        first = elements.indices[i_e][0]
        fraction = prediction_fraction(i_e, i_b, vertices, elements)
        qd.atomic_min(scale[vertices.entity_idx[first], i_b], fraction)


@qd.kernel
def apply_prediction(
    is_shell: bool,
    scale: qd.Tensor,
    vertices: V_ANNOTATION,
    collision: V_ANNOTATION,
    planes: V_ANNOTATION,
    errno: qd.Tensor,
):
    for i_v, i_b in vertices.pos:
        if (errno[None] & ErrorCode.INVALID_VBD_STATE) == 0 and vertices.is_volume[i_v] != is_shell:
            fraction = scale[vertices.entity_idx[i_v], i_b]
            displacement = fraction * (vertices.next_pos[i_v, i_b] - vertices.pos[i_v, i_b])
            if qd.static(collision.has_shell):  # noqa: SIM102
                if is_shell:
                    translation = qd.Vector.zero(gs.qd_float, 3)
                    i_entity = vertices.entity_idx[i_v]
                    if (
                        collision.translation[i_entity, i_b][3] == 0.0
                        and collision.contact_flags[i_entity, i_b] & 2 == 0
                    ):
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
                        if collision.contact_flags[i_entity, i_b] & 2 != 0:
                            # External contact retains the previous iterate to avoid unbalanced prediction impulses.
                            displacement = translation
                        else:
                            displacement -= translation
                            displacement *= qd.min(1.0, vertices.bound[i_v, i_b] / qd.max(displacement.norm(), 1e-30))
                            displacement += translation
                        if qd.static(planes.n_planes > 0):
                            # See initialize_shell_bounds for the plane-normal prediction limit.
                            for i_p in range(planes.n_planes):
                                normal = planes.normal[i_p]
                                gap = vertices.pos[i_v, i_b].dot(normal) - planes.offset[i_p] - vertices.radius[i_v]
                                displacement -= qd.min(0.0, displacement.dot(normal) + qd.max(0.0, gap)) * normal
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
        value, force, matrix = element_response(entry // 4, entry % 4, i_b, trial, dt, vertices, elements)
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
            stiffness = hinges.stiffness[i_h]
            force = qd.Vector.zero(gs.qd_float, 3)
            matrix = qd.Matrix.zero(gs.qd_float, 3, 3)
            if hinges.is_quadratic[i_h]:
                n0 = (positions[1, :] - positions[0, :]).cross(positions[2, :] - positions[0, :])
                n1 = (positions[2, :] - positions[3, :]).cross(positions[1, :] - positions[3, :])
                if qd.min(n0.norm_sqr(), n1.norm_sqr()) >= 1e-20:
                    weights = hinges.cotangent[i_h]
                    curvature = qd.Vector.zero(gs.qd_float, 3)
                    for corner in qd.static(range(1, 4)):
                        curvature += weights[corner] * (positions[corner, :] - positions[0, :])
                    energy += 0.5 * stiffness * curvature.norm_sqr()
                    force = stiffness * weights[i_local] * curvature
                    matrix = stiffness * weights[i_local] ** 2 * qd.Matrix.identity(gs.qd_float, 3)
            else:
                angle, q = hinge_angle_gradient(
                    i_local, positions[0, :], positions[1, :], positions[2, :], positions[3, :]
                )
                residual = wrapped_angle(angle - hinges.rest[i_h, i_b])
                energy += 0.5 * stiffness * residual**2
                force = stiffness * residual * q
                matrix = stiffness * q.outer_product(q)
            displacement = trial - vertices.prev[i_v, i_b]
            damping_matrix = (hinges.damping[i_h] / dt) * matrix
            energy += 0.5 * displacement.dot(damping_matrix @ displacement)
            gradient += force + damping_matrix @ displacement
            hessian += matrix + damping_matrix
            i_a += options.n_vertex_threads
    if qd.static(planes.n_planes > 0):
        i_p = i_t
        while i_p < planes.n_planes:
            normal = planes.normal[i_p]
            gap = trial.dot(normal) - planes.offset[i_p] - vertices.radius[i_v]
            stiffness = options.contact_stiffness
            distance = options.contact_d_hat
            friction = vertices.friction[i_v]
            epsilon = options.friction_epsilon * dt
            if qd.static(elements.has_shell) and not vertices.is_volume[i_v]:
                stiffness = options.shell_ground_stiffness
                if qd.static(options.shell_ground_d_hat >= 0.0):
                    gap += vertices.radius[i_v]
                    distance = options.shell_ground_d_hat
                epsilon = options.shell_ground_friction_epsilon * dt
                if qd.static(options.shell_ground_friction >= 0.0):
                    friction = options.shell_ground_friction
            value, force, matrix = contact_response(
                trial - vertices.prev[i_v, i_b],
                normal,
                gap,
                stiffness,
                friction,
                distance,
                epsilon,
                tangent_projection=qd.Matrix.identity(gs.qd_float, 3) - normal.outer_product(normal),
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
    options: V_ANNOTATION,
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
        if qd.static(options.is_contact_diagnostics_enabled and planes.n_planes > 0):
            for i_p in range(planes.n_planes):
                gap = vertices.pos[i_v, i_b].dot(planes.normal[i_p]) - planes.offset[i_p] - vertices.radius[i_v]
                if qd.static(options.shell_ground_d_hat >= 0.0) and not vertices.is_volume[i_v]:
                    gap += vertices.radius[i_v]
                if gap < 0.0:
                    qd.atomic_or(errno[None], ErrorCode.VBD_CONTACT_PENETRATION)
    if qd.static(options.is_contact_diagnostics_enabled):
        for i_c in range(pairs.shape[0]):
            _weights, _normal, gap, _displacement, _tangent_projection, is_active = contact_geometry(
                i_c, batches[i_c], -1, qd.Vector.zero(gs.qd_float, 3), pairs, kinds, vertices
            )
            if is_active and gap < 0.0:
                qd.atomic_or(errno[None], ErrorCode.VBD_CONTACT_PENETRATION)
    if qd.static(elements.has_plastic):
        for i_e, i_b in elements.plastic_strain:
            if elements.is_plastic[i_e] and (errno[None] & ErrorCode.INVALID_VBD_STATE) == 0:
                F = qd.Matrix.zero(gs.qd_float, 3, 3)
                origin = vertices.pos[elements.indices[i_e][0], i_b]
                for corner in qd.static(range(1, 4)):
                    F += (vertices.pos[elements.indices[i_e][corner], i_b] - origin).outer_product(
                        elements.gradient[i_e][corner, :]
                    )
                J = F.determinant()
                if qd.math.isnan(F).any() or qd.math.isinf(F).any() or qd.math.isnan(J) or qd.math.isinf(J):
                    qd.atomic_or(errno[None], ErrorCode.INVALID_VBD_STATE)
                # Plastic flow resumes after elastic forces restore positive orientation.
                elif J > 0.0:
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
        is_positive_required = not elements.is_volume[i_e]
        if qd.static(elements.has_log_neohookean):
            is_positive_required = is_positive_required or elements.is_log_neohookean[i_e]
        if (
            (is_positive_required and measure <= 0.0)
            or qd.math.isnan(measure)
            or qd.math.isinf(measure)
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
