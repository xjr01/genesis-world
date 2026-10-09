import enum
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import quadrants as qd
from scipy.spatial import ConvexHull

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.entities.pbd_entity import PBD3DSolidEntity
from genesis.engine.solvers.pbd_solver import (
    func_pbd_inertia_twist,
    func_pbd_polar_rotation,
    func_update_pbd_cluster_roots,
)
from genesis.utils.array_class import V_ANNOTATION
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity, RigidLink
    from genesis.engine.simulator import Simulator


class FragmentOwner(enum.IntEnum):
    """Which simulation authority currently advances a fragment: the PBD particle solver or the native rigid solver."""

    PBD = 0
    NATIVE = 1


@qd.func
def func_mat3_to_quat(R):
    """Unit quaternion (w, x, y, z) of a proper rotation matrix."""
    quat = qd.Vector.zero(qd.f64, 4)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = qd.sqrt(trace + 1.0) * 2.0
        quat = qd.Vector([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = qd.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        quat = qd.Vector([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = qd.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        quat = qd.Vector([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = qd.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        quat = qd.Vector([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return quat / quat.norm()


@qd.kernel
def kernel_set_fragment_owner(i_k: qd.i32, envs_idx: qd.types.ndarray(ndim=1), owner: qd.i32, bridge: V_ANNOTATION):
    for i_b_ in range(envs_idx.shape[0]):
        bridge.fragment_owner_q[i_k, envs_idx[i_b_]] = owner


@qd.kernel
def kernel_set_link_externally_driven(
    i_l: qd.i32,
    envs_idx: qd.types.ndarray(ndim=1),
    is_driven: qd.i32,
    dyn_state: array_class.DynState,
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        dyn_state.links.is_externally_driven[i_l, i_b] = is_driven


@qd.kernel
def kernel_set_link_contact_enabled(
    i_l: qd.i32,
    envs_idx: qd.types.ndarray(ndim=1),
    is_contact_enabled: qd.i32,
    dyn_state: array_class.DynState,
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        dyn_state.links.is_contact_enabled[i_l, i_b] = is_contact_enabled


@qd.kernel
def kernel_follow_pbd_fragments(bridge: V_ANNOTATION, pbd: V_ANNOTATION, rigid: V_ANNOTATION):
    """Write each driven fragment link's qpos from the PBD welded cluster fit of the previous substep.

    The welded `clusters_state` cm / rot of a cluster hold the connected-group transform evaluated at the cluster's
    unweighted rest centroid; the link frame sits at the fragment's mass-weighted rest center of mass, so the
    followed position shifts that centroid offset through the same rotation. Velocities stay zero: a shadow link is
    neither integrated nor contacted, and the bridge owns the current and next buffers through the copyback.
    """
    for i_k, i_b in qd.ndrange(bridge.n_fragments_q, pbd._B):
        i_l = bridge.fragment_link_q[i_k]
        if rigid.dyn_state.links.is_externally_driven[i_l, i_b]:
            i_c = bridge.fragment_cluster_q[i_k]
            rot = pbd.clusters_state[i_c, i_b].rot.cast(qd.f64)
            pos = pbd.clusters_state[i_c, i_b].cm.cast(qd.f64) + rot @ bridge.fragment_rest_cm_offset_q[i_k]
            quat = func_mat3_to_quat(rot)
            q_start = bridge.fragment_q_start_q[i_k]
            dof_start = bridge.fragment_dof_start_q[i_k]
            for j in qd.static(range(3)):
                rigid.rigid_info.qpos[q_start + j, i_b] = gs.qd_float(pos[j])
                rigid.rigid_info.qpos_next[q_start + j, i_b] = gs.qd_float(pos[j])
                rigid.dyn_state.dofs.vel[dof_start + j, i_b] = gs.qd_float(0.0)
                rigid.dyn_state.dofs.vel_next[dof_start + j, i_b] = gs.qd_float(0.0)
            for j in qd.static(range(4)):
                rigid.rigid_info.qpos[q_start + 3 + j, i_b] = gs.qd_float(quat[j])
                rigid.rigid_info.qpos_next[q_start + 3 + j, i_b] = gs.qd_float(quat[j])
            for j in qd.static(range(3)):
                rigid.dyn_state.dofs.vel[dof_start + 3 + j, i_b] = gs.qd_float(0.0)
                rigid.dyn_state.dofs.vel_next[dof_start + 3 + j, i_b] = gs.qd_float(0.0)


@qd.kernel
def kernel_scan_handoff_candidates(bridge: V_ANNOTATION, pbd: V_ANNOTATION):
    """Mark still-PBD fragments whose live-seam connected component shrank to themselves as handoff-pending.

    Runs at the substep tail, after the PBD solver finished the substep (bond death, contacts, state update). The
    scan rebuilds the cluster roots first: bonds may have died during this substep while the next welded fit that
    would refresh them only runs in the following one. Eligibility beyond isolation is static per fragment (free,
    unit stiffness, no plasticity) plus a runtime all-members-free-and-active check covering fixed, attached and
    deactivated particles (a fragment revoked to PBD by a public setter only re-enters native ownership once its
    members are all free and active again).
    """
    func_update_pbd_cluster_roots(pbd)
    for i_k, i_b in qd.ndrange(bridge.n_fragments_q, pbd._B):
        bridge.handoff_pending_q[i_k, i_b] = False
        if bridge.fragment_owner_q[i_k, i_b] == int(FragmentOwner.PBD) and bridge.fragment_is_handoff_eligible_q[i_k]:
            i_c = bridge.fragment_cluster_q[i_k]
            if qd.static(pbd._n_bonds > 0):
                if pbd.cluster_roots[i_c, i_b] == i_c:
                    is_singleton = True
                    for j_c in range(pbd._n_clusters):
                        if j_c != i_c and pbd.cluster_roots[j_c, i_b] == i_c:
                            is_singleton = False
                    if is_singleton:
                        is_admissible = True
                        for i_m_ in range(bridge.fragment_member_start_q[i_k], bridge.fragment_member_start_q[i_k + 1]):
                            i_p = bridge.fragment_member_particles_q[i_m_]
                            if not pbd.particles[i_p, i_b].free or not pbd.particles_ng[i_p, i_b].active:
                                is_admissible = False
                        if is_admissible:
                            bridge.handoff_pending_q[i_k, i_b] = True


@qd.kernel
def kernel_commit_fragment_handoffs(bridge: V_ANNOTATION, pbd: V_ANNOTATION, rigid: V_ANNOTATION):
    """Import every pending fragment to the native solver in one shot, before any solver advances this substep.

    Computes mass, center of mass, proper polar rotation, linear momentum and world angular momentum from the real
    particle members, resolves the spin as the minimum-norm solution of R I_rest R^T omega = L_world - CM x P, and
    writes the FREE-link state in the link's own frame (origin at the fragment's rest center of mass): world
    position CM, world quaternion R, world linear velocity v_CM and body-frame angular velocity R^T omega. The
    commit is atomic per fragment and environment: any non-finite or non-positive intermediate leaves the fragment
    PBD-owned and only bumps the observable rejection counter.
    """
    for i_k, i_b in qd.ndrange(bridge.n_fragments_q, pbd._B):
        if bridge.handoff_pending_q[i_k, i_b]:
            bridge.commit_errno_q[i_b] = 0
            member_start = bridge.fragment_member_start_q[i_k]
            member_end = bridge.fragment_member_start_q[i_k + 1]
            rest_cm = bridge.fragment_rest_cm_q[i_k]

            mass_sum = qd.f64(0.0)
            pos_sum = qd.Vector.zero(qd.f64, 3)
            vel_sum = qd.Vector.zero(qd.f64, 3)
            angular_sum = qd.Vector.zero(qd.f64, 3)
            rest_sum = qd.Vector.zero(qd.f64, 3)
            covariance = qd.Matrix.zero(qd.f64, 3, 3)
            for i_m_ in range(member_start, member_end):
                i_p = bridge.fragment_member_particles_q[i_m_]
                weight = qd.f64(pbd.particles_info[i_p].mass)
                pos = pbd.particles[i_p, i_b].pos.cast(qd.f64)
                vel = pbd.particles[i_p, i_b].vel.cast(qd.f64)
                rest = pbd.particles_solid_rest[i_p, i_b].cast(qd.f64) - rest_cm
                mass_sum += weight
                pos_sum += weight * pos
                vel_sum += weight * vel
                angular_sum += weight * pos.cross(vel)
                rest_sum += weight * rest
                covariance += weight * pos.outer_product(rest)

            is_valid = mass_sum > 0.0
            cm = pos_sum / mass_sum
            momentum = vel_sum
            angular_world = angular_sum
            covariance -= cm.outer_product(rest_sum)
            rot = func_pbd_polar_rotation(covariance)
            inertia_world = rot @ bridge.fragment_rest_inertia_q[i_k] @ rot.transpose()
            spin = angular_world - cm.cross(momentum)
            omega = func_pbd_inertia_twist(i_b, inertia_world, spin, bridge.commit_errno_q)
            quat = func_mat3_to_quat(rot)
            v_cm = momentum / mass_sum
            omega_body = rot.transpose() @ omega
            for j in qd.static(range(3)):
                is_valid &= not (qd.math.isnan(cm[j]) or qd.math.isinf(cm[j]))
                is_valid &= not (qd.math.isnan(v_cm[j]) or qd.math.isinf(v_cm[j]))
                is_valid &= not (qd.math.isnan(omega_body[j]) or qd.math.isinf(omega_body[j]))
            for j in qd.static(range(4)):
                is_valid &= not (qd.math.isnan(quat[j]) or qd.math.isinf(quat[j]))
            is_valid &= qd.abs(quat.norm() - 1.0) < qd.f64(1e-6)

            bridge.handoff_pending_q[i_k, i_b] = False
            if is_valid:
                i_l = bridge.fragment_link_q[i_k]
                q_start = bridge.fragment_q_start_q[i_k]
                dof_start = bridge.fragment_dof_start_q[i_k]
                for j in qd.static(range(3)):
                    rigid.rigid_info.qpos[q_start + j, i_b] = gs.qd_float(cm[j])
                    rigid.dyn_state.dofs.vel[dof_start + j, i_b] = gs.qd_float(v_cm[j])
                for j in qd.static(range(4)):
                    rigid.rigid_info.qpos[q_start + 3 + j, i_b] = gs.qd_float(quat[j])
                for j in qd.static(range(3)):
                    rigid.dyn_state.dofs.vel[dof_start + 3 + j, i_b] = gs.qd_float(omega_body[j])
                    rigid.dyn_state.dofs.acc[dof_start + j, i_b] = gs.qd_float(0.0)
                    rigid.dyn_state.dofs.acc[dof_start + 3 + j, i_b] = gs.qd_float(0.0)
                    rigid.dyn_state.links.cfrc_coupling_vel[i_l, i_b][j] = gs.qd_float(0.0)
                    rigid.dyn_state.links.cfrc_coupling_ang[i_l, i_b][j] = gs.qd_float(0.0)
                    rigid.dyn_state.links.contact_force[i_l, i_b][j] = gs.qd_float(0.0)
                rigid.dyn_state.links.is_externally_driven[i_l, i_b] = False
                rigid.dyn_state.links.is_contact_enabled[i_l, i_b] = True
                bridge.fragment_owner_q[i_k, i_b] = int(FragmentOwner.NATIVE)
                qd.atomic_add(bridge.commit_count_q[i_k, i_b], 1)
                for i_m_ in range(member_start, member_end):
                    i_p = bridge.fragment_member_particles_q[i_m_]
                    pbd.particles_ng[i_p, i_b].is_native_owned = True
                    for j in qd.static(range(3)):
                        bridge.commit_capture_pos_q[i_p, i_b, j] = pbd.particles[i_p, i_b].pos[j]
                        bridge.commit_capture_vel_q[i_p, i_b, j] = pbd.particles[i_p, i_b].vel[j]
                for j in qd.static(range(7)):
                    bridge.commit_capture_qpos_q[i_k, i_b, j] = rigid.rigid_info.qpos[q_start + j, i_b]
                for j in qd.static(range(6)):
                    bridge.commit_capture_qvel_q[i_k, i_b, j] = rigid.dyn_state.dofs.vel[dof_start + j, i_b]
                bridge.commit_capture_valid_q[i_k, i_b] = 1
                pbd.is_render_dirty[i_b] = True
            else:
                qd.atomic_add(bridge.commit_rejected_q[i_b], 1)


@qd.kernel
def kernel_sync_native_derived_particles(bridge: V_ANNOTATION, pbd: V_ANNOTATION, rigid: V_ANNOTATION):
    """Rewrite a native-owned fragment's particle pos / vel from its link pose at the substep tail.

    Keeps the public particle arrays a derived output of the native state (first-version substep-end sync): the
    compat particles track the fragment for getters without participating in any PBD response.
    """
    for i_k, i_b in qd.ndrange(bridge.n_fragments_q, pbd._B):
        if bridge.fragment_owner_q[i_k, i_b] == int(FragmentOwner.NATIVE):
            i_l = bridge.fragment_link_q[i_k]
            dof_start = bridge.fragment_dof_start_q[i_k]
            link_pos = rigid.dyn_state.links.pos[i_l, i_b].cast(qd.f64)
            link_quat = rigid.dyn_state.links.quat[i_l, i_b].cast(qd.f64)
            v_cm = qd.Vector(
                [
                    qd.f64(rigid.dyn_state.dofs.vel[dof_start, i_b]),
                    qd.f64(rigid.dyn_state.dofs.vel[dof_start + 1, i_b]),
                    qd.f64(rigid.dyn_state.dofs.vel[dof_start + 2, i_b]),
                ]
            )
            omega_body = qd.Vector(
                [
                    qd.f64(rigid.dyn_state.dofs.vel[dof_start + 3, i_b]),
                    qd.f64(rigid.dyn_state.dofs.vel[dof_start + 4, i_b]),
                    qd.f64(rigid.dyn_state.dofs.vel[dof_start + 5, i_b]),
                ]
            )
            omega_world = gu.qd_transform_by_quat(omega_body, link_quat)
            for i_m_ in range(bridge.fragment_member_start_q[i_k], bridge.fragment_member_start_q[i_k + 1]):
                i_p = bridge.fragment_member_particles_q[i_m_]
                rest = pbd.particles_solid_rest[i_p, i_b].cast(qd.f64) - bridge.fragment_rest_cm_q[i_k]
                offset = gu.qd_transform_by_quat(rest, link_quat)
                pos = link_pos + offset
                vel = v_cm + omega_world.cross(offset)
                for j in qd.static(range(3)):
                    pbd.particles[i_p, i_b].pos[j] = gs.qd_float(pos[j])
                    pbd.particles[i_p, i_b].vel[j] = gs.qd_float(vel[j])
            pbd.is_render_dirty[i_b] = True


@qd.kernel
def kernel_restore_fragment_ownership(
    envs_idx: qd.types.ndarray(ndim=1),
    bridge: V_ANNOTATION,
    rigid: V_ANNOTATION,
    pbd: V_ANNOTATION,
):
    """Rebuild bridge ownership and particle flags from the restored link lifecycle switches (set_state path).

    Ownership is exactly the restored driven switch: a fragment whose link is externally driven is PBD-owned
    (shadow), otherwise native. Pending is transient and cleared; the host mirrors are refreshed by the caller.
    """
    for i_k, i_b_ in qd.ndrange(bridge.n_fragments_q, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = bridge.fragment_link_q[i_k]
        is_native = not rigid.dyn_state.links.is_externally_driven[i_l, i_b]
        bridge.fragment_owner_q[i_k, i_b] = int(FragmentOwner.NATIVE) if is_native else int(FragmentOwner.PBD)
        bridge.handoff_pending_q[i_k, i_b] = False
        for i_m_ in range(bridge.fragment_member_start_q[i_k], bridge.fragment_member_start_q[i_k + 1]):
            pbd.particles_ng[bridge.fragment_member_particles_q[i_m_], i_b].is_native_owned = is_native


@qd.kernel
def kernel_revoke_fragment_ownership(
    fragments_idx: qd.types.ndarray(ndim=1),
    envs_idx: qd.types.ndarray(ndim=1),
    bridge: V_ANNOTATION,
    rigid: V_ANNOTATION,
    pbd: V_ANNOTATION,
):
    """Return native-owned fragments to PBD ownership (public particle-setter path).

    Re-flips the shadow lifecycle switches (driven on, native contact off), clears the members' native-owned
    flags, drops any pending handoff and invalidates the commit capture. The caller first re-derives the member
    particles from the current native link state, so the PBD solver resumes from the exported pose.
    """
    for i_k_, i_b_ in qd.ndrange(fragments_idx.shape[0], envs_idx.shape[0]):
        i_k = fragments_idx[i_k_]
        i_b = envs_idx[i_b_]
        i_l = bridge.fragment_link_q[i_k]
        rigid.dyn_state.links.is_externally_driven[i_l, i_b] = True
        rigid.dyn_state.links.is_contact_enabled[i_l, i_b] = False
        bridge.fragment_owner_q[i_k, i_b] = int(FragmentOwner.PBD)
        bridge.handoff_pending_q[i_k, i_b] = False
        bridge.commit_capture_valid_q[i_k, i_b] = 0
        for i_m_ in range(bridge.fragment_member_start_q[i_k], bridge.fragment_member_start_q[i_k + 1]):
            pbd.particles_ng[bridge.fragment_member_particles_q[i_m_], i_b].is_native_owned = False


@qd.kernel
def kernel_switch_parent_to_fragments(bridge: V_ANNOTATION, rigid: V_ANNOTATION):
    """One-shot import of every pre-cut fragment link from the parent's current state (stage B).

    Reads the parent FREE link's pose and velocity, transfers each fragment to
    pos_k = p + R dk, quat_k = quat, v_k = v + omega x (R dk), omega_k = R^T omega (mass, COM, P, L and
    KE stay continuous because the d_k sum to zero against the parent COM and the fragment inertias are
    the real particle-member inertias), then flips ownership in the same instant: fragments natively
    integrated and contacted, the parent frozen (externally driven) with native contacts off. Runs at the
    substep head; clears the pending flag itself so the hot path stays free of host readback.
    """
    for i_b in range(rigid._B):
        if bridge.parent_switch_pending_q[i_b] == 0:
            continue
        bridge.parent_switch_pending_q[i_b] = 0
        pq = bridge._parent_q_start
        pd = bridge._parent_dof_start
        c_p = qd.Vector.zero(qd.f64, 3)
        quat_p = qd.Vector.zero(qd.f64, 4)
        v_p = qd.Vector.zero(qd.f64, 3)
        w_body = qd.Vector.zero(qd.f64, 3)
        for j in qd.static(range(3)):
            c_p[j] = rigid.rigid_info.qpos[pq + j, i_b]
            v_p[j] = rigid.dyn_state.dofs.vel[pd + j, i_b]
        for j in qd.static(range(4)):
            quat_p[j] = rigid.rigid_info.qpos[pq + 3 + j, i_b]
        for j in qd.static(range(3)):
            w_body[j] = rigid.dyn_state.dofs.vel[pd + 3 + j, i_b]
        # rotation matrix of the parent quaternion, built inline (no helper call in kernel scope)
        qw, qx, qy, qz = quat_p[0], quat_p[1], quat_p[2], quat_p[3]
        R = qd.Matrix(
            [
                [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
                [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
                [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
            ]
        )
        w = R @ w_body
        for i_k in range(bridge.n_fragments_q):
            i_l = bridge.fragment_link_q[i_k]
            q_start = bridge.fragment_q_start_q[i_k]
            dof_start = bridge.fragment_dof_start_q[i_k]
            d_w = R @ bridge.frag_dk_q[i_k]
            pos_k = c_p + d_w
            v_k = v_p + w.cross(d_w)
            # the fragment link frames are spawn-aligned with the parent frame, so the inherited
            # body-frame angular velocity is the parent's own
            omega_body = w_body
            for j in qd.static(range(3)):
                rigid.rigid_info.qpos[q_start + j, i_b] = gs.qd_float(pos_k[j])
                rigid.dyn_state.dofs.vel[dof_start + j, i_b] = gs.qd_float(v_k[j])
            for j in qd.static(range(4)):
                rigid.rigid_info.qpos[q_start + 3 + j, i_b] = gs.qd_float(quat_p[j])
            for j in qd.static(range(3)):
                rigid.dyn_state.dofs.vel[dof_start + 3 + j, i_b] = gs.qd_float(omega_body[j])
            rigid.dyn_state.links.is_externally_driven[i_l, i_b] = False
            rigid.dyn_state.links.is_contact_enabled[i_l, i_b] = True
            for j in qd.static(range(3)):
                rigid.dyn_state.links.cfrc_coupling_vel[i_l, i_b][j] = gs.qd_float(0.0)
                rigid.dyn_state.links.cfrc_coupling_ang[i_l, i_b][j] = gs.qd_float(0.0)
                rigid.dyn_state.links.contact_force[i_l, i_b][j] = gs.qd_float(0.0)
        for i_pl_ in range(bridge.n_parent_links_q):
            i_pl = bridge.parent_links_q[i_pl_]
            rigid.dyn_state.links.is_externally_driven[i_pl, i_b] = True
            rigid.dyn_state.links.is_contact_enabled[i_pl, i_b] = False


@qd.kernel
def kernel_snapshot_parent_velocity(bridge: V_ANNOTATION, links_state: array_class.LinksState):
    """Copy the full parent's link velocities at the substep tail (stage-C trigger input).

    The tail velocities are exactly what the next substep's constraint solve consumes, so the trigger's closing
    speeds are evaluated on the pre-impact state instead of the post-resolution one the same tail would read.
    """
    for i_pl_, i_b in qd.ndrange(bridge.n_parent_links_q, bridge.impact_vel_snap_q.shape[1]):
        i_pl = bridge.parent_links_q[i_pl_]
        bridge.impact_vel_snap_q[i_pl_, i_b] = links_state.cd_vel[i_pl, i_b].cast(qd.f64)
        bridge.impact_ang_snap_q[i_pl_, i_b] = links_state.cd_ang[i_pl, i_b].cast(qd.f64)
        bridge.impact_com_snap_q[i_pl_, i_b] = links_state.root_COM[i_pl, i_b].cast(qd.f64)


@qd.kernel
def kernel_scan_impact_trigger(
    threshold: qd.f64,
    min_closing: qd.f64,
    bridge: V_ANNOTATION,
    rigid: V_ANNOTATION,
    collider_state: array_class.ColliderState,
):
    """Accumulate the closing-speed-gated compression impulse of the full parent's impact events (stage C).

    A substep contributes the normal compression impulse of every contact that touches a parent link and whose
    contact point still approaches along the push direction faster than `min_closing`, evaluated on the previous
    tail's velocity snapshot: that snapshot is the state the current substep's constraint solve consumed, while
    the tail's own `cd_vel` is already the post-resolution velocity the impact absorbed. Resting support and
    separating stabilization never pass the gate, and re-sampled contact manifolds cannot double-count because
    the gate reruns per substep on the live contact set instead of accumulating per persisted pair. A substep
    with no gate-passing contact closes the open event; a closed event whose total exceeds the calibrated
    threshold arms the one-shot switch for the next substep head and disarms the trigger. Runs at the substep
    tail on the contact forces the constraint solver just produced.
    """
    for i_b in range(rigid._B):
        if not bridge.impact_armed_q[i_b] or bridge.parent_switch_pending_q[i_b] != 0:
            continue
        h = rigid.rigid_info.substep_dt[None]
        contribution = 0.0
        for i_c in range(collider_state.n_contacts[i_b]):
            i_lp = -1
            i_row = -1
            side = 0
            for i_pl_ in range(bridge.n_parent_links_q):
                i_pl = bridge.parent_links_q[i_pl_]
                if collider_state.contact_data.link_a[i_c, i_b] == i_pl:
                    i_lp = i_pl
                    i_row = i_pl_
                    side = -1
                    break
                if collider_state.contact_data.link_b[i_c, i_b] == i_pl:
                    i_lp = i_pl
                    i_row = i_pl_
                    side = 1
                    break
            if i_lp < 0:
                continue
            force_p = collider_state.contact_data.force[i_c, i_b] * side
            f_mag = force_p.norm()
            if f_mag < 1e-9:
                continue
            push = force_p / f_mag
            v_point = bridge.impact_vel_snap_q[i_row, i_b] + bridge.impact_ang_snap_q[i_row, i_b].cross(
                collider_state.contact_data.pos[i_c, i_b].cast(qd.f64) - bridge.impact_com_snap_q[i_row, i_b]
            )
            if -v_point.dot(push) > min_closing:
                contribution += f_mag * qd.abs(push.dot(collider_state.contact_data.normal[i_c, i_b])) * h
        if contribution > 0.0:
            bridge.impact_accum_q[i_b] += contribution
            bridge.impact_event_open_q[i_b] = True
        elif bridge.impact_event_open_q[i_b]:
            if bridge.impact_accum_q[i_b] > threshold:
                bridge.parent_switch_pending_q[i_b] = 1
                bridge.impact_event_count_q[i_b] += 1
                bridge.impact_last_sum_q[i_b] = bridge.impact_accum_q[i_b]
                bridge.impact_armed_q[i_b] = False
            bridge.impact_accum_q[i_b] = 0.0
            bridge.impact_event_open_q[i_b] = False


class PBDRigidFragmentBridge:
    """Lifecycle owner of the native rigid links that shadow the fragments of a PBD fracturable solid.

    A registered fragment is followed (the bridge rewrites the link's qpos from the PBD welded fit at every substep
    head, and the native integrator and contact response skip the link) until its live-seam component shrinks to
    itself and the next substep head commits it: a one-shot momentum / inertia import that flips the fragment to
    exclusive native advancement with native contacts. The bridge only converts state; it never solves rigid-body
    dynamics or contacts itself. Fields are exact-size with no capacity padding: static layout K fragments /
    N member particles, per-environment dynamic state [K, B].

    Registration modes: `register_solid_fragments` wires a real `PBD3DSolidEntity` (the shadow FREE links, with
    explicit mass / center of mass / full inertia from the particle members and per-fragment convex collision
    proxies, are created at `Simulator.build` head); `register_fragments` wires pre-existing links with explicit
    shadow targets for rigid-only lifecycle tests.
    """

    def __init__(self, sim: "Simulator"):
        self.sim = sim
        self._solid_entity: "PBD3DSolidEntity | None" = None
        self._fragment_link: np.ndarray | None = None  # [K] rigid solver link idx of each fragment
        self._fragment_cluster: np.ndarray | None = None  # [K] global PBD cluster idx of each fragment
        self._particle_fragment: np.ndarray | None = None  # [N] fragment idx owning each particle
        self._fragment_q_start: np.ndarray | None = None  # [K] first qpos row of the fragment's link
        self._fragment_n_qs: np.ndarray | None = None  # [K] qpos rows of the fragment's link (7 for a FREE link)
        self._fragment_dof_start: np.ndarray | None = None  # [K] first dof of the fragment's link
        self._fragment_n_dofs: np.ndarray | None = None  # [K] dofs of the fragment's link (6 for a FREE link)
        self._fragment_link_objs: list["RigidLink"] = []
        self._fragment_mass: np.ndarray | None = None  # [K] f64 total particle mass
        self._fragment_rest_cm: np.ndarray | None = None  # [K, 3] f64 mass-weighted rest center of mass
        self._fragment_rest_cm_offset: np.ndarray | None = None  # [K, 3] f64 mass cm minus unweighted cluster cm
        self._fragment_rest_inertia: np.ndarray | None = None  # [K, 3, 3] f64 rest inertia about the mass cm
        self._fragment_member_start: np.ndarray | None = None  # [K + 1] CSR start into member particles
        self._fragment_member_particles: np.ndarray | None = None  # [sum_n_k] global particle idx per fragment
        self._fragment_is_handoff_eligible: np.ndarray | None = None  # [K] static eligibility (free/unit/no plastic)
        self._proxy_meshes: list[dict] = []  # per-fragment convex proxy record (path, vertices, faces)
        self._fragment_owner_cpu: np.ndarray | None = None  # [K, B] host mirror of fragment_owner_q
        self._fragment_driven_cpu: np.ndarray | None = None  # [K, B] host mirror gating the substep head
        self._shadow_pos: torch.Tensor | None = None  # [K, B, 3] manual-mode followed world position
        self._shadow_quat: torch.Tensor | None = None  # [K, B, 4] manual-mode followed quaternion (w, x, y, z)
        self._shadow_lin_vel: torch.Tensor | None = None  # [K, B, 3] manual-mode followed world linear velocity
        self._shadow_ang_vel: torch.Tensor | None = None  # [K, B, 3] manual-mode body-frame angular velocity
        self._fragment_q_idx: torch.Tensor | None = None  # [K, max_n_qs] qpos rows, masked for narrow links
        self._fragment_dof_idx: torch.Tensor | None = None  # [K, max_n_dofs] dof rows, masked for narrow links
        self._fragment_q_mask: torch.Tensor | None = None  # [K, max_n_qs] valid-row mask
        self._fragment_dof_mask: torch.Tensor | None = None  # [K, max_n_dofs] valid-row mask
        self.fragment_link_q = None  # qd int [K]
        self.fragment_cluster_q = None  # qd int [K]
        self.fragment_q_start_q = None  # qd int [K]
        self.fragment_dof_start_q = None  # qd int [K]
        self.fragment_rest_cm_q = None  # qd f64 vec3 [K]
        self.fragment_rest_cm_offset_q = None  # qd f64 vec3 [K]
        self.fragment_rest_inertia_q = None  # qd f64 mat3 [K]
        self.fragment_member_start_q = None  # qd int [K + 1]
        self.fragment_member_particles_q = None  # qd int [sum_n_k]
        self.fragment_is_handoff_eligible_q = None  # qd bool [K]
        self.fragment_owner_q = None  # qd int [K, B]
        self.handoff_pending_q = None  # qd bool [K, B]
        self.commit_count_q = None  # qd int [K, B]
        self.commit_rejected_q = None  # qd int [B]
        self.commit_errno_q = None  # qd int [B], scratch for the commit's twist solve (rejections stay non-fatal)
        self.commit_capture_pos_q = None  # qd float [N, B, 3] particle inputs at commit time (diagnostic)
        self.commit_capture_vel_q = None  # qd float [N, B, 3]
        self.commit_capture_qpos_q = None  # qd float [K, B, 7] imported link qpos at commit time
        self.commit_capture_qvel_q = None  # qd float [K, B, 6]
        self.commit_capture_valid_q = None  # qd int [K, B]
        self.n_fragments_q = 0  # kernel-visible fragment count
        # Stage-B full-native-parent switch: the parent entity's whole link tree owns the dynamics and contacts
        # until the one-shot switch imports the pre-cut fragment links from the root's state (same substep head,
        # mutually exclusive). All fields are None until `register_full_parent` runs.
        self._parent_entity: "RigidEntity | None" = None
        self._parent_link: int | None = None  # idx of the parent compound's FREE root link
        self._parent_links: np.ndarray | None = None  # [P] link idxs of the compound (root + welded children)
        self._parent_q_start: int | None = None
        self._parent_dof_start: int | None = None
        # Stage-C impact trigger: the calibrated one-shot release signal of the full parent. The threshold is
        # an effect parameter bound to this object's mass / scale / pre-cut and the validated scenario, never
        # a general material strength. All fields are None until `arm_impact_trigger` runs.
        self._impact_threshold: float | None = None  # N·s compression-impulse total an event must exceed
        self._impact_min_closing: float | None = None  # m/s closing-speed gate separating impacts from support
        self.impact_armed_q = None  # qd bool [B]
        self.impact_accum_q = None  # qd f64 [B] compression impulse of the currently open event
        self.impact_event_open_q = None  # qd bool [B]
        self.impact_event_count_q = None  # qd int [B] fired events (diagnostic)
        self.impact_last_sum_q = None  # qd f64 [B] compression impulse of the last fired event (diagnostic)
        self.impact_vel_snap_q = None  # qd f64 vec3 [P, B] parent link velocities at the last tail
        self.impact_ang_snap_q = None  # qd f64 vec3 [P, B] parent link angular velocities at the last tail
        self.impact_com_snap_q = None  # qd f64 vec3 [P, B] parent tree COMs at the last tail
        self._impact_armed_cpu: np.ndarray | None = None  # [B] host mirror gating the tail scan launch
        self._frag_dk: np.ndarray | None = None  # [K, 3] f64 fragment rest-CM offsets vs the parent COM
        self.parent_links_q = None  # qd int [P]
        self.n_parent_links_q = 0  # kernel-visible parent link count
        self.frag_dk_q = None  # qd f64 vec3 [K]
        self.parent_switch_pending_q = None  # qd int [B]

    @property
    def n_fragments(self) -> int:
        """Number of registered fragments (0 before registration)."""
        return 0 if self._fragment_link is None else len(self._fragment_link)

    def register_solid_fragments(self, entity, proxy_dir) -> None:
        """Collect the static fragment layout of a fracturable `PBD3DSolidEntity`, before `Simulator.build` runs.

        Reads the partition sampled at add_entity time: per-fragment member particles, total mass, mass-weighted
        rest center of mass, full rest inertia about that center (all from the particle mass members, never from a
        convex volume), and the mass-versus-unweighted centroid offset the welded follow corrects for. Writes one
        low-poly convex collision proxy per fragment (hull of the member rest positions, translated into the link
        frame) into `proxy_dir` for the shadow links' geoms. The shadow FREE links themselves are created by
        `register_shadow_links` at the build head.
        """
        if self._fragment_link is not None:
            gs.raise_exception("PBDRigidFragmentBridge fragments are already registered.")
        if not isinstance(entity, PBD3DSolidEntity):
            gs.raise_exception("register_solid_fragments requires a PBD3DSolidEntity.")
        if entity.n_fragments <= 1:
            gs.raise_exception("register_solid_fragments requires a fracturable entity (fracture_threshold > 0).")
        if entity.n_clusters != entity.n_fragments:
            gs.raise_exception("register_solid_fragments requires the one-cluster-per-fragment partition.")
        if not (entity._particle_member_counts == 1).all():
            gs.raise_exception("register_solid_fragments requires unique cluster membership per particle.")

        n_fragments = entity.n_fragments
        member_particles = [np.nonzero(entity._particle_fragments == i_k)[0] for i_k in range(n_fragments)]
        particle_mass = entity._particle_mass
        rest = entity._particles
        unweighted_cm = entity._cluster_rest_cm

        fragment_mass = np.empty(n_fragments, dtype=np.float64)
        rest_cm = np.empty((n_fragments, 3), dtype=np.float64)
        rest_inertia = np.empty((n_fragments, 3, 3), dtype=np.float64)
        member_start = np.zeros(n_fragments + 1, dtype=gs.np_int)
        proxies = []
        proxy_path = Path(proxy_dir)
        proxy_path.mkdir(parents=True, exist_ok=True)
        for i_k in range(n_fragments):
            members = member_particles[i_k]
            member_start[i_k + 1] = member_start[i_k] + len(members)
            mass = np.full(len(members), particle_mass, dtype=np.float64)
            fragment_mass[i_k] = mass.sum()
            rest_cm[i_k] = (mass[:, None] * rest[members]).sum(0) / fragment_mass[i_k]
            offsets = rest[members] - rest_cm[i_k]
            covariance = (mass[:, None] * offsets).T @ offsets
            rest_inertia[i_k] = np.eye(3) * np.trace(covariance) - covariance

            points = rest[members] - rest_cm[i_k]
            hull = ConvexHull(points)
            faces = hull.simplices.copy()
            cross = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
            inward = np.einsum("ij,ij->i", cross, points[faces].mean(1) - points.mean(0)) < 0
            faces[inward] = faces[inward][:, [0, 2, 1]]
            proxy_file = proxy_path / f"fragment_proxy_{i_k:02d}.obj"
            lines = ["# pbd fragment convex collision proxy"]
            for vertex in points:
                lines.append(f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}")
            for face in faces:
                lines.append(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}")
            proxy_file.write_text("\n".join(lines) + "\n", encoding="ascii")
            proxies.append(
                {
                    "fragment": i_k,
                    "path": str(proxy_file),
                    "hull_vertices": int(len(points)),
                    "hull_faces": int(len(faces)),
                    "member_particles": int(len(members)),
                }
            )

        self._solid_entity = entity
        self._particle_fragment = entity._particle_fragments.astype(gs.np_int, copy=True)
        self._fragment_cluster = entity._cluster_start + np.arange(n_fragments, dtype=gs.np_int)
        self._fragment_mass = fragment_mass
        self._fragment_rest_cm = rest_cm
        self._fragment_rest_cm_offset = rest_cm - unweighted_cm
        self._fragment_rest_inertia = rest_inertia
        self._fragment_member_start = member_start
        self._fragment_member_particles = entity._particle_start + np.concatenate(member_particles).astype(gs.np_int)
        self._fragment_is_handoff_eligible = np.array(
            [
                entity.material.stiffness == 1.0
                and entity.material.plastic_creep == 0.0
                and entity.material.plastic_flow_rate is None
            ]
            * n_fragments,
            dtype=bool,
        )
        self._proxy_meshes = proxies

    def register_fragments(self, fragment_links, fragment_clusters, particle_fragments) -> None:
        """Register pre-existing FREE links with explicit shadow targets, before `Simulator.build` runs.

        Rigid-only lifecycle mode: `fragment_links` are rigid solver link objects (one FREE link per fragment,
        homogeneous across envs), `fragment_clusters` the PBD cluster idx of each fragment, `particle_fragments`
        the per-particle fragment idx ([N] ints in [0, K), or empty when no particle ownership is tracked). Every
        fragment starts PBD-owned; the shadow lifecycle flags themselves are applied after the solver builds.
        """
        if self._fragment_link is not None:
            gs.raise_exception("PBDRigidFragmentBridge fragments are already registered.")
        n_fragments = len(fragment_links)
        if n_fragments == 0:
            gs.raise_exception("PBDRigidFragmentBridge requires at least one fragment link.")
        particle_fragments = np.asarray(particle_fragments, dtype=gs.np_int)
        fragment_clusters = np.asarray(fragment_clusters, dtype=gs.np_int)
        if len(fragment_clusters) != n_fragments:
            gs.raise_exception("fragment_clusters must hold one cluster idx per fragment link.")
        if particle_fragments.size and (particle_fragments.min() < 0 or particle_fragments.max() >= n_fragments):
            gs.raise_exception("particle_fragments holds a fragment idx outside [0, n_fragments).")

        self._wire_fragment_links(list(fragment_links), fragment_clusters, particle_fragments)

    def register_full_parent(self, parent_entity, frag_links, frag_offsets) -> None:
        """Register a full native parent body with its pre-cut fragment links (stage-B switch mode).

        `parent_entity` is the rigid entity whose FREE root link owns the dynamics and whose whole link tree
        carries the parent's collision geoms (one link per welded piece); the switch freezes and contact-gates
        every one of its links, so the entity must contain no link meant to keep moving past the switch.
        `frag_links` are the pre-cut fragment FREE links (dormant until the switch); `frag_offsets` are the
        fragments' rest-CM offsets relative to the parent COM in the spawn-aligned frame ([K, 3], f64).
        The fragment links are wired through the ordinary manual-mode path; the parent stays an ordinary
        natively-integrated body until `request_parent_switch` arms the one-shot kernel.
        """
        if self._parent_link is not None:
            gs.raise_exception("PBDRigidFragmentBridge full parent is already registered.")
        if self._fragment_link is not None:
            gs.raise_exception("PBDRigidFragmentBridge fragments are already registered.")
        root = parent_entity.links[0]
        d = np.asarray(frag_offsets, dtype=np.float64)
        if d.ndim != 2 or d.shape[1] != 3 or d.shape[0] != len(frag_links):
            gs.raise_exception("frag_offsets must be [n_fragments, 3].")
        if abs(np.asarray(frag_offsets).sum(0).max()) > 1e-8:
            pass  # offsets need not sum to zero for correctness; the switch transfer is per-fragment
        self._parent_entity = parent_entity
        self._parent_link = int(root.idx)
        self._parent_q_start = int(root.q_start)
        self._parent_dof_start = int(root.dof_start)
        self._parent_links = np.array([link.idx for link in parent_entity.links], dtype=gs.np_int)
        self._frag_dk = d
        self._wire_fragment_links(list(frag_links), np.arange(len(frag_links)), np.zeros(0, dtype=gs.np_int))

    def request_parent_switch(self, envs_idx=None) -> None:
        """Arm the one-shot parent→fragments switch for the given environments (applied at the next substep head)."""
        if self._parent_link is None:
            gs.raise_exception("PBDRigidFragmentBridge has no registered full parent.")
        envs_list = tensor_to_array(self.sim.scene._sanitize_envs_idx(envs_idx)).astype(np.int64).tolist()
        for i_b in envs_list:
            self.parent_switch_pending_q[i_b] = 1
        # The switch kernel flips the links' driven gate inside the next substep head, and the manual-mode follow
        # block gates on this host mirror within that same head: the mirror must drop the fragments here, or the
        # stale shadow target would overwrite the state the kernel transfers in that very substep.
        self._fragment_driven_cpu[:, envs_list] = False

    def arm_impact_trigger(self, threshold, min_closing_speed=0.1, envs_idx=None) -> None:
        """Arm the calibrated one-shot impact trigger of the registered full parent (stage C).

        An impact event is the maximal run of consecutive substeps whose contact set touches a parent link with a
        contact point still closing faster than `min_closing_speed`; the event's normal compression impulse total
        (N·s, compression component of F*h over the gate-passing contacts) arms `request_parent_switch` for the
        next substep head when it exceeds `threshold`. The threshold is an effect parameter calibrated per
        object mass / scale / pre-cut and per validated scenario - it is not a general material strength, and
        resting support or de-penetration stabilization cannot reach it because the closing gate excludes them
        per substep. The scan runs on the solved contact set at every substep tail while armed; a scene without
        an armed trigger launches nothing.
        """
        if self._parent_link is None:
            gs.raise_exception("PBDRigidFragmentBridge impact trigger requires a registered full parent.")
        if threshold <= 0.0:
            gs.raise_exception("PBDRigidFragmentBridge impact threshold must be positive (N·s).")
        if self.impact_armed_q is None:
            gs.raise_exception("PBDRigidFragmentBridge impact trigger state is allocated at scene build; build first.")
        envs_list = tensor_to_array(self.sim.scene._sanitize_envs_idx(envs_idx)).astype(np.int64).tolist()
        self._impact_threshold = float(threshold)
        self._impact_min_closing = float(min_closing_speed)
        for i_b in envs_list:
            self.impact_armed_q[i_b] = True
        self._impact_armed_cpu[envs_list] = True

    def read_impact_trigger(self) -> dict:
        """Host copy of the impact trigger state (diagnostic readback; syncs)."""
        if self.impact_armed_q is None:
            gs.raise_exception("PBDRigidFragmentBridge impact trigger state is allocated at scene build; build first.")
        return {
            "armed": qd_to_numpy(self.impact_armed_q),
            "accum": qd_to_numpy(self.impact_accum_q),
            "event_open": qd_to_numpy(self.impact_event_open_q),
            "event_count": qd_to_numpy(self.impact_event_count_q),
            "last_sum": qd_to_numpy(self.impact_last_sum_q),
        }

    def register_shadow_links(self) -> None:
        """`Simulator.build` head hook, before any solver allocates fields: create the shadow FREE links.

        For a registered solid entity this builds one MJCF body per fragment from the static partition: freejoint,
        explicit inertial (mass and full inertia about the fragment's mass-weighted rest center of mass, computed
        from the particle members) and the fragment's convex collision proxy geom carrying the material's kinetic
        friction (the stage-0 contract value; the static friction band is not representable natively). The created
        entity registers through the ordinary pre-build path, so pair allocation, sensors and state arrays see it.
        The shadow entity carries no visual geoms (`visualization=False`): the PBD render mesh is the single visual
        representation of the fragments, so the convex collision proxies must never draw. Manual-mode registrations
        only validate their links.
        """
        if self._fragment_link is not None:
            if self.sim.rigid_solver.n_links <= int(self._fragment_link.max()):
                gs.raise_exception("Registered fragment link idx exceeds the built rigid solver link count.")
            return
        if self._solid_entity is None:
            return

        entity = self._solid_entity
        model = ET.Element("mujoco", model="pbd_fragment_shadow_links")
        ET.SubElement(model, "compiler", angle="radian", inertiafromgeom="false")
        assets = ET.SubElement(model, "asset")
        world = ET.SubElement(model, "worldbody")
        friction = entity.material.kinetic_friction
        for proxy in self._proxy_meshes:
            i_k = proxy["fragment"]
            ET.SubElement(assets, "mesh", name=f"fragment_{i_k}", file=proxy["path"])
            body = ET.SubElement(
                world,
                "body",
                name=f"fragment_{i_k}",
                pos=" ".join(format(v, ".17g") for v in self._fragment_rest_cm[i_k]),
            )
            ET.SubElement(body, "freejoint", name=f"free_{i_k}")
            inertia = self._fragment_rest_inertia[i_k]
            ET.SubElement(
                body,
                "inertial",
                pos="0 0 0",
                mass=format(self._fragment_mass[i_k], ".17g"),
                fullinertia=" ".join(
                    format(v, ".17g")
                    for v in (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
                ),
            )
            ET.SubElement(
                body,
                "geom",
                type="mesh",
                mesh=f"fragment_{i_k}",
                friction=f"{friction:.17g} 0 0",
                condim="3",
            )
        xml = ET.tostring(model, encoding="unicode")
        shadow_entity = self.sim.scene.add_entity(
            morph=gs.morphs.MJCF(
                file=xml, convexify=False, decimate=False, recompute_inertia=False, align=False, visualization=False
            ),
            material=gs.materials.Rigid(),
        )
        n_fragments = len(self._proxy_meshes)
        links = [shadow_entity.get_link(name=f"fragment_{i_k}") for i_k in range(n_fragments)]
        self._wire_fragment_links(links, self._fragment_cluster, self._particle_fragment)

    def _wire_fragment_links(self, fragment_links, fragment_clusters, particle_fragments) -> None:
        links_by_idx = {link.idx: link for link in self.sim.rigid_solver.links}
        n_fragments = len(fragment_links)
        fragment_q_start = np.empty(n_fragments, dtype=gs.np_int)
        fragment_n_qs = np.empty(n_fragments, dtype=gs.np_int)
        fragment_dof_start = np.empty(n_fragments, dtype=gs.np_int)
        fragment_n_dofs = np.empty(n_fragments, dtype=gs.np_int)
        for i_k, link in enumerate(fragment_links):
            if link.idx not in links_by_idx:
                gs.raise_exception(f"Fragment link idx {link.idx} does not belong to the rigid solver.")
            fragment_q_start[i_k] = link.q_start
            fragment_n_qs[i_k] = link.n_qs
            fragment_dof_start[i_k] = link.dof_start
            fragment_n_dofs[i_k] = link.n_dofs
        self._fragment_link = np.array([link.idx for link in fragment_links], dtype=gs.np_int)
        self._fragment_cluster = np.asarray(fragment_clusters, dtype=gs.np_int)
        self._particle_fragment = np.asarray(particle_fragments, dtype=gs.np_int)
        self._fragment_q_start = fragment_q_start
        self._fragment_n_qs = fragment_n_qs
        self._fragment_dof_start = fragment_dof_start
        self._fragment_n_dofs = fragment_n_dofs
        self._fragment_link_objs = list(fragment_links)

        # Row-index grids for the vectorized manual-mode substep-head state write. Links narrower than the widest
        # layout address row 0 under a zero mask, so no foreign qpos row is ever read or written.
        q_offsets = torch.arange(int(fragment_n_qs.max()), device=gs.device)
        dof_offsets = torch.arange(int(fragment_n_dofs.max()), device=gs.device)
        self._fragment_q_mask = q_offsets[None, :] < torch.as_tensor(fragment_n_qs, device=gs.device)[:, None]
        self._fragment_dof_mask = dof_offsets[None, :] < torch.as_tensor(fragment_n_dofs, device=gs.device)[:, None]
        self._fragment_q_idx = (torch.as_tensor(fragment_q_start, device=gs.device)[:, None] + q_offsets) * (
            self._fragment_q_mask
        )
        self._fragment_dof_idx = (torch.as_tensor(fragment_dof_start, device=gs.device)[:, None] + dof_offsets) * (
            self._fragment_dof_mask
        )

    def enter_shadow_lifecycle(self) -> None:
        """`Simulator.build` tail hook, after the rigid solver built: put every registered fragment into shadow mode.

        Allocates the per-environment dynamic state (owner, pending, commit counters; the batch dim only exists
        once the scene is built), seeds the manual-mode followed pose from the solver's initial state, uploads the
        static fragment layout for the kernels, and turns the externally-driven gate on and native contacts off
        for the fragment links in all envs.
        """
        if self._fragment_link is None:
            return
        solver = self.sim.rigid_solver
        if not solver.is_active:
            gs.raise_exception("PBDRigidFragmentBridge registered fragments without an active rigid solver.")

        n_fragments = self.n_fragments
        n_envs = self.sim._B
        self.fragment_link_q = qd.field(dtype=gs.qd_int, shape=(n_fragments,))
        self.fragment_link_q.from_numpy(self._fragment_link)
        self.fragment_cluster_q = qd.field(dtype=gs.qd_int, shape=(n_fragments,))
        self.fragment_cluster_q.from_numpy(self._fragment_cluster)
        self.fragment_q_start_q = qd.field(dtype=gs.qd_int, shape=(n_fragments,))
        self.fragment_q_start_q.from_numpy(self._fragment_q_start)
        self.fragment_dof_start_q = qd.field(dtype=gs.qd_int, shape=(n_fragments,))
        self.fragment_dof_start_q.from_numpy(self._fragment_dof_start)
        if self._solid_entity is not None:
            rest_cm_field = qd.Vector.field(3, qd.f64, shape=(n_fragments,))
            rest_cm_field.from_numpy(self._fragment_rest_cm)
            self.fragment_rest_cm_q = rest_cm_field
            offset_field = qd.Vector.field(3, qd.f64, shape=(n_fragments,))
            offset_field.from_numpy(self._fragment_rest_cm_offset)
            self.fragment_rest_cm_offset_q = offset_field
            inertia_field = qd.Matrix.field(3, 3, qd.f64, shape=(n_fragments,))
            inertia_field.from_numpy(self._fragment_rest_inertia)
            self.fragment_rest_inertia_q = inertia_field
            self.fragment_member_start_q = qd.field(dtype=gs.qd_int, shape=(n_fragments + 1,))
            self.fragment_member_start_q.from_numpy(self._fragment_member_start)
            self.fragment_member_particles_q = qd.field(dtype=gs.qd_int, shape=(len(self._fragment_member_particles),))
            self.fragment_member_particles_q.from_numpy(self._fragment_member_particles)
            eligible_field = qd.field(dtype=gs.qd_bool, shape=(n_fragments,))
            eligible_field.from_numpy(self._fragment_is_handoff_eligible.astype(np.int32))
            self.fragment_is_handoff_eligible_q = eligible_field
            n_particles = self.sim.pbd_solver._n_particles
            self.commit_capture_pos_q = qd.field(dtype=gs.qd_float, shape=(n_particles, n_envs, 3))
            self.commit_capture_pos_q.fill(0.0)
            self.commit_capture_vel_q = qd.field(dtype=gs.qd_float, shape=(n_particles, n_envs, 3))
            self.commit_capture_vel_q.fill(0.0)
            self.commit_capture_qpos_q = qd.field(dtype=gs.qd_float, shape=(n_fragments, n_envs, 7))
            self.commit_capture_qpos_q.fill(0.0)
            self.commit_capture_qvel_q = qd.field(dtype=gs.qd_float, shape=(n_fragments, n_envs, 6))
            self.commit_capture_qvel_q.fill(0.0)
            self.commit_capture_valid_q = qd.field(dtype=gs.qd_int, shape=(n_fragments, n_envs))
            self.commit_capture_valid_q.fill(0)
        else:
            zero_cm = qd.Vector.field(3, qd.f64, shape=(n_fragments,))
            zero_cm.from_numpy(np.zeros((n_fragments, 3)))
            self.fragment_rest_cm_q = zero_cm
            self.fragment_rest_cm_offset_q = zero_cm
            zero_inertia = qd.Matrix.field(3, 3, qd.f64, shape=(n_fragments,))
            zero_inertia.from_numpy(np.zeros((n_fragments, 3, 3)))
            self.fragment_rest_inertia_q = zero_inertia
            member_start = np.arange(n_fragments + 1, dtype=gs.np_int)
            self.fragment_member_start_q = qd.field(dtype=gs.qd_int, shape=(n_fragments + 1,))
            self.fragment_member_start_q.from_numpy(member_start)
            self.fragment_member_particles_q = qd.field(dtype=gs.qd_int, shape=(n_fragments,))
            self.fragment_member_particles_q.from_numpy(np.zeros(n_fragments, dtype=gs.np_int))
            self.fragment_is_handoff_eligible_q = qd.field(dtype=gs.qd_bool, shape=(n_fragments,))
            self.fragment_is_handoff_eligible_q.from_numpy(np.zeros(n_fragments, dtype=np.int32))
        self.fragment_owner_q = qd.field(dtype=gs.qd_int, shape=(n_fragments, n_envs))
        self.fragment_owner_q.fill(int(FragmentOwner.PBD))
        self.handoff_pending_q = qd.field(dtype=gs.qd_bool, shape=(n_fragments, n_envs))
        self.handoff_pending_q.fill(False)
        if self._parent_link is not None:
            parent_links_field = qd.field(dtype=gs.qd_int, shape=(len(self._parent_links),))
            parent_links_field.from_numpy(self._parent_links)
            self.parent_links_q = parent_links_field
            self.n_parent_links_q = len(self._parent_links)
            dk_field = qd.Vector.field(3, qd.f64, shape=(n_fragments,))
            dk_field.from_numpy(self._frag_dk)
            self.frag_dk_q = dk_field
            self.impact_armed_q = qd.field(dtype=gs.qd_bool, shape=(n_envs,))
            self.impact_armed_q.fill(False)
            self.impact_accum_q = qd.field(dtype=gs.qd_float, shape=(n_envs,))
            self.impact_accum_q.fill(0.0)
            self.impact_event_open_q = qd.field(dtype=gs.qd_bool, shape=(n_envs,))
            self.impact_event_open_q.fill(False)
            self.impact_event_count_q = qd.field(dtype=gs.qd_int, shape=(n_envs,))
            self.impact_event_count_q.fill(0)
            self.impact_last_sum_q = qd.field(dtype=gs.qd_float, shape=(n_envs,))
            self.impact_last_sum_q.fill(0.0)
            self.impact_vel_snap_q = qd.Vector.field(3, qd.f64, shape=(len(self._parent_links), n_envs))
            self.impact_vel_snap_q.fill(0.0)
            self.impact_ang_snap_q = qd.Vector.field(3, qd.f64, shape=(len(self._parent_links), n_envs))
            self.impact_ang_snap_q.fill(0.0)
            self.impact_com_snap_q = qd.Vector.field(3, qd.f64, shape=(len(self._parent_links), n_envs))
            self.impact_com_snap_q.fill(0.0)
            self._impact_armed_cpu = np.zeros((n_envs,), dtype=bool)
        self.parent_switch_pending_q = qd.field(dtype=gs.qd_int, shape=(n_envs,))
        self.parent_switch_pending_q.fill(0)
        self.commit_count_q = qd.field(dtype=gs.qd_int, shape=(n_fragments, n_envs))
        self.commit_count_q.fill(0)
        self.commit_rejected_q = qd.field(dtype=gs.qd_int, shape=(n_envs,))
        self.commit_rejected_q.fill(0)
        self.commit_errno_q = qd.field(dtype=gs.qd_int, shape=(n_envs,))
        self.commit_errno_q.fill(0)
        self.n_fragments_q = n_fragments
        self._fragment_owner_cpu = np.full((n_fragments, n_envs), int(FragmentOwner.PBD), dtype=np.int64)
        self._fragment_driven_cpu = np.ones((n_fragments, n_envs), dtype=bool)

        self._shadow_pos = torch.zeros((n_fragments, n_envs, 3), dtype=gs.tc_float, device=gs.device)
        self._shadow_quat = torch.zeros((n_fragments, n_envs, 4), dtype=gs.tc_float, device=gs.device)
        self._shadow_quat[..., 0] = 1.0
        self._shadow_lin_vel = torch.zeros((n_fragments, n_envs, 3), dtype=gs.tc_float, device=gs.device)
        self._shadow_ang_vel = torch.zeros((n_fragments, n_envs, 3), dtype=gs.tc_float, device=gs.device)
        envs_rows = torch.arange(n_envs, device=gs.device)
        q_rows = self._fragment_q_idx[None].expand(n_envs, -1, -1)
        dof_rows = self._fragment_dof_idx[None].expand(n_envs, -1, -1)
        qpos = qd_to_torch(solver.rigid_info.qpos, transpose=True, copy=True)
        vel = qd_to_torch(solver.dyn_state.dofs.vel, transpose=True, copy=True)
        shadow_q = qpos[envs_rows[:, None, None], q_rows].permute(1, 0, 2)
        shadow_v = vel[envs_rows[:, None, None], dof_rows].permute(1, 0, 2)
        self._shadow_pos.copy_(shadow_q[..., :3])
        self._shadow_quat.copy_(shadow_q[..., 3:7])
        self._shadow_lin_vel.copy_(shadow_v[..., :3])
        self._shadow_ang_vel.copy_(shadow_v[..., 3:6])

        envs_idx = np.asarray(tensor_to_array(self.sim.scene._sanitize_envs_idx(None)), dtype=gs.np_int)
        for i_k in range(n_fragments):
            kernel_set_link_externally_driven(
                int(self._fragment_link[i_k]), envs_idx, is_driven=1, dyn_state=solver.dyn_state
            )
            kernel_set_link_contact_enabled(
                int(self._fragment_link[i_k]), envs_idx, is_contact_enabled=0, dyn_state=solver.dyn_state
            )

    def set_fragment_externally_driven(self, fragment_idx: int, envs_idx, is_driven: bool) -> None:
        """Flip the integrator gate of one fragment's link for the given environments.

        A driven link is also followed (the substep head keeps rewriting its pose and velocity), so flipping this
        flag off hands advancement back to the native integrator.
        """
        self._require_registry()
        envs_idx = self.sim.scene._sanitize_envs_idx(envs_idx)
        kernel_set_link_externally_driven(
            int(self._fragment_link[fragment_idx]),
            envs_idx,
            is_driven=int(is_driven),
            dyn_state=self.sim.rigid_solver.dyn_state,
        )
        self._fragment_driven_cpu[fragment_idx, envs_idx.tolist()] = bool(is_driven)

    def set_fragment_contact_enabled(self, fragment_idx: int, envs_idx, is_contact_enabled: bool) -> None:
        """Flip the runtime contact gate of one fragment's link for the given environments."""
        self._require_registry()
        envs_idx = self.sim.scene._sanitize_envs_idx(envs_idx)
        kernel_set_link_contact_enabled(
            int(self._fragment_link[fragment_idx]),
            envs_idx,
            is_contact_enabled=int(is_contact_enabled),
            dyn_state=self.sim.rigid_solver.dyn_state,
        )

    def set_shadow_target(self, fragment_idx: int, envs_idx, pos, quat, lin_vel, ang_vel) -> None:
        """Store the manual-mode followed pose and velocity of one fragment for the given environments.

        `pos` / `lin_vel` are world-frame [3], `quat` is world [w, x, y, z], `ang_vel` is body-frame [3] (the FREE
        joint's angular dofs are body-frame). Applied to the solver buffers at every substep head while the
        fragment's link stays externally driven. Ignored in solid mode, where the PBD welded fit is the source.
        """
        self._require_registry()
        envs_idx = self.sim.scene._sanitize_envs_idx(envs_idx)
        self._shadow_pos[fragment_idx, envs_idx] = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device)
        self._shadow_quat[fragment_idx, envs_idx] = torch.as_tensor(quat, dtype=gs.tc_float, device=gs.device)
        self._shadow_lin_vel[fragment_idx, envs_idx] = torch.as_tensor(lin_vel, dtype=gs.tc_float, device=gs.device)
        self._shadow_ang_vel[fragment_idx, envs_idx] = torch.as_tensor(ang_vel, dtype=gs.tc_float, device=gs.device)

    def substep_head(self, f: int) -> None:
        """Substep head, before any solver advances: commit pending handoffs, then follow driven fragment links.

        The commit kernel is unconditional (it no-ops without pending fragments, keeping the hot path free of host
        readback); the follow source is the PBD welded cluster fit in solid mode and the stored shadow target in
        manual mode. Both invalidate the solver's forward flags so the substep's kinematics refresh runs on the
        bridge-written state; the externally-driven integrator gate keeps those buffers authoritative through the
        copyback.
        """
        if self._fragment_link is None:
            return
        solver = self.sim.rigid_solver
        if not solver.is_active:
            return
        pbd = self.sim.pbd_solver
        if self._solid_entity is not None and pbd.is_active:
            kernel_commit_fragment_handoffs(self, pbd, solver)
            kernel_follow_pbd_fragments(self, pbd, solver)
            solver._is_forward_pos_updated = False
            solver._is_forward_vel_updated = False
            return
        kernel_switch_parent_to_fragments(self, solver)
        if not self._fragment_driven_cpu.any():
            return

        n_envs = self.sim._B
        envs_rows = torch.arange(n_envs, device=gs.device)
        q_rows = self._fragment_q_idx[None].expand(n_envs, -1, -1)
        dof_rows = self._fragment_dof_idx[None].expand(n_envs, -1, -1)
        is_q_target = (
            torch.as_tensor(self._fragment_driven_cpu, device=gs.device).t()[:, :, None] & (self._fragment_q_mask[None])
        )
        is_dof_target = (
            torch.as_tensor(self._fragment_driven_cpu, device=gs.device).t()[:, :, None]
            & (self._fragment_dof_mask[None])
        )
        shadow_q = torch.cat((self._shadow_pos, self._shadow_quat), dim=-1).permute(1, 0, 2)
        shadow_v = torch.cat((self._shadow_lin_vel, self._shadow_ang_vel), dim=-1).permute(1, 0, 2)

        qpos = qd_to_torch(solver.rigid_info.qpos, transpose=True, copy=False)
        qpos_next = qd_to_torch(solver.rigid_info.qpos_next, transpose=True, copy=False)
        vel = qd_to_torch(solver.dyn_state.dofs.vel, transpose=True, copy=False)
        vel_next = qd_to_torch(solver.dyn_state.dofs.vel_next, transpose=True, copy=False)
        # Native-owned rows keep their current solver values; only the driven rows take the shadow target.
        qpos[envs_rows[:, None, None], q_rows] = torch.where(
            is_q_target, shadow_q, qpos[envs_rows[:, None, None], q_rows]
        )
        qpos_next[envs_rows[:, None, None], q_rows] = torch.where(
            is_q_target, shadow_q, qpos_next[envs_rows[:, None, None], q_rows]
        )
        vel[envs_rows[:, None, None], dof_rows] = torch.where(
            is_dof_target, shadow_v, vel[envs_rows[:, None, None], dof_rows]
        )
        vel_next[envs_rows[:, None, None], dof_rows] = torch.where(
            is_dof_target, shadow_v, vel_next[envs_rows[:, None, None], dof_rows]
        )
        if gs.backend == gs.metal:
            torch.mps.synchronize()

        solver._is_forward_pos_updated = False
        solver._is_forward_vel_updated = False

    def substep_tail(self, f: int) -> None:
        """Substep tail, after every solver advanced: scan impact events, then isolated fragments, then sync
        derived particles of native-owned ones."""
        if self._fragment_link is None:
            return
        solver = self.sim.rigid_solver
        if not solver.is_active:
            return
        if self._impact_armed_cpu is not None and self._impact_armed_cpu.any():
            kernel_scan_impact_trigger(
                self._impact_threshold,
                self._impact_min_closing,
                self,
                solver,
                solver.collider._collider_state,
            )
            kernel_snapshot_parent_velocity(self, solver.dyn_state.links)
            # A kernel-fired switch (pending set inside the scan) flips the links' driven gate at the next
            # substep head, whose follow block gates on the host mirror: drop the fired envs here, or the
            # stale shadow target would overwrite the transferred state in that very head. The readback is
            # paid only while the trigger is armed, which ends at the fire.
            fired = np.nonzero(qd_to_numpy(self.parent_switch_pending_q) * self._impact_armed_cpu)[0]
            if len(fired) > 0:
                self._fragment_driven_cpu[:, fired] = False
                self._impact_armed_cpu[fired] = False
        pbd = self.sim.pbd_solver
        if self._solid_entity is None or not pbd.is_active:
            return
        kernel_scan_handoff_candidates(self, pbd)
        kernel_sync_native_derived_particles(self, pbd, self.sim.rigid_solver)

    def import_fragment_state(self, fragment_idx: int, envs_idx, pos, quat, lin_vel, ang_vel) -> None:
        """Import one fragment to native ownership from an explicit pose and velocity (manual mode).

        Writes the state through the public solver setters (which refresh forward kinematics and velocity caches),
        clears the externally-driven gate, re-enables native contacts for the fragment's link, and flips the owner
        to NATIVE for the given environments.
        """
        self._require_registry()
        envs_idx = self.sim.scene._sanitize_envs_idx(envs_idx)
        solver = self.sim.rigid_solver
        i_k = fragment_idx

        pos_t = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device).reshape(-1, 3)
        quat_t = torch.as_tensor(quat, dtype=gs.tc_float, device=gs.device).reshape(-1, 4)
        lin_vel_t = torch.as_tensor(lin_vel, dtype=gs.tc_float, device=gs.device).reshape(-1, 3)
        ang_vel_t = torch.as_tensor(ang_vel, dtype=gs.tc_float, device=gs.device).reshape(-1, 3)
        for tensor in (pos_t, quat_t, lin_vel_t, ang_vel_t):
            if not torch.isfinite(tensor).all():
                gs.raise_exception("PBDRigidFragmentBridge refuses to import a non-finite fragment state.")

        q_start = int(self._fragment_q_start[i_k])
        dof_start = int(self._fragment_dof_start[i_k])
        qpos = solver.get_qpos()
        dofs_vel = solver.get_dofs_velocity()
        if solver.n_envs == 0:
            qpos[q_start : q_start + 3] = pos_t[0]
            qpos[q_start + 3 : q_start + 7] = quat_t[0]
            dofs_vel[dof_start : dof_start + 3] = lin_vel_t[0]
            dofs_vel[dof_start + 3 : dof_start + 6] = ang_vel_t[0]
            solver.set_qpos(qpos)
            solver.set_dofs_velocity(dofs_vel)
        else:
            qpos[envs_idx, q_start : q_start + 3] = pos_t
            qpos[envs_idx, q_start + 3 : q_start + 7] = quat_t
            dofs_vel[envs_idx, dof_start : dof_start + 3] = lin_vel_t
            dofs_vel[envs_idx, dof_start + 3 : dof_start + 6] = ang_vel_t
            solver.set_qpos(qpos, envs_idx=envs_idx)
            solver.set_dofs_velocity(dofs_vel, envs_idx=envs_idx)

        envs_list = envs_idx.tolist()
        kernel_set_link_externally_driven(
            int(self._fragment_link[i_k]), envs_idx, is_driven=0, dyn_state=solver.dyn_state
        )
        kernel_set_link_contact_enabled(
            int(self._fragment_link[i_k]), envs_idx, is_contact_enabled=1, dyn_state=solver.dyn_state
        )
        kernel_set_fragment_owner(i_k, envs_idx, int(FragmentOwner.NATIVE), self)
        self._fragment_owner_cpu[i_k, envs_list] = int(FragmentOwner.NATIVE)
        self._fragment_driven_cpu[i_k, envs_list] = False

    def restore_lifecycle_from_links(self, envs_idx) -> None:
        """Rebuild ownership, pending and particle flags from the restored rigid link switches (set_state path).

        Runs after the rigid solver restored a state slice: fragments whose links are externally driven return to
        PBD ownership (shadow), the rest stay native, pending clears, and the host mirrors are refreshed from the
        GPU fields so the next substep head gates on current data.
        """
        if self._fragment_link is None:
            return
        solver = self.sim.rigid_solver
        if solver.n_envs == 0:
            # A non-parallelized scene still carries one implicit env slot; its reset path passes a
            # concrete envs_idx that _sanitize_envs_idx rejects, so resolve the single slot directly.
            envs_idx = torch.zeros(1, dtype=gs.tc_int, device=gs.device)
        else:
            envs_idx = self.sim.scene._sanitize_envs_idx(envs_idx)
        pbd = self.sim.pbd_solver
        if self._solid_entity is not None and pbd.is_active:
            kernel_restore_fragment_ownership(envs_idx, self, solver, pbd)
        elif self._parent_link is not None:
            # manual mode has no PBD side: ownership follows the restored link flags (driven = dormant
            # fragment, undriven = natively owned), keeping the mirror consistent for the next switch
            driven = qd_to_numpy(solver.dyn_state.links.is_externally_driven, transpose=True)
            envs_list = tensor_to_array(envs_idx).tolist()
            for i_k in range(self.n_fragments):
                i_l = int(self._fragment_link[i_k])
                for i_b in envs_list:
                    self.fragment_owner_q[i_k, i_b] = (
                        int(FragmentOwner.PBD) if driven[i_b, i_l] else int(FragmentOwner.NATIVE)
                    )
            # the restored world predates any open impact event: drop its partial accumulation so the trigger
            # judges only events of the restored timeline (the armed switch itself is a host decision)
            if self.impact_armed_q is not None:
                for i_b in envs_list:
                    self.impact_accum_q[i_b] = 0.0
                    self.impact_event_open_q[i_b] = False
        owner = qd_to_numpy(self.fragment_owner_q, transpose=True)
        self._fragment_owner_cpu[...] = owner.T.astype(np.int64)
        driven = qd_to_numpy(solver.dyn_state.links.is_externally_driven, transpose=True)
        self._fragment_driven_cpu[...] = driven[:, self._fragment_link].T.astype(bool)

    def revoke_native_ownership(self, entity, particles_idx_local, envs_idx) -> None:
        """Export and return to PBD every native-owned fragment touched by a public particle setter.

        Solid-mode contract for `set_particles_pos` / `set_particles_vel` / `set_particles_active` /
        `fix_particles_to_link`: the members' particle state is first re-derived from the current native link
        pose and velocity (so nothing is lost), then every touched native-owned fragment flips back to PBD
        ownership (shadow link driven again, native contacts off) and the caller's write lands on the
        authoritative PBD arrays instead of a derived view. A returned fragment re-enters native ownership
        only by passing the handoff admission scan again.
        """
        if self._fragment_link is None or self._solid_entity is not entity:
            return
        if self.fragment_owner_q is None:
            gs.raise_exception("PBDRigidFragmentBridge dynamic state is allocated at scene build; build first.")
        # The entity setters pass envs_idx already sanitized (a tensor covering the implicit env of a
        # non-parallelized scene as well), so it is used as-is here.
        envs_list = envs_idx.tolist()
        touched_fragments = np.zeros(self.n_fragments, dtype=bool)
        touched_fragments[self._particle_fragment[tensor_to_array(particles_idx_local).reshape(-1)]] = True
        owner = self.read_owner_mirror()
        touched_fragments &= (owner[:, envs_list] == int(FragmentOwner.NATIVE)).any(1)
        if not touched_fragments.any():
            return
        solver = self.sim.rigid_solver
        pbd = self.sim.pbd_solver
        kernel_sync_native_derived_particles(self, pbd, solver)
        fragments_idx = np.nonzero(touched_fragments)[0]
        kernel_revoke_fragment_ownership(fragments_idx, envs_idx, self, solver, pbd)
        for i_k in fragments_idx.tolist():
            self._fragment_owner_cpu[i_k, envs_list] = int(FragmentOwner.PBD)
            self._fragment_driven_cpu[i_k, envs_list] = True

    def read_owner_mirror(self) -> np.ndarray:
        """Host copy of the per-fragment, per-env owner enum (diagnostic readback; syncs)."""
        return qd_to_numpy(self.fragment_owner_q, transpose=True).T.astype(np.int64)

    def read_commit_counts(self) -> np.ndarray:
        """Host copy of the per-fragment, per-env commit counter (diagnostic readback; syncs)."""
        return qd_to_numpy(self.commit_count_q, transpose=True).T.astype(np.int64)

    def read_commit_rejected(self) -> np.ndarray:
        """Host copy of the per-env commit rejection counter (diagnostic readback; syncs)."""
        return qd_to_numpy(self.commit_rejected_q).astype(np.int64)

    def _require_registry(self) -> None:
        if self._fragment_link is None:
            gs.raise_exception("PBDRigidFragmentBridge has no registered fragments.")
        if self.fragment_owner_q is None:
            gs.raise_exception("PBDRigidFragmentBridge dynamic state is allocated at scene build; build first.")
