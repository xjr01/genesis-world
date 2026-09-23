#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline MF16 mug-wall geometric-pinning audit.

This diagnostic never advances Genesis.  It combines the strict history masks from the
no-adhesion/friction run with the shipped mug ball set, then compares the union-of-spheres
collision envelope against one denser/smaller-ball candidate.  The candidate is written under a
diagnostic-only name; it must not be used by the solver until boundary-ball mass is scaled for its
smaller surface-quadrature pitch.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS = os.path.join(WORKSPACE, "videos")
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

import mf16_balls as balls_source  # noqa: E402


PS_FLUID = 0.008
R_FLUID = 0.5 * PS_FLUID
R_VISIBLE = float(balls_source.BODIES["mug"]["r_out"])
Z_RIM = 0.316
CURRENT = dict(name="current", pitch=0.008, ball_radius=0.004, inset=0.004, layers=2)
CANDIDATE = dict(name="candidate_s4_rb3", pitch=0.004, ball_radius=0.003,
                 inset=0.003, layers=2)


def outer_mask(dump, t):
    key = f"{t:g}"
    pos = dump[key]
    radius = np.linalg.norm(pos[:, :2], axis=1)
    return (
        dump[f"valid_{key}"]
        & dump[f"not_jug_{key}"]
        & dump[f"mug_cleared_{key}"]
        & ~dump[f"mug_wall_tunnel_{key}"]
        & ~dump[f"mug_bottom_tunnel_{key}"]
        & (radius >= 0.166)
        & (radius <= 0.190)
        & (pos[:, 2] >= 0.020)
        & (pos[:, 2] <= Z_RIM)
    )


def persistent_contact_stats(dump_path, balls_path):
    dump = np.load(dump_path)
    balls = np.load(balls_path)["pos_local"].astype(np.float64)
    times = (16, 17, 18, 19, 20)
    masks = {t: outer_mask(dump, t) for t in times}
    persistent = np.logical_and.reduce([masks[t] for t in times])

    pos = dump["20"].astype(np.float64)
    vel = dump["vel_20"].astype(np.float64)
    p = pos[persistent]
    speed = np.linalg.norm(vel[persistent], axis=1)
    distance_k, nearest_k = cKDTree(balls).query(p, k=24, workers=-1)
    neighbour = balls[nearest_k]
    rel = p[:, None, :] - neighbour
    rel_unit = rel / np.linalg.norm(rel, axis=2)[:, :, None]
    distance = distance_k[:, 0]
    contact = distance <= 1.25 * PS_FLUID
    sphere_normal = rel_unit[:, 0]
    nz = sphere_normal[contact, 2]
    horizontal = np.sqrt(np.maximum(1.0 - nz * nz, 1.0e-15))
    slope = np.abs(nz) / horizontal
    upward = nz > 0.02

    z = p[:, 2]
    z_edges = np.array([0.02, 0.08, 0.14, 0.20, 0.26, Z_RIM + 1.0e-6])
    # Two practical smooth-normal references.  The weighted-vector average is cheap enough for a
    # collision kernel; PCA is a stronger offline plane-fit baseline.  Neither changes the sphere
    # union used for broad phase / CCD.
    # Gaussian weights stay strictly positive so diagnostic particles just outside the 1.25 ps
    # contact band cannot create a zero-weight row.  Reported normal statistics still use the
    # explicit `contact` selection below.
    weights = np.exp(-(distance_k / 0.012) ** 2)
    average_normal = (rel_unit * weights[:, :, None]).sum(axis=1) / weights.sum(axis=1)[:, None]
    average_normal /= np.linalg.norm(average_normal, axis=1)[:, None]
    pca_normal = np.empty_like(average_normal)
    radial = np.column_stack((p[:, :2] / np.linalg.norm(p[:, :2], axis=1)[:, None],
                              np.zeros(len(p))))
    for i, (centres, w) in enumerate(zip(neighbour, weights)):
        centre = (centres * w[:, None]).sum(axis=0) / w.sum()
        offset = centres - centre
        covariance = (offset * w[:, None]).T @ offset / w.sum()
        _, eigenvectors = np.linalg.eigh(covariance)
        n = eigenvectors[:, 0]
        pca_normal[i] = n if np.dot(n, radial[i]) >= 0.0 else -n

    def normal_summary(normal, selection):
        component = normal[selection, 2]
        alignment = np.sum(normal[selection] * radial[selection], axis=1)
        return {
            "normal_z_p01_p50_p99": np.percentile(component, [1, 50, 99]).tolist(),
            "abs_normal_z_p50_p90_p99": np.percentile(np.abs(component), [50, 90, 99]).tolist(),
            "fraction_abs_normal_z_gt_0p1": float(np.mean(np.abs(component) > 0.1)),
            "fraction_abs_normal_z_gt_0p25": float(np.mean(np.abs(component) > 0.25)),
            "radial_alignment_p01": float(np.percentile(alignment, 1)),
        }

    midbody = contact & (p[:, 2] <= 0.292)
    stats = {
        "outer_counts": {str(t): int(masks[t].sum()) for t in times},
        "persistent_16_to_20": int(persistent.sum()),
        "persistent_coffee": int(persistent[:49980].sum()),
        "persistent_milk": int(persistent[49980:].sum()),
        "z_p10_p50_p90": np.percentile(z, [10, 50, 90]).tolist(),
        "z_edges": z_edges.tolist(),
        "z_bin_counts": np.histogram(z, z_edges)[0].tolist(),
        "speed_p50_p90_p99": np.percentile(speed, [50, 90, 99]).tolist(),
        "nearest_distance_p50_p90": np.percentile(distance, [50, 90]).tolist(),
        "within_1p25ps": int(contact.sum()),
        "normal_z_p01_p10_p25_p50_p75_p90_p99":
            np.percentile(nz, [1, 10, 25, 50, 75, 90, 99]).tolist(),
        "fraction_normal_z_positive": float(np.mean(nz > 0.0)),
        "fraction_normal_z_gt_0p1": float(np.mean(nz > 0.1)),
        "fraction_normal_z_gt_0p25": float(np.mean(nz > 0.25)),
        "effective_abs_slope_p50_p90_p99": np.percentile(slope, [50, 90, 99]).tolist(),
        # A unilateral sphere contact can kinematically oppose gravity only on its upper
        # hemisphere. g/nz is the required normal acceleration in units of g.
        "required_normal_force_over_weight_p10_p50_p90":
            np.percentile(1.0 / nz[upward], [10, 50, 90]).tolist(),
        "normal_model_comparison": {
            "nearest_sphere_all": normal_summary(sphere_normal, contact),
            "nearest_sphere_midbody": normal_summary(sphere_normal, midbody),
            "neighbour_weighted_all": normal_summary(average_normal, contact),
            "neighbour_weighted_midbody": normal_summary(average_normal, midbody),
            "neighbour_pca_all": normal_summary(pca_normal, contact),
            "neighbour_pca_midbody": normal_summary(pca_normal, midbody),
            "analytic_cylinder_reference_midbody": normal_summary(radial, midbody),
        },
    }
    return stats, nz, average_normal[contact, 2], pca_normal[contact, 2], persistent, masks


def cylindrical_wall_centres(spec):
    cfg = balls_source.BODIES["mug"]
    z_rim = cfg["t_bottom"] + cfg["h_in"]
    builder = balls_source.Body("mug", ps=spec["pitch"], inset=spec["inset"],
                                layers=spec["layers"])
    builder.inner_wall(cfg["r_in"], cfg["t_bottom"], z_rim)
    builder.outer_wall(cfg["r_out"], z_rim)
    centres = np.concatenate([chunk[2] for chunk in builder.chunks])
    radius = np.linalg.norm(centres[:, :2], axis=1)
    outer_shell_r = cfg["r_out"] - spec["inset"]
    # Include every inner/outer-face row that lands on the outermost centre shell.  In the
    # shipped set the two faces interleave on that shell, so considering only outer_wall/L0
    # would exaggerate its corrugation.
    return centres[np.abs(radius - outer_shell_r) < 1.0e-7]


def envelope_stats(spec):
    centres = cylindrical_wall_centres(spec)
    tree = cKDTree(centres)
    theta = np.linspace(0.0, 2.0 * np.pi, 192, endpoint=False)
    z_values = np.arange(0.040, 0.2921, 0.001)
    ideal = R_VISIBLE + R_FLUID
    collision_radius = spec["ball_radius"] + R_FLUID
    envelope = []
    normal_z = []
    for z in z_values:
        unit = np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
        query = unit * ideal
        query[:, 2] = z
        _, idx = tree.query(query, k=12)
        candidates = centres[idx]
        radial_projection = (
            candidates[:, :, 0] * unit[:, None, 0]
            + candidates[:, :, 1] * unit[:, None, 1]
        )
        tangential = (
            candidates[:, :, 0] * -unit[:, None, 1]
            + candidates[:, :, 1] * unit[:, None, 0]
        )
        perpendicular2 = tangential**2 + (candidates[:, :, 2] - z) ** 2
        root2 = collision_radius**2 - perpendicular2
        roots = np.where(
            root2 >= 0.0,
            radial_projection + np.sqrt(np.maximum(root2, 0.0)),
            -np.inf,
        )
        best = np.argmax(roots, axis=1)
        rr = roots[np.arange(len(theta)), best]
        chosen = candidates[np.arange(len(theta)), best]
        assert np.all(np.isfinite(rr))
        envelope.append(rr)
        normal_z.append((z - chosen[:, 2]) / collision_radius)

    envelope = np.concatenate(envelope)
    normal_z = np.concatenate(normal_z)
    valley = ideal - envelope
    slope = np.abs(normal_z) / np.sqrt(np.maximum(1.0 - normal_z**2, 1.0e-15))
    return {
        "outer_shell_centres": int(len(centres)),
        "ideal_fluid_centre_radius": ideal,
        "collision_radius": collision_radius,
        "envelope_inward_valley_mm_p50_p90_max":
            (1000.0 * np.percentile(valley, [50, 90, 100])).tolist(),
        "abs_normal_z_p50_p90_max": np.percentile(np.abs(normal_z), [50, 90, 100]).tolist(),
        "effective_abs_slope_p50_p90_max": np.percentile(slope, [50, 90, 100]).tolist(),
    }, valley, normal_z


def disk_points(radius, z, spacing):
    radii = np.arange(0.0, radius + 1.0e-12, spacing)
    if radius - radii[-1] >= 0.5 * spacing:
        radii = np.append(radii, radius)
    chunks = [np.array([[0.0, 0.0, z]], dtype=np.float64)]
    for k, r in enumerate(radii[1:], start=1):
        n = balls_source.n_theta(float(r), spacing)
        chunks.append(balls_source.ring(float(r), z, n, balls_source.stagger(n, k)))
    return np.concatenate(chunks)


def write_candidate():
    paths = {}
    for body in ("mug", "jug"):
        # Exactly one inward-offset sample sheet for each existing visible face.  The old generic
        # layers=2 becomes redundant after halving the tangential pitch and would double-sample
        # intermediate shells.  Add the previously implicit outside bottom as its own true face.
        result = balls_source.build_body(
            body,
            balls_source.BODIES[body],
            CANDIDATE["pitch"],
            CANDIDATE["inset"],
            1,
            with_handle=True,
        )
        outer_bottom = disk_points(
            result["cfg"]["r_out"] - CANDIDATE["inset"],
            CANDIDATE["ball_radius"],
            CANDIDATE["pitch"],
        )
        pos = np.vstack((result["pos64"], outer_bottom))
        path = os.path.join(VIDEOS, f"mf16_balls_{body}_candidate_s4_rb3.npz")
        meta = {
            "diagnostic_only": True,
            "body": body,
            "source": "mf16_balls.py shared geometry dictionaries",
            "pitch": CANDIDATE["pitch"],
            "ball_radius": CANDIDATE["ball_radius"],
            "inset": CANDIDATE["inset"],
            "layers_per_visible_face": 1,
            "explicit_outer_bottom_count": int(len(outer_bottom)),
            "visible_outer_radius": float(result["cfg"]["r_out"]),
            "n_balls": int(len(pos)),
            "required_boundary_mass_scale": (CANDIDATE["pitch"] / PS_FLUID) ** 2,
            "warning": "Do not run until solver supports the recorded surface-quadrature mass scale.",
        }
        np.savez_compressed(path, pos_local=pos.astype(np.float32),
                            meta=json.dumps(meta, sort_keys=True))
        paths[body] = {"path": path, **meta}
    return paths


def main():
    dump_path = os.path.join(VIDEOS, "mf16_smoke_diag15_noadh_fric0_dumps.npz")
    mug_path = os.path.join(VIDEOS, "mf16_balls_mug.npz")
    actual, actual_nz, average_nz, pca_nz, _, _ = persistent_contact_stats(dump_path, mug_path)
    current_env, current_valley, current_nz = envelope_stats(CURRENT)
    candidate_env, candidate_valley, candidate_nz = envelope_stats(CANDIDATE)
    candidate_files = write_candidate()

    report = {
        "method": "union-of-spheres radial envelope; fixed theta/z grid, no GPU stepping",
        "fluid_particle_radius": R_FLUID,
        "visible_mug_outer_radius": R_VISIBLE,
        "actual_persistent_contact": actual,
        "current_envelope": {**CURRENT, **current_env},
        "candidate_envelope": {**CANDIDATE, **candidate_env},
        "candidate_files": candidate_files,
        "candidate_contract": {
            "same_visible_OBJ": True,
            "outer_fluid_centre_envelope_radius": R_VISIBLE + R_FLUID,
            "outer_peak_alignment_error":
                abs((R_VISIBLE - CANDIDATE["inset"])
                    + CANDIDATE["ball_radius"] + R_FLUID
                    - (R_VISIBLE + R_FLUID)),
            "inner_peak_alignment_error":
                abs((balls_source.BODIES["mug"]["r_in"] + CANDIDATE["inset"])
                    - (CANDIDATE["ball_radius"] + R_FLUID)
                    - (balls_source.BODIES["mug"]["r_in"] - R_FLUID)),
            "wall_opposite_face_centre_gap":
                balls_source.BODIES["mug"]["t_wall"] - 2.0 * CANDIDATE["inset"],
            "twice_collision_radius": 2.0 * (CANDIDATE["ball_radius"] + R_FLUID),
            "mass_scale_required": (CANDIDATE["pitch"] / PS_FLUID) ** 2,
            "not_gpu_approved": True,
        },
    }
    json_path = os.path.join(VIDEOS, "mf16_ball_pinning_audit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    csv_path = os.path.join(VIDEOS, "mf16_ball_pinning_envelope.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["geometry", "pitch", "ball_radius", "outer_shell_centres",
                         "valley_p50_mm", "valley_p90_mm", "valley_max_mm",
                         "abs_nz_p50", "abs_nz_p90", "abs_nz_max"])
        for spec, env in ((CURRENT, current_env), (CANDIDATE, candidate_env)):
            writer.writerow([
                spec["name"], spec["pitch"], spec["ball_radius"],
                env["outer_shell_centres"],
                *env["envelope_inward_valley_mm_p50_p90_max"],
                *env["abs_normal_z_p50_p90_max"],
            ])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].hist(actual_nz, bins=50, density=True, color="#b2182b", alpha=0.55,
                 label="nearest sphere")
    axes[0].hist(average_nz, bins=50, density=True, color="#2166ac", alpha=0.45,
                 label="neighbour average")
    axes[0].hist(pca_nz, bins=50, density=True, histtype="step", color="#1b7837", lw=1.5,
                 label="neighbour PCA")
    axes[0].axvline(0.0, color="black", lw=1)
    axes[0].set(title="D persistent layer\nnearest-ball normal z",
                xlabel="normal z component", ylabel="density")
    axes[0].legend(fontsize=8)
    axes[1].hist(1000.0 * current_valley, bins=50, density=True, alpha=0.65,
                 label="current s=8mm/rb=4mm")
    axes[1].hist(1000.0 * candidate_valley, bins=50, density=True, alpha=0.65,
                 label="candidate s=4mm/rb=3mm")
    axes[1].set(title="Collision-envelope corrugation", xlabel="inward valley (mm)",
                ylabel="density")
    axes[1].legend(fontsize=8)
    axes[2].hist(np.abs(current_nz), bins=40, density=True, alpha=0.65, label="current")
    axes[2].hist(np.abs(candidate_nz), bins=40, density=True, alpha=0.65,
                 label="candidate")
    axes[2].set(title="Envelope effective vertical normal", xlabel="|normal z|",
                ylabel="density")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle("MF16 mug virtual-ball geometric-pinning audit")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    png_path = os.path.join(VIDEOS, "mf16_ball_pinning_audit.png")
    fig.savefig(png_path, dpi=200)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote: {json_path}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {png_path}")


if __name__ == "__main__":
    main()
