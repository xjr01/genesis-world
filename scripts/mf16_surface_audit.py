"""Offline continuous-interface audit for the MF-16 mug surface.

This diagnostic is deliberately independent of the online ``coherent_mug_surface`` statistic.
It reconstructs a normalized color field on a fixed Cartesian grid with the PBD solver's
poly6 density kernel::

    rho_j = sum_k W_poly6(|x_j-x_k|, h)
    V_j = 1/rho_j
    phi(x) = sum_j V_j W_poly6(|x-x_j|, h),  h = 0.010 m.

The support is not ``2*ps``: Genesis PBD uses ``h*dist_scale`` with ``h=1`` and
``dist_scale=particle_radius/0.4=.010`` at ps=.008.  Current per-particle volume proxies prevent
compression (the target core reaches rho/rho0 about 1.14) from being misread as a thicker liquid
interface.  Marching cubes extracts the a-priori half-volume level ``phi=0.5``.  No fitted height
offset is applied.  The t=0 frame only validates flat-pool bias and establishes the separately
reported density ratio; it does not tune the target frames.

Of all mesh components, only the component anchored to the lower central mug bulk is retained.
The report includes its apex, robust apex, centre-column height, wall/contact-line azimuthal
coverage, near-rim coverage, an axisymmetric radial-curvature fit, and a poly6 core
density proxy normalized by the t=0 core median.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import generate_binary_structure, label as voxel_label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


PS = 0.008
# Exact physical support of PBDSolver.poly6 for ps=.008:
# particle_radius=.004, dist_scale=particle_radius/.4=.010, solver h=1.
H = 0.010
R_IN = 0.150
Z_FLOOR = 0.016
Z_RIM = 0.316
ISO_LEVEL = 0.5
STRICT_CORE_R = 0.080
GRID_DX = 0.5 * PS
K_NEIGHBORS = 96
N_AZIMUTH = 72

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_DUMPS = WORKSPACE / "videos" / "mf16_smoke_dumps.npz"
DEFAULT_SETTLED = WORKSPACE / "videos" / "mf16_settled.npz"
DEFAULT_CSV = WORKSPACE / "videos" / "mf16_surface_audit.csv"
DEFAULT_JSON = WORKSPACE / "videos" / "mf16_surface_audit_calibration.json"
DEFAULT_PROFILES = WORKSPACE / "videos" / "mf16_surface_audit_profiles.npz"
DEFAULT_SENSITIVITY = WORKSPACE / "videos" / "mf16_surface_audit_sensitivity.csv"
MUG_BALLS = WORKSPACE / "videos" / "mf16_balls_mug.npz"
JUG_BALLS = WORKSPACE / "videos" / "mf16_balls_jug.npz"


def poly6_kernel(distance: np.ndarray) -> np.ndarray:
    """Normalized Genesis-PBD poly6 shape on the exact physical support H.

    PBDSolver evaluates ``315/(64*pi) * (1-d_scaled**2)**3``.  The common coefficient is kept;
    Multiplying the solver's dimensionless shape by ``H^-3`` gives volume proxies in m^3;
    this common factor cancels from the Shepard color field and density ratios.
    """
    q = distance / H
    out = np.zeros_like(q, dtype=np.float64)
    inside = q < 1.0
    out[inside] = 315.0 / (64.0 * np.pi * H**3) * (1.0 - q[inside] ** 2) ** 3
    return out


def raw_field_values(tree: cKDTree, query: np.ndarray) -> np.ndarray:
    """Evaluate the unweighted PBD-poly6 neighbour sum."""
    answer = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), 20_000):
        stop = min(start + 20_000, len(query))
        dist, _ = tree.query(
            query[start:stop], k=K_NEIGHBORS, distance_upper_bound=H, workers=-1
        )
        answer[start:stop] = np.sum(poly6_kernel(dist), axis=1)
    return answer


def particle_volumes(tree: cKDTree, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return V_j=1/sum_k W_jk and the corresponding density proxies."""
    dist, _ = tree.query(source, k=K_NEIGHBORS, distance_upper_bound=H, workers=-1)
    rho = np.sum(poly6_kernel(dist), axis=1)
    if np.any(rho <= 0.0) or not np.isfinite(rho).all():
        raise RuntimeError("invalid particle volume proxy")
    return 1.0 / rho, rho


def shepard_field_values(
    tree: cKDTree, volumes: np.ndarray, query: np.ndarray
) -> np.ndarray:
    """Evaluate sum_j V_j W(x-x_j), using each frame's current volume proxies."""
    answer = np.empty(len(query), dtype=np.float64)
    padded = np.r_[volumes, 0.0]
    for start in range(0, len(query), 20_000):
        stop = min(start + 20_000, len(query))
        dist, idx = tree.query(
            query[start:stop], k=K_NEIGHBORS, distance_upper_bound=H, workers=-1
        )
        answer[start:stop] = np.sum(poly6_kernel(dist) * padded[idx], axis=1)
    return answer


def select_mug_sources(pos: np.ndarray, valid: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(pos[:, :2], axis=1)
    # One particle spacing above the visible rim admits the requested 0.5ps dome but rejects the
    # high incoming stream.  The radial cut lies inside the physical outer wall.
    return valid & (r <= R_IN + PS) & (pos[:, 2] >= 0.0) & (pos[:, 2] <= Z_RIM + PS)


def build_grid(tree: cKDTree, volumes: np.ndarray, dx: float = GRID_DX):
    axes = (
        np.arange(-R_IN - 2.0 * H, R_IN + 2.0 * H + 0.25 * dx, dx),
        np.arange(-R_IN - 2.0 * H, R_IN + 2.0 * H + 0.25 * dx, dx),
        np.arange(0.0, Z_RIM + 3.0 * H + 0.25 * dx, dx),
    )
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    query = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    phi = shepard_field_values(tree, volumes, query).reshape(xx.shape)
    radial = np.sqrt(xx * xx + yy * yy)
    phi[radial > R_IN + 2.0 * H] = 0.0
    return axes, phi


def anchored_component(vertices: np.ndarray, faces: np.ndarray):
    n = len(vertices)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    graph = graph + graph.T
    n_comp, labels = connected_components(graph.tocsr(), directed=False)
    r = np.linalg.norm(vertices[:, :2], axis=1)
    anchor = (r <= 0.5 * R_IN) & (vertices[:, 2] <= Z_FLOOR + 0.08)
    anchor_counts = np.bincount(labels[anchor], minlength=n_comp)
    sizes = np.bincount(labels, minlength=n_comp)
    chosen = int(np.argmax(anchor_counts)) if anchor_counts.max() > 0 else int(np.argmax(sizes))
    keep_v = labels == chosen
    remap = np.full(n, -1, dtype=np.int64)
    remap[keep_v] = np.arange(np.count_nonzero(keep_v))
    keep_f = np.all(keep_v[faces], axis=1)
    return vertices[keep_v], remap[faces[keep_f]], int(n_comp), int(sizes[chosen])


def top_envelope(vertices: np.ndarray):
    ij = np.floor((vertices[:, :2] + R_IN + PS) / GRID_DX + 1e-8).astype(np.int32)
    key = ij[:, 0].astype(np.int64) * 100_000 + ij[:, 1]
    order = np.lexsort((vertices[:, 2], key))
    sorted_key = key[order]
    last = np.r_[np.flatnonzero(sorted_key[1:] != sorted_key[:-1]), len(order) - 1]
    return vertices[order[last]]


def quiet_pool_reference(source: np.ndarray) -> tuple[float, int]:
    core = source[np.linalg.norm(source[:, :2], axis=1) <= 0.08]
    ij = np.floor((core[:, :2] + 0.08) / PS).astype(np.int32)
    key = ij[:, 0].astype(np.int64) * 10_000 + ij[:, 1]
    order = np.lexsort((core[:, 2], key))
    skey = key[order]
    starts = np.r_[0, 1 + np.flatnonzero(skey[1:] != skey[:-1])]
    ends = np.r_[starts[1:], len(order)]
    tops = np.asarray([core[order[e - 1], 2] for s, e in zip(starts, ends) if e - s >= 4])
    if len(tops) < 20:
        raise RuntimeError(f"t=0 has only {len(tops)} populated core columns")
    return float(np.median(tops) + 0.5 * PS), int(len(tops))


def radial_metrics(top: np.ndarray):
    r = np.linalg.norm(top[:, :2], axis=1)
    bins = np.floor(r / PS).astype(np.int32)
    rb, zb, nb = [], [], []
    for b in np.unique(bins):
        mask = (bins == b) & (r <= R_IN - PS)
        if np.count_nonzero(mask) >= 8:
            rb.append(float(np.median(r[mask])))
            zb.append(float(np.median(top[mask, 2])))
            nb.append(int(np.count_nonzero(mask)))
    rb, zb, nb = np.asarray(rb), np.asarray(zb), np.asarray(nb)
    fit = rb <= 0.11
    if np.count_nonzero(fit) >= 5:
        design = np.column_stack((rb[fit] ** 2, np.ones(np.count_nonzero(fit))))
        coef = np.linalg.lstsq(design * np.sqrt(nb[fit, None]), zb[fit] * np.sqrt(nb[fit]), rcond=None)[0]
        pred = design @ coef
        ss_res = float(np.sum((zb[fit] - pred) ** 2))
        ss_tot = float(np.sum((zb[fit] - np.mean(zb[fit])) ** 2))
        curvature = float(-2.0 * coef[0])  # positive means centre-high convexity
        r2 = 1.0 - ss_res / max(ss_tot, 1e-20)
    else:
        curvature = r2 = float("nan")
    return r, rb, zb, nb, curvature, r2


def robust_dome_shape(top: np.ndarray):
    """Robust centre-platform to inner-lip profile, including azimuth consistency.

    Medians, rather than maxima, prevent the known sparse r=.13-.15 wall-high particles from
    defining the shape.  ``radial_nonincrease_fraction`` permits 0.5 mm discretization noise
    per outward bin; its tolerance is fixed, not fitted to a target frame.
    """
    r = np.linalg.norm(top[:, :2], axis=1)
    plateau = top[r <= STRICT_CORE_R, 2]
    lip = top[(r >= 0.120) & (r <= R_IN - PS), 2]
    plateau_z = float(np.median(plateau)) if len(plateau) else float("nan")
    lip_z = float(np.median(lip)) if len(lip) else float("nan")
    _, rb, zb, _, _, _ = radial_metrics(top[r <= R_IN - PS])
    outer = (rb >= STRICT_CORE_R) & (rb <= R_IN - PS)
    oz = zb[outer]
    monotone = float(np.mean(np.diff(oz) <= 0.0005)) if len(oz) >= 2 else float("nan")

    theta = np.mod(np.arctan2(top[:, 1], top[:, 0]), 2.0 * np.pi)
    drops = []
    for sector in range(12):
        sm = (theta >= sector * 2.0 * np.pi / 12.0) & (theta < (sector + 1) * 2.0 * np.pi / 12.0)
        sc = top[sm & (r <= STRICT_CORE_R), 2]
        se = top[sm & (r >= 0.120) & (r <= R_IN - PS), 2]
        if len(sc) >= 8 and len(se) >= 8:
            drops.append(float(np.median(sc) - np.median(se)))
    drops = np.asarray(drops)
    return {
        "center_platform_z_median": plateau_z,
        "inner_lip_z_median": lip_z,
        "center_to_lip_drop": plateau_z - lip_z,
        "radial_nonincrease_fraction": monotone,
        "azimuth_center_above_lip_fraction": (
            float(np.mean(drops > 0.0)) if len(drops) else float("nan")
        ),
        "azimuth_drop_median": float(np.median(drops)) if len(drops) else float("nan"),
        "azimuth_drop_p10": float(np.percentile(drops, 10)) if len(drops) else float("nan"),
        "n_profile_sectors": int(len(drops)),
    }


def azimuth_coverage(top: np.ndarray):
    r = np.linalg.norm(top[:, :2], axis=1)
    annulus = (r >= R_IN - 2.0 * PS) & (r <= R_IN + PS)
    theta = np.mod(np.arctan2(top[:, 1], top[:, 0]), 2.0 * np.pi)
    bins = np.floor(theta / (2.0 * np.pi) * N_AZIMUTH).astype(np.int32) % N_AZIMUTH
    contact = np.unique(bins[annulus])
    near_rim = np.unique(bins[annulus & (top[:, 2] >= Z_RIM - PS)])
    return len(contact) / N_AZIMUTH, len(near_rim) / N_AZIMUTH


def core_density_ratio(
    source: np.ndarray, particle_rho: np.ndarray, surface_z: float, baseline: float | None
):
    r = np.linalg.norm(source[:, :2], axis=1)
    core_mask = (
        (r <= 0.06)
        & (source[:, 2] >= Z_FLOOR + 2.0 * H)
        & (source[:, 2] <= surface_z - 2.0 * H)
    )
    rho = particle_rho[core_mask]
    if len(rho) < 20:
        return float("nan"), float("nan"), float("nan"), 0
    med = float(np.median(rho))
    base = med if baseline is None else baseline
    return med / base, float(np.percentile(rho, 10) / base), float(np.percentile(rho, 90) / base), len(rho)


META_PREFIXES = ("c_", "vel_", "pose_", "valid_", "mug_resident_", "not_jug_",
                 "jug_wall_entered_", "jug_bottom_entered_", "mug_wall_entered_",
                 "mug_bottom_entered_", "mug_rim_crossed_", "mug_cleared_",
                 "mug_wall_tunnel_", "mug_bottom_tunnel_")


def transform_points(local: np.ndarray, pose: np.ndarray) -> np.ndarray:
    pos, q = pose[:3], pose[3:].copy()
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rot = np.array(
        [[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
         [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
         [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]],
        dtype=np.float64,
    )
    return local @ rot.T + pos


def fixed_xy_component(axes, phi: np.ndarray, iso: float):
    inside = phi >= iso
    labels, n_components = voxel_label(inside, structure=generate_binary_structure(3, 1))
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    anchor = (xx * xx + yy * yy <= (0.5 * R_IN) ** 2) & (zz <= Z_FLOOR + 0.08)
    candidates = labels[anchor & inside]
    candidates = candidates[candidates > 0]
    if not len(candidates):
        raise RuntimeError(f"iso={iso} has no lower-core anchored component")
    counts = np.bincount(candidates)
    chosen = int(np.argmax(counts))
    component = labels == chosen
    has = np.any(component, axis=2)
    rev = np.argmax(component[:, :, ::-1], axis=2)
    iz = component.shape[2] - 1 - rev
    ii, jj = np.nonzero(has)
    kk = iz[ii, jj]
    z0 = axes[2][kk]
    ztop = z0.copy()
    can_interp = kk + 1 < len(axes[2])
    if np.any(can_interp):
        a, b, k = ii[can_interp], jj[can_interp], kk[can_interp]
        f0, f1 = phi[a, b, k], phi[a, b, k + 1]
        frac = np.clip((iso - f0) / np.where(np.abs(f1 - f0) > 1e-12, f1 - f0, -1.0), 0.0, 1.0)
        ztop[can_interp] = axes[2][k] + frac * (axes[2][k + 1] - axes[2][k])
    top = np.column_stack((axes[0][ii], axes[1][jj], ztop))
    dx = float(axes[0][1] - axes[0][0])
    rr = np.sqrt(xx * xx + yy * yy)
    rim_vox = component & (np.abs(rr - R_IN) <= 1.5 * dx) & (np.abs(zz - Z_RIM) <= dx)
    angles = np.mod(np.arctan2(yy[rim_vox], xx[rim_vox]), 2.0 * np.pi)
    bins = np.floor(angles / (2.0 * np.pi) * N_AZIMUTH).astype(int) % N_AZIMUTH
    rim_coverage = len(np.unique(bins)) / N_AZIMUTH
    component_volume = float(np.count_nonzero(component) * dx**3)
    field_volume = float(np.sum(phi[component]) * dx**3)
    return top, component, int(n_components), int(np.count_nonzero(component)), rim_coverage, component_volume, field_volume


def directional_curvature(top: np.ndarray):
    _, rb, zb, nb, curvature, r2 = radial_metrics(top)
    theta = np.mod(np.arctan2(top[:, 1], top[:, 0]), 2.0 * np.pi)
    sector_curv, sector_r2 = [], []
    for sector in range(8):
        mask = (theta >= sector * np.pi / 4.0) & (theta < (sector + 1) * np.pi / 4.0)
        if np.count_nonzero(mask) >= 30:
            _, _, _, _, c, q = radial_metrics(top[mask])
            if np.isfinite(c) and np.isfinite(q):
                sector_curv.append(c)
                sector_r2.append(q)
    sc = np.asarray(sector_curv)
    sr = np.asarray(sector_r2)
    good = sr >= 0.5
    positive = float(np.mean(sc[good] > 0.0)) if np.any(good) else float("nan")
    spread = float(np.std(sc[good]) / max(abs(np.mean(sc[good])), 1e-12)) if np.any(good) else float("nan")
    return rb, zb, nb, curvature, r2, positive, spread, len(sc)


def actual_pbd_density(
    pos: np.ndarray, valid: np.ndarray, bulk_mask: np.ndarray, centre_z: float,
    mug_balls: np.ndarray, jug_balls: np.ndarray, rho0: float | None,
):
    fluid = pos[valid]
    geometry = np.vstack((fluid, mug_balls, jug_balls))
    tree = cKDTree(geometry)
    r = np.linalg.norm(pos[:, :2], axis=1)
    core_mask = valid & bulk_mask & (r <= 0.06) & (pos[:, 2] >= Z_FLOOR + 2.0 * H) \
        & (pos[:, 2] <= centre_z - 2.0 * H)
    core = pos[core_mask]
    if len(core) < 20:
        return float("nan"), float("nan"), float("nan"), 0, rho0
    rho = raw_field_values(tree, core) - float(poly6_kernel(np.array([0.0]))[0])
    baseline = float(np.median(rho)) if rho0 is None else rho0
    return (
        float(np.median(rho) / baseline), float(np.percentile(rho, 10) / baseline),
        float(np.percentile(rho, 90) / baseline), len(core), baseline,
    )


def reconstruct_grid(pos: np.ndarray, valid: np.ndarray, contributor: np.ndarray | None, dx: float):
    fluid_idx = np.flatnonzero(valid)
    fluid = pos[fluid_idx]
    tree_all = cKDTree(fluid)
    counts = tree_all.query_ball_point(fluid, H, return_length=True, workers=-1)
    if int(np.max(counts)) >= K_NEIGHBORS:
        raise RuntimeError(f"K_NEIGHBORS={K_NEIGHBORS} too small; observed {int(np.max(counts))}")
    volumes, shepard_rho = particle_volumes(tree_all, fluid)
    volume_full = np.zeros(len(pos), dtype=np.float64)
    volume_full[fluid_idx] = volumes
    if contributor is None:
        tree, weights = tree_all, volumes
    else:
        use = valid & contributor
        tree = cKDTree(pos[use])
        weights = volume_full[use]
    axes, phi = build_grid(tree, weights, dx)
    return axes, phi, volume_full, shepard_rho


def metric_row(time_s, iso, axes, phi, volume_full, bulk_history):
    top, component, n_comp, n_vox, rim_cov, iso_volume, field_volume = fixed_xy_component(axes, phi, iso)
    r = np.linalg.norm(top[:, :2], axis=1)
    # Strict apex/curvature evidence is deliberately central.  Slow particles in the
    # r=.13-.15 wall/contact annulus can sit anomalously high (.40 m in diag9) and must not be
    # reported as a dome apex.  Rim/contact-line coverage remains a separate all-radius metric.
    bulk = r <= STRICT_CORE_R
    bt = top[bulk, 2]
    centre = top[r <= PS, 2]
    profile = r <= R_IN - PS
    rb, zb, nb, curv, r2, sector_positive, sector_spread, n_sector = directional_curvature(top[profile])
    shape = robust_dome_shape(top)
    particle_volume = float(np.sum(volume_full[bulk_history]))
    result = {
        "t": time_s, "iso": iso, "n_components": n_comp, "component_voxels": n_vox,
        "bulk_apex_z": float(np.max(bt)), "bulk_p99_z": float(np.percentile(bt, 99)),
        "bulk_fraction_above_0320": float(np.mean(bt >= Z_RIM + 0.5 * PS)),
        "center_column_z_p95": float(np.percentile(centre, 95)),
        "rim_line_azimuth_coverage": rim_cov, "radial_curvature": curv,
        "radial_fit_r2": r2, "sector_positive_fraction": sector_positive,
        "sector_curvature_cv": sector_spread, "n_curvature_sectors": n_sector,
        "iso_volume": iso_volume, "field_volume": field_volume,
        "history_particle_volume": particle_volume,
        "iso_to_particle_volume": iso_volume / max(particle_volume, 1e-20),
        "field_to_particle_volume": field_volume / max(particle_volume, 1e-20),
    }
    result.update(shape)
    return result, (rb, zb, nb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dumps", type=Path, default=DEFAULT_DUMPS)
    parser.add_argument("--settled", type=Path, default=DEFAULT_SETTLED)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--sensitivity", type=Path, default=DEFAULT_SENSITIVITY)
    args = parser.parse_args()

    with np.load(args.settled) as settled:
        n_fluid = int(settled["n_fluid"])
        n_coffee = int(settled["n_coffee"])
    with np.load(MUG_BALLS) as data:
        mug_balls = np.asarray(data["pos_local"], dtype=np.float64)
    with np.load(JUG_BALLS) as data:
        jug_local = np.asarray(data["pos_local"], dtype=np.float64)
    dumps = np.load(args.dumps)
    times = sorted(float(k) for k in dumps.files if not k.startswith(META_PREFIXES))
    assert times and times[0] == 0.0

    rows, sensitivity_rows, profiles = [], [], {}
    rho0 = None
    calibration = None
    for time_s in times:
        key = f"{time_s:g}"
        pos = np.asarray(dumps[key], dtype=np.float64)
        valid = np.asarray(dumps[f"valid_{key}"], dtype=bool)
        resident = np.asarray(dumps[f"mug_resident_{key}"], dtype=bool)
        not_jug = np.asarray(dumps[f"not_jug_{key}"], dtype=bool)
        cleared = np.asarray(dumps[f"mug_cleared_{key}"], dtype=bool)
        bulk_history = valid & resident & not_jug & ~cleared
        pose = np.asarray(dumps[f"pose_{key}"], dtype=np.float64)
        jug_balls = transform_points(jug_local, pose)

        # Main geometry always estimates V_j from every valid liquid particle and uses every
        # valid liquid particle as a color-field contributor.  History masks only identify the
        # anchored mug volume and the density sample; they do not hard-clip the color source.
        axes, phi, volume_full, _ = reconstruct_grid(pos, valid, None, GRID_DX)
        frame_rows = []
        for iso in (0.45, 0.50, 0.55):
            row, profile = metric_row(time_s, iso, axes, phi, volume_full, bulk_history)
            if iso == 0.50:
                ratio, p10, p90, n_core, rho0 = actual_pbd_density(
                    pos, valid, bulk_history, row["center_column_z_p95"], mug_balls, jug_balls, rho0
                )
                row.update(core_pbd_rho_over_rho0=ratio, core_pbd_rho_p10=p10,
                           core_pbd_rho_p90=p90, n_core_density=n_core)
                profiles[f"r_{key}"] = profile[0]
                profiles[f"z_{key}"] = profile[1]
                profiles[f"n_{key}"] = profile[2]
            else:
                row.update(core_pbd_rho_over_rho0=float("nan"), core_pbd_rho_p10=float("nan"),
                           core_pbd_rho_p90=float("nan"), n_core_density=0)
            rows.append(row)
            frame_rows.append(row)

        main = frame_rows[1]
        if time_s == 0.0:
            ref, n_ref = quiet_pool_reference(pos[bulk_history])
            calibration = {
                "method": "all-valid-fluid current-volume Shepard field; fixed-XY highest crossing",
                "pbd_poly6_physical_support": H, "grid_dx": GRID_DX,
                "strict_apex_curvature_radius": STRICT_CORE_R,
                "robust_profile_radius": R_IN - PS,
                "profile_lip_annulus": [0.120, R_IN - PS],
                "radial_nonincrease_tolerance": 0.0005,
                "iso_levels": [0.45, 0.50, 0.55], "height_offset": 0.0,
                "t0_quiet_reference_z": ref, "t0_quiet_reference_columns": n_ref,
                "t0_iso50_center_z": main["center_column_z_p95"],
                "t0_iso50_minus_reference": main["center_column_z_p95"] - ref,
                "t0_global_particle_z99": float(np.percentile(pos[:n_coffee, 2], 99)),
                "geometry_self_term": "included only in V_j Shepard volume estimate",
                "pbd_density_self_term": "excluded; static mug and posed jug balls included",
                "target_frames_used_to_fit_parameters": False,
            }

        # Pre-declared sensitivities only: coarser dx and history-gated contributors at t0 and
        # three integer hold samples.  They never alter the main fixed configuration.
        if any(abs(time_s - s) < 1e-8 for s in (0.0, 12.0, 13.0, 14.0)):
            axes2, phi2, vol2, _ = reconstruct_grid(pos, valid, None, 0.005)
            row2, _ = metric_row(time_s, 0.50, axes2, phi2, vol2, bulk_history)
            row2.update(kind="dx_0.005")
            sensitivity_rows.append(row2)
            axes3, phi3, vol3, _ = reconstruct_grid(pos, valid, bulk_history, GRID_DX)
            row3, _ = metric_row(time_s, 0.50, axes3, phi3, vol3, bulk_history)
            row3.update(kind="history_gated_source")
            sensitivity_rows.append(row3)

        print(
            f"t={time_s:4.1f} iso50 apex={main['bulk_apex_z']:.6f} "
            f"p99={main['bulk_p99_z']:.6f} centre={main['center_column_z_p95']:.6f} "
            f"area>=.320={main['bulk_fraction_above_0320']:.3f} rim={main['rim_line_azimuth_coverage']:.3f} "
            f"platform/lip/drop={main['center_platform_z_median']:.6f}/"
            f"{main['inner_lip_z_median']:.6f}/{main['center_to_lip_drop']:+.6f} "
            f"mono={main['radial_nonincrease_fraction']:.3f} azi+={main['azimuth_center_above_lip_fraction']:.3f} "
            f"curv={main['radial_curvature']:+.4f}/R2={main['radial_fit_r2']:.3f} "
            f"sector+={main['sector_positive_fraction']:.3f} rho={main['core_pbd_rho_over_rho0']:.3f} "
            f"Vclose={main['iso_to_particle_volume']:.3f}/{main['field_to_particle_volume']:.3f}"
        )

    assert calibration is not None
    args.calibration_json.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(rows[0])
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    sensitivity_fields = list(sensitivity_rows[0])
    with args.sensitivity.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sensitivity_fields)
        writer.writeheader(); writer.writerows(sensitivity_rows)
    np.savez(args.profiles, **profiles)
    print("CALIBRATION", json.dumps(calibration, sort_keys=True))
    print(f"csv: {args.csv}")
    print(f"sensitivity: {args.sensitivity}")
    print(f"profiles: {args.profiles}")


if __name__ == "__main__":
    main()
