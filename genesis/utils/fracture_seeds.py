"""Volumetric variable-density fragment-seed sampling for pre-fragmented PBD solids.

Seeds are drawn inside the solid wall of a mesh body: a voxel-centre candidate pool is grown inward from
the surface voxelisation, then thinned by a variable-radius greedy Poisson-disk acceptance pass. Each
candidate draws a log-normal exclusion radius clipped to ``radius_range``; an impact lobe (a horizontal
direction restricted to a height band) multiplies the radius down, producing systematically finer
fragments near the impact site. The accepted points are returned in the mesh's local coordinate frame,
matching the convention expected by ``PBD.Solid(fragment_seeds=...)``.

Thin-wall bodies (wall thickness comparable to the voxel pitch) should use ``depth=2``; solid bodies
take a larger depth to fill the interior.
"""

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def sample_volume_seeds(
    mesh: trimesh.Trimesh,
    count_range: tuple[int, int] = (34, 44),
    radius_range: tuple[float, float] = (0.018, 0.05),
    radius_log_sigma: float = 0.55,
    impact_axis: tuple[float, float] = (-1.0, 0.0),
    impact_band: tuple[float, float] = (0.02, 0.10),
    impact_scale: float = 0.5,
    impact_sigma: float = 0.7,
    rng_seed: int = 3,
    overlap: float = 0.85,
    pitch: float = 0.002,
    depth: int = 2,
    max_attempts: int = 14,
    info_out: dict | None = None,
) -> np.ndarray:
    """Sample fragment seeds inside the solid volume of a mesh, in the mesh's local coordinate frame.

    Parameters
    ----------
    mesh
        Watertight (or near-watertight) surface mesh; seeds are drawn inside its wall or solid body.
    count_range
        Accepted seed-count window; the radius scale auto-rescales between attempts until the count
        lands inside this window.
    radius_range
        Exclusion-radius bounds in metres; each seed draws a log-normal radius clipped to this range.
    radius_log_sigma
        Log-normal spread; larger values produce stronger fragment-size variance.
    impact_axis
        Horizontal ``(x, y)`` direction of the impact lobe.
    impact_band
        Local-z height range the lobe acts on.
    impact_scale
        Radius shrink inside the lobe (0 = off, 1 = down to ``radius_range[0]``).
    impact_sigma
        Gaussian width of the lobe on ``impact_axis``.
    rng_seed
        Seed for the internal ``numpy.random.default_rng``; deterministic across calls.
    overlap
        Poisson-disk overlap factor: a candidate is rejected when it lies closer than
        ``overlap * (r_i + r_j)`` to an accepted seed.
    pitch
        Voxel edge length for the candidate pool (smaller = denser pool, slower).
    depth
        Number of voxel layers to grow inward from the surface (2 covers a thin wall).
    max_attempts
        Maximum number of radius-rescale attempts before returning the best result.
    info_out
        Optional dict; filled with ``count``, ``attempts``, and ``rescale_factor`` diagnostics.

    Returns
    -------
    numpy.ndarray
        ``(n, 3)`` float64 array of seed points in the mesh's local coordinate frame, suitable for
        ``PBD.Solid(fragment_seeds=...)``.

    Raises
    ------
    SystemExit
        If the interior candidate pool is too small (< 200 points), indicating that the mesh is not
        watertight or that ``pitch`` is too large for the wall thickness.
    """
    rng = np.random.default_rng(rng_seed)
    lo, hi = np.log(radius_range)
    axis = np.array([impact_axis[0], impact_axis[1], 0.0])
    axis /= np.linalg.norm(axis)
    axis_angle = np.arctan2(axis[1], axis[0])
    z_lo, z_hi = min(impact_band), max(impact_band)

    pool = _candidate_pool(mesh, pitch, depth, rng, int(np.mean(count_range)) * 40)
    if info_out is not None:
        info_out["pool_size"] = len(pool)
    if len(pool) < 200:
        raise SystemExit(
            f"sample_volume_seeds: interior candidate pool too small ({len(pool)} points); "
            "check that the mesh is watertight or reduce pitch"
        )

    rescale = 1.0
    attempts = []
    accepted = np.zeros((0, 3))
    accepted_r = np.zeros(0)
    for attempt in range(max_attempts):
        candidates = pool[rng.permutation(len(pool))][: int(len(pool) * 0.6)]
        log_r = 0.5 * (lo + hi) + radius_log_sigma * rng.standard_normal(len(candidates))
        radii = np.clip(np.exp(log_r) * rescale, radius_range[0], radius_range[1])
        d_axis = np.abs(np.arctan2(candidates[:, 1], candidates[:, 0]) - axis_angle + np.pi) % (2.0 * np.pi) - np.pi
        lobe = np.exp(-0.5 * (d_axis / impact_sigma) ** 2)
        in_band = (candidates[:, 2] >= z_lo) & (candidates[:, 2] <= z_hi)
        mult = np.where(in_band, 1.0 - impact_scale * lobe, 1.0)
        r_eff = np.maximum(radii * mult, radius_range[0] * 0.5)

        acc_pts = np.zeros((0, 3))
        acc_r = np.zeros(0)
        keep = []
        tree = None
        for i in np.argsort(-r_eff):
            if len(acc_pts):
                search_r = overlap * (r_eff[i] + acc_r.max())
                hits = tree.query_ball_point(candidates[i], search_r)
                blocked = any(
                    np.linalg.norm(candidates[i] - acc_pts[j]) < overlap * (r_eff[i] + acc_r[j]) for j in hits
                )
                if blocked:
                    continue
            keep.append(i)
            acc_pts = np.vstack([acc_pts, candidates[i][None]])
            acc_r = np.append(acc_r, r_eff[i])
            tree = cKDTree(acc_pts)
        order = np.argsort(keep)
        keep = np.asarray(keep)[order]
        accepted = candidates[keep]
        accepted_r = radii[keep]
        attempts.append({"attempt": attempt, "rescale": round(rescale, 4), "accepted": len(accepted)})
        if count_range[0] <= len(accepted) <= count_range[1]:
            break
        rescale *= (len(accepted) / np.mean(count_range)) ** 0.5
        if len(accepted) < count_range[0]:
            rescale = min(rescale, 0.95)
        else:
            rescale = max(rescale, 1.05)
    if info_out is not None:
        info_out["rescale_factor"] = round(float(rescale), 4)
        info_out["count"] = len(accepted)
        info_out["per_seed_radius"] = accepted_r.tolist()
        info_out["attempts"] = attempts
    return accepted


def _candidate_pool(mesh: trimesh.Trimesh, pitch: float, depth: int, rng: np.random.Generator, want: int) -> np.ndarray:
    """Interior candidate points: surface voxel centres grown ``depth`` voxels inward by binary dilation."""
    from scipy import ndimage

    vox = mesh.voxelized(pitch=pitch)
    mat = vox.matrix
    if depth > 1:
        mat = ndimage.binary_dilation(mat, iterations=depth - 1)
    pts = vox.indices_to_points(np.argwhere(mat))
    if len(pts) > want * 3:
        pts = pts[rng.choice(len(pts), size=int(want * 3), replace=False)]
    return pts
