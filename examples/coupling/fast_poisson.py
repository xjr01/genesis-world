"""Fast vectorized Poisson-disk sampler for DEM entities (user request 2026-08-27: poisson-sampled
sand bed).

The Genesis/C++ reference `DEMEntity._sample_poisson` is a faithful but pure-python Bridson
loop: for the 1.7x bed (~600-800k grains) it runs at ~80 grains/s -> ~2 h just to build the
scene. This module monkeypatches it with a batch-vectorized equivalent (no genesis-world
changes; installed at runtime by the probe/render scripts).

Algorithm: same Bridson poisson-disk scheme as the reference -- background grid with
cell = min_dist / sqrt(3) (one point per cell), candidates on the spherical shell
[min_dist, 2*min_dist] around active points, up to 300 consecutive failures before an active
point is retired, AABB-center seed point. The ONLY difference is execution: candidates are
generated in batches of 4096 and pre-filtered vectorized (bounds / inside-test / own-cell-empty /
5x5x5 neighbor scan); survivors are then accepted SEQUENTIALLY in random order with the exact
neighbor re-check against the updated grid, so the min_dist >= 2.01*radius guarantee holds
exactly (zero initial overlap, as the elastic contact model requires). The point ORDER and
stream of rng draws differ from the sequential reference, i.e. the packing is statistically
equivalent but not bit-identical (recorded deviation, same category as sampler_seed choice).

Usage: import fast_poisson; fast_poisson.install()  (before scene.build())
"""
import math

import numpy as np

BATCH = 4096
MAX_TRIES = 300  # consecutive failures before an active point is retired (reference value)
PREALLOC = 3_000_000

_offsets = np.array(
    [(dx, dy, dz) for dz in range(-2, 3) for dy in range(-2, 3) for dx in range(-2, 3)],
    dtype=np.int64,
)


def _fast_sample_poisson(trimesh, min_dist, rng, is_inside):
    d = float(min_dist)
    lower, upper = np.asarray(trimesh.bounds, dtype=np.float64)
    cell = d / math.sqrt(3.0)
    grid_size = np.maximum(((upper - lower) / cell).astype(np.int64), 1)
    gx, gy, gz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
    grid = np.full((gz, gy, gx), -1, dtype=np.int64)

    pts = np.zeros((PREALLOC, 3), dtype=np.float64)
    fails = np.zeros(PREALLOC, dtype=np.int32)
    n = 0

    def grid_idx(p):
        return ((p - lower) / cell).astype(np.int64)

    p0 = (lower + upper) * 0.5
    if not is_inside(p0[None, :])[0]:
        raise ValueError("Poisson sampler initial point (morph AABB center) is outside the morph surface.")
    pts[0] = p0
    g0 = grid_idx(p0)
    grid[g0[2], g0[1], g0[0]] = 0
    n = 1
    active = np.array([0], dtype=np.int64)

    two_pi = 2.0 * math.pi
    while active.size:
        if active.size > BATCH:
            sel = active[rng.choice(active.size, BATCH, replace=False)]
        else:
            sel = active
        base = pts[sel]
        b = base.shape[0]

        # shell candidates, same construction as the reference
        a1 = rng.random(b) * two_pi
        a2 = np.arccos(1.0 - 2.0 * rng.random(b))
        dist = d * (1.0 + rng.random(b))
        sa2 = np.sin(a2)
        cand = base + dist[:, None] * np.column_stack(
            [np.cos(a1) * sa2, np.sin(a1) * sa2, np.cos(a2)]
        )

        gi = grid_idx(cand)
        ok = (
            (gi >= 0).all(axis=1)
            & (gi < np.array([gx, gy, gz])).all(axis=1)
        )
        if ok.any():
            cand, gi, sel_ok = cand[ok], gi[ok], sel[ok]
            ok2 = is_inside(cand)
            cand, gi, sel_ok = cand[ok2], gi[ok2], sel_ok[ok2]
            # own cell must be empty (one point per cell invariant)
            own = grid[gi[:, 2], gi[:, 1], gi[:, 0]]
            keep = own == -1
            cand, gi, sel_ok = cand[keep], gi[keep], sel_ok[keep]
        else:
            cand = np.zeros((0, 3))
            sel_ok = np.zeros(0, dtype=np.int64)

        accepted_mask = np.zeros(sel.shape[0], dtype=bool)
        n_before = n
        if cand.shape[0]:
            # vectorized 5x5x5 neighbor pre-filter against the CURRENT grid
            mind2 = np.full(cand.shape[0], np.inf)
            for off in _offsets:
                gj = gi + off
                m = (
                    (gj[:, 0] >= 0) & (gj[:, 0] < gx)
                    & (gj[:, 1] >= 0) & (gj[:, 1] < gy)
                    & (gj[:, 2] >= 0) & (gj[:, 2] < gz)
                )
                if not m.any():
                    continue
                idx = grid[gj[m, 2], gj[m, 1], gj[m, 0]]
                hit = idx >= 0
                if not hit.any():
                    continue
                rows = np.nonzero(m)[0][hit]
                diff = pts[idx[hit]] - cand[rows]
                d2 = np.einsum("ij,ij->i", diff, diff)
                mind2[rows] = np.minimum(mind2[rows], d2)
            surv = mind2 >= d * d * (1.0 - 1e-12)
            cand, gi, sel_ok = cand[surv], gi[surv], sel_ok[surv]

            # sequential acceptance in random order with exact re-check (batch conflicts)
            for s in rng.permutation(cand.shape[0]):
                p = cand[s]
                g = gi[s]
                lo = np.maximum(g - 2, 0)
                hi = np.minimum(g + 3, [gx, gy, gz])
                block = grid[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]]
                idxs = np.unique(block)
                idxs = idxs[idxs != -1]
                if idxs.size:
                    diff = pts[idxs] - p
                    if (np.einsum("ij,ij->i", diff, diff) < d * d * (1.0 - 1e-12)).any():
                        continue
                pts[n] = p
                grid[g[2], g[1], g[0]] = n
                n += 1
                # mark the parent active point as having succeeded this round
                accepted_mask[np.nonzero(sel == sel_ok[s])[0][0]] = True

        # failure accounting: parents that produced a point stay active with the counter reset,
        # others accumulate consecutive failures and retire at MAX_TRIES (reference semantics);
        # points accepted this round join the active list
        fails[sel] += 1
        fails[sel[accepted_mask]] = 0
        survivors = active[fails[active] < MAX_TRIES]
        newly = np.arange(n_before, n, dtype=np.int64)
        active = np.concatenate([survivors, newly]) if newly.size else survivors

    return pts[:n].copy()


def install():
    """Monkeypatch DEMEntity._sample_poisson with the vectorized version (idempotent)."""
    from genesis.engine.entities.dem_entity import DEMEntity

    if getattr(DEMEntity, "_fast_poisson_installed", False):
        return
    DEMEntity._sample_poisson = staticmethod(_fast_sample_poisson)
    DEMEntity._fast_poisson_installed = True
    print("fast_poisson: vectorized poisson-disk sampler installed", flush=True)
