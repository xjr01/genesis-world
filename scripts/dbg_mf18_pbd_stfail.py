"""Offline re-computation of the PBD spherical-illumination surface classifier.

Reads the state dumped by mf18_realscale_pbd.py's build-failure diagnostics
(videos/mf18_pbd_stfail_state.npz) and re-runs the screen/classify math in numpy for a
sample of deep-interior and near-surface coffee particles, to decide whether the 56-63%
surface classification is real (state pathology) or a solver-side artifact.
"""

import os
import sys

import numpy as np

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(WORKSPACE, "videos", "mf18_pbd_stfail_state.npz")

N_THETA, N_PHI = 8, 16  # Shibata resolution used by the solvers (verify against engine)
RING_FACTOR = 3.0


def illumination_fraction(pos_i, others, ps):
    """Blocked-rectangle difference array + prefix sum, mirroring the solver kernels."""
    blocked = np.zeros((N_THETA + 1, N_PHI + 1), dtype=np.int32)
    ring = RING_FACTOR * ps
    d = others - pos_i
    dist = np.linalg.norm(d, axis=1)
    m = (dist > 1e-9) & (dist < ring)
    d, dist = d[m], dist[m]
    for delta, r in zip(d, dist):
        block_radius = min(0.5 * ps, 0.5 * r)
        delta_angle = np.arcsin(min(block_radius / r, 1.0))
        theta = np.arccos(np.clip(delta[1] / r, -1.0, 1.0))
        phi = np.arctan2(delta[2], delta[0])
        st_t = int(np.floor(max(theta - delta_angle, 0.0) / (np.pi / N_THETA)))
        en_t = int(np.ceil(min(theta + delta_angle, np.pi) / (np.pi / N_THETA)))
        st_t, en_t = min(st_t, N_THETA - 1), min(en_t, N_THETA)
        # phi rectangles (wrap at +-pi handled by two rectangles; small angles here: single)
        for lo, hi in [(phi - delta_angle, phi + delta_angle)]:
            lo = (lo + np.pi) % (2 * np.pi) - np.pi
            hi = (hi + np.pi) % (2 * np.pi) - np.pi
            if lo < hi:
                st_p = min(int(np.floor((lo + np.pi) / (2 * np.pi / N_PHI))), N_PHI - 1)
                en_p = min(int(np.ceil((hi + np.pi) / (2 * np.pi / N_PHI))), N_PHI)
                rects = [(st_t, st_p, en_p), (en_t, st_p, en_p)]
                # difference array corners: +1 at (st_t, st_p), -1 at (st_t, en_p),
                # -1 at (en_t, st_p), +1 at (en_t, en_p)
                blocked[st_t, st_p] += 1
                blocked[st_t, en_p] -= 1
                blocked[en_t, st_p] -= 1
                blocked[en_t, en_p] += 1
    # prefix sum over the cell grid, then illuminated weight
    cell = np.cumsum(np.cumsum(blocked[:N_THETA, :N_PHI], axis=0), axis=1)
    t = np.arange(N_THETA)
    weight = np.sin(np.pi / N_THETA * (t + 0.5))[:, None] * np.ones((1, N_PHI))
    lit = float(weight[cell == 0].sum() / weight.sum())
    n_blockers = int(m.sum())
    return lit, n_blockers


def main():
    data = np.load(NPZ)
    pos, on_surf, n_c, ps = data["pos"], data["on_surface"], int(data["n_coffee"]), float(data["ps"])
    coffee = pos[:n_c]
    others_full = pos
    r_cyl = np.linalg.norm(coffee[:, :2], axis=1)
    z = coffee[:, 2]
    z_lo, z_hi = np.percentile(z, 5), np.percentile(z, 95)

    # deep interior: far from the side wall and from both caps
    deep = np.flatnonzero((r_cyl < 0.020) & (z > z_lo + 0.02) & (z < z_hi - 0.02))
    # near surface: close to the side wall
    surf = np.flatnonzero(r_cyl > 0.030)
    rng = np.random.default_rng(0)
    deep_pick = rng.choice(deep, size=min(12, deep.size), replace=False)
    surf_pick = rng.choice(surf, size=min(12, surf.size), replace=False)

    print(f"state: {pos.shape[0]} particles, ps={ps}, engine on_surface={on_surf.mean():.1%}")
    for name, idx in [("deep-interior", deep_pick), ("near-surface", surf_pick)]:
        lits, nbs = [], []
        for i in idx:
            lit, nb = illumination_fraction(coffee[i], others_full, ps)
            lits.append(lit)
            nbs.append(nb)
        lits = np.array(lits)
        print(f"{name:13s} n={len(idx)}  blockers/particle avg={np.mean(nbs):.0f}  "
              f"lit fraction: min={lits.min():.2f} mean={lits.mean():.2f} max={lits.max():.2f}  "
              f"marked-surface(engine)={on_surf[:n_c][idx].mean():.0%}")
    print("threshold 1/9 -> marked surface iff lit >= 0.111; 1/3 iff lit >= 0.333")


if __name__ == "__main__":
    main()
