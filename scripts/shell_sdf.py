"""numpy mirror of TiltedCylinderShellBoundary._sdf (multiflow P2) for script-side metrics."""

import numpy as np


def shell_sdf_np(pos, O, A, r_in, length, t_wall, t_bottom, lip_round=0.0):
    """Exact shell SDF for points `pos` (n, 3): returns (d (n,), r (n,), s (n,)).

    d < 0 inside the (lip_round-dilated) shell solid; r/s are the cylindrical coordinates in
    the shell frame used for surface-feature classification (inner wall / outer wall / lip /
    underside).
    """
    r_out = r_in + t_wall
    O = np.asarray(O, dtype=float)
    A = np.asarray(A, dtype=float)
    rel = pos - O[None, :]
    s = rel @ A
    radial = rel - s[:, None] * A[None, :]
    r = np.linalg.norm(radial, axis=1)

    def box2(rr, ss, cr, cs, hr, hs):
        qr = np.abs(rr - cr) - hr
        qs = np.abs(ss - cs) - hs
        return np.sqrt(np.maximum(qr, 0.0) ** 2 + np.maximum(qs, 0.0) ** 2) + np.minimum(
            np.maximum(qr, qs), 0.0
        )

    sd_w = box2(r, s, 0.5 * (r_in + r_out), 0.5 * (length - t_bottom), 0.5 * t_wall,
                0.5 * (length + t_bottom))
    sd_f = box2(r, s, 0.5 * r_out, -0.5 * t_bottom, 0.5 * r_out, 0.5 * t_bottom)
    return np.minimum(sd_w, sd_f) - lip_round, r, s


def side_masks(pos, O, A, r_in, length, t_wall, t_bottom, r_p, lip_round=0.0):
    """Cavity-interior vs outside-alongside-the-wall masks (multiflow P2 n_leak metric).

    The per-substep impose sanitizes solid penetration every frame, so a leak cannot be seen
    as d < 0 at frame boundaries. A REAL tunnel shows as a side flip WITHOUT going over the
    rim: `inside_cav` (deep in the cavity, mid-height) vs `outside_wall` (outside the outer
    wall, alongside it — past-rim and below-bottom travelers excluded).
    """
    r_out = r_in + t_wall
    d, r, s = shell_sdf_np(pos, O, A, r_in, length, t_wall, t_bottom, lip_round)
    ps = 2.0 * r_p
    inside_cav = (r < r_in - r_p) & (s > ps) & (s < length - ps)
    outside_wall = (r > r_out + r_p) & (s > -t_bottom + ps) & (s < length - ps)
    return inside_cav, outside_wall
