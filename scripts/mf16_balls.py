#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mf16_balls.py -- pure-numpy ANALYTIC boundary-ball sampler for the mf16 milk->coffee demo.

Generates Akinci frozen-fluid boundary balls (spacing == particle spacing ps, 2 layers,
inset delta = 0.5*ps) for:

  mug  (static,  world frame   : origin = OUTER bottom center, +z up)
       -> videos/mf16_balls_mug.npz
  jug  (kinematic, jug-local frame : origin = jug outer bottom center, +z along the axis;
         the engine rigidly transforms these every frame via set_boundary_ball_pose)
       -> videos/mf16_balls_jug.npz

GEOMETRY SOURCE OF TRUTH -- this script imports the primitive constructors/default dictionaries
from gen_mug.py and gen_jug.py, writes both visible OBJs, and samples the boundary balls in the
same run. Every cylindrical ring is generated analytically with cos/sin on the TRUE circle.

  assets/gen_mug.py DEFAULTS (gen_mug.py:32-47)
      r_in=0.150 h_in=0.30 t_wall=0.016 t_bottom=0.016 n_seg=96
      handle: r_tube=0.013 z_lo=0.10 z_hi=0.22 handle_clear=0.055 m=72 k=16 psi_e_deg=70
  assets/gen_jug.py DEFAULTS (gen_jug.py:35-51) + M_SECTIONS/K_SEG/PSI_E_DEG (gen_jug.py:53)
      r_in=0.125 h_in=0.30 t_wall=0.016 t_bottom=0.016 n_seg=96
      handle: r_tube=0.011 z_lo=0.09 z_hi=0.21 handle_clear=0.050 m=72 k=16 psi_e_deg=70

Handle centerline is copied verbatim from gen_mug.build_handle (gen_mug.py:99-133):
    rho_end = 0.5*(r_in+r_out)                        # gen_mug.py:110  (cap center, mid-wall)
    b       = 0.5*(z_hi-z_lo)/sin(psi_e)              # gen_mug.py:111
    z_c     = 0.5*(z_hi+z_lo)                         # gen_mug.py:112
    a       = 0.998*(clear+r_out-r_tube-rho_end)/(1+cos(psi_e))   # gen_mug.py:114-116
    e       = rho_end + a*cos(psi_e)                  # gen_mug.py:117
    P(psi)  = (side*(e - a*cos psi), 0, z_c + b*sin psi),  psi in [psi_e, 2pi-psi_e]  # :122-123
    T       = dP/d|psi| ; N = (Tz, 0, -Tx) ; B = (0,1,0)                          # :125-127
    surface = P + r_tube*(cos th * N + sin th * B)                                # :130-133
The sampler re-derives a, b, e from the SAME formulas and samples the tube surface at
radii (r_tube - delta) and (r_tube - delta - ps); consistency with the shipped mesh is
proven in check_obj() for BOTH handles. The jug's shallow cos^2 beak is also applied to
the visible outer/rim surfaces and their collision samples from the same parameter dict.

FACES SAMPLED (per body):
  1 inner wall : r = R_IN+delta, R_IN+delta+ps          ; z = t_bottom+delta .. Z_RIM-delta (step ps, top ring truncated <= Z_RIM-delta)
  2 outer wall : r = R_OUT-delta, R_OUT-delta-ps        ; z = Z_RIM-delta .. 0 (step ps),
                 with the jug beak's shared cos^2 ruled-surface deformation
  3 rim top    : z = Z_RIM-delta, Z_RIM-delta-ps        ; r = R_IN .. R_OUT (step ps),
                 with the same shared beak interpolation for the jug
  4 inner floor: z = t_bottom-delta, t_bottom-delta-ps  ; r = 0 .. R_IN (step ps, incl. the r=0 single point)
  5 handle     : sections along the arc every ps        ; tube-surface radii r_tube-delta, r_tube-delta-ps
All neighboring rings/layers/sections are angularly staggered by half a step (hex-ish);
points closer than 0.2*ps are merged.

OUTPUT: npz with `pos_local (N,3) float32` (PBDOptions.boundary_ball_sets contract,
PLAN.md:398) + `meta` (JSON string: ps / inset / layers / counts_by_face / handle params).
Plus videos/mf16_balls_top.png, mug/jug front/side validation previews, and a self-check
report with hard assertions (exact shared-source OBJ reload, handle/spout presence, visible
surface collision coverage, true circles, budgets, no NaN). There is no warning-only path.

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 "$PY" mf16_balls.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
ASSETS = os.path.join(WORKSPACE, "assets")
VIDEOS = os.path.join(WORKSPACE, "videos")
if ASSETS not in sys.path:
    sys.path.insert(0, ASSETS)

# The visible meshes and the collision samples are built in this one process from the same
# primitive constructors and parameter dictionaries.  Importing these modules is deliberately
# stricter than copying their constants: a change to either visible-asset generator is picked up
# here immediately and the exact reload assertions below fail if the written OBJ diverges.
import gen_jug as jug_geometry  # noqa: E402
import gen_mug as mug_geometry  # noqa: E402

# ---------------------------------------------------------------- sampling spec (task F)
PS = 0.008           # ball spacing == fluid particle spacing [m]
INSET = 0.5 * PS     # delta: offset of layer 1 from the visible surface
LAYERS = 2           # layer 2 sits another 1*ps deeper
DEDUP_FRAC = 0.2     # merge points closer than 0.2*ps
CIRCLE_TOL = 1e-9    # assertion (1): radius spread of any sampled ring
N_SEG = int(mug_geometry.DEFAULTS["n_seg"])
H_IN = float(mug_geometry.DEFAULTS["h_in"])
PSI_E_DEG = float(mug_geometry.DEFAULTS["psi_e_deg"])
MUG_HANDLE_SIDE = -1.0
JUG_HANDLE_SIDE = float(jug_geometry.DEFAULTS["handle_side"])
JUG_SPOUT_CENTER_DEG = float(jug_geometry.DEFAULTS["spout_center_deg"])

assert N_SEG == int(jug_geometry.DEFAULTS["n_seg"])
assert H_IN == float(jug_geometry.DEFAULTS["h_in"])
assert PSI_E_DEG == float(jug_geometry.PSI_E_DEG)

# ------------------------------------------------- geometry, same family as the generators
BODIES = {
    "mug": dict(
        r_in=float(mug_geometry.DEFAULTS["r_in"]),
        r_out=float(mug_geometry.DEFAULTS["r_in"] + mug_geometry.DEFAULTS["t_wall"]),
        h_in=float(mug_geometry.DEFAULTS["h_in"]),
        t_wall=float(mug_geometry.DEFAULTS["t_wall"]),
        t_bottom=float(mug_geometry.DEFAULTS["t_bottom"]),
        n_seg=int(mug_geometry.DEFAULTS["n_seg"]),
        handle={k: float(mug_geometry.DEFAULTS[k]) for k in ("r_tube", "z_lo", "z_hi", "handle_clear")},
        handle_m=int(mug_geometry.DEFAULTS["m"]),
        handle_k=int(mug_geometry.DEFAULTS["k"]),
        psi_e_deg=float(mug_geometry.DEFAULTS["psi_e_deg"]),
        handle_side=MUG_HANDLE_SIDE,
        spout=None,
        obj=mug_geometry.DEFAULTS["out"],
    ),
    "jug": dict(
        r_in=float(jug_geometry.DEFAULTS["r_in"]),
        r_out=float(jug_geometry.DEFAULTS["r_in"] + jug_geometry.DEFAULTS["t_wall"]),
        h_in=float(jug_geometry.DEFAULTS["h_in"]),
        t_wall=float(jug_geometry.DEFAULTS["t_wall"]),
        t_bottom=float(jug_geometry.DEFAULTS["t_bottom"]),
        n_seg=int(jug_geometry.DEFAULTS["n_seg"]),
        handle={k: float(jug_geometry.DEFAULTS[k]) for k in ("r_tube", "z_lo", "z_hi", "handle_clear")},
        handle_m=int(jug_geometry.M_SECTIONS),
        handle_k=int(jug_geometry.K_SEG),
        psi_e_deg=float(jug_geometry.PSI_E_DEG),
        handle_side=JUG_HANDLE_SIDE,
        spout={
            **{k: float(jug_geometry.DEFAULTS[k]) for k in ("spout_dr", "spout_dz", "spout_half_deg")},
            "center_deg": JUG_SPOUT_CENTER_DEG,
        },
        obj=jug_geometry.DEFAULTS["out"],
    ),
}


# --------------------------------------------------------------------------- primitives
def n_theta(r: float, ps: float) -> int:
    """Azimuthal sample count of a circle of radius r so the chord step ~= ps."""
    return max(3, int(round(2.0 * math.pi * r / ps)))


def ring(r: float, z: float, n_th: int, phase: float) -> np.ndarray:
    """TRUE circle (analytic cos/sin) of radius r at height z, n_th points."""
    th = phase + 2.0 * np.pi * np.arange(n_th, dtype=np.float64) / n_th
    return np.stack([r * np.cos(th), r * np.sin(th), np.full(n_th, z)], axis=1)


def z_grid(z0: float, z1: float, ps: float) -> np.ndarray:
    """Exact multiples of ps from z0 toward z1 (both ends included when they are exact)."""
    n = int(math.floor(abs(z1 - z0) / ps + 1e-9))
    sgn = 1.0 if z1 >= z0 else -1.0
    return np.round(z0 + sgn * ps * np.arange(n + 1, dtype=np.float64), 9)


def stagger(n_th: int, parity: int) -> float:
    """Half-step angular offset; `parity` differs by 1 between adjacent rings/layers."""
    return 0.5 * (2.0 * np.pi / n_th) * (parity % 2)


# ------------------------------------------------------------------------------- builder
class Body:
    """Accumulates analytic ball chunks for one body, with per-face/per-layer bookkeeping."""

    def __init__(self, name: str, ps: float = PS, inset: float = INSET, layers: int = LAYERS):
        self.name, self.ps, self.inset, self.layers = name, ps, inset, layers
        self.chunks = []   # (face, layer, pts(N,3) float64, ring_meta | None)
        self.rings = []    # filled in finalize(): (face, layer, center|None, radius, a0, a1)

    # -- low level -------------------------------------------------------------------
    def _add(self, face, layer, pts, ring_meta=None):
        self.chunks.append((face, layer, np.asarray(pts, dtype=np.float64), ring_meta))

    def _axis_ring(self, face, layer, r, z, parity):
        n_th = n_theta(r, self.ps)
        pts = ring(r, z, n_th, stagger(n_th, parity))
        self._add(face, layer, pts, ring_meta=(face, layer, None, r))
        return len(pts)

    # -- faces 1-4 -------------------------------------------------------------------
    def inner_wall(self, r_in, t_bottom, z_rim):
        """Face 1: r = R_IN+delta (+ps); z = t_bottom+delta .. Z_RIM-delta (step ps)."""
        zs = z_grid(t_bottom + self.inset, z_rim - self.inset, self.ps)
        for L in range(self.layers):
            r = r_in + self.inset + L * self.ps
            for k, z in enumerate(zs):
                self._axis_ring("inner_wall", L, r, z, parity=L + k)
        return len(zs), float(zs[-1])

    def outer_wall(self, r_out, z_rim, spout=None):
        """Face 2, including the jug's shallow ruled-surface beak when requested.

        `gen_jug.make_spout_fn` moves the visible outer-top ring and leaves the outer-bottom
        ring fixed.  Linear interpolation in nominal height therefore samples the same ruled
        outer wall that the OBJ triangles represent.  The collision rows are offset radially
        into the solid by `delta + layer*ps`.
        """
        zs = z_grid(z_rim - self.inset, 0.0, self.ps)
        for L in range(self.layers):
            r = r_out - self.inset - L * self.ps
            for k, z in enumerate(zs):
                if spout is None:
                    self._axis_ring("outer_wall", L, r, z, parity=L + k)
                else:
                    n_th = n_theta(r, self.ps)
                    th = stagger(n_th, L + k) + 2.0 * np.pi * np.arange(n_th) / n_th
                    center = math.radians(spout["center_deg"])
                    d = np.arctan2(np.sin(th - center), np.cos(th - center))
                    half = math.radians(spout["spout_half_deg"])
                    w = np.where(np.abs(d) <= half, np.cos(0.5 * np.pi * d / half) ** 2, 0.0)
                    q = float(z / z_rim)
                    rr = r + spout["spout_dr"] * w * q
                    zz = z + spout["spout_dz"] * w * q
                    pts = np.stack([rr * np.cos(th), rr * np.sin(th), zz], axis=1)
                    self._add("outer_wall", L, pts)
        return len(zs), float(zs[-1])

    def rim_face(self, r_in, r_out, z_rim, spout=None):
        """Face 3, interpolated between the circular inner lip and the beaked outer lip.

        This is the same parameterization as the visible rim triangles: deformation is zero at
        the cavity lip and reaches the complete cos^2 beak at the outer lip.
        """
        radii, j = [], 0
        while r_in + j * self.ps <= r_out + 1e-9:
            radii.append(round(r_in + j * self.ps, 9))
            j += 1
        if radii[-1] < r_out - 1e-9:
            radii.append(float(r_out))
        for L in range(self.layers):
            z = z_rim - self.inset - L * self.ps
            for j, r in enumerate(radii):
                if spout is None:
                    self._axis_ring("rim_face", L, r, z, parity=L + j)
                else:
                    n_th = n_theta(r, self.ps)
                    th = stagger(n_th, L + j) + 2.0 * np.pi * np.arange(n_th) / n_th
                    center = math.radians(spout["center_deg"])
                    d = np.arctan2(np.sin(th - center), np.cos(th - center))
                    half = math.radians(spout["spout_half_deg"])
                    w = np.where(np.abs(d) <= half, np.cos(0.5 * np.pi * d / half) ** 2, 0.0)
                    q = float((r - r_in) / (r_out - r_in))
                    rr = r + spout["spout_dr"] * w * q
                    zz = z + spout["spout_dz"] * w * q
                    pts = np.stack([rr * np.cos(th), rr * np.sin(th), zz], axis=1)
                    self._add("rim_face", L, pts)
        return radii

    def floor_face(self, r_in, t_bottom, edge_ring=True):
        """Face 4: z = t_bottom-delta (-ps); rings r = 0 .. R_IN (step ps, incl. r=0 point).

        Inset goes INTO the solid (below the visible inner floor), same convention
        as the walls — main-agent correction 2026-09-13 of the original +delta spec
        (+delta left the liquid floating ~0.85*ps above the visible floor).
        edge_ring: when the ps grid cannot reach R_IN exactly and the leftover is
        >= 0.5*ps, one extra ring at exactly R_IN closes the floor/wall corner
        (documented deviation, see report).
        """
        radii = [0.0]
        k = 1
        while k * self.ps <= r_in + 1e-9:
            radii.append(round(k * self.ps, 9))
            k += 1
        leftover = round(r_in - radii[-1], 12)
        closing = None
        if edge_ring and leftover >= 0.5 * self.ps - 1e-12:
            closing = round(r_in, 9)
            radii.append(closing)
        for L in range(self.layers):
            z = t_bottom - self.inset - L * self.ps
            for j, r in enumerate(radii):
                if r == 0.0:  # single center point (avoids coincident-point NaN)
                    self._add("floor_face", L, np.array([[0.0, 0.0, z]]),
                              ring_meta=("floor_face", L, None, 0.0))
                else:
                    self._axis_ring("floor_face", L, r, z, parity=L + j)
        return radii, leftover, closing

    # -- face 5: handle --------------------------------------------------------------
    def handle(self, r_in, r_out, z_rim, hd, psi_e_deg=PSI_E_DEG, side=MUG_HANDLE_SIDE, dense=200001,
               force_two_layers=False):
        """C-tube handle: same elliptical-arc centerline as gen_mug.build_handle.

        Sections every ps along the arc; per section, n_th points on the tube surface at
        radii (r_tube-delta) and (r_tube-delta-ps). Buried end caps are sampled too.
        Layer 2 is dropped when its radius degenerates (see below / report deviation D2).
        """
        psi_e = math.radians(psi_e_deg)
        r_tube = hd["r_tube"]
        rho_end = 0.5 * (r_in + r_out)                                   # gen_mug.py:110
        b = 0.5 * (hd["z_hi"] - hd["z_lo"]) / math.sin(psi_e)            # gen_mug.py:111
        z_c = 0.5 * (hd["z_hi"] + hd["z_lo"])                            # gen_mug.py:112
        budget = hd["handle_clear"] + r_out - r_tube - rho_end           # gen_mug.py:114
        assert budget > 0.0, "no room for handle bulge"
        a = 0.998 * budget / (1.0 + math.cos(psi_e))                     # gen_mug.py:116
        e = rho_end + a * math.cos(psi_e)                                # gen_mug.py:117
        farthest = e + a + r_tube                                        # gen_mug.py:118
        assert farthest <= r_out + hd["handle_clear"] + 1e-9, "handle exceeds clearance"

        # arc-length parametrization of P(psi) (dense numeric quadrature -> uniform s steps)
        psi = np.linspace(psi_e, 2.0 * np.pi - psi_e, dense)
        speed = np.hypot(side * a * np.sin(psi), b * np.cos(psi))        # |dP/dpsi|
        s = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1])
                                             * np.diff(psi))])
        arc_len = float(s[-1])
        n_sec = int(math.floor(arc_len / self.ps + 1e-9)) + 1
        s_q = self.ps * np.arange(n_sec, dtype=np.float64)               # trailing (<ps) gap is
        psi_q = np.interp(s_q, s, psi)                                   # inside the buried cap

        cx = side * (e - a * np.cos(psi_q))                              # gen_mug.py:123
        cz = z_c + b * np.sin(psi_q)
        tx = side * a * np.sin(psi_q)                                    # gen_mug.py:124
        tz = b * np.cos(psi_q)
        norm = np.hypot(tx, tz)
        tx, tz = tx / norm, tz / norm                                    # T (gen_mug.py:125)
        nx, nz = tz, -tx                                                 # N (gen_mug.py:126)

        n_th = n_theta(r_tube - self.inset, self.ps)                     # spec: from layer-1 radius
        th0 = 2.0 * np.pi * np.arange(n_th, dtype=np.float64) / n_th
        rr2 = r_tube - self.inset - self.ps                              # layer-2 tube radius
        # Layer 2 is only a real circle while it stays outside the tube axis: for
        # r_tube = 0.013/0.011 and delta+ps = 0.012 it collapses to r=0.001 / -0.001,
        # where the generic 2-layer rule degenerates (and item-7 dedup would shred it).
        use_l2 = self.layers > 1 and (rr2 >= 0.5 * self.ps or force_two_layers)
        n_layers = 2 if use_l2 else min(1, self.layers)
        for i in range(n_sec):
            for L in range(n_layers):
                rr = r_tube - self.inset - L * self.ps
                th = stagger(n_th, i + L) + th0
                pts = np.stack([cx[i] + rr * np.cos(th) * nx[i],
                                rr * np.sin(th),                         # B = (0,1,0)
                                cz[i] + rr * np.cos(th) * nz[i]], axis=1)
                center = np.array([cx[i], 0.0, cz[i]])
                self._add("handle", L, pts, ring_meta=("handle", L, center, rr))
        return dict(a=a, b=b, e=e, rho_end=rho_end, z_c=z_c, r_tube=r_tube,
                    arc_len=arc_len, sweep_deg=math.degrees(2 * math.pi - 2 * psi_e),
                    n_sections=n_sec, n_theta=n_th, n_layers=n_layers,
                    layer2_radius=(rr2 if use_l2 else None),
                    farthest_radius=farthest,
                    farthest_limit=r_out + hd["handle_clear"],
                    attach_z=(z_c - b * math.sin(psi_e), z_c + b * math.sin(psi_e)),
                    rim_clearance=z_rim - (z_c + b) - r_tube)

    # -- assembly --------------------------------------------------------------------
    def finalize(self, dedup_frac=DEDUP_FRAC):
        pts64 = (np.vstack([c[2] for c in self.chunks]) if self.chunks
                 else np.zeros((0, 3), dtype=np.float64))
        # Keep the sampler's native provenance one-for-one with the coordinates.  Spatial
        # inference after generation is ambiguous at the shared wall radii and at face seams;
        # these arrays instead follow the exact chunk order and the exact de-duplication mask.
        face_pre = (np.concatenate([np.full(len(c[2]), c[0], dtype="<U16") for c in self.chunks])
                    if self.chunks else np.zeros(0, dtype="<U16"))
        layer_pre = (np.concatenate([np.full(len(c[2]), c[1], dtype=np.int8) for c in self.chunks])
                     if self.chunks else np.zeros(0, dtype=np.int8))
        off, self.rings = 0, []
        for face, layer, p, meta in self.chunks:
            if meta is not None:
                self.rings.append((meta[0], meta[1], meta[2], meta[3], off, off + len(p)))
            off += len(p)

        # assertion (1): every sampled ring is an exact circle (float64, pre-dedup)
        worst, worst_face = 0.0, ""
        for face, layer, center, radius, a0, a1 in self.rings:
            q = pts64[a0:a1]
            rr = (np.hypot(q[:, 0], q[:, 1]) if center is None
                  else np.linalg.norm(q - center, axis=1))
            spread = float(rr.max() - rr.min())
            if spread > worst:
                worst, worst_face = spread, f"{face}/L{layer}/r={radius:.6f}"
        assert worst < CIRCLE_TOL, f"ring is not a true circle: {worst_face} spread={worst:.3e}"

        n_pre = len(pts64)
        keep = dedup_near(pts64, dedup_frac * self.ps) if dedup_frac > 0 else \
            np.ones(n_pre, dtype=bool)
        pts64 = pts64[keep]
        face_label = face_pre[keep]
        layer_label = layer_pre[keep]

        # The mug's two shell directions sample the same two barrel radii: inner L0 and
        # outer L1 share r=.154, while inner L1 and outer L0 share r=.162.  Their z-staggered
        # samples form one surface quadrature, not two full-mass walls, so each copy carries
        # half weight.  Other faces, and the jug asset, retain the historical unit weight.
        mass_weight = np.ones(len(pts64), dtype=np.float32)
        if self.name == "mug":
            mass_weight[np.isin(face_label, ("inner_wall", "outer_wall"))] = 0.5

        # per-face / per-layer post-dedup counts (+ where the merges happened)
        kept_set = np.zeros(n_pre, dtype=bool)
        kept_set[np.flatnonzero(keep)] = True
        counts_post, merged_by, o0 = {}, {}, 0
        for face, layer, p, meta in self.chunks:
            counts_post.setdefault(face, [0] * self.layers)
            counts_post[face][layer] += int(kept_set[o0:o0 + len(p)].sum())
            merged_by[face] = merged_by.get(face, 0) + int((~kept_set[o0:o0 + len(p)]).sum())
            o0 += len(p)
        self.merged_by_face = merged_by

        assert np.isfinite(pts64).all(), "NaN/Inf in generated balls"
        assert np.isfinite(mass_weight).all() and np.all(mass_weight > 0.0)
        pos32 = pts64.astype(np.float32)
        return dict(pos=pos32, pos64=pts64, n_pre=n_pre, n_post=len(pts64),
                    face=face_label, layer=layer_label, mass_weight=mass_weight,
                    counts_post=counts_post, merged_by_face=self.merged_by_face,
                    worst_circle=worst, worst_circle_where=worst_face, rings=self.rings)


def dedup_near(points: np.ndarray, tol: float) -> np.ndarray:
    """Greedy near-duplicate removal: drop a point if a kept point is within `tol`.

    Voxel hash (cell = tol) + 27-neighbour scan; deterministic, pure numpy.
    """
    cell = tol
    key = np.floor(points / cell).astype(np.int64)
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    keep = np.zeros(len(points), dtype=bool)
    grid = {}
    tol2 = tol * tol
    for idx in order:
        kx, ky, kz = int(key[idx, 0]), int(key[idx, 1]), int(key[idx, 2])
        dup = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    lst = grid.get((kx + dx, ky + dy, kz + dz))
                    if lst:
                        d = points[lst] - points[idx]
                        if np.any(np.einsum("ij,ij->i", d, d) < tol2):
                            dup = True
                            break
                if dup:
                    break
            if dup:
                break
        if not dup:
            keep[idx] = True
            grid.setdefault((kx, ky, kz), []).append(idx)
    return keep


# ------------------------------------------------------------ visible OBJ + strict consistency
def load_obj(path: str):
    verts, faces = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def write_visible_obj(name: str, cfg: dict):
    """Generate the visible body and handle from the exact dictionaries used by the balls."""
    spout_fn = None
    if cfg["spout"] is not None:
        spout_fn = jug_geometry.make_spout_fn(
            cfg["r_out"], cfg["t_bottom"] + cfg["h_in"],
            cfg["spout"]["spout_dr"], cfg["spout"]["spout_dz"], cfg["spout"]["spout_half_deg"],
            center_deg=cfg["spout"]["center_deg"],
        )
    shell = mug_geometry.build_shell(
        cfg["r_in"], cfg["h_in"], cfg["t_wall"], cfg["t_bottom"], cfg["n_seg"],
        b_ring_fn=spout_fn,
    )
    handle, handle_info = mug_geometry.build_handle(
        cfg["r_in"], cfg["r_out"], cfg["t_bottom"] + cfg["h_in"],
        cfg["handle"]["r_tube"], cfg["handle"]["z_lo"], cfg["handle"]["z_hi"],
        cfg["handle"]["handle_clear"], m=cfg["handle_m"], k=cfg["handle_k"],
        psi_e_deg=cfg["psi_e_deg"], side=cfg["handle_side"],
    )
    assert shell.is_watertight and shell.is_winding_consistent and shell.is_volume
    assert handle.is_watertight and handle.is_winding_consistent and handle.is_volume
    verts, faces = mug_geometry.merge_meshes([shell, handle])
    mesh = mug_geometry.trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    os.makedirs(os.path.dirname(os.path.abspath(cfg["obj"])), exist_ok=True)
    mesh.export(cfg["obj"])
    # These previews are validation artifacts, not simulation renders.
    mug_geometry.ortho_png(verts, faces, os.path.splitext(cfg["obj"])[0] + "_front.png", axis="y")
    mug_geometry.ortho_png(verts, faces, os.path.splitext(cfg["obj"])[0] + "_side.png", axis="x")
    return dict(
        verts=np.asarray(verts, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64),
        shell_nverts=len(shell.vertices), shell_nfaces=len(shell.faces), handle_info=handle_info,
    )


def nearest_distance(queries: np.ndarray, sources: np.ndarray, chunk=128) -> np.ndarray:
    out = np.empty(len(queries), dtype=np.float64)
    for i in range(0, len(queries), chunk):
        q = queries[i:i + chunk]
        d2 = ((q[:, None, :] - sources[None, :, :]) ** 2).sum(axis=2)
        out[i:i + len(q)] = np.sqrt(d2.min(axis=1))
    return out


def check_obj(name: str, cfg: dict, ref: dict, balls: np.ndarray, ps: float):
    """Require exact generated-source identity plus geometric collision coverage."""
    verts, faces = load_obj(cfg["obj"])
    assert verts.shape == ref["verts"].shape, f"{name}: OBJ vertex count/source mismatch"
    assert faces.shape == ref["faces"].shape, f"{name}: OBJ face count/source mismatch"
    assert np.allclose(verts, ref["verts"], rtol=0.0, atol=1e-8), f"{name}: OBJ vertices diverge from shared source"
    assert np.array_equal(faces, ref["faces"]), f"{name}: OBJ faces diverge from shared source"

    n, z_rim = cfg["n_seg"], cfg["t_bottom"] + cfg["h_in"]
    shell_n = 4 * n + 2
    assert ref["shell_nverts"] == shell_n
    assert len(verts) == shell_n + cfg["handle_m"] * cfg["handle_k"] + 2
    r = np.linalg.norm(verts[:, :2], axis=1)
    B = verts[n:2 * n]
    C = verts[2 * n:3 * n]
    assert np.max(np.abs(np.linalg.norm(C[:, :2], axis=1) - cfg["r_in"])) < 1e-8
    assert np.max(np.abs(C[:, 2] - z_rim)) < 1e-8
    sag = cfg["r_in"] * (1.0 - math.cos(math.pi / n))
    assert sag < 1e-4

    if cfg["spout"] is None:
        assert np.max(np.abs(np.linalg.norm(B[:, :2], axis=1) - cfg["r_out"])) < 1e-8
        assert np.max(np.abs(B[:, 2] - z_rim)) < 1e-8
        spout_text = "none (mug)"
    else:
        sp = cfg["spout"]
        b_r = np.linalg.norm(B[:, :2], axis=1)
        tip = B[int(np.argmax(b_r))]
        assert float(b_r.max()) >= cfg["r_out"] + 0.99 * sp["spout_dr"]
        assert float(B[:, 2].min()) <= z_rim + 0.99 * sp["spout_dz"]
        assert float(B[:, 2].min()) >= z_rim - 0.002 - 1e-12
        assert tip[0] < -(cfg["r_out"] + 0.99 * sp["spout_dr"]), (
            f"{name}: spout tip must be local -x, got {tip}"
        )
        pos_x = B[B[:, 0] > 0.0]
        assert float(np.linalg.norm(pos_x[:, :2], axis=1).max()) <= cfg["r_out"] + 1e-8, (
            f"{name}: forbidden spout protrusion remains on local +x"
        )
        spout_text = f"tip_r={np.linalg.norm(B[:, :2], axis=1).max():.5f}, tip_z={B[:, 2].min():.5f}"

    # Handle vertices are ordered exactly as build_handle: M rings x K, then two cap centers.
    hv = verts[shell_n:]
    surf = hv[:cfg["handle_m"] * cfg["handle_k"]].reshape(cfg["handle_m"], cfg["handle_k"], 3)
    caps = hv[-2:]
    hd = cfg["handle"]
    psi_e = math.radians(cfg["psi_e_deg"])
    psi = np.linspace(psi_e, 2.0 * math.pi - psi_e, cfg["handle_m"])
    rho_end = 0.5 * (cfg["r_in"] + cfg["r_out"])
    b = 0.5 * (hd["z_hi"] - hd["z_lo"]) / math.sin(psi_e)
    z_c = 0.5 * (hd["z_hi"] + hd["z_lo"])
    a = 0.998 * (hd["handle_clear"] + cfg["r_out"] - hd["r_tube"] - rho_end) / (1.0 + math.cos(psi_e))
    e = rho_end + a * math.cos(psi_e)
    centers = np.stack([cfg["handle_side"] * (e - a * np.cos(psi)), np.zeros_like(psi), z_c + b * np.sin(psi)], axis=1)
    tube_dev = float(np.max(np.abs(np.linalg.norm(surf - centers[:, None, :], axis=2) - hd["r_tube"])))
    cap_dev = float(max(np.linalg.norm(caps[0] - centers[0]), np.linalg.norm(caps[1] - centers[-1])))
    assert tube_dev < 2e-8 and cap_dev < 2e-8
    if name == "jug":
        assert float(centers[:, 0].min()) > 0.0, "jug handle centreline must be local +x"

    # A visible surface vertex must lie within one fluid/ball center separation (ps) of a
    # boundary center.  This catches exactly the old failure: a visible handle/spout without
    # collision samples, or collision samples for geometry absent from the OBJ.
    coverage = nearest_distance(verts, balls)
    max_coverage = float(coverage.max())
    assert max_coverage <= ps * 1.01, (
        f"{name}: visible geometry is not covered by boundary balls; max distance "
        f"{max_coverage:.6f} > {ps * 1.01:.6f}"
    )
    print(
        f"  [{name}.obj] exact shared-source reload: verts={len(verts)} faces={len(faces)}; "
        f"inner lip sagitta={sag:.3e}; handle tube dev={tube_dev:.3e}; {spout_text}; "
        f"max visible->ball distance={max_coverage:.6f} <= {ps * 1.01:.6f}"
    )


# --------------------------------------------------------------------------- top view png
def top_view_png(sets, path, px_per_m=1400.0, pad=0.05):
    """sets: list of (label, pos(N,3) float64 in local frame, xy offset, rgb)."""
    from PIL import Image, ImageDraw

    placed = []
    for label, pos, off, rgb in sets:
        p = pos[:, :2] + np.asarray(off, dtype=np.float64)
        placed.append((label, p, np.asarray(off, dtype=np.float64), rgb))
    allp = np.vstack([p for _, p, _, _ in placed])
    lo = allp.min(axis=0) - pad
    hi = allp.max(axis=0) + pad
    w = int(round((hi[0] - lo[0]) * px_per_m)) + 1
    h = int(round((hi[1] - lo[1]) * px_per_m)) + 1
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    im = Image.fromarray(img)
    drw = ImageDraw.Draw(im)

    def to_px(xy):
        return (float((xy[0] - lo[0]) * px_per_m), float((hi[1] - xy[1]) * px_per_m))

    # faint true-circle references (r_in / r_out / handle reach) for eyeballing roundness
    for label, p, off, rgb in placed:
        for R, col in ((REF[label]["r_in"], (185, 185, 185)),
                       (REF[label]["r_out"], (185, 185, 185)),
                       (REF[label]["reach"], (215, 215, 215))):
            c = to_px(off)
            rr = R * px_per_m
            drw.ellipse([c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr], outline=col, width=1)
    arr = np.asarray(im).copy()
    for label, p, off, rgb in placed:
        ij = np.stack([np.round((p[:, 0] - lo[0]) * px_per_m).astype(int),
                       np.round((hi[1] - p[:, 1]) * px_per_m).astype(int)], axis=1)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
        arr[ij[ok, 1], ij[ok, 0]] = rgb
        print(f"  [png] {label}: {int(ok.sum())} px points (1 px each), color={rgb}")
    Image.fromarray(arr).save(path)
    return path


REF = {}


# ----------------------------------------------------------------------------------- main
def build_body(name, cfg, ps, inset, layers, with_handle, edge_ring=True,
               handle_two_layers=False):
    z_rim = cfg["t_bottom"] + cfg["h_in"]               # = 0.316 for both bodies
    b = Body(name, ps=ps, inset=inset, layers=layers)
    nz_in, z_in_top = b.inner_wall(cfg["r_in"], cfg["t_bottom"], z_rim)
    nz_out, z_out_bot = b.outer_wall(cfg["r_out"], z_rim, spout=cfg["spout"])
    rim_radii = b.rim_face(cfg["r_in"], cfg["r_out"], z_rim, spout=cfg["spout"])
    floor_radii, leftover, closing = b.floor_face(cfg["r_in"], cfg["t_bottom"],
                                                  edge_ring=edge_ring)
    hinfo = None
    if with_handle:
        hinfo = b.handle(cfg["r_in"], cfg["r_out"], z_rim, cfg["handle"],
                         psi_e_deg=cfg["psi_e_deg"], side=cfg["handle_side"],
                         force_two_layers=handle_two_layers)
    out = b.finalize()
    out.update(name=name, cfg=cfg, z_rim=z_rim, nz_in=nz_in, z_in_top=z_in_top,
               nz_out=nz_out, z_out_bot=z_out_bot, rim_radii=rim_radii,
               floor_radii=floor_radii, floor_leftover=leftover, floor_closing=closing,
               handle_info=hinfo)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=VIDEOS)
    ap.add_argument("--ps", type=float, default=PS)
    ap.add_argument("--inset", type=float, default=INSET)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--no-floor-edge-ring", action="store_true",
                    help="disable the corner-closing floor ring at r == R_IN")
    ap.add_argument("--handle-two-layers", action="store_true",
                    help="force the spec-literal 2nd handle layer even when its radius "
                         "degenerates (< 0.5*ps)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ps, inset, layers = args.ps, args.inset, args.layers
    z_rim = BODIES["mug"]["t_bottom"] + BODIES["mug"]["h_in"]
    print(f"== mf16_balls : analytic boundary-ball sampler ==")
    print(f"ps={ps}  inset(delta)={inset}  layers={layers}  z_rim={z_rim}  n_seg={N_SEG}  "
          f"psi_e={PSI_E_DEG} mug_handle_side={MUG_HANDLE_SIDE:+g} "
          f"jug_spout_center={JUG_SPOUT_CENTER_DEG:g}deg jug_handle_side={JUG_HANDLE_SIDE:+g}")

    # Generate the visible assets first from the same dictionaries used below.  A stale hand-made
    # OBJ can therefore never silently pass as "consistent".
    visible = {name: write_visible_obj(name, cfg) for name, cfg in BODIES.items()}

    results = {}
    for name, cfg in BODIES.items():
        with_handle = True
        print(f"\n-- {name}  (R_IN={cfg['r_in']} R_OUT={cfg['r_out']} "
              f"t_bottom={cfg['t_bottom']}, handle={'yes' if with_handle else 'no'})")
        res = build_body(name, cfg, ps, inset, layers, with_handle,
                         edge_ring=not args.no_floor_edge_ring,
                         handle_two_layers=args.handle_two_layers)
        results[name] = res
        counts = res["counts_post"]
        print(f"  inner wall : {res['nz_in']} z-rings  (z {cfg['t_bottom'] + inset:.3f} .. "
              f"{res['z_in_top']:.3f}, last ring truncated <= Z_RIM-delta="
              f"{z_rim - inset:.3f})")
        print(f"  outer wall : {res['nz_out']} z-rings  (z {z_rim - inset:.3f} .. "
              f"{res['z_out_bot']:.3f})")
        print(f"  rim face   : radii {res['rim_radii']}  at z {z_rim - inset:.3f}, "
              f"{z_rim - inset - ps:.3f}")
        print(f"  inner floor: {len(res['floor_radii'])} radii (0 .. {res['floor_radii'][-1]}), "
              f"grid leftover to R_IN = {res['floor_leftover']:.4f}"
              + (f", corner-closing ring at r={res['floor_closing']}" if res["floor_closing"]
                 else ""))
        if res["handle_info"] is not None:
            hi = res["handle_info"]
            print(f"  handle     : a={hi['a']:.6f} b={hi['b']:.6f} e={hi['e']:.6f} "
                  f"rho_end={hi['rho_end']:.6f} (gen_mug.py:110-117 formulas)")
            n_hb = hi["n_sections"] * hi["n_theta"] * hi["n_layers"]
            if hi["layer2_radius"] is None:
                note = (f" [layer-2 radius {hi['r_tube'] - inset - ps:+.4f} m < 0.5*ps "
                        f"-> dropped, deviation D2]")
            else:
                note = ""
            print(f"               sweep={hi['sweep_deg']:.0f} deg len={hi['arc_len']:.5f} m -> "
                  f"{hi['n_sections']} sections x {hi['n_theta']} th x "
                  f"{hi['n_layers']} layer(s) = {n_hb} balls{note}")
            print(f"               farthest r={hi['farthest_radius']:.5f} "
                  f"(limit {hi['farthest_limit']:.5f})  attach z="
                  f"{hi['attach_z'][0]:.3f}/{hi['attach_z'][1]:.3f}  "
                  f"rim clearance={hi['rim_clearance']:.3f}")
        tot = 0
        for face in ("inner_wall", "outer_wall", "rim_face", "floor_face", "handle"):
            if face not in counts:
                continue
            per = counts[face]
            print(f"  {face:11s}: per-layer {per}  sum={sum(per)}")
            tot += sum(per)
        print(f"  TOTAL {name}: {tot} balls (pre-dedup {res['n_pre']}, "
              f"merged {res['n_pre'] - res['n_post']} "
              f"{dict(res['merged_by_face'])})")
        print(f"  true-circle check: worst radius spread over {len(res['rings'])} rings = "
              f"{res['worst_circle']:.3e} (<{CIRCLE_TOL:g})  [{res['worst_circle_where']}]")

    # ---------------- strict OBJ/collision consistency (assertion 2)
    print("\n== strict visible/collision consistency ==")
    for name, cfg in BODIES.items():
        res = results[name]
        check_obj(name, cfg, visible[name], res["pos64"], ps)

    # ---------------- budgets (assertion 3)
    print("\n== budget report ==")
    budget = dict(mug=19000, jug=17000)
    for name, res in results.items():
        n = res["n_post"]
        ref = budget[name]
        frac = (n - ref) / ref
        flag = "OK" if frac <= 0.30 else "OVER 30%"
        print(f"  {name}: {n} balls  (reference ~{ref}, {frac * 100:+.1f}%)  {flag}")
        assert frac <= 0.30, f"{name}: boundary-ball budget exceeded by {frac * 100:.1f}%"

    # ---------------- write npz
    print("\n== npz output ==")
    for name, res in results.items():
        hmeta = None
        if res["handle_info"] is not None:
            hmeta = {k: (float(v) if isinstance(v, (int, float)) else v)
                     for k, v in res["handle_info"].items()}
        meta = dict(
            body=name, ps=ps, inset=inset, layers=layers,
            counts_by_face=dict(res["counts_post"]),
            n_balls=int(res["n_post"]),
            frame=("world (origin=outer bottom center, +z up)" if name == "mug"
                   else "jug-local (origin=outer bottom center, +z along axis)"),
            kinematic=(name == "jug"),
            geometry=dict(r_in=res["cfg"]["r_in"], r_out=res["cfg"]["r_out"],
                          h_in=res["cfg"]["h_in"], t_wall=res["cfg"]["t_wall"],
                          t_bottom=res["cfg"]["t_bottom"], z_rim=res["z_rim"],
                          handle=dict(res["cfg"]["handle"]), psi_e_deg=PSI_E_DEG,
                          side=res["cfg"]["handle_side"],
                          spout=res["cfg"]["spout"],
                          source="mf16_balls.py shared dictionaries + gen_mug/gen_jug constructors"),
            handle=hmeta,
            mass_weight_policy=(
                "mug inner_wall and outer_wall = 0.5; shared r=.154/.162 quadrature; "
                "rim_face/floor_face/handle = 1"
                if name == "mug" else "all faces = 1"
            ),
            effective_mass_weight_sum=float(res["mass_weight"].sum()),
        )
        path = os.path.join(args.out_dir, f"mf16_balls_{name}.npz")
        np.savez(path, pos_local=res["pos"],
                 face=res["face"], layer=res["layer"], mass_weight=res["mass_weight"],
                 meta=np.array(json.dumps(meta, ensure_ascii=False)))
        print(f"  {path}  pos_local{res['pos'].shape} {res['pos'].dtype}  "
              f"finite={bool(np.isfinite(res['pos']).all())}")

    # ---------------- top view png
    if not args.no_png:
        off = dict(mug=(-0.27, 0.0), jug=(0.27, 0.0))
        for name, res in results.items():
            reach = (res["handle_info"]["farthest_radius"] if res["handle_info"]
                     else res["cfg"]["r_out"])
            REF[name] = dict(r_in=res["cfg"]["r_in"], r_out=res["cfg"]["r_out"], reach=reach)
        png = top_view_png(
            [(n, r["pos64"], off[n], (200, 30, 30) if n == "mug" else (30, 90, 220))
             for n, r in results.items()],
            os.path.join(args.out_dir, "mf16_balls_top.png"))
        print(f"\n  png: {png}")

    print("\nMF16 ASSETS PASS (shared-source OBJ + boundary balls, strict coverage, true circles, no NaN)")


if __name__ == "__main__":
    main()
