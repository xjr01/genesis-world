#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mf17_balls.py -- pure-numpy ANALYTIC boundary-ball sampler for the mf17 plain-cup demo.

mf17 variant of mf16_balls.py: BOTH vessels are PLAIN ordinary straight cups --
NO spout, NO handle -- with the same proportions family as the mf16 mug/jug.

Generates Akinci frozen-fluid boundary balls (spacing == particle spacing ps, 2 layers,
inset delta = 0.5*ps) for:

  cup_big   (static, world frame; inner radius 0.150, inner height 0.300, wall 0.016)
            LIFTED 0.030 m off the table: the +0.030 is BAKED into the local z of both
            the ball npz and the OBJ.  Local frame origin stays at table level under the
            cup's axis, so the static set keeps identity-initial-pose compatibility
            (outer bottom at local z = +0.030, rim at local z = 0.300+0.016+0.030 = 0.346).
            -> videos/mf17_balls_cup_big.npz
  cup_small (kinematic, cup-local frame; inner radius 0.125, inner height 0.300, wall 0.016)
            Local origin at the OUTER bottom centre, NO lift baked (lift is applied later
            via the kinematic pose).  Rim at local z = 0.316.
            -> videos/mf17_balls_cup_small.npz

GEOMETRY SOURCE OF TRUTH -- the visible OBJ shells are built with the SAME
gen_mug.build_shell constructor that mf16 used for the mug shell (an open cup:
4 rings A/B/C/D + 2 disc centers O/O2, 7 faces per segment, closed watertight solid,
true circles with n_seg=96).  Every cylindrical ball ring is generated analytically
with cos/sin on the TRUE circle; ring spacing == ps; adjacent rings/layers are
angularly staggered by half a step; points closer than 0.2*ps are merged.

FACES SAMPLED (per body, identical conventions to mf16_balls.py):
  1 inner wall : r = R_IN+delta, R_IN+delta+ps   ; z = t_bottom+delta .. Z_RIM-delta (step ps)
  2 outer wall : r = R_OUT-delta, R_OUT-delta-ps ; z = Z_RIM-delta .. 0 (step ps)
  3 rim top    : z = Z_RIM-delta, Z_RIM-delta-ps ; r = R_IN .. R_OUT (step ps)
  4 inner floor + outer bottom: z = t_bottom-delta, t_bottom-delta-ps ; r = 0 .. R_OUT
     (step ps, incl. r=0 point).  The r <= R_IN part is the cavity-floor quadrature; the
     R_IN..R_OUT annulus is the collision coverage of the VISIBLE OUTER BOTTOM face
     (z = z_outer_bottom).  This seals the floor/wall corner corridor: with the baked
     +0.030 table lift the corridor under the outer-bottom edge is open to the world
     (mf16's mug sat on the table plane which closed that seam), and rebalanced
     floor-corner fluid was escaping through it.  Corner-closing rings are added at
     r == R_IN and r == R_OUT when the ps grid misses a landmark by >= 0.5*ps.

SCHEMA NOTES for the scene script (deviations from mf16):
  * No handle, no spout anywhere.  meta["handle"] and meta["geometry"]["handle"]/
    ["spout"] are present as keys but always None (mf16 had dicts); an explicit
    meta["geometry"]["plain"] = True flag is added.  Ball faces are only
    inner_wall / outer_wall / rim_face / floor_face (never "handle").
  * meta["geometry"]["lift"] records the baked table lift (0.030 for cup_big, 0.0
    for cup_small), together with the baked local z landmarks z_rim_local /
    z_floor_local / z_outer_bottom_local.
  * mass_weight policy: BOTH cups use the mf16-mug convention (inner_wall and
    outer_wall carry 0.5 weight because the two shell directions share the same
    two barrel radii); rim_face / floor_face carry 1.  floor_face now spans the
    whole disc r = 0..R_OUT: inner-floor quadrature (r <= R_IN) AND outer-bottom
    coverage (R_IN..R_OUT annulus), all at unit weight.

OUTPUT: npz with `pos_local (N,3) float32` (PBDOptions.boundary_ball_sets contract)
+ `face` / `layer` / `mass_weight` arrays + `meta` (JSON string).
Plus OBJ previews (front/side png), a top-view validation png, and a strict
self-check report printed and saved to videos/mf17_balls_report.json:
ball counts per face/layer, true-circle roundness, ring angular coverage,
inner/outer wall radii vs spec, rim/bottom z vs spec, npz<->OBJ surface
consistency, and a hard assertion that NO spout/handle geometry exists
(radial uniformity of every wall ring over all azimuths).  There is no
warning-only path.

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 "$PY" mf17_balls.py
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

# Same source-of-truth constructor as the mf16 mug shell (gen_mug.py:55-96).  Importing it
# (rather than copying constants) keeps the visible OBJ and the collision samples tied to
# one definition; the exact reload assertions below fail if the written OBJ ever diverges.
import gen_mug as cup_geometry  # noqa: E402

# ---------------------------------------------------------------- sampling spec
PS = 0.008           # ball spacing == fluid particle spacing [m]
INSET = 0.5 * PS     # delta: offset of layer 1 from the visible surface
LAYERS = 2           # layer 2 sits another 1*ps deeper
DEDUP_FRAC = 0.2     # merge points closer than 0.2*ps
CIRCLE_TOL = 1e-9    # assertion (1): radius spread of any sampled ring
COVERAGE_STEP_TOL = 1.5   # max angular gap tolerance, in units of the ring's nominal step
WALL_RADIAL_TOL = 1e-6    # wall-ball radius deviation from the spec layer radius
OBJ_COVERAGE_TOL_FRAC = 1.01   # visible->ball distance must be <= this fraction of ps
N_SEG = 96

# ------------------------------------------------------------- plain-cup bodies (owner spec)
BODIES = {
    "cup_big": dict(
        r_in=0.150, h_in=0.300, t_wall=0.016, t_bottom=0.016, n_seg=N_SEG,
        lift=0.030,                     # baked table lift (local z of npz AND obj)
        kinematic=False,
        frame=("world (origin at table level under the cup axis; +0.030 lift baked into "
               "local z: outer bottom z=0.030, rim z=0.346; floor_face spans r=0..r_out so "
               "the visible outer bottom face is ball-covered and the corner corridor sealed)"),
        obj=os.path.join(ASSETS, "mf17_cup_big.obj"),
    ),
    "cup_small": dict(
        r_in=0.125, h_in=0.300, t_wall=0.016, t_bottom=0.016, n_seg=N_SEG,
        lift=0.0,                       # no lift baked; kinematic pose applies it later
        kinematic=True,
        frame=("cup-local (origin=outer bottom center, +z along axis; no lift baked, "
               "outer bottom z=0, rim z=0.316; floor_face spans r=0..r_out so the visible "
               "outer bottom face is ball-covered and the corner corridor sealed)"),
        obj=os.path.join(ASSETS, "mf17_cup_small.obj"),
    ),
}
for _c in BODIES.values():
    _c["r_out"] = _c["r_in"] + _c["t_wall"]


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
    return 0.5 * (2.0 * math.pi / n_th) * (parity % 2)


# ------------------------------------------------------------------------------- builder
class Body:
    """Accumulates analytic ball chunks for one PLAIN cup, with per-face/per-layer bookkeeping.

    Identical sampling conventions to mf16_balls.Body faces 1-4; the handle face and the
    spout deformation do not exist here.  `z_off` is the baked table lift applied to every
    generated z (0.030 for cup_big, 0.0 for cup_small).
    """

    def __init__(self, name: str, ps: float = PS, inset: float = INSET,
                 layers: int = LAYERS, z_off: float = 0.0):
        self.name, self.ps, self.inset, self.layers, self.z_off = name, ps, inset, layers, z_off
        self.chunks = []   # (face, layer, pts(N,3) float64, ring_meta | None)
        self.rings = []    # filled in finalize(): (face, layer, center|None, radius, a0, a1)

    # -- low level -------------------------------------------------------------------
    def _add(self, face, layer, pts, ring_meta=None):
        self.chunks.append((face, layer, np.asarray(pts, dtype=np.float64), ring_meta))

    def _axis_ring(self, face, layer, r, z, parity):
        n_th = n_theta(r, self.ps)
        pts = ring(r, z + self.z_off, n_th, stagger(n_th, parity))
        self._add(face, layer, pts, ring_meta=(face, layer, None, r))
        return len(pts)

    # -- faces 1-4 (plain cup: no spout/handle variants) ------------------------------
    def inner_wall(self, r_in, t_bottom, z_rim):
        """Face 1: r = R_IN+delta (+ps); z = t_bottom+delta .. Z_RIM-delta (step ps)."""
        zs = z_grid(t_bottom + self.inset, z_rim - self.inset, self.ps)
        for L in range(self.layers):
            r = r_in + self.inset + L * self.ps
            for k, z in enumerate(zs):
                self._axis_ring("inner_wall", L, r, z, parity=L + k)
        return len(zs), float(zs[-1])

    def outer_wall(self, r_out, z_rim):
        """Face 2: r = R_OUT-delta (-ps); z = Z_RIM-delta .. 0 (step ps).  Plain: true circles."""
        zs = z_grid(z_rim - self.inset, 0.0, self.ps)
        for L in range(self.layers):
            r = r_out - self.inset - L * self.ps
            for k, z in enumerate(zs):
                self._axis_ring("outer_wall", L, r, z, parity=L + k)
        return len(zs), float(zs[-1])

    def rim_face(self, r_in, r_out, z_rim):
        """Face 3: z = Z_RIM-delta (-ps); r = R_IN .. R_OUT (step ps).  Plain: true circles."""
        radii, j = [], 0
        while r_in + j * self.ps <= r_out + 1e-9:
            radii.append(round(r_in + j * self.ps, 9))
            j += 1
        if radii[-1] < r_out - 1e-9:
            radii.append(float(r_out))
        for L in range(self.layers):
            z = z_rim - self.inset - L * self.ps
            for j, r in enumerate(radii):
                self._axis_ring("rim_face", L, r, z, parity=L + j)
        return radii

    def floor_face(self, r_in, r_out, t_bottom, edge_ring=True):
        """Face 4 (one disc): inner floor + outer bottom coverage.

        z = t_bottom-delta (layer 0) and t_bottom-delta-ps (layer 1) -- i.e. the two
        layers sit at z_outer_bottom+inset+ps and z_outer_bottom+inset, INSET into the
        bottom slab from both its faces (the cavity floor above and the visible outer
        bottom below are z=t_bottom and z=z_outer_bottom = t_bottom - (t_bottom) apart
        by exactly the slab thickness).  Rings r = 0 .. R_OUT (step ps, incl. r=0 point);
        the r <= R_IN part is the cavity-floor quadrature, the R_IN..R_OUT annulus
        covers the visible outer-bottom face and seals the floor/wall corner corridor.

        edge_ring: when the ps grid misses a landmark (R_IN and/or R_OUT) by >= 0.5*ps,
        a closing ring at exactly that radius seals the floor/wall and bottom/wall
        corners (documented deviation, same convention as mf16).
        """
        radii = [0.0]
        k = 1
        while k * self.ps <= r_out + 1e-9:
            radii.append(round(k * self.ps, 9))
            k += 1
        leftover = {}
        closings = []
        if edge_ring:
            for r_land in (r_in, r_out):
                below = max(r for r in radii if r <= r_land + 1e-12)
                lo = round(r_land - below, 12)
                leftover[f"r={r_land:g}"] = lo
                if lo >= 0.5 * self.ps - 1e-12:
                    closings.append(round(r_land, 9))
        radii = sorted(radii + closings)
        for L in range(self.layers):
            z = t_bottom - self.inset - L * self.ps
            for j, r in enumerate(radii):
                if r == 0.0:  # single center point (avoids coincident-point NaN)
                    self._add("floor_face", L,
                              np.array([[0.0, 0.0, z + self.z_off]]),
                              ring_meta=("floor_face", L, None, 0.0))
                else:
                    self._axis_ring("floor_face", L, r, z, parity=L + j)
        return radii, leftover, closings

    # -- assembly --------------------------------------------------------------------
    def finalize(self, dedup_frac=DEDUP_FRAC):
        pts64 = (np.vstack([c[2] for c in self.chunks]) if self.chunks
                 else np.zeros((0, 3), dtype=np.float64))
        # Keep the sampler's native provenance one-for-one with the coordinates (same
        # reasoning as mf16: spatial inference is ambiguous at shared radii and seams).
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

        # The two shell directions sample the same two barrel radii because
        # t_wall = 2*ps and delta = 0.5*ps make both pairs coincide:
        #   inner L0 (R_IN+delta)          == outer L1 (R_OUT-delta-ps = R_IN+ps-delta)
        #   inner L1 (R_IN+delta+ps)       == outer L0 (R_OUT-delta     = R_IN+3*ps.. = R_IN+ps+delta)
        # Their z-staggered samples form one surface quadrature, not two full-mass walls,
        # so each wall copy carries half weight -- exactly the mf16-mug convention.
        mass_weight = np.ones(len(pts64), dtype=np.float32)
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
    (Identical to mf16_balls.dedup_near.)
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
    """Build the plain open-cup shell from gen_mug.build_shell and bake the table lift.

    No handle is added: the mesh is exactly the 4-ring + 2-center shell
    (len(verts) == 4*n_seg + 2), which the assertions in check_obj verify.
    """
    shell = cup_geometry.build_shell(cfg["r_in"], cfg["h_in"], cfg["t_wall"],
                                     cfg["t_bottom"], cfg["n_seg"])
    assert shell.is_watertight and shell.is_winding_consistent and shell.is_volume
    verts = np.asarray(shell.vertices, dtype=np.float64)
    faces = np.asarray(shell.faces, dtype=np.int64)
    assert len(verts) == 4 * cfg["n_seg"] + 2, "plain cup shell must have no extra vertices"
    # Bake the table lift into the local frame (identity for cup_small).
    verts = verts + np.array([0.0, 0.0, cfg["lift"]])
    os.makedirs(os.path.dirname(os.path.abspath(cfg["obj"])), exist_ok=True)
    cup_geometry.trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(cfg["obj"])
    # These previews are validation artifacts, not simulation renders.
    cup_geometry.ortho_png(verts, faces, os.path.splitext(cfg["obj"])[0] + "_front.png",
                           axis="y")
    cup_geometry.ortho_png(verts, faces, os.path.splitext(cfg["obj"])[0] + "_side.png",
                           axis="x")
    return dict(verts=verts, faces=faces,
                shell_nverts=len(shell.vertices), shell_nfaces=len(shell.faces))


def nearest_distance(queries: np.ndarray, sources: np.ndarray, chunk=128) -> np.ndarray:
    out = np.empty(len(queries), dtype=np.float64)
    for i in range(0, len(queries), chunk):
        q = queries[i:i + chunk]
        d2 = ((q[:, None, :] - sources[None, :, :]) ** 2).sum(axis=2)
        out[i:i + len(q)] = np.sqrt(d2.min(axis=1))
    return out


def obj_surface_points(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Vertices + triangle centroids: a denser visible-surface sampling than vertices alone."""
    return np.vstack([verts, verts[faces].mean(axis=1)])


def ring_coverage_report(res: dict):
    """Max angular gap of every true-circle ring, in units of its nominal uniform step.

    A uniform ring of n points has all gaps == 2*pi/n exactly, so any angular defect
    (a spout/handle bump or a bad ring) shows up as a ratio >> 1.  Single-point floor
    centers (n == 1) carry no angular meaning and are skipped.
    """
    pts = res["pos64"]
    worst_ratio, worst_where, n_checked = 0.0, "", 0
    for face, layer, center, radius, a0, a1 in res["rings"]:
        q = pts[a0:a1]
        if len(q) < 3 or radius <= 0.0:
            continue
        ang = np.arctan2(q[:, 1], q[:, 0])
        ang.sort()
        gaps = np.diff(np.concatenate([ang, [ang[0] + 2.0 * math.pi]]))
        step = 2.0 * math.pi / len(q)
        ratio = float(gaps.max() / step)
        n_checked += 1
        if ratio > worst_ratio:
            worst_ratio, worst_where = ratio, f"{face}/L{layer}/r={radius:.6f}"
    assert worst_ratio < COVERAGE_STEP_TOL, (
        f"ring coverage failure: max angular gap ratio {worst_ratio:.3f} at {worst_where} "
        f">= {COVERAGE_STEP_TOL}"
    )
    return dict(max_gap_ratio=worst_ratio, max_gap_where=worst_where,
                n_rings_checked=n_checked,
                tol_ratio=COVERAGE_STEP_TOL)


def wall_radial_report(name: str, res: dict, cfg: dict):
    """Every wall ball must sit at its spec layer radius, at EVERY azimuth (no spout/handle).

    Groups the post-dedup wall balls by (face, layer) and compares against the analytic
    layer radii; then groups by z to prove ring-by-ring radial uniformity over all angles.
    """
    pos, face, layer = res["pos64"], res["face"], res["layer"]
    spec_r = {
        ("inner_wall", 0): cfg["r_in"] + INSET,
        ("inner_wall", 1): cfg["r_in"] + INSET + PS,
        ("outer_wall", 0): cfg["r_out"] - INSET,
        ("outer_wall", 1): cfg["r_out"] - INSET - PS,
    }
    dev_by_layer, z_uniformity = {}, 0.0
    for (f, L), r_spec in spec_r.items():
        m = (face == f) & (layer == L)
        assert m.sum() > 0, f"{name}: no balls for {f}/L{L}"
        dev = float(np.abs(np.hypot(pos[m, 0], pos[m, 1]) - r_spec).max())
        assert dev < WALL_RADIAL_TOL, f"{name}: {f}/L{L} radius dev {dev:.2e} >= {WALL_RADIAL_TOL:g}"
        dev_by_layer[f"{f}/L{L}"] = dict(expected=r_spec, max_dev=dev)
        # ring-by-ring radial uniformity at fixed z (this is what forbids a spout/handle):
        for z in np.unique(np.round(pos[m, 2], 9)):
            mm = m & (np.round(pos[:, 2], 9) == z)
            rr = np.hypot(pos[mm, 0], pos[mm, 1])
            z_uniformity = max(z_uniformity, float(rr.max() - rr.min()))
    assert z_uniformity < WALL_RADIAL_TOL, (
        f"{name}: wall rings are not radially uniform (max spread {z_uniformity:.2e}); "
        f"spout/handle geometry present?"
    )
    return dict(dev_by_layer=dev_by_layer,
                ring_radial_uniformity_max_spread=z_uniformity,
                tol=WALL_RADIAL_TOL)


def z_landmark_report(name: str, res: dict, cfg: dict, ref: dict):
    """Rim and bottom z of balls AND obj vs the owner spec."""
    z_rim = cfg["t_bottom"] + cfg["h_in"] + cfg["lift"]      # visible rim top (local)
    z_floor = cfg["t_bottom"] + cfg["lift"]                  # visible inner floor (local)
    z_ob = cfg["lift"]                                       # visible outer bottom (local)
    n = cfg["n_seg"]
    verts = ref["verts"]
    A = verts[0:n]        # outer bottom edge
    B = verts[n:2 * n]    # outer top edge
    C = verts[2 * n:3 * n]  # inner top edge
    D = verts[3 * n:4 * n]  # inner floor edge
    obj_checks = {
        "outer_top_ring_B": dict(expected_z=z_rim, measured=float(B[:, 2].max()),
                                 max_abs_dev=float(np.abs(B[:, 2] - z_rim).max())),
        "inner_top_ring_C": dict(expected_z=z_rim, measured=float(C[:, 2].max()),
                                 max_abs_dev=float(np.abs(C[:, 2] - z_rim).max())),
        "inner_floor_ring_D": dict(expected_z=z_floor, measured=float(D[:, 2].max()),
                                   max_abs_dev=float(np.abs(D[:, 2] - z_floor).max())),
        "outer_bottom_ring_A": dict(expected_z=z_ob, measured=float(A[:, 2].max()),
                                    max_abs_dev=float(np.abs(A[:, 2] - z_ob).max())),
    }
    for k, v in obj_checks.items():
        assert v["max_abs_dev"] < 1e-9, f"{name}: obj {k} z dev {v['max_abs_dev']:.2e}"
    # ball rim layers sit delta (and delta+ps) BELOW the visible rim, by construction;
    # assert they do so exactly.
    pos, face, layer = res["pos64"], res["face"], res["layer"]
    ball_rim = {}
    for L in range(LAYERS):
        m = (face == "rim_face") & (layer == L)
        assert m.sum() > 0
        z_spec = cfg["t_bottom"] + cfg["h_in"] - INSET - L * PS + cfg["lift"]
        dev = float(np.abs(pos[m, 2] - z_spec).max())
        assert dev < 1e-9, f"{name}: rim_face/L{L} z dev {dev:.2e}"
        ball_rim[f"L{L}"] = dict(expected_z=z_spec, max_abs_dev=dev)
    return dict(z_rim_local=z_rim, z_floor_local=z_floor, z_outer_bottom_local=z_ob,
                obj=obj_checks, ball_rim_layers=ball_rim)


def bottom_seal_report(name: str, res: dict, cfg: dict):
    """mf17 fix verification: the visible OUTER BOTTOM face is ball-covered and the
    floor/wall corner corridor is sealed.

    Three hard assertions:
      (a) lowest ball z == z_outer_bottom_local + inset (the deeper floor layer);
      (b) the R_IN..R_OUT annulus has balls in BOTH floor layers and every annulus
          ring passes the angular-coverage check (re-run here on the annulus subset);
      (c) corner-seam ray march: from cavity-corner starts (r ~= R_IN-ps, z ~= z_floor+ps),
          rays stepped down / down-out at 0.5*ps increments must pass within ps of a ball
          centre at EVERY sample until they drop below the lowest ball layer (z <
          z_outer_bottom+inset-0.5*ps) or leave the boundary region (r > R_OUT+ps).
          A single sample farther than ps from any ball is a traversable leak channel.
    """
    z_ob = cfg["lift"]
    z_floor = cfg["t_bottom"] + cfg["lift"]
    r_in, r_out = cfg["r_in"], cfg["r_out"]
    pos, face, layer = res["pos64"], res["face"], res["layer"]

    # (a) lowest balls: the deeper floor layer sits at z_outer_bottom+inset; the global
    # minimum is the outer-wall bottom ring, which mf16 convention samples at z = z_outer
    # bottom exactly (z = 0 un-lifted).
    z_min = float(pos[:, 2].min())
    z_min_spec = z_ob
    assert abs(z_min - z_min_spec) < 1e-9, (
        f"{name}: lowest ball z {z_min:.9f} != z_outer_bottom {z_min_spec:.9f}")
    fl = pos[face == "floor_face"]
    z_min_floor = float(fl[:, 2].min())
    z_min_floor_spec = z_ob + INSET
    assert abs(z_min_floor - z_min_floor_spec) < 1e-9, (
        f"{name}: lowest floor_face ball z {z_min_floor:.9f} != "
        f"z_outer_bottom+inset {z_min_floor_spec:.9f}")

    # (b) annulus presence + coverage
    rr = np.hypot(pos[:, 0], pos[:, 1])
    ann = (face == "floor_face") & (rr >= r_in - 1e-9)
    assert ann.sum() > 0, f"{name}: no floor balls in the R_IN..R_OUT outer-bottom annulus"
    ann_by_layer = {int(L): int((ann & (layer == L)).sum()) for L in np.unique(layer[ann])}
    assert set(ann_by_layer) == {0, 1}, f"{name}: annulus missing a layer: {ann_by_layer}"
    # per-ring coverage on the annulus rings
    worst_ratio, worst_ring = 0.0, ""
    for face_r, layer_r, center, radius, a0, a1 in res["rings"]:
        if face_r != "floor_face" or radius < r_in - 1e-9 or radius <= 0.0:
            continue
        q = pos[a0:a1]
        ang = np.arctan2(q[:, 1], q[:, 0])
        ang.sort()
        gaps = np.diff(np.concatenate([ang, [ang[0] + 2.0 * math.pi]]))
        ratio = float(gaps.max() / (2.0 * math.pi / len(q)))
        if ratio > worst_ratio:
            worst_ratio, worst_ring = ratio, f"r={radius:.6f}"
    assert worst_ratio < COVERAGE_STEP_TOL, (
        f"{name}: outer-bottom annulus ring coverage failure: ratio {worst_ratio:.3f} "
        f"at {worst_ring} >= {COVERAGE_STEP_TOL}")

    # (c) corner-seam ray march over the local corner neighbourhood
    nb_mask = (pos[:, 2] < z_floor + 2.0 * PS) & (rr < r_out + PS)
    nb = pos[nb_mask]
    z_stop = z_ob + INSET - 0.5 * PS
    start_r = r_in - PS
    start_z = z_floor + PS
    directions = [(0.0, -1.0), (1.0, -1.0), (1.0, -2.0), (2.0, -1.0), (1.0, -4.0)]
    n_az = 8
    n_v = int(round((r_out - start_r) / (0.5 * PS)))
    vertical_radii = [round(start_r + 0.5 * PS * k, 9) for k in range(n_v + 1)]
    vertical_radii = [r for r in vertical_radii if r <= r_out + 1e-9]
    worst_d, n_samples, n_paths = 0.0, 0, 0
    worst_where = ""

    def _probe(px, py, pz, tag):
        nonlocal worst_d, worst_where
        d = float(np.sqrt(((nb - np.array([px, py, pz])) ** 2).sum(axis=1)).min())
        if d > worst_d:
            worst_d, worst_where = d, tag
        assert d < PS, (
            f"{name}: corner corridor is traversable: sample {tag} at "
            f"({px:.5f},{py:.5f},{pz:.5f}) is {d:.6f} >= ps from every ball centre")

    n_skipped = 0   # samples in open fluid (cavity above the visible floor)
    for dr, dz in directions:
        norm = math.hypot(dr, dz)
        dr, dz = dr / norm, dz / norm
        for a in range(n_az):
            phi = 2.0 * math.pi * a / n_az
            ca, sa = math.cos(phi), math.sin(phi)
            s = 0.0
            while True:
                r = start_r + dr * s
                z = start_z + dz * s
                if z < z_stop or r > r_out + PS:
                    break
                # only the boundary-blocking region must be ball-dense: inside the cup
                # solid at/below the visible floor (z <= z_floor) and not beyond the
                # outer wall; samples above the floor are open fluid by design.
                if z <= z_floor + 1e-9:
                    _probe(r * ca, r * sa, z,
                           f"dir=({dr:.3f},{dz:.3f}) az={a} s={s:.4f}")
                    n_samples += 1
                else:
                    n_skipped += 1
                s += 0.5 * PS
            n_paths += 1
    for rv in vertical_radii:
        for a in range(n_az):
            phi = 2.0 * math.pi * a / n_az
            z = start_z
            while z >= z_stop:
                if z <= z_floor + 1e-9:
                    _probe(rv * math.cos(phi), rv * math.sin(phi), z,
                           f"vert r={rv:.4f} az={a} z={z:.5f}")
                    n_samples += 1
                else:
                    n_skipped += 1
                z -= 0.5 * PS
            n_paths += 1

    return dict(
        lowest_ball_z=z_min, lowest_ball_z_spec=z_min_spec,
        lowest_floor_ball_z=z_min_floor, lowest_floor_ball_z_spec=z_min_floor_spec,
        annulus_balls_by_layer=ann_by_layer,
        annulus_ring_coverage_max_gap_ratio=worst_ratio,
        annulus_ring_coverage_tol=COVERAGE_STEP_TOL,
        seam_raymarch=dict(
            n_paths=n_paths, n_samples=n_samples, n_openfluid_samples_skipped=n_skipped,
            worst_sample_to_ball_dist=worst_d, worst_where=worst_where, tol=PS,
            step=0.5 * PS, z_stop=z_stop,
            start=(start_r, start_z),
            directions=[(dr, dz) for dr, dz in directions],
            vertical_radii=vertical_radii, n_azimuths=n_az,
        ),
    )


def check_obj(name: str, cfg: dict, ref: dict, balls: np.ndarray, ps: float):
    """Require exact generated-source identity, plain-cup shape, and collision coverage."""
    verts, faces = load_obj(cfg["obj"])
    assert verts.shape == ref["verts"].shape, f"{name}: OBJ vertex count/source mismatch"
    assert faces.shape == ref["faces"].shape, f"{name}: OBJ face count/source mismatch"
    assert np.allclose(verts, ref["verts"], rtol=0.0, atol=1e-8), f"{name}: OBJ diverges from source"
    assert np.array_equal(faces, ref["faces"]), f"{name}: OBJ faces diverge from source"

    n = cfg["n_seg"]
    z_rim = cfg["t_bottom"] + cfg["h_in"] + cfg["lift"]
    assert ref["shell_nverts"] == 4 * n + 2
    assert len(verts) == 4 * n + 2, f"{name}: plain cup must have exactly {4*n+2} verts"

    B = verts[n:2 * n]
    C = verts[2 * n:3 * n]
    A = verts[0:n]
    # true circles at exact spec radii for ALL azimuths (no spout/handle permitted)
    for ring_name, q, r_spec in (("outer_top", B, cfg["r_out"]),
                                 ("inner_top", C, cfg["r_in"]),
                                 ("outer_bottom", A, cfg["r_out"])):
        dev_r = float(np.abs(np.linalg.norm(q[:, :2], axis=1) - r_spec).max())
        assert dev_r < 1e-8, f"{name}: {ring_name} ring radius dev {dev_r:.2e} (not a plain cup)"
        assert float(np.abs(q[:, 2] - q[0, 2]).max()) < 1e-12, f"{name}: {ring_name} ring not planar"
    assert abs(B[0, 2] - z_rim) < 1e-9 and abs(C[0, 2] - z_rim) < 1e-9
    sag = cfg["r_in"] * (1.0 - math.cos(math.pi / n))
    assert sag < 1e-4

    # visible-surface coverage: vertices + triangle centroids within one ps of a ball center
    surf = obj_surface_points(verts, faces)
    coverage = nearest_distance(surf, balls)
    max_cov_v = float(nearest_distance(verts, balls).max())
    max_cov = float(coverage.max())
    tol = ps * OBJ_COVERAGE_TOL_FRAC
    assert max_cov <= tol, (
        f"{name}: visible geometry not covered by boundary balls; max distance "
        f"{max_cov:.6f} > {tol:.6f}"
    )
    print(
        f"  [{name}.obj] exact shared-source reload: verts={len(verts)} faces={len(faces)}; "
        f"plain-cup rings (r_in/r_out exact, no spout/handle); sagitta={sag:.3e}; "
        f"max visible->ball dist: verts {max_cov_v:.6f}, verts+centroids {max_cov:.6f} "
        f"<= {tol:.6f}"
    )
    return dict(verts=len(verts), faces=len(faces),
                max_vertex_to_ball_dist=max_cov_v,
                max_surfacepoint_to_ball_dist=max_cov, tol=tol)


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

    # faint true-circle references (r_in / r_out) for eyeballing roundness
    for label, p, off, rgb in placed:
        for R, col in ((REF[label]["r_in"], (185, 185, 185)),
                       (REF[label]["r_out"], (185, 185, 185))):
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
def build_body(name, cfg, ps, inset, layers, edge_ring=True):
    z_rim = cfg["t_bottom"] + cfg["h_in"]               # un-lifted rim z
    b = Body(name, ps=ps, inset=inset, layers=layers, z_off=cfg["lift"])
    nz_in, z_in_top = b.inner_wall(cfg["r_in"], cfg["t_bottom"], z_rim)
    nz_out, z_out_bot = b.outer_wall(cfg["r_out"], z_rim)
    rim_radii = b.rim_face(cfg["r_in"], cfg["r_out"], z_rim)
    floor_radii, leftover, closings = b.floor_face(cfg["r_in"], cfg["r_out"], cfg["t_bottom"],
                                                   edge_ring=edge_ring)
    out = b.finalize()
    out.update(name=name, cfg=cfg, z_rim=z_rim, nz_in=nz_in, z_in_top=z_in_top,
               nz_out=nz_out, z_out_bot=z_out_bot, rim_radii=rim_radii,
               floor_radii=floor_radii, floor_leftover=leftover, floor_closing=closings)
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
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ps, inset, layers = args.ps, args.inset, args.layers
    print(f"== mf17_balls : analytic boundary-ball sampler, PLAIN straight cups ==")
    print(f"ps={ps}  inset(delta)={inset}  layers={layers}  n_seg={N_SEG}  "
          f"NO spout, NO handle (hard-asserted radial uniformity)")

    # Generate the visible assets first from the same constructor used below.  A stale
    # hand-made OBJ can therefore never silently pass as "consistent".
    visible = {name: write_visible_obj(name, cfg) for name, cfg in BODIES.items()}

    results, report = {}, dict(
        script="mf17_balls.py",
        ps=ps, inset=inset, layers=layers,
        tolerances=dict(circle_tol=CIRCLE_TOL,
                        coverage_step_tol=COVERAGE_STEP_TOL,
                        wall_radial_tol=WALL_RADIAL_TOL,
                        obj_coverage_tol_frac=OBJ_COVERAGE_TOL_FRAC),
        plain_cups=True,
        bodies={},
    )
    for name, cfg in BODIES.items():
        print(f"\n-- {name}  (R_IN={cfg['r_in']} R_OUT={cfg['r_out']} "
              f"t_bottom={cfg['t_bottom']} h_in={cfg['h_in']} lift={cfg['lift']} "
              f"kinematic={cfg['kinematic']})")
        res = build_body(name, cfg, ps, inset, layers,
                         edge_ring=not args.no_floor_edge_ring)
        results[name] = res
        counts = res["counts_post"]
        print(f"  inner wall : {res['nz_in']} z-rings  (z {cfg['t_bottom'] + inset + cfg['lift']:.3f} .. "
              f"{res['z_in_top'] + cfg['lift']:.3f}, truncated <= Z_RIM-delta="
              f"{res['z_rim'] - inset + cfg['lift']:.3f})")
        print(f"  outer wall : {res['nz_out']} z-rings  (z {res['z_rim'] - inset + cfg['lift']:.3f} .. "
              f"{res['z_out_bot'] + cfg['lift']:.3f})")
        print(f"  rim face   : radii {res['rim_radii']}  at z "
              f"{res['z_rim'] - inset + cfg['lift']:.3f}, {res['z_rim'] - inset - ps + cfg['lift']:.3f}")
        print(f"  inner floor+bottom: {len(res['floor_radii'])} radii (0 .. {res['floor_radii'][-1]}), "
              f"grid leftovers {res['floor_leftover']}"
              + (f", corner-closing rings at r={res['floor_closing']}" if res["floor_closing"]
                 else ""))
        tot = 0
        for face in ("inner_wall", "outer_wall", "rim_face", "floor_face"):
            per = counts[face]
            print(f"  {face:11s}: per-layer {per}  sum={sum(per)}")
            tot += sum(per)
        print(f"  TOTAL {name}: {tot} balls (pre-dedup {res['n_pre']}, "
              f"merged {res['n_pre'] - res['n_post']} {dict(res['merged_by_face'])})")
        print(f"  true-circle check: worst radius spread over {len(res['rings'])} rings = "
              f"{res['worst_circle']:.3e} (<{CIRCLE_TOL:g})  [{res['worst_circle_where']}]")

    # ---------------- strict OBJ/collision consistency + geometry assertions
    print("\n== strict visible/collision consistency + plain-cup assertions ==")
    for name, cfg in BODIES.items():
        res = results[name]
        obj_rep = check_obj(name, cfg, visible[name], res["pos64"], ps)
        cov_rep = ring_coverage_report(res)
        wall_rep = wall_radial_report(name, res, cfg)
        z_rep = z_landmark_report(name, res, cfg, visible[name])
        seal_rep = bottom_seal_report(name, res, cfg)
        print(f"  [{name}] ring coverage: max angular-gap ratio {cov_rep['max_gap_ratio']:.3f} "
              f"({cov_rep['n_rings_checked']} rings, tol <{COVERAGE_STEP_TOL}) "
              f"[{cov_rep['max_gap_where']}]")
        print(f"  [{name}] wall radial uniformity: max ring spread "
              f"{wall_rep['ring_radial_uniformity_max_spread']:.3e} (<{WALL_RADIAL_TOL:g}); "
              f"no spout/handle confirmed")
        print(f"  [{name}] z landmarks: rim={z_rep['z_rim_local']:.3f} "
              f"floor={z_rep['z_floor_local']:.3f} outer_bottom={z_rep['z_outer_bottom_local']:.3f}")
        print(f"  [{name}] outer-bottom seal: lowest ball z={seal_rep['lowest_ball_z']:.3f} "
              f"(outer-wall bottom ring == z_ob), lowest floor ball z="
              f"{seal_rep['lowest_floor_ball_z']:.3f} (== z_ob+inset "
              f"{seal_rep['lowest_floor_ball_z_spec']:.3f}); annulus balls/layer "
              f"{seal_rep['annulus_balls_by_layer']}; seam ray march worst sample-to-ball "
              f"dist {seal_rep['seam_raymarch']['worst_sample_to_ball_dist']:.6f} "
              f"(<{seal_rep['seam_raymarch']['tol']}, {seal_rep['seam_raymarch']['n_samples']} "
              f"samples on {seal_rep['seam_raymarch']['n_paths']} paths)")
        report["bodies"][name] = dict(
            geometry=dict(
                plain=True, spout=None, handle=None,
                r_in=cfg["r_in"], r_out=cfg["r_out"], h_in=cfg["h_in"],
                t_wall=cfg["t_wall"], t_bottom=cfg["t_bottom"], n_seg=cfg["n_seg"],
                lift=cfg["lift"], kinematic=cfg["kinematic"],
                z_rim_local=z_rep["z_rim_local"],
                z_floor_local=z_rep["z_floor_local"],
                z_outer_bottom_local=z_rep["z_outer_bottom_local"],
                frame=cfg["frame"],
            ),
            counts_by_face={k: list(v) for k, v in res["counts_post"].items()},
            n_balls=int(res["n_post"]),
            n_pre_dedup=int(res["n_pre"]),
            merged_by_face={k: int(v) for k, v in res["merged_by_face"].items()},
            mass_weight_effective_sum=float(res["mass_weight"].sum()),
            roundness=dict(worst_radius_spread=res["worst_circle"],
                           where=res["worst_circle_where"], tol=CIRCLE_TOL),
            ring_coverage=cov_rep,
            wall_radial=wall_rep,
            z_landmarks=z_rep,
            bottom_seal=seal_rep,
            obj=dict(path=cfg["obj"], **obj_rep),
            npz_path=os.path.join(args.out_dir, f"mf17_balls_{name}.npz"),
        )

    # ---------------- budgets (generous references; counts mirror mf16-mug scale)
    print("\n== budget report ==")
    budget = dict(cup_big=22000, cup_small=20000)
    for name, res in results.items():
        n = res["n_post"]
        ref = budget[name]
        frac = (n - ref) / ref
        flag = "OK" if frac <= 0.30 else "OVER 30%"
        print(f"  {name}: {n} balls  (reference ~{ref}, {frac * 100:+.1f}%)  {flag}")
        assert frac <= 0.30, f"{name}: boundary-ball budget exceeded by {frac * 100:.1f}%"
        report["bodies"][name]["budget"] = dict(reference=ref, n=n, over_frac=frac)

    # ---------------- write npz
    print("\n== npz output ==")
    for name, res in results.items():
        cfg = BODIES[name]
        meta = dict(
            body=name, ps=ps, inset=inset, layers=layers,
            counts_by_face=dict(res["counts_post"]),
            n_balls=int(res["n_post"]),
            frame=cfg["frame"],
            kinematic=cfg["kinematic"],
            geometry=dict(
                plain=True,                       # mf17: plain straight cup, no spout/handle
                spout=None,                       # key kept for mf16 schema parity, always None
                handle=None,                      # key kept for mf16 schema parity, always None
                r_in=cfg["r_in"], r_out=cfg["r_out"],
                h_in=cfg["h_in"], t_wall=cfg["t_wall"], t_bottom=cfg["t_bottom"],
                n_seg=cfg["n_seg"],
                lift=cfg["lift"],
                z_rim_local=cfg["t_bottom"] + cfg["h_in"] + cfg["lift"],
                z_floor_local=cfg["t_bottom"] + cfg["lift"],
                z_outer_bottom_local=cfg["lift"],
                outer_bottom_balls=True,   # mf17 fix: R_IN..R_OUT annulus of floor_face
                # covers the visible outer bottom face and seals the floor/wall corner
                source="mf17_balls.py + gen_mug.build_shell (same-source as mf16 mug shell)",
            ),
            handle=None,          # mf16 had a handle-info dict; plain cups have none
            mass_weight_policy=(
                "inner_wall and outer_wall = 0.5 (shared barrel radii, t_wall = 2*ps); "
                "rim_face and floor_face = 1; floor_face spans r=0..R_OUT (inner floor "
                "quadrature + outer-bottom annulus, all unit weight)"
            ),
            effective_mass_weight_sum=float(res["mass_weight"].sum()),
        )
        path = os.path.join(args.out_dir, f"mf17_balls_{name}.npz")
        np.savez(path, pos_local=res["pos"],
                 face=res["face"], layer=res["layer"], mass_weight=res["mass_weight"],
                 meta=np.array(json.dumps(meta, ensure_ascii=False)))
        print(f"  {path}  pos_local{res['pos'].shape} {res['pos'].dtype}  "
              f"finite={bool(np.isfinite(res['pos']).all())}")

    # ---------------- top view png
    if not args.no_png:
        off = dict(cup_big=(-0.25, 0.0), cup_small=(0.25, 0.0))
        for name, res in results.items():
            REF[name] = dict(r_in=res["cfg"]["r_in"], r_out=res["cfg"]["r_out"])
        png = top_view_png(
            [(n, r["pos64"], off[n], (200, 30, 30) if n == "cup_big" else (30, 90, 220))
             for n, r in results.items()],
            os.path.join(args.out_dir, "mf17_balls_top.png"))
        print(f"\n  png: {png}")

    report["pass"] = True
    report_path = os.path.join(args.out_dir, "mf17_balls_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n  report: {report_path}")

    print("\nMF17 ASSETS PASS (plain cups, shared-source OBJ + boundary balls, strict "
          "coverage, true circles, radial uniformity -> no spout/handle, no NaN)")


if __name__ == "__main__":
    main()
