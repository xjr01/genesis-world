"""Generate mug.obj: gen_cup shell + C-shaped arc-tube handle on the -x side.

Shell   : verbatim reuse of gen_cup's parameterized construction (4 rings + 2 centers,
          7 faces per segment, hand-wound CCW from outside) -> closed watertight solid.
          z=0 at the OUTER bottom, cavity floor z=t_bottom, rim top z=t_bottom+h_in.
Handle  : a C-shaped tube whose centerline is an elliptical arc in the x-z plane at
          azimuth 180 deg (-x). The arc sweeps > 180 deg (220 deg) so both ends dive
          into the wall solid: the end-cap centers sit at radius (r_in+r_out)/2 and the
          full end-cap discs stay inside the wall annulus -> no tip pokes into the cavity.
          M cross-sections (K segments each) + flat center-point caps = closed tube.
Merge   : plain vertex/face concatenation (index offset) of two INDEPENDENTLY watertight
          shells. Geometrically an overlapping-solids union, which is intentionally
          non-manifold as a whole -> assertions are run per shell, never on the merge.

Alignment: shell matches analytic boundary_cup_shell=(0,0,0.016,0.15,0.30,0.016,0.016,0.008)
           (x,y,z_floor,r_in,h_in,t_wall,t_bottom,particle_radius), z=0 at outer bottom.

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 "$PY" gen_mug.py
"""

import argparse
import os

import numpy as np
import trimesh

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
ASSETS = os.path.join(WORKSPACE, "assets")

# Analytic-shell-aligned defaults (meters): r_in / h_in / t_wall / t_bottom / n_seg
DEFAULTS = dict(
    r_in=0.15,
    h_in=0.30,       # cavity depth (floor -> rim)
    t_wall=0.016,    # = 2 * particle_size 0.008
    t_bottom=0.016,
    n_seg=96,
    # handle
    r_tube=0.013,
    z_lo=0.10,       # lower attachment z
    z_hi=0.22,       # upper attachment z
    handle_clear=0.055,   # farthest handle point <= r_out + handle_clear
    m=72,            # cross-sections along the arc (>= 48)
    k=16,            # segments per cross-section (>= 16)
    psi_e_deg=70.0,  # arc end angle from the inward direction -> arc = 220 deg > 180
    out=os.path.join(ASSETS, "mug.obj"),
)


def ring(radius, z, n):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n, z)], axis=1)


def build_shell(r_in, h_in, t_wall, t_bottom, n_seg, b_ring_fn=None):
    """gen_cup's shell, topology and winding untouched.

    b_ring_fn(B, theta) -> B may displace OUTER-TOP ring vertices only (jug spout);
    A/C/D/O/O2 (and hence the whole fluid-wetted cavity) are never touched.
    """
    n = n_seg
    r_out = r_in + t_wall
    z0 = 0.0                  # outer bottom
    z_floor = t_bottom        # inner floor
    z_rim = t_bottom + h_in   # rim height

    A = ring(r_out, z0, n)      # outer bottom edge
    B = ring(r_out, z_rim, n)   # outer top edge
    if b_ring_fn is not None:
        B = np.asarray(b_ring_fn(B, np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)))
    C = ring(r_in, z_rim, n)    # inner top edge
    D = ring(r_in, z_floor, n)  # inner floor edge
    O = np.array([[0.0, 0.0, z0]])
    O2 = np.array([[0.0, 0.0, z_floor]])

    verts = np.vstack([A, B, C, D, O, O2])
    iA, iB, iC, iD, iO, iO2 = 0, n, 2 * n, 3 * n, 4 * n, 4 * n + 1

    faces = []
    for i in range(n):
        j = (i + 1) % n
        # outer side wall, normal +radial
        faces.append([iA + i, iA + j, iB + j])
        faces.append([iA + i, iB + j, iB + i])
        # inner side wall, normal -radial (toward cavity axis)
        faces.append([iD + i, iC + i, iC + j])
        faces.append([iD + i, iC + j, iD + j])
        # top rim annulus, normal +z
        faces.append([iB + i, iB + j, iC + j])
        faces.append([iB + i, iC + j, iC + i])
        # bottom disc fan, normal -z
        faces.append([iO, iA + j, iA + i])
        # cavity floor disc fan, normal +z
        faces.append([iO2, iD + i, iD + j])

    return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64), process=False)


def build_handle(r_in, r_out, z_rim, r_tube, z_lo, z_hi, handle_clear,
                 m=72, k=16, psi_e_deg=70.0, side=-1.0):
    """Closed C-shaped tube (elliptical-arc centerline in the x-z plane, azimuth of `side`).

    End caps are buried in the wall solid: cap centers at rho_end=(r_in+r_out)/2 and the
    whole cap disc verified to stay inside the wall annulus [r_in, r_out].
    Returns (mesh, info dict).
    """
    assert side in (-1.0, 1.0)
    psi_e = np.radians(psi_e_deg)
    assert 45.0 < psi_e_deg < 90.0
    rho_end = 0.5 * (r_in + r_out)          # cap center radius: mid-wall
    b = (0.5 * (z_hi - z_lo)) / np.sin(psi_e)
    z_c = 0.5 * (z_hi + z_lo)

    budget = handle_clear + r_out - r_tube - rho_end   # horizontal room for the bulge
    assert budget > 0.0, "no room for handle bulge"
    a = 0.998 * budget / (1.0 + np.cos(psi_e))
    e = rho_end + a * np.cos(psi_e)
    extent = e + a + r_tube                 # farthest handle radius
    assert extent <= r_out + handle_clear + 1e-9

    # top end -> outward bulge -> bottom end (long way around, never through the axis side)
    psi = np.linspace(psi_e, 2.0 * np.pi - psi_e, m)
    P = np.stack([side * (e - a * np.cos(psi)), np.zeros(m), z_c + b * np.sin(psi)], axis=1)
    dP = np.stack([side * a * np.sin(psi), np.zeros(m), b * np.cos(psi)], axis=1)
    T = dP / np.linalg.norm(dP, axis=1, keepdims=True)
    N = np.stack([T[:, 2], np.zeros(m), -T[:, 0]], axis=1)   # in-plane normal, (T,N,B) right-handed
    Bv = np.tile(np.array([0.0, 1.0, 0.0]), (m, 1))
    assert np.allclose(np.cross(T, N), Bv, atol=1e-9)

    theta = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    verts3 = P[:, None, :] + r_tube * (
        np.cos(theta)[None, :, None] * N[:, None, :] + np.sin(theta)[None, :, None] * Bv[:, None, :]
    )
    verts = verts3.reshape(m * k, 3)        # idx = s*k + q

    def vid(s, q):
        return s * k + (q % k)

    faces = []
    for s in range(m - 1):
        for q in range(k):
            v00, v01 = vid(s, q), vid(s, q + 1)
            v10, v11 = vid(s + 1, q), vid(s + 1, q + 1)
            faces.append([v00, v11, v10])   # tube wall, normal outward
            faces.append([v00, v01, v11])
    c0 = len(verts)
    verts = np.vstack([verts, P[0], P[-1]])
    for q in range(k):
        faces.append([c0, vid(0, q + 1), vid(0, q)])            # start cap, normal -T
        faces.append([c0 + 1, vid(m - 1, q), vid(m - 1, q + 1)])  # end cap, normal +T

    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64), process=False)

    rad = np.linalg.norm(mesh.vertices[:, :2], axis=1)
    cap0 = rad[[vid(0, q) for q in range(k)]]
    cap1 = rad[[vid(m - 1, q) for q in range(k)]]
    cap_lo, cap_hi = float(min(cap0.min(), cap1.min())), float(max(cap0.max(), cap1.max()))
    attach_z = (float(P[-1, 2]), float(P[0, 2]))            # (z_lo, z_hi)
    info = dict(
        side=float(side),
        arc_deg=float(360.0 - 2.0 * psi_e_deg),
        farthest_radius=float(rad.max()),
        farthest_limit=float(r_out + handle_clear),
        attach_z=attach_z,
        rim_clearance=float(z_rim - max(P[:, 2]) - r_tube),
        cap_center_radius=float(rho_end),
        cap_penetration=float(r_out - rho_end),
        cap_rim_radius=(cap_lo, cap_hi),
    )
    return mesh, info


def merge_meshes(meshes):
    verts, faces, off = [], [], 0
    for m in meshes:
        verts.append(np.asarray(m.vertices, dtype=np.float64))
        faces.append(np.asarray(m.faces, dtype=np.int64) + off)
        off += len(m.vertices)
    return np.vstack(verts), np.vstack(faces)


def load_obj_manual(path):
    """Pure-python OBJ reader (v/f only)."""
    verts, faces = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def edge_report(faces):
    """Watertight: every undirected edge shared by exactly 2 faces.
    Winding:  every directed edge used exactly once (so each twin appears once)."""
    from collections import Counter

    directed = Counter()
    for f in np.asarray(faces):
        for a, b in zip(f, np.roll(f, -1)):
            directed[(int(a), int(b))] += 1
    undirected = Counter(frozenset(e) for e in directed)
    watertight = len(directed) > 0 and all(v == 2 for v in undirected.values())
    winding = all(v == 1 for v in directed.values())
    return watertight, winding


def check_mesh(name, mesh, expect_watertight=True):
    wt, wi = mesh.is_watertight, mesh.is_winding_consistent
    print(f"[{name:14s}] verts={len(mesh.vertices):5d} faces={len(mesh.faces):5d} "
          f"volume={mesh.volume:+.6e} watertight={wt} winding={wi} is_volume={mesh.is_volume}")
    if expect_watertight:
        assert wt and wi and mesh.is_volume, f"{name} shell is not a clean closed solid"
    return mesh.volume


def ortho_png(verts, faces, path, axis="y", res=520, pad=0.02):
    """Silhouette self-check (no GL): orthographic fill of all triangles via PIL."""
    from PIL import Image, ImageDraw

    u = verts[:, 0] if axis == "y" else verts[:, 1]
    v = verts[:, 2]
    lo = [u.min() - pad, v.min() - pad]
    hi = [u.max() + pad, v.max() + pad]
    scale = (res - 1) / max(hi[0] - lo[0], hi[1] - lo[1])
    w = int(round((hi[0] - lo[0]) * scale)) + 1
    h = int(round((hi[1] - lo[1]) * scale)) + 1
    img = Image.new("RGB", (w, h), "white")
    drw = ImageDraw.Draw(img)
    pts = np.stack([(u - lo[0]) * scale, (hi[1] - v) * scale], axis=1)
    for f in np.asarray(faces):
        drw.polygon([tuple(pts[i]) for i in f], fill=(118, 130, 150))
    img.save(path)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r-in", type=float, default=DEFAULTS["r_in"])
    p.add_argument("--h-in", type=float, default=DEFAULTS["h_in"])
    p.add_argument("--t-wall", type=float, default=DEFAULTS["t_wall"])
    p.add_argument("--t-bottom", type=float, default=DEFAULTS["t_bottom"])
    p.add_argument("--n-seg", type=int, default=DEFAULTS["n_seg"])
    p.add_argument("--r-tube", type=float, default=DEFAULTS["r_tube"])
    p.add_argument("--z-lo", type=float, default=DEFAULTS["z_lo"])
    p.add_argument("--z-hi", type=float, default=DEFAULTS["z_hi"])
    p.add_argument("--handle-clear", type=float, default=DEFAULTS["handle_clear"])
    p.add_argument("--out", default=DEFAULTS["out"])
    args = p.parse_args()

    r_out = args.r_in + args.t_wall
    z_floor = args.t_bottom
    z_rim = args.t_bottom + args.h_in

    shell = build_shell(args.r_in, args.h_in, args.t_wall, args.t_bottom, args.n_seg)
    handle, hinfo = build_handle(args.r_in, r_out, z_rim, args.r_tube, args.z_lo, args.z_hi,
                                 args.handle_clear, m=DEFAULTS["m"], k=DEFAULTS["k"],
                                 psi_e_deg=DEFAULTS["psi_e_deg"], side=-1.0)

    print(f"parameters       : r_in={args.r_in} h_in={args.h_in} t_wall={args.t_wall} "
          f"t_bottom={args.t_bottom} n_seg={args.n_seg} | z_floor={z_floor} z_rim={z_rim} r_out={r_out}")
    print("== mug.obj geometry table ==")
    vol_shell = check_mesh("shell:cup", shell)
    vol_handle = check_mesh("shell:handle", handle)
    exp_vol = np.pi * (r_out ** 2 * z_rim - args.r_in ** 2 * args.h_in)
    print(f"[cup-volume    ] {vol_shell:.6e} (analytic {exp_vol:.6e}, "
          f"rel err {abs(vol_shell - exp_vol) / exp_vol:.2e})")
    assert abs(vol_shell - exp_vol) / exp_vol < 1e-3

    # handle embedding / clearance proofs
    assert hinfo["farthest_radius"] <= hinfo["farthest_limit"]
    assert hinfo["cap_penetration"] >= 0.004
    assert hinfo["cap_rim_radius"][0] > args.r_in and hinfo["cap_rim_radius"][1] < r_out
    assert hinfo["attach_z"][1] <= z_rim - 0.05
    print(f"[handle-arc    ] sweep={hinfo['arc_deg']:.1f} deg (>180)  tube r={args.r_tube}  "
          f"M={DEFAULTS['m']} sections x K={DEFAULTS['k']}")
    print(f"[handle-extent ] farthest radius={hinfo['farthest_radius']:.5f} "
          f"(limit r_out+{args.handle_clear}={hinfo['farthest_limit']:.5f})  "
          f"rim clearance={hinfo['rim_clearance']:.5f} (>=0.05)")
    print(f"[handle-attach ] z_lo={hinfo['attach_z'][0]:.5f} z_hi={hinfo['attach_z'][1]:.5f} "
          f"(both <= rim {z_rim:.3f} - 0.05)")
    print(f"[handle-embed  ] cap center radius={hinfo['cap_center_radius']:.5f} "
          f"-> penetration {hinfo['cap_penetration']:.5f} (>=0.004)  "
          f"cap rim radius {hinfo['cap_rim_radius'][0]:.5f}..{hinfo['cap_rim_radius'][1]:.5f} "
          f"inside wall annulus [{args.r_in:.5f}, {r_out:.5f}]")

    verts, faces = merge_meshes([shell, handle])
    print(f"[merged        ] verts={len(verts)} faces={len(faces)} "
          f"(union of 2 watertight shells -> per-shell assertions only)")
    print(f"[merged-bbox   ] min={verts.min(axis=0)} max={verts.max(axis=0)}")
    assert abs(verts[:, 2].min() - 0.0) < 1e-12 and abs(verts[:, 2].max() - z_rim) < 1e-12
    assert len(verts) < 20000 and len(faces) < 20000

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(args.out)
    print(f"exported         : {args.out}")

    # reload with the pure-python parser: per-shell + merged checks
    rv, rf = load_obj_manual(args.out)
    wt, wi = edge_report(rf)
    print(f"[reload-manual ] verts={len(rv)} faces={len(rf)} watertight={wt} winding={wi}")
    assert len(rv) == len(verts) and len(rf) == len(faces) and wt and wi
    assert np.allclose(rv, verts, atol=1e-12)
    wts = edge_report(np.asarray(shell.faces))
    wth = edge_report(np.asarray(handle.faces))
    print(f"[reload-shells ] cup watertight={wts[0]} winding={wts[1]} | "
          f"handle watertight={wth[0]} winding={wth[1]}")
    assert wts[0] and wts[1] and wth[0] and wth[1]
    re = trimesh.load(args.out, force="mesh")
    print(f"[reload-trimesh] watertight={re.is_watertight} winding={re.is_winding_consistent} "
          f"bodies={re.body_count} volume={re.volume:.6e}")

    try:
        pv = ortho_png(rv, rf, os.path.splitext(args.out)[0] + "_front.png", axis="y")
        ps = ortho_png(rv, rf, os.path.splitext(args.out)[0] + "_side.png", axis="x")
        print(f"preview png     : {pv} , {ps}")
    except Exception as exc:  # pragma: no cover
        print(f"preview png     : skipped ({exc})")

    print("WATERTIGHT PASS (both shells)")


if __name__ == "__main__":
    main()
