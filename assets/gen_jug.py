"""Generate jug.obj: gen_cup shell + shallow beak spout (-x) + C-tube handle (+x).

Body    : same parameterized shell as gen_cup/gen_mug (4 rings + 2 centers, same
          winding), z=0 at the OUTER bottom, cavity floor z=t_bottom, rim z=t_bottom+h_in.
Spout   : shallow beak pouring lip at azimuth pi (-x). Implementation = vertex-only
          displacement of the OUTER-TOP ring (B): radial outward + slight drop inside a
          smooth cos^2 azimuth window. Topology untouched -> the shell stays a single
          watertight solid and the fluid-wetted cavity (C/D rings, floor) is bit-identical
          to the analytic cylinder.
          HARD CONSTRAINT: spout-tip lowest z >= z_rim - 0.002, i.e. flush with the
          analytic lip circle, so fluid leaving the analytic rim can never be seen
          escaping "behind" the spout. Tip protrusion <= 0.030, half-width 20 deg
          (arc length <= 0.10 at the rim circle).
Handle  : same C-tube construction as gen_mug, scaled to the jug (r_tube 0.011,
          farthest point <= r_out + 0.05, attachments z 0.09 / 0.21).

Alignment: shell geometry matches analytic boundary_pitcher_shell params
           (r_in=0.125, h_in=0.30, t_wall=0.016, t_bottom=0.016, particle 0.008);
           the pitcher's pose is owned by the scene script.

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 "$PY" gen_jug.py
"""

import argparse
import os

import numpy as np
import trimesh

from gen_mug import (ASSETS, build_handle, build_shell, check_mesh, edge_report,
                     load_obj_manual, merge_meshes, ortho_png)

# Analytic-shell-aligned defaults (meters)
DEFAULTS = dict(
    r_in=0.125,
    h_in=0.30,
    t_wall=0.016,    # = 2 * particle_size 0.008
    t_bottom=0.016,
    n_seg=96,
    spout_center_deg=180.0,
    spout_dr=0.030,
    spout_dz=-0.002,     # >= -0.002 hard limit (flush with the analytic lip circle)
    spout_half_deg=20.0,  # half width -> rim arc length 2*20deg*0.141 = 0.0985 <= 0.10
    handle_side=1.0,
    r_tube=0.011,
    z_lo=0.09,
    z_hi=0.21,
    handle_clear=0.05,
    out=os.path.join(ASSETS, "jug.obj"),
)

M_SECTIONS, K_SEG, PSI_E_DEG = 72, 16, 70.0


def make_spout_fn(r_out, z_rim, dr, dz, half_deg, center_deg=180.0):
    """Vertex-only B-ring displacer: smooth cos^2 beak around ``center_deg``."""
    half = np.radians(half_deg)
    center = np.radians(center_deg)

    def fn(B, theta):
        d = np.arctan2(np.sin(theta - center), np.cos(theta - center))
        w = np.where(np.abs(d) <= half, np.cos(0.5 * np.pi * d / half) ** 2, 0.0)
        r = np.linalg.norm(B[:, :2], axis=1) + dr * w      # radial outward only
        out = np.stack([r * np.cos(theta), r * np.sin(theta), B[:, 2] + dz * w], axis=1)
        assert out[:, 2].min() >= z_rim + dz - 1e-12       # hard spout-tip constraint
        return out

    return fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r-in", type=float, default=DEFAULTS["r_in"])
    p.add_argument("--h-in", type=float, default=DEFAULTS["h_in"])
    p.add_argument("--t-wall", type=float, default=DEFAULTS["t_wall"])
    p.add_argument("--t-bottom", type=float, default=DEFAULTS["t_bottom"])
    p.add_argument("--n-seg", type=int, default=DEFAULTS["n_seg"])
    p.add_argument("--spout-dr", type=float, default=DEFAULTS["spout_dr"])
    p.add_argument("--spout-dz", type=float, default=DEFAULTS["spout_dz"])
    p.add_argument("--spout-half-deg", type=float, default=DEFAULTS["spout_half_deg"])
    p.add_argument("--r-tube", type=float, default=DEFAULTS["r_tube"])
    p.add_argument("--z-lo", type=float, default=DEFAULTS["z_lo"])
    p.add_argument("--z-hi", type=float, default=DEFAULTS["z_hi"])
    p.add_argument("--handle-clear", type=float, default=DEFAULTS["handle_clear"])
    p.add_argument("--out", default=DEFAULTS["out"])
    args = p.parse_args()

    r_out = args.r_in + args.t_wall
    z_floor = args.t_bottom
    z_rim = args.t_bottom + args.h_in
    assert args.spout_dz >= -0.002, "spout tip must not sink more than 0.002 below the lip"

    spout = make_spout_fn(r_out, z_rim, args.spout_dr, args.spout_dz, args.spout_half_deg,
                           center_deg=DEFAULTS["spout_center_deg"])
    shell = build_shell(args.r_in, args.h_in, args.t_wall, args.t_bottom, args.n_seg, b_ring_fn=spout)
    handle, hinfo = build_handle(args.r_in, r_out, z_rim, args.r_tube, args.z_lo, args.z_hi,
                                 args.handle_clear, m=M_SECTIONS, k=K_SEG,
                                 psi_e_deg=PSI_E_DEG, side=DEFAULTS["handle_side"])

    print(f"parameters       : r_in={args.r_in} h_in={args.h_in} t_wall={args.t_wall} "
          f"t_bottom={args.t_bottom} n_seg={args.n_seg} | z_floor={z_floor} z_rim={z_rim} r_out={r_out}")
    print("== jug.obj geometry table ==")
    check_mesh("shell:jug+spout", shell)
    check_mesh("shell:handle", handle)
    exp_vol = np.pi * (r_out ** 2 * z_rim - args.r_in ** 2 * args.h_in)
    print(f"[volume        ] shell={shell.volume:.6e} (plain-shell analytic {exp_vol:.6e}; "
          f"spout adds a small wedge on top)")

    # spout proofs: tip flush with the analytic lip circle, cavity untouched
    theta = np.linspace(0.0, 2.0 * np.pi, args.n_seg, endpoint=False)
    Bm = np.asarray(spout(np.stack([r_out * np.cos(theta), r_out * np.sin(theta),
                                    np.full(args.n_seg, z_rim)], axis=1), theta))
    tip_z, tip_r = float(Bm[:, 2].min()), float(np.linalg.norm(Bm[:, :2], axis=1).max())
    center = np.radians(DEFAULTS["spout_center_deg"])
    sector = np.abs(np.arctan2(np.sin(theta - center), np.cos(theta - center))) <= np.radians(args.spout_half_deg)
    arc_len = float(2.0 * np.radians(args.spout_half_deg) * r_out)
    assert tip_z >= z_rim - 0.002
    assert tip_r <= r_out + args.spout_dr + 1e-12
    assert arc_len <= 0.10 + 1e-12
    tip_i = int(np.argmax(np.linalg.norm(Bm[:, :2], axis=1)))
    assert Bm[tip_i, 0] < 0.0, "MF-16 jug spout must be on local -x"
    assert hinfo["side"] > 0.0, "MF-16 jug handle must be on local +x"
    print(f"[spout         ] tip z={tip_z:.5f} >= z_rim-0.002={z_rim - 0.002:.5f} "
          f"(flush with analytic lip, dz={args.spout_dz:+.5f})")
    print(f"[spout         ] tip radius={tip_r:.5f} (<= r_out+{args.spout_dr}={r_out + args.spout_dr:.5f})  "
          f"half width=+-{args.spout_half_deg} deg -> rim arc {arc_len:.5f} (<=0.10)  "
          f"vertices moved={int(sector.sum())}/{args.n_seg} (topology unchanged)")

    # handle proofs
    assert hinfo["farthest_radius"] <= hinfo["farthest_limit"]
    assert hinfo["cap_penetration"] >= 0.004
    assert hinfo["cap_rim_radius"][0] > args.r_in and hinfo["cap_rim_radius"][1] < r_out
    assert hinfo["attach_z"][1] <= z_rim - 0.05
    print(f"[handle-arc    ] sweep={hinfo['arc_deg']:.1f} deg (>180)  tube r={args.r_tube}  "
          f"M={M_SECTIONS} sections x K={K_SEG}")
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

    rv, rf = load_obj_manual(args.out)
    wt, wi = edge_report(rf)
    print(f"[reload-manual ] verts={len(rv)} faces={len(rf)} watertight={wt} winding={wi}")
    assert len(rv) == len(verts) and len(rf) == len(faces) and wt and wi
    assert np.allclose(rv, verts, atol=1e-12)
    wts, wth = edge_report(np.asarray(shell.faces)), edge_report(np.asarray(handle.faces))
    print(f"[reload-shells ] body watertight={wts[0]} winding={wts[1]} | "
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
