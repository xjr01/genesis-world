"""Generate an open-top cylindrical cup mesh (side wall + bottom, no lid) for MF-3 / MF-4.

Parametric construction (deterministic, no boolean engine needed):
  - outer side wall : radius R_out, z in [0, z_rim]
  - inner side wall : radius R_in,  z in [z_floor, z_rim]
  - top rim annulus : z = z_rim, radius in [R_in, R_out]
  - bottom disc     : z = 0, radius R_out
All faces wound CCW seen from outside -> normals point away from the cup solid,
which is watertight and suitable for SDF-based collision after convexify.

Defaults reproduce the original MF-3 cup (assets/cup.obj). For MF-4's tall cup:
  "$PY" multiflow/assets/gen_cup.py --r-in 0.12 --h-in 0.75 --out multiflow/assets/cup_tall.obj

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; "$PY" multiflow/assets/gen_cup.py
"""

import argparse
import os

import numpy as np
import trimesh

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/

# Default parameters (meters), per MF-3 spec
DEFAULTS = dict(
    r_in=0.15,       # inner radius
    h_in=0.6,        # inner height (floor -> rim)
    t_wall=0.02,     # side wall thickness (= 2 * particle_size 0.01)
    t_bottom=0.02,   # bottom thickness
    n_seg=96,        # radial segments (>= 64)
    out=os.path.join(WORKSPACE, "assets", "cup.obj"),
)


def ring(radius, z, n):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n, z)], axis=1)


def build_cup(r_in, h_in, t_wall, t_bottom, n_seg):
    n = n_seg
    r_out = r_in + t_wall
    z0 = 0.0                  # outer bottom
    z_floor = t_bottom        # inner floor
    z_rim = t_bottom + h_in   # rim height

    A = ring(r_out, z0, n)      # outer bottom
    B = ring(r_out, z_rim, n)   # outer top
    C = ring(r_in, z_rim, n)    # inner top
    D = ring(r_in, z_floor, n)  # inner floor edge
    O = np.array([[0.0, 0.0, z0]])  # bottom center (outer, z=0)
    O2 = np.array([[0.0, 0.0, z_floor]])  # cavity floor center (z=t_bottom)

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r-in", type=float, default=DEFAULTS["r_in"], help="inner radius")
    p.add_argument("--h-in", type=float, default=DEFAULTS["h_in"], help="inner height (floor -> rim)")
    p.add_argument("--t-wall", type=float, default=DEFAULTS["t_wall"], help="side wall thickness")
    p.add_argument("--t-bottom", type=float, default=DEFAULTS["t_bottom"], help="bottom thickness")
    p.add_argument("--n-seg", type=int, default=DEFAULTS["n_seg"], help="radial segments")
    p.add_argument("--out", default=DEFAULTS["out"], help="output .obj path")
    args = p.parse_args()

    mesh = build_cup(args.r_in, args.h_in, args.t_wall, args.t_bottom, args.n_seg)

    r_out = args.r_in + args.t_wall
    z_rim = args.t_bottom + args.h_in
    expected_vol = np.pi * (r_out**2 * z_rim - args.r_in**2 * args.h_in)
    print(f"parameters       : r_in={args.r_in} h_in={args.h_in} t_wall={args.t_wall} "
          f"t_bottom={args.t_bottom} n_seg={args.n_seg}")
    print(f"aspect (h/2r)    : {args.h_in / (2.0 * args.r_in):.3f}")
    print(f"vertices         : {len(mesh.vertices)}")
    print(f"faces            : {len(mesh.faces)}")
    print(f"is_watertight    : {mesh.is_watertight}")
    print(f"is_winding_consistent: {mesh.is_winding_consistent}")
    print(f"is_volume        : {mesh.is_volume}")
    print(f"volume           : {mesh.volume:.6e} (expected {expected_vol:.6e}, "
          f"rel err {abs(mesh.volume - expected_vol) / expected_vol:.2e})")
    print(f"bounds           : min {mesh.bounds[0]}, max {mesh.bounds[1]}")

    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.is_volume
    assert abs(mesh.volume - expected_vol) / expected_vol < 1e-3

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mesh.export(args.out)
    print(f"exported         : {args.out}")

    # re-load sanity check (what Genesis will see)
    re = trimesh.load(args.out, force="mesh")
    print(f"reload ok        : watertight={re.is_watertight}, volume={re.volume:.6e}")


if __name__ == "__main__":
    main()
