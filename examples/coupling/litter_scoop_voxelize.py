"""
Phase 8c: voxelize the aligned litter-scoop mesh into an SDF grid for the DEM mesh obstacle
(dem.set_sdf_obstacle). The aligned glb is glTF Y-up; rotating +90 deg about x recovers the
blade frame (pan floor in xy, tip +x, handle along the 40 deg direction) that the DEM
obstacle pose is expressed in.

Output: experiments/assets/litter_scoop_sdf.npz  (sdf_val (nx,ny,nz) numpy C-order, origin, cell;
grid nodes are at origin + (i + 0.5) * cell, matching the kernel's -0.5 offset convention)
"""

import os
import time

import numpy as np
import trimesh
import igl

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_GLB = os.path.join(HERE, "assets", "litter_scoop_aligned.glb")
OUT_NPZ = os.path.join(HERE, "assets", "litter_scoop_sdf.npz")

PARTICLE_RADIUS = 3.125e-3
CELL = 1.5e-3  # ~half particle radius: resolves the pan walls (3-4 mm)
PAD = 2.0 * PARTICLE_RADIUS


def main():
    m = trimesh.load(MESH_GLB, force="mesh")
    # glTF Y-up file -> blade frame (z-up): rotate +90 deg about x
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0]))
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces)
    lo, hi = m.bounds
    print(f"mesh bounds (blade frame): {np.round(lo, 4).tolist()} .. {np.round(hi, 4).tolist()}", flush=True)

    origin = lo - PAD
    dims = np.ceil((hi - lo + 2.0 * PAD) / CELL).astype(int)
    print(f"grid dims {dims.tolist()} -> {int(np.prod(dims)) / 1e6:.1f} M cells", flush=True)
    axes = [origin[a] + (np.arange(dims[a]) + 0.5) * CELL for a in range(3)]
    GX, GY, GZ = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    pts = np.stack([GX, GY, GZ], axis=-1).reshape(-1, 3)

    t0 = time.time()
    # pseudo-normal sign: correct for the thin open shell (slots in the pan floor); the default
    # winding-number sign mislabels the thin floor slab's interior as outside and grains fall through
    sdf, *_ = igl.signed_distance(pts, V, F, sign_type=igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
    sdf = sdf.reshape(dims).astype(np.float32)
    print(f"igl.signed_distance: {time.time() - t0:.0f}s, sdf range [{sdf.min():.4f}, {sdf.max():.4f}]", flush=True)

    np.savez_compressed(
        OUT_NPZ,
        sdf_val=sdf,
        dims=dims.astype(np.int32),
        origin=origin.astype(np.float32),
        cell=np.array([CELL, CELL, CELL], dtype=np.float32),
    )
    print(f"saved {OUT_NPZ}", flush=True)


if __name__ == "__main__":
    main()
