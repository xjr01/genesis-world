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

    # fill the slots in the PHYSICS SDF (the visual mesh keeps them): our grains (6.25 mm) are half
    # the slot width — they jam into the slots, sift through, then get crushed under the pan and
    # erupt. Union in an analytic thin plate covering the pan floor slab (z ~0.008..0.018).
    PLATE_X = (-0.145, 0.140)
    PLATE_Y = (-0.110, 0.110)
    PLATE_Z = (0.008, 0.018)
    # exact box SDF: outside distance + inside (max of signed face distances)
    dx = np.maximum(PLATE_X[0] - pts[:, 0], pts[:, 0] - PLATE_X[1])
    dy = np.maximum(PLATE_Y[0] - pts[:, 1], pts[:, 1] - PLATE_Y[1])
    dz = np.maximum(PLATE_Z[0] - pts[:, 2], pts[:, 2] - PLATE_Z[1])
    plate = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2 + np.maximum(dz, 0)**2) + np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0.0)
    sdf = np.minimum(sdf, plate.reshape(dims).astype(np.float32))
    print(f"slots filled with a plate {PLATE_X}x{PLATE_Y}x{PLATE_Z}; sdf range now [{sdf.min():.4f}, {sdf.max():.4f}]", flush=True)

    # Topological sign correction: the pseudo-normal sign is unreliable on this asset — it paints a
    # wrong-sign NEGATIVE halo around the handle in open space (~56k cells, reachable from the grid
    # boundary), which violently ejects grains that touch it. The magnitude from igl is fine, so fix
    # the sign topologically: cells with |phi| <= 1 cell are barriers; flood fill the rest from the
    # grid boundary — reachable => positive (outside), unreachable => negative (inside solid);
    # barrier cells inherit the sign of their nearest non-barrier cell.
    from scipy import ndimage

    free = np.abs(sdf) > 1.0 * CELL
    lab, _ = ndimage.label(free)
    bnd = set(
        np.unique(
            np.concatenate(
                [
                    lab[0, :, :].ravel(), lab[-1, :, :].ravel(),
                    lab[:, 0, :].ravel(), lab[:, -1, :].ravel(),
                    lab[:, :, 0].ravel(), lab[:, :, -1].ravel(),
                ]
            )
        )
    )
    bnd.discard(0)
    outside_free = np.isin(lab, list(bnd))
    ind = ndimage.distance_transform_edt(~free, return_distances=False, return_indices=True)
    outside = outside_free[tuple(ind)]
    n_flip = int(np.count_nonzero((sdf < 0) & outside) + np.count_nonzero((sdf > 0) & ~outside))
    sdf = np.where(outside, np.abs(sdf), -np.abs(sdf)).astype(np.float32)
    print(f"flood-fill sign fix: {n_flip} cells flipped; sdf range now [{sdf.min():.4f}, {sdf.max():.4f}]", flush=True)

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
