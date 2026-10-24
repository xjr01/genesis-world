"""Normalize the porous bread mesh and classify its crust."""

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, binary_dilation, distance_transform_edt


def prepare_bread(source, out):
    data = np.load(source)
    v = data["vertices"].copy()
    f = data["faces"].copy()
    lo = v.min(0)
    hi = v.max(0)
    v = (v - lo) / (hi - lo)
    # Bread fits the existing task workspace; its upper cut face remains z=0.
    v = (v - np.array([0.5, 0.5, 1], np.float32)) * np.array([0.14, 0.114, 0.016], np.float32)
    # VDB density and signed-distance conventions may reverse winding.
    cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    volume = float(np.einsum("ij,ij->i", v[f[:, 0]], cross).sum() / 6)
    if volume < 0:
        f = f[:, [0, 2, 1]]
        cross = -cross
    normals = np.zeros_like(v)
    for i in range(3):
        np.add.at(normals, f[:, i], cross)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    # Microscopic relief breaks perfect isosurface smoothness. The full
    # upstream resolution preserves fine walls that coarse VDB resampling lost.
    relief = (np.sin(v[:, 0] * 23000 + v[:, 1] * 3100) * np.sin(v[:, 1] * 19000 + v[:, 2] * 4500)) * 0.000015
    v += normals * relief[:, None]
    res = 768
    ids = np.clip(((v[:, :2] / [0.14, 0.114] + 0.5) * (res - 17) + 8).astype(int), 0, res - 1)
    mask = np.zeros((res, res), bool)
    mask[ids[:, 0], ids[:, 1]] = True
    mask = binary_fill_holes(binary_closing(binary_dilation(mask, iterations=2), iterations=4))
    dist = distance_transform_edt(mask, sampling=(0.14 / (res - 17), 0.114 / (res - 17)))
    depth = dist[ids[:, 0], ids[:, 1]]
    crust = depth[f].mean(axis=1) < 0.0019
    # UVs are retained for later refinement; the cut crumb uses a scattering
    # material and actual pores, rather than the v2 baked photograph.
    uv = v[:, :2] / [0.14, 0.114] + 0.5
    np.savez_compressed(
        out,
        vertices=v.astype("float32"),
        faces=f.astype("int32"),
        normals=normals.astype("float32"),
        uv=uv.astype("float32"),
        crust_faces=np.flatnonzero(crust).astype("int32"),
    )
