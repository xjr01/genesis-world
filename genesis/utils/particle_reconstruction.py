"""Reconstruct particle surfaces in a lightweight, isolated process."""

import argparse
import pickle
import sys
from typing import NamedTuple

import numpy as np

try:
    import pysplashsurf
except ImportError:
    pysplashsurf = None


class ReconstructedSurface(NamedTuple):
    vertices: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray


def reconstruct_surface(positions, radius, radius_scale=1.0, smoothing_iterations=25):
    if pysplashsurf is None:
        raise ImportError("The splashsurf reconstruction backend requires pysplashsurf.")
    mesh, _ = pysplashsurf.reconstruction_pipeline(
        positions,
        particle_radius=radius * radius_scale,
        smoothing_length=2.0,
        cube_size=0.8,
        iso_surface_threshold=0.6,
        mesh_smoothing_weights=True,
        mesh_smoothing_iters=smoothing_iterations,
        normals_smoothing_iters=10,
        mesh_cleanup=True,
        compute_normals=True,
        multi_threading=True,
    )
    return ReconstructedSurface(mesh.mesh.vertices, mesh.mesh.triangles, mesh.point_attributes["normals"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("radius", type=float)
    parser.add_argument("radius_scale", type=float)
    parser.add_argument("smoothing_iterations", type=int)
    args = parser.parse_args()
    positions = pickle.load(sys.stdin.buffer)
    surface = reconstruct_surface(positions, args.radius, args.radius_scale, args.smoothing_iterations)
    pickle.dump(tuple(surface), sys.stdout.buffer, protocol=pickle.HIGHEST_PROTOCOL)
