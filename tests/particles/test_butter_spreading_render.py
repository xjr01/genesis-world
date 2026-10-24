import numpy as np
import pytest

from examples.butter_spreading.render_geometry import bind_bread, deform_bread


@pytest.mark.parametrize("backend", [None])
def test_bread_display_preserves_rigid_and_affine_material_motion(backend):
    spacing = 0.002
    x, y, z = np.meshgrid(np.arange(10), np.arange(8), np.arange(4), indexing="ij")
    particles = np.stack((x, y, z), axis=-1).reshape((-1, 3)) * spacing
    binding = bind_bread(particles, spacing)
    rest, rest_normals = deform_bread(binding, particles, particles)
    np.testing.assert_allclose(rest.min(axis=0), particles.min(axis=0) - spacing / 2, atol=1e-14)
    np.testing.assert_allclose(rest.max(axis=0), particles.max(axis=0) + spacing / 2, atol=1e-14)
    matrix = np.array(((0.97, 0.12, 0.0), (0.0, 1.03, 0.08), (0.0, 0.0, 0.72)))
    shift = np.array((0.04, -0.03, 0.01))
    moved, normals = deform_bread(binding, particles, particles @ matrix.T + shift)
    np.testing.assert_allclose(moved, rest @ matrix.T + shift, atol=1e-14)
    expected = rest_normals @ np.linalg.inv(matrix)
    expected /= np.linalg.norm(expected, axis=1)[:, None]
    np.testing.assert_allclose(normals, expected, atol=1e-12)
    # Particle row ordering must not change the bound material geometry.
    permutation = np.random.default_rng(17).permutation(len(particles))
    rebound = bind_bread(particles[permutation], spacing)
    reordered, _ = deform_bread(rebound, particles[permutation], (particles @ matrix.T + shift)[permutation])
    np.testing.assert_allclose(reordered, moved, atol=1e-14)
