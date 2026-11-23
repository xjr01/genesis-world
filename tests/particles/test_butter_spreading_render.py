import numpy as np
import pytest

from examples.butter_spreading.bread_deformation import BreadDeformation


@pytest.mark.parametrize("backend", [None])
def test_porous_bread_material_motion_and_normals(backend):
    rng = np.random.default_rng(31)
    axes = [np.linspace(-0.015, 0.015, n) for n in (13, 11, 9)]
    particles = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3)).astype(np.float32)
    rng.shuffle(particles)
    extent = np.array([0.015 + 0.5 * (a[1] - a[0]) for a in axes])
    vertices = rng.uniform(-extent, extent, (2000, 3)).astype(np.float32)
    normals = rng.normal(size=vertices.shape)
    normals = (normals / np.linalg.norm(normals, axis=1, keepdims=True)).astype(np.float32)
    binding = BreadDeformation(particles, vertices, normals)
    matrix = np.array(((0.97, 0.12, 0), (0, 1.03, 0.08), (0, 0, 0.72)))
    shift = np.array((0.002, -0.004, 0.001))
    moved, directions, determinant = binding.deform(particles @ matrix.T + shift)
    expected_normals = normals @ np.linalg.inv(matrix)
    expected_normals /= np.linalg.norm(expected_normals, axis=1, keepdims=True)
    np.testing.assert_allclose(moved, vertices @ matrix.T + shift, atol=1e-8)
    np.testing.assert_allclose(directions, expected_normals, atol=5e-6)
    assert abs(determinant - np.linalg.det(matrix)) < 5e-6
    np.testing.assert_array_equal(binding.deform(particles)[0], vertices)
    with pytest.raises(ValueError, match="Folded"):
        binding.deform(particles * np.array((1, 1, -1)))
