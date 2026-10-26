from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.butter_cut_spread.blade_film import BladeFilm
from examples.butter_cut_spread.density import splat_density
from tests.utils import assert_allclose


@pytest.mark.parametrize("backend", [None])
def test_blade_film_conserves_volume_and_rotates_both_faces(backend, tmp_path):
    axis = np.arange(-0.01, 0.01001, 0.0001)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    np.savez(
        tmp_path / "blade-face.npz",
        origin=np.array([-0.01, -0.01]),
        spacing=0.0001,
        distance=0.009 - np.maximum(abs(x), abs(y)),
    )
    film = BladeFilm(tmp_path, 0.0006, 0.0008)
    points = np.array([[-0.003, 0.0, -0.0007], [0.003, 0.0, 0.0007]])
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    result = film.partition(points, np.zeros(3), identity)
    np.testing.assert_allclose(result.bulk_weights, 0)
    np.testing.assert_allclose(result.thickness.sum(axis=(1, 2)) * film.settings.pixel**2, film.spacing**3, rtol=0.001)
    np.testing.assert_allclose(result.volumes[0], result.volumes[1])
    narrow = BladeFilm(tmp_path, film.spacing, 2 * film.half, replace(film.settings, sheet_kernel=film.settings.kernel))
    narrow_result = narrow.partition(points, np.zeros(3), identity)
    assert_allclose(result.bulk_weights, narrow_result.bulk_weights, atol=1e-14)
    assert_allclose(result.volumes, narrow_result.volumes, atol=1e-18)
    assert result.thickness.max() < narrow_result.thickness.max()
    assert np.count_nonzero(film.opacity(result.thickness)) > np.count_nonzero(narrow.opacity(narrow_result.thickness))
    refined = BladeFilm(tmp_path, film.spacing / 2, 2 * film.half)
    assert_allclose(refined.opacity(result.thickness), film.opacity(result.thickness), atol=1e-15)
    orientation = Rotation.from_euler("xyz", [0.8, -1.1, 2.3])
    center = np.array([0.1, -0.3, 0.6])
    rotated = film.partition(orientation.apply(points) + center, center, orientation.as_quat()[[3, 0, 1, 2]])
    np.testing.assert_allclose(rotated.thickness, result.thickness, atol=1e-15)
    np.testing.assert_allclose(
        rotated.bulk_weights.sum() * film.spacing**3 + sum(rotated.volumes), len(points) * film.spacing**3
    )
    empty = film.partition(points + (0, 0.03, 0), np.zeros(3), identity)
    assert not empty.thickness.any()
    assert empty.bulk_weights.min() == 1


@pytest.mark.parametrize("backend", [None])
def test_thick_column_stays_bulk_and_weighted_density_retains_complement(backend, tmp_path):
    np.savez(
        tmp_path / "blade-face.npz", origin=np.array([-0.01, -0.01]), spacing=0.001, distance=np.full((21, 21), 0.01)
    )
    film = BladeFilm(tmp_path, 0.0006, 0.0008)
    coordinates = np.arange(-4, 5) * film.spacing
    x, y, z = np.meshgrid(coordinates, coordinates, np.arange(10) * film.spacing + 0.0007, indexing="ij")
    points = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    result = film.partition(points, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    central = np.max(abs(points[:, :2]), axis=1) < 0.001
    np.testing.assert_allclose(result.bulk_weights[central], 1)
    origin, voxel, shape = np.full(3, -0.005), 0.0002, (81, 81, 81)
    density = splat_density(points, origin, shape, voxel, 0.0008, film.spacing**3, particle_weights=result.bulk_weights)
    np.testing.assert_allclose(density.sum() * voxel**3 + sum(result.volumes), len(points) * film.spacing**3, rtol=1e-5)
