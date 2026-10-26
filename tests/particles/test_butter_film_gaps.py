import numpy as np
import pytest

from examples.butter_cut_spread.film import repair_sampling_gaps


@pytest.mark.parametrize("backend", [None])
@pytest.mark.parametrize("amplitude", [0.6, 1.2])
@pytest.mark.parametrize("support", ["density", "particles", "none"])
def test_film_repairs_sampling_gap_preserving_volume_and_large_openings(backend, amplitude, support):
    voxel, spacing = 0.0002, 0.0006
    origin = np.array([-0.01, -0.01, 0.0])
    field = np.zeros((101, 101, 21), dtype=np.float32)
    field[10:91, 10:91, 7:14] = amplitude
    x, y = np.meshgrid(origin[0] + np.arange(101) * voxel, origin[1] + np.arange(101) * voxel, indexing="ij")
    small = (x + 0.004) ** 2 + y**2 < 0.0005**2
    large = (x - 0.004) ** 2 + y**2 < 0.0022**2
    field[small | large] = 0
    field[48:53, 75:, :] = 0
    before = field.copy()
    points = None
    if support != "density":
        occupied = field.max(axis=2) > 0.5
        points = np.column_stack((x[occupied], y[occupied], np.full(occupied.sum(), 0.002)))
        if support == "none":
            points[:, 2] = 0.02
    result = repair_sampling_gaps(field, origin, voxel, spacing, (0, 0, 0.002), (0.025, 0.025, 0.004), points)
    np.testing.assert_allclose(result.sum(dtype=np.float64), before.sum(dtype=np.float64), rtol=1e-6)
    if amplitude > 0.65 and support != "none":
        assert np.all(result[small].max(axis=1) > 0.5)
    else:
        np.testing.assert_array_equal(result, before)
    assert not result[large].any()
    assert not result[48:53, 75:, :].any()
    assert np.all(result >= 0)
