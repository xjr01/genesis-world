import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.butter_cut_spread.shape_audit import deformation


@pytest.mark.parametrize("backend", [None])
def test_slice_deformation_removes_rigid_motion(backend):
    points = np.array([[x, y, z] for x in (-0.005, 0.005) for y in (-0.013, 0.013) for z in (-0.02, 0.02)])
    target = Rotation.from_euler("xyz", (0.3, -0.8, 1.2)).apply(points) + (0.2, -0.1, 0.08)
    rms, p99 = deformation(points, target)
    assert rms < 1e-15
    assert p99 < 1e-15


@pytest.mark.parametrize("backend", [None])
def test_slice_deformation_detects_compression(backend):
    points = np.array([[x, y, z] for x in (-0.005, 0.005) for y in (-0.013, 0.013) for z in (-0.02, 0.02)])
    target = points * (1, 1, 0.9)
    rms, p99 = deformation(points, target)
    np.testing.assert_allclose((rms, p99), (0.002, 0.002), rtol=0, atol=1e-15)
