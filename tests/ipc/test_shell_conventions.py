import numpy as np
import pytest

from genesis.engine.couplers.ipc_coupler.coupler import (
    _shell_bending_stiffness_for_uipc,
    _shell_hinge_areas,
    _shell_membrane_stiffness_for_uipc,
)


@pytest.mark.parametrize(
    ("version", "expected"),
    (("0.0.25", 40.0), ("0.0.26", 0.004), ("0.0.28", 0.004), ("0.0.28.dev1", 0.004)),
)
def test_shell_bending_stiffness_tracks_uipc_convention(version, expected):
    assert _shell_bending_stiffness_for_uipc(40.0, 1.0e-4, version) == pytest.approx(expected)


def test_shell_hinge_area_preserves_legacy_bending_energy():
    positions = np.array(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 1.0, 0.0)))
    faces = np.array(((0, 1, 2), (1, 3, 2)))
    edges = np.array(((3, 2), (2, 1), (0, 2), (1, 3), (1, 0)))
    areas = _shell_hinge_areas(positions, faces, edges)
    np.testing.assert_allclose(areas, (1.0, 2.0, 1.0, 1.0, 1.0))
    # Uniform geometric scaling must preserve the source's area-weighted metric.
    np.testing.assert_allclose(_shell_hinge_areas(positions * 0.001, faces, edges), areas * 1.0e-6)


@pytest.mark.parametrize("version", ("0.0.25", "0.0.26", "0.0.28"))
def test_shell_membrane_energy_coefficients_preserve_source_calibration(version):
    thickness = 1.0e-4
    lame_lambda = 20000.0 * 0.49 / (1.0 - 0.49**2)
    lame_mu = 20000.0 / (2.0 * (1.0 + 0.49))
    stretch, shear = _shell_membrane_stiffness_for_uipc(lame_lambda, lame_mu, thickness, version)
    backend_measure = 2.0 * thickness if version == "0.0.25" else 1.0
    assert stretch * backend_measure == pytest.approx(2.57928674825635)
    assert shear * backend_measure == pytest.approx(1.34228187919463)
