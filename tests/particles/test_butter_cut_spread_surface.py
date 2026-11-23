from pathlib import Path

import numpy as np
import pytest
import trimesh
from skimage.measure import marching_cubes

from examples.butter_cut_spread.config import Config
from examples.butter_cut_spread.density import splat_density
from examples.butter_cut_spread.surface import closed_density_surface, coating_density
from examples.butter_cut_spread.visibility import visible_holes
from examples.butter_spreading.surface import splat
from tests.utils import assert_allclose, assert_equal


@pytest.mark.parametrize("backend", [None])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_density_preserves_accumulation_and_anisotropic_particle_volume(backend, dtype):
    spacing, voxel, width = 0.0004, 0.0001, 0.00085
    origin, shape = np.full(3, -0.003), (61, 61, 61)
    points = np.array(((0, 0, 0), (0.000173, -0.000281, 0.000399), (-0.003, 0.0029, 0), (0.004, 0, 0)), dtype=dtype)
    assert_equal(
        splat_density(points, origin, shape, voxel, width, spacing**3),
        splat(points, origin, shape, voxel, width, spacing**3),
    )
    points = np.zeros((1, 3), dtype=dtype)
    density = splat_density(points, origin, shape, voxel, width, spacing**3, np.array((0.0002,)), np.array((0.0008,)))
    assert_allclose(density.sum(dtype=np.float64) * voxel**3, spacing**3, rtol=1e-6, atol=1e-16)
    coordinate = origin[0] + np.arange(shape[0]) * voxel
    normal_moment = np.dot(density.sum(axis=(0, 1), dtype=np.float64), coordinate**2)
    tangent_moment = np.dot(density.sum(axis=(1, 2), dtype=np.float64), coordinate**2)
    assert_allclose(tangent_moment / normal_moment, 16.0, rtol=1e-6)


@pytest.mark.parametrize("backend", [None])
@pytest.mark.parametrize("voxel, normal_floor_ratio", [(0.0002, 1.0), (0.0001, 0.25)])
def test_supported_single_layer_retains_mass_and_real_gaps(backend, voxel, normal_floor_ratio):
    config = Config()
    spacing, width = config.particle_size, 0.00085
    axis = (np.arange(31) - 15) * spacing
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((x.reshape(-1) + config.bread_center[0], y.reshape(-1), np.full(x.size, 0.0142)))
    origin = points.min(axis=0) - 2 * width
    shape = tuple(np.ceil((points.max(axis=0) + 2 * width - origin) / voxel).astype(int) + 1)
    field = coating_density(points, origin, shape, voxel, width, config, normal_floor_ratio)
    assert abs(field.sum() * voxel**3 / (len(points) * spacing**3) - 1) < 0.02
    indices = np.round((points - origin) / voxel).astype(int)
    central = (abs(points[:, 0] - config.bread_center[0]) < 0.004) & (abs(points[:, 1]) < 0.004)
    assert np.all(field.max(axis=2)[indices[central, 0], indices[central, 1]] > 0.5)
    gap = np.linalg.norm(points[:, :2] - config.bread_center[:2], axis=1) < 0.002
    field = coating_density(points[~gap], origin, shape, voxel, width, config, normal_floor_ratio)
    center = np.round((np.array(config.bread_center[:2]) - origin[:2]) / voxel).astype(int)
    assert field[center[0], center[1]].max() < 0.01


@pytest.mark.parametrize("backend", [None])
def test_visible_coating_audit_detects_bread_protruding_through_a_closed_butter_mesh(backend):
    config = Config()
    butter = trimesh.creation.box(extents=(0.01, 0.01, 0.001))
    butter.apply_translation((config.bread_center[0], 0, 0.0155))
    bread = trimesh.creation.box(extents=config.bread_size)
    bread.apply_translation(config.bread_center)
    result = visible_holes(butter.vertices, butter.faces, bread.vertices, bread.faces, config)
    assert result.enclosed_visible_hole_area_mm2 == 0
    raised = trimesh.creation.box(extents=(0.002, 0.002, 0.003))
    raised.apply_translation((config.bread_center[0], 0, 0.0155))
    bread = trimesh.util.concatenate((bread, raised))
    result = visible_holes(butter.vertices, butter.faces, bread.vertices, bread.faces, config)
    assert 3.5 < result.enclosed_visible_hole_area_mm2 < 5


@pytest.mark.parametrize("backend", [None])
def test_isovalue_coincidence_in_recorded_contact_frame_is_resolved_without_hole_filling(backend):
    # Retain the grid offset: float32 marching-cubes coordinates merge a saddle edge at this scale.
    fixture = Path(__file__).parent / "fixtures/butter_cut_spread_iso_contact.npz"
    with np.load(fixture) as sample:
        field = np.pad(sample["field"], tuple((offset, 2) for offset in sample["offset"]))
    voxel = 0.0002
    vertices, faces, _, _ = marching_cubes(
        field, 0.5, spacing=(voxel,) * 3, gradient_direction="ascent", allow_degenerate=False
    )
    original = trimesh.Trimesh(vertices, faces, process=False)
    assert not original.is_watertight
    result = closed_density_surface(field, np.zeros(3), voxel)
    assert result.mesh.is_volume
    assert 0 < result.level - 0.5 < 0.000011
    assert abs(result.mesh.volume - abs(original.volume)) < abs(original.volume) * 0.001
    np.testing.assert_allclose(result.mesh.bounds, original.bounds, atol=voxel / 100, rtol=0)
