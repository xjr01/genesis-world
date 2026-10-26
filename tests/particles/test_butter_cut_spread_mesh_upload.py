import numpy as np
import pytest
import trimesh

try:
    from pxr import Usd, UsdGeom
except ImportError:
    pytest.skip("USD is not supported because 'pxr' module is not available.", allow_module_level=True)

from examples.butter_cut_spread.mesh_upload import upload_triangle_mesh
from tests.utils import assert_equal


@pytest.mark.parametrize("backend", [None])
def test_triangle_upload_preserves_vertices_normals_winding_and_extent(backend):
    stage = Usd.Stage.CreateInMemory()
    target = UsdGeom.Mesh.Define(stage, "/Butter")
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.012)
    mesh.apply_translation((0.13, -0.02, 0.007))
    upload_triangle_mesh(target, mesh)
    assert_equal(np.array(target.GetPointsAttr().Get()), mesh.vertices.astype(np.float32))
    assert_equal(np.array(target.GetNormalsAttr().Get()), mesh.vertex_normals.astype(np.float32))
    assert_equal(np.array(target.GetFaceVertexCountsAttr().Get()), np.full(len(mesh.faces), 3, dtype=np.int32))
    assert_equal(np.array(target.GetFaceVertexIndicesAttr().Get()), np.concatenate(mesh.faces).astype(np.int32))
    assert_equal(
        np.array(target.GetExtentAttr().Get()),
        np.stack((mesh.vertices.min(axis=0), mesh.vertices.max(axis=0))).astype(np.float32),
    )
