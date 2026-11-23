"""Upload triangle buffers without iterating over tracked face-array rows."""

import numpy as np
from pxr import Vt


def upload_triangle_mesh(target, mesh):
    """Preserve the recorded mesh, normals and extent in the USD buffers."""
    vectors = (
        (target.GetPointsAttr(), mesh.vertices),
        (target.GetNormalsAttr(), mesh.vertex_normals),
        (target.GetExtentAttr(), np.stack((mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)))),
    )
    for attribute, values in vectors:
        attribute.Set(Vt.Vec3fArray.FromNumpy(values.astype(np.float32)))
    target.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, dtype=np.int32)))
    target.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(mesh.faces.reshape(-1).astype(np.int32)))
