"""Build and validate the shared jelly simulation surface in meters with Z up."""

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

import igl
import pymeshlab
import tetgen
import trimesh

import genesis as gs


@dataclass
class MeshQuality:
    vertices: int
    triangles: int
    volume_m3: float
    minimum_triangle_angle_deg: float
    percentile_1_triangle_angle_deg: float
    triangles_below_5_deg: int
    minimum_edge_length_m: float
    maximum_edge_length_m: float
    tetrahedral_vertices: int
    tetrahedra: int
    minimum_tetrahedron_volume_m3: float
    minimum_tetrahedron_mean_ratio: float
    percentile_1_tetrahedron_mean_ratio: float


@dataclass
class RemeshReport:
    source: str
    source_sha256: str
    source_node: str
    asset: str
    asset_sha256: str
    pymeshlab_version: str
    tetgen_version: str
    target_edge_length_m: float
    maximum_local_surface_distance_m: float
    remeshing_iterations: int
    tetrahedralization_switches: str
    original: MeshQuality
    remeshed: MeshQuality
    maximum_sampled_surface_distance_m: float
    relative_volume_change: float


def mesh_quality(mesh, switches):
    """Measure surface angles and tetrahedral mean ratios, rejecting invalid closed-volume meshes."""
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.euler_number != 2 or mesh.volume <= 0:
        raise ValueError("The jelly surface must be a closed, consistently oriented genus-zero mesh.")
    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    meshset.compute_selection_by_self_intersections_per_face()
    if meshset.current_mesh().selected_face_number():
        raise ValueError("The jelly surface contains self-intersections.")
    tetrahedralizer = tetgen.TetGen(mesh.vertices, mesh.faces.astype(np.int32))
    vertices, elements, *_ = tetrahedralizer.tetrahedralize(switches=switches)
    tets = vertices[elements]
    volumes = np.linalg.det(tets[:, 1:] - tets[:, :1]) / 6
    if not np.isfinite(volumes).all() or (volumes <= 0).any():
        raise ValueError("The initial tetrahedral mesh must have strictly positive volumes.")
    edges = tets[:, [1, 2, 3, 2, 3, 3]] - tets[:, [0, 0, 0, 1, 1, 2]]
    mean_ratio = 12 * (3 * volumes) ** (2 / 3) / np.sum(edges**2, axis=(1, 2))
    angles = np.rad2deg(mesh.face_angles.min(axis=1))
    return MeshQuality(
        len(mesh.vertices),
        len(mesh.faces),
        float(mesh.volume),
        float(angles.min()),
        float(np.percentile(angles, 1)),
        int(np.count_nonzero(angles < 5)),
        float(mesh.edges_unique_length.min()),
        float(mesh.edges_unique_length.max()),
        len(vertices),
        len(elements),
        float(volumes.min()),
        float(mean_ratio.min()),
        float(np.percentile(mean_ratio, 1)),
    )


def main():
    asset_root = Path(gs.__file__).parent / "assets"
    source_path = asset_root / "meshes" / "jelly" / "jelly.glb"
    output_dir = asset_root / "meshes" / "jelly"
    asset_path = output_dir / "remeshed.obj"
    source = trimesh.load(source_path, force="scene", process=False)
    transform, geometry = source.graph["Jelly_Fluted_12Lobes"]
    mesh = source.geometry[geometry].copy()
    mesh.apply_transform(transform)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    edge_length, surface_distance, iterations = 0.002, 0.0001, 15
    switches = "pq1.5/10Ya0.0000001Q"
    original_quality = mesh_quality(mesh, switches)
    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    meshset.meshing_isotropic_explicit_remeshing(
        iterations=iterations,
        targetlen=pymeshlab.PureValue(edge_length),
        maxsurfdist=pymeshlab.PureValue(surface_distance),
    )
    remeshed = trimesh.Trimesh(
        meshset.current_mesh().vertex_matrix(), meshset.current_mesh().face_matrix(), process=False
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_text = remeshed.export(file_type="obj", digits=12)
    # Validate the serialized coordinates used by every simulation.
    candidate_path = output_dir / "remeshed.candidate.obj"
    candidate_path.write_text(asset_text, encoding="ascii")
    remeshed = trimesh.load(candidate_path, force="mesh", process=False)
    remeshed_quality = mesh_quality(remeshed, switches)
    maximum_distance = 0.0
    for query, reference in ((mesh, remeshed), (remeshed, mesh)):
        points = np.concatenate((query.vertices, query.triangles_center))
        distances, _, _ = igl.point_mesh_squared_distance(points, reference.vertices, reference.faces)
        maximum_distance = max(maximum_distance, float(np.sqrt(distances.max())))
    volume_change = remeshed.volume / mesh.volume - 1
    if remeshed_quality.minimum_triangle_angle_deg < 15 or maximum_distance > 0.00025 or abs(volume_change) > 0.001:
        raise ValueError("Remeshing failed the surface angle, sampled shape distance, or volume preservation checks.")
    candidate_path.replace(asset_path)
    report = RemeshReport(
        source_path.relative_to(asset_root).as_posix(),
        hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "Jelly_Fluted_12Lobes",
        asset_path.relative_to(asset_root).as_posix(),
        hashlib.sha256(asset_path.read_bytes()).hexdigest(),
        version("pymeshlab"),
        version("tetgen"),
        edge_length,
        surface_distance,
        iterations,
        switches,
        original_quality,
        remeshed_quality,
        maximum_distance,
        float(volume_change),
    )
    report_text = json.dumps(asdict(report), indent=2)
    (output_dir / "quality.json").write_text(report_text + "\n", encoding="ascii")
    print(report_text)


if __name__ == "__main__":
    main()
